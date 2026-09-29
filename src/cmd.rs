//! Bounded subprocess runs: `std::process::Command` blocks forever by
//! default, and several probes hang on hostile files (FFMetrics #4, heavy
//! 4K AV1 / 50 GB ffv1 inputs). Every blocking spawn in the app goes
//! through [`output_timeout`] so nothing waits unbounded — except metric
//! runs, which stay user-stoppable via Stop but get a bounded reap (see
//! `pump_process` / `abort_worker`).

use std::io;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use wait_timeout::ChildExt;

/// Preset bounds (FFMetrics timeout history: 2s → 5s → 10s upstream).
/// Version probes answer instantly when healthy; the JSON probe and the
/// thumbnail extract touch real media; packet-count scans whole files.
pub const VERSION_TIMEOUT: Duration = Duration::from_secs(5);
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
pub const PACKET_COUNT_TIMEOUT: Duration = Duration::from_secs(60);
pub const THUMB_TIMEOUT: Duration = Duration::from_secs(15);
/// Single accurate-seek bad-frame extract (original `BadFrames.Timeout`).
pub const BADFRAME_TIMEOUT: Duration = Duration::from_secs(60);
/// Backstop for reaping an already-finished child (metric runs,
/// Stop/Reset); a healthy reap returns in ms.
pub const REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// Shared ffmpeg/ffprobe probe window (FFMetrics.conf `Metric.Template` /
/// `Thumbnail.Template` parity): larger window for sparse headers
/// (ts/m2ts/mxf). Single source so the six call sites can't drift.
pub const FFMPEG_PROBESIZE: &str = "50M";

/// Windows: spawn CLI children (ffmpeg, ffprobe, FFVship) with
/// `CREATE_NO_WINDOW` so version checks and probes don't each flash a
/// console window (the app itself is already windowed in release).
/// No-op on other platforms.
pub fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        cmd.creation_flags(0x0800_0000);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

/// `Command::output()` with a wall-clock bound: stdin is nulled like
/// `.output()` does, pipes are captured the same way, but if the child
/// doesn't exit within `timeout` it is killed and reaped (no
/// zombies/orphans) and `Err` with `ErrorKind::TimedOut` is returned.
/// Spawn failures pass through as their `io::Error`.
pub fn output_timeout(mut cmd: Command, timeout: Duration) -> io::Result<Output> {
    // Match `.output()` plumbing (callers never set stdio themselves).
    hide_console(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    // Drain pipes concurrently: a child writing more than the pipe buffer
    // (~64KB Linux, ~4KB Windows anonymous pipe) blocks on write until
    // someone reads, so waiting before reading deadlocks a healthy chatty
    // child (large ffprobe JSON, thumb PNG) into a false TimedOut.
    let mut so = child.stdout.take();
    let mut se = child.stderr.take();
    let t_out = std::thread::spawn(move || {
        use std::io::Read as _;
        let mut v = Vec::new();
        if let Some(p) = so.as_mut() {
            let _ = p.read_to_end(&mut v);
        }
        v
    });
    let t_err = std::thread::spawn(move || {
        use std::io::Read as _;
        let mut v = Vec::new();
        if let Some(p) = se.as_mut() {
            let _ = p.read_to_end(&mut v);
        }
        v
    });
    match child.wait_timeout(timeout)? {
        Some(status) => Ok(Output {
            status,
            stdout: t_out.join().unwrap_or_default(),
            stderr: t_err.join().unwrap_or_default(),
        }),
        None => {
            let _ = child.kill();
            // Reap after kill so no zombie/defunct entry lingers. Detach
            // readers (no join): a grandchild inheriting the pipes (cmd
            // -> ping) keeps them open past the kill, so joining here
            // would block until it exits; detached readers end at EOF
            // on their own.
            let _ = child.wait();
            drop(t_out);
            drop(t_err);
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("command exceeded {timeout:?}"),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Portable quick command: the test harness itself listing tests —
    /// exits 0 at once on every platform.
    fn quick_cmd() -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--list", "--format=terse"]);
        c
    }

    /// Portable sleeper: blocks on stdin, which we null — stdin closes at
    /// once, so instead hang on a child that never exits on its own.
    /// `cmd /c pause` is interactive; use a ping burst long enough to
    /// always exceed the test timeout (localhost, no network needed).
    #[cfg(windows)]
    fn hang_cmd() -> Command {
        let mut c = Command::new("cmd");
        c.args(["/c", "ping -n 30 127.0.0.1 >nul"]);
        c
    }

    /// Portable sleeper on unix.
    #[cfg(not(windows))]
    fn hang_cmd() -> Command {
        let mut c = Command::new("sleep");
        c.arg("30");
        c
    }

    #[test]
    fn quick_command_succeeds_with_output() {
        let out = output_timeout(quick_cmd(), Duration::from_secs(10))
            .expect("current exe --version must succeed");
        assert!(out.status.success());
    }

    #[test]
    fn hanging_command_times_out_and_reaps() {
        let start = std::time::Instant::now();
        let err = output_timeout(hang_cmd(), Duration::from_millis(300)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(start.elapsed() < Duration::from_secs(25));
    }

    /// Chatty child: >pipe buffer on both streams at once (~325KB each).
    /// Old wait-then-drain deadlocked this into a false TimedOut.
    #[cfg(windows)]
    fn chatty_cmd() -> Command {
        let mut c = Command::new("cmd");
        c.args([
            "/c",
            "for /L %i in (1,1,5000) do @(echo 0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF & echo ERR0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF 1>&2)",
        ]);
        c
    }

    /// Chatty child on unix.
    #[cfg(not(windows))]
    fn chatty_cmd() -> Command {
        let mut c = Command::new("sh");
        c.args([
            "-c",
            "i=0; while [ $i -lt 5000 ]; do echo 0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF; echo ERR0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF >&2; i=$((i+1)); done",
        ]);
        c
    }

    #[test]
    fn chatty_child_does_not_deadlock() {
        let out =
            output_timeout(chatty_cmd(), Duration::from_secs(10)).expect("chatty child must exit");
        assert!(out.status.success());
        assert!(out.stdout.len() > 200_000, "stdout {}", out.stdout.len());
        assert!(out.stderr.len() > 200_000, "stderr {}", out.stderr.len());
    }
}
