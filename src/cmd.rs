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

/// `Command::output()` with a wall-clock bound: stdin is nulled like
/// `.output()` does, pipes are captured the same way, but if the child
/// neither exits nor fills its pipes within `timeout` it is killed and
/// reaped (no zombies/orphans) and `Err` with `ErrorKind::TimedOut` is
/// returned. Spawn failures pass through as their `io::Error`.
pub fn output_timeout(mut cmd: Command, timeout: Duration) -> io::Result<Output> {
    // Match `.output()` plumbing (callers never set stdio themselves).
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    match child.wait_timeout(timeout)? {
        Some(status) => {
            // Exited in time: drain the pipes (no deadlock — the child is
            // gone, so reads terminate at EOF) and report like `.output()`.
            use std::io::Read as _;
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(mut o) = child.stdout.take() {
                let _ = o.read_to_end(&mut stdout);
            }
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_end(&mut stderr);
            }
            Ok(Output {
                status,
                stdout,
                stderr,
            })
        }
        None => {
            let _ = child.kill();
            // Reap after kill so no zombie/defunct entry lingers.
            let _ = child.wait();
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
}
