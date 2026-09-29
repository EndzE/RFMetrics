use std::path::PathBuf;
use std::sync::OnceLock;

use flexi_logger::{Cleanup, Criterion, FileSpec, Logger, LoggerHandle, Naming};

/// Log file basename next to the executable: `<exe_dir>/rfmetrics.log`.
const BASENAME: &str = "rfmetrics";

/// Rotation bound: the current file rolls at 5 MB, keeping 4 rotated
/// siblings (~25 MB worst case). Info-level volume makes one file last
/// months; without this the single log grows forever.
const MAX_LOG_BYTES: u64 = 5_000_000;
const KEEP_ROTATED: usize = 4;

/// Logger handle for an explicit flush at shutdown (a forgotten handle
/// leaves the buffered tail to the OS instead).
static HANDLE: OnceLock<LoggerHandle> = OnceLock::new();

/// Directory for the log file: first writable of exe dir → cwd → temp,
/// so a read-only install still logs instead of failing silently.
fn log_dir() -> PathBuf {
    crate::binaries::writable_app_dir()
}

/// Initialize file-only logging once per process. Safe to call from tests:
/// a second init is a no-op instead of a panic.
pub fn init_logging() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let dir = log_dir();
        let spec = FileSpec::default()
            .directory(&dir)
            .basename(BASENAME)
            .suffix("log")
            .suppress_timestamp();
        match Logger::try_with_str("info").and_then(|l| {
            l.log_to_file(spec)
                .rotate(
                    Criterion::Size(MAX_LOG_BYTES),
                    Naming::Numbers,
                    Cleanup::KeepLogFiles(KEEP_ROTATED),
                )
                .append()
                .start()
        }) {
            Ok(handle) => {
                if HANDLE.set(handle).is_err() {
                    log::warn!("logging already initialized; discarding duplicate handle");
                }
                log::info!(
                    "RFMetrics {} logging to {}",
                    env!("CARGO_PKG_VERSION"),
                    dir.join(format!("{BASENAME}.log")).display()
                );
            }
            Err(e) => {
                eprintln!("logging init failed ({e}); continuing without log file");
            }
        }
    });
}

/// Flush buffered log lines; safe before init (no-op). Called from
/// `on_exit` so the shutdown tail reaches disk.
pub fn flush_logging() {
    if let Some(handle) = HANDLE.get() {
        handle.flush();
    }
}
#[cfg(test)]
#[path = "tests/test_logging.rs"]
mod tests;
