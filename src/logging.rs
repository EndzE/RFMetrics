use std::path::PathBuf;

use flexi_logger::{FileSpec, Logger};

/// Log file basename next to the executable: `<exe_dir>/rfmetrics.log`.
const BASENAME: &str = "rfmetrics";

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
        match Logger::try_with_str("info").and_then(|l| l.log_to_file(spec).append().start()) {
            Ok(_handle) => {
                // ponytail: leak handle; dropping it would flush + close the file
                std::mem::forget(_handle);
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
#[cfg(test)]
#[path = "tests/test_logging.rs"]
mod tests;
