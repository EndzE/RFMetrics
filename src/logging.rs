use std::path::PathBuf;

use flexi_logger::{FileSpec, Logger};

/// Log file basename next to the executable: `<exe_dir>/rfmetrics.log`.
const BASENAME: &str = "rfmetrics";

/// Directory for the log file: next to the exe, falling back to the
/// current dir and then the temp dir when the exe dir is unresolvable.
fn log_dir() -> PathBuf {
    if let Some(dir) = crate::binaries::exe_dir() {
        return dir;
    }
    std::env::current_dir()
        .ok()
        .unwrap_or_else(std::env::temp_dir)
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
