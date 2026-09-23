//! `ffmetrics-state.json` persistence (Python `_load_state`/`_save_state`
//! parity): ref path, trim boxes, queue paths + include flags, header
//! toggles, VMAF options. Metric results, row selection, and window
//! geometry are session state and never touch disk.
//!
//! Every persisted section is optional on load: a corrupt file behaves as
//! no file, and a missing key keeps the live default (so hand-edited or
//! Python-written files degrade gracefully — Python's `files` string list
//! loads as included rows).

use std::path::PathBuf;

use serde::Deserialize as _;

/// Debounce: a detected change is written this long after it settles.
pub const SAVE_DEBOUNCE_SECS: f64 = 1.0;

/// State file name next to the exe (Python `_STATE_FILE` parity).
const FILE_NAME: &str = "rfmetrics-state.json";

/// Temp suffix for atomic writes (Python `.json.tmp` parity).
const TMP_SUFFIX: &str = ".tmp";

fn default_true() -> bool {
    true
}

/// One queue row on disk: path + first-column include flag (an rfmetrics
/// extension — Python saves paths only).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    pub path: String,
    #[serde(default = "default_true")]
    pub include: bool,
}

/// Python shape (`"files": ["a.mp4"]`) vs ours (`[{path, include}]`).
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum FileEntryRaw {
    Path(String),
    Full(FileEntry),
}

impl From<FileEntryRaw> for FileEntry {
    fn from(raw: FileEntryRaw) -> Self {
        match raw {
            FileEntryRaw::Path(path) => Self {
                path,
                include: true,
            },
            FileEntryRaw::Full(e) => e,
        }
    }
}

fn de_files<'de, D>(d: D) -> Result<Option<Vec<FileEntry>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<Vec<FileEntryRaw>>::deserialize(d)
        .map(|opt| opt.map(|v| v.into_iter().map(FileEntry::from).collect()))
}

/// Header toggles; `None` = key absent, keep the live default.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MetricsState {
    pub psnr: Option<bool>,
    pub ssim: Option<bool>,
    pub vmaf: Option<bool>,
    pub xpsnr: Option<bool>,
    pub ssim2: Option<bool>,
    pub butteraugli: Option<bool>,
    pub cvvdp: Option<bool>,
}

/// VMAF options; `None` = key absent, keep the live default.
/// `threads` is rfmetrics-only (no Python key).
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VmafState {
    pub model: Option<String>,
    pub phone: Option<bool>,
    pub scale: Option<bool>,
    pub pooling: Option<String>,
    pub subsample: Option<String>,
    pub threads: Option<String>,
}

/// Global options; `None` = key absent, keep the live default.
/// `scaling`/`plot_size` are rfmetrics-only UI labels (no Python keys).
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OptionsState {
    pub scaling: Option<String>,
    pub fps_mode: Option<String>,
    pub cell_stat: Option<String>,
    pub cell_precision: Option<String>,
    pub plot_at_start: Option<bool>,
    pub plot_size: Option<String>,
    pub csv_export: Option<bool>,
    pub csv_dir: Option<String>,
    pub badframes_count: Option<String>,
    pub badframes_export_dir: Option<String>,
    pub results_autosave: Option<bool>,
    pub results_path: Option<String>,
}

/// The whole persisted snapshot: verbatim boxes, optional queue, and
/// optional per-key overrides.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AppState {
    #[serde(default)]
    pub ref_path: String,
    #[serde(default)]
    pub skip: String,
    #[serde(default)]
    pub duration: String,
    #[serde(deserialize_with = "de_files", default)]
    pub files: Option<Vec<FileEntry>>,
    pub metrics: MetricsState,
    pub vmaf: VmafState,
    pub options: OptionsState,
}

/// State file location: next to the exe (Python `app_dir` parity),
/// falling back like the logger when unresolvable.
pub fn state_path() -> PathBuf {
    crate::binaries::app_dir().join(FILE_NAME)
}

/// Tolerant load: missing/corrupt/non-object files behave as no file.
pub fn load() -> Option<AppState> {
    let path = state_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            log::debug!(target: "rfmetrics::state", "no state file {}: {e}", path.display());
            return None;
        }
    };
    match serde_json::from_str::<AppState>(&text) {
        Ok(s) => {
            log::info!(target: "rfmetrics::state", "loaded {}", path.display());
            Some(s)
        }
        Err(e) => {
            log::warn!(target: "rfmetrics::state", "ignoring corrupt {}: {e}", path.display());
            None
        }
    }
}

/// Atomic save (tmp + rename, Python parity); failures are logged, never
/// toasted — losing UI persistence must not interrupt a run.
pub fn save(state: &AppState) {
    save_to(state, &state_path());
}

fn save_to(state: &AppState, path: &PathBuf) {
    let text = match serde_json::to_string_pretty(state) {
        Ok(t) => t,
        Err(e) => {
            log::warn!(target: "rfmetrics::state", "serialize failed: {e}");
            return;
        }
    };
    let tmp = path.with_extension(format!("json{TMP_SUFFIX}"));
    if let Err(e) = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        log::warn!(target: "rfmetrics::state", "save {} failed: {e}", path.display());
        return;
    }
    log::debug!(target: "rfmetrics::state", "saved {}", path.display());
}
#[cfg(test)]
#[path = "tests/test_state.rs"]
mod tests;
