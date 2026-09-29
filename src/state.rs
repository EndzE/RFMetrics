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
    pub ref_pixfmt: Option<String>,
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
    pub ref_path: String,
    pub skip: String,
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

/// Every place the state file may live, exe-first (parity order): load
/// reads the first one that exists, save writes the first writable one.
fn candidate_paths() -> Vec<PathBuf> {
    crate::binaries::candidate_dirs()
        .into_iter()
        .map(|d| d.join(FILE_NAME))
        .collect()
}

/// One section of a parsed state object: deserialize or fall back to
/// the live default, warning so hand-edit typos stay diagnosable.
fn section<T>(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> T
where
    T: serde::de::DeserializeOwned + Default,
{
    match obj.get(key) {
        None => T::default(),
        Some(v) => serde_json::from_value(v.clone()).unwrap_or_else(|e| {
            log::warn!(target: "rfmetrics::state", "ignoring bad section {key:?}: {e}");
            T::default()
        }),
    }
}

/// Per-section salvage of a state file: a single type mismatch (a null
/// in `files`, a mistyped toggle) drops only its own section instead of
/// failing the whole `AppState` parse. `None` = not JSON / not an object.
fn salvage(text: &str) -> Option<AppState> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = value.as_object()?;
    let files: Option<Vec<FileEntry>> = match obj.get("files") {
        None => None,
        Some(v) => Option::<Vec<FileEntryRaw>>::deserialize(v.clone())
            .map(|opt| opt.map(|list| list.into_iter().map(FileEntry::from).collect()))
            .unwrap_or_else(|e| {
                log::warn!(target: "rfmetrics::state", "ignoring bad section \"files\": {e}");
                None
            }),
    };
    Some(AppState {
        ref_path: section(obj, "ref_path"),
        skip: section(obj, "skip"),
        duration: section(obj, "duration"),
        files,
        metrics: section(obj, "metrics"),
        vmaf: section(obj, "vmaf"),
        options: section(obj, "options"),
    })
}

/// Rename a corrupt candidate aside so the next save doesn't destroy
/// evidence (best-effort: a read-only dir just logs).
fn quarantine(path: &PathBuf) {
    let mut bak = path.as_os_str().to_owned();
    bak.push(".bak");
    let bak = PathBuf::from(bak);
    match std::fs::rename(path, &bak) {
        Ok(()) => {
            log::warn!(target: "rfmetrics::state", "moved corrupt {} to {}", path.display(), bak.display())
        }
        Err(e) => {
            log::warn!(target: "rfmetrics::state", "cannot quarantine corrupt {}: {e}", path.display())
        }
    }
}

/// Tolerant load over explicit candidates: missing files are skipped, a
/// fully valid file loads as-is, anything else is salvaged per section
/// (or skipped) and quarantined to `.bak` so it survives the next save.
fn load_from(paths: &[PathBuf]) -> Option<AppState> {
    for path in paths {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        // Fast path: clean files keep their exact shape, no rename.
        if let Ok(s) = serde_json::from_str::<AppState>(&text) {
            log::info!(target: "rfmetrics::state", "loaded {}", path.display());
            return Some(s);
        }
        match salvage(&text) {
            Some(s) => {
                log::warn!(target: "rfmetrics::state", "loaded partial {}", path.display());
                quarantine(path);
                return Some(s);
            }
            None => {
                log::warn!(target: "rfmetrics::state", "ignoring corrupt {}", path.display());
                quarantine(path);
            }
        }
    }
    log::debug!(target: "rfmetrics::state", "no state file found");
    None
}

/// Tolerant load: missing/corrupt/non-object files behave as no file.
/// Searches exe dir → cwd → temp so a fallback save is found again.
pub fn load() -> Option<AppState> {
    load_from(&candidate_paths())
}

/// Atomic save (tmp + rename, Python parity); failures are logged, never
/// toasted — losing UI persistence must not interrupt a run.
/// Writes the first writable candidate (exe → cwd → temp) and returns
/// where it landed (`None` when everywhere failed, so the caller can
/// toast once instead of losing persistence silently).
pub fn save(state: &AppState) -> Option<PathBuf> {
    let text = match serde_json::to_string_pretty(state) {
        Ok(t) => t,
        Err(e) => {
            log::warn!(target: "rfmetrics::state", "serialize failed: {e}");
            return None;
        }
    };
    for path in candidate_paths() {
        if write_atomic(&text, &path) {
            if path != state_path() {
                log::warn!(target: "rfmetrics::state", "saved fallback {}", path.display());
            } else {
                log::debug!(target: "rfmetrics::state", "saved {}", path.display());
            }
            return Some(path);
        }
    }
    log::warn!(target: "rfmetrics::state", "save failed in every candidate dir");
    None
}

fn write_atomic(text: &str, path: &PathBuf) -> bool {
    let tmp = path.with_extension(format!("json{TMP_SUFFIX}"));
    if std::fs::write(&tmp, text)
        .and_then(|()| std::fs::rename(&tmp, path))
        .is_err()
    {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

#[cfg(test)]
fn save_to(state: &AppState, path: &PathBuf) {
    let text = match serde_json::to_string_pretty(state) {
        Ok(t) => t,
        Err(e) => {
            log::warn!(target: "rfmetrics::state", "serialize failed: {e}");
            return;
        }
    };
    if !write_atomic(&text, path) {
        log::warn!(target: "rfmetrics::state", "save {} failed", path.display());
    }
}
#[cfg(test)]
#[path = "tests/test_state.rs"]
mod tests;
