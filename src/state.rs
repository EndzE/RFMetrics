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
/// `scaling` is an rfmetrics-only UI label (no Python key).
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OptionsState {
    pub scaling: Option<String>,
    pub plot_at_start: Option<bool>,
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
    let dir = crate::binaries::exe_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(std::env::temp_dir);
    dir.join(FILE_NAME)
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
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s = AppState {
            ref_path: "C:/vids/ref.mp4".to_owned(),
            skip: "5".to_owned(),
            duration: "00:10".to_owned(),
            files: Some(vec![
                FileEntry {
                    path: "C:/vids/a.mp4".to_owned(),
                    include: true,
                },
                FileEntry {
                    path: "C:/vids/b.mp4".to_owned(),
                    include: false,
                },
            ]),
            metrics: MetricsState {
                psnr: Some(true),
                vmaf: Some(false),
                ..Default::default()
            },
            vmaf: VmafState {
                model: Some("vmaf_v0.6.1.json".to_owned()),
                threads: Some("auto".to_owned()),
                ..Default::default()
            },
            options: OptionsState {
                scaling: Some("Bicubic".to_owned()),
                plot_at_start: Some(true),
            },
        };
        let back: AppState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn python_shaped_files_load_included() {
        let s: AppState = serde_json::from_str(
            r#"{"ref": "x", "ref_path": "C:/r.mp4", "files": ["C:/a.mp4", {"path": "C:/b.mp4", "include": false}]}"#,
        )
        .unwrap();
        // Unknown keys ("ref") are ignored; strings load as included rows.
        assert_eq!(s.ref_path, "C:/r.mp4");
        assert_eq!(
            s.files,
            Some(vec![
                FileEntry {
                    path: "C:/a.mp4".to_owned(),
                    include: true,
                },
                FileEntry {
                    path: "C:/b.mp4".to_owned(),
                    include: false,
                },
            ])
        );
        // Absent sections stay None so live defaults survive.
        assert_eq!(s.metrics, MetricsState::default());
        assert_eq!(s.vmaf, VmafState::default());
        assert_eq!(s.options, OptionsState::default());
    }

    #[test]
    fn options_scaling_round_trips_and_rejects_unknown() {
        let s: AppState = serde_json::from_str(r#"{"options": {"scaling": "Lanczos"}}"#).unwrap();
        assert_eq!(
            s.options.scaling,
            Some("Lanczos".to_owned()),
            "label persists verbatim; from_label validates on apply"
        );
        // Unknown section keys are ignored, like Python's unknown keys.
        let s: AppState =
            serde_json::from_str(r#"{"options": {"scaling": "Lanczos", "other": 1}}"#).unwrap();
        assert_eq!(s.options.scaling, Some("Lanczos".to_owned()));
    }

    #[test]
    fn tmp_suffix_targets_state_file() {
        let tmp = PathBuf::from("ffmetrics-state.json").with_extension(format!("json{TMP_SUFFIX}"));
        assert_eq!(tmp.to_string_lossy(), "ffmetrics-state.json.tmp");
    }

    #[test]
    fn save_round_trip_is_atomic() {
        let dir = std::env::temp_dir().join(format!("rfmetrics-state-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ffmetrics-state.json");
        let s = AppState {
            ref_path: "C:/r.mp4".to_owned(),
            metrics: MetricsState {
                vmaf: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        save_to(&s, &path);
        let back: AppState =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(back, s);
        // No temp file left behind.
        assert!(!path.with_extension(format!("json{TMP_SUFFIX}")).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
