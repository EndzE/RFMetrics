//! Persisted run configuration for `RFMetricsApp`.
//!
//! Everything here round-trips through `ffmetrics-state.json` (see
//! `persist.rs` + `crate::state`), except `VmafOpts::models`, which is a
//! read-only discovery catalog. All other app state (queue internals,
//! worker channels, viewer textures) is session-only and lives in the
//! domain sub-structs next to the code that drives it.

use crate::metrics::CellStat;
use crate::metrics::ffmpeg::{InputFpsMode, RefPixFmt, ScaleMethod};
use crate::plot::PlotSize;

/// Reference file + trim boxes (Skip-row inputs).
pub(crate) struct ReferenceInputs {
    pub(crate) path: String,
    pub(crate) duration: String,
    pub(crate) skip: String,
    /// Pixel format both legs converge on (Skip-row combobox, default No
    /// conversion = legacy dist→ref-native legs). Run input: locked
    /// mid-run, stamped onto `Done` cells like `scaler`.
    pub(crate) pixfmt: RefPixFmt,
}

/// Header metric toggles (queue table checkboxes).
pub(crate) struct MetricToggles {
    pub(crate) psnr: bool,
    pub(crate) ssim: bool,
    pub(crate) vmaf: bool,
    pub(crate) xpsnr: bool,
    pub(crate) ssim2: bool,
    pub(crate) butteraugli: bool,
    pub(crate) cvvdp: bool,
}

/// VMAF options (Options panel). `models` is the discovered catalog
/// (read-only, never snapshotted); the rest persist per key.
pub(crate) struct VmafOpts {
    pub(crate) model: String,
    pub(crate) phone: bool,
    pub(crate) scale: bool,
    pub(crate) pooling: String,
    pub(crate) subsample: String,
    pub(crate) threads: String,
    pub(crate) models: Vec<String>,
}

/// Display/run-shape options (Options panel).
pub(crate) struct ViewOpts {
    /// Global scaling method for every `scale=` the app emits.
    pub(crate) scale_method: ScaleMethod,
    /// Input framerate mode for every `-i` the app emits (FFMetrics #111).
    pub(crate) fps_mode: InputFpsMode,
    /// Which `DoneStats` stat metric cells display, sort by, and copy
    /// (Options combobox, default Avg).
    pub(crate) cell_stat: CellStat,
    /// Decimals for metric cell display + Copy value (Options combobox,
    /// default 4). Frozen Avg texts re-freeze on change.
    pub(crate) cell_precision: u8,
    /// Save PNG / Copy image size preset (Options combobox).
    pub(crate) plot_size: PlotSize,
    /// Open the plot viewport when a run starts (Options checkbox).
    pub(crate) plot_at_start: bool,
}

/// Export options (Options panel).
pub(crate) struct ExportOpts {
    /// Save per-frame metric CSVs on Done (Options checkbox).
    pub(crate) csv_export: bool,
    /// CSV output folder; empty = beside the distorted file (Options).
    pub(crate) csv_dir: String,
    /// Append results rows to the results file when each run ends.
    pub(crate) results_autosave: bool,
    /// Results file path; empty = `RFMetrics.Results.csv` next to the exe.
    pub(crate) results_path: String,
    /// Worst frames saved per metric/file by Extract bad frames (Options
    /// combobox, original `BadFrames.Count` parity, default 5).
    pub(crate) badframes_count: String,
    /// Bad-frames export folder; empty = beside each distorted file.
    pub(crate) badframes_export_dir: String,
}

/// Everything `ffmetrics-state.json` persists, grouped by Options panel
/// section. Mirrors `crate::state::AppState` via `persist.rs`.
pub(crate) struct Config {
    pub(crate) reference: ReferenceInputs,
    pub(crate) metrics: MetricToggles,
    pub(crate) vmaf: VmafOpts,
    pub(crate) view: ViewOpts,
    pub(crate) export: ExportOpts,
}

/// Parse a trim box; empty means no trim. `None` = invalid ("bad time").
pub(crate) fn trim_opt(raw: &str) -> Option<Option<f64>> {
    if raw.trim().is_empty() {
        Some(None)
    } else {
        crate::metrics::parse_time_spec(raw).map(Some)
    }
}

impl VmafOpts {
    /// Validated VMAF settings snapshot (Python `vmaf_cfg`): subsample
    /// parses to u32 with max(1, …), pooling maps the UI strings to the
    /// enum. Single source for `start_run` and the stale-cell badge so
    /// the two can never disagree on what "current settings" means.
    pub(crate) fn current_vmaf_cfg(&self) -> crate::metrics::vmaf::VmafCfg {
        crate::metrics::vmaf::VmafCfg {
            model: self.model.clone(),
            phone: self.phone,
            scale: self.scale,
            pooling: if self.pooling == "Harmonic Mean" {
                crate::metrics::vmaf::Pooling::HarmonicMean
            } else {
                crate::metrics::vmaf::Pooling::Mean
            },
            subsample: self.subsample.parse::<u32>().unwrap_or(1).max(1),
            // "auto" (or garbage) follows the system CPU, as before.
            n_threads: match self.threads.parse::<u32>() {
                Ok(n) => n.max(1),
                Err(_) => crate::metrics::vmaf::system_threads(),
            },
        }
    }
}
