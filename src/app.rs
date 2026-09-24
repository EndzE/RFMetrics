pub(crate) use crate::app_queue::{
    DropAction, METRIC_COLUMNS, QueueRow, SortColumn, SortDir, VIDEO_EXTS, apply_alt_include,
    cycle_sort, done_is_stale, norm_key, reveal_in_explorer, route_drop, shift_include_range,
    sort_view,
};
use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

// Queue domain lives in `crate::app_queue` (re-exported above).

/// Retrieves the cursor position in egui's logical point coordinates.
/// During Windows OLE file drags, winit omits pointer move events, so egui's
/// internal pointer state is None/stale. We query the OS cursor directly.
#[cfg(windows)]
pub(crate) fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    #[repr(C)]
    struct Point {
        x: i32,
        y: i32,
    }
    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetCursorPos(lpPoint: *mut Point) -> i32;
    }

    let mut pt = Point { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut pt) } != 0 {
        let ppp = ctx.pixels_per_point();
        let screen_pos = egui::pos2(pt.x as f32 / ppp, pt.y as f32 / ppp);
        if let Some(inner_rect) = ctx.input(|i| i.viewport().inner_rect) {
            return Some(egui::pos2(
                screen_pos.x - inner_rect.min.x,
                screen_pos.y - inner_rect.min.y,
            ));
        }
    }
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

#[cfg(not(windows))]
pub(crate) fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

/// Hover highlight delay (seconds) so passing over rows while aiming
/// at text to copy doesn't flash each row.
pub(crate) const ROW_HOVER_DELAY: f64 = 0.1;

/// How long the drop toast (e.g. ignored extra reference files) stays up.
pub(crate) const TOAST_SECS: f64 = 3.0;

/// Toast severity; drives the outline color. `Info` keeps the default
/// popup outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToastKind {
    Info,
    Warning,
    Error,
}

impl ToastKind {
    pub(crate) fn outline(self) -> Option<egui::Color32> {
        match self {
            ToastKind::Info => None,
            ToastKind::Warning => Some(egui::Color32::from_rgb(0xD9, 0xA4, 0x06)),
            ToastKind::Error => Some(egui::Color32::from_rgb(0xE0, 0x4B, 0x4B)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Toast {
    pub(crate) text: String,
    pub(crate) until: f64,
    pub(crate) kind: ToastKind,
}

// Worker message + plan types live in their domain modules (re-exported
// so `super::ProbeMsg` etc. in child tests keeps working).
pub(crate) use crate::app_badframes::{BadframeExportPending, BadframeMsg};
pub(crate) use crate::app_plots::{PlotExport, PngSaveMsg};
pub(crate) use crate::app_run::{BinProbe, MetricMsg, ProbeMsg, ThumbMsg, wall_now_string};

pub struct RFMetricsApp {
    pub(crate) ref_path: String,
    pub(crate) duration: String,
    pub(crate) skip: String,
    pub(crate) m_psnr: bool,
    pub(crate) m_ssim: bool,
    pub(crate) m_vmaf: bool,
    pub(crate) m_xpsnr: bool,
    pub(crate) m_ssim2: bool,
    pub(crate) m_but: bool,
    pub(crate) m_cvvdp: bool,
    pub(crate) vmaf_model: String,
    pub(crate) vmaf_phone: bool,
    pub(crate) vmaf_scale: bool,
    pub(crate) vmaf_pooling: String,
    pub(crate) vmaf_subsample: String,
    pub(crate) vmaf_threads: String,
    pub(crate) vmaf_models: Vec<String>,
    /// Global scaling method for every `scale=` the app emits.
    pub(crate) scale_method: ScaleMethod,
    /// Input framerate mode for every `-i` the app emits (FFMetrics #111).
    pub(crate) fps_mode: crate::metrics::ffmpeg::InputFpsMode,
    /// Which `DoneStats` stat metric cells display, sort by, and copy
    /// (Options combobox, default Avg).
    pub(crate) cell_stat: crate::metrics::CellStat,
    /// Decimals for metric cell display + Copy value (Options combobox,
    /// default 4). Frozen Avg texts re-freeze on change (see below).
    pub(crate) cell_precision: u8,
    /// Pixel format both legs converge on (Skip-row combobox, default No
    /// conversion = legacy dist→ref-native legs). Run input: locked
    /// mid-run, stamped onto `Done` cells like `scaler`.
    pub(crate) ref_pixfmt: crate::metrics::ffmpeg::RefPixFmt,
    /// Open the plot viewport when a run starts (Options checkbox).
    pub(crate) plot_at_start: bool,
    /// Save per-frame metric CSVs on Done (Options checkbox).
    pub(crate) csv_export: bool,
    /// CSV output folder; empty = beside the distorted file (Options).
    pub(crate) csv_dir: String,
    /// Append results rows to the results file when each run ends.
    pub(crate) results_autosave: bool,
    /// Results file path; empty = `RFMetrics.Results.csv` next to the exe.
    pub(crate) results_path: String,
    /// Save PNG / Copy image size preset (Options combobox).
    pub(crate) plot_size: crate::plot::PlotSize,
    /// Worst frames saved per metric/file by Extract bad frames (Options
    /// combobox, original `BadFrames.Count` parity, default 5).
    pub(crate) badframes_count: String,
    /// Bad-frames export folder; empty = beside each distorted file.
    pub(crate) badframes_export_dir: String,
    pub(crate) rows: Vec<QueueRow>,
    pub(crate) ffmpeg: crate::binaries::BinaryInfo,
    pub(crate) ffvship: crate::binaries::BinaryInfo,
    pub(crate) ffprobe: Option<std::path::PathBuf>,
    pub(crate) ref_info: String,
    /// Probed reference stream; feeds metric filtergraphs (scale/format).
    pub(crate) ref_info_data: Option<crate::probe::MediaInfo>,
    pub(crate) ref_rect: Option<egui::Rect>,
    pub(crate) table_rect: Option<egui::Rect>,
    pub(crate) hover_row: Option<usize>,
    pub(crate) hover_since: Option<f64>,
    pub(crate) hovered_now: Option<usize>,
    /// Shift+click range anchor: last clicked include-checkbox row
    /// (session-only, like hover/selection — never persisted).
    pub(crate) include_anchor: Option<usize>,
    /// Shift+click range anchor: last free-space-clicked row for the
    /// `selected` removal set (session-only, never persisted).
    pub(crate) selected_anchor: Option<usize>,
    /// Active table sort, if any (session-only, never persisted — the
    /// state file and run order always keep insertion order).
    pub(crate) sort_spec: Option<(SortColumn, SortDir)>,
    pub(crate) toast: Option<Toast>,
    /// A probe worker hit its timeout; the drain records the display name
    /// here and the update loop toasts it once (single slot, like `toast`).
    /// Session-only, never persisted.
    pub(crate) probe_timeout_note: Option<String>,
    /// Pending CSV summary, set by the CsvReport drain arm and toasted
    /// with a real timestamp at the next UI frame (drain has none).
    /// `(files_written, error_strings)`.
    pub(crate) csv_report: Option<(usize, Vec<String>)>,
    /// Results auto-save owed: set by the Finished drain arm when the
    /// option is on (aborted runs included), consumed with a timestamp
    /// at the next UI frame like `csv_report` above.
    pub(crate) results_autosave_pending: bool,
    pub(crate) probe_tx: Sender<ProbeMsg>,
    pub(crate) probe_rx: Receiver<ProbeMsg>,
    /// Startup binary probe channel: one `(ffmpeg, ffvship, ffprobe)`
    /// tuple lands after the version probes finish off the UI thread.
    /// `bins_probing` gates `start_run` with an accurate toast meanwhile
    /// (instead of a misleading "not found").
    pub(crate) bin_rx: Receiver<BinProbe>,
    pub(crate) bins_probing: bool,
    pub(crate) metric_tx: Sender<MetricMsg>,
    pub(crate) metric_rx: Receiver<MetricMsg>,
    /// True while the metric worker runs; the button flips Start↔Stop then.
    pub(crate) measuring: bool,
    /// Stop flag shared with the worker (checked between jobs + in `run_psnr`).
    pub(crate) abort: Arc<AtomicBool>,
    /// The live ffmpeg child, so Stop can kill the in-flight run.
    pub(crate) current_child: Arc<Mutex<Option<std::process::Child>>>,
    /// Jobs still Blocking; `Finished` alone clears `measuring` (the
    /// last `Done` must not: `Finished`/`CsvReport` still follow it).
    pub(crate) pending: usize,
    /// Next plot-color slot; bumped per queued file, never reused.
    pub(crate) next_color_idx: usize,
    /// Next row-probe token; bumped per queued file, never reused.
    pub(crate) next_probe_seq: u64,
    /// Bumped per run; late worker messages after a Reset are stale.
    pub(crate) run_generation: u64,
    /// Path last handed to a probe worker (or resolved cheaply without one).
    pub(crate) last_spawned_ref: String,
    /// Path the current `ref_info_data` was probed from (set when its
    /// worker result lands). The thumbnail worker reuses its duration only
    /// on a match — `ref_info_data` alone lags one probe behind on ref
    /// change and can't say which path it belongs to.
    pub(crate) ref_info_path: String,
    /// Bumped on every ref change; worker results with an older generation
    /// are stale (typed-through) and discarded.
    pub(crate) ref_generation: u64,
    pub(crate) thumb_tx: Sender<ThumbMsg>,
    pub(crate) thumb_rx: Receiver<ThumbMsg>,
    pub(crate) thumb_tex: Option<egui::TextureHandle>,
    pub(crate) thumb_loading: bool,
    pub(crate) last_thumb_path: String,
    pub(crate) thumb_generation: u64,
    /// Last state actually written to `ffmetrics-state.json`; the per-frame
    /// snapshot compares against this so only real changes arm a write.
    pub(crate) saved_snapshot: crate::state::AppState,
    /// Egui time of the first unsaved change (`None` = clean).
    pub(crate) pending_save_since: Option<f64>,
    /// One-shot: the state-fallback/failure toast already fired, so a
    /// read-only install toasts once instead of every debounce.
    pub(crate) state_fallback_toasted: bool,
    /// Metric kind of the currently executing job (last kind seen on the
    /// Progress/Series feed); drives plot tab-follow while measuring.
    pub(crate) live_kind: Option<MetricKind>,
    /// Queue key (`QueueRow::key`) of the currently executing job (last
    /// key seen on the Progress/Series feed, cleared when its `Done`
    /// lands). The worker runs jobs sequentially but every queued cell
    /// is marked `Running` upfront, so the sweep animates only this
    /// cell — the rest stay static until their turn.
    pub(crate) live_key: Option<String>,
    /// PSNR plot viewport open (Python `plot["win"]` parity: closing the
    /// window withdraws it, Plot reopens it).
    pub(crate) show_plot: bool,
    /// Selected plot viewport tab (session-only, like the Python window).
    pub(crate) plot_tab: MetricKind,
    /// Last measured tab-strip box width, for centering the strip
    /// (session-only; texts are static so it converges in one frame).
    pub(crate) plot_tabs_w: f32,
    /// Grow-only live fit per open tab while any series is running;
    /// cleared once all settle, so finished graphs fit exactly again.
    pub(crate) plot_live_fit: Option<(MetricKind, crate::plot::FitBounds)>,
    /// Follow poke still owed: set when a live phase starts without plot
    /// memory present (window just opened), retried until it lands.
    pub(crate) plot_follow_pending: bool,
    /// Reset-view click still owed: the help-bar `Ui` scopes persistent
    /// ids differently than the canvas `Ui`, so the button only arms this
    /// flag and the central panel (plot id scope) executes the poke.
    /// Retried until plot memory exists, like the follow poke.
    pub(crate) plot_reset_pending: bool,
    /// Snap-to-data lock (plot window checkbox, session-only): panning is
    /// clamped to the first/last frame on x and the plotted min/max on y;
    /// zooming and in-limits panning stay free.
    pub(crate) plot_snap: bool,
    /// Pending plot export (Save PNG / Copy button), executed in the
    /// central panel where the plot id scope lives.
    pub(crate) plot_save_pending: Option<PlotExport>,
    /// Plot export worker channel + busy flag: while `png_saving` both the
    /// Save PNG and Copy buttons are disabled so 5 s renders can't overlap.
    pub(crate) png_tx: Sender<PngSaveMsg>,
    pub(crate) png_rx: Receiver<PngSaveMsg>,
    pub(crate) png_saving: bool,
    /// Bad-frames worker channel + state: `badframes_busy` while accurate
    /// seeks run, `badframe_done/total` for the button label, `badframe_abort`
    /// for Stop-between-frames (mid-seek ffmpeg is bounded by
    /// `BADFRAME_TIMEOUT`, so no child kill needed).
    pub(crate) badframe_tx: Sender<BadframeMsg>,
    pub(crate) badframe_rx: Receiver<BadframeMsg>,
    pub(crate) badframes_busy: bool,
    pub(crate) badframe_done: usize,
    pub(crate) badframe_total: usize,
    pub(crate) badframe_abort: Arc<AtomicBool>,
    /// Pending bad-frames summary, toasted at the next UI frame like
    /// `csv_report` above. `(files_written, error_strings)`.
    pub(crate) badframe_report: Option<(usize, Vec<String>)>,
    /// Bad-frames viewer window (own OS viewport like the plot window).
    /// Frames live as tmp PNGs (`badframe_tmp`); only the visible dist/ref
    /// pair is uploaded as textures, keyed by `badframe_tex_key`.
    pub(crate) show_badframes: bool,
    pub(crate) badframe_tab: MetricKind,
    /// Selected queue-row key for the viewer (None = auto-pick first).
    pub(crate) badframe_file: Option<String>,
    /// Position in the worst-N list for `(file, tab)`.
    pub(crate) badframe_frame_pos: usize,
    /// One-frame view reset for the viewer plots (Reset view button /
    /// selection change): applies `Plot::reset()`, which also clears the
    /// shared link-group bounds a fresh plot id alone would keep.
    pub(crate) badframe_reset_once: bool,
    /// Last selection the viewer plots were fit for; a change arms
    /// `badframe_reset_once` so every tab/file/frame lands fit.
    pub(crate) badframe_view_key: Option<(MetricKind, String, usize)>,
    /// Overlay compare mode: false = side-by-side plots (default),
    /// true = single wipe view with a draggable divider.
    pub(crate) badframe_slider: bool,
    /// Wipe divider fraction (0..1, ref on the left). Drag-only.
    pub(crate) badframe_split: f32,
    /// Last-frame divider screen x for pre-show pan suppression
    /// (NaN until the wipe plot paints once).
    pub(crate) badframe_div_sx: f32,
    /// Divider drag in progress: keeps plot pan off while held.
    pub(crate) badframe_div_drag: bool,
    pub(crate) badframe_tex_dist: Option<egui::TextureHandle>,
    pub(crate) badframe_tex_ref: Option<egui::TextureHandle>,
    pub(crate) badframe_tex_key: Option<(String, MetricKind, usize)>,
    /// Tmp dir holding this run's viewer PNGs (per-process).
    pub(crate) badframe_tmp: std::path::PathBuf,
    /// Close requested while a bad-frames worker runs: tmp deletion waits
    /// for its `Finished` drain (the worker reads/writes tmp until then).
    /// Session-only, never persisted.
    pub(crate) badframe_tmp_cleanup_pending: bool,
    /// All tmp PNGs from the last Extract run.
    pub(crate) badframe_files: Vec<std::path::PathBuf>,
    /// In-flight export summary (copies done, worker PNGs pending).
    /// Session-only, never persisted.
    pub(crate) badframe_export_pending: Option<BadframeExportPending>,
}

impl Default for RFMetricsApp {
    fn default() -> Self {
        // Version probes can each block up to `VERSION_TIMEOUT` (wedged
        // exe, AV stall): run them on a worker, never the UI thread.
        // Tests stay synchronous + hermetic (no host timing in asserts).
        let (ffmpeg, ffvship, ffprobe, bin_rx, bins_probing) = if cfg!(test) {
            let ffmpeg = crate::binaries::ffmpeg_info();
            let ffvship = crate::binaries::ffvship_info();
            let ffprobe = crate::binaries::ffprobe_path(ffmpeg.path.as_deref());
            let (_, bin_rx) = std::sync::mpsc::channel();
            (ffmpeg, ffvship, ffprobe, bin_rx, false)
        } else {
            let (bin_tx, bin_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let ffmpeg = crate::binaries::ffmpeg_info();
                let ffvship = crate::binaries::ffvship_info();
                let ffprobe = crate::binaries::ffprobe_path(ffmpeg.path.as_deref());
                let _ = bin_tx.send((ffmpeg, ffvship, ffprobe));
            });
            (
                crate::binaries::BinaryInfo::probing("Probing for ffmpeg…"),
                crate::binaries::BinaryInfo::probing("Probing for FFVship…"),
                None,
                bin_rx,
                true,
            )
        };
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (metric_tx, metric_rx) = std::sync::mpsc::channel();
        let (png_tx, png_rx) = std::sync::mpsc::channel();
        let (badframe_tx, badframe_rx) = std::sync::mpsc::channel();
        let mut app = Self {
            ref_path: String::new(),
            duration: String::new(),
            skip: String::new(),
            m_psnr: false,
            m_ssim: false,
            m_vmaf: true,
            m_xpsnr: false,
            m_ssim2: false,
            m_but: false,
            m_cvvdp: false,
            vmaf_model: "vmaf_v0.6.1.json".to_owned(),
            vmaf_phone: false,
            vmaf_scale: false,
            vmaf_pooling: "Mean".to_owned(),
            vmaf_subsample: "1".to_owned(),
            vmaf_threads: "auto".to_owned(),
            vmaf_models: crate::metrics::vmaf::list_models(
                &crate::metrics::vmaf::vmaf_home().join("vmaf-models"),
            ),
            scale_method: ScaleMethod::default(),
            fps_mode: crate::metrics::ffmpeg::InputFpsMode::default(),
            cell_stat: crate::metrics::CellStat::default(),
            cell_precision: crate::metrics::DEFAULT_PRECISION as u8,
            ref_pixfmt: crate::metrics::ffmpeg::RefPixFmt::default(),
            plot_at_start: false,
            csv_export: false,
            csv_dir: String::new(),
            results_autosave: false,
            results_path: String::new(),
            plot_size: crate::plot::PlotSize::default(),
            badframes_count: "5".to_owned(),
            badframes_export_dir: String::new(),
            rows: Vec::new(),
            ffmpeg,
            ffvship,
            ffprobe,
            ref_info: crate::probe::reference_media_text("", None).0,
            ref_info_data: None,
            ref_rect: None,
            table_rect: None,
            hover_row: None,
            hover_since: None,
            hovered_now: None,
            include_anchor: None,
            selected_anchor: None,
            sort_spec: None,
            toast: None,
            probe_timeout_note: None,
            csv_report: None,
            results_autosave_pending: false,
            probe_tx,
            probe_rx,
            bin_rx,
            bins_probing,
            metric_tx,
            metric_rx,
            measuring: false,
            abort: Arc::new(AtomicBool::new(false)),
            current_child: Arc::new(Mutex::new(None)),
            pending: 0,
            next_color_idx: 0,
            next_probe_seq: 0,
            run_generation: 0,
            last_spawned_ref: String::new(),
            ref_info_path: String::new(),
            ref_generation: 0,
            thumb_tx,
            thumb_rx,
            thumb_tex: None,
            thumb_loading: false,
            last_thumb_path: String::new(),
            thumb_generation: 0,
            saved_snapshot: crate::state::AppState::default(),
            pending_save_since: None,
            state_fallback_toasted: false,
            live_kind: None,
            live_key: None,
            show_plot: false,
            plot_tab: MetricKind::Psnr,
            plot_tabs_w: 0.0,
            plot_live_fit: None,
            plot_follow_pending: false,
            plot_reset_pending: false,
            plot_snap: false,
            plot_save_pending: None,
            png_tx,
            png_rx,
            png_saving: false,
            badframe_tx,
            badframe_rx,
            badframes_busy: false,
            badframe_done: 0,
            badframe_total: 0,
            badframe_abort: Arc::new(AtomicBool::new(false)),
            badframe_report: None,
            show_badframes: false,
            badframe_tab: MetricKind::Psnr,
            badframe_file: None,
            badframe_frame_pos: 0,
            badframe_reset_once: false,
            badframe_view_key: None,
            badframe_slider: false,
            badframe_split: 0.5,
            badframe_div_sx: f32::NAN,
            badframe_div_drag: false,
            badframe_tex_dist: None,
            badframe_tex_ref: None,
            badframe_tex_key: None,
            badframe_tmp: crate::metrics::badframes::tmp_dir(),
            badframe_tmp_cleanup_pending: false,
            badframe_files: Vec::new(),
            badframe_export_pending: None,
        };
        // Hermetic tests: the developer's own state file must not leak
        // into assertions about defaults.
        let loaded = if cfg!(test) {
            None
        } else {
            crate::state::load()
        };
        app.apply_state(loaded);
        app.saved_snapshot = app.snapshot();
        app
    }
}

impl RFMetricsApp {
    /// Everything `ffmetrics-state.json` persists, read off the live UI.
    pub(crate) fn snapshot(&self) -> crate::state::AppState {
        crate::state::AppState {
            ref_path: self.ref_path.clone(),
            skip: self.skip.clone(),
            duration: self.duration.clone(),
            files: Some(
                self.rows
                    .iter()
                    .map(|r| crate::state::FileEntry {
                        path: r.path.clone(),
                        include: r.include,
                    })
                    .collect(),
            ),
            metrics: crate::state::MetricsState {
                psnr: Some(self.m_psnr),
                ssim: Some(self.m_ssim),
                vmaf: Some(self.m_vmaf),
                xpsnr: Some(self.m_xpsnr),
                ssim2: Some(self.m_ssim2),
                butteraugli: Some(self.m_but),
                cvvdp: Some(self.m_cvvdp),
            },
            vmaf: crate::state::VmafState {
                model: Some(self.vmaf_model.clone()),
                phone: Some(self.vmaf_phone),
                scale: Some(self.vmaf_scale),
                pooling: Some(self.vmaf_pooling.clone()),
                subsample: Some(self.vmaf_subsample.clone()),
                threads: Some(self.vmaf_threads.clone()),
            },
            options: crate::state::OptionsState {
                scaling: Some(self.scale_method.label().to_owned()),
                fps_mode: Some(self.fps_mode.label().to_owned()),
                cell_stat: Some(self.cell_stat.label().to_owned()),
                cell_precision: Some(self.cell_precision.to_string()),
                ref_pixfmt: Some(self.ref_pixfmt.label().to_owned()),
                plot_at_start: Some(self.plot_at_start),
                plot_size: Some(self.plot_size.label().to_owned()),
                csv_export: Some(self.csv_export),
                csv_dir: Some(self.csv_dir.clone()),
                badframes_count: Some(self.badframes_count.clone()),
                badframes_export_dir: Some(self.badframes_export_dir.clone()),
                results_autosave: Some(self.results_autosave),
                results_path: Some(self.results_path.clone()),
            },
        }
    }

    /// Issue #7: force off restored/default ticks for filters this ffmpeg
    /// build lacks (their header checkboxes are disabled, so they were
    /// never ticked live). Same for the FFVship family when the binary is
    /// absent or unusable (wrong-GPU build): its header checkboxes render
    /// disabled, so restored ticks are cleared too. Session-only
    /// capability, never persisted.
    /// Idempotent: safe to run for both the no-file and restored paths.
    pub(crate) fn untick_unsupported_metrics(&mut self) {
        let supported = &self.ffmpeg.supported_metrics;
        if !supported.contains(&MetricKind::Psnr) {
            self.m_psnr = false;
        }
        if !supported.contains(&MetricKind::Ssim) {
            self.m_ssim = false;
        }
        if !supported.contains(&MetricKind::Vmaf) {
            self.m_vmaf = false;
        }
        if !supported.contains(&MetricKind::Xpsnr) {
            self.m_xpsnr = false;
        }
        if !self.ffvship.usable {
            self.m_ssim2 = false;
            self.m_but = false;
            self.m_cvvdp = false;
        }
    }

    /// Apply a loaded state file (tolerant per-key; absent keys keep live
    /// defaults, saved models must still be on disk, queue entries must
    /// still be files). Restored rows probe through the normal path.
    pub(crate) fn apply_state(&mut self, loaded: Option<crate::state::AppState>) {
        // Defaults (VMAF-on) obey capability even with no state file.
        self.untick_unsupported_metrics();
        let Some(s) = loaded else {
            return;
        };
        self.ref_path = s.ref_path;
        self.skip = s.skip;
        self.duration = s.duration;
        if let Some(files) = s.files {
            let live: Vec<&crate::state::FileEntry> = files
                .iter()
                .filter(|e| std::path::Path::new(&e.path).is_file())
                .collect();
            self.add_queue_files(
                live.iter()
                    .map(|e| std::path::PathBuf::from(&e.path))
                    .collect(),
            );
            for e in live {
                let ekey = norm_key(&e.path);
                if let Some(row) = self.rows.iter_mut().find(|r| r.key == ekey) {
                    row.include = e.include;
                }
            }
        }
        let m = s.metrics;
        if let Some(v) = m.psnr {
            self.m_psnr = v;
        }
        if let Some(v) = m.ssim {
            self.m_ssim = v;
        }
        if let Some(v) = m.vmaf {
            self.m_vmaf = v;
        }
        if let Some(v) = m.xpsnr {
            self.m_xpsnr = v;
        }
        if let Some(v) = m.ssim2 {
            self.m_ssim2 = v;
        }
        if let Some(v) = m.butteraugli {
            self.m_but = v;
        }
        if let Some(v) = m.cvvdp {
            self.m_cvvdp = v;
        }
        let v = s.vmaf;
        if let Some(model) = v.model
            && self.vmaf_models.contains(&model)
        {
            self.vmaf_model = model;
        }
        if let Some(phone) = v.phone {
            self.vmaf_phone = phone;
        }
        if let Some(scale) = v.scale {
            self.vmaf_scale = scale;
        }
        if let Some(pooling) = v.pooling
            && ["Mean", "Harmonic Mean"].contains(&pooling.as_str())
        {
            self.vmaf_pooling = pooling;
        }
        if let Some(subsample) = v.subsample
            && ["1", "2", "3", "5", "10", "15"].contains(&subsample.as_str())
        {
            self.vmaf_subsample = subsample;
        }
        if let Some(threads) = v.threads {
            self.vmaf_threads = threads;
        }
        if let Some(scaling) = s.options.scaling
            && let Some(m) = ScaleMethod::from_label(&scaling)
        {
            self.scale_method = m;
        }
        if let Some(fps_mode) = s.options.fps_mode
            && let Some(m) = crate::metrics::ffmpeg::InputFpsMode::from_label(&fps_mode)
        {
            self.fps_mode = m;
        }
        if let Some(cell_stat) = s.options.cell_stat
            && let Some(m) = crate::metrics::CellStat::from_label(&cell_stat)
        {
            self.cell_stat = m;
        }
        if let Some(ref_pixfmt) = s.options.ref_pixfmt
            && let Some(m) = crate::metrics::ffmpeg::RefPixFmt::from_label(&ref_pixfmt)
        {
            self.ref_pixfmt = m;
        }
        if let Some(cell_precision) = s.options.cell_precision
            && let Ok(p) = cell_precision.parse::<u8>()
            && p as usize <= crate::metrics::MAX_PRECISION
        {
            self.cell_precision = p;
        }
        if let Some(plot_at_start) = s.options.plot_at_start {
            self.plot_at_start = plot_at_start;
        }
        if let Some(plot_size) = s.options.plot_size
            && let Some(m) = crate::plot::PlotSize::from_label(&plot_size)
        {
            self.plot_size = m;
        }
        if let Some(csv_export) = s.options.csv_export {
            self.csv_export = csv_export;
        }
        if let Some(csv_dir) = s.options.csv_dir {
            self.csv_dir = csv_dir;
        }
        if let Some(badframes_count) = s.options.badframes_count
            && crate::metrics::badframes::COUNT_LABELS.contains(&badframes_count.as_str())
        {
            self.badframes_count = badframes_count;
        }
        if let Some(badframes_export_dir) = s.options.badframes_export_dir {
            self.badframes_export_dir = badframes_export_dir;
        }
        if let Some(results_autosave) = s.options.results_autosave {
            self.results_autosave = results_autosave;
        }
        if let Some(results_path) = s.options.results_path {
            self.results_path = results_path;
        }
        // Restored ticks obey capability (see helper docs).
        self.untick_unsupported_metrics();
    }

    /// Allocation-free dirty check mirroring `snapshot() != saved_snapshot`
    /// without building `AppState`. Covers every field `snapshot()` sets;
    /// a new persisted field must be added here too, or edits to it will
    /// silently stop saving.
    pub(crate) fn is_state_dirty(&self) -> bool {
        let s = &self.saved_snapshot;
        if self.ref_path != s.ref_path || self.skip != s.skip || self.duration != s.duration {
            return true;
        }
        // `snapshot()` always emits `Some(vec)`; a `None` here can only mean
        // "never saved this shape", which never equals a snapshot.
        match &s.files {
            Some(files) => {
                if files.len() != self.rows.len() {
                    return true;
                }
                for (live, saved) in self.rows.iter().zip(files.iter()) {
                    if live.path != saved.path || live.include != saved.include {
                        return true;
                    }
                }
            }
            None => return true,
        }
        let m = &s.metrics;
        if m.psnr != Some(self.m_psnr)
            || m.ssim != Some(self.m_ssim)
            || m.vmaf != Some(self.m_vmaf)
            || m.xpsnr != Some(self.m_xpsnr)
            || m.ssim2 != Some(self.m_ssim2)
            || m.butteraugli != Some(self.m_but)
            || m.cvvdp != Some(self.m_cvvdp)
        {
            return true;
        }
        let v = &s.vmaf;
        if v.model.as_deref() != Some(self.vmaf_model.as_str())
            || v.phone != Some(self.vmaf_phone)
            || v.scale != Some(self.vmaf_scale)
            || v.pooling.as_deref() != Some(self.vmaf_pooling.as_str())
            || v.subsample.as_deref() != Some(self.vmaf_subsample.as_str())
            || v.threads.as_deref() != Some(self.vmaf_threads.as_str())
        {
            return true;
        }
        let o = &s.options;
        if o.scaling.as_deref() != Some(self.scale_method.label())
            || o.fps_mode.as_deref() != Some(self.fps_mode.label())
            || o.cell_stat.as_deref() != Some(self.cell_stat.label())
            || o.cell_precision
                .as_deref()
                .and_then(|s| s.parse::<u8>().ok())
                != Some(self.cell_precision)
            || o.ref_pixfmt.as_deref() != Some(self.ref_pixfmt.label())
            || o.plot_at_start != Some(self.plot_at_start)
            || o.plot_size.as_deref() != Some(self.plot_size.label())
            || o.csv_export != Some(self.csv_export)
            || o.csv_dir.as_deref() != Some(self.csv_dir.as_str())
            || o.badframes_count.as_deref() != Some(self.badframes_count.as_str())
            || o.badframes_export_dir.as_deref() != Some(self.badframes_export_dir.as_str())
            || o.results_autosave != Some(self.results_autosave)
            || o.results_path.as_deref() != Some(self.results_path.as_str())
        {
            return true;
        }
        false
    }

    /// Debounced state write (1s after the last detected change): compare
    /// the live snapshot against the last write, arm/re-arm a single
    /// wake-up while dirty, save once it settles.
    pub(crate) fn autosave_tick(&mut self, ctx: &egui::Context, now: f64) {
        use crate::state::SAVE_DEBOUNCE_SECS;
        if !self.is_state_dirty() {
            self.pending_save_since = None;
            return;
        }
        match self.pending_save_since {
            None => {
                self.pending_save_since = Some(now);
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(SAVE_DEBOUNCE_SECS));
            }
            Some(since) if now - since >= SAVE_DEBOUNCE_SECS => {
                let snap = self.snapshot();
                match crate::state::save(&snap) {
                    Some(path) => {
                        // Read-only install: the save fell back (or the log
                        // did) — toast once so persistence loss is visible.
                        if path != crate::state::state_path() && !self.state_fallback_toasted {
                            self.state_fallback_toasted = true;
                            self.toast(
                                now,
                                format!(
                                    "App folder not writable — state saves to {}",
                                    path.display()
                                ),
                                ToastKind::Warning,
                            );
                        }
                        self.saved_snapshot = snap;
                        self.pending_save_since = None;
                    }
                    None => {
                        // Transient (locked/full disk): retry next debounce,
                        // toast once.
                        if !self.state_fallback_toasted {
                            self.state_fallback_toasted = true;
                            self.toast(
                                now,
                                "State save failed everywhere — retrying".to_owned(),
                                ToastKind::Warning,
                            );
                        }
                        self.pending_save_since = Some(now);
                        ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                            SAVE_DEBOUNCE_SECS,
                        ));
                    }
                }
            }
            Some(since) => {
                let remaining = (SAVE_DEBOUNCE_SECS - (now - since)).max(0.0);
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(remaining));
            }
        }
    }

    /// Flush a run-end auto-save (armed by the Finished drain arm):
    /// resolves the configured path or the exe-dir default, exports via
    /// the manual path below, and disarms. Headless-testable: the UI
    /// frame only supplies `now`.
    pub(crate) fn consume_autosave(&mut self, now: f64) {
        if !self.results_autosave_pending {
            return;
        }
        self.results_autosave_pending = false;
        let path = if self.results_path.trim().is_empty() {
            crate::metrics::results::default_results_path()
        } else {
            std::path::PathBuf::from(&self.results_path)
        };
        self.save_results(now, path);
    }

    /// Export the results summary CSV (bottom-bar "Save results"): one
    /// row per queued row in table order, appended to the chosen file
    /// (header only when new/empty). Unscored cells export as empty
    /// blocks; the caller gates on `!run_locked`.
    pub(crate) fn save_results(&mut self, now: f64, path: std::path::PathBuf) {
        use crate::metrics::results::{Block, ORDER, ResultsRow};
        let stamp = wall_now_string();
        let app_version = env!("CARGO_PKG_VERSION");
        let ffmpeg_version = self.ffmpeg.ffmpeg_version.clone().unwrap_or_default();
        let lines: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let blocks: [Block; 7] = std::array::from_fn(|i| {
                    let kind = ORDER[i];
                    match r.cell(kind) {
                        crate::metrics::MetricCell::Done {
                            avg,
                            skip,
                            clip_dur,
                            vmaf_cfg,
                            ..
                        } => {
                            let cached = r.cached(kind);
                            let vmaf = (kind == MetricKind::Vmaf)
                                .then_some(vmaf_cfg.as_ref())
                                .flatten()
                                .map(|c| (c.model.as_str(), c.pooling));
                            Block {
                                avg: Some(*avg),
                                stats: cached.stats.as_ref(),
                                finished: cached.finished.as_deref(),
                                options: crate::metrics::results::options_for(
                                    kind, *skip, *clip_dur, vmaf,
                                ),
                            }
                        }
                        _ => Block {
                            avg: None,
                            stats: None,
                            finished: None,
                            options: String::new(),
                        },
                    }
                });
                let frames = ORDER
                    .iter()
                    .find_map(|k| match r.cell(*k) {
                        crate::metrics::MetricCell::Done { values, .. } if !values.is_empty() => {
                            Some(values.len().to_string())
                        }
                        _ => None,
                    })
                    .unwrap_or_default();
                ResultsRow {
                    blocks,
                    frames,
                    frame: crate::probe::results_media_text(r.info.as_ref()),
                    bitrate: r
                        .info
                        .as_ref()
                        .and_then(|info| info.bitrate_kbps)
                        .map(|kbps| kbps.to_string())
                        .unwrap_or_default(),
                    path: &r.path,
                }
            })
            .map(|data| crate::metrics::results::row(&stamp, &data, app_version, &ffmpeg_version))
            .collect();
        match crate::metrics::results::append(&path, &lines) {
            Ok(n) => {
                let s = if n == 1 { "" } else { "s" };
                log::info!(target: "rfmetrics::app", "saved {n} result row{s} to {}", path.display());
                self.toast(
                    now,
                    format!("Appended {n} row{s} to {}", path.display()),
                    ToastKind::Info,
                );
            }
            Err(e) => {
                log::error!(target: "rfmetrics::app", "save results failed: {e}");
                self.toast(
                    now,
                    format!("Could not save results: {e}"),
                    ToastKind::Error,
                );
            }
        }
    }
}
/// 1px vertical divider in an exact 3px grid column.
pub(crate) fn vline(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, color),
    );
}

/// Vector sort-direction mark: painted triangles, not text — the bundled
/// UI font has no ▲▼⇅ glyphs (tofu squares). Active direction in text
/// color, inactive as a faint up+down pair (sortable affordance).
/// Returns the click response; the caller attaches hover text + action.
pub(crate) fn sort_mark(ui: &mut egui::Ui, dir: Option<SortDir>) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let c = rect.center();
        let visuals = ui.visuals();
        // (pointing-up, y-offset, half-size).
        let tris: &[(bool, f32, f32)] = match dir {
            Some(SortDir::Asc) => &[(true, 0.0, 4.5)],
            Some(SortDir::Desc) => &[(false, 0.0, 4.5)],
            None => &[(true, -2.6, 3.0), (false, 2.6, 3.0)],
        };
        let color = match dir {
            Some(_) => visuals.text_color(),
            None => visuals.weak_text_color(),
        };
        for &(up, dy, r) in tris {
            let cy = c.y + dy;
            let pts = if up {
                vec![
                    egui::pos2(c.x - r, cy + r * 0.8),
                    egui::pos2(c.x + r, cy + r * 0.8),
                    egui::pos2(c.x, cy - r * 0.8),
                ]
            } else {
                vec![
                    egui::pos2(c.x - r, cy - r * 0.8),
                    egui::pos2(c.x + r, cy - r * 0.8),
                    egui::pos2(c.x, cy + r * 0.8),
                ]
            };
            ui.painter()
                .add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        }
    }
    resp
}

/// Panel frame with Python's drag-enter green (#2FA572) while hovered.
pub(crate) fn panel_frame(ui: &egui::Ui, hovering: bool) -> egui::Frame {
    let mut frame = egui::Frame::group(ui.style());
    if hovering {
        frame = frame.stroke(egui::Stroke::new(
            1.5,
            egui::Color32::from_rgb(0x2F, 0xA5, 0x72),
        ));
    }
    frame
}

/// Screenshot green/red fills, muted for the dark theme (light text stays
/// readable): best green, worst red, all-tied dim yellow. Colors apply only
/// with 2+ scored rows; a lone result stays uncolored.
pub(crate) const BEST_FILL: egui::Color32 = egui::Color32::from_rgb(0x2E, 0x6B, 0x3E);
pub(crate) const WORST_FILL: egui::Color32 = egui::Color32::from_rgb(0x7A, 0x36, 0x36);
pub(crate) const TIE_FILL: egui::Color32 = egui::Color32::from_rgb(0x6B, 0x5F, 0x2A);
/// Cross-format warning tint for Media cells (upstream #47): readable on
/// the dark default theme without touching layout.
pub(crate) const WARN_TEXT: egui::Color32 = egui::Color32::from_rgb(0xE5, 0xA6, 0x3B);

/// Cell/chip background for a stat rank; `None` = no highlight.
pub(crate) fn rank_fill(rank: crate::metrics::StatRank) -> Option<egui::Color32> {
    match rank {
        crate::metrics::StatRank::Best => Some(BEST_FILL),
        crate::metrics::StatRank::Worst => Some(WORST_FILL),
        crate::metrics::StatRank::Tie => Some(TIE_FILL),
        crate::metrics::StatRank::Plain => None,
    }
}

/// Indeterminate bounce position for `Running` cells: triangle wave
/// `0 → 1 → 0`, one leg per `LEG_S` seconds. Pure (no `Ui`) so tests
/// cover the ping-pong without a GUI harness.
pub(crate) fn running_sweep_pos(time_s: f64) -> f32 {
    const LEG_S: f64 = 0.7;
    let phase = (time_s / LEG_S).rem_euclid(2.0);
    (if phase < 1.0 { phase } else { 2.0 - phase }) as f32
}

/// Bold green sweep behind a `Running` cell's text: a full-height
/// translucent `#2FA572` segment bouncing left ↔ right. Painted before
/// the label so the `Frame: N` text stays on top; driven by the
/// existing ~10Hz measuring heartbeat, so no extra repaint cost.
pub(crate) fn paint_running_sweep(ui: &mut egui::Ui) {
    let rect = ui.available_rect_before_wrap();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let x01 = running_sweep_pos(ui.input(|i| i.time));
    let seg_w = (rect.width() * 0.28).clamp(14.0, 24.0);
    let bar = egui::Rect::from_min_size(
        egui::pos2(rect.left() + (rect.width() - seg_w) * x01, rect.top()),
        egui::vec2(seg_w, rect.height()),
    );
    ui.painter().rect_filled(
        bar,
        3.0,
        egui::Color32::from_rgba_unmultiplied(0x2F, 0xA5, 0x72, 110),
    );
}

/// Filter-metric Done tooltip in FFMetrics order: Avg, Exec, Frames, a blank
/// line, Mean..StdDev, another blank line, then Percentiles. Each comparable
/// value is chipped by its cross-row rank; Exec time and Frames count are
/// display-only (no chip).
/// Plain horizontal rows with content-hugging widths: grids and expanding
/// layouts feed back into the tooltip auto-size and balloon while hovered.
pub(crate) fn metric_stat_tooltip(
    ui: &mut egui::Ui,
    title: &str,
    stats: &crate::metrics::DoneStats,
    ranks: &[crate::metrics::StatRank; 10],
    sel: crate::metrics::CellStat,
) {
    ui.label(egui::RichText::new(title).strong());
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 1.0);
        // Bold the row the cell-value selector shows.
        let lbl = |k: usize, label: &'static str| {
            if k == sel.index() {
                egui::RichText::new(label).strong()
            } else {
                egui::RichText::new(label)
            }
        };
        let comp = stats.comparable();
        let (label, v, _) = comp[0];
        tip_stat_row(ui, lbl(0, label), &format!("{v:.6}"), ranks[0]);
        tip_plain_row(ui, "Exec time:", &crate::metrics::format_exec(stats.exec_s));
        tip_plain_row(ui, "Frames count:", &stats.frames.to_string());
        ui.add_space(5.0);
        for k in 1..=5 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, lbl(k, label), &format!("{v:.6}"), ranks[k]);
        }
        ui.add_space(5.0);
        for k in 6..10 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, lbl(k, label), &format!("{v:.6}"), ranks[k]);
        }
    });
}

/// One tooltip row: fixed label + right-aligned value, chipped when ranked.
pub(crate) fn tip_stat_row(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    val: &str,
    rank: crate::metrics::StatRank,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [96.0, 15.0],
            egui::Label::new(label).halign(egui::Align::LEFT),
        );
        let val = egui::Label::new(val).halign(egui::Align::RIGHT);
        match rank_fill(rank) {
            Some(fill) => {
                egui::Frame::NONE
                    .fill(fill)
                    .inner_margin(egui::Margin::symmetric(4, 0))
                    .show(ui, |ui| {
                        ui.add_sized([80.0, 15.0], val);
                    });
            }
            None => {
                ui.add_sized([88.0, 15.0], val);
            }
        }
    });
}

/// Display-only tooltip row (Exec time, Frames count): never chipped.
pub(crate) fn tip_plain_row(ui: &mut egui::Ui, label: &str, val: &str) {
    tip_stat_row(ui, label, val, crate::metrics::StatRank::Plain);
}

impl eframe::App for RFMetricsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let (hovering, dropped) = ui.ctx().input(|i| {
            (
                !i.raw.hovered_files.is_empty(),
                i.raw
                    .dropped_files
                    .iter()
                    .map(|f| f.path().to_path_buf())
                    .collect::<Vec<_>>(),
            )
        });

        // Drops are disabled mid-run (guard rail): ignore the drag state so
        // targets never outline and `route_drop` below can only Block.
        let hovering = hovering && !self.measuring;

        // Continuously repaint while dragging so hover outlines update smoothly
        if hovering {
            ui.ctx().request_repaint();
        }

        let cursor_pos = get_cursor_pos(ui.ctx());
        let now = ui.ctx().input(|i| i.time);
        // While the metric worker runs, run-scoped inputs lock: dimmed and
        // unclickable so paths, trim, queue, and toggles can't shift mid-run.
        let run_locked = self.measuring || self.badframes_busy;

        // Direct OS-cursor hit test; winit gives no position during OLE drags.
        let is_over_ref =
            matches!((cursor_pos, self.ref_rect), (Some(pos), Some(rect)) if rect.contains(pos));
        let is_over_table =
            matches!((cursor_pos, self.table_rect), (Some(pos), Some(rect)) if rect.contains(pos));

        // Strict target routing (Python parity: a drop outside a target
        // does nothing). The reference box takes one file; extras are
        // reported via toast instead of silently vanishing.
        match route_drop(self.measuring, is_over_ref, is_over_table, dropped) {
            DropAction::Ignore => {}
            DropAction::Blocked => {
                self.toast(
                    now,
                    "Stop the run before changing files".to_owned(),
                    ToastKind::Info,
                );
            }
            DropAction::SetRef { first, extra } => {
                self.ref_path = first.to_string_lossy().into_owned();
                if extra > 0 {
                    self.toast(
                        now,
                        format!("Reference takes one file — kept the first, ignored {extra} more"),
                        ToastKind::Warning,
                    );
                }
            }
            DropAction::Queue(paths) => self.add_queue_files(paths),
        }
        // Repaint only on frames that drained worker traffic, so live
        // `Frame: N` progress and arriving results render while data flows
        // and gaps (VMAF startup, slow encodes, between-jobs) no longer
        // spin the full table rebuild at 60fps. `|` (not `||`) keeps every
        // drain running. Spawns above stem from input frames, which repaint
        // on their own; thumb workers also wake the UI themselves.
        let live = self.refresh_ref_info();
        // Startup version probes land here (off-UI-thread, see `Default`).
        let live = self.drain_bin_results() | live;
        // Probe timeouts surface once as a warning toast (the drain only
        // records the name; the slot keeps the latest like `toast`).
        if let Some(name) = self.probe_timeout_note.take() {
            self.toast(now, format!("Probe timed out: {name}"), ToastKind::Warning);
        }
        let ctx = ui.ctx().clone();
        let live = self.refresh_thumbnail(&ctx) | live;
        let live = self.drain_metric_results() | live;
        if let Some((ok, errors)) = self.csv_report.take() {
            if errors.is_empty() {
                let s = if ok == 1 { "" } else { "s" };
                self.toast(now, format!("Saved {ok} CSV file{s}"), ToastKind::Info);
            } else {
                let mut first = errors[0].clone();
                if first.chars().count() > 80 {
                    first = format!("{}…", first.chars().take(79).collect::<String>());
                }
                let s = if errors.len() == 1 { "" } else { "s" };
                self.toast(
                    now,
                    format!("CSV export failed for {} file{s}: {first}", errors.len()),
                    ToastKind::Error,
                );
            }
        }
        self.consume_autosave(now);
        let live = self.drain_png_results(&ctx, now) | live;
        let live = self.drain_badframe_results(now) | live;
        if live {
            ui.ctx().request_repaint();
        } else if self.measuring || self.badframes_busy {
            // Heartbeat: metric/probe workers never wake the UI, so without
            // a running frame their traffic would strand in the channel
            // (stuck `Frame: 0` until the next mouse move). 10Hz keeps
            // counters live; traffic frames repaint immediately above.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Debounced `ffmetrics-state.json` write (Python parity).
        self.autosave_tick(ui.ctx(), now);

        let ref_hover = hovering && is_over_ref;
        let table_hover = hovering && is_over_table;

        // ---- Reference (top, fixed) ----
        let ref_resp = egui::Panel::top("reference").show(ui, |ui| {
            ui.add(egui::Label::new("Reference").selectable(false));
            panel_frame(ui, ref_hover).show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let preview_w = 136.0;
                    let total = ui.available_width();
                    ui.vertical(|ui| {
                        ui.set_width((total - preview_w - 12.0).max(0.0));
                        // Path row
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Path to file:").selectable(false));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let browse = ui.add_enabled_ui(!run_locked, |ui| {
                                        ui.add_sized([90.0, 24.0], egui::Button::new("Browse"))
                                    });
                                    if browse.inner.clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_title("Select reference video")
                                            .add_filter("Video files", VIDEO_EXTS)
                                            .pick_file()
                                    {
                                        self.ref_path = path.to_string_lossy().into_owned();
                                    }
                                    let _ = ui.add_enabled(
                                        !run_locked,
                                        egui::TextEdit::singleline(&mut self.ref_path)
                                            .desired_width(f32::INFINITY),
                                    );
                                },
                            );
                        });
                        ui.label(&self.ref_info);
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Duration:").selectable(false));
                            let _ = ui
                                .add_enabled(
                                    !run_locked,
                                    egui::TextEdit::singleline(&mut self.duration)
                                        .hint_text("00:00.000")
                                        .desired_width(110.0),
                                )
                                .on_hover_text(
                                    "Clip length to measure: seconds (10) or hh:mm:ss (.000)",
                                );
                            ui.add(egui::Label::new("Skip:").selectable(false));
                            let _ = ui
                                .add_enabled(
                                    !run_locked,
                                    egui::TextEdit::singleline(&mut self.skip)
                                        .hint_text("00:00.000")
                                        .desired_width(110.0),
                                )
                                .on_hover_text("Skip from start: seconds (5) or hh:mm:ss (.000)");
                            ui.add(egui::Label::new("Pixel Format:").selectable(false));
                            ui.add_enabled_ui(!run_locked, |ui| {
                                let _ = egui::ComboBox::from_id_salt("ref_pixfmt")
                                    .width(150.0)
                                    .selected_text(self.ref_pixfmt.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::metrics::ffmpeg::RefPixFmt::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.ref_pixfmt,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Pixel format the reference is converted to (both legs converge on it; \
                                         No conversion keeps the legacy distorted-to-reference legs). The selection \
                                         is ignored for metrics that don't support it (e.g. VMAF requires YUV); \
                                         FFVship metrics always compare unconverted inputs",
                                    );
                            });
                        });
                    });
                    // Reference thumbnail 136x76 (black box parity with Python).
                    egui::Frame::NONE
                        .fill(egui::Color32::BLACK)
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(60)))
                        .show(ui, |ui| {
                            ui.set_min_size(egui::vec2(preview_w, 76.0));
                            if let Some(tex) = &self.thumb_tex {
                                let size = tex.size_vec2();
                                ui.centered_and_justified(|ui| {
                                    ui.image((tex.id(), size));
                                });
                            } else if self.thumb_loading {
                                ui.centered_and_justified(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new("Loading…")
                                                .small()
                                                .color(egui::Color32::from_gray(160)),
                                        )
                                        .selectable(false),
                                    );
                                });
                            } else {
                                ui.centered_and_justified(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new("No preview")
                                                .small()
                                                .color(egui::Color32::from_gray(120)),
                                        )
                                        .selectable(false),
                                    );
                                });
                            }
                        });
                });
            });
        });
        self.ref_rect = Some(ref_resp.response.rect);

        // ---- Bottom action bar (bottommost) ----
        egui::Panel::bottom("actions").show(ui, |ui| {
            ui.horizontal(|ui| {
                let run_label = if self.measuring { "Stop" } else { "Start" };
                if ui
                    .add_sized([90.0, 24.0], egui::Button::new(run_label))
                    .clicked()
                {
                    if self.measuring {
                        self.stop_psnr();
                    } else {
                        self.start_run(now);
                    }
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([90.0, 24.0], egui::Button::new("Reset"))
                    })
                    .inner
                    .clicked()
                {
                    self.reset_psnr();
                }
                if ui
                    .add_sized([90.0, 24.0], egui::Button::new("Plot"))
                    .clicked()
                {
                    self.show_plot = true;
                }
                // Bad-frames viewer window (tmp-backed, tabs per metric).
                let bf_enabled = !self.measuring && self.ffmpeg.path.is_some();
                if ui
                    .add_enabled_ui(bf_enabled, |ui| {
                        ui.add_sized([130.0, 24.0], egui::Button::new("Bad frames"))
                    })
                    .inner
                    .on_hover_text("Open the worst-frame viewer (dist/ref side by side)")
                    .clicked()
                {
                    self.show_badframes = true;
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([110.0, 24.0], egui::Button::new("Save results"))
                    })
                    .inner
                    .on_hover_text("Append one row per queued file to RFMetrics.Results.csv")
                    .clicked()
                    && let Some(mut path) = rfd::FileDialog::new()
                        .set_title("Save results CSV")
                        .set_file_name(crate::metrics::results::RESULTS_FILE_NAME)
                        .add_filter("CSV file", &["csv"])
                        .save_file()
                {
                    path.set_extension("csv");
                    self.save_results(now, path);
                }
                ui.label(&self.ffmpeg.short)
                    .on_hover_text(&self.ffmpeg.detail);
                ui.add(egui::Label::new("|").selectable(false));
                ui.label(&self.ffvship.short)
                    .on_hover_text(&self.ffvship.detail);
            });
        });

        // ---- VMAF options + Options (just above bottom bar) ----
        egui::Panel::bottom("vmaf").show(ui, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.add(egui::Label::new("VMAF options").selectable(false));
            // Dim when running or when the VMAF header checkbox is off
            // (todo.txt:1 parity with the run_locked inputs above).
            let vmaf_enabled = !run_locked && self.m_vmaf;
            ui.add_enabled_ui(vmaf_enabled, |ui| {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Model").selectable(false));
                    let models = self.vmaf_models.clone();
                    // Tall enough for every model: the default max menu
                    // height scrolls past ~10 entries. This is a ceiling —
                    // the popup still shrinks to its content.
                    let menu_h = models.len() as f32 * 24.0 + 16.0;
                    let _ = egui::ComboBox::from_id_salt("vmaf_model")
                        .width(220.0)
                        .height(menu_h)
                        .selected_text(&self.vmaf_model)
                            .show_ui(ui, |ui| {
                                for m in &models {
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_model,
                                        m.clone(),
                                        m.as_str(),
                                    );
                                }
                            });
                        // v1 phone is the separate `5d0h` file, never the flag:
                        // block the box for v1 models (run-time guard is the
                        // backstop). Unchecking here is visible and keeps the
                        // Start snapshot truthful — no silent coercion.
                        let v1 = crate::metrics::vmaf::is_v1_model(&self.vmaf_model);
                        if v1 {
                            self.vmaf_phone = false;
                        }
                        let _ = ui
                            .add_enabled(
                                !v1,
                                egui::Checkbox::new(&mut self.vmaf_phone, "Phone"),
                            )
                            .on_hover_text(if v1 {
                                "v1 phone is the separate 5d0h model file — pick it with Phone unticked"
                            } else {
                                "Phone viewing-condition transform (v0.6.1 model only)"
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 18.0], egui::Label::new(""));
                        let _ = ui.add(egui::Checkbox::new(
                            &mut self.vmaf_scale,
                            "Scale to model's resolution",
                        ));
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 18.0], egui::Label::new("Pooling").selectable(false));
                        let _ = egui::ComboBox::from_id_salt("vmaf_pooling")
                            .width(220.0)
                            .selected_text(&self.vmaf_pooling)
                            .show_ui(ui, |ui| {
                                let _ = ui.selectable_value(
                                    &mut self.vmaf_pooling,
                                    "Mean".to_owned(),
                                    "Mean",
                                );
                                let _ = ui.selectable_value(
                                    &mut self.vmaf_pooling,
                                    "Harmonic Mean".to_owned(),
                                    "Harmonic Mean",
                                );
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [70.0, 18.0],
                            egui::Label::new("Subsample").selectable(false),
                        );
                        let _ = egui::ComboBox::from_id_salt("vmaf_subsample")
                            .width(220.0)
                            .selected_text(&self.vmaf_subsample)
                            .show_ui(ui, |ui| {
                                for v in ["1", "2", "3", "5", "10", "15"] {
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_subsample,
                                        v.to_owned(),
                                        v,
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 18.0], egui::Label::new("Threads").selectable(false));
                        let _ = egui::ComboBox::from_id_salt("vmaf_threads")
                            .width(220.0)
                            .selected_text(&self.vmaf_threads)
                            .show_ui(ui, |ui| {
                                for v in ["auto", "1", "2", "4", "8", "16", "32"] {
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_threads,
                                        v.to_owned(),
                                        v,
                                    );
                                }
                            })
                            .response
                            .on_hover_text("auto follows the system CPU count");
                    });
                });
            });
                });
                // Global options box, right of VMAF options. Gated on
                // `!run_locked` only (not on `m_vmaf`): the scaler feeds
                // every ffmpeg-backed metric.
                ui.vertical(|ui| {
                    ui.add(egui::Label::new("Options").selectable(false));
                    ui.add_enabled_ui(!run_locked, |ui| {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal_top(|ui| {
                                ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Scaling").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("scale_method")
                                    .width(220.0)
                                    .selected_text(self.scale_method.label())
                                    .show_ui(ui, |ui| {
                                        for m in ScaleMethod::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.scale_method,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "sws scaler for every scale filter the app emits; \
                                         FFmpeg default omits flags (bicubic in practice)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Framerate").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("fps_mode")
                                    .width(220.0)
                                    .selected_text(self.fps_mode.label())
                                    .show_ui(ui, |ui| {
                                        use crate::metrics::ffmpeg::InputFpsMode;
                                        for m in InputFpsMode::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.fps_mode,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "which -r rate ffmpeg forces on each input, e.g. ref 23.98 / \
                                         dist 23.81: Reference emits -r 23.98 before both inputs, so \
                                         a one-sided VFR misread cannot desync the pair \
                                         (FFMetrics #111); Per-input emits -r 23.81 then -r 23.98 \
                                         (upstream 1.4.5 parity — disagreeing detections desync \
                                         scores); Off emits no -r and trusts container timestamps",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Plot size").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("plot_size")
                                    .width(220.0)
                                    .selected_text(self.plot_size.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::plot::PlotSize::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.plot_size,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Image dimensions for Save PNG and Copy (current plot view)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Cell value").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("cell_stat")
                                    .width(220.0)
                                    .selected_text(self.cell_stat.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::metrics::CellStat::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.cell_stat,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Which per-run stat metric cells show, sort by, and copy (default Avg)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Precision").selectable(false),
                                );
                                let prev_precision = self.cell_precision;
                                let _ = egui::ComboBox::from_id_salt("cell_precision")
                                    .width(220.0)
                                    .selected_text(self.cell_precision.to_string())
                                    .show_ui(ui, |ui| {
                                        for p in 0..=crate::metrics::MAX_PRECISION as u8 {
                                            let _ = ui.selectable_value(
                                                &mut self.cell_precision,
                                                p,
                                                p.to_string(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Decimals for metric cell display and Copy value (default 4)",
                                    );
                                // Frozen Avg texts embed the old precision —
                                // re-freeze them once on change (non-Avg
                                // selectors format live and follow for free).
                                if self.cell_precision != prev_precision {
                                    self.refreeze_cell_texts();
                                }
                            });
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.plot_at_start,
                                    "Plot window at start",
                                ))
                                .on_hover_text(
                                    "Open the plot window automatically when a run starts",
                                );
                            if ui
                                .add(egui::Button::new("Refresh Files Media Info"))
                                .on_hover_text(
                                    "Re-probe the reference and every queued file (media info only)",
                                )
                                .clicked()
                            {
                                self.refresh_media_info();
                            }
                                });
                                ui.vertical(|ui| {
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.csv_export,
                                    "Save frames metrics to CSV files",
                                ))
                                .on_hover_text(
                                    "Write one <name>.<METRIC>.csv per finished run \
                                     (TAB-separated per-frame values), to the chosen CSV folder \
                                     or beside each distorted file when empty",
                                );
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("CSV folder").selectable(false),
                                );
                                // Bounded display (full path stays in the hover);
                                // the worker snapshots the real string at Start.
                                let full = self.csv_dir.clone();
                                let shown = if full.trim().is_empty() {
                                    "Beside distorted files".to_owned()
                                } else if full.chars().count() > 40 {
                                    format!(
                                        "…{}",
                                        full.chars().skip(full.chars().count() - 39).collect::<String>()
                                    )
                                } else {
                                    full.clone()
                                };
                                ui.label(shown).on_hover_text(if full.trim().is_empty() {
                                    "Empty: each CSV lands next to its distorted file".to_owned()
                                } else {
                                    full
                                });
                                if ui.button("Browse…").clicked()
                                    && let Some(dir) = rfd::FileDialog::new()
                                        .set_title("CSV output folder")
                                        .pick_folder()
                                {
                                    self.csv_dir = dir.to_string_lossy().into_owned();
                                }
                                if ui
                                    .button("Clear")
                                    .on_hover_text("Back to beside-the-distorted-file")
                                    .clicked()
                                {
                                    self.csv_dir.clear();
                                }
                            });
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.results_autosave,
                                    "Auto-save results",
                                ))
                                .on_hover_text(
                                    "Append one row per queued file to the results file \
                                     when each run ends, stopped runs included",
                                );
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Results file").selectable(false),
                                );
                                // Bounded display (full path stays in the hover);
                                // resolved (custom or exe-dir default) at save time.
                                let full = self.results_path.clone();
                                let empty = full.trim().is_empty();
                                let shown = if empty {
                                    "RFMetrics.Results.csv next to exe".to_owned()
                                } else if full.chars().count() > 40 {
                                    format!(
                                        "…{}",
                                        full.chars().skip(full.chars().count() - 39).collect::<String>()
                                    )
                                } else {
                                    full.clone()
                                };
                                ui.label(shown).on_hover_text(if empty {
                                    crate::metrics::results::default_results_path()
                                        .to_string_lossy()
                                        .into_owned()
                                } else {
                                    full.clone()
                                });
                                if ui.button("Browse…").clicked() {
                                    let mut dialog = rfd::FileDialog::new()
                                        .set_title("Results file")
                                        .add_filter("CSV file", &["csv"]);
                                    if empty {
                                        dialog = dialog.set_file_name(
                                            crate::metrics::results::RESULTS_FILE_NAME,
                                        );
                                    } else {
                                        dialog = dialog.set_file_name(&full);
                                    }
                                    if let Some(mut path) = dialog.save_file() {
                                        path.set_extension("csv");
                                        self.results_path =
                                            path.to_string_lossy().into_owned();
                                    }
                                }
                                if ui
                                    .button("Clear")
                                    .on_hover_text("Back to next-to-the-exe default")
                                    .clicked()
                                {
                                    self.results_path.clear();
                                }
                            });
                                });
                            });
                        });
                    });
                });
            });
        });

        // ---- File queue (center, expanding) ----
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([110.0, 24.0], egui::Button::new("Add files"))
                    })
                    .inner
                    .clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .set_title("Select video files")
                        .add_filter("Video files", VIDEO_EXTS)
                        .add_filter("All files", &["*"])
                        .pick_files()
                {
                    self.add_queue_files(paths);
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([130.0, 24.0], egui::Button::new("Remove Selected"))
                    })
                    .inner
                    .clicked()
                {
                    self.rows.retain(|r| !r.selected);
                    self.refresh_queue_names();
                    // The scored set may have shrunk: re-rank all columns.
                    for kind in MetricKind::ALL {
                        self.refresh_ranks(kind);
                    }
                }
            });
            ui.add_space(4.0);
            self.hovered_now = None;
            let table_resp = panel_frame(ui, table_hover).show(ui, |ui| {
                // Keep the drop box a stable size: at least full width × 160
                // even when the table content is smaller (e.g. one row).
                ui.set_min_size(egui::vec2(ui.available_width(), 160.0));
                if self.rows.is_empty() {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new("No files yet — drag & drop video files here")
                                .weak(),
                        )
                        .selectable(false),
                    );
                    return;
                }
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                    // Metric stats + ranks come from the per-row cache (H1:
                    // computed on result arrival, never per frame).
                    let mut table = egui_extras::TableBuilder::new(ui)
                        .striped(false)
                        .resizable(false)
                        .sense(egui::Sense::click())
                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                        .column(egui_extras::Column::exact(22.0))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::exact(20.0))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::remainder().clip(true))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::exact(240.0));
                    for _ in 0..7 {
                        table = table
                            .column(egui_extras::Column::exact(3.0))
                            .column(egui_extras::Column::exact(82.0));
                    }
                    // Row-click side effects deferred past the loop (L3): the
                    // hot path borrows rows read-only, so no per-frame clones
                    // are needed to dodge the borrow checker.
                    let mut toggle_row: Option<usize> = None;
                    let mut open_path: Option<String> = None;
                    let mut reveal_path: Option<String> = None;
                    let mut hovered_next: Option<usize> = None;
                    // Alt+click solo target: set inside the row loop when a
                    // checkbox flips with Alt held, applied once below.
                    let mut alt_solo: Option<usize> = None;
                    // Shift+click range fill: `(lo, hi, value)` span plus
                    // the anchor update, both applied once below.
                    let mut shift_range: Option<(usize, usize, bool)> = None;
                    let mut anchor_next: Option<usize> = None;
                    // Range anchor snapshot (stale = dangling past the
                    // queue → plain toggle); copied before the `&mut`
                    // flag borrows below so no shared borrow lives on.
                    let anchor = self.include_anchor.filter(|&a| a < self.rows.len());
                    let sel_anchor = self.selected_anchor.filter(|&a| a < self.rows.len());
                    // Sorted display order as underlying indices (identity
                    // when unsorted); anchors/toggles below stay underlying
                    // so they survive re-sorts without invalidation.
                    let view: Vec<usize> = sort_view(&self.rows, self.sort_spec, self.cell_stat);
                    let sort_spec = self.sort_spec;
                    // Header sort click, applied once below the loop.
                    let mut sort_click: Option<SortColumn> = None;
                    // Header per-metric Reset, applied once below the loop
                    // (deferred so the `&mut` flag borrows above don't
                    // conflict with the `&mut self` reset call).
                    let mut reset_click: Option<MetricKind> = None;
                    // Current-settings snapshot for the stale badge: one
                    // trim parse per frame, then comparisons only per cell.
                    // Invalid boxes ("bad time") disable the badge —
                    // `start_run` reports those as errors instead.
                    let stale_cur = match (
                        Self::trim_opt(&self.skip),
                        Self::trim_opt(&self.duration),
                    ) {
                        (Some(s), Some(c)) => Some((
                            s,
                            c,
                            self.current_vmaf_cfg(),
                            self.scale_method,
                            self.fps_mode,
                            self.ref_pixfmt,
                        )),
                        _ => None,
                    };
                    table
                        .header(18.0, |mut header| {
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                // Sort click (3-state: A-Z → Z-A → insertion);
                                // the mark is a painted vector triangle (see
                                // `sort_mark` — the UI font lacks ▲▼⇅).
                                let dir = match sort_spec {
                                    Some((SortColumn::Path, d)) => Some(d),
                                    _ => None,
                                };
                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("Path").strong())
                                            .frame(false),
                                    )
                                    .on_hover_text("Sort by path (A-Z, Z-A, insertion order)")
                                    .clicked()
                                    || sort_mark(ui, dir)
                                        .on_hover_text("Sort by path (A-Z, Z-A, insertion order)")
                                        .clicked()
                                {
                                    sort_click = Some(SortColumn::Path);
                                }
                            });
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("Media info").strong())
                                        .selectable(false),
                                );
                            });
                            // Issue #7: copy support out first — the loop takes
                            // `&mut` flag borrows, so no shared `self` borrow
                            // may live across it. FFVship readiness rides the
                            // same pattern (`usable` covers absent + wrong-GPU
                            // builds); its disabled hover shows the binary
                            // detail (path / GPU-mismatch hint) instead of
                            // the ffmpeg filter text below.
                            let (psnr_ok, ssim_ok, vmaf_ok, xpsnr_ok) = {
                                let sup = &self.ffmpeg.supported_metrics;
                                (
                                    sup.contains(&MetricKind::Psnr),
                                    sup.contains(&MetricKind::Ssim),
                                    sup.contains(&MetricKind::Vmaf),
                                    sup.contains(&MetricKind::Xpsnr),
                                )
                            };
                            let ffvship_ok = self.ffvship.usable;
                            let ffvship_detail = self.ffvship.detail.clone();
                            for (flag, name, ok, kind) in [
                                (&mut self.m_psnr, "PSNR", psnr_ok, MetricKind::Psnr),
                                (&mut self.m_ssim, "SSIM", ssim_ok, MetricKind::Ssim),
                                (&mut self.m_vmaf, "VMAF", vmaf_ok, MetricKind::Vmaf),
                                (&mut self.m_xpsnr, "XPSNR", xpsnr_ok, MetricKind::Xpsnr),
                                (&mut self.m_ssim2, "SSIM2", ffvship_ok, MetricKind::Ssim2),
                                (&mut self.m_but, "BUTTER", ffvship_ok, MetricKind::But),
                                (&mut self.m_cvvdp, "CVVDP", ffvship_ok, MetricKind::Cvvdp),
                            ] {
                                header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                                header.col(|ui| {
                                    // Sort click (3-state: best first →
                                    // reversed → insertion); the checkbox
                                    // keeps its enable/disable job. The mark
                                    // is a painted vector triangle (see
                                    // `sort_mark` — the UI font lacks ▲▼⇅).
                                    let col = SortColumn::Metric(kind);
                                    let dir = match sort_spec {
                                        Some((c, d)) if c == col => Some(d),
                                        _ => None,
                                    };
                                    if sort_mark(ui, dir)
                                        .on_hover_text(format!(
                                            "Sort by {name} (best first, reversed, insertion order)"
                                        ))
                                        .clicked()
                                    {
                                        sort_click = Some(col);
                                    }
                                    let mut resp = ui.add_enabled(
                                        ok && !run_locked,
                                        egui::Checkbox::new(flag, name),
                                    );
                                    if !ok {
                                        resp = if kind.is_ffvship() {
                                            resp.on_hover_text(&ffvship_detail)
                                        } else {
                                            resp.on_hover_text(format!(
                                                "{name} filter not supported by this ffmpeg build"
                                            ))
                                        };
                                    }
                                    resp.context_menu(|ui| {
                                        if ui
                                            .add_enabled(
                                                !run_locked,
                                                egui::Button::new(format!("Reset {name}")),
                                            )
                                            .clicked()
                                        {
                                            reset_click = Some(kind);
                                            ui.close();
                                        }
                                    });
                                });
                            }
                        })
                        .body(|body| {
                            body.rows(20.0, self.rows.len(), |mut row| {
                                let i = row.index();
                                // Display position → underlying row: every
                                // `vi` use below addresses the real row, so
                                // selection/anchors survive re-sorts.
                                let vi = view[i];
                                row.set_selected(self.rows[vi].selected);
                                // Delayed hover: only outline after the pointer
                                // rests on the row, so passing over rows while
                                // aiming at text doesn't flash each one.
                                let hover_delayed = self.hover_row == Some(vi)
                                    && self.hover_since.is_some_and(|t| now - t >= ROW_HOVER_DELAY);
                                row.set_hovered(hover_delayed);
                                // Free-space click toggles selection; widget clicks
                                // (checkbox, play, text drag-select) must not.
                                // (`toggle_row` etc. are set here, applied below.)
                                row.col(|ui| {
                                    let alt = ui.input(|i| i.modifiers.alt);
                                    let shift = ui.input(|i| i.modifiers.shift);
                                    let resp = ui
                                        .checkbox(&mut self.rows[vi].include, "")
                                        .on_hover_text("Include in run and plot");
                                    if resp.changed() {
                                        if alt {
                                            // Alt wins over Shift on combo.
                                            alt_solo = Some(vi);
                                        } else if shift {
                                            // Gmail-style: the span takes
                                            // the clicked box's post-toggle
                                            // value. The span is VIEW
                                            // positions (`i` already is one;
                                            // the underlying anchor maps
                                            // through `view`) and is mapped
                                            // back at apply time; a stale
                                            // anchor falls through to a
                                            // plain toggle.
                                            if let Some((lo, hi)) = anchor
                                                .and_then(|a| view.iter().position(|&u| u == a))
                                                .and_then(|ap| {
                                                    shift_include_range(view.len(), ap, i)
                                                })
                                            {
                                                shift_range = Some((lo, hi, self.rows[vi].include));
                                            }
                                        }
                                        // Every click moves the anchor, so
                                        // chained Shift+clicks extend from
                                        // the last clicked row.
                                        anchor_next = Some(vi);
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        open_path = Some(self.rows[vi].path.clone());
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) = row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let row_data = &self.rows[vi];
                                    // Plain non-selectable text: no button hover
                                    // outline; copy lives in the right-click menu
                                    // and the full path shows as tooltip (Python parity).
                                    ui.add(egui::Label::new(&row_data.display).selectable(false))
                                        .on_hover_text(&row_data.path);
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Show in explorer").clicked() {
                                        reveal_path = Some(self.rows[vi].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy path").clicked() {
                                        ui.ctx().copy_text(self.rows[vi].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy filename").clicked() {
                                        let name = Path::new(&self.rows[vi].path)
                                            .file_name()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| self.rows[vi].display.clone());
                                        ui.ctx().copy_text(name);
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) = row.col(|ui| {
                                    let row_data = &self.rows[vi];
                                    // Cross-format warning (upstream #47):
                                    // the runners silently scale/convert a
                                    // distorted leg to the reference (FFVship
                                    // normalizes nothing at all), so a row
                                    // whose probe differs from the reference
                                    // ambers with the conversions listed in
                                    // its tooltip. Suppressed while a pixel
                                    // format target is selected (the
                                    // convergence is explicitly requested).
                                    // Comparisons only when matched; strings
                                    // alloc on mismatch.
                                    let warns = match (
                                        self.ref_info_data.as_ref(),
                                        row_data.info.as_ref(),
                                    ) {
                                        (Some(r), Some(d))
                                            if self.ref_pixfmt
                                                == crate::metrics::ffmpeg::RefPixFmt::NoConversion =>
                                        {
                                            crate::metrics::ffmpeg::conversion_warnings(r, d)
                                        }
                                        _ => Vec::new(),
                                    };
                                    let rich = if warns.is_empty() {
                                        egui::RichText::new(&row_data.media)
                                    } else {
                                        egui::RichText::new(&row_data.media).color(WARN_TEXT)
                                    };
                                    let resp = ui.add(
                                        egui::Label::new(rich).selectable(false),
                                    );
                                    if warns.is_empty() {
                                        resp.on_hover_text(&row_data.media_tip);
                                    } else {
                                        resp.on_hover_ui(|ui| {
                                            ui.label(&row_data.media_tip);
                                            ui.add_space(5.0);
                                            ui.label(
                                                egui::RichText::new(
                                                    "Cross-format vs reference:",
                                                )
                                                .strong(),
                                            );
                                            for w in &warns {
                                                ui.label(w.describe());
                                            }
                                        });
                                    }
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Copy summary").clicked() {
                                        ui.ctx().copy_text(crate::probe::table_media_text(
                                            self.rows[vi].info.as_ref(),
                                        ));
                                        ui.close();
                                    }
                                    if ui.button("Copy details").clicked() {
                                        ui.ctx().copy_text(self.rows[vi].media_tip.clone());
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                // Metric columns in METRIC_COLUMNS order: live state
                                // text on a rank fill (best green, worst red,
                                // tie dim yellow) with per-stat chip tooltips
                                // for Done cells.
                                for (kind_opt, title) in METRIC_COLUMNS {
                                    let Some(kind) = kind_opt else {
                                        continue;
                                    };
                                    let (_, r) =
                                        row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                    if r.clicked() {
                                        toggle_row = Some(vi);
                                    }
                                    let (_, r) = row.col(|ui| {
                                        let row_data = &self.rows[vi];
                                        let cell = row_data.cell(kind);
                                        let cached = row_data.cached(kind);
                                        // Stale `Done` values (stamped options
                                        // no longer match current settings)
                                        // keep their value and rank fill —
                                        // only struck-through (issue #92).
                                        let stale = match &stale_cur {
                                            Some((s, c, vmaf, scaler, fps, pf)) => done_is_stale(
                                                kind, cell, *s, *c, vmaf, *scaler, *fps, *pf,
                                            ),
                                            None => false,
                                        };
                                        // Idle borrows a static, Avg-selected Done
                                        // borrows the arrival-frozen text, other
                                        // selectors format from the cached stats
                                        // (one short format per visible cell per
                                        // frame, only when non-Avg is selected),
                                        // Error borrows its message; only live
                                        // Running frames format per frame
                                        // (they change anyway).
                                        let sel = self.cell_stat;
                                        let prec = self.cell_precision as usize;
                                        let running;
                                        let selected;
                                        let text: &str = match cell {
                                            crate::metrics::MetricCell::Idle => "N/A",
                                            crate::metrics::MetricCell::Running {
                                                frame, ..
                                            } => {
                                                running = format!("Frame: {frame}");
                                                &running
                                            }
                                            crate::metrics::MetricCell::Done { .. }
                                                if sel == crate::metrics::CellStat::Avg =>
                                            {
                                                &cached.text
                                            }
                                            crate::metrics::MetricCell::Done { .. } => {
                                                match cached.stats.as_ref() {
                                                    Some(s) => {
                                                        selected = format!(
                                                            "{:.prec$}",
                                                            s.value(sel),
                                                            prec = prec
                                                        );
                                                        &selected
                                                    }
                                                    None => &cached.text,
                                                }
                                            }
                                            crate::metrics::MetricCell::Error { msg } => msg,
                                        };
                                        // Borrow the cached stats when scored;
                                        // unscored cells build their one-line tip
                                        // below, and only while hovered.
                                        let stats = cached.stats.as_ref();
                                        let ranks = cached.ranks;
                                        let mut cell_frame = egui::Frame::NONE;
                                        if let Some(fill) = rank_fill(ranks[sel.index()]) {
                                            cell_frame = cell_frame.fill(fill);
                                        }
                                        // Only the job on the Progress/Series feed
                                        // sweeps; queued `Running` cells wait
                                        // statically until their turn.
                                        let is_live = matches!(
                                            cell,
                                            crate::metrics::MetricCell::Running { .. }
                                        ) && self.live_kind == Some(kind)
                                            && self.live_key.as_deref()
                                                == Some(row_data.key.as_str());
                                        cell_frame.show(ui, |ui| {
                                            ui.set_width(ui.available_width());
                                            ui.centered_and_justified(|ui| {
                                                if is_live {
                                                    paint_running_sweep(ui);
                                                }
                                                // Plain non-selectable text, like the
                                                // Path/Media columns (copy lives in
                                                // the right-click menu); stale
                                                // values strike through.
                                                let rich = if stale {
                                                    egui::RichText::new(text).strikethrough()
                                                } else {
                                                    egui::RichText::new(text)
                                                };
                                                let resp = ui
                                                    .add(egui::Label::new(rich).selectable(false));
                                                match stats {
                                                    Some(stats) => {
                                                        resp.on_hover_ui(|ui| {
                                                            if stale {
                                                                ui.label("Stale settings - rerun to refresh");
                                                            }
                                                            metric_stat_tooltip(
                                                                ui, title, stats, &ranks, sel,
                                                            );
                                                        });
                                                    }
                                                    None => {
                                                        if resp.hovered() {
                                                            resp.on_hover_text(cell.tooltip(title));
                                                        }
                                                    }
                                                }
                                            });
                                        });
                                    });
                                    // Right-click copies (Media-column parity):
                                    // value = the visible selector stat,
                                    // summary = the whole tooltip stats block.
                                    r.context_menu(|ui| {
                                        if ui.button("Copy value").clicked() {
                                            ui.ctx().copy_text(
                                                self.rows[vi].cell(kind).cell_stat_text_prec(
                                                    self.cell_stat,
                                                    self.cell_precision as usize,
                                                ),
                                            );
                                            ui.close();
                                        }
                                        if ui.button("Copy summary").clicked() {
                                            ui.ctx()
                                                .copy_text(self.rows[vi].cell(kind).tooltip(title));
                                            ui.close();
                                        }
                                    });
                                    if r.clicked() {
                                        toggle_row = Some(vi);
                                    }
                                }
                                if row.response().hovered() {
                                    hovered_next = Some(vi);
                                }
                            });
                        });
                    // Deferred row-click side effects (L3): selection
                    // toggle, open-in-player (error toast needs `now`),
                    // reveal-in-explorer, and hover tracking all land after
                    // the loop. Modifiers are read here, in the same frame
                    // as the clicks above, so they match the click-time
                    // state without threading through every cell.
                    let (sel_alt, sel_shift) = ui.input(|i| (i.modifiers.alt, i.modifiers.shift));
                    if let Some(i) = toggle_row
                        && i < self.rows.len()
                    {
                        if sel_alt {
                            // Alt-solo/select-all, mirroring the include
                            // column: the helper's call depends only on
                            // other-rows state, so it runs directly on the
                            // unflipped column (rare action: one small
                            // alloc is fine — never per frame).
                            let mut flags: Vec<bool> =
                                self.rows.iter().map(|r| r.selected).collect();
                            apply_alt_include(&mut flags, i);
                            for (r, v) in self.rows.iter_mut().zip(flags) {
                                r.selected = v;
                            }
                        } else if sel_shift {
                            // Checkbox-style: the span takes the clicked
                            // row's post-toggle state — Shift+clicking a
                            // selected row unselects the range, an
                            // unselected one selects it. The span is VIEW
                            // positions (mapped back through `view`);
                            // stale/missing anchor falls through to a
                            // plain toggle.
                            let click_pos = view.iter().position(|&u| u == i);
                            let anchor_pos =
                                sel_anchor.and_then(|a| view.iter().position(|&u| u == a));
                            match (anchor_pos, click_pos) {
                                (Some(ap), Some(cp)) => {
                                    let (lo, hi) = (ap.min(cp), ap.max(cp));
                                    let v = !self.rows[i].selected;
                                    for &u in &view[lo..=hi] {
                                        self.rows[u].selected = v;
                                    }
                                }
                                _ => {
                                    self.rows[i].selected = !self.rows[i].selected;
                                }
                            }
                        } else {
                            self.rows[i].selected = !self.rows[i].selected;
                        }
                        // Every selection click moves the anchor, so chained
                        // Shift+clicks extend from the last clicked row.
                        self.selected_anchor = Some(i);
                    }
                    // Alt+click solo/select-all: the clicked box already
                    // flipped above; the helper overwrites the whole column
                    // from that post-toggle state (rare action: one small
                    // alloc is fine — never on the per-frame hot path).
                    if let Some(i) = alt_solo
                        && i < self.rows.len()
                    {
                        let mut flags: Vec<bool> = self.rows.iter().map(|r| r.include).collect();
                        apply_alt_include(&mut flags, i);
                        for (r, v) in self.rows.iter_mut().zip(flags) {
                            r.include = v;
                        }
                    }
                    // Shift+click range fill skipped when Alt solo ran.
                    // The span is VIEW positions, mapped back through `view`.
                    if alt_solo.is_none()
                        && let Some((lo, hi, v)) = shift_range
                        && hi < view.len()
                    {
                        for &u in &view[lo..=hi] {
                            self.rows[u].include = v;
                        }
                    }
                    if let Some(a) = anchor_next {
                        self.include_anchor = Some(a);
                    }
                    // Header sort click: cycle best-first → reversed →
                    // insertion order (session-only; run/state order stays
                    // insertion).
                    if let Some(col) = sort_click {
                        self.sort_spec = cycle_sort(self.sort_spec, col, self.cell_stat);
                    }
                    // Header per-metric Reset: one column back to Idle.
                    if let Some(kind) = reset_click {
                        self.reset_metric(kind);
                    }
                    if let Some(path) = open_path
                        && let Err(e) = open::that(&path)
                    {
                        log::error!(target: "rfmetrics::app", "open \"{path}\" failed: {e}");
                        self.toast = Some(Toast {
                            text: format!("Could not open file: {e}"),
                            until: now + TOAST_SECS,
                            kind: ToastKind::Error,
                        });
                    }
                    if let Some(path) = reveal_path
                        && let Err(e) = reveal_in_explorer(&path)
                    {
                        log::error!(target: "rfmetrics::app", "reveal \"{path}\" failed: {e}");
                        self.toast = Some(Toast {
                            text: format!("Could not show in explorer: {e}"),
                            until: now + TOAST_SECS,
                            kind: ToastKind::Error,
                        });
                    }
                    self.hovered_now = hovered_next;
                });
            });
            self.table_rect = Some(table_resp.response.rect);
            // Ctrl/Cmd+A select-all for the removal set, scoped to
            // pointer-over-table (the "focused" proxy — egui tables take no
            // keyboard focus). Runs after the table so a focused TextEdit
            // (ref path/trim boxes, built above) consumes the key first and
            // keeps its select-all-text behavior.
            if !self.rows.is_empty()
                && self.table_rect.is_some_and(|r| ui.rect_contains_pointer(r))
                && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::A))
            {
                for r in &mut self.rows {
                    r.selected = true;
                }
                // Anchor at the end: a follow-up Shift+click on a selected
                // row unselects the trailing range (checkbox-style).
                self.selected_anchor = Some(self.rows.len() - 1);
            }
            // Roll the delayed-hover timer forward; schedule one wake-up
            // for when the delay elapses so the outline appears without
            // moving (no every-frame spin while pending).
            if self.hovered_now != self.hover_row {
                self.hover_row = self.hovered_now;
                self.hover_since = self.hover_row.map(|_| now);
            }
            let hover_pending = matches!(
                (self.hover_row, self.hovered_now, self.hover_since),
                (Some(a), Some(b), Some(t)) if a == b && now - t < ROW_HOVER_DELAY
            );
            if hover_pending {
                // `hover_pending` implies `hover_since` is `Some`.
                let remaining = self
                    .hover_since
                    .map(|t| (ROW_HOVER_DELAY - (now - t)).max(0.0))
                    .unwrap_or(ROW_HOVER_DELAY);
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(remaining));
            }
        });

        // Drop hint overlay pinned over the active drop target
        if hovering {
            let active_hint = if ref_hover {
                self.ref_rect
                    .map(|r| ("drop_hint_ref", r, "Drop video to set as reference"))
            } else if table_hover {
                self.table_rect
                    .map(|r| ("drop_hint_table", r, "Drop files to queue"))
            } else {
                None
            };

            if let Some((id, r, text)) = active_hint {
                egui::Area::new(egui::Id::new(id))
                    .fixed_pos(r.center_top())
                    .pivot(egui::Align2::CENTER_TOP)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.label(text);
                        });
                    });
            }
        }

        // Transient toast (e.g. extra files dropped on the reference box).
        // Static text needs exactly two frames (show + hide): schedule one
        // wake-up at expiry instead of full-rate repaints for 3 s.
        if let Some(toast) = self.toast.clone() {
            if now < toast.until {
                let remaining = (toast.until - now).max(0.0);
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(remaining));
                let corner = ui.max_rect().right_bottom();
                let mut frame = egui::Frame::popup(ui.style());
                if let Some(outline) = toast.kind.outline() {
                    frame = frame.stroke(egui::Stroke::new(1.5, outline));
                }
                egui::Area::new(egui::Id::new("toast"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(corner + egui::vec2(-10.0, -10.0))
                    .pivot(egui::Align2::RIGHT_BOTTOM)
                    .show(ui.ctx(), |ui| {
                        frame.show(ui, |ui| {
                            ui.label(&toast.text);
                        });
                    });
            } else {
                self.toast = None;
            }
        }

        // Metric plot viewport (own OS window while open).
        self.show_plots(ui.ctx());
        // Bad-frames viewer viewport (own OS window while open).
        self.show_badframes(ui.ctx());
    }

    fn on_exit(&mut self) {
        // Debounced writes can still be pending: flush the final second.
        // `save` is atomic (tmp + rename) and log-only, so this never
        // toasts or blocks shutdown.
        if self.is_state_dirty() {
            let snap = self.snapshot();
            let _ = crate::state::save(&snap);
            self.saved_snapshot = snap;
        }
    }
}
#[cfg(test)]
#[path = "tests/test_app.rs"]
mod tests;
