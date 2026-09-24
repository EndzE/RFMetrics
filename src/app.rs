pub(crate) use crate::app_queue::{
    DropAction, METRIC_COLUMNS, QueueRow, SortColumn, SortDir, VIDEO_EXTS, apply_alt_include,
    cycle_sort, done_is_stale, norm_key, reveal_in_explorer, route_drop, shift_include_range,
    sort_view,
};
use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

// Queue domain lives in `crate::app_queue` (re-exported above).
// UI primitives live in `crate::app_widgets` (re-exported below).
pub(crate) use crate::app_widgets::{
    ROW_HOVER_DELAY, TOAST_SECS, Toast, ToastKind, WARN_TEXT, get_cursor_pos, metric_stat_tooltip,
    paint_running_sweep, panel_frame, rank_fill, sort_mark, vline,
};

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

// State persistence (`snapshot`/`apply_state`/`is_state_dirty`/`autosave_tick`
// /`consume_autosave`/`save_results`/`untick_unsupported_metrics`) lives in
// `crate::app_state`.
// UI primitives (`vline`/`sort_mark`/…) live in `crate::app_widgets`.
// (moved to `crate::app_widgets`; re-exported above)

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

        self.show_reference(ui, run_locked, ref_hover);

        self.show_actions(ui, now, run_locked);

        self.show_options(ui, run_locked);

        // ---- File queue (center, expanding) ----
        self.show_queue(ui, now, run_locked, table_hover);

        self.show_overlays(ui, now, hovering, ref_hover, table_hover);

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
