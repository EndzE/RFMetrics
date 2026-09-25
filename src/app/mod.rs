pub(crate) mod badframes;
pub(crate) mod config;
pub(crate) mod panels;
pub(crate) mod persist;
pub(crate) mod plots;
pub(crate) mod queue;
pub(crate) mod run;
pub(crate) mod widgets;

pub(crate) use crate::app::queue::{
    DropAction, METRIC_COLUMNS, SortColumn, VIDEO_EXTS, apply_alt_include, cycle_sort,
    done_is_stale, norm_key, reveal_in_explorer, route_drop, shift_include_range, sort_view,
};
use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

// Queue domain lives in `crate::app::queue` (re-exported above).
// UI primitives live in `crate::app::widgets` (re-exported below).
pub(crate) use crate::app::widgets::{
    ROW_HOVER_DELAY, TOAST_SECS, Toast, ToastKind, WARN_TEXT, get_cursor_pos, metric_stat_tooltip,
    paint_running_sweep, panel_frame, rank_fill, sort_mark, vline,
};

// Shared worker-clock helper, re-exported for the domain modules.
pub(crate) use crate::app::run::wall_now_string;

/// Session-only UI chrome + persistence bookkeeping. Never persisted
/// itself — it *holds* the last-written snapshot for dirty-checking.
pub(crate) struct UiState {
    pub(crate) toast: Option<Toast>,
    /// Last state actually written to `ffmetrics-state.json`; the per-frame
    /// snapshot compares against this so only real changes arm a write.
    pub(crate) saved_snapshot: crate::state::AppState,
    /// Egui time of the first unsaved change (`None` = clean).
    pub(crate) pending_save_since: Option<f64>,
    /// One-shot: the state-fallback/failure toast already fired, so a
    /// read-only install toasts once instead of every debounce.
    pub(crate) state_fallback_toasted: bool,
}

impl UiState {
    pub(crate) fn toast(&mut self, now: f64, text: String, kind: crate::app::ToastKind) {
        match kind {
            crate::app::ToastKind::Info => {
                log::info!(target: "rfmetrics::app", "toast info: {text}")
            }
            crate::app::ToastKind::Warning => {
                log::warn!(target: "rfmetrics::app", "toast warning: {text}")
            }
            crate::app::ToastKind::Error => {
                log::error!(target: "rfmetrics::app", "toast error: {text}")
            }
        }
        self.toast = Some(crate::app::Toast {
            text,
            until: now + crate::app::TOAST_SECS,
            kind,
        });
    }
}

/// Facade: pure composition + cross-cutting orchestration. Leaf state and
/// its operations live on the sub-structs (`Config`, `QueueModel`,
/// `Binaries`, `RefProbe`, `RunRuntime`, `PlotRuntime`,
/// `BadframesRuntime`, `UiState`); the `impl RFMetricsApp` blocks that
/// remain (worker drains, `start_run`, persistence glue, `show_*` views)
/// inherently span several sub-structs, so `&mut self` is their honest
/// signature — not a regression to the god object.
pub struct RFMetricsApp {
    /// Persisted run configuration (reference, metrics, VMAF, view, export).
    pub(crate) config: config::Config,
    /// Queue rows + session-only table chrome.
    pub(crate) queue: queue::QueueModel,
    /// Discovered binaries + startup probe channel.
    pub(crate) binaries: run::Binaries,
    /// Reference probe + thumbnail state.
    pub(crate) ref_probe: run::RefProbe,
    /// Metric-run runtime (channels, progress, pending summaries).
    pub(crate) run: run::RunRuntime,
    /// Plot viewport state.
    pub(crate) plots: plots::PlotRuntime,
    /// Bad-frames extract + viewer state.
    pub(crate) badframes: badframes::BadframesRuntime,
    /// Toast + state-file bookkeeping.
    pub(crate) ui: UiState,
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
            config: config::Config {
                reference: config::ReferenceInputs {
                    path: String::new(),
                    duration: String::new(),
                    skip: String::new(),
                    pixfmt: crate::metrics::ffmpeg::RefPixFmt::default(),
                },
                metrics: config::MetricToggles {
                    psnr: false,
                    ssim: false,
                    vmaf: true,
                    xpsnr: false,
                    ssim2: false,
                    butteraugli: false,
                    cvvdp: false,
                },
                vmaf: config::VmafOpts {
                    model: "vmaf_v0.6.1.json".to_owned(),
                    phone: false,
                    scale: false,
                    pooling: "Mean".to_owned(),
                    subsample: "1".to_owned(),
                    threads: "auto".to_owned(),
                    models: crate::metrics::vmaf::list_models(
                        &crate::metrics::vmaf::vmaf_home().join("vmaf-models"),
                    ),
                },
                view: config::ViewOpts {
                    scale_method: ScaleMethod::default(),
                    fps_mode: crate::metrics::ffmpeg::InputFpsMode::default(),
                    cell_stat: crate::metrics::CellStat::default(),
                    cell_precision: crate::metrics::DEFAULT_PRECISION as u8,
                    plot_size: crate::plot::PlotSize::default(),
                    plot_at_start: false,
                },
                export: config::ExportOpts {
                    csv_export: false,
                    csv_dir: String::new(),
                    results_autosave: false,
                    results_path: String::new(),
                    badframes_count: "5".to_owned(),
                    badframes_export_dir: String::new(),
                },
            },
            queue: queue::QueueModel {
                rows: Vec::new(),
                next_color_idx: 0,
                next_probe_seq: 0,
                sort_spec: None,
                include_anchor: None,
                selected_anchor: None,
                hover_row: None,
                hover_since: None,
                hovered_now: None,
                ref_rect: None,
                table_rect: None,
            },
            binaries: run::Binaries {
                ffmpeg,
                ffvship,
                ffprobe,
                probing: bins_probing,
                bin_rx,
            },
            ref_probe: run::RefProbe {
                info: crate::probe::reference_media_text("", None).0,
                info_data: None,
                info_path: String::new(),
                last_spawned: String::new(),
                generation: 0,
                probe_tx,
                probe_rx,
                timeout_note: None,
                thumb_tx,
                thumb_rx,
                thumb_tex: None,
                thumb_loading: false,
                last_thumb_path: String::new(),
                thumb_generation: 0,
            },
            run: run::RunRuntime {
                measuring: false,
                abort: Arc::new(AtomicBool::new(false)),
                current_child: Arc::new(Mutex::new(None)),
                pending: 0,
                generation: 0,
                live_kind: None,
                live_key: None,
                metric_tx,
                metric_rx,
                csv_report: None,
                results_autosave_pending: false,
            },
            plots: plots::PlotRuntime {
                open: false,
                tab: MetricKind::Psnr,
                tabs_w: 0.0,
                live_fit: None,
                follow_pending: false,
                reset_pending: false,
                snap: false,
                save_pending: None,
                png_tx,
                png_rx,
                png_saving: false,
            },
            badframes: badframes::BadframesRuntime {
                tx: badframe_tx,
                rx: badframe_rx,
                busy: false,
                done: 0,
                total: 0,
                abort: Arc::new(AtomicBool::new(false)),
                report: None,
                open: false,
                tab: MetricKind::Psnr,
                file: None,
                frame_pos: 0,
                reset_once: false,
                view_key: None,
                slider: false,
                split: 0.5,
                div_sx: f32::NAN,
                div_drag: false,
                tex_dist: None,
                tex_ref: None,
                tex_key: None,
                show_diff: false,
                tex_diff: None,
                tmp: crate::metrics::badframes::tmp_dir(),
                tmp_cleanup_pending: false,
                files: Vec::new(),
                export_pending: None,
            },
            ui: UiState {
                toast: None,
                saved_snapshot: crate::state::AppState::default(),
                pending_save_since: None,
                state_fallback_toasted: false,
            },
        };
        // Hermetic tests: the developer's own state file must not leak
        // into assertions about defaults.
        let loaded = if cfg!(test) {
            None
        } else {
            crate::state::load()
        };
        app.apply_state(loaded);
        app.ui.saved_snapshot = app.snapshot();
        app
    }
}

// State persistence lives in `crate::app::persist`; UI primitives in
// `crate::app::widgets`.

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
        let hovering = hovering && !self.run.measuring;

        // Continuously repaint while dragging so hover outlines update smoothly
        if hovering {
            ui.ctx().request_repaint();
        }

        let cursor_pos = get_cursor_pos(ui.ctx());
        let now = ui.ctx().input(|i| i.time);
        // While the metric worker runs, run-scoped inputs lock: dimmed and
        // unclickable so paths, trim, queue, and toggles can't shift mid-run.
        let run_locked = self.run.measuring || self.badframes.busy;

        // Direct OS-cursor hit test; winit gives no position during OLE drags.
        let is_over_ref = matches!((cursor_pos, self.queue.ref_rect), (Some(pos), Some(rect)) if rect.contains(pos));
        let is_over_table = matches!((cursor_pos, self.queue.table_rect), (Some(pos), Some(rect)) if rect.contains(pos));

        // Strict target routing (Python parity: a drop outside a target
        // does nothing). The reference box takes one file; extras are
        // reported via toast instead of silently vanishing.
        match route_drop(self.run.measuring, is_over_ref, is_over_table, dropped) {
            DropAction::Ignore => {}
            DropAction::Blocked => {
                self.ui.toast(
                    now,
                    "Stop the run before changing files".to_owned(),
                    ToastKind::Info,
                );
            }
            DropAction::SetRef { first, extra } => {
                self.config.reference.path = first.to_string_lossy().into_owned();
                if extra > 0 {
                    self.ui.toast(
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
        if let Some(name) = self.ref_probe.timeout_note.take() {
            self.ui
                .toast(now, format!("Probe timed out: {name}"), ToastKind::Warning);
        }
        let ctx = ui.ctx().clone();
        let live = self.refresh_thumbnail(&ctx) | live;
        let live = self.drain_metric_results() | live;
        if let Some((ok, errors)) = self.run.csv_report.take() {
            if errors.is_empty() {
                let s = if ok == 1 { "" } else { "s" };
                self.ui
                    .toast(now, format!("Saved {ok} CSV file{s}"), ToastKind::Info);
            } else {
                let mut first = errors[0].clone();
                if first.chars().count() > 80 {
                    first = format!("{}…", first.chars().take(79).collect::<String>());
                }
                let s = if errors.len() == 1 { "" } else { "s" };
                self.ui.toast(
                    now,
                    format!("CSV export failed for {} file{s}: {first}", errors.len()),
                    ToastKind::Error,
                );
            }
        }
        self.consume_autosave(now);
        let live = self.plots.drain_png_results(&ctx, now, &mut self.ui) | live;
        let live = self.drain_badframe_results(now) | live;
        if live {
            ui.ctx().request_repaint();
        } else if self.run.measuring || self.badframes.busy {
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
            self.ui.saved_snapshot = snap;
        }
    }
}
#[cfg(test)]
#[path = "../tests/test_app.rs"]
mod tests;
