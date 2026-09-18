use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct QueueRow {
    path: String,
    /// `norm_key(path)` computed once at insert; `path` is never mutated
    /// after push, so worker-message routing compares this instead of
    /// re-normalizing (and re-hitting `current_dir()`) per row per message.
    key: String,
    display: String,
    include: bool,
    selected: bool,
    media: String,
    media_tip: String,
    info: Option<crate::probe::MediaInfo>,
    psnr: crate::metrics::MetricCell,
    ssim: crate::metrics::MetricCell,
    vmaf: crate::metrics::MetricCell,
    xpsnr: crate::metrics::MetricCell,
    ssim2: crate::metrics::MetricCell,
    butter: crate::metrics::MetricCell,
    cvvdp: crate::metrics::MetricCell,
    psnr_cache: CachedStats,
    ssim_cache: CachedStats,
    vmaf_cache: CachedStats,
    xpsnr_cache: CachedStats,
    ssim2_cache: CachedStats,
    butter_cache: CachedStats,
    cvvdp_cache: CachedStats,
}

/// Cached per-row stats + cross-row ranks for one metric column (H1: the
/// values-vec clone+sort in `DoneStats::new` and the rank scan run on
/// result arrival, not per frame; the render loop only reads).
/// `points` is the same idea for the plot: built/extended on arrival,
/// so the render loop never rebuilds `PlotPoints` per frame.
#[derive(Debug, Clone, Default)]
struct CachedStats {
    stats: Option<crate::metrics::DoneStats>,
    ranks: [crate::metrics::StatRank; 10],
    points: Vec<egui_plot::PlotPoint>,
    /// Rendered Done text (`format!("{avg:.4}")`), frozen at Done arrival
    /// so the table loop never formats per frame. Cleared wherever `stats`
    /// is cleared (rerun start, Reset via wholesale `default()`).
    text: String,
}

impl QueueRow {
    fn cell(&self, kind: MetricKind) -> &crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &self.psnr,
            MetricKind::Ssim => &self.ssim,
            MetricKind::Vmaf => &self.vmaf,
            MetricKind::Xpsnr => &self.xpsnr,
            MetricKind::Ssim2 => &self.ssim2,
            MetricKind::But => &self.butter,
            MetricKind::Cvvdp => &self.cvvdp,
        }
    }

    fn cell_mut(&mut self, kind: MetricKind) -> &mut crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &mut self.psnr,
            MetricKind::Ssim => &mut self.ssim,
            MetricKind::Vmaf => &mut self.vmaf,
            MetricKind::Xpsnr => &mut self.xpsnr,
            MetricKind::Ssim2 => &mut self.ssim2,
            MetricKind::But => &mut self.butter,
            MetricKind::Cvvdp => &mut self.cvvdp,
        }
    }

    fn cached(&self, kind: MetricKind) -> &CachedStats {
        match kind {
            MetricKind::Psnr => &self.psnr_cache,
            MetricKind::Ssim => &self.ssim_cache,
            MetricKind::Vmaf => &self.vmaf_cache,
            MetricKind::Xpsnr => &self.xpsnr_cache,
            MetricKind::Ssim2 => &self.ssim2_cache,
            MetricKind::But => &self.butter_cache,
            MetricKind::Cvvdp => &self.cvvdp_cache,
        }
    }

    fn cached_mut(&mut self, kind: MetricKind) -> &mut CachedStats {
        match kind {
            MetricKind::Psnr => &mut self.psnr_cache,
            MetricKind::Ssim => &mut self.ssim_cache,
            MetricKind::Vmaf => &mut self.vmaf_cache,
            MetricKind::Xpsnr => &mut self.xpsnr_cache,
            MetricKind::Ssim2 => &mut self.ssim2_cache,
            MetricKind::But => &mut self.butter_cache,
            MetricKind::Cvvdp => &mut self.cvvdp_cache,
        }
    }
}

/// File picker extensions (FFMetrics.conf `VideoFilesList` parity).
const VIDEO_EXTS: &[&str] = &[
    "264", "avi", "avs", "h264", "hevc", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts",
    "mxf", "ts", "webm",
];

/// Python `normcase(abspath)` equivalent for the same-file guard rail.
fn norm_key(p: &str) -> String {
    let path = Path::new(p);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let s = abs.to_string_lossy().replace('/', "\\");
    #[cfg(windows)]
    let s = s.to_lowercase();
    s
}

/// Reveal a queued file in the OS file manager without blocking the UI.
/// Windows selects the file (`explorer /select,`); other platforms open the
/// containing folder (select-on-open has no portable equivalent).
/// Spawn-only: never waits on the child, so a slow Explorer can't freeze a frame.
fn reveal_in_explorer(path: &str) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer")
            .args(["/select,", path])
            .spawn()
            .map(|_| ())
    }
    #[cfg(not(windows))]
    {
        let target = Path::new(path)
            .parent()
            .map(|p| p.as_os_str().to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| path.to_owned());
        open::that(&target)
    }
}

/// Thumbnail seek duration: reuse the completed reference probe's
/// duration when it belongs to the current path; otherwise `None` and the
/// worker falls back to a dedicated probe (`media_duration`).
fn thumb_duration(ref_path: &str, ref_info_path: &str, probed: Option<f64>) -> Option<f64> {
    if !ref_path.is_empty() && ref_path == ref_info_path {
        probed.filter(|&d| d > 0.0)
    } else {
        None
    }
}

/// Shortest unique trailing-path suffix per entry (Python `_display_names`).
fn display_names(paths: &[String]) -> Vec<String> {
    let parts: Vec<Vec<String>> = paths
        .iter()
        .map(|p| {
            Path::new(p)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .collect();
    let sep = std::path::MAIN_SEPARATOR.to_string();
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| {
            for n in 1..=part.len() {
                let cand = &part[part.len() - n..];
                let unique = parts.iter().enumerate().all(|(j, q)| {
                    j == i || {
                        let tail = if q.len() >= n {
                            &q[q.len() - n..]
                        } else {
                            &q[..]
                        };
                        tail != cand
                    }
                });
                if unique {
                    return cand.join(&sep);
                }
            }
            part.join(&sep)
        })
        .collect()
}

/// Retrieves the cursor position in egui's logical point coordinates.
/// During Windows OLE file drags, winit omits pointer move events, so egui's
/// internal pointer state is None/stale. We query the OS cursor directly.
#[cfg(windows)]
fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
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
fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

/// Hover highlight delay (seconds) so passing over rows while aiming
/// at text to copy doesn't flash each row.
const ROW_HOVER_DELAY: f64 = 0.1;

/// How long the drop toast (e.g. ignored extra reference files) stays up.
const TOAST_SECS: f64 = 3.0;

/// Toast severity; drives the outline color. `Info` keeps the default
/// popup outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToastKind {
    Info,
    Warning,
    Error,
}

impl ToastKind {
    fn outline(self) -> Option<egui::Color32> {
        match self {
            ToastKind::Info => None,
            ToastKind::Warning => Some(egui::Color32::from_rgb(0xD9, 0xA4, 0x06)),
            ToastKind::Error => Some(egui::Color32::from_rgb(0xE0, 0x4B, 0x4B)),
        }
    }
}

#[derive(Debug, Clone)]
struct Toast {
    text: String,
    until: f64,
    kind: ToastKind,
}

/// Results sent back from background probe threads. The UI thread never
/// blocks on ffprobe; it drains these each frame via `try_recv`.
#[derive(Debug)]
enum ProbeMsg {
    Reference {
        generation: u64,
        text: String,
        info: Option<crate::probe::MediaInfo>,
    },
    RowMedia {
        key: String,
        media: String,
        tip: String,
        info: Option<crate::probe::MediaInfo>,
    },
}

/// Thumbnail result from the dedicated ffmpeg worker (separate channel so
/// slow frame extracts never block fast ffprobe text results).
struct ThumbMsg {
    generation: u64,
    image: Option<egui::ColorImage>,
}

/// PNG export result from the one-shot saver thread. The supersampled
/// render + Lanczos3 downscale blocks for seconds, so it never runs on the
/// UI thread; the worker sends the outcome back here for a toast. Copy jobs
/// send pixels back because `ctx.copy_image()` must run on the UI thread
/// (winit executes it as a frame-end `OutputCommand`).
enum PngSaveMsg {
    Saved { path: std::path::PathBuf },
    CopyReady { w: u32, h: u32, rgba: Vec<u8> },
    SaveFailed { err: String },
    CopyFailed { err: String },
}

/// Pending plot export: file save (filename captured at click time) or
/// clipboard copy. Executed in the central panel where the plot id scope
/// (for the current view bounds) lives.
enum PlotExport {
    Save { name: String },
    Copy,
}

/// Progress + results from the single sequential metric worker (Python
/// `_worker` parity: one thread, checked metrics in order, never on the UI
/// thread). `generation` drops late messages after a Reset starts a new run.
#[derive(Debug)]
enum MetricMsg {
    Progress {
        generation: u64,
        kind: MetricKind,
        key: String,
        frame: u64,
    },
    /// Live per-frame value deltas for the running job's plot curve
    /// (throttled worker-side); appended to `Running.values` in arrival
    /// order, replaced by the strict full series on `Done`.
    Series {
        generation: u64,
        kind: MetricKind,
        key: String,
        new_values: Vec<f64>,
    },
    Done {
        generation: u64,
        kind: MetricKind,
        key: String,
        values: Vec<f64>,
        avg: Option<f64>,
        exec_s: f64,
        error: Option<String>,
        /// Trim settings the run used; stamped onto the `Done` cell so a
        /// rerun under different skip/clip recomputes instead of skipping.
        skip: Option<f64>,
        clip_dur: Option<f64>,
        /// VMAF settings the run used (`Some` for VMAF jobs only); stamped
        /// onto the `Done` cell so an options change recomputes VMAF alone.
        vmaf_cfg: Option<crate::metrics::vmaf::VmafCfg>,
        /// Scaling method the run used; stamped onto the `Done` cell so a
        /// method change recomputes every ffmpeg-backed column (FFVship
        /// has no scale stage and ignores it at compare time).
        scaler: ScaleMethod,
    },
    /// End of the worker loop; `aborted` settles still-Running cells to
    /// Idle while keeping finished (`Done`) results on screen.
    Finished { generation: u64, aborted: bool },
    /// CSV export summary from the worker (sent once before `Finished`
    /// when export was enabled): files written vs. error strings.
    CsvReport {
        generation: u64,
        ok: usize,
        errors: Vec<String>,
    },
}

pub struct RFMetricsApp {
    ref_path: String,
    duration: String,
    skip: String,
    m_psnr: bool,
    m_ssim: bool,
    m_vmaf: bool,
    m_xpsnr: bool,
    m_ssim2: bool,
    m_but: bool,
    m_cvvdp: bool,
    vmaf_model: String,
    vmaf_phone: bool,
    vmaf_scale: bool,
    vmaf_pooling: String,
    vmaf_subsample: String,
    vmaf_threads: String,
    vmaf_models: Vec<String>,
    /// Global scaling method for every `scale=` the app emits.
    scale_method: ScaleMethod,
    /// Open the plot viewport when a run starts (Options checkbox).
    plot_at_start: bool,
    /// Save per-frame metric CSVs on Done (Options checkbox).
    csv_export: bool,
    /// CSV output folder; empty = beside the distorted file (Options).
    csv_dir: String,
    /// Save PNG / Copy image size preset (Options combobox).
    plot_size: crate::plot::PlotSize,
    rows: Vec<QueueRow>,
    ffmpeg: crate::binaries::BinaryInfo,
    ffvship: crate::binaries::BinaryInfo,
    ffprobe: Option<std::path::PathBuf>,
    ref_info: String,
    /// Probed reference stream; feeds metric filtergraphs (scale/format).
    ref_info_data: Option<crate::probe::MediaInfo>,
    ref_rect: Option<egui::Rect>,
    table_rect: Option<egui::Rect>,
    hover_row: Option<usize>,
    hover_since: Option<f64>,
    hovered_now: Option<usize>,
    toast: Option<Toast>,
    /// Pending CSV summary, set by the CsvReport drain arm and toasted
    /// with a real timestamp at the next UI frame (drain has none).
    /// `(files_written, error_strings)`.
    csv_report: Option<(usize, Vec<String>)>,
    probe_tx: Sender<ProbeMsg>,
    probe_rx: Receiver<ProbeMsg>,
    metric_tx: Sender<MetricMsg>,
    metric_rx: Receiver<MetricMsg>,
    /// True while the metric worker runs; the button flips Start↔Stop then.
    measuring: bool,
    /// Stop flag shared with the worker (checked between jobs + in `run_psnr`).
    abort: Arc<AtomicBool>,
    /// The live ffmpeg child, so Stop can kill the in-flight run.
    current_child: Arc<Mutex<Option<std::process::Child>>>,
    /// Jobs still Blocking; last `Done` clears `measuring`.
    pending: usize,
    /// Bumped per run; late worker messages after a Reset are stale.
    run_generation: u64,
    /// Path last handed to a probe worker (or resolved cheaply without one).
    last_spawned_ref: String,
    /// Path the current `ref_info_data` was probed from (set when its
    /// worker result lands). The thumbnail worker reuses its duration only
    /// on a match — `ref_info_data` alone lags one probe behind on ref
    /// change and can't say which path it belongs to.
    ref_info_path: String,
    /// Bumped on every ref change; worker results with an older generation
    /// are stale (typed-through) and discarded.
    ref_generation: u64,
    thumb_tx: Sender<ThumbMsg>,
    thumb_rx: Receiver<ThumbMsg>,
    thumb_tex: Option<egui::TextureHandle>,
    thumb_loading: bool,
    last_thumb_path: String,
    thumb_generation: u64,
    /// Last state actually written to `ffmetrics-state.json`; the per-frame
    /// snapshot compares against this so only real changes arm a write.
    saved_snapshot: crate::state::AppState,
    /// Egui time of the first unsaved change (`None` = clean).
    pending_save_since: Option<f64>,
    /// Metric kind of the currently executing job (last kind seen on the
    /// Progress/Series feed); drives plot tab-follow while measuring.
    live_kind: Option<MetricKind>,
    /// PSNR plot viewport open (Python `plot["win"]` parity: closing the
    /// window withdraws it, Plot reopens it).
    show_plot: bool,
    /// Selected plot viewport tab (session-only, like the Python window).
    plot_tab: MetricKind,
    /// Last measured tab-strip box width, for centering the strip
    /// (session-only; texts are static so it converges in one frame).
    plot_tabs_w: f32,
    /// Grow-only live fit per open tab while any series is running;
    /// cleared once all settle, so finished graphs fit exactly again.
    plot_live_fit: Option<(MetricKind, crate::plot::FitBounds)>,
    /// Follow poke still owed: set when a live phase starts without plot
    /// memory present (window just opened), retried until it lands.
    plot_follow_pending: bool,
    /// Reset-view click still owed: the help-bar `Ui` scopes persistent
    /// ids differently than the canvas `Ui`, so the button only arms this
    /// flag and the central panel (plot id scope) executes the poke.
    /// Retried until plot memory exists, like the follow poke.
    plot_reset_pending: bool,
    /// Snap-to-data lock (plot window checkbox, session-only): panning is
    /// clamped to the first/last frame on x and the plotted min/max on y;
    /// zooming and in-limits panning stay free.
    plot_snap: bool,
    /// Pending plot export (Save PNG / Copy button), executed in the
    /// central panel where the plot id scope lives.
    plot_save_pending: Option<PlotExport>,
    /// Plot export worker channel + busy flag: while `png_saving` both the
    /// Save PNG and Copy buttons are disabled so 5 s renders can't overlap.
    png_tx: Sender<PngSaveMsg>,
    png_rx: Receiver<PngSaveMsg>,
    png_saving: bool,
}

impl Default for RFMetricsApp {
    fn default() -> Self {
        let ffmpeg = crate::binaries::ffmpeg_info();
        let ffvship = crate::binaries::ffvship_info();
        let ffprobe = crate::binaries::ffprobe_path(ffmpeg.path.as_deref());
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (metric_tx, metric_rx) = std::sync::mpsc::channel();
        let (png_tx, png_rx) = std::sync::mpsc::channel();
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
            plot_at_start: false,
            csv_export: false,
            csv_dir: String::new(),
            plot_size: crate::plot::PlotSize::default(),
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
            toast: None,
            csv_report: None,
            probe_tx,
            probe_rx,
            metric_tx,
            metric_rx,
            measuring: false,
            abort: Arc::new(AtomicBool::new(false)),
            current_child: Arc::new(Mutex::new(None)),
            pending: 0,
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
            live_kind: None,
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
    /// Apply any probe results that arrived since the last frame. Stale
    /// reference results (typed-through while a worker was running) are
    /// dropped via the generation check.
    /// Drains the probe channel; returns whether any message arrived (even
    /// a stale one — callers use it to decide on a repaint, and one extra
    /// frame on a rare stale message is harmless).
    fn drain_probe_results(&mut self) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.probe_rx.try_recv() {
            activity = true;
            match msg {
                ProbeMsg::Reference {
                    generation,
                    text,
                    info,
                } => {
                    if generation == self.ref_generation {
                        self.ref_info = text;
                        self.ref_info_data = info;
                        // No newer spawn happened since (same generation),
                        // so `last_spawned_ref` is the path this probed.
                        self.ref_info_path = self.last_spawned_ref.clone();
                    } else {
                        log::debug!(target: "rfmetrics::app", "discarded stale ref probe (gen {generation})");
                    }
                }
                ProbeMsg::RowMedia {
                    key,
                    media,
                    tip,
                    info,
                } => {
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key) {
                        row.media = media;
                        row.media_tip = tip;
                        row.info = info;
                    }
                }
            }
        }
        activity
    }

    /// Re-probe only when the path actually changed, and only off the UI
    /// thread: cheap cases (empty/missing/no ffprobe) resolve inline, an
    /// existing file spawns a worker and shows "Probing…" meanwhile.
    /// Returns the probe drain flag (spawns stem from input frames, which
    /// repaint on their own).
    fn refresh_ref_info(&mut self) -> bool {
        let activity = self.drain_probe_results();
        if self.ref_path == self.last_spawned_ref {
            return activity;
        }
        self.last_spawned_ref = self.ref_path.clone();
        self.ref_generation = self.ref_generation.wrapping_add(1);
        if self.ref_path.trim().is_empty() {
            self.ref_info =
                "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-"
                    .to_owned();
            self.ref_info_data = None;
            return activity;
        }
        if !Path::new(&self.ref_path).is_file() {
            self.ref_info = "File not found".to_owned();
            self.ref_info_data = None;
            return activity;
        }
        if self.ffprobe.is_none() {
            self.ref_info = "ffprobe not found".to_owned();
            self.ref_info_data = None;
            return activity;
        }
        self.ref_info = "Probing…".to_owned();
        let tx = self.probe_tx.clone();
        let generation = self.ref_generation;
        let path = self.ref_path.clone();
        let exe = self.ffprobe.clone();
        std::thread::spawn(move || {
            let (text, info) = crate::probe::reference_media_text(&path, exe.as_deref());
            let _ = tx.send(ProbeMsg::Reference {
                generation,
                text,
                info,
            });
        });
        activity
    }

    /// Apply arrived thumbnails; stale generations (typed-through) are dropped.
    /// Returns whether any message arrived (see `drain_probe_results`).
    fn drain_thumbs(&mut self, ctx: &egui::Context) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.thumb_rx.try_recv() {
            activity = true;
            if msg.generation != self.thumb_generation {
                log::debug!(target: "rfmetrics::app", "discarded stale thumbnail (gen {})", msg.generation);
                continue;
            }
            self.thumb_loading = false;
            match msg.image {
                Some(img) => {
                    self.thumb_tex =
                        Some(ctx.load_texture("ref_thumb", img, egui::TextureOptions::LINEAR));
                }
                None => self.thumb_tex = None,
            }
        }
        activity
    }

    /// Spawn a dedicated ffmpeg worker when the ref path changed. Cheap cases
    /// clear inline; the worker sends duration-aware extracts back on the
    /// thumb channel and repaints via the cloned ctx. Returns the thumb
    /// drain flag (spawns stem from input frames, which repaint on their own).
    fn refresh_thumbnail(&mut self, ctx: &egui::Context) -> bool {
        let activity = self.drain_thumbs(ctx);
        if self.ref_path == self.last_thumb_path {
            return activity;
        }
        self.last_thumb_path = self.ref_path.clone();
        self.thumb_generation = self.thumb_generation.wrapping_add(1);
        self.thumb_tex = None;
        if self.ref_path.trim().is_empty() || !Path::new(&self.ref_path).is_file() {
            self.thumb_loading = false;
            return activity;
        }
        let Some(ffmpeg_exe) = self.ffmpeg.path.clone() else {
            self.thumb_loading = false;
            return activity;
        };
        self.thumb_loading = true;
        let tx = self.thumb_tx.clone();
        let generation = self.thumb_generation;
        let path = self.ref_path.clone();
        let ffprobe_exe = self.ffprobe.clone();
        // Prefer the completed reference probe's duration (same path only);
        // the worker probes itself when the ref probe hasn't landed yet.
        let duration = thumb_duration(
            &self.ref_path,
            &self.ref_info_path,
            self.ref_info_data.as_ref().and_then(|i| i.duration),
        );
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let duration =
                duration.or_else(|| crate::probe::media_duration(&path, ffprobe_exe.as_deref()));
            let image = crate::preview::extract_thumbnail(&ffmpeg_exe, &path, duration);
            let _ = tx.send(ThumbMsg { generation, image });
            ctx.request_repaint();
        });
        activity
    }

    /// Short display names for all rows (Python `_refresh_names`).
    fn refresh_queue_names(&mut self) {
        let paths: Vec<String> = self.rows.iter().map(|r| r.path.clone()).collect();
        for (row, name) in self.rows.iter_mut().zip(display_names(&paths)) {
            row.display = name;
        }
        // Row indices may have shifted; drop stale hover state.
        self.hover_row = None;
        self.hover_since = None;
    }

    /// Re-probe everything (Options "Refresh Files Media Info"): the
    /// reference text + thumbnail and every queue row's media text + raw
    /// info. Same worker channels as the initial probes, so the window
    /// never blocks; results (not reruns of finished metrics) update.
    fn refresh_media_info(&mut self) {
        // Forget the last-spawned markers: the per-frame refreshers see a
        // mismatch and re-probe through the normal path (cheap inline
        // cases resolve without a worker, as before).
        self.last_spawned_ref.clear();
        self.last_thumb_path.clear();
        if self.rows.is_empty() {
            return;
        }
        for row in &mut self.rows {
            row.media = "Probing…".to_owned();
            row.media_tip = "Probing…".to_owned();
        }
        let tx = self.probe_tx.clone();
        let exe = self.ffprobe.clone();
        let paths: Vec<(String, String)> = self
            .rows
            .iter()
            .map(|r| (r.key.clone(), r.path.clone()))
            .collect();
        std::thread::spawn(move || {
            for (key, s) in paths {
                let (media, tip, info) = crate::probe::probe_table_text(&s, exe.as_deref());
                let _ = tx.send(ProbeMsg::RowMedia {
                    key,
                    media,
                    tip,
                    info,
                });
            }
        });
    }

    /// Queue picked files, silently skipping ones already present.
    /// Media probing runs on a worker thread; rows show "Probing…"
    /// until their results arrive, so drops never freeze the window.
    fn add_queue_files(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut seen: HashSet<String> = self.rows.iter().map(|r| r.key.clone()).collect();
        let mut fresh: Vec<(String, String)> = Vec::new();
        for p in paths {
            let s = p.to_string_lossy().into_owned();
            let key = norm_key(&s);
            if !seen.insert(key.clone()) {
                continue; // guard rail: same file already queued
            }
            self.rows.push(QueueRow {
                path: s.clone(),
                key: key.clone(),
                display: String::new(),
                include: true,
                selected: false,
                media: "Probing…".to_owned(),
                media_tip: "Probing…".to_owned(),
                info: None,
                psnr: crate::metrics::MetricCell::Idle,
                ssim: crate::metrics::MetricCell::Idle,
                vmaf: crate::metrics::MetricCell::Idle,
                xpsnr: crate::metrics::MetricCell::Idle,
                ssim2: crate::metrics::MetricCell::Idle,
                butter: crate::metrics::MetricCell::Idle,
                cvvdp: crate::metrics::MetricCell::Idle,
                psnr_cache: CachedStats::default(),
                ssim_cache: CachedStats::default(),
                vmaf_cache: CachedStats::default(),
                xpsnr_cache: CachedStats::default(),
                ssim2_cache: CachedStats::default(),
                butter_cache: CachedStats::default(),
                cvvdp_cache: CachedStats::default(),
            });
            fresh.push((key, s));
        }
        self.refresh_queue_names();
        if fresh.is_empty() {
            return;
        }
        let tx = self.probe_tx.clone();
        let exe = self.ffprobe.clone();
        std::thread::spawn(move || {
            for (key, s) in fresh {
                let (media, tip, info) = crate::probe::probe_table_text(&s, exe.as_deref());
                let _ = tx.send(ProbeMsg::RowMedia {
                    key,
                    media,
                    tip,
                    info,
                });
            }
        });
    }

    /// Apply metric worker results; stale generations (post-Reset) drop.
    /// Progress keeps the max frame per row (dual stdout/stderr feeds).
    /// Returns whether any message arrived (see `drain_probe_results`).
    fn drain_metric_results(&mut self) -> bool {
        let mut scored_changed = false;
        let mut activity = false;
        while let Ok(msg) = self.metric_rx.try_recv() {
            activity = true;
            match msg {
                MetricMsg::Progress {
                    generation,
                    kind,
                    key,
                    frame,
                } => {
                    if generation != self.run_generation {
                        continue;
                    }
                    // The job emitting progress is the live one: the plot
                    // tab follows it while measuring.
                    self.live_kind = Some(kind);
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key)
                        && let crate::metrics::MetricCell::Running { frame: cur, .. } =
                            row.cell_mut(kind)
                        && frame > *cur
                    {
                        *cur = frame;
                    }
                }
                MetricMsg::Series {
                    generation,
                    kind,
                    key,
                    new_values,
                } => {
                    if generation != self.run_generation || new_values.is_empty() {
                        continue;
                    }
                    self.live_kind = Some(kind);
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key)
                        && let crate::metrics::MetricCell::Running { values, .. } =
                            row.cell_mut(kind)
                    {
                        // Points mirror values 1:1 (x = 1-based frame), so
                        // the plot borrows them instead of rebuilding.
                        let base = values.len() as f64;
                        values.extend_from_slice(&new_values);
                        row.cached_mut(kind).points.extend(
                            new_values
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| egui_plot::PlotPoint::new(base + i as f64 + 1.0, v)),
                        );
                    }
                }
                MetricMsg::Done {
                    generation,
                    kind,
                    key,
                    values,
                    avg,
                    exec_s,
                    error,
                    skip,
                    clip_dur,
                    vmaf_cfg,
                    scaler,
                } => {
                    if generation != self.run_generation {
                        log::debug!(target: "rfmetrics::app", "discarded stale {} result", kind.name());
                        continue;
                    }
                    self.pending = self.pending.saturating_sub(1);
                    // First real data for a no-live-feed tab (VMAF): it sat
                    // on the empty default all run, so owe one auto-follow
                    // poke and Done snaps into view. Live-feed metrics
                    // follow mid-run already — refitting those here would
                    // yank a zoom the user is examining. Only when the plot
                    // window is open on this tab and no sibling row shows
                    // data yet (later rows must not disturb the first fit).
                    if !kind.streams_live_values()
                        && self.show_plot
                        && kind == self.plot_tab
                        && error.is_none()
                        && !values.is_empty()
                        && !self.rows.iter().any(|r| {
                            r.key != key
                                && matches!(
                                    r.cell(kind),
                                    crate::metrics::MetricCell::Done { values, .. }
                                    if !values.is_empty()
                                )
                        })
                    {
                        self.plot_follow_pending = true;
                    }
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key) {
                        *row.cell_mut(kind) = match error {
                            // Killed by Stop: settle quietly like unstarted
                            // rows (H4); `Finished{aborted}` below handles
                            // the still-Running ones.
                            Some(msg) if msg == "aborted" => crate::metrics::MetricCell::Idle,
                            Some(msg) => crate::metrics::MetricCell::Error { msg },
                            None => crate::metrics::MetricCell::Done {
                                avg: avg.unwrap_or_else(|| crate::metrics::mean(&values)),
                                values,
                                exec_s,
                                skip,
                                clip_dur,
                                vmaf_cfg,
                                scaler,
                            },
                        };
                        // Cache the stats once (clone+sort lives here, not
                        // per frame); ranks refresh below for this metric.
                        // Points likewise: the plot borrows them instead of
                        // rebuilding `PlotPoints` every frame. Anything but
                        // `Done` clears (stale partials must never render).
                        let stats = row.cell(kind).done_stats();
                        let points = match row.cell(kind) {
                            crate::metrics::MetricCell::Done { values, .. } => values
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| egui_plot::PlotPoint::new(i as f64 + 1.0, v))
                                .collect(),
                            _ => Vec::new(),
                        };
                        // Rendered text frozen once per result (the table loop
                        // borrows it instead of formatting per frame).
                        let text = row.cell(kind).cell_text();
                        row.cached_mut(kind).stats = stats;
                        row.cached_mut(kind).points = points;
                        row.cached_mut(kind).text = text;
                        scored_changed = true;
                    }
                    if self.pending == 0 {
                        self.measuring = false;
                    }
                }
                MetricMsg::Finished {
                    generation,
                    aborted,
                } => {
                    if generation != self.run_generation {
                        continue;
                    }
                    // Aborted runs: unstarted/killed rows were left Running;
                    // settle them to Idle. Finished (`Done`) cells are kept.
                    if aborted {
                        for row in &mut self.rows {
                            for kind in MetricKind::ALL {
                                let cell = row.cell_mut(kind);
                                if matches!(cell, crate::metrics::MetricCell::Running { .. }) {
                                    *cell = crate::metrics::MetricCell::Idle;
                                }
                            }
                        }
                    }
                    self.pending = 0;
                    self.measuring = false;
                }
                MetricMsg::CsvReport {
                    generation,
                    ok,
                    errors,
                } => {
                    if generation != self.run_generation {
                        continue;
                    }
                    // Toasted with a timestamp at the next UI frame below;
                    // all-quiet reports (aborted run, nothing written) stay silent.
                    if ok > 0 || !errors.is_empty() {
                        self.csv_report = Some((ok, errors));
                    }
                }
            }
        }
        // Ranks depend on the whole scored set, so refresh after applying
        // the batch — not per message, and never per frame. The scan itself
        // is trivial (min/max over 10 scalars per scored row, no sorting).
        if scored_changed {
            for kind in MetricKind::ALL {
                self.refresh_ranks(kind);
            }
        }
        activity
    }

    /// Recompute cross-row ranks for one metric from the cached stats.
    /// Call whenever the scored set changes: `Done` landing, a rerun
    /// marking cells `Running`, Reset, or row removal.
    fn refresh_ranks(&mut self, kind: MetricKind) {
        let mut stat_lo = [f64::INFINITY; 10];
        let mut stat_hi = [f64::NEG_INFINITY; 10];
        let mut scored = 0usize;
        for row in &self.rows {
            if let Some(s) = &row.cached(kind).stats {
                scored += 1;
                for (k, (_, v, _)) in s.comparable().iter().enumerate() {
                    stat_lo[k] = stat_lo[k].min(*v);
                    stat_hi[k] = stat_hi[k].max(*v);
                }
            }
        }
        for row in &mut self.rows {
            let cached = row.cached_mut(kind);
            let mut ranks = [crate::metrics::StatRank::Plain; 10];
            if scored >= 2
                && let Some(s) = &cached.stats
            {
                let comp = s.comparable();
                for k in 0..10 {
                    let (_, v, lower_better) = comp[k];
                    // BUTTERAUGLI is lower-is-better on every stat (Python
                    // "lower is better, min 0"); StdDev already is.
                    ranks[k] = if lower_better || kind == MetricKind::But {
                        crate::metrics::rank_low(v, stat_lo[k], stat_hi[k])
                    } else {
                        crate::metrics::rank(v, stat_lo[k], stat_hi[k])
                    };
                }
            }
            cached.ranks = ranks;
        }
    }

    /// Apply plot export thread results; clears the Saving…/Copying…
    /// lock so the buttons re-arm. Copy pixels land here because
    /// `ctx.copy_image()` must run on the UI thread. Runs on the main
    /// viewport each frame. Returns whether any message arrived.
    fn drain_png_results(&mut self, ctx: &egui::Context, now: f64) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.png_rx.try_recv() {
            activity = true;
            self.png_saving = false;
            match msg {
                PngSaveMsg::Saved { path } => {
                    self.toast(
                        now,
                        format!("Plot saved to {}", path.display()),
                        ToastKind::Info,
                    );
                }
                PngSaveMsg::CopyReady { w, h, rgba } => {
                    ctx.copy_image(egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        &rgba,
                    ));
                    self.toast(now, "Plot copied to clipboard".to_owned(), ToastKind::Info);
                }
                PngSaveMsg::SaveFailed { err } => {
                    self.toast(now, format!("Could not save plot: {err}"), ToastKind::Error);
                }
                PngSaveMsg::CopyFailed { err } => {
                    self.toast(now, format!("Could not copy plot: {err}"), ToastKind::Error);
                }
            }
        }
        activity
    }

    fn toast(&mut self, now: f64, text: String, kind: ToastKind) {
        match kind {
            ToastKind::Info => log::info!(target: "rfmetrics::app", "toast info: {text}"),
            ToastKind::Warning => log::warn!(target: "rfmetrics::app", "toast warning: {text}"),
            ToastKind::Error => log::error!(target: "rfmetrics::app", "toast error: {text}"),
        }
        self.toast = Some(Toast {
            text,
            until: now + TOAST_SECS,
            kind,
        });
    }

    /// Parse a trim box; empty means no trim. `None` = invalid ("bad time").
    fn trim_opt(raw: &str) -> Option<Option<f64>> {
        if raw.trim().is_empty() {
            Some(None)
        } else {
            crate::metrics::parse_time_spec(raw).map(Some)
        }
    }

    /// Everything `ffmetrics-state.json` persists, read off the live UI.
    fn snapshot(&self) -> crate::state::AppState {
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
                plot_at_start: Some(self.plot_at_start),
                plot_size: Some(self.plot_size.label().to_owned()),
                csv_export: Some(self.csv_export),
                csv_dir: Some(self.csv_dir.clone()),
            },
        }
    }

    /// Issue #7: force off restored/default ticks for filters this ffmpeg
    /// build lacks (their header checkboxes are disabled, so they were
    /// never ticked live). Session-only capability, never persisted.
    /// Idempotent: safe to run for both the no-file and restored paths.
    fn untick_unsupported_metrics(&mut self) {
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
    }

    /// Apply a loaded state file (tolerant per-key; absent keys keep live
    /// defaults, saved models must still be on disk, queue entries must
    /// still be files). Restored rows probe through the normal path.
    fn apply_state(&mut self, loaded: Option<crate::state::AppState>) {
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
        // Restored ticks obey capability (see helper docs).
        self.untick_unsupported_metrics();
    }

    /// Allocation-free dirty check mirroring `snapshot() != saved_snapshot`
    /// without building `AppState`. Covers every field `snapshot()` sets;
    /// a new persisted field must be added here too, or edits to it will
    /// silently stop saving.
    fn is_state_dirty(&self) -> bool {
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
            || o.plot_at_start != Some(self.plot_at_start)
            || o.plot_size.as_deref() != Some(self.plot_size.label())
            || o.csv_export != Some(self.csv_export)
            || o.csv_dir.as_deref() != Some(self.csv_dir.as_str())
        {
            return true;
        }
        false
    }

    /// Debounced state write (1s after the last detected change): compare
    /// the live snapshot against the last write, arm/re-arm a single
    /// wake-up while dirty, save once it settles.
    fn autosave_tick(&mut self, ctx: &egui::Context, now: f64) {
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
                crate::state::save(&snap);
                self.saved_snapshot = snap;
                self.pending_save_since = None;
            }
            Some(since) => {
                let remaining = (SAVE_DEBOUNCE_SECS - (now - since)).max(0.0);
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(remaining));
            }
        }
    }

    /// Start a run over included rows on one worker thread: each checked
    /// metric runs sequentially in Python `METRICS` order (Python
    /// `start`/`_worker` parity). Pre-flight failures land in the cells
    /// as errors, mirroring Python's `"bad time"` / `"probe failed"` text.
    fn start_run(&mut self, now: f64) {
        if self.measuring {
            return;
        }
        let kinds: Vec<MetricKind> = MetricKind::ALL
            .into_iter()
            .filter(|k| match k {
                // Issue #7 backstop: restored/default ticks for missing
                // filters are forced off at startup, but a ticked-yet-
                // unsupported metric must never reach the worker either.
                MetricKind::Psnr => self.m_psnr && self.ffmpeg.supported_metrics.contains(k),
                MetricKind::Ssim => self.m_ssim && self.ffmpeg.supported_metrics.contains(k),
                MetricKind::Vmaf => self.m_vmaf && self.ffmpeg.supported_metrics.contains(k),
                MetricKind::Xpsnr => self.m_xpsnr && self.ffmpeg.supported_metrics.contains(k),
                MetricKind::Ssim2 => self.m_ssim2,
                MetricKind::But => self.m_but,
                MetricKind::Cvvdp => self.m_cvvdp,
            })
            .collect();
        if kinds.is_empty() {
            self.toast(
                now,
                "Tick a metric in the table header to run it".to_owned(),
                ToastKind::Info,
            );
            return;
        }
        let targets: Vec<usize> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.include)
            .map(|(i, _)| i)
            .collect();
        if targets.is_empty() {
            self.toast(
                now,
                "Nothing to run — tick the checkbox in the first column".to_owned(),
                ToastKind::Info,
            );
            return;
        }
        if self.ref_path.trim().is_empty() || !Path::new(&self.ref_path).is_file() {
            for &i in &targets {
                for &kind in &kinds {
                    *self.rows[i].cell_mut(kind) = crate::metrics::MetricCell::Error {
                        msg: "no ref".to_owned(),
                    };
                }
            }
            self.toast(
                now,
                "Set a reference file first".to_owned(),
                ToastKind::Error,
            );
            return;
        }
        let (Some(skip), Some(clip_dur)) = (
            Self::trim_opt(&self.skip.clone()),
            Self::trim_opt(&self.duration.clone()),
        ) else {
            for &i in &targets {
                for &kind in &kinds {
                    *self.rows[i].cell_mut(kind) = crate::metrics::MetricCell::Error {
                        msg: "bad time".to_owned(),
                    };
                }
            }
            self.toast(
                now,
                "Skip/Duration is not a valid time".to_owned(),
                ToastKind::Error,
            );
            return;
        };
        // Validated VMAF snapshot (Python `vmaf_cfg`): subsample parses to
        // u32 with max(1, …), pooling maps the UI strings to the enum.
        // Snapshotted before the partition so VMAF `Done` stamps compare
        // against the settings this run will use.
        let vmaf_cfg = crate::metrics::vmaf::VmafCfg {
            model: self.vmaf_model.clone(),
            phone: self.vmaf_phone,
            scale: self.vmaf_scale,
            pooling: if self.vmaf_pooling == "Harmonic Mean" {
                crate::metrics::vmaf::Pooling::HarmonicMean
            } else {
                crate::metrics::vmaf::Pooling::Mean
            },
            subsample: self.vmaf_subsample.parse::<u32>().unwrap_or(1).max(1),
            // "auto" (or garbage) follows the system CPU, as before.
            n_threads: match self.vmaf_threads.parse::<u32>() {
                Ok(n) => n.max(1),
                Err(_) => crate::metrics::vmaf::system_threads(),
            },
        };
        // Per metric: rows already holding a valid value sit the rerun out —
        // but only when the trim settings still match: a value computed
        // under a different skip/clip is stale and must recompute. VMAF
        // additionally compares its options stamp, so an options change
        // recomputes just the VMAF column while other metrics keep skipping.
        // Every ffmpeg-backed column also compares the scaling stamp, so a
        // method change recomputes them (FFVship has no scale stage).
        // Pre-flight error cells above touch `targets` (settings
        // uncomparable there); everything below touches `fresh` only.
        let scaler = self.scale_method;
        let mut work: Vec<(MetricKind, Vec<usize>, Vec<String>)> = Vec::new();
        for &kind in &kinds {
            let mut skipped = Vec::new();
            let mut fresh = Vec::new();
            for &i in &targets {
                if let crate::metrics::MetricCell::Done {
                    skip: s,
                    clip_dur: c,
                    vmaf_cfg: v,
                    scaler: sc,
                    ..
                } = self.rows[i].cell(kind)
                    && *s == skip
                    && *c == clip_dur
                    && (kind != MetricKind::Vmaf || v.as_ref() == Some(&vmaf_cfg))
                    && (kind.is_ffvship() || *sc == scaler)
                {
                    skipped.push(self.rows[i].display.clone());
                } else {
                    fresh.push(i);
                }
            }
            work.push((kind, fresh, skipped));
        }
        let fresh_total: usize = work.iter().map(|(_, f, _)| f.len()).sum();
        if fresh_total == 0 {
            // One combined toast: the slot holds a single message, so per-kind
            // toasts would overwrite each other and only the last survive.
            // Identical skip sets merge (`skip_groups`) so shared filenames
            // print once instead of repeating per metric.
            let msg = skip_groups(&work)
                .iter()
                .map(|(names, skipped)| {
                    format!(
                        "Skipped {} with existing {}",
                        skipped.len(),
                        names.join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            self.toast(now, format!("{msg} (Reset to recompute)"), ToastKind::Info);
            return;
        }
        // Per-family binary gates (Python parity: per-row "ffmpeg not
        // found" / "FFVship not found"). A wrong-GPU FFVship build has a
        // path but no usable version, so it gates on `usable` as well.
        // Families are independent: an FFVship-only run needs no ffmpeg.
        let ffmpeg_exe = self.ffmpeg.path.clone();
        let ffvship_exe = if self.ffvship.usable {
            self.ffvship.path.clone()
        } else {
            None
        };
        let mut missing: Vec<&str> = Vec::new();
        for (kind, fresh, _) in &work {
            if fresh.is_empty() {
                continue;
            }
            let (exe, label) = if kind.is_ffvship() {
                (&ffvship_exe, "FFVship not found")
            } else {
                (&ffmpeg_exe, "ffmpeg not found")
            };
            if exe.is_none() {
                for &i in fresh {
                    *self.rows[i].cell_mut(*kind) = crate::metrics::MetricCell::Error {
                        msg: label.to_owned(),
                    };
                }
                if !missing.contains(&label) {
                    missing.push(label);
                }
            }
        }
        if !missing.is_empty() {
            self.toast(now, missing.join(" + "), ToastKind::Error);
        }
        let Some(ref_info) = self.ref_info_data.clone() else {
            self.toast(
                now,
                "Reference is still probing — try again in a moment".to_owned(),
                ToastKind::Info,
            );
            return;
        };
        // Rows whose probe hasn't landed yet sit this run out (cells stay
        // as-is); running the ready ones beats failing the whole batch.
        // NOTE: `fresh`, not `targets` — Done rows were partitioned out
        // above and must never be marked Running here.
        let mut jobs = Vec::new();
        for (kind, fresh, _) in &work {
            // Kinds whose binary is missing were errored above; they
            // contribute no jobs but must not block the runnable ones.
            let Some(exe) = (if kind.is_ffvship() {
                &ffvship_exe
            } else {
                &ffmpeg_exe
            })
            .clone() else {
                continue;
            };
            for &i in fresh {
                if let Some(info) = self.rows[i].info.clone() {
                    jobs.push((
                        *kind,
                        self.rows[i].key.clone(),
                        self.rows[i].path.clone(),
                        info,
                        exe.clone(),
                    ));
                    *self.rows[i].cell_mut(*kind) = crate::metrics::MetricCell::Running {
                        frame: 0,
                        values: Vec::new(),
                    };
                    // Leaving the scored set: drop the cached stats now so
                    // the refresh below can't rank a stale value, and drop
                    // cached points (capacity kept for the rerun).
                    self.rows[i].cached_mut(*kind).stats = None;
                    self.rows[i].cached_mut(*kind).points.clear();
                    self.rows[i].cached_mut(*kind).text.clear();
                }
            }
        }
        for &kind in &kinds {
            self.refresh_ranks(kind);
        }
        if jobs.is_empty() {
            // An exe-gated family already toasted above; only complain
            // about probing when binaries were fine.
            if missing.is_empty() {
                self.toast(
                    now,
                    "Files are still probing — try again in a moment".to_owned(),
                    ToastKind::Info,
                );
            }
            return;
        }
        self.run_generation = self.run_generation.wrapping_add(1);
        self.pending = jobs.len();
        self.measuring = true;
        // Fresh run: tab-follow restarts from the first live job.
        self.live_kind = None;
        if self.plot_at_start {
            self.show_plot = true;
        }
        self.abort.store(false, std::sync::atomic::Ordering::SeqCst);
        let tx = self.metric_tx.clone();
        let generation = self.run_generation;
        let ref_path = self.ref_path.clone();
        // CSV setting frozen for the run (mid-run toggles must not half-apply).
        let csv_cfg = crate::metrics::csv::CsvCfg {
            enabled: self.csv_export,
            dir: self.csv_dir.clone(),
        };
        let abort = Arc::clone(&self.abort);
        let child_slot = Arc::clone(&self.current_child);
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            let mut csv_ok = 0usize;
            let mut csv_errors: Vec<String> = Vec::new();
            for (kind, key, dist_path, dist_info, exe) in jobs {
                if abort.load(Ordering::SeqCst) {
                    break;
                }
                let txp = tx.clone();
                let keyp = key.clone();
                let txs = tx.clone();
                let keys = key.clone();
                let job = crate::metrics::ffmpeg::RunInputs {
                    kind,
                    exe: &exe,
                    ref_path: &ref_path,
                    dist_path: &dist_path,
                    ref_info: &ref_info,
                    dist_info: &dist_info,
                    skip,
                    clip_dur,
                    scaler,
                    abort: &abort,
                    child_slot: &child_slot,
                };
                let progress = |f| {
                    let _ = txp.send(MetricMsg::Progress {
                        generation,
                        kind,
                        key: keyp.clone(),
                        frame: f,
                    });
                };
                // Live-curve batches stream regardless of the plot window:
                // rendering is gated on visibility, but opening Plot
                // mid-run must show history, so the buffer always grows.
                let series = |vals: &[f64]| {
                    let _ = txs.send(MetricMsg::Series {
                        generation,
                        kind,
                        key: keys.clone(),
                        new_values: vals.to_vec(),
                    });
                };
                let out = if kind == MetricKind::Vmaf {
                    crate::metrics::vmaf::run_vmaf(&job, &vmaf_cfg, &progress)
                } else if let Some(fkind) = kind.ffvship_kind() {
                    crate::metrics::ffvship::run_ffvship(&job, fkind, &progress, &series)
                } else {
                    crate::metrics::ffmpeg::run_metric(&job, &progress, &series)
                };
                // CSV export rides the worker (never the UI thread); the
                // one-line summary lands before Finished.
                if csv_cfg.enabled && out.error.is_none() && !out.values.is_empty() {
                    match crate::metrics::csv::write_metric_csv(&csv_cfg, kind, &dist_path, &out) {
                        Ok(path) => {
                            csv_ok += 1;
                            log::info!(target: "rfmetrics::csv", "wrote {}", path.display());
                        }
                        Err(e) => {
                            log::warn!(target: "rfmetrics::csv", "export failed: {e}");
                            csv_errors.push(e);
                        }
                    }
                }
                let _ = tx.send(MetricMsg::Done {
                    generation,
                    kind,
                    key,
                    values: out.values,
                    avg: out.avg,
                    exec_s: out.exec_s,
                    error: out.error,
                    skip,
                    clip_dur,
                    scaler,
                    vmaf_cfg: if kind == MetricKind::Vmaf {
                        Some(vmaf_cfg.clone())
                    } else {
                        None
                    },
                });
            }
            let _ = tx.send(MetricMsg::Finished {
                generation,
                aborted: abort.load(Ordering::SeqCst),
            });
            if csv_cfg.enabled {
                let _ = tx.send(MetricMsg::CsvReport {
                    generation,
                    ok: csv_ok,
                    errors: csv_errors,
                });
            }
        });
        // One combined toast (see above), with identical skip sets merged so
        // shared filenames print once instead of repeating per metric.
        let parts: Vec<String> = skip_groups(&work)
            .into_iter()
            .map(|(names, skipped)| {
                let mut list = skipped.join(", ");
                if list.chars().count() > 80 {
                    list = format!("{}…", list.chars().take(79).collect::<String>());
                }
                format!(
                    "Skipped {} with existing {}: {list}",
                    skipped.len(),
                    names.join(", ")
                )
            })
            .collect();
        if !parts.is_empty() {
            self.toast(now, parts.join("\n"), ToastKind::Info);
        }
    }

    /// Signal the worker to stop and kill the in-flight ffmpeg, if any.
    /// Shared with `reset_psnr` so Reset never leaves an orphaned run.
    fn abort_worker(&self) {
        use std::sync::atomic::Ordering;
        self.abort.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = self.current_child.lock()
            && let Some(mut child) = slot.take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Stop button: abort all runners, keep finished results on screen.
    /// Still-Running cells settle to Idle when the worker's `Finished`
    /// lands in `drain_metric_results`.
    fn stop_psnr(&mut self) {
        if !self.measuring {
            return;
        }
        self.abort_worker();
        log::info!(target: "rfmetrics::app", "run aborted by user");
    }

    /// Clear all metric cells. Aborts a running worker first so Reset never
    /// leaves an orphaned ffmpeg burning CPU in the background.
    fn reset_psnr(&mut self) {
        self.abort_worker();
        self.run_generation = self.run_generation.wrapping_add(1);
        self.pending = 0;
        self.measuring = false;
        for row in &mut self.rows {
            for kind in MetricKind::ALL {
                *row.cell_mut(kind) = crate::metrics::MetricCell::Idle;
                *row.cached_mut(kind) = CachedStats::default();
            }
        }
        log::info!(target: "rfmetrics::app", "metric results cleared");
    }

    /// Metric plots in their own OS window (Python `show_plot` parity,
    /// all 7 tabs). Series are read live from `rows` every frame, so the
    /// viewport needs no update plumbing: curves appear on Done data and
    /// empty on Reset by themselves. Interaction stays on the stock
    /// `egui_plot` binds (drag pan, box-zoom select, ctrl+scroll zoom,
    /// double-click reset); the Python custom keybinds are out of scope.
    fn show_plots(&mut self, ctx: &egui::Context) {
        if !self.show_plot {
            return;
        }
        let id = egui::ViewportId::from_hash_of("metrics_plot");
        let builder = egui::ViewportBuilder::default()
            .with_title("Metrics")
            .with_inner_size([1100.0, 700.0]);
        ctx.show_viewport_immediate(id, builder, |vui, _class| {
            // Window-manager close withdraws (Python `withdraw` parity);
            // Plot reopens it.
            if vui.input(|i| i.viewport().close_requested()) {
                self.show_plot = false;
            }
            // While measuring, follow the live job's tab so its growing
            // curve is visible; idle windows stay user-driven.
            self.plot_tab = crate::plot::follow_live_tab(
                self.measuring,
                self.live_kind,
                self.plot_tab,
            );
            let kind = self.plot_tab;
            let def = crate::plot::plot_def(kind);
            // Finished series plus live `Running` buffers, so curves grow
            // mid-run (a cell is ever only one of the two — no dupes).
            // Streaming runs whether the window is open or not, so a
            // mid-run Plot click shows history; painting itself only
            // happens here, i.e. never unseen.
            let mut any_running = false;
            // Names + values feed fit/hover; `points` feeds the lines
            // directly from cache (built on arrival — zero per-frame
            // allocs). Invariant: points mirrors values for Done/Running
            // cells (drain maintains both; anything else is ignored).
            let done: Vec<(&str, &[f64], &[egui_plot::PlotPoint])> = self
                .rows
                .iter()
                .filter_map(|r| match r.cell(kind) {
                    crate::metrics::MetricCell::Done { values, .. }
                        if !values.is_empty() =>
                    {
                        Some((
                            r.display.as_str(),
                            values.as_slice(),
                            r.cached(kind).points.as_slice(),
                        ))
                    }
                    crate::metrics::MetricCell::Running { values, .. }
                        if !values.is_empty() =>
                    {
                        any_running = true;
                        Some((
                            r.display.as_str(),
                            values.as_slice(),
                            r.cached(kind).points.as_slice(),
                        ))
                    }
                    _ => None,
                })
                .collect();
            let borrowed: Vec<&[f64]> = done.iter().map(|(_, v, _)| *v).collect();
            let fresh = crate::plot::fit_limits(&borrowed, def.lo, def.hi);
            // Grow-only live bounds: axes expand with arriving points but
            // never jump inward mid-run; cleared once all settle so the
            // finished graph fits exactly again. `follow` arms the
            // one-shot auto-follow poke below (new live phase on this tab,
            // or a still-owed retry).
            let (follow, (xlim, (ymin, ymax))) = if any_running {
                let grown = match self.plot_live_fit {
                    Some((t, prev)) if t == kind => crate::plot::union_bounds(prev, fresh),
                    _ => fresh,
                };
                let follow = !matches!(self.plot_live_fit, Some((t, _)) if t == kind)
                    || self.plot_follow_pending;
                self.plot_live_fit = Some((kind, grown));
                (follow, grown)
            } else {
                self.plot_live_fit = None;
                // One-shot re-fit owed by a no-live-feed first Done (VMAF):
                // consumed like the follow retry above, so a stale arm can
                // never yank a later zoom.
                let follow = self.plot_follow_pending;
                self.plot_follow_pending = false;
                (follow, fresh)
            };
            // Empty plot (no Done data): axes only, y on the metric
            // default range; x falls back to a unit span.
            let (xmin, xmax) = xlim.unwrap_or((0.0, 1.0));
            // Per-tab plot id (zoom state persists per metric); hoisted so
            // the help-bar Reset below pokes the same memory entry the
            // canvas, snap clamp, and export snapshot use.
            let plot_id = format!("plot-{}", crate::plot::tab_title(kind).to_lowercase());
            // Help bar pinned to the bottom (Python `side="bottom"` parity).
            egui::Panel::bottom("plot_help").show(vui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(
                            "Drag: pan • Right-drag select: box zoom • Ctrl+scroll: zoom • Double-click: reset",
                        )
                        .selectable(false),
                    );
                    ui.checkbox(&mut self.plot_snap, "Snap to data").on_hover_text(
                        "Lock panning to the first/last frame and the plotted min/max; zoom and pan inside freely",
                    );
                    // Explicit re-fit (double-click parity): only arms the
                    // flag — the poke runs in the central panel below,
                    // where the plot id scope lives (a help-bar `Ui`
                    // derives different persistent ids than the canvas).
                    if ui
                        .add_enabled(
                            !borrowed.is_empty(),
                            egui::Button::new("Reset view"),
                        )
                        .on_hover_text("Fit the whole series (same as double-click)")
                        .clicked()
                    {
                        self.plot_reset_pending = true;
                    }
                    let save_label = if self.png_saving { "Saving…" } else { "Save PNG" };
                    let save_hover = if self.png_saving {
                        "Writing PNG in the background…"
                    } else {
                        "Save the current view as a PNG file (legend and axes included)"
                    };
                    let save_btn = ui
                        .add_enabled(!self.png_saving, egui::Button::new(save_label))
                        .on_hover_text(save_hover);
                    if save_btn.clicked() && !self.png_saving {
                        // Filename captured now; the export itself runs in
                        // the central panel below, where the plot id scope
                        // (for the current view bounds) lives.
                        self.plot_save_pending = Some(PlotExport::Save {
                            name: format!("{}.png", crate::plot::tab_title(self.plot_tab)),
                        });
                    }
                    let copy_label = if self.png_saving { "Copying…" } else { "Copy" };
                    let copy_hover = if self.png_saving {
                        "Rendering plot in the background…"
                    } else {
                        "Copy the current view as an image to the clipboard (legend and axes included)"
                    };
                    let copy_btn = ui
                        .add_enabled(!self.png_saving, egui::Button::new(copy_label))
                        .on_hover_text(copy_hover);
                    if copy_btn.clicked() && !self.png_saving {
                        self.plot_save_pending = Some(PlotExport::Copy);
                    }
                });
            });
            egui::CentralPanel::default().show(vui, |ui| {
                // FPS HUD (egui demo pattern): smoothed frame rate
                // top-right. Full repaint rate only while measuring (live
                // curves need it); idle repaints at ~10 Hz plus
                // input-driven ones — a static plot at 60 fps is pure
                // main+plot re-render cost, and immediate viewports
                // repaint the parent together with the child.
                if self.measuring {
                    ui.ctx().request_repaint();
                } else {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                let fps = 1.0 / ui.input(|i| i.stable_dt);
                egui::Area::new(egui::Id::new("plot_fps"))
                    .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            // Single-line HUD: never wrap the counter.
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.label(format!("FPS: {fps:.0}"));
                        });
                    });
                // Tab strip (Python `CTkTabview` parity): all 7 tabs
                // always visible; empty tabs show empty axes. Compact
                // box hugging the buttons, centered via last frame's
                // measured width: egui cannot center content of unknown
                // width upfront (`with_layout`/`horizontal_centered`
                // reserve the full remaining rect and starve the plot),
                // but tab texts are static so one measured offset stays
                // pixel-exact. First frame falls back to the left edge.
                let pad = if self.plot_tabs_w <= 0.0 {
                    0.0
                } else {
                    ((ui.available_width() - self.plot_tabs_w) / 2.0).max(0.0)
                };
                ui.horizontal(|ui| {
                    if pad > 0.0 {
                        ui.add_space(pad);
                    }
                    let frame_resp = egui::Frame::group(ui.style()).show(ui, |ui| {
                        egui::Grid::new("plot_tabs").show(ui, |ui| {
                            for tab in MetricKind::ALL {
                                let title = crate::plot::tab_title(tab);
                                let btn =
                                    egui::Button::new(title).selected(self.plot_tab == tab);
                                if ui.add(btn).clicked() {
                                    self.plot_tab = tab;
                                }
                            }
                            ui.end_row();
                        });
                    });
                    self.plot_tabs_w = frame_resp.response.rect.width();
                });
                // Per-tab plot id shared with the help-bar Reset above.
                // One-shot live-follow: explicit default bounds seed fresh
                // PlotMemory with auto OFF, freezing the first-shown
                // (often still empty) view until a double-click. Flip auto
                // back on once per live phase so bounds track the growing
                // fit; user pan/zoom afterwards still takes over (it flips
                // auto off again). Retried while memory is missing: on the
                // opening frame there is nothing to poke yet, memory
                // appears on the next shown frame.
                if follow {
                    // NOTE: the id must be derived exactly like
                    // `Plot::show` does (`new` stores `Id::new(source)`,
                    // show hashes *that*); hashing the raw string hits a
                    // different memory entry and the poke never lands.
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        mem.auto_bounds = true.into();
                        mem.store(ui.ctx(), pid);
                        self.plot_follow_pending = false;
                    } else {
                        self.plot_follow_pending = true;
                    }
                }
                // Snap-to-data lock: clamp the stored view into the data
                // extent ([1, N] frames, fit min/max) before show, so the
                // user cannot pan past the first/last frame or leave the
                // plotted min/max — zooming and in-limits panning stay
                // free. Done on the stored bounds (not via
                // `set_plot_bounds`) so auto-follow keeps working.
                if self.plot_snap && !borrowed.is_empty() {
                    let n = borrowed.iter().map(|s| s.len()).max().unwrap_or(0);
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        let b = mem.bounds();
                        let (cx0, cx1) = if n >= 2 {
                            crate::plot::clamp_range((b.min()[0], b.max()[0]), (1.0, n as f64))
                        } else {
                            (b.min()[0], b.max()[0])
                        };
                        let (cy0, cy1) = crate::plot::clamp_range(
                            (b.min()[1], b.max()[1]),
                            (ymin, ymax),
                        );
                        mem.set_bounds(egui_plot::PlotBounds::from_min_max(
                            [cx0, cy0],
                            [cx1, cy1],
                        ));
                        mem.store(ui.ctx(), pid);
                    }
                }
                // Reset-view button (help bar arms the flag — this `Ui`
                // owns the plot id scope): write the computed fit into
                // this tab's stored bounds and take over from auto-follow
                // (a user takeover, like pan/zoom). After snap so the
                // exact fit wins over the clamp. Retried while memory is
                // missing, like the follow poke.
                if self.plot_reset_pending && !borrowed.is_empty() {
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        mem.auto_bounds = false.into();
                        mem.set_bounds(egui_plot::PlotBounds::from_min_max(
                            [xmin, ymin],
                            [xmax, ymax],
                        ));
                        mem.store(ui.ctx(), pid);
                        self.plot_reset_pending = false;
                    }
                }
                // Draw budget: ~2 points per horizontal pixel (the y-axis
                // gutter makes this a slight over-estimate, harmless).
                let target = (ui.available_width() as usize * 2).clamp(512, 8192);
                // Pending plot export (Save PNG / Copy button): snapshot
                // the CURRENT view (stored bounds when strict, else the
                // fit) plus owned series data, then render on a one-shot
                // worker thread. Crosshair/tooltip never enter: this is a
                // fresh render, not a screenshot. Direct field writes below
                // (not `self.toast()`): `done` still borrows rows here.
                if let Some(job) = self.plot_save_pending.take() {
                    // Re-entrant click while an export is in flight: drop
                    // it (both buttons are disabled, so this is a guard).
                    if !self.png_saving {
                        let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                        let ((vx0, vx1), (vy0, vy1)) =
                            match egui_plot::PlotMemory::load(ui.ctx(), pid) {
                                Some(mem) => {
                                    let b = mem.bounds();
                                    let (a0, a1) = (b.min()[0], b.max()[0]);
                                    let (c0, c1) = (b.min()[1], b.max()[1]);
                                    (
                                        if a1 > a0 { (a0, a1) } else { (xmin, xmax) },
                                        if c1 > c0 { (c0, c1) } else { (ymin, ymax) },
                                    )
                                }
                                None => ((xmin, xmax), (ymin, ymax)),
                            };
                        // Owned snapshot: the worker outlives this frame and
                        // cannot borrow `done`/`self.rows`.
                        let owned: Vec<(String, Vec<f64>)> = done
                            .iter()
                            .map(|(n, v, _)| ((*n).to_owned(), (*v).to_vec()))
                            .collect();
                        let title = crate::plot::tab_title(kind).to_owned();
                        let y_label = def.label.to_owned();
                        let view = ((vx0, vx1), (vy0, vy1));
                        // Size preset snapshot: a mid-render combobox change
                        // only affects the next export.
                        let size = self.plot_size.dims();
                        match job {
                            PlotExport::Save { name } => {
                                // Picker cancelled: silent no-op. Runs on
                                // the UI thread (native modal); only the
                                // render moves off.
                                if let Some(mut path) = rfd::FileDialog::new()
                                    .set_title("Save plot as PNG")
                                    .set_file_name(&name)
                                    .add_filter("PNG image", &["png"])
                                    .save_file()
                                {
                                    path.set_extension("png");
                                    self.png_saving = true;
                                    let tx = self.png_tx.clone();
                                    let ctx = ui.ctx().clone();
                                    std::thread::spawn(move || {
                                        let series: Vec<(&str, &[f64])> = owned
                                            .iter()
                                            .map(|(n, v)| (n.as_str(), v.as_slice()))
                                            .collect();
                                        let msg = match crate::plot::export_png(
                                            &path,
                                            &title,
                                            &y_label,
                                            &series,
                                            view,
                                            size,
                                        ) {
                                            Ok(()) => {
                                                log::info!(target: "rfmetrics::plot", "plot saved to {}", path.display());
                                                PngSaveMsg::Saved { path }
                                            }
                                            Err(e) => {
                                                log::warn!(target: "rfmetrics::plot", "plot save failed: {e}");
                                                PngSaveMsg::SaveFailed { err: e }
                                            }
                                        };
                                        let _ = tx.send(msg);
                                        ctx.request_repaint();
                                    });
                                }
                            }
                            PlotExport::Copy => {
                                self.png_saving = true;
                                let tx = self.png_tx.clone();
                                let ctx = ui.ctx().clone();
                                std::thread::spawn(move || {
                                    let series: Vec<(&str, &[f64])> = owned
                                        .iter()
                                        .map(|(n, v)| (n.as_str(), v.as_slice()))
                                        .collect();
                                    let msg = match crate::plot::render_rgba(
                                        &title, &y_label, &series, view, size,
                                    ) {
                                        Ok((w, h, rgba)) => {
                                            log::info!(target: "rfmetrics::plot", "plot rendered for clipboard ({w}x{h})");
                                            PngSaveMsg::CopyReady { w, h, rgba }
                                        }
                                        Err(e) => {
                                            log::warn!(target: "rfmetrics::plot", "plot copy failed: {e}");
                                            PngSaveMsg::CopyFailed { err: e }
                                        }
                                    };
                                    let _ = tx.send(msg);
                                    ctx.request_repaint();
                                });
                            }
                        }
                    }
                }
                let plot_resp = egui_plot::Plot::new(plot_id)
                    .x_axis_label("Frames")
                    .y_axis_label(def.label)
                    .legend(
                        egui_plot::Legend::default()
                            .position(egui_plot::Corner::RightBottom),
                    )
                    .default_x_bounds(xmin, xmax)
                    .default_y_bounds(ymin, ymax)
                    .show(ui, |plot_ui| {
                        // Lines borrow cached points, min-max decimated to
                        // ~2 px buckets (values still feed fit + hover at
                        // full resolution).
                        for (name, _, points) in &done {
                            let thin = crate::plot::decimate_minmax(points, target);
                            plot_ui.line(egui_plot::Line::new(*name, thin));
                        }
                        // Hover inspect (Python `_on_hover` parity):
                        // nearest data point within 30 screen px gets a
                        // crosshair; the `{name}\nFrame=N, Metric=V.4f`
                        // text returns to the caller, which draws it as a
                        // native tooltip (plot-canvas text is tiny and has
                        // no background). Suppressed while
                        // panning/zooming, like the ref.
                        let hovering = plot_ui.response().hovered()
                            && !plot_ui.response().dragged();
                        let hover_pos = plot_ui.response().hover_pos();
                        let ptr = plot_ui.pointer_coordinate();
                        if let (true, Some(mouse), Some(p)) = (hovering, hover_pos, ptr) {
                            let to_screen = |fx: f64, fy: f64| {
                                let sp = plot_ui.screen_from_plot(
                                    egui_plot::PlotPoint::new(fx, fy),
                                );
                                (sp.x, sp.y)
                            };
                            if let Some((si, frame, value)) = crate::plot::nearest_hover(
                                &borrowed,
                                p.x,
                                to_screen,
                                (mouse.x, mouse.y),
                            ) {
                                let fx = frame as f64;
                                plot_ui.vline(egui_plot::VLine::new("", fx));
                                plot_ui.hline(egui_plot::HLine::new("", value));
                                return Some(format!(
                                    "{}\nFrame={frame}, Metric={value:.4}",
                                    done[si].0
                                ));
                            }
                        }
                        None
                    });
                // Native tooltip at the pointer: readable body text on a
                // theme background (Python yellow annotation-box parity).
                if let Some(text) = plot_resp.inner {
                    egui::Tooltip::for_widget(&plot_resp.response)
                        .at_pointer()
                        .gap(12.0)
                        .show(|ui| {
                            ui.label(text);
                        });
                }
                // Mirror the main-window toast here (PNG saver results land
                // while this OS window has focus; the main toast behind it
                // is invisible). Expiry is owned by the main viewport.
                if let Some(toast) = self.toast.clone()
                    && ui.input(|i| i.time) < toast.until
                {
                    let corner = ui.max_rect().right_bottom();
                    let mut frame = egui::Frame::popup(ui.style());
                    if let Some(outline) = toast.kind.outline() {
                        frame = frame.stroke(egui::Stroke::new(1.5, outline));
                    }
                    egui::Area::new(egui::Id::new("plot_toast"))
                        .order(egui::Order::Foreground)
                        .fixed_pos(corner + egui::vec2(-10.0, -10.0))
                        .pivot(egui::Align2::RIGHT_BOTTOM)
                        .show(ui.ctx(), |ui| {
                            frame.show(ui, |ui| {
                                ui.label(&toast.text);
                            });
                        });
                }
            });
        });
    }
}
/// Kinds sharing an identical skip set merge into one toast line
/// ("Skipped 2 with existing PSNR, SSIM: a, b") so filenames print once
/// instead of repeating per metric. First-seen kind order is kept.
fn skip_groups(work: &[(MetricKind, Vec<usize>, Vec<String>)]) -> Vec<(Vec<&str>, &Vec<String>)> {
    let mut groups: Vec<(Vec<&str>, &Vec<String>)> = Vec::new();
    for (kind, _, skipped) in work {
        if skipped.is_empty() {
            continue;
        }
        if let Some(g) = groups
            .iter_mut()
            .find(|(_, s)| s.as_slice() == skipped.as_slice())
        {
            g.0.push(kind.name());
        } else {
            groups.push((vec![kind.name()], skipped));
        }
    }
    groups
}

/// 1px vertical divider in an exact 3px grid column.
fn vline(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, color),
    );
}

/// Panel frame with Python's drag-enter green (#2FA572) while hovered.
fn panel_frame(ui: &egui::Ui, hovering: bool) -> egui::Frame {
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
const BEST_FILL: egui::Color32 = egui::Color32::from_rgb(0x2E, 0x6B, 0x3E);
const WORST_FILL: egui::Color32 = egui::Color32::from_rgb(0x7A, 0x36, 0x36);
const TIE_FILL: egui::Color32 = egui::Color32::from_rgb(0x6B, 0x5F, 0x2A);

/// Cell/chip background for a stat rank; `None` = no highlight.
fn rank_fill(rank: crate::metrics::StatRank) -> Option<egui::Color32> {
    match rank {
        crate::metrics::StatRank::Best => Some(BEST_FILL),
        crate::metrics::StatRank::Worst => Some(WORST_FILL),
        crate::metrics::StatRank::Tie => Some(TIE_FILL),
        crate::metrics::StatRank::Plain => None,
    }
}

/// Table metric-column layout, left to right — MUST match the header
/// checkbox order.
const METRIC_COLUMNS: [(Option<MetricKind>, &str); 7] = [
    (Some(MetricKind::Psnr), "PSNR"),
    (Some(MetricKind::Ssim), "SSIM"),
    (Some(MetricKind::Vmaf), "VMAF"),
    (Some(MetricKind::Xpsnr), "XPSNR"),
    (Some(MetricKind::Ssim2), "SSIM2"),
    (Some(MetricKind::But), "BUTTER"),
    (Some(MetricKind::Cvvdp), "CVVDP"),
];

/// Filter-metric Done tooltip in FFMetrics order: Avg, Exec, Frames, a blank
/// line, Mean..StdDev, another blank line, then Percentiles. Each comparable
/// value is chipped by its cross-row rank; Exec time and Frames count are
/// display-only (no chip).
/// Plain horizontal rows with content-hugging widths: grids and expanding
/// layouts feed back into the tooltip auto-size and balloon while hovered.
fn metric_stat_tooltip(
    ui: &mut egui::Ui,
    title: &str,
    stats: &crate::metrics::DoneStats,
    ranks: &[crate::metrics::StatRank; 10],
) {
    ui.label(egui::RichText::new(title).strong());
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 1.0);
        let comp = stats.comparable();
        let (label, v, _) = comp[0];
        tip_stat_row(ui, label, &format!("{v:.6}"), ranks[0]);
        tip_plain_row(ui, "Exec time:", &crate::metrics::format_exec(stats.exec_s));
        tip_plain_row(ui, "Frames count:", &stats.frames.to_string());
        ui.add_space(5.0);
        for k in 1..=5 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, label, &format!("{v:.6}"), ranks[k]);
        }
        ui.add_space(5.0);
        for k in 6..10 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, label, &format!("{v:.6}"), ranks[k]);
        }
    });
}

/// One tooltip row: fixed label + right-aligned value, chipped when ranked.
fn tip_stat_row(ui: &mut egui::Ui, label: &str, val: &str, rank: crate::metrics::StatRank) {
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
fn tip_plain_row(ui: &mut egui::Ui, label: &str, val: &str) {
    tip_stat_row(ui, label, val, crate::metrics::StatRank::Plain);
}

/// Drop routing decision: pure so the guard rails stay unit-tested.
/// Mid-run drops are `Blocked` (toast) — the worker snapshotted its jobs
/// at Start, so ref/queue changes must wait for Stop.
#[derive(Debug, PartialEq, Eq)]
enum DropAction {
    Ignore,
    Blocked,
    SetRef {
        first: std::path::PathBuf,
        extra: usize,
    },
    Queue(Vec<std::path::PathBuf>),
}

fn route_drop(
    measuring: bool,
    is_over_ref: bool,
    is_over_table: bool,
    dropped: Vec<std::path::PathBuf>,
) -> DropAction {
    if dropped.is_empty() {
        return DropAction::Ignore;
    }
    if measuring {
        return DropAction::Blocked;
    }
    if is_over_ref {
        let mut iter = dropped.into_iter();
        // `dropped` is non-empty (checked above), so `first` exists.
        let first = iter.next().unwrap_or_default();
        let extra = iter.len();
        DropAction::SetRef { first, extra }
    } else if is_over_table {
        DropAction::Queue(dropped)
    } else {
        DropAction::Ignore
    }
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
        let run_locked = self.measuring;

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
        let live = self.drain_png_results(&ctx, now) | live;
        if live {
            ui.ctx().request_repaint();
        } else if self.measuring {
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
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.plot_at_start,
                                    "Plot window at start",
                                ))
                                .on_hover_text(
                                    "Open the plot window automatically when a run starts",
                                );
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.csv_export,
                                    "Save frames metrics to CSV files",
                                ))
                                .on_hover_text(
                                    "Write one <name>.<METRIC>.csv per finished run \
                                     (TAB-separated per-frame values), to the folder below \
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
                    table
                        .header(18.0, |mut header| {
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("Path").strong())
                                        .selectable(false),
                                );
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
                            // may live across it. FFVship metrics ride the
                            // binary-level `usable` gate instead (start_run).
                            let (psnr_ok, ssim_ok, vmaf_ok, xpsnr_ok) = {
                                let sup = &self.ffmpeg.supported_metrics;
                                (
                                    sup.contains(&MetricKind::Psnr),
                                    sup.contains(&MetricKind::Ssim),
                                    sup.contains(&MetricKind::Vmaf),
                                    sup.contains(&MetricKind::Xpsnr),
                                )
                            };
                            for (flag, name, ok) in [
                                (&mut self.m_psnr, "PSNR", psnr_ok),
                                (&mut self.m_ssim, "SSIM", ssim_ok),
                                (&mut self.m_vmaf, "VMAF", vmaf_ok),
                                (&mut self.m_xpsnr, "XPSNR", xpsnr_ok),
                                (&mut self.m_ssim2, "SSIM2", true),
                                (&mut self.m_but, "BUTTER", true),
                                (&mut self.m_cvvdp, "CVVDP", true),
                            ] {
                                header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                                header.col(|ui| {
                                    let resp = ui.add_enabled(
                                        ok && !run_locked,
                                        egui::Checkbox::new(flag, name),
                                    );
                                    if !ok {
                                        resp.on_hover_text(format!(
                                            "{name} filter not supported by this ffmpeg build"
                                        ));
                                    }
                                });
                            }
                        })
                        .body(|body| {
                            body.rows(20.0, self.rows.len(), |mut row| {
                                let i = row.index();
                                row.set_selected(self.rows[i].selected);
                                // Delayed hover: only outline after the pointer
                                // rests on the row, so passing over rows while
                                // aiming at text doesn't flash each one.
                                let hover_delayed = self.hover_row == Some(i)
                                    && self.hover_since.is_some_and(|t| now - t >= ROW_HOVER_DELAY);
                                row.set_hovered(hover_delayed);
                                // Free-space click toggles selection; widget clicks
                                // (checkbox, play, text drag-select) must not.
                                // (`toggle_row` etc. are set here, applied below.)
                                row.col(|ui| {
                                    ui.checkbox(&mut self.rows[i].include, "");
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(i);
                                }
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        open_path = Some(self.rows[i].path.clone());
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(i);
                                }
                                let (_, r) = row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let row_data = &self.rows[i];
                                    // Plain non-selectable text: no button hover
                                    // outline; copy lives in the right-click menu
                                    // and the full path shows as tooltip (Python parity).
                                    ui.add(egui::Label::new(&row_data.display).selectable(false))
                                        .on_hover_text(&row_data.path);
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Show in explorer").clicked() {
                                        reveal_path = Some(self.rows[i].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy Path").clicked() {
                                        ui.ctx().copy_text(self.rows[i].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy filename").clicked() {
                                        let name = Path::new(&self.rows[i].path)
                                            .file_name()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| self.rows[i].display.clone());
                                        ui.ctx().copy_text(name);
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(i);
                                }
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(i);
                                }
                                let (_, r) = row.col(|ui| {
                                    let row_data = &self.rows[i];
                                    ui.add(egui::Label::new(&row_data.media).selectable(false))
                                        .on_hover_text(&row_data.media_tip);
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Copy summary").clicked() {
                                        ui.ctx().copy_text(crate::probe::table_media_text(
                                            self.rows[i].info.as_ref(),
                                        ));
                                        ui.close();
                                    }
                                    if ui.button("Copy details").clicked() {
                                        ui.ctx().copy_text(self.rows[i].media_tip.clone());
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(i);
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
                                        toggle_row = Some(i);
                                    }
                                    let (_, r) = row.col(|ui| {
                                        let row_data = &self.rows[i];
                                        let cell = row_data.cell(kind);
                                        let cached = row_data.cached(kind);
                                        // Idle borrows a static, Done borrows
                                        // the arrival-frozen text, Error borrows
                                        // its message; only live Running frames
                                        // format per frame (they change anyway).
                                        let running;
                                        let text: &str = match cell {
                                            crate::metrics::MetricCell::Idle => "N/A",
                                            crate::metrics::MetricCell::Running {
                                                frame, ..
                                            } => {
                                                running = format!("Frame: {frame}");
                                                &running
                                            }
                                            crate::metrics::MetricCell::Done { .. } => &cached.text,
                                            crate::metrics::MetricCell::Error { msg } => msg,
                                        };
                                        // Borrow the cached stats when scored;
                                        // unscored cells build their one-line tip
                                        // below, and only while hovered.
                                        let stats = cached.stats.as_ref();
                                        let ranks = cached.ranks;
                                        let mut cell_frame = egui::Frame::NONE;
                                        if let Some(fill) = rank_fill(ranks[0]) {
                                            cell_frame = cell_frame.fill(fill);
                                        }
                                        cell_frame.show(ui, |ui| {
                                            ui.set_width(ui.available_width());
                                            ui.centered_and_justified(|ui| {
                                                let resp = ui.label(text);
                                                match stats {
                                                    Some(stats) => {
                                                        resp.on_hover_ui(|ui| {
                                                            metric_stat_tooltip(
                                                                ui, title, stats, &ranks,
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
                                    if r.clicked() {
                                        toggle_row = Some(i);
                                    }
                                }
                                if row.response().hovered() {
                                    hovered_next = Some(i);
                                }
                            });
                        });
                    // Deferred row-click side effects (L3): selection
                    // toggle, open-in-player (error toast needs `now`),
                    // reveal-in-explorer, and hover tracking all land after
                    // the loop.
                    if let Some(i) = toggle_row {
                        self.rows[i].selected = !self.rows[i].selected;
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
    }
}

#[cfg(test)]
mod tests {
    use super::ScaleMethod;
    use super::{
        CachedStats, DropAction, METRIC_COLUMNS, ProbeMsg, QueueRow, RFMetricsApp, display_names,
        norm_key, route_drop,
    };

    #[test]
    fn metric_columns_keep_live_cells_under_their_headers() {
        use crate::metrics::ffmpeg::MetricKind;
        // Body order must mirror the header checkboxes (PSNR SSIM VMAF
        // XPSNR SSIM2 BUTTER CVVDP); every slot is live, so no column can
        // slide under the wrong header.
        let kinds: Vec<_> = METRIC_COLUMNS.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            [
                Some(MetricKind::Psnr),
                Some(MetricKind::Ssim),
                Some(MetricKind::Vmaf),
                Some(MetricKind::Xpsnr),
                Some(MetricKind::Ssim2),
                Some(MetricKind::But),
                Some(MetricKind::Cvvdp),
            ]
        );
        let titles: Vec<_> = METRIC_COLUMNS.iter().map(|(_, t)| *t).collect();
        assert_eq!(
            titles,
            ["PSNR", "SSIM", "VMAF", "XPSNR", "SSIM2", "BUTTER", "CVVDP"]
        );
    }

    #[test]
    fn drop_routing() {
        use std::path::PathBuf;
        let files = || vec![PathBuf::from("C:/v/a.mp4"), PathBuf::from("C:/v/b.mp4")];
        // Empty drop: nothing, even mid-run.
        assert_eq!(route_drop(false, true, true, vec![]), DropAction::Ignore);
        assert_eq!(route_drop(true, true, true, vec![]), DropAction::Ignore);
        // Mid-run drops block with a toast instead of mutating state.
        assert_eq!(route_drop(true, true, false, files()), DropAction::Blocked);
        assert_eq!(route_drop(true, false, true, files()), DropAction::Blocked);
        // Reference takes one file; extras are reported, not lost.
        assert_eq!(
            route_drop(false, true, false, files()),
            DropAction::SetRef {
                first: PathBuf::from("C:/v/a.mp4"),
                extra: 1,
            }
        );
        // Table queues everything; outside any target is ignored.
        assert_eq!(
            route_drop(false, false, true, files()),
            DropAction::Queue(files())
        );
        assert_eq!(route_drop(false, false, false, files()), DropAction::Ignore);
    }

    #[test]
    fn same_file_keys_equal() {
        assert_eq!(norm_key("C:/Vids/a.mp4"), norm_key("c:\\vids\\A.MP4"));
        assert_ne!(norm_key("C:/Vids/a.mp4"), norm_key("C:/Vids/b.mp4"));
    }

    #[test]
    fn single_name_is_basename() {
        assert_eq!(
            display_names(&["C:/a/b/c.mp4".to_owned()]),
            vec!["c.mp4".to_owned()]
        );
    }

    #[test]
    fn sibling_names_disambiguate() {
        let names = display_names(&[
            "C:/b/output tq 70.mkv".to_owned(),
            "C:/b/output tq 75.mkv".to_owned(),
        ]);
        assert_eq!(names, vec!["output tq 70.mkv", "output tq 75.mkv"]);
    }

    #[test]
    fn same_basename_keeps_parent() {
        let names = display_names(&["C:/a/x.mp4".to_owned(), "C:/b/x.mp4".to_owned()]);
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(names, vec![format!("a{sep}x.mp4"), format!("b{sep}x.mp4")]);
    }

    /// Refresh re-probes the reference and every row through the normal
    /// worker channels: markers clear, rows show the placeholder, and
    /// results land via drain (missing files resolve without a process).
    #[test]
    fn refresh_media_info_reprobes_ref_and_rows() {
        // Hermetic: no ffprobe, so workers resolve to text without spawning.
        let mut app = RFMetricsApp {
            ffprobe: None,
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/no/such/a.mp4", true));
        app.rows[0].media = "old".to_owned();
        app.last_spawned_ref = "sentinel".to_owned();
        app.last_thumb_path = "sentinel".to_owned();
        app.ref_path = "C:/no/such/ref.mp4".to_owned();
        app.refresh_media_info();
        assert!(app.last_spawned_ref.is_empty());
        assert!(app.last_thumb_path.is_empty());
        assert_eq!(app.rows[0].media, "Probing…");
        for _ in 0..200 {
            app.drain_probe_results();
            if app.rows[0].media != "Probing…" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_ne!(app.rows[0].media, "Probing…");
        // Reference cheap case resolves inline on the next refresh tick.
        app.refresh_ref_info();
        assert_eq!(app.ref_info, "File not found");
    }

    #[test]
    fn refresh_media_info_empty_queue_only_clears_markers() {
        let mut app = RFMetricsApp {
            last_spawned_ref: "sentinel".to_owned(),
            ..RFMetricsApp::default()
        };
        app.refresh_media_info();
        assert!(app.last_spawned_ref.is_empty());
        assert!(app.probe_rx.try_recv().is_err());
    }

    #[test]
    fn ref_cheap_cases_stay_synchronous() {
        let mut app = RFMetricsApp {
            ref_path: String::new(),
            ..RFMetricsApp::default()
        };
        app.refresh_ref_info();
        assert!(app.ref_info.contains("-unknown-"));
        app.ref_path = "C:/no/such/file.mp4".to_owned();
        app.refresh_ref_info();
        assert_eq!(app.ref_info, "File not found");
        // Neither case spawns a worker: the channel stays empty.
        assert!(app.probe_rx.try_recv().is_err());
    }

    #[test]
    fn stale_reference_result_discarded() {
        let mut app = RFMetricsApp {
            ref_info: "sentinel".to_owned(),
            ..RFMetricsApp::default()
        };
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: 999,
                text: "stale".to_owned(),
                info: None,
            })
            .unwrap();
        app.refresh_ref_info();
        assert_eq!(app.ref_info, "sentinel");
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: app.ref_generation,
                text: "fresh".to_owned(),
                info: None,
            })
            .unwrap();
        app.refresh_ref_info();
        assert_eq!(app.ref_info, "fresh");
    }

    #[test]
    fn row_media_applies_by_key() {
        let mut app = RFMetricsApp::default();
        app.rows.push(QueueRow {
            path: "C:/vids/a.mp4".to_owned(),
            key: norm_key("C:/vids/a.mp4"),
            display: "a.mp4".to_owned(),
            include: true,
            selected: false,
            media: "Probing…".to_owned(),
            media_tip: "Probing…".to_owned(),
            info: None,
            psnr: crate::metrics::MetricCell::Idle,
            ssim: crate::metrics::MetricCell::Idle,
            vmaf: crate::metrics::MetricCell::Idle,
            xpsnr: crate::metrics::MetricCell::Idle,
            ssim2: crate::metrics::MetricCell::Idle,
            butter: crate::metrics::MetricCell::Idle,
            cvvdp: crate::metrics::MetricCell::Idle,
            psnr_cache: CachedStats::default(),
            ssim_cache: CachedStats::default(),
            vmaf_cache: CachedStats::default(),
            xpsnr_cache: CachedStats::default(),
            ssim2_cache: CachedStats::default(),
            butter_cache: CachedStats::default(),
            cvvdp_cache: CachedStats::default(),
        });
        let key = norm_key("C:/vids/a.mp4");
        app.probe_tx
            .send(ProbeMsg::RowMedia {
                key,
                media: "h264, 1080p".to_owned(),
                tip: "tip".to_owned(),
                info: None,
            })
            .unwrap();
        app.probe_tx
            .send(ProbeMsg::RowMedia {
                key: "nope".to_owned(),
                media: "x".to_owned(),
                tip: "y".to_owned(),
                info: None,
            })
            .unwrap();
        app.refresh_ref_info();
        assert_eq!(app.rows[0].media, "h264, 1080p");
        assert_eq!(app.rows[0].media_tip, "tip");
    }

    #[test]
    fn queue_shows_probing_placeholder() {
        let mut app = RFMetricsApp::default();
        app.add_queue_files(vec![std::path::PathBuf::from("C:/no/such/file.mp4")]);
        assert_eq!(app.rows.len(), 1);
        assert_eq!(app.rows[0].media, "Probing…");
        // Missing files resolve without spawning ffprobe; poll briefly.
        for _ in 0..200 {
            app.drain_probe_results();
            if app.rows[0].media != "Probing…" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(app.rows[0].media.contains("-unknown-"));
    }

    fn psnr_test_row(path: &str, include: bool) -> QueueRow {
        QueueRow {
            path: path.to_owned(),
            key: norm_key(path),
            display: "a.mp4".to_owned(),
            include,
            selected: false,
            media: "h264, 1080p".to_owned(),
            media_tip: "tip".to_owned(),
            info: None,
            psnr: crate::metrics::MetricCell::Idle,
            ssim: crate::metrics::MetricCell::Idle,
            vmaf: crate::metrics::MetricCell::Idle,
            xpsnr: crate::metrics::MetricCell::Idle,
            ssim2: crate::metrics::MetricCell::Idle,
            butter: crate::metrics::MetricCell::Idle,
            cvvdp: crate::metrics::MetricCell::Idle,
            psnr_cache: CachedStats::default(),
            ssim_cache: CachedStats::default(),
            vmaf_cache: CachedStats::default(),
            xpsnr_cache: CachedStats::default(),
            ssim2_cache: CachedStats::default(),
            butter_cache: CachedStats::default(),
            cvvdp_cache: CachedStats::default(),
        }
    }

    #[test]
    fn start_psnr_gated_on_checkbox() {
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.m_psnr = false;
        app.m_vmaf = false;
        app.start_run(0.0);
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].psnr, crate::metrics::MetricCell::Idle));
        assert!(app.toast.is_some());
    }

    #[test]
    fn start_psnr_needs_included_rows() {
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ..RFMetricsApp::default()
        };
        // Unchecked include box: the row must not be processed.
        app.rows.push(psnr_test_row("C:/vids/a.mp4", false));
        app.start_run(0.0);
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].psnr, crate::metrics::MetricCell::Idle));
    }

    #[test]
    fn start_psnr_bad_time_marks_cells() {
        let p = std::env::temp_dir().join("rfmetrics-psnr-ref.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_ssim: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            skip: "abc".to_owned(),
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(matches!(
            &app.rows[0].psnr,
            crate::metrics::MetricCell::Error { msg } if msg == "bad time"
        ));
        assert!(matches!(
            &app.rows[0].ssim,
            crate::metrics::MetricCell::Error { msg } if msg == "bad time"
        ));
    }

    #[test]
    fn psnr_progress_keeps_max_and_done_clears() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Running {
            frame: 10,
            values: Vec::new(),
        };
        app.run_generation = 1;
        app.pending = 1;
        app.measuring = true;
        let key = norm_key("C:/vids/a.mp4");
        // Stale frame ignored, fresh frame applied.
        app.metric_tx
            .send(MetricMsg::Progress {
                generation: 1,
                kind: crate::metrics::ffmpeg::MetricKind::Psnr,
                key: key.clone(),
                frame: 5,
            })
            .unwrap();
        app.metric_tx
            .send(MetricMsg::Progress {
                generation: 1,
                kind: crate::metrics::ffmpeg::MetricKind::Psnr,
                key: key.clone(),
                frame: 25,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(
            app.rows[0].psnr,
            MetricCell::Running { frame: 25, .. }
        ));
        // No-summary avg falls back to the arithmetic mean; run ends.
        // Settings stamp through: the cell remembers this trim.
        app.metric_tx
            .send(MetricMsg::Done {
                generation: 1,
                kind: crate::metrics::ffmpeg::MetricKind::Psnr,
                key,
                values: vec![30.0, 32.0],
                avg: None,
                exec_s: 1.5,
                error: None,
                skip: None,
                clip_dur: Some(5.0),
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(
            &app.rows[0].psnr,
            MetricCell::Done { avg, skip, clip_dur, .. }
                if (*avg - 31.0).abs() < 1e-9 && skip.is_none() && *clip_dur == Some(5.0)
        ));
        assert!(!app.measuring);
    }

    #[test]
    fn series_appends_in_order_and_done_replaces() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Running {
            frame: 0,
            values: Vec::new(),
        };
        app.run_generation = 1;
        let key = norm_key("C:/vids/a.mp4");
        let series = |generation: u64, vals: Vec<f64>| MetricMsg::Series {
            generation,
            kind: crate::metrics::ffmpeg::MetricKind::Psnr,
            key: key.clone(),
            new_values: vals,
        };
        // Stale generation and empty batches drop silently.
        app.metric_tx.send(series(0, vec![99.0])).unwrap();
        app.metric_tx.send(series(1, vec![])).unwrap();
        app.metric_tx.send(series(1, vec![30.0, 31.0])).unwrap();
        app.metric_tx.send(series(1, vec![32.0])).unwrap();
        app.drain_metric_results();
        assert!(matches!(
            &app.rows[0].psnr,
            MetricCell::Running { values, .. } if values == &[30.0, 31.0, 32.0]
        ));
        // Cached points mirror values 1:1 (plot borrows them, no rebuild).
        let pts = &app.rows[0].psnr_cache.points;
        assert_eq!(pts.len(), 3);
        assert_eq!((pts[0].x, pts[0].y), (1.0, 30.0));
        assert_eq!((pts[2].x, pts[2].y), (3.0, 32.0));
        // Batches for a settled cell are ignored, not resurrected.
        app.rows[0].psnr = MetricCell::Idle;
        app.metric_tx.send(series(1, vec![33.0])).unwrap();
        app.drain_metric_results();
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        // Done replaces the live buffer with the strict series.
        app.rows[0].psnr = MetricCell::Running {
            frame: 3,
            values: vec![30.0, 31.0, 32.0],
        };
        app.metric_tx
            .send(MetricMsg::Done {
                generation: 1,
                kind: crate::metrics::ffmpeg::MetricKind::Psnr,
                key: key.clone(),
                values: vec![29.0, 31.0],
                avg: Some(30.0),
                exec_s: 1.0,
                error: None,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(
            &app.rows[0].psnr,
            MetricCell::Done { values, .. } if values == &[29.0, 31.0]
        ));
        // Done rebuilds points from the strict series (not the partials).
        let pts = &app.rows[0].psnr_cache.points;
        assert_eq!(pts.len(), 2);
        assert_eq!((pts[0].x, pts[0].y), (1.0, 29.0));
        assert_eq!((pts[1].x, pts[1].y), (2.0, 31.0));
    }

    #[test]
    fn progress_and_series_track_live_kind() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        use crate::metrics::ffmpeg::MetricKind;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Running {
            frame: 0,
            values: Vec::new(),
        };
        app.run_generation = 1;
        let key = norm_key("C:/vids/a.mp4");
        assert_eq!(app.live_kind, None);
        // Stale generation touches nothing.
        app.metric_tx
            .send(MetricMsg::Progress {
                generation: 0,
                kind: MetricKind::Xpsnr,
                key: key.clone(),
                frame: 5,
            })
            .unwrap();
        app.drain_metric_results();
        assert_eq!(app.live_kind, None);
        // Live feeds record the executing job's kind.
        app.metric_tx
            .send(MetricMsg::Progress {
                generation: 1,
                kind: MetricKind::Psnr,
                key: key.clone(),
                frame: 5,
            })
            .unwrap();
        app.drain_metric_results();
        assert_eq!(app.live_kind, Some(MetricKind::Psnr));
        app.metric_tx
            .send(MetricMsg::Series {
                generation: 1,
                kind: MetricKind::Ssim,
                key,
                new_values: vec![0.9],
            })
            .unwrap();
        app.drain_metric_results();
        assert_eq!(app.live_kind, Some(MetricKind::Ssim));
    }

    #[test]
    fn psnr_stale_generation_dropped() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.run_generation = 2; // run 1's messages are orphans after Reset
        app.metric_tx
            .send(MetricMsg::Done {
                generation: 1,
                kind: crate::metrics::ffmpeg::MetricKind::Psnr,
                key: norm_key("C:/vids/a.mp4"),
                values: vec![30.0],
                avg: Some(30.0),
                exec_s: 1.0,
                error: None,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
    }

    #[test]
    fn stop_is_noop_when_idle() {
        use std::sync::atomic::Ordering;
        let mut app = RFMetricsApp::default();
        app.stop_psnr();
        assert!(!app.abort.load(Ordering::SeqCst));
    }

    #[test]
    fn drain_caches_stats_and_ranks_once() {
        use super::MetricMsg;
        use crate::metrics::StatRank;
        use crate::metrics::ffmpeg::MetricKind;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        app.run_generation = 1;
        for (path, avg) in [("C:/vids/a.mp4", 30.0), ("C:/vids/b.mp4", 40.0)] {
            app.metric_tx
                .send(MetricMsg::Done {
                    generation: 1,
                    kind: MetricKind::Psnr,
                    key: norm_key(path),
                    values: vec![avg - 1.0, avg, avg + 1.0],
                    avg: Some(avg),
                    exec_s: 1.0,
                    error: None,
                    skip: None,
                    clip_dur: None,
                    vmaf_cfg: None,
                    scaler: ScaleMethod::Bicubic,
                })
                .unwrap();
        }
        app.drain_metric_results();
        // Stats cached on arrival (no per-frame clone+sort in the render).
        assert!(app.rows[0].psnr_cache.stats.is_some());
        assert!(app.rows[1].psnr_cache.stats.is_some());
        // Ranks resolved across the scored set: avg index 0 decides the cell.
        assert_eq!(app.rows[0].psnr_cache.ranks[0], StatRank::Worst);
        assert_eq!(app.rows[1].psnr_cache.ranks[0], StatRank::Best);
        // Reset clears the cache back to Plain.
        app.reset_psnr();
        assert!(app.rows[0].psnr_cache.stats.is_none());
        assert_eq!(app.rows[0].psnr_cache.ranks, [StatRank::Plain; 10]);
    }

    #[test]
    fn refresh_ranks_single_row_stays_plain() {
        use crate::metrics::ffmpeg::MetricKind;
        use crate::metrics::{MetricCell, StatRank};
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0, 31.0],
            avg: 30.5,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        let stats = app.rows[0].psnr.done_stats();
        app.rows[0].psnr_cache.stats = stats;
        app.refresh_ranks(MetricKind::Psnr);
        assert_eq!(app.rows[0].psnr_cache.ranks, [StatRank::Plain; 10]);
    }

    #[test]
    fn stop_keeps_done_and_settles_running_to_idle() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        use std::sync::atomic::Ordering;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        // a finished before Stop, b was in flight.
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[1].psnr = MetricCell::Running {
            frame: 12,
            values: Vec::new(),
        };
        app.rows[1].ssim = MetricCell::Running {
            frame: 3,
            values: Vec::new(),
        };
        app.measuring = true;
        app.pending = 1;
        app.run_generation = 1;

        app.stop_psnr();
        assert!(app.abort.load(Ordering::SeqCst));

        app.metric_tx
            .send(MetricMsg::Finished {
                generation: 1,
                aborted: true,
            })
            .unwrap();
        app.drain_metric_results();
        // Processed result kept; unprocessed settled; button flips back.
        assert!(matches!(
            &app.rows[0].psnr,
            MetricCell::Done { avg, .. } if (*avg - 30.0).abs() < 1e-9
        ));
        assert!(matches!(app.rows[1].psnr, MetricCell::Idle));
        assert!(matches!(app.rows[1].ssim, MetricCell::Idle));
        assert!(!app.measuring);
        assert_eq!(app.pending, 0);
    }

    #[test]
    fn stop_settles_killed_cell_to_idle() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        use crate::metrics::ffmpeg::MetricKind;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        // a is the killed in-flight job, b never started.
        app.rows[0].psnr = MetricCell::Running {
            frame: 42,
            values: Vec::new(),
        };
        app.rows[1].psnr = MetricCell::Running {
            frame: 0,
            values: Vec::new(),
        };
        app.measuring = true;
        app.pending = 2;
        app.run_generation = 1;

        // The killed job reports back first: quiet settle, not an error.
        app.metric_tx
            .send(MetricMsg::Done {
                generation: 1,
                kind: MetricKind::Psnr,
                key: norm_key("C:/vids/a.mp4"),
                values: Vec::new(),
                avg: None,
                exec_s: 3.5,
                error: Some("aborted".to_owned()),
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));

        // `Finished` settles the unstarted row; the run ends.
        app.metric_tx
            .send(MetricMsg::Finished {
                generation: 1,
                aborted: true,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(matches!(app.rows[1].psnr, MetricCell::Idle));
        assert!(!app.measuring);
        assert_eq!(app.pending, 0);
    }

    #[test]
    fn clean_finish_leaves_cells_and_clears_measuring() {
        use super::MetricMsg;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.measuring = true;
        app.run_generation = 1;
        app.metric_tx
            .send(MetricMsg::Finished {
                generation: 1,
                aborted: false,
            })
            .unwrap();
        app.drain_metric_results();
        assert!(!app.measuring);
    }

    #[test]
    fn start_psnr_bad_time_marks_all_included() {
        use crate::metrics::MetricCell;
        let p = std::env::temp_dir().join("rfmetrics-psnr-skip.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_ssim: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            skip: "abc".to_owned(), // unparseable: settings uncomparable
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        // Garbage settings can't be compared against the stored trim, so
        // even valid rows take the error (Python writes all targets too).
        for i in 0..2 {
            for cell in [&app.rows[i].psnr, &app.rows[i].ssim] {
                assert!(
                    matches!(cell, MetricCell::Error { msg } if msg == "bad time"),
                    "row {i} should be bad time, got {cell:?}",
                );
            }
        }
        assert!(!app.measuring);
    }

    #[test]
    fn start_psnr_all_done_toasts_without_running() {
        use crate::metrics::MetricCell;
        let p = std::env::temp_dir().join("rfmetrics-psnr-skipall.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(app.pending == 0);
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert!(toast.text.contains("Skipped 1 with existing PSNR"));
    }

    /// Checked "Plot window at start" opens the plot viewport when a run
    /// launches; unchecked (default) leaves it closed unless Plot is pressed.
    #[test]
    fn start_run_plot_at_start_opens_plot() {
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-plot-at-start.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            plot_at_start: true,
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        assert!(!app.show_plot);
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(app.show_plot);
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
    }

    #[test]
    fn start_run_plot_stays_closed_by_default() {
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-plot-default-closed.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        assert!(!app.plot_at_start);
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(!app.show_plot);
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
    }

    /// Regression: the jobs loop must iterate `fresh`, never `targets`.
    /// A Done row is skipped AND stays Done; only the fresh row runs.
    #[test]
    fn start_psnr_rerun_leaves_done_row_untouched() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-psnr-rerun.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        // Fabricate everything past the pre-flights so the run reaches the
        // jobs loop; the ffmpeg binary doesn't exist, so the worker fails
        // the spawn asynchronously and the test stays headless-safe.
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.rows[1].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        // Sync state right after Start: Done row untouched, fresh Running.
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Done { avg, .. } if (*avg - 30.0).abs() < 1e-9),
            "Done row must never re-enter Running, got {:?}",
            app.rows[0].psnr,
        );
        assert!(matches!(
            app.rows[1].psnr,
            MetricCell::Running { frame: 0, .. }
        ));
        assert!(app.measuring);
        // Let the doomed worker land, then settle.
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
            "Done row must survive the whole rerun, got {:?}",
            app.rows[0].psnr,
        );
        assert!(matches!(&app.rows[1].psnr, MetricCell::Error { .. }));
    }

    /// Guard rail: a Done value stamped with different trim settings is
    /// stale — changing Duration/Skip must recompute, never skip.
    #[test]
    fn start_psnr_changed_trim_recomputes_done_row() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-psnr-staletrim.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            duration: "10".to_owned(), // value was computed with clip 5
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: Some(5.0),
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        // Stale trim: the row re-enters the run instead of skipping.
        assert!(
            matches!(app.rows[0].psnr, MetricCell::Running { .. }),
            "stale-trim Done must recompute, got {:?}",
            app.rows[0].psnr,
        );
        assert!(app.measuring);
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
    }

    /// Same trim stamp still skips, even with nonzero settings.
    #[test]
    fn start_psnr_matching_trim_still_skips() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-psnr-sametrim.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            duration: "10".to_owned(),
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: Some(10.0),
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
            "matching-trim Done must skip, got {:?}",
            app.rows[0].psnr,
        );
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert!(toast.text.contains("Skipped 1 with existing PSNR"));
    }

    /// VMAF options change invalidates VMAF alone: the stale-stamped VMAF
    /// cell recomputes while a valid PSNR cell on the same row keeps
    /// skipping (no Reset needed).
    #[test]
    fn start_run_vmaf_settings_change_recomputes_vmaf_only() {
        use crate::metrics::MetricCell;
        use crate::metrics::vmaf::{Pooling, VmafCfg};
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-vmaf-restamp.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        // Current UI settings snapshot to subsample 1; the stored VMAF
        // value was computed under subsample 5.
        app.vmaf_subsample = "1".to_owned();
        app.vmaf_threads = "4".to_owned();
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].vmaf = MetricCell::Done {
            values: vec![90.0],
            avg: 90.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: Some(VmafCfg {
                model: "vmaf_v0.6.1.json".to_owned(),
                phone: false,
                scale: false,
                pooling: Pooling::Mean,
                subsample: 5,
                n_threads: 4,
            }),
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
            "valid PSNR must keep skipping, got {:?}",
            app.rows[0].psnr,
        );
        assert!(
            matches!(&app.rows[0].vmaf, MetricCell::Running { .. }),
            "stale-stamped VMAF must recompute, got {:?}",
            app.rows[0].vmaf,
        );
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        // Spawn fails headless (bogus binary): the cell records the error
        // while PSNR still holds its skipped value.
        assert!(matches!(&app.rows[0].vmaf, MetricCell::Error { .. }));
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
    }

    /// Matching VMAF stamp still skips: unchanged options recompute nothing.
    #[test]
    fn start_run_vmaf_matching_settings_still_skips() {
        use crate::metrics::MetricCell;
        use crate::metrics::vmaf::{Pooling, VmafCfg};
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-vmaf-samestamp.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.vmaf_subsample = "1".to_owned();
        app.vmaf_threads = "4".to_owned();
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].vmaf = MetricCell::Done {
            values: vec![90.0],
            avg: 90.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: Some(VmafCfg {
                model: "vmaf_v0.6.1.json".to_owned(),
                phone: false,
                scale: false,
                pooling: Pooling::Mean,
                subsample: 1,
                n_threads: 4,
            }),
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
        assert!(matches!(&app.rows[0].vmaf, MetricCell::Done { .. }));
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert_eq!(
            toast.text,
            "Skipped 1 with existing PSNR, VMAF (Reset to recompute)"
        );
    }

    /// Scaler change recomputes ffmpeg-backed columns but leaves FFVship
    /// ones alone (no scale stage there): a Bicubic-stamped PSNR cell
    /// recomputes under Lanczos while a Bicubic-stamped SSIM2 cell keeps
    /// skipping, no Reset needed.
    #[test]
    fn start_run_scaler_change_recomputes_ffmpeg_only() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-scaler-restamp.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            m_ssim2: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.scale_method = ScaleMethod::Lanczos;
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        let done = |avg: f64| MetricCell::Done {
            values: vec![avg],
            avg,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].psnr = done(30.0);
        app.rows[0].ssim2 = done(80.0);
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Running { .. }),
            "stale-stamped PSNR must recompute, got {:?}",
            app.rows[0].psnr,
        );
        assert!(
            matches!(&app.rows[0].ssim2, MetricCell::Done { .. }),
            "FFVship SSIM2 ignores the scaler, got {:?}",
            app.rows[0].ssim2,
        );
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        // Spawn fails headless (bogus binary): the cell records the error
        // while SSIM2 still holds its skipped value.
        assert!(matches!(&app.rows[0].psnr, MetricCell::Error { .. }));
        assert!(matches!(&app.rows[0].ssim2, MetricCell::Done { .. }));
    }

    /// Matching scaler stamp still skips: unchanged method recomputes nothing.
    #[test]
    fn start_run_matching_scaler_still_skips() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-scaler-samestamp.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.ref_info_data = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert!(toast.text.contains("Skipped 1 with existing PSNR"));
    }

    /// All-metrics skip lists every metric: the single toast slot must name
    /// PSNR, SSIM, and VMAF instead of collapsing to the last one (VMAF).
    #[test]
    fn start_run_skip_toast_lists_all_metrics() {
        use crate::metrics::MetricCell;
        use crate::metrics::vmaf::{Pooling, VmafCfg};
        let p = std::env::temp_dir().join("rfmetrics-skipall-metrics.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_ssim: true,
            m_vmaf: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        // Pin threads: the "auto" default resolves machine-dependently.
        app.vmaf_threads = "4".to_owned();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        let done = |avg: f64| MetricCell::Done {
            values: vec![avg],
            avg,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].psnr = done(30.0);
        app.rows[0].ssim = done(0.9);
        app.rows[0].vmaf = MetricCell::Done {
            values: vec![90.0],
            avg: 90.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: Some(VmafCfg {
                model: "vmaf_v0.6.1.json".to_owned(),
                phone: false,
                scale: false,
                pooling: Pooling::Mean,
                subsample: 1,
                n_threads: 4,
            }),
            scaler: ScaleMethod::Bicubic,
        };
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        // Shared skip set merges into one line instead of repeating it
        // per metric (previously only the VMAF line survived).
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert_eq!(
            toast.text,
            "Skipped 1 with existing PSNR, SSIM, VMAF (Reset to recompute)"
        );
    }

    /// VMAF recompute alongside skipped metrics lists shared filenames
    /// once: 2 rows skip PSNR+SSIM while VMAF alone runs on both.
    #[test]
    fn start_run_skip_toast_merges_shared_rows() {
        use crate::metrics::MetricCell;
        use crate::metrics::vmaf::{Pooling, VmafCfg};
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-skip-merge.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_ssim: true,
            m_vmaf: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.vmaf_subsample = "1".to_owned();
        app.vmaf_threads = "4".to_owned();
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        app.rows[1].display = "b.mp4".to_owned();
        let stale_vmaf = || MetricCell::Done {
            values: vec![90.0],
            avg: 90.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: Some(VmafCfg {
                model: "vmaf_v0.6.1.json".to_owned(),
                phone: false,
                scale: false,
                pooling: Pooling::Mean,
                subsample: 5,
                n_threads: 4,
            }),
            scaler: ScaleMethod::Bicubic,
        };
        for i in 0..2 {
            app.rows[i].psnr = MetricCell::Done {
                values: vec![30.0],
                avg: 30.0,
                exec_s: 1.0,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            };
            app.rows[i].ssim = MetricCell::Done {
                values: vec![0.9],
                avg: 0.9,
                exec_s: 1.0,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            };
            app.rows[i].vmaf = stale_vmaf();
            app.rows[i].info = Some(MediaInfo::default());
        }
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        // One line, filenames once — not repeated per metric.
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert_eq!(
            toast.text,
            "Skipped 2 with existing PSNR, SSIM: a.mp4, b.mp4"
        );
        assert!(matches!(&app.rows[0].vmaf, MetricCell::Running { .. }));
        assert!(matches!(&app.rows[1].vmaf, MetricCell::Running { .. }));
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
        assert!(matches!(&app.rows[1].ssim, MetricCell::Done { .. }));
    }

    /// SSIM-only run: only the SSIM cell enters the run, PSNR stays Idle.
    #[test]
    fn start_run_ssim_only_runs_ssim_cell() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-ssim-only.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_ssim: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        // Fabricate past the pre-flights; the binary doesn't exist so the
        // worker fails the spawn asynchronously (headless-safe).
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        assert!(matches!(
            app.rows[0].ssim,
            MetricCell::Running { frame: 0, .. }
        ));
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        assert!(matches!(&app.rows[0].ssim, MetricCell::Error { .. }));
    }

    /// Mixed run: valid PSNR skips while fresh SSIM on the same row runs.
    #[test]
    fn start_run_skips_done_psnr_but_runs_fresh_ssim() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-ssim-mixed.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_ssim: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].psnr = MetricCell::Done {
            values: vec![30.0],
            avg: 30.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(
            matches!(&app.rows[0].psnr, MetricCell::Done { .. }),
            "valid PSNR must skip, got {:?}",
            app.rows[0].psnr,
        );
        assert!(matches!(
            app.rows[0].ssim,
            MetricCell::Running { frame: 0, .. }
        ));
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
    }

    /// XPSNR-only run: only the XPSNR cell enters the run, the rest stay Idle.
    #[test]
    fn start_run_xpsnr_only_runs_xpsnr_cell() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-xpsnr-only.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_xpsnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        // Fabricate past the pre-flights; the binary doesn't exist so the
        // worker fails the spawn asynchronously (headless-safe).
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        assert!(matches!(app.rows[0].ssim, MetricCell::Idle));
        assert!(matches!(
            app.rows[0].xpsnr,
            MetricCell::Running { frame: 0, .. }
        ));
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        assert!(matches!(&app.rows[0].xpsnr, MetricCell::Error { .. }));
    }

    /// Mixed run: valid XPSNR skips while fresh PSNR on the same row runs.
    #[test]
    fn start_run_skips_done_xpsnr_but_runs_fresh_psnr() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-xpsnr-mixed.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_xpsnr: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].xpsnr = MetricCell::Done {
            values: vec![40.0],
            avg: 40.0,
            exec_s: 1.0,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(matches!(
            app.rows[0].psnr,
            MetricCell::Running { frame: 0, .. }
        ));
        assert!(
            matches!(&app.rows[0].xpsnr, MetricCell::Done { .. }),
            "valid XPSNR must skip, got {:?}",
            app.rows[0].xpsnr,
        );
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].xpsnr, MetricCell::Done { .. }));
    }

    /// SSIM2-only run: only the SSIM2 cell enters the run, the rest stay
    /// Idle. The binary doesn't exist so the worker fails the spawn
    /// asynchronously (headless-safe).
    #[test]
    fn start_run_ssim2_only_runs_ssim2_cell() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-ssim2-only.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_ssim2: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.ffvship.path = Some(std::path::PathBuf::from("rfmetrics-no-such-binary"));
        app.ffvship.usable = true;
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(app.measuring);
        assert!(matches!(app.rows[0].psnr, MetricCell::Idle));
        assert!(matches!(
            app.rows[0].ssim2,
            MetricCell::Running { frame: 0, .. }
        ));
        for _ in 0..200 {
            app.drain_metric_results();
            if !app.measuring {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.measuring);
        assert!(matches!(&app.rows[0].ssim2, MetricCell::Error { .. }));
    }

    /// Missing FFVship binary: FFVship cells error out without running,
    /// while an ffmpeg-only selection is unaffected by the missing binary.
    #[test]
    fn start_run_ffvship_missing_binary_errors_cells() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-ffvship-missing.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_ssim2: true,
            m_but: true,
            m_cvvdp: true,
            m_vmaf: false,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.ffvship.path = None;
        app.ffvship.usable = false;
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        for cell in [&app.rows[0].ssim2, &app.rows[0].butter, &app.rows[0].cvvdp] {
            assert!(
                matches!(cell, MetricCell::Error { msg } if msg == "FFVship not found"),
                "expected FFVship gate, got {cell:?}"
            );
        }
        assert!(app.toast.is_some());
    }

    /// BUTTER ranks min-wins on every stat (lower is better); other
    /// metrics keep max-wins.
    #[test]
    fn butter_rank_is_min_wins() {
        use crate::metrics::{MetricCell, StatRank};
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        for (i, avg) in [3.0, 5.0].into_iter().enumerate() {
            app.rows[i].butter = MetricCell::Done {
                values: vec![avg, avg],
                avg,
                exec_s: 1.0,
                skip: None,
                clip_dur: None,
                vmaf_cfg: None,
                scaler: ScaleMethod::Bicubic,
            };
            let stats = app.rows[i].butter.done_stats();
            app.rows[i].butter_cache.stats = stats;
        }
        app.refresh_ranks(crate::metrics::ffmpeg::MetricKind::But);
        assert_eq!(app.rows[0].butter_cache.ranks[0], StatRank::Best);
        assert_eq!(app.rows[1].butter_cache.ranks[0], StatRank::Worst);
    }

    /// State round-trip: snapshot captures boxes, toggles, options, and
    /// per-row include flags; apply restores them onto a fresh app.
    #[test]
    fn state_snapshot_apply_round_trip() {
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-state-rt.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            ref_path: "C:/vids/ref.mp4".to_owned(),
            skip: "5".to_owned(),
            duration: "00:10".to_owned(),
            m_psnr: true,
            m_vmaf: false,
            m_ssim2: true,
            vmaf_phone: true,
            vmaf_pooling: "Harmonic Mean".to_owned(),
            vmaf_threads: "4".to_owned(),
            scale_method: ScaleMethod::Lanczos,
            plot_at_start: true,
            plot_size: crate::plot::PlotSize::S1600,
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows.push(psnr_test_row("C:/vids/b.mp4", false));
        app.rows[0].info = Some(MediaInfo::default());
        let snap = app.snapshot();
        std::fs::remove_file(&p).ok();

        let mut fresh = RFMetricsApp::default();
        fresh.apply_state(Some(snap));
        assert_eq!(fresh.ref_path, "C:/vids/ref.mp4");
        assert_eq!(fresh.skip, "5");
        assert_eq!(fresh.duration, "00:10");
        assert!(fresh.m_psnr && !fresh.m_vmaf && fresh.m_ssim2);
        assert!(fresh.vmaf_phone);
        assert_eq!(fresh.vmaf_pooling, "Harmonic Mean");
        assert_eq!(fresh.vmaf_threads, "4");
        assert_eq!(fresh.scale_method, ScaleMethod::Lanczos);
        assert!(fresh.plot_at_start);
        assert_eq!(fresh.plot_size, crate::plot::PlotSize::S1600);
        // psnr_test_row paths don't exist on disk: only pre-existing rows
        // could restore, so the queue stays empty here.
        assert!(fresh.rows.is_empty());
    }

    /// State apply restores live queue files with their include flags and
    /// drops missing ones.
    #[test]
    fn state_apply_restores_files_with_include() {
        let a = std::env::temp_dir().join("rfmetrics-state-a.tmp");
        let b = std::env::temp_dir().join("rfmetrics-state-b.tmp");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let state = crate::state::AppState {
            files: Some(vec![
                crate::state::FileEntry {
                    path: a.to_string_lossy().into_owned(),
                    include: true,
                },
                crate::state::FileEntry {
                    path: b.to_string_lossy().into_owned(),
                    include: false,
                },
                crate::state::FileEntry {
                    path: "C:/no/such/file.mp4".to_owned(),
                    include: true,
                },
            ]),
            ..Default::default()
        };
        let mut app = RFMetricsApp::default();
        app.apply_state(Some(state));
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
        assert_eq!(app.rows.len(), 2);
        assert!(app.rows[0].include);
        assert!(!app.rows[1].include);
        // Absent keys keep live defaults (VMAF-only).
        assert!(!app.m_psnr && app.m_vmaf);
    }

    /// State apply validates options: unknown models/pooling/subsamples
    /// keep live values instead of poisoning the run.
    #[test]
    fn state_apply_validates_vmaf_options() {
        let mut app = RFMetricsApp::default();
        let state = crate::state::AppState {
            vmaf: crate::state::VmafState {
                model: Some("evil.json".to_owned()),
                pooling: Some("Median".to_owned()),
                subsample: Some("7".to_owned()),
                phone: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        app.apply_state(Some(state));
        assert_eq!(app.vmaf_model, "vmaf_v0.6.1.json");
        assert_eq!(app.vmaf_pooling, "Mean");
        assert_eq!(app.vmaf_subsample, "1");
        assert!(app.vmaf_phone);
    }

    /// Issue #7: restored ticks for filters this ffmpeg build lacks are
    /// forced off (their header checkboxes render disabled).
    #[test]
    fn state_apply_unticks_unsupported_filters() {
        use crate::metrics::ffmpeg::MetricKind;
        let mut app = RFMetricsApp::default();
        // Simulate a w32threads-style build: everything but libvmaf.
        app.ffmpeg.supported_metrics = vec![MetricKind::Psnr, MetricKind::Ssim, MetricKind::Xpsnr];
        let state = crate::state::AppState {
            metrics: crate::state::MetricsState {
                psnr: Some(true),
                ssim: Some(true),
                vmaf: Some(true),
                xpsnr: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        app.apply_state(Some(state));
        assert!(app.m_psnr && app.m_ssim && app.m_xpsnr);
        assert!(!app.m_vmaf);
    }

    /// Issue #7 backstop: a ticked-but-unsupported metric never reaches
    /// the worker — with nothing runnable Start is a no-op toast.
    #[test]
    fn start_run_unsupported_vmaf_is_noop() {
        use crate::metrics::ffmpeg::MetricKind;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-unsupported-vmaf.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: false,
            m_ssim: false,
            m_vmaf: true,
            ref_path: p.to_string_lossy().into_owned(),
            ..RFMetricsApp::default()
        };
        app.ffmpeg.supported_metrics = vec![MetricKind::Psnr, MetricKind::Ssim, MetricKind::Xpsnr];
        app.ref_info_data = Some(MediaInfo::default());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].vmaf, crate::metrics::MetricCell::Idle));
    }

    /// CsvReport stashes the summary; the UI frame toasts it (drain has
    /// no timestamp). Stale generations stay silent.
    #[test]
    fn csv_report_stash_and_generation() {
        use super::MetricMsg;
        let mut app = RFMetricsApp::default();
        app.metric_tx
            .send(MetricMsg::CsvReport {
                generation: app.run_generation,
                ok: 3,
                errors: Vec::new(),
            })
            .unwrap();
        assert!(app.drain_metric_results());
        assert_eq!(
            app.csv_report,
            Some((3, Vec::new())),
            "fresh report must stash"
        );
        // Stale report: dropped without touching the stash.
        app.run_generation = app.run_generation.wrapping_add(1);
        app.metric_tx
            .send(MetricMsg::CsvReport {
                generation: app.run_generation.wrapping_sub(1),
                ok: 9,
                errors: vec!["x".to_owned()],
            })
            .unwrap();
        assert!(app.drain_metric_results());
        assert_eq!(app.csv_report, Some((3, Vec::new())));
    }

    /// CSV options persist and restore like the other Options keys.
    #[test]
    fn state_apply_restores_csv_options() {
        let mut app = RFMetricsApp::default();
        let state = crate::state::AppState {
            options: crate::state::OptionsState {
                csv_export: Some(true),
                csv_dir: Some("D:/csv".to_owned()),
                ..Default::default()
            },
            ..Default::default()
        };
        app.apply_state(Some(state));
        assert!(app.csv_export);
        assert_eq!(app.csv_dir, "D:/csv");
        assert!(app.snapshot().options.csv_export == Some(true));
    }

    /// `is_state_dirty` must agree with `snapshot() != saved_snapshot` for
    /// every persisted field; a missed field silently stops saving.
    #[test]
    fn dirty_check_matches_snapshot_compare() {
        let mut app = RFMetricsApp::default();
        app.saved_snapshot = app.snapshot();
        assert!(!app.is_state_dirty());

        // One mutation at a time: each must read dirty under both paths,
        // and re-saving must read clean.
        let check = |app: &mut RFMetricsApp| {
            assert!(app.is_state_dirty());
            assert_ne!(app.snapshot(), app.saved_snapshot);
            app.saved_snapshot = app.snapshot();
            assert!(!app.is_state_dirty());
        };
        app.ref_path = "C:/vids/ref.mp4".to_owned();
        check(&mut app);
        app.skip = "5".to_owned();
        check(&mut app);
        app.duration = "00:10".to_owned();
        check(&mut app);
        app.m_psnr = !app.m_psnr;
        check(&mut app);
        app.m_ssim = !app.m_ssim;
        check(&mut app);
        app.m_vmaf = !app.m_vmaf;
        check(&mut app);
        app.m_xpsnr = !app.m_xpsnr;
        check(&mut app);
        app.m_ssim2 = !app.m_ssim2;
        check(&mut app);
        app.m_but = !app.m_but;
        check(&mut app);
        app.m_cvvdp = !app.m_cvvdp;
        check(&mut app);
        app.vmaf_model = "other.json".to_owned();
        check(&mut app);
        app.vmaf_phone = !app.vmaf_phone;
        check(&mut app);
        app.vmaf_scale = !app.vmaf_scale;
        check(&mut app);
        app.vmaf_pooling = "Harmonic Mean".to_owned();
        check(&mut app);
        app.vmaf_subsample = "2".to_owned();
        check(&mut app);
        app.vmaf_threads = "4".to_owned();
        check(&mut app);
        app.scale_method = ScaleMethod::Lanczos;
        check(&mut app);
        app.plot_at_start = !app.plot_at_start;
        check(&mut app);
        app.plot_size = crate::plot::PlotSize::S1600;
        check(&mut app);
        app.csv_export = !app.csv_export;
        check(&mut app);
        app.csv_dir = "D:/csv".to_owned();
        check(&mut app);
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        check(&mut app);
        app.rows[0].include = false;
        check(&mut app);
        app.rows[0].path = "C:/vids/b.mp4".to_owned();
        app.rows[0].key = norm_key("C:/vids/b.mp4");
        check(&mut app);
        // Order-sensitive like the snapshot vec.
        app.rows.push(psnr_test_row("C:/vids/c.mp4", true));
        check(&mut app);
        app.rows.swap(0, 1);
        assert!(app.is_state_dirty());
        assert_ne!(app.snapshot(), app.saved_snapshot);
    }

    /// Before/after timing: full `snapshot()` clone+compare (old per-frame
    /// path) vs `is_state_dirty` (new path) over a 200-row queue.
    #[test]
    fn dirty_check_perf() {
        let mut app = RFMetricsApp::default();
        for i in 0..200 {
            app.rows.push(psnr_test_row(
                &format!("C:/vids/clip_{i:04}.mp4"),
                i % 2 == 0,
            ));
        }
        app.saved_snapshot = app.snapshot();
        assert!(!app.is_state_dirty());

        let n = 2000;
        let t0 = std::time::Instant::now();
        let mut dirty_old = false;
        for _ in 0..n {
            dirty_old |= app.snapshot() != app.saved_snapshot;
        }
        let old_ms = t0.elapsed();
        let t1 = std::time::Instant::now();
        let mut dirty_new = false;
        for _ in 0..n {
            dirty_new |= app.is_state_dirty();
        }
        let new_ms = t1.elapsed();
        assert_eq!(dirty_old, dirty_new);
        eprintln!(
            "autosave dirty-check ({n} iters, 200 rows): snapshot-clone {old_ms:?} vs field-compare {new_ms:?}"
        );
    }

    /// Before/after timing for worker-message routing: re-normalizing every
    /// row per lookup (old `Progress`-rate path) vs stored-key compare.
    #[test]
    fn row_routing_perf() {
        let mut app = RFMetricsApp::default();
        for i in 0..200 {
            app.rows.push(psnr_test_row(
                &format!("C:/vids/clip_{i:04}.mp4"),
                i % 2 == 0,
            ));
        }
        let key = app.rows[100].key.clone();
        let n = 5000;
        let t0 = std::time::Instant::now();
        let mut found_old = 0;
        for _ in 0..n {
            if let Some(r) = app.rows.iter().find(|r| norm_key(&r.path) == key) {
                found_old += r.path.len();
            }
        }
        let old = t0.elapsed();
        let t1 = std::time::Instant::now();
        let mut found_new = 0;
        for _ in 0..n {
            if let Some(r) = app.rows.iter().find(|r| r.key == key) {
                found_new += r.path.len();
            }
        }
        let new = t1.elapsed();
        assert_eq!(found_old, found_new);
        eprintln!(
            "row routing ({n} lookups, 200 rows): norm_key-scan {old:?} vs stored-key {new:?}"
        );
    }

    /// Done/Error arrival freezes the rendered cell text (equal to
    /// `cell_text()` by construction); Reset clears it.
    #[test]
    fn cell_text_cache_set_and_cleared() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        use crate::metrics::ffmpeg::MetricKind;
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.run_generation = 1;
        let done = |error: Option<String>| MetricMsg::Done {
            generation: 1,
            kind: MetricKind::Psnr,
            key: norm_key("C:/vids/a.mp4"),
            values: vec![29.0, 30.0, 31.0],
            avg: Some(30.123_456),
            exec_s: 1.0,
            error,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        app.metric_tx.send(done(None)).unwrap();
        app.drain_metric_results();
        assert_eq!(app.rows[0].psnr_cache.text, "30.1235");
        assert_eq!(app.rows[0].psnr_cache.text, app.rows[0].psnr.cell_text());
        app.metric_tx.send(done(Some("boom".to_owned()))).unwrap();
        app.drain_metric_results();
        assert!(matches!(app.rows[0].psnr, MetricCell::Error { .. }));
        assert_eq!(app.rows[0].psnr_cache.text, "boom");
        app.reset_psnr();
        assert!(app.rows[0].psnr_cache.text.is_empty());
    }

    /// No-live-feed first Done (VMAF) on the shown tab owes one auto-fit
    /// poke; a later sibling row must not disturb the first fit.
    #[test]
    fn vmaf_first_done_arms_follow() {
        use super::MetricMsg;
        use crate::metrics::MetricCell;
        use crate::metrics::ffmpeg::MetricKind;
        let done = |key: &str| MetricMsg::Done {
            generation: 0,
            kind: MetricKind::Vmaf,
            key: norm_key(key),
            values: vec![90.0, 91.0],
            avg: Some(90.5),
            exec_s: 1.0,
            error: None,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        let mut app = RFMetricsApp {
            show_plot: true,
            plot_tab: MetricKind::Vmaf,
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].vmaf = MetricCell::Running {
            frame: 0,
            values: Vec::new(),
        };
        app.metric_tx.send(done("C:/vids/a.mp4")).unwrap();
        app.drain_metric_results();
        assert!(
            app.plot_follow_pending,
            "first VMAF Done on the shown tab must arm a refit"
        );
        app.plot_follow_pending = false;
        app.rows.push(psnr_test_row("C:/vids/b.mp4", true));
        app.metric_tx.send(done("C:/vids/b.mp4")).unwrap();
        app.drain_metric_results();
        assert!(
            !app.plot_follow_pending,
            "later rows must not disturb the first fit"
        );
    }

    /// Arming rules: live-feed metrics, hidden tabs, closed window, and
    /// errored runs never arm the poke.
    #[test]
    fn follow_arm_rules() {
        use super::MetricMsg;
        use crate::metrics::ffmpeg::MetricKind;
        let done = |kind, key: &str, error: Option<String>| MetricMsg::Done {
            generation: 0,
            kind,
            key: norm_key(key),
            values: vec![30.0],
            avg: Some(30.0),
            exec_s: 1.0,
            error,
            skip: None,
            clip_dur: None,
            vmaf_cfg: None,
            scaler: ScaleMethod::Bicubic,
        };
        // Live-feed metric never arms, even first on the shown tab.
        let mut app = RFMetricsApp {
            show_plot: true,
            plot_tab: MetricKind::Psnr,
            ..RFMetricsApp::default()
        };
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.metric_tx
            .send(done(MetricKind::Psnr, "C:/vids/a.mp4", None))
            .unwrap();
        app.drain_metric_results();
        assert!(!app.plot_follow_pending);
        // First VMAF data on a hidden tab never arms.
        app.metric_tx
            .send(done(MetricKind::Vmaf, "C:/vids/a.mp4", None))
            .unwrap();
        app.drain_metric_results();
        assert!(!app.plot_follow_pending);
        // Closed window never arms.
        app.show_plot = false;
        app.plot_tab = MetricKind::Vmaf;
        app.metric_tx
            .send(done(MetricKind::Vmaf, "C:/vids/a.mp4", None))
            .unwrap();
        app.drain_metric_results();
        assert!(!app.plot_follow_pending);
        // Errored VMAF never arms.
        app.show_plot = true;
        app.metric_tx
            .send(done(
                MetricKind::Vmaf,
                "C:/vids/a.mp4",
                Some("boom".to_owned()),
            ))
            .unwrap();
        app.drain_metric_results();
        assert!(!app.plot_follow_pending);
    }

    /// Rerun start clears the frozen text synchronously (asserted before the
    /// doomed worker's error reply is ever drained, so no race).
    #[test]
    fn cell_text_cache_cleared_on_rerun() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let dir = std::env::temp_dir();
        let ref_file = dir.join("rfmetrics-rerun-ref.tmp");
        let fake_exe = dir.join("rfmetrics-rerun-exe.tmp");
        std::fs::write(&ref_file, b"x").unwrap();
        std::fs::write(&fake_exe, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_psnr: true,
            m_vmaf: false,
            ref_path: ref_file.to_string_lossy().into_owned(),
            ref_info_data: Some(MediaInfo::default()),
            ..RFMetricsApp::default()
        };
        app.ffmpeg.path = Some(fake_exe.clone());
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.rows[0].info = Some(MediaInfo::default());
        app.rows[0].psnr_cache.text = "30.1235".to_owned();
        app.start_run(0.0);
        assert!(matches!(app.rows[0].psnr, MetricCell::Running { .. }));
        assert!(app.rows[0].psnr_cache.text.is_empty());
        assert!(app.rows[0].psnr_cache.stats.is_none());
        app.reset_psnr();
        std::fs::remove_file(&ref_file).ok();
        std::fs::remove_file(&fake_exe).ok();
    }

    /// Before/after timing for the table cell string work: per-frame
    /// `cell_text()` + `tooltip()` (old) vs borrowing statics (new).
    /// Models 200 rows × 7 idle columns.
    #[test]
    fn cell_text_perf() {
        use crate::metrics::{MetricCell, ffmpeg::MetricKind};
        use std::hint::black_box;
        let kinds = MetricKind::ALL;
        let cells: Vec<MetricCell> = (0..200 * kinds.len()).map(|_| MetricCell::Idle).collect();
        // Borrowed text must read identically to the formatted one.
        assert_eq!(cells[0].cell_text(), "N/A");
        let n = 500;
        let t0 = std::time::Instant::now();
        let mut len_old = 0;
        for _ in 0..n {
            for (cell, kind) in cells.iter().zip(kinds.iter().cycle()) {
                let text = cell.cell_text();
                let tip = cell.tooltip(kind.name());
                len_old += text.len() + tip.len();
            }
        }
        let old = t0.elapsed();
        black_box(len_old);
        let t1 = std::time::Instant::now();
        let mut len_new = 0;
        for _ in 0..n {
            for cell in &cells {
                let text: &str = match cell {
                    MetricCell::Idle => "N/A",
                    _ => unreachable!("idle-only fixture"),
                };
                len_new += text.len();
            }
        }
        let new = t1.elapsed();
        black_box(len_new);
        eprintln!(
            "cell text ({n} frames, 1400 idle cells): format-per-frame {old:?} vs borrow {new:?}"
        );
    }

    /// Idle-heartbeat budget: our-code CPU of one heartbeat frame (dirty
    /// check + borrowed cell text for 200×7 idle cells + fit/decimate of
    /// 5×5000 plot series) ×2/s, vs zero frames without the heartbeat.
    /// egui layout/tessellation is not included (headless can't render).
    #[test]
    fn idle_heartbeat_budget() {
        use crate::metrics::{MetricCell, ffmpeg::MetricKind};
        use std::hint::black_box;
        let mut app = RFMetricsApp::default();
        for i in 0..200 {
            app.rows.push(psnr_test_row(
                &format!("C:/vids/clip_{i:04}.mp4"),
                i % 2 == 0,
            ));
        }
        app.saved_snapshot = app.snapshot();
        let raw: Vec<Vec<f64>> = (0..5)
            .map(|s| {
                (0..5000)
                    .map(|i| 30.0 + s as f64 + (i as f64 * 0.01).sin() * 5.0)
                    .collect()
            })
            .collect();
        let borrowed: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
        let points: Vec<Vec<egui_plot::PlotPoint>> = raw
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(i, &y)| egui_plot::PlotPoint::new(i as f64 + 1.0, y))
                    .collect()
            })
            .collect();
        let n = 200;
        let t0 = std::time::Instant::now();
        let mut acc = 0usize;
        for _ in 0..n {
            acc += app.is_state_dirty() as usize;
            for row in &app.rows {
                for kind in MetricKind::ALL {
                    let text: &str = match row.cell(kind) {
                        MetricCell::Idle => "N/A",
                        MetricCell::Running { .. } => unreachable!("idle fixture"),
                        MetricCell::Done { .. } => &row.cached(kind).text,
                        MetricCell::Error { msg } => msg,
                    };
                    acc += text.len();
                }
            }
            black_box(crate::plot::fit_limits(
                &borrowed,
                crate::plot::PSNR_LO,
                crate::plot::PSNR_HI,
            ));
            for p in &points {
                acc += crate::plot::decimate_minmax(p, 1000).points().len();
            }
        }
        let per_frame = t0.elapsed() / n;
        black_box(acc);
        let per_sec = per_frame * 2;
        eprintln!(
            "idle heartbeat: {per_frame:?}/frame our-code ×2/s = {per_sec:?}/s (~{:.3}% of a core); without heartbeat: 0 frames",
            per_sec.as_secs_f64() * 100.0,
        );
    }

    /// Drain activity flags: queued message → `true`; empty channel → `false`.
    /// Stale messages count as traffic (one harmless extra frame).
    #[test]
    fn drains_report_activity() {
        use super::{MetricMsg, PngSaveMsg, ThumbMsg};
        use crate::metrics::ffmpeg::MetricKind;
        let ctx = egui::Context::default();
        let mut app = RFMetricsApp::default();
        assert!(!app.drain_probe_results());
        assert!(!app.drain_metric_results());
        assert!(!app.drain_png_results(&ctx, 0.0));
        assert!(!app.drain_thumbs(&ctx));
        // Stale metric message (wrong generation) is still traffic.
        app.metric_tx
            .send(MetricMsg::Progress {
                generation: 999,
                kind: MetricKind::Psnr,
                key: "gone".to_owned(),
                frame: 1,
            })
            .unwrap();
        assert!(app.drain_metric_results());
        assert!(!app.drain_metric_results());
        // Stale probe message likewise.
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: 999,
                text: "stale".to_owned(),
                info: None,
            })
            .unwrap();
        assert!(app.drain_probe_results());
        assert!(!app.drain_probe_results());
        // PNG worker reply lands a toast.
        app.png_tx
            .send(PngSaveMsg::Saved {
                path: std::path::PathBuf::from("C:/vids/plot.png"),
            })
            .unwrap();
        assert!(app.drain_png_results(&ctx, 0.0));
        assert!(app.toast.is_some());
        assert!(!app.drain_png_results(&ctx, 0.0));
        // Empty-image thumbnail clears without touching the GPU path.
        app.thumb_tx
            .send(ThumbMsg {
                generation: app.thumb_generation,
                image: None,
            })
            .unwrap();
        assert!(app.drain_thumbs(&ctx));
        assert!(!app.drain_thumbs(&ctx));
        // Refresh helpers propagate their drain flags (paths unchanged, so
        // no worker spawns — pure drain reporting).
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: 999,
                text: "stale".to_owned(),
                info: None,
            })
            .unwrap();
        assert!(app.refresh_ref_info());
        assert!(!app.refresh_ref_info());
        app.thumb_tx
            .send(ThumbMsg {
                generation: app.thumb_generation,
                image: None,
            })
            .unwrap();
        assert!(app.refresh_thumbnail(&ctx));
        assert!(!app.refresh_thumbnail(&ctx));
    }

    /// FPS-counter cost across regimes, one run on identical fixtures
    /// (200 idle rows, 5×5000 plot series, debug build = upper bound):
    /// old frame (snapshot clone+compare, per-cell format+tooltip,
    /// fit/decimate, FPS format every frame) vs new frame (borrowed text,
    /// dirty check, fit/decimate, FPS format), plus the counter label
    /// alone (format every frame vs only when the rounded value changes).
    #[test]
    fn fps_counter_budget() {
        use crate::metrics::{MetricCell, ffmpeg::MetricKind};
        use std::hint::black_box;
        let mut app = RFMetricsApp::default();
        for i in 0..200 {
            app.rows.push(psnr_test_row(
                &format!("C:/vids/clip_{i:04}.mp4"),
                i % 2 == 0,
            ));
        }
        app.saved_snapshot = app.snapshot();
        let kinds = MetricKind::ALL;
        let cells: Vec<MetricCell> = (0..200 * kinds.len()).map(|_| MetricCell::Idle).collect();
        let raw: Vec<Vec<f64>> = (0..5)
            .map(|s| {
                (0..5000)
                    .map(|i| 30.0 + s as f64 + (i as f64 * 0.01).sin() * 5.0)
                    .collect()
            })
            .collect();
        let borrowed: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
        let points: Vec<Vec<egui_plot::PlotPoint>> = raw
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(i, &y)| egui_plot::PlotPoint::new(i as f64 + 1.0, y))
                    .collect()
            })
            .collect();
        let n = 100;
        // Old frame: everything formatted per frame.
        let t0 = std::time::Instant::now();
        let mut acc = 0usize;
        for _ in 0..n {
            acc += (app.snapshot() != app.saved_snapshot) as usize;
            for (cell, kind) in cells.iter().zip(kinds.iter().cycle()) {
                let text = cell.cell_text();
                let tip = cell.tooltip(kind.name());
                acc += text.len() + tip.len();
            }
            black_box(crate::plot::fit_limits(
                &borrowed,
                crate::plot::PSNR_LO,
                crate::plot::PSNR_HI,
            ));
            for p in &points {
                acc += crate::plot::decimate_minmax(p, 1000).points().len();
            }
            acc += format!("FPS: {:.0}", black_box(59.7)).len();
        }
        let old_frame = t0.elapsed() / n;
        // New frame: borrows + dirty check.
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            acc += app.is_state_dirty() as usize;
            for cell in &cells {
                let text: &str = match cell {
                    MetricCell::Idle => "N/A",
                    _ => unreachable!("idle fixture"),
                };
                acc += text.len();
            }
            black_box(crate::plot::fit_limits(
                &borrowed,
                crate::plot::PSNR_LO,
                crate::plot::PSNR_HI,
            ));
            for p in &points {
                acc += crate::plot::decimate_minmax(p, 1000).points().len();
            }
            acc += format!("FPS: {:.0}", black_box(59.7)).len();
        }
        let new_frame = t1.elapsed() / n;
        black_box(acc);
        // Counter label alone over a jittering fps stream.
        let stream: Vec<f64> = (0..10_000)
            .map(|i| 59.5 + (i as f64 * 0.37).sin())
            .collect();
        let t2 = std::time::Instant::now();
        let mut chars_a = 0;
        for &f in &stream {
            chars_a += format!("FPS: {f:.0}").len();
        }
        let always = t2.elapsed();
        black_box(chars_a);
        let t3 = std::time::Instant::now();
        let mut chars_b = 0;
        let mut last = i64::MIN;
        for &f in &stream {
            let r = f.round() as i64;
            if r != last {
                last = r;
                chars_b += format!("FPS: {f:.0}").len();
            }
        }
        let on_change = t3.elapsed();
        black_box(chars_b);
        let pct = |per_frame: std::time::Duration, hz: f64| per_frame.as_secs_f64() * hz * 100.0;
        eprintln!(
            "fps counter: old frame {old_frame:?} | new frame {new_frame:?}\n\
             regimes (% of a core, our-code only): old-measuring 60Hz {:.2}% | old-idle 10Hz {:.2}% | \
             new-measuring 10Hz {:.2}% | new-idle 2Hz {:.3}%\n\
             counter label 10k samples: format-always {always:?} vs on-change {on_change:?}",
            pct(old_frame, 60.0),
            pct(old_frame, 10.0),
            pct(new_frame, 10.0),
            pct(new_frame, 2.0),
        );
    }

    /// Thumbnail duration reuse: only the completed probe's own path
    /// qualifies; anything else (or no/zero duration) falls back.
    #[test]
    fn thumb_duration_reuse_rules() {
        use super::thumb_duration;
        assert_eq!(
            thumb_duration("C:/r.mp4", "C:/r.mp4", Some(63.0)),
            Some(63.0)
        );
        // Stale cache (other path), missing, or non-positive: re-probe.
        assert_eq!(thumb_duration("C:/r.mp4", "C:/old.mp4", Some(63.0)), None);
        assert_eq!(thumb_duration("C:/r.mp4", "", Some(63.0)), None);
        assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", None), None);
        assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", Some(0.0)), None);
        assert_eq!(thumb_duration("C:/r.mp4", "C:/r.mp4", Some(-1.0)), None);
        assert_eq!(thumb_duration("", "", Some(63.0)), None);
    }

    /// The reference drain records which path its info was probed from.
    #[test]
    fn ref_drain_tracks_info_path() {
        let mut app = RFMetricsApp {
            last_spawned_ref: "C:/vids/r.mp4".to_owned(),
            ..RFMetricsApp::default()
        };
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: app.ref_generation,
                text: "info".to_owned(),
                info: None,
            })
            .unwrap();
        app.drain_probe_results();
        assert_eq!(app.ref_info_path, "C:/vids/r.mp4");
        // Stale generation leaves the tracked path alone.
        app.last_spawned_ref = "C:/vids/new.mp4".to_owned();
        app.ref_generation = app.ref_generation.wrapping_add(1);
        app.probe_tx
            .send(ProbeMsg::Reference {
                generation: 0,
                text: "stale".to_owned(),
                info: None,
            })
            .unwrap();
        app.drain_probe_results();
        assert_eq!(app.ref_info_path, "C:/vids/r.mp4");
    }
}
