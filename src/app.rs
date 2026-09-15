use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::metrics::ffmpeg::MetricKind;

#[derive(Debug)]
struct QueueRow {
    path: String,
    display: String,
    include: bool,
    selected: bool,
    media: String,
    media_tip: String,
    info: Option<crate::probe::MediaInfo>,
    psnr: crate::metrics::MetricCell,
    ssim: crate::metrics::MetricCell,
}

impl QueueRow {
    fn cell(&self, kind: MetricKind) -> &crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &self.psnr,
            MetricKind::Ssim => &self.ssim,
        }
    }

    fn cell_mut(&mut self, kind: MetricKind) -> &mut crate::metrics::MetricCell {
        match kind {
            MetricKind::Psnr => &mut self.psnr,
            MetricKind::Ssim => &mut self.ssim,
        }
    }
}

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
    },
    /// End of the worker loop; `aborted` settles still-Running cells to
    /// Idle while keeping finished (`Done`) results on screen.
    Finished { generation: u64, aborted: bool },
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
    /// Bumped on every ref change; worker results with an older generation
    /// are stale (typed-through) and discarded.
    ref_generation: u64,
    thumb_tx: Sender<ThumbMsg>,
    thumb_rx: Receiver<ThumbMsg>,
    thumb_tex: Option<egui::TextureHandle>,
    thumb_loading: bool,
    last_thumb_path: String,
    thumb_generation: u64,
}

impl Default for RFMetricsApp {
    fn default() -> Self {
        let ffmpeg = crate::binaries::ffmpeg_info();
        let ffvship = crate::binaries::ffvship_info();
        let ffprobe = crate::binaries::ffprobe_path(ffmpeg.path.as_deref());
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let (metric_tx, metric_rx) = std::sync::mpsc::channel();
        Self {
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
            vmaf_phone: true,
            vmaf_scale: false,
            vmaf_pooling: "Mean".to_owned(),
            vmaf_subsample: "1".to_owned(),
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
            ref_generation: 0,
            thumb_tx,
            thumb_rx,
            thumb_tex: None,
            thumb_loading: false,
            last_thumb_path: String::new(),
            thumb_generation: 0,
        }
    }
}

impl RFMetricsApp {
    /// Apply any probe results that arrived since the last frame. Stale
    /// reference results (typed-through while a worker was running) are
    /// dropped via the generation check.
    fn drain_probe_results(&mut self) {
        while let Ok(msg) = self.probe_rx.try_recv() {
            match msg {
                ProbeMsg::Reference {
                    generation,
                    text,
                    info,
                } => {
                    if generation == self.ref_generation {
                        self.ref_info = text;
                        self.ref_info_data = info;
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
                    if let Some(row) = self.rows.iter_mut().find(|r| norm_key(&r.path) == key) {
                        row.media = media;
                        row.media_tip = tip;
                        row.info = info;
                    }
                }
            }
        }
    }

    /// Re-probe only when the path actually changed, and only off the UI
    /// thread: cheap cases (empty/missing/no ffprobe) resolve inline, an
    /// existing file spawns a worker and shows "Probing…" meanwhile.
    fn refresh_ref_info(&mut self) {
        self.drain_probe_results();
        if self.ref_path == self.last_spawned_ref {
            return;
        }
        self.last_spawned_ref = self.ref_path.clone();
        self.ref_generation = self.ref_generation.wrapping_add(1);
        if self.ref_path.trim().is_empty() {
            self.ref_info =
                "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-"
                    .to_owned();
            self.ref_info_data = None;
            return;
        }
        if !Path::new(&self.ref_path).is_file() {
            self.ref_info = "File not found".to_owned();
            self.ref_info_data = None;
            return;
        }
        if self.ffprobe.is_none() {
            self.ref_info = "ffprobe not found".to_owned();
            self.ref_info_data = None;
            return;
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
    }

    /// Apply arrived thumbnails; stale generations (typed-through) are dropped.
    fn drain_thumbs(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.thumb_rx.try_recv() {
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
    }

    /// Spawn a dedicated ffmpeg worker when the ref path changed. Cheap cases
    /// clear inline; the worker sends duration-aware extracts back on the
    /// thumb channel and repaints via the cloned ctx.
    fn refresh_thumbnail(&mut self, ctx: &egui::Context) {
        self.drain_thumbs(ctx);
        if self.ref_path == self.last_thumb_path {
            return;
        }
        self.last_thumb_path = self.ref_path.clone();
        self.thumb_generation = self.thumb_generation.wrapping_add(1);
        self.thumb_tex = None;
        if self.ref_path.trim().is_empty() || !Path::new(&self.ref_path).is_file() {
            self.thumb_loading = false;
            return;
        }
        let Some(ffmpeg_exe) = self.ffmpeg.path.clone() else {
            self.thumb_loading = false;
            return;
        };
        self.thumb_loading = true;
        let tx = self.thumb_tx.clone();
        let generation = self.thumb_generation;
        let path = self.ref_path.clone();
        let ffprobe_exe = self.ffprobe.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let duration = crate::probe::media_duration(&path, ffprobe_exe.as_deref());
            let image = crate::preview::extract_thumbnail(&ffmpeg_exe, &path, duration);
            let _ = tx.send(ThumbMsg { generation, image });
            ctx.request_repaint();
        });
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

    /// Queue picked files, silently skipping ones already present.
    /// Media probing runs on a worker thread; rows show "Probing…"
    /// until their results arrive, so drops never freeze the window.
    fn add_queue_files(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut seen: HashSet<String> = self.rows.iter().map(|r| norm_key(&r.path)).collect();
        let mut fresh: Vec<(String, String)> = Vec::new();
        for p in paths {
            let s = p.to_string_lossy().into_owned();
            let key = norm_key(&s);
            if !seen.insert(key.clone()) {
                continue; // guard rail: same file already queued
            }
            self.rows.push(QueueRow {
                path: s.clone(),
                display: String::new(),
                include: true,
                selected: false,
                media: "Probing…".to_owned(),
                media_tip: "Probing…".to_owned(),
                info: None,
                psnr: crate::metrics::MetricCell::Idle,
                ssim: crate::metrics::MetricCell::Idle,
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
    fn drain_metric_results(&mut self) {
        while let Ok(msg) = self.metric_rx.try_recv() {
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
                    if let Some(row) = self.rows.iter_mut().find(|r| norm_key(&r.path) == key)
                        && let crate::metrics::MetricCell::Running { frame: cur } =
                            row.cell_mut(kind)
                        && frame > *cur
                    {
                        *cur = frame;
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
                } => {
                    if generation != self.run_generation {
                        log::debug!(target: "rfmetrics::app", "discarded stale {} result", kind.name());
                        continue;
                    }
                    self.pending = self.pending.saturating_sub(1);
                    if let Some(row) = self.rows.iter_mut().find(|r| norm_key(&r.path) == key) {
                        *row.cell_mut(kind) = match error {
                            Some(msg) => crate::metrics::MetricCell::Error { msg },
                            None => crate::metrics::MetricCell::Done {
                                avg: avg.unwrap_or_else(|| crate::metrics::mean(&values)),
                                values,
                                exec_s,
                                skip,
                                clip_dur,
                            },
                        };
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
                            for cell in [&mut row.psnr, &mut row.ssim] {
                                if matches!(cell, crate::metrics::MetricCell::Running { .. }) {
                                    *cell = crate::metrics::MetricCell::Idle;
                                }
                            }
                        }
                    }
                    self.pending = 0;
                    self.measuring = false;
                }
            }
        }
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

    /// Start a run over included rows on one worker thread: each checked
    /// metric runs sequentially in Python `METRICS` order (Python
    /// `start`/`_worker` parity). Pre-flight failures land in the cells
    /// as errors, mirroring Python's `"bad time"` / `"probe failed"` text.
    fn start_run(&mut self, now: f64) {
        if self.measuring {
            return;
        }
        let kinds: Vec<MetricKind> = [MetricKind::Psnr, MetricKind::Ssim]
            .into_iter()
            .filter(|k| match k {
                MetricKind::Psnr => self.m_psnr,
                MetricKind::Ssim => self.m_ssim,
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
        // Per metric: rows already holding a valid value sit the rerun out —
        // but only when the trim settings still match: a value computed
        // under a different skip/clip is stale and must recompute.
        // Pre-flight error cells above touch `targets` (settings
        // uncomparable there); everything below touches `fresh` only.
        let mut work: Vec<(MetricKind, Vec<usize>, Vec<String>)> = Vec::new();
        for &kind in &kinds {
            let mut skipped = Vec::new();
            let mut fresh = Vec::new();
            for &i in &targets {
                if let crate::metrics::MetricCell::Done {
                    skip: s,
                    clip_dur: c,
                    ..
                } = self.rows[i].cell(kind)
                    && *s == skip
                    && *c == clip_dur
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
            for (kind, _, skipped) in &work {
                self.toast(
                    now,
                    format!(
                        "Skipped {} with existing {} (Reset to recompute)",
                        skipped.len(),
                        kind.name(),
                    ),
                    ToastKind::Info,
                );
            }
            return;
        }
        let Some(ffmpeg_exe) = self.ffmpeg.path.clone() else {
            for (kind, fresh, _) in &work {
                for &i in fresh {
                    *self.rows[i].cell_mut(*kind) = crate::metrics::MetricCell::Error {
                        msg: "ffmpeg not found".to_owned(),
                    };
                }
            }
            self.toast(now, "ffmpeg not found".to_owned(), ToastKind::Error);
            return;
        };
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
            for &i in fresh {
                if let Some(info) = self.rows[i].info.clone() {
                    jobs.push((
                        *kind,
                        norm_key(&self.rows[i].path),
                        self.rows[i].path.clone(),
                        info,
                    ));
                    *self.rows[i].cell_mut(*kind) =
                        crate::metrics::MetricCell::Running { frame: 0 };
                }
            }
        }
        if jobs.is_empty() {
            self.toast(
                now,
                "Files are still probing — try again in a moment".to_owned(),
                ToastKind::Info,
            );
            return;
        }
        self.run_generation = self.run_generation.wrapping_add(1);
        self.pending = jobs.len();
        self.measuring = true;
        self.abort.store(false, std::sync::atomic::Ordering::SeqCst);
        let tx = self.metric_tx.clone();
        let generation = self.run_generation;
        let ref_path = self.ref_path.clone();
        let abort = Arc::clone(&self.abort);
        let child_slot = Arc::clone(&self.current_child);
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            for (kind, key, dist_path, dist_info) in jobs {
                if abort.load(Ordering::SeqCst) {
                    break;
                }
                let txp = tx.clone();
                let keyp = key.clone();
                let job = crate::metrics::ffmpeg::RunInputs {
                    kind,
                    exe: &ffmpeg_exe,
                    ref_path: &ref_path,
                    dist_path: &dist_path,
                    ref_info: &ref_info,
                    dist_info: &dist_info,
                    skip,
                    clip_dur,
                    abort: &abort,
                    child_slot: &child_slot,
                };
                let out = crate::metrics::ffmpeg::run_metric(&job, &|f| {
                    let _ = txp.send(MetricMsg::Progress {
                        generation,
                        kind,
                        key: keyp.clone(),
                        frame: f,
                    });
                });
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
                });
            }
            let _ = tx.send(MetricMsg::Finished {
                generation,
                aborted: abort.load(Ordering::SeqCst),
            });
        });
        for (kind, _, skipped) in &work {
            if skipped.is_empty() {
                continue;
            }
            let mut list = skipped.join(", ");
            if list.chars().count() > 80 {
                list = format!("{}…", list.chars().take(79).collect::<String>());
            }
            self.toast(
                now,
                format!(
                    "Skipped {} with existing {}: {list}",
                    skipped.len(),
                    kind.name(),
                ),
                ToastKind::Info,
            );
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
            row.psnr = crate::metrics::MetricCell::Idle;
            row.ssim = crate::metrics::MetricCell::Idle;
        }
        log::info!(target: "rfmetrics::app", "metric results cleared");
    }
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

/// Per-row (stats, ranks) for one metric column; ranks stay Plain unless
/// 2+ rows scored (colors need a comparison).
fn rank_details(
    rows: &[QueueRow],
    pick: impl Fn(&QueueRow) -> Option<crate::metrics::DoneStats>,
) -> Vec<Option<(crate::metrics::DoneStats, [crate::metrics::StatRank; 10])>> {
    let scored: Vec<(usize, crate::metrics::DoneStats)> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| pick(r).map(|s| (i, s)))
        .collect();
    let comparable = scored.len() >= 2;
    let mut stat_lo = [f64::INFINITY; 10];
    let mut stat_hi = [f64::NEG_INFINITY; 10];
    for (_, s) in &scored {
        for (k, (_, v, _)) in s.comparable().iter().enumerate() {
            stat_lo[k] = stat_lo[k].min(*v);
            stat_hi[k] = stat_hi[k].max(*v);
        }
    }
    let mut detail = vec![None; rows.len()];
    for (i, s) in &scored {
        let comp = s.comparable();
        let mut ranks = [crate::metrics::StatRank::Plain; 10];
        if comparable {
            for k in 0..10 {
                let (_, v, lower_better) = comp[k];
                ranks[k] = if lower_better {
                    crate::metrics::rank_low(v, stat_lo[k], stat_hi[k])
                } else {
                    crate::metrics::rank(v, stat_lo[k], stat_hi[k])
                };
            }
        }
        detail[*i] = Some((s.clone(), ranks));
    }
    detail
}

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
        self.refresh_ref_info();
        let ctx = ui.ctx().clone();
        self.refresh_thumbnail(&ctx);
        self.drain_metric_results();
        // Live `Frame: N` progress while the metric worker runs.
        if self.measuring {
            ui.ctx().request_repaint();
        }

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
                                            .add_filter(
                                                "Video files",
                                                &[
                                                    "mp4", "mkv", "mov", "avi", "webm", "m2ts",
                                                    "ts", "m4v",
                                                ],
                                            )
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
                let _ = ui.add_sized([90.0, 24.0], egui::Button::new("Plot"));
                ui.label(&self.ffmpeg.short)
                    .on_hover_text(&self.ffmpeg.detail);
                ui.add(egui::Label::new("|").selectable(false));
                ui.label(&self.ffvship.short)
                    .on_hover_text(&self.ffvship.detail);
            });
        });

        // ---- VMAF options (just above bottom bar) ----
        egui::Panel::bottom("vmaf").show(ui, |ui| {
            ui.add(egui::Label::new("VMAF options").selectable(false));
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Model").selectable(false));
                    let _ = egui::ComboBox::from_id_salt("vmaf_model")
                        .width(220.0)
                        .selected_text(&self.vmaf_model)
                        .show_ui(ui, |ui| {
                            let _ = ui.selectable_value(
                                &mut self.vmaf_model,
                                "vmaf_v0.6.1.json".to_owned(),
                                "vmaf_v0.6.1.json",
                            );
                        });
                    let _ = ui.add(egui::Checkbox::new(&mut self.vmaf_phone, "Phone"));
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
                                let _ =
                                    ui.selectable_value(&mut self.vmaf_subsample, v.to_owned(), v);
                            }
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
                        .add_filter(
                            "Video files",
                            &["mp4", "mkv", "mov", "avi", "webm", "m2ts", "ts", "m4v"],
                        )
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
                    // Screenshot green/red rules: per-stat column extremes
                    // across scored rows rank every metric cell + tooltip chip.
                    let psnr_detail = rank_details(&self.rows, |r| r.psnr.done_stats());
                    let ssim_detail = rank_details(&self.rows, |r| r.ssim.done_stats());
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
                            for (flag, name) in [
                                (&mut self.m_psnr, "PSNR"),
                                (&mut self.m_ssim, "SSIM"),
                                (&mut self.m_vmaf, "VMAF"),
                                (&mut self.m_xpsnr, "XPSNR"),
                                (&mut self.m_ssim2, "SSIM2"),
                                (&mut self.m_but, "BUTTER"),
                                (&mut self.m_cvvdp, "CVVDP"),
                            ] {
                                header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                                header.col(|ui| {
                                    ui.add_enabled(!run_locked, egui::Checkbox::new(flag, name));
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
                                let mut label_clicked = false;
                                let mut bg_clicked = false;
                                row.col(|ui| {
                                    ui.checkbox(&mut self.rows[i].include, "");
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        let path = self.rows[i].path.clone();
                                        if let Err(e) = open::that(&path) {
                                            log::error!(target: "rfmetrics::app", "open \"{path}\" failed: {e}");
                                            self.toast = Some(Toast {
                                                text: format!("Could not open file: {e}"),
                                                until: now + TOAST_SECS,
                                                kind: ToastKind::Error,
                                            });
                                        }
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                let (_, r) = row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let (display, path) = {
                                        let r = &self.rows[i];
                                        (r.display.clone(), r.path.clone())
                                    };
                                    // Plain selectable text: no button hover
                                    // outline; drag-select/copy still works and
                                    // the full path shows as tooltip (Python parity).
                                    ui.add(egui::Label::new(&display).selectable(true))
                                        .on_hover_text(&path);
                                });
                                if r.clicked() {
                                    label_clicked = true;
                                }
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                let (_, r) = row.col(|ui| {
                                    let (media, tip) = {
                                        let r = &self.rows[i];
                                        (r.media.clone(), r.media_tip.clone())
                                    };
                                    ui.label(&media).on_hover_text(&tip);
                                });
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                // Filter-metric columns: live state text on a rank
                                // fill (best green, worst red, tie dim yellow),
                                // per-stat chip grid tooltip for Done cells.
                                for (kind, title, detail) in [
                                    (MetricKind::Psnr, "PSNR", &psnr_detail),
                                    (MetricKind::Ssim, "SSIM", &ssim_detail),
                                ] {
                                    let (_, r) = row.col(|ui| {
                                        vline(ui, egui::Color32::from_gray(0x38))
                                    });
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                    let (_, r) = row.col(|ui| {
                                        let (text, tip, cell_detail) = {
                                            let cell = &self.rows[i].cell(kind);
                                            (
                                                cell.cell_text(),
                                                cell.tooltip(title),
                                                detail[i].clone(),
                                            )
                                        };
                                        let mut cell_frame = egui::Frame::NONE;
                                        if let Some(fill) = cell_detail
                                            .as_ref()
                                            .and_then(|(_, r)| rank_fill(r[0]))
                                        {
                                            cell_frame = cell_frame.fill(fill);
                                        }
                                        cell_frame.show(ui, |ui| {
                                            ui.set_width(ui.available_width());
                                            ui.centered_and_justified(|ui| {
                                                let resp = ui.label(&text);
                                                match cell_detail {
                                                    Some((stats, ranks)) => {
                                                        resp.on_hover_ui(|ui| {
                                                            metric_stat_tooltip(
                                                                ui, title, &stats, &ranks,
                                                            );
                                                        });
                                                    }
                                                    None => {
                                                        resp.on_hover_text(&tip);
                                                    }
                                                }
                                            });
                                        });
                                    });
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                }
                                for _ in 0..5 {
                                    let (_, r) =
                                        row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                    let (_, r) = row.col(|ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.label("N/A");
                                        });
                                    });
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                }
                                if label_clicked || bg_clicked {
                                    self.rows[i].selected = !self.rows[i].selected;
                                }
                                if row.response().hovered() {
                                    self.hovered_now = Some(i);
                                }
                            });
                        });
                });
            });
            self.table_rect = Some(table_resp.response.rect);
            // Roll the delayed-hover timer forward; repaint while the
            // delay is pending so the outline appears without moving.
            if self.hovered_now != self.hover_row {
                self.hover_row = self.hovered_now;
                self.hover_since = self.hover_row.map(|_| now);
            }
            let hover_pending = matches!(
                (self.hover_row, self.hovered_now, self.hover_since),
                (Some(a), Some(b), Some(t)) if a == b && now - t < ROW_HOVER_DELAY
            );
            if hover_pending {
                ui.ctx().request_repaint();
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
        if let Some(toast) = self.toast.clone() {
            if now < toast.until {
                ui.ctx().request_repaint();
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
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DropAction, ProbeMsg, QueueRow, RFMetricsApp, display_names, norm_key, route_drop,
    };

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
            display: "a.mp4".to_owned(),
            include: true,
            selected: false,
            media: "Probing…".to_owned(),
            media_tip: "Probing…".to_owned(),
            info: None,
            psnr: crate::metrics::MetricCell::Idle,
            ssim: crate::metrics::MetricCell::Idle,
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
            display: "a.mp4".to_owned(),
            include,
            selected: false,
            media: "h264, 1080p".to_owned(),
            media_tip: "tip".to_owned(),
            info: None,
            psnr: crate::metrics::MetricCell::Idle,
            ssim: crate::metrics::MetricCell::Idle,
        }
    }

    #[test]
    fn start_psnr_gated_on_checkbox() {
        let mut app = RFMetricsApp::default();
        app.rows.push(psnr_test_row("C:/vids/a.mp4", true));
        app.m_psnr = false;
        app.start_run(0.0);
        assert!(!app.measuring);
        assert!(matches!(app.rows[0].psnr, crate::metrics::MetricCell::Idle));
        assert!(app.toast.is_some());
    }

    #[test]
    fn start_psnr_needs_included_rows() {
        let mut app = RFMetricsApp {
            m_psnr: true,
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
        app.rows[0].psnr = MetricCell::Running { frame: 10 };
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
            MetricCell::Running { frame: 25 }
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
        };
        app.rows[1].psnr = MetricCell::Running { frame: 12 };
        app.rows[1].ssim = MetricCell::Running { frame: 3 };
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
        };
        app.start_run(0.0);
        std::fs::remove_file(&p).ok();
        assert!(!app.measuring);
        assert!(app.pending == 0);
        let toast = app.toast.as_ref().expect("skip toast shown");
        assert!(toast.text.contains("Skipped 1 with existing PSNR"));
        assert!(matches!(&app.rows[0].psnr, MetricCell::Done { .. }));
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
        assert!(matches!(app.rows[1].psnr, MetricCell::Running { frame: 0 }));
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

    /// SSIM-only run: only the SSIM cell enters the run, PSNR stays Idle.
    #[test]
    fn start_run_ssim_only_runs_ssim_cell() {
        use crate::metrics::MetricCell;
        use crate::probe::MediaInfo;
        let p = std::env::temp_dir().join("rfmetrics-ssim-only.tmp");
        std::fs::write(&p, b"x").unwrap();
        let mut app = RFMetricsApp {
            m_ssim: true,
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
        assert!(matches!(app.rows[0].ssim, MetricCell::Running { frame: 0 }));
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
        assert!(matches!(app.rows[0].ssim, MetricCell::Running { frame: 0 }));
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
}
