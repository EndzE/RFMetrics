//! Run worker domain: message types for probe/thumb/metric channels.
//!
//! Extracted from `app.rs` (High 1 split). Worker threads only `send`;
//! the UI thread drains via `try_recv`. Method bodies move here in a
//! follow-up step once `RFMetricsApp` fields turn `pub(crate)`.

use crate::metrics::ffmpeg::MetricKind;
use crate::metrics::ffmpeg::ScaleMethod;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Results sent back from background probe threads. The UI thread never
/// blocks on ffprobe; it drains these each frame via `try_recv`.
#[derive(Debug)]
pub(crate) enum ProbeMsg {
    Reference {
        generation: u64,
        text: String,
        info: Option<crate::probe::MediaInfo>,
        /// The ffprobe call timed out (the drain surfaces it as a toast).
        timed_out: bool,
    },
    RowMedia {
        key: String,
        /// Row token at spawn; applies only if the row still holds it
        /// (a remove/re-add orphan carries the old one).
        probe_gen: u64,
        media: String,
        tip: String,
        info: Option<crate::probe::MediaInfo>,
        /// The ffprobe call timed out (the drain surfaces it as a toast).
        timed_out: bool,
    },
}

/// Thumbnail result from the dedicated ffmpeg worker (separate channel so
/// slow frame extracts never block fast ffprobe text results).
pub(crate) struct ThumbMsg {
    pub(crate) generation: u64,
    pub(crate) image: Option<egui::ColorImage>,
}

/// Startup binary probe result: the version probes (`-version` × 2 +
/// `-filters`) run on a worker thread so a wedged exe can't freeze the
/// window; exactly one tuple is sent.
pub(crate) type BinProbe = (
    crate::binaries::BinaryInfo,
    crate::binaries::BinaryInfo,
    Option<std::path::PathBuf>,
);

/// Progress + results from the single sequential metric worker (Python
/// `_worker` parity: one thread, checked metrics in order, never on the UI
/// thread). `generation` drops late messages after a Reset starts a new run.
#[derive(Debug)]
pub(crate) enum MetricMsg {
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
        /// Input framerate mode the run used; stamped like `scaler` so a
        /// mode change recomputes every ffmpeg-backed column (FFVship
        /// has no `-r` stage and ignores it at compare time).
        fps_mode: crate::metrics::ffmpeg::InputFpsMode,
        /// Reference pixel-format target the run used; stamped like
        /// `scaler` so a target change recomputes every ffmpeg-backed
        /// column (FFVship has no `format=` stage and ignores it at
        /// compare time).
        ref_pixfmt: crate::metrics::ffmpeg::RefPixFmt,
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

/// `DateTime` columns (original `DateTime` parity).
pub(crate) fn wall_now_string() -> String {
    jiff::Timestamp::now()
        .to_zoned(jiff::tz::TimeZone::system())
        .strftime("%Y-%m-%d %H:%M:%S")
        .to_string()
}
/// Kinds sharing an identical skip set merge into one toast line
/// ("Skipped 2 with existing PSNR, SSIM: a, b") so filenames print once
/// instead of repeating per metric. First-seen kind order is kept.
pub(crate) fn skip_groups(
    work: &[(MetricKind, Vec<usize>, Vec<String>)],
) -> Vec<(Vec<&str>, &Vec<String>)> {
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

impl crate::app::RFMetricsApp {
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

    /// Short display name for timeout toasts: filename when available,
    /// full path otherwise (never empty — falls back to a placeholder).
    pub(crate) fn timeout_display(path: &str) -> String {
        std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(unknown file)".to_owned())
    }

    /// Parse a trim box; empty means no trim. `None` = invalid ("bad time").
    pub(crate) fn trim_opt(raw: &str) -> Option<Option<f64>> {
        if raw.trim().is_empty() {
            Some(None)
        } else {
            crate::metrics::parse_time_spec(raw).map(Some)
        }
    }

    /// Validated VMAF settings snapshot (Python `vmaf_cfg`): subsample
    /// parses to u32 with max(1, …), pooling maps the UI strings to the
    /// enum. Single source for `start_run` and the stale-cell badge so
    /// the two can never disagree on what "current settings" means.
    pub(crate) fn current_vmaf_cfg(&self) -> crate::metrics::vmaf::VmafCfg {
        crate::metrics::vmaf::VmafCfg {
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
        }
    }

    pub(crate) fn abort_worker(&self) {
        use std::sync::atomic::Ordering;
        self.abort.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = self.current_child.lock()
            && let Some(mut child) = slot.take()
        {
            let _ = child.kill();
            std::thread::spawn(move || {
                match wait_timeout::ChildExt::wait_timeout(&mut child, crate::cmd::REAP_TIMEOUT) {
                    Ok(Some(_)) => {}
                    _ => {
                        log::error!(target: "rfmetrics::app", "stop reap timed out after {:?} — child still alive", crate::cmd::REAP_TIMEOUT);
                        let _ = child.wait();
                    }
                }
            });
        }
    }

    /// Stop button: abort all runners, keep finished results on screen.
    /// Still-Running cells settle to Idle when the worker's `Finished`
    /// lands in `drain_metric_results`.
    pub(crate) fn stop_psnr(&mut self) {
        if !self.measuring {
            return;
        }
        self.abort_worker();
        log::info!(target: "rfmetrics::app", "run aborted by user");
    }

    /// Clear all metric cells. Aborts a running worker first so Reset never
    /// leaves an orphaned ffmpeg burning CPU in the background.
    pub(crate) fn reset_psnr(&mut self) {
        self.abort_worker();
        self.run_generation = self.run_generation.wrapping_add(1);
        self.pending = 0;
        self.measuring = false;
        self.live_kind = None;
        self.live_key = None;
        for row in &mut self.rows {
            for kind in MetricKind::ALL {
                *row.cell_mut(kind) = crate::metrics::MetricCell::Idle;
                *row.cached_mut(kind) = crate::app_queue::CachedStats::default();
            }
        }
        log::info!(target: "rfmetrics::app", "metric results cleared");
    }

    /// Clear one metric column (header Reset). No abort/generation bump:
    /// the menu item is disabled while running, so nothing is in flight.
    pub(crate) fn reset_metric(&mut self, kind: MetricKind) {
        for row in &mut self.rows {
            *row.cell_mut(kind) = crate::metrics::MetricCell::Idle;
            *row.cached_mut(kind) = crate::app_queue::CachedStats::default();
        }
        self.refresh_ranks(kind);
        if self.live_kind == Some(kind) {
            self.live_kind = None;
            self.live_key = None;
        }
        log::info!(target: "rfmetrics::app", "{} results cleared", kind.name());
    }

    /// Re-freeze every `Done` cell's Avg text at the current Precision.
    /// Runs only on discrete Precision changes (Options combobox), so the
    /// per-frame table loop keeps borrowing without formatting.
    pub(crate) fn refreeze_cell_texts(&mut self) {
        let prec = self.cell_precision as usize;
        for row in &mut self.rows {
            for kind in MetricKind::ALL {
                if matches!(row.cell(kind), crate::metrics::MetricCell::Done { .. }) {
                    row.cached_mut(kind).text = row.cell(kind).cell_text_prec(prec);
                }
            }
        }
    }

    pub(crate) fn refresh_queue_names(&mut self) {
        let paths: Vec<String> = self.rows.iter().map(|r| r.path.clone()).collect();
        for (row, name) in self
            .rows
            .iter_mut()
            .zip(crate::app_queue::display_names(&paths))
        {
            row.display = name;
        }
        // Row indices may have shifted; drop stale hover state.
        self.hover_row = None;
        self.hover_since = None;
        self.include_anchor = None;
        self.selected_anchor = None;
    }

    /// Re-probe everything (Options "Refresh Files Media Info"): the
    /// reference text + thumbnail and every queue row's media text + raw
    /// info. Same worker channels as the initial probes, so the window
    /// never blocks; results (not reruns of finished metrics) update.
    pub(crate) fn refresh_media_info(&mut self) {
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
        let paths: Vec<(String, String, u64)> = self
            .rows
            .iter()
            .map(|r| (r.key.clone(), r.path.clone(), r.probe_gen))
            .collect();
        std::thread::spawn(move || {
            for (key, s, probe_gen) in paths {
                let (media, tip, info, timed_out) =
                    crate::probe::probe_table_text(&s, exe.as_deref());
                let _ = tx.send(ProbeMsg::RowMedia {
                    key,
                    probe_gen,
                    media,
                    tip,
                    info,
                    timed_out,
                });
            }
        });
    }

    /// Queue picked files, silently skipping ones already present.
    /// Media probing runs on a worker thread; rows show "Probing…"
    /// until their results arrive, so drops never freeze the window.
    pub(crate) fn add_queue_files(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut seen: HashSet<String> = self.rows.iter().map(|r| r.key.clone()).collect();
        let mut fresh: Vec<(String, String, u64)> = Vec::new();
        for p in paths {
            let s = p.to_string_lossy().into_owned();
            let key = crate::app_queue::norm_key(&s);
            if !seen.insert(key.clone()) {
                continue; // guard rail: same file already queued
            }
            let color_idx = self.next_color_idx;
            self.next_color_idx += 1;
            let probe_gen = self.next_probe_seq;
            self.next_probe_seq = self.next_probe_seq.wrapping_add(1);
            self.rows.push(crate::app_queue::QueueRow {
                path: s.clone(),
                key: key.clone(),
                display: String::new(),
                include: true,
                color_idx,
                probe_gen,
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
                psnr_cache: crate::app_queue::CachedStats::default(),
                ssim_cache: crate::app_queue::CachedStats::default(),
                vmaf_cache: crate::app_queue::CachedStats::default(),
                xpsnr_cache: crate::app_queue::CachedStats::default(),
                ssim2_cache: crate::app_queue::CachedStats::default(),
                butter_cache: crate::app_queue::CachedStats::default(),
                cvvdp_cache: crate::app_queue::CachedStats::default(),
            });
            fresh.push((key, s, probe_gen));
        }
        self.refresh_queue_names();
        if fresh.is_empty() {
            return;
        }
        let tx = self.probe_tx.clone();
        let exe = self.ffprobe.clone();
        std::thread::spawn(move || {
            for (key, s, probe_gen) in fresh {
                let (media, tip, info, timed_out) =
                    crate::probe::probe_table_text(&s, exe.as_deref());
                let _ = tx.send(ProbeMsg::RowMedia {
                    key,
                    probe_gen,
                    media,
                    tip,
                    info,
                    timed_out,
                });
            }
        });
    }

    /// Apply any probe results that arrived since the last frame. Stale
    /// reference results (typed-through while a worker was running) are
    /// dropped via the generation check.
    /// Drains the probe channel; returns whether any message arrived (even
    /// a stale one — callers use it to decide on a repaint, and one extra
    /// frame on a rare stale message is harmless).
    pub(crate) fn drain_probe_results(&mut self) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.probe_rx.try_recv() {
            activity = true;
            match msg {
                ProbeMsg::Reference {
                    generation,
                    text,
                    info,
                    timed_out,
                } => {
                    if generation == self.ref_generation {
                        self.ref_info = text;
                        self.ref_info_data = info;
                        // No newer spawn happened since (same generation),
                        // so `last_spawned_ref` is the path this probed.
                        self.ref_info_path = self.last_spawned_ref.clone();
                        if timed_out {
                            self.probe_timeout_note =
                                Some(Self::timeout_display(&self.last_spawned_ref));
                        }
                    } else {
                        log::debug!(target: "rfmetrics::app", "discarded stale ref probe (gen {generation})");
                    }
                }
                ProbeMsg::RowMedia {
                    key,
                    probe_gen,
                    media,
                    tip,
                    info,
                    timed_out,
                } => {
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key) {
                        // Remove/re-add orphans carry the old token: only the
                        // row this probe spawned for may consume it.
                        if row.probe_gen == probe_gen {
                            row.media = media;
                            row.media_tip = tip;
                            row.info = info;
                            if timed_out {
                                self.probe_timeout_note = Some(row.display.clone());
                            }
                        } else {
                            log::debug!(target: "rfmetrics::app", "discarded stale row probe for {key}");
                        }
                    }
                }
            }
        }
        activity
    }

    /// Startup binary probe drain: swaps the `Probing…` placeholders for
    /// the real version results, then re-arms anything that resolved
    /// while the binaries were unknown. Generations bump so stale
    /// no-binary results ("ffprobe not found") drop instead of winning
    /// the race against the re-probes.
    pub(crate) fn drain_bin_results(&mut self) -> bool {
        let Ok((ffmpeg, ffvship, ffprobe)) = self.bin_rx.try_recv() else {
            return false;
        };
        // Only one tuple is ever sent; drop duplicates if any.
        while self.bin_rx.try_recv().is_ok() {}
        self.ffmpeg = ffmpeg;
        self.ffvship = ffvship;
        self.ffprobe = ffprobe;
        self.bins_probing = false;
        self.untick_unsupported_metrics();
        // Reference + thumbnail re-probe through the normal path next frame.
        self.ref_generation = self.ref_generation.wrapping_add(1);
        self.last_spawned_ref.clear();
        self.thumb_generation = self.thumb_generation.wrapping_add(1);
        self.last_thumb_path.clear();
        // Queue rows that settled (or are still settling) without a binary
        // re-probe with fresh tokens; stale workers carry the old token.
        let mut stale: Vec<(String, String, u64)> = Vec::new();
        for row in &mut self.rows {
            if row.info.is_none() {
                let token = self.next_probe_seq;
                self.next_probe_seq = self.next_probe_seq.wrapping_add(1);
                row.probe_gen = token;
                row.media = "Probing…".to_owned();
                row.media_tip = "Probing…".to_owned();
                stale.push((row.key.clone(), row.path.clone(), token));
            }
        }
        if !stale.is_empty() {
            let tx = self.probe_tx.clone();
            let exe = self.ffprobe.clone();
            std::thread::spawn(move || {
                for (key, s, probe_gen) in stale {
                    let (media, tip, info, timed_out) =
                        crate::probe::probe_table_text(&s, exe.as_deref());
                    let _ = tx.send(ProbeMsg::RowMedia {
                        key,
                        probe_gen,
                        media,
                        tip,
                        info,
                        timed_out,
                    });
                }
            });
        }
        true
    }

    /// Re-probe only when the path actually changed, and only off the UI
    /// thread: cheap cases (empty/missing/no ffprobe) resolve inline, an
    /// existing file spawns a worker and shows "Probing…" meanwhile.
    /// Returns the probe drain flag (spawns stem from input frames, which
    /// repaint on their own).
    pub(crate) fn refresh_ref_info(&mut self) -> bool {
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
            let (text, info, timed_out) = crate::probe::reference_media_text(&path, exe.as_deref());
            let _ = tx.send(ProbeMsg::Reference {
                generation,
                text,
                info,
                timed_out,
            });
        });
        activity
    }

    /// Apply arrived thumbnails; stale generations (typed-through) are dropped.
    /// Returns whether any message arrived (see `drain_probe_results`).
    pub(crate) fn drain_thumbs(&mut self, ctx: &egui::Context) -> bool {
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
    pub(crate) fn refresh_thumbnail(&mut self, ctx: &egui::Context) -> bool {
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
        let duration = crate::app_queue::thumb_duration(
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

    /// Apply metric worker results; stale generations (post-Reset) drop.
    /// Progress keeps the max frame per row (dual stdout/stderr feeds).
    /// Returns whether any message arrived (see `drain_probe_results`).
    pub(crate) fn drain_metric_results(&mut self) -> bool {
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
                    // tab follows it while measuring, and only its cell
                    // animates the sweep (the rest wait statically).
                    self.live_kind = Some(kind);
                    self.live_key = Some(key.clone());
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
                    self.live_key = Some(key.clone());
                    if let Some(row) = self.rows.iter_mut().find(|r| r.key == key)
                        && let crate::metrics::MetricCell::Running { values, .. } =
                            row.cell_mut(kind)
                    {
                        // Live curve feeds straight from `values` (plot
                        // decimates at draw time, x = 1-based frame).
                        values.extend_from_slice(&new_values);
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
                    fps_mode,
                    ref_pixfmt,
                } => {
                    if generation != self.run_generation {
                        log::debug!(target: "rfmetrics::app", "discarded stale {} result", kind.name());
                        continue;
                    }
                    // The finished job stops being live; the next job
                    // takes over on its first Progress/Series (until
                    // then no cell sweeps — the gap shows static text).
                    // `live_kind` stays for plot tab-follow.
                    if self.live_kind == Some(kind)
                        && self.live_key.as_deref() == Some(key.as_str())
                    {
                        self.live_key = None;
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
                                fps_mode,
                                ref_pixfmt,
                            },
                        };
                        // Cache the stats once (clone+sort lives here, not
                        // per frame); ranks refresh below for this metric.
                        let stats = row.cell(kind).done_stats();
                        // Rendered text frozen once per result (the table loop
                        // borrows it instead of formatting per frame).
                        let text = row.cell(kind).cell_text_prec(self.cell_precision as usize);
                        row.cached_mut(kind).stats = stats;
                        row.cached_mut(kind).text = text;
                        row.cached_mut(kind).finished =
                            matches!(row.cell(kind), crate::metrics::MetricCell::Done { .. })
                                .then(wall_now_string);
                        scored_changed = true;
                    }
                    // `Finished` alone clears `measuring` below: the worker
                    // sends it (then `CsvReport`) after the last `Done`, so
                    // clearing here would reopen Start a frame early and
                    // orphan those terminal messages as stale.
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
                    // Results auto-save (option): exported with a timestamp
                    // at the next UI frame, stopped runs included — their
                    // finished cells still count.
                    if self.results_autosave {
                        self.results_autosave_pending = true;
                    }
                    self.pending = 0;
                    self.measuring = false;
                    self.live_key = None;
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

    /// Start a run over included rows on one worker thread: each checked
    /// metric runs sequentially in Python `METRICS` order (Python
    /// `start`/`_worker` parity). Pre-flight failures land in the cells
    /// as errors, mirroring Python's `"bad time"` / `"probe failed"` text.
    pub(crate) fn start_run(&mut self, now: f64) {
        if self.measuring {
            return;
        }
        if self.bins_probing {
            self.toast(
                now,
                "Binaries still probing — try again in a moment".to_owned(),
                crate::app::ToastKind::Info,
            );
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
                crate::app::ToastKind::Info,
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
                crate::app::ToastKind::Info,
            );
            return;
        }
        if self.ref_path.trim().is_empty() || !Path::new(&self.ref_path).is_file() {
            for &i in &targets {
                for &kind in &kinds {
                    *self.rows[i].cell_mut(kind) = crate::metrics::MetricCell::Error {
                        msg: "no ref".to_owned(),
                    };
                    self.rows[i].clear_cached(kind);
                }
            }
            for &kind in &kinds {
                self.refresh_ranks(kind);
            }
            self.toast(
                now,
                "Set a reference file first".to_owned(),
                crate::app::ToastKind::Error,
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
                    self.rows[i].clear_cached(kind);
                }
            }
            for &kind in &kinds {
                self.refresh_ranks(kind);
            }
            self.toast(
                now,
                "Skip/Duration is not a valid time".to_owned(),
                crate::app::ToastKind::Error,
            );
            return;
        };
        // Validated VMAF snapshot: snapshotted before the partition so
        // VMAF `Done` stamps compare against the settings this run uses.
        let vmaf_cfg = self.current_vmaf_cfg();
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
        let fps_mode = self.fps_mode;
        let ref_pixfmt = self.ref_pixfmt;
        let mut work: Vec<(MetricKind, Vec<usize>, Vec<String>)> = Vec::new();
        for &kind in &kinds {
            let mut skipped = Vec::new();
            let mut fresh = Vec::new();
            for &i in &targets {
                if !crate::app_queue::done_is_stale(
                    kind,
                    self.rows[i].cell(kind),
                    skip,
                    clip_dur,
                    &vmaf_cfg,
                    scaler,
                    fps_mode,
                    ref_pixfmt,
                ) && matches!(
                    self.rows[i].cell(kind),
                    crate::metrics::MetricCell::Done { .. }
                ) {
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
            self.toast(
                now,
                format!("{msg} (Reset to recompute)"),
                crate::app::ToastKind::Info,
            );
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
                    // Ranks refresh with the rest below, after `Running`
                    // cells are marked (their caches clear there too).
                    self.rows[i].clear_cached(*kind);
                }
                if !missing.contains(&label) {
                    missing.push(label);
                }
            }
        }
        if !missing.is_empty() {
            self.toast(now, missing.join(" + "), crate::app::ToastKind::Error);
            // Downstream early-returns (probing ref, empty jobs) skip the
            // post-marking refresh, so settle ranks here.
            for (kind, _, _) in &work {
                self.refresh_ranks(*kind);
            }
        }
        let Some(ref_info) = self.ref_info_data.clone() else {
            self.toast(
                now,
                "Reference is still probing — try again in a moment".to_owned(),
                crate::app::ToastKind::Info,
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
                    // the refresh below can't rank a stale value.
                    self.rows[i].cached_mut(*kind).stats = None;
                    self.rows[i].cached_mut(*kind).text.clear();
                    self.rows[i].cached_mut(*kind).finished = None;
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
                    crate::app::ToastKind::Info,
                );
            }
            return;
        }
        self.run_generation = self.run_generation.wrapping_add(1);
        self.pending = jobs.len();
        self.measuring = true;
        // Fresh run: tab-follow restarts from the first live job.
        self.live_kind = None;
        self.live_key = None;
        if self.plot_at_start {
            self.show_plot = true;
        }
        // Fresh Arcs per run: a zombie from Reset keeps the old Arc
        // (still aborted) instead of observing a shared store(false).
        self.abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.current_child = Arc::new(Mutex::new(None));
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
                    fps_mode,
                    ref_pixfmt,
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
                    fps_mode,
                    ref_pixfmt,
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
            self.toast(now, parts.join("\n"), crate::app::ToastKind::Info);
        }
    }
}
