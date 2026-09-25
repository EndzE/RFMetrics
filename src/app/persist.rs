//! State-file glue for `RFMetricsApp`: the `ffmetrics-state.json`
//! snapshot / apply / dirty-check / debounced autosave plus the
//! results-CSV export.

use crate::app::{ToastKind, norm_key, wall_now_string};
use crate::metrics::ffmpeg::{MetricKind, ScaleMethod};

impl crate::app::config::Config {
    /// Everything `ffmetrics-state.json` persists, read off the live UI
    /// plus the queue's paths + include flags.
    pub(crate) fn snapshot(&self, queue: &crate::app::queue::QueueModel) -> crate::state::AppState {
        crate::state::AppState {
            ref_path: self.reference.path.clone(),
            skip: self.reference.skip.clone(),
            duration: self.reference.duration.clone(),
            files: Some(
                queue
                    .rows
                    .iter()
                    .map(|r| crate::state::FileEntry {
                        path: r.path.clone(),
                        include: r.include,
                    })
                    .collect(),
            ),
            metrics: crate::state::MetricsState {
                psnr: Some(self.metrics.psnr),
                ssim: Some(self.metrics.ssim),
                vmaf: Some(self.metrics.vmaf),
                xpsnr: Some(self.metrics.xpsnr),
                ssim2: Some(self.metrics.ssim2),
                butteraugli: Some(self.metrics.butteraugli),
                cvvdp: Some(self.metrics.cvvdp),
            },
            vmaf: crate::state::VmafState {
                model: Some(self.vmaf.model.clone()),
                phone: Some(self.vmaf.phone),
                scale: Some(self.vmaf.scale),
                pooling: Some(self.vmaf.pooling.clone()),
                subsample: Some(self.vmaf.subsample.clone()),
                threads: Some(self.vmaf.threads.clone()),
            },
            options: crate::state::OptionsState {
                scaling: Some(self.view.scale_method.label().to_owned()),
                fps_mode: Some(self.view.fps_mode.label().to_owned()),
                cell_stat: Some(self.view.cell_stat.label().to_owned()),
                cell_precision: Some(self.view.cell_precision.to_string()),
                ref_pixfmt: Some(self.reference.pixfmt.label().to_owned()),
                plot_at_start: Some(self.view.plot_at_start),
                plot_size: Some(self.view.plot_size.label().to_owned()),
                csv_export: Some(self.export.csv_export),
                csv_dir: Some(self.export.csv_dir.clone()),
                badframes_count: Some(self.export.badframes_count.clone()),
                badframes_export_dir: Some(self.export.badframes_export_dir.clone()),
                results_autosave: Some(self.export.results_autosave),
                results_path: Some(self.export.results_path.clone()),
            },
        }
    }
}

impl crate::app::RFMetricsApp {
    /// Everything `ffmetrics-state.json` persists, read off the live UI.
    pub(crate) fn snapshot(&self) -> crate::state::AppState {
        self.config.snapshot(&self.queue)
    }

    /// Issue #7: force off restored/default ticks for filters this ffmpeg
    /// build lacks (their header checkboxes are disabled, so they were
    /// never ticked live). Same for the FFVship family when the binary is
    /// absent or unusable (wrong-GPU build): its header checkboxes render
    /// disabled, so restored ticks are cleared too. Session-only
    /// capability, never persisted.
    /// Idempotent: safe to run for both the no-file and restored paths.
    pub(crate) fn untick_unsupported_metrics(&mut self) {
        // Capability unknown while the version probes are in flight:
        // clearing here would wipe restored ticks (and the VMAF-on
        // default) against the empty placeholder set. `drain_bin_results`
        // enforces this once the real results land, so wait for it.
        if self.binaries.probing {
            return;
        }
        let supported = &self.binaries.ffmpeg.supported_metrics;
        if !supported.contains(&MetricKind::Psnr) {
            self.config.metrics.psnr = false;
        }
        if !supported.contains(&MetricKind::Ssim) {
            self.config.metrics.ssim = false;
        }
        if !supported.contains(&MetricKind::Vmaf) {
            self.config.metrics.vmaf = false;
        }
        if !supported.contains(&MetricKind::Xpsnr) {
            self.config.metrics.xpsnr = false;
        }
        if !self.binaries.ffvship.usable {
            self.config.metrics.ssim2 = false;
            self.config.metrics.butteraugli = false;
            self.config.metrics.cvvdp = false;
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
        self.config.reference.path = s.ref_path;
        self.config.reference.skip = s.skip;
        self.config.reference.duration = s.duration;
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
                if let Some(row) = self.queue.rows.iter_mut().find(|r| r.key == ekey) {
                    row.include = e.include;
                }
            }
        }
        let m = s.metrics;
        if let Some(v) = m.psnr {
            self.config.metrics.psnr = v;
        }
        if let Some(v) = m.ssim {
            self.config.metrics.ssim = v;
        }
        if let Some(v) = m.vmaf {
            self.config.metrics.vmaf = v;
        }
        if let Some(v) = m.xpsnr {
            self.config.metrics.xpsnr = v;
        }
        if let Some(v) = m.ssim2 {
            self.config.metrics.ssim2 = v;
        }
        if let Some(v) = m.butteraugli {
            self.config.metrics.butteraugli = v;
        }
        if let Some(v) = m.cvvdp {
            self.config.metrics.cvvdp = v;
        }
        let v = s.vmaf;
        if let Some(model) = v.model
            && self.config.vmaf.models.contains(&model)
        {
            self.config.vmaf.model = model;
        }
        if let Some(phone) = v.phone {
            self.config.vmaf.phone = phone;
        }
        if let Some(scale) = v.scale {
            self.config.vmaf.scale = scale;
        }
        if let Some(pooling) = v.pooling
            && ["Mean", "Harmonic Mean"].contains(&pooling.as_str())
        {
            self.config.vmaf.pooling = pooling;
        }
        if let Some(subsample) = v.subsample
            && ["1", "2", "3", "5", "10", "15"].contains(&subsample.as_str())
        {
            self.config.vmaf.subsample = subsample;
        }
        if let Some(threads) = v.threads {
            self.config.vmaf.threads = threads;
        }
        if let Some(scaling) = s.options.scaling
            && let Some(m) = ScaleMethod::from_label(&scaling)
        {
            self.config.view.scale_method = m;
        }
        if let Some(fps_mode) = s.options.fps_mode
            && let Some(m) = crate::metrics::ffmpeg::InputFpsMode::from_label(&fps_mode)
        {
            self.config.view.fps_mode = m;
        }
        if let Some(cell_stat) = s.options.cell_stat
            && let Some(m) = crate::metrics::CellStat::from_label(&cell_stat)
        {
            self.config.view.cell_stat = m;
        }
        if let Some(ref_pixfmt) = s.options.ref_pixfmt
            && let Some(m) = crate::metrics::ffmpeg::RefPixFmt::from_label(&ref_pixfmt)
        {
            self.config.reference.pixfmt = m;
        }
        if let Some(cell_precision) = s.options.cell_precision
            && let Ok(p) = cell_precision.parse::<u8>()
            && p as usize <= crate::metrics::MAX_PRECISION
        {
            self.config.view.cell_precision = p;
        }
        if let Some(plot_at_start) = s.options.plot_at_start {
            self.config.view.plot_at_start = plot_at_start;
        }
        if let Some(plot_size) = s.options.plot_size
            && let Some(m) = crate::plot::PlotSize::from_label(&plot_size)
        {
            self.config.view.plot_size = m;
        }
        if let Some(csv_export) = s.options.csv_export {
            self.config.export.csv_export = csv_export;
        }
        if let Some(csv_dir) = s.options.csv_dir {
            self.config.export.csv_dir = csv_dir;
        }
        if let Some(badframes_count) = s.options.badframes_count
            && crate::metrics::badframes::COUNT_LABELS.contains(&badframes_count.as_str())
        {
            self.config.export.badframes_count = badframes_count;
        }
        if let Some(badframes_export_dir) = s.options.badframes_export_dir {
            self.config.export.badframes_export_dir = badframes_export_dir;
        }
        if let Some(results_autosave) = s.options.results_autosave {
            self.config.export.results_autosave = results_autosave;
        }
        if let Some(results_path) = s.options.results_path {
            self.config.export.results_path = results_path;
        }
        // Restored ticks obey capability (see helper docs).
        self.untick_unsupported_metrics();
    }

    /// Allocation-free dirty check mirroring `snapshot() != saved_snapshot`
    /// without building `AppState`. Covers every field `snapshot()` sets;
    /// a new persisted field must be added here too, or edits to it will
    /// silently stop saving.
    pub(crate) fn is_state_dirty(&self) -> bool {
        let s = &self.ui.saved_snapshot;
        if self.config.reference.path != s.ref_path
            || self.config.reference.skip != s.skip
            || self.config.reference.duration != s.duration
        {
            return true;
        }
        // `snapshot()` always emits `Some(vec)`; a `None` here can only mean
        // "never saved this shape", which never equals a snapshot.
        match &s.files {
            Some(files) => {
                if files.len() != self.queue.rows.len() {
                    return true;
                }
                for (live, saved) in self.queue.rows.iter().zip(files.iter()) {
                    if live.path != saved.path || live.include != saved.include {
                        return true;
                    }
                }
            }
            None => return true,
        }
        let m = &s.metrics;
        if m.psnr != Some(self.config.metrics.psnr)
            || m.ssim != Some(self.config.metrics.ssim)
            || m.vmaf != Some(self.config.metrics.vmaf)
            || m.xpsnr != Some(self.config.metrics.xpsnr)
            || m.ssim2 != Some(self.config.metrics.ssim2)
            || m.butteraugli != Some(self.config.metrics.butteraugli)
            || m.cvvdp != Some(self.config.metrics.cvvdp)
        {
            return true;
        }
        let v = &s.vmaf;
        if v.model.as_deref() != Some(self.config.vmaf.model.as_str())
            || v.phone != Some(self.config.vmaf.phone)
            || v.scale != Some(self.config.vmaf.scale)
            || v.pooling.as_deref() != Some(self.config.vmaf.pooling.as_str())
            || v.subsample.as_deref() != Some(self.config.vmaf.subsample.as_str())
            || v.threads.as_deref() != Some(self.config.vmaf.threads.as_str())
        {
            return true;
        }
        let o = &s.options;
        if o.scaling.as_deref() != Some(self.config.view.scale_method.label())
            || o.fps_mode.as_deref() != Some(self.config.view.fps_mode.label())
            || o.cell_stat.as_deref() != Some(self.config.view.cell_stat.label())
            || o.cell_precision
                .as_deref()
                .and_then(|s| s.parse::<u8>().ok())
                != Some(self.config.view.cell_precision)
            || o.ref_pixfmt.as_deref() != Some(self.config.reference.pixfmt.label())
            || o.plot_at_start != Some(self.config.view.plot_at_start)
            || o.plot_size.as_deref() != Some(self.config.view.plot_size.label())
            || o.csv_export != Some(self.config.export.csv_export)
            || o.csv_dir.as_deref() != Some(self.config.export.csv_dir.as_str())
            || o.badframes_count.as_deref() != Some(self.config.export.badframes_count.as_str())
            || o.badframes_export_dir.as_deref()
                != Some(self.config.export.badframes_export_dir.as_str())
            || o.results_autosave != Some(self.config.export.results_autosave)
            || o.results_path.as_deref() != Some(self.config.export.results_path.as_str())
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
            self.ui.pending_save_since = None;
            return;
        }
        match self.ui.pending_save_since {
            None => {
                self.ui.pending_save_since = Some(now);
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(SAVE_DEBOUNCE_SECS));
            }
            Some(since) if now - since >= SAVE_DEBOUNCE_SECS => {
                let snap = self.snapshot();
                match crate::state::save(&snap) {
                    Some(path) => {
                        // Read-only install: the save fell back (or the log
                        // did) — toast once so persistence loss is visible.
                        if path != crate::state::state_path() && !self.ui.state_fallback_toasted {
                            self.ui.state_fallback_toasted = true;
                            self.ui.toast(
                                now,
                                format!(
                                    "App folder not writable — state saves to {}",
                                    path.display()
                                ),
                                ToastKind::Warning,
                            );
                        }
                        self.ui.saved_snapshot = snap;
                        self.ui.pending_save_since = None;
                    }
                    None => {
                        // Transient (locked/full disk): retry next debounce,
                        // toast once.
                        if !self.ui.state_fallback_toasted {
                            self.ui.state_fallback_toasted = true;
                            self.ui.toast(
                                now,
                                "State save failed everywhere — retrying".to_owned(),
                                ToastKind::Warning,
                            );
                        }
                        self.ui.pending_save_since = Some(now);
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
        if !self.run.results_autosave_pending {
            return;
        }
        self.run.results_autosave_pending = false;
        let path = if self.config.export.results_path.trim().is_empty() {
            crate::metrics::results::default_results_path()
        } else {
            std::path::PathBuf::from(&self.config.export.results_path)
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
        let ffmpeg_version = self
            .binaries
            .ffmpeg
            .ffmpeg_version
            .clone()
            .unwrap_or_default();
        let lines: Vec<String> = self
            .queue
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
                self.ui.toast(
                    now,
                    format!("Appended {n} row{s} to {}", path.display()),
                    ToastKind::Info,
                );
            }
            Err(e) => {
                log::error!(target: "rfmetrics::app", "save results failed: {e}");
                self.ui.toast(
                    now,
                    format!("Could not save results: {e}"),
                    ToastKind::Error,
                );
            }
        }
    }
}
