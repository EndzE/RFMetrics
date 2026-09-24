//! Bad-frames viewer domain: worker message + plan types.
//!
//! Extracted from `app.rs` (High 1 split). Viewer/export method bodies
//! move here in a follow-up step.

use crate::metrics::ffmpeg::MetricKind;

/// Progress + summary from the bad-frames worker (one thread, sequential
/// accurate seeks; abort stops between frames).
#[derive(Debug)]
pub(crate) enum BadframeMsg {
    Progress { done: usize, total: usize },
    Finished { ok: usize, errors: Vec<String> },
}

/// One PNG to extract (owned snapshot for the worker thread).
pub(crate) struct BadframeJob {
    pub(crate) kind: MetricKind,
    pub(crate) dist_path: String,
    pub(crate) dist_fps: f64,
    pub(crate) frame: usize,
    pub(crate) offset: f64,
}

/// Frozen-at-click export plan: no UI borrows cross into the thread.
pub(crate) struct BadframePlan {
    pub(crate) ffmpeg: std::path::PathBuf,
    pub(crate) ref_path: String,
    pub(crate) ref_fps: f64,
    pub(crate) tmp: std::path::PathBuf,
    pub(crate) jobs: Vec<BadframeJob>,
}

/// Export scope for the bad-frames Export buttons (bf_opts row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BadframeExportScope {
    Pair,
    Metric,
    All,
}

/// One worst-frame pair to export (owned snapshot for the worker).
pub(crate) struct BadframeExportPair {
    pub(crate) kind: MetricKind,
    pub(crate) dist_path: String,
    pub(crate) dist_fps: f64,
    pub(crate) frame: usize,
    pub(crate) offset: f64,
}

/// Direct-to-destination extract (viewer tmp untouched, so no wipe and
/// no stale-tmp risk; overwrites like the old Save-all copy).
pub(crate) struct BadframeExportJob {
    pub(crate) kind: MetricKind,
    pub(crate) dist_src: String,
    pub(crate) dist_fps: f64,
    pub(crate) frame: usize,
    pub(crate) offset: f64,
    pub(crate) dest_dist: std::path::PathBuf,
    pub(crate) dest_ref: std::path::PathBuf,
}

/// Tmp-to-destination copy for an already-extracted pair (exports exactly
/// what the viewer shows). Runs in the export worker, never on the UI
/// thread — batch scopes copy hundreds of multi-MB PNGs.
pub(crate) struct BadframeExportCopy {
    pub(crate) tmp_dist: std::path::PathBuf,
    pub(crate) tmp_ref: std::path::PathBuf,
    pub(crate) dest_dist: std::path::PathBuf,
    pub(crate) dest_ref: std::path::PathBuf,
    pub(crate) name: &'static str,
    pub(crate) frame: usize,
}

/// Frozen-at-click export plan: no UI borrows cross into the thread.
pub(crate) struct BadframeExportPlan {
    pub(crate) ffmpeg: std::path::PathBuf,
    pub(crate) ref_src: String,
    pub(crate) ref_fps: f64,
    pub(crate) jobs: Vec<BadframeExportJob>,
}

/// Pending export summary: worker copies + PNGs land here on Finished,
/// then toasted (viewer tmp/textures untouched).
pub(crate) struct BadframeExportPending {
    pub(crate) copied: usize,
    pub(crate) failed: Vec<String>,
    pub(crate) dest_note: String,
}

impl crate::app::RFMetricsApp {
    /// Snapshot of finished cells for the bad-frames worker (owned so
    /// the thread never touches UI state). Current viewer tab only.
    pub(crate) fn badframe_jobs_for(&self, kind: MetricKind) -> Option<BadframePlan> {
        use crate::metrics::badframes;
        let ffmpeg = self.ffmpeg.path.clone()?;
        let ref_path = self.ref_path.clone();
        if ref_path.trim().is_empty() || !std::path::Path::new(&ref_path).is_file() {
            return None;
        }
        let skip = Self::trim_opt(&self.skip)?.unwrap_or(0.0);
        let ref_fps = self
            .ref_info_data
            .as_ref()
            .and_then(|m| m.fps)
            .filter(|f| *f > 0.0);
        let n: usize = self.badframes_count.parse().ok().filter(|n| *n >= 1)?;
        let mut jobs: Vec<BadframeJob> = Vec::new();
        for row in &self.rows {
            if !row.include {
                continue;
            }
            let dist_fps = row.info.as_ref().and_then(|m| m.fps).filter(|f| *f > 0.0);
            let Some(fps) = ref_fps.or(dist_fps) else {
                continue;
            };
            let (values, vmaf_cfg) = match row.cell(kind) {
                crate::metrics::MetricCell::Done {
                    values, vmaf_cfg, ..
                } if !values.is_empty() => (values.clone(), vmaf_cfg.clone()),
                _ => continue,
            };
            let picks = badframes::worst_n(&values, n, kind == MetricKind::But);
            let stride = badframes::stride_for(kind, vmaf_cfg.as_ref());
            for (idx, _) in picks {
                let frame = idx.saturating_mul(stride);
                jobs.push(BadframeJob {
                    kind,
                    dist_path: row.path.clone(),
                    dist_fps: dist_fps.unwrap_or(fps),
                    frame,
                    offset: badframes::frame_offset(skip, frame, fps),
                });
            }
        }
        if jobs.is_empty() {
            return None;
        }
        Some(BadframePlan {
            ffmpeg,
            ref_path,
            ref_fps: ref_fps.unwrap_or_else(|| {
                self.rows
                    .iter()
                    .filter_map(|r| r.info.as_ref().and_then(|m| m.fps))
                    .next()
                    .unwrap_or(30.0)
            }),
            tmp: self.badframe_tmp.clone(),
            jobs,
        })
    }

    /// Worst list for one viewer `(file key, tab)`: `(actual_frame, value,
    /// offset)`, worst-first. Empty when the cell isn't Done or fps unknown.
    pub(crate) fn badframe_picks(&self, kind: MetricKind, key: &str) -> Vec<(usize, f64, f64)> {
        use crate::metrics::badframes;
        let row = match self.rows.iter().find(|r| r.key == key) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let skip = match Self::trim_opt(&self.skip) {
            Some(v) => v.unwrap_or(0.0),
            None => return Vec::new(),
        };
        let ref_fps = self
            .ref_info_data
            .as_ref()
            .and_then(|m| m.fps)
            .filter(|f| *f > 0.0);
        let dist_fps = row.info.as_ref().and_then(|m| m.fps).filter(|f| *f > 0.0);
        let Some(fps) = ref_fps.or(dist_fps) else {
            return Vec::new();
        };
        let (values, vmaf_cfg) = match row.cell(kind) {
            crate::metrics::MetricCell::Done {
                values, vmaf_cfg, ..
            } if !values.is_empty() => (values, vmaf_cfg),
            _ => return Vec::new(),
        };
        let n: usize = self
            .badframes_count
            .parse()
            .ok()
            .filter(|n| *n >= 1)
            .unwrap_or(5);
        let stride = badframes::stride_for(kind, vmaf_cfg.as_ref());
        badframes::worst_n(values, n, kind == MetricKind::But)
            .into_iter()
            .map(|(idx, v)| {
                let frame = idx.saturating_mul(stride);
                (frame, v, badframes::frame_offset(skip, frame, fps))
            })
            .collect()
    }

    /// Queue rows with a finished cell for the viewer tab (file picker).
    pub(crate) fn badframe_files_for(&self, kind: MetricKind) -> Vec<(String, String)> {
        self.rows
            .iter()
            .filter(|r| r.include)
            .filter(|r| {
                matches!(r.cell(kind), crate::metrics::MetricCell::Done { values, .. } if !values.is_empty())
            })
            .map(|r| (r.key.clone(), r.display.clone()))
            .collect()
    }

    /// Spawn the bad-frames worker for the viewer tab: sequential accurate
    /// seeks into the run tmp dir, dist + ref per frame. Abort stops between
    /// frames (mid-seek ffmpeg is bounded by `BADFRAME_TIMEOUT`).
    /// ponytail: abort between frames, not mid-seek; a Stop click waits out
    /// at most one single-frame extract.
    pub(crate) fn start_badframes(&mut self, now: f64) {
        use std::sync::atomic::Ordering;
        if self.measuring || self.badframes_busy {
            return;
        }
        let kind = self.badframe_tab;
        let Some(plan) = self.badframe_jobs_for(kind) else {
            self.toast(
                now,
                "Nothing to export: run a metric first".to_owned(),
                crate::app::ToastKind::Info,
            );
            return;
        };
        let _ = std::fs::remove_dir_all(&plan.tmp);
        if let Err(e) = std::fs::create_dir_all(&plan.tmp) {
            self.toast(
                now,
                format!("Could not create tmp dir: {e}"),
                crate::app::ToastKind::Error,
            );
            return;
        }
        let total = plan.jobs.len() * 2;
        self.badframes_busy = true;
        self.badframe_done = 0;
        self.badframe_total = total;
        self.badframe_abort.store(false, Ordering::SeqCst);
        let tx = self.badframe_tx.clone();
        let abort = self.badframe_abort.clone();
        std::thread::spawn(move || {
            use crate::metrics::badframes;
            let mut ok = 0usize;
            let mut errors: Vec<String> = Vec::new();
            let mut done = 0usize;
            for job in &plan.jobs {
                for (src, fps, dest) in [
                    (
                        job.dist_path.as_str(),
                        job.dist_fps,
                        badframes::tmp_dest_for(
                            &plan.tmp,
                            &job.dist_path,
                            job.kind.name(),
                            job.frame,
                        ),
                    ),
                    (
                        plan.ref_path.as_str(),
                        plan.ref_fps,
                        badframes::tmp_dest_ref_for(
                            &plan.tmp,
                            &job.dist_path,
                            job.kind.name(),
                            job.frame,
                        ),
                    ),
                ] {
                    if abort.load(Ordering::SeqCst) {
                        break;
                    }
                    if badframes::extract_one(&plan.ffmpeg, src, &dest, job.offset, fps) {
                        ok += 1;
                    } else {
                        errors.push(format!("{} frame {}", job.kind.name(), job.frame));
                    }
                    done += 1;
                    let _ = tx.send(BadframeMsg::Progress { done, total });
                }
                if abort.load(Ordering::SeqCst) {
                    break;
                }
            }
            let _ = tx.send(BadframeMsg::Finished { ok, errors });
        });
        log::info!(target: "rfmetrics::app", "bad-frames started: {} extracts", total);
    }

    /// Stop an in-flight bad-frames export (checked between seeks).
    pub(crate) fn stop_badframes(&mut self) {
        use std::sync::atomic::Ordering;
        if !self.badframes_busy {
            return;
        }
        self.badframe_abort.store(true, Ordering::SeqCst);
        log::info!(target: "rfmetrics::app", "bad-frames aborted by user");
    }

    /// Worst-frame pairs for an export scope: Pair = the open pair only,
    /// Metric = every file × worst-N of the current tab, All = every tab
    /// with finished values. Rows without usable fps are skipped, like the
    /// Extract worker.
    pub(crate) fn export_pairs(&self, scope: BadframeExportScope) -> Vec<BadframeExportPair> {
        let kinds: Vec<MetricKind> = match scope {
            BadframeExportScope::Pair | BadframeExportScope::Metric => vec![self.badframe_tab],
            BadframeExportScope::All => MetricKind::ALL.to_vec(),
        };
        let ref_fps = self
            .ref_info_data
            .as_ref()
            .and_then(|m| m.fps)
            .filter(|f| *f > 0.0);
        let mut out = Vec::new();
        for kind in kinds {
            let keys: Vec<String> = match scope {
                BadframeExportScope::Pair => self.badframe_file.clone().into_iter().collect(),
                BadframeExportScope::Metric | BadframeExportScope::All => self
                    .badframe_files_for(kind)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect(),
            };
            for key in keys {
                let Some(row) = self.rows.iter().find(|r| r.key == key) else {
                    continue;
                };
                let picks = self.badframe_picks(kind, &key);
                let frames: Vec<(usize, f64)> = match scope {
                    BadframeExportScope::Pair => picks
                        .get(self.badframe_frame_pos)
                        .map(|(f, _, o)| (*f, *o))
                        .into_iter()
                        .collect(),
                    BadframeExportScope::Metric | BadframeExportScope::All => {
                        picks.into_iter().map(|(f, _, o)| (f, o)).collect()
                    }
                };
                if frames.is_empty() {
                    continue;
                }
                let dist_fps = row.info.as_ref().and_then(|m| m.fps).filter(|f| *f > 0.0);
                let Some(dfps) = dist_fps.or(ref_fps) else {
                    continue;
                };
                for (frame, offset) in frames {
                    out.push(BadframeExportPair {
                        kind,
                        dist_path: row.path.clone(),
                        dist_fps: dfps,
                        frame,
                        offset,
                    });
                }
            }
        }
        out
    }

    /// Toast an export outcome (PNG counts; worker PNGs + tmp copies).
    pub(crate) fn toast_export(&mut self, now: f64, saved: usize, failed: Vec<String>, dest: &str) {
        if failed.is_empty() {
            self.toast(
                now,
                format!("Exported {saved} PNGs to {dest}"),
                crate::app::ToastKind::Info,
            );
        } else {
            let first = failed[0].clone();
            let s = if failed.len() == 1 { "" } else { "s" };
            self.toast(
                now,
                format!(
                    "Export: {saved} saved, {} failed{s} ({first}) → {dest}",
                    failed.len()
                ),
                crate::app::ToastKind::Error,
            );
        }
    }

    /// Export worst-frame PNGs: tmp copies when the pair is already
    /// extracted (exports exactly what the viewer shows), otherwise a
    /// background worker extracting straight to the destination (viewer
    /// tmp untouched). Empty browse line = beside each distorted file.
    pub(crate) fn start_export(&mut self, scope: BadframeExportScope, now: f64) {
        use std::sync::atomic::Ordering;
        if self.measuring || self.badframes_busy {
            return;
        }
        let Some(ffmpeg) = self.ffmpeg.path.clone() else {
            self.toast(
                now,
                "Export needs ffmpeg".to_owned(),
                crate::app::ToastKind::Error,
            );
            return;
        };
        let ref_path = self.ref_path.clone();
        if ref_path.trim().is_empty() || !std::path::Path::new(&ref_path).is_file() {
            self.toast(
                now,
                "Export needs the reference file".to_owned(),
                crate::app::ToastKind::Error,
            );
            return;
        }
        let pairs = self.export_pairs(scope);
        if pairs.is_empty() {
            self.toast(
                now,
                "Nothing to export: run a metric first".to_owned(),
                crate::app::ToastKind::Info,
            );
            return;
        }
        let export_dir = self.badframes_export_dir.clone();
        if !export_dir.trim().is_empty()
            && let Err(e) = std::fs::create_dir_all(&export_dir)
        {
            self.toast(
                now,
                format!("Could not create export folder: {e}"),
                crate::app::ToastKind::Error,
            );
            return;
        }
        let ref_fps = self
            .ref_info_data
            .as_ref()
            .and_then(|m| m.fps)
            .filter(|f| *f > 0.0)
            .unwrap_or_else(|| {
                self.rows
                    .iter()
                    .filter_map(|r| r.info.as_ref().and_then(|m| m.fps))
                    .next()
                    .unwrap_or(30.0)
            });
        let mut copied = Vec::new();
        let mut jobs: Vec<BadframeExportJob> = Vec::new();
        for p in pairs {
            let name = p.kind.name();
            let tmp_d = crate::metrics::badframes::tmp_dest_for(
                &self.badframe_tmp,
                &p.dist_path,
                name,
                p.frame,
            );
            let tmp_r = crate::metrics::badframes::tmp_dest_ref_for(
                &self.badframe_tmp,
                &p.dist_path,
                name,
                p.frame,
            );
            let dest_d = crate::metrics::badframes::export_dest_for(
                &export_dir,
                &p.dist_path,
                name,
                p.frame,
                false,
            );
            let dest_r = crate::metrics::badframes::export_dest_for(
                &export_dir,
                &p.dist_path,
                name,
                p.frame,
                true,
            );
            if tmp_d.is_file() && tmp_r.is_file() {
                // Collected for the worker below: batch scopes copy
                // hundreds of multi-MB PNGs, never on the UI thread.
                copied.push(BadframeExportCopy {
                    tmp_dist: tmp_d,
                    tmp_ref: tmp_r,
                    dest_dist: dest_d,
                    dest_ref: dest_r,
                    name,
                    frame: p.frame,
                });
            } else {
                jobs.push(BadframeExportJob {
                    kind: p.kind,
                    dist_src: p.dist_path,
                    dist_fps: p.dist_fps,
                    frame: p.frame,
                    offset: p.offset,
                    dest_dist: dest_d,
                    dest_ref: dest_r,
                });
            }
        }
        let dest_note = if export_dir.trim().is_empty() {
            "beside each file".to_owned()
        } else {
            export_dir
        };
        if jobs.is_empty() && copied.is_empty() {
            self.toast_export(now, 0, Vec::new(), &dest_note);
            return;
        }
        let plan = BadframeExportPlan {
            ffmpeg,
            ref_src: ref_path,
            ref_fps,
            jobs,
        };
        let total = copied.len() * 2 + plan.jobs.len() * 2;
        self.badframes_busy = true;
        self.badframe_done = 0;
        self.badframe_total = total;
        self.badframe_abort.store(false, Ordering::SeqCst);
        self.badframe_export_pending = Some(BadframeExportPending {
            copied: 0,
            failed: Vec::new(),
            dest_note,
        });
        let tx = self.badframe_tx.clone();
        let abort = self.badframe_abort.clone();
        std::thread::spawn(move || {
            use crate::metrics::badframes;
            let mut ok = 0usize;
            let mut errors: Vec<String> = Vec::new();
            let mut done = 0usize;
            // Tmp copies first: local and quick, so no abort gate (Stop
            // semantics today only interrupt extracts between frames).
            for c in &copied {
                match (
                    std::fs::copy(&c.tmp_dist, &c.dest_dist),
                    std::fs::copy(&c.tmp_ref, &c.dest_ref),
                ) {
                    (Ok(_), Ok(_)) => ok += 2,
                    _ => errors.push(format!("{} frame {}", c.name, c.frame)),
                }
                done += 2;
                let _ = tx.send(BadframeMsg::Progress { done, total });
            }
            for job in &plan.jobs {
                for (src, fps, dest) in [
                    (job.dist_src.as_str(), job.dist_fps, job.dest_dist.clone()),
                    (plan.ref_src.as_str(), plan.ref_fps, job.dest_ref.clone()),
                ] {
                    if abort.load(Ordering::SeqCst) {
                        break;
                    }
                    if badframes::extract_one(&plan.ffmpeg, src, &dest, job.offset, fps) {
                        ok += 1;
                    } else {
                        errors.push(format!("{} frame {}", job.kind.name(), job.frame));
                    }
                    done += 1;
                    let _ = tx.send(BadframeMsg::Progress { done, total });
                }
                if abort.load(Ordering::SeqCst) {
                    break;
                }
            }
            let _ = tx.send(BadframeMsg::Finished { ok, errors });
        });
        log::info!(target: "rfmetrics::app", "bad-frames export started: {} files", total);
    }

    /// Drain bad-frames progress; returns true on activity. Viewer runs
    /// rescan tmp, drop textures and auto-select the first file; export
    /// runs toast the combined copy + extract outcome instead.
    pub(crate) fn drain_badframe_results(&mut self, now: f64) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.badframe_rx.try_recv() {
            activity = true;
            match msg {
                BadframeMsg::Progress { done, total } => {
                    self.badframe_done = done;
                    self.badframe_total = total;
                }
                BadframeMsg::Finished { ok, errors } => {
                    self.badframes_busy = false;
                    self.badframe_done = 0;
                    self.badframe_total = 0;
                    // Deferred close cleanup: the window closed mid-run and
                    // tmp stayed alive for the worker until now (`Finished`
                    // is its last send, so nothing touches tmp afterwards).
                    if self.badframe_tmp_cleanup_pending {
                        self.badframe_tmp_cleanup_pending = false;
                        let _ = std::fs::remove_dir_all(&self.badframe_tmp);
                    }
                    // Export extracts went straight to the destination:
                    // toast the combined outcome, viewer tmp untouched.
                    if let Some(pending) = self.badframe_export_pending.take() {
                        let mut failed = pending.failed;
                        failed.extend(errors);
                        self.toast_export(now, pending.copied + ok, failed, &pending.dest_note);
                        continue;
                    }
                    self.badframe_files = std::fs::read_dir(&self.badframe_tmp)
                        .map(|entries| {
                            let mut v: Vec<std::path::PathBuf> = entries
                                .filter_map(|e| e.ok().map(|e| e.path()))
                                .filter(|p| p.extension().is_some_and(|x| x == "png"))
                                .collect();
                            v.sort();
                            v
                        })
                        .unwrap_or_default();
                    self.badframe_tex_dist = None;
                    self.badframe_tex_ref = None;
                    self.badframe_tex_key = None;
                    self.badframe_frame_pos = 0;
                    if self
                        .badframe_file
                        .as_ref()
                        .is_none_or(|k| !self.rows.iter().any(|r| &r.key == k))
                    {
                        self.badframe_file = self
                            .badframe_files_for(self.badframe_tab)
                            .into_iter()
                            .next()
                            .map(|(k, _)| k);
                    }
                    self.badframe_report = Some((ok, errors));
                }
            }
        }
        if let Some((ok, errors)) = self.badframe_report.take() {
            if errors.is_empty() {
                let s = if ok == 1 { "" } else { "s" };
                self.toast(
                    now,
                    format!("Extracted {ok} bad-frame PNG{s}"),
                    crate::app::ToastKind::Info,
                );
            } else {
                let first = errors[0].clone();
                let s = if errors.len() == 1 { "" } else { "s" };
                self.toast(
                    now,
                    format!(
                        "Bad frames: {ok} saved, {} failed{s} ({first})",
                        errors.len()
                    ),
                    crate::app::ToastKind::Error,
                );
            }
        }
        activity
    }

    /// Upload the visible viewer pair as textures when the selection
    /// changed. Full resolution (inspection needs detail); only two
    /// textures are ever cached.
    pub(crate) fn refresh_viewer_textures(&mut self, ctx: &egui::Context) {
        let key = match &self.badframe_file {
            Some(k) => (k.clone(), self.badframe_tab, self.badframe_frame_pos),
            None => return,
        };
        if self.badframe_tex_key.as_ref() == Some(&key) {
            return;
        }
        self.badframe_tex_dist = None;
        self.badframe_tex_ref = None;
        let picks = self.badframe_picks(key.1, &key.0);
        let Some((frame, _, _)) = picks.get(key.2).copied() else {
            return;
        };
        let row_path = match self.rows.iter().find(|r| r.key == key.0) {
            Some(r) => r.path.clone(),
            None => return,
        };
        let load = |p: std::path::PathBuf| -> Option<egui::TextureHandle> {
            let bytes = std::fs::read(&p).ok()?;
            if bytes.len() <= 100 {
                return None;
            }
            let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
            let (w, h) = (img.width(), img.height());
            if w == 0 || h == 0 {
                return None;
            }
            Some(ctx.load_texture(
                p.to_string_lossy().into_owned(),
                egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img.into_raw()),
                egui::TextureOptions::LINEAR,
            ))
        };
        self.badframe_tex_dist = load(crate::metrics::badframes::tmp_dest_for(
            &self.badframe_tmp,
            &row_path,
            key.1.name(),
            frame,
        ));
        self.badframe_tex_ref = load(crate::metrics::badframes::tmp_dest_ref_for(
            &self.badframe_tmp,
            &row_path,
            key.1.name(),
            frame,
        ));
        self.badframe_tex_key = Some(key);
    }

    /// Bad-frames viewer in its own OS window (mirrors `show_plots`):
    /// per-metric tabs, file picker, worst-frame stepper, dist/ref pair
    /// side by side with shared zoom + scroll-pan, current-tab extractor,
    /// and an options box with count + save-all-to-folder.
    ///
    /// Close while a worker runs defers tmp deletion until its `Finished`
    /// drains (no abort: an export launched from the viewer may be using
    /// tmp, and the run is bounded anyway).
    pub(crate) fn close_badframes(&mut self) {
        self.show_badframes = false;
        if self.badframes_busy {
            self.badframe_tmp_cleanup_pending = true;
        } else {
            // Best-effort tmp cleanup; save-all must happen while open.
            let _ = std::fs::remove_dir_all(&self.badframe_tmp);
        }
        self.badframe_files.clear();
        self.badframe_tex_dist = None;
        self.badframe_tex_ref = None;
        self.badframe_tex_key = None;
    }

    pub(crate) fn show_badframes(&mut self, ctx: &egui::Context) {
        if !self.show_badframes {
            return;
        }
        let id = egui::ViewportId::from_hash_of("badframes_view");
        let builder = egui::ViewportBuilder::default()
            .with_title("Bad frames")
            .with_inner_size([1100.0, 700.0]);
        ctx.show_viewport_immediate(id, builder, |vui, _class| {
            if vui.input(|i| i.viewport().close_requested()) {
                self.close_badframes();
                return;
            }
            let vnow = vui.input(|i| i.time);
            let kind = self.badframe_tab;
            // Tab strip (all 7, like the plot window).
            egui::Panel::top("bf_tabs").show(vui, |ui| {
                ui.horizontal(|ui| {
                    for tab in MetricKind::ALL {
                        let title = crate::plot::tab_title(tab);
                        if ui
                            .add(egui::Button::new(title).selected(self.badframe_tab == tab))
                            .clicked()
                        {
                            self.badframe_tab = tab;
                            self.badframe_frame_pos = 0;
                            self.badframe_tex_key = None;
                        }
                    }
                });
            });
            // Runner row: extract current tab / stop + progress.
            egui::Panel::top("bf_run").show(vui, |ui| {
                ui.horizontal(|ui| {
                    if self.badframes_busy {
                        if ui
                            .add_sized([110.0, 24.0], egui::Button::new("Stop"))
                            .clicked()
                        {
                            self.stop_badframes();
                        }
                        ui.label(format!(
                            "{} {}/{}",
                            if self.badframe_export_pending.is_some() {
                                "Exporting"
                            } else {
                                "Extracting"
                            },
                            self.badframe_done,
                            self.badframe_total
                        ));
                    } else {
                        let can_run =
                            self.ffmpeg.path.is_some() && !self.badframe_files_for(kind).is_empty();
                        if ui
                            .add_enabled_ui(can_run, |ui| {
                                ui.add_sized([110.0, 24.0], egui::Button::new("Extract"))
                            })
                            .inner
                            .on_hover_text("Extract worst frames for this tab into tmp")
                            .clicked()
                        {
                            self.start_badframes(vnow);
                        }
                        if self.badframe_files_for(kind).is_empty() {
                            ui.label("Run a metric first");
                        }
                    }
                });
            });
            // File + frame controls.
            let files = self.badframe_files_for(kind);
            if !files.iter().any(|(k, _)| Some(k) == self.badframe_file.as_ref()) {
                self.badframe_file = files.first().map(|(k, _)| k.clone());
                self.badframe_frame_pos = 0;
                self.badframe_tex_key = None;
            }
            egui::Panel::top("bf_pick").show(vui, |ui| {
                ui.horizontal(|ui| {
                    let current = self
                        .badframe_file
                        .as_ref()
                        .and_then(|k| files.iter().find(|(fk, _)| fk == k))
                        .map(|(_, d)| d.clone())
                        .unwrap_or_else(|| "No file".to_owned());
                    egui::ComboBox::from_id_salt("bf_file")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for (k, d) in &files {
                                let _ = ui.selectable_value(
                                    self.badframe_file.get_or_insert_with(|| k.clone()),
                                    k.clone(),
                                    d.as_str(),
                                );
                            }
                        });
                    let picks = self
                        .badframe_file
                        .as_ref()
                        .map(|k| self.badframe_picks(kind, k))
                        .unwrap_or_default();
                    let max_pos = picks.len().saturating_sub(1);
                    if self.badframe_frame_pos > max_pos {
                        self.badframe_frame_pos = max_pos;
                        self.badframe_tex_key = None;
                    }
                    if ui.add_enabled(self.badframe_frame_pos > 0, egui::Button::new("◀")).clicked() {
                        self.badframe_frame_pos = self.badframe_frame_pos.saturating_sub(1);
                        self.badframe_tex_key = None;
                    }
                    let mut pos = self.badframe_frame_pos;
                    if !picks.is_empty() {
                        ui.add(
                            egui::Slider::new(&mut pos, 0..=max_pos)
                                .show_value(false)
                                .trailing_fill(true),
                        );
                        if pos != self.badframe_frame_pos {
                            self.badframe_frame_pos = pos;
                            self.badframe_tex_key = None;
                        }
                    }
                    if ui
                        .add_enabled(self.badframe_frame_pos < max_pos, egui::Button::new("▶"))
                        .clicked()
                    {
                        self.badframe_frame_pos = self.badframe_frame_pos.saturating_add(1).min(max_pos);
                        self.badframe_tex_key = None;
                    }
                    if let Some((frame, value, _)) = picks.get(self.badframe_frame_pos).copied() {
                        ui.label(format!("Frame {frame} ({value:.4})"));
                    }
                    if ui
                        .button("Reset view")
                        .on_hover_text("Fit both images (zoom/pan stay linked)")
                        .clicked()
                    {
                        self.badframe_reset_once = true;
                    }
                    ui.separator();
                    ui.selectable_value(&mut self.badframe_slider, false, "Side")
                        .on_hover_text("Distorted and reference side by side");
                    ui.selectable_value(&mut self.badframe_slider, true, "Slider")
                        .on_hover_text("Before/after wipe — drag the divider");
                });
            });
            // Options box first: egui requires CentralPanel after all
            // other panels, otherwise the bottom panel gets zero space.
            egui::Panel::bottom("bf_opts").show(vui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new("Bad frames").selectable(false));
                    let _ = egui::ComboBox::from_id_salt("bf_count")
                        .selected_text(self.badframes_count.as_str())
                        .show_ui(ui, |ui| {
                            for v in crate::metrics::badframes::COUNT_LABELS {
                                let _ = ui.selectable_value(
                                    &mut self.badframes_count,
                                    v.to_owned(),
                                    v,
                                );
                            }
                        });
                    let busy = self.badframes_busy || self.measuring;
                    let can_pair = !busy
                        && self.badframe_file.as_ref().is_some_and(|k| {
                            self.badframe_picks(kind, k)
                                .get(self.badframe_frame_pos)
                                .is_some()
                        });
                    let can_metric =
                        !busy && !self.badframe_files_for(kind).is_empty();
                    let can_all = !busy
                        && MetricKind::ALL
                            .iter()
                            .any(|k| !self.badframe_files_for(*k).is_empty());
                    if ui
                        .add_enabled(can_pair, egui::Button::new("Export pair"))
                        .on_hover_text("Export the open pair (this tab, file and frame)")
                        .clicked()
                    {
                        self.start_export(BadframeExportScope::Pair, vnow);
                    }
                    if ui
                        .add_enabled(can_metric, egui::Button::new("Export metric"))
                        .on_hover_text("Export all worst-frame pairs of this tab")
                        .clicked()
                    {
                        self.start_export(BadframeExportScope::Metric, vnow);
                    }
                    if ui
                        .add_enabled(can_all, egui::Button::new("Export all"))
                        .on_hover_text(
                            "Extract missing frames for every finished metric, then export all pairs",
                        )
                        .clicked()
                    {
                        self.start_export(BadframeExportScope::All, vnow);
                    }
                    ui.label(format!("{} PNGs in tmp", self.badframe_files.len()));
                });
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new("Export folder").selectable(false));
                    // Bounded display (full path stays in the hover).
                    let full = self.badframes_export_dir.clone();
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
                        "Empty: each pair lands next to its distorted file".to_owned()
                    } else {
                        full
                    });
                    if ui.button("Browse…").clicked()
                        && let Some(dir) = rfd::FileDialog::new()
                            .set_title("Bad-frames export folder")
                            .pick_folder()
                    {
                        self.badframes_export_dir = dir.to_string_lossy().into_owned();
                    }
                    if ui
                        .button("Clear")
                        .on_hover_text("Back to beside-the-distorted-file")
                        .clicked()
                    {
                        self.badframes_export_dir.clear();
                    }
                });
            });
            // Side-by-side pair as linked plots (shared zoom/pan): both
            // images centered at the origin at true pixel size, so one view
            // transform fits both. Stock plot gestures: drag pans, wheel
            // zooms, box-select zooms.
            self.refresh_viewer_textures(vui);
            // Any selection change re-fits: `Plot::reset()` clears both the
            // stored bounds and the shared link-group entry (a fresh plot id
            // alone would inherit the group's zoom).
            let view_key = (
                kind,
                self.badframe_file.clone().unwrap_or_default(),
                self.badframe_frame_pos,
            );
            if self.badframe_view_key.as_ref() != Some(&view_key) {
                self.badframe_view_key = Some(view_key);
                self.badframe_reset_once = true;
            }
            egui::CentralPanel::default().show(vui, |ui| {
                // Overlay wipe as a single plot: UV-cropped halves tile
                // exactly at the divider (ref left, dist right), so stock
                // plot gestures give pan (drag), zoom (scroll/box) and
                // double-click fit. Divider drag suppresses pan via last
                // frame's divider screen x. No default bounds: auto-bounds
                // + expanding aspect contain-fits the pair (never crops),
                // on first show, Reset view, double-click and selection
                // change alike.
                if self.badframe_slider {
                    let dist = self.badframe_tex_dist.clone();
                    let refr = self.badframe_tex_ref.clone();
                    match (dist, refr) {
                        (Some(d), Some(r)) => {
                            ui.label("Reference (left) | Distorted (right) — drag divider to compare · drag to pan · scroll to zoom · double-click to fit");
                            let (ds, rs) = (d.size(), r.size());
                            let w = ds[0].max(rs[0]) as f64;
                            let h = ds[1].max(rs[1]) as f64;
                            let lay = crate::metrics::badframes::wipe_layout(
                                w,
                                self.badframe_split,
                            );
                            let hover_x = ui
                                .ctx()
                                .pointer_hover_pos()
                                .map(|p| p.x)
                                .unwrap_or(f32::NAN);
                            let suppress = self.badframe_div_drag
                                || (self.badframe_div_sx.is_finite()
                                    && (hover_x - self.badframe_div_sx).abs() <= 10.0);
                            let do_reset = self.badframe_reset_once;
                            let plot = egui_plot::Plot::new("bf-wipe")
                                .data_aspect(1.0)
                                .show_grid(false)
                                .show_axes(false)
                                .show_crosshair(false)
                                .allow_drag(!suppress);
                            let plot = if do_reset { plot.reset() } else { plot };
                            let u = lay.u;
                            let resp = plot.show(ui, |plot_ui| {
                                plot_ui.image(
                                    egui_plot::PlotImage::new(
                                        "bf-wipe-ref",
                                        r.id(),
                                        egui_plot::PlotPoint::new(lay.left_cx, 0.0),
                                        egui::Vec2::new(lay.left_w as f32, h as f32),
                                    )
                                    .uv(egui::Rect::from_min_max(
                                        egui::Pos2::new(0.0, 0.0),
                                        egui::Pos2::new(u, 1.0),
                                    ))
                                    .allow_hover(false),
                                );
                                plot_ui.image(
                                    egui_plot::PlotImage::new(
                                        "bf-wipe-dist",
                                        d.id(),
                                        egui_plot::PlotPoint::new(lay.right_cx, 0.0),
                                        egui::Vec2::new(lay.right_w as f32, h as f32),
                                    )
                                    .uv(egui::Rect::from_min_max(
                                        egui::Pos2::new(u, 0.0),
                                        egui::Pos2::new(1.0, 1.0),
                                    ))
                                    .allow_hover(false),
                                );
                                plot_ui.line(
                                    egui_plot::Line::new(
                                        "bf-wipe-div",
                                        egui_plot::PlotPoints::new(vec![
                                            [lay.div_x, -h / 2.0],
                                            [lay.div_x, h / 2.0],
                                        ]),
                                    )
                                    .color(egui::Color32::WHITE)
                                    .width(2.0)
                                    .allow_hover(false),
                                );
                            });
                            let p0 = resp.transform.position_from_point(
                                &egui_plot::PlotPoint::new(-w / 2.0, -h / 2.0),
                            );
                            let p1 = resp.transform.position_from_point(
                                &egui_plot::PlotPoint::new(w / 2.0, h / 2.0),
                            );
                            let img = egui::Rect::from_two_pos(p0, p1);
                            let sx = resp
                                .transform
                                .position_from_point(&egui_plot::PlotPoint::new(
                                    lay.div_x, 0.0,
                                ))
                                .x;
                            self.badframe_div_sx = sx;
                            // Overlay follows the visible image area so tags,
                            // handle and grab stay on screen while zoomed or
                            // panned (half-centers drift off-screen).
                            let frame = resp.response.rect;
                            let vis = img.intersect(frame);
                            let painter =
                                ui.painter_at(frame).with_clip_rect(frame);
                            let font = egui::TextStyle::Small.resolve(ui.style());
                            if vis.is_positive() {
                                let y = vis.min.y + 16.0;
                                for (lo, hi, tag) in [
                                    (img.min.x, sx, "REF"),
                                    (sx, img.max.x, "DIST"),
                                ] {
                                    let (vlo, vhi) =
                                        (lo.max(vis.min.x), hi.min(vis.max.x));
                                    let galley = painter.layout_no_wrap(
                                        tag.to_owned(),
                                        font.clone(),
                                        egui::Color32::WHITE,
                                    );
                                    let half = galley.size().x / 2.0 + 8.0;
                                    if vhi - vlo < half * 2.0 + 4.0 {
                                        continue;
                                    }
                                    let c = egui::Pos2::new((vlo + vhi) / 2.0, y);
                                    let bg = egui::Rect::from_center_size(
                                        c,
                                        galley.size() + egui::Vec2::new(12.0, 4.0),
                                    );
                                    painter.rect_filled(
                                        bg,
                                        4.0,
                                        egui::Color32::from_black_alpha(150),
                                    );
                                    painter.text(
                                        c,
                                        egui::Align2::CENTER_CENTER,
                                        tag,
                                        font.clone(),
                                        egui::Color32::WHITE,
                                    );
                                }
                                if sx >= vis.min.x && sx <= vis.max.x {
                                    let hy = img
                                        .center()
                                        .y
                                        .clamp(vis.min.y, vis.max.y);
                                    painter.circle_filled(
                                        egui::Pos2::new(sx, hy),
                                        9.0,
                                        egui::Color32::from_black_alpha(160),
                                    );
                                    painter.circle_stroke(
                                        egui::Pos2::new(sx, hy),
                                        9.0,
                                        egui::Stroke::new(1.5, egui::Color32::WHITE),
                                    );
                                }
                            }
                            let grab = egui::Rect::from_x_y_ranges(
                                (sx - 8.0)..=(sx + 8.0),
                                vis.y_range(),
                            )
                            .intersect(frame);
                            let grab = if grab.is_positive() {
                                grab
                            } else {
                                egui::Rect::from_center_size(
                                    frame.center(),
                                    egui::Vec2::ZERO,
                                )
                            };
                            let grab_resp = ui.interact(
                                grab,
                                ui.id().with("bf_wipe_grab"),
                                egui::Sense::drag(),
                            );
                            if grab_resp.hovered() || grab_resp.dragged() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeColumn);
                            }
                            if grab_resp.dragged()
                                && let Some(pos) = grab_resp.interact_pointer_pos()
                                && img.width() > 10.0
                            {
                                self.badframe_split =
                                    crate::metrics::badframes::clamp_split(
                                        (pos.x - img.min.x) / img.width(),
                                    );
                            }
                            self.badframe_div_drag = grab_resp.dragged();
                            if do_reset {
                                self.badframe_reset_once = false;
                            }
                        }
                        (Some(d), None) => {
                            let _ = (d,);
                            ui.label("Reference — not extracted");
                        }
                        (None, Some(r)) => {
                            let _ = (r,);
                            ui.label("Distorted — not extracted");
                        }
                        (None, None) => {
                            ui.label("Extract frames to compare");
                        }
                    }
                    return;
                }
                // Union-fit defaults (stored memory wins once the user
                // pans/zooms within a selection).
                let (dxmin, dxmax, dymin, dymax) =
                    match (&self.badframe_tex_dist, &self.badframe_tex_ref) {
                        (Some(d), Some(r)) => {
                            let (ds, rs) = (d.size(), r.size());
                            crate::metrics::badframes::viewer_fit(
                                ds[0] as u32,
                                ds[1] as u32,
                                rs[0] as u32,
                                rs[1] as u32,
                            )
                        }
                        _ => (-1.0, 1.0, -1.0, 1.0),
                    };
                let do_reset = self.badframe_reset_once;
                ui.columns(2, |cols| {
                    if let Some(tex) = &self.badframe_tex_ref {
                        let (w, h) = (tex.size()[0] as f32, tex.size()[1] as f32);
                        cols[0].label("Reference");
                        let plot = egui_plot::Plot::new("bf-ref")
                            .link_axis("bf_img", true)
                            .data_aspect(1.0)
                            .show_grid(false)
                            .show_axes(false)
                            .show_crosshair(false)
                            .default_x_bounds(dxmin, dxmax)
                            .default_y_bounds(dymin, dymax);
                        let plot = if do_reset { plot.reset() } else { plot };
                        plot.show(&mut cols[0], |plot_ui| {
                            plot_ui.image(
                                egui_plot::PlotImage::new(
                                    "bf-ref-img",
                                    tex.id(),
                                    egui_plot::PlotPoint::new(0.0, 0.0),
                                    egui::Vec2::new(w, h),
                                )
                                .allow_hover(false),
                            );
                        });
                    } else {
                        cols[0].label("Reference — not extracted");
                    }
                    if let Some(tex) = &self.badframe_tex_dist {
                        let (w, h) = (tex.size()[0] as f32, tex.size()[1] as f32);
                        cols[1].label("Distorted");
                        let plot = egui_plot::Plot::new("bf-dist")
                            .link_axis("bf_img", true)
                            .data_aspect(1.0)
                            .show_grid(false)
                            .show_axes(false)
                            .show_crosshair(false)
                            .default_x_bounds(dxmin, dxmax)
                            .default_y_bounds(dymin, dymax);
                        let plot = if do_reset { plot.reset() } else { plot };
                        plot.show(&mut cols[1], |plot_ui| {
                            plot_ui.image(
                                egui_plot::PlotImage::new(
                                    "bf-dist-img",
                                    tex.id(),
                                    egui_plot::PlotPoint::new(0.0, 0.0),
                                    egui::Vec2::new(w, h),
                                )
                                .allow_hover(false),
                            );
                        });
                    } else {
                        cols[1].label("Distorted — not extracted");
                    }
                });
                if do_reset {
                    self.badframe_reset_once = false;
                }
            });
            // Keep progress live while the worker runs (viewport repaints
            // with the parent only on input otherwise).
            if self.badframes_busy {
                vui.request_repaint_after(std::time::Duration::from_millis(100));
            }
        });
    }
}
