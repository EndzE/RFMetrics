//! VMAF metric (Python `_compute_vmaf_series` parity): libvmaf filtergraph,
//! exe-dir JSON log file, tolerant log parsing. Rides the shared worker
//! plumbing (`RunInputs`, abort slot, progress channel); only the filter,
//! the temp log, and the parsers are VMAF-specific.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::metrics::ffmpeg::{
    FrameDetail, NORM, RunInputs, RunOutcome, ScaleMethod, last_err_line, no_data_outcome,
    pump_process, rate_args, scale_filter, setrange_segment, trim_window,
};
use crate::probe::MediaInfo;

/// Pooling selector (UI strings `Mean`/`Harmonic Mean` map here at Start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    #[default]
    Mean,
    HarmonicMean,
}

impl Pooling {
    pub(crate) fn as_filter_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::HarmonicMean => "harmonic_mean",
        }
    }
}

/// Validated VMAF settings snapshot, taken at Start (Python `vmaf_cfg`).
/// Stamped onto `Done` cells so a settings change invalidates them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmafCfg {
    pub model: String,
    pub phone: bool,
    pub scale: bool,
    pub pooling: Pooling,
    pub subsample: u32,
    /// Resolved thread count (`0` omits the option).
    pub n_threads: u32,
}

/// System thread count for libvmaf (`0` when undetectable → option omitted).
pub fn system_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0)
}

/// Home directory for models + temp logs: next to the exe (Python
/// `app_dir` parity), falling back like the logger when unresolvable.
pub(crate) fn vmaf_home() -> PathBuf {
    crate::binaries::app_dir()
}

/// `vmaf-models/*.json` sorted, or the Python `"No models found"` sentinel.
pub fn list_models(models_dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(models_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && p.extension().is_some_and(|x| x == "json")
                        && p.file_name()
                            .is_some_and(|n| n.to_string_lossy() != "No models found")
                })
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    if names.is_empty() {
        names.push("No models found".to_owned());
    }
    names
}

/// Model filename validity (Python `re.fullmatch(r"[\w.\-]+", name)`).
fn valid_model_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// Resolve the `model=` filter option (Python `_vmaf_model_option`
/// parity): validated name → `path=vmaf-models/{name}` (relative: the
/// child runs with cwd at the exe dir), else the `vmaf_v0.6.1.json`
/// fallback chain, else the built-in `version=vmaf_v0.6.1`.
/// Returns `(model_opt, model_name)`; empty name means the builtin fallback.
pub fn resolve_model(name: &str, models_dir: &Path) -> (String, String) {
    if valid_model_name(name) && models_dir.join(name).is_file() {
        return (format!("path=vmaf-models/{name}"), name.to_owned());
    }
    let mut fallbacks = vec!["vmaf_v0.6.1.json".to_owned()];
    fallbacks.extend(list_models(models_dir));
    for fallback in fallbacks {
        if valid_model_name(&fallback) && models_dir.join(&fallback).is_file() {
            return (format!("path=vmaf-models/{fallback}"), fallback);
        }
    }
    ("version=vmaf_v0.6.1".to_owned(), String::new())
}

/// `ScaleThreshold` from the original's FFMetrics.conf: fraction of the
/// model height. A leg's model-scale applies only past this difference.
const SCALE_THRESHOLD: f64 = 0.1;

/// Per-leg model-fit scale for the ticked checkbox (original parity,
/// read off its FFMetrics.log): `Some((w, h))` aspect-preserving
/// fit-inside dims iff the leg's own height differs from the model height
/// by more than SCALE_THRESHOLD, else `None` (pass through). Width alone
/// never triggers it (1600x1080 vs 1920x1080 passes through), while
/// 1066x720 → `scale=1599:1080` and 1760x720 → `scale=1920:785` exactly
/// (rounded to nearest — truncation would give 1919 for the latter).
/// Legs are evaluated independently with no cross-leg equalization —
/// mismatched aspects fail in libvmaf, as in the original. Unknown dims
/// fall back to exact model dims (can't fit without them).
fn leg_model_scale(w: Option<i64>, h: Option<i64>, mw: u32, mh: u32) -> Option<(i64, i64)> {
    match (w, h) {
        (Some(w), Some(h)) if w > 0 && h > 0 => {
            if (h - mh as i64).abs() as f64 / mh as f64 > SCALE_THRESHOLD {
                let s = (mw as f64 / w as f64).min(mh as f64 / h as f64);
                Some(((w as f64 * s).round() as i64, (h as f64 * s).round() as i64))
            } else {
                None
            }
        }
        _ => Some((mw as i64, mh as i64)),
    }
}

/// Native resolution of a model (`*4k*`/`*2160*` → 2160p, else 1080p).
/// The `*2160*` arm covers the v1 (`vmaf_v1.0.16_*_2160`, incl. `_hfr_`)
/// and any future 4K files that don't carry a `4k` marker.
pub fn model_resolution(model_name: &str) -> (u32, u32) {
    let lower = model_name.to_lowercase();
    if lower.contains("4k") || lower.contains("2160") {
        (3840, 2160)
    } else {
        (1920, 1080)
    }
}

/// v1-generation model (`vmaf_v1.0.16_*`, incl. `_hfr_` variants): phone
/// is a separate `5d0h` file there, never the `enable_transform` flag.
pub fn is_v1_model(model_name: &str) -> bool {
    model_name.to_lowercase().contains("v1.0")
}

/// Top of the score range for a model: the v1 4K-consumer model
/// (`3d0h_2160`, incl. `_hfr_`) operates on [0, 110] (models_v1.md);
/// everything else on [0, 100].
pub fn model_score_max(model_name: &str) -> f64 {
    if model_name.to_lowercase().contains("3d0h_2160") {
        110.0
    } else {
        100.0
    }
}

/// libvmaf filtergraph (Python filter construction parity). Returns the
/// full `-filter_complex` string or a fatal per-file error (Phone on a
/// neg/4k/v1 model — the v1 branch has no Python equivalent, v1 postdates
/// the rev).
#[allow(clippy::too_many_arguments)]
pub fn build_filter(
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
    cfg: &VmafCfg,
    models_dir: &Path,
    logname: &str,
    n_threads: u32,
    scaler: ScaleMethod,
    ref_pixfmt: super::ffmpeg::RefPixFmt,
) -> Result<String, String> {
    let (mut model_opt, model_name) = resolve_model(&cfg.model, models_dir);
    if cfg.phone {
        // Phone transform exists only on standard v0.6.1 1080p models.
        // v1 phone is the separate `5d0h` file: the flag parses but is a
        // silent no-op there (verified live), so v1+phone is rejected
        // outright — including the `5d0h` file itself, for which the flag
        // would be meaningless.
        if is_v1_model(&model_name) {
            return Err(format!(
                "Model '{model_name}' has no Phone transform (v1 uses the separate 5d0h phone file)"
            ));
        }
        let lower = model_name.to_lowercase();
        if lower.contains("neg") || lower.contains("4k") {
            return Err(format!(
                "Model '{model_name}' has no Phone transform (use v0.6.1)"
            ));
        }
        // `\\:` (two backslashes) survives both filtergraph parse levels
        // (Python `f"{model_opt}\\\\:enable_transform=true"` parity): one
        // backslash would be consumed by the outer filtergraph, leaving a
        // bare `:` that libvmaf splits into an unknown top-level
        // `enable_transform` option ("Option not found").
        model_opt = format!("{model_opt}\\\\:enable_transform=true");
    }
    let window = trim_window(skip, clip_dur);
    let mut main_pre: Vec<String> = window.iter().map(|s| s.to_string()).collect();
    main_pre.push(NORM.to_owned());
    let mut ref_pre: Vec<String> = window;
    ref_pre.push(NORM.to_owned());
    if cfg.scale {
        let (mw, mh) = model_resolution(&model_name);
        if let Some((fw, fh)) = leg_model_scale(dist_info.width, dist_info.height, mw, mh) {
            main_pre.push(scale_filter(fw, fh, scaler));
        }
        if let Some((fw, fh)) = leg_model_scale(ref_info.width, ref_info.height, mw, mh) {
            ref_pre.push(scale_filter(fw, fh, scaler));
        }
    } else if (dist_info.width, dist_info.height) != (ref_info.width, ref_info.height)
        && let (Some(w), Some(h)) = (ref_info.width, ref_info.height)
    {
        main_pre.push(scale_filter(w, h, scaler));
    }
    // Colour-range legs differ: tag each side with its own range (conf
    // `scale,setrange,format` order); matching/unknown ranges emit nothing.
    let range_differs = ref_info.range_tag.as_deref() != dist_info.range_tag.as_deref();
    if range_differs && let Some(s) = setrange_segment(dist_info.range_tag.as_deref()) {
        main_pre.push(s);
    }
    if range_differs && let Some(s) = setrange_segment(ref_info.range_tag.as_deref()) {
        ref_pre.push(s);
    }
    // Pixel-format target converges both legs (unsupported selections,
    // e.g. RGB for VMAF, fall back to the map canonicalization).
    let (dist_fmt, ref_fmt) = super::ffmpeg::format_legs(
        ref_info.pix_fmt.as_deref(),
        dist_info.pix_fmt.as_deref(),
        ref_pixfmt,
        super::ffmpeg::MetricKind::Vmaf,
    );
    if let Some(s) = dist_fmt {
        main_pre.push(s);
    }
    if let Some(s) = ref_fmt {
        ref_pre.push(s);
    }
    // `model={model_opt}` unquoted so `\\:` passes both parse levels.
    Ok(format!(
        "[0:v]{}[main];[1:v]{}[ref];[main][ref]libvmaf=eof_action=endall:model={model_opt}:pool={}:n_subsample={}{}:log_path={logname}:log_fmt=json",
        main_pre.join(","),
        ref_pre.join(","),
        cfg.pooling.as_filter_str(),
        cfg.subsample.max(1),
        if n_threads > 0 {
            format!(":n_threads={n_threads}")
        } else {
            String::new()
        },
    ))
}

/// Tolerant float: JSON numbers or numeric strings (Python `float(...)`).
fn tolerant_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Parsed VMAF JSON log: per-frame scores + pooled summaries.
#[derive(Debug, Default)]
pub struct VmafLog {
    pub values: Vec<f64>,
    pub mean: Option<f64>,
    pub harmonic_mean: Option<f64>,
    /// CSV columns (libvmaf emission order, `vmaf` pinned last) + raw
    /// unclamped feature rows, aligned 1:1 with `values`.
    pub cols: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

/// Parse a libvmaf JSON log (Python `_parse_vmaf_log` parity):
/// `frames[].metrics.vmaf` per frame (sanitized + clamped 0–`max_score`),
/// `pooled_metrics.vmaf.{mean,harmonic_mean}` pooled (sanitized like the
/// series: `nan` drops to `None` so the frame mean wins downstream, `inf`
/// saturates at `max_score`).
/// `max_score` is `model_score_max` for the run's model (110 for v1 4K 3H).
pub fn parse_vmaf_log(text: &str, max_score: f64) -> Option<VmafLog> {
    let data: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut log = VmafLog::default();
    if let Some(frames) = data.get("frames").and_then(|f| f.as_array()) {
        for f in frames {
            let metrics = f.get("metrics").and_then(|m| m.as_object());
            let Some(metrics) = metrics else {
                continue;
            };
            // Column order from the first frame (libvmaf emission order
            // with `preserve_order`); `vmaf` pinned last like the samples.
            if log.cols.is_empty() {
                log.cols.extend(
                    metrics
                        .keys()
                        .filter(|k| *k != "vmaf")
                        .map(|k| k.to_owned()),
                );
                if metrics.contains_key("vmaf") {
                    log.cols.push("vmaf".to_owned());
                }
            }
            let v = metrics.get("vmaf").and_then(tolerant_f64);
            if let Some(v) = v {
                // `inf` (identical files) saturates at the top of the
                // model's range; `nan` sanitizes to 0 downstream.
                let v = if v.is_infinite() {
                    max_score
                } else {
                    crate::metrics::ffmpeg::sanitize_db(v).clamp(0.0, max_score)
                };
                if v.is_finite() {
                    log.values.push(v);
                    // Raw features (no clamp — samples show values like
                    // 1.007); only non-finite junk sanitizes. A frame
                    // missing a column can never align, so it drops the
                    // whole row (values + detail stay 1:1).
                    let row: Option<Vec<f64>> = log
                        .cols
                        .iter()
                        .map(|k| {
                            metrics
                                .get(k)
                                .and_then(tolerant_f64)
                                .map(crate::metrics::ffmpeg::sanitize_db)
                        })
                        .collect();
                    if let Some(row) = row {
                        log.rows.push(row);
                    } else {
                        log.values.pop();
                    }
                }
            }
        }
    }
    if let Some(pooled) = data.get("pooled_metrics").and_then(|p| p.get("vmaf")) {
        let sanitize_pooled =
            |v: Option<f64>| v.filter(|v| !v.is_nan()).map(|v| v.clamp(0.0, max_score));
        log.mean = sanitize_pooled(pooled.get("mean").and_then(tolerant_f64));
        log.harmonic_mean = sanitize_pooled(pooled.get("harmonic_mean").and_then(tolerant_f64));
    }
    Some(log)
}

/// Unique temp-log reservation next to the exe (Python `mkstemp` parity):
/// `vmaf_log_<pid>_<n>.json`, created empty so concurrent runs can't clash.
fn reserve_log_path(home: &Path) -> PathBuf {
    static CTR: AtomicU64 = AtomicU64::new(0);
    loop {
        let path = home.join(format!(
            "vmaf_log_{}_{}.json",
            std::process::id(),
            CTR.fetch_add(1, Ordering::SeqCst)
        ));
        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .is_ok()
        {
            return path;
        }
    }
}

/// Delete-on-drop guard: the temp log vanishes on success, error, and abort
/// (Python `finally: unlink` parity).
struct DeleteOnDrop(PathBuf);

impl Drop for DeleteOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Substrings identifying a fatal ffmpeg error line (missing-`libvmaf`
/// builds surface as raw ffmpeg output; Python reverse-scans for these).
fn is_fatal_line(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("option '")
        || l.contains("not found")
        || l.contains("error")
        || l.contains("could not")
}

/// Blocking VMAF run; call off the UI thread. Progress comes from stderr
/// `frame=` lines only (no `stats_file` feed exists for libvmaf).
/// ponytail: no wait-timeout (worker-only pattern, like `run_metric`).
pub fn run_vmaf(job: &RunInputs, cfg: &VmafCfg, on_progress: &(dyn Fn(u64) + Sync)) -> RunOutcome {
    use crate::metrics::ffmpeg::RunInputs;
    let RunInputs {
        exe,
        ref_path,
        dist_path,
        ref_info,
        dist_info,
        skip,
        clip_dur,
        scaler,
        fps_mode,
        ref_pixfmt,
        abort,
        child_slot,
        ..
    } = *job;
    let fail = |msg: String| RunOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
        detail: FrameDetail::None,
    };
    let home = vmaf_home();
    let models_dir = home.join("vmaf-models");
    let log_path = reserve_log_path(&home);
    let _cleanup = DeleteOnDrop(log_path.clone());
    // Basename only: the child runs with cwd at the exe dir, so both the
    // relative `path=vmaf-models/…` and `log_path=` resolve (Python parity).
    let logname = log_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vmaf_log.json".to_owned());
    let filt = match build_filter(
        ref_info,
        dist_info,
        skip,
        clip_dur,
        cfg,
        &models_dir,
        &logname,
        cfg.n_threads,
        scaler,
        ref_pixfmt,
    ) {
        Ok(f) => f,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "VMAF {e}");
            return fail(e);
        }
    };
    let mut args = vec![
        "-hide_banner".to_owned(),
        "-nostdin".to_owned(),
        // FFMetrics.conf parity (see `build_args`): sparse-header probe window.
        "-probesize".to_owned(),
        "50M".to_owned(),
    ];
    args.extend(rate_args(fps_mode, ref_info, dist_info, true));
    args.push("-i".to_owned());
    args.push(dist_path.to_owned());
    args.extend(rate_args(fps_mode, ref_info, dist_info, false));
    args.push("-i".to_owned());
    args.push(ref_path.to_owned());
    args.push("-filter_complex".to_owned());
    args.push(filt);
    args.extend(["-f".to_owned(), "null".to_owned(), "-".to_owned()]);
    // `info`: see `run_metric` — the repro command belongs in issue reports.
    log::info!(target: "rfmetrics::metric", "run: \"{}\" {}", exe.display(), args.join(" "));
    // Stdout carries no VMAF scores; drain it so the pipe never blocks.
    let pumped = match pump_process(
        exe,
        &args,
        Some(&home),
        abort,
        child_slot,
        |_| {},
        on_progress,
    ) {
        Ok(p) => p,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "VMAF {e}");
            return fail(e);
        }
    };
    if pumped.aborted {
        log::info!(target: "rfmetrics::metric", "VMAF \"{dist_path}\" aborted after {:.1}s", pumped.exec_s);
        return RunOutcome {
            values: Vec::new(),
            avg: None,
            exec_s: pumped.exec_s,
            error: Some("aborted".to_owned()),
            detail: FrameDetail::None,
        };
    }
    let log_text = std::fs::read_to_string(&log_path).unwrap_or_default();
    if pumped.code != Some(0) || log_text.trim().is_empty() {
        // Surface the meaningful ffmpeg line (e.g. missing-libvmaf build).
        let msg = pumped
            .stderr
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty() && is_fatal_line(l))
            .or_else(|| last_err_line(&pumped.stderr))
            .unwrap_or(&format!("FFmpeg failed with code {:?}", pumped.code))
            .to_owned();
        let dump = crate::metrics::ffmpeg::stderr_tail(
            &pumped.stderr,
            crate::metrics::ffmpeg::STDERR_TAIL_LINES,
        );
        log::warn!(target: "rfmetrics::metric", "VMAF no data for \"{dist_path}\" (exit {:?}, {:.1}s): {msg}\n{dump}", pumped.code, pumped.exec_s);
        return RunOutcome {
            values: Vec::new(),
            avg: None,
            exec_s: pumped.exec_s,
            error: Some(msg),
            detail: FrameDetail::None,
        };
    }
    // Re-resolve for the score range (one extra tiny dir read per run;
    // cheaper than threading the name out of `build_filter`).
    let max_score = model_score_max(&resolve_model(&cfg.model, &models_dir).1);
    let (values, avg, detail) = match parse_vmaf_log(&log_text, max_score) {
        Some(log) if !log.values.is_empty() => {
            let pooled = match cfg.pooling {
                Pooling::HarmonicMean => log.harmonic_mean,
                Pooling::Mean => log.mean,
            };
            let detail = FrameDetail::Vmaf {
                cols: log.cols,
                rows: log.rows,
            };
            (log.values, pooled, detail)
        }
        _ => (Vec::new(), None, FrameDetail::None),
    };
    if values.is_empty() {
        return no_data_outcome(
            "VMAF",
            dist_path,
            pumped.code,
            pumped.exec_s,
            &pumped.stderr,
            "no VMAF data",
        );
    }
    let show = avg.unwrap_or_else(|| values.iter().sum::<f64>() / values.len() as f64);
    log::info!(
        target: "rfmetrics::metric",
        "VMAF \"{dist_path}\" → {show:.4} ({} frames, exit {:?}, {:.1}s)",
        values.len(),
        pumped.code,
        pumped.exec_s,
    );
    RunOutcome {
        values,
        avg,
        exec_s: pumped.exec_s,
        error: None,
        detail,
    }
}
#[cfg(test)]
#[path = "../tests/test_metrics_vmaf.rs"]
mod tests;
