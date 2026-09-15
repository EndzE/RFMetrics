//! VMAF metric (Python `_compute_vmaf_series` parity): libvmaf filtergraph,
//! exe-dir JSON log file, tolerant log parsing. Rides the shared worker
//! plumbing (`RunInputs`, abort slot, progress channel); only the filter,
//! the temp log, and the parsers are VMAF-specific.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::metrics::ffmpeg::{NORM, RunInputs, RunOutcome, pump_process, rate_args, trim_window};
use crate::probe::MediaInfo;

/// Pooling selector (UI strings `Mean`/`Harmonic Mean` map here at Start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    #[default]
    Mean,
    HarmonicMean,
}

impl Pooling {
    fn as_filter_str(self) -> &'static str {
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
}

/// Home directory for models + temp logs: next to the exe (Python
/// `app_dir` parity), falling back like the logger when unresolvable.
pub(crate) fn vmaf_home() -> PathBuf {
    if let Some(dir) = crate::binaries::exe_dir() {
        return dir;
    }
    std::env::current_dir()
        .ok()
        .unwrap_or_else(std::env::temp_dir)
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

/// Native resolution of a model (`*4k*` → 2160p, else 1080p).
pub fn model_resolution(model_name: &str) -> (u32, u32) {
    if model_name.to_lowercase().contains("4k") {
        (3840, 2160)
    } else {
        (1920, 1080)
    }
}

/// libvmaf filtergraph (Python filter construction parity). Returns the
/// full `-filter_complex` string or a fatal per-file error (Phone on a
/// neg/4k model, mirroring the exact Python message).
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
) -> Result<String, String> {
    let (mut model_opt, model_name) = resolve_model(&cfg.model, models_dir);
    if cfg.phone {
        // Phone transform exists only on standard v0.6.1 1080p models.
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
            main_pre.push(format!("scale={fw}:{fh}:flags=bicubic"));
        }
        if let Some((fw, fh)) = leg_model_scale(ref_info.width, ref_info.height, mw, mh) {
            ref_pre.push(format!("scale={fw}:{fh}:flags=bicubic"));
        }
    } else if (dist_info.width, dist_info.height) != (ref_info.width, ref_info.height)
        && let (Some(w), Some(h)) = (ref_info.width, ref_info.height)
    {
        main_pre.push(format!("scale={w}:{h}"));
    }
    if ref_info.pix_fmt.is_some() && dist_info.pix_fmt != ref_info.pix_fmt {
        main_pre.push(format!("format={}", ref_info.pix_fmt.as_deref().unwrap()));
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
}

/// Parse a libvmaf JSON log (Python `_parse_vmaf_log` parity):
/// `frames[].metrics.vmaf` per frame (sanitized + clamped 0–100),
/// `pooled_metrics.vmaf.{mean,harmonic_mean}` pooled.
pub fn parse_vmaf_log(text: &str) -> Option<VmafLog> {
    let data: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut log = VmafLog::default();
    if let Some(frames) = data.get("frames").and_then(|f| f.as_array()) {
        for f in frames {
            let v = f
                .get("metrics")
                .and_then(|m| m.get("vmaf"))
                .and_then(tolerant_f64);
            if let Some(v) = v {
                let v = crate::metrics::ffmpeg::sanitize_db(v).clamp(0.0, 100.0);
                if v.is_finite() {
                    log.values.push(v);
                }
            }
        }
    }
    if let Some(pooled) = data.get("pooled_metrics").and_then(|p| p.get("vmaf")) {
        log.mean = pooled.get("mean").and_then(tolerant_f64);
        log.harmonic_mean = pooled.get("harmonic_mean").and_then(tolerant_f64);
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
        abort,
        child_slot,
        ..
    } = *job;
    let fail = |msg: String| RunOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
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
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);
    let filt = match build_filter(
        ref_info,
        dist_info,
        skip,
        clip_dur,
        cfg,
        &models_dir,
        &logname,
        n_threads,
    ) {
        Ok(f) => f,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "VMAF {e}");
            return fail(e);
        }
    };
    let mut args = vec!["-hide_banner".to_owned(), "-nostdin".to_owned()];
    args.extend(rate_args(ref_info, dist_info));
    args.push("-i".to_owned());
    args.push(dist_path.to_owned());
    args.extend(rate_args(ref_info, dist_info));
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
            .or_else(|| {
                pumped
                    .stderr
                    .lines()
                    .map(str::trim)
                    .rfind(|l| !l.is_empty())
            })
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
        };
    }
    let (values, avg) = match parse_vmaf_log(&log_text) {
        Some(log) if !log.values.is_empty() => {
            let pooled = match cfg.pooling {
                Pooling::HarmonicMean => log.harmonic_mean,
                Pooling::Mean => log.mean,
            };
            (log.values, pooled)
        }
        _ => (Vec::new(), None),
    };
    if values.is_empty() {
        let tail = pumped
            .stderr
            .lines()
            .map(str::trim)
            .rfind(|l| !l.is_empty());
        let msg = tail.unwrap_or("no VMAF data").to_owned();
        let dump = crate::metrics::ffmpeg::stderr_tail(
            &pumped.stderr,
            crate::metrics::ffmpeg::STDERR_TAIL_LINES,
        );
        log::warn!(target: "rfmetrics::metric", "VMAF no data for \"{dist_path}\" (exit {:?}, {:.1}s): {msg}\n{dump}", pumped.code, pumped.exec_s);
        return RunOutcome {
            values,
            avg: None,
            exec_s: pumped.exec_s,
            error: Some(msg),
        };
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_info() -> MediaInfo {
        MediaInfo {
            width: Some(1920),
            height: Some(1080),
            fps: Some(25.0),
            pix_fmt: Some("yuv420p".to_owned()),
            ..Default::default()
        }
    }

    fn cfg() -> VmafCfg {
        VmafCfg {
            model: "vmaf_v0.6.1.json".to_owned(),
            phone: false,
            scale: false,
            pooling: Pooling::Mean,
            subsample: 1,
        }
    }

    fn models_dir(names: &[&str]) -> (TempfileGuard, PathBuf) {
        // Creates <temp>/vmaf-models/<names>; guard deletes on drop.
        let base = std::env::temp_dir().join(format!(
            "rfmetrics-vmaf-test-{}-{}",
            std::process::id(),
            CTR.fetch_add(1, Ordering::SeqCst)
        ));
        let dir = base.join("vmaf-models");
        std::fs::create_dir_all(&dir).unwrap();
        for n in names {
            std::fs::write(dir.join(n), "{}").unwrap();
        }
        (TempfileGuard(base), dir)
    }

    struct TempfileGuard(PathBuf);
    impl Drop for TempfileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    static CTR: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn models_list_sorted_or_sentinel() {
        let (_g, dir) = models_dir(&["vmaf_4k_v0.6.1.json", "vmaf_v0.6.1.json"]);
        assert_eq!(
            list_models(&dir),
            vec![
                "vmaf_4k_v0.6.1.json".to_owned(),
                "vmaf_v0.6.1.json".to_owned()
            ]
        );
        let (_g2, empty) = models_dir(&[]);
        // Only non-JSON files: still the sentinel.
        std::fs::write(empty.join("readme.txt"), "x").unwrap();
        assert_eq!(list_models(&empty), vec!["No models found".to_owned()]);
    }

    #[test]
    fn resolver_prefers_valid_then_fallbacks() {
        let (_g, dir) = models_dir(&["custom.json", "vmaf_v0.6.1.json"]);
        assert_eq!(
            resolve_model("custom.json", &dir),
            (
                "path=vmaf-models/custom.json".to_owned(),
                "custom.json".to_owned()
            )
        );
        // Unsafe names and missing files fall back to vmaf_v0.6.1.json.
        assert_eq!(resolve_model("../../evil.json", &dir).1, "vmaf_v0.6.1.json");
        assert_eq!(resolve_model("gone.json", &dir).1, "vmaf_v0.6.1.json");
        // Nothing on disk: builtin version fallback, empty model name.
        let (_g2, empty) = models_dir(&[]);
        assert_eq!(
            resolve_model("vmaf_v0.6.1.json", &empty),
            ("version=vmaf_v0.6.1".to_owned(), String::new())
        );
    }

    #[test]
    fn model_resolution_picks_4k() {
        assert_eq!(model_resolution("vmaf_4k_v0.6.1.json"), (3840, 2160));
        assert_eq!(model_resolution("VMAF_4K_v0.6.1neg.json"), (3840, 2160));
        assert_eq!(model_resolution("vmaf_v0.6.1.json"), (1920, 1080));
        assert_eq!(model_resolution("vmaf_v0.6.1neg.json"), (1920, 1080));
        assert_eq!(model_resolution(""), (1920, 1080));
    }

    #[test]
    fn filter_baseline_matches_python_shape() {
        let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
        let f = build_filter(
            &ref_info(),
            &ref_info(),
            None,
            None,
            &cfg(),
            &dir,
            "vmaf_log_1.json",
            8,
        )
        .unwrap();
        assert_eq!(
            f,
            "[0:v]settb=AVTB,setpts=PTS-STARTPTS[main];\
             [1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];\
             [main][ref]libvmaf=eof_action=endall:model=path=vmaf-models/vmaf_v0.6.1.json:pool=mean:n_subsample=1:n_threads=8:log_path=vmaf_log_1.json:log_fmt=json"
        );
    }

    #[test]
    fn filter_phone_scale_pool_subsample() {
        let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
        let mut c = cfg();
        c.phone = true;
        c.scale = true;
        c.pooling = Pooling::HarmonicMean;
        c.subsample = 5;
        // Above model resolution: both legs scale down to model native.
        let ref4k = MediaInfo {
            width: Some(3840),
            height: Some(2160),
            ..ref_info()
        };
        let f = build_filter(&ref4k, &ref4k, None, None, &c, &dir, "v.json", 0).unwrap();
        // Two backslashes before the colon (Python parity): the outer
        // filtergraph consumes one, libvmaf's model parser the other.
        // A single backslash would split `enable_transform` into an
        // unknown top-level libvmaf option ("Option not found").
        assert!(f.contains("model=path=vmaf-models/vmaf_v0.6.1.json\\\\:enable_transform=true"));
        assert!(
            f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[main]")
        );
        assert!(
            f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[ref]")
        );
        assert!(f.contains(":pool=harmonic_mean:"));
        assert!(f.contains(":n_subsample=5:"));
        assert!(!f.contains("n_threads"));
        // Subsample clamps to >= 1.
        c.subsample = 0;
        let f = build_filter(&ref_info(), &ref_info(), None, None, &c, &dir, "v.json", 4).unwrap();
        assert!(f.contains(":n_subsample=1:n_threads=4:"));
    }

    #[test]
    fn scale_follows_model_height_threshold() {
        // Original `ScaleThreshold: 0.1`: both legs scale to model native
        // iff ref height differs from model height by >10%.
        // 1600x1080 is the live parity case (no scale filter emitted).
        let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
        let mut c = cfg();
        c.scale = true;
        let at = |w: i64, h: i64| MediaInfo {
            width: Some(w),
            height: Some(h),
            ..ref_info()
        };
        // Same height, narrower width: no scale (width alone never triggers).
        let small = at(1600, 1080);
        let f = build_filter(&small, &small, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(!f.contains("scale="), "must not upscale: {f}");
        assert!(f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS[main]"));
        assert!(f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref]"));
        // Exact model resolution is a no-op too, never a scaler pass.
        let f = build_filter(&ref_info(), &ref_info(), None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(!f.contains("scale="), "must not rescale in place: {f}");
        // 7.4% under: within threshold, no scale.
        let near = at(1920, 1000);
        let f = build_filter(&near, &near, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(!f.contains("scale="), "within threshold: {f}");
        // 33% under: upscale both legs to model native.
        let tiny = at(1280, 720);
        let f = build_filter(&tiny, &tiny, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(
            f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[main]"),
            "{f}"
        );
        assert!(
            f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic[ref]"),
            "{f}"
        );
        // Per-leg aspect fit-inside, read off the original's log verbatim:
        // 1066x720 → 1599:1080, 1760x720 → 1920:785 (width 8.3% under
        // alone would not trigger; the 33% height gap does).
        let a = at(1066, 720);
        let f = build_filter(&a, &a, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(f.contains("scale=1599:1080:flags=bicubic"), "{f}");
        let b = at(1760, 720);
        let f = build_filter(&b, &b, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(f.contains("scale=1920:785:flags=bicubic"), "{f}");
        // Mixed aspects scale independently with no equalization (the
        // original emits this shape too; libvmaf then fails the pair).
        let wide = at(1600, 1080);
        let f = build_filter(&wide, &a, None, None, &c, &dir, "v.json", 0).unwrap();
        assert!(
            f.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1599:1080:flags=bicubic[main]"),
            "{f}"
        );
        assert!(
            f.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref]"),
            "{f}"
        );
    }

    #[test]
    fn phone_guard_rejects_neg_and_4k() {
        let (_g, dir) = models_dir(&["vmaf_v0.6.1neg.json", "vmaf_4k_v0.6.1.json"]);
        let mut c = cfg();
        c.phone = true;
        c.model = "vmaf_v0.6.1neg.json".to_owned();
        assert_eq!(
            build_filter(&ref_info(), &ref_info(), None, None, &c, &dir, "v.json", 0),
            Err("Model 'vmaf_v0.6.1neg.json' has no Phone transform (use v0.6.1)".to_owned())
        );
        c.model = "vmaf_4k_v0.6.1.json".to_owned();
        assert!(build_filter(&ref_info(), &ref_info(), None, None, &c, &dir, "v.json", 0).is_err());
    }

    #[test]
    fn filter_trims_and_scales_dist_only() {
        let (_g, dir) = models_dir(&["vmaf_v0.6.1.json"]);
        let mut dist = ref_info();
        dist.width = Some(1280);
        dist.height = Some(720);
        let f = build_filter(
            &ref_info(),
            &dist,
            Some(5.0),
            Some(10.0),
            &cfg(),
            &dir,
            "v.json",
            0,
        )
        .unwrap();
        assert!(f.contains(
            "[0:v]trim=start=5:end=15,settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080[main]"
        ));
        assert!(f.contains("[1:v]trim=start=5:end=15,settb=AVTB,setpts=PTS-STARTPTS[ref]"));
    }

    #[test]
    fn log_parses_frames_and_pooled() {
        let text = r#"{
            "frames": [
                {"metrics": {"vmaf": 90.5}},
                {"metrics": {"vmaf": "91.25"}},
                {"metrics": {"psnr": 40.0}},
                {"nope": 1}
            ],
            "pooled_metrics": {"vmaf": {"mean": 91.0, "harmonic_mean": 90.8}}
        }"#;
        let log = parse_vmaf_log(text).unwrap();
        assert_eq!(log.values, vec![90.5, 91.25]);
        assert_eq!(log.mean, Some(91.0));
        assert_eq!(log.harmonic_mean, Some(90.8));
    }

    #[test]
    fn log_sanitizes_and_clamps() {
        let text = r#"{"frames": [
            {"metrics": {"vmaf": "inf"}},
            {"metrics": {"vmaf": "nan"}},
            {"metrics": {"vmaf": 150.0}},
            {"metrics": {"vmaf": -5.0}}
        ]}"#;
        let log = parse_vmaf_log(text).unwrap();
        assert_eq!(log.values, vec![100.0, 0.0, 100.0, 0.0]);
    }

    #[test]
    fn log_rejects_garbage() {
        assert!(parse_vmaf_log("not json").is_none());
        assert!(parse_vmaf_log("{}").unwrap().values.is_empty());
    }
}
