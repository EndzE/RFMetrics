use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::probe::MediaInfo;

/// Filter-based metric sharing the `_compute_series` engine: identical
/// trim/scale/format legs, worker, and stats. The filter name, the input
/// order (XPSNR alone inverts to `[ref][main]`), and the output parsers
/// differ per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    Psnr,
    Ssim,
    Xpsnr,
    Vmaf,
}

impl MetricKind {
    /// All ffmpeg-backed metrics, in Python `METRICS` run order.
    pub const ALL: [MetricKind; 4] = [
        MetricKind::Psnr,
        MetricKind::Ssim,
        MetricKind::Vmaf,
        MetricKind::Xpsnr,
    ];

    /// Display name for cells, tooltips, and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Psnr => "PSNR",
            Self::Ssim => "SSIM",
            Self::Xpsnr => "XPSNR",
            Self::Vmaf => "VMAF",
        }
    }

    /// ffmpeg filter name in the `<order><filter>=…` segment.
    fn filter(self) -> &'static str {
        match self {
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
            Self::Xpsnr => "xpsnr",
            // VMAF never uses `filtergraph` (own libvmaf builder in vmaf.rs).
            Self::Vmaf => "libvmaf",
        }
    }

    /// Input order segment: XPSNR alone inverts to `[ref][main]`.
    fn order(self) -> &'static str {
        match self {
            Self::Xpsnr => "[ref][main]",
            _ => "[main][ref]",
        }
    }
}

fn frame_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"psnr_avg:(\S+)").unwrap())
}

fn ssim_frame_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"All:(\S+)").unwrap())
}

fn progress_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\bn:\s*(\d+)").unwrap())
}

fn summary_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"average:(\S+)").unwrap())
}

fn ssim_summary_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"All:(\S+)").unwrap())
}

/// Matches `_XPSNR_NUM`: plain/scientific floats plus `inf`/`nan`.
const XPSNR_NUM: &str = r"[-+]?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|nan)";

fn xpsnr_frame_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(&format!(
            r"(?i)XPSNR\s+y:\s*({XPSNR_NUM})\s+XPSNR\s+u:\s*({XPSNR_NUM})\s+XPSNR\s+v:\s*({XPSNR_NUM})"
        ))
        .unwrap()
    })
}

fn xpsnr_summary_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(&format!(
            r"(?i)XPSNR\s+y:\s*({XPSNR_NUM})\s+u:\s*({XPSNR_NUM})\s+v:\s*({XPSNR_NUM})"
        ))
        .unwrap()
    })
}

fn err_progress_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"frame=\s*(\d+)").unwrap())
}

/// Max `frame=` progress number in one stderr segment (`frame=  12`).
/// A segment can hold several updates, so all matches are scanned —
/// first-match-only scanning stuck VMAF at Frame: 0/1 on `\r`-joined
/// progress blobs.
fn max_frame_in(text: &str) -> Option<u64> {
    err_progress_re()
        .captures_iter(text)
        .filter_map(|c| c.get(1)?.as_str().parse::<u64>().ok())
        .max()
}

/// Cap for the stderr dump appended to failure log lines (original
/// `ERROR:` + stderr parity without progress-meter flooding).
pub(crate) const STDERR_TAIL_LINES: usize = 30;

/// Last `n` non-empty stderr lines, chronological, for the log file.
pub(crate) fn stderr_tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Per-frame value from a `stats_file=-` stdout line.
/// PSNR (`n:1 ... psnr_avg:34.12 ...`) clamps to 0–100; SSIM
/// (`n:1 ... All:0.985210 ...`, Y/U/V ignored) clamps to 0–1 and strips a
/// trailing `)` (Python parity; `inf` clamps, `nan`/garbage skipped).
pub fn parse_frame_line(line: &str, kind: MetricKind) -> Option<f64> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    let (raw, hi) = match kind {
        MetricKind::Psnr => (frame_re().captures(line)?.get(1)?.as_str(), 100.0),
        MetricKind::Ssim => (
            ssim_frame_re()
                .captures(line)?
                .get(1)?
                .as_str()
                .trim_end_matches(')'),
            1.0,
        ),
        // XPSNR combines three planes with weights: use parse_xpsnr_frame_line.
        // VMAF parses its JSON log instead: use vmaf::parse_vmaf_log.
        MetricKind::Xpsnr | MetricKind::Vmaf => return None,
    };
    let v: f64 = raw.parse().ok()?;
    if v.is_nan() {
        return None;
    }
    Some(v.clamp(0.0, hi))
}

/// Frame counter from a stats line (`n:12 ...`) for progress.
pub fn parse_progress(line: &str) -> Option<u64> {
    progress_re().captures(line)?.get(1)?.as_str().parse().ok()
}

/// Pooled average from ffmpeg's stderr summary, scanned bottom-up.
/// PSNR (`... PSNR ... average:33.98 ...`) matches `average:` on any line
/// mentioning PSNR; SSIM (`SSIM ... All:0.99 ...`) matches `All:` on lines
/// mentioning SSIM that are not per-frame `n:` lines (Python parity,
/// unclamped — `inf` stays `inf`).
pub fn parse_summary(text: &str, kind: MetricKind) -> Option<f64> {
    for line in text.lines().rev() {
        match kind {
            MetricKind::Psnr => {
                if line.contains("PSNR")
                    && let Some(c) = summary_re().captures(line)
                    && let Ok(v) = c.get(1).unwrap().as_str().parse::<f64>()
                {
                    return Some(v);
                }
            }
            MetricKind::Ssim => {
                let s = line.trim();
                if s.contains("SSIM")
                    && !s.starts_with("n:")
                    && let Some(c) = ssim_summary_re().captures(s)
                    && let Ok(v) = c.get(1).unwrap().as_str().parse::<f64>()
                {
                    return Some(v);
                }
            }
            // XPSNR combines three planes with weights: use parse_xpsnr_summary.
            // VMAF parses its JSON log instead: use vmaf::parse_vmaf_log.
            MetricKind::Xpsnr | MetricKind::Vmaf => {}
        }
    }
    None
}

/// Timestamp reset shared by every filtergraph leg.
pub(crate) const NORM: &str = "settb=AVTB,setpts=PTS-STARTPTS";

/// Trim window shared by every filtergraph (Python `if skip or clip_dur:`
/// parity — 0.0 is falsy, so a zero skip/clip disables trim instead of
/// producing an empty `trim=start=0:end=0`).
pub(crate) fn trim_window(skip: Option<f64>, clip_dur: Option<f64>) -> Vec<String> {
    let mut window = Vec::new();
    if skip.is_some_and(|v| v != 0.0) || clip_dur.is_some_and(|v| v != 0.0) {
        let start = skip.unwrap_or(0.0);
        let end = clip_dur
            .filter(|&d| d != 0.0)
            .map(|d| format!(":end={}", start + d))
            .unwrap_or_default();
        window.push(format!("trim=start={start}{end}"));
    }
    window
}

/// `-r` input flags shared by every ffmpeg invocation (Python parity: the
/// same rate feeds both inputs).
pub(crate) fn rate_args(ref_info: &MediaInfo, dist_info: &MediaInfo) -> Vec<String> {
    match ref_info.fps.or(dist_info.fps) {
        Some(fps) => vec!["-r".to_owned(), crate::probe::format_fps(fps)],
        None => Vec::new(),
    }
}

/// Python `_compute_series` filtergraph for filter-based metrics. Inputs are
/// inverted vs. the arg order: `-i dist` is `[0:v]`/main, `-i ref` is
/// `[1:v]`/ref, and the filter is `[main][ref]<filter>=…` (same order for
/// PSNR and SSIM; only XPSNR inverts). The distorted leg is scaled/converted
/// up to the reference when they differ; the reference leg only gets trim.
pub fn filtergraph(
    kind: MetricKind,
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
) -> String {
    // Python `if skip or clip_dur:` — 0.0 is falsy, so a zero skip/clip
    // disables trim instead of producing an empty `trim=start=0:end=0`.
    let window = trim_window(skip, clip_dur);
    let mut pre: Vec<String> = window.iter().map(|s| s.to_string()).collect();
    pre.push(NORM.to_owned());
    if (dist_info.width, dist_info.height) != (ref_info.width, ref_info.height)
        && let (Some(w), Some(h)) = (ref_info.width, ref_info.height)
    {
        pre.push(format!("scale={w}:{h}"));
    }
    if ref_info.pix_fmt.is_some() && dist_info.pix_fmt != ref_info.pix_fmt {
        pre.push(format!("format={}", ref_info.pix_fmt.as_deref().unwrap()));
    }
    let mut ref_leg: Vec<String> = window;
    ref_leg.push(NORM.to_owned());
    format!(
        "[0:v]{}[main];[1:v]{}[ref];{}{}=eof_action=endall:stats_file=-",
        pre.join(","),
        ref_leg.join(","),
        kind.order(),
        kind.filter(),
    )
}

/// Plane weights as `(y, u, v)` sample counts (Python
/// `_xpsnr_plane_weights` parity): exact counts when the reference
/// dimensions are known (`w*h`, `ceil(w/2)*ceil(h/2)`), 4:1:1 / 2:1:1
/// ratios without them, `1.0` default (incl. unknown formats).
/// Format comes from the reference first, the distorted as fallback.
pub fn xpsnr_weights(
    pix_fmt: Option<&str>,
    width: Option<i64>,
    height: Option<i64>,
) -> (f64, f64, f64) {
    let fmt = pix_fmt.unwrap_or_default().to_lowercase();
    if let (Some(w), Some(h)) = (width, height)
        && w > 0
        && h > 0
    {
        if fmt.contains("420") || fmt.contains("nv12") || fmt.contains("nv21") {
            let uv = (((w + 1) / 2) * ((h + 1) / 2)) as f64;
            if uv > 0.0 {
                return ((w * h) as f64, uv, uv);
            }
        }
        if fmt.contains("422") {
            let uv = (((w + 1) / 2) * h) as f64;
            if uv > 0.0 {
                return ((w * h) as f64, uv, uv);
            }
        }
        if fmt.contains("444") {
            return (1.0, 1.0, 1.0);
        }
    }
    if fmt.contains("420") || fmt.contains("nv12") || fmt.contains("nv21") {
        return (4.0, 1.0, 1.0);
    }
    if fmt.contains("422") {
        return (2.0, 1.0, 1.0);
    }
    (1.0, 1.0, 1.0)
}

/// Per-plane sanitize (Python `_sanitize_db` parity): identical files
/// yield `inf`/`nan` from ffmpeg; those become 100.0/0.0. Otherwise
/// unclamped — negative dB stays negative.
pub fn sanitize_db(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value.is_infinite() {
        100.0
    } else {
        value
    }
}

fn weighted_avg(y: f64, u: f64, v: f64, weights: (f64, f64, f64)) -> f64 {
    let (wy, wu, wv) = weights;
    let total = wy + wu + wv;
    let total = if total > 0.0 { total } else { 1.0 };
    (wy * y + wu * u + wv * v) / total
}

/// Per-frame XPSNR value from a stats line
/// (`n:1 XPSNR y: 42.1 XPSNR u: 45.0 XPSNR v: 44.2`), plane-weighted.
/// Frame regex requires the `XPSNR` prefix on every plane.
pub fn parse_xpsnr_frame_line(line: &str, weights: (f64, f64, f64)) -> Option<f64> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    let c = xpsnr_frame_re().captures(line)?;
    let y = sanitize_db(c.get(1)?.as_str().parse().ok()?);
    let u = sanitize_db(c.get(2)?.as_str().parse().ok()?);
    let v = sanitize_db(c.get(3)?.as_str().parse().ok()?);
    Some(weighted_avg(y, u, v, weights))
}

/// Pooled XPSNR from ffmpeg's stderr summary
/// (`... XPSNR y: 43.1 u: 44.0 v: 43.8 ...`), scanned bottom-up.
/// Summary form has no `XPSNR` prefix on u/v; the scan requires
/// case-sensitive `XPSNR` in the line and skips `n:` lines.
pub fn parse_xpsnr_summary(text: &str, weights: (f64, f64, f64)) -> Option<f64> {
    for line in text.lines().rev() {
        if !line.contains("XPSNR") || line.trim_start().starts_with("n:") {
            continue;
        }
        if let Some(c) = xpsnr_summary_re().captures(line) {
            // Unparseable numbers skip the line (Python `except
            // ValueError: continue`), not the whole scan.
            let (Ok(y), Ok(u), Ok(v)) = (
                c.get(1).unwrap().as_str().parse::<f64>(),
                c.get(2).unwrap().as_str().parse::<f64>(),
                c.get(3).unwrap().as_str().parse::<f64>(),
            ) else {
                continue;
            };
            // Python also sanitizes the summary planes (inf/nan safety).
            return Some(weighted_avg(
                sanitize_db(y),
                sanitize_db(u),
                sanitize_db(v),
                weights,
            ));
        }
    }
    None
}

/// Full ffmpeg argv (minus the exe) for a filter-metric run.
pub fn build_args(
    kind: MetricKind,
    ref_path: &str,
    dist_path: &str,
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
) -> Vec<String> {
    let mut args = vec!["-hide_banner".to_owned(), "-nostdin".to_owned()];
    args.extend(rate_args(ref_info, dist_info));
    args.push("-i".to_owned());
    args.push(dist_path.to_owned());
    args.extend(rate_args(ref_info, dist_info));
    args.push("-i".to_owned());
    args.push(ref_path.to_owned());
    args.push("-filter_complex".to_owned());
    args.push(filtergraph(kind, ref_info, dist_info, skip, clip_dur));
    args.extend(["-f".to_owned(), "null".to_owned(), "-".to_owned()]);
    args
}

/// Everything a filter-metric run needs; bundled so `run_metric` stays lean.
pub struct RunInputs<'a> {
    pub kind: MetricKind,
    pub exe: &'a Path,
    pub ref_path: &'a str,
    pub dist_path: &'a str,
    pub ref_info: &'a MediaInfo,
    pub dist_info: &'a MediaInfo,
    pub skip: Option<f64>,
    pub clip_dur: Option<f64>,
    /// Set by Stop; the run reports "aborted" and drops partial values.
    pub abort: &'a AtomicBool,
    /// Holds the live child so Stop can kill it; `None` when idle/reaped.
    pub child_slot: &'a Mutex<Option<Child>>,
}

pub struct RunOutcome {
    pub values: Vec<f64>,
    pub avg: Option<f64>,
    pub exec_s: f64,
    pub error: Option<String>,
}

/// Result of pumping one ffmpeg child to completion.
pub(crate) struct Pumped {
    pub code: Option<i32>,
    pub stderr: String,
    pub exec_s: f64,
    pub aborted: bool,
}

/// Shared process skeleton for every ffmpeg metric (Python parity: piped
/// stdout/stderr, live `frame=` + `n:` progress, abort-slot publish/kill/
/// reap). The caller owns parsing: stdout lines stream into `on_stdout_line`
/// (VMAF ignores them but the pipe must still drain), progress from both
/// feeds is deduplicated through a shared high-water mark.
/// ponytail: plain wait(), no wait-timeout dep — a stuck ffmpeg only stalls
/// this worker thread, never the UI.
pub(crate) fn pump_process(
    exe: &Path,
    args: &[String],
    cwd: Option<&Path>,
    abort: &AtomicBool,
    child_slot: &Mutex<Option<Child>>,
    mut on_stdout_line: impl FnMut(&str) + Send,
    on_progress: &(dyn Fn(u64) + Sync),
) -> Result<Pumped, String> {
    let start = std::time::Instant::now();
    let mut cmd = Command::new(exe);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return Err(format!("spawn failed: {e}")),
    };
    // Take the pipes before publishing: the child moves into the slot next.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    // Publish the child so Stop can kill it; reaped below once the pipes
    // drain (or already reaped by Stop, which sets `abort` first).
    if child_slot
        .lock()
        .map(|mut slot| slot.replace(child))
        .is_err()
    {
        return Err("internal lock error".to_owned());
    }
    // Readers: live progress feed + full stderr capture.
    // Scoped threads so borrowed callbacks need not be 'static.
    let max_sent = Arc::new(AtomicU64::new(0));
    let err_text = std::thread::scope(|s| {
        let err_max = Arc::clone(&max_sent);
        let err_handle = stderr.map(|err| {
            s.spawn(move || {
                use std::io::Read;
                let mut lines = Vec::new();
                let mut seg: Vec<u8> = Vec::new();
                // ffmpeg draws its `frame=` meter with `\r` (no `\n` until
                // exit); Python reads stderr in universal-newlines mode, so
                // `\r` is a line boundary there too. Splitting on `\n` only
                // yields zero complete lines mid-run — VMAF progress stuck
                // at Frame: 0 with no stdout `n:` feed to fall back on.
                let flush = |seg: &mut Vec<u8>, lines: &mut Vec<String>| {
                    if seg.is_empty() {
                        return;
                    }
                    let text = String::from_utf8_lossy(seg).into_owned();
                    if let Some(f) = max_frame_in(&text)
                        && f > err_max.fetch_max(f, Ordering::SeqCst)
                    {
                        on_progress(f);
                    }
                    lines.push(text);
                    seg.clear();
                };
                let mut reader = BufReader::new(err);
                let mut tmp = [0u8; 4096];
                loop {
                    match reader.read(&mut tmp) {
                        Ok(0) => break,
                        Ok(n) => {
                            for &b in &tmp[..n] {
                                if b == b'\r' || b == b'\n' {
                                    flush(&mut seg, &mut lines);
                                } else {
                                    seg.push(b);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                flush(&mut seg, &mut lines);
                lines.join("\n")
            })
        });
        if let Some(out) = stdout {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(f) = parse_progress(&line)
                    && f > max_sent.fetch_max(f, Ordering::SeqCst)
                {
                    on_progress(f);
                }
                on_stdout_line(&line);
            }
        }
        err_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default()
    });
    // Reap: Stop takes + kills + waits ahead of us when aborting (it sets
    // the flag first, so a missing child always means "aborted").
    let status = child_slot
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
        .map(|mut c| c.wait());
    let aborted = abort.load(Ordering::SeqCst);
    Ok(Pumped {
        code: status.and_then(|s| s.ok()).and_then(|s| s.code()),
        stderr: err_text,
        exec_s: start.elapsed().as_secs_f64(),
        aborted,
    })
}

/// Blocking filter-metric run; call off the UI thread. The UI keeps the max
/// per row from the progress feed.
pub fn run_metric(job: &RunInputs<'_>, on_progress: &(dyn Fn(u64) + Sync)) -> RunOutcome {
    let RunInputs {
        kind,
        exe,
        ref_path,
        dist_path,
        ref_info,
        dist_info,
        skip,
        clip_dur,
        abort,
        child_slot,
    } = *job;
    let name = kind.name();
    // XPSNR weights come from the reference (pix_fmt, then dims), with the
    // distorted pix_fmt as fallback — computed once per run, not per line.
    let weights = xpsnr_weights(
        ref_info.pix_fmt.as_deref().or(dist_info.pix_fmt.as_deref()),
        ref_info.width,
        ref_info.height,
    );
    let fail = |msg: String| RunOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
    };
    let args = build_args(
        kind, ref_path, dist_path, ref_info, dist_info, skip, clip_dur,
    );
    // `info`: the exact repro command is the core artifact of an issue
    // report (FFMetrics.log parity) — one line per metric job.
    log::info!(target: "rfmetrics::metric", "run: \"{}\" {}", exe.display(), args.join(" "));
    let mut values = Vec::new();
    let pumped = match pump_process(
        exe,
        &args,
        None,
        abort,
        child_slot,
        |line| {
            if let Some(v) = match kind {
                MetricKind::Xpsnr => parse_xpsnr_frame_line(line, weights),
                kind => parse_frame_line(line, kind),
            } {
                values.push(v);
            }
        },
        on_progress,
    ) {
        Ok(p) => p,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} {e}");
            return fail(e);
        }
    };
    let (code, err_text, exec_s) = (pumped.code, pumped.stderr, pumped.exec_s);
    if pumped.aborted {
        log::info!(target: "rfmetrics::metric", "{name} \"{dist_path}\" aborted after {exec_s:.1}s");
        return RunOutcome {
            values: Vec::new(),
            avg: None,
            exec_s,
            error: Some("aborted".to_owned()),
        };
    }
    if values.is_empty() {
        let tail = err_text.lines().map(str::trim).rfind(|l| !l.is_empty());
        let msg = tail.unwrap_or(&format!("no {name} data")).to_owned();
        let dump = stderr_tail(&err_text, STDERR_TAIL_LINES);
        log::warn!(target: "rfmetrics::metric", "{name} no data for \"{dist_path}\" (exit {code:?}, {exec_s:.1}s): {msg}\n{dump}");
        return RunOutcome {
            values,
            avg: None,
            exec_s,
            error: Some(msg),
        };
    }
    let avg = match kind {
        MetricKind::Xpsnr => parse_xpsnr_summary(&err_text, weights),
        kind => parse_summary(&err_text, kind),
    };
    let show = avg.unwrap_or_else(|| values.iter().sum::<f64>() / values.len() as f64);
    log::info!(
        target: "rfmetrics::metric",
        "{name} \"{dist_path}\" → {show:.4} ({} frames, exit {code:?}, {exec_s:.1}s)",
        values.len(),
    );
    RunOutcome {
        values,
        avg,
        exec_s,
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

    #[test]
    fn stderr_progress_scans_all_matches_per_segment() {
        // ffmpeg draws its progress meter with `\r`: one read can hold many
        // `frame=` updates. First-match-only scanning reported just the
        // first and VMAF (no stdout `n:` feed) stuck at Frame: 0.
        let blob = "frame= 1 fps=100 q=-0.0 size=N/A time=00:00:01 bitrate=N/A speed=4x\rframe= 27 fps=110 q=-0.0 size=N/A time=00:00:02 bitrate=N/A speed=4x";
        assert_eq!(max_frame_in(blob), Some(27));
        assert_eq!(max_frame_in("frame=  3 fps=25"), Some(3));
        assert_eq!(max_frame_in("no progress here"), None);
    }

    #[test]
    fn stderr_tail_keeps_last_non_empty_lines() {
        assert_eq!(stderr_tail("", 30), "");
        assert_eq!(stderr_tail("a\n\nb\n", 30), "a\nb");
        let many: String = (1..=40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tail = stderr_tail(&many, 30);
        assert_eq!(tail.lines().count(), 30);
        assert!(tail.starts_with("line 11\n"));
        assert!(tail.ends_with("line 40"));
    }

    #[test]
    fn frame_lines() {
        use super::MetricKind::Psnr;
        assert_eq!(
            parse_frame_line("n:1 mse_avg:12.3 psnr_avg:34.1234 mse_y:1.0", Psnr),
            Some(34.1234)
        );
        assert_eq!(parse_frame_line("frame= 12 fps=25", Psnr), None);
        assert_eq!(parse_frame_line("n:2 no avg here", Psnr), None);
        assert_eq!(parse_frame_line("n:3 psnr_avg:inf", Psnr), Some(100.0));
        assert_eq!(parse_frame_line("n:3 psnr_avg:-inf", Psnr), Some(0.0));
        assert_eq!(parse_frame_line("n:3 psnr_avg:nan", Psnr), None);
        assert_eq!(parse_frame_line("n:3 psnr_avg:garbage", Psnr), None);
    }

    #[test]
    fn ssim_frame_lines() {
        use super::MetricKind::Ssim;
        assert_eq!(
            parse_frame_line(
                "n:1 Y:0.991234 U:0.987654 V:0.976543 All:0.985210 (parsed)",
                Ssim
            ),
            Some(0.98521)
        );
        // Trailing `)` stripped (Python `rstrip(")")` parity).
        assert_eq!(
            parse_frame_line("n:2 Y:1 U:1 V:1 All:1.000000)", Ssim),
            Some(1.0)
        );
        assert_eq!(parse_frame_line("frame= 12 fps=25", Ssim), None);
        assert_eq!(parse_frame_line("n:3 no All here", Ssim), None);
        assert_eq!(parse_frame_line("n:3 All:inf", Ssim), Some(1.0));
        assert_eq!(parse_frame_line("n:3 All:-inf", Ssim), Some(0.0));
        assert_eq!(parse_frame_line("n:3 All:nan", Ssim), None);
        assert_eq!(parse_frame_line("n:3 All:garbage", Ssim), None);
        // PSNR field ignored under SSIM and vice versa.
        assert_eq!(parse_frame_line("n:4 psnr_avg:34.1", Ssim), None);
        assert_eq!(
            parse_frame_line("n:4 Y:0.9 U:0.9 V:0.9 All:0.9", super::MetricKind::Psnr),
            None
        );
    }

    #[test]
    fn summary_scans_bottom_up() {
        use super::MetricKind::Psnr;
        let err = "[Parsed_psnr_0] PSNR y:1 u:2 v:3 average:30.0 min:1 max:2\n\
                   [Parsed_psnr_0] PSNR y:1 u:2 v:3 average:33.98 min:1 max:2\n";
        assert_eq!(parse_summary(err, Psnr), Some(33.98));
        assert_eq!(parse_summary("nothing here", Psnr), None);
    }

    #[test]
    fn ssim_summary_scans_bottom_up() {
        use super::MetricKind::Ssim;
        let err = "[Parsed_ssim_0 @ 0x123] SSIM Y:0.97 U:0.98 V:0.99 All:0.975 (dB 16.02)\n\
                   [Parsed_ssim_0 @ 0x123] SSIM Y:0.98 U:0.99 V:0.99 All:0.986 (dB 18.55)\n";
        assert_eq!(parse_summary(err, Ssim), Some(0.986));
        // Per-frame `n:` lines never count as summaries.
        assert_eq!(
            parse_summary("n:7 Y:0.9 U:0.9 V:0.9 All:0.9 SSIM", Ssim),
            None
        );
        assert_eq!(parse_summary("nothing here", Ssim), None);
    }

    #[test]
    fn graph_scales_dist_to_ref() {
        use super::MetricKind::Psnr;
        let mut dist = ref_info();
        dist.width = Some(1280);
        dist.height = Some(720);
        dist.pix_fmt = Some("yuv444p".to_owned());
        let g = filtergraph(Psnr, &ref_info(), &dist, None, None);
        assert_eq!(
            g,
            "[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080,format=yuv420p[main];\
             [1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];\
             [main][ref]psnr=eof_action=endall:stats_file=-"
        );
    }

    #[test]
    fn graph_matching_streams_have_no_scale() {
        use super::MetricKind::Psnr;
        let g = filtergraph(Psnr, &ref_info(), &ref_info(), Some(5.0), Some(10.0));
        assert!(g.contains("[0:v]trim=start=5:end=15,"));
        assert!(g.contains("[1:v]trim=start=5:end=15,"));
        assert!(!g.contains("scale="));
        assert!(!g.contains("format="));
    }

    #[test]
    fn graph_zero_skip_and_clip_disable_trim() {
        use super::MetricKind::Psnr;
        // Python `if skip or clip_dur:` — 0.0 is falsy, so zero values
        // measure the full video instead of an empty clip.
        let plain = filtergraph(Psnr, &ref_info(), &ref_info(), None, None);
        assert_eq!(
            filtergraph(Psnr, &ref_info(), &ref_info(), Some(0.0), Some(0.0)),
            plain
        );
        assert!(!plain.contains("trim="));
        // Mixed: zero side drops out, nonzero side applies.
        let g = filtergraph(Psnr, &ref_info(), &ref_info(), Some(0.0), Some(10.0));
        assert!(g.contains("trim=start=0:end=10"));
        let g = filtergraph(Psnr, &ref_info(), &ref_info(), Some(5.0), Some(0.0));
        assert!(g.contains("[0:v]trim=start=5,"));
        assert!(!g.contains(":end="));
    }

    #[test]
    fn ssim_graph_differs_only_by_filter_name() {
        use super::MetricKind::{Psnr, Ssim};
        let psnr = filtergraph(Psnr, &ref_info(), &ref_info(), Some(5.0), Some(10.0));
        let ssim = filtergraph(Ssim, &ref_info(), &ref_info(), Some(5.0), Some(10.0));
        // Same legs, same order — only the filter segment differs.
        assert_eq!(
            ssim,
            psnr.replace(
                "[main][ref]psnr=eof_action=endall:stats_file=-",
                "[main][ref]ssim=eof_action=endall:stats_file=-"
            )
        );
        let a = build_args(
            Ssim,
            "ref.mp4",
            "dist.mp4",
            &ref_info(),
            &ref_info(),
            None,
            None,
        );
        assert!(a.iter().any(|x| x.contains("[main][ref]ssim=")));
    }

    #[test]
    fn args_order_is_dist_then_ref() {
        use super::MetricKind::Psnr;
        let a = build_args(
            Psnr,
            "ref.mp4",
            "dist.mp4",
            &ref_info(),
            &ref_info(),
            None,
            None,
        );
        let i1 = a.iter().position(|x| x == "dist.mp4").unwrap();
        let i2 = a.iter().position(|x| x == "ref.mp4").unwrap();
        assert!(i1 < i2);
        assert!(a.contains(&"-r".to_owned()) && a.contains(&"25".to_owned()));
        assert_eq!(a.last().unwrap(), "-");
    }

    #[test]
    fn xpsnr_weights_match_python_branches() {
        // Exact sample counts for 1920x1080 4:2:0 (not just 4:1:1).
        assert_eq!(
            xpsnr_weights(Some("yuv420p"), Some(1920), Some(1080)),
            (2073600.0, 518400.0, 518400.0)
        );
        // Odd dims use ceil halves: 5x5 -> y=25, uv=3x3=9.
        assert_eq!(
            xpsnr_weights(Some("yuv420p"), Some(5), Some(5)),
            (25.0, 9.0, 9.0)
        );
        // nv12/nv21 aliases of 420; 422 halves width only.
        assert_eq!(
            xpsnr_weights(Some("nv12"), Some(1920), Some(1080)),
            (2073600.0, 518400.0, 518400.0)
        );
        assert_eq!(
            xpsnr_weights(Some("yuv422p"), Some(1920), Some(1080)),
            (2073600.0, 1036800.0, 1036800.0)
        );
        // 444 and unknown formats weigh planes equally.
        assert_eq!(
            xpsnr_weights(Some("yuv444p"), Some(1920), Some(1080)),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            xpsnr_weights(Some("rgb24"), Some(1920), Some(1080)),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(xpsnr_weights(None, Some(1920), Some(1080)), (1.0, 1.0, 1.0));
        // Unknown dims fall back to ratios.
        assert_eq!(xpsnr_weights(Some("yuv420p"), None, None), (4.0, 1.0, 1.0));
        assert_eq!(xpsnr_weights(Some("NV21"), None, None), (4.0, 1.0, 1.0));
        assert_eq!(
            xpsnr_weights(Some("yuv422p10le"), None, None),
            (2.0, 1.0, 1.0)
        );
        assert_eq!(xpsnr_weights(None, None, None), (1.0, 1.0, 1.0));
    }

    #[test]
    fn sanitize_db_pins_inf_and_nan() {
        assert_eq!(sanitize_db(f64::INFINITY), 100.0);
        assert_eq!(sanitize_db(f64::NEG_INFINITY), 100.0);
        assert_eq!(sanitize_db(f64::NAN), 0.0);
        assert_eq!(sanitize_db(42.5), 42.5);
        assert_eq!(sanitize_db(-3.0), -3.0); // unclamped otherwise
    }

    #[test]
    fn xpsnr_frame_lines() {
        let w = (4.0, 1.0, 1.0);
        assert_eq!(
            parse_xpsnr_frame_line("n:1 XPSNR y: 42.0 XPSNR u: 45.0 XPSNR v: 44.0", w),
            Some((4.0 * 42.0 + 45.0 + 44.0) / 6.0)
        );
        // Non-n: lines and missing planes are skipped.
        assert_eq!(parse_xpsnr_frame_line("frame= 12 fps=25", w), None);
        assert_eq!(parse_xpsnr_frame_line("n:2 XPSNR y: 42.0", w), None);
        // Summary form (no XPSNR prefix on u/v) is not a frame line.
        assert_eq!(
            parse_xpsnr_frame_line("n:3 XPSNR y: 42.0 u: 45.0 v: 44.0", w),
            None
        );
        // Identical files: inf -> 100, nan -> 0 per plane.
        assert_eq!(
            parse_xpsnr_frame_line("n:4 XPSNR y: inf XPSNR u: inf XPSNR v: inf", w),
            Some(100.0)
        );
        assert_eq!(
            parse_xpsnr_frame_line("n:5 XPSNR y: nan XPSNR u: 45.0 XPSNR v: 44.0", w),
            Some((45.0 + 44.0) / 6.0)
        );
        // Case-insensitive like the Python regex.
        assert!(
            parse_xpsnr_frame_line("n:6 xpsnr y: 40.0 xpsnr u: 40.0 xpsnr v: 40.0", w).is_some()
        );
    }

    #[test]
    fn xpsnr_summary_scans_bottom_up() {
        let w = (4.0, 1.0, 1.0);
        let err = "[Parsed_xpsnr_0] XPSNR y: 40.0 u: 41.0 v: 42.0\n\
                   [Parsed_xpsnr_0] XPSNR y: 43.0 u: 44.0 v: 45.0\n";
        assert_eq!(
            parse_xpsnr_summary(err, w),
            Some((4.0 * 43.0 + 44.0 + 45.0) / 6.0)
        );
        // Per-frame lines never count, even mentioning XPSNR planes.
        assert_eq!(
            parse_xpsnr_summary("n:7 XPSNR y: 1.0 XPSNR u: 1.0 XPSNR v: 1.0", w),
            None
        );
        assert_eq!(parse_xpsnr_summary("nothing here", w), None);
    }

    #[test]
    fn xpsnr_graph_inverts_input_order() {
        use super::MetricKind::{Psnr, Xpsnr};
        let psnr = filtergraph(Psnr, &ref_info(), &ref_info(), Some(5.0), Some(10.0));
        let xpsnr = filtergraph(Xpsnr, &ref_info(), &ref_info(), Some(5.0), Some(10.0));
        // Same legs — only the order segment and filter name differ.
        assert_eq!(
            xpsnr,
            psnr.replace(
                "[main][ref]psnr=eof_action=endall:stats_file=-",
                "[ref][main]xpsnr=eof_action=endall:stats_file=-"
            )
        );
    }
}
