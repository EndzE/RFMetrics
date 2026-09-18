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
    Ssim2,
    But,
    Cvvdp,
}

impl MetricKind {
    /// All metrics, in Python `METRICS` run order (ffmpeg-backed first,
    /// then FFVship).
    pub const ALL: [MetricKind; 7] = [
        MetricKind::Psnr,
        MetricKind::Ssim,
        MetricKind::Vmaf,
        MetricKind::Xpsnr,
        MetricKind::Ssim2,
        MetricKind::But,
        MetricKind::Cvvdp,
    ];

    /// Display name for cells, tooltips, and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Psnr => "PSNR",
            Self::Ssim => "SSIM",
            Self::Xpsnr => "XPSNR",
            Self::Vmaf => "VMAF",
            Self::Ssim2 => "SSIM2",
            Self::But => "BUTTER",
            Self::Cvvdp => "CVVDP",
        }
    }

    /// FFVship-backed metrics ride `run_ffvship`, not the ffmpeg engine.
    pub fn is_ffvship(self) -> bool {
        matches!(self, Self::Ssim2 | Self::But | Self::Cvvdp)
    }

    /// Metrics whose runner streams per-frame live values (the plot
    /// follows them mid-run). VMAF has no live feed — scores exist only
    /// at `Done` — so its tab never auto-fits without a Done-triggered
    /// poke. A future no-live-feed metric must join the `Vmaf` arm.
    pub fn streams_live_values(self) -> bool {
        !matches!(self, Self::Vmaf)
    }

    /// The FFVship sub-kind (metric name, arity, pooling); `None` for
    /// ffmpeg-backed metrics.
    pub fn ffvship_kind(self) -> Option<crate::metrics::ffvship::FfvshipKind> {
        use crate::metrics::ffvship::FfvshipKind;
        match self {
            Self::Ssim2 => Some(FfvshipKind::Ssimulacra2),
            Self::But => Some(FfvshipKind::Butteraugli),
            Self::Cvvdp => Some(FfvshipKind::Cvvdp),
            Self::Psnr | Self::Ssim | Self::Xpsnr | Self::Vmaf => None,
        }
    }

    /// ffmpeg filter name in the `<order><filter>=…` segment, also used
    /// for the startup `-filters` capability probe in `binaries.rs`.
    /// Unreachable for FFVship metrics (no filtergraph); the worker
    /// dispatches on `is_ffvship()` first.
    pub(crate) fn filter(self) -> &'static str {
        match self {
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
            Self::Xpsnr => "xpsnr",
            // VMAF never uses `filtergraph` (own libvmaf builder in vmaf.rs).
            Self::Vmaf => "libvmaf",
            Self::Ssim2 | Self::But | Self::Cvvdp => {
                unreachable!("FFVship metrics have no ffmpeg filter")
            }
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

/// Live-curve throttle: a Series batch ships when it holds this many
/// values or this much time passed since the last batch — whichever
/// first. Plot repaints ride the existing measuring-repaint driver.
/// Live-curve throttle (shared with `run_ffvship`): a Series batch ships
/// when it holds this many values or this much time passed since the last
/// batch — whichever first. Plot repaints ride the existing
/// measuring-repaint driver.
pub(crate) const SERIES_BATCH: usize = 64;
pub(crate) const SERIES_THROTTLE: std::time::Duration = std::time::Duration::from_millis(150);

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
        // FFVship parses live stdout instead: use ffvship::parse_live_rows.
        MetricKind::Xpsnr
        | MetricKind::Vmaf
        | MetricKind::Ssim2
        | MetricKind::But
        | MetricKind::Cvvdp => {
            return None;
        }
    };
    let v: f64 = raw.parse().ok()?;
    if v.is_nan() {
        return None;
    }
    Some(v.clamp(0.0, hi))
}

/// Raw number after a `key:` token on a stats line (`psnr_y:43.93`,
/// `Y:0.98`). `None` when the key is absent, unparseable, or `nan` —
/// a nan plane poisons the row, mirroring the pooled-value skip, so
/// detail rows stay 1:1 with `values`. Order-independent (unlike the
/// single-regex frame parsers above).
fn stat_num(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|tok| {
        tok.strip_prefix(key)?
            .strip_prefix(':')?
            .trim_end_matches(')')
            .parse::<f64>()
            .ok()
            .filter(|v| !v.is_nan())
    })
}

/// PSNR planes per frame: [avg, y, u, v], sanitized like the series
/// (`inf` → 100; a surviving line never holds `nan`).
pub fn parse_psnr_planes(line: &str) -> Option<[f64; 4]> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    Some([
        sanitize_db(stat_num(line, "psnr_avg")?),
        sanitize_db(stat_num(line, "psnr_y")?),
        sanitize_db(stat_num(line, "psnr_u")?),
        sanitize_db(stat_num(line, "psnr_v")?),
    ])
}

/// SSIM planes per frame: [Y, U, V, All], clamped like the series.
pub fn parse_ssim_planes(line: &str) -> Option<[f64; 4]> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    let plane = |key: &str| stat_num(line, key).map(|v| v.clamp(0.0, 1.0));
    Some([plane("Y")?, plane("U")?, plane("V")?, plane("All")?])
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
            // FFVship pools live stdout instead: see ffvship::run_ffvship.
            MetricKind::Xpsnr
            | MetricKind::Vmaf
            | MetricKind::Ssim2
            | MetricKind::But
            | MetricKind::Cvvdp => {}
        }
    }
    None
}

/// Global scaling method for every `scale=` the app emits (metric
/// upscaling legs + VMAF model-fit legs). Flag names verified against
/// `ffmpeg -h full` (`sws_flags`); the sws default is bicubic, so an
/// explicit `flags=bicubic` renders pixel-identical to the old flagless
/// graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleMethod {
    #[default]
    Bicubic,
    FfmpegDefault,
    Neighbor,
    Gauss,
    Bilinear,
    Lanczos,
    Spline,
    Sinc,
}

impl ScaleMethod {
    /// Combo order: the default first, then the requested list order.
    pub const ALL: [ScaleMethod; 8] = [
        ScaleMethod::Bicubic,
        ScaleMethod::FfmpegDefault,
        ScaleMethod::Neighbor,
        ScaleMethod::Gauss,
        ScaleMethod::Bilinear,
        ScaleMethod::Lanczos,
        ScaleMethod::Spline,
        ScaleMethod::Sinc,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Bicubic => "Bicubic",
            Self::FfmpegDefault => "FFmpeg default",
            Self::Neighbor => "Nearest neighbor",
            Self::Gauss => "Gauss",
            Self::Bilinear => "Bilinear",
            Self::Lanczos => "Lanczos",
            Self::Spline => "Spline",
            Self::Sinc => "Sinc",
        }
    }

    /// libswscale flag, or `None` for "FFmpeg default" (omit `:flags=` —
    /// today's exact strings).
    pub fn flag(self) -> Option<&'static str> {
        match self {
            Self::Bicubic => Some("bicubic"),
            Self::FfmpegDefault => None,
            Self::Neighbor => Some("neighbor"),
            Self::Gauss => Some("gauss"),
            Self::Bilinear => Some("bilinear"),
            Self::Lanczos => Some("lanczos"),
            Self::Spline => Some("spline"),
            Self::Sinc => Some("sinc"),
        }
    }

    /// State-file validation (unknown labels keep the live default).
    pub fn from_label(s: &str) -> Option<ScaleMethod> {
        Self::ALL.into_iter().find(|m| m.label() == s)
    }
}

/// `scale=w:h` with the global method (`:flags=` omitted for FFmpeg default).
pub fn scale_filter(w: i64, h: i64, method: ScaleMethod) -> String {
    match method.flag() {
        Some(f) => format!("scale={w}:{h}:flags={f}"),
        None => format!("scale={w}:{h}"),
    }
}

/// Timestamp reset shared by every filtergraph leg.
pub(crate) const NORM: &str = "settb=AVTB,setpts=PTS-STARTPTS";

/// `setrange` segment for one leg (FFMetrics.conf `{{main-setrange}}` /
/// `{{ref-setrange}}` parity): tags the leg's own probed colour range so a
/// downstream `format` interprets levels correctly. Only known ffmpeg range
/// tokens pass through; anything else (or unknown) is `None` = no segment.
/// Callers emit it only when the legs' tags differ — matching-range content
/// shows no setrange segment in the original's FFMetrics.log.
pub(crate) fn setrange_segment(range_tag: Option<&str>) -> Option<String> {
    match range_tag {
        Some("tv" | "pc" | "limited" | "full") => {
            Some(format!("setrange=range={}", range_tag.unwrap()))
        }
        _ => None,
    }
}

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
    scaler: ScaleMethod,
) -> String {
    // Python `if skip or clip_dur:` — 0.0 is falsy, so a zero skip/clip
    // disables trim instead of producing an empty `trim=start=0:end=0`.
    let window = trim_window(skip, clip_dur);
    let mut pre: Vec<String> = window.iter().map(|s| s.to_string()).collect();
    pre.push(NORM.to_owned());
    if (dist_info.width, dist_info.height) != (ref_info.width, ref_info.height)
        && let (Some(w), Some(h)) = (ref_info.width, ref_info.height)
    {
        pre.push(scale_filter(w, h, scaler));
    }
    // Colour-range legs differ: tag each side with its own range (conf
    // `scale,setrange,format` order); matching/unknown ranges emit nothing.
    let range_differs = ref_info.range_tag.as_deref() != dist_info.range_tag.as_deref();
    if range_differs && let Some(s) = setrange_segment(dist_info.range_tag.as_deref()) {
        pre.push(s);
    }
    if ref_info.pix_fmt.is_some() && dist_info.pix_fmt != ref_info.pix_fmt {
        pre.push(format!("format={}", ref_info.pix_fmt.as_deref().unwrap()));
    }
    let mut ref_leg: Vec<String> = window;
    ref_leg.push(NORM.to_owned());
    if range_differs && let Some(s) = setrange_segment(ref_info.range_tag.as_deref()) {
        ref_leg.push(s);
    }
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

/// Raw XPSNR planes from a stats line
/// (`n:1 XPSNR y: 42.1 XPSNR u: 45.0 XPSNR v: 44.2`), sanitized.
/// Frame regex requires the `XPSNR` prefix on every plane.
pub fn parse_xpsnr_planes(line: &str) -> Option<[f64; 3]> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    let c = xpsnr_frame_re().captures(line)?;
    Some([
        sanitize_db(c.get(1)?.as_str().parse().ok()?),
        sanitize_db(c.get(2)?.as_str().parse().ok()?),
        sanitize_db(c.get(3)?.as_str().parse().ok()?),
    ])
}

/// Per-frame XPSNR value from a stats line, plane-weighted.
pub fn parse_xpsnr_frame_line(line: &str, weights: (f64, f64, f64)) -> Option<f64> {
    let [y, u, v] = parse_xpsnr_planes(line)?;
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
#[allow(clippy::too_many_arguments)]
pub fn build_args(
    kind: MetricKind,
    ref_path: &str,
    dist_path: &str,
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
    scaler: ScaleMethod,
) -> Vec<String> {
    let mut args = vec![
        "-hide_banner".to_owned(),
        "-nostdin".to_owned(),
        // FFMetrics.conf `Metric.Template` parity: larger probe window for
        // sparse headers (ts/m2ts/mxf); the original's log shows it live.
        "-probesize".to_owned(),
        "50M".to_owned(),
    ];
    args.extend(rate_args(ref_info, dist_info));
    args.push("-i".to_owned());
    args.push(dist_path.to_owned());
    args.extend(rate_args(ref_info, dist_info));
    args.push("-i".to_owned());
    args.push(ref_path.to_owned());
    args.push("-filter_complex".to_owned());
    args.push(filtergraph(
        kind, ref_info, dist_info, skip, clip_dur, scaler,
    ));
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
    /// Global scaling method; selects the `:flags=` on every `scale=`
    /// this run emits (FFVship jobs ignore it — no ffmpeg stage).
    pub scaler: ScaleMethod,
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
    /// Per-frame detail rows for CSV export, aligned 1:1 with `values`
    /// (row `i` describes `values[i]`). The pooled series stays the only
    /// source for stats/plots; this never enters the UI data path.
    pub detail: FrameDetail,
}

/// Per-frame detail rows for CSV export (raw planes/features; pooled
/// `values` stay the single source for stats/plots).
#[derive(Debug, Clone, Default)]
pub enum FrameDetail {
    /// Errors, and metrics without a CSV shape (FFVship kinds).
    #[default]
    None,
    /// PSNR planes per frame: [avg, y, u, v] (sanitized like the series).
    Psnr(Vec<[f64; 4]>),
    /// SSIM planes per frame: [Y, U, V, All] (clamped like the series).
    Ssim(Vec<[f64; 4]>),
    /// XPSNR raw planes per frame: [y, u, v] (sanitized, unweighted).
    Xpsnr(Vec<[f64; 3]>),
    /// VMAF: feature columns in libvmaf emission order with `vmaf` pinned
    /// last; rows hold raw (unclamped) feature values.
    Vmaf {
        cols: Vec<String>,
        rows: Vec<Vec<f64>>,
    },
    /// FFVship multi-score rows in live-output order (`idx`-sorted by the
    /// parser); headers come from `FfvshipKind::csv_cols`.
    Scores {
        cols: Vec<String>,
        rows: Vec<Vec<f64>>,
    },
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
pub fn run_metric(
    job: &RunInputs<'_>,
    on_progress: &(dyn Fn(u64) + Sync),
    on_series: &(dyn Fn(&[f64]) + Sync),
) -> RunOutcome {
    let RunInputs {
        kind,
        exe,
        ref_path,
        dist_path,
        ref_info,
        dist_info,
        skip,
        clip_dur,
        scaler,
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
        detail: FrameDetail::None,
    };
    let args = build_args(
        kind, ref_path, dist_path, ref_info, dist_info, skip, clip_dur, scaler,
    );
    // `info`: the exact repro command is the core artifact of an issue
    // report (FFMetrics.log parity) — one line per metric job.
    log::info!(target: "rfmetrics::metric", "run: \"{}\" {}", exe.display(), args.join(" "));
    let mut values = Vec::new();
    // CSV detail rows, pushed in lockstep with `values` (row `i`
    // describes `values[i]`); the writer zips defensively anyway.
    let mut detail = match kind {
        MetricKind::Psnr => FrameDetail::Psnr(Vec::new()),
        MetricKind::Ssim => FrameDetail::Ssim(Vec::new()),
        MetricKind::Xpsnr => FrameDetail::Xpsnr(Vec::new()),
        // VMAF/FFVship fill detail in their own runners.
        _ => FrameDetail::None,
    };
    // Live-curve tap: per-frame values stream to the plot in throttled
    // batches (the strict full series still lands on `Done`). No final
    // flush: `Done` arrives right behind and replaces the buffer.
    let mut pending: Vec<f64> = Vec::new();
    let mut last_emit = std::time::Instant::now();
    let emit = |pending: &mut Vec<f64>, last_emit: &mut std::time::Instant| {
        if !pending.is_empty()
            && (pending.len() >= SERIES_BATCH || last_emit.elapsed() >= SERIES_THROTTLE)
        {
            on_series(pending);
            pending.clear();
            *last_emit = std::time::Instant::now();
        }
    };
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
                pending.push(v);
                emit(&mut pending, &mut last_emit);
                // A pooled value implies parseable planes (same line), so
                // a missed row here is only theoretical; the writer zips.
                match (&mut detail, kind) {
                    (FrameDetail::Psnr(rows), MetricKind::Psnr) => {
                        if let Some(p) = parse_psnr_planes(line) {
                            rows.push(p);
                        }
                    }
                    (FrameDetail::Ssim(rows), MetricKind::Ssim) => {
                        if let Some(p) = parse_ssim_planes(line) {
                            rows.push(p);
                        }
                    }
                    (FrameDetail::Xpsnr(rows), MetricKind::Xpsnr) => {
                        if let Some(p) = parse_xpsnr_planes(line) {
                            rows.push(p);
                        }
                    }
                    _ => {}
                }
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
            detail: FrameDetail::None,
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
            detail: FrameDetail::None,
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
        detail,
    }
}
#[cfg(test)]
#[path = "../tests/test_metrics_ffmpeg.rs"]
mod tests;
