use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::probe::MediaInfo;

/// Filter-based metric sharing the `_compute_series` engine: identical
/// `[main][ref]` order, trim/scale/format legs, worker, and stats.
/// Only the filter name and the output parsers differ per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    Psnr,
    Ssim,
}

impl MetricKind {
    /// Display name for cells, tooltips, and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Psnr => "PSNR",
            Self::Ssim => "SSIM",
        }
    }

    /// ffmpeg filter name in the `[main][ref]<filter>=…` segment.
    fn filter(self) -> &'static str {
        match self {
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
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

fn err_progress_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"frame=\s*(\d+)").unwrap())
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
        }
    }
    None
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
    const NORM: &str = "settb=AVTB,setpts=PTS-STARTPTS";
    // Python `if skip or clip_dur:` — 0.0 is falsy, so a zero skip/clip
    // disables trim instead of producing an empty `trim=start=0:end=0`.
    let mut window = Vec::new();
    if skip.is_some_and(|v| v != 0.0) || clip_dur.is_some_and(|v| v != 0.0) {
        let start = skip.unwrap_or(0.0);
        let end = clip_dur
            .filter(|&d| d != 0.0)
            .map(|d| format!(":end={}", start + d))
            .unwrap_or_default();
        window.push(format!("trim=start={start}{end}"));
    }
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
        "[0:v]{}[main];[1:v]{}[ref];[main][ref]{}=eof_action=endall:stats_file=-",
        pre.join(","),
        ref_leg.join(","),
        kind.filter(),
    )
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
    if let Some(fps) = ref_info.fps.or(dist_info.fps) {
        args.push("-r".to_owned());
        args.push(crate::probe::format_fps(fps));
    }
    args.push("-i".to_owned());
    args.push(dist_path.to_owned());
    if let Some(fps) = ref_info.fps.or(dist_info.fps) {
        args.push("-r".to_owned());
        args.push(crate::probe::format_fps(fps));
    }
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

/// Blocking filter-metric run; call off the UI thread. Progress is
/// monotonic-ish: only frame numbers above the shared high-water mark are
/// reported, from both the stdout `n:` feed and the stderr `frame=` feed
/// (Python dual-feed parity); the UI keeps the max per row.
/// ponytail: plain wait(), no wait-timeout dep — a stuck ffmpeg only stalls
/// this worker thread, never the UI.
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
    let fail = |msg: String| RunOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
    };
    let start = std::time::Instant::now();
    let args = build_args(
        kind, ref_path, dist_path, ref_info, dist_info, skip, clip_dur,
    );
    log::debug!(target: "rfmetrics::metric", "run: \"{}\" {}", exe.display(), args.join(" "));
    let mut child = match Command::new(exe)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            log::warn!(target: "rfmetrics::metric", "{name} spawn failed: {e}");
            return fail(format!("spawn failed: {e}"));
        }
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
        return fail("internal lock error".to_owned());
    }
    // Stderr reader: live `frame=` progress feed + summary capture.
    // Scoped thread so the borrowed progress callback need not be 'static.
    let max_sent = Arc::new(AtomicU64::new(0));
    let (values, err_text) = std::thread::scope(|s| {
        let err_max = Arc::clone(&max_sent);
        let err_handle = stderr.map(|err| {
            s.spawn(move || {
                let mut lines = Vec::new();
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if let Some(f) = err_progress_re()
                        .captures(&line)
                        .and_then(|c| c.get(1)?.as_str().parse::<u64>().ok())
                        && f > err_max.fetch_max(f, Ordering::SeqCst)
                    {
                        on_progress(f);
                    }
                    lines.push(line);
                }
                lines.join("\n")
            })
        });
        let mut values = Vec::new();
        if let Some(out) = stdout {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(f) = parse_progress(&line)
                    && f > max_sent.fetch_max(f, Ordering::SeqCst)
                {
                    on_progress(f);
                }
                if let Some(v) = parse_frame_line(&line, kind) {
                    values.push(v);
                }
            }
        }
        let err_text = err_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        (values, err_text)
    });
    // Reap: Stop takes + kills + waits ahead of us when aborting (it sets
    // the flag first, so a missing child always means "aborted").
    let status = child_slot
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
        .map(|mut c| c.wait());
    let exec_s = start.elapsed().as_secs_f64();
    if abort.load(Ordering::SeqCst) {
        log::info!(target: "rfmetrics::metric", "{name} \"{dist_path}\" aborted after {exec_s:.1}s");
        return RunOutcome {
            values: Vec::new(),
            avg: None,
            exec_s,
            error: Some("aborted".to_owned()),
        };
    }
    let code = status.and_then(|s| s.ok()).and_then(|s| s.code());
    if values.is_empty() {
        let tail = err_text.lines().map(str::trim).rfind(|l| !l.is_empty());
        let msg = tail.unwrap_or(&format!("no {name} data")).to_owned();
        log::warn!(target: "rfmetrics::metric", "{name} no data for \"{dist_path}\" (exit {code:?}, {exec_s:.1}s): {msg}");
        return RunOutcome {
            values,
            avg: None,
            exec_s,
            error: Some(msg),
        };
    }
    let avg = parse_summary(&err_text, kind);
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
}
