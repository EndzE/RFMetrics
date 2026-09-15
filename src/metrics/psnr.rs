use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::probe::MediaInfo;

fn frame_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"psnr_avg:(\S+)").unwrap())
}

fn progress_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\bn:\s*(\d+)").unwrap())
}

fn summary_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"average:(\S+)").unwrap())
}

fn err_progress_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"frame=\s*(\d+)").unwrap())
}

/// Per-frame value from a `stats_file=-` stdout line
/// (`n:1 ... psnr_avg:34.12 ...`), clamped to 0–100 (Python parity;
/// `inf` clamps, `nan`/garbage lines are skipped).
pub fn parse_frame_line(line: &str) -> Option<f64> {
    if !line.trim_start().starts_with("n:") {
        return None;
    }
    let v: f64 = frame_re().captures(line)?.get(1)?.as_str().parse().ok()?;
    if v.is_nan() {
        return None;
    }
    Some(v.clamp(0.0, 100.0))
}

/// Frame counter from a stats line (`n:12 ...`) for progress.
pub fn parse_progress(line: &str) -> Option<u64> {
    progress_re().captures(line)?.get(1)?.as_str().parse().ok()
}

/// Pooled average from ffmpeg's stderr summary
/// (`... PSNR ... average:33.98 ...`), scanned bottom-up (Python parity,
/// unclamped — `inf` stays `inf`).
pub fn parse_summary(text: &str) -> Option<f64> {
    for line in text.lines().rev() {
        if line.contains("PSNR")
            && let Some(c) = summary_re().captures(line)
            && let Ok(v) = c.get(1).unwrap().as_str().parse::<f64>()
        {
            return Some(v);
        }
    }
    None
}

/// Python `_compute_series` filtergraph for PSNR. Inputs are inverted vs.
/// the arg order: `-i dist` is `[0:v]`/main, `-i ref` is `[1:v]`/ref, and
/// the filter is `[main][ref]psnr=…`. The distorted leg is scaled/converted
/// up to the reference when they differ; the reference leg only gets trim.
pub fn filtergraph(
    ref_info: &MediaInfo,
    dist_info: &MediaInfo,
    skip: Option<f64>,
    clip_dur: Option<f64>,
) -> String {
    const NORM: &str = "settb=AVTB,setpts=PTS-STARTPTS";
    let mut window = Vec::new();
    if skip.is_some() || clip_dur.is_some() {
        let start = skip.unwrap_or(0.0);
        let end = clip_dur
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
        "[0:v]{}[main];[1:v]{}[ref];[main][ref]psnr=eof_action=endall:stats_file=-",
        pre.join(","),
        ref_leg.join(","),
    )
}

/// Full ffmpeg argv (minus the exe) for a PSNR run.
pub fn build_args(
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
    args.push(filtergraph(ref_info, dist_info, skip, clip_dur));
    args.extend(["-f".to_owned(), "null".to_owned(), "-".to_owned()]);
    args
}

/// Everything a PSNR run needs; bundled so `run_psnr` stays lean and
/// later metrics (SSIM/XPSNR take the same inputs) can reuse the shape.
pub struct PsnrInputs<'a> {
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

pub struct PsnrOutcome {
    pub values: Vec<f64>,
    pub avg: Option<f64>,
    pub exec_s: f64,
    pub error: Option<String>,
}

/// Blocking PSNR run; call off the UI thread. Progress is monotonic-ish:
/// only frame numbers above the shared high-water mark are reported, from
/// both the stdout `n:` feed and the stderr `frame=` feed (Python dual-feed
/// parity); the UI keeps the max per row.
/// ponytail: plain wait(), no wait-timeout dep — a stuck ffmpeg only stalls
/// this worker thread, never the UI.
pub fn run_psnr(job: &PsnrInputs<'_>, on_progress: &(dyn Fn(u64) + Sync)) -> PsnrOutcome {
    let PsnrInputs {
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
    let fail = |msg: String| PsnrOutcome {
        values: Vec::new(),
        avg: None,
        exec_s: 0.0,
        error: Some(msg),
    };
    let start = std::time::Instant::now();
    let args = build_args(ref_path, dist_path, ref_info, dist_info, skip, clip_dur);
    log::debug!(target: "rfmetrics::psnr", "run: \"{}\" {}", exe.display(), args.join(" "));
    let mut child = match Command::new(exe)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            log::warn!(target: "rfmetrics::psnr", "spawn failed: {e}");
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
                if let Some(v) = parse_frame_line(&line) {
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
        log::info!(target: "rfmetrics::psnr", "\"{dist_path}\" aborted after {exec_s:.1}s");
        return PsnrOutcome {
            values: Vec::new(),
            avg: None,
            exec_s,
            error: Some("aborted".to_owned()),
        };
    }
    let code = status.and_then(|s| s.ok()).and_then(|s| s.code());
    if values.is_empty() {
        let tail = err_text.lines().map(str::trim).rfind(|l| !l.is_empty());
        let msg = tail.unwrap_or("no PSNR data").to_owned();
        log::warn!(target: "rfmetrics::psnr", "no data for \"{dist_path}\" (exit {code:?}, {exec_s:.1}s): {msg}");
        return PsnrOutcome {
            values,
            avg: None,
            exec_s,
            error: Some(msg),
        };
    }
    let avg = parse_summary(&err_text);
    let show = avg.unwrap_or_else(|| values.iter().sum::<f64>() / values.len() as f64);
    log::info!(
        target: "rfmetrics::psnr",
        "\"{dist_path}\" → {show:.4} ({} frames, exit {code:?}, {exec_s:.1}s)",
        values.len(),
    );
    PsnrOutcome {
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
        assert_eq!(
            parse_frame_line("n:1 mse_avg:12.3 psnr_avg:34.1234 mse_y:1.0"),
            Some(34.1234)
        );
        assert_eq!(parse_frame_line("frame= 12 fps=25"), None);
        assert_eq!(parse_frame_line("n:2 no avg here"), None);
        assert_eq!(parse_frame_line("n:3 psnr_avg:inf"), Some(100.0));
        assert_eq!(parse_frame_line("n:3 psnr_avg:-inf"), Some(0.0));
        assert_eq!(parse_frame_line("n:3 psnr_avg:nan"), None);
        assert_eq!(parse_frame_line("n:3 psnr_avg:garbage"), None);
    }

    #[test]
    fn summary_scans_bottom_up() {
        let err = "[Parsed_psnr_0] PSNR y:1 u:2 v:3 average:30.0 min:1 max:2\n\
                   [Parsed_psnr_0] PSNR y:1 u:2 v:3 average:33.98 min:1 max:2\n";
        assert_eq!(parse_summary(err), Some(33.98));
        assert_eq!(parse_summary("nothing here"), None);
    }

    #[test]
    fn graph_scales_dist_to_ref() {
        let mut dist = ref_info();
        dist.width = Some(1280);
        dist.height = Some(720);
        dist.pix_fmt = Some("yuv444p".to_owned());
        let g = filtergraph(&ref_info(), &dist, None, None);
        assert_eq!(
            g,
            "[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080,format=yuv420p[main];\
             [1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];\
             [main][ref]psnr=eof_action=endall:stats_file=-"
        );
    }

    #[test]
    fn graph_matching_streams_have_no_scale() {
        let g = filtergraph(&ref_info(), &ref_info(), Some(5.0), Some(10.0));
        assert!(g.contains("[0:v]trim=start=5:end=15,"));
        assert!(g.contains("[1:v]trim=start=5:end=15,"));
        assert!(!g.contains("scale="));
        assert!(!g.contains("format="));
    }

    #[test]
    fn args_order_is_dist_then_ref() {
        let a = build_args("ref.mp4", "dist.mp4", &ref_info(), &ref_info(), None, None);
        let i1 = a.iter().position(|x| x == "dist.mp4").unwrap();
        let i2 = a.iter().position(|x| x == "ref.mp4").unwrap();
        assert!(i1 < i2);
        assert!(a.contains(&"-r".to_owned()) && a.contains(&"25".to_owned()));
        assert_eq!(a.last().unwrap(), "-");
    }
}
