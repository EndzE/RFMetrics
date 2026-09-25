use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Stream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    codec_tag_string: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    pix_fmt: Option<String>,
    color_range: Option<String>,
    bit_rate: Option<String>,
    duration: Option<String>,
    nb_frames: Option<String>,
    field_order: Option<String>,
    tags: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Format {
    bit_rate: Option<String>,
    duration: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ProbeOutput {
    streams: Vec<Stream>,
    format: Format,
}

#[derive(Debug, Default, Clone)]
pub struct MediaInfo {
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub fps: Option<f64>,
    pub pix_fmt: Option<String>,
    pub range_tag: Option<String>,
    pub bitrate_kbps: Option<i64>,
    pub is_container_rate: bool,
    pub duration: Option<f64>,
    pub total_frames: Option<i64>,
    pub encoder: Option<String>,
    pub interlaced: bool,
}

fn parse_fps(rate: &str) -> Option<f64> {
    if rate.is_empty() || rate == "0/0" {
        return None;
    }
    if let Some((n, d)) = rate.split_once('/') {
        let denom: f64 = d.parse().ok()?;
        if denom == 0.0 {
            return None;
        }
        return n.parse::<f64>().ok().map(|num| num / denom);
    }
    rate.parse().ok()
}

pub(crate) fn format_fps(fps: f64) -> String {
    if (fps - fps.round()).abs() < 0.005 {
        format!("{}", fps.round() as i64)
    } else {
        format!("{fps:.2}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

fn format_duration(seconds: f64) -> String {
    let total = (seconds * 100.0).round() / 100.0;
    let mut h = (total / 3600.0).floor() as i64;
    let mut m = ((total % 3600.0) / 60.0).floor() as i64;
    let mut s = total - h as f64 * 3600.0 - m as f64 * 60.0;
    if s >= 59.995 {
        s = 0.0;
        m += 1;
        if m >= 60 {
            m = 0;
            h += 1;
        }
    }
    format!("{h:02}:{m:02}:{s:05.2}")
}

/// `pix_fmt` range suffix (`yuv420p(tv)`): compiled once, not per probe
/// (same `OnceLock` pattern as the metric parsers in `metrics/ffmpeg.rs`).
fn pix_fmt_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"([a-zA-Z0-9_]+)\s*\(([^)]+)\)").unwrap())
}

pub fn parse_media(v: &Stream, fmt: &Format) -> MediaInfo {
    let mut fps = None;
    for key in [v.avg_frame_rate.as_deref(), v.r_frame_rate.as_deref()] {
        if let Some(val) = key
            && let Some(f) = parse_fps(val)
            && f > 0.0
        {
            fps = Some(f);
            break;
        }
    }

    let mut pix_fmt = v.pix_fmt.clone();
    let mut range_tag: Option<String> = None;
    if let Some(cr) = v.color_range.as_deref() {
        match cr.trim().to_lowercase().as_str() {
            "tv" | "mpeg" => range_tag = Some("tv".to_owned()),
            "pc" | "jpeg" => range_tag = Some("pc".to_owned()),
            "" | "unknown" | "n/a" | "unspecified" | "none" => {}
            other => range_tag = Some(other.to_owned()),
        }
    }
    if let Some(pf) = pix_fmt.clone()
        && pf.contains('(')
        && let Some(c) = pix_fmt_re().captures(&pf)
    {
        let inner = c[2].to_lowercase();
        if inner.contains("tv") {
            range_tag = Some("tv".to_owned());
        } else if inner.contains("pc") {
            range_tag = Some("pc".to_owned());
        }
        pix_fmt = Some(c[1].to_owned());
    }
    if pix_fmt.as_deref().unwrap_or("").is_empty() {
        pix_fmt = None;
    }

    let br_raw = [v.bit_rate.as_deref(), fmt.bit_rate.as_deref()]
        .into_iter()
        .flatten()
        .find(|s| !s.is_empty());
    let mut bitrate_kbps = None;
    if let Some(raw) = br_raw
        && let Ok(br) = raw.parse::<f64>()
        && br > 0.0
    {
        bitrate_kbps = Some(br as i64 / 1000);
    }

    let mut duration = None;
    for src in [fmt.duration.as_deref(), v.duration.as_deref()]
        .into_iter()
        .flatten()
    {
        if let Ok(d) = src.parse::<f64>() {
            duration = Some(d);
            break;
        }
    }

    // Prefer ffprobe's own frame count; fall back to duration × fps.
    // (`nb_frames` is often missing/"N/A", and `parse` rejects those.)
    // The estimate is a last resort only: `probe_media` overrides it with
    // a packet count whenever `nb_frames` is unusable.
    let mut total_frames = v
        .nb_frames
        .as_deref()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|&n| n > 0);
    if total_frames.is_none()
        && let (Some(d), Some(f)) = (duration, fps)
        && d > 0.0
        && f > 0.0
    {
        total_frames = Some((d * f).round() as i64);
    }

    let mut encoder = None;
    if let Some(c) = v.codec_name.as_deref().filter(|s| !s.is_empty()) {
        encoder = Some(c.to_owned());
    } else if let Some(c) = v.codec_tag_string.as_deref().filter(|s| !s.is_empty()) {
        encoder = Some(c.to_owned());
    } else if let Some(raw) = v.tags.get("encoder").and_then(|t| t.as_str())
        && !raw.is_empty()
    {
        encoder = raw.split_whitespace().last().map(str::to_owned);
    }

    MediaInfo {
        width: v.width,
        height: v.height,
        fps,
        pix_fmt,
        range_tag,
        bitrate_kbps,
        is_container_rate: v.bit_rate.is_none() && fmt.bit_rate.is_some(),
        duration,
        total_frames,
        encoder,
        interlaced: matches!(
            v.field_order.as_deref(),
            Some("interlaced" | "tt" | "bb" | "tb" | "bt")
        ),
    }
}

/// Accurate frame count via packet scan, for streams without `nb_frames`.
/// Reads packet headers only (no decoding), so it is fast but not free —
/// called only when the cheap JSON probe has no usable count.
fn count_packets(exe: &Path, path: &str) -> Option<i64> {
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::probe", "count_packets: \"{}\" -count_packets \"{path}\"", exe.display());
    // Packet scans walk whole files: generous bound, warn on timeout.
    let mut cmd = Command::new(exe);
    cmd.args([
        "-v",
        "error",
        // FFMetrics.conf parity: larger probe window for sparse headers.
        "-probesize",
        "50M",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]);
    let out = match crate::cmd::output_timeout(cmd, crate::cmd::PACKET_COUNT_TIMEOUT) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
            log::warn!(target: "rfmetrics::probe", "count_packets \"{path}\" timed out after {:?} ({}ms)", crate::cmd::PACKET_COUNT_TIMEOUT, start.elapsed().as_millis());
            return None;
        }
        Err(_) => return None,
    };
    let n = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<i64>()
        .ok()
        .filter(|&n| n > 0);
    log::info!(
        target: "rfmetrics::probe",
        "count_packets \"{path}\" → {} ({}ms)",
        n.map_or("none".to_owned(), |n| n.to_string()),
        start.elapsed().as_millis(),
    );
    n
}

/// Spawn failure classes of the shared ffprobe core, so `probe_media`
/// (silent `None`) and `reference_media_text` (user-facing strings) share
/// the spawn + parse + stream-pick + packet-count fallback while keeping
/// their exact texts and log lines.
#[derive(Debug)]
enum ProbeFail {
    Spawn(String),
    Timeout,
    InvalidJson(String),
    NoVideo,
}

/// Shared ffprobe spawn + JSON parse + video-stream pick + packet-count
/// fallback (for streams without usable `nb_frames`). Callers time this
/// call themselves, so their ms logs stay equivalent, and map `ProbeFail`
/// to their own user-facing text.
fn probe_once(path: &str, exe: &Path) -> Result<MediaInfo, ProbeFail> {
    let mut cmd = Command::new(exe);
    cmd.args([
        "-v",
        "quiet",
        // FFMetrics.conf parity: larger probe window for sparse headers.
        "-probesize",
        "50M",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]);
    let out = match crate::cmd::output_timeout(cmd, crate::cmd::PROBE_TIMEOUT) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => return Err(ProbeFail::Timeout),
        Err(e) => return Err(ProbeFail::Spawn(e.to_string())),
    };
    let data: ProbeOutput =
        serde_json::from_slice(&out.stdout).map_err(|e| ProbeFail::InvalidJson(e.to_string()))?;
    let Some(v) = data
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .or_else(|| data.streams.first())
    else {
        return Err(ProbeFail::NoVideo);
    };
    let mut info = parse_media(v, &data.format);
    // `duration × fps` is only an estimate (wrong on VFR/long-GOP files),
    // so when `nb_frames` gave nothing usable, count packets instead.
    let has_nb = v
        .nb_frames
        .as_deref()
        .and_then(|s| s.parse::<i64>().ok())
        .is_some_and(|n| n > 0);
    if !has_nb && let Some(n) = count_packets(exe, path) {
        info.total_frames = Some(n);
    }
    Ok(info)
}

/// Empty/missing guard shared by probe, thumbnail, run, and badframes
/// call sites: non-blank path to an existing file.
pub(crate) fn path_usable(path: &str) -> bool {
    !path.trim().is_empty() && Path::new(path).is_file()
}

/// Shared ffprobe spawn + parse; `None` = no usable video stream.
pub(crate) fn probe_media(path: &str, ffprobe: Option<&Path>) -> Option<MediaInfo> {
    if !path_usable(path) {
        return None;
    }
    let exe = ffprobe?;
    probe_media_fail(path, exe).0
}

/// Same as [`probe_media`], plus whether the probe timed out (the queue
/// worker surfaces timeouts as toasts; `probe_media` callers don't care).
fn probe_media_fail(path: &str, exe: &Path) -> (Option<MediaInfo>, bool) {
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::probe", "probe: \"{}\" -show_format -show_streams \"{path}\"", exe.display());
    let info = match probe_once(path, exe) {
        Ok(info) => info,
        // Today's exact behavior: spawn failure is silent (`ok()?`).
        Err(ProbeFail::Spawn(_)) => return (None, false),
        Err(ProbeFail::Timeout) => {
            log::warn!(target: "rfmetrics::probe", "probe \"{path}\" timed out after {:?} ({}ms)", crate::cmd::PROBE_TIMEOUT, start.elapsed().as_millis());
            return (None, true);
        }
        Err(ProbeFail::InvalidJson(e)) => {
            log::warn!(target: "rfmetrics::probe", "probe \"{path}\" invalid JSON: {e} ({}ms)", start.elapsed().as_millis());
            return (None, false);
        }
        Err(ProbeFail::NoVideo) => {
            log::warn!(target: "rfmetrics::probe", "probe \"{path}\" no video stream ({}ms)", start.elapsed().as_millis());
            return (None, false);
        }
    };
    log::info!(
        target: "rfmetrics::probe",
        "probe \"{path}\" → {}x{} {:?} ({}ms)",
        info.width.unwrap_or(-1),
        info.height.unwrap_or(-1),
        info.encoder,
        start.elapsed().as_millis(),
    );
    (Some(info), false)
}

/// Duration helper for the thumbnail worker (Python `_get_media_duration`).
pub(crate) fn media_duration(path: &str, ffprobe: Option<&Path>) -> Option<f64> {
    probe_media(path, ffprobe)?.duration.filter(|&d| d > 0.0)
}

/// Python `table_media_text`: `{enc}, {height+suffix}, {PIX}, {bitrate}`.
pub fn table_media_text(info: Option<&MediaInfo>) -> String {
    let u = "-unknown-";
    let Some(info) = info else {
        return format!("{u}, {u}, {u}, {u}");
    };
    let suffix = if info.interlaced { "i" } else { "p" };
    let height_s = match info.height {
        Some(h) => format!("{h}{suffix}"),
        None => u.to_owned(),
    };
    let pix_s = match info.pix_fmt.as_deref() {
        Some(p) => {
            let up = p.to_uppercase();
            up.strip_suffix('P').unwrap_or(&up).to_owned()
        }
        None => u.to_owned(),
    };
    let bit_s = match info.bitrate_kbps {
        Some(kbps) => format!("{kbps} kb/s"),
        None => u.to_owned(),
    };
    let enc_s = info
        .encoder
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(u);
    format!("{enc_s}, {height_s}, {pix_s}, {bit_s}")
}

/// Results-CSV `Frame` column (`1600x1080-60p, yuv420p10le (tv)`):
/// size + rate + field suffix, then pixfmt + range. Unknown parts are
/// skipped; fully unknown media is `-unknown-`.
pub fn results_media_text(info: Option<&MediaInfo>) -> String {
    let u = "-unknown-";
    let Some(info) = info else {
        return u.to_owned();
    };
    let mut head = String::new();
    if let (Some(w), Some(h)) = (info.width, info.height) {
        head.push_str(&format!("{w}x{h}"));
    }
    if let Some(fps) = info.fps {
        let suffix = if info.interlaced { "i" } else { "p" };
        if !head.is_empty() {
            head.push('-');
        }
        head.push_str(&format!("{}{suffix}", format_fps(fps)));
    }
    let mut pix = info.pix_fmt.clone().unwrap_or_default();
    if let Some(rt) = info.range_tag.as_deref()
        && !pix.is_empty()
    {
        pix.push_str(&format!(" ({rt})"));
    }
    match (head.is_empty(), pix.is_empty()) {
        (false, false) => format!("{head}, {pix}"),
        (false, true) => head,
        (true, false) => pix,
        (true, true) => u.to_owned(),
    }
}

/// Python `_cell_media_text`: comma-aware truncation to `limit` chars.
pub fn cell_media_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let truncated: String = text.chars().take(limit).collect();
    match truncated.rsplit_once(", ") {
        Some((head, _)) => format!("{head}…"),
        None => {
            let h: String = truncated.chars().take(limit.saturating_sub(1)).collect();
            format!("{}…", h.trim_end())
        }
    }
}

/// Python `table_media_tooltip`: 9-line hover detail (note `Colour` spelling).
pub fn table_media_tooltip(info: Option<&MediaInfo>) -> String {
    let u = "-unknown-";
    let (enc_s, size_s, rate_s, field_s, pix_s, range_s, bit_s, dur_s, frames_s) = match info {
        None => (
            u.to_owned(),
            u.to_owned(),
            u.to_owned(),
            "Progressive".to_owned(),
            u.to_owned(),
            u.to_owned(),
            u.to_owned(),
            u.to_owned(),
            u.to_owned(),
        ),
        Some(info) => {
            let enc_s = info
                .encoder
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or(u)
                .to_owned();
            let size_s = match (info.width, info.height) {
                (Some(w), Some(h)) => format!("{w}x{h}"),
                _ => u.to_owned(),
            };
            let rate_s = match info.fps {
                Some(fps) => format!("{} fps", format_fps(fps)),
                None => u.to_owned(),
            };
            let field_s = if info.interlaced {
                "Interlaced"
            } else {
                "Progressive"
            }
            .to_owned();
            let pix_s = info.pix_fmt.as_deref().unwrap_or(u).to_owned();
            let range_s = info
                .range_tag
                .as_deref()
                .map(|r| r.to_uppercase())
                .unwrap_or_else(|| u.to_owned());
            let bit_s = match info.bitrate_kbps {
                Some(kbps) => {
                    let star = if info.is_container_rate { "*" } else { "" };
                    format!("{kbps} kb/s{star}")
                }
                None => u.to_owned(),
            };
            let dur_s = match info.duration {
                Some(d) if d > 0.0 => format_duration(d),
                _ => u.to_owned(),
            };
            let frames_s = match info.total_frames {
                Some(n) => n.to_string(),
                None => u.to_owned(),
            };
            (
                enc_s, size_s, rate_s, field_s, pix_s, range_s, bit_s, dur_s, frames_s,
            )
        }
    };
    format!(
        "Encoder: {enc_s}\nFrame size: {size_s}\nFrame Rate: {rate_s}\nField Type: {field_s}\n\
         Pixel Format: {pix_s}\nColour Range: {range_s}\nBitrate: {bit_s}\nDuration: {dur_s}\nTotal Frames: {frames_s}"
    )
}

/// Single spawn returning truncated cell text + full tooltip for a queue row,
/// plus the raw info for metric runners (filtergraph scale/format decisions).
pub fn probe_table_text(
    path: &str,
    ffprobe: Option<&Path>,
) -> (String, String, Option<MediaInfo>, bool) {
    // Same guards as `probe_media`; the flag reports an ffprobe timeout
    // (the queue worker surfaces it as a toast).
    let (info, timed_out) = match (!path_usable(path), ffprobe) {
        (true, _) | (_, None) => (None, false),
        (false, Some(exe)) => probe_media_fail(path, exe),
    };
    let full = table_media_text(info.as_ref());
    let cell = cell_media_text(&full, 38);
    let tip = table_media_tooltip(info.as_ref());
    (cell, tip, info, timed_out)
}

/// Single-line reference info, mirroring Python `reference_media_text`.
/// Spawns ffprobe only for existing files; anything else is a cheap string.
/// The returned info feeds metric runners (filtergraph scale/format decisions).
/// The trailing flag reports an ffprobe timeout (the worker surfaces it as
/// a toast; the text itself stays user-facing like the other failures).
pub fn reference_media_text(
    path: &str,
    ffprobe: Option<&Path>,
) -> (String, Option<MediaInfo>, bool) {
    if path.trim().is_empty() {
        return (
            "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-"
                .to_owned(),
            None,
            false,
        );
    }
    if !Path::new(path).is_file() {
        return ("File not found".to_owned(), None, false);
    }
    let Some(exe) = ffprobe else {
        return ("ffprobe not found".to_owned(), None, false);
    };
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::probe", "probe ref: \"{}\" -show_format -show_streams \"{path}\"", exe.display());
    let info = match probe_once(path, exe) {
        Ok(info) => info,
        Err(ProbeFail::Spawn(e)) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" spawn failed: {e}");
            return ("Probe failed: {e}".to_owned(), None, false);
        }
        Err(ProbeFail::Timeout) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" timed out after {:?} ({}ms)", crate::cmd::PROBE_TIMEOUT, start.elapsed().as_millis());
            return ("Probe timed out".to_owned(), None, true);
        }
        Err(ProbeFail::InvalidJson(_)) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" invalid output ({}ms)", start.elapsed().as_millis());
            return ("Probe failed: invalid output".to_owned(), None, false);
        }
        Err(ProbeFail::NoVideo) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" no video stream ({}ms)", start.elapsed().as_millis());
            return ("No video stream".to_owned(), None, false);
        }
    };

    let mut parts: Vec<String> = Vec::new();
    if let Some(e) = info.encoder.as_deref().filter(|s| !s.is_empty()) {
        parts.push(format!("Encoder: {e}"));
    }
    let suffix = if info.interlaced { "i" } else { "p" };
    match (info.width, info.height, info.fps) {
        (Some(w), Some(h), Some(fps)) => {
            parts.push(format!("Frame: {w}x{h}-{}{suffix}", format_fps(fps)));
        }
        (Some(w), Some(h), None) => parts.push(format!("Frame: {w}x{h}")),
        (_, _, Some(fps)) => parts.push(format!("Frame: {}{suffix}", format_fps(fps))),
        _ => {}
    }
    if let Some(pf) = info.pix_fmt.as_deref() {
        if let Some(rt) = info.range_tag.as_deref() {
            parts.push(format!("{pf} ({rt})"));
        } else {
            parts.push(pf.to_owned());
        }
    }
    if let Some(kbps) = info.bitrate_kbps {
        let star = if info.is_container_rate { "*" } else { "" };
        parts.push(format!("Bitrate: {kbps} kb/s{star}"));
    }
    if let Some(d) = info.duration
        && d > 0.0
    {
        parts.push(format!("Duration: {}", format_duration(d)));
    }
    if let Some(n) = info.total_frames {
        parts.push(format!("Total Frames: {n}"));
    }
    if parts.is_empty() {
        log::info!(target: "rfmetrics::probe", "probe ref \"{path}\" → no info ({}ms)", start.elapsed().as_millis());
        ("—".to_owned(), Some(info), false)
    } else {
        let text = parts.join(", ");
        log::info!(target: "rfmetrics::probe", "probe ref \"{path}\" → {text} ({}ms)", start.elapsed().as_millis());
        (text, Some(info), false)
    }
}
#[cfg(test)]
#[path = "tests/test_probe.rs"]
mod tests;
