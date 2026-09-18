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
    let out = Command::new(exe)
        .args([
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
        ])
        .output()
        .ok()?;
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
    InvalidJson(String),
    NoVideo,
}

/// Shared ffprobe spawn + JSON parse + video-stream pick + packet-count
/// fallback (for streams without usable `nb_frames`). Callers time this
/// call themselves, so their ms logs stay equivalent, and map `ProbeFail`
/// to their own user-facing text.
fn probe_once(path: &str, exe: &Path) -> Result<MediaInfo, ProbeFail> {
    let out = Command::new(exe)
        .args([
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
        ])
        .output()
        .map_err(|e| ProbeFail::Spawn(e.to_string()))?;
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

/// Shared ffprobe spawn + parse; `None` = no usable video stream.
pub(crate) fn probe_media(path: &str, ffprobe: Option<&Path>) -> Option<MediaInfo> {
    if path.trim().is_empty() || !Path::new(path).is_file() {
        return None;
    }
    let exe = ffprobe?;
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::probe", "probe: \"{}\" -show_format -show_streams \"{path}\"", exe.display());
    let info = match probe_once(path, exe) {
        Ok(info) => info,
        // Today's exact behavior: spawn failure is silent (`ok()?`).
        Err(ProbeFail::Spawn(_)) => return None,
        Err(ProbeFail::InvalidJson(e)) => {
            log::warn!(target: "rfmetrics::probe", "probe \"{path}\" invalid JSON: {e} ({}ms)", start.elapsed().as_millis());
            return None;
        }
        Err(ProbeFail::NoVideo) => {
            log::warn!(target: "rfmetrics::probe", "probe \"{path}\" no video stream ({}ms)", start.elapsed().as_millis());
            return None;
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
    Some(info)
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

/// Python `table_media_tooltip`: 8-line hover detail (note `Colour` spelling).
pub fn table_media_tooltip(info: Option<&MediaInfo>) -> String {
    let u = "-unknown-";
    let (size_s, rate_s, field_s, pix_s, range_s, bit_s, dur_s, frames_s) = match info {
        None => (
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
                size_s, rate_s, field_s, pix_s, range_s, bit_s, dur_s, frames_s,
            )
        }
    };
    format!(
        "Frame size: {size_s}\nFrame Rate: {rate_s}\nField Type: {field_s}\n\
         Pixel Format: {pix_s}\nColour Range: {range_s}\nBitrate: {bit_s}\nDuration: {dur_s}\nTotal Frames: {frames_s}"
    )
}

/// Single spawn returning truncated cell text + full tooltip for a queue row,
/// plus the raw info for metric runners (filtergraph scale/format decisions).
pub fn probe_table_text(path: &str, ffprobe: Option<&Path>) -> (String, String, Option<MediaInfo>) {
    let info = probe_media(path, ffprobe);
    let full = table_media_text(info.as_ref());
    let cell = cell_media_text(&full, 38);
    let tip = table_media_tooltip(info.as_ref());
    (cell, tip, info)
}

/// Single-line reference info, mirroring Python `reference_media_text`.
/// Spawns ffprobe only for existing files; anything else is a cheap string.
/// The returned info feeds metric runners (filtergraph scale/format decisions).
pub fn reference_media_text(path: &str, ffprobe: Option<&Path>) -> (String, Option<MediaInfo>) {
    if path.trim().is_empty() {
        return (
            "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-"
                .to_owned(),
            None,
        );
    }
    if !Path::new(path).is_file() {
        return ("File not found".to_owned(), None);
    }
    let Some(exe) = ffprobe else {
        return ("ffprobe not found".to_owned(), None);
    };
    let start = std::time::Instant::now();
    log::debug!(target: "rfmetrics::probe", "probe ref: \"{}\" -show_format -show_streams \"{path}\"", exe.display());
    let info = match probe_once(path, exe) {
        Ok(info) => info,
        Err(ProbeFail::Spawn(e)) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" spawn failed: {e}");
            return (format!("Probe failed: {e}"), None);
        }
        Err(ProbeFail::InvalidJson(_)) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" invalid output ({}ms)", start.elapsed().as_millis());
            return ("Probe failed: invalid output".to_owned(), None);
        }
        Err(ProbeFail::NoVideo) => {
            log::warn!(target: "rfmetrics::probe", "probe ref \"{path}\" no video stream ({}ms)", start.elapsed().as_millis());
            return ("No video stream".to_owned(), None);
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
        ("—".to_owned(), Some(info))
    } else {
        let text = parts.join(", ");
        log::info!(target: "rfmetrics::probe", "probe ref \"{path}\" → {text} ({}ms)", start.elapsed().as_millis());
        (text, Some(info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_fraction() {
        let f = parse_fps("30000/1001").unwrap();
        assert!((f - 29.970_029_97).abs() < 1e-6);
    }

    #[test]
    fn fps_rejects() {
        assert_eq!(parse_fps("0/0"), None);
        assert_eq!(parse_fps(""), None);
        assert_eq!(parse_fps("abc"), None);
        assert_eq!(parse_fps("25"), Some(25.0));
    }

    #[test]
    fn fps_display() {
        assert_eq!(format_fps(25.0), "25");
        assert_eq!(format_fps(29.97), "29.97");
        assert_eq!(format_fps(23.976), "23.98");
    }

    #[test]
    fn duration_display() {
        assert_eq!(format_duration(3661.5), "01:01:01.50");
        assert_eq!(format_duration(59.996), "00:01:00.00");
    }

    fn fixture() -> (Stream, Format) {
        let v: Stream = serde_json::from_str(
            r#"{
                "codec_type": "video", "codec_name": "h264",
                "width": 1920, "height": 1080,
                "avg_frame_rate": "30000/1001", "pix_fmt": "yuv420p",
                "color_range": "tv", "bit_rate": "5000000",
                "field_order": "progressive"
            }"#,
        )
        .unwrap();
        let f: Format =
            serde_json::from_str(r#"{"bit_rate": "5200000", "duration": "63.04"}"#).unwrap();
        (v, f)
    }

    #[test]
    fn media_from_fixture() {
        let (v, f) = fixture();
        let m = parse_media(&v, &f);
        assert_eq!(m.width, Some(1920));
        assert!((m.fps.unwrap() - 29.970_029_97).abs() < 1e-6);
        assert_eq!(m.pix_fmt.as_deref(), Some("yuv420p"));
        assert_eq!(m.range_tag.as_deref(), Some("tv"));
        assert_eq!(m.bitrate_kbps, Some(5000));
        assert!(!m.is_container_rate);
        assert_eq!(m.duration, Some(63.04));
        assert_eq!(m.encoder.as_deref(), Some("h264"));
        assert!(!m.interlaced);
    }

    #[test]
    fn container_bitrate_flag() {
        let (mut v, f) = fixture();
        v.bit_rate = None;
        let m = parse_media(&v, &f);
        assert_eq!(m.bitrate_kbps, Some(5200));
        assert!(m.is_container_rate);
    }

    #[test]
    fn text_edge_cases() {
        let (text, info) = reference_media_text("", None);
        assert!(text.contains("-unknown-"));
        assert!(info.is_none());
        assert_eq!(
            reference_media_text("C:/no/such/file.mp4", None).0,
            "File not found"
        );
    }

    #[test]
    fn text_needs_ffprobe() {
        let p = std::env::temp_dir().join("rfmetrics-probe-test.tmp");
        std::fs::write(&p, b"x").unwrap();
        let (s, info) = reference_media_text(&p.to_string_lossy(), None);
        std::fs::remove_file(&p).ok();
        assert_eq!(s, "ffprobe not found");
        assert!(info.is_none());
    }

    #[test]
    fn table_text_from_fixture() {
        let (v, f) = fixture();
        let m = parse_media(&v, &f);
        assert_eq!(table_media_text(Some(&m)), "h264, 1080p, YUV420, 5000 kb/s");
    }

    #[test]
    fn table_text_unknown() {
        assert_eq!(
            table_media_text(None),
            "-unknown-, -unknown-, -unknown-, -unknown-"
        );
    }

    #[test]
    fn table_pix_strip() {
        let (v, f) = fixture();
        let mut m = parse_media(&v, &f);
        m.pix_fmt = Some("rgb24".to_owned());
        assert!(table_media_text(Some(&m)).contains("RGB24"));
        m.pix_fmt = None;
        assert!(table_media_text(Some(&m)).contains("-unknown-"));
    }

    /// `pix_fmt` range suffix (`yuv420p(tv)`) splits off the range tag and
    /// the base format (exercises the shared `pix_fmt_re` static).
    #[test]
    fn pix_fmt_parens_range() {
        let (mut v, f) = fixture();
        v.pix_fmt = Some("yuv420p(tv)".to_owned());
        v.color_range = None;
        let t0 = std::time::Instant::now();
        let m = parse_media(&v, &f);
        eprintln!("pix_fmt parens parse: {:?}", t0.elapsed());
        assert_eq!(m.pix_fmt.as_deref(), Some("yuv420p"));
        assert_eq!(m.range_tag.as_deref(), Some("tv"));
        v.pix_fmt = Some("yuv420p(pc)".to_owned());
        let m = parse_media(&v, &f);
        assert_eq!(m.range_tag.as_deref(), Some("pc"));
        // No parens: untouched, no range inferred.
        v.pix_fmt = Some("yuv420p".to_owned());
        let m = parse_media(&v, &f);
        assert_eq!(m.pix_fmt.as_deref(), Some("yuv420p"));
        assert_eq!(m.range_tag, None);
    }

    #[test]
    fn table_star_in_tooltip_not_cell() {
        let (mut v, f) = fixture();
        v.bit_rate = None; // force container rate
        let m = parse_media(&v, &f);
        assert!(m.is_container_rate);
        assert!(!table_media_text(Some(&m)).contains('*'));
        assert!(table_media_tooltip(Some(&m)).contains("5200 kb/s*"));
    }

    #[test]
    fn cell_truncation() {
        let short = "h264, 1080p, YUV420, 5000 kb/s";
        assert_eq!(cell_media_text(short, 38), short);
        let long = "av1, 2160p, YUV420P10LE, 12345 kb/s, extra";
        let cell = cell_media_text(long, 38);
        assert!(cell.chars().count() <= 39);
        assert!(cell.ends_with('…'));
        // comma-aware: cuts at a ", " boundary, not mid-token
        assert!(!cell.contains("extra"));
        let no_comma = "x".repeat(50);
        assert_eq!(
            cell_media_text(&no_comma, 38),
            format!("{}…", "x".repeat(37))
        );
    }

    #[test]
    fn table_tooltip_exact() {
        let (v, f) = fixture();
        let m = parse_media(&v, &f);
        // 63.04s × 29.97fps ≈ 1889 frames (computed fallback, no nb_frames)
        assert_eq!(
            table_media_tooltip(Some(&m)),
            "Frame size: 1920x1080\nFrame Rate: 29.97 fps\nField Type: Progressive\n\
             Pixel Format: yuv420p\nColour Range: TV\nBitrate: 5000 kb/s\nDuration: 00:01:03.04\nTotal Frames: 1889"
        );
        assert!(table_media_tooltip(None).contains("Field Type: Progressive"));
        assert!(table_media_tooltip(None).contains("Total Frames: -unknown-"));
    }

    #[test]
    fn total_frames_prefers_nb_frames() {
        let (mut v, f) = fixture();
        v.nb_frames = Some("1500".to_owned());
        let m = parse_media(&v, &f);
        assert_eq!(m.total_frames, Some(1500));
        assert!(table_media_tooltip(Some(&m)).contains("Total Frames: 1500"));
    }

    #[test]
    fn total_frames_rejects_garbage() {
        let (mut v, f) = fixture();
        v.nb_frames = Some("N/A".to_owned());
        let m = parse_media(&v, &f);
        assert_eq!(m.total_frames, Some(1889)); // falls back to duration × fps
    }
}
