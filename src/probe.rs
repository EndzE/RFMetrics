use std::path::Path;
use std::process::Command;

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

fn format_fps(fps: f64) -> String {
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
        && let Ok(re) = regex::Regex::new(r"([a-zA-Z0-9_]+)\s*\(([^)]+)\)")
        && let Some(c) = re.captures(&pf)
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
        encoder,
        interlaced: matches!(
            v.field_order.as_deref(),
            Some("interlaced" | "tt" | "bb" | "tb" | "bt")
        ),
    }
}

/// Single-line reference info, mirroring Python `reference_media_text`.
/// Spawns ffprobe only for existing files; anything else is a cheap string.
pub fn reference_media_text(path: &str, ffprobe: Option<&Path>) -> String {
    if path.trim().is_empty() {
        return "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-"
            .to_owned();
    }
    if !Path::new(path).is_file() {
        return "File not found".to_owned();
    }
    let Some(exe) = ffprobe else {
        return "ffprobe not found".to_owned();
    };
    let out = match Command::new(exe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()
    {
        Ok(o) => o,
        Err(e) => return format!("Probe failed: {e}"),
    };
    let data: ProbeOutput = match serde_json::from_slice(&out.stdout) {
        Ok(d) => d,
        Err(_) => return "Probe failed: invalid output".to_owned(),
    };
    let streams = &data.streams;
    let Some(v) = streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .or_else(|| streams.first())
    else {
        return "No video stream".to_owned();
    };
    let info = parse_media(v, &data.format);

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
    if parts.is_empty() {
        "—".to_owned()
    } else {
        parts.join(", ")
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
        assert!(reference_media_text("", None).contains("-unknown-"));
        assert_eq!(
            reference_media_text("C:/no/such/file.mp4", None),
            "File not found"
        );
    }

    #[test]
    fn text_needs_ffprobe() {
        let p = std::env::temp_dir().join("rfmetrics-probe-test.tmp");
        std::fs::write(&p, b"x").unwrap();
        let s = reference_media_text(&p.to_string_lossy(), None);
        std::fs::remove_file(&p).ok();
        assert_eq!(s, "ffprobe not found");
    }
}
