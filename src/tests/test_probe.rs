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
    let (text, info, timed_out) = reference_media_text("", None);
    assert!(text.contains("-unknown-"));
    assert!(info.is_none());
    assert!(!timed_out);
    assert_eq!(
        reference_media_text("C:/no/such/file.mp4", None).0,
        "File not found"
    );
}

#[test]
fn text_needs_ffprobe() {
    let p = std::env::temp_dir().join("rfmetrics-probe-test.tmp");
    std::fs::write(&p, b"x").unwrap();
    let (s, info, timed_out) = reference_media_text(&p.to_string_lossy(), None);
    std::fs::remove_file(&p).ok();
    assert_eq!(s, "ffprobe not found");
    assert!(info.is_none());
    assert!(!timed_out);
}

#[test]
fn table_text_from_fixture() {
    let (v, f) = fixture();
    let m = parse_media(&v, &f);
    assert_eq!(table_media_text(Some(&m)), "h264, 1080p, YUV420, 5000 kb/s");
}

#[test]
fn results_media_line() {
    // Fixture: 1920x1080, 29.97fps progressive, yuv420p tv.
    let (v, f) = fixture();
    let m = parse_media(&v, &f);
    assert_eq!(
        results_media_text(Some(&m)),
        "1920x1080-29.97p, yuv420p (tv)"
    );
    assert_eq!(results_media_text(None), "-unknown-");
    let mut interlaced = m;
    interlaced.interlaced = true;
    interlaced.pix_fmt = None;
    assert!(results_media_text(Some(&interlaced)).ends_with("i"));
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
        "Encoder: h264\nFrame size: 1920x1080\nFrame Rate: 29.97 fps\nField Type: Progressive\n\
             Pixel Format: yuv420p\nColour Range: TV\nBitrate: 5000 kb/s\nDuration: 00:01:03.04\nTotal Frames: 1889"
    );
    assert!(table_media_tooltip(None).contains("Encoder: -unknown-"));
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

#[test]
fn path_usable_guard() {
    assert!(!path_usable(""));
    assert!(!path_usable("   "));
    assert!(!path_usable("C:/no/such/rfmetrics-file.mp4"));
    let p = std::env::temp_dir().join("rfmetrics-path-usable.tmp");
    std::fs::write(&p, b"x").unwrap();
    assert!(path_usable(&p.to_string_lossy()));
    std::fs::remove_file(&p).ok();
    assert!(!path_usable(&std::env::temp_dir().to_string_lossy()));
}
