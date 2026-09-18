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
    let g = filtergraph(Psnr, &ref_info(), &dist, None, None, ScaleMethod::default());
    assert_eq!(
        g,
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic,format=yuv420p[main];\
             [1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];\
             [main][ref]psnr=eof_action=endall:stats_file=-"
    );
}

#[test]
fn graph_matching_streams_have_no_scale() {
    use super::MetricKind::Psnr;
    let g = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(10.0),
        ScaleMethod::default(),
    );
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
    let plain = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        None,
        None,
        ScaleMethod::default(),
    );
    assert_eq!(
        filtergraph(
            Psnr,
            &ref_info(),
            &ref_info(),
            Some(0.0),
            Some(0.0),
            ScaleMethod::default()
        ),
        plain
    );
    assert!(!plain.contains("trim="));
    // Mixed: zero side drops out, nonzero side applies.
    let g = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        Some(0.0),
        Some(10.0),
        ScaleMethod::default(),
    );
    assert!(g.contains("trim=start=0:end=10"));
    let g = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(0.0),
        ScaleMethod::default(),
    );
    assert!(g.contains("[0:v]trim=start=5,"));
    assert!(!g.contains(":end="));
}

#[test]
fn ssim_graph_differs_only_by_filter_name() {
    use super::MetricKind::{Psnr, Ssim};
    let psnr = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(10.0),
        ScaleMethod::default(),
    );
    let ssim = filtergraph(
        Ssim,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(10.0),
        ScaleMethod::default(),
    );
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
        ScaleMethod::default(),
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
        ScaleMethod::default(),
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
fn live_feed_flags() {
    use super::MetricKind::*;
    assert!(!Vmaf.streams_live_values());
    for k in [Psnr, Ssim, Xpsnr, Ssim2, But, Cvvdp] {
        assert!(k.streams_live_values());
    }
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
    assert!(parse_xpsnr_frame_line("n:6 xpsnr y: 40.0 xpsnr u: 40.0 xpsnr v: 40.0", w).is_some());
}

#[test]
fn plane_rows_for_csv() {
    assert_eq!(
        parse_psnr_planes("n:1 mse_avg:0.5 psnr_avg:45.42 psnr_y:43.93 psnr_u:52.94 psnr_v:52.73"),
        Some([45.42, 43.93, 52.94, 52.73])
    );
    assert_eq!(parse_psnr_planes("frame= 12 fps=25"), None);
    assert_eq!(parse_psnr_planes("n:1 psnr_avg:45.42"), None);
    assert_eq!(
        parse_ssim_planes("n:1 Y:0.984065 U:0.995422 V:0.995221 All:0.987817"),
        Some([0.984065, 0.995422, 0.995221, 0.987817])
    );
    assert_eq!(parse_ssim_planes("n:1 Y:0.9 All:0.95"), None);
    assert_eq!(
        parse_xpsnr_planes("n:1 XPSNR y: 48.1827 XPSNR u: 56.2272 XPSNR v: 55.9481"),
        Some([48.1827, 56.2272, 55.9481])
    );
    // inf sanitizes like the series.
    assert_eq!(
        parse_psnr_planes("n:1 psnr_avg:inf psnr_y:inf psnr_u:inf psnr_v:inf"),
        Some([100.0, 100.0, 100.0, 100.0])
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
fn xpsnr_summary_v9_shape() {
    // Real ffmpeg 9.0.1 stderr tail: poolable planes ride the
    // `[Parsed_xpsnr_*]` line (the `XPSNR average, N frames y:` line
    // carries y only and must not match).
    let w = (4.0, 1.0, 1.0);
    let err = "[Parsed_xpsnr_7 @ 000002296aacdc00] XPSNR  y: 39.6885  u: 44.3744  v: 44.0219  (minimum: 39.6885)\n\
                   XPSNR average, 300 frames  y: 39.6885\n";
    assert_eq!(
        parse_xpsnr_summary(err, w),
        Some((4.0 * 39.6885 + 44.3744 + 44.0219) / 6.0)
    );
    assert_eq!(
        parse_xpsnr_summary("XPSNR average, 300 frames  y: 39.6885\n", w),
        None
    );
}

#[test]
fn xpsnr_graph_inverts_input_order() {
    use super::MetricKind::{Psnr, Xpsnr};
    let psnr = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(10.0),
        ScaleMethod::default(),
    );
    let xpsnr = filtergraph(
        Xpsnr,
        &ref_info(),
        &ref_info(),
        Some(5.0),
        Some(10.0),
        ScaleMethod::default(),
    );
    // Same legs — only the order segment and filter name differ.
    assert_eq!(
        xpsnr,
        psnr.replace(
            "[main][ref]psnr=eof_action=endall:stats_file=-",
            "[ref][main]xpsnr=eof_action=endall:stats_file=-"
        )
    );
}

#[test]
fn setrange_tokens() {
    assert_eq!(
        setrange_segment(Some("tv")),
        Some("setrange=range=tv".to_owned())
    );
    assert_eq!(
        setrange_segment(Some("pc")),
        Some("setrange=range=pc".to_owned())
    );
    assert_eq!(setrange_segment(None), None);
    assert_eq!(setrange_segment(Some("mystery")), None);
}

#[test]
fn setrange_only_when_ranges_differ() {
    use super::MetricKind::Psnr;
    // Matching or unknown ranges: no segment (FFMetrics.log parity).
    let g = filtergraph(
        Psnr,
        &ref_info(),
        &ref_info(),
        None,
        None,
        ScaleMethod::default(),
    );
    assert!(!g.contains("setrange"));
    // tv vs pc: each leg tagged with its own range.
    let mut rf = ref_info();
    rf.range_tag = Some("tv".to_owned());
    let mut dist = ref_info();
    dist.range_tag = Some("pc".to_owned());
    let g = filtergraph(Psnr, &rf, &dist, None, None, ScaleMethod::default());
    assert!(g.contains("[0:v]settb=AVTB,setpts=PTS-STARTPTS,setrange=range=pc[main]"));
    assert!(g.contains("[1:v]settb=AVTB,setpts=PTS-STARTPTS,setrange=range=tv[ref]"));
}

#[test]
fn setrange_sits_between_scale_and_format() {
    use super::MetricKind::Psnr;
    let mut rf = ref_info();
    rf.range_tag = Some("tv".to_owned());
    let mut dist = ref_info();
    dist.range_tag = Some("pc".to_owned());
    dist.width = Some(1280);
    dist.height = Some(720);
    dist.pix_fmt = Some("yuv444p".to_owned());
    let g = filtergraph(Psnr, &rf, &dist, None, None, ScaleMethod::default());
    assert!(g.contains(
            "[0:v]settb=AVTB,setpts=PTS-STARTPTS,scale=1920:1080:flags=bicubic,setrange=range=pc,format=yuv420p[main]"
        ));
}

#[test]
fn args_start_with_probesize() {
    use super::MetricKind::Psnr;
    let a = build_args(
        Psnr,
        "ref.mp4",
        "dist.mp4",
        &ref_info(),
        &ref_info(),
        None,
        None,
        ScaleMethod::default(),
    );
    assert_eq!(&a[..4], &["-hide_banner", "-nostdin", "-probesize", "50M"]);
}

#[test]
fn scaler_labels_flags_and_lookup() {
    // Default is Bicubic; every UI label round-trips.
    assert_eq!(ScaleMethod::default(), ScaleMethod::Bicubic);
    assert_eq!(ScaleMethod::ALL.len(), 8);
    for m in ScaleMethod::ALL {
        assert_eq!(ScaleMethod::from_label(m.label()), Some(m));
    }
    assert_eq!(ScaleMethod::from_label("Nope"), None);
    // Flag names verified against `ffmpeg -h full` (sws_flags).
    assert_eq!(ScaleMethod::Bicubic.flag(), Some("bicubic"));
    assert_eq!(ScaleMethod::Neighbor.flag(), Some("neighbor"));
    assert_eq!(ScaleMethod::Gauss.flag(), Some("gauss"));
    assert_eq!(ScaleMethod::Bilinear.flag(), Some("bilinear"));
    assert_eq!(ScaleMethod::Lanczos.flag(), Some("lanczos"));
    assert_eq!(ScaleMethod::Spline.flag(), Some("spline"));
    assert_eq!(ScaleMethod::Sinc.flag(), Some("sinc"));
    // "FFmpeg default" omits `:flags=` (today's exact strings).
    assert_eq!(ScaleMethod::FfmpegDefault.flag(), None);
    assert_eq!(
        scale_filter(1920, 1080, ScaleMethod::Bicubic),
        "scale=1920:1080:flags=bicubic"
    );
    assert_eq!(
        scale_filter(1920, 1080, ScaleMethod::FfmpegDefault),
        "scale=1920:1080"
    );
}

#[test]
fn graph_scaler_selects_flags() {
    use super::MetricKind::Psnr;
    let mut dist = ref_info();
    dist.width = Some(1280);
    dist.height = Some(720);
    let g = filtergraph(Psnr, &ref_info(), &dist, None, None, ScaleMethod::Lanczos);
    assert!(g.contains("scale=1920:1080:flags=lanczos[main]"));
    // FFmpeg default: today's flagless strings.
    let g = filtergraph(
        Psnr,
        &ref_info(),
        &dist,
        None,
        None,
        ScaleMethod::FfmpegDefault,
    );
    assert!(g.contains("scale=1920:1080[main]"));
    assert!(!g.contains("flags="));
}
