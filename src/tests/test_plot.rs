use super::*;

#[test]
fn plot_size_labels_round_trip() {
    for m in PlotSize::ALL {
        assert_eq!(PlotSize::from_label(m.label()), Some(m));
    }
    assert_eq!(PlotSize::from_label("Nope"), None);
    assert_eq!(PlotSize::default().dims(), (3200, 800));
}

#[test]
fn empty_keeps_metric_defaults() {
    assert_eq!(
        fit_limits(&[], PSNR_LO, PSNR_HI),
        (None, (PSNR_LO, PSNR_HI))
    );
    let empty: &[f64] = &[];
    assert_eq!(
        fit_limits(&[empty, empty], PSNR_LO, PSNR_HI),
        (None, (PSNR_LO, PSNR_HI))
    );
}

#[test]
fn span_gets_five_percent_pad() {
    // min 30, max 40, span 10 -> pad 0.5.
    let (x, (lo, hi)) = fit_limits(&[&[30.0, 40.0, 35.0]], PSNR_LO, PSNR_HI);
    assert_eq!(x, Some((1.0, 3.0)));
    assert!((lo - 29.5).abs() < 1e-9);
    assert!((hi - 40.5).abs() < 1e-9);
}

#[test]
fn flat_series_pads_by_magnitude() {
    // span 0, |40| * 2% = 0.8.
    let (x, (lo, hi)) = fit_limits(&[&[40.0, 40.0]], PSNR_LO, PSNR_HI);
    assert_eq!(x, Some((1.0, 2.0)));
    assert!((lo - 39.2).abs() < 1e-9);
    assert!((hi - 40.8).abs() < 1e-9);
}

#[test]
fn flat_zero_series_pads_by_half() {
    // Python `0.0 or 0.5` fallback.
    let (x, (lo, hi)) = fit_limits(&[&[0.0, 0.0]], PSNR_LO, PSNR_HI);
    assert_eq!(x, Some((1.0, 2.0)));
    assert_eq!((lo, hi), (-0.5, 0.5));
}

#[test]
fn x_covers_longest_series() {
    let a = [30.0, 31.0, 32.0];
    let b = [28.0, 29.0, 30.0, 31.0, 33.0];
    let (x, (lo, hi)) = fit_limits(&[&a, &b], PSNR_LO, PSNR_HI);
    assert_eq!(x, Some((1.0, 5.0)));
    // all values min 28, max 33, span 5 -> pad 0.25.
    assert!((lo - 27.75).abs() < 1e-9);
    assert!((hi - 33.25).abs() < 1e-9);
}

#[test]
fn single_point_centers_x_bounds() {
    // egui_plot panics on degenerate min >= max bounds: the first live
    // batch (or a 1-frame run) must still yield a strict span.
    let (x, (lo, hi)) = fit_limits(&[&[42.0]], PSNR_LO, PSNR_HI);
    assert_eq!(x, Some((0.5, 1.5)));
    assert!(lo < hi);
}

/// Identity-ish screen map: 10 px per frame, 1 px per unit.
fn scr(fx: f64, fy: f64) -> (f32, f32) {
    (fx as f32 * 10.0, fy as f32)
}

#[test]
fn hover_hits_nearest_point() {
    let a = [10.0, 20.0, 30.0];
    let b = [15.0, 25.0, 35.0];
    // Pointer x on frame 2, mouse exactly on a's point.
    assert_eq!(
        nearest_hover(&[&a, &b], 2.0, scr, (20.0, 20.0)),
        Some((0, 2, 20.0))
    );
    // Same x, mouse closer to b's point (25) than a's (20).
    assert_eq!(
        nearest_hover(&[&a, &b], 2.0, scr, (20.0, 24.0)),
        Some((1, 2, 25.0))
    );
}

#[test]
fn hover_misses_beyond_30px() {
    let a = [10.0, 20.0, 30.0];
    // 31 px away: hidden (Python `> 30` hides).
    assert_eq!(nearest_hover(&[&a], 2.0, scr, (20.0, 51.0)), None);
    // Exactly 30 px: still shown.
    assert_eq!(
        nearest_hover(&[&a], 2.0, scr, (20.0, 50.0)),
        Some((0, 2, 20.0))
    );
}

#[test]
fn hover_skips_empty_series() {
    let empty: &[f64] = &[];
    let a = [10.0, 20.0];
    assert_eq!(
        nearest_hover(&[empty, &a], 1.0, scr, (10.0, 10.0)),
        Some((1, 1, 10.0))
    );
    assert_eq!(nearest_hover(&[empty], 1.0, scr, (10.0, 10.0)), None);
}

#[test]
fn defs_match_python_plot_defs() {
    use crate::metrics::ffmpeg::MetricKind;
    let def = plot_def(MetricKind::Psnr);
    assert_eq!(def.label, PSNR_LABEL);
    assert_eq!((def.lo, def.hi), (PSNR_LO, PSNR_HI));
    let def = plot_def(MetricKind::Ssim);
    assert_eq!(def.label, "SSIM (higher is better, min 0, max 1)");
    assert_eq!((def.lo, def.hi), (0.0, 1.0));
    let def = plot_def(MetricKind::But);
    assert_eq!(def.label, "BUTTERAUGLI (lower is better, min 0)");
    assert_eq!((def.lo, def.hi), (0.0, 10.0));
    let def = plot_def(MetricKind::Cvvdp);
    assert_eq!(
        def.label,
        "CVVDP (cumulative JOD, higher is better, 10 = identical)"
    );
    assert_eq!((def.lo, def.hi), (0.0, 10.0));
}

#[test]
fn tab_titles_match_python_tabs() {
    use crate::metrics::ffmpeg::MetricKind;
    let titles: Vec<_> = MetricKind::ALL.iter().map(|k| tab_title(*k)).collect();
    assert_eq!(
        titles,
        [
            "PSNR",
            "SSIM",
            "VMAF",
            "XPSNR",
            "SSIM2",
            "BUTTERAUGLI",
            "CVVDP"
        ]
    );
}

#[test]
fn union_bounds_only_grows() {
    let a = (Some((1.0, 10.0)), (20.0, 30.0));
    let b = (Some((1.0, 20.0)), (25.0, 28.0));
    // x max and y min expand; y max and x min hold.
    assert_eq!(union_bounds(a, b), (Some((1.0, 20.0)), (20.0, 30.0)));
    // Empty-vs-data either way keeps the data side whole.
    assert_eq!(union_bounds((None, (0.0, 100.0)), a), a);
    assert_eq!(union_bounds(a, (None, (0.0, 100.0))), a);
}

#[test]
fn follow_live_tab_switches_only_while_measuring() {
    use crate::metrics::ffmpeg::MetricKind;
    // Live job wins while measuring, even over another tab.
    assert_eq!(
        follow_live_tab(true, Some(MetricKind::Xpsnr), MetricKind::Psnr),
        MetricKind::Xpsnr
    );
    // No live job yet: stays put.
    assert_eq!(
        follow_live_tab(true, None, MetricKind::Psnr),
        MetricKind::Psnr
    );
    // Idle: user-driven, never switched.
    assert_eq!(
        follow_live_tab(false, Some(MetricKind::Xpsnr), MetricKind::Psnr),
        MetricKind::Psnr
    );
}

fn dec_pts(ys: &[f64]) -> Vec<egui_plot::PlotPoint> {
    ys.iter()
        .enumerate()
        .map(|(i, &y)| egui_plot::PlotPoint::new(i as f64 + 1.0, y))
        .collect()
}

fn ys(points: &egui_plot::PlotPoints) -> Vec<f64> {
    points.points().iter().map(|p| p.y).collect()
}

#[test]
fn decimate_passes_through_small_series() {
    let pts = dec_pts(&[1.0, 2.0, 3.0]);
    assert_eq!(ys(&decimate_minmax(&pts, 512)), vec![1.0, 2.0, 3.0]);
    assert!(decimate_minmax(&[], 512).points().is_empty());
}

/// Per-frame plot CPU budget check (5 rows × 5000 frames, the queue
/// shape the render loop re-scans every frame): `fit_limits` over the
/// value slices, `decimate_minmax` per series at viewport width, and
/// the two collected-vec builds. Measurement only — decides whether
/// caching the fit/decimation is worth any risk.
#[test]
fn per_frame_rebuild_budget() {
    use std::hint::black_box;
    let raw: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..5000)
                .map(|i| 30.0 + s as f64 + (i as f64 * 0.01).sin() * 5.0)
                .collect()
        })
        .collect();
    let borrowed: Vec<&[f64]> = raw.iter().map(Vec::as_slice).collect();
    let points: Vec<Vec<egui_plot::PlotPoint>> = raw.iter().map(|v| dec_pts(v)).collect();
    let n = 100;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        black_box(fit_limits(&borrowed, PSNR_LO, PSNR_HI));
    }
    let fit = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut lens = 0;
    for _ in 0..n {
        for p in &points {
            lens += decimate_minmax(p, 1000).points().len();
        }
    }
    let dec = t1.elapsed();
    black_box(lens);
    let t2 = std::time::Instant::now();
    let mut rows = 0;
    for _ in 0..n {
        let done: Vec<(&str, &[f64])> = raw.iter().map(|v| ("clip.mp4", v.as_slice())).collect();
        let thin: Vec<&[f64]> = done.iter().map(|(_, v)| *v).collect();
        rows += done.len() + thin.len();
    }
    let vecs = t2.elapsed();
    black_box(rows);
    eprintln!(
        "plot rebuild ({n} frames, 5x5000): fit {fit:?} | decimate {dec:?} | vec-builds {vecs:?}"
    );
}

#[test]
fn decimate_keeps_endpoints_and_spikes() {
    // 1001 points with a spike and a dip: thinned to ~100.
    let mut raw: Vec<f64> = (0..1001).map(|i| 40.0 + (i as f64 * 0.01).sin()).collect();
    raw[500] = 90.0;
    raw[700] = 10.0;
    let pts = dec_pts(&raw);
    let thin = decimate_minmax(&pts, 100);
    let got = thin.points();
    assert!(got.len() <= 104, "len {}", got.len());
    // Endpoints preserved (line meets the axes where it should).
    assert_eq!((got[0].x, got[got.len() - 1].x), (1.0, 1001.0));
    // Envelope preserved: spike and dip survive decimation.
    assert!(got.iter().any(|p| p.y == 90.0));
    assert!(got.iter().any(|p| p.y == 10.0));
    // Index order kept (valid line strip, no zigzag).
    assert!(got.windows(2).all(|w| w[0].x <= w[1].x));
}

#[test]
fn clamp_range_keeps_span_inside_limits() {
    // Inside: untouched. Past either edge: shifted, span kept.
    assert_eq!(clamp_range((2.0, 5.0), (1.0, 10.0)), (2.0, 5.0));
    assert_eq!(clamp_range((-3.0, 2.0), (1.0, 10.0)), (1.0, 6.0));
    assert_eq!(clamp_range((8.0, 15.0), (1.0, 10.0)), (3.0, 10.0));
    // Wider than limits (zoomed out): snap exactly to limits.
    assert_eq!(clamp_range((0.0, 100.0), (1.0, 10.0)), (1.0, 10.0));
    // Degenerate view or limits: snap / disable respectively.
    assert_eq!(clamp_range((5.0, 5.0), (1.0, 10.0)), (1.0, 10.0));
    assert_eq!(clamp_range((2.0, 5.0), (7.0, 7.0)), (2.0, 5.0));
}

#[test]
fn series_colors_are_stable_slots_not_positions() {
    // Removing/hiding the first curve must not recolor the rest: the
    // color derives from the permanent queue slot, never the visible
    // index. Slot 0 is the old red, slot 1 the old blue.
    assert_eq!(series_egui_color(1), series_egui_color(1));
    assert_eq!(series_plot_color(1), series_plot_color(1));
    // Live and export twins agree (same hue).
    let egui = series_egui_color(1);
    let plotters::style::RGBColor(r, g, b) = series_plot_color(1);
    assert_eq!((egui.r(), egui.g(), egui.b()), (r, g, b));
    // Golden-ratio neighbors stay distinct even with ~20 series.
    let slots: Vec<_> = (0..20).map(series_egui_color).collect();
    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            assert_ne!(slots[i], slots[j], "slots {i} and {j} collide");
        }
    }
}

fn png_bytes(path: &std::path::Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"), "PNG magic");
    assert!(bytes.len() > 1024, "non-trivial image");
    bytes
}

#[test]
fn export_png_writes_valid_image() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-png-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("PSNR.png");
    let a = [46.0, 48.5, 47.0, 49.0];
    let b = [44.0, 45.0, 46.5, 45.5];
    export_png(
        &path,
        "PSNR",
        "PSNR (higher is better, min 0, max 100)",
        &[("a.mkv", 0, &a[..]), ("b.mkv", 1, &b[..])],
        ((1.0, 4.0), (44.0, 49.0)),
        (400, 300),
    )
    .unwrap();
    png_bytes(&path);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn smallest_preset_renders_exact_dims() {
    // 1280x320 exercises the scaled-down fonts/margins/legend (would
    // panic on zero-size text); long series name stresses the legend.
    let vals: Vec<f64> = (0..300)
        .map(|i| 45.0 + (i as f64 * 0.37).sin() * 2.0)
        .collect();
    let (w, h, rgba) = render_rgba(
        "PSNR",
        "PSNR (higher is better, min 0, max 100)",
        &[(
            "output-[2026-08-28] Sample_Encode_Test q70.mkv",
            0,
            &vals[..],
        )],
        ((1.0, 300.0), (43.0, 49.0)),
        PlotSize::S1280.dims(),
    )
    .unwrap();
    assert_eq!((w, h), (1280, 320));
    assert_eq!(rgba.len(), 1280 * 320 * 4);
}

#[test]
fn export_png_tolerates_empty_and_degenerate() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-png-edge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // No series: axes only. Degenerate view: sanitized, no panic.
    let empty: &[(&str, usize, &[f64])] = &[];
    export_png(
        &dir.join("empty.png"),
        "SSIM",
        "SSIM",
        empty,
        ((0.0, 1.0), (0.0, 1.0)),
        (400, 300),
    )
    .unwrap();
    let one = [0.9];
    export_png(
        &dir.join("one.png"),
        "SSIM",
        "SSIM",
        &[("a.mkv", 0, &one[..])],
        ((1.0, 1.0), (0.9, 0.9)),
        (400, 300),
    )
    .unwrap();
    png_bytes(&dir.join("empty.png"));
    png_bytes(&dir.join("one.png"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn view_window_widens_by_one_point() {
    // (start, end) as slice bounds over 1-based frame x.
    assert_eq!(view_window(10, 1.0, 10.0), (0, 10)); // full range
    assert_eq!(view_window(10, 3.0, 5.0), (1, 6)); // ±1 margin
    assert_eq!(view_window(10, 9.0, 20.0), (7, 10)); // past the end
    assert_eq!(view_window(10, -5.0, 2.0), (0, 3)); // before the start
    assert_eq!(view_window(0, 1.0, 5.0), (0, 0)); // empty
    assert_eq!(view_window(1, 5.0, 6.0), (0, 1)); // single point
}

/// Stable export line color for the `"a.mkv"` series used below.
fn line_rgb() -> (f32, f32, f32) {
    let plotters::style::RGBColor(r, g, b) = series_plot_color(0);
    (r as f32, g as f32, b as f32)
}

/// `(dist², t)` from a pixel to the bg→line segment: anti-aliased edge
/// blends sit near this segment with `0 < t < 1`; bg, grid, line core
/// and white text do not.
fn seg_dt2(p: &image::Rgb<u8>, bg: (f32, f32, f32), line: (f32, f32, f32)) -> (f32, f32) {
    let ab = (line.0 - bg.0, line.1 - bg.1, line.2 - bg.2);
    let ap = (p[0] as f32 - bg.0, p[1] as f32 - bg.1, p[2] as f32 - bg.2);
    let len2 = ab.0 * ab.0 + ab.1 * ab.1 + ab.2 * ab.2;
    let t = ((ap.0 * ab.0 + ap.1 * ab.1 + ap.2 * ab.2) / len2).clamp(0.0, 1.0);
    let d2 = (ap.0 - t * ab.0).powi(2) + (ap.1 - t * ab.1).powi(2) + (ap.2 - t * ab.2).powi(2);
    (d2, t)
}

#[test]
fn export_aa_blends_line_edges() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-png-aa-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Diagonal-heavy series: long slanted segments show AA blends.
    let vals: Vec<f64> = (0..500).map(|i| 45.0 + i as f64 * 0.008).collect();
    let path = dir.join("aa.png");
    export_png(
        &path,
        "PSNR",
        "PSNR",
        &[("a.mkv", 0, &vals[..])],
        ((1.0, 500.0), (44.0, 50.0)),
        (800, 400),
    )
    .unwrap();
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    // Edge blends: near the bg→line segment but strictly between the
    // endpoints — neither bg (20s), grid (42s), line core nor white text.
    let (bg, line) = ((20.0, 20.0, 20.0), line_rgb());
    let mut blends = 0u64;
    for x in 100..w - 40 {
        for y in 40..h - 40 {
            let (d2, t) = seg_dt2(&img[(x, y)], bg, line);
            if d2 < 25.0 * 25.0 && (0.05..0.95).contains(&t) {
                blends += 1;
            }
        }
    }
    assert!(blends > 300, "supersampled edges expected");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fullrange_export_stays_readable_when_dense() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-png-blob-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Dense spiky full-range series like a zoomed-out PSNR run.
    let n = 5550usize;
    let vals: Vec<f64> = (0..n)
        .map(|i| 47.0 + ((i as f64 * 0.13).sin() * 1.8 + (i % 29) as f64 * 0.03))
        .collect();
    let path = dir.join("full.png");
    export_png(
        &path,
        "PSNR",
        "PSNR",
        &[("a.mkv", 0, &vals[..])],
        ((1.0, n as f64), (44.0, 51.0)),
        (1600, 400),
    )
    .unwrap();
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let line = line_rgb();
    let is_line = |p: &image::Rgb<u8>| {
        let d2 = (p[0] as f32 - line.0).powi(2)
            + (p[1] as f32 - line.1).powi(2)
            + (p[2] as f32 - line.2).powi(2);
        d2 < 60.0 * 60.0
    };
    // Interior plot columns only (skip y-label gutter + legend corner).
    let mut red = 0u64;
    let mut tot = 0u64;
    for x in 150..w - 60 {
        for y in 60..h - 60 {
            tot += 1;
            if is_line(&img[(x, y)]) {
                red += 1;
            }
        }
    }
    let frac = red as f64 / tot as f64;
    // Thinned width-2 strokes stay readable; full-res width-3 hit ~0.28.
    assert!(frac < 0.25, "dense full-range export overpaints: {frac:.3}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn zoomed_export_has_no_edge_waterfalls() {
    let dir = std::env::temp_dir().join(format!("rfmetrics-png-zoom-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Spiky series like PSNR, zoomed into the middle third: out-of-view
    // points must not pile onto the plot-area edges.
    let n = 2000usize;
    let vals: Vec<f64> = (0..n)
        .map(|i| 47.0 + ((i as f64 * 0.11).sin() * 1.5 + (i % 37) as f64 * 0.02))
        .collect();
    let path = dir.join("zoom.png");
    export_png(
        &path,
        "PSNR",
        "PSNR",
        &[("a.mkv", 0, &vals[..])],
        ((700.0, 1300.0), (44.0, 51.0)),
        (800, 400),
    )
    .unwrap();
    let img = image::open(&path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let line = line_rgb();
    let is_line = |p: &image::Rgb<u8>| {
        let d2 = (p[0] as f32 - line.0).powi(2)
            + (p[1] as f32 - line.1).powi(2)
            + (p[2] as f32 - line.2).powi(2);
        d2 < 60.0 * 60.0
    };
    let mut max_col = 0u32;
    let mut mid = 0u32;
    for x in 0..w {
        let mut c = 0u32;
        for y in 0..h {
            if is_line(&img[(x, y)]) {
                c += 1;
            }
        }
        max_col = max_col.max(c);
        if (w / 2..w / 2 + 30).contains(&x) {
            mid += c;
        }
    }
    // Curve is really drawn, but no column carries a streak (the bug
    // stacked ~100+ red px on the edge columns).
    assert!(mid > 100, "curve drawn, mid30={mid}");
    assert!(max_col <= 80, "no waterfall column, max={max_col}");
    std::fs::remove_dir_all(&dir).ok();
}
