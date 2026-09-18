//! Plot helpers: `_fit_limits` (+ `_y_fit` padding rule) / `PLOT_DEFS` parity with
//! `FFMetrics-rev/main.py`.

use crate::metrics::ffmpeg::MetricKind;

/// Python `PLOT_DEFS` PSNR entry: label and fallback range.
pub const PSNR_LABEL: &str = "PSNR (higher is better, min 0, max 100)";
pub const PSNR_LO: f64 = 0.0;
pub const PSNR_HI: f64 = 100.0;

/// One Python `PLOT_DEFS` row: axis label + default y range.
pub struct PlotDef {
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
}

/// Python `PLOT_DEFS` verbatim (label, `lo`, `hi` per metric).
pub fn plot_def(kind: MetricKind) -> PlotDef {
    match kind {
        MetricKind::Psnr => PlotDef {
            label: PSNR_LABEL,
            lo: PSNR_LO,
            hi: PSNR_HI,
        },
        MetricKind::Ssim => PlotDef {
            label: "SSIM (higher is better, min 0, max 1)",
            lo: 0.0,
            hi: 1.0,
        },
        MetricKind::Vmaf => PlotDef {
            label: "VMAF (higher is better, min 0, max 100)",
            lo: 0.0,
            hi: 100.0,
        },
        MetricKind::Xpsnr => PlotDef {
            label: "XPSNR (higher is better, min 0, max 100)",
            lo: 0.0,
            hi: 100.0,
        },
        MetricKind::Ssim2 => PlotDef {
            label: "SSIMULACRA2 (higher is better, min 0, max 100)",
            lo: 0.0,
            hi: 100.0,
        },
        MetricKind::But => PlotDef {
            label: "BUTTERAUGLI (lower is better, min 0)",
            lo: 0.0,
            hi: 10.0,
        },
        MetricKind::Cvvdp => PlotDef {
            label: "CVVDP (cumulative JOD, higher is better, 10 = identical)",
            lo: 0.0,
            hi: 10.0,
        },
    }
}

/// Python tab titles (`"SSIM2"` for ssimulacra2, else the uppercased
/// key — note `BUTTERAUGLI`, not the queue header's `BUTTER`).
pub fn tab_title(kind: MetricKind) -> &'static str {
    match kind {
        MetricKind::Psnr => "PSNR",
        MetricKind::Ssim => "SSIM",
        MetricKind::Vmaf => "VMAF",
        MetricKind::Xpsnr => "XPSNR",
        MetricKind::Ssim2 => "SSIM2",
        MetricKind::But => "BUTTERAUGLI",
        MetricKind::Cvvdp => "CVVDP",
    }
}

/// Python `_y_fit` padding rule for a data range (min/max known): 5% of
/// the span; a flat series pads by `|max| * 2%`, falling back to `0.5`
/// at zero (Python `abs(mx) * 0.02 or 0.5`).
fn pad_for(mn: f64, mx: f64) -> f64 {
    let span = mx - mn;
    if span > 0.0 {
        span * 0.05
    } else {
        // Python `abs(mx) * 0.02 or 0.5`: falsy (0.0) falls back to 0.5.
        let p = mx.abs() * 0.02;
        if p > 0.0 { p } else { 0.5 }
    }
}

/// Python `_fit_limits`: x is `(1, N)` over the longest series, y is the
/// `_y_fit` rule over all values. No values at all yields no x fit and
/// the metric default y range (empty plot, axes only).
/// Allocation-free: scans borrowed slices, never concatenates.
pub fn fit_limits(series: &[&[f64]], lo: f64, hi: f64) -> FitBounds {
    let mut n = 0usize;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut count = 0usize;
    for s in series {
        n = n.max(s.len());
        for &v in *s {
            mn = mn.min(v);
            mx = mx.max(v);
            count += 1;
        }
    }
    if count == 0 {
        return (None, (lo, hi));
    }
    // egui_plot panics on degenerate `min >= max` bounds: a single point
    // (first live batch, 1-frame run) centers in (0.5, 1.5) instead.
    let x = Some(if n <= 1 { (0.5, 1.5) } else { (1.0, n as f64) });
    let pad = pad_for(mn, mx);
    (x, (mn - pad, mx + pad))
}

/// Fitted view bounds: optional x span (absent when dataless) plus y span.
pub type FitBounds = (Option<(f64, f64)>, (f64, f64));

/// Tab-follow while a run is live: switch to the metric currently being
/// computed so its growing curve is visible; untouched otherwise (user
/// picks freely when idle, and after the run the last tab stays put).
pub fn follow_live_tab(
    measuring: bool,
    live: Option<MetricKind>,
    current: MetricKind,
) -> MetricKind {
    if measuring {
        live.unwrap_or(current)
    } else {
        current
    }
}

/// Min-max decimation for drawing: buckets the points and keeps each
/// bucket's min and max (in index order), so spikes survive while the
/// tessellator sees ~`target` points instead of the full series. Returns
/// a borrow when already small. Hover/fit keep using the full values —
/// only the drawn line is thinned.
pub fn decimate_minmax(
    points: &[egui_plot::PlotPoint],
    target: usize,
) -> egui_plot::PlotPoints<'_> {
    let target = target.max(4);
    if points.len() <= target {
        return egui_plot::PlotPoints::Borrowed(points);
    }
    let n = points.len();
    let buckets = target / 2;
    let mut out = Vec::with_capacity(target);
    out.push(points[0]);
    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1).min(n);
        let (mut lo, mut hi) = (start, start);
        for i in start + 1..end {
            if points[i].y < points[lo].y {
                lo = i;
            }
            if points[i].y > points[hi].y {
                hi = i;
            }
        }
        if lo < hi {
            out.push(points[lo]);
            out.push(points[hi]);
        } else if hi < lo {
            out.push(points[hi]);
            out.push(points[lo]);
        } else {
            out.push(points[lo]);
        }
    }
    if out.last() != Some(&points[n - 1]) {
        out.push(points[n - 1]);
    }
    egui_plot::PlotPoints::Owned(out)
}

/// Clamp a view range into hard limits (snap-to-data lock): the view
/// keeps its span and is shifted inside; a wider-than-limits (or
/// degenerate) view snaps exactly to the limits; degenerate limits
/// disable clamping on that axis.
pub fn clamp_range(view: (f64, f64), lim: (f64, f64)) -> (f64, f64) {
    let (v0, v1) = view;
    let (l0, l1) = lim;
    // Degenerate (or NaN) limits: no clamping on this axis.
    if l0.is_nan() || l1.is_nan() || l1 <= l0 {
        return view;
    }
    let span = v1 - v0;
    // Degenerate, NaN, or wider-than-limits view: snap exactly.
    if v0.is_nan() || v1.is_nan() || span <= 0.0 || span >= l1 - l0 {
        return lim;
    }
    let s0 = v0.clamp(l0, l1 - span);
    (s0, s0 + span)
}

/// Max screen distance for a hover hit (Python `best[0] > 30` parity:
/// anything farther hides the crosshair).
pub const HOVER_MAX_PX: f32 = 30.0;

/// Python `_on_hover` nearest-point search over 1-based frame series:
/// candidates within ±2 frames of the pointer x in every series, ranked
/// by Manhattan screen distance through `to_screen`. A hit within
/// `HOVER_MAX_PX` yields `(series_idx, frame_1based, value)`.
pub fn nearest_hover(
    series: &[&[f64]],
    x: f64,
    to_screen: impl Fn(f64, f64) -> (f32, f32),
    mouse: (f32, f32),
) -> Option<(usize, i64, f64)> {
    let mut best: Option<(f32, usize, i64, f64)> = None;
    for (si, values) in series.iter().enumerate() {
        if values.is_empty() {
            continue;
        }
        let center = (x.round() as isize - 1).clamp(0, values.len() as isize - 1);
        for j in (center - 2).max(0)..=(center + 2).min(values.len() as isize - 1) {
            let fx = (j + 1) as f64;
            let fy = values[j as usize];
            let (sx, sy) = to_screen(fx, fy);
            let d = (sx - mouse.0).abs() + (sy - mouse.1).abs();
            if best.is_none_or(|(bd, _, _, _)| d < bd) {
                best = Some((d, si, (j + 1) as i64, fy));
            }
        }
    }
    match best {
        Some((d, si, frame, value)) if d <= HOVER_MAX_PX => Some((si, frame, value)),
        _ => None,
    }
}
/// Union of two fitted ranges, per axis: bounds only ever grow. Used
/// for live curves so axes don't jump inward as new points arrive. A
/// side with no data (`None` x fit) contributes nothing, so an empty
/// tab can't dilute a fitted one.
pub fn union_bounds(a: FitBounds, b: FitBounds) -> FitBounds {
    match (a.0, b.0) {
        (Some((a0, a1)), Some((b0, b1))) => (
            Some((a0.min(b0), a1.max(b1))),
            ((a.1).0.min((b.1).0), (a.1).1.max((b.1).1)),
        ),
        (Some(_), None) => a,
        (None, Some(_)) => b,
        (None, None) => a,
    }
}

/// egui_plot line auto-color, same hue ladder as `PlotUi::auto_color`
/// (`Hsva::new(i * (φ-1), …)`), but brightened for export contrast: the
/// verbatim value (v=0.5) renders nearly invisible dark red on a dark
/// canvas in stills.
pub fn egui_auto_color(i: usize) -> plotters::style::RGBColor {
    use std::f32::consts::GOLDEN_RATIO;
    let h = (i as f32 * (GOLDEN_RATIO - 1.0)) % 1.0;
    let (r, g, b) = hsv_to_rgb(h, 0.9, 0.85);
    plotters::style::RGBColor(r, g, b)
}

/// `h` in [0, 1): standard HSV→RGB, full opacity.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0).round() as u8,
        ((g + m) * 255.0).round() as u8,
        ((b + m) * 255.0).round() as u8,
    )
}

/// Indices of `values` (1-based frame x) overlapping `[x0, x1]`, widened
/// by one point on each side so edge-crossing segments still render.
/// Points outside would otherwise pile onto the plot-area edge (plotters
/// clamps out-of-range coordinates), drawing the "waterfall" streaks.
fn view_window(n: usize, x0: f64, x1: f64) -> (usize, usize) {
    let mut start = 0usize;
    while start < n && ((start + 1) as f64) < x0 {
        start += 1;
    }
    let start = start.saturating_sub(1);
    let mut end = start;
    while end < n && ((end + 1) as f64) <= x1 {
        end += 1;
    }
    (start, (end + 1).min(n))
}

/// Export image size presets for Save PNG / Copy (Options combobox).
/// All wide 4:1 — plots are frames-wide, so height stays small.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlotSize {
    S1280,
    S1600,
    S2400,
    #[default]
    S3200,
}

impl PlotSize {
    /// Combo order: smallest first, current default last.
    pub const ALL: [PlotSize; 4] = [
        PlotSize::S1280,
        PlotSize::S1600,
        PlotSize::S2400,
        PlotSize::S3200,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::S1280 => "1280×320",
            Self::S1600 => "1600×400",
            Self::S2400 => "2400×600",
            Self::S3200 => "3200×800",
        }
    }

    pub fn dims(self) -> (u32, u32) {
        match self {
            Self::S1280 => (1280, 320),
            Self::S1600 => (1600, 400),
            Self::S2400 => (2400, 600),
            Self::S3200 => (3200, 800),
        }
    }

    /// State-file validation (unknown labels keep the live default).
    pub fn from_label(s: &str) -> Option<PlotSize> {
        Self::ALL.into_iter().find(|m| m.label() == s)
    }
}

/// Render the current tab to a PNG file in egui-plot style (dark canvas,
/// white axes, same auto-colors and lower-right legend). `view` is the
/// on-screen range (pan/zoom respected); degenerate spans fall back to a
/// unit span instead of erroring. Empty series still produce axes.
///
/// Anti-aliasing: plotters' bitmap backend draws aliased strokes, so the
/// chart renders supersampled (2x at default size, more when small) and
/// Lanczos-downscales (real supersampled edges).
pub fn export_png(
    path: &std::path::Path,
    title: &str,
    y_label: &str,
    series: &[(&str, &[f64])],
    view: ((f64, f64), (f64, f64)),
    size: (u32, u32),
) -> Result<(), String> {
    let (w, h, rgba) = render_rgba(title, y_label, series, view, size)?;
    let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(w, h, rgba)
        .ok_or_else(|| "render buffer mismatch".to_owned())?;
    img.save(path).map_err(|e| e.to_string())?;
    Ok(())
}

/// Same render as [`export_png`], but returns raw RGBA pixels at `size`
/// instead of writing a file — feeds `ctx.copy_image()` without touching
/// disk. Returns `(width, height, rgba_bytes)` with opaque alpha.
pub fn render_rgba(
    title: &str,
    y_label: &str,
    series: &[(&str, &[f64])],
    view: ((f64, f64), (f64, f64)),
    size: (u32, u32),
) -> Result<(u32, u32, Vec<u8>), String> {
    use plotters::prelude::*;
    let ((x0, x1), (y0, y1)) = view;
    // Degenerate spans fall back instead of erroring (plotters requires
    // strict ranges); a finite point expands around itself.
    fn strict_span(v: (f64, f64)) -> (f64, f64) {
        let (a, b) = v;
        if !a.is_nan() && !b.is_nan() && b > a {
            (a, b)
        } else if a.is_finite() {
            (a, a + 1.0)
        } else {
            (0.0, 1.0)
        }
    }
    let (x0, x1) = strict_span((x0, x1));
    let (y0, y1) = strict_span((y0, y1));
    // Supersample, then Lanczos-downscale (plotters draws aliased
    // strokes): real anti-aliased edges like matplotlib's. All pixel
    // sizes below are pre-scale.
    //
    // Adaptive factor: small presets supersample more (4x at 1280-wide)
    // so downscaled small text stays crisp instead of pixelated. Capped
    // so the buffer never exceeds the 3200-wide default's (~10M px):
    // smaller exports stay faster than the default while matching its
    // per-pixel sample count.
    let ssaa: u32 = (5120 / size.0.max(1)).clamp(2, 4);
    // Layout scale: font/margin sizes are tuned for the 3200-wide
    // default; smaller presets shrink them proportionally so axis labels
    // and the legend fit instead of overflowing. All presets share the
    // 4:1 aspect, so width alone sets the factor (1.0 at default keeps
    // the old pixels exactly).
    let layout = size.0 as f32 / 3200.0;
    let fp = |n: u32| ((n as f32 * layout).round() as u32).max(1) * ssaa;
    let (bw, bh) = (size.0 * ssaa, size.1 * ssaa);
    let mut buf = vec![0u8; (bw * bh * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buf, (bw, bh)).into_drawing_area();
        root.fill(&RGBColor(20, 20, 20))
            .map_err(|e| e.to_string())?;
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", fp(40)).into_font().color(&WHITE))
            .margin(fp(12))
            .x_label_area_size(fp(48))
            .y_label_area_size(fp(100))
            .build_cartesian_2d(x0..x1, y0..y1)
            .map_err(|e| e.to_string())?;
        chart
            .configure_mesh()
            .x_desc("Frames")
            .y_desc(y_label)
            .x_labels(16)
            .y_labels(8)
            .axis_style(WHITE)
            .label_style(("sans-serif", fp(24)).into_font().color(&WHITE))
            .light_line_style(RGBColor(42, 42, 42))
            .draw()
            .map_err(|e| e.to_string())?;
        for (i, (name, values)) in series.iter().enumerate() {
            let color = egui_auto_color(i);
            // Draw only the visible window (±1 point for clean edge
            // crossings): out-of-range points pile onto the plot-area edge.
            let (lo, hi) = view_window(values.len(), x0, x1);
            let slice = &values[lo..hi];
            // Thin to ~1 point per pixel like the live view: a dense spiky
            // series at full resolution overpaints every column into a
            // solid band. Hover/fit are unaffected.
            let pts: Vec<egui_plot::PlotPoint> = slice
                .iter()
                .enumerate()
                .map(|(j, &v)| egui_plot::PlotPoint::new(lo as f64 + j as f64 + 1.0, v))
                .collect();
            let thin = decimate_minmax(&pts, size.0 as usize);
            chart
                .draw_series(LineSeries::new(
                    thin.points().iter().map(|p| (p.x, p.y)),
                    color.stroke_width(fp(2)),
                ))
                .map_err(|e| e.to_string())?
                .label(*name)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + fp(30) as i32, y)], color)
                });
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::LowerRight)
            .background_style(RGBColor(30, 30, 30).mix(0.85))
            .border_style(WHITE)
            .label_font(("sans-serif", fp(24)).into_font().color(&WHITE))
            .draw()
            .map_err(|e| e.to_string())?;
        root.present().map_err(|e| e.to_string())?;
    }
    let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(bw, bh, buf)
        .ok_or_else(|| "supersample buffer mismatch".to_owned())?;
    let small =
        image::imageops::resize(&img, size.0, size.1, image::imageops::FilterType::Lanczos3);
    // Opaque alpha: the plot canvas has no transparency.
    let mut rgba = Vec::with_capacity((size.0 * size.1 * 4) as usize);
    for p in small.pixels() {
        rgba.extend_from_slice(&[p[0], p[1], p[2], 0xFF]);
    }
    Ok((size.0, size.1, rgba))
}

#[cfg(test)]
mod tests {
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
            let done: Vec<(&str, &[f64])> =
                raw.iter().map(|v| ("clip.mp4", v.as_slice())).collect();
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
    fn auto_colors_follow_egui_hue_ladder() {
        use plotters::style::RGBColor;
        // i=0: hue 0 (red); i=1: hue φ-1 ≈ 0.618 (blue). Brightened for
        // export contrast, hue order kept.
        assert_eq!(egui_auto_color(0), RGBColor(217, 22, 22));
        let RGBColor(r1, g1, b1) = egui_auto_color(1);
        assert!(b1 > r1 && b1 > g1, "second curve is blue-ish");
        // Deterministic per index.
        assert_eq!(egui_auto_color(3), egui_auto_color(3));
        assert_ne!(egui_auto_color(0), egui_auto_color(1));
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
            &[("a.mkv", &a[..]), ("b.mkv", &b[..])],
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
        let empty: &[(&str, &[f64])] = &[];
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
            &[("a.mkv", &one[..])],
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
            &[("a.mkv", &vals[..])],
            ((1.0, 500.0), (44.0, 50.0)),
            (800, 400),
        )
        .unwrap();
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        // Mid-tone reds: neither bg (20s), grid (42s), line core (217,22,22)
        // nor white text — only anti-aliased edge blends land here.
        let mut blends = 0u64;
        for x in 100..w - 40 {
            for y in 40..h - 40 {
                let p = &img[(x, y)];
                if (60..190).contains(&p[0]) && p[1] < 60 && p[2] < 60 {
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
            &[("a.mkv", &vals[..])],
            ((1.0, n as f64), (44.0, 51.0)),
            (1600, 400),
        )
        .unwrap();
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        let is_red = |p: &image::Rgb<u8>| p[0] > 150 && p[1] < 80 && p[2] < 80;
        // Interior plot columns only (skip y-label gutter + legend corner).
        let mut red = 0u64;
        let mut tot = 0u64;
        for x in 150..w - 60 {
            for y in 60..h - 60 {
                tot += 1;
                if is_red(&img[(x, y)]) {
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
            &[("a.mkv", &vals[..])],
            ((700.0, 1300.0), (44.0, 51.0)),
            (800, 400),
        )
        .unwrap();
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        let is_red = |p: &image::Rgb<u8>| p[0] > 150 && p[1] < 80 && p[2] < 80;
        let mut max_col = 0u32;
        let mut mid = 0u32;
        for x in 0..w {
            let mut c = 0u32;
            for y in 0..h {
                if is_red(&img[(x, y)]) {
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
}
