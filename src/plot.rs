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

/// Stable per-file color from a permanent queue slot (assigned at insert,
/// never reused): hiding or removing one curve never recolors the rest.
/// Hue walks the golden-ratio ladder — the same sequence as egui_plot's
/// `auto_color` — so neighbors stay maximally distinct even with ~20
/// series. Same saturation/brightness as before for export contrast.
pub fn series_plot_color(slot: usize) -> plotters::style::RGBColor {
    use std::f32::consts::GOLDEN_RATIO;
    let h = (slot as f32 * (GOLDEN_RATIO - 1.0)) % 1.0;
    let (r, g, b) = hsv_to_rgb(h, 0.9, 0.85);
    plotters::style::RGBColor(r, g, b)
}

/// egui twin of `series_plot_color` (same hue): live lines match exports.
pub fn series_egui_color(slot: usize) -> egui::Color32 {
    use std::f32::consts::GOLDEN_RATIO;
    let h = (slot as f32 * (GOLDEN_RATIO - 1.0)) % 1.0;
    let (r, g, b) = hsv_to_rgb(h, 0.9, 0.85);
    egui::Color32::from_rgb(r, g, b)
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
/// white axes, same stable per-file colors as the live view, lower-right
/// legend). `view` is the on-screen range (pan/zoom respected); degenerate
/// spans fall back to a unit span instead of erroring. Empty series still
/// produce axes.
///
/// `series` is `(legend name, color slot, values)`: the slot (queue
/// `color_idx`) picks the color so exports match the live lines even when
/// some rows are hidden; the name is display-only.
///
/// Anti-aliasing: the chart renders to an in-memory SVG vector and
/// rasterizes it at target size via resvg/tiny-skia (exact subpixel
/// geometric coverage, no supersample buffer).
pub fn export_png(
    path: &std::path::Path,
    title: &str,
    y_label: &str,
    series: &[(&str, usize, &[f64])],
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
    series: &[(&str, usize, &[f64])],
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
    // Layout scale: font/margin sizes are tuned for the 3200-wide
    // default; smaller presets shrink them proportionally so axis labels
    // and the legend fit instead of overflowing. All presets share the
    // 4:1 aspect, so width alone sets the factor (1.0 at default keeps
    // the old pixels exactly).
    let layout = size.0 as f32 / 3200.0;
    let fp = |n: u32| ((n as f32 * layout).round() as u32).max(1);
    // Floors so the bottom stack (tick labels top-anchored, "Frames"
    // desc bottom-anchored) never converges: both only bite below
    // 2400-wide, 2400/3200 keep their exact current pixels.
    let tick = fp(24).max(13);
    let x_area = fp(48).max(36);
    // 1. Render chart to an in-memory SVG string (vector paths with
    // exact stroke widths, bypassing the bitmap backend's integer
    // snapping).
    let mut svg_buffer = String::with_capacity(64 * 1024);
    {
        let root = plotters_svg::SVGBackend::with_string(&mut svg_buffer, size).into_drawing_area();
        root.fill(&RGBColor(20, 20, 20))
            .map_err(|e| e.to_string())?;
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", fp(40)).into_font().color(&WHITE))
            .margin(fp(12))
            .x_label_area_size(x_area)
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
            .label_style(("sans-serif", tick).into_font().color(&WHITE))
            .light_line_style(RGBColor(42, 42, 42))
            .draw()
            .map_err(|e| e.to_string())?;
        for (name, slot, values) in series.iter() {
            let color = series_plot_color(*slot);
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
            .label_font(("sans-serif", tick).into_font().color(&WHITE))
            .draw()
            .map_err(|e| e.to_string())?;
        root.present().map_err(|e| e.to_string())?;
    }
    // 2. Rasterize the SVG with resvg (analytical coverage
    // anti-aliasing). Small presets rasterize at 2x and downscale:
    // sub-20px glyphs are inherently crunchy at 1x, while 2x hinted
    // glyphs downscale smooth (the old SSAA text effect at a quarter
    // of its cost). 2400-wide and up stay exact 1x.
    // ponytail: system-font scan once per process; reload per call if
    // missing glyphs ever appear in exports.
    static FONTDB: std::sync::OnceLock<std::sync::Arc<resvg::usvg::fontdb::Database>> =
        std::sync::OnceLock::new();
    let fontdb = FONTDB
        .get_or_init(|| {
            let mut db = resvg::usvg::fontdb::Database::new();
            db.load_system_fonts();
            std::sync::Arc::new(db)
        })
        .clone();
    let opt = resvg::usvg::Options {
        fontdb,
        ..resvg::usvg::Options::default()
    };
    let tree = resvg::usvg::Tree::from_str(&svg_buffer, &opt)
        .map_err(|e| format!("SVG parse error: {e}"))?;
    let k: u32 = if size.0 < 2400 { 2 } else { 1 };
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.0 * k, size.1 * k)
        .ok_or_else(|| "Failed to allocate rasterizer buffer".to_owned())?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(k as f32, k as f32),
        &mut pixmap.as_mut(),
    );
    // 3. Extract pixels. The canvas has an opaque background
    // (RGB(20,20,20)), so premultiplied RGBA from tiny-skia is identical
    // to straight RGBA.
    if k == 1 {
        let rgba = pixmap.take();
        return Ok((size.0, size.1, rgba));
    }
    let big =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(size.0 * k, size.1 * k, pixmap.take())
            .ok_or_else(|| "rasterizer buffer mismatch".to_owned())?;
    let small =
        image::imageops::resize(&big, size.0, size.1, image::imageops::FilterType::Lanczos3);
    Ok((size.0, size.1, small.into_raw()))
}
#[cfg(test)]
#[path = "tests/test_plot.rs"]
mod tests;
