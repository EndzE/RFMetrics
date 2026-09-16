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

#[cfg(test)]
mod tests {
    use super::*;

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
}
