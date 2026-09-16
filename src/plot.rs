//! Plot helpers: `_y_fit` / `_fit_limits` / `PLOT_DEFS` parity with
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
        MetricKind::But => "BUTTER",
        MetricKind::Cvvdp => "CVVDP",
    }
}

/// Python `_y_fit`: data min/max padded by 5% of the span; a flat
/// series pads by `|max| * 2%`, falling back to `0.5` at zero.
/// Empty input keeps the metric default `(lo, hi)`.
pub fn y_fit(values: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    if values.is_empty() {
        return (lo, hi);
    }
    let mut mn = values[0];
    let mut mx = values[0];
    for &v in &values[1..] {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let span = mx - mn;
    let pad = if span > 0.0 {
        span * 0.05
    } else {
        // Python `abs(mx) * 0.02 or 0.5`: falsy (0.0) falls back to 0.5.
        let p = mx.abs() * 0.02;
        if p > 0.0 { p } else { 0.5 }
    };
    (mn - pad, mx + pad)
}

/// Python `_fit_limits`: x is `(1, N)` over the longest series, y is
/// `y_fit` over all values concatenated. No values at all yields no
/// x fit and the metric default y range (empty plot, axes only).
pub fn fit_limits(series: &[&[f64]], lo: f64, hi: f64) -> (Option<(f64, f64)>, (f64, f64)) {
    let mut n = 0usize;
    let mut count = 0usize;
    for s in series {
        n = n.max(s.len());
        count += s.len();
    }
    if count == 0 {
        return (None, (lo, hi));
    }
    let mut all = Vec::with_capacity(count);
    for s in series {
        all.extend_from_slice(s);
    }
    (Some((1.0, n as f64)), y_fit(&all, lo, hi))
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_keeps_metric_defaults() {
        assert_eq!(y_fit(&[], PSNR_LO, PSNR_HI), (0.0, 100.0));
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
        let (lo, hi) = y_fit(&[30.0, 40.0, 35.0], PSNR_LO, PSNR_HI);
        assert!((lo - 29.5).abs() < 1e-9);
        assert!((hi - 40.5).abs() < 1e-9);
    }

    #[test]
    fn flat_series_pads_by_magnitude() {
        // span 0, |40| * 2% = 0.8.
        let (lo, hi) = y_fit(&[40.0, 40.0], PSNR_LO, PSNR_HI);
        assert!((lo - 39.2).abs() < 1e-9);
        assert!((hi - 40.8).abs() < 1e-9);
    }

    #[test]
    fn flat_zero_series_pads_by_half() {
        // Python `0.0 or 0.5` fallback.
        assert_eq!(y_fit(&[0.0, 0.0], PSNR_LO, PSNR_HI), (-0.5, 0.5));
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
}
