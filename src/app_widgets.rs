//! Shared UI primitives for the `RFMetricsApp` windows.
//!
//! Extracted from `app.rs`: toast types, cursor helper, and the small pure
//! `egui` helpers (`vline`, `sort_mark`, `panel_frame`, rank fills, running
//! sweep, stat tooltips). `app.rs` re-exports these so existing
//! `crate::app::…` paths keep working.

use crate::app_queue::SortDir;

/// Retrieves the cursor position in egui's logical point coordinates.
/// During Windows OLE file drags, winit omits pointer move events, so egui's
/// internal pointer state is None/stale. We query the OS cursor directly.
#[cfg(windows)]
pub(crate) fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    #[repr(C)]
    struct Point {
        x: i32,
        y: i32,
    }
    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetCursorPos(lpPoint: *mut Point) -> i32;
    }

    let mut pt = Point { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut pt) } != 0 {
        let ppp = ctx.pixels_per_point();
        let screen_pos = egui::pos2(pt.x as f32 / ppp, pt.y as f32 / ppp);
        if let Some(inner_rect) = ctx.input(|i| i.viewport().inner_rect) {
            return Some(egui::pos2(
                screen_pos.x - inner_rect.min.x,
                screen_pos.y - inner_rect.min.y,
            ));
        }
    }
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

#[cfg(not(windows))]
pub(crate) fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

/// Hover highlight delay (seconds) so passing over rows while aiming
/// at text to copy doesn't flash each row.
pub(crate) const ROW_HOVER_DELAY: f64 = 0.1;

/// How long the drop toast (e.g. ignored extra reference files) stays up.
pub(crate) const TOAST_SECS: f64 = 3.0;

/// Toast severity; drives the outline color. `Info` keeps the default
/// popup outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToastKind {
    Info,
    Warning,
    Error,
}

impl ToastKind {
    pub(crate) fn outline(self) -> Option<egui::Color32> {
        match self {
            ToastKind::Info => None,
            ToastKind::Warning => Some(egui::Color32::from_rgb(0xD9, 0xA4, 0x06)),
            ToastKind::Error => Some(egui::Color32::from_rgb(0xE0, 0x4B, 0x4B)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Toast {
    pub(crate) text: String,
    pub(crate) until: f64,
    pub(crate) kind: ToastKind,
}

/// 1px vertical divider in an exact 3px grid column.
pub(crate) fn vline(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, color),
    );
}

/// Vector sort-direction mark: painted triangles, not text — the bundled
/// UI font has no ▲▼⇅ glyphs (tofu squares). Active direction in text
/// color, inactive as a faint up+down pair (sortable affordance).
/// Returns the click response; the caller attaches hover text + action.
pub(crate) fn sort_mark(ui: &mut egui::Ui, dir: Option<SortDir>) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let c = rect.center();
        let visuals = ui.visuals();
        // (pointing-up, y-offset, half-size).
        let tris: &[(bool, f32, f32)] = match dir {
            Some(SortDir::Asc) => &[(true, 0.0, 4.5)],
            Some(SortDir::Desc) => &[(false, 0.0, 4.5)],
            None => &[(true, -2.6, 3.0), (false, 2.6, 3.0)],
        };
        let color = match dir {
            Some(_) => visuals.text_color(),
            None => visuals.weak_text_color(),
        };
        for &(up, dy, r) in tris {
            let cy = c.y + dy;
            let pts = if up {
                vec![
                    egui::pos2(c.x - r, cy + r * 0.8),
                    egui::pos2(c.x + r, cy + r * 0.8),
                    egui::pos2(c.x, cy - r * 0.8),
                ]
            } else {
                vec![
                    egui::pos2(c.x - r, cy - r * 0.8),
                    egui::pos2(c.x + r, cy - r * 0.8),
                    egui::pos2(c.x, cy + r * 0.8),
                ]
            };
            ui.painter()
                .add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        }
    }
    resp
}

/// Panel frame with Python's drag-enter green (#2FA572) while hovered.
pub(crate) fn panel_frame(ui: &egui::Ui, hovering: bool) -> egui::Frame {
    let mut frame = egui::Frame::group(ui.style());
    if hovering {
        frame = frame.stroke(egui::Stroke::new(
            1.5,
            egui::Color32::from_rgb(0x2F, 0xA5, 0x72),
        ));
    }
    frame
}

/// Screenshot green/red fills, muted for the dark theme (light text stays
/// readable): best green, worst red, all-tied dim yellow. Colors apply only
/// with 2+ scored rows; a lone result stays uncolored.
pub(crate) const BEST_FILL: egui::Color32 = egui::Color32::from_rgb(0x2E, 0x6B, 0x3E);
pub(crate) const WORST_FILL: egui::Color32 = egui::Color32::from_rgb(0x7A, 0x36, 0x36);
pub(crate) const TIE_FILL: egui::Color32 = egui::Color32::from_rgb(0x6B, 0x5F, 0x2A);
/// Cross-format warning tint for Media cells (upstream #47): readable on
/// the dark default theme without touching layout.
pub(crate) const WARN_TEXT: egui::Color32 = egui::Color32::from_rgb(0xE5, 0xA6, 0x3B);

/// Cell/chip background for a stat rank; `None` = no highlight.
pub(crate) fn rank_fill(rank: crate::metrics::StatRank) -> Option<egui::Color32> {
    match rank {
        crate::metrics::StatRank::Best => Some(BEST_FILL),
        crate::metrics::StatRank::Worst => Some(WORST_FILL),
        crate::metrics::StatRank::Tie => Some(TIE_FILL),
        crate::metrics::StatRank::Plain => None,
    }
}

/// Indeterminate bounce position for `Running` cells: triangle wave
/// `0 → 1 → 0`, one leg per `LEG_S` seconds. Pure (no `Ui`) so tests
/// cover the ping-pong without a GUI harness.
pub(crate) fn running_sweep_pos(time_s: f64) -> f32 {
    const LEG_S: f64 = 0.7;
    let phase = (time_s / LEG_S).rem_euclid(2.0);
    (if phase < 1.0 { phase } else { 2.0 - phase }) as f32
}

/// Bold green sweep behind a `Running` cell's text: a full-height
/// translucent `#2FA572` segment bouncing left ↔ right. Painted before
/// the label so the `Frame: N` text stays on top; driven by the
/// existing ~10Hz measuring heartbeat, so no extra repaint cost.
pub(crate) fn paint_running_sweep(ui: &mut egui::Ui) {
    let rect = ui.available_rect_before_wrap();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let x01 = running_sweep_pos(ui.input(|i| i.time));
    let seg_w = (rect.width() * 0.28).clamp(14.0, 24.0);
    let bar = egui::Rect::from_min_size(
        egui::pos2(rect.left() + (rect.width() - seg_w) * x01, rect.top()),
        egui::vec2(seg_w, rect.height()),
    );
    ui.painter().rect_filled(
        bar,
        3.0,
        egui::Color32::from_rgba_unmultiplied(0x2F, 0xA5, 0x72, 110),
    );
}

/// Filter-metric Done tooltip in FFMetrics order: Avg, Exec, Frames, a blank
/// line, Mean..StdDev, another blank line, then Percentiles. Each comparable
/// value is chipped by its cross-row rank; Exec time and Frames count are
/// display-only (no chip).
/// Plain horizontal rows with content-hugging widths: grids and expanding
/// layouts feed back into the tooltip auto-size and balloon while hovered.
pub(crate) fn metric_stat_tooltip(
    ui: &mut egui::Ui,
    title: &str,
    stats: &crate::metrics::DoneStats,
    ranks: &[crate::metrics::StatRank; 10],
    sel: crate::metrics::CellStat,
) {
    ui.label(egui::RichText::new(title).strong());
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 1.0);
        // Bold the row the cell-value selector shows.
        let lbl = |k: usize, label: &'static str| {
            if k == sel.index() {
                egui::RichText::new(label).strong()
            } else {
                egui::RichText::new(label)
            }
        };
        let comp = stats.comparable();
        let (label, v, _) = comp[0];
        tip_stat_row(ui, lbl(0, label), &format!("{v:.6}"), ranks[0]);
        tip_plain_row(ui, "Exec time:", &crate::metrics::format_exec(stats.exec_s));
        tip_plain_row(ui, "Frames count:", &stats.frames.to_string());
        ui.add_space(5.0);
        for k in 1..=5 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, lbl(k, label), &format!("{v:.6}"), ranks[k]);
        }
        ui.add_space(5.0);
        for k in 6..10 {
            let (label, v, _) = comp[k];
            tip_stat_row(ui, lbl(k, label), &format!("{v:.6}"), ranks[k]);
        }
    });
}

/// One tooltip row: fixed label + right-aligned value, chipped when ranked.
pub(crate) fn tip_stat_row(
    ui: &mut egui::Ui,
    label: impl Into<egui::WidgetText>,
    val: &str,
    rank: crate::metrics::StatRank,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [96.0, 15.0],
            egui::Label::new(label).halign(egui::Align::LEFT),
        );
        let val = egui::Label::new(val).halign(egui::Align::RIGHT);
        match rank_fill(rank) {
            Some(fill) => {
                egui::Frame::NONE
                    .fill(fill)
                    .inner_margin(egui::Margin::symmetric(4, 0))
                    .show(ui, |ui| {
                        ui.add_sized([80.0, 15.0], val);
                    });
            }
            None => {
                ui.add_sized([88.0, 15.0], val);
            }
        }
    });
}

/// Display-only tooltip row (Exec time, Frames count): never chipped.
pub(crate) fn tip_plain_row(ui: &mut egui::Ui, label: &str, val: &str) {
    tip_stat_row(ui, label, val, crate::metrics::StatRank::Plain);
}
