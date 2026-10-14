//! Shared UI primitives for the `RFMetricsApp` windows: toast types,
//! cursor helper, and the small pure `egui` helpers (`vline`,
//! `sort_mark`, `panel_frame`, rank fills, running sweep, stat tooltips).

use crate::app::queue::SortDir;

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

/// Painted up/down stepper chevron: stroked ^/v glyph, not text — the
/// bundled UI font has no ▲▼ glyphs (tofu squares; see `sort_mark`).
/// Bare on the spinbox field (no button chrome; faint fill on
/// hover/press only), stroked chevron in the matching text color so it
/// dims with the field when disabled. `held` is the caller's own hold
/// latch: egui retires its hover/press flags after `max_click_duration`,
/// which would un-highlight a still-repeating button — a latched hold
/// forces active visuals instead. Returns the click response, so it
/// drops into `clicked` / hold-to-repeat call sites like a text button.
pub(crate) fn step_button(ui: &mut egui::Ui, up: bool, held: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(18.0, 11.0), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let pressed = held && resp.enabled();
        let v = if pressed {
            &ui.visuals().widgets.active
        } else {
            ui.style().interact(&resp)
        };
        if pressed || resp.is_pointer_button_down_on() {
            ui.painter().rect_filled(rect, v.corner_radius, v.bg_fill);
        } else if resp.hovered() {
            ui.painter()
                .rect_filled(rect, v.corner_radius, v.weak_bg_fill);
        }
        let c = rect.center();
        let (r, h) = (3.5, 2.2);
        let pts = if up {
            vec![
                egui::pos2(c.x - r, c.y + h),
                egui::pos2(c.x, c.y - h),
                egui::pos2(c.x + r, c.y + h),
            ]
        } else {
            vec![
                egui::pos2(c.x - r, c.y - h),
                egui::pos2(c.x, c.y + h),
                egui::pos2(c.x + r, c.y - h),
            ]
        };
        ui.painter().add(egui::Shape::line(
            pts,
            egui::Stroke::new(1.5, v.text_color()),
        ));
    }
    resp
}

/// Chevron click step with range clamp (spinbox behavior, pure for tests).
pub(crate) fn spin_step(
    value: f64,
    up: bool,
    step: f64,
    range: std::ops::RangeInclusive<f64>,
) -> f64 {
    (if up { value + step } else { value - step }).clamp(*range.start(), *range.end())
}

/// Display formatting for spinbox buffers: fixed decimals with the
/// DragValue look (no trailing zeros: "24", not "24.0").
pub(crate) fn fmt_spin(v: f64, decimals: usize) -> String {
    let s = format!("{v:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s
    }
}

/// Hold-to-repeat cadence for spinbox chevrons (Plot goto parity:
/// first repeat after a short delay, then a steady cadence).
const SPIN_FIRST_DELAY_S: f64 = 0.4;
const SPIN_REPEAT_S: f64 = 0.05;

/// Go-to-frame-style spinbox (Plot parity): frameless typed field plus
/// bare ^/v chevrons in one text-edit-styled frame. No dragging — a
/// press steps at once and holding repeats on the cadence above;
/// typed text parses live (unparseable keeps the old value). The hold
/// latch lives in egui temp data per field, so callers keep no state.
/// Returns the outer frame response for hover tips.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spinbox(
    ui: &mut egui::Ui,
    id: &'static str,
    text: &mut String,
    value: &mut f64,
    step: f64,
    range: std::ops::RangeInclusive<f64>,
    decimals: usize,
    field_w: f32,
) -> egui::Response {
    let tid = egui::Id::new(("spinbox", id));
    // Previous-frame focus (same one-frame lag as the goto field: the
    // frame must draw before this frame's field response exists).
    let focused = ui.memory(|mem| mem.has_focus(tid));
    let frame = egui::Frame::new()
        .fill(ui.visuals().text_edit_bg_color())
        .corner_radius(ui.visuals().widgets.inactive.corner_radius)
        .stroke(if focused {
            ui.visuals().selection.stroke
        } else {
            egui::Stroke::new(1.0, egui::Color32::TRANSPARENT)
        })
        .inner_margin(egui::Margin {
            left: 6,
            right: 3,
            top: 2,
            bottom: 2,
        });
    frame
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let field = ui.add(
                    egui::TextEdit::singleline(text)
                        .id(tid)
                        .frame(egui::Frame::NONE)
                        .desired_width(field_w),
                );
                if field.changed()
                    && let Ok(v) = text.trim().parse::<f64>()
                {
                    *value = v.clamp(*range.start(), *range.end());
                }
                // Hold-to-repeat (goto parity): a fresh press steps at
                // once and latches; the latch fires on cadence off the
                // physical button state and drops on release, so the next
                // press steps at once. `clicked` covers keyboard
                // activation; the latch guard keeps a mouse release from
                // double-stepping. Repaints run continuously while held
                // so the cadence fires on an idle tick.
                let now = ui.input(|i| i.time);
                let down = ui.input(|i| i.pointer.primary_down());
                let mut latch: Option<(i8, f64)> = ui.data(|data| data.get_temp(tid));
                let mut held = latch.map(|(dir, _)| dir);
                let up = step_button(ui, true, held == Some(1));
                let dn = step_button(ui, false, held == Some(-1));
                let mut step_dir = 0i8;
                for (resp, dir) in [(&up, 1i8), (&dn, -1i8)] {
                    if resp.is_pointer_button_down_on() && held != Some(dir) {
                        step_dir = dir;
                        held = Some(dir);
                        latch = Some((dir, now + SPIN_FIRST_DELAY_S));
                    } else if resp.clicked() && held != Some(dir) {
                        step_dir = dir;
                    }
                }
                match (down, held) {
                    (true, Some(_)) => {
                        if latch.is_some_and(|(_, next)| now >= next) {
                            step_dir = held.unwrap_or(0);
                            latch = held.map(|dir| (dir, now + SPIN_REPEAT_S));
                        }
                        ui.ctx().request_repaint();
                    }
                    _ => {
                        latch = None;
                    }
                }
                if step_dir != 0 {
                    *value = spin_step(*value, step_dir > 0, step, range.clone());
                    *text = fmt_spin(*value, decimals);
                }
                match latch {
                    Some(held) => {
                        ui.data_mut(|data| data.insert_temp(tid, held));
                    }
                    None => {
                        ui.data_mut(|data| data.remove_temp::<(i8, f64)>(tid));
                    }
                }
            });
        })
        .response
}

/// Hover tooltip that wraps instead of clipping: viewport-bound popups
/// cut long single-line texts (e.g. in the narrow preset editor), so
/// informational hovers go through this. Fixed max width, full text kept.
pub(crate) fn hover_wrap(response: egui::Response, text: &str) -> egui::Response {
    response.on_hover_ui(|ui| {
        ui.set_max_width(260.0);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        ui.label(text);
    })
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
