use std::collections::HashSet;
use std::path::Path;

#[derive(Debug)]
struct QueueRow {
    path: String,
    display: String,
    include: bool,
    selected: bool,
    media: String,
    media_tip: String,
}

/// Python `normcase(abspath)` equivalent for the same-file guard rail.
fn norm_key(p: &str) -> String {
    let path = Path::new(p);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let s = abs.to_string_lossy().replace('/', "\\");
    #[cfg(windows)]
    let s = s.to_lowercase();
    s
}

/// Shortest unique trailing-path suffix per entry (Python `_display_names`).
fn display_names(paths: &[String]) -> Vec<String> {
    let parts: Vec<Vec<String>> = paths
        .iter()
        .map(|p| {
            Path::new(p)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .collect();
    let sep = std::path::MAIN_SEPARATOR.to_string();
    parts
        .iter()
        .enumerate()
        .map(|(i, part)| {
            for n in 1..=part.len() {
                let cand = &part[part.len() - n..];
                let unique = parts.iter().enumerate().all(|(j, q)| {
                    j == i || {
                        let tail = if q.len() >= n {
                            &q[q.len() - n..]
                        } else {
                            &q[..]
                        };
                        tail != cand
                    }
                });
                if unique {
                    return cand.join(&sep);
                }
            }
            part.join(&sep)
        })
        .collect()
}

/// Retrieves the cursor position in egui's logical point coordinates.
/// During Windows OLE file drags, winit omits pointer move events, so egui's
/// internal pointer state is None/stale. We query the OS cursor directly.
#[cfg(windows)]
fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
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
fn get_cursor_pos(ctx: &egui::Context) -> Option<egui::Pos2> {
    ctx.input(|i| i.pointer.hover_pos().or(i.pointer.latest_pos()))
}

/// Hover highlight delay (seconds) so passing over rows while aiming
/// at text to copy doesn't flash each row.
const ROW_HOVER_DELAY: f64 = 0.1;

/// How long the drop toast (e.g. ignored extra reference files) stays up.
const TOAST_SECS: f64 = 3.0;

/// Toast severity; drives the outline color. `Info` keeps the default
/// popup outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    dead_code,
    reason = "Info/Error have no producer yet; kept for probe/run errors"
)]
enum ToastKind {
    Info,
    Warning,
    Error,
}

impl ToastKind {
    fn outline(self) -> Option<egui::Color32> {
        match self {
            ToastKind::Info => None,
            ToastKind::Warning => Some(egui::Color32::from_rgb(0xD9, 0xA4, 0x06)),
            ToastKind::Error => Some(egui::Color32::from_rgb(0xE0, 0x4B, 0x4B)),
        }
    }
}

#[derive(Debug, Clone)]
struct Toast {
    text: String,
    until: f64,
    kind: ToastKind,
}

#[derive(Debug)]
pub struct RFMetricsApp {
    ref_path: String,
    duration: String,
    skip: String,
    m_psnr: bool,
    m_ssim: bool,
    m_vmaf: bool,
    m_xpsnr: bool,
    m_ssim2: bool,
    m_but: bool,
    m_cvvdp: bool,
    vmaf_model: String,
    vmaf_phone: bool,
    vmaf_scale: bool,
    vmaf_pooling: String,
    vmaf_subsample: String,
    rows: Vec<QueueRow>,
    ffmpeg: crate::binaries::BinaryInfo,
    ffvship: crate::binaries::BinaryInfo,
    ffprobe: Option<std::path::PathBuf>,
    ref_info: String,
    last_probed_ref: String,
    ref_rect: Option<egui::Rect>,
    table_rect: Option<egui::Rect>,
    hover_row: Option<usize>,
    hover_since: Option<f64>,
    hovered_now: Option<usize>,
    toast: Option<Toast>,
}

impl Default for RFMetricsApp {
    fn default() -> Self {
        let ffmpeg = crate::binaries::ffmpeg_info();
        let ffvship = crate::binaries::ffvship_info();
        let ffprobe = crate::binaries::ffprobe_path(ffmpeg.path.as_deref());
        Self {
            ref_path: String::new(),
            duration: String::new(),
            skip: String::new(),
            m_psnr: false,
            m_ssim: false,
            m_vmaf: true,
            m_xpsnr: false,
            m_ssim2: false,
            m_but: false,
            m_cvvdp: false,
            vmaf_model: "vmaf_v0.6.1.json".to_owned(),
            vmaf_phone: true,
            vmaf_scale: false,
            vmaf_pooling: "Mean".to_owned(),
            vmaf_subsample: "1".to_owned(),
            rows: Vec::new(),
            ffmpeg,
            ffvship,
            ffprobe,
            ref_info: crate::probe::reference_media_text("", None),
            last_probed_ref: String::new(),
            ref_rect: None,
            table_rect: None,
            hover_row: None,
            hover_since: None,
            hovered_now: None,
            toast: None,
        }
    }
}

impl RFMetricsApp {
    /// Re-probe only when the path actually changed.
    fn refresh_ref_info(&mut self) {
        if self.ref_path == self.last_probed_ref {
            return;
        }
        self.last_probed_ref = self.ref_path.clone();
        let path = self.ref_path.clone();
        let exe = self.ffprobe.clone();
        self.ref_info = crate::probe::reference_media_text(&path, exe.as_deref());
    }

    /// Short display names for all rows (Python `_refresh_names`).
    fn refresh_queue_names(&mut self) {
        let paths: Vec<String> = self.rows.iter().map(|r| r.path.clone()).collect();
        for (row, name) in self.rows.iter_mut().zip(display_names(&paths)) {
            row.display = name;
        }
        // Row indices may have shifted; drop stale hover state.
        self.hover_row = None;
        self.hover_since = None;
    }

    /// Queue picked files, silently skipping ones already present.
    fn add_queue_files(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut seen: HashSet<String> = self.rows.iter().map(|r| norm_key(&r.path)).collect();
        let exe = self.ffprobe.clone();
        for p in paths {
            let s = p.to_string_lossy().into_owned();
            if !seen.insert(norm_key(&s)) {
                continue; // guard rail: same file already queued
            }
            let (media, media_tip) = crate::probe::probe_table_text(&s, exe.as_deref());
            self.rows.push(QueueRow {
                path: s,
                display: String::new(),
                include: true,
                selected: false,
                media,
                media_tip,
            });
        }
        self.refresh_queue_names();
    }
}

/// 1px vertical divider in an exact 3px grid column.
fn vline(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, color),
    );
}

/// Panel frame with Python's drag-enter green (#2FA572) while hovered.
fn panel_frame(ui: &egui::Ui, hovering: bool) -> egui::Frame {
    let mut frame = egui::Frame::group(ui.style());
    if hovering {
        frame = frame.stroke(egui::Stroke::new(
            1.5,
            egui::Color32::from_rgb(0x2F, 0xA5, 0x72),
        ));
    }
    frame
}

impl eframe::App for RFMetricsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let (hovering, dropped) = ui.ctx().input(|i| {
            (
                !i.raw.hovered_files.is_empty(),
                i.raw
                    .dropped_files
                    .iter()
                    .map(|f| f.path().to_path_buf())
                    .collect::<Vec<_>>(),
            )
        });

        // Continuously repaint while dragging so hover outlines update smoothly
        if hovering {
            ui.ctx().request_repaint();
        }

        let cursor_pos = get_cursor_pos(ui.ctx());
        let now = ui.ctx().input(|i| i.time);

        // Direct OS-cursor hit test; winit gives no position during OLE drags.
        let is_over_ref =
            matches!((cursor_pos, self.ref_rect), (Some(pos), Some(rect)) if rect.contains(pos));
        let is_over_table =
            matches!((cursor_pos, self.table_rect), (Some(pos), Some(rect)) if rect.contains(pos));

        // Strict target routing (Python parity: a drop outside a target
        // does nothing). The reference box takes one file; extras are
        // reported via toast instead of silently vanishing.
        if !dropped.is_empty() {
            if is_over_ref {
                let mut iter = dropped.into_iter();
                if let Some(first) = iter.next() {
                    self.ref_path = first.to_string_lossy().into_owned();
                }
                let extra = iter.len();
                if extra > 0 {
                    self.toast = Some(Toast {
                        text: format!(
                            "Reference takes one file — kept the first, ignored {extra} more"
                        ),
                        until: now + TOAST_SECS,
                        kind: ToastKind::Warning,
                    });
                }
            } else if is_over_table {
                self.add_queue_files(dropped);
            }
        }
        self.refresh_ref_info();

        let ref_hover = hovering && is_over_ref;
        let table_hover = hovering && is_over_table;

        // ---- Reference (top, fixed) ----
        let ref_resp = egui::Panel::top("reference").show(ui, |ui| {
            ui.add(egui::Label::new("Reference").selectable(false));
            panel_frame(ui, ref_hover).show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let preview_w = 136.0;
                    let total = ui.available_width();
                    ui.vertical(|ui| {
                        ui.set_width((total - preview_w - 12.0).max(0.0));
                        // Path row
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Path to file:").selectable(false));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add_sized([90.0, 24.0], egui::Button::new("Browse"))
                                        .clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_title("Select reference video")
                                            .add_filter(
                                                "Video files",
                                                &[
                                                    "mp4", "mkv", "mov", "avi", "webm", "m2ts",
                                                    "ts", "m4v",
                                                ],
                                            )
                                            .pick_file()
                                    {
                                        self.ref_path = path.to_string_lossy().into_owned();
                                    }
                                    let _ = ui.add(
                                        egui::TextEdit::singleline(&mut self.ref_path)
                                            .desired_width(f32::INFINITY),
                                    );
                                },
                            );
                        });
                        ui.label(&self.ref_info);
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Duration:").selectable(false));
                            let _ = ui.add(
                                egui::TextEdit::singleline(&mut self.duration)
                                    .hint_text("00:00.000")
                                    .desired_width(110.0),
                            );
                            ui.add(egui::Label::new("Skip:").selectable(false));
                            let _ = ui.add(
                                egui::TextEdit::singleline(&mut self.skip)
                                    .hint_text("00:00.000")
                                    .desired_width(110.0),
                            );
                        });
                    });
                    // Thumbnail placeholder 136x76
                    egui::Frame::NONE
                        .fill(egui::Color32::BLACK)
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(60)))
                        .show(ui, |ui| {
                            ui.allocate_space(egui::vec2(preview_w, 76.0));
                        });
                });
            });
        });
        self.ref_rect = Some(ref_resp.response.rect);

        // ---- Bottom action bar (bottommost) ----
        egui::Panel::bottom("actions").show(ui, |ui| {
            ui.horizontal(|ui| {
                let _ = ui.add_sized([90.0, 24.0], egui::Button::new("Start"));
                let _ = ui.add_sized([90.0, 24.0], egui::Button::new("Reset"));
                let _ = ui.add_sized([90.0, 24.0], egui::Button::new("Plot"));
                ui.label(&self.ffmpeg.short)
                    .on_hover_text(&self.ffmpeg.detail);
                ui.add(egui::Label::new("|").selectable(false));
                ui.label(&self.ffvship.short)
                    .on_hover_text(&self.ffvship.detail);
            });
        });

        // ---- VMAF options (just above bottom bar) ----
        egui::Panel::bottom("vmaf").show(ui, |ui| {
            ui.add(egui::Label::new("VMAF options").selectable(false));
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Model").selectable(false));
                    let _ = egui::ComboBox::from_id_salt("vmaf_model")
                        .width(220.0)
                        .selected_text(&self.vmaf_model)
                        .show_ui(ui, |ui| {
                            let _ = ui.selectable_value(
                                &mut self.vmaf_model,
                                "vmaf_v0.6.1.json".to_owned(),
                                "vmaf_v0.6.1.json",
                            );
                        });
                    let _ = ui.add(egui::Checkbox::new(&mut self.vmaf_phone, "Phone"));
                });
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new(""));
                    let _ = ui.add(egui::Checkbox::new(
                        &mut self.vmaf_scale,
                        "Scale to model's resolution",
                    ));
                });
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Pooling").selectable(false));
                    let _ = egui::ComboBox::from_id_salt("vmaf_pooling")
                        .width(220.0)
                        .selected_text(&self.vmaf_pooling)
                        .show_ui(ui, |ui| {
                            let _ = ui.selectable_value(
                                &mut self.vmaf_pooling,
                                "Mean".to_owned(),
                                "Mean",
                            );
                            let _ = ui.selectable_value(
                                &mut self.vmaf_pooling,
                                "Harmonic Mean".to_owned(),
                                "Harmonic Mean",
                            );
                        });
                });
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [70.0, 18.0],
                        egui::Label::new("Subsample").selectable(false),
                    );
                    let _ = egui::ComboBox::from_id_salt("vmaf_subsample")
                        .width(220.0)
                        .selected_text(&self.vmaf_subsample)
                        .show_ui(ui, |ui| {
                            for v in ["1", "2", "3", "5", "10", "15"] {
                                let _ =
                                    ui.selectable_value(&mut self.vmaf_subsample, v.to_owned(), v);
                            }
                        });
                });
            });
        });

        // ---- File queue (center, expanding) ----
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_sized([110.0, 24.0], egui::Button::new("Add files"))
                    .clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .set_title("Select video files")
                        .add_filter(
                            "Video files",
                            &["mp4", "mkv", "mov", "avi", "webm", "m2ts", "ts", "m4v"],
                        )
                        .add_filter("All files", &["*"])
                        .pick_files()
                {
                    self.add_queue_files(paths);
                }
                if ui
                    .add_sized([130.0, 24.0], egui::Button::new("Remove Selected"))
                    .clicked()
                {
                    self.rows.retain(|r| !r.selected);
                    self.refresh_queue_names();
                }
            });
            ui.add_space(4.0);
            self.hovered_now = None;
            let table_resp = panel_frame(ui, table_hover).show(ui, |ui| {
                // Keep the drop box a stable size: at least full width × 160
                // even when the table content is smaller (e.g. one row).
                ui.set_min_size(egui::vec2(ui.available_width(), 160.0));
                if self.rows.is_empty() {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new("No files yet — drag & drop video files here")
                                .weak(),
                        )
                        .selectable(false),
                    );
                    return;
                }
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                    let mut table = egui_extras::TableBuilder::new(ui)
                        .striped(false)
                        .resizable(false)
                        .sense(egui::Sense::click())
                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                        .column(egui_extras::Column::exact(22.0))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::exact(20.0))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::remainder().clip(true))
                        .column(egui_extras::Column::exact(3.0))
                        .column(egui_extras::Column::exact(240.0));
                    for _ in 0..7 {
                        table = table
                            .column(egui_extras::Column::exact(3.0))
                            .column(egui_extras::Column::exact(82.0));
                    }
                    table
                        .header(18.0, |mut header| {
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("Path").strong())
                                        .selectable(false),
                                );
                            });
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("Media info").strong())
                                        .selectable(false),
                                );
                            });
                            for (flag, name) in [
                                (&mut self.m_psnr, "PSNR"),
                                (&mut self.m_ssim, "SSIM"),
                                (&mut self.m_vmaf, "VMAF"),
                                (&mut self.m_xpsnr, "XPSNR"),
                                (&mut self.m_ssim2, "SSIM2"),
                                (&mut self.m_but, "BUTTER"),
                                (&mut self.m_cvvdp, "CVVDP"),
                            ] {
                                header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                                header.col(|ui| {
                                    ui.checkbox(flag, name);
                                });
                            }
                        })
                        .body(|body| {
                            body.rows(20.0, self.rows.len(), |mut row| {
                                let i = row.index();
                                row.set_selected(self.rows[i].selected);
                                // Delayed hover: only outline after the pointer
                                // rests on the row, so passing over rows while
                                // aiming at text doesn't flash each one.
                                let hover_delayed = self.hover_row == Some(i)
                                    && self.hover_since.is_some_and(|t| now - t >= ROW_HOVER_DELAY);
                                row.set_hovered(hover_delayed);
                                // Free-space click toggles selection; widget clicks
                                // (checkbox, play, text drag-select) must not.
                                let mut label_clicked = false;
                                let mut bg_clicked = false;
                                row.col(|ui| {
                                    ui.checkbox(&mut self.rows[i].include, "");
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        let path = self.rows[i].path.clone();
                                        let _ = open::that(&path);
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                let (_, r) = row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let (display, path) = {
                                        let r = &self.rows[i];
                                        (r.display.clone(), r.path.clone())
                                    };
                                    // Plain selectable text: no button hover
                                    // outline; drag-select/copy still works and
                                    // the full path shows as tooltip (Python parity).
                                    ui.add(egui::Label::new(&display).selectable(true))
                                        .on_hover_text(&path);
                                });
                                if r.clicked() {
                                    label_clicked = true;
                                }
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                let (_, r) = row.col(|ui| {
                                    let (media, tip) = {
                                        let r = &self.rows[i];
                                        (r.media.clone(), r.media_tip.clone())
                                    };
                                    ui.label(&media).on_hover_text(&tip);
                                });
                                if r.clicked() {
                                    bg_clicked = true;
                                }
                                for _ in 0..7 {
                                    let (_, r) =
                                        row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                    let (_, r) = row.col(|ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.label("N/A");
                                        });
                                    });
                                    if r.clicked() {
                                        bg_clicked = true;
                                    }
                                }
                                if label_clicked || bg_clicked {
                                    self.rows[i].selected = !self.rows[i].selected;
                                }
                                if row.response().hovered() {
                                    self.hovered_now = Some(i);
                                }
                            });
                        });
                });
            });
            self.table_rect = Some(table_resp.response.rect);
            // Roll the delayed-hover timer forward; repaint while the
            // delay is pending so the outline appears without moving.
            if self.hovered_now != self.hover_row {
                self.hover_row = self.hovered_now;
                self.hover_since = self.hover_row.map(|_| now);
            }
            let hover_pending = matches!(
                (self.hover_row, self.hovered_now, self.hover_since),
                (Some(a), Some(b), Some(t)) if a == b && now - t < ROW_HOVER_DELAY
            );
            if hover_pending {
                ui.ctx().request_repaint();
            }
        });

        // Drop hint overlay pinned over the active drop target
        if hovering {
            let active_hint = if ref_hover {
                self.ref_rect
                    .map(|r| ("drop_hint_ref", r, "Drop video to set as reference"))
            } else if table_hover {
                self.table_rect
                    .map(|r| ("drop_hint_table", r, "Drop files to queue"))
            } else {
                None
            };

            if let Some((id, r, text)) = active_hint {
                egui::Area::new(egui::Id::new(id))
                    .fixed_pos(r.center_top())
                    .pivot(egui::Align2::CENTER_TOP)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.label(text);
                        });
                    });
            }
        }

        // Transient toast (e.g. extra files dropped on the reference box).
        if let Some(toast) = self.toast.clone() {
            if now < toast.until {
                ui.ctx().request_repaint();
                let corner = ui.max_rect().right_bottom();
                let mut frame = egui::Frame::popup(ui.style());
                if let Some(outline) = toast.kind.outline() {
                    frame = frame.stroke(egui::Stroke::new(1.5, outline));
                }
                egui::Area::new(egui::Id::new("toast"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(corner + egui::vec2(-10.0, -10.0))
                    .pivot(egui::Align2::RIGHT_BOTTOM)
                    .show(ui.ctx(), |ui| {
                        frame.show(ui, |ui| {
                            ui.label(&toast.text);
                        });
                    });
            } else {
                self.toast = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{display_names, norm_key};

    #[test]
    fn same_file_keys_equal() {
        assert_eq!(norm_key("C:/Vids/a.mp4"), norm_key("c:\\vids\\A.MP4"));
        assert_ne!(norm_key("C:/Vids/a.mp4"), norm_key("C:/Vids/b.mp4"));
    }

    #[test]
    fn single_name_is_basename() {
        assert_eq!(
            display_names(&["C:/a/b/c.mp4".to_owned()]),
            vec!["c.mp4".to_owned()]
        );
    }

    #[test]
    fn sibling_names_disambiguate() {
        let names = display_names(&[
            "C:/b/output tq 70.mkv".to_owned(),
            "C:/b/output tq 75.mkv".to_owned(),
        ]);
        assert_eq!(names, vec!["output tq 70.mkv", "output tq 75.mkv"]);
    }

    #[test]
    fn same_basename_keeps_parent() {
        let names = display_names(&["C:/a/x.mp4".to_owned(), "C:/b/x.mp4".to_owned()]);
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(names, vec![format!("a{sep}x.mp4"), format!("b{sep}x.mp4")]);
    }
}
