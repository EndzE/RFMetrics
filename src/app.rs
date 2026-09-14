use std::collections::HashSet;
use std::path::Path;

#[derive(Debug)]
struct QueueRow {
    path: String,
    display: String,
    include: bool,
    selected: bool,
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
        }
    }
}

impl RFMetricsApp {
    /// Re-probe only when the path actually changed. Cheap for typed text
    /// (nonexistent paths never spawn ffprobe); one spawn per Browse pick.
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
    }

    /// Queue picked files, silently skipping ones already present.
    fn add_queue_files(&mut self, paths: Vec<std::path::PathBuf>) {
        let mut seen: HashSet<String> = self.rows.iter().map(|r| norm_key(&r.path)).collect();
        for p in paths {
            let s = p.to_string_lossy().into_owned();
            if !seen.insert(norm_key(&s)) {
                continue; // guard rail: same file already queued
            }
            self.rows.push(QueueRow {
                path: s,
                display: String::new(),
                include: true,
                selected: false,
            });
        }
        self.refresh_queue_names();
    }
}

/// 1px vertical divider in an exact 3px grid column. (The Separator widget
/// sizes to spacing units and broke the table width math; Python draws plain
/// 1px tk frames here.)
fn vline(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, color),
    );
}

impl eframe::App for RFMetricsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // OS file-drag state. `hovering` is window-wide (Python paints the
        // boxes on drag-enter anywhere); the drop itself is routed by rect.
        let (hovering, dropped, pointer) = ui.ctx().input(|i| {
            (
                !i.raw.hovered_files.is_empty(),
                i.raw
                    .dropped_files
                    .iter()
                    .map(|f| f.path().to_path_buf())
                    .collect::<Vec<_>>(),
                i.pointer.latest_pos(),
            )
        });
        // Route by rect when the cursor pos is known; if it isn't (drop
        // frame without a preceding move event), accept anyway — reference
        // is currently the only drop target.
        let over_ref = match (pointer, self.ref_rect) {
            (Some(pos), Some(rect)) => rect.contains(pos),
            _ => true,
        };
        if let Some(first) = dropped.into_iter().next()
            && over_ref
        {
            self.ref_path = first.to_string_lossy().into_owned();
        }
        self.refresh_ref_info();
        // ---- Reference (top, fixed) ----
        let ref_resp = egui::Panel::top("reference").show(ui, |ui| {
            ui.label("Reference");
            let mut ref_frame = egui::Frame::group(ui.style());
            if hovering {
                // ponytail: Python's drag-enter green (#2FA572)
                ref_frame = ref_frame.stroke(egui::Stroke::new(
                    1.5,
                    egui::Color32::from_rgb(0x2F, 0xA5, 0x72),
                ));
            }
            ref_frame.show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let preview_w = 136.0;
                    let total = ui.available_width();
                    ui.vertical(|ui| {
                        ui.set_width(total - preview_w - 12.0);
                        // Path row
                        ui.horizontal(|ui| {
                            ui.label("Path to file:");
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
                            ui.label("Duration:");
                            let _ = ui.add(
                                egui::TextEdit::singleline(&mut self.duration)
                                    .hint_text("00:00.000")
                                    .desired_width(110.0),
                            );
                            ui.label("Skip:");
                            let _ = ui.add(
                                egui::TextEdit::singleline(&mut self.skip)
                                    .hint_text("00:00.000")
                                    .desired_width(110.0),
                            );
                        });
                    });
                    // Thumbnail placeholder 136x76, black like Python preview_box
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
                ui.label("|");
                ui.label(&self.ffvship.short)
                    .on_hover_text(&self.ffvship.detail);
            });
        });

        // ---- VMAF options (just above bottom bar) ----
        egui::Panel::bottom("vmaf").show(ui, |ui| {
            ui.label("VMAF options");
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Model"));
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
                    ui.add_sized([70.0, 18.0], egui::Label::new("Pooling"));
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
                    ui.add_sized([70.0, 18.0], egui::Label::new("Subsample"));
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
            egui::Frame::group(ui.style()).show(ui, |ui| {
                if self.rows.is_empty() {
                    // ponytail: claim full width/height so the empty box
                    // matches the table dimensions instead of shrinking
                    ui.set_min_size(egui::vec2(ui.available_width(), 160.0));
                    ui.weak("No files yet — drag & drop video files here");
                    return;
                }
                // ponytail: TableBuilder owns column geometry; the remainder
                // Path column replaces all hand-rolled width math.
                // Tight gaps like Python's padx (default 8px gaps would eat
                // ~160px across 21 columns).
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                    let mut table = egui_extras::TableBuilder::new(ui)
                        .striped(false)
                        .resizable(false)
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
                                ui.label(egui::RichText::new("Path").strong());
                            });
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.label(egui::RichText::new("Media info").strong());
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
                                row.col(|ui| {
                                    ui.checkbox(&mut self.rows[i].include, "");
                                });
                                row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        // ponytail: Python ignores play errors too
                                        let path = self.rows[i].path.clone();
                                        let _ = open::that(&path);
                                    }
                                });
                                row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                // Left-aligned selectable label
                                // (Python: anchor="w"); the clipped
                                // remainder column truncates long names.
                                row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let (selected, display) = {
                                        let r = &self.rows[i];
                                        (r.selected, r.display.clone())
                                    };
                                    if ui.selectable_label(selected, &display).clicked() {
                                        self.rows[i].selected = !selected;
                                    }
                                });
                                row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                row.col(|ui| {
                                    ui.label("N/A");
                                });
                                for _ in 0..7 {
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                    // ponytail: Python centers metric cells too
                                    row.col(|ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.label("N/A");
                                        });
                                    });
                                }
                            });
                        });
                });
            });
        });

        if hovering {
            egui::Area::new(egui::Id::new("drop_hint"))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.label("Drop file to set as reference");
                    });
                });
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
