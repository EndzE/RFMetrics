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
    files: Vec<String>,
    ffmpeg: crate::binaries::BinaryInfo,
    ffvship: crate::binaries::BinaryInfo,
    #[allow(dead_code)] // consumed by the upcoming probe step
    ffprobe: Option<std::path::PathBuf>,
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
            files: Vec::new(),
            ffmpeg,
            ffvship,
            ffprobe,
        }
    }
}

fn vsep(ui: &mut egui::Ui) {
    let _ = ui.add(egui::Separator::default().vertical());
}

impl eframe::App for RFMetricsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // ---- Reference (top, fixed) ----
        egui::Panel::top("reference").show(ui, |ui| {
            ui.label("Reference");
            egui::Frame::group(ui.style()).show(ui, |ui| {
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
                                        .add_sized(
                                            [90.0, 24.0],
                                            egui::Button::new("Browse"),
                                        )
                                        .clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_title("Select reference video")
                                            .add_filter(
                                                "Video files",
                                                &[
                                                    "mp4", "mkv", "mov", "avi",
                                                    "webm", "m2ts", "ts", "m4v",
                                                ],
                                            )
                                            .pick_file()
                                    {
                                        self.ref_path =
                                            path.to_string_lossy().into_owned();
                                    }
                                    let _ = ui.add(
                                        egui::TextEdit::singleline(&mut self.ref_path)
                                            .desired_width(f32::INFINITY),
                                    );
                                },
                            );
                        });
                        ui.label(
                            "Encoder: -unknown-, Frame: -unknown-, Bitrate: -unknown-, Duration: -unknown-",
                        );
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
                        .stroke(egui::Stroke::new(
                            1.0,
                            egui::Color32::from_gray(60),
                        ))
                        .show(ui, |ui| {
                            ui.allocate_space(egui::vec2(preview_w, 76.0));
                        });
                });
            });
        });

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
                let _ = ui.add_sized([110.0, 24.0], egui::Button::new("Add files"));
                let _ = ui.add_sized([130.0, 24.0], egui::Button::new("Remove Selected"));
            });
            ui.add_space(4.0);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                // ponytail: single horizontal scroll fallback so CVVDP can never clip
                egui::ScrollArea::horizontal()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Header mirrors Python widths: sel 22, btn 20, media 240, metric 82.
                        // Tight spacing like Python's padx=1..4 (egui default 8 overflows ~50px).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.allocate_space(egui::vec2(22.0, 18.0));
                            vsep(ui);
                            ui.allocate_space(egui::vec2(20.0, 18.0));
                            vsep(ui);
                            // Path takes the remaining width; reserve accounts for
                            // media + 7 metrics + 8 separators + gaps at 4px spacing.
                            let reserve = 240.0 + 7.0 * 82.0 + 8.0 * 5.0 + 16.0 * 4.0;
                            let path_w = (ui.available_width() - reserve).max(80.0);
                            ui.add_sized(
                                [path_w, 18.0],
                                egui::Label::new(egui::RichText::new("Path").strong()),
                            );
                            vsep(ui);
                            ui.add_sized(
                                [240.0, 18.0],
                                egui::Label::new(egui::RichText::new("Media info").strong()),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_psnr, "PSNR"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_ssim, "SSIM"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_vmaf, "VMAF"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_xpsnr, "XPSNR"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_ssim2, "SSIM2"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_but, "BUT"),
                            );
                            vsep(ui);
                            let _ = ui.add_sized(
                                [82.0, 18.0],
                                egui::Checkbox::new(&mut self.m_cvvdp, "CVVDP"),
                            );
                        });
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.set_min_height(160.0);
                                if self.files.is_empty() {
                                    ui.weak("No files yet — drag & drop video files here");
                                } else {
                                    for f in &self.files {
                                        ui.label(f);
                                    }
                                }
                            });
                    });
            });
        });
    }
}
