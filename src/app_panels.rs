//! Main-window panels for `RFMetricsApp`.
//!
//! Extracted from `app.rs` (`eframe::App::ui`): reference panel, bottom
//! action bar, VMAF/options panels, the file-queue center table, and the
//! drop-hint/toast overlays. Pure `impl RFMetricsApp` moves — the thin
//! `ui` orchestrator in `app.rs` (drop routing + worker drains + repaint)
//! calls these; no behavior change.

use crate::app::{
    METRIC_COLUMNS, ROW_HOVER_DELAY, SortColumn, TOAST_SECS, Toast, ToastKind, VIDEO_EXTS,
    WARN_TEXT, apply_alt_include, cycle_sort, done_is_stale, metric_stat_tooltip,
    paint_running_sweep, panel_frame, rank_fill, reveal_in_explorer, shift_include_range,
    sort_mark, sort_view, vline,
};
use crate::metrics::ffmpeg::{MetricKind, ScaleMethod};
use std::path::Path;

impl crate::app::RFMetricsApp {
    /// Reference picker (top, fixed): path row + Browse, probed media
    /// text, Duration/Skip/Pixel-Format boxes, and the 136x76 thumbnail.
    pub(crate) fn show_reference(&mut self, ui: &mut egui::Ui, run_locked: bool, ref_hover: bool) {
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
                                    let browse = ui.add_enabled_ui(!run_locked, |ui| {
                                        ui.add_sized([90.0, 24.0], egui::Button::new("Browse"))
                                    });
                                    if browse.inner.clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .set_title("Select reference video")
                                            .add_filter("Video files", VIDEO_EXTS)
                                            .pick_file()
                                    {
                                        self.ref_path = path.to_string_lossy().into_owned();
                                    }
                                    let _ = ui.add_enabled(
                                        !run_locked,
                                        egui::TextEdit::singleline(&mut self.ref_path)
                                            .desired_width(f32::INFINITY),
                                    );
                                },
                            );
                        });
                        ui.label(&self.ref_info);
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Duration:").selectable(false));
                            let _ = ui
                                .add_enabled(
                                    !run_locked,
                                    egui::TextEdit::singleline(&mut self.duration)
                                        .hint_text("00:00.000")
                                        .desired_width(110.0),
                                )
                                .on_hover_text(
                                    "Clip length to measure: seconds (10) or hh:mm:ss (.000)",
                                );
                            ui.add(egui::Label::new("Skip:").selectable(false));
                            let _ = ui
                                .add_enabled(
                                    !run_locked,
                                    egui::TextEdit::singleline(&mut self.skip)
                                        .hint_text("00:00.000")
                                        .desired_width(110.0),
                                )
                                .on_hover_text("Skip from start: seconds (5) or hh:mm:ss (.000)");
                            ui.add(egui::Label::new("Pixel Format:").selectable(false));
                            ui.add_enabled_ui(!run_locked, |ui| {
                                let _ = egui::ComboBox::from_id_salt("ref_pixfmt")
                                    .width(150.0)
                                    .selected_text(self.ref_pixfmt.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::metrics::ffmpeg::RefPixFmt::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.ref_pixfmt,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Pixel format the reference is converted to (both legs converge on it; \
                                         No conversion keeps the legacy distorted-to-reference legs). The selection \
                                         is ignored for metrics that don't support it (e.g. VMAF requires YUV); \
                                         FFVship metrics always compare unconverted inputs",
                                    );
                            });
                        });
                    });
                    // Reference thumbnail 136x76 (black box parity with Python).
                    egui::Frame::NONE
                        .fill(egui::Color32::BLACK)
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(60)))
                        .show(ui, |ui| {
                            ui.set_min_size(egui::vec2(preview_w, 76.0));
                            if let Some(tex) = &self.thumb_tex {
                                let size = tex.size_vec2();
                                ui.centered_and_justified(|ui| {
                                    ui.image((tex.id(), size));
                                });
                            } else if self.thumb_loading {
                                ui.centered_and_justified(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new("Loading…")
                                                .small()
                                                .color(egui::Color32::from_gray(160)),
                                        )
                                        .selectable(false),
                                    );
                                });
                            } else {
                                ui.centered_and_justified(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new("No preview")
                                                .small()
                                                .color(egui::Color32::from_gray(120)),
                                        )
                                        .selectable(false),
                                    );
                                });
                            }
                        });
                });
            });
        });
        self.ref_rect = Some(ref_resp.response.rect);
    }

    /// Bottom action bar (bottommost): Start/Stop, Reset, Plot,
    /// Bad frames, Save results, plus binary version labels.
    pub(crate) fn show_actions(&mut self, ui: &mut egui::Ui, now: f64, run_locked: bool) {
        // ---- Bottom action bar (bottommost) ----
        egui::Panel::bottom("actions").show(ui, |ui| {
            ui.horizontal(|ui| {
                let run_label = if self.measuring { "Stop" } else { "Start" };
                if ui
                    .add_sized([90.0, 24.0], egui::Button::new(run_label))
                    .clicked()
                {
                    if self.measuring {
                        self.stop_psnr();
                    } else {
                        self.start_run(now);
                    }
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([90.0, 24.0], egui::Button::new("Reset"))
                    })
                    .inner
                    .clicked()
                {
                    self.reset_psnr();
                }
                if ui
                    .add_sized([90.0, 24.0], egui::Button::new("Plot"))
                    .clicked()
                {
                    self.show_plot = true;
                }
                // Bad-frames viewer window (tmp-backed, tabs per metric).
                let bf_enabled = !self.measuring && self.ffmpeg.path.is_some();
                if ui
                    .add_enabled_ui(bf_enabled, |ui| {
                        ui.add_sized([130.0, 24.0], egui::Button::new("Bad frames"))
                    })
                    .inner
                    .on_hover_text("Open the worst-frame viewer (dist/ref side by side)")
                    .clicked()
                {
                    self.show_badframes = true;
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([110.0, 24.0], egui::Button::new("Save results"))
                    })
                    .inner
                    .on_hover_text("Append one row per queued file to RFMetrics.Results.csv")
                    .clicked()
                    && let Some(mut path) = rfd::FileDialog::new()
                        .set_title("Save results CSV")
                        .set_file_name(crate::metrics::results::RESULTS_FILE_NAME)
                        .add_filter("CSV file", &["csv"])
                        .save_file()
                {
                    path.set_extension("csv");
                    self.save_results(now, path);
                }
                ui.label(&self.ffmpeg.short)
                    .on_hover_text(&self.ffmpeg.detail);
                ui.add(egui::Label::new("|").selectable(false));
                ui.label(&self.ffvship.short)
                    .on_hover_text(&self.ffvship.detail);
            });
        });
    }

    /// VMAF options + Options (just above bottom bar).
    pub(crate) fn show_options(&mut self, ui: &mut egui::Ui, run_locked: bool) {
        // ---- VMAF options + Options (just above bottom bar) ----
        egui::Panel::bottom("vmaf").show(ui, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.add(egui::Label::new("VMAF options").selectable(false));
            // Dim when running or when the VMAF header checkbox is off
            // (todo.txt:1 parity with the run_locked inputs above).
            let vmaf_enabled = !run_locked && self.m_vmaf;
            ui.add_enabled_ui(vmaf_enabled, |ui| {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                    ui.add_sized([70.0, 18.0], egui::Label::new("Model").selectable(false));
                    let models = self.vmaf_models.clone();
                    // Tall enough for every model: the default max menu
                    // height scrolls past ~10 entries. This is a ceiling —
                    // the popup still shrinks to its content.
                    let menu_h = models.len() as f32 * 24.0 + 16.0;
                    let _ = egui::ComboBox::from_id_salt("vmaf_model")
                        .width(220.0)
                        .height(menu_h)
                        .selected_text(&self.vmaf_model)
                            .show_ui(ui, |ui| {
                                for m in &models {
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_model,
                                        m.clone(),
                                        m.as_str(),
                                    );
                                }
                            });
                        // v1 phone is the separate `5d0h` file, never the flag:
                        // block the box for v1 models (run-time guard is the
                        // backstop). Unchecking here is visible and keeps the
                        // Start snapshot truthful — no silent coercion.
                        let v1 = crate::metrics::vmaf::is_v1_model(&self.vmaf_model);
                        if v1 {
                            self.vmaf_phone = false;
                        }
                        let _ = ui
                            .add_enabled(
                                !v1,
                                egui::Checkbox::new(&mut self.vmaf_phone, "Phone"),
                            )
                            .on_hover_text(if v1 {
                                "v1 phone is the separate 5d0h model file — pick it with Phone unticked"
                            } else {
                                "Phone viewing-condition transform (v0.6.1 model only)"
                            });
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
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_subsample,
                                        v.to_owned(),
                                        v,
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 18.0], egui::Label::new("Threads").selectable(false));
                        let _ = egui::ComboBox::from_id_salt("vmaf_threads")
                            .width(220.0)
                            .selected_text(&self.vmaf_threads)
                            .show_ui(ui, |ui| {
                                for v in ["auto", "1", "2", "4", "8", "16", "32"] {
                                    let _ = ui.selectable_value(
                                        &mut self.vmaf_threads,
                                        v.to_owned(),
                                        v,
                                    );
                                }
                            })
                            .response
                            .on_hover_text("auto follows the system CPU count");
                    });
                });
            });
                });
                // Global options box, right of VMAF options. Gated on
                // `!run_locked` only (not on `m_vmaf`): the scaler feeds
                // every ffmpeg-backed metric.
                ui.vertical(|ui| {
                    ui.add(egui::Label::new("Options").selectable(false));
                    ui.add_enabled_ui(!run_locked, |ui| {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal_top(|ui| {
                                ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Scaling").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("scale_method")
                                    .width(220.0)
                                    .selected_text(self.scale_method.label())
                                    .show_ui(ui, |ui| {
                                        for m in ScaleMethod::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.scale_method,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "sws scaler for every scale filter the app emits; \
                                         FFmpeg default omits flags (bicubic in practice)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Framerate").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("fps_mode")
                                    .width(220.0)
                                    .selected_text(self.fps_mode.label())
                                    .show_ui(ui, |ui| {
                                        use crate::metrics::ffmpeg::InputFpsMode;
                                        for m in InputFpsMode::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.fps_mode,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "which -r rate ffmpeg forces on each input, e.g. ref 23.98 / \
                                         dist 23.81: Reference emits -r 23.98 before both inputs, so \
                                         a one-sided VFR misread cannot desync the pair \
                                         (FFMetrics #111); Per-input emits -r 23.81 then -r 23.98 \
                                         (upstream 1.4.5 parity — disagreeing detections desync \
                                         scores); Off emits no -r and trusts container timestamps",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Plot size").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("plot_size")
                                    .width(220.0)
                                    .selected_text(self.plot_size.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::plot::PlotSize::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.plot_size,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Image dimensions for Save PNG and Copy (current plot view)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Cell value").selectable(false),
                                );
                                let _ = egui::ComboBox::from_id_salt("cell_stat")
                                    .width(220.0)
                                    .selected_text(self.cell_stat.label())
                                    .show_ui(ui, |ui| {
                                        for m in crate::metrics::CellStat::ALL {
                                            let _ = ui.selectable_value(
                                                &mut self.cell_stat,
                                                m,
                                                m.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Which per-run stat metric cells show, sort by, and copy (default Avg)",
                                    );
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Precision").selectable(false),
                                );
                                let prev_precision = self.cell_precision;
                                let _ = egui::ComboBox::from_id_salt("cell_precision")
                                    .width(220.0)
                                    .selected_text(self.cell_precision.to_string())
                                    .show_ui(ui, |ui| {
                                        for p in 0..=crate::metrics::MAX_PRECISION as u8 {
                                            let _ = ui.selectable_value(
                                                &mut self.cell_precision,
                                                p,
                                                p.to_string(),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Decimals for metric cell display and Copy value (default 4)",
                                    );
                                // Frozen Avg texts embed the old precision —
                                // re-freeze them once on change (non-Avg
                                // selectors format live and follow for free).
                                if self.cell_precision != prev_precision {
                                    self.refreeze_cell_texts();
                                }
                            });
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.plot_at_start,
                                    "Plot window at start",
                                ))
                                .on_hover_text(
                                    "Open the plot window automatically when a run starts",
                                );
                            if ui
                                .add(egui::Button::new("Refresh Files Media Info"))
                                .on_hover_text(
                                    "Re-probe the reference and every queued file (media info only)",
                                )
                                .clicked()
                            {
                                self.refresh_media_info();
                            }
                                });
                                ui.vertical(|ui| {
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.csv_export,
                                    "Save frames metrics to CSV files",
                                ))
                                .on_hover_text(
                                    "Write one <name>.<METRIC>.csv per finished run \
                                     (TAB-separated per-frame values), to the chosen CSV folder \
                                     or beside each distorted file when empty",
                                );
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("CSV folder").selectable(false),
                                );
                                // Bounded display (full path stays in the hover);
                                // the worker snapshots the real string at Start.
                                let full = self.csv_dir.clone();
                                let shown = if full.trim().is_empty() {
                                    "Beside distorted files".to_owned()
                                } else if full.chars().count() > 40 {
                                    format!(
                                        "…{}",
                                        full.chars().skip(full.chars().count() - 39).collect::<String>()
                                    )
                                } else {
                                    full.clone()
                                };
                                ui.label(shown).on_hover_text(if full.trim().is_empty() {
                                    "Empty: each CSV lands next to its distorted file".to_owned()
                                } else {
                                    full
                                });
                                if ui.button("Browse…").clicked()
                                    && let Some(dir) = rfd::FileDialog::new()
                                        .set_title("CSV output folder")
                                        .pick_folder()
                                {
                                    self.csv_dir = dir.to_string_lossy().into_owned();
                                }
                                if ui
                                    .button("Clear")
                                    .on_hover_text("Back to beside-the-distorted-file")
                                    .clicked()
                                {
                                    self.csv_dir.clear();
                                }
                            });
                            let _ = ui
                                .add(egui::Checkbox::new(
                                    &mut self.results_autosave,
                                    "Auto-save results",
                                ))
                                .on_hover_text(
                                    "Append one row per queued file to the results file \
                                     when each run ends, stopped runs included",
                                );
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [70.0, 18.0],
                                    egui::Label::new("Results file").selectable(false),
                                );
                                // Bounded display (full path stays in the hover);
                                // resolved (custom or exe-dir default) at save time.
                                let full = self.results_path.clone();
                                let empty = full.trim().is_empty();
                                let shown = if empty {
                                    "RFMetrics.Results.csv next to exe".to_owned()
                                } else if full.chars().count() > 40 {
                                    format!(
                                        "…{}",
                                        full.chars().skip(full.chars().count() - 39).collect::<String>()
                                    )
                                } else {
                                    full.clone()
                                };
                                ui.label(shown).on_hover_text(if empty {
                                    crate::metrics::results::default_results_path()
                                        .to_string_lossy()
                                        .into_owned()
                                } else {
                                    full.clone()
                                });
                                if ui.button("Browse…").clicked() {
                                    let mut dialog = rfd::FileDialog::new()
                                        .set_title("Results file")
                                        .add_filter("CSV file", &["csv"]);
                                    if empty {
                                        dialog = dialog.set_file_name(
                                            crate::metrics::results::RESULTS_FILE_NAME,
                                        );
                                    } else {
                                        dialog = dialog.set_file_name(&full);
                                    }
                                    if let Some(mut path) = dialog.save_file() {
                                        path.set_extension("csv");
                                        self.results_path =
                                            path.to_string_lossy().into_owned();
                                    }
                                }
                                if ui
                                    .button("Clear")
                                    .on_hover_text("Back to next-to-the-exe default")
                                    .clicked()
                                {
                                    self.results_path.clear();
                                }
                            });
                                });
                            });
                        });
                    });
                });
            });
        });
    }

    /// File queue (center, expanding): Add/Remove buttons plus the sortable
    /// metric table with rank fills, stale badges, and context menus.
    pub(crate) fn show_queue(
        &mut self,
        ui: &mut egui::Ui,
        now: f64,
        run_locked: bool,
        table_hover: bool,
    ) {
        // ---- File queue (center, expanding) ----
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([110.0, 24.0], egui::Button::new("Add files"))
                    })
                    .inner
                    .clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .set_title("Select video files")
                        .add_filter("Video files", VIDEO_EXTS)
                        .add_filter("All files", &["*"])
                        .pick_files()
                {
                    self.add_queue_files(paths);
                }
                if ui
                    .add_enabled_ui(!run_locked, |ui| {
                        ui.add_sized([130.0, 24.0], egui::Button::new("Remove Selected"))
                    })
                    .inner
                    .clicked()
                {
                    self.rows.retain(|r| !r.selected);
                    self.refresh_queue_names();
                    // The scored set may have shrunk: re-rank all columns.
                    for kind in MetricKind::ALL {
                        self.refresh_ranks(kind);
                    }
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
                    // Metric stats + ranks come from the per-row cache (H1:
                    // computed on result arrival, never per frame).
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
                    // Row-click side effects deferred past the loop (L3): the
                    // hot path borrows rows read-only, so no per-frame clones
                    // are needed to dodge the borrow checker.
                    let mut toggle_row: Option<usize> = None;
                    let mut open_path: Option<String> = None;
                    let mut reveal_path: Option<String> = None;
                    let mut hovered_next: Option<usize> = None;
                    // Alt+click solo target: set inside the row loop when a
                    // checkbox flips with Alt held, applied once below.
                    let mut alt_solo: Option<usize> = None;
                    // Shift+click range fill: `(lo, hi, value)` span plus
                    // the anchor update, both applied once below.
                    let mut shift_range: Option<(usize, usize, bool)> = None;
                    let mut anchor_next: Option<usize> = None;
                    // Range anchor snapshot (stale = dangling past the
                    // queue → plain toggle); copied before the `&mut`
                    // flag borrows below so no shared borrow lives on.
                    let anchor = self.include_anchor.filter(|&a| a < self.rows.len());
                    let sel_anchor = self.selected_anchor.filter(|&a| a < self.rows.len());
                    // Sorted display order as underlying indices (identity
                    // when unsorted); anchors/toggles below stay underlying
                    // so they survive re-sorts without invalidation.
                    let view: Vec<usize> = sort_view(&self.rows, self.sort_spec, self.cell_stat);
                    let sort_spec = self.sort_spec;
                    // Header sort click, applied once below the loop.
                    let mut sort_click: Option<SortColumn> = None;
                    // Header per-metric Reset, applied once below the loop
                    // (deferred so the `&mut` flag borrows above don't
                    // conflict with the `&mut self` reset call).
                    let mut reset_click: Option<MetricKind> = None;
                    // Current-settings snapshot for the stale badge: one
                    // trim parse per frame, then comparisons only per cell.
                    // Invalid boxes ("bad time") disable the badge —
                    // `start_run` reports those as errors instead.
                    let stale_cur = match (
                        Self::trim_opt(&self.skip),
                        Self::trim_opt(&self.duration),
                    ) {
                        (Some(s), Some(c)) => Some((
                            s,
                            c,
                            self.current_vmaf_cfg(),
                            self.scale_method,
                            self.fps_mode,
                            self.ref_pixfmt,
                        )),
                        _ => None,
                    };
                    table
                        .header(18.0, |mut header| {
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|_| {});
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                // Sort click (3-state: A-Z → Z-A → insertion);
                                // the mark is a painted vector triangle (see
                                // `sort_mark` — the UI font lacks ▲▼⇅).
                                let dir = match sort_spec {
                                    Some((SortColumn::Path, d)) => Some(d),
                                    _ => None,
                                };
                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("Path").strong())
                                            .frame(false),
                                    )
                                    .on_hover_text("Sort by path (A-Z, Z-A, insertion order)")
                                    .clicked()
                                    || sort_mark(ui, dir)
                                        .on_hover_text("Sort by path (A-Z, Z-A, insertion order)")
                                        .clicked()
                                {
                                    sort_click = Some(SortColumn::Path);
                                }
                            });
                            header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                            header.col(|ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("Media info").strong())
                                        .selectable(false),
                                );
                            });
                            // Issue #7: copy support out first — the loop takes
                            // `&mut` flag borrows, so no shared `self` borrow
                            // may live across it. FFVship readiness rides the
                            // same pattern (`usable` covers absent + wrong-GPU
                            // builds); its disabled hover shows the binary
                            // detail (path / GPU-mismatch hint) instead of
                            // the ffmpeg filter text below.
                            let (psnr_ok, ssim_ok, vmaf_ok, xpsnr_ok) = {
                                let sup = &self.ffmpeg.supported_metrics;
                                (
                                    sup.contains(&MetricKind::Psnr),
                                    sup.contains(&MetricKind::Ssim),
                                    sup.contains(&MetricKind::Vmaf),
                                    sup.contains(&MetricKind::Xpsnr),
                                )
                            };
                            let ffvship_ok = self.ffvship.usable;
                            let ffvship_detail = self.ffvship.detail.clone();
                            for (flag, name, ok, kind) in [
                                (&mut self.m_psnr, "PSNR", psnr_ok, MetricKind::Psnr),
                                (&mut self.m_ssim, "SSIM", ssim_ok, MetricKind::Ssim),
                                (&mut self.m_vmaf, "VMAF", vmaf_ok, MetricKind::Vmaf),
                                (&mut self.m_xpsnr, "XPSNR", xpsnr_ok, MetricKind::Xpsnr),
                                (&mut self.m_ssim2, "SSIM2", ffvship_ok, MetricKind::Ssim2),
                                (&mut self.m_but, "BUTTER", ffvship_ok, MetricKind::But),
                                (&mut self.m_cvvdp, "CVVDP", ffvship_ok, MetricKind::Cvvdp),
                            ] {
                                header.col(|ui| vline(ui, egui::Color32::from_gray(0x8A)));
                                header.col(|ui| {
                                    // Sort click (3-state: best first →
                                    // reversed → insertion); the checkbox
                                    // keeps its enable/disable job. The mark
                                    // is a painted vector triangle (see
                                    // `sort_mark` — the UI font lacks ▲▼⇅).
                                    let col = SortColumn::Metric(kind);
                                    let dir = match sort_spec {
                                        Some((c, d)) if c == col => Some(d),
                                        _ => None,
                                    };
                                    if sort_mark(ui, dir)
                                        .on_hover_text(format!(
                                            "Sort by {name} (best first, reversed, insertion order)"
                                        ))
                                        .clicked()
                                    {
                                        sort_click = Some(col);
                                    }
                                    let mut resp = ui.add_enabled(
                                        ok && !run_locked,
                                        egui::Checkbox::new(flag, name),
                                    );
                                    if !ok {
                                        resp = if kind.is_ffvship() {
                                            resp.on_hover_text(&ffvship_detail)
                                        } else {
                                            resp.on_hover_text(format!(
                                                "{name} filter not supported by this ffmpeg build"
                                            ))
                                        };
                                    }
                                    resp.context_menu(|ui| {
                                        if ui
                                            .add_enabled(
                                                !run_locked,
                                                egui::Button::new(format!("Reset {name}")),
                                            )
                                            .clicked()
                                        {
                                            reset_click = Some(kind);
                                            ui.close();
                                        }
                                    });
                                });
                            }
                        })
                        .body(|body| {
                            body.rows(20.0, self.rows.len(), |mut row| {
                                let i = row.index();
                                // Display position → underlying row: every
                                // `vi` use below addresses the real row, so
                                // selection/anchors survive re-sorts.
                                let vi = view[i];
                                row.set_selected(self.rows[vi].selected);
                                // Delayed hover: only outline after the pointer
                                // rests on the row, so passing over rows while
                                // aiming at text doesn't flash each one.
                                let hover_delayed = self.hover_row == Some(vi)
                                    && self.hover_since.is_some_and(|t| now - t >= ROW_HOVER_DELAY);
                                row.set_hovered(hover_delayed);
                                // Free-space click toggles selection; widget clicks
                                // (checkbox, play, text drag-select) must not.
                                // (`toggle_row` etc. are set here, applied below.)
                                row.col(|ui| {
                                    let alt = ui.input(|i| i.modifiers.alt);
                                    let shift = ui.input(|i| i.modifiers.shift);
                                    let resp = ui
                                        .checkbox(&mut self.rows[vi].include, "")
                                        .on_hover_text("Include in run and plot");
                                    if resp.changed() {
                                        if alt {
                                            // Alt wins over Shift on combo.
                                            alt_solo = Some(vi);
                                        } else if shift {
                                            // Gmail-style: the span takes
                                            // the clicked box's post-toggle
                                            // value. The span is VIEW
                                            // positions (`i` already is one;
                                            // the underlying anchor maps
                                            // through `view`) and is mapped
                                            // back at apply time; a stale
                                            // anchor falls through to a
                                            // plain toggle.
                                            if let Some((lo, hi)) = anchor
                                                .and_then(|a| view.iter().position(|&u| u == a))
                                                .and_then(|ap| {
                                                    shift_include_range(view.len(), ap, i)
                                                })
                                            {
                                                shift_range = Some((lo, hi, self.rows[vi].include));
                                            }
                                        }
                                        // Every click moves the anchor, so
                                        // chained Shift+clicks extend from
                                        // the last clicked row.
                                        anchor_next = Some(vi);
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                row.col(|ui| {
                                    if ui.button("▶").clicked() {
                                        open_path = Some(self.rows[vi].path.clone());
                                    }
                                });
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) = row.col(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                    let row_data = &self.rows[vi];
                                    // Plain non-selectable text: no button hover
                                    // outline; copy lives in the right-click menu
                                    // and the full path shows as tooltip (Python parity).
                                    ui.add(egui::Label::new(&row_data.display).selectable(false))
                                        .on_hover_text(&row_data.path);
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Show in explorer").clicked() {
                                        reveal_path = Some(self.rows[vi].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy path").clicked() {
                                        ui.ctx().copy_text(self.rows[vi].path.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy filename").clicked() {
                                        let name = Path::new(&self.rows[vi].path)
                                            .file_name()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| self.rows[vi].display.clone());
                                        ui.ctx().copy_text(name);
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) =
                                    row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                let (_, r) = row.col(|ui| {
                                    let row_data = &self.rows[vi];
                                    // Cross-format warning (upstream #47):
                                    // the runners silently scale/convert a
                                    // distorted leg to the reference (FFVship
                                    // normalizes nothing at all), so a row
                                    // whose probe differs from the reference
                                    // ambers with the conversions listed in
                                    // its tooltip. Suppressed while a pixel
                                    // format target is selected (the
                                    // convergence is explicitly requested).
                                    // Comparisons only when matched; strings
                                    // alloc on mismatch.
                                    let warns = match (
                                        self.ref_info_data.as_ref(),
                                        row_data.info.as_ref(),
                                    ) {
                                        (Some(r), Some(d))
                                            if self.ref_pixfmt
                                                == crate::metrics::ffmpeg::RefPixFmt::NoConversion =>
                                        {
                                            crate::metrics::ffmpeg::conversion_warnings(r, d)
                                        }
                                        _ => Vec::new(),
                                    };
                                    let rich = if warns.is_empty() {
                                        egui::RichText::new(&row_data.media)
                                    } else {
                                        egui::RichText::new(&row_data.media).color(WARN_TEXT)
                                    };
                                    let resp = ui.add(
                                        egui::Label::new(rich).selectable(false),
                                    );
                                    if warns.is_empty() {
                                        resp.on_hover_text(&row_data.media_tip);
                                    } else {
                                        resp.on_hover_ui(|ui| {
                                            ui.label(&row_data.media_tip);
                                            ui.add_space(5.0);
                                            ui.label(
                                                egui::RichText::new(
                                                    "Cross-format vs reference:",
                                                )
                                                .strong(),
                                            );
                                            for w in &warns {
                                                ui.label(w.describe());
                                            }
                                        });
                                    }
                                });
                                r.context_menu(|ui| {
                                    if ui.button("Copy summary").clicked() {
                                        ui.ctx().copy_text(crate::probe::table_media_text(
                                            self.rows[vi].info.as_ref(),
                                        ));
                                        ui.close();
                                    }
                                    if ui.button("Copy details").clicked() {
                                        ui.ctx().copy_text(self.rows[vi].media_tip.clone());
                                        ui.close();
                                    }
                                });
                                if r.clicked() {
                                    toggle_row = Some(vi);
                                }
                                // Metric columns in METRIC_COLUMNS order: live state
                                // text on a rank fill (best green, worst red,
                                // tie dim yellow) with per-stat chip tooltips
                                // for Done cells.
                                for (kind_opt, title) in METRIC_COLUMNS {
                                    let Some(kind) = kind_opt else {
                                        continue;
                                    };
                                    let (_, r) =
                                        row.col(|ui| vline(ui, egui::Color32::from_gray(0x38)));
                                    if r.clicked() {
                                        toggle_row = Some(vi);
                                    }
                                    let (_, r) = row.col(|ui| {
                                        let row_data = &self.rows[vi];
                                        let cell = row_data.cell(kind);
                                        let cached = row_data.cached(kind);
                                        // Stale `Done` values (stamped options
                                        // no longer match current settings)
                                        // keep their value and rank fill —
                                        // only struck-through (issue #92).
                                        let stale = match &stale_cur {
                                            Some((s, c, vmaf, scaler, fps, pf)) => done_is_stale(
                                                kind, cell, *s, *c, vmaf, *scaler, *fps, *pf,
                                            ),
                                            None => false,
                                        };
                                        // Idle borrows a static, Avg-selected Done
                                        // borrows the arrival-frozen text, other
                                        // selectors format from the cached stats
                                        // (one short format per visible cell per
                                        // frame, only when non-Avg is selected),
                                        // Error borrows its message; only live
                                        // Running frames format per frame
                                        // (they change anyway).
                                        let sel = self.cell_stat;
                                        let prec = self.cell_precision as usize;
                                        let running;
                                        let selected;
                                        let text: &str = match cell {
                                            crate::metrics::MetricCell::Idle => "N/A",
                                            crate::metrics::MetricCell::Running {
                                                frame, ..
                                            } => {
                                                running = format!("Frame: {frame}");
                                                &running
                                            }
                                            crate::metrics::MetricCell::Done { .. }
                                                if sel == crate::metrics::CellStat::Avg =>
                                            {
                                                &cached.text
                                            }
                                            crate::metrics::MetricCell::Done { .. } => {
                                                match cached.stats.as_ref() {
                                                    Some(s) => {
                                                        selected = format!(
                                                            "{:.prec$}",
                                                            s.value(sel),
                                                            prec = prec
                                                        );
                                                        &selected
                                                    }
                                                    None => &cached.text,
                                                }
                                            }
                                            crate::metrics::MetricCell::Error { msg } => msg,
                                        };
                                        // Borrow the cached stats when scored;
                                        // unscored cells build their one-line tip
                                        // below, and only while hovered.
                                        let stats = cached.stats.as_ref();
                                        let ranks = cached.ranks;
                                        let mut cell_frame = egui::Frame::NONE;
                                        if let Some(fill) = rank_fill(ranks[sel.index()]) {
                                            cell_frame = cell_frame.fill(fill);
                                        }
                                        // Only the job on the Progress/Series feed
                                        // sweeps; queued `Running` cells wait
                                        // statically until their turn.
                                        let is_live = matches!(
                                            cell,
                                            crate::metrics::MetricCell::Running { .. }
                                        ) && self.live_kind == Some(kind)
                                            && self.live_key.as_deref()
                                                == Some(row_data.key.as_str());
                                        cell_frame.show(ui, |ui| {
                                            ui.set_width(ui.available_width());
                                            ui.centered_and_justified(|ui| {
                                                if is_live {
                                                    paint_running_sweep(ui);
                                                }
                                                // Plain non-selectable text, like the
                                                // Path/Media columns (copy lives in
                                                // the right-click menu); stale
                                                // values strike through.
                                                let rich = if stale {
                                                    egui::RichText::new(text).strikethrough()
                                                } else {
                                                    egui::RichText::new(text)
                                                };
                                                let resp = ui
                                                    .add(egui::Label::new(rich).selectable(false));
                                                match stats {
                                                    Some(stats) => {
                                                        resp.on_hover_ui(|ui| {
                                                            if stale {
                                                                ui.label("Stale settings - rerun to refresh");
                                                            }
                                                            metric_stat_tooltip(
                                                                ui, title, stats, &ranks, sel,
                                                            );
                                                        });
                                                    }
                                                    None => {
                                                        if resp.hovered() {
                                                            resp.on_hover_text(cell.tooltip(title));
                                                        }
                                                    }
                                                }
                                            });
                                        });
                                    });
                                    // Right-click copies (Media-column parity):
                                    // value = the visible selector stat,
                                    // summary = the whole tooltip stats block.
                                    r.context_menu(|ui| {
                                        if ui.button("Copy value").clicked() {
                                            ui.ctx().copy_text(
                                                self.rows[vi].cell(kind).cell_stat_text_prec(
                                                    self.cell_stat,
                                                    self.cell_precision as usize,
                                                ),
                                            );
                                            ui.close();
                                        }
                                        if ui.button("Copy summary").clicked() {
                                            ui.ctx()
                                                .copy_text(self.rows[vi].cell(kind).tooltip(title));
                                            ui.close();
                                        }
                                    });
                                    if r.clicked() {
                                        toggle_row = Some(vi);
                                    }
                                }
                                if row.response().hovered() {
                                    hovered_next = Some(vi);
                                }
                            });
                        });
                    // Deferred row-click side effects (L3): selection
                    // toggle, open-in-player (error toast needs `now`),
                    // reveal-in-explorer, and hover tracking all land after
                    // the loop. Modifiers are read here, in the same frame
                    // as the clicks above, so they match the click-time
                    // state without threading through every cell.
                    let (sel_alt, sel_shift) = ui.input(|i| (i.modifiers.alt, i.modifiers.shift));
                    if let Some(i) = toggle_row
                        && i < self.rows.len()
                    {
                        if sel_alt {
                            // Alt-solo/select-all, mirroring the include
                            // column: the helper's call depends only on
                            // other-rows state, so it runs directly on the
                            // unflipped column (rare action: one small
                            // alloc is fine — never per frame).
                            let mut flags: Vec<bool> =
                                self.rows.iter().map(|r| r.selected).collect();
                            apply_alt_include(&mut flags, i);
                            for (r, v) in self.rows.iter_mut().zip(flags) {
                                r.selected = v;
                            }
                        } else if sel_shift {
                            // Checkbox-style: the span takes the clicked
                            // row's post-toggle state — Shift+clicking a
                            // selected row unselects the range, an
                            // unselected one selects it. The span is VIEW
                            // positions (mapped back through `view`);
                            // stale/missing anchor falls through to a
                            // plain toggle.
                            let click_pos = view.iter().position(|&u| u == i);
                            let anchor_pos =
                                sel_anchor.and_then(|a| view.iter().position(|&u| u == a));
                            match (anchor_pos, click_pos) {
                                (Some(ap), Some(cp)) => {
                                    let (lo, hi) = (ap.min(cp), ap.max(cp));
                                    let v = !self.rows[i].selected;
                                    for &u in &view[lo..=hi] {
                                        self.rows[u].selected = v;
                                    }
                                }
                                _ => {
                                    self.rows[i].selected = !self.rows[i].selected;
                                }
                            }
                        } else {
                            self.rows[i].selected = !self.rows[i].selected;
                        }
                        // Every selection click moves the anchor, so chained
                        // Shift+clicks extend from the last clicked row.
                        self.selected_anchor = Some(i);
                    }
                    // Alt+click solo/select-all: the clicked box already
                    // flipped above; the helper overwrites the whole column
                    // from that post-toggle state (rare action: one small
                    // alloc is fine — never on the per-frame hot path).
                    if let Some(i) = alt_solo
                        && i < self.rows.len()
                    {
                        let mut flags: Vec<bool> = self.rows.iter().map(|r| r.include).collect();
                        apply_alt_include(&mut flags, i);
                        for (r, v) in self.rows.iter_mut().zip(flags) {
                            r.include = v;
                        }
                    }
                    // Shift+click range fill skipped when Alt solo ran.
                    // The span is VIEW positions, mapped back through `view`.
                    if alt_solo.is_none()
                        && let Some((lo, hi, v)) = shift_range
                        && hi < view.len()
                    {
                        for &u in &view[lo..=hi] {
                            self.rows[u].include = v;
                        }
                    }
                    if let Some(a) = anchor_next {
                        self.include_anchor = Some(a);
                    }
                    // Header sort click: cycle best-first → reversed →
                    // insertion order (session-only; run/state order stays
                    // insertion).
                    if let Some(col) = sort_click {
                        self.sort_spec = cycle_sort(self.sort_spec, col, self.cell_stat);
                    }
                    // Header per-metric Reset: one column back to Idle.
                    if let Some(kind) = reset_click {
                        self.reset_metric(kind);
                    }
                    if let Some(path) = open_path
                        && let Err(e) = open::that(&path)
                    {
                        log::error!(target: "rfmetrics::app", "open \"{path}\" failed: {e}");
                        self.toast = Some(Toast {
                            text: format!("Could not open file: {e}"),
                            until: now + TOAST_SECS,
                            kind: ToastKind::Error,
                        });
                    }
                    if let Some(path) = reveal_path
                        && let Err(e) = reveal_in_explorer(&path)
                    {
                        log::error!(target: "rfmetrics::app", "reveal \"{path}\" failed: {e}");
                        self.toast = Some(Toast {
                            text: format!("Could not show in explorer: {e}"),
                            until: now + TOAST_SECS,
                            kind: ToastKind::Error,
                        });
                    }
                    self.hovered_now = hovered_next;
                });
            });
            self.table_rect = Some(table_resp.response.rect);
            // Ctrl/Cmd+A select-all for the removal set, scoped to
            // pointer-over-table (the "focused" proxy — egui tables take no
            // keyboard focus). Runs after the table so a focused TextEdit
            // (ref path/trim boxes, built above) consumes the key first and
            // keeps its select-all-text behavior.
            if !self.rows.is_empty()
                && self.table_rect.is_some_and(|r| ui.rect_contains_pointer(r))
                && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::A))
            {
                for r in &mut self.rows {
                    r.selected = true;
                }
                // Anchor at the end: a follow-up Shift+click on a selected
                // row unselects the trailing range (checkbox-style).
                self.selected_anchor = Some(self.rows.len() - 1);
            }
            // Roll the delayed-hover timer forward; schedule one wake-up
            // for when the delay elapses so the outline appears without
            // moving (no every-frame spin while pending).
            if self.hovered_now != self.hover_row {
                self.hover_row = self.hovered_now;
                self.hover_since = self.hover_row.map(|_| now);
            }
            let hover_pending = matches!(
                (self.hover_row, self.hovered_now, self.hover_since),
                (Some(a), Some(b), Some(t)) if a == b && now - t < ROW_HOVER_DELAY
            );
            if hover_pending {
                // `hover_pending` implies `hover_since` is `Some`.
                let remaining = self
                    .hover_since
                    .map(|t| (ROW_HOVER_DELAY - (now - t)).max(0.0))
                    .unwrap_or(ROW_HOVER_DELAY);
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(remaining));
            }
        });
    }

    /// Drop-hint overlay over the active drop target + the transient toast.
    pub(crate) fn show_overlays(
        &mut self,
        ui: &mut egui::Ui,
        now: f64,
        hovering: bool,
        ref_hover: bool,
        table_hover: bool,
    ) {
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
        // Static text needs exactly two frames (show + hide): schedule one
        // wake-up at expiry instead of full-rate repaints for 3 s.
        if let Some(toast) = self.toast.clone() {
            if now < toast.until {
                let remaining = (toast.until - now).max(0.0);
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(remaining));
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
