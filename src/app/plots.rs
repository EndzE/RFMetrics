//! Plot viewport domain: `PlotRuntime` state, PNG export message types,
//! and the metrics viewport (own OS window).

use crate::metrics::ffmpeg::MetricKind;
use std::sync::mpsc::{Receiver, Sender};

/// Metric plot viewport state (own OS viewport while open).
/// Session-only, like the Python window — never persisted.
pub(crate) struct PlotRuntime {
    /// PSNR plot viewport open (Python `plot["win"]` parity: closing the
    /// window withdraws it, Plot reopens it).
    pub(crate) open: bool,
    /// Selected plot viewport tab (session-only, like the Python window).
    pub(crate) tab: MetricKind,
    /// Last measured tab-strip box width, for centering the strip
    /// (session-only; texts are static so it converges in one frame).
    pub(crate) tabs_w: f32,
    /// Grow-only live fit per open tab while any series is running;
    /// cleared once all settle, so finished graphs fit exactly again.
    pub(crate) live_fit: Option<(MetricKind, crate::plot::FitBounds)>,
    /// Follow poke still owed: set when a live phase starts without plot
    /// memory present (window just opened), retried until it lands.
    pub(crate) follow_pending: bool,
    /// Reset-view click still owed: the help-bar `Ui` scopes persistent
    /// ids differently than the canvas `Ui`, so the button only arms this
    /// flag and the central panel (plot id scope) executes the poke.
    /// Retried until plot memory exists, like the follow poke.
    pub(crate) reset_pending: bool,
    /// Snap-to-data lock (plot window checkbox, session-only): panning is
    /// clamped to the first/last frame on x and the plotted min/max on y;
    /// zooming and in-limits panning stay free.
    pub(crate) snap: bool,
    /// Pending plot export (Save PNG / Copy button), executed in the
    /// central panel where the plot id scope lives.
    pub(crate) save_pending: Option<PlotExport>,
    /// Plot export worker channel + busy flag: while `png_saving` both the
    /// Save PNG and Copy buttons are disabled so 5 s renders can't overlap.
    pub(crate) png_tx: Sender<PngSaveMsg>,
    pub(crate) png_rx: Receiver<PngSaveMsg>,
    pub(crate) png_saving: bool,
}

/// PNG export result from the one-shot saver thread. The supersampled
/// render + Lanczos3 downscale blocks for seconds, so it never runs on the
/// UI thread; the worker sends the outcome back here for a toast. Copy jobs
/// send pixels back because `ctx.copy_image()` must run on the UI thread
/// (winit executes it as a frame-end `OutputCommand`).
pub(crate) enum PngSaveMsg {
    Saved { path: std::path::PathBuf },
    CopyReady { w: u32, h: u32, rgba: Vec<u8> },
    SaveFailed { err: String },
    CopyFailed { err: String },
}

/// Pending plot export: file save (filename captured at click time) or
/// clipboard copy. Executed in the central panel where the plot id scope
/// (for the current view bounds) lives.
pub(crate) enum PlotExport {
    Save { name: String },
    Copy,
}

impl PlotRuntime {
    /// Apply plot export thread results; clears the Saving…/Copying…
    /// lock so the buttons re-arm. Copy pixels land here because
    /// `ctx.copy_image()` must run on the UI thread. Runs on the main
    /// viewport each frame. Returns whether any message arrived.
    pub(crate) fn drain_png_results(
        &mut self,
        ctx: &egui::Context,
        now: f64,
        ui: &mut crate::app::UiState,
    ) -> bool {
        let mut activity = false;
        while let Ok(msg) = self.png_rx.try_recv() {
            activity = true;
            self.png_saving = false;
            match msg {
                PngSaveMsg::Saved { path } => {
                    ui.toast(
                        now,
                        format!("Plot saved to {}", path.display()),
                        crate::app::ToastKind::Info,
                    );
                }
                PngSaveMsg::CopyReady { w, h, rgba } => {
                    ctx.copy_image(egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        &rgba,
                    ));
                    ui.toast(
                        now,
                        "Plot copied to clipboard".to_owned(),
                        crate::app::ToastKind::Info,
                    );
                }
                PngSaveMsg::SaveFailed { err } => {
                    ui.toast(
                        now,
                        format!("Could not save plot: {err}"),
                        crate::app::ToastKind::Error,
                    );
                }
                PngSaveMsg::CopyFailed { err } => {
                    ui.toast(
                        now,
                        format!("Could not copy plot: {err}"),
                        crate::app::ToastKind::Error,
                    );
                }
            }
        }
        activity
    }
}

impl crate::app::RFMetricsApp {
    /// Metric plots in their own OS window (Python `show_plot` parity,
    /// all 7 tabs). Series are read live from `rows` every frame, so the
    /// viewport needs no update plumbing: curves appear on Done data and
    /// empty on Reset by themselves. Interaction stays on the stock
    /// `egui_plot` binds (drag pan, box-zoom select, ctrl+scroll zoom,
    /// double-click reset); the Python custom keybinds are out of scope.
    pub(crate) fn show_plots(&mut self, ctx: &egui::Context) {
        if !self.plots.open {
            return;
        }
        let id = egui::ViewportId::from_hash_of("metrics_plot");
        let builder = egui::ViewportBuilder::default()
            .with_title("Metrics")
            .with_inner_size([1100.0, 700.0]);
        ctx.show_viewport_immediate(id, builder, |vui, _class| {
            // Window-manager close withdraws (Python `withdraw` parity);
            // Plot reopens it.
            if vui.input(|i| i.viewport().close_requested()) {
                self.plots.open = false;
            }
            // While measuring, follow the live job's tab so its growing
            // curve is visible; idle windows stay user-driven.
            self.plots.tab = crate::plot::follow_live_tab(
                self.run.measuring,
                self.run.live_kind,
                self.plots.tab,
            );
            let kind = self.plots.tab;
            let def = crate::plot::plot_def(kind);
            // Finished series plus live `Running` buffers, so curves grow
            // mid-run (a cell is ever only one of the two — no dupes).
            // Streaming runs whether the window is open or not, so a
            // mid-run Plot click shows history; painting itself only
            // happens here, i.e. never unseen.
            let mut any_running = false;
            // Names + values feed fit/hover and the drawn lines (decimated
            // at draw time, x = 1-based frame — no full-res point cache).
            // First-column `include` doubles as plot visibility (#3):
            // unchecked rows are excluded from runs (start_run) and hidden
            // here, so fit/hover/export below re-fit to visible only.
            // Data keeps streaming in the background, so re-checking shows
            // history instantly, including mid-run Running curves.
            // `done` carries the row's permanent color slot alongside the
            // display name so lines/export use the stable per-file color
            // (hiding or removing one curve never recolors the rest).
            let done: Vec<(&str, usize, &[f64])> = self
                .queue.rows
                .iter()
                .filter(|r| r.include)
                .filter_map(|r| match r.cell(kind) {
                    crate::metrics::MetricCell::Done { values, .. }
                        if !values.is_empty() =>
                    {
                        Some((r.display.as_str(), r.color_idx, values.as_slice()))
                    }
                    crate::metrics::MetricCell::Running { values, .. }
                        if !values.is_empty() =>
                    {
                        any_running = true;
                        Some((r.display.as_str(), r.color_idx, values.as_slice()))
                    }
                    _ => None,
                })
                .collect();
            let borrowed: Vec<&[f64]> = done.iter().map(|(_, _, v)| *v).collect();
            let fresh = crate::plot::fit_limits(&borrowed, def.lo, def.hi);
            // Grow-only live bounds: axes expand with arriving points but
            // never jump inward mid-run; cleared once all settle so the
            // finished graph fits exactly again. `follow` arms the
            // one-shot auto-follow poke below (new live phase on this tab,
            // or a still-owed retry).
            let (follow, (xlim, (ymin, ymax))) = if any_running {
                let grown = match self.plots.live_fit {
                    Some((t, prev)) if t == kind => crate::plot::union_bounds(prev, fresh),
                    _ => fresh,
                };
                let follow = !matches!(self.plots.live_fit, Some((t, _)) if t == kind)
                    || self.plots.follow_pending;
                self.plots.live_fit = Some((kind, grown));
                (follow, grown)
            } else {
                self.plots.live_fit = None;
                // One-shot re-fit owed by a no-live-feed first Done (VMAF):
                // consumed like the follow retry above, so a stale arm can
                // never yank a later zoom.
                let follow = self.plots.follow_pending;
                self.plots.follow_pending = false;
                (follow, fresh)
            };
            // Empty plot (no Done data): axes only, y on the metric
            // default range; x falls back to a unit span.
            let (xmin, xmax) = xlim.unwrap_or((0.0, 1.0));
            // Per-tab plot id (zoom state persists per metric); hoisted so
            // the help-bar Reset below pokes the same memory entry the
            // canvas, snap clamp, and export snapshot use.
            let plot_id = format!("plot-{}", crate::plot::tab_title(kind).to_lowercase());
            // Help bar pinned to the bottom (Python `side="bottom"` parity).
            egui::Panel::bottom("plot_help").show(vui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(
                            "Drag: pan • Right-drag select: box zoom • Ctrl+scroll: zoom • Double-click: reset",
                        )
                        .selectable(false),
                    );
                    ui.checkbox(&mut self.plots.snap, "Snap to data").on_hover_text(
                        "Lock panning to the first/last frame and the plotted min/max; zoom and pan inside freely",
                    );
                    // Explicit re-fit (double-click parity): only arms the
                    // flag — the poke runs in the central panel below,
                    // where the plot id scope lives (a help-bar `Ui`
                    // derives different persistent ids than the canvas).
                    if ui
                        .add_enabled(
                            !borrowed.is_empty(),
                            egui::Button::new("Reset view"),
                        )
                        .on_hover_text("Fit the whole series (same as double-click)")
                        .clicked()
                    {
                        self.plots.reset_pending = true;
                    }
                    let save_label = if self.plots.png_saving { "Saving…" } else { "Save PNG" };
                    let save_hover = if self.plots.png_saving {
                        "Writing PNG in the background…"
                    } else {
                        "Save the current view as a PNG file (legend and axes included)"
                    };
                    let save_btn = ui
                        .add_enabled(!self.plots.png_saving, egui::Button::new(save_label))
                        .on_hover_text(save_hover);
                    if save_btn.clicked() && !self.plots.png_saving {
                        // Filename captured now; the export itself runs in
                        // the central panel below, where the plot id scope
                        // (for the current view bounds) lives.
                        self.plots.save_pending = Some(PlotExport::Save {
                            name: format!("{}.png", crate::plot::tab_title(self.plots.tab)),
                        });
                    }
                    let copy_label = if self.plots.png_saving { "Copying…" } else { "Copy" };
                    let copy_hover = if self.plots.png_saving {
                        "Rendering plot in the background…"
                    } else {
                        "Copy the current view as an image to the clipboard (legend and axes included)"
                    };
                    let copy_btn = ui
                        .add_enabled(!self.plots.png_saving, egui::Button::new(copy_label))
                        .on_hover_text(copy_hover);
                    if copy_btn.clicked() && !self.plots.png_saving {
                        self.plots.save_pending = Some(PlotExport::Copy);
                    }
                });
            });
            egui::CentralPanel::default().show(vui, |ui| {
                // FPS HUD (egui demo pattern): smoothed frame rate
                // top-right. Full repaint rate only while measuring (live
                // curves need it); idle repaints at ~10 Hz plus
                // input-driven ones — a static plot at 60 fps is pure
                // main+plot re-render cost, and immediate viewports
                // repaint the parent together with the child.
                if self.run.measuring {
                    ui.ctx().request_repaint();
                } else {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
                let fps = 1.0 / ui.input(|i| i.stable_dt);
                egui::Area::new(egui::Id::new("plot_fps"))
                    .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            // Single-line HUD: never wrap the counter.
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.label(format!("FPS: {fps:.0}"));
                        });
                    });
                // Tab strip (Python `CTkTabview` parity): all 7 tabs
                // always visible; empty tabs show empty axes. Compact
                // box hugging the buttons, centered via last frame's
                // measured width: egui cannot center content of unknown
                // width upfront (`with_layout`/`horizontal_centered`
                // reserve the full remaining rect and starve the plot),
                // but tab texts are static so one measured offset stays
                // pixel-exact. First frame falls back to the left edge.
                let pad = if self.plots.tabs_w <= 0.0 {
                    0.0
                } else {
                    ((ui.available_width() - self.plots.tabs_w) / 2.0).max(0.0)
                };
                ui.horizontal(|ui| {
                    if pad > 0.0 {
                        ui.add_space(pad);
                    }
                    let frame_resp = egui::Frame::group(ui.style()).show(ui, |ui| {
                        egui::Grid::new("plot_tabs").show(ui, |ui| {
                            for tab in MetricKind::ALL {
                                let title = crate::plot::tab_title(tab);
                                let btn =
                                    egui::Button::new(title).selected(self.plots.tab == tab);
                                if ui.add(btn).clicked() {
                                    self.plots.tab = tab;
                                }
                            }
                            ui.end_row();
                        });
                    });
                    self.plots.tabs_w = frame_resp.response.rect.width();
                });
                // Per-tab plot id shared with the help-bar Reset above.
                // One-shot live-follow: explicit default bounds seed fresh
                // PlotMemory with auto OFF, freezing the first-shown
                // (often still empty) view until a double-click. Flip auto
                // back on once per live phase so bounds track the growing
                // fit; user pan/zoom afterwards still takes over (it flips
                // auto off again). Retried while memory is missing: on the
                // opening frame there is nothing to poke yet, memory
                // appears on the next shown frame.
                if follow {
                    // NOTE: the id must be derived exactly like
                    // `Plot::show` does (`new` stores `Id::new(source)`,
                    // show hashes *that*); hashing the raw string hits a
                    // different memory entry and the poke never lands.
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        mem.auto_bounds = true.into();
                        mem.store(ui.ctx(), pid);
                        self.plots.follow_pending = false;
                    } else {
                        self.plots.follow_pending = true;
                    }
                }
                // Snap-to-data lock: clamp the stored view into the data
                // extent ([1, N] frames, fit min/max) before show, so the
                // user cannot pan past the first/last frame or leave the
                // plotted min/max — zooming and in-limits panning stay
                // free. Done on the stored bounds (not via
                // `set_plot_bounds`) so auto-follow keeps working.
                if self.plots.snap && !borrowed.is_empty() {
                    let n = borrowed.iter().map(|s| s.len()).max().unwrap_or(0);
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        let b = mem.bounds();
                        let (cx0, cx1) = if n >= 2 {
                            crate::plot::clamp_range((b.min()[0], b.max()[0]), (1.0, n as f64))
                        } else {
                            (b.min()[0], b.max()[0])
                        };
                        let (cy0, cy1) = crate::plot::clamp_range(
                            (b.min()[1], b.max()[1]),
                            (ymin, ymax),
                        );
                        mem.set_bounds(egui_plot::PlotBounds::from_min_max(
                            [cx0, cy0],
                            [cx1, cy1],
                        ));
                        mem.store(ui.ctx(), pid);
                    }
                }
                // Reset-view button (help bar arms the flag — this `Ui`
                // owns the plot id scope): write the computed fit into
                // this tab's stored bounds and take over from auto-follow
                // (a user takeover, like pan/zoom). After snap so the
                // exact fit wins over the clamp. Retried while memory is
                // missing, like the follow poke.
                if self.plots.reset_pending && !borrowed.is_empty() {
                    let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                    if let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), pid) {
                        mem.auto_bounds = false.into();
                        mem.set_bounds(egui_plot::PlotBounds::from_min_max(
                            [xmin, ymin],
                            [xmax, ymax],
                        ));
                        mem.store(ui.ctx(), pid);
                        self.plots.reset_pending = false;
                    }
                }
                // Draw budget: ~2 points per horizontal pixel (the y-axis
                // gutter makes this a slight over-estimate, harmless).
                let target = (ui.available_width() as usize * 2).clamp(512, 8192);
                // Pending plot export (Save PNG / Copy button): snapshot
                // the CURRENT view (stored bounds when strict, else the
                // fit) plus owned series data, then render on a one-shot
                // worker thread. Crosshair/tooltip never enter: this is a
                // fresh render, not a screenshot. Direct field writes below
                // (not `self.ui.toast()`): `done` still borrows rows here.
                if let Some(job) = self.plots.save_pending.take() {
                    // Re-entrant click while an export is in flight: drop
                    // it (both buttons are disabled, so this is a guard).
                    if !self.plots.png_saving {
                        let pid = ui.make_persistent_id(egui::Id::new(plot_id.clone()));
                        let ((vx0, vx1), (vy0, vy1)) =
                            match egui_plot::PlotMemory::load(ui.ctx(), pid) {
                                Some(mem) => {
                                    let b = mem.bounds();
                                    let (a0, a1) = (b.min()[0], b.max()[0]);
                                    let (c0, c1) = (b.min()[1], b.max()[1]);
                                    (
                                        if a1 > a0 { (a0, a1) } else { (xmin, xmax) },
                                        if c1 > c0 { (c0, c1) } else { (ymin, ymax) },
                                    )
                                }
                                None => ((xmin, xmax), (ymin, ymax)),
                            };
                        // Owned snapshot: the worker outlives this frame and
                        // cannot borrow `done`/`self.queue.rows`.
                        let owned: Vec<(String, usize, Vec<f64>)> = done
                            .iter()
                            .map(|(n, s, v)| ((*n).to_owned(), *s, (*v).to_vec()))
                            .collect();
                        let title = crate::plot::tab_title(kind).to_owned();
                        let y_label = def.label.to_owned();
                        let view = ((vx0, vx1), (vy0, vy1));
                        // Size preset snapshot: a mid-render combobox change
                        // only affects the next export.
                        let size = self.config.view.plot_size.dims();
                        match job {
                            PlotExport::Save { name } => {
                                // Picker cancelled: silent no-op. Runs on
                                // the UI thread (native modal); only the
                                // render moves off.
                                if let Some(mut path) = rfd::FileDialog::new()
                                    .set_title("Save plot as PNG")
                                    .set_file_name(&name)
                                    .add_filter("PNG image", &["png"])
                                    .save_file()
                                {
                                    path.set_extension("png");
                                    self.plots.png_saving = true;
                                    let tx = self.plots.png_tx.clone();
                                    let ctx = ui.ctx().clone();
                                    std::thread::spawn(move || {
                                        let series: Vec<(&str, usize, &[f64])> = owned
                                            .iter()
                                            .map(|(n, s, v)| (n.as_str(), *s, v.as_slice()))
                                            .collect();
                                        let msg = match crate::plot::export_png(
                                            &path,
                                            &title,
                                            &y_label,
                                            &series,
                                            view,
                                            size,
                                        ) {
                                            Ok(()) => {
                                                log::info!(target: "rfmetrics::plot", "plot saved to {}", path.display());
                                                PngSaveMsg::Saved { path }
                                            }
                                            Err(e) => {
                                                log::warn!(target: "rfmetrics::plot", "plot save failed: {e}");
                                                PngSaveMsg::SaveFailed { err: e }
                                            }
                                        };
                                        let _ = tx.send(msg);
                                        ctx.request_repaint();
                                    });
                                }
                            }
                            PlotExport::Copy => {
                                self.plots.png_saving = true;
                                let tx = self.plots.png_tx.clone();
                                let ctx = ui.ctx().clone();
                                std::thread::spawn(move || {
                                    let series: Vec<(&str, usize, &[f64])> = owned
                                        .iter()
                                        .map(|(n, s, v)| (n.as_str(), *s, v.as_slice()))
                                        .collect();
                                    let msg = match crate::plot::render_rgba(
                                        &title, &y_label, &series, view, size,
                                    ) {
                                        Ok((w, h, rgba)) => {
                                            log::info!(target: "rfmetrics::plot", "plot rendered for clipboard ({w}x{h})");
                                            PngSaveMsg::CopyReady { w, h, rgba }
                                        }
                                        Err(e) => {
                                            log::warn!(target: "rfmetrics::plot", "plot copy failed: {e}");
                                            PngSaveMsg::CopyFailed { err: e }
                                        }
                                    };
                                    let _ = tx.send(msg);
                                    ctx.request_repaint();
                                });
                            }
                        }
                    }
                }
                let plot_resp = egui_plot::Plot::new(plot_id)
                    .x_axis_label("Frames")
                    .y_axis_label(def.label)
                    .legend(
                        egui_plot::Legend::default()
                            .position(egui_plot::Corner::RightBottom),
                    )
                    .default_x_bounds(xmin, xmax)
                    .default_y_bounds(ymin, ymax)
                    .show(ui, |plot_ui| {
                        // Lines decimate from values to ~2 px buckets (fit +
                        // hover still use full resolution).
                        // Stable per-file colors: keyed by permanent queue
                        // slot, so hiding/removing one curve never recolors
                        // the rest.
                        for (name, slot, values) in &done {
                            let thin = crate::plot::decimate_minmax(values, target);
                            plot_ui.line(
                                egui_plot::Line::new(*name, thin)
                                    .color(crate::plot::series_egui_color(*slot)),
                            );
                        }
                        // Hover inspect (Python `_on_hover` parity):
                        // nearest data point within 30 screen px gets a
                        // crosshair; the `{name}\nFrame=N, Metric=V.4f`
                        // text returns to the caller, which draws it as a
                        // native tooltip (plot-canvas text is tiny and has
                        // no background). Suppressed while
                        // panning/zooming, like the ref.
                        let hovering = plot_ui.response().hovered()
                            && !plot_ui.response().dragged();
                        let hover_pos = plot_ui.response().hover_pos();
                        let ptr = plot_ui.pointer_coordinate();
                        if let (true, Some(mouse), Some(p)) = (hovering, hover_pos, ptr) {
                            let to_screen = |fx: f64, fy: f64| {
                                let sp = plot_ui.screen_from_plot(
                                    egui_plot::PlotPoint::new(fx, fy),
                                );
                                (sp.x, sp.y)
                            };
                            if let Some((si, frame, value)) = crate::plot::nearest_hover(
                                &borrowed,
                                p.x,
                                to_screen,
                                (mouse.x, mouse.y),
                            ) {
                                let fx = frame as f64;
                                plot_ui.vline(egui_plot::VLine::new("", fx));
                                plot_ui.hline(egui_plot::HLine::new("", value));
                                return Some(format!(
                                    "{}\nFrame={frame}, Metric={value:.4}",
                                    done[si].0
                                ));
                            }
                        }
                        None
                    });
                // Native tooltip at the pointer: readable body text on a
                // theme background (Python yellow annotation-box parity).
                if let Some(text) = plot_resp.inner {
                    egui::Tooltip::for_widget(&plot_resp.response)
                        .at_pointer()
                        .gap(12.0)
                        .show(|ui| {
                            ui.label(text);
                        });
                }
                // Mirror the main-window toast here (PNG saver results land
                // while this OS window has focus; the main toast behind it
                // is invisible). Expiry is owned by the main viewport.
                if let Some(toast) = self.ui.toast.clone()
                    && ui.input(|i| i.time) < toast.until
                {
                    let corner = ui.max_rect().right_bottom();
                    let mut frame = egui::Frame::popup(ui.style());
                    if let Some(outline) = toast.kind.outline() {
                        frame = frame.stroke(egui::Stroke::new(1.5, outline));
                    }
                    egui::Area::new(egui::Id::new("plot_toast"))
                        .order(egui::Order::Foreground)
                        .fixed_pos(corner + egui::vec2(-10.0, -10.0))
                        .pivot(egui::Align2::RIGHT_BOTTOM)
                        .show(ui.ctx(), |ui| {
                            frame.show(ui, |ui| {
                                ui.label(&toast.text);
                            });
                        });
                }
            });
        });
    }
}
