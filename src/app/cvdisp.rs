//! CVVDP custom display presets: editor dialog session state, preset
//! list ops, and the dialog view (VideoMetricsLab preset-management
//! parity, global scope: one display applies to all runs, so there is no
//! per-row assignment, no apply-without-saving, and no "Custom" state —
//! every flow ends in a named preset).

use crate::metrics::ffvship::CustomDisplay;

/// Editor window size: wide enough for the four-button bar (~470px of
/// buttons), which is wider than the form; the bar sets the width.
/// Height stays snug (a taller window leaves a black void) — the window
/// is user-resizable either way.
const EDITOR_SIZE: [f32; 2] = [480.0, 315.0];

/// Draft display values as edited (floats throughout, like the boxes;
/// resolution rounds to ints and reflectivity edits as percent at save).
#[derive(Debug, Clone, Default)]
pub(crate) struct DisplayDraft {
    pub name: String,
    pub width: f64,
    pub height: f64,
    pub diagonal: f64,
    pub distance: f64,
    pub peak: f64,
    pub contrast: f64,
    pub ambient: f64,
    pub reflect_pct: f64,
    pub exposure: f64,
    pub colorspace: String,
}

type DisplayMap = serde_json::Map<String, serde_json::Value>;

impl DisplayDraft {
    /// Draft from a stored display object (editor open).
    pub(crate) fn from_map(
        name: &str,
        display: &serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        let num = |f: &str| display.get(f).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let res = display.get("resolution").and_then(|r| r.as_array());
        Self {
            name: name.to_owned(),
            width: res
                .and_then(|r| r.first())
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            height: res
                .and_then(|r| r.get(1))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            diagonal: num("diagonal_size_inches"),
            distance: num("viewing_distance_meters"),
            peak: num("max_luminance"),
            contrast: num("contrast"),
            ambient: num("E_ambient"),
            reflect_pct: num("k_refl") * 100.0,
            exposure: num("exposure"),
            colorspace: display
                .get("colorspace")
                .and_then(|c| c.as_str())
                .unwrap_or("SDR")
                .to_owned(),
        }
    }

    /// Canonical display object for saving (resolution rounded to ints,
    /// reflectivity back to a fraction). Whole floats serialize as ints
    /// so a draft round-trips its source object exactly.
    pub(crate) fn into_map(self) -> DisplayMap {
        let int = |v: f64| serde_json::Value::from(v.round() as i64);
        let num = |v: f64| {
            if v.fract() == 0.0 && v.abs() < 9.0e15 {
                serde_json::Value::from(v as i64)
            } else {
                serde_json::Number::from_f64(v)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::from(0))
            }
        };
        let mut m = DisplayMap::new();
        m.insert(
            "name".to_owned(),
            serde_json::Value::from(self.name.clone()),
        );
        m.insert(
            "colorspace".to_owned(),
            serde_json::Value::from(self.colorspace),
        );
        m.insert(
            "resolution".to_owned(),
            serde_json::Value::Array(vec![int(self.width), int(self.height)]),
        );
        m.insert("viewing_distance_meters".to_owned(), num(self.distance));
        m.insert("diagonal_size_inches".to_owned(), num(self.diagonal));
        m.insert("max_luminance".to_owned(), num(self.peak));
        m.insert("contrast".to_owned(), num(self.contrast));
        m.insert("E_ambient".to_owned(), num(self.ambient));
        m.insert("k_refl".to_owned(), num(self.reflect_pct / 100.0));
        m.insert("exposure".to_owned(), num(self.exposure));
        m.insert(
            "source".to_owned(),
            serde_json::Value::from("rfmetrics-custom"),
        );
        m
    }
}

/// Keep the original raw value for a field the user did not touch
/// (VideoMetricsLab parity: boxes show fewer decimals than a display can
/// hold, and 0.7472 read back as 0.747 would silently change the display
/// identity). Both values in box units; `decimals` is the box precision.
fn preserve(
    dst: &mut DisplayMap,
    key: &str,
    box_val: f64,
    orig_box_units: f64,
    orig_map_val: f64,
    decimals: u32,
) {
    let shown =
        (orig_box_units * 10f64.powi(decimals as i32)).round() / 10f64.powi(decimals as i32);
    if (box_val - shown).abs() < 10f64.powi(-(decimals as i32) - 2)
        && let Some(n) = serde_json::Number::from_f64(orig_map_val)
    {
        dst.insert(key.to_owned(), serde_json::Value::Number(n));
    }
}

/// Save a validated draft (VideoMetricsLab `with_user_preset` parity):
/// replace the `replacing` preset if given (rename when the name
/// changed), else append. Returns the saved (trimmed) name.
pub(crate) fn save_custom(
    list: &mut Vec<CustomDisplay>,
    name: &str,
    display: DisplayMap,
    replacing: Option<&str>,
) -> Result<String, String> {
    crate::metrics::ffvship::validate_custom_display(name, &display, replacing, list)?;
    if let Some(own) = replacing {
        list.retain(|c| c.name != own);
    }
    let name = name.trim().to_owned();
    list.push(CustomDisplay {
        name: name.clone(),
        display,
    });
    Ok(name)
}

/// Delete a custom preset by name; `true` when one was there.
pub(crate) fn delete_custom(list: &mut Vec<CustomDisplay>, name: &str) -> bool {
    let n = list.len();
    list.retain(|c| c.name != name);
    list.len() != n
}

/// Editor dialog session state (like the plot runtime: session-only,
// never persisted — the presets themselves persist via the state file).
#[derive(Debug, Default)]
pub(crate) struct CvdispEditor {
    pub open: bool,
    pub confirm_delete: bool,
    adding: bool,
    /// Center over the main window on the next shown frame (one-shot, so
    /// the user can move it freely afterwards).
    center_once: bool,
    own: Option<String>,
    draft: DisplayDraft,
    original: DisplayMap,
    warning: String,
}

impl CvdispEditor {
    /// "Add preset…" from the given display: name empty, save-as-new only.
    pub(crate) fn open_add(&mut self, display: &DisplayMap) {
        self.open_editor(display, "", None, true);
    }

    /// "Edit preset…" from the given display: `own` is the custom preset
    /// it came from, if any (built-ins offer save-as-new only).
    pub(crate) fn open_edit(&mut self, display: &DisplayMap, own: Option<String>) {
        let name = own.clone().unwrap_or_default();
        self.open_editor(display, &name, own, false);
    }

    fn open_editor(&mut self, display: &DisplayMap, name: &str, own: Option<String>, adding: bool) {
        self.draft = DisplayDraft::from_map(name, display);
        self.original = display.clone();
        self.own = own;
        self.adding = adding;
        self.warning.clear();
        self.open = true;
        self.center_once = true;
    }

    /// Canonical map for the current draft, preserving untouched
    /// originals (see `preserve`).
    fn built_map(&self) -> DisplayMap {
        let d = &self.draft;
        let mut map = d.clone().into_map();
        let orig = |f: &str| {
            self.original
                .get(f)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::NAN)
        };
        preserve(
            &mut map,
            "diagonal_size_inches",
            d.diagonal,
            orig("diagonal_size_inches"),
            orig("diagonal_size_inches"),
            1,
        );
        preserve(
            &mut map,
            "viewing_distance_meters",
            d.distance,
            orig("viewing_distance_meters"),
            orig("viewing_distance_meters"),
            4,
        );
        preserve(
            &mut map,
            "max_luminance",
            d.peak,
            orig("max_luminance"),
            orig("max_luminance"),
            0,
        );
        preserve(
            &mut map,
            "contrast",
            d.contrast,
            orig("contrast"),
            orig("contrast"),
            0,
        );
        preserve(
            &mut map,
            "E_ambient",
            d.ambient,
            orig("E_ambient"),
            orig("E_ambient"),
            1,
        );
        preserve(
            &mut map,
            "exposure",
            d.exposure,
            orig("exposure"),
            orig("exposure"),
            2,
        );
        // Reflectivity edits as percent; the map holds a fraction.
        preserve(
            &mut map,
            "k_refl",
            d.reflect_pct,
            orig("k_refl") * 100.0,
            orig("k_refl"),
            2,
        );
        map
    }
}

impl crate::app::RFMetricsApp {
    /// Add/Edit/Delete preset buttons for the CVVDP options box (call
    /// inside the box scope; inherits its enablement).
    pub(crate) fn cvdisp_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_sized([70.0, 18.0], egui::Label::new(""));
            if ui.button("Add preset…").clicked() {
                let map = self.current_display_map();
                self.cvdisp.open_add(&map);
            }
            if ui.button("Edit preset…").clicked() {
                let display = self.config.cvvdp.display.clone();
                let own = self
                    .config
                    .cvvdp
                    .custom
                    .iter()
                    .any(|c| c.name == display)
                    .then(|| display.clone());
                let map = self.current_display_map();
                self.cvdisp.open_edit(&map, own);
            }
            let can_delete = self
                .config
                .cvvdp
                .custom
                .iter()
                .any(|c| c.name == self.config.cvvdp.display);
            if ui
                .add_enabled(can_delete, egui::Button::new("Delete preset"))
                .on_hover_text("Delete this preset of yours. Built-in presets cannot be deleted.")
                .clicked()
            {
                self.cvdisp.confirm_delete = true;
            }
        });
    }

    /// Current effective display object: temp values while an
    /// Apply-without-saving is active (validated selection ⇒ lookup hits
    /// otherwise; the binary default map is the unreachable fallback).
    fn current_display_map(&self) -> DisplayMap {
        if let Some(t) = &self.config.cvvdp.temp {
            return t.clone();
        }
        crate::metrics::ffvship::lookup_display(
            &self.config.cvvdp.display,
            &self.config.cvvdp.custom,
        )
        .map(|(_, d)| d.clone())
        .unwrap_or_default()
    }

    /// Delete-preset confirm on the main window (modals don't cross
    /// into the editor viewport). Call unconditionally; no-op unless
    /// armed.
    pub(crate) fn cvdisp_delete_modal(&mut self, ui: &mut egui::Ui) {
        if !self.cvdisp.confirm_delete {
            return;
        }
        let name = self.config.cvvdp.display.clone();
        let r = egui::Modal::new(egui::Id::new("cvvdp_delete")).show(ui.ctx(), |ui| {
            ui.label(format!(
                "Delete your preset \"{name}\"?\n\nThe display resets to standard_fhd."
            ));
            ui.horizontal(|ui| {
                let yes = ui.button("Delete").clicked();
                let no = ui.button("Cancel").clicked();
                (yes, no)
            })
            .inner
        });
        if r.inner.0 {
            delete_custom(&mut self.config.cvvdp.custom, &name);
            self.config.cvvdp.display = crate::metrics::ffvship::DEFAULT_DISPLAY_KEY.to_owned();
            let now = ui.input(|i| i.time);
            self.ui.toast(
                now,
                format!("Deleted the CVVDP preset \"{name}\"."),
                crate::app::ToastKind::Info,
            );
            self.cvdisp.confirm_delete = false;
        } else if r.inner.1 || r.should_close() {
            self.cvdisp.confirm_delete = false;
        }
    }

    /// Preset editor in its own OS window (plot-window pattern: native
    /// drag + close; the frameless inline window couldn't be dragged).
    /// Call every frame; no-op unless open.
    pub(crate) fn show_cvdisp_editor(&mut self, ctx: &egui::Context) {
        if !self.cvdisp.open {
            return;
        }
        let title = if self.cvdisp.adding {
            "New preset"
        } else {
            "Display"
        };
        let id = egui::ViewportId::from_hash_of("cvvdp_editor");
        let mut builder = egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size(EDITOR_SIZE);
        // Center over the main window once on open — never every frame,
        // or the user could not move it. A missing rect retries.
        if self.cvdisp.center_once
            && let Some(r) = ctx.input(|i| i.viewport().outer_rect)
        {
            let c = r.center();
            builder = builder.with_position(egui::pos2(
                c.x - EDITOR_SIZE[0] / 2.0,
                c.y - EDITOR_SIZE[1] / 2.0,
            ));
            self.cvdisp.center_once = false;
        }
        let mut open = true;
        let mut close = false;
        ctx.show_viewport_immediate(id, builder, |vui, _class| {
            // Window-manager close withdraws; the buttons below use the
            // `close` flag instead (same writeback).
            if vui.input(|i| i.viewport().close_requested()) {
                open = false;
                return;
            }
            // Control buttons docked to the bottom (dialog button bar);
            // the form keeps the central area.
            egui::Panel::bottom("cvdisp_buttons").show(vui, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let now = ui.input(|i| i.time);
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    if ui.button("Save as new preset").clicked() {
                        self.cvdisp_save(now, true, &mut close);
                    }
                    if !self.cvdisp.adding && ui.button("Apply without saving").clicked() {
                        self.cvdisp_apply(&mut close);
                    }
                    let can_save = self.cvdisp.own.is_some() && !self.cvdisp.adding;
                    if can_save && ui.button("Save preset").clicked() {
                        self.cvdisp_save(now, false, &mut close);
                    }
                });
            });
            // CentralPanel: main-window background + margins (a bare
            // viewport Ui starts at the raw origin on flat black).
            egui::CentralPanel::default().show(vui, |ui| {
                self.cvdisp_form(ui);
            });
        });
        self.cvdisp.open = open && !close;
    }

    /// The editor form (VideoMetricsLab `CvvdpDisplayDialog` parity:
    /// fields, ranges, units, and the live screen-height note). Control
    /// buttons live in the bottom bar below, not here.
    fn cvdisp_form(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("cvdisp_form")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Preset name:");
                let name_tip = if self.cvdisp.own.is_some() {
                    "Change it and click Save preset to rename this preset."
                } else {
                    "Type a name and click Save as new preset to keep this display as a preset."
                };
                crate::app::widgets::hover_wrap(
                    ui.text_edit_singleline(&mut self.cvdisp.draft.name),
                    name_tip,
                );
                ui.end_row();
                ui.label("Resolution:");
                ui.horizontal(|ui| {
                    crate::app::widgets::hover_wrap(
                        ui.add(
                            egui::DragValue::new(&mut self.cvdisp.draft.width)
                                .range(16.0..=8192.0)
                                .speed(1.0)
                                .max_decimals(0),
                        ),
                        "The display's own resolution in pixels (not the video's).",
                    );
                    ui.label("x");
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.height)
                            .range(16.0..=8192.0)
                            .speed(1.0)
                            .max_decimals(0),
                    );
                });
                ui.end_row();
                ui.label("Screen size:");
                ui.add(
                    egui::DragValue::new(&mut self.cvdisp.draft.diagonal)
                        .range(1.0..=1000.0)
                        .speed(0.1)
                        .max_decimals(1)
                        .suffix(" in"),
                )
                .on_hover_text("The screen's diagonal size.");
                ui.end_row();
                ui.label("Viewing distance:");
                ui.horizontal(|ui| {
                    crate::app::widgets::hover_wrap(
                        ui.add(
                            egui::DragValue::new(&mut self.cvdisp.draft.distance)
                                .range(0.05..=50.0)
                                .speed(0.05)
                                .max_decimals(4)
                                .suffix(" m"),
                        ),
                        "How far the viewer's eyes are from the screen. Closer makes \
                     small artifacts easier to see.",
                    );
                    let r = crate::metrics::ffvship::heights_ratio(
                        self.cvdisp.draft.width,
                        self.cvdisp.draft.height,
                        self.cvdisp.draft.diagonal,
                        self.cvdisp.draft.distance,
                    );
                    ui.label(if r.is_finite() {
                        format!("= {r:.2} x screen height")
                    } else {
                        "—".to_owned()
                    });
                });
                ui.end_row();
                ui.label("Peak brightness:");
                crate::app::widgets::hover_wrap(
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.peak)
                            .range(1.0..=10000.0)
                            .speed(10.0)
                            .max_decimals(0)
                            .suffix(" nits"),
                    ),
                    "The display's peak brightness: about 200 for an office monitor, \
                 600-1500 for an HDR monitor, 1000-4000 for an HDR TV.",
                );
                ui.end_row();
                ui.label("Contrast:");
                crate::app::widgets::hover_wrap(
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.contrast)
                            .range(1.0..=10_000_000.0)
                            .speed(100.0)
                            .max_decimals(0)
                            .suffix(" : 1"),
                    ),
                    "Peak to black: about 1000:1 for a typical LCD, 1,000,000:1 \
                 for OLED or the official HDR displays.",
                );
                ui.end_row();
                ui.label("Room light:");
                crate::app::widgets::hover_wrap(
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.ambient)
                            .range(0.0..=100_000.0)
                            .speed(10.0)
                            .max_decimals(1)
                            .suffix(" lux"),
                    ),
                    "Light falling on the screen: about 250 lux in an office, 5-10 \
                 watching a film with the lights low, 0 in the dark.",
                );
                ui.end_row();
                ui.label("Screen reflectivity:");
                crate::app::widgets::hover_wrap(
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.reflect_pct)
                            .range(0.0..=99.9)
                            .speed(0.1)
                            .max_decimals(2)
                            .suffix(" %"),
                    ),
                    "How much of the room light the screen reflects back at the \
                 viewer; 0.5% is the official models' value.",
                );
                ui.end_row();
                ui.label("Exposure:");
                crate::app::widgets::hover_wrap(
                    ui.add(
                        egui::DragValue::new(&mut self.cvdisp.draft.exposure)
                            .range(0.01..=100.0)
                            .speed(0.1)
                            .max_decimals(2),
                    ),
                    "Brightness multiplier for the pictures; 1 shows them as encoded.",
                );
                ui.end_row();
                ui.label("Colorspace:");
                crate::app::widgets::hover_wrap(
                    egui::ComboBox::from_id_salt("cvdisp_colorspace")
                        .selected_text(self.cvdisp.draft.colorspace.as_str())
                        .show_ui(ui, |ui| {
                            for c in crate::metrics::ffvship::DISPLAY_COLORSPACES {
                                crate::app::widgets::hover_wrap(
                                    ui.selectable_value(
                                        &mut self.cvdisp.draft.colorspace,
                                        (*c).to_owned(),
                                        *c,
                                    ),
                                    colorspace_tip(c),
                                );
                            }
                        })
                        .response,
                    "Transfer function of the content: SDR for regular videos, \
                     the HDR one matching the footage, linear for absolute \
                     linear-light frames.",
                );
                ui.end_row();
            });
        if !self.cvdisp.warning.is_empty() {
            ui.colored_label(ui.visuals().error_fg_color, &self.cvdisp.warning);
        }
    }

    /// Run one save flow: build + preserve + validate, then store,
    /// always select the saved preset, toast, and close. `as_new`:
    /// Save-as-new button (else the Save-preset button).
    fn cvdisp_save(&mut self, now: f64, as_new: bool, close: &mut bool) {
        let replacing = if as_new {
            None
        } else {
            self.cvdisp.own.clone()
        };
        let name = self.cvdisp.draft.name.clone();
        let map = self.cvdisp.built_map();
        match save_custom(
            &mut self.config.cvvdp.custom,
            &name,
            map,
            replacing.as_deref(),
        ) {
            Ok(saved) => {
                let renamed = replacing.as_deref().is_some_and(|o| o != saved);
                self.config.cvvdp.select(saved.clone());
                self.ui.toast(
                    now,
                    if renamed {
                        format!(
                            "Renamed your CVVDP preset \"{}\" to \"{saved}\" and saved it.",
                            replacing.as_deref().unwrap_or_default()
                        )
                    } else if as_new {
                        format!("Saved the CVVDP preset \"{saved}\".")
                    } else {
                        format!("Saved your CVVDP preset \"{saved}\".")
                    },
                    crate::app::ToastKind::Info,
                );
                *close = true;
            }
            Err(e) => self.cvdisp.warning = e,
        }
    }

    /// Apply-without-saving (Edit mode only): validated draft values take
    /// effect at once without creating a preset. Values matching a known
    /// preset select it instead of minting temp state; anything else lands
    /// in the session-only temp slot (never persisted).
    fn cvdisp_apply(&mut self, close: &mut bool) {
        let map = self.cvdisp.built_map();
        if let Err(e) = crate::metrics::ffvship::validate_display_values(&map) {
            self.cvdisp.warning = e;
            return;
        }
        if let Some(key) = crate::metrics::ffvship::match_display(&map, &self.config.cvvdp.custom) {
            self.config.cvvdp.select(key);
        } else {
            self.config.cvvdp.temp = Some(map);
        }
        *close = true;
    }
}

/// What each colorspace choice means, in HDR/SDR terms (pycvvdp guidance:
/// the transfer must match the content).
fn colorspace_tip(c: &str) -> &'static str {
    match c {
        "BT.2020-PQ" => "HDR10 content with the PQ (ST 2084) transfer — HDR monitors and TVs.",
        "BT.2020-HLG" => {
            "HLG broadcast content (cameras, phones) — same display, different transfer."
        }
        "BT.709-linear" => "Absolute linear-light frames (OpenEXR); values are scene nits.",
        _ => "Standard content: office monitors, phones, SDR TVs.",
    }
}

#[cfg(test)]
#[path = "../tests/test_cvdisp.rs"]
mod tests;
