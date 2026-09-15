mod app;
mod binaries;
mod logging;
mod preview;
mod probe;

fn main() -> eframe::Result<()> {
    logging::init_logging();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1300.0, 600.0])
            // Windows file-drop is opt-in; without this no hover/drop events arrive
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "RFMetrics",
        options,
        Box::new(|_cc| Ok(Box::new(app::RFMetricsApp::default()))),
    )
}
