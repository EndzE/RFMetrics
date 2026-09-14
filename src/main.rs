mod app;
mod binaries;
mod probe;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1300.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "RFMetrics",
        options,
        Box::new(|_cc| Ok(Box::new(app::RFMetricsApp::default()))),
    )
}
