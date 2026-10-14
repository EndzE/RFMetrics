#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

mod app;
mod binaries;
mod cmd;
mod logging;
mod metrics;
mod plot;
mod preview;
mod probe;
mod state;

fn main() -> eframe::Result<()> {
    logging::init_logging();
    // Crash orphans (never wiped, never adopted thanks to the random
    // suffix); 24 h grace protects a live second instance.
    metrics::badframes::sweep_stale_tmp();
    // Default to DirectX 12: wgpu's auto-picked Vulkan backend stalls
    // surface presents across window resizes (persistent black box on
    // the grown area, quiet CPU/GPU, nothing in the log). `WGPU_BACKEND`
    // still overrides (run-vulkan.bat uses it for backend testing).
    let mut wgpu_setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.instance_descriptor.backends = eframe::egui_wgpu::wgpu::Backends::from_env()
        .unwrap_or(eframe::egui_wgpu::wgpu::Backends::DX12);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1300.0, 600.0])
            // Windows file-drop is opt-in; without this no hover/drop events arrive
            .with_drag_and_drop(true),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: wgpu_setup.into(),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "RFMetrics",
        options,
        Box::new(|_cc| Ok(Box::new(app::RFMetricsApp::default()))),
    )
}
