#![windows_subsystem = "windows"]

use eframe::egui;
use inferno_app::InfernoApp;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1300.0, 900.0])
            .with_min_inner_size([1000.0, 700.0])
            .with_title("Inferno Analyzer"),
        ..Default::default()
    };
    eframe::run_native(
        "Inferno Analyzer",
        options,
        Box::new(|cc| Ok(Box::new(InfernoApp::new(cc)))),
    )
}
