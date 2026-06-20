use eframe::egui;

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

struct InfernoApp;

impl InfernoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut visuals = egui::Visuals::dark();

        // Dark theme: #1a1a2e background
        let bg = egui::Color32::from_rgb(0x1a, 0x1a, 0x2e);
        visuals.panel_fill = bg;
        visuals.window_fill = bg;
        visuals.extreme_bg_color = egui::Color32::from_rgb(0x12, 0x12, 0x22);
        visuals.faint_bg_color = egui::Color32::from_rgb(0x22, 0x22, 0x3a);

        cc.egui_ctx.set_visuals(visuals);

        Self
    }
}

impl eframe::App for InfernoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.heading("Inferno Analyzer");
            });
        });
    }
}
