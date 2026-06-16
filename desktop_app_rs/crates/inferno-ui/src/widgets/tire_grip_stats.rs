use egui::RichText;
use inferno_core::analysis::tire_grip::{MetricMode, TireGripResult};

use crate::theme;

/// Popup window displaying tire grip statistics.
#[derive(Default)]
pub struct TireGripStatsWindow {
    pub open: bool,
}

impl TireGripStatsWindow {
    pub fn show(&mut self, ctx: &egui::Context, result: &TireGripResult) {
        egui::Window::new("Tire Grip Statistics")
            .open(&mut self.open)
            .default_width(600.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    summary_section(ui, result);
                    ui.add_space(12.0);
                    interpretation_section(ui);
                });
            });
    }
}

fn summary_section(ui: &mut egui::Ui, result: &TireGripResult) {
    let metric_name = match result.metric_mode {
        MetricMode::Pressure => "Pressure",
        MetricMode::Temperature => "Temperature",
    };
    let mu = if result.metric_unit.is_empty() {
        String::new()
    } else {
        format!(" ({})", result.metric_unit)
    };
    let au = if result.accel_unit.is_empty() {
        String::new()
    } else {
        format!(" ({})", result.accel_unit)
    };

    ui.heading("Summary Statistics");
    ui.separator();
    egui::Grid::new("tire_grip_summary_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Wheel").strong());
            ui.label(RichText::new(format!("Mean G{au}")).strong());
            ui.label(RichText::new(format!("Std G{au}")).strong());
            ui.label(RichText::new(format!("Mean {metric_name}{mu}")).strong());
            ui.label(RichText::new(format!("Std {metric_name}{mu}")).strong());
            ui.label(RichText::new("Buckets").strong());
            ui.end_row();

            for w in &result.wheels {
                ui.label(&w.name);
                ui.label(format!("{:.3}", w.mean_g));
                ui.label(format!("{:.3}", w.std_g));
                ui.label(format!("{:.2}", w.mean_metric));
                ui.label(format!("{:.2}", w.std_metric));
                ui.label(format!("{}", w.bucket_centers.len()));
                ui.end_row();
            }
        });

    // Cross-wheel comparison
    ui.add_space(8.0);
    let [ref fl, ref fr, ref rl, ref rr] = result.wheels;

    let front_mean = (fl.mean_g + fr.mean_g) / 2.0;
    let rear_mean = (rl.mean_g + rr.mean_g) / 2.0;
    let left_mean = (fl.mean_g + rl.mean_g) / 2.0;
    let right_mean = (fr.mean_g + rr.mean_g) / 2.0;

    if (front_mean - rear_mean).abs() > 0.01 {
        ui.label(
            RichText::new(format!(
                "Front avg G: {front_mean:.3} vs Rear: {rear_mean:.3} (Δ {:.3})",
                front_mean - rear_mean
            ))
            .color(theme::TEXT_SECONDARY),
        );
    }
    if (left_mean - right_mean).abs() > 0.01 {
        ui.label(
            RichText::new(format!(
                "Left avg G: {left_mean:.3} vs Right: {right_mean:.3} (Δ {:.3})",
                left_mean - right_mean
            ))
            .color(theme::TEXT_SECONDARY),
        );
    }
}

fn interpretation_section(ui: &mut egui::Ui) {
    ui.heading("Interpretation Guide");
    ui.separator();

    ui.label(RichText::new("Total Acceleration").strong());
    ui.label("Combined lateral and longitudinal G-force: sqrt(lat² + inline²). Represents total grip demand on tires.");
    ui.add_space(4.0);

    ui.label(RichText::new("Grip Envelope").strong());
    ui.label("The P99.9 line shows the near-maximum G achievable at each pressure/temperature. A peak in the curve indicates the optimal tire operating window.");
    ui.add_space(4.0);

    ui.label(RichText::new("Pressure vs Temperature").strong());
    ui.label("Pressure mode shows how inflation affects grip. Temperature mode shows the thermal operating window. Both help identify optimal tire setup.");
}
