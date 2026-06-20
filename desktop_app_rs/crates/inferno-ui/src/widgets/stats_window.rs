use egui::RichText;
use inferno_core::analysis::driver_consistency::DriverConsistencyResult;

use crate::theme;

/// Popup window displaying driver consistency statistics.
#[derive(Default)]
pub struct StatsWindow {
    pub open: bool,
}

impl StatsWindow {
    /// Show the statistics window. Call this from the top-level `update()`.
    pub fn show(&mut self, ctx: &egui::Context, result: &DriverConsistencyResult) {
        egui::Window::new("Statistics")
            .open(&mut self.open)
            .default_width(700.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    entry_consistency_section(ui, result);
                    ui.add_space(12.0);
                    corner_speed_section(ui, result);
                    ui.add_space(12.0);
                    opportunity_section(ui, result);
                    ui.add_space(12.0);
                    interpretation_section(ui);
                });
            });
    }
}

fn entry_consistency_section(ui: &mut egui::Ui, result: &DriverConsistencyResult) {
    ui.heading("Entry Consistency");
    ui.separator();
    egui::Grid::new("entry_consistency_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Corner").strong());
            ui.label(RichText::new("TA Mean").strong());
            ui.label(RichText::new("TA Std").strong());
            ui.label(RichText::new("BP Std").strong());
            ui.label(RichText::new("Laps").strong());
            ui.end_row();

            for cd in &result.corner_data {
                ui.label(&cd.corner.name);
                ui.label(format!("{:.1}%", cd.ta_mean));
                ui.label(format!("{:.1}", cd.ta_std));
                ui.label(format!("{:.1}m", cd.bp_std));
                ui.label(format!("{}", cd.ta_values.len()));
                ui.end_row();
            }
        });
}

fn corner_speed_section(ui: &mut egui::Ui, result: &DriverConsistencyResult) {
    ui.heading("Corner Speed & Exit");
    ui.separator();
    egui::Grid::new("corner_speed_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Corner").strong());
            ui.label(RichText::new("Min Spd").strong());
            ui.label(RichText::new("Min Std").strong());
            ui.label(RichText::new("Exit Spd").strong());
            ui.label(RichText::new("Exit Std").strong());
            ui.label(RichText::new("Laps").strong());
            ui.end_row();

            for cd in &result.corner_data {
                ui.label(&cd.corner.name);
                ui.label(format!("{:.1}", cd.speed_mean));
                ui.label(format!("{:.1}", cd.speed_std));
                ui.label(format!("{:.1}", cd.exit_speed_mean));
                ui.label(format!("{:.1}", cd.exit_speed_std));
                ui.label(format!("{}", cd.speed_values.len()));
                ui.end_row();
            }
        });
}

fn opportunity_section(ui: &mut egui::Ui, result: &DriverConsistencyResult) {
    ui.heading("Opportunity Ranking");
    ui.separator();

    let mut ranked: Vec<_> = result.corner_data.iter().collect();
    ranked.sort_by(|a, b| {
        b.opportunity_score
            .partial_cmp(&a.opportunity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    egui::Grid::new("opportunity_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Rank").strong());
            ui.label(RichText::new("Corner").strong());
            ui.label(RichText::new("Exit \u{03c3}").strong());
            ui.label(RichText::new("Accel Zone").strong());
            ui.label(RichText::new("Score").strong());
            ui.end_row();

            for (rank, cd) in ranked.iter().enumerate() {
                ui.label(format!("{}", rank + 1));
                ui.label(RichText::new(&cd.corner.name).color(theme::DARKORANGE));
                ui.label(format!("{:.2}", cd.exit_speed_std));
                ui.label(format!("{:.0}m", cd.accel_zone_length));
                ui.label(format!("{:.1}", cd.opportunity_score));
                ui.end_row();
            }
        });
}

fn interpretation_section(ui: &mut egui::Ui) {
    ui.heading("Interpretation Guide");
    ui.separator();

    ui.label(RichText::new("Throttle Acceptance (TA)").strong());
    ui.label("Percentage of peak lateral G at which the driver goes to full throttle. Higher = more aggressive mid-corner throttle application.");
    ui.add_space(4.0);

    ui.label(RichText::new("Braking Point Std (BP Std)").strong());
    ui.label(
        "Standard deviation of braking start distance across laps. Lower = more consistent braking.",
    );
    ui.add_space(4.0);

    ui.label(RichText::new("Exit Speed Std").strong());
    ui.label("Standard deviation of corner exit speed. Lower = more consistent exits.");
    ui.add_space(4.0);

    ui.label(RichText::new("Opportunity Score").strong());
    ui.label(
        "Exit speed std \u{00d7} acceleration zone length. Higher scores indicate corners where improving consistency yields the most lap time.",
    );
}
