use egui::RichText;
use inferno_core::analysis::suspension::SuspensionResult;

use crate::theme;

/// Popup window displaying suspension velocity statistics.
#[derive(Default)]
pub struct SuspensionStatsWindow {
    pub open: bool,
}

impl SuspensionStatsWindow {
    /// Show the statistics window. Call from the top-level `update()`.
    pub fn show(&mut self, ctx: &egui::Context, result: &SuspensionResult) {
        egui::Window::new("Suspension Statistics")
            .open(&mut self.open)
            .default_width(700.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    summary_section(ui, result);
                    ui.add_space(12.0);
                    velocity_range_section(ui, result);
                    ui.add_space(12.0);
                    balance_section(ui, result);
                    ui.add_space(12.0);
                    interpretation_section(ui);
                });
            });
    }
}

fn summary_section(ui: &mut egui::Ui, result: &SuspensionResult) {
    ui.heading("Summary Statistics");
    ui.separator();
    egui::Grid::new("susp_summary_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Wheel").strong());
            ui.label(RichText::new("Skew").strong());
            ui.label(RichText::new("Kurtosis").strong());
            ui.label(RichText::new("Mean (mm/s)").strong());
            ui.label(RichText::new("Std (mm/s)").strong());
            ui.end_row();

            for w in &result.wheels {
                ui.label(&w.name);
                ui.label(format!("{:.3}", w.skew));
                ui.label(format!("{:.3}", w.kurtosis));
                ui.label(format!("{:.1}", w.mean));
                ui.label(format!("{:.1}", w.std));
                ui.end_row();
            }
        });
}

fn velocity_range_section(ui: &mut egui::Ui, result: &SuspensionResult) {
    ui.heading("Velocity Range Distribution");
    ui.separator();
    egui::Grid::new("susp_range_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Wheel").strong());
            ui.label(RichText::new("Friction %").strong());
            ui.label(RichText::new("Slow Bump %").strong());
            ui.label(RichText::new("Slow Reb %").strong());
            ui.label(RichText::new("Fast Bump %").strong());
            ui.label(RichText::new("Fast Reb %").strong());
            ui.label(RichText::new("Curb %").strong());
            ui.end_row();

            for w in &result.wheels {
                ui.label(&w.name);
                ui.label(format!("{:.1}", w.pct_friction));
                ui.label(format!("{:.1}", w.pct_slow_bump));
                ui.label(format!("{:.1}", w.pct_slow_rebound));
                ui.label(format!("{:.1}", w.pct_fast_bump));
                ui.label(format!("{:.1}", w.pct_fast_rebound));
                ui.label(format!("{:.1}", w.pct_curb));
                ui.end_row();
            }
        });
}

fn balance_section(ui: &mut egui::Ui, result: &SuspensionResult) {
    ui.heading("Balance Analysis");
    ui.separator();

    let [ref fl, ref fr, ref rl, ref rr] = result.wheels;

    let front_skew = (fl.skew + fr.skew) / 2.0;
    let rear_skew = (rl.skew + rr.skew) / 2.0;
    let left_skew = (fl.skew + rl.skew) / 2.0;
    let right_skew = (fr.skew + rr.skew) / 2.0;

    egui::Grid::new("susp_balance_grid")
        .striped(true)
        .show(ui, |ui| {
            ui.label(RichText::new("Comparison").strong());
            ui.label(RichText::new("Avg Skew").strong());
            ui.label(RichText::new("Interpretation").strong());
            ui.end_row();

            ui.label("Front");
            ui.label(format!("{front_skew:.3}"));
            ui.label(skew_interpretation(front_skew));
            ui.end_row();

            ui.label("Rear");
            ui.label(format!("{rear_skew:.3}"));
            ui.label(skew_interpretation(rear_skew));
            ui.end_row();

            ui.label("Left (FL+RL)");
            ui.label(format!("{left_skew:.3}"));
            ui.label(skew_interpretation(left_skew));
            ui.end_row();

            ui.label("Right (FR+RR)");
            ui.label(format!("{right_skew:.3}"));
            ui.label(skew_interpretation(right_skew));
            ui.end_row();
        });

    ui.add_space(4.0);

    let fr_diff = (front_skew - rear_skew).abs();
    let lr_diff = (left_skew - right_skew).abs();
    if fr_diff > 0.1 {
        ui.label(
            RichText::new(format!(
                "Front/Rear skew difference: {:.3} — {}",
                fr_diff,
                if front_skew > rear_skew {
                    "front biased toward rebound"
                } else {
                    "rear biased toward rebound"
                }
            ))
            .color(theme::DARKORANGE),
        );
    }
    if lr_diff > 0.1 {
        ui.label(
            RichText::new(format!(
                "Left/Right skew difference: {:.3} — {}",
                lr_diff,
                if left_skew > right_skew {
                    "left side biased toward rebound"
                } else {
                    "right side biased toward rebound"
                }
            ))
            .color(theme::DARKORANGE),
        );
    }
}

fn skew_interpretation(skew: f64) -> String {
    if skew.abs() < 0.05 {
        "Balanced".to_string()
    } else if skew > 0.0 {
        format!("Rebound biased (+{skew:.3})")
    } else {
        format!("Bump biased ({skew:.3})")
    }
}

fn interpretation_section(ui: &mut egui::Ui) {
    ui.heading("Interpretation Guide");
    ui.separator();

    ui.label(RichText::new("Skewness").strong());
    ui.label("Positive skew = more time in rebound (extension). Negative = more bump (compression). Balanced suspension has skew near zero.");
    ui.add_space(4.0);

    ui.label(RichText::new("Kurtosis").strong());
    ui.label("Higher kurtosis = sharper peak with heavy tails (more extreme velocities). Negative = flatter distribution.");
    ui.add_space(4.0);

    ui.label(RichText::new("Standard Deviation").strong());
    ui.label("Higher std = more suspension activity. Compare across wheels to identify imbalance.");
    ui.add_space(4.0);

    ui.label(RichText::new("Velocity Ranges").strong());
    ui.label("Friction (<5 mm/s): near-static, dominated by stiction. Slow (5-25): gentle damping. Fast (25-200): aggressive response. Curb (>200): impacts/curbs.");
    ui.add_space(4.0);

    ui.label(RichText::new("Balance").strong());
    ui.label("Compare front vs rear and left vs right skew. Large differences suggest asymmetric damping or setup issues.");
}
