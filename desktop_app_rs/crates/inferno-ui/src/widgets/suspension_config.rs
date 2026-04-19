use std::collections::HashMap;

use inferno_core::analysis::suspension::{SuspensionConfig, VelocityRanges};
use inferno_core::profile::{self, MotionRatios, VehicleProfile};

/// Configuration panel for suspension velocity analysis.
pub struct SuspensionConfigPanel {
    pub channel_names: HashMap<String, String>,
    pub motion_ratios: MotionRatios,
    pub velocity_ranges: VelocityRanges,
    pub smoothing_window: usize,
    pub bin_size: f64,
    pub save_requested: bool,
}

impl Default for SuspensionConfigPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl SuspensionConfigPanel {
    pub fn new() -> Self {
        Self {
            channel_names: profile::default_channel_names(),
            motion_ratios: MotionRatios::default(),
            velocity_ranges: VelocityRanges::default(),
            smoothing_window: 5,
            bin_size: 10.0,
            save_requested: false,
        }
    }

    /// Render the config panel. Returns true if any value changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        self.save_requested = false;

        ui.group(|ui| {
            ui.heading("Suspension Config");
            ui.separator();

            // Shock channel names
            ui.collapsing("Shock Channels", |ui| {
                changed |= channel_input(ui, &mut self.channel_names, "shock_fl", "FL");
                changed |= channel_input(ui, &mut self.channel_names, "shock_fr", "FR");
                changed |= channel_input(ui, &mut self.channel_names, "shock_rl", "RL");
                changed |= channel_input(ui, &mut self.channel_names, "shock_rr", "RR");
            });

            ui.add_space(4.0);

            // Motion ratios
            ui.collapsing("Motion Ratios", |ui| {
                changed |= ratio_input(ui, &mut self.motion_ratios.front_left, "FL");
                changed |= ratio_input(ui, &mut self.motion_ratios.front_right, "FR");
                changed |= ratio_input(ui, &mut self.motion_ratios.rear_left, "RL");
                changed |= ratio_input(ui, &mut self.motion_ratios.rear_right, "RR");
            });

            ui.add_space(4.0);

            // Velocity ranges & analysis params
            ui.collapsing("Thresholds", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Friction (mm/s):");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.velocity_ranges.friction)
                                .speed(0.5)
                                .range(0.0..=50.0),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Slow (mm/s):");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.velocity_ranges.slow)
                                .speed(1.0)
                                .range(1.0..=200.0),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Fast (mm/s):");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.velocity_ranges.fast)
                                .speed(5.0)
                                .range(10.0..=500.0),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Smoothing:");
                    let mut sw = self.smoothing_window as f64;
                    if ui
                        .add(egui::DragValue::new(&mut sw).speed(1.0).range(1.0..=20.0))
                        .changed()
                    {
                        self.smoothing_window = sw as usize;
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Bin size (mm/s):");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.bin_size)
                                .speed(1.0)
                                .range(1.0..=50.0),
                        )
                        .changed();
                });
            });

            ui.add_space(4.0);

            if ui.button("Save Profile").clicked() {
                self.save_requested = true;
            }
        });

        changed
    }

    /// Populate from a vehicle profile.
    pub fn set_from_profile(&mut self, profile: &VehicleProfile) {
        self.channel_names = profile.channel_names.clone();
        self.motion_ratios = profile.motion_ratios.clone();
    }

    /// Build a SuspensionConfig from current panel values.
    pub fn to_suspension_config(&self) -> SuspensionConfig {
        let get = |key: &str| self.channel_names.get(key).cloned().unwrap_or_default();
        SuspensionConfig {
            shock_fl: get("shock_fl"),
            shock_fr: get("shock_fr"),
            shock_rl: get("shock_rl"),
            shock_rr: get("shock_rr"),
            motion_ratios: self.motion_ratios.clone(),
            velocity_ranges: self.velocity_ranges.clone(),
            smoothing_window: self.smoothing_window,
            bin_size: self.bin_size,
            max_velocity: 300.0,
        }
    }
}

fn channel_input(
    ui: &mut egui::Ui,
    map: &mut HashMap<String, String>,
    key: &str,
    label: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        let entry = map.entry(key.to_string()).or_default();
        changed = ui.text_edit_singleline(entry).changed();
    });
    changed
}

fn ratio_input(ui: &mut egui::Ui, value: &mut f64, label: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        changed = ui
            .add(
                egui::DragValue::new(value)
                    .speed(0.01)
                    .range(0.1..=2.0)
                    .max_decimals(3),
            )
            .changed();
    });
    changed
}
