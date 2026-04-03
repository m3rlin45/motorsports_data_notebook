use std::collections::HashMap;

use inferno_core::analysis::tire_grip::{MetricMode, TireGripConfig};
use inferno_core::profile::{self, VehicleProfile};

/// Configuration panel for tire grip analysis.
pub struct TireGripConfigPanel {
    pub channel_names: HashMap<String, String>,
    pub metric_mode: MetricMode,
    pub percentile: f64,
    pub save_requested: bool,
}

impl Default for TireGripConfigPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl TireGripConfigPanel {
    pub fn new() -> Self {
        Self {
            channel_names: profile::default_channel_names(),
            metric_mode: MetricMode::Pressure,
            percentile: 99.9,
            save_requested: false,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        self.save_requested = false;

        ui.group(|ui| {
            ui.heading("Tire Grip Config");
            ui.separator();

            // Metric mode toggle
            ui.horizontal(|ui| {
                ui.label("Mode:");
                if ui
                    .selectable_value(&mut self.metric_mode, MetricMode::Pressure, "Pressure")
                    .changed()
                {
                    changed = true;
                }
                if ui
                    .selectable_value(
                        &mut self.metric_mode,
                        MetricMode::Temperature,
                        "Temperature",
                    )
                    .changed()
                {
                    changed = true;
                }
            });

            ui.add_space(4.0);

            // Percentile
            ui.horizontal(|ui| {
                ui.label("Percentile:");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.percentile)
                            .speed(0.1)
                            .range(50.0..=100.0)
                            .max_decimals(1),
                    )
                    .changed();
            });

            ui.add_space(4.0);

            // Channel names (show relevant set based on mode)
            ui.collapsing("Accel Channels", |ui| {
                changed |= channel_input(ui, &mut self.channel_names, "lateral_g", "Lateral G");
                changed |= channel_input(ui, &mut self.channel_names, "inline_g", "Inline G");
            });

            match self.metric_mode {
                MetricMode::Pressure => {
                    ui.collapsing("Pressure Channels", |ui| {
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_press_fl", "FL");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_press_fr", "FR");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_press_rl", "RL");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_press_rr", "RR");
                    });
                }
                MetricMode::Temperature => {
                    ui.collapsing("Temperature Channels", |ui| {
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_temp_fl", "FL");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_temp_fr", "FR");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_temp_rl", "RL");
                        changed |=
                            channel_input(ui, &mut self.channel_names, "tpms_temp_rr", "RR");
                    });
                }
            }

            ui.add_space(4.0);

            if ui.button("Save Profile").clicked() {
                self.save_requested = true;
            }
        });

        changed
    }

    pub fn set_from_profile(&mut self, profile: &VehicleProfile) {
        self.channel_names = profile.channel_names.clone();
    }

    pub fn to_tire_grip_config(&self) -> TireGripConfig {
        let get = |key: &str| self.channel_names.get(key).cloned().unwrap_or_default();
        TireGripConfig {
            lateral_g: get("lateral_g"),
            inline_g: get("inline_g"),
            tpms_press_fl: get("tpms_press_fl"),
            tpms_press_fr: get("tpms_press_fr"),
            tpms_press_rl: get("tpms_press_rl"),
            tpms_press_rr: get("tpms_press_rr"),
            tpms_temp_fl: get("tpms_temp_fl"),
            tpms_temp_fr: get("tpms_temp_fr"),
            tpms_temp_rl: get("tpms_temp_rl"),
            tpms_temp_rr: get("tpms_temp_rr"),
            metric_mode: self.metric_mode,
            num_buckets: 20,
            percentile: self.percentile,
            min_count: 5,
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
