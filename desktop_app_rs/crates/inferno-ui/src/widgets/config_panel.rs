use std::collections::HashMap;

use inferno_core::analysis::driver_consistency::ChannelConfig;
use inferno_core::profile::{self, VehicleProfile};

/// Configuration panel for channel names and analysis thresholds.
pub struct ConfigPanel {
    pub channel_names: HashMap<String, String>,
    pub corner_threshold: f64,
    pub throttle_threshold: f64,
    pub sustain_time_ms: f64,
    pub save_requested: bool,
}

impl Default for ConfigPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigPanel {
    pub fn new() -> Self {
        Self {
            channel_names: profile::default_channel_names(),
            corner_threshold: 0.006,
            throttle_threshold: 98.0,
            sustain_time_ms: 500.0,
            save_requested: false,
        }
    }

    /// Render the config panel. Returns true if any value changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        self.save_requested = false;

        ui.group(|ui| {
            ui.heading("Configuration");
            ui.separator();

            // Channel names
            ui.collapsing("Channel Names", |ui| {
                changed |= channel_input(ui, &mut self.channel_names, "throttle", "Throttle");
                changed |= channel_input(ui, &mut self.channel_names, "brake", "Brake");
                changed |= channel_input(ui, &mut self.channel_names, "lateral_g", "Lateral G");
                changed |= channel_input(ui, &mut self.channel_names, "gps_latitude", "GPS Lat");
                changed |= channel_input(ui, &mut self.channel_names, "gps_longitude", "GPS Lon");
                changed |= channel_input(ui, &mut self.channel_names, "gps_speed", "GPS Speed");
            });

            ui.add_space(4.0);

            // Thresholds
            ui.collapsing("Thresholds", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Corner detection:");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.corner_threshold).speed(0.0001))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Throttle %:");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.throttle_threshold)
                                .speed(0.5)
                                .range(0.0..=100.0),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Sustain (ms):");
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.sustain_time_ms)
                                .speed(10.0)
                                .range(0.0..=5000.0),
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
    }

    /// Convert current channel names to a ChannelConfig for analysis.
    pub fn to_channel_config(&self) -> ChannelConfig {
        let get = |key: &str| self.channel_names.get(key).cloned().unwrap_or_default();
        ChannelConfig {
            throttle: get("throttle"),
            brake: get("brake"),
            lateral_g: get("lateral_g"),
            gps_lat: get("gps_latitude"),
            gps_lon: get("gps_longitude"),
            gps_speed: get("gps_speed"),
        }
    }
}

/// Helper: render a labeled text input for a channel name. Returns true if changed.
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
