use std::path::PathBuf;
use std::sync::Arc;

use egui::RichText;
use inferno_core::lap::Lap;
use inferno_core::session::Session;

/// Response events from the session panel.
pub struct SessionPanelResponse {
    /// A new session was loaded.
    pub file_loaded: Option<Arc<Session>>,
    /// Lap selection changed.
    pub selection_changed: bool,
}

/// Panel for loading telemetry files and selecting laps.
pub struct SessionPanel {
    pub title: String,
    pub file_path: Option<PathBuf>,
    pub session: Option<Arc<Session>>,
    pub lap_selected: Vec<bool>,
    pub file_dialog_open: bool,
}

impl SessionPanel {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            file_path: None,
            session: None,
            lap_selected: Vec::new(),
            file_dialog_open: false,
        }
    }

    /// Render the session panel and return response events.
    pub fn show(&mut self, ui: &mut egui::Ui) -> SessionPanelResponse {
        let mut response = SessionPanelResponse {
            file_loaded: None,
            selection_changed: false,
        };

        ui.group(|ui| {
            ui.heading(&self.title);
            ui.separator();

            // File loading
            ui.horizontal(|ui| {
                if ui.button("Browse...").clicked() && !self.file_dialog_open {
                    self.file_dialog_open = true;
                    let path = rfd::FileDialog::new()
                        .add_filter("Telemetry", &["xrk", "xrz", "ibt"])
                        .pick_file();
                    self.file_dialog_open = false;

                    if let Some(path) = path {
                        self.load_file(&path);
                        response.file_loaded = self.session.clone();
                    }
                }

                if let Some(path) = &self.file_path {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    ui.label(name);
                } else {
                    ui.label(RichText::new("No file loaded").color(crate::theme::TEXT_SECONDARY));
                }
            });

            // Handle dropped files
            let dropped_files = ui.input(|i| i.raw.dropped_files.clone());
            for file in &dropped_files {
                if let Some(path) = &file.path {
                    self.load_file(path);
                    response.file_loaded = self.session.clone();
                    break;
                }
            }

            // Lap selection
            if let Some(session) = &self.session {
                ui.add_space(8.0);
                ui.label(
                    RichText::new(format!("{} laps", session.laps.len()))
                        .color(crate::theme::STEELBLUE),
                );

                ui.horizontal(|ui| {
                    if ui.button("Select All").clicked() {
                        self.lap_selected.fill(true);
                        response.selection_changed = true;
                    }
                    if ui.button("Clear").clicked() {
                        self.lap_selected.fill(false);
                        response.selection_changed = true;
                    }
                });

                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for (i, lap) in session.laps.iter().enumerate() {
                            if i < self.lap_selected.len() {
                                let label = format_lap_label(lap);
                                if ui.checkbox(&mut self.lap_selected[i], label).changed() {
                                    response.selection_changed = true;
                                }
                            }
                        }
                    });
            }
        });

        response
    }

    /// Get the list of selected lap numbers.
    pub fn selected_laps(&self) -> Vec<i32> {
        self.session
            .as_ref()
            .map(|s| {
                s.laps
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.lap_selected.get(*i).copied().unwrap_or(false))
                    .map(|(_, lap)| lap.num)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn load_file(&mut self, path: &std::path::Path) {
        match Session::open(path) {
            Ok(session) => {
                let num_laps = session.laps.len();
                self.file_path = Some(path.to_path_buf());
                self.session = Some(Arc::new(session));
                self.lap_selected = vec![true; num_laps];
            }
            Err(e) => {
                eprintln!("Failed to load session: {e}");
            }
        }
    }
}

fn format_lap_label(lap: &Lap) -> String {
    let duration_ms = lap.duration_ms();
    let mins = duration_ms / 60_000;
    let secs = (duration_ms % 60_000) as f64 / 1000.0;
    format!("Lap {} ({mins}:{secs:06.3})", lap.num)
}
