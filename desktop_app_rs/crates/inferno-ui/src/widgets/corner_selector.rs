use inferno_core::analysis::corners::Corner;

/// View mode: summary of all corners, detail of a single corner, or full track map.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewMode {
    Summary,
    Detail(usize),
    TrackMap,
}

/// Sidebar widget for selecting view mode and individual corners.
pub struct CornerSelector {
    pub view_mode: ViewMode,
    pub corners: Vec<(String, char)>,
    selected_corner: usize,
}

impl Default for CornerSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl CornerSelector {
    pub fn new() -> Self {
        Self {
            view_mode: ViewMode::Summary,
            corners: Vec::new(),
            selected_corner: 0,
        }
    }

    /// Render the corner selector. Returns true if selection changed.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.group(|ui| {
            ui.heading("View");
            ui.separator();

            // Summary / Detail radio
            if ui
                .radio_value(&mut self.view_mode, ViewMode::Summary, "Summary")
                .changed()
            {
                changed = true;
            }
            let detail_mode = ViewMode::Detail(self.selected_corner);
            if ui
                .radio_value(&mut self.view_mode, detail_mode, "Detail")
                .changed()
            {
                changed = true;
            }

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Corners:")
                    .strong()
                    .color(crate::theme::STEELBLUE),
            );

            if self.corners.is_empty() {
                ui.label(
                    egui::RichText::new("Load a session to detect corners")
                        .color(crate::theme::TEXT_SECONDARY),
                );
            } else {
                egui::ScrollArea::vertical()
                    .max_height(400.0)
                    .show(ui, |ui| {
                        for (i, (name, dir)) in self.corners.iter().enumerate() {
                            let label = format!("{name} ({dir})");
                            let response = ui.radio_value(&mut self.selected_corner, i, label);
                            if response.changed() {
                                // Clicking a corner auto-switches to Detail mode
                                self.view_mode = ViewMode::Detail(i);
                                changed = true;
                            }
                        }
                    });
            }

            // Keep Detail mode in sync with selected corner
            if matches!(self.view_mode, ViewMode::Detail(_)) {
                self.view_mode = ViewMode::Detail(self.selected_corner);
            }
        });

        changed
    }

    /// Update the corner list from analysis results.
    pub fn update_corners(&mut self, corners: &[Corner]) {
        self.corners = corners
            .iter()
            .map(|c| (c.name.clone(), c.direction))
            .collect();
        self.selected_corner = 0;
        if !corners.is_empty() {
            self.view_mode = ViewMode::Summary;
        }
    }

    /// Reset to empty state.
    pub fn clear(&mut self) {
        self.corners.clear();
        self.selected_corner = 0;
        self.view_mode = ViewMode::Summary;
    }

    /// Get the currently selected corner index (if in Detail mode).
    pub fn selected_corner_index(&self) -> Option<usize> {
        match self.view_mode {
            ViewMode::Detail(i) => Some(i),
            ViewMode::Summary | ViewMode::TrackMap => None,
        }
    }
}
