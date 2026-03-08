use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;

use inferno_core::analysis::driver_consistency::{
    analyze_driver_consistency, DriverConsistencyResult,
};
use inferno_core::channel;
use inferno_core::error::Error;
use inferno_core::lap::get_top_laps;
use inferno_core::profile;
use inferno_core::session::Session;

use inferno_ui::charts;
use inferno_ui::widgets::config_panel::ConfigPanel;
use inferno_ui::widgets::corner_selector::{CornerSelector, ViewMode};
use inferno_ui::widgets::session_panel::SessionPanel;
use inferno_ui::widgets::stats_window::StatsWindow;

const DEBOUNCE_MS: u64 = 300;

type AnalysisResult = Result<DriverConsistencyResult, Error>;

pub struct InfernoApp {
    // Widgets
    session_a: SessionPanel,
    session_b: SessionPanel,
    config: ConfigPanel,
    pub corner_selector: CornerSelector,
    stats_window: StatsWindow,

    // Analysis results
    result_a: Option<DriverConsistencyResult>,
    result_b: Option<DriverConsistencyResult>,

    // Background analysis receivers
    rx_a: Option<mpsc::Receiver<AnalysisResult>>,
    rx_b: Option<mpsc::Receiver<AnalysisResult>>,

    // State
    status: String,
    analyzing: bool,
    pending_a: bool,
    pending_b: bool,
    last_change: Option<Instant>,
    pub top_collapsed: bool,
}

impl InfernoApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut visuals = egui::Visuals::dark();
        let bg = egui::Color32::from_rgb(0x1a, 0x1a, 0x2e);
        visuals.panel_fill = bg;
        visuals.window_fill = bg;
        visuals.extreme_bg_color = egui::Color32::from_rgb(0x12, 0x12, 0x22);
        visuals.faint_bg_color = egui::Color32::from_rgb(0x22, 0x22, 0x3a);
        cc.egui_ctx.set_visuals(visuals);

        Self {
            session_a: SessionPanel::new("Session A"),
            session_b: SessionPanel::new("Session B"),
            config: ConfigPanel::new(),
            corner_selector: CornerSelector::new(),
            stats_window: StatsWindow::default(),
            result_a: None,
            result_b: None,
            rx_a: None,
            rx_b: None,
            status: "Load a telemetry file to begin".into(),
            analyzing: false,
            pending_a: false,
            pending_b: false,
            last_change: None,
            top_collapsed: false,
        }
    }

    /// Pre-load a session, run analysis synchronously, and populate state.
    /// Used by snapshot tests to render charts without user interaction.
    pub fn load_and_analyze(&mut self, path: &Path) {
        let session = Arc::new(Session::open(path).expect("failed to open session"));

        // Auto-detect profile and set channel names
        let logger_id = profile::get_logger_id(&session);
        if let Some(prof) = profile::get_profile_for_logger(&logger_id) {
            self.config.set_from_profile(&prof);
        }

        // Auto-detect throttle threshold: 95% of peak throttle value
        let throttle_name = self
            .config
            .channel_names
            .get("throttle")
            .cloned()
            .unwrap_or_default();
        if let Some(batch) = session.channels.get(&throttle_name) {
            if let Ok(values) = channel::get_values_f64(batch) {
                let peak = values
                    .values()
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                if peak > 0.0 {
                    self.config.throttle_threshold = 0.95 * peak;
                }
            }
        }

        // Select top 103% laps
        let top = get_top_laps(&session.laps, 1.03);
        let selected_laps: Vec<i32> = top.iter().map(|l| l.num).collect();

        // Store session in session_a panel
        self.session_a.lap_selected = vec![false; session.laps.len()];
        for (i, lap) in session.laps.iter().enumerate() {
            if selected_laps.contains(&lap.num) {
                self.session_a.lap_selected[i] = true;
            }
        }
        self.session_a.session = Some(session.clone());
        self.session_a.file_path = Some(path.to_path_buf());

        // Run analysis synchronously
        let config = self.config.to_channel_config();
        let result = analyze_driver_consistency(
            &session,
            &selected_laps,
            &config,
            self.config.corner_threshold,
            self.config.throttle_threshold,
            self.config.sustain_time_ms,
        )
        .expect("analysis failed");

        self.corner_selector.update_corners(&result.corners);
        self.status = format!("{} corners detected", result.corner_data.len());
        self.result_a = Some(result);
    }

    fn schedule_analysis(&mut self, target_b: bool) {
        if target_b {
            self.pending_b = true;
        } else {
            self.pending_a = true;
        }
        self.last_change = Some(Instant::now());
    }

    fn check_debounce(&mut self, ctx: &egui::Context) {
        let Some(last) = self.last_change else {
            return;
        };
        if last.elapsed() < Duration::from_millis(DEBOUNCE_MS) {
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        }
        self.last_change = None;
        if self.pending_a {
            self.pending_a = false;
            self.fire_analysis(ctx, false);
        }
        if self.pending_b {
            self.pending_b = false;
            self.fire_analysis(ctx, true);
        }
    }

    fn fire_analysis(&mut self, ctx: &egui::Context, target_b: bool) {
        let panel = if target_b {
            &self.session_b
        } else {
            &self.session_a
        };
        let session = match panel.session.clone() {
            Some(s) => s,
            None => return,
        };
        let laps = panel.selected_laps();
        if laps.is_empty() {
            if !target_b {
                self.status = "No laps selected".into();
            }
            return;
        }

        let config = self.config.to_channel_config();
        let ct = self.config.corner_threshold;
        let tt = self.config.throttle_threshold;
        let st = self.config.sustain_time_ms;

        let (tx, rx) = mpsc::channel();
        if target_b {
            self.rx_b = Some(rx);
        } else {
            self.rx_a = Some(rx);
        }
        self.analyzing = true;
        if !target_b {
            self.status = "Analyzing...".into();
        }

        let ctx_c = ctx.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                analyze_driver_consistency(&session, &laps, &config, ct, tt, st)
            }));
            let result = match result {
                Ok(r) => r,
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    Err(Error::Other(format!("Analysis panicked: {msg}")))
                }
            };
            let _ = tx.send(result);
            ctx_c.request_repaint();
        });
    }

    fn poll_results(&mut self) {
        if let Some(rx) = &self.rx_a {
            match rx.try_recv() {
                Ok(result) => {
                    self.rx_a = None;
                    match result {
                        Ok(dr) => {
                            self.corner_selector.update_corners(&dr.corners);
                            self.status = format!("{} corners detected", dr.corner_data.len());
                            self.result_a = Some(dr);
                        }
                        Err(e) => {
                            self.status = format!("Error: {e}");
                            self.result_a = None;
                            self.corner_selector.clear();
                        }
                    }
                    self.analyzing = self.rx_b.is_some();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_a = None;
                    self.status = "Analysis failed unexpectedly".into();
                    self.analyzing = self.rx_b.is_some();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        if let Some(rx) = &self.rx_b {
            match rx.try_recv() {
                Ok(result) => {
                    self.rx_b = None;
                    match result {
                        Ok(dr) => self.result_b = Some(dr),
                        Err(_) => self.result_b = None,
                    }
                    self.analyzing = self.rx_a.is_some();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_b = None;
                    self.analyzing = self.rx_a.is_some();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }
}

impl eframe::App for InfernoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_results();
        self.check_debounce(ctx);

        // Event flags — set inside UI closures, handled after all panels render
        let mut trigger_a = false;
        let mut trigger_b = false;
        let mut config_changed = false;
        let mut save_profile = false;
        let mut new_session_a = false;

        // Stats popup window
        if let Some(result) = &self.result_a {
            self.stats_window.show(ctx, result);
        }

        // Bottom status bar
        egui::TopBottomPanel::bottom("status")
            .exact_height(28.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    if self.analyzing {
                        ui.spinner();
                    }
                    ui.label(&self.status);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.result_a.is_some() && ui.button("Statistics").clicked() {
                            self.stats_window.open = !self.stats_window.open;
                        }
                        let label = if self.top_collapsed {
                            "\u{25bc} Show"
                        } else {
                            "\u{25b2} Hide"
                        };
                        if ui.button(label).clicked() {
                            self.top_collapsed = !self.top_collapsed;
                        }
                    });
                });
            });

        // Top panel — session loading + config (collapsible)
        if !self.top_collapsed {
            egui::TopBottomPanel::top("config")
                .resizable(true)
                .default_height(220.0)
                .show(ctx, |ui| {
                    ui.columns(3, |cols| {
                        // Session A
                        let resp = self.session_a.show(&mut cols[0]);
                        if resp.file_loaded.is_some() {
                            new_session_a = true;
                            trigger_a = true;
                        }
                        if resp.selection_changed {
                            trigger_a = true;
                        }

                        // Session B
                        let resp = self.session_b.show(&mut cols[1]);
                        if resp.file_loaded.is_some() {
                            trigger_b = true;
                        }
                        if resp.selection_changed {
                            trigger_b = true;
                        }

                        // Config
                        config_changed = self.config.show(&mut cols[2]);
                        save_profile = self.config.save_requested;
                    });
                });
        }

        // Handle events outside closures to avoid borrow conflicts
        if new_session_a {
            if let Some(session) = &self.session_a.session {
                let logger_id = profile::get_logger_id(session);
                if let Some(prof) = profile::get_profile_for_logger(&logger_id) {
                    self.status = format!("Loaded — logger {} | profile: {}", logger_id, prof.name);
                    self.config.set_from_profile(&prof);
                } else {
                    self.status =
                        format!("Loaded — logger {logger_id} | no profile found, using defaults");
                }

                // Auto-detect throttle threshold: 95% of peak throttle value
                let throttle_name = self
                    .config
                    .channel_names
                    .get("throttle")
                    .cloned()
                    .unwrap_or_default();
                if let Some(batch) = session.channels.get(&throttle_name) {
                    if let Ok(values) = channel::get_values_f64(batch) {
                        let peak = values
                            .values()
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        if peak > 0.0 {
                            self.config.throttle_threshold = 0.95 * peak;
                        }
                    }
                }
            }
        }
        if trigger_a || config_changed {
            self.schedule_analysis(false);
        }
        if trigger_b || (config_changed && self.session_b.session.is_some()) {
            self.schedule_analysis(true);
        }
        if save_profile {
            if let Some(session) = &self.session_a.session {
                let lid = profile::get_logger_id(session);
                let prof = inferno_core::profile::VehicleProfile {
                    name: lid.clone(),
                    channel_names: self.config.channel_names.clone(),
                    motion_ratios: Default::default(),
                };
                match profile::save_profile_for_logger(&lid, &lid, &prof) {
                    Ok(()) => self.status = "Profile saved".into(),
                    Err(e) => self.status = format!("Save failed: {e}"),
                }
            }
        }

        // Corner selector sidebar
        egui::SidePanel::left("corners")
            .default_width(150.0)
            .show(ctx, |ui| {
                self.corner_selector.show(ui);

                // Track map thumbnail at the bottom of the sidebar
                if let Some(result) = &self.result_a {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("Track Map")
                            .strong()
                            .color(inferno_ui::theme::STEELBLUE),
                    );
                    if charts::track_map::draw_track_map_thumbnail(ui, result) {
                        self.corner_selector.view_mode = ViewMode::TrackMap;
                    }
                }
            });

        // Central chart area
        egui::CentralPanel::default().show(ctx, |ui| match &self.corner_selector.view_mode {
            ViewMode::Summary => {
                if let Some(result) = &self.result_a {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if let Some(rb) = &self.result_b {
                            charts::summary::draw_summary_comparison(
                                ui,
                                &result.corner_data,
                                &rb.corner_data,
                            );
                        } else {
                            charts::summary::draw_summary(ui, &result.corner_data);
                        }
                    });
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Load a telemetry file to begin analysis");
                    });
                }
            }
            ViewMode::Detail(idx) => {
                if let Some(result) = &self.result_a {
                    if let Some(cd) = result.corner_data.get(*idx) {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            charts::detail::draw_detail(ui, cd);
                        });
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Load a telemetry file to begin analysis");
                    });
                }
            }
            ViewMode::TrackMap => {
                if let Some(result) = &self.result_a {
                    charts::track_map::draw_track_map(ui, result);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Load a telemetry file to begin analysis");
                    });
                }
            }
        });
    }
}
