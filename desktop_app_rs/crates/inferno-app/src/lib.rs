use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;

use inferno_core::analysis::driver_consistency::{
    analyze_driver_consistency, DriverConsistencyResult,
};
use inferno_core::analysis::suspension::{analyze_suspension_velocity, SuspensionResult};
use inferno_core::analysis::tire_grip::{analyze_tire_grip, TireGripResult};
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
use inferno_ui::widgets::suspension_config::SuspensionConfigPanel;
use inferno_ui::widgets::suspension_stats::SuspensionStatsWindow;
use inferno_ui::widgets::tire_grip_config::TireGripConfigPanel;
use inferno_ui::widgets::tire_grip_stats::TireGripStatsWindow;

const DEBOUNCE_MS: u64 = 300;

type DriverAnalysisResult = Result<DriverConsistencyResult, Error>;
type SuspensionAnalysisResult = Result<SuspensionResult, Error>;
type TireGripAnalysisResult = Result<TireGripResult, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    DriverConsistency,
    Suspension,
    TireGrip,
}

pub struct InfernoApp {
    // Tab state
    pub active_tab: ActiveTab,

    // Shared widgets
    session_a: SessionPanel,
    session_b: SessionPanel,

    // Driver consistency tab
    config: ConfigPanel,
    pub corner_selector: CornerSelector,
    stats_window: StatsWindow,
    result_a: Option<DriverConsistencyResult>,
    result_b: Option<DriverConsistencyResult>,
    rx_a: Option<mpsc::Receiver<DriverAnalysisResult>>,
    rx_b: Option<mpsc::Receiver<DriverAnalysisResult>>,

    // Suspension tab
    susp_config: SuspensionConfigPanel,
    susp_stats: SuspensionStatsWindow,
    susp_result_a: Option<SuspensionResult>,
    susp_result_b: Option<SuspensionResult>,
    susp_rx_a: Option<mpsc::Receiver<SuspensionAnalysisResult>>,
    susp_rx_b: Option<mpsc::Receiver<SuspensionAnalysisResult>>,

    // Tire grip tab
    tire_config: TireGripConfigPanel,
    tire_stats: TireGripStatsWindow,
    tire_result_a: Option<TireGripResult>,
    tire_result_b: Option<TireGripResult>,
    tire_rx_a: Option<mpsc::Receiver<TireGripAnalysisResult>>,
    tire_rx_b: Option<mpsc::Receiver<TireGripAnalysisResult>>,

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
            active_tab: ActiveTab::DriverConsistency,
            session_a: SessionPanel::new("Session A"),
            session_b: SessionPanel::new("Session B"),
            config: ConfigPanel::new(),
            corner_selector: CornerSelector::new(),
            stats_window: StatsWindow::default(),
            result_a: None,
            result_b: None,
            rx_a: None,
            rx_b: None,
            susp_config: SuspensionConfigPanel::new(),
            susp_stats: SuspensionStatsWindow::default(),
            susp_result_a: None,
            susp_result_b: None,
            susp_rx_a: None,
            susp_rx_b: None,
            tire_config: TireGripConfigPanel::new(),
            tire_stats: TireGripStatsWindow::default(),
            tire_result_a: None,
            tire_result_b: None,
            tire_rx_a: None,
            tire_rx_b: None,
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
            self.susp_config.set_from_profile(&prof);
            self.tire_config.set_from_profile(&prof);
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

        // Run driver consistency analysis synchronously
        let driver_config = self.config.to_channel_config();
        let result = analyze_driver_consistency(
            &session,
            &selected_laps,
            &driver_config,
            self.config.corner_threshold,
            self.config.throttle_threshold,
            self.config.sustain_time_ms,
        )
        .expect("analysis failed");

        self.corner_selector.update_corners(&result.corners);
        self.status = format!("{} corners detected", result.corner_data.len());
        self.result_a = Some(result);

        // Run suspension analysis synchronously
        let susp_config = self.susp_config.to_suspension_config();
        if let Ok(susp_result) = analyze_suspension_velocity(&session, &selected_laps, &susp_config)
        {
            self.susp_result_a = Some(susp_result);
        }

        // Run tire grip analysis synchronously
        let tire_config = self.tire_config.to_tire_grip_config();
        if let Ok(tire_result) = analyze_tire_grip(&session, &selected_laps, &tire_config) {
            self.tire_result_a = Some(tire_result);
        }
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
            self.fire_driver_analysis(ctx, false);
            self.fire_suspension_analysis(ctx, false);
            self.fire_tire_grip_analysis(ctx, false);
        }
        if self.pending_b {
            self.pending_b = false;
            self.fire_driver_analysis(ctx, true);
            self.fire_suspension_analysis(ctx, true);
            self.fire_tire_grip_analysis(ctx, true);
        }
    }

    fn fire_driver_analysis(&mut self, ctx: &egui::Context, target_b: bool) {
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

    fn fire_suspension_analysis(&mut self, ctx: &egui::Context, target_b: bool) {
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
            return;
        }

        let config = self.susp_config.to_suspension_config();
        let (tx, rx) = mpsc::channel();
        if target_b {
            self.susp_rx_b = Some(rx);
        } else {
            self.susp_rx_a = Some(rx);
        }

        let ctx_c = ctx.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                analyze_suspension_velocity(&session, &laps, &config)
            }));
            let result = match result {
                Ok(r) => r,
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    Err(Error::Other(format!("Suspension analysis panicked: {msg}")))
                }
            };
            let _ = tx.send(result);
            ctx_c.request_repaint();
        });
    }

    fn fire_tire_grip_analysis(&mut self, ctx: &egui::Context, target_b: bool) {
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
            return;
        }

        let config = self.tire_config.to_tire_grip_config();
        let (tx, rx) = mpsc::channel();
        if target_b {
            self.tire_rx_b = Some(rx);
        } else {
            self.tire_rx_a = Some(rx);
        }

        let ctx_c = ctx.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                analyze_tire_grip(&session, &laps, &config)
            }));
            let result = match result {
                Ok(r) => r,
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    Err(Error::Other(format!("Tire grip analysis panicked: {msg}")))
                }
            };
            let _ = tx.send(result);
            ctx_c.request_repaint();
        });
    }

    fn poll_results(&mut self) {
        // Driver consistency results
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
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_a = None;
                    self.status = "Analysis failed unexpectedly".into();
                    self.update_analyzing();
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
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_b = None;
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        // Suspension results
        if let Some(rx) = &self.susp_rx_a {
            match rx.try_recv() {
                Ok(result) => {
                    self.susp_rx_a = None;
                    match result {
                        Ok(sr) => self.susp_result_a = Some(sr),
                        Err(_) => self.susp_result_a = None,
                    }
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.susp_rx_a = None;
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        if let Some(rx) = &self.susp_rx_b {
            match rx.try_recv() {
                Ok(result) => {
                    self.susp_rx_b = None;
                    match result {
                        Ok(sr) => self.susp_result_b = Some(sr),
                        Err(_) => self.susp_result_b = None,
                    }
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.susp_rx_b = None;
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        // Tire grip results
        if let Some(rx) = &self.tire_rx_a {
            match rx.try_recv() {
                Ok(result) => {
                    self.tire_rx_a = None;
                    match result {
                        Ok(tr) => self.tire_result_a = Some(tr),
                        Err(_) => self.tire_result_a = None,
                    }
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.tire_rx_a = None;
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        if let Some(rx) = &self.tire_rx_b {
            match rx.try_recv() {
                Ok(result) => {
                    self.tire_rx_b = None;
                    match result {
                        Ok(tr) => self.tire_result_b = Some(tr),
                        Err(_) => self.tire_result_b = None,
                    }
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.tire_rx_b = None;
                    self.update_analyzing();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn update_analyzing(&mut self) {
        self.analyzing = self.rx_a.is_some()
            || self.rx_b.is_some()
            || self.susp_rx_a.is_some()
            || self.susp_rx_b.is_some()
            || self.tire_rx_a.is_some()
            || self.tire_rx_b.is_some();
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

        // Stats popup windows (show for whichever tab is active)
        match self.active_tab {
            ActiveTab::DriverConsistency => {
                if let Some(result) = &self.result_a {
                    self.stats_window.show(ctx, result);
                }
            }
            ActiveTab::Suspension => {
                if let Some(result) = &self.susp_result_a {
                    self.susp_stats.show(ctx, result);
                }
            }
            ActiveTab::TireGrip => {
                if let Some(result) = &self.tire_result_a {
                    self.tire_stats.show(ctx, result);
                }
            }
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
                        let has_stats = match self.active_tab {
                            ActiveTab::DriverConsistency => self.result_a.is_some(),
                            ActiveTab::Suspension => self.susp_result_a.is_some(),
                            ActiveTab::TireGrip => self.tire_result_a.is_some(),
                        };
                        if has_stats && ui.button("Statistics").clicked() {
                            match self.active_tab {
                                ActiveTab::DriverConsistency => {
                                    self.stats_window.open = !self.stats_window.open;
                                }
                                ActiveTab::Suspension => {
                                    self.susp_stats.open = !self.susp_stats.open;
                                }
                                ActiveTab::TireGrip => {
                                    self.tire_stats.open = !self.tire_stats.open;
                                }
                            }
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

        // Top panel — tab selector + session loading + config (collapsible)
        if !self.top_collapsed {
            let panel_resp = egui::TopBottomPanel::top("config")
                .exact_height(220.0)
                .show(ctx, |ui| {
                    // Tab selector bar
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.active_tab,
                            ActiveTab::DriverConsistency,
                            "Driver Consistency",
                        );
                        ui.selectable_value(
                            &mut self.active_tab,
                            ActiveTab::Suspension,
                            "Suspension Velocity",
                        );
                        ui.selectable_value(
                            &mut self.active_tab,
                            ActiveTab::TireGrip,
                            "Tire Grip",
                        );
                    });
                    ui.separator();

                    // Save the 3rd column rect for the config overlay
                    let mut config_col_rect = egui::Rect::NOTHING;
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

                        // 3rd column: just capture its rect
                        config_col_rect = cols[2].available_rect_before_wrap();
                    });

                    config_col_rect
                });

            // Config overlay — foreground Area at the 3rd column position,
            // can extend below the top panel over the chart area
            let col_rect = panel_resp.inner;
            egui::Area::new(egui::Id::new("config_overlay"))
                .fixed_pos(col_rect.left_top())
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(ui.visuals().panel_fill)
                        .inner_margin(4.0)
                        .show(ui, |ui| {
                            ui.set_width(col_rect.width() - 8.0);
                            match self.active_tab {
                                ActiveTab::DriverConsistency => {
                                    config_changed = self.config.show(ui);
                                    save_profile = self.config.save_requested;
                                }
                                ActiveTab::Suspension => {
                                    config_changed = self.susp_config.show(ui);
                                    save_profile = self.susp_config.save_requested;
                                }
                                ActiveTab::TireGrip => {
                                    config_changed = self.tire_config.show(ui);
                                    save_profile = self.tire_config.save_requested;
                                }
                            }
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
                    self.susp_config.set_from_profile(&prof);
                    self.tire_config.set_from_profile(&prof);
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
                    motion_ratios: self.susp_config.motion_ratios.clone(),
                };
                match profile::save_profile_for_logger(&lid, &lid, &prof) {
                    Ok(()) => self.status = "Profile saved".into(),
                    Err(e) => self.status = format!("Save failed: {e}"),
                }
            }
        }

        // Left sidebar — only shown for Driver Consistency tab
        if self.active_tab == ActiveTab::DriverConsistency {
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
                        if charts::track_map::draw_track_map_thumbnail(ui, result, self.corner_selector.hovered_corner) {
                            self.corner_selector.view_mode = ViewMode::TrackMap;
                        }
                    }
                });
        }

        // Central chart area
        egui::CentralPanel::default().show(ctx, |ui| match self.active_tab {
            ActiveTab::DriverConsistency => self.draw_driver_tab(ui),
            ActiveTab::Suspension => self.draw_suspension_tab(ui),
            ActiveTab::TireGrip => self.draw_tire_grip_tab(ui),
        });
    }
}

impl InfernoApp {
    fn draw_driver_tab(&mut self, ui: &mut egui::Ui) {
        match &self.corner_selector.view_mode {
            ViewMode::Summary => {
                if let Some(result) = &self.result_a {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let h = if let Some(rb) = &self.result_b {
                            charts::summary::draw_summary_comparison(
                                ui,
                                &result.corner_data,
                                &rb.corner_data,
                            )
                        } else {
                            charts::summary::draw_summary(ui, &result.corner_data)
                        };
                        self.corner_selector.hovered_corner = h;
                    });
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Load a telemetry file to begin analysis");
                    });
                }
            }
            ViewMode::Detail(idx) => {
                self.corner_selector.hovered_corner = None;
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
                self.corner_selector.hovered_corner = None;
                if let Some(result) = &self.result_a {
                    charts::track_map::draw_track_map(ui, result);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Load a telemetry file to begin analysis");
                    });
                }
            }
        }
    }

    fn draw_suspension_tab(&self, ui: &mut egui::Ui) {
        if let Some(result) = &self.susp_result_a {
            if let Some(rb) = &self.susp_result_b {
                charts::histogram::draw_histograms_comparison(ui, result, rb);
            } else {
                charts::histogram::draw_histograms(ui, result);
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.heading("Load a telemetry file to begin analysis");
            });
        }
    }

    fn draw_tire_grip_tab(&self, ui: &mut egui::Ui) {
        if let Some(result) = &self.tire_result_a {
            if let Some(rb) = &self.tire_result_b {
                charts::grip::draw_grip_comparison(ui, result, rb);
            } else {
                charts::grip::draw_grip(ui, result);
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.heading("Load a telemetry file to begin analysis");
            });
        }
    }
}
