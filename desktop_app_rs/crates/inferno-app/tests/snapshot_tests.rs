use std::path::Path;

use egui_kittest::Harness;
use inferno_app::{ActiveTab, InfernoApp};
use inferno_ui::widgets::corner_selector::ViewMode;

const TEST_DATA: &str =
    "../../../workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz";

fn create_app_with_data() -> Harness<'static, InfernoApp> {
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1300.0, 900.0))
        .build_eframe(|cc| {
            let mut app = InfernoApp::new(cc);
            app.load_and_analyze(Path::new(TEST_DATA));
            app.top_collapsed = true;
            app
        });
    // Run a few frames to let layout settle
    harness.run();
    harness
}

fn create_app_light() -> Harness<'static, InfernoApp> {
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1300.0, 900.0))
        .build_eframe(|cc| {
            let mut app = InfernoApp::new(cc);
            // Override dark theme with light visuals
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            app.load_and_analyze(Path::new(TEST_DATA));
            app.top_collapsed = true;
            app
        });
    harness.run();
    harness
}

// === Dark mode (default) ===

#[test]
fn snapshot_summary_view() {
    let mut harness = create_app_with_data();
    harness.try_snapshot("summary").expect("snapshot failed");
}

#[test]
fn snapshot_detail_view() {
    let mut harness = create_app_with_data();
    harness.state_mut().corner_selector.view_mode = ViewMode::Detail(0);
    harness.run();
    harness
        .try_snapshot("detail_corner0")
        .expect("snapshot failed");
}

#[test]
fn snapshot_track_map() {
    let mut harness = create_app_with_data();
    harness.state_mut().corner_selector.view_mode = ViewMode::TrackMap;
    harness.run();
    harness.try_snapshot("track_map").expect("snapshot failed");
}

// === Light mode ===

#[test]
fn snapshot_summary_light() {
    let mut harness = create_app_light();
    harness
        .try_snapshot("summary_light")
        .expect("snapshot failed");
}

#[test]
fn snapshot_detail_light() {
    let mut harness = create_app_light();
    harness.state_mut().corner_selector.view_mode = ViewMode::Detail(0);
    harness.run();
    harness
        .try_snapshot("detail_corner0_light")
        .expect("snapshot failed");
}

#[test]
fn snapshot_track_map_light() {
    let mut harness = create_app_light();
    harness.state_mut().corner_selector.view_mode = ViewMode::TrackMap;
    harness.run();
    harness
        .try_snapshot("track_map_light")
        .expect("snapshot failed");
}

// === Suspension tab (dark) ===

#[test]
fn snapshot_suspension() {
    let mut harness = create_app_with_data();
    harness.state_mut().active_tab = ActiveTab::Suspension;
    harness.run();
    harness.try_snapshot("suspension").expect("snapshot failed");
}

// === Suspension tab (light) ===

#[test]
fn snapshot_suspension_light() {
    let mut harness = create_app_light();
    harness.state_mut().active_tab = ActiveTab::Suspension;
    harness.run();
    harness
        .try_snapshot("suspension_light")
        .expect("snapshot failed");
}
