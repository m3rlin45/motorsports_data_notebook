use egui::{RichText, Ui};
use egui_plot::{Line, MarkerShape, Plot, PlotBounds, PlotPoints, Points};

use inferno_core::analysis::tire_grip::{MetricMode, TireGripResult, WheelGripData};

use crate::theme;

/// Compute unified axis bounds across all wheels (and optionally a second result).
fn compute_bounds(
    result: &TireGripResult,
    result_b: Option<&TireGripResult>,
) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    let mut update = |w: &WheelGripData| {
        for &x in &w.bucket_centers {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
        }
        for &y in &w.bucket_values {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    };

    for w in &result.wheels {
        update(w);
    }
    if let Some(rb) = result_b {
        for w in &rb.wheels {
            update(w);
        }
    }

    // Add 5% padding
    let x_pad = (x_max - x_min).abs() * 0.05;
    let y_pad = (y_max - y_min).abs() * 0.05;
    (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
}

/// Draw 2x2 tire grip scatter/line plots (single session).
pub fn draw_grip(ui: &mut Ui, result: &TireGripResult) {
    let [ref fl, ref fr, ref rl, ref rr] = result.wheels;
    let x_label = metric_label(result);
    let y_label = accel_label(result);
    let bounds = compute_bounds(result, None);

    let avail = ui.available_size();
    let cell_w = avail.x / 2.0;
    let cell_h = avail.y / 2.0;

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, fl, &x_label, &y_label, None, bounds);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, fr, &x_label, &y_label, None, bounds);
        });
    });
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, rl, &x_label, &y_label, None, bounds);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, rr, &x_label, &y_label, None, bounds);
        });
    });
}

/// Draw 2x2 tire grip scatter/line plots (A/B comparison).
pub fn draw_grip_comparison(ui: &mut Ui, result_a: &TireGripResult, result_b: &TireGripResult) {
    let x_label = metric_label(result_a);
    let y_label = accel_label(result_a);
    let bounds = compute_bounds(result_a, Some(result_b));

    let avail = ui.available_size();
    let cell_w = avail.x / 2.0;
    let cell_h = avail.y / 2.0;

    let pairs: [(_, _); 4] = std::array::from_fn(|i| (&result_a.wheels[i], &result_b.wheels[i]));

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, pairs[0].0, &x_label, &y_label, Some(pairs[0].1), bounds);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, pairs[1].0, &x_label, &y_label, Some(pairs[1].1), bounds);
        });
    });
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, pairs[2].0, &x_label, &y_label, Some(pairs[2].1), bounds);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_grip(ui, pairs[3].0, &x_label, &y_label, Some(pairs[3].1), bounds);
        });
    });
}

fn draw_single_grip(
    ui: &mut Ui,
    data: &WheelGripData,
    x_label: &str,
    y_label: &str,
    data_b: Option<&WheelGripData>,
    bounds: (f64, f64, f64, f64),
) {
    let n: u64 = data.bucket_counts.iter().sum();

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(&data.name)
                .strong()
                .size(14.0)
                .color(theme::STEELBLUE),
        );
        ui.label(
            RichText::new(format!("n={n}  mean G: {:.3}", data.mean_g))
                .small()
                .color(theme::TEXT_SECONDARY),
        );
        if let Some(b) = data_b {
            let nb: u64 = b.bucket_counts.iter().sum();
            ui.label(
                RichText::new(format!("| B n={nb}  mean G: {:.3}", b.mean_g))
                    .small()
                    .color(theme::DARKORANGE),
            );
        }
    });

    let plot_id = if data_b.is_some() {
        format!("grip_cmp_{}", data.name)
    } else {
        format!("grip_{}", data.name)
    };

    let (x_min, x_max, y_min, y_max) = bounds;

    Plot::new(plot_id)
        .allow_drag(true)
        .allow_zoom(true)
        .allow_scroll(true)
        .x_axis_label(x_label)
        .y_axis_label(y_label)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max)
        .show(ui, |plot_ui| {
            // Force the initial view to the shared bounds
            plot_ui.set_plot_bounds(PlotBounds::from_min_max([x_min, y_min], [x_max, y_max]));

            // Session A
            if !data.bucket_centers.is_empty() {
                let pts: Vec<[f64; 2]> = data
                    .bucket_centers
                    .iter()
                    .zip(data.bucket_values.iter())
                    .map(|(&x, &y)| [x, y])
                    .collect();

                let label = if data_b.is_some() {
                    "Session A"
                } else {
                    &data.name
                };

                plot_ui.line(
                    Line::new(label, PlotPoints::new(pts.clone()))
                        .color(theme::STEELBLUE)
                        .width(2.0),
                );
                plot_ui.points(
                    Points::new(format!("{label}_pts"), PlotPoints::new(pts))
                        .color(theme::STEELBLUE)
                        .radius(4.0)
                        .shape(MarkerShape::Circle),
                );
            }

            // Session B
            if let Some(b) = data_b {
                if !b.bucket_centers.is_empty() {
                    let pts: Vec<[f64; 2]> = b
                        .bucket_centers
                        .iter()
                        .zip(b.bucket_values.iter())
                        .map(|(&x, &y)| [x, y])
                        .collect();

                    plot_ui.line(
                        Line::new("Session B", PlotPoints::new(pts.clone()))
                            .color(theme::DARKORANGE)
                            .width(2.0),
                    );
                    plot_ui.points(
                        Points::new("Session B_pts", PlotPoints::new(pts))
                            .color(theme::DARKORANGE)
                            .radius(4.0)
                            .shape(MarkerShape::Circle),
                    );
                }
            }
        });
}

fn metric_label(result: &TireGripResult) -> String {
    let mode = match result.metric_mode {
        MetricMode::Pressure => "Pressure",
        MetricMode::Temperature => "Temperature",
    };
    if result.metric_unit.is_empty() {
        mode.to_string()
    } else {
        format!("{mode} ({0})", result.metric_unit)
    }
}

fn accel_label(result: &TireGripResult) -> String {
    if result.accel_unit.is_empty() {
        "Total Accel".to_string()
    } else {
        format!("Total Accel ({0})", result.accel_unit)
    }
}
