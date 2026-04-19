use egui::{Color32, RichText, Ui};
use egui_plot::{Bar, BarChart, Plot, PlotPoint, PlotPoints, Polygon, Text, VLine};

use inferno_core::analysis::suspension::{SuspensionResult, WheelVelocityData};

use crate::theme;

const BUMP_COLOR: Color32 = Color32::from_rgb(70, 130, 180); // steelblue
const REBOUND_COLOR: Color32 = Color32::from_rgb(205, 92, 92); // indianred

fn friction_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(128, 128, 128, 25)
}
fn slow_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(100, 149, 237, 20)
}
fn fast_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(100, 200, 100, 15)
}
fn curb_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(205, 92, 92, 15)
}

/// Draw 2x2 suspension velocity histograms (single session).
pub fn draw_histograms(ui: &mut Ui, result: &SuspensionResult) {
    let [ref fl, ref fr, ref rl, ref rr] = result.wheels;
    let ranges = &result.velocity_ranges;

    let avail = ui.available_size();
    let cell_w = avail.x / 2.0;
    let cell_h = avail.y / 2.0;

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, fl, ranges, None);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, fr, ranges, None);
        });
    });
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, rl, ranges, None);
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, rr, ranges, None);
        });
    });
}

/// Draw 2x2 suspension velocity histograms (A/B comparison).
pub fn draw_histograms_comparison(
    ui: &mut Ui,
    result_a: &SuspensionResult,
    result_b: &SuspensionResult,
) {
    let ranges = &result_a.velocity_ranges;
    let avail = ui.available_size();
    let cell_w = avail.x / 2.0;
    let cell_h = avail.y / 2.0;

    let pairs = [
        (&result_a.wheels[0], &result_b.wheels[0]),
        (&result_a.wheels[1], &result_b.wheels[1]),
        (&result_a.wheels[2], &result_b.wheels[2]),
        (&result_a.wheels[3], &result_b.wheels[3]),
    ];

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, pairs[0].0, ranges, Some(pairs[0].1));
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, pairs[1].0, ranges, Some(pairs[1].1));
        });
    });
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, pairs[2].0, ranges, Some(pairs[2].1));
        });
        ui.vertical(|ui| {
            ui.set_width(cell_w);
            ui.set_height(cell_h);
            draw_single_histogram(ui, pairs[3].0, ranges, Some(pairs[3].1));
        });
    });
}

/// Draw a single wheel histogram. If `data_b` is Some, draws grouped bars for comparison.
fn draw_single_histogram(
    ui: &mut Ui,
    data: &WheelVelocityData,
    ranges: &inferno_core::analysis::suspension::VelocityRanges,
    data_b: Option<&WheelVelocityData>,
) {
    let is_comparison = data_b.is_some();

    // Title with stats
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(&data.name)
                .strong()
                .size(14.0)
                .color(theme::STEELBLUE),
        );
        ui.label(
            RichText::new(format!("skew: {:.2}  std: {:.1} mm/s", data.skew, data.std))
                .small()
                .color(theme::TEXT_SECONDARY),
        );
        if let Some(b) = data_b {
            ui.label(
                RichText::new(format!("| B skew: {:.2}  std: {:.1}", b.skew, b.std))
                    .small()
                    .color(theme::DARKORANGE),
            );
        }
    });

    let plot_id = if is_comparison {
        format!("hist_cmp_{}", data.name)
    } else {
        format!("hist_{}", data.name)
    };

    Plot::new(plot_id)
        .allow_drag(true)
        .allow_zoom(true)
        .allow_scroll(true)
        .x_axis_label("Velocity (mm/s)")
        .y_axis_label("Time (%)")
        .show(ui, |plot_ui| {
            // Background velocity range shading
            draw_range_shading(plot_ui, ranges, &data.name);

            // Zero reference line
            plot_ui.vline(
                VLine::new("zero", 0.0)
                    .color(Color32::from_rgba_unmultiplied(255, 255, 255, 60))
                    .width(1.0),
            );

            // Session A bars
            let bin_width = if data.bin_centers.len() > 1 {
                (data.bin_centers[1] - data.bin_centers[0]).abs()
            } else {
                5.0
            };

            let offset = if is_comparison { bin_width * 0.25 } else { 0.0 };

            let bars_a: Vec<Bar> = data
                .bin_centers
                .iter()
                .zip(data.histogram.iter())
                .map(|(&center, &pct)| {
                    let color = if center >= 0.0 {
                        BUMP_COLOR
                    } else {
                        REBOUND_COLOR
                    };
                    Bar::new(center - offset, pct)
                        .width(if is_comparison {
                            bin_width * 0.45
                        } else {
                            bin_width * 0.9
                        })
                        .fill(color)
                })
                .collect();

            let chart_name = if is_comparison {
                "Session A".to_string()
            } else {
                data.name.clone()
            };
            plot_ui.bar_chart(BarChart::new(chart_name, bars_a).allow_hover(true));

            // Session B bars (comparison mode)
            if let Some(b) = data_b {
                let bars_b: Vec<Bar> = b
                    .bin_centers
                    .iter()
                    .zip(b.histogram.iter())
                    .map(|(&center, &pct)| {
                        Bar::new(center + offset, pct)
                            .width(bin_width * 0.45)
                            .fill(theme::DARKORANGE)
                    })
                    .collect();

                plot_ui.bar_chart(BarChart::new("Session B", bars_b).allow_hover(true));
            }

            // Skew annotation in the plot
            let annotation = format!("Skew: {:.2}", data.skew);
            plot_ui.text(Text::new(
                "skew_label",
                PlotPoint::new(-250.0, plot_max_y(data, data_b) * 0.9),
                annotation,
            ));
        });
}

/// Get a reasonable Y max for annotation placement.
fn plot_max_y(data: &WheelVelocityData, data_b: Option<&WheelVelocityData>) -> f64 {
    let max_a = data.histogram.iter().cloned().fold(0.0f64, f64::max);
    let max_b = data_b
        .map(|b| b.histogram.iter().cloned().fold(0.0f64, f64::max))
        .unwrap_or(0.0);
    max_a.max(max_b).max(1.0)
}

/// Draw semi-transparent velocity range background regions.
fn draw_range_shading(
    plot_ui: &mut egui_plot::PlotUi,
    ranges: &inferno_core::analysis::suspension::VelocityRanges,
    wheel_name: &str,
) {
    let y_hi = 100.0;
    let y_lo = 0.0;

    draw_rect(
        plot_ui,
        &format!("{wheel_name}_friction"),
        -ranges.friction,
        ranges.friction,
        y_lo,
        y_hi,
        friction_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_slow_p"),
        ranges.friction,
        ranges.slow,
        y_lo,
        y_hi,
        slow_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_slow_n"),
        -ranges.slow,
        -ranges.friction,
        y_lo,
        y_hi,
        slow_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_fast_p"),
        ranges.slow,
        ranges.fast,
        y_lo,
        y_hi,
        fast_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_fast_n"),
        -ranges.fast,
        -ranges.slow,
        y_lo,
        y_hi,
        fast_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_curb_p"),
        ranges.fast,
        300.0,
        y_lo,
        y_hi,
        curb_fill(),
    );
    draw_rect(
        plot_ui,
        &format!("{wheel_name}_curb_n"),
        -300.0,
        -ranges.fast,
        y_lo,
        y_hi,
        curb_fill(),
    );
}

/// Draw a filled rectangle in plot space.
fn draw_rect(
    plot_ui: &mut egui_plot::PlotUi,
    name: &str,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    fill: Color32,
) {
    let points = PlotPoints::new(vec![
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]);
    plot_ui.polygon(
        Polygon::new(name, points)
            .fill_color(fill)
            .stroke(egui::Stroke::NONE),
    );
}
