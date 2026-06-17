use egui::Ui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints, Text, VLine};

use inferno_core::analysis::driver_consistency::CornerConsistencyData;

use super::colors;

/// Draw the 3 linked detail line plots for a single corner:
/// Throttle, Brake, Lateral G — with per-lap viridis coloring and VLine markers.
pub fn draw_detail(ui: &mut Ui, corner_data: &CornerConsistencyData) {
    let corner = &corner_data.corner;
    let traces = &corner_data.lap_traces;
    let n_laps = traces.len();

    if n_laps == 0 {
        ui.label("No lap trace data");
        return;
    }

    // Axis/cursor linking IDs
    let link_group = ui.id().with("detail_link");
    let cursor_group = ui.id().with("detail_cursor");

    // VLine marker positions
    let entry_dist = corner.start_dist;
    let apex_dist = corner.apex_dist;
    let exit_dist = corner.end_dist;
    let braking_start = corner_data.braking_start;

    // TA annotation text
    let ta_text = if corner_data.ta_mean > 0.0 {
        format!(
            "TA: {:.1}% ± {:.1}%",
            corner_data.ta_mean, corner_data.ta_std
        )
    } else {
        String::new()
    };

    // --- Throttle ---
    ui.label("Throttle (%)");
    Plot::new(format!("detail_throttle_{}", corner.id))
        .height(120.0)
        .link_axis(link_group, egui::Vec2b::new(true, false))
        .link_cursor(cursor_group, egui::Vec2b::new(true, false))
        .allow_drag(true)
        .allow_zoom(true)
        .cursor_color(egui::Color32::TRANSPARENT)
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            draw_vline_markers(plot_ui, braking_start, entry_dist, apex_dist, exit_dist);

            for (i, trace) in traces.iter().enumerate() {
                let color = colors::viridis_for_lap(i, n_laps);
                let points: Vec<[f64; 2]> = trace
                    .distance
                    .iter()
                    .zip(&trace.throttle)
                    .map(|(&d, &t)| [d, t])
                    .collect();
                let name = format!("Lap {}", trace.lap_num);
                plot_ui.line(
                    Line::new(name, PlotPoints::new(points))
                        .color(color)
                        .width(1.5),
                );
            }

            // TA annotation
            if !ta_text.is_empty() {
                if let Some(d_min) = traces
                    .iter()
                    .flat_map(|t| t.distance.first())
                    .copied()
                    .reduce(f64::min)
                {
                    plot_ui.text(
                        Text::new(
                            "ta_annotation",
                            egui_plot::PlotPoint::new(d_min + 5.0, 95.0),
                            &ta_text,
                        )
                        .color(egui::Color32::DARK_GRAY),
                    );
                }
            }
        })
        .response
        .on_hover_cursor(egui::CursorIcon::Default);

    // --- Brake ---
    ui.label("Brake Pressure");
    Plot::new(format!("detail_brake_{}", corner.id))
        .height(120.0)
        .link_axis(link_group, egui::Vec2b::new(true, false))
        .link_cursor(cursor_group, egui::Vec2b::new(true, false))
        .allow_drag(true)
        .allow_zoom(true)
        .cursor_color(egui::Color32::TRANSPARENT)
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            draw_vline_markers(plot_ui, braking_start, entry_dist, apex_dist, exit_dist);

            for (i, trace) in traces.iter().enumerate() {
                let color = colors::viridis_for_lap(i, n_laps);
                let points: Vec<[f64; 2]> = trace
                    .distance
                    .iter()
                    .zip(&trace.brake)
                    .map(|(&d, &b)| [d, b])
                    .collect();
                let name = format!("Lap {}", trace.lap_num);
                plot_ui.line(
                    Line::new(name, PlotPoints::new(points))
                        .color(color)
                        .width(1.5),
                );
            }
        })
        .response
        .on_hover_cursor(egui::CursorIcon::Default);

    // --- Lateral G ---
    ui.label("Lateral G");
    Plot::new(format!("detail_lat_g_{}", corner.id))
        .height(120.0)
        .link_axis(link_group, egui::Vec2b::new(true, false))
        .link_cursor(cursor_group, egui::Vec2b::new(true, false))
        .allow_drag(true)
        .allow_zoom(true)
        .cursor_color(egui::Color32::TRANSPARENT)
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            draw_vline_markers(plot_ui, braking_start, entry_dist, apex_dist, exit_dist);

            for (i, trace) in traces.iter().enumerate() {
                let color = colors::viridis_for_lap(i, n_laps);
                let points: Vec<[f64; 2]> = trace
                    .distance
                    .iter()
                    .zip(&trace.lateral_g)
                    .map(|(&d, &g)| [d, g])
                    .collect();
                let name = format!("Lap {}", trace.lap_num);
                plot_ui.line(
                    Line::new(name, PlotPoints::new(points))
                        .color(color)
                        .width(1.5),
                );
            }
        })
        .response
        .on_hover_cursor(egui::CursorIcon::Default);
}

/// Draw VLine markers for braking start, corner entry, apex, and corner exit.
fn draw_vline_markers(
    plot_ui: &mut egui_plot::PlotUi,
    braking_start: Option<f64>,
    entry_dist: f64,
    apex_dist: f64,
    exit_dist: f64,
) {
    // Braking start — cyan dotted
    if let Some(bs) = braking_start {
        plot_ui.vline(
            VLine::new("Braking", bs)
                .color(colors::BRAKING_START_COLOR)
                .style(LineStyle::Dotted { spacing: 3.0 })
                .width(1.5),
        );
    }

    // Corner entry — yellow dashed
    plot_ui.vline(
        VLine::new("Entry", entry_dist)
            .color(colors::CORNER_BOUNDARY_COLOR)
            .style(LineStyle::Dashed { length: 5.0 })
            .width(1.5),
    );

    // Apex — red solid
    plot_ui.vline(
        VLine::new("Apex", apex_dist)
            .color(colors::APEX_VLINE_COLOR)
            .width(2.0),
    );

    // Corner exit — yellow dashed
    plot_ui.vline(
        VLine::new("Exit", exit_dist)
            .color(colors::CORNER_BOUNDARY_COLOR)
            .style(LineStyle::Dashed { length: 5.0 })
            .width(1.5),
    );
}
