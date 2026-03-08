use egui::{Stroke, Ui};
use egui_plot::{BoxElem, BoxPlot, BoxSpread, GridInput, GridMark, Plot, PlotPoints, Polygon};

use inferno_core::analysis::driver_consistency::CornerConsistencyData;

use super::colors;

/// Stroke for box outlines, whiskers, and median line (dark for visibility on light backgrounds).
const BOX_STROKE: fn() -> Stroke = || Stroke::new(1.5, egui::Color32::from_rgb(0x22, 0x22, 0x22));
const BOX_STROKE_A: fn() -> Stroke = || Stroke::new(1.5, egui::Color32::from_rgb(0x20, 0x40, 0x80));
const BOX_STROKE_B: fn() -> Stroke = || Stroke::new(1.5, egui::Color32::from_rgb(0x80, 0x40, 0x00));

/// Semi-transparent gold for opportunity highlight bands.
const OPP_FILL: fn() -> egui::Color32 = || egui::Color32::from_rgba_unmultiplied(255, 200, 50, 30);

/// Compute box spread statistics from a slice of values.
fn compute_box_spread(values: &[f64]) -> Option<BoxSpread> {
    if values.is_empty() {
        return None;
    }

    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let median = percentile_sorted(&sorted, 0.5);
    let q1 = percentile_sorted(&sorted, 0.25);
    let q3 = percentile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;

    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let lower_whisker = sorted
        .iter()
        .find(|&&v| v >= lower_fence)
        .copied()
        .unwrap_or(sorted[0]);
    let upper_whisker = sorted
        .iter()
        .rev()
        .find(|&&v| v <= upper_fence)
        .copied()
        .unwrap_or(sorted[n - 1]);

    Some(BoxSpread::new(lower_whisker, q1, median, q3, upper_whisker))
}

/// Linear interpolation percentile on a sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Find the indices of the top-N corners by opportunity score.
fn top_opportunity_indices(corner_data: &[CornerConsistencyData], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = corner_data
        .iter()
        .enumerate()
        .map(|(i, cd)| (i, cd.opportunity_score))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(n).map(|(i, _)| i).collect()
}

/// Build corner name list for X-axis labels.
fn corner_names(corner_data: &[CornerConsistencyData]) -> Vec<String> {
    corner_data
        .iter()
        .map(|cd| cd.corner.name.clone())
        .collect()
}

/// X-axis formatter closure: shows corner names at integer positions.
fn x_formatter(
    names: &[String],
) -> impl Fn(GridMark, &std::ops::RangeInclusive<f64>) -> String + '_ {
    move |mark, _range| {
        let idx = mark.value.round() as usize;
        if idx >= 1 && idx <= names.len() && (mark.value - idx as f64).abs() < 0.01 {
            names[idx - 1].clone()
        } else {
            String::new()
        }
    }
}

/// X-axis grid spacer: one grid mark per corner position (1, 2, 3, ...).
fn corner_spacer(n: usize) -> impl Fn(GridInput) -> Vec<GridMark> {
    move |input| {
        let start = (input.bounds.0.ceil() as usize).max(1);
        let end = (input.bounds.1.floor() as usize).min(n);
        (start..=end)
            .map(|i| GridMark {
                value: i as f64,
                step_size: 1.0,
            })
            .collect()
    }
}

/// Configure a summary plot with corner-name X-axis and consistent style.
fn summary_plot<'a>(id: &str, names: &'a [String]) -> Plot<'a> {
    Plot::new(id)
        .height(200.0)
        .allow_drag(true)
        .allow_zoom(true)
        .show_axes([true, true])
        .x_axis_formatter(x_formatter(names))
        .x_grid_spacer(corner_spacer(names.len()))
}

/// Compute the Y range (min whisker, max whisker) from a set of BoxSpreads.
fn y_range(spreads: &[Option<BoxSpread>]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for s in spreads.iter().flatten() {
        lo = lo.min(s.lower_whisker);
        hi = hi.max(s.upper_whisker);
    }
    if lo.is_infinite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

/// Draw semi-transparent gold highlight bands at the top-3 opportunity corners.
/// Draw BEFORE box plots so bands appear behind the boxes.
fn draw_opportunity_bands(plot_ui: &mut egui_plot::PlotUi, top3: &[usize], y_min: f64, y_max: f64) {
    for (rank, &idx) in top3.iter().enumerate() {
        let x = (idx + 1) as f64;
        let half = 0.45;
        let pts: PlotPoints = vec![
            [x - half, y_min],
            [x + half, y_min],
            [x + half, y_max],
            [x - half, y_max],
        ]
        .into();
        plot_ui.polygon(
            Polygon::new(format!("Opp #{}", rank + 1), pts)
                .fill_color(OPP_FILL())
                .stroke(Stroke::NONE),
        );
    }
}

/// Draw the 3-stacked summary box plots: Braking Points, Throttle Acceptance, Exit Speed.
pub fn draw_summary(ui: &mut Ui, corner_data: &[CornerConsistencyData]) {
    if corner_data.is_empty() {
        ui.label("No corner data available");
        return;
    }

    let top3 = top_opportunity_indices(corner_data, 3);
    let names = corner_names(corner_data);

    let max_opp = corner_data
        .iter()
        .map(|cd| cd.opportunity_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_opp = corner_data
        .iter()
        .map(|cd| cd.opportunity_score)
        .fold(f64::INFINITY, f64::min);
    let opp_range = (max_opp - min_opp).max(1e-6);

    // --- Braking Points (centered around mean) ---
    ui.label("Braking Points (\u{0394} from mean, m)");
    {
        let spreads: Vec<Option<BoxSpread>> = corner_data
            .iter()
            .map(|cd| {
                let m = inferno_core::analysis::math::mean(&cd.bp_values);
                let centered: Vec<f64> = cd.bp_values.iter().map(|v| v - m).collect();
                compute_box_spread(&centered)
            })
            .collect();
        let (y_lo, y_hi) = y_range(&spreads);

        summary_plot("summary_bp", &names).show(ui, |plot_ui| {
            draw_opportunity_bands(plot_ui, &top3, y_lo, y_hi);

            let mut elems = Vec::new();
            for (i, (cd, spread)) in corner_data.iter().zip(spreads.iter()).enumerate() {
                if let Some(spread) = spread {
                    elems.push(
                        BoxElem::new((i + 1) as f64, spread.clone())
                            .name(&cd.corner.name)
                            .fill(colors::DARKORANGE)
                            .stroke(BOX_STROKE())
                            .box_width(0.5)
                            .whisker_width(0.3),
                    );
                }
            }
            if !elems.is_empty() {
                plot_ui.box_plot(BoxPlot::new("BP", elems));
            }
        });
    }

    // --- Throttle Acceptance ---
    ui.label("Throttle Acceptance (%)");
    {
        let spreads: Vec<Option<BoxSpread>> = corner_data
            .iter()
            .map(|cd| compute_box_spread(&cd.ta_values))
            .collect();
        let (y_lo, y_hi) = y_range(&spreads);

        summary_plot("summary_ta", &names).show(ui, |plot_ui| {
            draw_opportunity_bands(plot_ui, &top3, y_lo, y_hi);

            let mut elems = Vec::new();
            for (i, (cd, spread)) in corner_data.iter().zip(spreads.iter()).enumerate() {
                if let Some(spread) = spread {
                    elems.push(
                        BoxElem::new((i + 1) as f64, spread.clone())
                            .name(&cd.corner.name)
                            .fill(colors::STEELBLUE)
                            .stroke(BOX_STROKE())
                            .box_width(0.5)
                            .whisker_width(0.3),
                    );
                }
            }
            if !elems.is_empty() {
                plot_ui.box_plot(BoxPlot::new("TA%", elems));
            }
        });
    }

    // --- Exit Speed ---
    ui.label("Exit Speed (km/h)");
    {
        let spreads: Vec<Option<BoxSpread>> = corner_data
            .iter()
            .map(|cd| compute_box_spread(&cd.exit_speed_values))
            .collect();
        let (y_lo, y_hi) = y_range(&spreads);

        summary_plot("summary_exit_speed", &names).show(ui, |plot_ui| {
            draw_opportunity_bands(plot_ui, &top3, y_lo, y_hi);

            let mut elems = Vec::new();
            for (i, (cd, spread)) in corner_data.iter().zip(spreads.iter()).enumerate() {
                if let Some(spread) = spread {
                    let t = (cd.opportunity_score - min_opp) / opp_range;
                    let color = colors::opportunity_gradient(t);
                    elems.push(
                        BoxElem::new((i + 1) as f64, spread.clone())
                            .name(&cd.corner.name)
                            .fill(color)
                            .stroke(BOX_STROKE())
                            .box_width(0.5)
                            .whisker_width(0.3),
                    );
                }
            }
            if !elems.is_empty() {
                plot_ui.box_plot(BoxPlot::new("Exit Speed", elems));
            }
        });
    }
}

/// Draw the summary with comparison mode (A vs B side-by-side boxes).
pub fn draw_summary_comparison(
    ui: &mut Ui,
    corner_data_a: &[CornerConsistencyData],
    corner_data_b: &[CornerConsistencyData],
) {
    if corner_data_a.is_empty() && corner_data_b.is_empty() {
        ui.label("No corner data available");
        return;
    }

    let n = corner_data_a.len().max(corner_data_b.len());
    let offset = 0.2;

    let label_data = if !corner_data_a.is_empty() {
        corner_data_a
    } else {
        corner_data_b
    };
    let names = corner_names(label_data);

    // --- Braking Points ---
    ui.label("Braking Points (\u{0394} from mean, m)");
    summary_plot("summary_bp_cmp", &names).show(ui, |plot_ui| {
        let mut elems_a = Vec::new();
        let mut elems_b = Vec::new();

        for i in 0..n {
            let x = (i + 1) as f64;
            if let Some(cd) = corner_data_a.get(i) {
                let m = inferno_core::analysis::math::mean(&cd.bp_values);
                let centered: Vec<f64> = cd.bp_values.iter().map(|v| v - m).collect();
                if let Some(spread) = compute_box_spread(&centered) {
                    elems_a.push(
                        BoxElem::new(x - offset, spread)
                            .name(format!("{} (A)", cd.corner.name))
                            .fill(colors::STEELBLUE)
                            .stroke(BOX_STROKE_A())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
            if let Some(cd) = corner_data_b.get(i) {
                let m = inferno_core::analysis::math::mean(&cd.bp_values);
                let centered: Vec<f64> = cd.bp_values.iter().map(|v| v - m).collect();
                if let Some(spread) = compute_box_spread(&centered) {
                    elems_b.push(
                        BoxElem::new(x + offset, spread)
                            .name(format!("{} (B)", cd.corner.name))
                            .fill(colors::DARKORANGE)
                            .stroke(BOX_STROKE_B())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
        }

        if !elems_a.is_empty() {
            plot_ui.box_plot(BoxPlot::new("A", elems_a));
        }
        if !elems_b.is_empty() {
            plot_ui.box_plot(BoxPlot::new("B", elems_b));
        }
    });

    // --- Throttle Acceptance ---
    ui.label("Throttle Acceptance (%)");
    summary_plot("summary_ta_cmp", &names).show(ui, |plot_ui| {
        let mut elems_a = Vec::new();
        let mut elems_b = Vec::new();

        for i in 0..n {
            let x = (i + 1) as f64;
            if let Some(cd) = corner_data_a.get(i) {
                if let Some(spread) = compute_box_spread(&cd.ta_values) {
                    elems_a.push(
                        BoxElem::new(x - offset, spread)
                            .name(format!("{} (A)", cd.corner.name))
                            .fill(colors::STEELBLUE)
                            .stroke(BOX_STROKE_A())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
            if let Some(cd) = corner_data_b.get(i) {
                if let Some(spread) = compute_box_spread(&cd.ta_values) {
                    elems_b.push(
                        BoxElem::new(x + offset, spread)
                            .name(format!("{} (B)", cd.corner.name))
                            .fill(colors::DARKORANGE)
                            .stroke(BOX_STROKE_B())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
        }

        if !elems_a.is_empty() {
            plot_ui.box_plot(BoxPlot::new("A", elems_a));
        }
        if !elems_b.is_empty() {
            plot_ui.box_plot(BoxPlot::new("B", elems_b));
        }
    });

    // --- Exit Speed ---
    ui.label("Exit Speed (km/h)");
    summary_plot("summary_exit_cmp", &names).show(ui, |plot_ui| {
        let mut elems_a = Vec::new();
        let mut elems_b = Vec::new();

        for i in 0..n {
            let x = (i + 1) as f64;
            if let Some(cd) = corner_data_a.get(i) {
                if let Some(spread) = compute_box_spread(&cd.exit_speed_values) {
                    elems_a.push(
                        BoxElem::new(x - offset, spread)
                            .name(format!("{} (A)", cd.corner.name))
                            .fill(colors::STEELBLUE)
                            .stroke(BOX_STROKE_A())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
            if let Some(cd) = corner_data_b.get(i) {
                if let Some(spread) = compute_box_spread(&cd.exit_speed_values) {
                    elems_b.push(
                        BoxElem::new(x + offset, spread)
                            .name(format!("{} (B)", cd.corner.name))
                            .fill(colors::DARKORANGE)
                            .stroke(BOX_STROKE_B())
                            .box_width(0.35)
                            .whisker_width(0.2),
                    );
                }
            }
        }

        if !elems_a.is_empty() {
            plot_ui.box_plot(BoxPlot::new("A", elems_a));
        }
        if !elems_b.is_empty() {
            plot_ui.box_plot(BoxPlot::new("B", elems_b));
        }
    });
}
