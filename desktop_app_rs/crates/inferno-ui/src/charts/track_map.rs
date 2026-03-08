use egui::Ui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoint, PlotPoints, Points, Text};

use inferno_core::analysis::corners::gps_to_local_xy;
use inferno_core::analysis::driver_consistency::{CornerConsistencyData, DriverConsistencyResult};
use inferno_core::analysis::math::searchsorted;
use inferno_core::analysis::zones::SegmentType;

use super::colors;

/// Draw the track map with segment coloring, apex markers, and opportunity stars.
pub fn draw_track_map(ui: &mut Ui, result: &DriverConsistencyResult) {
    if result.ref_lat.is_empty() || result.ref_lon.is_empty() {
        ui.label("No GPS data available");
        return;
    }

    let (x, y) = gps_to_local_xy(&result.ref_lat, &result.ref_lon);
    let ref_dist = &result.ref_distance;

    // Find top-3 opportunity corners
    let top3 = top_opportunity_indices(&result.corner_data, 3);

    // Compute opportunity score range for linewidth scaling
    let max_opp = result
        .corner_data
        .iter()
        .map(|cd| cd.opportunity_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_opp = result
        .corner_data
        .iter()
        .map(|cd| cd.opportunity_score)
        .fold(f64::INFINITY, f64::min);
    let opp_range = (max_opp - min_opp).max(1e-6);

    Plot::new("track_map")
        .data_aspect(1.0)
        .allow_drag(true)
        .allow_zoom(true)
        .show_axes([false, false])
        .cursor_color(egui::Color32::TRANSPARENT)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            // Base track line (gray)
            let base_pts: Vec<[f64; 2]> =
                x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();
            plot_ui.line(
                Line::new("Track", PlotPoints::new(base_pts))
                    .color(colors::segment::TRACK_BASE)
                    .width(3.0),
            );

            // Draw each segment with its color
            for seg in &result.segments {
                let si = searchsorted(ref_dist, seg.start_dist);
                let ei = searchsorted(ref_dist, seg.end_dist).min(x.len());
                if si >= ei || si >= x.len() {
                    continue;
                }

                let seg_pts: Vec<[f64; 2]> = (si..ei).map(|j| [x[j], y[j]]).collect();
                if seg_pts.is_empty() {
                    continue;
                }

                let (color, base_width) = match seg.segment_type {
                    SegmentType::Braking => (colors::segment::BRAKING, 4.0),
                    SegmentType::Corner => (colors::segment::CORNER, 4.0),
                    SegmentType::Acceleration => {
                        let opp = result
                            .corner_data
                            .iter()
                            .find(|cd| cd.corner.id == seg.corner_id)
                            .map(|cd| cd.opportunity_score)
                            .unwrap_or(0.0);
                        let t = (opp - min_opp) / opp_range;
                        let width = 4.0 + t * 6.0; // Scale from 4 to 10
                        (colors::segment::ACCELERATION, width as f32)
                    }
                };

                plot_ui.line(
                    Line::new(&seg.name, PlotPoints::new(seg_pts))
                        .color(color)
                        .width(base_width),
                );
            }

            // Apex markers (dark red circles with corner name labels)
            for corner in &result.corners {
                if corner.apex_idx < x.len() {
                    let ax = x[corner.apex_idx];
                    let ay = y[corner.apex_idx];

                    plot_ui.points(
                        Points::new(&corner.name, PlotPoints::new(vec![[ax, ay]]))
                            .color(colors::segment::APEX_MARKER)
                            .radius(5.0)
                            .shape(MarkerShape::Circle),
                    );

                    plot_ui.text(
                        Text::new(
                            format!("label_{}", corner.id),
                            PlotPoint::new(ax, ay),
                            &corner.name,
                        )
                        .color(egui::Color32::WHITE),
                    );
                }
            }

            // Gold stars on top-3 opportunity corners
            for &idx in &top3 {
                if let Some(cd) = result.corner_data.get(idx) {
                    let apex_idx = cd.corner.apex_idx;
                    if apex_idx < x.len() {
                        plot_ui.points(
                            Points::new(
                                format!("★ {}", cd.corner.name),
                                PlotPoints::new(vec![[x[apex_idx], y[apex_idx]]]),
                            )
                            .color(colors::GOLD)
                            .radius(8.0)
                            .shape(MarkerShape::Diamond),
                        );
                    }
                }
            }
        })
        .response
        .on_hover_cursor(egui::CursorIcon::Default);
}

/// Draw a small clickable track map thumbnail. Returns true if clicked.
pub fn draw_track_map_thumbnail(ui: &mut Ui, result: &DriverConsistencyResult) -> bool {
    if result.ref_lat.is_empty() || result.ref_lon.is_empty() {
        return false;
    }

    let (x, y) = gps_to_local_xy(&result.ref_lat, &result.ref_lon);
    let ref_dist = &result.ref_distance;

    let resp = Plot::new("track_map_thumb")
        .data_aspect(1.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show_x(false)
        .show_y(false)
        .show_axes([false, false])
        .show_grid(false)
        .show(ui, |plot_ui| {
            // Base track line
            let base_pts: Vec<[f64; 2]> =
                x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();
            plot_ui.line(
                Line::new("Track", PlotPoints::new(base_pts))
                    .color(colors::segment::TRACK_BASE)
                    .width(2.0),
            );

            // Segment coloring (simplified — no legend names)
            for seg in &result.segments {
                let si = searchsorted(ref_dist, seg.start_dist);
                let ei = searchsorted(ref_dist, seg.end_dist).min(x.len());
                if si >= ei || si >= x.len() {
                    continue;
                }
                let seg_pts: Vec<[f64; 2]> = (si..ei).map(|j| [x[j], y[j]]).collect();
                if seg_pts.is_empty() {
                    continue;
                }
                let color = match seg.segment_type {
                    SegmentType::Braking => colors::segment::BRAKING,
                    SegmentType::Corner => colors::segment::CORNER,
                    SegmentType::Acceleration => colors::segment::ACCELERATION,
                };
                plot_ui.line(
                    Line::new(&seg.name, PlotPoints::new(seg_pts))
                        .color(color)
                        .width(3.0),
                );
            }
        });

    resp.response.clicked()
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
