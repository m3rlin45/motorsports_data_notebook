use super::math;

const EARTH_RADIUS_M: f64 = 6371000.0;
const DEFAULT_POS_SMOOTH: usize = 15;
const DEFAULT_CURV_SMOOTH: usize = 30;
const DEFAULT_MIN_CORNER_LENGTH: f64 = 15.0;
const DEFAULT_MIN_GAP: f64 = 80.0;

/// A detected corner on the track.
#[derive(Debug, Clone)]
pub struct Corner {
    pub id: usize,
    pub name: String,
    pub direction: char, // 'L' or 'R'
    pub start_idx: usize,
    pub end_idx: usize,
    pub start_dist: f64,
    pub end_dist: f64,
    pub apex_idx: usize,
    pub apex_dist: f64,
    pub max_curvature: f64,
}

impl Corner {
    pub fn length(&self) -> f64 {
        self.end_dist - self.start_dist
    }

    pub fn radius(&self) -> f64 {
        if self.max_curvature > 1e-6 {
            1.0 / self.max_curvature
        } else {
            10000.0
        }
    }
}

/// Convert GPS lat/lon to local XY using equirectangular projection.
pub fn gps_to_local_xy(lat: &[f64], lon: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = lat.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let lat_rad: Vec<f64> = lat.iter().map(|&l| l.to_radians()).collect();
    let lon_rad: Vec<f64> = lon.iter().map(|&l| l.to_radians()).collect();

    let lat0 = math::mean(&lat_rad);
    let lon0 = math::mean(&lon_rad);

    let cos_lat0 = lat0.cos();
    let x: Vec<f64> = lon_rad
        .iter()
        .map(|&lo| EARTH_RADIUS_M * (lo - lon0) * cos_lat0)
        .collect();
    let y: Vec<f64> = lat_rad
        .iter()
        .map(|&la| EARTH_RADIUS_M * (la - lat0))
        .collect();

    (x, y)
}

/// Compute curvature from XY coordinates with smoothing.
/// Returns (abs_curvature, signed_curvature).
pub fn compute_curvature(
    x: &[f64],
    y: &[f64],
    pos_smooth_window: usize,
    curv_smooth_window: usize,
) -> (Vec<f64>, Vec<f64>) {
    let x_smooth = math::rolling_mean(x, pos_smooth_window);
    let y_smooth = math::rolling_mean(y, pos_smooth_window);

    let dx = math::gradient_uniform(&x_smooth);
    let dy = math::gradient_uniform(&y_smooth);
    let ddx = math::gradient_uniform(&dx);
    let ddy = math::gradient_uniform(&dy);

    let n = x.len();
    let mut signed_curvature = Vec::with_capacity(n);

    for i in 0..n {
        let numerator = dx[i] * ddy[i] - dy[i] * ddx[i];
        let denom = (dx[i] * dx[i] + dy[i] * dy[i]).powf(1.5);
        if denom.abs() > 1e-10 {
            signed_curvature.push(numerator / denom);
        } else {
            signed_curvature.push(0.0);
        }
    }

    let signed_curvature = math::rolling_mean(&signed_curvature, curv_smooth_window);
    let abs_curvature: Vec<f64> = signed_curvature.iter().map(|&c| c.abs()).collect();

    (abs_curvature, signed_curvature)
}

/// Identify corners from GPS lat/lon.
pub fn identify_corners(lat: &[f64], lon: &[f64], threshold: f64) -> Vec<Corner> {
    identify_corners_with_params(
        lat,
        lon,
        threshold,
        DEFAULT_MIN_CORNER_LENGTH,
        DEFAULT_MIN_GAP,
        DEFAULT_POS_SMOOTH,
        DEFAULT_CURV_SMOOTH,
    )
}

pub fn identify_corners_with_params(
    lat: &[f64],
    lon: &[f64],
    threshold: f64,
    min_corner_length: f64,
    min_gap: f64,
    pos_smooth_window: usize,
    curv_smooth_window: usize,
) -> Vec<Corner> {
    // Filter invalid GPS points
    let valid: Vec<usize> = (0..lat.len())
        .filter(|&i| lat[i] != 0.0 || lon[i] != 0.0)
        .collect();

    if valid.len() < 3 {
        return vec![];
    }

    let lat_valid: Vec<f64> = valid.iter().map(|&i| lat[i]).collect();
    let lon_valid: Vec<f64> = valid.iter().map(|&i| lon[i]).collect();

    let (x, y) = gps_to_local_xy(&lat_valid, &lon_valid);
    let distance = math::cumulative_distance(&x, &y);
    let (abs_curv, signed_curv) = compute_curvature(&x, &y, pos_smooth_window, curv_smooth_window);

    identify_corners_from_curvature(
        &distance,
        &abs_curv,
        &signed_curv,
        threshold,
        min_corner_length,
        min_gap,
    )
}

/// Identify corners from pre-computed curvature and distance.
pub fn identify_corners_from_curvature(
    distance: &[f64],
    curvature: &[f64],
    signed_curvature: &[f64],
    threshold: f64,
    min_corner_length: f64,
    min_gap: f64,
) -> Vec<Corner> {
    let n = curvature.len();
    if n == 0 {
        return vec![];
    }

    // Determine direction at each point: +1 = left, -1 = right
    let direction: Vec<i8> = signed_curvature
        .iter()
        .map(|&c| {
            if c > 0.0 {
                1
            } else if c < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect();

    // Find corner segments where curvature > threshold
    let mut raw_corners: Vec<(usize, usize, i8)> = Vec::new(); // (start, end, direction)
    let mut corner_start: Option<usize> = None;
    let mut current_dir: i8 = 0;

    for i in 0..n {
        if curvature[i] > threshold {
            match corner_start {
                None => {
                    corner_start = Some(i);
                    current_dir = direction[i];
                }
                Some(_start) => {
                    // Direction change within a corner → split
                    if direction[i] != 0 && direction[i] != current_dir {
                        raw_corners.push((_start, i, current_dir));
                        corner_start = Some(i);
                        current_dir = direction[i];
                    }
                }
            }
        } else if let Some(start) = corner_start.take() {
            raw_corners.push((start, i, current_dir));
        }
    }
    // Close final corner
    if let Some(start) = corner_start {
        raw_corners.push((start, n - 1, current_dir));
    }

    // Filter by minimum length
    let raw_corners: Vec<(usize, usize, i8)> = raw_corners
        .into_iter()
        .filter(|&(s, e, _)| distance[e] - distance[s] >= min_corner_length)
        .collect();

    // Find apex for each corner
    let corners_with_apex: Vec<(usize, usize, i8, usize, f64)> = raw_corners
        .iter()
        .map(|&(s, e, dir)| {
            let (apex_offset, &max_curv) = curvature[s..=e]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (s, e, dir, s + apex_offset, max_curv)
        })
        .collect();

    // Merge same-direction corners separated by < min_gap
    let mut merged: Vec<(usize, usize, i8, usize, f64)> = Vec::new();
    for corner in corners_with_apex {
        if let Some(last) = merged.last_mut() {
            if last.2 == corner.2 && distance[corner.0] - distance[last.1] < min_gap {
                // Merge: extend end, update apex if new corner has higher curvature
                last.1 = corner.1;
                if corner.4 > last.4 {
                    last.3 = corner.3;
                    last.4 = corner.4;
                }
                continue;
            }
        }
        merged.push(corner);
    }

    // Build Corner structs with 1-indexed IDs
    merged
        .into_iter()
        .enumerate()
        .map(|(idx, (s, e, dir, apex, max_curv))| Corner {
            id: idx + 1,
            name: format!("Turn {}", idx + 1),
            direction: if dir >= 0 { 'L' } else { 'R' },
            start_idx: s,
            end_idx: e,
            start_dist: distance[s],
            end_dist: distance[e],
            apex_idx: apex,
            apex_dist: distance[apex],
            max_curvature: max_curv,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gps_to_local_xy() {
        // Two points ~111km apart in latitude
        let lat = vec![35.0, 36.0];
        let lon = vec![139.0, 139.0];
        let (x, y) = gps_to_local_xy(&lat, &lon);

        assert_eq!(x.len(), 2);
        // x should be ~0 (same longitude)
        assert!(x[0].abs() < 1.0);
        // y difference should be ~111km
        let dy = (y[1] - y[0]).abs();
        assert!(dy > 110_000.0 && dy < 112_000.0);
    }

    #[test]
    fn test_identify_corners_from_curvature() {
        // Simulate a track with two corners
        let n = 200;
        let mut distance = Vec::with_capacity(n);
        let mut curvature = Vec::with_capacity(n);
        let mut signed_curv = Vec::with_capacity(n);

        for i in 0..n {
            distance.push(i as f64 * 5.0); // 5m spacing, 1000m track

            // Corner 1 at 200-300m (left turn)
            // Corner 2 at 600-700m (right turn)
            let d = i as f64 * 5.0;
            if (200.0..300.0).contains(&d) {
                curvature.push(0.01);
                signed_curv.push(0.01);
            } else if (600.0..700.0).contains(&d) {
                curvature.push(0.008);
                signed_curv.push(-0.008);
            } else {
                curvature.push(0.001);
                signed_curv.push(0.001);
            }
        }

        let corners =
            identify_corners_from_curvature(&distance, &curvature, &signed_curv, 0.006, 15.0, 80.0);

        assert_eq!(corners.len(), 2);
        assert_eq!(corners[0].direction, 'L');
        assert_eq!(corners[1].direction, 'R');
        assert_eq!(corners[0].id, 1);
        assert_eq!(corners[1].id, 2);
        assert!(corners[0].start_dist >= 200.0 && corners[0].end_dist <= 300.0);
    }

    #[test]
    fn test_merge_same_direction() {
        // Two left corners separated by < 80m should merge
        let n = 100;
        let mut distance = Vec::with_capacity(n);
        let mut curvature = Vec::with_capacity(n);
        let mut signed_curv = Vec::with_capacity(n);

        for i in 0..n {
            let d = i as f64 * 5.0;
            distance.push(d);

            if (100.0..150.0).contains(&d) || (200.0..250.0).contains(&d) {
                curvature.push(0.01);
                signed_curv.push(0.01);
            } else {
                curvature.push(0.001);
                signed_curv.push(0.001);
            }
        }

        let corners =
            identify_corners_from_curvature(&distance, &curvature, &signed_curv, 0.006, 15.0, 80.0);

        // Gap is 50m < 80m, same direction → should merge to 1 corner
        assert_eq!(corners.len(), 1);
        assert!(corners[0].start_dist >= 100.0);
        assert!(corners[0].end_dist <= 250.0);
    }

    #[test]
    fn test_corner_radius() {
        let corner = Corner {
            id: 1,
            name: "Turn 1".into(),
            direction: 'L',
            start_idx: 0,
            end_idx: 10,
            start_dist: 0.0,
            end_dist: 100.0,
            apex_idx: 5,
            apex_dist: 50.0,
            max_curvature: 0.01,
        };
        assert!((corner.radius() - 100.0).abs() < 1e-10);
        assert!((corner.length() - 100.0).abs() < 1e-10);
    }
}
