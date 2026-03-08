use super::corners::Corner;
use super::math;

/// A track segment (braking zone, corner, or acceleration zone).
#[derive(Debug, Clone)]
pub struct TrackSegment {
    pub id: usize,
    pub segment_type: SegmentType,
    pub start_dist: f64,
    pub end_dist: f64,
    pub name: String,
    pub corner_id: usize,
    pub apex_dist: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    Braking,
    Corner,
    Acceleration,
}

impl TrackSegment {
    pub fn length(&self) -> f64 {
        self.end_dist - self.start_dist
    }
}

/// A zone is a (start_dist, end_dist) range.
pub type Zone = (f64, f64);

/// Detect braking and acceleration zones for a single lap.
pub fn identify_zones_single_lap(
    distance: &[f64],
    brake_press: &[f64],
    throttle: &[f64],
    speed: &[f64],
    brake_threshold: f64,
    throttle_threshold: f64,
    gear_change_time: f64,
) -> (Vec<Zone>, Vec<Zone>) {
    let n = distance.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Find braking zones
    let braking_zones = find_zones(distance, &|i| brake_press[i] > brake_threshold);

    // Find acceleration zones (throttle above threshold AND brake below threshold)
    let is_accel: Vec<bool> = (0..n)
        .map(|i| throttle[i] > throttle_threshold && brake_press[i] <= brake_threshold)
        .collect();
    let raw_accel_zones = find_zones(distance, &|i| is_accel[i]);

    // Merge acceleration zones separated by short time gaps (gear changes)
    let accel_zones = merge_accel_zones_by_time(
        &raw_accel_zones,
        distance,
        speed,
        &braking_zones,
        gear_change_time,
    );

    (braking_zones, accel_zones)
}

/// Find contiguous zones where a predicate is true.
fn find_zones(distance: &[f64], predicate: &dyn Fn(usize) -> bool) -> Vec<Zone> {
    let n = distance.len();
    let mut zones = Vec::new();
    let mut start: Option<usize> = None;

    for i in 0..n {
        if predicate(i) {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            zones.push((distance[s], distance[i - 1]));
        }
    }
    if let Some(s) = start {
        zones.push((distance[s], distance[n - 1]));
    }

    zones
}

/// Merge acceleration zones separated by short time gaps.
fn merge_accel_zones_by_time(
    zones: &[Zone],
    distance: &[f64],
    speed: &[f64],
    braking_zones: &[Zone],
    gear_change_time: f64,
) -> Vec<Zone> {
    if zones.is_empty() {
        return vec![];
    }

    let mut merged = vec![zones[0]];

    for zone in &zones[1..] {
        let prev = merged.last().unwrap();
        let gap_dist = zone.0 - prev.1;

        if gap_dist <= 0.0 {
            // Overlapping — merge
            let last = merged.last_mut().unwrap();
            last.1 = zone.1;
            continue;
        }

        // Estimate gap time from speed
        let gap_start_idx = math::searchsorted(distance, prev.1);
        let gap_end_idx = math::searchsorted(distance, zone.0);

        if gap_start_idx >= gap_end_idx || gap_start_idx >= speed.len() {
            merged.push(*zone);
            continue;
        }

        let end = gap_end_idx.min(speed.len());
        let avg_speed = math::mean(&speed[gap_start_idx..end]);
        let gap_time = if avg_speed > 1.0 {
            gap_dist / avg_speed
        } else {
            f64::MAX
        };

        // Check if there's braking in the gap
        let has_braking = braking_zones
            .iter()
            .any(|bz| bz.0 < zone.0 && bz.1 > prev.1);

        if gap_time <= gear_change_time && !has_braking {
            let last = merged.last_mut().unwrap();
            last.1 = zone.1;
        } else {
            merged.push(*zone);
        }
    }

    merged
}

/// Average zones across multiple laps using grid-based voting.
pub fn average_zones_across_laps(
    all_zones: &[Vec<Zone>],
    track_length: f64,
    resolution: f64,
    threshold: f64,
) -> Vec<Zone> {
    let n_laps = all_zones.len();
    if n_laps == 0 || track_length <= 0.0 {
        return vec![];
    }

    let n_points = (track_length / resolution) as usize + 1;
    let mut counts = vec![0u32; n_points];

    for lap_zones in all_zones {
        for &(start, end) in lap_zones {
            let si = (start / resolution) as usize;
            let ei = ((end / resolution) as usize + 1).min(n_points);
            for count in counts[si..ei].iter_mut() {
                *count += 1;
            }
        }
    }

    let min_count = (n_laps as f64 * threshold).ceil() as u32;

    // Extract zones from grid
    let mut zones = Vec::new();
    let mut start: Option<usize> = None;

    for (i, &count) in counts.iter().enumerate() {
        if count >= min_count {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            zones.push((s as f64 * resolution, (i - 1) as f64 * resolution));
        }
    }
    if let Some(s) = start {
        zones.push((s as f64 * resolution, (n_points - 1) as f64 * resolution));
    }

    zones
}

/// Detect zones from multiple laps of data.
#[allow(clippy::too_many_arguments)]
pub fn detect_zones_from_arrays(
    distances: &[Vec<f64>],
    brake_presses: &[Vec<f64>],
    throttles: &[Vec<f64>],
    speeds: &[Vec<f64>],
    resolution: f64,
    threshold: f64,
    max_gap_time: f64,
    brake_threshold: Option<f64>,
    throttle_threshold: Option<f64>,
) -> (Vec<Zone>, Vec<Zone>) {
    let n_laps = distances.len();
    if n_laps == 0 {
        return (vec![], vec![]);
    }

    // Auto-detect thresholds from first lap
    let bt = brake_threshold.unwrap_or_else(|| {
        let max_brake = brake_presses[0]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        0.05 * max_brake
    });
    let tt = throttle_threshold.unwrap_or_else(|| {
        let max_throttle = throttles[0]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        0.20 * max_throttle
    });

    let mut all_braking = Vec::with_capacity(n_laps);
    let mut all_accel = Vec::with_capacity(n_laps);

    for i in 0..n_laps {
        let (braking, accel) = identify_zones_single_lap(
            &distances[i],
            &brake_presses[i],
            &throttles[i],
            &speeds[i],
            bt,
            tt,
            max_gap_time,
        );
        all_braking.push(braking);
        all_accel.push(accel);
    }

    // Determine track length
    let track_length = distances
        .iter()
        .filter_map(|d| d.last().copied())
        .fold(f64::NEG_INFINITY, f64::max);

    let braking = average_zones_across_laps(&all_braking, track_length, resolution, threshold);
    let accel = average_zones_across_laps(&all_accel, track_length, resolution, threshold);

    (braking, accel)
}

/// Create track segments from corners and zones.
pub fn create_track_segments(
    corners: &[Corner],
    braking_zones: &[Zone],
    accel_zones: &[Zone],
    _track_length: f64,
) -> Vec<TrackSegment> {
    let mut segments = Vec::new();
    let mut seg_id = 1;

    let mut sorted_corners: Vec<&Corner> = corners.iter().collect();
    sorted_corners.sort_by(|a, b| a.start_dist.partial_cmp(&b.start_dist).unwrap());

    for corner in &sorted_corners {
        // Find braking zone that ends near corner start (within 100m)
        let brake_start = braking_zones
            .iter()
            .find(|bz| bz.1 >= corner.start_dist - 100.0 && bz.1 <= corner.start_dist + 50.0)
            .map(|bz| bz.0)
            .unwrap_or_else(|| (corner.start_dist - 100.0).max(0.0));

        // Braking segment
        if brake_start < corner.start_dist {
            segments.push(TrackSegment {
                id: seg_id,
                segment_type: SegmentType::Braking,
                start_dist: brake_start,
                end_dist: corner.start_dist,
                name: format!("{} Brake", corner.name),
                corner_id: corner.id,
                apex_dist: None,
            });
            seg_id += 1;
        }

        // Corner segment
        segments.push(TrackSegment {
            id: seg_id,
            segment_type: SegmentType::Corner,
            start_dist: corner.start_dist,
            end_dist: corner.end_dist,
            name: corner.name.clone(),
            corner_id: corner.id,
            apex_dist: Some(corner.apex_dist),
        });
        seg_id += 1;

        // Find acceleration zone after corner
        let accel_end = accel_zones
            .iter()
            .find(|az| az.0 <= corner.end_dist + 50.0 && az.1 > corner.end_dist)
            .map(|az| az.1)
            .unwrap_or(corner.end_dist + 50.0);

        // Cap at next braking zone start
        let next_brake_start = braking_zones
            .iter()
            .filter(|bz| bz.0 > corner.end_dist)
            .map(|bz| bz.0)
            .reduce(f64::min);

        let accel_end = match next_brake_start {
            Some(nbs) => accel_end.min(nbs),
            None => accel_end,
        };

        if accel_end > corner.end_dist {
            segments.push(TrackSegment {
                id: seg_id,
                segment_type: SegmentType::Acceleration,
                start_dist: corner.end_dist,
                end_dist: accel_end,
                name: format!("{} Accel", corner.name),
                corner_id: corner.id,
                apex_dist: None,
            });
            seg_id += 1;
        }
    }

    segments.sort_by(|a, b| a.start_dist.partial_cmp(&b.start_dist).unwrap());
    segments
}

/// Per-lap statistics for a segment.
#[derive(Debug, Clone)]
pub struct SegmentStat {
    pub segment_id: usize,
    pub segment_name: String,
    pub segment_type: SegmentType,
    pub corner_id: usize,
    pub lap_num: i32,
    pub braking_point: Option<f64>,
    pub min_speed: Option<f64>,
    pub exit_speed: Option<f64>,
}

/// Compute per-segment statistics for each lap.
pub fn compute_segment_stats_from_arrays(
    distances: &[&[f64]],
    speeds: &[&[f64]],
    brakes: &[&[f64]],
    lap_nums: &[i32],
    segments: &[TrackSegment],
    brake_threshold: f64,
) -> Vec<SegmentStat> {
    let mut results = Vec::new();

    for (lap_idx, &lap_num) in lap_nums.iter().enumerate() {
        let dist = distances[lap_idx];
        let speed = speeds[lap_idx];
        let brake = brakes[lap_idx];

        for seg in segments {
            let si = math::searchsorted(dist, seg.start_dist);
            let ei = math::searchsorted_right(dist, seg.end_dist).min(dist.len());

            if si >= ei {
                continue;
            }

            let mut stat = SegmentStat {
                segment_id: seg.id,
                segment_name: seg.name.clone(),
                segment_type: seg.segment_type,
                corner_id: seg.corner_id,
                lap_num,
                braking_point: None,
                min_speed: None,
                exit_speed: None,
            };

            match seg.segment_type {
                SegmentType::Braking => {
                    // First point where brake > threshold
                    for j in si..ei {
                        if brake[j] > brake_threshold {
                            stat.braking_point = Some(dist[j]);
                            break;
                        }
                    }
                }
                SegmentType::Corner => {
                    // Minimum speed in segment
                    stat.min_speed = speed[si..ei].iter().cloned().reduce(f64::min);
                    // Exit speed: speed at segment end
                    if ei > si {
                        stat.exit_speed = Some(speed[ei - 1]);
                    }
                }
                SegmentType::Acceleration => {
                    // Nothing specific for acceleration segments in stats
                }
            }

            results.push(stat);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_zones() {
        let distance = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0];
        let values = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
        let zones = find_zones(&distance, &|i| values[i] > 1.0);
        assert_eq!(zones.len(), 1);
        assert!((zones[0].0 - 20.0).abs() < 1e-10);
        assert!((zones[0].1 - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_zones() {
        let all_zones = vec![
            vec![(100.0, 200.0), (500.0, 600.0)],
            vec![(100.0, 200.0), (500.0, 600.0)],
            vec![(100.0, 200.0)], // third lap missing second zone
        ];

        let averaged = average_zones_across_laps(&all_zones, 1000.0, 1.0, 0.5);
        // First zone: present in all 3 laps (3/3 >= 0.5) → included
        assert!(!averaged.is_empty());
        // Second zone: present in 2/3 laps (0.67 >= 0.5) → included
        assert!(averaged.len() >= 2);
    }

    #[test]
    fn test_create_track_segments() {
        let corners = vec![Corner {
            id: 1,
            name: "Turn 1".into(),
            direction: 'L',
            start_idx: 10,
            end_idx: 20,
            start_dist: 200.0,
            end_dist: 300.0,
            apex_idx: 15,
            apex_dist: 250.0,
            max_curvature: 0.01,
        }];
        let braking = vec![(150.0, 200.0)];
        let accel = vec![(280.0, 500.0)];

        let segments = create_track_segments(&corners, &braking, &accel, 1000.0);

        assert!(segments.len() >= 2); // at least corner + something
        let corner_seg = segments
            .iter()
            .find(|s| s.segment_type == SegmentType::Corner);
        assert!(corner_seg.is_some());
        assert_eq!(corner_seg.unwrap().apex_dist, Some(250.0));
    }
}
