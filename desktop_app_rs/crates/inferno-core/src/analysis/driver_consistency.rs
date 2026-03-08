use rayon::prelude::*;

use super::corners::{self, Corner};
use super::math;
use super::throttle_acceptance;
use super::zones::{self, SegmentType, TrackSegment};
use crate::channel;
use crate::error::{Error, Result};
use crate::lap::Lap;
use crate::session::{LapData, Session};

/// Per-lap trace data for visualization overlays.
#[derive(Debug, Clone)]
pub struct LapTraceData {
    pub lap_num: i32,
    pub distance: Vec<f64>,
    pub throttle: Vec<f64>,
    pub brake: Vec<f64>,
    pub lateral_g: Vec<f64>,
}

/// Per-corner aggregated consistency data.
#[derive(Debug, Clone)]
pub struct CornerConsistencyData {
    pub corner: Corner,
    pub ta_values: Vec<f64>,
    pub ta_mean: f64,
    pub ta_std: f64,
    pub bp_values: Vec<f64>,
    pub bp_std: f64,
    pub speed_values: Vec<f64>,
    pub speed_mean: f64,
    pub speed_std: f64,
    pub exit_speed_values: Vec<f64>,
    pub exit_speed_mean: f64,
    pub exit_speed_std: f64,
    pub accel_zone_length: f64,
    pub opportunity_score: f64,
    pub lap_traces: Vec<LapTraceData>,
    pub braking_start: Option<f64>,
}

/// Full driver consistency analysis result.
pub struct DriverConsistencyResult {
    pub corners: Vec<Corner>,
    pub corner_data: Vec<CornerConsistencyData>,
    pub segments: Vec<TrackSegment>,
    pub ref_lat: Vec<f64>,
    pub ref_lon: Vec<f64>,
    pub ref_distance: Vec<f64>,
}

/// Channel name mapping for analysis.
pub struct ChannelConfig {
    pub throttle: String,
    pub brake: String,
    pub lateral_g: String,
    pub gps_lat: String,
    pub gps_lon: String,
    pub gps_speed: String,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            throttle: "PPS".into(),
            brake: "BrakePress".into(),
            lateral_g: "LateralAcc".into(),
            gps_lat: "GPS Latitude".into(),
            gps_lon: "GPS Longitude".into(),
            gps_speed: "GPS Speed".into(),
        }
    }
}

/// Run the full driver consistency analysis pipeline.
pub fn analyze_driver_consistency(
    session: &Session,
    selected_laps: &[i32],
    config: &ChannelConfig,
    corner_threshold: f64,
    throttle_threshold: f64,
    sustain_time_ms: f64,
) -> Result<DriverConsistencyResult> {
    // Find selected laps
    let laps: Vec<&Lap> = session
        .laps
        .iter()
        .filter(|l| selected_laps.contains(&l.num))
        .collect();

    if laps.is_empty() {
        return Err(Error::NoValidLaps);
    }

    // Get best lap for GPS reference
    let best_lap = laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .min_by_key(|l| l.duration_ms())
        .ok_or(Error::NoValidLaps)?;

    // Extract GPS data from best lap
    let channel_names = [
        config.gps_lat.as_str(),
        config.gps_lon.as_str(),
        config.gps_speed.as_str(),
        config.throttle.as_str(),
        config.brake.as_str(),
        config.lateral_g.as_str(),
        "speed_kmh",
    ];

    let ref_data = LapData::extract(session, best_lap, &channel_names)?;
    let ref_lat = ref_data
        .get(&config.gps_lat)
        .ok_or_else(|| Error::MissingChannel(config.gps_lat.clone()))?;
    let ref_lon = ref_data
        .get(&config.gps_lon)
        .ok_or_else(|| Error::MissingChannel(config.gps_lon.clone()))?;

    // Filter valid GPS
    let valid_mask: Vec<bool> = ref_lat
        .iter()
        .zip(ref_lon.iter())
        .map(|(&lat, &lon)| lat != 0.0 || lon != 0.0)
        .collect();

    let valid_lat: Vec<f64> = ref_lat
        .iter()
        .zip(&valid_mask)
        .filter(|(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();
    let valid_lon: Vec<f64> = ref_lon
        .iter()
        .zip(&valid_mask)
        .filter(|(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();

    // Identify corners
    let corners = corners::identify_corners(&valid_lat, &valid_lon, corner_threshold);
    if corners.is_empty() {
        return Err(Error::Other("No corners detected".into()));
    }

    // Extract per-lap data (parallelized)
    let lap_data: Vec<LapData> = laps
        .par_iter()
        .filter_map(|lap| LapData::extract(session, lap, &channel_names).ok())
        .collect();

    if lap_data.is_empty() {
        return Err(Error::NoValidLaps);
    }

    // Check which channels are available (brake/throttle are optional for GPS-only loggers)
    if !lap_data.iter().any(|ld| ld.get("speed_kmh").is_some()) {
        let available: Vec<&str> = session.channel_names();
        return Err(Error::MissingChannel(format!(
            "speed_kmh (derived) not found. Available: {}",
            available.join(", ")
        )));
    }

    // Collect arrays for zone detection
    let distances: Vec<Vec<f64>> = lap_data.iter().map(|ld| ld.dist().to_vec()).collect();
    let brake_presses: Vec<Vec<f64>> = lap_data
        .iter()
        .map(|ld| ld.get(&config.brake).unwrap_or(&[]).to_vec())
        .collect();
    let throttles: Vec<Vec<f64>> = lap_data
        .iter()
        .map(|ld| ld.get(&config.throttle).unwrap_or(&[]).to_vec())
        .collect();
    let speeds: Vec<Vec<f64>> = lap_data
        .iter()
        .map(|ld| ld.get("speed_kmh").unwrap_or(&[]).to_vec())
        .collect();

    // Detect zones
    let (braking_zones, accel_zones) = zones::detect_zones_from_arrays(
        &distances,
        &brake_presses,
        &throttles,
        &speeds,
        1.0,
        0.5,
        1.5,
        None,
        None,
    );

    // Create segments
    let track_length = distances
        .iter()
        .filter_map(|d| d.last().copied())
        .fold(0.0f64, f64::max);

    let segments =
        zones::create_track_segments(&corners, &braking_zones, &accel_zones, track_length);

    // Compute segment stats
    let dist_refs: Vec<&[f64]> = distances.iter().map(|d| d.as_slice()).collect();
    let speed_refs: Vec<&[f64]> = speeds.iter().map(|s| s.as_slice()).collect();
    let brake_refs: Vec<&[f64]> = brake_presses.iter().map(|b| b.as_slice()).collect();
    let lap_nums: Vec<i32> = lap_data.iter().map(|ld| ld.lap_num).collect();

    // Auto-detect brake threshold for stats
    let brake_threshold = brake_presses
        .first()
        .map(|b| {
            let max_brake = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            0.05 * max_brake
        })
        .unwrap_or(5.0);

    let segment_stats = zones::compute_segment_stats_from_arrays(
        &dist_refs,
        &speed_refs,
        &brake_refs,
        &lap_nums,
        &segments,
        brake_threshold,
    );

    // Build per-corner consistency data
    let trace_margin = 50.0;
    let mut corner_data = Vec::with_capacity(corners.len());

    for corner in &corners {
        // Collect braking point stats
        let bp_values: Vec<f64> = segment_stats
            .iter()
            .filter(|s| s.corner_id == corner.id && s.segment_type == SegmentType::Braking)
            .filter_map(|s| s.braking_point)
            .collect();
        let bp_std = math::std_dev(&bp_values);

        // Collect speed stats
        let speed_values: Vec<f64> = segment_stats
            .iter()
            .filter(|s| s.corner_id == corner.id && s.segment_type == SegmentType::Corner)
            .filter_map(|s| s.min_speed)
            .collect();
        let speed_mean = math::mean(&speed_values);
        let speed_std = math::std_dev(&speed_values);

        // Collect exit speed stats
        let exit_speed_values: Vec<f64> = segment_stats
            .iter()
            .filter(|s| s.corner_id == corner.id && s.segment_type == SegmentType::Corner)
            .filter_map(|s| s.exit_speed)
            .collect();
        let exit_speed_mean = math::mean(&exit_speed_values);
        let exit_speed_std = math::std_dev(&exit_speed_values);

        // Acceleration zone length
        let accel_zone_length = segments
            .iter()
            .find(|s| s.corner_id == corner.id && s.segment_type == SegmentType::Acceleration)
            .map(|s| s.length())
            .unwrap_or(0.0);

        let opportunity_score = exit_speed_std * accel_zone_length;

        // Braking start distance
        let braking_start = segments
            .iter()
            .find(|s| s.corner_id == corner.id && s.segment_type == SegmentType::Braking)
            .map(|s| s.start_dist);

        // Extract lap traces for this corner
        let trace_start = braking_start
            .unwrap_or(corner.start_dist)
            .min(corner.start_dist)
            - trace_margin;
        let trace_end = corner.end_dist + accel_zone_length + trace_margin;

        let mut lap_traces = Vec::new();
        let mut ta_values = Vec::new();

        for ld in &lap_data {
            let dist = ld.dist();
            let si = math::searchsorted(dist, trace_start);
            let ei = math::searchsorted_right(dist, trace_end).min(dist.len());

            if si >= ei {
                continue;
            }

            lap_traces.push(LapTraceData {
                lap_num: ld.lap_num,
                distance: dist[si..ei].to_vec(),
                throttle: ld
                    .get(&config.throttle)
                    .map(|a| a[si..ei].to_vec())
                    .unwrap_or_default(),
                brake: ld
                    .get(&config.brake)
                    .map(|a| a[si..ei].to_vec())
                    .unwrap_or_default(),
                lateral_g: ld
                    .get(&config.lateral_g)
                    .map(|a| a[si..ei].to_vec())
                    .unwrap_or_default(),
            });

            // Compute throttle acceptance for this corner+lap
            if let (Some(throttle_arr), Some(lat_g_arr)) =
                (ld.get(&config.throttle), ld.get(&config.lateral_g))
            {
                let timecodes_batch = session.channels.get("distance_m").and_then(|b| {
                    channel::filter_by_lap(
                        b,
                        session.laps.iter().find(|l| l.num == ld.lap_num).unwrap(),
                    )
                    .ok()
                });

                if let Some(tc_batch) = timecodes_batch {
                    if let Ok(tc) = channel::get_timecodes(&tc_batch) {
                        let tc_slice: Vec<i64> = tc.values().to_vec();

                        let (smoothed, eff_threshold) =
                            throttle_acceptance::prepare_throttle_acceptance(
                                throttle_arr,
                                lat_g_arr,
                                Some(throttle_threshold),
                                25,
                            );

                        if let Some(ta) = throttle_acceptance::find_throttle_acceptance(
                            dist,
                            &tc_slice,
                            throttle_arr,
                            &smoothed,
                            eff_threshold,
                            corner,
                            sustain_time_ms,
                        ) {
                            ta_values.push(ta.throttle_acceptance_pct);
                        }
                    }
                }
            }
        }

        let ta_mean = math::mean(&ta_values);
        let ta_std = math::std_dev(&ta_values);

        corner_data.push(CornerConsistencyData {
            corner: corner.clone(),
            ta_values,
            ta_mean,
            ta_std,
            bp_values,
            bp_std,
            speed_values,
            speed_mean,
            speed_std,
            exit_speed_values,
            exit_speed_mean,
            exit_speed_std,
            accel_zone_length,
            opportunity_score,
            lap_traces,
            braking_start,
        });
    }

    Ok(DriverConsistencyResult {
        corners,
        corner_data,
        segments,
        ref_lat: valid_lat,
        ref_lon: valid_lon,
        ref_distance: ref_data.dist().to_vec(),
    })
}
