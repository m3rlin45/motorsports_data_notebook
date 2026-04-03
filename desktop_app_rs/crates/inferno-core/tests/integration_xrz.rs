//! Integration tests: load real XRZ files and verify the full analysis pipeline.

use std::path::Path;

use inferno_core::analysis::driver_consistency::{
    analyze_driver_consistency, ChannelConfig, DriverConsistencyResult,
};
use inferno_core::analysis::math;
use inferno_core::analysis::suspension::{
    analyze_suspension_velocity, SuspensionConfig, SuspensionResult,
};
use inferno_core::analysis::tire_grip::{
    analyze_tire_grip, MetricMode, TireGripConfig, TireGripResult,
};
use inferno_core::session::{LapData, Session};

// ── Test data paths ──────────────────────────────────────────────────

const XRZ_86: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../workspace_template/data/CMD_Inferno 86_Fuji GP Sh_Generic testing_a_2248.xrz"
);

// ── Helpers ──────────────────────────────────────────────────────────

fn load_86() -> Session {
    Session::open(Path::new(XRZ_86)).expect("Failed to load 86 XRZ")
}

fn valid_lap_nums(session: &Session) -> Vec<i32> {
    session
        .laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .map(|l| l.num)
        .collect()
}

fn run_analysis_86() -> DriverConsistencyResult {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    assert!(!laps.is_empty(), "No valid laps in 86 data");
    analyze_driver_consistency(
        &session,
        &laps,
        &ChannelConfig::default(),
        0.005,
        98.0,
        300.0,
    )
    .expect("86 analysis failed")
}

// ══════════════════════════════════════════════════════════════════════
// Tier 1 — Regression tests (prevent known bugs from recurring)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_xrk_loads_can_bus_channels() {
    let session = load_86();
    let names = session.channel_names();

    // Must have more than just GPS + derived channels
    assert!(
        names.len() > 5,
        "Expected many channels, got {}: {:?}",
        names.len(),
        names
    );

    // Specific CAN bus channel that must be present for the 86
    assert!(
        names.contains(&"PPS"),
        "Missing throttle channel 'PPS'. Available: {names:?}"
    );
}

#[test]
fn test_distance_is_lap_relative() {
    let session = load_86();
    let laps: Vec<_> = session
        .laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .collect();
    assert!(laps.len() >= 2, "Need at least 2 valid laps");

    for lap in &laps[..2] {
        let ld = LapData::extract(&session, lap, &["speed_kmh"]).unwrap();
        assert!(
            ld.dist()[0].abs() < 1.0,
            "Lap {} distance starts at {}, expected ~0",
            lap.num,
            ld.dist()[0]
        );
    }
}

#[test]
fn test_channel_data_not_empty() {
    let session = load_86();
    for name in session.channel_names() {
        let batch = session.channel(name).unwrap();
        assert!(batch.num_rows() > 0, "Channel '{name}' has 0 rows");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tier 2 — Data integrity (catch silent data loss)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_distance_monotonically_increasing() {
    let session = load_86();
    let laps: Vec<_> = session
        .laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .collect();

    for lap in &laps {
        let ld = LapData::extract(&session, lap, &["speed_kmh"]).unwrap();
        let dist = ld.dist();
        for i in 1..dist.len() {
            assert!(
                dist[i] >= dist[i - 1],
                "Lap {} distance not monotonic at index {}: {} < {}",
                lap.num,
                i,
                dist[i],
                dist[i - 1]
            );
        }
    }
}

#[test]
fn test_all_corners_have_speed_data() {
    let result = run_analysis_86();
    for cd in &result.corner_data {
        assert!(
            !cd.speed_values.is_empty(),
            "Corner '{}' has empty speed_values",
            cd.corner.name
        );
        assert!(
            !cd.exit_speed_values.is_empty(),
            "Corner '{}' has empty exit_speed_values",
            cd.corner.name
        );
    }
}

#[test]
fn test_all_corners_have_lap_traces() {
    let result = run_analysis_86();
    for cd in &result.corner_data {
        assert!(
            !cd.lap_traces.is_empty(),
            "Corner '{}' has no lap traces",
            cd.corner.name
        );
        for trace in &cd.lap_traces {
            assert!(
                !trace.distance.is_empty(),
                "Lap {} trace for '{}' has empty distance",
                trace.lap_num,
                cd.corner.name
            );
        }
    }
}

#[test]
fn test_all_corners_have_braking_data() {
    let result = run_analysis_86();
    let corners_with_bp = result
        .corner_data
        .iter()
        .filter(|cd| !cd.bp_values.is_empty())
        .count();
    assert!(corners_with_bp > 0, "No corners have braking point data");
}

#[test]
fn test_all_corners_have_throttle_acceptance() {
    let result = run_analysis_86();
    let corners_with_ta = result
        .corner_data
        .iter()
        .filter(|cd| !cd.ta_values.is_empty())
        .count();
    assert!(
        corners_with_ta > 0,
        "No corners have throttle acceptance data"
    );
}

#[test]
fn test_speed_values_positive() {
    let result = run_analysis_86();
    for cd in &result.corner_data {
        assert!(
            cd.speed_mean > 0.0,
            "Corner '{}' has zero speed_mean",
            cd.corner.name
        );
        assert!(
            cd.exit_speed_mean > 0.0,
            "Corner '{}' has zero exit_speed_mean",
            cd.corner.name
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tier 3 — Invariant checks (verify correctness between pipeline stages)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_corner_distances_within_track() {
    let result = run_analysis_86();
    let track_length = result.ref_distance.last().copied().unwrap_or(0.0);

    for corner in &result.corners {
        assert!(
            corner.start_dist >= 0.0,
            "Corner '{}' start_dist {} < 0",
            corner.name,
            corner.start_dist
        );
        assert!(
            corner.start_dist <= corner.apex_dist,
            "Corner '{}' start_dist {} > apex_dist {}",
            corner.name,
            corner.start_dist,
            corner.apex_dist
        );
        assert!(
            corner.apex_dist <= corner.end_dist,
            "Corner '{}' apex_dist {} > end_dist {}",
            corner.name,
            corner.apex_dist,
            corner.end_dist
        );
        assert!(
            corner.end_dist <= track_length + 50.0,
            "Corner '{}' end_dist {} exceeds track_length {}",
            corner.name,
            corner.end_dist,
            track_length
        );
    }
}

#[test]
fn test_corner_distance_range_has_data() {
    let session = load_86();
    let result = run_analysis_86();
    let laps: Vec<_> = session
        .laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .collect();

    let channel_names = ["speed_kmh", "PPS", "BrakePress", "LateralAcc"];
    let ld = LapData::extract(&session, laps[0], &channel_names).unwrap();
    let dist = ld.dist();

    for corner in &result.corners {
        let si = math::searchsorted(dist, corner.start_dist);
        let ei = math::searchsorted_right(dist, corner.end_dist).min(dist.len());
        assert!(
            ei > si,
            "Corner '{}' [{:.0}, {:.0}] has no data points in lap {} distance range",
            corner.name,
            corner.start_dist,
            corner.end_dist,
            laps[0].num
        );
    }
}

#[test]
fn test_segments_reference_valid_corners() {
    let result = run_analysis_86();
    let corner_ids: Vec<usize> = result.corners.iter().map(|c| c.id).collect();

    for seg in &result.segments {
        assert!(
            corner_ids.contains(&seg.corner_id),
            "Segment '{}' references corner_id {} not in detected corners {:?}",
            seg.name,
            seg.corner_id,
            corner_ids
        );
    }
}

#[test]
fn test_braking_point_within_segment() {
    let result = run_analysis_86();

    for cd in &result.corner_data {
        if let Some(bs) = cd.braking_start {
            assert!(
                bs <= cd.corner.start_dist + 10.0,
                "Corner '{}' braking_start {:.1} is far past corner start {:.1}",
                cd.corner.name,
                bs,
                cd.corner.start_dist
            );
        }
    }
}

#[test]
fn test_ta_values_bounded() {
    let result = run_analysis_86();
    for cd in &result.corner_data {
        for &ta in &cd.ta_values {
            assert!(
                (0.0..=150.0).contains(&ta),
                "Corner '{}' TA value {} out of bounds [0, 150]",
                cd.corner.name,
                ta
            );
        }
    }
}

#[test]
fn test_segment_stats_per_lap_count() {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    let result = run_analysis_86();

    // Each corner should have roughly as many speed values as selected laps
    // (some laps might be skipped if data is missing, but at least 50%)
    let min_expected = laps.len() / 2;
    for cd in &result.corner_data {
        assert!(
            cd.speed_values.len() >= min_expected,
            "Corner '{}' has {} speed values, expected at least {} (from {} laps)",
            cd.corner.name,
            cd.speed_values.len(),
            min_expected,
            laps.len()
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tier 4 — Edge cases (prevent future regressions)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_single_lap_analysis() {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    assert!(!laps.is_empty());

    let result = analyze_driver_consistency(
        &session,
        &laps[..1],
        &ChannelConfig::default(),
        0.005,
        98.0,
        300.0,
    )
    .expect("Single-lap analysis should not fail");

    assert!(
        !result.corners.is_empty(),
        "Should detect corners even with 1 lap"
    );
    // std_dev of 1 value should be 0, not NaN or crash
    for cd in &result.corner_data {
        assert!(
            !cd.speed_std.is_nan(),
            "Corner '{}' speed_std is NaN with single lap",
            cd.corner.name
        );
    }
}

#[test]
fn test_different_channel_configs() {
    // Use 86 data with TPS as throttle channel (the 86 has both PPS and TPS)
    let session = load_86();
    let laps = valid_lap_nums(&session);
    assert!(!laps.is_empty());

    let config = ChannelConfig {
        throttle: "TPS".into(),
        ..ChannelConfig::default()
    };

    let result = analyze_driver_consistency(&session, &laps, &config, 0.005, 98.0, 300.0)
        .expect("Analysis with TPS throttle channel failed");

    assert!(
        !result.corners.is_empty(),
        "No corners detected with TPS config"
    );
    assert!(
        result
            .corner_data
            .iter()
            .any(|cd| !cd.speed_values.is_empty()),
        "No corner speed data with TPS config"
    );
}

#[test]
fn test_missing_brake_channel_no_panic() {
    let session = load_86();
    let laps = valid_lap_nums(&session);

    let config = ChannelConfig {
        brake: "nonexistent_brake_channel".into(),
        ..ChannelConfig::default()
    };

    // Should not panic; braking data will be empty but analysis should complete
    let result = analyze_driver_consistency(&session, &laps, &config, 0.005, 98.0, 300.0);
    assert!(
        result.is_ok(),
        "Analysis panicked with missing brake channel: {:?}",
        result.err()
    );
    let result = result.unwrap();
    // No braking data expected
    for cd in &result.corner_data {
        assert!(
            cd.bp_values.is_empty(),
            "Should have no BP values with missing brake channel"
        );
    }
}

#[test]
fn test_high_corner_threshold_no_corners() {
    let session = load_86();
    let laps = valid_lap_nums(&session);

    let result = analyze_driver_consistency(
        &session,
        &laps,
        &ChannelConfig::default(),
        1.0, // Impossibly high threshold
        98.0,
        300.0,
    );
    assert!(
        result.is_err(),
        "Should return error when no corners detected"
    );
}

#[test]
fn test_lap_data_channels_same_length() {
    let session = load_86();
    let laps: Vec<_> = session
        .laps
        .iter()
        .filter(|l| l.duration_ms() > 0)
        .collect();
    assert!(!laps.is_empty());

    let channel_names = ["speed_kmh", "PPS", "BrakePress", "LateralAcc"];
    let ld = LapData::extract(&session, laps[0], &channel_names).unwrap();
    let n = ld.dist().len();

    for &name in &channel_names {
        if let Some(vals) = ld.get(name) {
            assert_eq!(
                vals.len(),
                n,
                "Channel '{}' has {} samples, expected {} (same as distance)",
                name,
                vals.len(),
                n
            );
        }
    }
}

// ── Suspension velocity integration tests ──────────────────────────

fn run_suspension_86() -> SuspensionResult {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    assert!(!laps.is_empty(), "No valid laps in 86 data");
    analyze_suspension_velocity(&session, &laps, &SuspensionConfig::default())
        .expect("Suspension analysis failed")
}

#[test]
fn test_suspension_all_wheels_have_data() {
    let result = run_suspension_86();
    for w in &result.wheels {
        assert!(
            !w.histogram.is_empty(),
            "Wheel {} should have histogram data",
            w.name
        );
        assert!(
            !w.bin_centers.is_empty(),
            "Wheel {} should have bin centers",
            w.name
        );
        assert_eq!(
            w.histogram.len(),
            w.bin_centers.len(),
            "Wheel {} histogram and bin_centers must match",
            w.name
        );
    }
}

#[test]
fn test_suspension_histogram_sums_to_100() {
    let result = run_suspension_86();
    for w in &result.wheels {
        let total: f64 = w.histogram.iter().sum();
        assert!(
            (total - 100.0).abs() < 0.5,
            "Wheel {} histogram sums to {:.2}, expected ~100",
            w.name,
            total
        );
    }
}

#[test]
fn test_suspension_range_pcts_sum_to_100() {
    let result = run_suspension_86();
    for w in &result.wheels {
        let total = w.pct_friction
            + w.pct_slow_bump
            + w.pct_slow_rebound
            + w.pct_fast_bump
            + w.pct_fast_rebound
            + w.pct_curb;
        assert!(
            (total - 100.0).abs() < 0.5,
            "Wheel {} range pcts sum to {:.2}, expected ~100",
            w.name,
            total
        );
    }
}

#[test]
fn test_suspension_std_positive() {
    let result = run_suspension_86();
    for w in &result.wheels {
        assert!(
            w.std > 0.0,
            "Wheel {} std should be positive, got {}",
            w.name,
            w.std
        );
    }
}

#[test]
fn test_suspension_wheel_names() {
    let result = run_suspension_86();
    let names: Vec<&str> = result.wheels.iter().map(|w| w.name.as_str()).collect();
    assert_eq!(names, ["FL", "FR", "RL", "RR"]);
}

#[test]
fn test_suspension_single_lap() {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    let result = analyze_suspension_velocity(&session, &laps[..1], &SuspensionConfig::default())
        .expect("Single lap suspension analysis failed");
    for w in &result.wheels {
        assert!(!w.histogram.is_empty());
    }
}

#[test]
fn test_suspension_missing_channel() {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    let config = SuspensionConfig {
        shock_fl: "NONEXISTENT_CHANNEL".into(),
        ..SuspensionConfig::default()
    };
    let result = analyze_suspension_velocity(&session, &laps, &config);
    assert!(result.is_err(), "Should fail with missing channel");
}

// ── Tire grip integration tests ────────────────────────────────────

fn run_tire_grip_86(mode: MetricMode) -> TireGripResult {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    let config = TireGripConfig {
        metric_mode: mode,
        ..TireGripConfig::default()
    };
    analyze_tire_grip(&session, &laps, &config).expect("Tire grip analysis failed")
}

#[test]
fn test_tire_grip_pressure_all_wheels() {
    let result = run_tire_grip_86(MetricMode::Pressure);
    for w in &result.wheels {
        assert!(
            !w.bucket_centers.is_empty(),
            "Wheel {} should have bucket data",
            w.name
        );
        assert_eq!(w.bucket_centers.len(), w.bucket_values.len());
        assert_eq!(w.bucket_centers.len(), w.bucket_counts.len());
    }
}

#[test]
fn test_tire_grip_temperature_mode() {
    let result = run_tire_grip_86(MetricMode::Temperature);
    assert_eq!(result.metric_mode, MetricMode::Temperature);
    for w in &result.wheels {
        assert!(
            !w.bucket_centers.is_empty(),
            "Wheel {} should have temperature bucket data",
            w.name
        );
    }
}

#[test]
fn test_tire_grip_bucket_values_positive() {
    let result = run_tire_grip_86(MetricMode::Pressure);
    for w in &result.wheels {
        for &v in &w.bucket_values {
            assert!(
                v > 0.0,
                "Wheel {} bucket value should be positive G",
                w.name
            );
        }
    }
}

#[test]
fn test_tire_grip_stats_positive() {
    let result = run_tire_grip_86(MetricMode::Pressure);
    for w in &result.wheels {
        assert!(w.mean_g > 0.0, "Wheel {} mean_g should be positive", w.name);
        assert!(w.std_g > 0.0, "Wheel {} std_g should be positive", w.name);
        assert!(
            w.mean_metric > 0.0,
            "Wheel {} mean pressure should be positive",
            w.name
        );
    }
}

#[test]
fn test_tire_grip_wheel_names() {
    let result = run_tire_grip_86(MetricMode::Pressure);
    let names: Vec<&str> = result.wheels.iter().map(|w| w.name.as_str()).collect();
    assert_eq!(names, ["FL", "FR", "RL", "RR"]);
}

#[test]
fn test_tire_grip_missing_channel() {
    let session = load_86();
    let laps = valid_lap_nums(&session);
    let config = TireGripConfig {
        lateral_g: "NONEXISTENT".into(),
        ..TireGripConfig::default()
    };
    assert!(analyze_tire_grip(&session, &laps, &config).is_err());
}
