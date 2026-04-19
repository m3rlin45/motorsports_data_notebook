use super::math;
use crate::channel;
use crate::error::{Error, Result};
use crate::profile::MotionRatios;
use crate::session::Session;

/// Velocity range thresholds for categorizing suspension motion (mm/s, absolute).
#[derive(Debug, Clone)]
pub struct VelocityRanges {
    /// Velocities below this are in the friction/static range.
    pub friction: f64,
    /// Velocities below this (above friction) are slow.
    pub slow: f64,
    /// Velocities below this (above slow) are fast. Above this is curb/high-speed.
    pub fast: f64,
}

impl Default for VelocityRanges {
    fn default() -> Self {
        Self {
            friction: 5.0,
            slow: 25.0,
            fast: 200.0,
        }
    }
}

/// Channel names and analysis parameters for suspension analysis.
pub struct SuspensionConfig {
    pub shock_fl: String,
    pub shock_fr: String,
    pub shock_rl: String,
    pub shock_rr: String,
    pub motion_ratios: MotionRatios,
    pub velocity_ranges: VelocityRanges,
    pub smoothing_window: usize,
    pub bin_size: f64,
    pub max_velocity: f64,
}

impl Default for SuspensionConfig {
    fn default() -> Self {
        Self {
            shock_fl: "LF_Shock_Pot".into(),
            shock_fr: "RF_Shock_Pot".into(),
            shock_rl: "LR_Shock_Pot".into(),
            shock_rr: "RR_Shock_Pot".into(),
            motion_ratios: MotionRatios::default(),
            velocity_ranges: VelocityRanges::default(),
            smoothing_window: 5,
            bin_size: 10.0,
            max_velocity: 300.0,
        }
    }
}

/// Velocity histogram and statistics for a single wheel.
#[derive(Debug, Clone)]
pub struct WheelVelocityData {
    pub name: String,
    pub bin_centers: Vec<f64>,
    pub histogram: Vec<f64>,
    pub skew: f64,
    pub kurtosis: f64,
    pub mean: f64,
    pub std: f64,
    pub pct_friction: f64,
    pub pct_slow_bump: f64,
    pub pct_slow_rebound: f64,
    pub pct_fast_bump: f64,
    pub pct_fast_rebound: f64,
    pub pct_curb: f64,
}

/// Complete suspension velocity analysis result for all four wheels.
#[derive(Debug, Clone)]
pub struct SuspensionResult {
    /// [FL, FR, RL, RR]
    pub wheels: [WheelVelocityData; 4],
    pub velocity_ranges: VelocityRanges,
}

/// Compute zero-centered velocity histogram.
///
/// Bins are arranged so zero is at the center of a bin:
/// edges at [-max-half, ..., -half, half, ..., max+half], centers at [..., -bin, 0, bin, ...]
fn compute_velocity_histogram(
    velocity: &[f64],
    bin_size: f64,
    max_velocity: f64,
) -> (Vec<f64>, Vec<f64>) {
    let half_bin = bin_size / 2.0;

    // Build positive edges: half_bin, half_bin+bin_size, ...
    let mut positive_edges = Vec::new();
    let mut edge = half_bin;
    while edge <= max_velocity + bin_size {
        positive_edges.push(edge);
        edge += bin_size;
    }

    // Negative edges are the reverse negation of positive
    let mut bin_edges: Vec<f64> = positive_edges.iter().rev().map(|&e| -e).collect();
    bin_edges.extend_from_slice(&positive_edges);

    // Compute bin centers
    let bin_centers: Vec<f64> = bin_edges.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    let n_bins = bin_centers.len();
    let mut counts = vec![0u64; n_bins];
    let total = velocity.len() as f64;

    for &v in velocity {
        let clamped = v.clamp(-max_velocity, max_velocity);
        // Binary search for the right bin
        let idx = bin_edges.partition_point(|&e| e <= clamped);
        // idx is the first edge > clamped; the bin index is idx-1, clamped to valid range
        let bin = if idx == 0 {
            0
        } else if idx > n_bins {
            n_bins - 1
        } else {
            idx - 1
        };
        counts[bin] += 1;
    }

    let histogram: Vec<f64> = if total > 0.0 {
        counts.iter().map(|&c| c as f64 / total * 100.0).collect()
    } else {
        vec![0.0; n_bins]
    };

    (histogram, bin_centers)
}

/// Compute velocity range percentages.
fn compute_velocity_range_pcts(velocity: &[f64], ranges: &VelocityRanges) -> [f64; 6] {
    let n = velocity.len() as f64;
    if n == 0.0 {
        return [0.0; 6];
    }

    let (mut friction, mut slow_bump, mut slow_rebound, mut fast_bump, mut fast_rebound, mut curb) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);

    for &v in velocity {
        let abs_v = v.abs();
        if abs_v < ranges.friction {
            friction += 1;
        } else if abs_v < ranges.slow {
            if v > 0.0 {
                slow_bump += 1;
            } else {
                slow_rebound += 1;
            }
        } else if abs_v < ranges.fast {
            if v > 0.0 {
                fast_bump += 1;
            } else {
                fast_rebound += 1;
            }
        } else {
            curb += 1;
        }
    }

    [
        friction as f64 / n * 100.0,
        slow_bump as f64 / n * 100.0,
        slow_rebound as f64 / n * 100.0,
        fast_bump as f64 / n * 100.0,
        fast_rebound as f64 / n * 100.0,
        curb as f64 / n * 100.0,
    ]
}

/// Build WheelVelocityData from a velocity array (mm/s).
fn build_wheel_data(
    velocity: &[f64],
    name: &str,
    bin_size: f64,
    max_velocity: f64,
    ranges: &VelocityRanges,
) -> WheelVelocityData {
    let (histogram, bin_centers) = compute_velocity_histogram(velocity, bin_size, max_velocity);
    let pcts = compute_velocity_range_pcts(velocity, ranges);

    WheelVelocityData {
        name: name.to_string(),
        bin_centers,
        histogram,
        skew: math::skewness(velocity),
        kurtosis: math::kurtosis(velocity),
        mean: math::mean(velocity),
        std: math::std_dev(velocity),
        pct_friction: pcts[0],
        pct_slow_bump: pcts[1],
        pct_slow_rebound: pcts[2],
        pct_fast_bump: pcts[3],
        pct_fast_rebound: pcts[4],
        pct_curb: pcts[5],
    }
}

/// Compute shock velocity from displacement and timecodes.
///
/// Uses central-difference gradient with time in seconds, then optional rolling-mean smoothing.
/// Returns velocity in mm/s (same units as displacement per second).
fn compute_shock_velocity(
    displacement: &[f64],
    timecodes_ms: &[i64],
    smoothing_window: usize,
) -> Vec<f64> {
    // Convert timecodes to seconds
    let time_s: Vec<f64> = timecodes_ms.iter().map(|&t| t as f64 / 1000.0).collect();

    let mut velocity = math::gradient(displacement, &time_s);

    if smoothing_window > 1 && velocity.len() >= smoothing_window {
        velocity = math::rolling_mean(&velocity, smoothing_window);
    }

    velocity
}

/// Run suspension velocity analysis across multiple laps.
///
/// For each lap, extracts shock pot channels, computes velocity (time-derivative
/// of displacement), applies motion ratios, then concatenates across laps before
/// computing histograms. This gives correct time-weighted distributions.
pub fn analyze_suspension_velocity(
    session: &Session,
    selected_laps: &[i32],
    config: &SuspensionConfig,
) -> Result<SuspensionResult> {
    if selected_laps.is_empty() {
        return Err(Error::Other("No laps selected".into()));
    }

    let shock_names = [
        &config.shock_fl,
        &config.shock_fr,
        &config.shock_rl,
        &config.shock_rr,
    ];
    let wheel_labels = ["FL", "FR", "RL", "RR"];
    let mr_values = [
        config.motion_ratios.front_left,
        config.motion_ratios.front_right,
        config.motion_ratios.rear_left,
        config.motion_ratios.rear_right,
    ];

    // Validate that at least the FL shock channel exists
    if !session.channels.contains_key(shock_names[0].as_str()) {
        return Err(Error::MissingChannel(shock_names[0].clone()));
    }

    // Detect if channels are velocity (iRacing: units contain "/s") or displacement (AIM)
    let is_velocity = session
        .channels
        .get(shock_names[0].as_str())
        .and_then(|batch| {
            batch
                .schema()
                .field(1)
                .metadata()
                .get("units")
                .map(|u| u.contains("/s"))
        })
        .unwrap_or(false);

    // Collect velocity data (mm/s) per wheel across all laps
    let mut all_velocities: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];

    for &lap_num in selected_laps {
        let lap = match session.laps.iter().find(|l| l.num == lap_num) {
            Some(l) => l,
            None => continue,
        };

        // Get FL channel filtered to this lap — use its timecodes as reference
        let fl_batch = match session.channels.get(shock_names[0].as_str()) {
            Some(b) => b,
            None => continue,
        };
        let fl_lap = match channel::filter_by_lap(fl_batch, lap) {
            Ok(b) if b.num_rows() > 1 => b,
            _ => continue,
        };
        let target_tc = match channel::get_timecodes(&fl_lap) {
            Ok(tc) => tc.clone(),
            Err(_) => continue,
        };

        for (i, name) in shock_names.iter().enumerate() {
            let batch = match session.channels.get(name.as_str()) {
                Some(b) => b,
                None => continue,
            };
            let lap_batch = match channel::filter_by_lap(batch, lap) {
                Ok(b) if b.num_rows() > 1 => b,
                _ => continue,
            };

            let values = match channel::resample_to_timecodes(&lap_batch, &target_tc) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let vals = values.values();

            if is_velocity {
                // Already velocity (m/s), convert to mm/s
                all_velocities[i].extend(vals.iter().map(|&v| v * 1000.0));
            } else {
                // Displacement (mm): derive velocity, apply motion ratio
                let tc_slice = target_tc.values();
                let shock_vel = compute_shock_velocity(vals, tc_slice, config.smoothing_window);
                let mr = mr_values[i];
                all_velocities[i].extend(shock_vel.iter().map(|&v| v / mr));
            }
        }
    }

    // Build per-wheel results from concatenated velocity data
    let wheels = std::array::from_fn(|i| {
        build_wheel_data(
            &all_velocities[i],
            wheel_labels[i],
            config.bin_size,
            config.max_velocity,
            &config.velocity_ranges,
        )
    });

    Ok(SuspensionResult {
        wheels,
        velocity_ranges: config.velocity_ranges.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_velocity_histogram_symmetric() {
        // Simple symmetric data around zero
        let vel: Vec<f64> = (-50..=50).map(|x| x as f64).collect();
        let (hist, centers) = compute_velocity_histogram(&vel, 10.0, 300.0);

        assert_eq!(hist.len(), centers.len());
        // Sum should be approximately 100%
        let sum: f64 = hist.iter().sum();
        assert!((sum - 100.0).abs() < 0.1);
        // Centers should include 0
        assert!(centers.iter().any(|&c| c.abs() < 5.1));
    }

    #[test]
    fn test_compute_velocity_histogram_zero_centered() {
        // With bin_size=10, we expect centers at ..., -5, 5, 15, ...
        let vel = vec![0.0; 100];
        let (hist, centers) = compute_velocity_histogram(&vel, 10.0, 300.0);
        // All data at 0 should be in the center bin
        let max_idx = hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(centers[max_idx].abs() < 5.1); // center bin near 0
        assert!((hist[max_idx] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_velocity_range_pcts() {
        let ranges = VelocityRanges {
            friction: 5.0,
            slow: 25.0,
            fast: 200.0,
        };
        // 100 values: 20 friction, 20 slow bump, 20 slow rebound, 20 fast bump, 10 fast rebound, 10 curb
        let mut vel = vec![];
        vel.extend(std::iter::repeat_n(2.0, 20)); // friction
        vel.extend(std::iter::repeat_n(15.0, 20)); // slow bump
        vel.extend(std::iter::repeat_n(-15.0, 20)); // slow rebound
        vel.extend(std::iter::repeat_n(100.0, 20)); // fast bump
        vel.extend(std::iter::repeat_n(-100.0, 10)); // fast rebound
        vel.extend(std::iter::repeat_n(250.0, 10)); // curb

        let pcts = compute_velocity_range_pcts(&vel, &ranges);
        assert!((pcts[0] - 20.0).abs() < 0.01); // friction
        assert!((pcts[1] - 20.0).abs() < 0.01); // slow bump
        assert!((pcts[2] - 20.0).abs() < 0.01); // slow rebound
        assert!((pcts[3] - 20.0).abs() < 0.01); // fast bump
        assert!((pcts[4] - 10.0).abs() < 0.01); // fast rebound
        assert!((pcts[5] - 10.0).abs() < 0.01); // curb
    }

    #[test]
    fn test_build_wheel_data() {
        let vel: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) * 0.1).collect();
        let data = build_wheel_data(&vel, "FL", 5.0, 300.0, &VelocityRanges::default());
        assert_eq!(data.name, "FL");
        assert!(!data.histogram.is_empty());
        assert!(!data.bin_centers.is_empty());
        // Mean should be near 0 for symmetric data
        assert!(data.mean.abs() < 0.1);
        // Skew should be near 0 for symmetric data
        assert!(data.skew.abs() < 0.1);
    }

    #[test]
    fn test_compute_shock_velocity() {
        // Linear displacement: 0, 1, 2, 3, 4 mm over 0, 100, 200, 300, 400 ms
        // Expected velocity: 10 mm/s constant
        let disp = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tc = vec![0i64, 100, 200, 300, 400];
        let vel = compute_shock_velocity(&disp, &tc, 1); // no smoothing
        for v in &vel {
            assert!((*v - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_compute_shock_velocity_smoothed() {
        // Noisy signal: smoothing should reduce variance
        let disp: Vec<f64> = (0..100).map(|i| i as f64 + (i % 3) as f64 * 0.5).collect();
        let tc: Vec<i64> = (0..100).map(|i| i * 10).collect();
        let vel_raw = compute_shock_velocity(&disp, &tc, 1);
        let vel_smooth = compute_shock_velocity(&disp, &tc, 5);
        // Smoothed should have lower variance
        let std_raw = math::std_dev(&vel_raw);
        let std_smooth = math::std_dev(&vel_smooth);
        assert!(std_smooth < std_raw);
    }

    #[test]
    fn test_velocity_range_pcts_empty() {
        let pcts = compute_velocity_range_pcts(&[], &VelocityRanges::default());
        assert_eq!(pcts, [0.0; 6]);
    }
}
