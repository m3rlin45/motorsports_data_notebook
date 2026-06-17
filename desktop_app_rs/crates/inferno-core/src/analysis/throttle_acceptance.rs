use super::corners::Corner;
use super::math;

const MIN_PEAK_LATERAL_G: f64 = 0.1;

/// Result of throttle acceptance computation for a single corner+lap.
#[derive(Debug, Clone)]
pub struct ThrottleAcceptanceResult {
    pub throttle_acceptance_pct: f64,
    pub lateral_g_at_throttle: f64,
    pub peak_lateral_g: f64,
    pub full_throttle_dist: f64,
}

/// Pre-compute shared data for throttle acceptance across corners.
/// Returns (smoothed_lateral_g, effective_threshold).
pub fn prepare_throttle_acceptance(
    throttle: &[f64],
    lateral_g: &[f64],
    throttle_threshold: Option<f64>,
    smoothing_window: usize,
) -> (Vec<f64>, f64) {
    // Smooth absolute lateral G
    let abs_lat_g: Vec<f64> = lateral_g.iter().map(|&g| g.abs()).collect();
    let smoothed = math::rolling_mean(&abs_lat_g, smoothing_window);

    // Determine effective threshold
    let max_throttle = throttle.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let threshold = throttle_threshold.unwrap_or(0.98 * max_throttle);

    let effective = if (90.0..100.0).contains(&max_throttle) {
        max_throttle * (threshold / 100.0)
    } else {
        threshold
    };

    (smoothed, effective)
}

/// Find throttle acceptance for a specific corner in a lap.
pub fn find_throttle_acceptance(
    distance: &[f64],
    timecodes: &[i64],
    throttle: &[f64],
    smoothed_lateral_g: &[f64],
    effective_threshold: f64,
    corner: &Corner,
    sustain_time_ms: f64,
) -> Option<ThrottleAcceptanceResult> {
    let c_si = math::searchsorted(distance, corner.start_dist);
    let c_ei = math::searchsorted_right(distance, corner.end_dist).min(distance.len());

    if c_si >= c_ei {
        return None;
    }

    // Peak lateral G in corner
    let peak_lateral_g = smoothed_lateral_g[c_si..c_ei]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if peak_lateral_g < MIN_PEAK_LATERAL_G {
        return None;
    }

    // Exit zone: apex to corner end
    let a_si = math::searchsorted(distance, corner.apex_dist);
    if a_si >= c_ei {
        return None;
    }

    let exit_throttle = &throttle[a_si..c_ei];
    let exit_timecodes = &timecodes[a_si..c_ei];
    let exit_smoothed = &smoothed_lateral_g[a_si..c_ei];
    let exit_distance = &distance[a_si..c_ei];
    let n_exit = exit_throttle.len();

    // Find first sustained full-throttle point
    for j in 0..n_exit {
        if exit_throttle[j] >= effective_threshold {
            let start_time = exit_timecodes[j];
            let end_time = start_time + sustain_time_ms as i64;
            let mut sustained = true;

            for k in (j + 1)..n_exit {
                if exit_timecodes[k] > end_time {
                    break;
                }
                if exit_throttle[k] < effective_threshold {
                    sustained = false;
                    break;
                }
            }

            if sustained {
                let ta_pct = (exit_smoothed[j] / peak_lateral_g) * 100.0;
                return Some(ThrottleAcceptanceResult {
                    throttle_acceptance_pct: ta_pct,
                    lateral_g_at_throttle: exit_smoothed[j],
                    peak_lateral_g,
                    full_throttle_dist: exit_distance[j],
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_throttle_acceptance() {
        let throttle = vec![0.0, 50.0, 98.0, 100.0, 95.0];
        let lateral_g = vec![0.5, -0.8, 1.0, -0.3, 0.2];
        let (smoothed, threshold) = prepare_throttle_acceptance(&throttle, &lateral_g, None, 3);

        assert_eq!(smoothed.len(), 5);
        // All smoothed values should be positive (absolute)
        assert!(smoothed.iter().all(|&v| v >= 0.0));
        // Auto-threshold should be 0.98 * 100 = 98
        assert!((threshold - 98.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_throttle_acceptance_basic() {
        // Simulate a corner exit where throttle goes to 100% at 0.5g lateral
        let n = 100;
        let distance: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect(); // 0-198m
        let timecodes: Vec<i64> = (0..n).map(|i| i as i64 * 20).collect(); // 0-1980ms

        let mut throttle = vec![0.0; n];
        let mut lateral_g = vec![0.0; n];

        // Corner from 20-120m, apex at 70m
        let corner = Corner {
            id: 1,
            name: "T1".into(),
            direction: 'L',
            start_idx: 10,
            end_idx: 60,
            start_dist: 20.0,
            end_dist: 120.0,
            apex_idx: 35,
            apex_dist: 70.0,
            max_curvature: 0.01,
        };

        // Set lateral G: peak at 1.0 in corner
        for val in lateral_g.iter_mut().take(60).skip(10) {
            *val = 1.0;
        }

        // Set throttle: goes to 100% after apex (at distance 90m = index 45)
        for val in throttle.iter_mut().skip(45) {
            *val = 100.0;
        }

        let (smoothed, threshold) =
            prepare_throttle_acceptance(&throttle, &lateral_g, Some(98.0), 3);

        let result = find_throttle_acceptance(
            &distance, &timecodes, &throttle, &smoothed, threshold, &corner, 500.0,
        );

        assert!(result.is_some());
        let ta = result.unwrap();
        assert!(ta.throttle_acceptance_pct > 0.0);
        assert!(ta.full_throttle_dist >= 70.0); // After apex
    }
}
