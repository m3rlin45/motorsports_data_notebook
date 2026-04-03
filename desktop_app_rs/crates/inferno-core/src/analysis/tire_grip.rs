use super::math;
use crate::channel;
use crate::error::{Error, Result};
use crate::session::Session;

/// Whether to analyze pressure or temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricMode {
    Pressure,
    Temperature,
}

/// Channel names and parameters for tire grip analysis.
pub struct TireGripConfig {
    pub lateral_g: String,
    pub inline_g: String,
    pub tpms_press_fl: String,
    pub tpms_press_fr: String,
    pub tpms_press_rl: String,
    pub tpms_press_rr: String,
    pub tpms_temp_fl: String,
    pub tpms_temp_fr: String,
    pub tpms_temp_rl: String,
    pub tpms_temp_rr: String,
    pub metric_mode: MetricMode,
    pub num_buckets: usize,
    pub percentile: f64,
    pub min_count: usize,
}

impl Default for TireGripConfig {
    fn default() -> Self {
        Self {
            lateral_g: "LateralAcc".into(),
            inline_g: "InlineAcc".into(),
            tpms_press_fl: "TPMS_Press_LF".into(),
            tpms_press_fr: "TPMS_Press_RF".into(),
            tpms_press_rl: "TPMS_Press_LR".into(),
            tpms_press_rr: "TPMS_Press_RR".into(),
            tpms_temp_fl: "TPMS_Temp_LF".into(),
            tpms_temp_fr: "TPMS_Temp_RF".into(),
            tpms_temp_rl: "TPMS_Temp_LR".into(),
            tpms_temp_rr: "TPMS_Temp_RR".into(),
            metric_mode: MetricMode::Pressure,
            num_buckets: 20,
            percentile: 99.9,
            min_count: 5,
        }
    }
}

impl TireGripConfig {
    /// Return the 4 metric channel names for the current mode.
    fn metric_channels(&self) -> [&str; 4] {
        match self.metric_mode {
            MetricMode::Pressure => [
                &self.tpms_press_fl,
                &self.tpms_press_fr,
                &self.tpms_press_rl,
                &self.tpms_press_rr,
            ],
            MetricMode::Temperature => [
                &self.tpms_temp_fl,
                &self.tpms_temp_fr,
                &self.tpms_temp_rl,
                &self.tpms_temp_rr,
            ],
        }
    }
}

/// Per-wheel grip analysis result.
#[derive(Debug, Clone)]
pub struct WheelGripData {
    pub name: String,
    pub mean_g: f64,
    pub std_g: f64,
    pub mean_metric: f64,
    pub std_metric: f64,
    pub bucket_centers: Vec<f64>,
    pub bucket_values: Vec<f64>,
    pub bucket_counts: Vec<u64>,
    pub percentile: f64,
}

/// Complete tire grip analysis result.
#[derive(Debug, Clone)]
pub struct TireGripResult {
    /// [FL, FR, RL, RR]
    pub wheels: [WheelGripData; 4],
    pub metric_mode: MetricMode,
    pub metric_unit: String,
    pub accel_unit: String,
}

/// Compute bucketed percentile for a single wheel.
fn compute_wheel_grip(
    total_g: &[f64],
    tire_metric: &[f64],
    name: &str,
    num_buckets: usize,
    pct: f64,
    min_count: usize,
) -> WheelGripData {
    let n = total_g.len();
    if n == 0 || tire_metric.is_empty() {
        return WheelGripData {
            name: name.to_string(),
            mean_g: 0.0,
            std_g: 0.0,
            mean_metric: 0.0,
            std_metric: 0.0,
            bucket_centers: vec![],
            bucket_values: vec![],
            bucket_counts: vec![],
            percentile: pct,
        };
    }

    let mean_g = math::mean(total_g);
    let std_g = math::std_dev(total_g);
    let mean_metric = math::mean(tire_metric);
    let std_metric = math::std_dev(tire_metric);

    // Create evenly spaced bin edges across metric range
    let metric_min = tire_metric.iter().cloned().fold(f64::INFINITY, f64::min);
    let metric_max = tire_metric
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (metric_max - metric_min).abs() < 1e-12 || num_buckets == 0 {
        return WheelGripData {
            name: name.to_string(),
            mean_g,
            std_g,
            mean_metric,
            std_metric,
            bucket_centers: vec![],
            bucket_values: vec![],
            bucket_counts: vec![],
            percentile: pct,
        };
    }

    let step = (metric_max - metric_min) / num_buckets as f64;
    // edges[0..=num_buckets] — num_buckets+1 edges
    let edges: Vec<f64> = (0..=num_buckets)
        .map(|i| metric_min + i as f64 * step)
        .collect();

    // Assign each sample to a bucket (1-indexed, clipped to [1, num_buckets])
    // numpy.digitize returns 1..=num_buckets for values in range
    let bin_indices: Vec<usize> = tire_metric
        .iter()
        .map(|&v| {
            let idx = edges.partition_point(|&e| e < v);
            idx.clamp(1, num_buckets)
        })
        .collect();

    // Compute percentile per bucket
    let mut centers = Vec::new();
    let mut values = Vec::new();
    let mut counts = Vec::new();

    for bucket in 1..=num_buckets {
        // Collect total_g values for this bucket
        let mut g_in_bucket: Vec<f64> = bin_indices
            .iter()
            .zip(total_g.iter())
            .filter(|(&bi, _)| bi == bucket)
            .map(|(_, &g)| g)
            .collect();

        if g_in_bucket.len() >= min_count {
            centers.push((edges[bucket - 1] + edges[bucket]) / 2.0);
            values.push(math::percentile(&mut g_in_bucket, pct));
            counts.push(g_in_bucket.len() as u64);
        }
    }

    WheelGripData {
        name: name.to_string(),
        mean_g,
        std_g,
        mean_metric,
        std_metric,
        bucket_centers: centers,
        bucket_values: values,
        bucket_counts: counts,
        percentile: pct,
    }
}

/// Extract unit string from a channel's Arrow metadata.
fn get_channel_unit(session: &Session, channel_name: &str) -> String {
    session
        .channels
        .get(channel_name)
        .and_then(|batch| batch.schema().field(1).metadata().get("units").cloned())
        .unwrap_or_default()
}

/// Run tire grip analysis across multiple laps.
///
/// Computes total acceleration from lateral + inline G, then buckets tire
/// pressure or temperature and computes the percentile of total G per bucket.
pub fn analyze_tire_grip(
    session: &Session,
    selected_laps: &[i32],
    config: &TireGripConfig,
) -> Result<TireGripResult> {
    if selected_laps.is_empty() {
        return Err(Error::Other("No laps selected".into()));
    }

    let metric_channels = config.metric_channels();
    let wheel_labels = ["FL", "FR", "RL", "RR"];

    // Validate lateral_g channel exists
    if !session.channels.contains_key(&config.lateral_g) {
        return Err(Error::MissingChannel(config.lateral_g.clone()));
    }
    // Validate first metric channel exists
    if !session.channels.contains_key(metric_channels[0]) {
        return Err(Error::MissingChannel(metric_channels[0].to_string()));
    }

    // Collect per-wheel: all total_g values and tire_metric values across laps
    let mut all_total_g: Vec<f64> = Vec::new();
    let mut all_metric: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];

    for &lap_num in selected_laps {
        let lap = match session.laps.iter().find(|l| l.num == lap_num) {
            Some(l) => l,
            None => continue,
        };

        // Get lateral_g filtered to this lap — use its timecodes as reference
        let lat_batch = match session.channels.get(&config.lateral_g) {
            Some(b) => b,
            None => continue,
        };
        let lat_lap = match channel::filter_by_lap(lat_batch, lap) {
            Ok(b) if b.num_rows() > 1 => b,
            _ => continue,
        };
        let target_tc = match channel::get_timecodes(&lat_lap) {
            Ok(tc) => tc.clone(),
            Err(_) => continue,
        };
        let lat_vals = match channel::get_values_f64(&lat_lap) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Get inline_g resampled to same timebase
        let inline_vals = match session.channels.get(&config.inline_g) {
            Some(batch) => {
                let lap_batch = channel::filter_by_lap(batch, lap)?;
                channel::resample_to_timecodes(&lap_batch, &target_tc)?
            }
            None => {
                // If no inline_g, use zeros (lateral-only)
                arrow::array::Float64Array::from(vec![0.0; target_tc.len()])
            }
        };

        // Compute total_g = sqrt(lat² + inline²)
        let lat_s = lat_vals.values();
        let inl_s = inline_vals.values();
        let total_g: Vec<f64> = lat_s
            .iter()
            .zip(inl_s.iter())
            .map(|(&l, &i)| (l * l + i * i).sqrt())
            .collect();

        all_total_g.extend_from_slice(&total_g);

        // Extract each metric channel resampled to same timebase
        for (i, &ch_name) in metric_channels.iter().enumerate() {
            if let Some(batch) = session.channels.get(ch_name) {
                if let Ok(lap_batch) = channel::filter_by_lap(batch, lap) {
                    if let Ok(resampled) = channel::resample_to_timecodes(&lap_batch, &target_tc) {
                        all_metric[i].extend_from_slice(resampled.values());
                        continue;
                    }
                }
            }
            // If channel missing/failed, pad with NaN (will be excluded from stats)
            all_metric[i].extend(std::iter::repeat_n(f64::NAN, total_g.len()));
        }
    }

    if all_total_g.is_empty() {
        return Err(Error::Other("No data extracted from selected laps".into()));
    }

    // Build per-wheel results
    let wheels = std::array::from_fn(|i| {
        // Filter out NaN metric values
        let (g_clean, m_clean): (Vec<f64>, Vec<f64>) = all_total_g
            .iter()
            .zip(all_metric[i].iter())
            .filter(|(_, m)| m.is_finite())
            .map(|(&g, &m)| (g, m))
            .unzip();

        compute_wheel_grip(
            &g_clean,
            &m_clean,
            wheel_labels[i],
            config.num_buckets,
            config.percentile,
            config.min_count,
        )
    });

    let metric_unit = get_channel_unit(session, metric_channels[0]);
    let accel_unit = get_channel_unit(session, &config.lateral_g);

    Ok(TireGripResult {
        wheels,
        metric_mode: config.metric_mode,
        metric_unit,
        accel_unit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_wheel_grip_empty() {
        let data = compute_wheel_grip(&[], &[], "FL", 20, 99.9, 5);
        assert_eq!(data.name, "FL");
        assert!(data.bucket_centers.is_empty());
        assert_eq!(data.mean_g, 0.0);
    }

    #[test]
    fn test_compute_wheel_grip_basic() {
        // 200 samples with linear relationship: metric=0..10, g=metric*0.1
        let n = 200;
        let metric: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 10.0).collect();
        let total_g: Vec<f64> = metric.iter().map(|&m| m * 0.1).collect();

        let data = compute_wheel_grip(&total_g, &metric, "FL", 10, 99.0, 5);
        assert!(!data.bucket_centers.is_empty());
        assert_eq!(data.bucket_centers.len(), data.bucket_values.len());
        assert_eq!(data.bucket_centers.len(), data.bucket_counts.len());
        // Higher metric values should have higher G
        if data.bucket_values.len() >= 2 {
            assert!(
                data.bucket_values.last().unwrap() > data.bucket_values.first().unwrap(),
                "Higher metric should correlate with higher G"
            );
        }
    }

    #[test]
    fn test_compute_wheel_grip_min_count() {
        // Only 3 samples per bucket (below min_count=5) → empty
        let metric = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let total_g = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let data = compute_wheel_grip(&total_g, &metric, "FL", 6, 99.9, 5);
        // With 6 samples in 6 buckets, each bucket has ~1 sample → all filtered out
        assert!(data.bucket_centers.is_empty());
    }

    #[test]
    fn test_compute_wheel_grip_constant_metric() {
        // All same metric value → no bins possible
        let metric = vec![5.0; 100];
        let total_g = vec![1.0; 100];
        let data = compute_wheel_grip(&total_g, &metric, "FL", 20, 99.9, 5);
        assert!(data.bucket_centers.is_empty());
    }
}
