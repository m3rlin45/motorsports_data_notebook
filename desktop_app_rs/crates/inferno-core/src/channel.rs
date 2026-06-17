use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};

use crate::error::{Error, Result};
use crate::lap::Lap;

/// Extract timecodes (column 0) from a channel RecordBatch as Int64Array.
pub fn get_timecodes(batch: &RecordBatch) -> Result<&Int64Array> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::Other("Timecodes column is not Int64".into()))
}

/// Extract values (column 1) from a channel RecordBatch as Float64Array,
/// casting if necessary.
pub fn get_values_f64(batch: &RecordBatch) -> Result<Float64Array> {
    let col = batch.column(1);
    if col.data_type() == &DataType::Float64 {
        Ok(col
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("checked type")
            .clone())
    } else {
        let casted = cast(col, &DataType::Float64)?;
        Ok(casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cast to Float64")
            .clone())
    }
}

/// Filter a channel RecordBatch to a time range [start, end).
pub fn filter_by_time_range(batch: &RecordBatch, start: i64, end: i64) -> Result<RecordBatch> {
    let tc = get_timecodes(batch)?;
    let values = tc.values();

    // Binary search for start/end indices
    let start_idx = values.partition_point(|&t| t < start);
    let end_idx = values.partition_point(|&t| t < end);

    if start_idx >= end_idx {
        // Empty result — return zero-row batch with same schema
        return Ok(batch.slice(0, 0));
    }

    Ok(batch.slice(start_idx, end_idx - start_idx))
}

/// Filter a channel RecordBatch to a specific lap's time range.
pub fn filter_by_lap(batch: &RecordBatch, lap: &Lap) -> Result<RecordBatch> {
    filter_by_time_range(batch, lap.start_time, lap.end_time)
}

/// Resample values to target timecodes using linear interpolation.
/// Source batch has 2 columns: [timecodes (Int64), values (any numeric → cast to Float64)].
/// Returns a Float64Array aligned to `target_tc`.
pub fn resample_to_timecodes(batch: &RecordBatch, target_tc: &Int64Array) -> Result<Float64Array> {
    let src_tc = get_timecodes(batch)?;
    let src_vals = get_values_f64(batch)?;

    let src_t = src_tc.values();
    let src_v = src_vals.values();
    let n_src = src_t.len();
    let n_tgt = target_tc.len();

    if n_src == 0 {
        return Ok(Float64Array::from(vec![0.0; n_tgt]));
    }

    let tgt_t = target_tc.values();
    let mut result = Vec::with_capacity(n_tgt);
    let mut j = 0usize; // current position in source

    for i in 0..n_tgt {
        let t = tgt_t[i];

        // Advance j to find the bracket [j-1, j] around t
        while j < n_src && src_t[j] < t {
            j += 1;
        }

        let val = if j == 0 {
            // Before first source point — use first value
            src_v[0]
        } else if j >= n_src {
            // After last source point — use last value
            src_v[n_src - 1]
        } else if src_t[j] == t {
            // Exact match
            src_v[j]
        } else {
            // Linear interpolation between j-1 and j
            let t0 = src_t[j - 1] as f64;
            let t1 = src_t[j] as f64;
            let v0 = src_v[j - 1];
            let v1 = src_v[j];
            let frac = (t as f64 - t0) / (t1 - t0);
            v0 + frac * (v1 - v0)
        };

        result.push(val);
    }

    Ok(Float64Array::from(result))
}

/// Build a 2-column RecordBatch (timecodes + values) from arrays.
pub fn make_channel_batch(
    name: &str,
    timecodes: Arc<Int64Array>,
    values: Float64Array,
) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timecodes", DataType::Int64, false),
        Field::new(name, DataType::Float64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(timecodes.as_ref().clone()), Arc::new(values)],
    )?)
}

/// Compute cumulative distance from GPS speed (m/s) and timecodes (ms).
/// Returns (distance_m array, timecodes).
pub fn compute_distance_from_speed(
    speed_batch: &RecordBatch,
) -> Result<(Float64Array, Int64Array)> {
    let tc = get_timecodes(speed_batch)?;
    let speed = get_values_f64(speed_batch)?;

    let t = tc.values();
    let v = speed.values();
    let n = t.len();

    let mut dist = Vec::with_capacity(n);
    dist.push(0.0);

    for i in 1..n {
        let dt = (t[i] - t[i - 1]) as f64 / 1000.0; // seconds
        let avg_speed = (v[i] + v[i - 1]) / 2.0; // m/s
        dist.push(dist[i - 1] + avg_speed * dt);
    }

    Ok((Float64Array::from(dist), tc.clone()))
}

/// Compute speed in km/h from GPS speed in m/s.
pub fn speed_ms_to_kmh(speed_ms: &Float64Array) -> Float64Array {
    let values: Vec<f64> = speed_ms.values().iter().map(|v| v * 3.6).collect();
    Float64Array::from(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch(timecodes: &[i64], values: &[f64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timecodes", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(timecodes.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_filter_by_time_range() {
        let batch = make_batch(&[100, 200, 300, 400, 500], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let filtered = filter_by_time_range(&batch, 200, 400).unwrap();
        assert_eq!(filtered.num_rows(), 2);
        let tc = get_timecodes(&filtered).unwrap();
        assert_eq!(tc.value(0), 200);
        assert_eq!(tc.value(1), 300);
    }

    #[test]
    fn test_resample_exact_match() {
        let batch = make_batch(&[0, 100, 200], &[10.0, 20.0, 30.0]);
        let target = Int64Array::from(vec![0, 100, 200]);
        let result = resample_to_timecodes(&batch, &target).unwrap();
        assert_eq!(result.value(0), 10.0);
        assert_eq!(result.value(1), 20.0);
        assert_eq!(result.value(2), 30.0);
    }

    #[test]
    fn test_resample_interpolation() {
        let batch = make_batch(&[0, 100, 200], &[10.0, 20.0, 30.0]);
        let target = Int64Array::from(vec![50, 150]);
        let result = resample_to_timecodes(&batch, &target).unwrap();
        assert!((result.value(0) - 15.0).abs() < 1e-10);
        assert!((result.value(1) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_extrapolation() {
        let batch = make_batch(&[100, 200], &[10.0, 20.0]);
        let target = Int64Array::from(vec![50, 250]);
        let result = resample_to_timecodes(&batch, &target).unwrap();
        // Before first → first value
        assert_eq!(result.value(0), 10.0);
        // After last → last value
        assert_eq!(result.value(1), 20.0);
    }

    #[test]
    fn test_compute_distance() {
        // Constant speed of 10 m/s for 1 second
        let batch = make_batch(&[0, 500, 1000], &[10.0, 10.0, 10.0]);
        let (dist, _tc) = compute_distance_from_speed(&batch).unwrap();
        assert_eq!(dist.value(0), 0.0);
        assert!((dist.value(1) - 5.0).abs() < 1e-10);
        assert!((dist.value(2) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_speed_conversion() {
        let speed_ms = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let speed_kmh = speed_ms_to_kmh(&speed_ms);
        assert!((speed_kmh.value(0) - 36.0).abs() < 1e-10);
        assert!((speed_kmh.value(1) - 72.0).abs() < 1e-10);
    }
}
