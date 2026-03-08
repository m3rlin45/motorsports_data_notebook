use arrow::array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};

/// A single lap boundary.
#[derive(Debug, Clone)]
pub struct Lap {
    pub num: i32,
    pub start_time: i64,
    pub end_time: i64,
    pub lap_type: Option<String>,
    pub session: Option<i32>,
}

impl Lap {
    pub fn duration_ms(&self) -> i64 {
        self.end_time - self.start_time
    }
}

/// Extract laps from an Arrow RecordBatch (columns: num, start_time, end_time,
/// and optionally lap_type, session).
pub fn laps_from_batch(batch: &RecordBatch) -> Vec<Lap> {
    let nums = batch
        .column_by_name("num")
        .expect("laps batch missing 'num' column")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("'num' column is not Int32");
    let starts = batch
        .column_by_name("start_time")
        .expect("laps batch missing 'start_time' column")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("'start_time' column is not Int64");
    let ends = batch
        .column_by_name("end_time")
        .expect("laps batch missing 'end_time' column")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("'end_time' column is not Int64");

    let lap_types = batch
        .column_by_name("lap_type")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned());
    let sessions = batch
        .column_by_name("session")
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>().cloned());

    (0..batch.num_rows())
        .map(|i| Lap {
            num: nums.value(i),
            start_time: starts.value(i),
            end_time: ends.value(i),
            lap_type: lap_types.as_ref().and_then(|lt| {
                if lt.is_null(i) {
                    None
                } else {
                    Some(lt.value(i).to_string())
                }
            }),
            session: sessions.as_ref().map(|s| s.value(i)),
        })
        .collect()
}

/// Find the best (fastest) lap from a list.
/// For IBT files (laps with lap_type), only "full" laps are considered.
/// For AIM files (no lap_type), excludes first and last laps.
pub fn get_best_lap(laps: &[Lap]) -> Option<&Lap> {
    let candidates = filter_valid_laps(laps);
    candidates
        .into_iter()
        .filter(|lap| lap.duration_ms() > 0)
        .min_by_key(|lap| lap.duration_ms())
}

/// Find laps within `threshold_pct` of the best lap time.
/// Default threshold is 1.03 (103% of best).
pub fn get_top_laps(laps: &[Lap], threshold_pct: f64) -> Vec<&Lap> {
    let candidates = filter_valid_laps(laps);
    let valid: Vec<&Lap> = candidates
        .into_iter()
        .filter(|lap| lap.duration_ms() > 0)
        .collect();

    let best_time = match valid.iter().map(|l| l.duration_ms()).min() {
        Some(t) => t,
        None => return vec![],
    };

    let cutoff = (best_time as f64 * threshold_pct) as i64;
    valid
        .into_iter()
        .filter(|lap| lap.duration_ms() <= cutoff)
        .collect()
}

/// Filter to valid laps based on type information.
fn filter_valid_laps(laps: &[Lap]) -> Vec<&Lap> {
    let has_lap_type = laps.iter().any(|l| l.lap_type.is_some());

    if has_lap_type {
        // IBT: only "full" laps
        laps.iter()
            .filter(|l| l.lap_type.as_deref() == Some("full"))
            .collect()
    } else {
        // AIM: exclude first and last laps
        if laps.len() <= 2 {
            return vec![];
        }
        laps[1..laps.len() - 1].iter().collect()
    }
}

/// Infer whether a channel uses 0-1 or 0-100 scale.
/// Returns a multiplier to normalize to 0-100.
pub fn infer_channel_scale(values: &Float64Array) -> f64 {
    let max_val = values.iter().flatten().fold(f64::NEG_INFINITY, f64::max);

    if max_val <= 1.5 {
        100.0 // 0-1 scale → multiply by 100
    } else {
        1.0 // already 0-100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_laps(durations: &[(i32, i64)]) -> Vec<Lap> {
        let mut start = 0i64;
        durations
            .iter()
            .map(|&(num, dur)| {
                let lap = Lap {
                    num,
                    start_time: start,
                    end_time: start + dur,
                    lap_type: None,
                    session: None,
                };
                start += dur;
                lap
            })
            .collect()
    }

    fn make_ibt_laps(data: &[(i32, i64, &str)]) -> Vec<Lap> {
        let mut start = 0i64;
        data.iter()
            .map(|&(num, dur, lt)| {
                let lap = Lap {
                    num,
                    start_time: start,
                    end_time: start + dur,
                    lap_type: Some(lt.to_string()),
                    session: Some(0),
                };
                start += dur;
                lap
            })
            .collect()
    }

    #[test]
    fn test_best_lap_aim() {
        // AIM: excludes first and last
        let laps = make_laps(&[(0, 90000), (1, 80000), (2, 85000), (3, 82000), (4, 95000)]);
        let best = get_best_lap(&laps).unwrap();
        assert_eq!(best.num, 1);
    }

    #[test]
    fn test_best_lap_ibt() {
        let laps = make_ibt_laps(&[
            (0, 90000, "out"),
            (1, 80000, "full"),
            (2, 75000, "full"),
            (3, 95000, "in"),
        ]);
        let best = get_best_lap(&laps).unwrap();
        assert_eq!(best.num, 2);
    }

    #[test]
    fn test_top_laps() {
        let laps = make_laps(&[
            (0, 90000),
            (1, 80000),
            (2, 82000),
            (3, 85000),
            (4, 100000),
            (5, 95000),
        ]);
        let top = get_top_laps(&laps, 1.03);
        let nums: Vec<i32> = top.iter().map(|l| l.num).collect();
        assert!(nums.contains(&1));
        assert!(nums.contains(&2));
        // 85000 / 80000 = 1.0625, above 1.03 cutoff
        assert!(!nums.contains(&3));
    }

    #[test]
    fn test_infer_channel_scale_0_1() {
        let arr = Float64Array::from(vec![0.0, 0.5, 0.98, 1.0]);
        assert_eq!(infer_channel_scale(&arr), 100.0);
    }

    #[test]
    fn test_infer_channel_scale_0_100() {
        let arr = Float64Array::from(vec![0.0, 50.0, 98.0, 100.0]);
        assert_eq!(infer_channel_scale(&arr), 1.0);
    }
}
