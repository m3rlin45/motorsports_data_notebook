use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch};

use crate::channel;
use crate::error::{Error, Result};
use crate::lap::{self, Lap};

/// Metadata extracted from a telemetry file.
#[derive(Debug, Clone, Default)]
pub struct SessionMetadata {
    pub driver: Option<String>,
    pub vehicle: Option<String>,
    pub venue: Option<String>,
    pub log_date: Option<String>,
    pub log_time: Option<String>,
    pub session_name: Option<String>,
    pub logger_id: Option<u32>,
    pub file_name: String,
    pub file_type: FileType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileType {
    #[default]
    Xrk,
    Ibt,
}

/// A loaded telemetry session.
/// Channels are stored as 2-column Arrow RecordBatches (timecodes: Int64, value: <typed>).
pub struct Session {
    pub channels: HashMap<String, RecordBatch>,
    pub laps: Vec<Lap>,
    pub metadata: SessionMetadata,
}

impl Session {
    /// Load a session from a file, detecting format by extension.
    pub fn open(path: &Path) -> Result<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("xrk" | "xrz" | "XRK" | "XRZ") => Self::from_xrk(path),
            Some("ibt" | "IBT") => Self::from_ibt(path),
            Some(ext) => Err(Error::UnsupportedFileType(ext.to_string())),
            None => Err(Error::UnsupportedFileType("(no extension)".to_string())),
        }
    }

    /// Load from an AIM XRK/XRZ file.
    pub fn from_xrk(path: &Path) -> Result<Self> {
        let xrk = libxrk::read_xrk_file(path)?;

        // Build channel RecordBatches from parsed channels.
        // Note: xrk.raw.channel_data is empty (std::mem::take'd by read_xrk),
        // so we must use xrk.channels which has the actual decoded data.
        let mut channels: HashMap<String, RecordBatch> = HashMap::new();
        for ch in xrk.channels {
            let mut metadata = HashMap::new();
            metadata.insert("units".to_string(), ch.units.clone());
            metadata.insert("dec_pts".to_string(), ch.dec_pts.to_string());
            metadata.insert(
                "interpolate".to_string(),
                if ch.interpolate { "True" } else { "False" }.to_string(),
            );

            let ch_data = libxrk::ChannelData {
                timecodes: ch.timecodes,
                values: ch.values,
            };
            let batch = libxrk::arrow::build_channel_batch(&ch.name, ch_data, metadata)?;
            channels.insert(ch.name, batch);
        }

        // Add GPS channels if available
        if let Some(gps) = xrk.gps {
            let gps_batches = libxrk::arrow::build_gps_channel_batches(gps, |f| format!("{f}"))?;
            for (name, batch) in gps_batches {
                channels.insert(name, batch);
            }
        }

        // Build laps
        let laps_batch = libxrk::arrow::build_laps_batch(&xrk.laps)?;
        let mut laps = lap::laps_from_batch(&laps_batch);
        clean_laps_aim(&mut laps);

        // Compute derived channels
        compute_derived_channels(&mut channels, "GPS Speed")?;

        let metadata = SessionMetadata {
            driver: xrk.metadata.driver,
            vehicle: xrk.metadata.vehicle,
            venue: xrk.metadata.venue,
            log_date: xrk.metadata.log_date,
            log_time: xrk.metadata.log_time,
            session_name: xrk.metadata.session,
            logger_id: xrk.metadata.logger_id,
            file_name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            file_type: FileType::Xrk,
        };

        Ok(Session {
            channels,
            laps,
            metadata,
        })
    }

    /// Load from an iRacing IBT file.
    pub fn from_ibt(path: &Path) -> Result<Self> {
        let ibt = libibt::IbtFile::open(path)?;
        let timecodes = ibt.build_timecodes()?;

        // Build all channel RecordBatches
        let channel_batches = ibt.all_channels_to_arrow(&timecodes)?;
        let mut channels: HashMap<String, RecordBatch> = channel_batches.into_iter().collect();

        // Extract laps
        let laps_batch = ibt.extract_laps(&timecodes)?;
        let laps = lap::laps_from_batch(&laps_batch);

        // Compute derived channels from Speed (m/s)
        compute_derived_channels(&mut channels, "Speed")?;

        let metadata = SessionMetadata {
            file_name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            file_type: FileType::Ibt,
            ..Default::default()
        };

        Ok(Session {
            channels,
            laps,
            metadata,
        })
    }

    /// Get a channel RecordBatch by name.
    pub fn channel(&self, name: &str) -> Option<&RecordBatch> {
        self.channels.get(name)
    }

    /// List all channel names.
    pub fn channel_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.channels.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Get channel values for a specific lap, resampled to a target timebase.
    /// Returns (target_timecodes, values) where values are Float64.
    pub fn lap_channel(
        &self,
        channel_name: &str,
        lap: &Lap,
        target_tc: &Int64Array,
    ) -> Result<Float64Array> {
        let batch = self
            .channels
            .get(channel_name)
            .ok_or_else(|| Error::MissingChannel(channel_name.to_string()))?;

        let filtered = channel::filter_by_lap(batch, lap)?;
        channel::resample_to_timecodes(&filtered, target_tc)
    }
}

/// Compute speed_kmh and distance_m from a speed channel (m/s).
fn compute_derived_channels(
    channels: &mut HashMap<String, RecordBatch>,
    speed_channel: &str,
) -> Result<()> {
    let speed_batch = match channels.get(speed_channel) {
        Some(b) => b.clone(),
        None => return Ok(()),
    };

    let speed_vals = channel::get_values_f64(&speed_batch)?;
    let speed_tc = channel::get_timecodes(&speed_batch)?.clone();

    let speed_kmh = channel::speed_ms_to_kmh(&speed_vals);
    let speed_kmh_batch = channel::make_channel_batch("speed_kmh", Arc::new(speed_tc), speed_kmh)?;
    channels.insert("speed_kmh".to_string(), speed_kmh_batch);

    let (dist, dist_tc) = channel::compute_distance_from_speed(&speed_batch)?;
    let dist_batch = channel::make_channel_batch("distance_m", Arc::new(dist_tc), dist)?;
    channels.insert("distance_m".to_string(), dist_batch);

    Ok(())
}

/// Clean AIM laps: remove lap 0 (out lap from pit) if it has negative or zero duration.
fn clean_laps_aim(laps: &mut Vec<Lap>) {
    laps.retain(|lap| lap.duration_ms() > 0);
}

/// Aligned per-lap channel data, resampled to distance timebase.
pub struct LapData {
    pub lap_num: i32,
    pub distance: Float64Array,
    pub channels: HashMap<String, Float64Array>,
}

impl LapData {
    /// Extract and align channels for a single lap, resampled to the distance channel's timebase.
    pub fn extract(session: &Session, lap: &Lap, channel_names: &[&str]) -> Result<Self> {
        // Get distance channel for this lap
        let dist_batch = session
            .channels
            .get("distance_m")
            .ok_or_else(|| Error::MissingChannel("distance_m".to_string()))?;

        let lap_dist = channel::filter_by_lap(dist_batch, lap)?;
        if lap_dist.num_rows() == 0 {
            return Err(Error::Other(format!("No data for lap {}", lap.num)));
        }

        let target_tc = channel::get_timecodes(&lap_dist)?.clone();
        let raw_distance = channel::get_values_f64(&lap_dist)?;

        // Zero-base distance so each lap starts at 0m (matches corner detection's GPS distance scale)
        let first_dist = raw_distance.value(0);
        let distance = Float64Array::from(
            raw_distance
                .values()
                .iter()
                .map(|d| d - first_dist)
                .collect::<Vec<f64>>(),
        );

        let mut channels = HashMap::new();
        for &name in channel_names {
            if let Some(batch) = session.channels.get(name) {
                let filtered = channel::filter_by_lap(batch, lap)?;
                let resampled = channel::resample_to_timecodes(&filtered, &target_tc)?;
                channels.insert(name.to_string(), resampled);
            }
        }

        Ok(LapData {
            lap_num: lap.num,
            distance,
            channels,
        })
    }

    /// Get a channel's values as a float slice.
    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.channels.get(name).map(|arr| arr.values().as_ref())
    }

    /// Get distance values as a float slice.
    pub fn dist(&self) -> &[f64] {
        self.distance.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_detection() {
        let err = Session::open(Path::new("test.xyz"));
        assert!(matches!(err, Err(Error::UnsupportedFileType(_))));
    }

    #[test]
    fn test_clean_laps() {
        let mut laps = vec![
            Lap {
                num: 0,
                start_time: 0,
                end_time: 0,
                lap_type: None,
                session: None,
            },
            Lap {
                num: 1,
                start_time: 0,
                end_time: 80000,
                lap_type: None,
                session: None,
            },
        ];
        clean_laps_aim(&mut laps);
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].num, 1);
    }
}
