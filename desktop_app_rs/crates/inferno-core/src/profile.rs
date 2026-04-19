use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::session::{FileType, Session};

/// Default channel name mappings (AIM devices).
pub fn default_channel_names() -> HashMap<String, String> {
    [
        ("throttle", "PPS"),
        ("brake", "BrakePress"),
        ("gps_speed", "GPS Speed"),
        ("gps_latitude", "GPS Latitude"),
        ("gps_longitude", "GPS Longitude"),
        ("lateral_g", "LateralAcc"),
        ("inline_g", "InlineAcc"),
        ("steering", "SteerAngle"),
        ("shock_fl", "LF_Shock_Pot"),
        ("shock_fr", "RF_Shock_Pot"),
        ("shock_rl", "LR_Shock_Pot"),
        ("shock_rr", "RR_Shock_Pot"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionRatios {
    pub front_left: f64,
    pub front_right: f64,
    pub rear_left: f64,
    pub rear_right: f64,
}

impl Default for MotionRatios {
    fn default() -> Self {
        Self {
            front_left: 1.0,
            front_right: 1.0,
            rear_left: 1.0,
            rear_right: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleProfile {
    pub name: String,
    pub channel_names: HashMap<String, String>,
    #[serde(default)]
    pub motion_ratios: MotionRatios,
}

impl Default for VehicleProfile {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            channel_names: default_channel_names(),
            motion_ratios: MotionRatios::default(),
        }
    }
}

impl VehicleProfile {
    /// Get a subset of channel names for the given keys.
    pub fn get_channel_subset(&self, keys: &[&str]) -> HashMap<String, String> {
        keys.iter()
            .filter_map(|&key| {
                self.channel_names
                    .get(key)
                    .map(|v| (key.to_string(), v.clone()))
            })
            .collect()
    }
}

/// YAML format matching the Python app's profiles.yaml.
#[derive(Debug, Serialize, Deserialize)]
struct ProfilesFile {
    #[serde(default)]
    profiles: HashMap<String, VehicleProfile>,
    #[serde(default)]
    logger_map: HashMap<String, String>,
}

/// Load profiles from a YAML file.
fn load_profiles_from_file(path: &Path) -> Result<ProfilesFile> {
    let content = std::fs::read_to_string(path)?;
    let file: ProfilesFile = serde_yaml::from_str(&content)
        .map_err(|e| crate::error::Error::Other(format!("YAML parse error: {e}")))?;
    Ok(file)
}

/// Get the user profiles directory.
fn user_profiles_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("motorsports_data_notebook"))
}

/// Get the user profiles file path.
fn user_profiles_path() -> Option<PathBuf> {
    user_profiles_dir().map(|d| d.join("profiles.yaml"))
}

/// Load the user's saved profiles.
pub fn load_user_profiles() -> HashMap<String, VehicleProfile> {
    let path = match user_profiles_path() {
        Some(p) if p.exists() => p,
        _ => return HashMap::new(),
    };

    match load_profiles_from_file(&path) {
        Ok(file) => file.profiles,
        Err(_) => HashMap::new(),
    }
}

/// Load the logger-to-profile mapping from user profiles.
pub fn load_user_logger_map() -> HashMap<String, String> {
    let path = match user_profiles_path() {
        Some(p) if p.exists() => p,
        _ => return HashMap::new(),
    };

    match load_profiles_from_file(&path) {
        Ok(file) => file.logger_map,
        Err(_) => HashMap::new(),
    }
}

/// Look up a profile by logger ID string.
pub fn get_profile_for_logger(logger_id: &str) -> Option<VehicleProfile> {
    let path = user_profiles_path()?;
    if !path.exists() {
        return None;
    }

    let file = load_profiles_from_file(&path).ok()?;
    let profile_id = file.logger_map.get(logger_id)?;
    let mut profile = file.profiles.get(profile_id).cloned()?;

    // Fill missing keys from defaults
    let defaults = default_channel_names();
    for (key, val) in &defaults {
        profile
            .channel_names
            .entry(key.clone())
            .or_insert_with(|| val.clone());
    }

    Some(profile)
}

/// Save a profile for a logger ID.
pub fn save_profile_for_logger(
    logger_id: &str,
    profile_id: &str,
    profile: &VehicleProfile,
) -> Result<()> {
    let dir = user_profiles_dir()
        .ok_or_else(|| crate::error::Error::Other("Cannot determine config directory".into()))?;
    std::fs::create_dir_all(&dir)?;

    let path = dir.join("profiles.yaml");
    let mut file = if path.exists() {
        load_profiles_from_file(&path).unwrap_or(ProfilesFile {
            profiles: HashMap::new(),
            logger_map: HashMap::new(),
        })
    } else {
        ProfilesFile {
            profiles: HashMap::new(),
            logger_map: HashMap::new(),
        }
    };

    file.profiles
        .insert(profile_id.to_string(), profile.clone());
    file.logger_map
        .insert(logger_id.to_string(), profile_id.to_string());

    let yaml = serde_yaml::to_string(&file)
        .map_err(|e| crate::error::Error::Other(format!("YAML serialize error: {e}")))?;
    std::fs::write(&path, yaml)?;

    Ok(())
}

/// Extract logger ID from a session.
pub fn get_logger_id(session: &Session) -> String {
    match session.metadata.file_type {
        FileType::Ibt => "iracing".to_string(),
        FileType::Xrk => session
            .metadata
            .logger_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_channel_names() {
        let names = default_channel_names();
        assert_eq!(names.get("throttle").unwrap(), "PPS");
        assert_eq!(names.get("brake").unwrap(), "BrakePress");
        assert_eq!(names.get("gps_latitude").unwrap(), "GPS Latitude");
    }

    #[test]
    fn test_vehicle_profile_defaults() {
        let profile = VehicleProfile::default();
        assert_eq!(profile.name, "Default");
        assert!(profile.channel_names.contains_key("throttle"));
    }

    #[test]
    fn test_get_channel_subset() {
        let profile = VehicleProfile::default();
        let subset = profile.get_channel_subset(&["throttle", "brake"]);
        assert_eq!(subset.len(), 2);
        assert_eq!(subset.get("throttle").unwrap(), "PPS");
    }

    #[test]
    fn test_yaml_roundtrip() {
        let profile = VehicleProfile::default();
        let file = ProfilesFile {
            profiles: [("test".to_string(), profile)].into_iter().collect(),
            logger_map: [("12345".to_string(), "test".to_string())]
                .into_iter()
                .collect(),
        };

        let yaml = serde_yaml::to_string(&file).unwrap();
        let parsed: ProfilesFile = serde_yaml::from_str(&yaml).unwrap();
        assert!(parsed.profiles.contains_key("test"));
        assert_eq!(parsed.profiles["test"].channel_names["throttle"], "PPS");
        assert_eq!(parsed.logger_map["12345"], "test");
    }
}
