"""Vehicle profile mapping for AIM telemetry channel names.

Provides a two-tier YAML profile system: reusable vehicle profiles (channel names
and motion ratios) mapped to logger serial numbers. Profiles can be shipped as
builtins or saved per-user.
"""

from __future__ import annotations

import importlib.resources
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from ._types import LogFile
    from .suspension import MotionRatios


# All canonical channel keys with their default values (Toyota 86 ZN6 / AiM MXP defaults)
DEFAULT_CHANNEL_NAMES: dict[str, str] = {
    "throttle": "PPS",
    "brake": "BrakePress",
    "brake_rear": "",
    "gps_speed": "GPS Speed",
    "gps_latitude": "GPS Latitude",
    "gps_longitude": "GPS Longitude",
    "lateral_g": "LateralAcc",
    "inline_g": "InlineAcc",
    "steering": "SteerAngle",
    "shock_fl": "LF_Shock_Pot",
    "shock_fr": "RF_Shock_Pot",
    "shock_rl": "LR_Shock_Pot",
    "shock_rr": "RR_Shock_Pot",
    "tpms_press_fl": "TPMS_Press_LF",
    "tpms_press_fr": "TPMS_Press_RF",
    "tpms_press_rl": "TPMS_Press_LR",
    "tpms_press_rr": "TPMS_Press_RR",
    "tpms_temp_fl": "TPMS_Temp_LF",
    "tpms_temp_fr": "TPMS_Temp_RF",
    "tpms_temp_rl": "TPMS_Temp_LR",
    "tpms_temp_rr": "TPMS_Temp_RR",
    **{f"tire_temp_fl_{i}": f"FL_Ch{i}" for i in range(1, 9)},
    **{f"tire_temp_fr_{i}": f"FR_Ch{i}" for i in range(1, 9)},
    **{f"tire_temp_rl_{i}": f"RL_Ch{i}" for i in range(1, 9)},
    **{f"tire_temp_rr_{i}": f"RR_Ch{i}" for i in range(1, 9)},
}

ALL_CANONICAL_KEYS: list[str] = list(DEFAULT_CHANNEL_NAMES.keys())


@dataclass
class VehicleProfile:
    """A vehicle's channel name mapping and motion ratios.

    Attributes
    ----------
    name : str
        Human-readable vehicle name.
    channel_names : dict[str, str]
        Mapping from canonical keys to actual AIM channel names.
    motion_ratios : MotionRatios
        Shock-to-wheel motion ratios for each corner.
    """

    name: str
    channel_names: dict[str, str] = field(default_factory=lambda: DEFAULT_CHANNEL_NAMES.copy())
    motion_ratios: MotionRatios = field(default_factory=lambda: _default_motion_ratios())

    def get_channel_subset(self, keys: list[str]) -> dict[str, str]:
        """Return a subset of channel_names for the given canonical keys.

        Parameters
        ----------
        keys : list[str]
            Canonical keys to include.

        Returns
        -------
        dict[str, str]
            Subset of channel_names containing only the requested keys.
        """
        return {k: self.channel_names[k] for k in keys if k in self.channel_names}


def _default_motion_ratios() -> MotionRatios:
    """Create default MotionRatios (deferred import to avoid circular dependency)."""
    from .suspension import MotionRatios

    return MotionRatios()


def _profile_to_dict(profile: VehicleProfile) -> dict:
    """Serialize a VehicleProfile to a dict suitable for YAML output."""
    return {
        "name": profile.name,
        "channel_names": dict(profile.channel_names),
        "motion_ratios": {
            "front_left": profile.motion_ratios.front_left,
            "front_right": profile.motion_ratios.front_right,
            "rear_left": profile.motion_ratios.rear_left,
            "rear_right": profile.motion_ratios.rear_right,
        },
    }


def _profile_from_dict(data: dict) -> VehicleProfile:
    """Deserialize a VehicleProfile from a YAML dict.

    Missing channel keys are filled from DEFAULT_CHANNEL_NAMES for forward
    compatibility — new canonical keys added in future versions won't break
    existing profile files.
    """
    channel_names = DEFAULT_CHANNEL_NAMES.copy()
    if "channel_names" in data:
        channel_names.update(data["channel_names"])

    from .suspension import MotionRatios

    ratios_data = data.get("motion_ratios", {})
    motion_ratios = MotionRatios(
        front_left=float(ratios_data.get("front_left", 0.997)),
        front_right=float(ratios_data.get("front_right", 0.997)),
        rear_left=float(ratios_data.get("rear_left", 0.768)),
        rear_right=float(ratios_data.get("rear_right", 0.768)),
    )

    return VehicleProfile(
        name=data.get("name", "Unknown"),
        channel_names=channel_names,
        motion_ratios=motion_ratios,
    )


def _load_profiles_file(path: Path) -> dict[str, VehicleProfile]:
    """Parse a two-tier YAML profiles file.

    Returns a dict keyed by logger ID, with each value being the resolved
    VehicleProfile from the profiles section.

    Parameters
    ----------
    path : Path
        Path to the YAML profiles file.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from logger ID strings to VehicleProfile instances.
    """
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    # Parse profile definitions
    profiles_section = data.get("profiles", {})
    if not isinstance(profiles_section, dict):
        profiles_section = {}

    parsed_profiles: dict[str, VehicleProfile] = {}
    for profile_id, profile_data in profiles_section.items():
        if isinstance(profile_data, dict):
            parsed_profiles[profile_id] = _profile_from_dict(profile_data)

    # Resolve logger_map -> profiles
    logger_map = data.get("logger_map", {})
    if not isinstance(logger_map, dict):
        logger_map = {}

    result: dict[str, VehicleProfile] = {}
    for logger_id, profile_id in logger_map.items():
        logger_id_str = str(logger_id)
        if profile_id in parsed_profiles:
            result[logger_id_str] = parsed_profiles[profile_id]

    return result


def _get_user_profiles_path() -> Path:
    """Return the path to the user's profiles YAML file.

    Linux/macOS: ~/.config/motorsports_data_notebook/profiles.yaml
    Windows: %APPDATA%/motorsports_data_notebook/profiles.yaml
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "motorsports_data_notebook" / "profiles.yaml"


def load_builtin_profiles() -> dict[str, VehicleProfile]:
    """Load profiles shipped with the package.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from logger ID strings to VehicleProfile instances.
    """
    # Try importlib.resources first (normal Python installs)
    try:
        ref = importlib.resources.files("motorsports_data_notebook.data").joinpath(
            "builtin_profiles.yaml"
        )
        with importlib.resources.as_file(ref) as path:
            result = _load_profiles_file(path)
            if result:
                return result
    except Exception:
        pass

    # Fallback: relative to this file
    path = Path(__file__).parent / "data" / "builtin_profiles.yaml"
    result = _load_profiles_file(path)
    if result:
        return result

    # Fallback: PyInstaller _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        path = Path(meipass) / "motorsports_data_notebook" / "data" / "builtin_profiles.yaml"
        result = _load_profiles_file(path)
        if result:
            return result

    return {}


def load_user_profiles() -> dict[str, VehicleProfile]:
    """Load profiles from the user's config directory.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from logger ID strings to VehicleProfile instances.
        Empty dict if file doesn't exist or is invalid.
    """
    return _load_profiles_file(_get_user_profiles_path())


def save_user_profiles(profiles: dict[str, VehicleProfile]) -> None:
    """Save profiles to the user's config directory.

    Parameters
    ----------
    profiles : dict[str, VehicleProfile]
        Mapping from logger ID strings to VehicleProfile instances.
    """
    path = _get_user_profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group profiles by identity to avoid duplicating profile definitions
    # Use profile name as the dedup key
    profile_defs: dict[str, dict] = {}
    profile_id_map: dict[str, str] = {}  # name -> profile_id

    logger_map: dict[str, str] = {}

    for logger_id, profile in profiles.items():
        # Create a stable profile_id from the name
        profile_id = profile.name.lower().replace(" ", "_")
        if profile_id not in profile_defs:
            profile_defs[profile_id] = _profile_to_dict(profile)
            profile_id_map[profile.name] = profile_id
        logger_map[logger_id] = profile_id_map[profile.name]

    data = {
        "profiles": profile_defs,
        "logger_map": logger_map,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_profile_for_logger(logger_id: str) -> VehicleProfile | None:
    """Look up a vehicle profile by logger ID.

    Checks user profiles first, then builtin profiles.

    Parameters
    ----------
    logger_id : str
        The logger serial number.

    Returns
    -------
    VehicleProfile | None
        The matching profile, or None if not found.
    """
    # User profiles take precedence
    user_profiles = load_user_profiles()
    if logger_id in user_profiles:
        return user_profiles[logger_id]

    builtin_profiles = load_builtin_profiles()
    if logger_id in builtin_profiles:
        return builtin_profiles[logger_id]

    return None


def save_profile_for_logger(logger_id: str, profile: VehicleProfile) -> None:
    """Save a profile for a logger ID to the user's config.

    Merges with existing user profiles.

    Parameters
    ----------
    logger_id : str
        The logger serial number.
    profile : VehicleProfile
        The profile to save.
    """
    profiles = load_user_profiles()
    profiles[logger_id] = profile
    save_user_profiles(profiles)


def get_logger_id(log: LogFile) -> str | None:
    """Extract the logger serial number from a LogFile's metadata.

    For AIM files, returns the logger serial number.
    For iRacing IBT files, returns ``"iracing"`` (all iRacing cars use
    the same channel names).

    Parameters
    ----------
    log : LogFile
        A loaded telemetry log file.

    Returns
    -------
    str | None
        The logger ID string, or None if not available.
    """
    if not hasattr(log, "metadata") or not isinstance(log.metadata, dict):
        return None
    # AIM logger ID
    val = log.metadata.get("Logger ID") or log.metadata.get("logger_id")
    if val is not None:
        return str(val)
    # iRacing: all IBT files use the same channel names
    if "session_info_yaml" in log.metadata:
        return "iracing"
    return None


def is_iracing_session(log: LogFile) -> bool:
    """Check if a LogFile was loaded from an iRacing IBT file.

    Parameters
    ----------
    log : LogFile
        A loaded telemetry log file.

    Returns
    -------
    bool
        True if the log file contains iRacing session metadata.
    """
    return get_logger_id(log) == "iracing"
