"""Vehicle profile mapping for AIM telemetry channel names.

Provides a two-tier YAML profile system: reusable vehicle profiles (channel names
and motion ratios) mapped to logger serial numbers. Profiles can be shipped as
builtins or saved per-user.

A logger that is moved between cars (same serial number, different Race Studio
configuration) can be disambiguated by vehicle name::

    logger_map:
      8401203:
        default: kk_s2
        by_vehicle:
          KK-SII: kk_s2
          KK-F: kk_f

The vehicle name is read from the file metadata (``Vehicle``, set in Race
Studio and present in .xrk files) and, when that is empty (typical for .xrz
exports), from the Race Studio file name ``<Racer>_<Vehicle>_<Track>_...``.
"""

from __future__ import annotations

import importlib.resources
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from ._types import LogFile
    from .roll_gradient import TrackWidths
    from .suspension import MotionRatios, ShockCalibration


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

# Separator between logger ID and vehicle name in composite profile keys
PROFILE_KEY_SEPARATOR = "/"


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
        Pot-to-wheel motion ratios for each corner, as applied to the logged
        shock pot channel (``wheel = pot_channel / ratio``). This is the only
        ratio the suspension analysis uses.
    shock_calibration : ShockCalibration or None
        Stroke-sensor calibration slopes per axle (wheel per pot mm, damper per
        pot mm), kept for reference and future use. When a profile file defines
        it without explicit ``motion_ratios``, the pot/wheel ratios are derived
        from it.
    track_widths : TrackWidths or None
        Front/rear track widths in mm, used by the roll gradient analysis.
    """

    name: str
    channel_names: dict[str, str] = field(default_factory=lambda: DEFAULT_CHANNEL_NAMES.copy())
    motion_ratios: MotionRatios = field(default_factory=lambda: _default_motion_ratios())
    shock_calibration: ShockCalibration | None = None
    track_widths: TrackWidths | None = None

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


def _axle_calibration_to_dict(axle: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "wheel_per_sensor": float(axle.wheel_per_sensor),
        "damper_per_sensor": float(axle.damper_per_sensor),
    }
    if axle.damper_per_wheel_fit is not None:
        data["damper_per_wheel"] = float(axle.damper_per_wheel_fit)
    if axle.wheel_r2 is not None:
        data["wheel_r2"] = float(axle.wheel_r2)
    if axle.damper_r2 is not None:
        data["damper_r2"] = float(axle.damper_r2)
    if axle.damper_wheel_r2 is not None:
        data["damper_wheel_r2"] = float(axle.damper_wheel_r2)
    if axle.points:
        data["points"] = [[float(v) for v in row] for row in axle.points]
    if axle.excluded_points:
        data["excluded_points"] = [int(i) for i in axle.excluded_points]
    return data


def _axle_calibration_from_dict(data: Any) -> Any:
    from .suspension import AxleShockCalibration

    if not isinstance(data, dict):
        raise ValueError("axle calibration must be a mapping")
    r2w = data.get("wheel_r2")
    r2d = data.get("damper_r2")
    r2dw = data.get("damper_wheel_r2")
    dpw = data.get("damper_per_wheel")
    return AxleShockCalibration(
        wheel_per_sensor=float(data["wheel_per_sensor"]),
        damper_per_sensor=float(data["damper_per_sensor"]),
        damper_per_wheel_fit=float(dpw) if dpw is not None else None,
        wheel_r2=float(r2w) if r2w is not None else None,
        damper_r2=float(r2d) if r2d is not None else None,
        damper_wheel_r2=float(r2dw) if r2dw is not None else None,
        points=[[float(v) for v in row] for row in (data.get("points") or [])],
        excluded_points=[int(i) for i in (data.get("excluded_points") or [])],
    )


def _profile_to_dict(profile: VehicleProfile) -> dict:
    """Serialize a VehicleProfile to a dict suitable for YAML output."""
    data: dict[str, Any] = {
        "name": profile.name,
        "channel_names": dict(profile.channel_names),
        "motion_ratios": {
            "front_left": profile.motion_ratios.front_left,
            "front_right": profile.motion_ratios.front_right,
            "rear_left": profile.motion_ratios.rear_left,
            "rear_right": profile.motion_ratios.rear_right,
        },
    }
    if profile.shock_calibration is not None:
        data["shock_calibration"] = {
            "front": _axle_calibration_to_dict(profile.shock_calibration.front),
            "rear": _axle_calibration_to_dict(profile.shock_calibration.rear),
        }
    if profile.track_widths is not None:
        data["track_width_mm"] = {
            "front": float(profile.track_widths.front_mm),
            "rear": float(profile.track_widths.rear_mm),
        }
    return data


def _profile_from_dict(data: dict) -> VehicleProfile:
    """Deserialize a VehicleProfile from a YAML dict.

    Missing channel keys are filled from DEFAULT_CHANNEL_NAMES for forward
    compatibility — new canonical keys added in future versions won't break
    existing profile files.

    ``motion_ratios`` (pot/wheel) default to ``1 / wheel_per_sensor`` of
    ``shock_calibration`` when that section is present, and to the Toyota 86
    values otherwise.
    Explicit ``motion_ratios`` entries override the derived values.
    """
    channel_names = DEFAULT_CHANNEL_NAMES.copy()
    if "channel_names" in data:
        channel_names.update(data["channel_names"])

    from .suspension import MotionRatios, ShockCalibration

    shock_calibration: ShockCalibration | None = None
    cal_data = data.get("shock_calibration")
    if isinstance(cal_data, dict):
        try:
            shock_calibration = ShockCalibration(
                front=_axle_calibration_from_dict(cal_data.get("front")),
                rear=_axle_calibration_from_dict(cal_data.get("rear")),
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            shock_calibration = None

    if shock_calibration is not None:
        defaults = shock_calibration.pot_wheel_ratios()
    else:
        defaults = MotionRatios.toyota_86_zn6()

    ratios_data = data.get("motion_ratios") or {}
    motion_ratios = MotionRatios(
        front_left=float(ratios_data.get("front_left", defaults.front_left)),
        front_right=float(ratios_data.get("front_right", defaults.front_right)),
        rear_left=float(ratios_data.get("rear_left", defaults.rear_left)),
        rear_right=float(ratios_data.get("rear_right", defaults.rear_right)),
    )

    track_widths = None
    tw_data = data.get("track_width_mm")
    if isinstance(tw_data, dict):
        from .roll_gradient import TrackWidths

        try:
            track_widths = TrackWidths(front_mm=tw_data["front"], rear_mm=tw_data["rear"])
        except (KeyError, TypeError, ValueError):
            track_widths = None

    return VehicleProfile(
        name=data.get("name", "Unknown"),
        channel_names=channel_names,
        motion_ratios=motion_ratios,
        shock_calibration=shock_calibration,
        track_widths=track_widths,
    )


def profile_key(logger_id: str, vehicle: str | None = None) -> str:
    """Build the lookup key for a logger ID and optional vehicle name.

    Parameters
    ----------
    logger_id : str
        Logger serial number.
    vehicle : str or None
        Vehicle name used to disambiguate loggers shared between cars.

    Returns
    -------
    str
        ``"<logger_id>"`` or ``"<logger_id>/<vehicle>"``.
    """
    logger_id = str(logger_id).strip()
    if vehicle:
        return f"{logger_id}{PROFILE_KEY_SEPARATOR}{vehicle.strip()}"
    return logger_id


def split_profile_key(key: str) -> tuple[str, str | None]:
    """Inverse of :func:`profile_key`."""
    if PROFILE_KEY_SEPARATOR in key:
        logger_id, vehicle = key.split(PROFILE_KEY_SEPARATOR, 1)
        return logger_id, vehicle or None
    return key, None


def _parse_profiles_yaml(data: Any) -> tuple[dict[str, VehicleProfile], dict[str, VehicleProfile]]:
    """Parse the YAML document into (profile definitions, resolved logger map).

    The logger map is keyed by :func:`profile_key` strings. Each ``logger_map``
    value is either a profile id (string) or a mapping with an optional
    ``default`` profile id and a ``by_vehicle`` mapping of vehicle name ->
    profile id.
    """
    if not isinstance(data, dict):
        return {}, {}

    profiles_section = data.get("profiles", {})
    if not isinstance(profiles_section, dict):
        profiles_section = {}

    parsed_profiles: dict[str, VehicleProfile] = {}
    for profile_id, profile_data in profiles_section.items():
        if isinstance(profile_data, dict):
            parsed_profiles[str(profile_id)] = _profile_from_dict(profile_data)

    logger_map = data.get("logger_map", {})
    if not isinstance(logger_map, dict):
        logger_map = {}

    result: dict[str, VehicleProfile] = {}
    for logger_id, entry in logger_map.items():
        logger_id_str = str(logger_id)
        if isinstance(entry, dict):
            default_id = entry.get("default")
            if default_id in parsed_profiles:
                result[profile_key(logger_id_str)] = parsed_profiles[default_id]
            by_vehicle = entry.get("by_vehicle", {})
            if isinstance(by_vehicle, dict):
                for vehicle, pid in by_vehicle.items():
                    if pid in parsed_profiles and vehicle is not None:
                        result[profile_key(logger_id_str, str(vehicle))] = parsed_profiles[pid]
        elif entry in parsed_profiles:
            result[profile_key(logger_id_str)] = parsed_profiles[entry]

    return parsed_profiles, result


def _read_profiles_yaml(path: Path) -> Any:
    """Read a YAML file, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _load_profiles_file(path: Path) -> dict[str, VehicleProfile]:
    """Parse a two-tier YAML profiles file.

    Returns a dict keyed by profile key (logger ID, or ``logger ID/vehicle``
    for vehicle-specific entries), with each value being the resolved
    VehicleProfile from the profiles section.

    Parameters
    ----------
    path : Path
        Path to the YAML profiles file.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from profile key strings to VehicleProfile instances.
    """
    _, resolved = _parse_profiles_yaml(_read_profiles_yaml(path))
    return resolved


def _load_profile_definitions_file(path: Path) -> dict[str, VehicleProfile]:
    """Parse only the ``profiles`` section of a YAML profiles file (keyed by profile id)."""
    definitions, _ = _parse_profiles_yaml(_read_profiles_yaml(path))
    return definitions


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


def _builtin_profiles_candidates() -> list[Path]:
    """Candidate locations of the builtin profiles YAML, in priority order."""
    candidates: list[Path] = []
    # Try importlib.resources first (normal Python installs)
    try:
        ref = importlib.resources.files("motorsports_data_notebook.data").joinpath(
            "builtin_profiles.yaml"
        )
        with importlib.resources.as_file(ref) as path:
            candidates.append(Path(path))
    except Exception:
        pass
    # Fallback: relative to this file
    candidates.append(Path(__file__).parent / "data" / "builtin_profiles.yaml")
    # Fallback: PyInstaller _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(
            Path(meipass) / "motorsports_data_notebook" / "data" / "builtin_profiles.yaml"
        )
    return candidates


def load_builtin_profiles() -> dict[str, VehicleProfile]:
    """Load profiles shipped with the package.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from profile key strings (logger ID or ``logger ID/vehicle``)
        to VehicleProfile instances.
    """
    for path in _builtin_profiles_candidates():
        result = _load_profiles_file(path)
        if result:
            return result
    return {}


def load_profile_definitions() -> dict[str, VehicleProfile]:
    """Load all profile definitions keyed by profile id (e.g. ``"kk_f"``).

    User definitions override builtin ones with the same id.
    """
    definitions: dict[str, VehicleProfile] = {}
    for path in _builtin_profiles_candidates():
        builtin = _load_profile_definitions_file(path)
        if builtin:
            definitions.update(builtin)
            break
    definitions.update(_load_profile_definitions_file(_get_user_profiles_path()))
    return definitions


def load_user_profiles() -> dict[str, VehicleProfile]:
    """Load profiles from the user's config directory.

    Returns
    -------
    dict[str, VehicleProfile]
        Mapping from profile key strings to VehicleProfile instances.
        Empty dict if file doesn't exist or is invalid.
    """
    return _load_profiles_file(_get_user_profiles_path())


def save_user_profiles(profiles: dict[str, VehicleProfile]) -> None:
    """Save profiles to the user's config directory.

    Parameters
    ----------
    profiles : dict[str, VehicleProfile]
        Mapping from profile key strings (see :func:`profile_key`) to
        VehicleProfile instances.
    """
    path = _get_user_profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group profiles by identity to avoid duplicating profile definitions
    # Use profile name as the dedup key
    profile_defs: dict[str, dict] = {}
    profile_id_map: dict[str, str] = {}  # name -> profile_id

    logger_map: dict[str, Any] = {}

    for key, profile in profiles.items():
        # Create a stable profile_id from the name
        profile_id = profile.name.lower().replace(" ", "_")
        if profile_id not in profile_defs:
            profile_defs[profile_id] = _profile_to_dict(profile)
            profile_id_map[profile.name] = profile_id
        pid = profile_id_map[profile.name]

        logger_id, vehicle = split_profile_key(key)
        existing = logger_map.get(logger_id)
        if vehicle is None:
            if isinstance(existing, dict):
                existing["default"] = pid
            else:
                logger_map[logger_id] = pid
        else:
            if not isinstance(existing, dict):
                entry: dict[str, Any] = {}
                if isinstance(existing, str):
                    entry["default"] = existing
                logger_map[logger_id] = entry
                existing = entry
            existing.setdefault("by_vehicle", {})[vehicle] = pid

    data = {
        "profiles": profile_defs,
        "logger_map": logger_map,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _lookup_vehicle(
    profiles: dict[str, VehicleProfile], logger_id: str, vehicle: str
) -> VehicleProfile | None:
    """Case-insensitive lookup of a vehicle-specific profile key."""
    wanted = profile_key(logger_id, vehicle).lower()
    for key, profile in profiles.items():
        if key.lower() == wanted:
            return profile
    return None


def get_profile_for_logger(logger_id: str, vehicle: str | None = None) -> VehicleProfile | None:
    """Look up a vehicle profile by logger ID and optional vehicle name.

    Resolution order: vehicle-specific user profile, vehicle-specific builtin
    profile, then the logger's default user profile, then the default builtin
    profile. Vehicle names are matched case-insensitively.

    Parameters
    ----------
    logger_id : str
        The logger serial number.
    vehicle : str or None, optional
        Vehicle name (from file metadata or file name) used to disambiguate
        a logger shared between cars.

    Returns
    -------
    VehicleProfile | None
        The matching profile, or None if not found.
    """
    logger_id = str(logger_id)
    user_profiles = load_user_profiles()
    builtin_profiles = load_builtin_profiles()

    if vehicle:
        for source in (user_profiles, builtin_profiles):
            match = _lookup_vehicle(source, logger_id, vehicle)
            if match is not None:
                return match

    for source in (user_profiles, builtin_profiles):
        if logger_id in source:
            return source[logger_id]

    return None


def save_profile_for_logger(
    logger_id: str, profile: VehicleProfile, vehicle: str | None = None
) -> None:
    """Save a profile for a logger ID (and optional vehicle) to the user's config.

    Merges with existing user profiles.

    Parameters
    ----------
    logger_id : str
        The logger serial number.
    profile : VehicleProfile
        The profile to save.
    vehicle : str or None, optional
        Vehicle name. When given, the profile is saved as the vehicle-specific
        entry for that logger; otherwise as the logger's default.
    """
    profiles = load_user_profiles()
    profiles[profile_key(logger_id, vehicle)] = profile
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


def _vehicle_from_file_name(file_name: str) -> str | None:
    """Extract the vehicle token from a Race Studio 3 file name.

    Race Studio names exports ``<Racer>_<Vehicle>_<Track>_<Session>_a_<NNNN>``
    (e.g. ``CMD_KK-F_Suzuka West_Generic testing_a_0351.xrz``). Returns the
    second underscore-delimited token, or None when the name does not look
    like that pattern.
    """
    stem = Path(file_name).name
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    vehicle = parts[1].strip()
    return vehicle or None


def get_vehicle_name(log: LogFile, file_name: str | None = None) -> str | None:
    """Extract the vehicle name used to disambiguate shared loggers.

    Uses the ``Vehicle`` metadata field when set (Race Studio writes it to
    .xrk files). Falls back to the vehicle token of the Race Studio file name
    for AIM files whose metadata field is empty (typical for .xrz exports).

    Parameters
    ----------
    log : LogFile
        A loaded telemetry log file.
    file_name : str or None, optional
        File name to use for the fallback when the log has none (e.g. it was
        loaded from bytes). ``log.file_name`` is used when available.

    Returns
    -------
    str | None
        The vehicle name, or None if not available.
    """
    metadata = getattr(log, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("Vehicle", "Vehicle Name", "vehicle_name", "vehicle"):
        val = metadata.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Only AIM files follow the Race Studio naming scheme; "present" means the
    # same thing as in get_logger_id (an empty logger ID counts as absent).
    if not (metadata.get("Logger ID") or metadata.get("logger_id")):
        return None
    log_file_name = getattr(log, "file_name", None)
    if isinstance(log_file_name, str) and log_file_name:
        vehicle = _vehicle_from_file_name(log_file_name)
        if vehicle:
            return vehicle
    if file_name:
        return _vehicle_from_file_name(file_name)
    return None


def get_profile_for_log(log: LogFile) -> VehicleProfile | None:
    """Resolve the vehicle profile for a loaded session (logger ID + vehicle name).

    Parameters
    ----------
    log : LogFile
        A loaded telemetry log file.

    Returns
    -------
    VehicleProfile | None
        The matching profile, or None if the logger ID is unknown or unmapped.
    """
    logger_id = get_logger_id(log)
    if not logger_id:
        return None
    return get_profile_for_logger(logger_id, get_vehicle_name(log))


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
