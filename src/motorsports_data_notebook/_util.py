"""Internal utilities shared across modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from .corners import compute_lap_distance

if TYPE_CHECKING:
    from ._types import LogFile

# IBT file magic bytes: little-endian uint32 with value 2
_IBT_MAGIC = b"\x02\x00\x00\x00"


def infer_channel_scale(data: np.ndarray) -> float:
    """Infer whether channel data uses 0-1 or 0-100 scale.

    Parameters
    ----------
    data : np.ndarray
        Channel data array (e.g., throttle or brake values).

    Returns
    -------
    float
        ``1.0`` for 0-1 scale data, ``100.0`` for 0-100 scale data.
    """
    max_val = float(np.max(np.abs(data)))
    return 1.0 if max_val <= 1.5 else 100.0


def detect_file_type(file_data: str | bytes) -> Literal["aim", "ibt"]:
    """Detect telemetry file type from path or bytes.

    Parameters
    ----------
    file_data : str or bytes
        File path string or raw file bytes.

    Returns
    -------
    Literal["aim", "ibt"]
        ``"aim"`` for XRK/XRZ files, ``"ibt"`` for iRacing IBT files.
    """
    if isinstance(file_data, str):
        lower = file_data.lower()
        if lower.endswith(".ibt"):
            return "ibt"
        return "aim"

    # bytes: check IBT magic header
    if isinstance(file_data, (bytes, bytearray)) and len(file_data) >= 4:
        if file_data[:4] == _IBT_MAGIC:
            return "ibt"
    return "aim"


def validate_channel_names(channel_names: dict, required_keys: list[str], func_name: str) -> None:
    """Validate that required keys are present in channel_names dict.

    Parameters
    ----------
    channel_names : dict
        Channel name mapping from canonical names to actual channel names.
    required_keys : list[str]
        List of required keys that must be present.
    func_name : str
        Name of the calling function (for error messages).

    Raises
    ------
    KeyError
        If any required key is missing from channel_names.
    """
    missing = [key for key in required_keys if key not in channel_names]
    if missing:
        raise KeyError(
            f"{func_name}() requires channel_names to have keys: {required_keys}. "
            f"Missing: {missing}"
        )


def get_channel_unit(table: pa.Table, channel_name: str) -> str:
    """Get the unit string for a channel from its PyArrow field metadata.

    Parameters
    ----------
    table : pa.Table
        PyArrow table containing the channel.
    channel_name : str
        Name of the channel field.

    Returns
    -------
    str
        Unit string (e.g., "g", "kPa", "m/s"), or empty string if not available.
    """
    from libxrk import ChannelMetadata

    try:
        meta = ChannelMetadata.from_field(table.schema.field(channel_name))
        return meta.units
    except (KeyError, AttributeError):
        return ""


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------


def load_session(file_data: str | bytes) -> "LogFile":
    """Load and prepare session data from a telemetry file.

    Supports AIM (XRK/XRZ) and iRacing (IBT) file formats. Automatically
    detects the file type and dispatches to the appropriate loader.

    Adds derived columns:

    - speed_kmh: Speed in km/h
    - distance_m: Per-lap cumulative distance in meters
    - lap_time: Lap duration as timedelta (added to laps table)

    Parameters
    ----------
    file_data : str or bytes
        Path to the telemetry file, or bytes containing file data.

    Returns
    -------
    LogFile
        The enriched LogFile object with derived channels and lap_time column.
    """
    log: LogFile
    if detect_file_type(file_data) == "ibt":
        from libibt import ibt

        log = ibt(file_data)
    else:
        from libxrk import aim_xrk

        log = aim_xrk(file_data)
    return _post_process_session(log)


def _add_lap_time(log: "LogFile") -> None:
    """Compute lap_time column and append to log.laps (mutates in place)."""
    laps_table = log.laps
    start_times = laps_table.column("start_time")
    end_times = laps_table.column("end_time")
    lap_time_ms = pc.subtract(end_times, start_times)
    lap_time_ns = pc.multiply(lap_time_ms, 1000000)  # ms -> ns
    lap_time_duration = lap_time_ns.cast(pa.duration("ns"))
    log.laps = laps_table.append_column("lap_time", lap_time_duration)


def _add_distance_channel(
    log: "LogFile",
    speed_channel: str,
    timecodes: pa.ChunkedArray,
    speed_ms: pa.ChunkedArray,
) -> None:
    """Compute per-lap cumulative distance_m and add as a channel."""
    timecodes_np = timecodes.to_numpy()
    speed_np = speed_ms.to_numpy(zero_copy_only=False).astype(np.float64)

    start_times_np = log.laps.column("start_time").to_numpy()
    end_times_np = log.laps.column("end_time").to_numpy()

    distance_m = np.zeros(len(timecodes_np))
    for i in range(len(start_times_np)):
        lap_mask = (timecodes_np >= start_times_np[i]) & (timecodes_np < end_times_np[i])
        lap_indices = np.where(lap_mask)[0]
        if len(lap_indices) > 0:
            distance_m[lap_indices] = compute_lap_distance(
                timecodes_np[lap_indices], speed_np[lap_indices]
            )

    distance_table = pa.table({"timecodes": timecodes, "distance_m": distance_m})
    log.channels["distance_m"] = distance_table


def clean_laps(laps_table: pa.Table) -> pa.Table:
    """Remove lap 0 and deduplicate lap numbers (keeps longest).

    Parameters
    ----------
    laps_table : pa.Table
        Laps table with 'num', 'start_time', 'end_time' columns.

    Returns
    -------
    pa.Table
        Cleaned laps table.
    """
    if len(laps_table) == 0:
        return laps_table

    start_times = laps_table.column("start_time").to_numpy()
    end_times = laps_table.column("end_time").to_numpy()
    durations_ms = end_times - start_times
    lap_nums = laps_table.column("num").to_pylist()
    n = len(lap_nums)

    # Deduplicate — for each lap number, keep the longest; skip lap 0
    best_by_num: dict[int, tuple[int, float]] = {}
    for i in range(n):
        num = lap_nums[i]
        if num == 0:
            continue
        dur = float(durations_ms[i])
        if num not in best_by_num or dur > best_by_num[num][1]:
            best_by_num[num] = (i, dur)

    keep_indices = sorted(idx for idx, _ in best_by_num.values())
    if not keep_indices:
        return laps_table  # Don't filter if it would remove everything
    return laps_table.take(keep_indices)


def _post_process_session(log: "LogFile") -> "LogFile":
    """Add derived channels and clean laps for any telemetry format.

    Resolves the speed channel name via the profile system, then adds
    speed_kmh, distance_m, and lap_time columns. Filters to full laps
    if a ``lap_type`` column is present.
    """
    from .profiles import DEFAULT_CHANNEL_NAMES, get_logger_id, get_profile_for_logger

    # Resolve speed channel name from profile (or default)
    speed_channel = DEFAULT_CHANNEL_NAMES["gps_speed"]
    logger_id = get_logger_id(log)
    if logger_id is not None:
        profile = get_profile_for_logger(logger_id)
        if profile is not None:
            speed_channel = profile.channel_names.get("gps_speed", speed_channel)

    has_speed = speed_channel in log.channels
    if has_speed:
        speed_table = log.channels[speed_channel]
        timecodes = speed_table.column("timecodes")
        speed_ms = speed_table.column(speed_channel)

        speed_kmh = pc.multiply(speed_ms, 3.6)
        log.channels["speed_kmh"] = pa.table({"timecodes": timecodes, "speed_kmh": speed_kmh})

    _add_lap_time(log)

    if has_speed:
        _add_distance_channel(log, speed_channel, timecodes, speed_ms)

    # Filter to full laps if lap_type column exists (e.g. iRacing IBT)
    if "lap_type" in log.laps.column_names:
        log.laps = log.laps.filter(pc.equal(log.laps.column("lap_type"), "full"))

    log.laps = clean_laps(log.laps)
    return log
