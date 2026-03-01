"""Session loading utilities for desktop app (no IPython dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from typing import Union

    from libxrk.base import LogFile as AimLogFile
    from libibt.base import LogFile as IbtLogFile

    LogFile = Union[AimLogFile, IbtLogFile]


def load_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from a telemetry file.

    Supports AIM (XRK/XRZ) and iRacing (IBT) file formats.

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
    from motorsports_data_notebook._util import clean_laps_table, detect_file_type

    if detect_file_type(file_data) == "ibt":
        log = _load_ibt_session(file_data)
    else:
        log = _load_aim_session(file_data)

    log.laps = clean_laps_table(log.laps, log.channels)
    return log


def _load_aim_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from an AIM XRK/XRZ file."""
    from libxrk import aim_xrk

    log = aim_xrk(file_data)

    # Check if GPS Speed channel exists
    has_gps_speed = "GPS Speed" in log.channels

    if has_gps_speed:
        gps_speed_table = log.channels["GPS Speed"]
        timecodes = gps_speed_table.column("timecodes")
        gps_speed = gps_speed_table.column("GPS Speed")

        # Add speed_kmh channel
        speed_kmh = pc.multiply(gps_speed, 3.6)
        speed_kmh_table = pa.table({"timecodes": timecodes, "speed_kmh": speed_kmh})
        log.channels["speed_kmh"] = speed_kmh_table

    # Compute lap_time for laps table (end_time - start_time in ms -> timedelta)
    laps_table = log.laps
    start_times = laps_table.column("start_time")
    end_times = laps_table.column("end_time")
    lap_time_ms = pc.subtract(end_times, start_times)
    # Convert to duration in milliseconds
    lap_time_duration = pc.multiply(lap_time_ms, 1000000)  # ms to nanoseconds
    lap_time_duration = lap_time_duration.cast(pa.duration("ns"))
    log.laps = laps_table.append_column("lap_time", lap_time_duration)

    # Compute distance_m for each lap
    if has_gps_speed:
        timecodes_np = timecodes.to_numpy()
        gps_speed_np = gps_speed.to_numpy()
        start_times_np = start_times.to_numpy()
        end_times_np = end_times.to_numpy()

        distance_m = np.zeros(len(timecodes_np))

        for i in range(len(start_times_np)):
            start_time = start_times_np[i]
            end_time = end_times_np[i]

            lap_mask = (timecodes_np >= start_time) & (timecodes_np <= end_time)
            lap_indices = np.where(lap_mask)[0]

            if len(lap_indices) > 0:
                lap_timecodes = timecodes_np[lap_indices]
                lap_speed = gps_speed_np[lap_indices]
                distance_values = _compute_lap_distance(lap_timecodes, lap_speed)
                distance_m[lap_indices] = distance_values

        # Add distance_m channel
        distance_table = pa.table({"timecodes": timecodes, "distance_m": distance_m})
        log.channels["distance_m"] = distance_table

    return log


def _load_ibt_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from an iRacing IBT file."""
    from libibt import ibt

    log = ibt(file_data)

    # iRacing Speed channel is in m/s — add speed_kmh
    has_speed = "Speed" in log.channels
    if has_speed:
        speed_table = log.channels["Speed"]
        timecodes = speed_table.column("timecodes")
        speed_ms = speed_table.column("Speed")

        speed_kmh = pc.multiply(speed_ms, 3.6)
        speed_kmh_table = pa.table({"timecodes": timecodes, "speed_kmh": speed_kmh})
        log.channels["speed_kmh"] = speed_kmh_table

    # Compute per-lap distance_m from Speed (same approach as AIM).
    # iRacing's LapDist wraps at the S/F line, causing discontinuities
    # when lap boundaries don't perfectly align with the wrap point.
    if has_speed:
        timecodes_np = timecodes.to_numpy()
        speed_np = speed_ms.to_numpy(zero_copy_only=False).astype(np.float64)

        laps_table_tmp = log.laps
        start_times_np = laps_table_tmp.column("start_time").to_numpy()
        end_times_np = laps_table_tmp.column("end_time").to_numpy()

        distance_m = np.zeros(len(timecodes_np))
        for i in range(len(start_times_np)):
            lap_mask = (timecodes_np >= start_times_np[i]) & (timecodes_np < end_times_np[i])
            lap_indices = np.where(lap_mask)[0]
            if len(lap_indices) > 0:
                distance_m[lap_indices] = _compute_lap_distance(
                    timecodes_np[lap_indices], speed_np[lap_indices]
                )

        distance_table = pa.table({"timecodes": timecodes, "distance_m": distance_m})
        log.channels["distance_m"] = distance_table

    # Compute lap_time for laps table (same logic as AIM)
    laps_table = log.laps
    start_times = laps_table.column("start_time")
    end_times = laps_table.column("end_time")
    lap_time_ms = pc.subtract(end_times, start_times)
    lap_time_duration = pc.multiply(lap_time_ms, 1000000)  # ms to nanoseconds
    lap_time_duration = lap_time_duration.cast(pa.duration("ns"))
    log.laps = laps_table.append_column("lap_time", lap_time_duration)

    return log


def _compute_lap_distance(timecodes: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Compute cumulative distance for a single lap.

    Parameters
    ----------
    timecodes : np.ndarray
        Timestamps in milliseconds.
    speed : np.ndarray
        Speed in m/s.

    Returns
    -------
    np.ndarray
        Cumulative distance in meters.
    """
    if len(timecodes) < 2:
        return np.zeros(len(timecodes))

    # Convert timecodes to seconds
    time_seconds = timecodes / 1000.0

    # Compute time deltas
    dt = np.diff(time_seconds)

    # Average speed between points
    avg_speed = (speed[:-1] + speed[1:]) / 2

    # Distance increments
    d_distance = avg_speed * dt

    # Cumulative distance
    distance = np.zeros(len(timecodes))
    distance[1:] = np.cumsum(d_distance)

    return distance
