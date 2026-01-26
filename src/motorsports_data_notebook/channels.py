"""Channel extraction and interpolation utilities.

This module provides functions for efficiently extracting and aligning channel data
from log files without the expensive get_channels_as_table() merge operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from libxrk.base import LogFile


def get_lap_channels(
    log: "LogFile",
    channel_names: list[str],
    start_time: int,
    end_time: int,
) -> dict[str, pa.Table]:
    """Extract channels filtered to a time range, each at their native timebase.

    This is a fast alternative to get_channels_as_table() when you only need
    specific channels and don't need them merged onto a common timebase.

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    channel_names : list[str]
        Names of channels to extract (e.g., ["GPS Latitude", "GPS Longitude", "speed_kmh"]).
    start_time : int
        Start timestamp in milliseconds (inclusive).
    end_time : int
        End timestamp in milliseconds (exclusive).

    Returns
    -------
    dict[str, pa.Table]
        Dictionary mapping channel names to PyArrow tables, same format as log.channels.
        Each table has columns: 'timecodes' (int64) and the channel value column.

    Examples
    --------
    >>> channels = get_lap_channels(log, ["GPS Latitude", "GPS Longitude"], 60000, 120000)
    >>> lat = channels["GPS Latitude"].column("GPS Latitude").to_numpy()
    """
    result = {}
    for name in channel_names:
        if name not in log.channels:
            raise KeyError(
                f"Channel '{name}' not found in log. Available: {list(log.channels.keys())}"
            )

        table = log.channels[name]
        timecodes = table.column("timecodes")

        # Create filter mask: start_time <= timecodes < end_time
        mask = pc.and_(
            pc.greater_equal(timecodes, start_time),
            pc.less(timecodes, end_time),
        )

        # Filter the table
        result[name] = table.filter(mask)

    return result


def get_best_lap(laps_df: pd.DataFrame) -> pd.Series:
    """Find the best (fastest) lap by duration.

    Excludes the first and last laps to avoid pit entry/exit laps.

    Parameters
    ----------
    laps_df : pandas.DataFrame
        Laps table with 'start_time', 'end_time' columns.

    Returns
    -------
    pandas.Series
        The row corresponding to the best lap.
    """
    if not {"start_time", "end_time"}.issubset(laps_df.columns):
        raise ValueError("Expected start_time and end_time columns in laps table")

    if len(laps_df) <= 2:
        raise ValueError("Need at least 3 laps to exclude first and last laps")

    # Exclude first and last laps
    laps_subset = laps_df.iloc[1:-1].copy()

    laps_subset["lap_duration_ms"] = laps_subset["end_time"] - laps_subset["start_time"]
    best_idx = laps_subset["lap_duration_ms"].idxmin()
    best_lap: pd.Series = laps_subset.loc[best_idx]  # type: ignore[assignment]
    return best_lap


def get_best_lap_channels(
    log: "LogFile",
    laps: pd.DataFrame,
    channel_names: list[str],
) -> tuple[pd.Series, dict[str, pa.Table]]:
    """Extract best lap info and channels at their native timebases.

    This is a fast alternative to get_best_lap_data() that avoids the expensive
    get_channels_as_table() merge operation.

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    laps : pd.DataFrame
        Laps table with 'start_time', 'end_time' columns.
    channel_names : list[str]
        Names of channels to extract.

    Returns
    -------
    tuple[pd.Series, dict[str, pa.Table]]
        (best_lap, channels) - best lap info and channel tables for that lap.
        Each channel table has columns: 'timecodes' (int64) and the channel value.

    Examples
    --------
    >>> best_lap, channels = get_best_lap_channels(log, laps, ["GPS Latitude", "GPS Longitude", "speed_kmh"])
    >>> lat = channels["GPS Latitude"].column("GPS Latitude").to_numpy()
    """
    best_lap = get_best_lap(laps)
    start_ts = best_lap["start_time"]
    end_ts = best_lap["end_time"]

    channels = get_lap_channels(log, channel_names, start_ts, end_ts)
    return best_lap, channels


def interpolate_channels(
    channels: dict[str, pa.Table],
    reference_channel: str,
) -> dict[str, pa.Table]:
    """Interpolate all channels to a reference channel's timebase.

    Use this to align channels with different sample rates to a common timebase.
    For example, to align brake pressure (sampled at 100Hz) to GPS timestamps (10Hz).

    The reference channel is copied unchanged. All other channels are linearly
    interpolated to match the reference channel's timecodes.

    Parameters
    ----------
    channels : dict[str, pa.Table]
        Dictionary of channel tables (as returned by get_lap_channels).
    reference_channel : str
        Name of channel whose timebase to use as the target.

    Returns
    -------
    dict[str, pa.Table]
        New dictionary with all channels interpolated to the reference timebase.
        Each table has columns: 'timecodes' (int64) and the channel value.

    Examples
    --------
    >>> best_lap, channels = get_best_lap_channels(log, laps,
    ...     ["GPS Latitude", "GPS Longitude", "speed_kmh", "BrakePress"])
    >>> aligned = interpolate_channels(channels, reference_channel="GPS Latitude")
    >>> # Now all channels share the same timecodes as GPS Latitude
    """
    if reference_channel not in channels:
        raise KeyError(
            f"Reference channel '{reference_channel}' not found. "
            f"Available: {list(channels.keys())}"
        )

    # Get reference timebase
    ref_table = channels[reference_channel]
    ref_times = ref_table.column("timecodes").to_numpy()

    result: dict[str, pa.Table] = {}

    for name, table in channels.items():
        if name == reference_channel:
            # Copy reference channel unchanged
            result[name] = table
        else:
            # Interpolate to reference timebase
            source_times = table.column("timecodes").to_numpy()
            source_values = table.column(name).to_numpy()
            interpolated = np.interp(ref_times, source_times, source_values)

            # Create new table with reference timecodes
            result[name] = pa.table(
                {
                    "timecodes": pa.array(ref_times, type=pa.int64()),
                    name: pa.array(interpolated),
                }
            )

    return result


def get_top_laps(laps: pd.DataFrame, threshold_pct: float = 1.03) -> pd.DataFrame:
    """Get laps within threshold percentage of best lap time.

    Excludes first and last laps, and laps with zero or negative duration.

    Parameters
    ----------
    laps : pd.DataFrame
        Laps table with 'lap_time' column (as Timedelta).
    threshold_pct : float, default=1.03
        Threshold as multiplier (e.g., 1.03 for within 103% of best).

    Returns
    -------
    pd.DataFrame
        DataFrame of qualifying laps.

    Examples
    --------
    >>> top_laps = get_top_laps(laps, threshold_pct=1.03)
    >>> print(f"Using {len(top_laps)} laps for analysis")
    """
    if "lap_time" not in laps.columns:
        raise ValueError("Expected lap_time column in laps table")

    # Exclude first/last laps and zero-duration laps
    valid_laps: pd.DataFrame = laps[laps["lap_time"] > pd.Timedelta(0)][1:-1].copy()

    if len(valid_laps) == 0:
        return valid_laps

    best_lap_time = valid_laps["lap_time"].min()
    threshold_time = best_lap_time * threshold_pct
    top_laps: pd.DataFrame = valid_laps[valid_laps["lap_time"] <= threshold_time]

    return top_laps
