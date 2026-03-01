"""Channel extraction and lap filtering utilities.

This module provides functions for finding best/top laps and extracting channel data
using libxrk 0.5.0's built-in filtering and resampling methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pyarrow as pa

    from ._types import LogFile


def get_best_lap(laps_df: pd.DataFrame) -> pd.Series:
    """Find the best (fastest) lap by duration.

    If a ``lap_type`` column is present (IBT files), filters to ``"full"`` laps.
    Otherwise falls back to excluding the first and last laps (AIM files).

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

    if "lap_type" in laps_df.columns:
        laps_subset = laps_df[laps_df["lap_type"] == "full"].copy()
        if len(laps_subset) == 0:
            raise ValueError("No full laps found")
    else:
        if len(laps_df) <= 2:
            raise ValueError("Need at least 3 laps to exclude first and last laps")
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

    Uses libxrk 0.5.0's filter_by_lap and select_channels methods for efficient
    extraction without expensive merge operations.

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    laps : pd.DataFrame
        Laps table with 'start_time', 'end_time', 'num' columns.
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
    lap_num = int(best_lap["num"])
    channels = log.filter_by_lap(lap_num).select_channels(channel_names).channels
    return best_lap, channels


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

    positive = laps[laps["lap_time"] > pd.Timedelta(0)]
    if "lap_type" in laps.columns:
        valid_laps: pd.DataFrame = positive[positive["lap_type"] == "full"].copy()
    else:
        # Fallback for AIM files: exclude first/last laps
        valid_laps = positive.iloc[1:-1].copy()

    if len(valid_laps) == 0:
        return valid_laps

    best_lap_time = valid_laps["lap_time"].min()
    threshold_time = best_lap_time * threshold_pct
    top_laps: pd.DataFrame = valid_laps[valid_laps["lap_time"] <= threshold_time]

    return top_laps
