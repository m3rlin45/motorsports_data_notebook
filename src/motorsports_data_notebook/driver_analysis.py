"""Driver performance analysis functions.

This module provides functions for analyzing driver behavior and performance,
such as throttle acceptance during corner exits.
"""

from typing import cast

import pandas as pd

from .corners import Corner


def _validate_channel_names(channel_names: dict, required_keys: list[str], func_name: str) -> None:
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


def find_throttle_acceptance(
    lap_data: pd.DataFrame,
    corner: Corner,
    channel_names: dict,
    throttle_threshold: float = 98.0,
    sustain_time_ms: float = 500.0,
    smoothing_window: int = 25,
) -> dict | None:
    """
    Find the throttle acceptance for a corner exit.

    Throttle acceptance is the lateral G at which the driver reaches and maintains
    full throttle during corner exit, expressed as a percentage of the peak lateral G
    of the corner.

    Parameters
    ----------
    lap_data : pd.DataFrame
        Lap data with columns matching channel_names values plus 'distance_m' and 'timecodes'.
    corner : Corner
        Corner object defining the corner boundaries.
    channel_names : dict
        Channel name mapping. Required keys:
        - "throttle": Throttle position column name in lap_data (e.g., "PPS")
        - "lateral_g": Lateral acceleration column name in lap_data (e.g., "LateralAcc")
    throttle_threshold : float, default=98.0
        Throttle percentage to consider as "full throttle".
    sustain_time_ms : float, default=500.0
        Time in milliseconds that throttle must be sustained to count as "maintained".
    smoothing_window : int, default=10
        Number of samples for rolling average smoothing of lateral G.

    Returns
    -------
    dict or None
        Dictionary with:
        - throttle_acceptance_pct: Lateral G at full throttle as % of peak lateral G
        - lateral_g_at_throttle: Smoothed absolute lateral G when full throttle was reached
        - peak_lateral_g: Peak smoothed absolute lateral G in the corner
        - full_throttle_dist: Distance where sustained full throttle began
        Returns None if full throttle is not sustained within the exit zone.

    Raises
    ------
    KeyError
        If required keys are missing from channel_names.
    """
    # Validate required channel names
    _validate_channel_names(channel_names, ["throttle", "lateral_g"], "find_throttle_acceptance")

    throttle_col = channel_names["throttle"]
    lateral_g_col = channel_names["lateral_g"]

    # Apply smoothing to lateral G (rolling average on absolute value)
    lap_data = lap_data.copy()
    lap_data["LateralAcc_smooth"] = (
        lap_data[lateral_g_col]
        .abs()
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )

    # Get corner data for peak lateral G calculation
    corner_mask = (lap_data["distance_m"] >= corner.start_dist) & (
        lap_data["distance_m"] <= corner.end_dist
    )
    corner_data = lap_data[corner_mask]

    if len(corner_data) == 0:
        return None

    # Peak lateral G in corner (smoothed absolute value)
    peak_lateral_g = corner_data["LateralAcc_smooth"].max()

    if peak_lateral_g < 0.1:  # Skip if negligible lateral G
        return None

    # Get exit zone data (apex to corner end)
    exit_mask = (lap_data["distance_m"] >= corner.apex_dist) & (
        lap_data["distance_m"] <= corner.end_dist
    )
    exit_data = lap_data[exit_mask].copy()

    if len(exit_data) == 0:
        return None

    # Find first point where throttle >= threshold and is sustained for sustain_time_ms
    exit_data = exit_data.sort_values("timecodes").reset_index(drop=True)

    for i in range(len(exit_data)):
        if cast(float, exit_data.loc[i, throttle_col]) >= throttle_threshold:
            start_time = cast(float, exit_data.loc[i, "timecodes"])
            end_time = start_time + sustain_time_ms

            # Check if throttle stays above threshold for sustain_time_ms
            timecodes = cast("pd.Series[float]", exit_data["timecodes"])
            sustain_mask = (timecodes >= start_time) & (timecodes <= end_time)
            sustain_data = exit_data[sustain_mask]

            if len(sustain_data) == 0:
                continue

            # Check if all points in the sustain window are above threshold
            if (sustain_data[throttle_col] >= throttle_threshold).all():
                # Found sustained full throttle - use smoothed lateral G
                lateral_g_at_throttle = exit_data.loc[i, "LateralAcc_smooth"]
                throttle_acceptance_pct = (lateral_g_at_throttle / peak_lateral_g) * 100

                return {
                    "throttle_acceptance_pct": throttle_acceptance_pct,
                    "lateral_g_at_throttle": lateral_g_at_throttle,
                    "peak_lateral_g": peak_lateral_g,
                    "full_throttle_dist": exit_data.loc[i, "distance_m"],
                }

    # Full throttle not sustained within exit zone
    return None
