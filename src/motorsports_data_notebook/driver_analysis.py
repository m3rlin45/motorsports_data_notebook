"""Driver performance analysis functions.

This module provides functions for analyzing driver behavior and performance,
such as throttle acceptance during corner exits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from ._util import infer_channel_scale as _infer_channel_scale
from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from .corners import Corner


def prepare_throttle_acceptance(
    throttle: np.ndarray,
    lateral_g: np.ndarray,
    throttle_threshold: float | None = None,
    smoothing_window: int = 25,
) -> tuple[np.ndarray, float]:
    """Compute per-lap shared data for throttle acceptance.

    Call once per lap, then pass results to find_throttle_acceptance_from_arrays()
    for each corner in that lap.

    Parameters
    ----------
    throttle : np.ndarray
        Full-lap throttle position values.
    lateral_g : np.ndarray
        Full-lap lateral acceleration values.
    throttle_threshold : float or None, default=None
        Throttle value to consider as "full throttle".
        If None, auto-detected from data scale (98% of scale).
    smoothing_window : int, default=25
        Number of samples for rolling average smoothing of lateral G.

    Returns
    -------
    tuple[np.ndarray, float]
        (smoothed_lateral_g, effective_threshold)
    """
    if throttle_threshold is None:
        throttle_threshold = 0.98 * _infer_channel_scale(throttle)

    # Smooth lateral G (convolution-based rolling average on absolute value)
    abs_lateral = np.abs(lateral_g)
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed_lateral_g = np.convolve(abs_lateral, kernel, mode="same")

    # Adapt threshold for sensors that don't reach 100%
    max_throttle = float(throttle.max())
    if 90.0 <= max_throttle < 100.0:
        effective_threshold = max_throttle * (throttle_threshold / 100.0)
    else:
        effective_threshold = throttle_threshold

    return smoothed_lateral_g, effective_threshold


def find_throttle_acceptance_from_arrays(
    distance: np.ndarray,
    timecodes: np.ndarray,
    throttle: np.ndarray,
    lateral_g: np.ndarray,
    corner: "Corner",
    *,
    smoothed_lateral_g: np.ndarray | None = None,
    effective_threshold: float | None = None,
    throttle_threshold: float | None = None,
    sustain_time_ms: float = 500.0,
    smoothing_window: int = 25,
) -> dict | None:
    """Find throttle acceptance from numpy arrays.

    If smoothed_lateral_g / effective_threshold are provided (from
    prepare_throttle_acceptance), uses them directly — avoids redundant
    per-call smoothing. Otherwise computes inline (for standalone use).

    Parameters
    ----------
    distance : np.ndarray
        Full-lap distance values (meters).
    timecodes : np.ndarray
        Full-lap timecodes (milliseconds).
    throttle : np.ndarray
        Full-lap throttle position values.
    lateral_g : np.ndarray
        Full-lap lateral acceleration values.
    corner : Corner
        Corner object defining the corner boundaries.
    smoothed_lateral_g : np.ndarray or None
        Pre-computed smoothed |lateral_g| from prepare_throttle_acceptance().
    effective_threshold : float or None
        Pre-computed effective throttle threshold from prepare_throttle_acceptance().
    throttle_threshold : float or None, default=None
        Raw throttle threshold (used only when effective_threshold is None).
    sustain_time_ms : float, default=500.0
        Time in milliseconds that throttle must be sustained.
    smoothing_window : int, default=25
        Number of samples for smoothing (used only when smoothed_lateral_g is None).

    Returns
    -------
    dict or None
        Dictionary with throttle_acceptance_pct, lateral_g_at_throttle,
        peak_lateral_g, full_throttle_dist. None if not sustained.
    """
    # Compute shared data if not provided
    if smoothed_lateral_g is None or effective_threshold is None:
        smoothed_lateral_g, effective_threshold = prepare_throttle_acceptance(
            throttle,
            lateral_g,
            throttle_threshold=throttle_threshold,
            smoothing_window=smoothing_window,
        )

    # Peak lateral G in corner region
    c_si = np.searchsorted(distance, corner.start_dist)
    c_ei = np.searchsorted(distance, corner.end_dist, side="right")
    if c_si >= c_ei:
        return None

    peak_lateral_g = float(smoothed_lateral_g[c_si:c_ei].max())
    if peak_lateral_g < 0.1:
        return None

    # Exit zone: apex to corner end
    a_si = np.searchsorted(distance, corner.apex_dist)
    if a_si >= c_ei:
        return None

    exit_throttle = throttle[a_si:c_ei]
    exit_timecodes = timecodes[a_si:c_ei]
    exit_smoothed = smoothed_lateral_g[a_si:c_ei]
    exit_distance = distance[a_si:c_ei]
    n_exit = len(exit_throttle)

    above = exit_throttle >= effective_threshold
    for j in range(n_exit):
        if not above[j]:
            continue
        start_time = exit_timecodes[j]
        end_time = start_time + sustain_time_ms
        sustained = True
        for k in range(j + 1, n_exit):
            if exit_timecodes[k] > end_time:
                break
            if not above[k]:
                sustained = False
                break
        if sustained:
            ta_pct = (exit_smoothed[j] / peak_lateral_g) * 100
            return {
                "throttle_acceptance_pct": float(ta_pct),
                "lateral_g_at_throttle": float(exit_smoothed[j]),
                "peak_lateral_g": peak_lateral_g,
                "full_throttle_dist": float(exit_distance[j]),
            }

    return None


def find_throttle_acceptance(
    lap_data: pd.DataFrame,
    corner: "Corner",
    channel_names: dict,
    throttle_threshold: float | None = None,
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
    throttle_threshold : float or None, default=None
        Throttle value to consider as "full throttle".
        If None, auto-detected from data scale (98% of scale).
        When the sensor's maximum output is 90-100% of scale, the threshold
        is automatically scaled proportionally (e.g. 98% threshold with a
        98.7% max sensor becomes 98.7 * 98/100 = 96.7%).
    sustain_time_ms : float, default=500.0
        Time in milliseconds that throttle must be sustained to count as "maintained".
    smoothing_window : int, default=25
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
    _validate_channel_names(channel_names, ["throttle", "lateral_g"], "find_throttle_acceptance")

    throttle_col = channel_names["throttle"]
    lateral_g_col = channel_names["lateral_g"]

    return find_throttle_acceptance_from_arrays(
        distance=lap_data["distance_m"].to_numpy(),
        timecodes=lap_data["timecodes"].to_numpy(),
        throttle=lap_data[throttle_col].to_numpy(),
        lateral_g=lap_data[lateral_g_col].to_numpy(),
        corner=corner,
        throttle_threshold=throttle_threshold,
        sustain_time_ms=sustain_time_ms,
        smoothing_window=smoothing_window,
    )
