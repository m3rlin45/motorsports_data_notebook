"""Tire grip analysis module.

Analyzes the relationship between total acceleration (combined lateral and
longitudinal G-forces) and tire pressure or temperature for each corner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import pyarrow as pa

from ._util import get_channel_unit as _get_channel_unit
from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from ._types import LogFile


@dataclass
class CornerTireGripData:
    """Tire grip data for a single corner.

    Attributes
    ----------
    corner_name : str
        Name of the corner ("FL", "FR", "RL", "RR").
    total_g : np.ndarray
        Total acceleration: sqrt(lateral_g² + inline_g²).
    tire_metric : np.ndarray
        Tire pressure (psi) or temperature (°F) values.
    mean_g : float
        Mean total G.
    mean_metric : float
        Mean tire metric value.
    std_g : float
        Standard deviation of total G.
    std_metric : float
        Standard deviation of tire metric.
    bucket_centers : np.ndarray
        Center of each bucket on the X axis (tire metric).
    bucket_values : np.ndarray
        Percentile of total_g per bucket.
    bucket_counts : np.ndarray
        Sample count per bucket.
    percentile : float
        Which percentile was computed (e.g. 99.9).
    """

    corner_name: str
    total_g: np.ndarray
    tire_metric: np.ndarray
    mean_g: float
    mean_metric: float
    std_g: float
    std_metric: float
    bucket_centers: np.ndarray
    bucket_values: np.ndarray
    bucket_counts: np.ndarray
    percentile: float


@dataclass
class TireGripResult:
    """Complete tire grip analysis for all four corners.

    Attributes
    ----------
    front_left : CornerTireGripData
        Front left corner data.
    front_right : CornerTireGripData
        Front right corner data.
    rear_left : CornerTireGripData
        Rear left corner data.
    rear_right : CornerTireGripData
        Rear right corner data.
    metric_mode : str
        "pressure" or "temperature".
    metric_unit : str
        Unit string for the tire metric, read from channel metadata
        (e.g., "bar", "kPa", "C").
    accel_unit : str
        Unit string for acceleration, read from channel metadata
        (e.g., "g", "m/s^2").
    """

    front_left: CornerTireGripData
    front_right: CornerTireGripData
    rear_left: CornerTireGripData
    rear_right: CornerTireGripData
    metric_mode: str
    metric_unit: str
    accel_unit: str


# Channel keys for each mode
_PRESSURE_KEYS = ["tpms_press_fl", "tpms_press_fr", "tpms_press_rl", "tpms_press_rr"]
_TEMPERATURE_KEYS = ["tpms_temp_fl", "tpms_temp_fr", "tpms_temp_rl", "tpms_temp_rr"]
_ACCEL_KEYS = ["lateral_g", "inline_g"]


def _compute_corner(
    total_g: np.ndarray,
    tire_metric: np.ndarray,
    corner_name: str,
    num_buckets: int = 20,
    percentile: float = 99.9,
    min_count: int = 5,
) -> CornerTireGripData:
    """Compute statistics and bucketed percentile for a single corner."""
    n = len(total_g)
    if n == 0:
        return CornerTireGripData(
            corner_name=corner_name,
            total_g=total_g,
            tire_metric=tire_metric,
            mean_g=0.0,
            mean_metric=0.0,
            std_g=0.0,
            std_metric=0.0,
            bucket_centers=np.array([]),
            bucket_values=np.array([]),
            bucket_counts=np.array([], dtype=np.int64),
            percentile=percentile,
        )

    mean_g = float(np.mean(total_g))
    mean_metric = float(np.mean(tire_metric))
    std_g = float(np.std(total_g))
    std_metric = float(np.std(tire_metric))

    # Bucketed percentile computation
    edges = np.linspace(tire_metric.min(), tire_metric.max(), num_buckets + 1)
    bin_indices = np.clip(np.digitize(tire_metric, edges), 1, num_buckets)

    centers_list: list[float] = []
    values_list: list[float] = []
    counts_list: list[int] = []

    for i in range(1, num_buckets + 1):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count >= min_count:
            centers_list.append(float((edges[i - 1] + edges[i]) / 2))
            values_list.append(float(np.percentile(total_g[mask], percentile)))
            counts_list.append(count)

    return CornerTireGripData(
        corner_name=corner_name,
        total_g=total_g,
        tire_metric=tire_metric,
        mean_g=mean_g,
        mean_metric=mean_metric,
        std_g=std_g,
        std_metric=std_metric,
        bucket_centers=np.array(centers_list),
        bucket_values=np.array(values_list),
        bucket_counts=np.array(counts_list, dtype=np.int64),
        percentile=percentile,
    )


def analyze_tire_grip(
    lap_log: "LogFile",
    channel_names: dict[str, str],
    metric_mode: str = "pressure",
    num_buckets: int = 20,
    percentile: float = 99.9,
) -> TireGripResult:
    """Analyze tire grip: total G vs tire pressure or temperature.

    The caller must filter the LogFile to a single lap before calling this
    function using log.filter_by_lap(lap_num).

    Parameters
    ----------
    lap_log : LogFile
        The LogFile pre-filtered to a single lap (via log.filter_by_lap()).
    channel_names : dict[str, str]
        Channel name mapping. Required keys:
        - "lateral_g", "inline_g" (acceleration channels)
        - "tpms_press_fl/fr/rl/rr" (for pressure mode)
        - "tpms_temp_fl/fr/rl/rr" (for temperature mode)
    metric_mode : str, default="pressure"
        "pressure" or "temperature".
    num_buckets : int, default=20
        Number of buckets for percentile analysis.
    percentile : float, default=99.9
        Percentile of total G to compute per bucket.

    Returns
    -------
    TireGripResult
        Analysis results for all four corners.

    Raises
    ------
    KeyError
        If required channel names are missing.
    ValueError
        If metric_mode is not "pressure" or "temperature".
    """
    if metric_mode not in ("pressure", "temperature"):
        raise ValueError(f"metric_mode must be 'pressure' or 'temperature', got '{metric_mode}'")

    # Determine which tire metric channels to use
    metric_keys = _PRESSURE_KEYS if metric_mode == "pressure" else _TEMPERATURE_KEYS

    required_keys = _ACCEL_KEYS + metric_keys
    _validate_channel_names(channel_names, required_keys, "analyze_tire_grip")

    # Resolve actual channel names
    lat_g_ch = channel_names["lateral_g"]
    inline_g_ch = channel_names["inline_g"]
    metric_channels = [channel_names[k] for k in metric_keys]

    all_channels = [lat_g_ch, inline_g_ch] + metric_channels

    # Select and resample channels
    aligned = lap_log.select_channels(all_channels).resample_to_channel(lat_g_ch).channels

    # Read units from channel metadata
    metric_unit = _get_channel_unit(aligned[metric_channels[0]], metric_channels[0])
    accel_unit = _get_channel_unit(aligned[lat_g_ch], lat_g_ch)

    # Extract acceleration arrays
    lateral_g = aligned[lat_g_ch].column(lat_g_ch).to_numpy()
    inline_g = aligned[inline_g_ch].column(inline_g_ch).to_numpy()
    total_g = np.sqrt(lateral_g**2 + inline_g**2)

    # Extract tire metric arrays per corner
    corner_specs = [
        ("FL", metric_channels[0]),
        ("FR", metric_channels[1]),
        ("RL", metric_channels[2]),
        ("RR", metric_channels[3]),
    ]

    corners: dict[str, CornerTireGripData] = {}
    for corner_name, metric_ch in corner_specs:
        tire_metric = aligned[metric_ch].column(metric_ch).to_numpy()
        corners[corner_name] = _compute_corner(
            total_g,
            tire_metric,
            corner_name,
            num_buckets=num_buckets,
            percentile=percentile,
        )

    return TireGripResult(
        front_left=corners["FL"],
        front_right=corners["FR"],
        rear_left=corners["RL"],
        rear_right=corners["RR"],
        metric_mode=metric_mode,
        metric_unit=metric_unit,
        accel_unit=accel_unit,
    )


def analyze_tire_grip_multi_lap(
    log: "LogFile",
    lap_numbers: list[int],
    channel_names: dict[str, str],
    metric_mode: str = "pressure",
    num_buckets: int = 20,
    percentile: float = 99.9,
) -> TireGripResult:
    """Analyze tire grip across multiple laps.

    Concatenates total G and tire metric data from all selected laps
    before computing per-corner statistics, ensuring correct weighting
    (longer laps contribute more data points).

    Parameters
    ----------
    log : LogFile
        The LogFile containing all laps.
    lap_numbers : list[int]
        Lap numbers to include in the analysis.
    channel_names : dict[str, str]
        Channel name mapping. Required keys:
        - "lateral_g", "inline_g" (acceleration channels)
        - "tpms_press_fl/fr/rl/rr" (for pressure mode)
        - "tpms_temp_fl/fr/rl/rr" (for temperature mode)
    metric_mode : str, default="pressure"
        "pressure" or "temperature".
    num_buckets : int, default=20
        Number of buckets for percentile analysis.
    percentile : float, default=99.9
        Percentile of total G to compute per bucket.

    Returns
    -------
    TireGripResult
        Complete analysis results for all four corners, aggregated
        across all selected laps.

    Raises
    ------
    KeyError
        If required channel names are missing.
    ValueError
        If lap_numbers is empty or metric_mode is invalid.
    """
    if not lap_numbers:
        raise ValueError("lap_numbers cannot be empty")

    if metric_mode not in ("pressure", "temperature"):
        raise ValueError(f"metric_mode must be 'pressure' or 'temperature', got '{metric_mode}'")

    # Determine which tire metric channels to use
    metric_keys = _PRESSURE_KEYS if metric_mode == "pressure" else _TEMPERATURE_KEYS

    required_keys = _ACCEL_KEYS + metric_keys
    _validate_channel_names(channel_names, required_keys, "analyze_tire_grip_multi_lap")

    # Resolve actual channel names
    lat_g_ch = channel_names["lateral_g"]
    inline_g_ch = channel_names["inline_g"]
    metric_channels = [channel_names[k] for k in metric_keys]

    all_channels = [lat_g_ch, inline_g_ch] + metric_channels

    corner_specs = [
        ("FL", metric_channels[0]),
        ("FR", metric_channels[1]),
        ("RL", metric_channels[2]),
        ("RR", metric_channels[3]),
    ]

    # Collect data from all laps
    all_total_g: list[np.ndarray] = []
    all_metrics: dict[str, list[np.ndarray]] = {
        "FL": [],
        "FR": [],
        "RL": [],
        "RR": [],
    }
    metric_unit = ""
    accel_unit = ""

    for lap_num in lap_numbers:
        try:
            lap_log = log.filter_by_lap(lap_num)
            aligned = lap_log.select_channels(all_channels).resample_to_channel(lat_g_ch).channels

            # Read units from the first successful lap
            if not metric_unit:
                metric_unit = _get_channel_unit(aligned[metric_channels[0]], metric_channels[0])
            if not accel_unit:
                accel_unit = _get_channel_unit(aligned[lat_g_ch], lat_g_ch)

            lateral_g = aligned[lat_g_ch].column(lat_g_ch).to_numpy()
            inline_g = aligned[inline_g_ch].column(inline_g_ch).to_numpy()
            total_g = np.sqrt(lateral_g**2 + inline_g**2)

            all_total_g.append(total_g)

            for corner_name, metric_ch in corner_specs:
                tire_metric = aligned[metric_ch].column(metric_ch).to_numpy()
                all_metrics[corner_name].append(tire_metric)
        except Exception:
            # Skip laps that fail to process (e.g., missing data)
            continue

    # Concatenate data from all laps
    combined_total_g = np.concatenate(all_total_g) if all_total_g else np.array([])

    # Process each corner with combined data
    corners: dict[str, CornerTireGripData] = {}
    for corner_name, _ in corner_specs:
        combined_metric = (
            np.concatenate(all_metrics[corner_name]) if all_metrics[corner_name] else np.array([])
        )
        corners[corner_name] = _compute_corner(
            combined_total_g,
            combined_metric,
            corner_name,
            num_buckets=num_buckets,
            percentile=percentile,
        )

    return TireGripResult(
        front_left=corners["FL"],
        front_right=corners["FR"],
        rear_left=corners["RL"],
        rear_right=corners["RR"],
        metric_mode=metric_mode,
        metric_unit=metric_unit,
        accel_unit=accel_unit,
    )


def format_tire_grip_stats_table(result: TireGripResult) -> pd.DataFrame:
    """Format tire grip statistics as a DataFrame.

    Parameters
    ----------
    result : TireGripResult
        Analysis results from analyze_tire_grip().

    Returns
    -------
    pd.DataFrame
        DataFrame with per-corner statistics.
    """
    corners = {
        "FL": result.front_left,
        "FR": result.front_right,
        "RL": result.rear_left,
        "RR": result.rear_right,
    }

    metric_label = "Pressure" if result.metric_mode == "pressure" else "Temperature"
    metric_col = f"Mean {metric_label}"
    metric_std_col = f"Std {metric_label}"
    if result.metric_unit:
        metric_col += f" ({result.metric_unit})"
        metric_std_col += f" ({result.metric_unit})"

    accel_col = "Mean Accel"
    accel_std_col = "Std Accel"
    if result.accel_unit:
        accel_col += f" ({result.accel_unit})"
        accel_std_col += f" ({result.accel_unit})"

    data = []
    for corner_name, corner_data in corners.items():
        data.append(
            {
                "Corner": corner_name,
                accel_col: corner_data.mean_g,
                accel_std_col: corner_data.std_g,
                metric_col: corner_data.mean_metric,
                metric_std_col: corner_data.std_metric,
            }
        )

    return pd.DataFrame(data)
