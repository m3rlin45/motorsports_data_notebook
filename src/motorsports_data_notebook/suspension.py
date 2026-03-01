"""Suspension velocity analysis module.

This module provides functions for computing wheel velocity from shock pot
displacement data and analyzing velocity distributions for suspension setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from libxrk import ChannelMetadata

from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from typing import Union

    from libxrk.base import LogFile as AimLogFile
    from libibt.base import LogFile as IbtLogFile

    LogFile = Union[AimLogFile, IbtLogFile]


@dataclass
class MotionRatios:
    """Motion ratios for converting shock velocity to wheel velocity.

    Motion ratio = shock_travel / wheel_travel
    wheel_velocity = shock_velocity / motion_ratio

    Attributes
    ----------
    front_left : float
        Front left motion ratio.
    front_right : float
        Front right motion ratio.
    rear_left : float
        Rear left motion ratio.
    rear_right : float
        Rear right motion ratio.
    """

    front_left: float = 0.997
    front_right: float = 0.997
    rear_left: float = 0.768
    rear_right: float = 0.768

    @classmethod
    def toyota_86_zn6(cls) -> "MotionRatios":
        """Return default motion ratios for Toyota 86 ZN6.

        Front (MacPherson strut): 0.997 (nearly 1:1)
        Rear (Multi-link): 0.768
        """
        return cls(
            front_left=0.997,
            front_right=0.997,
            rear_left=0.768,
            rear_right=0.768,
        )


@dataclass
class VelocityRanges:
    """Velocity thresholds for categorizing suspension motion.

    All values are in mm/s (absolute velocity).

    Attributes
    ----------
    friction : float
        Velocities below this are in the friction/static range.
    slow : float
        Velocities below this (but above friction) are in the slow range.
    fast : float
        Velocities below this (but above slow) are in the fast range.
        Velocities above this are in the curb/high-speed range.
    """

    friction: float = 5.0
    slow: float = 25.0
    fast: float = 200.0


@dataclass
class CornerVelocityData:
    """Velocity histogram and statistics for a single corner.

    Attributes
    ----------
    corner_name : str
        Name of the corner (e.g., "FL", "FR", "RL", "RR").
    velocity : np.ndarray
        Raw velocity values in mm/s.
    bin_edges : np.ndarray
        Histogram bin edges in mm/s.
    bin_centers : np.ndarray
        Histogram bin centers in mm/s.
    histogram : np.ndarray
        Histogram values as percentage of time.
    skew : float
        Distribution skewness (positive = more rebound, negative = more bump).
    kurtosis : float
        Distribution kurtosis (excess kurtosis, 0 for normal).
    mean : float
        Mean velocity in mm/s.
    std : float
        Standard deviation in mm/s.
    pct_friction : float
        Percentage of time in friction range (|v| < friction threshold).
    pct_slow_bump : float
        Percentage of time in slow bump range.
    pct_slow_rebound : float
        Percentage of time in slow rebound range.
    pct_fast_bump : float
        Percentage of time in fast bump range.
    pct_fast_rebound : float
        Percentage of time in fast rebound range.
    pct_curb : float
        Percentage of time in curb/high-speed range (|v| > fast threshold).
    pct_zero_bin : float
        Percentage of time in the center histogram bin (centered at 0).
    """

    corner_name: str
    velocity: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    histogram: np.ndarray
    skew: float
    kurtosis: float
    mean: float
    std: float
    pct_friction: float
    pct_slow_bump: float
    pct_slow_rebound: float
    pct_fast_bump: float
    pct_fast_rebound: float
    pct_curb: float
    pct_zero_bin: float


@dataclass
class VelocityHistogramResult:
    """Complete velocity histogram analysis for all four corners.

    Attributes
    ----------
    front_left : CornerVelocityData
        Front left corner data.
    front_right : CornerVelocityData
        Front right corner data.
    rear_left : CornerVelocityData
        Rear left corner data.
    rear_right : CornerVelocityData
        Rear right corner data.
    velocity_ranges : VelocityRanges
        Velocity thresholds used for categorization.
    """

    front_left: CornerVelocityData
    front_right: CornerVelocityData
    rear_left: CornerVelocityData
    rear_right: CornerVelocityData
    velocity_ranges: VelocityRanges = field(default_factory=VelocityRanges)


# Default channel names for suspension analysis
SUSPENSION_CHANNEL_NAMES = {
    "shock_fl": "LF_Shock_Pot",
    "shock_fr": "RF_Shock_Pot",
    "shock_rl": "LR_Shock_Pot",
    "shock_rr": "RR_Shock_Pot",
}


def compute_shock_velocity(
    displacement: np.ndarray,
    timecodes: np.ndarray,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Compute shock velocity from displacement data.

    Uses numpy gradient with time-based differentiation to handle varying
    sample rates. Optional smoothing reduces noise from differentiation.

    Parameters
    ----------
    displacement : np.ndarray
        Shock pot displacement values in mm.
    timecodes : np.ndarray
        Timestamps in milliseconds.
    smoothing_window : int, default=5
        Rolling average window size for smoothing. Set to 1 to disable.

    Returns
    -------
    np.ndarray
        Shock velocity in mm/s. Positive = bump (compression),
        negative = rebound (extension).

    Examples
    --------
    >>> displacement = np.array([0, 1, 3, 6, 10])
    >>> timecodes = np.array([0, 100, 200, 300, 400])
    >>> velocity = compute_shock_velocity(displacement, timecodes, smoothing_window=1)
    """
    # Convert timecodes from ms to seconds for proper velocity units
    time_seconds = timecodes / 1000.0

    # Compute velocity using numpy gradient (handles varying sample rates)
    velocity: np.ndarray = np.gradient(displacement, time_seconds)

    # Apply smoothing if requested
    if smoothing_window > 1 and len(velocity) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        # Use 'same' mode and handle edges by padding
        velocity_padded = np.pad(
            velocity, (smoothing_window // 2, smoothing_window // 2), mode="edge"
        )
        velocity = np.asarray(np.convolve(velocity_padded, kernel, mode="valid"))

    return velocity


def compute_wheel_velocity(
    shock_velocity: np.ndarray,
    motion_ratio: float,
) -> np.ndarray:
    """Convert shock velocity to wheel velocity using motion ratio.

    Parameters
    ----------
    shock_velocity : np.ndarray
        Shock velocity in mm/s.
    motion_ratio : float
        Motion ratio (shock_travel / wheel_travel).

    Returns
    -------
    np.ndarray
        Wheel velocity in mm/s.

    Examples
    --------
    >>> shock_vel = np.array([10, 20, 30])
    >>> wheel_vel = compute_wheel_velocity(shock_vel, motion_ratio=0.768)
    >>> # wheel_vel = [13.02, 26.04, 39.06] (approximately)
    """
    return shock_velocity / motion_ratio


def compute_velocity_histogram(
    velocity: np.ndarray,
    bin_size: float = 5.0,
    max_velocity: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute zero-centered velocity histogram.

    Creates bins centered on zero (e.g., -5 to 5, not 0 to 10).

    Parameters
    ----------
    velocity : np.ndarray
        Velocity values in mm/s.
    bin_size : float, default=5.0
        Size of each histogram bin in mm/s.
    max_velocity : float, default=300.0
        Maximum velocity to include in histogram.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (histogram, bin_edges, bin_centers)
        - histogram: Percentage of time in each bin (sums to 100).
        - bin_edges: Bin edge values in mm/s.
        - bin_centers: Bin center values in mm/s.

    Examples
    --------
    >>> velocity = np.random.normal(0, 50, 1000)
    >>> hist, edges, centers = compute_velocity_histogram(velocity)
    >>> assert np.isclose(hist.sum(), 100.0)
    """
    # Create zero-centered bins
    # Bins are arranged so that zero is at the center of a bin
    # e.g., with bin_size=10: edges at -5, 5, 15, ... so centers are at 0, 10, 20, ...
    half_bin = bin_size / 2
    n_bins_half = int(max_velocity / bin_size)
    # Create edges: -max - half_bin, ..., -half_bin, half_bin, ..., max + half_bin
    positive_edges = np.arange(half_bin, max_velocity + bin_size, bin_size)
    negative_edges = -positive_edges[::-1]
    bin_edges = np.concatenate([negative_edges, positive_edges])

    # Clip velocities to the histogram range
    clipped_velocity = np.clip(velocity, -max_velocity, max_velocity)

    # Compute histogram
    counts, _ = np.histogram(clipped_velocity, bins=bin_edges)

    # Convert to percentage of time
    total_samples = len(velocity)
    if total_samples > 0:
        histogram = counts / total_samples * 100.0
    else:
        histogram = counts.astype(float)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_edges, bin_centers


def compute_velocity_stats(
    velocity: np.ndarray,
    ranges: VelocityRanges,
) -> dict[str, float]:
    """Compute velocity distribution statistics.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity values in mm/s.
    ranges : VelocityRanges
        Velocity thresholds for categorization.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - skew: Distribution skewness
        - kurtosis: Excess kurtosis (0 for normal distribution)
        - mean: Mean velocity
        - std: Standard deviation
        - pct_friction: % time in friction range
        - pct_slow_bump: % time in slow bump range
        - pct_slow_rebound: % time in slow rebound range
        - pct_fast_bump: % time in fast bump range
        - pct_fast_rebound: % time in fast rebound range
        - pct_curb: % time in curb/high-speed range

    Examples
    --------
    >>> velocity = np.random.normal(0, 50, 1000)
    >>> stats = compute_velocity_stats(velocity, VelocityRanges())
    >>> print(f"Skew: {stats['skew']:.2f}")
    """
    n = len(velocity)
    if n == 0:
        return {
            "skew": 0.0,
            "kurtosis": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "pct_friction": 0.0,
            "pct_slow_bump": 0.0,
            "pct_slow_rebound": 0.0,
            "pct_fast_bump": 0.0,
            "pct_fast_rebound": 0.0,
            "pct_curb": 0.0,
        }

    # Basic statistics
    mean = float(np.mean(velocity))
    std = float(np.std(velocity))

    # Skewness and kurtosis
    if std > 0:
        centered = velocity - mean
        skew = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3)  # Excess kurtosis
    else:
        skew = 0.0
        kurtosis = 0.0

    # Velocity categorization
    abs_velocity = np.abs(velocity)

    # Friction range: |v| < friction threshold
    friction_mask = abs_velocity < ranges.friction
    pct_friction = float(np.sum(friction_mask) / n * 100)

    # Slow range: friction <= |v| < slow
    slow_mask = (abs_velocity >= ranges.friction) & (abs_velocity < ranges.slow)
    slow_bump_mask = slow_mask & (velocity > 0)
    slow_rebound_mask = slow_mask & (velocity < 0)
    pct_slow_bump = float(np.sum(slow_bump_mask) / n * 100)
    pct_slow_rebound = float(np.sum(slow_rebound_mask) / n * 100)

    # Fast range: slow <= |v| < fast
    fast_mask = (abs_velocity >= ranges.slow) & (abs_velocity < ranges.fast)
    fast_bump_mask = fast_mask & (velocity > 0)
    fast_rebound_mask = fast_mask & (velocity < 0)
    pct_fast_bump = float(np.sum(fast_bump_mask) / n * 100)
    pct_fast_rebound = float(np.sum(fast_rebound_mask) / n * 100)

    # Curb range: |v| >= fast
    curb_mask = abs_velocity >= ranges.fast
    pct_curb = float(np.sum(curb_mask) / n * 100)

    return {
        "skew": skew,
        "kurtosis": kurtosis,
        "mean": mean,
        "std": std,
        "pct_friction": pct_friction,
        "pct_slow_bump": pct_slow_bump,
        "pct_slow_rebound": pct_slow_rebound,
        "pct_fast_bump": pct_fast_bump,
        "pct_fast_rebound": pct_fast_rebound,
        "pct_curb": pct_curb,
    }


def _channels_are_velocity(log: "LogFile", channel_name: str) -> bool:
    """Check if shock channel contains velocity data (vs displacement).

    iRacing provides shock velocity directly (m/s), while AIM loggers
    provide displacement (mm) requiring velocity derivation.
    """
    table = log.channels.get(channel_name)
    if table is None:
        return False
    try:
        meta = ChannelMetadata.from_field(table.schema.field(channel_name))
    except (KeyError, AttributeError):
        return False
    return "/s" in meta.units  # m/s, mm/s, etc.


def _process_corner(
    displacement: np.ndarray,
    timecodes: np.ndarray,
    motion_ratio: float,
    corner_name: str,
    smoothing_window: int,
    bin_size: float,
    max_velocity: float,
    ranges: VelocityRanges,
) -> CornerVelocityData:
    """Process a single corner's suspension data."""
    # Compute shock velocity
    shock_velocity = compute_shock_velocity(displacement, timecodes, smoothing_window)

    # Convert to wheel velocity
    wheel_velocity = compute_wheel_velocity(shock_velocity, motion_ratio)

    # Compute histogram
    histogram, bin_edges, bin_centers = compute_velocity_histogram(
        wheel_velocity, bin_size, max_velocity
    )

    # Find zero bin percentage (the bin centered at 0)
    zero_bin_idx = int(np.argmin(np.abs(bin_centers)))
    pct_zero_bin = float(histogram[zero_bin_idx]) if len(histogram) > 0 else 0.0

    # Compute statistics
    stats = compute_velocity_stats(wheel_velocity, ranges)

    return CornerVelocityData(
        corner_name=corner_name,
        velocity=wheel_velocity,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        histogram=histogram,
        skew=stats["skew"],
        kurtosis=stats["kurtosis"],
        mean=stats["mean"],
        std=stats["std"],
        pct_friction=stats["pct_friction"],
        pct_slow_bump=stats["pct_slow_bump"],
        pct_slow_rebound=stats["pct_slow_rebound"],
        pct_fast_bump=stats["pct_fast_bump"],
        pct_fast_rebound=stats["pct_fast_rebound"],
        pct_curb=stats["pct_curb"],
        pct_zero_bin=pct_zero_bin,
    )


def _process_corner_velocity(
    velocity_ms: np.ndarray,
    corner_name: str,
    bin_size: float,
    max_velocity: float,
    ranges: VelocityRanges,
) -> CornerVelocityData:
    """Process a single corner's suspension data when velocity is already available.

    Used for iRacing data where shock velocity channels provide velocity directly
    in m/s, bypassing displacement-to-velocity derivation and motion ratio conversion.
    """
    # Convert m/s to mm/s
    velocity_mms = velocity_ms * 1000.0

    # Compute histogram
    histogram, bin_edges, bin_centers = compute_velocity_histogram(
        velocity_mms, bin_size, max_velocity
    )

    # Find zero bin percentage
    zero_bin_idx = int(np.argmin(np.abs(bin_centers)))
    pct_zero_bin = float(histogram[zero_bin_idx]) if len(histogram) > 0 else 0.0

    # Compute statistics
    stats = compute_velocity_stats(velocity_mms, ranges)

    return CornerVelocityData(
        corner_name=corner_name,
        velocity=velocity_mms,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        histogram=histogram,
        skew=stats["skew"],
        kurtosis=stats["kurtosis"],
        mean=stats["mean"],
        std=stats["std"],
        pct_friction=stats["pct_friction"],
        pct_slow_bump=stats["pct_slow_bump"],
        pct_slow_rebound=stats["pct_slow_rebound"],
        pct_fast_bump=stats["pct_fast_bump"],
        pct_fast_rebound=stats["pct_fast_rebound"],
        pct_curb=stats["pct_curb"],
        pct_zero_bin=pct_zero_bin,
    )


def analyze_suspension_velocity(
    log: "LogFile",
    channel_names: dict | None = None,
    motion_ratios: MotionRatios | None = None,
    velocity_ranges: VelocityRanges | None = None,
    smoothing_window: int = 5,
    bin_size: float = 10.0,
    max_velocity: float = 300.0,
) -> VelocityHistogramResult:
    """Analyze suspension velocity distribution for a lap.

    Main analysis function that processes all four corners and returns
    comprehensive velocity histogram data and statistics.

    The caller must filter the LogFile to a single lap before calling this function
    using log.filter_by_lap(lap_num).

    Parameters
    ----------
    log : LogFile
        The LogFile pre-filtered to a single lap (via log.filter_by_lap()).
    channel_names : dict, optional
        Channel name mapping. Required keys:
        - "shock_fl": Front left shock pot channel
        - "shock_fr": Front right shock pot channel
        - "shock_rl": Rear left shock pot channel
        - "shock_rr": Rear right shock pot channel
        If None, uses SUSPENSION_CHANNEL_NAMES defaults.
    motion_ratios : MotionRatios, optional
        Motion ratios for each corner. If None, uses Toyota 86 ZN6 defaults.
    velocity_ranges : VelocityRanges, optional
        Velocity thresholds for categorization. If None, uses defaults.
    smoothing_window : int, default=5
        Rolling average window size for velocity smoothing.
    bin_size : float, default=10.0
        Histogram bin size in mm/s.
    max_velocity : float, default=300.0
        Maximum velocity for histogram range.

    Returns
    -------
    VelocityHistogramResult
        Complete analysis results for all four corners.

    Raises
    ------
    KeyError
        If required channel names are missing.

    Examples
    --------
    >>> from motorsports_data_notebook.channels import get_best_lap
    >>> best_lap = get_best_lap(laps)
    >>> lap_log = log.filter_by_lap(int(best_lap["num"]))
    >>> result = analyze_suspension_velocity(lap_log)
    >>> print(f"FL skew: {result.front_left.skew:.2f}")
    """
    # Use defaults if not provided
    if channel_names is None:
        channel_names = SUSPENSION_CHANNEL_NAMES.copy()
    if motion_ratios is None:
        motion_ratios = MotionRatios.toyota_86_zn6()
    if velocity_ranges is None:
        velocity_ranges = VelocityRanges()

    # Validate channel names
    required_keys = ["shock_fl", "shock_fr", "shock_rl", "shock_rr"]
    _validate_channel_names(channel_names, required_keys, "analyze_suspension_velocity")

    # Extract channel data
    shock_fl_name = channel_names["shock_fl"]
    shock_fr_name = channel_names["shock_fr"]
    shock_rl_name = channel_names["shock_rl"]
    shock_rr_name = channel_names["shock_rr"]

    # Use libxrk 0.5.0 methods to select and resample channels
    aligned = (
        log.select_channels([shock_fl_name, shock_fr_name, shock_rl_name, shock_rr_name])
        .resample_to_channel(shock_fl_name)
        .channels
    )

    # Check if channels contain velocity data (iRacing) or displacement (AIM)
    is_velocity = _channels_are_velocity(log, shock_fl_name)

    if is_velocity:
        # Velocity channels (iRacing): convert m/s to mm/s, skip derivation
        fl_vel = aligned[shock_fl_name].column(shock_fl_name).to_numpy()
        fr_vel = aligned[shock_fr_name].column(shock_fr_name).to_numpy()
        rl_vel = aligned[shock_rl_name].column(shock_rl_name).to_numpy()
        rr_vel = aligned[shock_rr_name].column(shock_rr_name).to_numpy()

        fl_data = _process_corner_velocity(fl_vel, "FL", bin_size, max_velocity, velocity_ranges)
        fr_data = _process_corner_velocity(fr_vel, "FR", bin_size, max_velocity, velocity_ranges)
        rl_data = _process_corner_velocity(rl_vel, "RL", bin_size, max_velocity, velocity_ranges)
        rr_data = _process_corner_velocity(rr_vel, "RR", bin_size, max_velocity, velocity_ranges)
    else:
        # Displacement channels (AIM): derive velocity and apply motion ratios
        timecodes = aligned[shock_fl_name].column("timecodes").to_numpy()
        fl_disp = aligned[shock_fl_name].column(shock_fl_name).to_numpy()
        fr_disp = aligned[shock_fr_name].column(shock_fr_name).to_numpy()
        rl_disp = aligned[shock_rl_name].column(shock_rl_name).to_numpy()
        rr_disp = aligned[shock_rr_name].column(shock_rr_name).to_numpy()

        fl_data = _process_corner(
            fl_disp,
            timecodes,
            motion_ratios.front_left,
            "FL",
            smoothing_window,
            bin_size,
            max_velocity,
            velocity_ranges,
        )
        fr_data = _process_corner(
            fr_disp,
            timecodes,
            motion_ratios.front_right,
            "FR",
            smoothing_window,
            bin_size,
            max_velocity,
            velocity_ranges,
        )
        rl_data = _process_corner(
            rl_disp,
            timecodes,
            motion_ratios.rear_left,
            "RL",
            smoothing_window,
            bin_size,
            max_velocity,
            velocity_ranges,
        )
        rr_data = _process_corner(
            rr_disp,
            timecodes,
            motion_ratios.rear_right,
            "RR",
            smoothing_window,
            bin_size,
            max_velocity,
            velocity_ranges,
        )

    return VelocityHistogramResult(
        front_left=fl_data,
        front_right=fr_data,
        rear_left=rl_data,
        rear_right=rr_data,
        velocity_ranges=velocity_ranges,
    )


def analyze_suspension_velocity_multi_lap(
    log: "LogFile",
    lap_numbers: list[int],
    channel_names: dict | None = None,
    motion_ratios: MotionRatios | None = None,
    velocity_ranges: VelocityRanges | None = None,
    smoothing_window: int = 5,
    bin_size: float = 10.0,
    max_velocity: float = 300.0,
) -> VelocityHistogramResult:
    """Analyze suspension velocity across multiple laps.

    Concatenates velocity data from all selected laps before computing
    the histogram, ensuring correct weighting (longer laps contribute
    more data points).

    Automatically detects whether channels contain velocity data (iRacing)
    or displacement data (AIM) and processes accordingly.

    Parameters
    ----------
    log : LogFile
        The LogFile containing all laps (unfiltered).
    lap_numbers : list[int]
        Lap numbers to include in the analysis.
    channel_names : dict, optional
        Channel name mapping. Required keys:
        - "shock_fl": Front left shock pot channel
        - "shock_fr": Front right shock pot channel
        - "shock_rl": Rear left shock pot channel
        - "shock_rr": Rear right shock pot channel
        If None, uses SUSPENSION_CHANNEL_NAMES defaults.
    motion_ratios : MotionRatios, optional
        Motion ratios for each corner. If None, uses Toyota 86 ZN6 defaults.
    velocity_ranges : VelocityRanges, optional
        Velocity thresholds for categorization. If None, uses defaults.
    smoothing_window : int, default=5
        Rolling average window size for velocity smoothing.
    bin_size : float, default=10.0
        Histogram bin size in mm/s.
    max_velocity : float, default=300.0
        Maximum velocity for histogram range.

    Returns
    -------
    VelocityHistogramResult
        Complete analysis results for all four corners, aggregated
        across all selected laps.

    Raises
    ------
    KeyError
        If required channel names are missing.
    ValueError
        If lap_numbers is empty.
    """
    if not lap_numbers:
        raise ValueError("lap_numbers cannot be empty")

    if channel_names is None:
        channel_names = SUSPENSION_CHANNEL_NAMES.copy()
    if motion_ratios is None:
        motion_ratios = MotionRatios.toyota_86_zn6()
    if velocity_ranges is None:
        velocity_ranges = VelocityRanges()

    required_keys = ["shock_fl", "shock_fr", "shock_rl", "shock_rr"]
    _validate_channel_names(channel_names, required_keys, "analyze_suspension_velocity_multi_lap")

    shock_fl_name = channel_names["shock_fl"]
    shock_fr_name = channel_names["shock_fr"]
    shock_rl_name = channel_names["shock_rl"]
    shock_rr_name = channel_names["shock_rr"]
    shock_names = [shock_fl_name, shock_fr_name, shock_rl_name, shock_rr_name]
    corner_labels = ["FL", "FR", "RL", "RR"]
    mr_values = [
        motion_ratios.front_left,
        motion_ratios.front_right,
        motion_ratios.rear_left,
        motion_ratios.rear_right,
    ]

    is_velocity = _channels_are_velocity(log, shock_fl_name)

    # Collect velocity data (in mm/s) from all laps
    all_velocities: dict[str, list[np.ndarray]] = {c: [] for c in corner_labels}

    for lap_num in lap_numbers:
        try:
            lap_log = log.filter_by_lap(lap_num)
            aligned = (
                lap_log.select_channels(shock_names).resample_to_channel(shock_fl_name).channels
            )

            if is_velocity:
                # iRacing: channels are velocity in m/s, convert to mm/s
                for name, label in zip(shock_names, corner_labels):
                    vel_ms = aligned[name].column(name).to_numpy()
                    all_velocities[label].append(vel_ms * 1000.0)
            else:
                # AIM: channels are displacement in mm, derive velocity
                timecodes = aligned[shock_fl_name].column("timecodes").to_numpy()
                for name, label, mr in zip(shock_names, corner_labels, mr_values):
                    disp = aligned[name].column(name).to_numpy()
                    shock_vel = compute_shock_velocity(disp, timecodes, smoothing_window)
                    wheel_vel = compute_wheel_velocity(shock_vel, mr)
                    all_velocities[label].append(wheel_vel)

        except Exception:
            continue

    # Concatenate and compute histograms
    results = {}
    for label in corner_labels:
        vel_list = all_velocities[label]
        combined = np.concatenate(vel_list) if vel_list else np.array([])

        if len(combined) == 0:
            histogram, bin_edges, bin_centers = compute_velocity_histogram(
                np.array([0.0]), bin_size, max_velocity
            )
            results[label] = CornerVelocityData(
                corner_name=label,
                velocity=combined,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                histogram=np.zeros_like(histogram),
                skew=0.0,
                kurtosis=0.0,
                mean=0.0,
                std=0.0,
                pct_friction=0.0,
                pct_slow_bump=0.0,
                pct_slow_rebound=0.0,
                pct_fast_bump=0.0,
                pct_fast_rebound=0.0,
                pct_curb=0.0,
                pct_zero_bin=0.0,
            )
        else:
            histogram, bin_edges, bin_centers = compute_velocity_histogram(
                combined, bin_size, max_velocity
            )
            zero_bin_idx = int(np.argmin(np.abs(bin_centers)))
            pct_zero_bin = float(histogram[zero_bin_idx]) if len(histogram) > 0 else 0.0
            stats = compute_velocity_stats(combined, velocity_ranges)
            results[label] = CornerVelocityData(
                corner_name=label,
                velocity=combined,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                histogram=histogram,
                skew=stats["skew"],
                kurtosis=stats["kurtosis"],
                mean=stats["mean"],
                std=stats["std"],
                pct_friction=stats["pct_friction"],
                pct_slow_bump=stats["pct_slow_bump"],
                pct_slow_rebound=stats["pct_slow_rebound"],
                pct_fast_bump=stats["pct_fast_bump"],
                pct_fast_rebound=stats["pct_fast_rebound"],
                pct_curb=stats["pct_curb"],
                pct_zero_bin=pct_zero_bin,
            )

    return VelocityHistogramResult(
        front_left=results["FL"],
        front_right=results["FR"],
        rear_left=results["RL"],
        rear_right=results["RR"],
        velocity_ranges=velocity_ranges,
    )


def format_suspension_stats_table(
    result: VelocityHistogramResult,
) -> pd.DataFrame:
    """Format suspension statistics as a DataFrame.

    Creates a summary table with per-corner statistics for easy comparison.

    Parameters
    ----------
    result : VelocityHistogramResult
        Analysis results from analyze_suspension_velocity().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each corner and rows for each statistic.

    Examples
    --------
    >>> result = analyze_suspension_velocity(log, lap_start, lap_end)
    >>> stats_df = format_suspension_stats_table(result)
    >>> display(stats_df)
    """
    corners = {
        "FL": result.front_left,
        "FR": result.front_right,
        "RL": result.rear_left,
        "RR": result.rear_right,
    }

    data = []
    for corner_name, corner_data in corners.items():
        data.append(
            {
                "Corner": corner_name,
                "Skew": corner_data.skew,
                "Kurtosis": corner_data.kurtosis,
                "Mean (mm/s)": corner_data.mean,
                "Std (mm/s)": corner_data.std,
                "Zero Bin %": corner_data.pct_zero_bin,
                "Friction %": corner_data.pct_friction,
                "Slow Bump %": corner_data.pct_slow_bump,
                "Slow Rebound %": corner_data.pct_slow_rebound,
                "Fast Bump %": corner_data.pct_fast_bump,
                "Fast Rebound %": corner_data.pct_fast_rebound,
                "Curb %": corner_data.pct_curb,
            }
        )

    return pd.DataFrame(data)


def format_symmetry_table(result: VelocityHistogramResult) -> pd.DataFrame:
    """Format bump vs rebound symmetry comparison.

    Parameters
    ----------
    result : VelocityHistogramResult
        Analysis results from analyze_suspension_velocity().

    Returns
    -------
    pd.DataFrame
        DataFrame comparing bump and rebound percentages for each corner.
    """
    corners = {
        "FL": result.front_left,
        "FR": result.front_right,
        "RL": result.rear_left,
        "RR": result.rear_right,
    }

    data = []
    for corner_name, corner_data in corners.items():
        total_bump = corner_data.pct_slow_bump + corner_data.pct_fast_bump
        total_rebound = corner_data.pct_slow_rebound + corner_data.pct_fast_rebound
        symmetry = (
            total_bump / (total_bump + total_rebound) * 100
            if (total_bump + total_rebound) > 0
            else 50.0
        )

        data.append(
            {
                "Corner": corner_name,
                "Total Bump %": total_bump,
                "Total Rebound %": total_rebound,
                "Bump/Total %": symmetry,
                "Slow Bump %": corner_data.pct_slow_bump,
                "Slow Rebound %": corner_data.pct_slow_rebound,
                "Fast Bump %": corner_data.pct_fast_bump,
                "Fast Rebound %": corner_data.pct_fast_rebound,
            }
        )

    return pd.DataFrame(data)


def format_comparison_table(result: VelocityHistogramResult) -> pd.DataFrame:
    """Format left/right and front/rear comparison table.

    Combines wheels into groups and shows the difference in velocity range
    percentages between groups:
    - Left (FL + RL) vs Right (FR + RR)
    - Front (FL + FR) vs Rear (RL + RR)

    Parameters
    ----------
    result : VelocityHistogramResult
        Analysis results from analyze_suspension_velocity().

    Returns
    -------
    pd.DataFrame
        DataFrame with combined group comparisons showing percentage differences.
    """
    fl = result.front_left
    fr = result.front_right
    rl = result.rear_left
    rr = result.rear_right

    # Compute averages for each group
    # Left (FL + RL) vs Right (FR + RR)
    left_zero = (fl.pct_zero_bin + rl.pct_zero_bin) / 2
    right_zero = (fr.pct_zero_bin + rr.pct_zero_bin) / 2
    left_friction = (fl.pct_friction + rl.pct_friction) / 2
    right_friction = (fr.pct_friction + rr.pct_friction) / 2
    left_slow_bump = (fl.pct_slow_bump + rl.pct_slow_bump) / 2
    right_slow_bump = (fr.pct_slow_bump + rr.pct_slow_bump) / 2
    left_slow_rebound = (fl.pct_slow_rebound + rl.pct_slow_rebound) / 2
    right_slow_rebound = (fr.pct_slow_rebound + rr.pct_slow_rebound) / 2
    left_fast_bump = (fl.pct_fast_bump + rl.pct_fast_bump) / 2
    right_fast_bump = (fr.pct_fast_bump + rr.pct_fast_bump) / 2
    left_fast_rebound = (fl.pct_fast_rebound + rl.pct_fast_rebound) / 2
    right_fast_rebound = (fr.pct_fast_rebound + rr.pct_fast_rebound) / 2
    left_curb = (fl.pct_curb + rl.pct_curb) / 2
    right_curb = (fr.pct_curb + rr.pct_curb) / 2

    # Front (FL + FR) vs Rear (RL + RR)
    front_zero = (fl.pct_zero_bin + fr.pct_zero_bin) / 2
    rear_zero = (rl.pct_zero_bin + rr.pct_zero_bin) / 2
    front_friction = (fl.pct_friction + fr.pct_friction) / 2
    rear_friction = (rl.pct_friction + rr.pct_friction) / 2
    front_slow_bump = (fl.pct_slow_bump + fr.pct_slow_bump) / 2
    rear_slow_bump = (rl.pct_slow_bump + rr.pct_slow_bump) / 2
    front_slow_rebound = (fl.pct_slow_rebound + fr.pct_slow_rebound) / 2
    rear_slow_rebound = (rl.pct_slow_rebound + rr.pct_slow_rebound) / 2
    front_fast_bump = (fl.pct_fast_bump + fr.pct_fast_bump) / 2
    rear_fast_bump = (rl.pct_fast_bump + rr.pct_fast_bump) / 2
    front_fast_rebound = (fl.pct_fast_rebound + fr.pct_fast_rebound) / 2
    rear_fast_rebound = (rl.pct_fast_rebound + rr.pct_fast_rebound) / 2
    front_curb = (fl.pct_curb + fr.pct_curb) / 2
    rear_curb = (rl.pct_curb + rr.pct_curb) / 2

    data = [
        {
            "Comparison": "Left vs Right",
            "Zero Bin Diff": left_zero - right_zero,
            "Friction Diff": left_friction - right_friction,
            "Slow Bump Diff": left_slow_bump - right_slow_bump,
            "Slow Rebound Diff": left_slow_rebound - right_slow_rebound,
            "Fast Bump Diff": left_fast_bump - right_fast_bump,
            "Fast Rebound Diff": left_fast_rebound - right_fast_rebound,
            "Curb Diff": left_curb - right_curb,
        },
        {
            "Comparison": "Front vs Rear",
            "Zero Bin Diff": front_zero - rear_zero,
            "Friction Diff": front_friction - rear_friction,
            "Slow Bump Diff": front_slow_bump - rear_slow_bump,
            "Slow Rebound Diff": front_slow_rebound - rear_slow_rebound,
            "Fast Bump Diff": front_fast_bump - rear_fast_bump,
            "Fast Rebound Diff": front_fast_rebound - rear_fast_rebound,
            "Curb Diff": front_curb - rear_curb,
        },
    ]

    return pd.DataFrame(data)
