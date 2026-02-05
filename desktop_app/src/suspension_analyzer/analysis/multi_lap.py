"""Multi-lap suspension velocity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from motorsports_data_notebook.suspension import (
    CornerVelocityData,
    MotionRatios,
    SUSPENSION_CHANNEL_NAMES,
    VelocityHistogramResult,
    VelocityRanges,
    _validate_channel_names,
    compute_shock_velocity,
    compute_velocity_histogram,
    compute_velocity_stats,
    compute_wheel_velocity,
)

if TYPE_CHECKING:
    from libxrk.base import LogFile


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

    Parameters
    ----------
    log : LogFile
        The LogFile containing all laps.
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

    Examples
    --------
    >>> result = analyze_suspension_velocity_multi_lap(
    ...     log, [1, 2, 3],
    ...     motion_ratios=MotionRatios.toyota_86_zn6(),
    ... )
    >>> print(f"FL skew: {result.front_left.skew:.2f}")
    """
    if not lap_numbers:
        raise ValueError("lap_numbers cannot be empty")

    # Use defaults if not provided
    if channel_names is None:
        channel_names = SUSPENSION_CHANNEL_NAMES.copy()
    if motion_ratios is None:
        motion_ratios = MotionRatios.toyota_86_zn6()
    if velocity_ranges is None:
        velocity_ranges = VelocityRanges()

    # Validate channel names
    required_keys = ["shock_fl", "shock_fr", "shock_rl", "shock_rr"]
    _validate_channel_names(channel_names, required_keys, "analyze_suspension_velocity_multi_lap")

    shock_fl_name = channel_names["shock_fl"]
    shock_fr_name = channel_names["shock_fr"]
    shock_rl_name = channel_names["shock_rl"]
    shock_rr_name = channel_names["shock_rr"]

    # Collect velocity data from all laps
    all_velocities = {
        "FL": [],
        "FR": [],
        "RL": [],
        "RR": [],
    }

    for lap_num in lap_numbers:
        try:
            # Filter to single lap
            lap_log = log.filter_by_lap(lap_num)

            # Select and resample channels
            aligned = (
                lap_log.select_channels(
                    [shock_fl_name, shock_fr_name, shock_rl_name, shock_rr_name]
                )
                .resample_to_channel(shock_fl_name)
                .channels
            )

            # Extract arrays
            timecodes = aligned[shock_fl_name].column("timecodes").to_numpy()
            fl_disp = aligned[shock_fl_name].column(shock_fl_name).to_numpy()
            fr_disp = aligned[shock_fr_name].column(shock_fr_name).to_numpy()
            rl_disp = aligned[shock_rl_name].column(shock_rl_name).to_numpy()
            rr_disp = aligned[shock_rr_name].column(shock_rr_name).to_numpy()

            # Compute velocities for this lap
            corners = [
                ("FL", fl_disp, motion_ratios.front_left),
                ("FR", fr_disp, motion_ratios.front_right),
                ("RL", rl_disp, motion_ratios.rear_left),
                ("RR", rr_disp, motion_ratios.rear_right),
            ]

            for corner_name, displacement, motion_ratio in corners:
                shock_vel = compute_shock_velocity(displacement, timecodes, smoothing_window)
                wheel_vel = compute_wheel_velocity(shock_vel, motion_ratio)
                all_velocities[corner_name].append(wheel_vel)

        except Exception:
            # Skip laps that fail to process (e.g., missing data)
            continue

    # Concatenate velocities from all laps
    combined_velocities = {
        corner: np.concatenate(vel_list) if vel_list else np.array([])
        for corner, vel_list in all_velocities.items()
    }

    # Process each corner with combined velocities
    fl_data = _process_corner_from_velocity(
        combined_velocities["FL"],
        "FL",
        bin_size,
        max_velocity,
        velocity_ranges,
    )
    fr_data = _process_corner_from_velocity(
        combined_velocities["FR"],
        "FR",
        bin_size,
        max_velocity,
        velocity_ranges,
    )
    rl_data = _process_corner_from_velocity(
        combined_velocities["RL"],
        "RL",
        bin_size,
        max_velocity,
        velocity_ranges,
    )
    rr_data = _process_corner_from_velocity(
        combined_velocities["RR"],
        "RR",
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


def _process_corner_from_velocity(
    velocity: np.ndarray,
    corner_name: str,
    bin_size: float,
    max_velocity: float,
    ranges: VelocityRanges,
) -> CornerVelocityData:
    """Process a single corner from pre-computed velocity data.

    Parameters
    ----------
    velocity : np.ndarray
        Wheel velocity in mm/s (already computed and concatenated).
    corner_name : str
        Name of the corner ("FL", "FR", "RL", "RR").
    bin_size : float
        Histogram bin size in mm/s.
    max_velocity : float
        Maximum velocity for histogram range.
    ranges : VelocityRanges
        Velocity thresholds for categorization.

    Returns
    -------
    CornerVelocityData
        Processed corner data with histogram and statistics.
    """
    # Handle empty velocity array
    if len(velocity) == 0:
        # Return zeros for empty data
        histogram, bin_edges, bin_centers = compute_velocity_histogram(
            np.array([0.0]), bin_size, max_velocity
        )
        return CornerVelocityData(
            corner_name=corner_name,
            velocity=velocity,
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

    # Compute histogram
    histogram, bin_edges, bin_centers = compute_velocity_histogram(
        velocity, bin_size, max_velocity
    )

    # Find zero bin percentage
    zero_bin_idx = int(np.argmin(np.abs(bin_centers)))
    pct_zero_bin = float(histogram[zero_bin_idx]) if len(histogram) > 0 else 0.0

    # Compute statistics
    stats = compute_velocity_stats(velocity, ranges)

    return CornerVelocityData(
        corner_name=corner_name,
        velocity=velocity,
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
