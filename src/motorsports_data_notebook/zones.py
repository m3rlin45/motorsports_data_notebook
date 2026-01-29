"""Braking and acceleration zone detection and analysis.

This module provides functions for detecting braking and acceleration zones
from telemetry data and building track segment definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from libxrk.base import LogFile

    from .corners import Corner


@dataclass
class TrackSegment:
    """Represents a segment of the track (braking zone, corner, or acceleration zone).

    Attributes
    ----------
    id : int
        Segment ID (1-indexed).
    segment_type : str
        Type of segment: 'braking', 'corner', or 'acceleration'.
    start_dist : float
        Distance along track where segment begins (meters).
    end_dist : float
        Distance along track where segment ends (meters).
    name : str
        Segment name (e.g., "Turn 1 Braking", "Turn 1", "Turn 1 Exit").
    corner_id : int
        ID of the associated corner.
    apex_dist : float, optional
        Distance along track of the apex (only for corner segments).
    """

    id: int
    segment_type: str  # 'braking', 'corner', or 'acceleration'
    start_dist: float
    end_dist: float
    name: str
    corner_id: int
    apex_dist: float | None = None  # Only for corner segments

    @property
    def length(self) -> float:
        """Segment length in meters."""
        return self.end_dist - self.start_dist


def get_segment_mask(
    channels: pd.DataFrame,
    segment: TrackSegment,
    *,
    margin: float = 0.0,
) -> pd.Series:
    """Return a boolean mask for rows within a segment's distance range.

    The caller is responsible for pre-filtering by lap (e.g., using timecodes).
    This function only filters by distance within the segment boundaries.

    Parameters
    ----------
    channels : pd.DataFrame
        Channel data with a 'distance_m' column.
    segment : TrackSegment
        The segment to create a mask for.
    margin : float, default=0.0
        Extra distance (meters) to include before and after the segment.

    Returns
    -------
    pd.Series
        Boolean mask where True indicates rows within the segment range.

    Examples
    --------
    >>> # Get data for Turn 1 with 50m margin
    >>> mask = get_segment_mask(lap_channels, turn1_segment, margin=50)
    >>> turn1_data = lap_channels[mask]
    """
    return (channels["distance_m"] >= segment.start_dist - margin) & (
        channels["distance_m"] <= segment.end_dist + margin
    )


# Channels typically needed for corner data analysis
CORNER_DATA_CHANNELS = ["distance_m", "PPS", "BrakePress", "LateralAcc", "SteerAngle"]


def get_corner_data(
    log: "LogFile",
    corner: "Corner",
    channel_names: list[str] | None = None,
    *,
    margin: float = 50.0,
) -> pd.DataFrame:
    """Get channel data for a specific corner from a pre-filtered LogFile.

    The caller must filter the LogFile to a single lap before calling this function
    using log.filter_by_lap(lap_num).

    Parameters
    ----------
    log : LogFile
        The LogFile pre-filtered to a single lap (via log.filter_by_lap()).
    corner : Corner
        The corner object with start_dist and end_dist.
    channel_names : list[str], optional
        Names of channels to extract. If None, uses CORNER_DATA_CHANNELS.
    margin : float, default=50.0
        Extra distance (meters) to include before and after the corner.

    Returns
    -------
    pd.DataFrame
        Filtered channel data for the specified corner.

    Examples
    --------
    >>> lap_log = log.filter_by_lap(3)
    >>> corner_data = get_corner_data(lap_log, corners[0], margin=50)
    >>> fig = visualize_throttle_acceptance(
    ...     distance=corner_data["distance_m"],
    ...     throttle=corner_data["PPS"],
    ...     lateral_g=corner_data["LateralAcc"].abs(),
    ...     corner=corners[0],
    ...     throttle_acceptance_result=result,
    ... )
    """
    # Use default channels if not specified
    if channel_names is None:
        channel_names = CORNER_DATA_CHANNELS.copy()

    # Use libxrk 0.5.0 methods to select and resample channels
    aligned = log.select_channels(channel_names).resample_to_channel("distance_m").channels

    # Convert to DataFrame
    lap_data = pd.DataFrame({name: aligned[name].column(name).to_numpy() for name in channel_names})

    # Filter by corner distance
    corner_mask = (lap_data["distance_m"] >= corner.start_dist - margin) & (
        lap_data["distance_m"] <= corner.end_dist + margin
    )

    result: pd.DataFrame = lap_data[corner_mask].copy()
    return result


def identify_zones_single_lap(
    distance: np.ndarray,
    brake_press: np.ndarray,
    throttle: np.ndarray,
    speed: np.ndarray,
    brake_threshold: float = 5,
    throttle_threshold: float = 20,
    gear_change_time: float = 1.5,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Identify braking and acceleration zones for a single lap.

    Parameters
    ----------
    distance : array-like
        Distance along track in meters.
    brake_press : array-like
        Brake pressure % (0-100).
    throttle : array-like
        Throttle position % (0-100).
    speed : array-like
        Speed in m/s.
    brake_threshold : float, default=5
        Minimum brake pressure % to consider as braking.
    throttle_threshold : float, default=20
        Minimum throttle % to consider as accelerating.
    gear_change_time : float, default=1.5
        Maximum time (seconds) to bridge across gear changes in accel zones.

    Returns
    -------
    tuple[list[tuple[float, float]], list[tuple[float, float]]]
        (braking_zones, accel_zones) - lists of (start_dist, end_dist) tuples.

    Examples
    --------
    >>> braking, accel = identify_zones_single_lap(
    ...     distance=lap_data['distance_m'].values,
    ...     brake_press=lap_data['BrakePress'].values,
    ...     throttle=lap_data['PPS'].values,
    ...     speed=lap_data['GPS Speed'].values
    ... )
    """
    distance = np.array(distance)
    brake_press = np.array(brake_press)
    throttle = np.array(throttle)
    speed = np.array(speed)

    braking_zones = []
    accel_zones = []

    # Find braking zones
    is_braking = brake_press > brake_threshold
    brake_start = None
    for i in range(len(is_braking)):
        if is_braking[i] and brake_start is None:
            brake_start = i
        elif not is_braking[i] and brake_start is not None:
            braking_zones.append((distance[brake_start], distance[i - 1]))
            brake_start = None
    if brake_start is not None:
        braking_zones.append((distance[brake_start], distance[-1]))

    # Find acceleration zones (throttle high, brake low)
    # Also track the end index for time-based gap calculation
    is_accel = (throttle > throttle_threshold) & (brake_press < brake_threshold)
    accel_start = None
    accel_zones_with_idx = []  # Store (start_dist, end_dist, end_idx)
    for i in range(len(is_accel)):
        if is_accel[i] and accel_start is None:
            accel_start = i
        elif not is_accel[i] and accel_start is not None:
            accel_zones_with_idx.append((distance[accel_start], distance[i - 1], i - 1))
            accel_start = None
    if accel_start is not None:
        accel_zones_with_idx.append((distance[accel_start], distance[-1], len(distance) - 1))

    # Merge acceleration zones that are close together (gear change gaps)
    # Use time-based threshold: gap_time = gap_distance / avg_speed_in_gap
    if len(accel_zones_with_idx) > 1:
        merged_accel = [accel_zones_with_idx[0]]
        for start, end, end_idx in accel_zones_with_idx[1:]:
            prev_start, prev_end, prev_end_idx = merged_accel[-1]

            # Calculate time to cross the gap based on average speed in the gap
            gap_distance = start - prev_end
            # Find start index of current zone
            start_idx = np.searchsorted(distance, start)
            if start_idx > prev_end_idx and start_idx < len(speed):
                # Average speed in the gap region
                gap_speed = np.mean(speed[prev_end_idx : start_idx + 1])
                if gap_speed > 0:
                    gap_time = gap_distance / gap_speed
                else:
                    gap_time = float("inf")
            else:
                gap_time = float("inf")

            # If gap time is small (gear change) and no braking in between, merge
            if gap_time <= gear_change_time:
                # Check no significant braking in the gap
                gap_has_braking = False
                for bz_start, bz_end in braking_zones:
                    if bz_start < start and bz_end > prev_end:
                        gap_has_braking = True
                        break
                if not gap_has_braking:
                    merged_accel[-1] = (prev_start, end, end_idx)
                    continue
            merged_accel.append((start, end, end_idx))
        accel_zones = [(s, e) for s, e, _ in merged_accel]
    else:
        accel_zones = [(s, e) for s, e, _ in accel_zones_with_idx]

    return braking_zones, accel_zones


def average_zones_across_laps(
    all_braking_zones: list[list[tuple[float, float]]],
    all_accel_zones: list[list[tuple[float, float]]],
    track_length: float,
    resolution: float = 1.0,
    threshold: float = 0.5,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Average braking and acceleration zones across multiple laps.

    Uses a grid-based voting system where each point on the track is marked
    as braking/accelerating if at least threshold fraction of laps agree.

    Parameters
    ----------
    all_braking_zones : list[list[tuple[float, float]]]
        List of braking zone lists from each lap.
    all_accel_zones : list[list[tuple[float, float]]]
        List of acceleration zone lists from each lap.
    track_length : float
        Total track length in meters.
    resolution : float, default=1.0
        Distance resolution for averaging (meters).
    threshold : float, default=0.5
        Fraction of laps that must be braking/accelerating at a point to include it.

    Returns
    -------
    tuple[list[tuple[float, float]], list[tuple[float, float]]]
        (braking_zones, accel_zones) - averaged zones.

    Examples
    --------
    >>> braking_zones, accel_zones = average_zones_across_laps(
    ...     all_braking_zones, all_accel_zones,
    ...     track_length=4500, resolution=1.0, threshold=0.5
    ... )
    """
    n_laps = len(all_braking_zones)
    n_points = int(track_length / resolution) + 1

    # Create arrays to count how many laps are braking/accelerating at each point
    brake_counts = np.zeros(n_points)
    accel_counts = np.zeros(n_points)

    for lap_braking in all_braking_zones:
        for start_dist, end_dist in lap_braking:
            start_idx = int(start_dist / resolution)
            end_idx = min(int(end_dist / resolution) + 1, n_points)
            brake_counts[start_idx:end_idx] += 1

    for lap_accel in all_accel_zones:
        for start_dist, end_dist in lap_accel:
            start_idx = int(start_dist / resolution)
            end_idx = min(int(end_dist / resolution) + 1, n_points)
            accel_counts[start_idx:end_idx] += 1

    # Convert to zones where at least threshold fraction of laps agree
    min_laps = n_laps * threshold

    def extract_zones(counts, min_count, res):
        zones = []
        is_zone = counts >= min_count
        zone_start = None
        for i in range(len(is_zone)):
            if is_zone[i] and zone_start is None:
                zone_start = i
            elif not is_zone[i] and zone_start is not None:
                zones.append((zone_start * res, (i - 1) * res))
                zone_start = None
        if zone_start is not None:
            zones.append((zone_start * res, (len(is_zone) - 1) * res))
        return zones

    braking_zones = extract_zones(brake_counts, min_laps, resolution)
    accel_zones = extract_zones(accel_counts, min_laps, resolution)

    return braking_zones, accel_zones


def merge_accel_zones_by_time(
    accel_zones: list[tuple[float, float]],
    braking_zones: list[tuple[float, float]],
    distance_arr: np.ndarray,
    speed_arr: np.ndarray,
    max_gap_time: float = 1.5,
) -> list[tuple[float, float]]:
    """
    Merge acceleration zones separated by short time gaps (gear changes).

    This post-processing step merges acceleration zones that are separated
    by gaps that would take less than max_gap_time seconds to traverse,
    provided there's no braking zone in the gap.

    Parameters
    ----------
    accel_zones : list[tuple[float, float]]
        List of (start_dist, end_dist) tuples for acceleration zones.
    braking_zones : list[tuple[float, float]]
        List of (start_dist, end_dist) tuples for braking zones.
    distance_arr : array-like
        Distance array from reference lap (meters).
    speed_arr : array-like
        Speed array from reference lap (m/s).
    max_gap_time : float, default=1.5
        Maximum time (seconds) to bridge across gear changes.

    Returns
    -------
    list[tuple[float, float]]
        Merged acceleration zones.

    Examples
    --------
    >>> accel_zones = merge_accel_zones_by_time(
    ...     accel_zones, braking_zones,
    ...     lap_channels['distance_m'].values,
    ...     lap_channels['GPS Speed'].values,
    ...     max_gap_time=1.5
    ... )
    """
    if len(accel_zones) <= 1:
        return accel_zones

    distance_arr = np.array(distance_arr)
    speed_arr = np.array(speed_arr)

    merged = [list(accel_zones[0])]
    for start, end in accel_zones[1:]:
        prev_start, prev_end = merged[-1]
        gap_distance = start - prev_end

        # Find indices for the gap region
        prev_end_idx = np.searchsorted(distance_arr, prev_end)
        start_idx = np.searchsorted(distance_arr, start)

        # Calculate time to cross the gap
        if start_idx > prev_end_idx and start_idx < len(speed_arr):
            gap_speeds = speed_arr[prev_end_idx : start_idx + 1]
            avg_speed = np.mean(gap_speeds[gap_speeds > 0]) if np.any(gap_speeds > 0) else 0
            gap_time = gap_distance / avg_speed if avg_speed > 0 else float("inf")
        else:
            gap_time = float("inf")

        # Check if there's braking in the gap
        gap_has_braking = False
        for bz_start, bz_end in braking_zones:
            # Braking zone overlaps with gap
            if bz_start < start and bz_end > prev_end:
                gap_has_braking = True
                break

        # Merge if gap is short and no braking
        if gap_time <= max_gap_time and not gap_has_braking:
            merged[-1][1] = end  # Extend previous zone
        else:
            merged.append([start, end])

    return [(z[0], z[1]) for z in merged]


def create_track_segments(
    corners: list[Corner],
    braking_zones: list[tuple[float, float]],
    accel_zones: list[tuple[float, float]],
    track_length: float,
) -> list[TrackSegment]:
    """
    Create fixed segment definitions for the track.

    Each corner gets a braking zone before it and an acceleration zone after.
    Uses actual detected accel_zones to determine acceleration zone extents.

    Parameters
    ----------
    corners : list[Corner]
        List of detected corners from identify_corners() or identify_corners_from_curvature().
    braking_zones : list[tuple[float, float]]
        List of (start_dist, end_dist) tuples for braking zones.
    accel_zones : list[tuple[float, float]]
        List of (start_dist, end_dist) tuples for acceleration zones.
    track_length : float
        Total track length in meters.

    Returns
    -------
    list[TrackSegment]
        List of TrackSegment dataclass instances, sorted by start distance.

    Examples
    --------
    >>> segments = create_track_segments(corners, braking_zones, accel_zones, track_length)
    >>> for seg in segments:
    ...     print(f"[{seg.segment_type}] {seg.name}: {seg.start_dist:.0f}m - {seg.end_dist:.0f}m")
    """
    segments = []
    segment_id = 0

    # Sort corners by start distance
    sorted_corners = sorted(corners, key=lambda c: c.start_dist)

    for corner in sorted_corners:
        # Find braking zone that ends near/at this corner
        brake_start = None
        for bz_start, bz_end in braking_zones:
            # Braking zone should end within 100m of corner start or overlap
            if bz_end >= corner.start_dist - 100 and bz_start < corner.start_dist:
                brake_start = bz_start
                break

        # If no braking zone found, use 100m before corner as default
        if brake_start is None:
            brake_start = max(0, corner.start_dist - 100)

        # Braking segment
        segment_id += 1
        segments.append(
            TrackSegment(
                id=segment_id,
                segment_type="braking",
                start_dist=brake_start,
                end_dist=corner.start_dist,
                name=f"{corner.name} Braking",
                corner_id=corner.id,
            )
        )

        # Corner segment
        segment_id += 1
        segments.append(
            TrackSegment(
                id=segment_id,
                segment_type="corner",
                start_dist=corner.start_dist,
                end_dist=corner.end_dist,
                name=corner.name,
                corner_id=corner.id,
                apex_dist=corner.apex_dist,
            )
        )

        # Acceleration segment - use actual accel_zones data
        # Find the accel zone that starts near this corner's exit
        accel_end = corner.end_dist + 50  # Minimum default
        for az_start, az_end in accel_zones:
            # Accel zone should start near/within corner exit and extend beyond it
            if az_start <= corner.end_dist + 50 and az_end > corner.end_dist:
                # Use the full extent of this accel zone
                accel_end = az_end
                break

        # Don't extend past the next braking zone
        for bz_start, bz_end in braking_zones:
            if bz_start > corner.end_dist and bz_start < accel_end:
                accel_end = bz_start
                break

        segment_id += 1
        segments.append(
            TrackSegment(
                id=segment_id,
                segment_type="acceleration",
                start_dist=corner.end_dist,
                end_dist=max(corner.end_dist + 10, accel_end),  # At least 10m
                name=f"{corner.name} Exit",
                corner_id=corner.id,
            )
        )

    # Sort segments by start distance
    segments = sorted(segments, key=lambda s: s.start_dist)

    return segments


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


def detect_zones_averaged(
    log: "LogFile",
    top_laps: pd.DataFrame,
    channel_names: dict,
    resolution: float = 1.0,
    threshold: float = 0.5,
    max_gap_time: float = 1.5,
    brake_threshold: float = 5,
    throttle_threshold: float = 20,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Detect and average braking/acceleration zones across top laps.

    This is a convenience function that combines the full zone detection pipeline:
    1. Iterate over top laps and extract only required channels per-lap
    2. Call identify_zones_single_lap() for each lap
    3. Average zones across laps with average_zones_across_laps()
    4. Merge acceleration zones with merge_accel_zones_by_time()

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    top_laps : pd.DataFrame
        Laps to analyze (typically from get_top_laps()).
    channel_names : dict
        Channel name mapping. Required keys:
        - "throttle": Throttle position channel name (e.g., "PPS")
        - "brake": Brake pressure channel name (e.g., "BrakePress")
        - "gps_speed": GPS speed channel name (e.g., "GPS Speed")
    resolution : float, default=1.0
        Distance resolution for averaging (meters).
    threshold : float, default=0.5
        Fraction of laps that must agree for a zone to be included.
    max_gap_time : float, default=1.5
        Maximum time (seconds) to bridge across gear changes in accel zones.
    brake_threshold : float, default=5
        Minimum brake pressure % to consider as braking.
    throttle_threshold : float, default=20
        Minimum throttle % to consider as accelerating.

    Returns
    -------
    tuple[list[tuple[float, float]], list[tuple[float, float]]]
        (braking_zones, accel_zones) - averaged and post-processed zones.

    Raises
    ------
    KeyError
        If required keys are missing from channel_names.

    Examples
    --------
    >>> channel_names = {"throttle": "PPS", "brake": "BrakePress", "gps_speed": "GPS Speed"}
    >>> top_laps = get_top_laps(laps)
    >>> braking_zones, accel_zones = detect_zones_averaged(log, top_laps, channel_names)
    """
    # Validate required channel names
    _validate_channel_names(
        channel_names, ["throttle", "brake", "gps_speed"], "detect_zones_averaged"
    )

    # Build list of channels to extract
    throttle_ch = channel_names["throttle"]
    brake_ch = channel_names["brake"]
    speed_ch = channel_names["gps_speed"]
    zone_channels = ["distance_m", brake_ch, throttle_ch, speed_ch]

    all_braking_zones: list[list[tuple[float, float]]] = []
    all_accel_zones: list[list[tuple[float, float]]] = []

    # Store reference lap data for post-processing (first lap processed)
    reference_distance: np.ndarray | None = None
    reference_speed: np.ndarray | None = None

    for _, lap in top_laps.iterrows():
        lap_num = int(lap["num"])

        # Use libxrk 0.5.0 methods to filter by lap, select channels, and resample
        aligned = (
            log.filter_by_lap(lap_num)
            .select_channels(zone_channels)
            .resample_to_channel("distance_m")
            .channels
        )

        # Extract arrays
        distance = aligned["distance_m"].column("distance_m").to_numpy()
        brake_press = aligned[brake_ch].column(brake_ch).to_numpy()
        pps = aligned[throttle_ch].column(throttle_ch).to_numpy()
        speed = aligned[speed_ch].column(speed_ch).to_numpy()

        if len(distance) < 10:
            continue

        # Store first lap as reference for post-processing
        if reference_distance is None:
            reference_distance = distance
            reference_speed = speed

        braking, accel = identify_zones_single_lap(
            distance,
            brake_press,
            pps,
            speed,
            brake_threshold=brake_threshold,
            throttle_threshold=throttle_threshold,
        )
        all_braking_zones.append(braking)
        all_accel_zones.append(accel)

    if not all_braking_zones or reference_distance is None:
        return [], []

    # Average zones across laps
    track_length = float(reference_distance.max())
    braking_zones, accel_zones = average_zones_across_laps(
        all_braking_zones,
        all_accel_zones,
        track_length=track_length,
        resolution=resolution,
        threshold=threshold,
    )

    # Post-process: merge acceleration zones separated by short time gaps
    assert reference_speed is not None
    accel_zones = merge_accel_zones_by_time(
        accel_zones,
        braking_zones,
        reference_distance,
        reference_speed,
        max_gap_time=max_gap_time,
    )

    return braking_zones, accel_zones


def _find_braking_point(
    lap_data: pd.DataFrame,
    segment: TrackSegment,
    brake_col: str,
    brake_threshold: float = 5,
) -> float | None:
    """Find the distance where braking starts within a segment."""
    mask = (lap_data["distance_m"] >= segment.start_dist) & (
        lap_data["distance_m"] <= segment.end_dist
    )
    seg_data = lap_data[mask]

    if len(seg_data) == 0:
        return None

    brake_points = seg_data[seg_data[brake_col] > brake_threshold]
    if len(brake_points) > 0:
        return float(brake_points["distance_m"].iloc[0])
    return None


def _find_throttle_point(
    lap_data: pd.DataFrame,
    segment: TrackSegment,
    throttle_col: str,
    brake_col: str,
    throttle_threshold: float = 20,
    brake_threshold: float = 5,
) -> float | None:
    """Find the distance where throttle application starts within a segment."""
    mask = (lap_data["distance_m"] >= segment.start_dist) & (
        lap_data["distance_m"] <= segment.end_dist
    )
    seg_data = lap_data[mask]

    if len(seg_data) == 0:
        return None

    throttle_points = seg_data[
        (seg_data[throttle_col] > throttle_threshold) & (seg_data[brake_col] < brake_threshold)
    ]
    if len(throttle_points) > 0:
        return float(throttle_points["distance_m"].iloc[0])
    return None


def _find_min_speed(lap_data: pd.DataFrame, segment: TrackSegment, speed_col: str) -> float | None:
    """Find minimum speed within a segment (for corners)."""
    mask = (lap_data["distance_m"] >= segment.start_dist) & (
        lap_data["distance_m"] <= segment.end_dist
    )
    seg_data = lap_data[mask]

    if len(seg_data) == 0:
        return None

    return float(seg_data[speed_col].min())


def compute_segment_stats(
    log: "LogFile",
    laps: pd.DataFrame,
    segments: list[TrackSegment],
    channel_names: dict,
    brake_threshold: float = 5,
    throttle_threshold: float = 20,
) -> pd.DataFrame:
    """Compute per-lap statistics for each track segment.

    Computes:
    - For braking segments: braking point distance and offset from segment start
    - For corner segments: minimum speed through the corner
    - For acceleration segments: throttle application point and offset

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    laps : pd.DataFrame
        Laps to analyze (with 'start_time', 'end_time', 'num', 'lap_time' columns).
    segments : list[TrackSegment]
        Track segments to compute statistics for.
    channel_names : dict
        Channel name mapping. Required keys:
        - "throttle": Throttle position channel name (e.g., "PPS")
        - "brake": Brake pressure channel name (e.g., "BrakePress")
    brake_threshold : float, default=5
        Minimum brake pressure % to consider as braking.
    throttle_threshold : float, default=20
        Minimum throttle % to consider as accelerating.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - segment_id, segment_name, segment_type, corner_id
        - lap_num, lap_time
        - braking_point, brake_offset (for braking segments)
        - min_speed (for corner segments)
        - throttle_point, throttle_offset (for acceleration segments)

    Raises
    ------
    KeyError
        If required keys are missing from channel_names.

    Examples
    --------
    >>> channel_names = {"throttle": "PPS", "brake": "BrakePress"}
    >>> top_laps = get_top_laps(laps)
    >>> stats_df = compute_segment_stats(log, top_laps, segments, channel_names)
    >>> braking_stats = stats_df[stats_df["segment_type"] == "braking"]
    """
    # Validate required channel names
    _validate_channel_names(channel_names, ["throttle", "brake"], "compute_segment_stats")

    # Get channel names
    throttle_ch = channel_names["throttle"]
    brake_ch = channel_names["brake"]
    stats_channels = ["distance_m", "speed_kmh", brake_ch, throttle_ch]

    all_lap_stats = []

    for _, lap in laps.iterrows():
        lap_num = int(lap["num"])

        # Use libxrk 0.5.0 methods to filter by lap, select channels, and resample
        aligned = (
            log.filter_by_lap(lap_num)
            .select_channels(stats_channels)
            .resample_to_channel("distance_m")
            .channels
        )

        # Convert to DataFrame for the helper functions
        lap_data = pd.DataFrame(
            {
                "distance_m": aligned["distance_m"].column("distance_m").to_numpy(),
                "speed_kmh": aligned["speed_kmh"].column("speed_kmh").to_numpy(),
                brake_ch: aligned[brake_ch].column(brake_ch).to_numpy(),
                throttle_ch: aligned[throttle_ch].column(throttle_ch).to_numpy(),
            }
        )

        if len(lap_data) < 10:
            continue

        for seg in segments:
            stat: dict = {
                "segment_id": seg.id,
                "segment_name": seg.name,
                "segment_type": seg.segment_type,
                "corner_id": seg.corner_id,
                "lap_num": lap["num"],
                "lap_time": lap["lap_time"],
            }

            if seg.segment_type == "braking":
                braking_point = _find_braking_point(lap_data, seg, brake_ch, brake_threshold)
                stat["braking_point"] = braking_point
                if braking_point is not None:
                    stat["brake_offset"] = braking_point - seg.start_dist

            elif seg.segment_type == "corner":
                stat["min_speed"] = _find_min_speed(lap_data, seg, "speed_kmh")

            elif seg.segment_type == "acceleration":
                throttle_point = _find_throttle_point(
                    lap_data, seg, throttle_ch, brake_ch, throttle_threshold, brake_threshold
                )
                stat["throttle_point"] = throttle_point
                if throttle_point is not None:
                    stat["throttle_offset"] = throttle_point - seg.start_dist

            all_lap_stats.append(stat)

    return pd.DataFrame(all_lap_stats)
