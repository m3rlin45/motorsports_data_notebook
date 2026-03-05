"""Braking and acceleration zone detection and analysis.

This module provides functions for detecting braking and acceleration zones
from telemetry data and building track segment definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


from ._util import infer_channel_scale as _infer_channel_scale
from ._util import validate_channel_names as _validate_channel_names

if TYPE_CHECKING:
    from ._types import LogFile
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
    brake_threshold: float | None = None,
    throttle_threshold: float | None = None,
    gear_change_time: float = 1.5,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Identify braking and acceleration zones for a single lap.

    Parameters
    ----------
    distance : array-like
        Distance along track in meters.
    brake_press : array-like
        Brake pressure values.
    throttle : array-like
        Throttle position values.
    speed : array-like
        Speed in m/s.
    brake_threshold : float or None, default=None
        Minimum brake pressure to consider as braking.
        If None, auto-detected from data scale (5% of scale).
    throttle_threshold : float or None, default=None
        Minimum throttle to consider as accelerating.
        If None, auto-detected from data scale (20% of scale).
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

    if brake_threshold is None:
        brake_threshold = 0.05 * _infer_channel_scale(brake_press)
    if throttle_threshold is None:
        throttle_threshold = 0.20 * _infer_channel_scale(throttle)

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


def detect_zones_from_arrays(
    distances: list[np.ndarray],
    brake_presses: list[np.ndarray],
    throttles: list[np.ndarray],
    speeds: list[np.ndarray],
    resolution: float = 1.0,
    threshold: float = 0.5,
    max_gap_time: float = 1.5,
    brake_threshold: float | None = None,
    throttle_threshold: float | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Detect and average braking/acceleration zones from pre-extracted arrays.

    Same as detect_zones_averaged() but accepts numpy arrays directly
    instead of a LogFile. Each list has one entry per lap.

    Parameters
    ----------
    distances : list[np.ndarray]
        Per-lap distance arrays (meters).
    brake_presses : list[np.ndarray]
        Per-lap brake pressure arrays.
    throttles : list[np.ndarray]
        Per-lap throttle position arrays.
    speeds : list[np.ndarray]
        Per-lap speed arrays (m/s).
    resolution : float, default=1.0
        Distance resolution for averaging (meters).
    threshold : float, default=0.5
        Fraction of laps that must agree for a zone to be included.
    max_gap_time : float, default=1.5
        Maximum time (seconds) to bridge across gear changes in accel zones.
    brake_threshold : float or None, default=None
        Minimum brake pressure to consider as braking.
        If None, auto-detected from data scale.
    throttle_threshold : float or None, default=None
        Minimum throttle to consider as accelerating.
        If None, auto-detected from data scale.

    Returns
    -------
    tuple[list[tuple[float, float]], list[tuple[float, float]]]
        (braking_zones, accel_zones) - averaged and post-processed zones.
    """
    all_braking_zones: list[list[tuple[float, float]]] = []
    all_accel_zones: list[list[tuple[float, float]]] = []

    reference_distance: np.ndarray | None = None
    reference_speed: np.ndarray | None = None

    for distance, brake_press, pps, speed in zip(distances, brake_presses, throttles, speeds):
        if len(distance) < 10:
            continue

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

    track_length = float(reference_distance.max())
    braking_zones, accel_zones = average_zones_across_laps(
        all_braking_zones,
        all_accel_zones,
        track_length=track_length,
        resolution=resolution,
        threshold=threshold,
    )

    assert reference_speed is not None
    accel_zones = merge_accel_zones_by_time(
        accel_zones,
        braking_zones,
        reference_distance,
        reference_speed,
        max_gap_time=max_gap_time,
    )

    return braking_zones, accel_zones


def detect_zones_averaged(
    log: "LogFile",
    top_laps: pd.DataFrame,
    channel_names: dict,
    resolution: float = 1.0,
    threshold: float = 0.5,
    max_gap_time: float = 1.5,
    brake_threshold: float | None = None,
    throttle_threshold: float | None = None,
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
    brake_threshold : float or None, default=None
        Minimum brake pressure to consider as braking.
        If None, auto-detected from data scale.
    throttle_threshold : float or None, default=None
        Minimum throttle to consider as accelerating.
        If None, auto-detected from data scale.

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
    _validate_channel_names(
        channel_names, ["throttle", "brake", "gps_speed"], "detect_zones_averaged"
    )

    throttle_ch = channel_names["throttle"]
    brake_ch = channel_names["brake"]
    speed_ch = channel_names["gps_speed"]
    zone_channels = ["distance_m", brake_ch, throttle_ch, speed_ch]

    all_distances: list[np.ndarray] = []
    all_brake_presses: list[np.ndarray] = []
    all_throttles: list[np.ndarray] = []
    all_speeds: list[np.ndarray] = []

    for _, lap in top_laps.iterrows():
        lap_num = int(lap["num"])
        aligned = (
            log.filter_by_lap(lap_num)
            .select_channels(zone_channels)
            .resample_to_channel("distance_m")
            .channels
        )
        all_distances.append(aligned["distance_m"].column("distance_m").to_numpy())
        all_brake_presses.append(aligned[brake_ch].column(brake_ch).to_numpy())
        all_throttles.append(aligned[throttle_ch].column(throttle_ch).to_numpy())
        all_speeds.append(aligned[speed_ch].column(speed_ch).to_numpy())

    return detect_zones_from_arrays(
        all_distances,
        all_brake_presses,
        all_throttles,
        all_speeds,
        resolution=resolution,
        threshold=threshold,
        max_gap_time=max_gap_time,
        brake_threshold=brake_threshold,
        throttle_threshold=throttle_threshold,
    )


def _find_braking_point_np(
    distance: np.ndarray,
    brake: np.ndarray,
    start_dist: float,
    end_dist: float,
    brake_threshold: float,
) -> float | None:
    """Find the distance where braking starts within a segment (numpy)."""
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    if si >= ei:
        return None
    seg_brake = brake[si:ei]
    mask = seg_brake > brake_threshold
    idx = np.argmax(mask)
    return float(distance[si + idx]) if mask[idx] else None


def _find_peak_brake_np(
    brake: np.ndarray,
    distance: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> tuple[float, float] | None:
    """Find peak brake pressure and its distance within a segment.

    Parameters
    ----------
    brake : np.ndarray
        Brake pressure array.
    distance : np.ndarray
        Distance array (meters).
    start_dist : float
        Start of segment.
    end_dist : float
        End of segment.

    Returns
    -------
    tuple[float, float] | None
        (peak_value, peak_dist) or None if segment is empty.
    """
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    if si >= ei:
        return None
    seg_brake = brake[si:ei]
    peak_idx = np.argmax(seg_brake)
    return float(seg_brake[peak_idx]), float(distance[si + peak_idx])


def _find_brake_release_np(
    brake: np.ndarray,
    distance: np.ndarray,
    start_dist: float,
    end_dist: float,
    threshold_pct: float = 0.10,
) -> float | None:
    """Find where brake pressure drops below threshold_pct * peak_value.

    Searches forward from peak brake position to end of segment.

    Parameters
    ----------
    brake : np.ndarray
        Brake pressure array.
    distance : np.ndarray
        Distance array (meters).
    start_dist : float
        Start of segment.
    end_dist : float
        End of segment.
    threshold_pct : float, default=0.10
        Fraction of peak below which brake is considered released.

    Returns
    -------
    float | None
        Distance at release point, or None if brake held through segment.
    """
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    if si >= ei:
        return None
    seg_brake = brake[si:ei]
    peak_idx = np.argmax(seg_brake)
    peak_value = seg_brake[peak_idx]
    if peak_value <= 0:
        return None
    release_threshold = threshold_pct * peak_value
    # Search forward from peak to end of segment
    after_peak = seg_brake[peak_idx:]
    release_mask = after_peak < release_threshold
    if not np.any(release_mask):
        return None
    release_idx = np.argmax(release_mask)
    return float(distance[si + peak_idx + release_idx])


def _compute_deceleration(
    speed: np.ndarray,
    distance: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> float | None:
    """Compute mean deceleration in g between two distances.

    Uses v² = v0² - 2*a*d to compute deceleration from speed change
    over distance. Speed is expected in km/h.

    Parameters
    ----------
    speed : np.ndarray
        Speed array (km/h).
    distance : np.ndarray
        Distance array (meters).
    start_dist : float
        Start distance (braking point).
    end_dist : float
        End distance (brake release).

    Returns
    -------
    float | None
        Mean deceleration in g (positive value), or None if insufficient data.
    """
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    if si >= ei or ei > len(speed):
        return None
    v0 = speed[si] / 3.6  # km/h -> m/s
    v1 = speed[ei - 1] / 3.6
    d = distance[ei - 1] - distance[si]
    if d <= 0:
        return None
    # v1² = v0² - 2*a*d  =>  a = (v0² - v1²) / (2*d)
    decel = (v0**2 - v1**2) / (2 * d) / 9.81
    return float(decel) if decel > 0 else None


def _find_throttle_point_np(
    distance: np.ndarray,
    throttle: np.ndarray,
    brake: np.ndarray,
    start_dist: float,
    end_dist: float,
    throttle_threshold: float,
    brake_threshold: float,
) -> float | None:
    """Find the distance where throttle application starts within a segment (numpy)."""
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    if si >= ei:
        return None
    mask = (throttle[si:ei] > throttle_threshold) & (brake[si:ei] < brake_threshold)
    idx = np.argmax(mask)
    return float(distance[si + idx]) if mask[idx] else None


def _find_min_speed_np(
    distance: np.ndarray,
    speed: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> float | None:
    """Find minimum speed within a segment (numpy)."""
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    return float(speed[si:ei].min()) if si < ei else None


def _find_exit_speed_np(
    distance: np.ndarray,
    speed: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> float | None:
    """Find speed at the exit (end) of a segment (numpy)."""
    si = np.searchsorted(distance, start_dist)
    ei = np.searchsorted(distance, end_dist, side="right")
    return float(speed[ei - 1]) if si < ei else None


def _find_braking_point(
    lap_data: pd.DataFrame,
    segment: TrackSegment,
    brake_col: str,
    brake_threshold: float = 5,
) -> float | None:
    """Find the distance where braking starts within a segment."""
    return _find_braking_point_np(
        lap_data["distance_m"].to_numpy(),
        lap_data[brake_col].to_numpy(),
        segment.start_dist,
        segment.end_dist,
        brake_threshold,
    )


def _find_throttle_point(
    lap_data: pd.DataFrame,
    segment: TrackSegment,
    throttle_col: str,
    brake_col: str,
    throttle_threshold: float = 20,
    brake_threshold: float = 5,
) -> float | None:
    """Find the distance where throttle application starts within a segment."""
    return _find_throttle_point_np(
        lap_data["distance_m"].to_numpy(),
        lap_data[throttle_col].to_numpy(),
        lap_data[brake_col].to_numpy(),
        segment.start_dist,
        segment.end_dist,
        throttle_threshold,
        brake_threshold,
    )


def _find_min_speed(lap_data: pd.DataFrame, segment: TrackSegment, speed_col: str) -> float | None:
    """Find minimum speed within a segment (for corners)."""
    return _find_min_speed_np(
        lap_data["distance_m"].to_numpy(),
        lap_data[speed_col].to_numpy(),
        segment.start_dist,
        segment.end_dist,
    )


def _find_exit_speed(lap_data: pd.DataFrame, segment: TrackSegment, speed_col: str) -> float | None:
    """Find speed at the exit (end) of a segment."""
    return _find_exit_speed_np(
        lap_data["distance_m"].to_numpy(),
        lap_data[speed_col].to_numpy(),
        segment.start_dist,
        segment.end_dist,
    )


def compute_segment_stats_from_arrays(
    distances: list[np.ndarray],
    speeds: list[np.ndarray],
    brakes: list[np.ndarray],
    throttles: list[np.ndarray],
    lap_nums: list[int],
    lap_times: list[float],
    segments: list[TrackSegment],
    brake_threshold: float | None = None,
    throttle_threshold: float | None = None,
) -> pd.DataFrame:
    """Compute per-lap segment stats from pre-extracted numpy arrays.

    Same output as compute_segment_stats() but no LogFile needed.

    Parameters
    ----------
    distances : list[np.ndarray]
        Per-lap distance arrays (meters).
    speeds : list[np.ndarray]
        Per-lap speed arrays (km/h).
    brakes : list[np.ndarray]
        Per-lap brake pressure arrays.
    throttles : list[np.ndarray]
        Per-lap throttle position arrays.
    lap_nums : list[int]
        Lap numbers corresponding to each array set.
    lap_times : list[float]
        Lap times corresponding to each array set.
    segments : list[TrackSegment]
        Track segments to compute statistics for.
    brake_threshold : float or None, default=None
        Minimum brake pressure to consider as braking.
        If None, auto-detected from data scale.
    throttle_threshold : float or None, default=None
        Minimum throttle to consider as accelerating.
        If None, auto-detected from data scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - segment_id, segment_name, segment_type, corner_id
        - lap_num, lap_time
        - braking_point, brake_offset (for braking segments)
        - peak_brake, peak_brake_dist (for braking segments)
        - brake_release_point, brake_release_offset (for braking segments)
        - entry_speed (km/h at braking point, for braking segments)
        - braking_distance, mean_decel_g (for braking segments)
        - min_speed, exit_speed (for corner segments)
        - throttle_point, throttle_offset (for acceleration segments)
    """
    all_lap_stats = []
    thresholds_resolved = False

    for distance, speed, brake, throttle, lap_num, lap_time in zip(
        distances, speeds, brakes, throttles, lap_nums, lap_times
    ):
        if len(distance) < 10:
            continue

        if not thresholds_resolved:
            if brake_threshold is None:
                brake_threshold = 0.05 * _infer_channel_scale(brake)
            if throttle_threshold is None:
                throttle_threshold = 0.20 * _infer_channel_scale(throttle)
            thresholds_resolved = True

        assert brake_threshold is not None
        assert throttle_threshold is not None
        for seg_idx, seg in enumerate(segments):
            stat: dict = {
                "segment_id": seg.id,
                "segment_name": seg.name,
                "segment_type": seg.segment_type,
                "corner_id": seg.corner_id,
                "lap_num": lap_num,
                "lap_time": lap_time,
            }

            if seg.segment_type == "braking":
                braking_point = _find_braking_point_np(
                    distance, brake, seg.start_dist, seg.end_dist, brake_threshold
                )
                stat["braking_point"] = braking_point
                if braking_point is not None:
                    stat["brake_offset"] = braking_point - seg.start_dist

                # Peak brake pressure
                peak_result = _find_peak_brake_np(brake, distance, seg.start_dist, seg.end_dist)
                if peak_result is not None:
                    stat["peak_brake"] = peak_result[0]
                    stat["peak_brake_dist"] = peak_result[1]

                # Brake release point — extend search into next corner (trail braking)
                release_end = seg.end_dist
                if seg_idx + 1 < len(segments) and segments[seg_idx + 1].segment_type == "corner":
                    release_end = segments[seg_idx + 1].end_dist
                release_dist = _find_brake_release_np(brake, distance, seg.start_dist, release_end)
                stat["brake_release_point"] = release_dist
                if release_dist is not None:
                    stat["brake_release_offset"] = release_dist - seg.start_dist

                # Entry speed (speed at braking point)
                if braking_point is not None:
                    bp_idx = np.searchsorted(distance, braking_point)
                    if bp_idx < len(speed):
                        stat["entry_speed"] = float(speed[bp_idx])

                # Braking distance and mean deceleration
                if braking_point is not None and release_dist is not None:
                    stat["braking_distance"] = release_dist - braking_point
                    decel = _compute_deceleration(speed, distance, braking_point, release_dist)
                    stat["mean_decel_g"] = decel

            elif seg.segment_type == "corner":
                stat["min_speed"] = _find_min_speed_np(
                    distance, speed, seg.start_dist, seg.end_dist
                )
                stat["exit_speed"] = _find_exit_speed_np(
                    distance, speed, seg.start_dist, seg.end_dist
                )

            elif seg.segment_type == "acceleration":
                throttle_point = _find_throttle_point_np(
                    distance,
                    throttle,
                    brake,
                    seg.start_dist,
                    seg.end_dist,
                    throttle_threshold,
                    brake_threshold,
                )
                stat["throttle_point"] = throttle_point
                if throttle_point is not None:
                    stat["throttle_offset"] = throttle_point - seg.start_dist

            all_lap_stats.append(stat)

    return pd.DataFrame(all_lap_stats)


def compute_segment_stats(
    log: "LogFile",
    laps: pd.DataFrame,
    segments: list[TrackSegment],
    channel_names: dict,
    brake_threshold: float | None = None,
    throttle_threshold: float | None = None,
) -> pd.DataFrame:
    """Compute per-lap statistics for each track segment.

    Computes:
    - For braking segments: braking point distance and offset from segment start
    - For corner segments: minimum speed and exit speed through the corner
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
    brake_threshold : float or None, default=None
        Minimum brake pressure to consider as braking.
        If None, auto-detected from data scale.
    throttle_threshold : float or None, default=None
        Minimum throttle to consider as accelerating.
        If None, auto-detected from data scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - segment_id, segment_name, segment_type, corner_id
        - lap_num, lap_time
        - braking_point, brake_offset (for braking segments)
        - peak_brake, peak_brake_dist (for braking segments)
        - brake_release_point, brake_release_offset (for braking segments)
        - entry_speed (km/h at braking point, for braking segments)
        - braking_distance, mean_decel_g (for braking segments)
        - min_speed, exit_speed (for corner segments)
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
    _validate_channel_names(channel_names, ["throttle", "brake"], "compute_segment_stats")

    throttle_ch = channel_names["throttle"]
    brake_ch = channel_names["brake"]
    stats_channels = ["distance_m", "speed_kmh", brake_ch, throttle_ch]

    all_distances: list[np.ndarray] = []
    all_speeds: list[np.ndarray] = []
    all_brakes: list[np.ndarray] = []
    all_throttles: list[np.ndarray] = []
    lap_nums: list[int] = []
    lap_times: list[float] = []

    for _, lap in laps.iterrows():
        lap_num = int(lap["num"])
        aligned = (
            log.filter_by_lap(lap_num)
            .select_channels(stats_channels)
            .resample_to_channel("distance_m")
            .channels
        )
        all_distances.append(aligned["distance_m"].column("distance_m").to_numpy())
        all_speeds.append(aligned["speed_kmh"].column("speed_kmh").to_numpy())
        all_brakes.append(aligned[brake_ch].column(brake_ch).to_numpy())
        all_throttles.append(aligned[throttle_ch].column(throttle_ch).to_numpy())
        lap_nums.append(lap_num)
        lap_times.append(lap["lap_time"])

    return compute_segment_stats_from_arrays(
        all_distances,
        all_speeds,
        all_brakes,
        all_throttles,
        lap_nums,
        lap_times,
        segments,
        brake_threshold=brake_threshold,
        throttle_threshold=throttle_threshold,
    )


def compute_g_utilization(
    distances: list[np.ndarray],
    speeds: list[np.ndarray],
    lateral_gs: list[np.ndarray],
    inline_gs: list[np.ndarray] | None,
    lap_nums: list[int],
    segments: list[TrackSegment],
    corners: list["Corner"],
) -> pd.DataFrame:
    """Compute G utilization metrics per corner per lap.

    Total G = sqrt(lateral_g² + inline_g²) measures grip usage.
    When inline_gs is None, longitudinal G is derived from speed.

    Parameters
    ----------
    distances : list[np.ndarray]
        Per-lap distance arrays (meters).
    speeds : list[np.ndarray]
        Per-lap speed arrays (km/h).
    lateral_gs : list[np.ndarray]
        Per-lap lateral G arrays.
    inline_gs : list[np.ndarray] | None
        Per-lap inline G arrays, or None to derive from speed.
    lap_nums : list[int]
        Lap numbers.
    segments : list[TrackSegment]
        Track segments.
    corners : list[Corner]
        Corner definitions.

    Returns
    -------
    pd.DataFrame
        Per-corner, per-lap G utilization with columns:
        corner_id, corner_name, lap_num,
        total_g_mean, total_g_max, total_g_min, total_g_min_phase,
        g_utilization_pct, early_braking_coast_m, braking_g_mean,
        entry_g_mean, mid_g_mean, exit_g_mean
    """
    # Build lookup: corner_id -> (braking_segment, accel_segment)
    braking_segs: dict[int, TrackSegment] = {}
    accel_segs: dict[int, TrackSegment] = {}
    for seg in segments:
        if seg.segment_type == "braking":
            braking_segs[seg.corner_id] = seg
        elif seg.segment_type == "acceleration":
            accel_segs[seg.corner_id] = seg

    rows: list[dict] = []

    for lap_idx, (dist, spd, lat_g, lap_num) in enumerate(
        zip(distances, speeds, lateral_gs, lap_nums)
    ):
        if len(dist) < 10:
            continue

        # Compute or use inline G
        if inline_gs is not None:
            inl_g = inline_gs[lap_idx]
        else:
            # Derive longitudinal G from speed
            spd_ms = spd / 3.6
            # Estimate dt from distance differences and speed
            dd = np.diff(dist)
            # Average speed between consecutive points
            avg_spd = (spd_ms[:-1] + spd_ms[1:]) / 2
            # Avoid division by zero
            safe_avg = np.where(avg_spd > 0.1, avg_spd, 0.1)
            dt = dd / safe_avg
            dv = np.diff(spd_ms)
            # Avoid division by zero in dt
            safe_dt = np.where(dt > 1e-6, dt, 1e-6)
            accel = dv / safe_dt / 9.81
            # Pad to same length and smooth with 5-point rolling window
            accel = np.concatenate([[accel[0]], accel])
            kernel = np.ones(5) / 5
            inl_g = np.convolve(accel, kernel, mode="same")

        total_g = np.sqrt(lat_g**2 + inl_g**2)

        for corner in corners:
            brake_seg = braking_segs.get(corner.id)
            accel_seg = accel_segs.get(corner.id)

            # Full corner complex: from braking start to accel end
            complex_start = brake_seg.start_dist if brake_seg else corner.start_dist
            complex_end = accel_seg.end_dist if accel_seg else corner.end_dist

            # Get indices for the full complex
            ci = np.searchsorted(dist, complex_start)
            ce = np.searchsorted(dist, complex_end, side="right")
            if ci >= ce:
                continue

            complex_total_g = total_g[ci:ce]
            complex_dist = dist[ci:ce]
            total_g_mean = float(np.mean(complex_total_g))
            total_g_max = float(np.max(complex_total_g))

            # Find the G hole: valley between braking peak and cornering peak.
            # The braking peak is the max total G before the corner starts,
            # the cornering peak is the max total G within the corner.
            # The G hole is the minimum between these two peaks.
            corner_si = np.searchsorted(dist, corner.start_dist)
            corner_ei = np.searchsorted(dist, corner.end_dist, side="right")

            # Braking peak: max total G from complex start to corner start
            brake_peak_idx = ci  # fallback
            if corner_si > ci:
                brake_slice = total_g[ci:corner_si]
                brake_peak_idx = ci + int(np.argmax(brake_slice))

            # Cornering peak: max total G within the corner
            corner_peak_idx = corner_si  # fallback
            if corner_ei > corner_si:
                corner_slice = total_g[corner_si:corner_ei]
                corner_peak_idx = corner_si + int(np.argmax(corner_slice))

            # G hole: minimum between the two peaks (by index order)
            peak_lo = min(brake_peak_idx, corner_peak_idx)
            peak_hi = max(brake_peak_idx, corner_peak_idx)
            if peak_hi > peak_lo + 1:
                valley_slice = total_g[peak_lo:peak_hi]
                valley_min_idx = peak_lo + int(np.argmin(valley_slice))
            else:
                # Peaks are adjacent — use the lower one
                valley_min_idx = peak_lo if total_g[peak_lo] < total_g[peak_hi] else peak_hi

            total_g_min_val = float(total_g[valley_min_idx])
            total_g_min_dist = float(dist[valley_min_idx])

            # Define phase boundaries
            corner_len = corner.end_dist - corner.start_dist
            mid_half = 0.10 * corner_len
            apex = (
                corner.apex_dist
                if corner.apex_dist is not None
                else (corner.start_dist + corner_len / 2)
            )

            phase_defs = {
                "braking": (complex_start, corner.start_dist),
                "entry": (corner.start_dist, apex),
                "mid": (apex - mid_half, apex + mid_half),
                "exit": (apex + mid_half, corner.end_dist),
            }

            # Determine which phase the G hole falls in
            min_phase = "entry"
            for phase_name, (ps, pe) in phase_defs.items():
                if ps <= total_g_min_dist <= pe:
                    min_phase = phase_name
                    break

            phase_means: dict[str, float | None] = {}
            for phase_name, (ps, pe) in phase_defs.items():
                pi = np.searchsorted(dist, ps)
                pj = np.searchsorted(dist, pe, side="right")
                if pi < pj:
                    phase_means[phase_name] = float(np.mean(total_g[pi:pj]))
                else:
                    phase_means[phase_name] = None

            # G utilization: valley depth relative to the lower of the two peaks
            peak_g = min(float(total_g[brake_peak_idx]), float(total_g[corner_peak_idx]))
            g_util_pct = total_g_min_val / max(peak_g, 1e-6) * 100 if peak_g > 0 else 0.0

            # Early braking detection: when G hole is in the braking phase,
            # measure the "coast" distance from where braking G fades to
            # the corner start.  This tells the driver how early they braked.
            early_braking_coast_m: float | None = None
            if min_phase == "braking" and corner_si > brake_peak_idx:
                # Find where total G first drops below 50% of braking peak
                # after the peak but before the corner start.
                brake_peak_g = float(total_g[brake_peak_idx])
                threshold = brake_peak_g * 0.5
                post_peak = total_g[brake_peak_idx:corner_si]
                below = np.where(post_peak < threshold)[0]
                if len(below) > 0:
                    fade_idx = brake_peak_idx + int(below[0])
                    fade_dist = float(dist[fade_idx])
                    coast = corner.start_dist - fade_dist
                    if coast > 0:
                        early_braking_coast_m = round(coast, 1)

            rows.append(
                {
                    "corner_id": corner.id,
                    "corner_name": corner.name,
                    "lap_num": lap_num,
                    "total_g_mean": total_g_mean,
                    "total_g_max": total_g_max,
                    "total_g_min": total_g_min_val,
                    "total_g_min_dist": total_g_min_dist,
                    "total_g_min_phase": min_phase,
                    "g_utilization_pct": g_util_pct,
                    "early_braking_coast_m": early_braking_coast_m,
                    "braking_g_mean": phase_means.get("braking"),
                    "entry_g_mean": phase_means.get("entry"),
                    "mid_g_mean": phase_means.get("mid"),
                    "exit_g_mean": phase_means.get("exit"),
                }
            )

    return pd.DataFrame(rows)
