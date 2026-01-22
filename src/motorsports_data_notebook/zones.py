"""Braking and acceleration zone detection and analysis.

This module provides functions for detecting braking and acceleration zones
from telemetry data and building track segment definitions.
"""

from dataclasses import dataclass

import numpy as np

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
