"""Corner detection and curvature analysis for race tracks.

This module provides functions for detecting corners from GPS data using
curvature analysis.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Corner:
    """Represents a detected corner on a race track.

    Attributes
    ----------
    id : int
        Corner number (1-indexed).
    name : str
        Corner name (e.g., "Turn 1").
    direction : Literal['L', 'R']
        Turn direction: 'L' for left, 'R' for right.
    start_idx : int
        Array index where corner begins.
    end_idx : int
        Array index where corner ends.
    start_dist : float
        Distance along track where corner begins (meters).
    end_dist : float
        Distance along track where corner ends (meters).
    apex_idx : int
        Array index of the apex (maximum curvature point).
    apex_dist : float
        Distance along track of the apex (meters).
    max_curvature : float
        Maximum curvature value at apex (1/m).
    """

    id: int
    name: str
    direction: Literal["L", "R"]
    start_idx: int
    end_idx: int
    start_dist: float
    end_dist: float
    apex_idx: int
    apex_dist: float
    max_curvature: float

    @property
    def length(self) -> float:
        """Corner length in meters."""
        return self.end_dist - self.start_dist

    @property
    def radius(self) -> float:
        """Approximate corner radius in meters (at apex)."""
        if self.max_curvature > 1e-6:
            return 1.0 / self.max_curvature
        return 10000.0


def compute_lap_distance(timecodes, speed):
    """
    Compute cumulative distance traveled along the lap.

    Integrates speed over time to compute distance at each point.

    Parameters
    ----------
    timecodes : pandas.Series or array-like
        Timestamps in milliseconds.
    speed : pandas.Series or array-like
        Speed values in m/s.

    Returns
    -------
    numpy.ndarray
        Cumulative distance in meters at each point.

    Examples
    --------
    >>> distance_m = compute_lap_distance(lap_data['timecodes'], lap_data['GPS Speed'])
    """
    timecodes = pd.Series(timecodes)
    speed = pd.Series(speed)

    # Convert timecodes to seconds from start
    time_s = (timecodes - timecodes.iloc[0]) / 1000.0

    # Compute time deltas
    dt = time_s.diff().fillna(0)

    # Integrate speed over time to get distance
    distance_m = (speed * dt).cumsum()

    return distance_m.values


def gps_to_local_xy(lat, lon):
    """
    Convert GPS lat/lon to local XY coordinates in meters.

    Uses a simple equirectangular projection centered on the track.

    Parameters
    ----------
    lat : pandas.Series or array-like
        Latitude values in degrees.
    lon : pandas.Series or array-like
        Longitude values in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        (x, y) coordinates in meters relative to track center.

    Examples
    --------
    >>> x, y = gps_to_local_xy(lap_data['GPS Latitude'], lap_data['GPS Longitude'])
    """
    lat = np.array(lat)
    lon = np.array(lon)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Reference point (center of track)
    lat0 = np.radians(lat.mean())
    lon0 = np.radians(lon.mean())

    # Earth radius in meters
    R = 6371000

    # Convert to local XY (meters)
    x = R * (lon_rad - lon0) * np.cos(lat0)
    y = R * (lat_rad - lat0)

    return x, y


def compute_curvature(x, y, pos_smooth_window=15, curv_smooth_window=30):
    """
    Compute curvature from XY coordinates.

    Uses the formula: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)

    Parameters
    ----------
    x : array-like
        X position coordinates in meters.
    y : array-like
        Y position coordinates in meters.
    pos_smooth_window : int, default=15
        Rolling average window for position smoothing.
    curv_smooth_window : int, default=30
        Rolling average window for curvature output smoothing.

    Returns
    -------
    tuple of numpy.ndarray
        (curvature, signed_curvature, radius) where:
        - curvature: Smoothed absolute curvature values (1/radius in 1/m)
        - signed_curvature: Signed curvature (positive=left, negative=right)
        - radius: Radius of curvature in meters (capped at 10000m for straights)

    Examples
    --------
    >>> x, y = gps_to_local_xy(lat, lon)
    >>> curvature, signed_curvature, radius = compute_curvature(x, y)
    """
    x = np.array(x)
    y = np.array(y)

    # Apply rolling average smoothing to positions
    if pos_smooth_window > 1:
        x_smooth = pd.Series(x).rolling(pos_smooth_window, center=True, min_periods=1).mean().values
        y_smooth = pd.Series(y).rolling(pos_smooth_window, center=True, min_periods=1).mean().values
    else:
        x_smooth, y_smooth = x, y

    # First derivatives (velocity)
    dx = np.gradient(x_smooth)  # type: ignore[arg-type]
    dy = np.gradient(y_smooth)  # type: ignore[arg-type]

    # Second derivatives (acceleration)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Signed curvature formula: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    # Sign indicates direction: positive = turning left, negative = turning right
    numerator_signed = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5

    # Avoid division by zero
    signed_curvature = np.where(denominator > 1e-10, numerator_signed / denominator, 0)

    # Apply additional smoothing to curvature output to reduce GPS noise
    if curv_smooth_window > 1:
        signed_curvature = (
            pd.Series(signed_curvature)
            .rolling(curv_smooth_window, center=True, min_periods=1)
            .mean()
            .values  # type: ignore[assignment]
        )

    # Absolute curvature
    curvature = np.abs(signed_curvature)

    # Radius = 1/curvature (cap at 10000m for near-straight sections)
    radius = np.where(curvature > 1e-4, 1 / curvature, 10000)
    radius = np.clip(radius, 0, 10000)

    return curvature, signed_curvature, radius


def identify_corners(
    lat,
    lon,
    threshold: float = 0.006,
    min_corner_length: float = 15,
    min_gap: float = 80,
    pos_smooth_window: int = 15,
    curv_smooth_window: int = 30,
) -> list[Corner]:
    """
    Identify corners from GPS latitude/longitude data.

    This is a convenience function that performs all the steps needed to go from
    raw GPS coordinates to detected corners:
    1. Convert GPS to local XY coordinates
    2. Compute curvature from XY positions
    3. Identify corners from curvature

    Parameters
    ----------
    lat : pandas.Series or array-like
        Latitude values in degrees.
    lon : pandas.Series or array-like
        Longitude values in degrees.
    threshold : float, default=0.006
        Minimum curvature to consider as corner (1/m). 0.006 ≈ 167m radius.
    min_corner_length : float, default=15
        Minimum corner length in meters.
    min_gap : float, default=80
        Minimum gap between same-direction corners to keep them separate.
        Corners closer than this with same direction will be merged.
    pos_smooth_window : int, default=15
        Rolling average window for position smoothing in curvature computation.
    curv_smooth_window : int, default=30
        Rolling average window for curvature output smoothing.

    Returns
    -------
    list[Corner]
        List of Corner dataclass instances.

    Examples
    --------
    >>> corners = identify_corners(
    ...     lat=lap_data['GPS Latitude'],
    ...     lon=lap_data['GPS Longitude']
    ... )
    >>> for c in corners:
    ...     print(f"{c.name} ({c.direction}): {c.start_dist:.0f}m - {c.end_dist:.0f}m")
    """
    # Step 1: Convert GPS to local XY coordinates
    lat = np.array(lat, dtype=np.float64)
    lon = np.array(lon, dtype=np.float64)

    # Filter out invalid GPS samples (e.g. iRacing reports (0,0) before car is on track)
    valid = (lat != 0.0) | (lon != 0.0)
    if not np.all(valid):
        lat = lat[valid]
        lon = lon[valid]

    x, y = gps_to_local_xy(lat, lon)

    # Step 2: Compute distance along track
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    distance = np.cumsum(np.sqrt(dx**2 + dy**2))

    # Step 3: Compute curvature
    curvature, signed_curvature, _ = compute_curvature(
        x, y, pos_smooth_window=pos_smooth_window, curv_smooth_window=curv_smooth_window
    )

    # Step 4: Identify corners from curvature
    return identify_corners_from_curvature(
        distance=distance,
        curvature=curvature,
        signed_curvature=signed_curvature,
        threshold=threshold,
        min_corner_length=min_corner_length,
        min_gap=min_gap,
    )


def identify_corners_from_curvature(
    distance, curvature, signed_curvature, threshold=0.006, min_corner_length=15, min_gap=80
) -> list[Corner]:
    """
    Identify corners where curvature exceeds threshold.

    Uses signed curvature to separate corners that change direction (left vs right).
    Same-direction corners within min_gap distance are merged.

    Parameters
    ----------
    distance : array-like
        Distance along track in meters.
    curvature : array-like
        Absolute curvature values (1/m).
    signed_curvature : array-like
        Signed curvature (positive=left, negative=right).
    threshold : float, default=0.006
        Minimum curvature to consider as corner (1/m). 0.006 ≈ 167m radius.
    min_corner_length : float, default=15
        Minimum corner length in meters.
    min_gap : float, default=80
        Minimum gap between same-direction corners to keep them separate.
        Corners closer than this with same direction will be merged.

    Returns
    -------
    list[Corner]
        List of Corner dataclass instances.

    Examples
    --------
    >>> corners = identify_corners_from_curvature(
    ...     distance=lap_data['distance_m'].values,
    ...     curvature=lap_data['curvature'].values,
    ...     signed_curvature=lap_data['signed_curvature'].values
    ... )
    >>> for c in corners:
    ...     print(f"{c.name} ({c.direction}): {c.start_dist:.0f}m - {c.end_dist:.0f}m")
    """
    distance = np.array(distance)
    curvature = np.array(curvature)
    signed_curvature = np.array(signed_curvature)

    # Find points above threshold
    in_corner = curvature > threshold

    # Determine direction at each point (1=left, -1=right, 0=straight)
    direction = np.sign(signed_curvature)

    # Find corner boundaries (transitions) - split on direction change too
    corners = []
    corner_start = None
    corner_direction = None

    for i in range(len(in_corner)):
        if in_corner[i]:
            current_dir = direction[i]

            if corner_start is None:
                # Start new corner
                corner_start = i
                corner_direction = current_dir
            elif current_dir != corner_direction and current_dir != 0:
                # Direction changed - end current corner and start new one
                corner_end = i - 1
                corner_length = distance[corner_end] - distance[corner_start]

                if corner_length >= min_corner_length:
                    apex_idx = corner_start + np.argmax(curvature[corner_start : corner_end + 1])
                    corners.append(
                        Corner(
                            id=0,  # Will be set after merging
                            name="",  # Will be set after merging
                            direction="L" if corner_direction > 0 else "R",  # type: ignore[operator]
                            start_idx=corner_start,
                            end_idx=corner_end,
                            start_dist=float(distance[corner_start]),
                            end_dist=float(distance[corner_end]),
                            apex_idx=int(apex_idx),
                            apex_dist=float(distance[apex_idx]),
                            max_curvature=float(curvature[apex_idx]),
                        )
                    )

                # Start new corner with new direction
                corner_start = i
                corner_direction = current_dir

        elif corner_start is not None:
            # Exited corner region
            corner_end = i - 1
            corner_length = distance[corner_end] - distance[corner_start]

            if corner_length >= min_corner_length:
                apex_idx = corner_start + np.argmax(curvature[corner_start : corner_end + 1])
                corners.append(
                    Corner(
                        id=0,  # Will be set after merging
                        name="",  # Will be set after merging
                        direction="L" if corner_direction > 0 else "R",  # type: ignore[operator]
                        start_idx=corner_start,
                        end_idx=corner_end,
                        start_dist=float(distance[corner_start]),
                        end_dist=float(distance[corner_end]),
                        apex_idx=int(apex_idx),
                        apex_dist=float(distance[apex_idx]),
                        max_curvature=float(curvature[apex_idx]),
                    )
                )
            corner_start = None
            corner_direction = None

    # Handle corner at end of lap
    if corner_start is not None:
        corner_end = len(distance) - 1
        corner_length = distance[corner_end] - distance[corner_start]
        if corner_length >= min_corner_length:
            apex_idx = corner_start + np.argmax(curvature[corner_start : corner_end + 1])
            corners.append(
                Corner(
                    id=0,  # Will be set after merging
                    name="",  # Will be set after merging
                    direction="L" if corner_direction > 0 else "R",  # type: ignore[operator]
                    start_idx=corner_start,
                    end_idx=corner_end,
                    start_dist=float(distance[corner_start]),
                    end_dist=float(distance[corner_end]),
                    apex_idx=int(apex_idx),
                    apex_dist=float(distance[apex_idx]),
                    max_curvature=float(curvature[apex_idx]),
                )
            )

    # Merge corners that are too close together AND same direction
    merged_corners: list[Corner] = []
    for corner in corners:
        if (
            merged_corners
            and (corner.start_dist - merged_corners[-1].end_dist) < min_gap
            and corner.direction == merged_corners[-1].direction
        ):
            # Merge with previous corner (same direction and close)
            prev = merged_corners[-1]
            prev.end_dist = corner.end_dist
            prev.end_idx = corner.end_idx
            if corner.max_curvature > prev.max_curvature:
                prev.apex_dist = corner.apex_dist
                prev.apex_idx = corner.apex_idx
                prev.max_curvature = corner.max_curvature
        else:
            # Create a copy of the corner
            merged_corners.append(
                Corner(
                    id=corner.id,
                    name=corner.name,
                    direction=corner.direction,
                    start_idx=corner.start_idx,
                    end_idx=corner.end_idx,
                    start_dist=corner.start_dist,
                    end_dist=corner.end_dist,
                    apex_idx=corner.apex_idx,
                    apex_dist=corner.apex_dist,
                    max_curvature=corner.max_curvature,
                )
            )

    # Add corner numbers/names
    for i, corner in enumerate(merged_corners):
        corner.id = i + 1
        corner.name = f"Turn {i + 1}"

    return merged_corners
