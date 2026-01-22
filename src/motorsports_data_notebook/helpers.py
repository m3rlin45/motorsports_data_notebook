"""Helper functions for motorsports data analysis.

This module provides utility functions for working with lap data and creating
GPS track visualizations.
"""

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display

if TYPE_CHECKING:
    import ipywidgets as widgets


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


def find_throttle_acceptance(
    lap_data: pd.DataFrame,
    corner: Corner,
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
        Lap data with columns: distance_m, timecodes, PPS, LateralAcc.
    corner : Corner
        Corner object defining the corner boundaries.
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
    """
    # Apply smoothing to lateral G (rolling average on absolute value)
    lap_data = lap_data.copy()
    lap_data["LateralAcc_smooth"] = (
        lap_data["LateralAcc"]
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
        if cast(float, exit_data.loc[i, "PPS"]) >= throttle_threshold:
            start_time = cast(float, exit_data.loc[i, "timecodes"])
            end_time = start_time + sustain_time_ms

            # Check if throttle stays above threshold for sustain_time_ms
            timecodes = cast("pd.Series[float]", exit_data["timecodes"])
            sustain_mask = (timecodes >= start_time) & (timecodes <= end_time)
            sustain_data = exit_data[sustain_mask]

            if len(sustain_data) == 0:
                continue

            # Check if all points in the sustain window are above threshold
            if (sustain_data["PPS"] >= throttle_threshold).all():
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


def show_fig(fig):
    """
    Display a Plotly figure with automatic environment detection.

    Handles both standard JupyterLab and JupyterLite (Pyodide) environments.
    In JupyterLite, figures are rendered via HTML since the standard Plotly
    renderer is not available.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure to display.

    Returns
    -------
    None
        Displays the figure inline.

    Examples
    --------
    >>> fig = px.scatter(df, x='x', y='y')
    >>> show_fig(fig)
    """
    if "pyodide" in sys.modules:
        # In Pyodide/JupyterLite, use HTML display
        from IPython.display import HTML, display

        html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        display(HTML(html))
    else:
        # Standard environment
        fig.show()


def get_best_lap(laps_df):
    """
    Find the best (fastest) lap by duration.

    Excludes the first and last laps to avoid pit entry/exit laps.

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

    if len(laps_df) <= 2:
        raise ValueError("Need at least 3 laps to exclude first and last laps")

    # Exclude first and last laps
    laps_subset = laps_df.iloc[1:-1].copy()

    laps_subset["lap_duration_ms"] = laps_subset["end_time"] - laps_subset["start_time"]
    best_idx = laps_subset["lap_duration_ms"].idxmin()
    return laps_subset.loc[best_idx]


def compute_start_line(lat, lon, ahead_points=100, scale=0.02):
    """
    Compute endpoints for a perpendicular start/finish line at the beginning of the track.

    Parameters
    ----------
    lat : pandas.Series
        Latitude values along the track.
    lon : pandas.Series
        Longitude values along the track.
    ahead_points : int, default=100
        Number of points ahead to use for computing the heading direction.
    scale : float, default=0.02
        Line length as a fraction of track size (lon/lat range).

    Returns
    -------
    tuple of tuples
        ((lat_a, lon_a), (lat_b, lon_b)) - endpoints of the perpendicular line.
    """
    lat1 = lat.iloc[0]
    lon1 = lon.iloc[0]
    idx2 = min(ahead_points, len(lat) - 1)
    lat2 = lat.iloc[idx2]
    lon2 = lon.iloc[idx2]
    heading_vec = (lon2 - lon1, lat2 - lat1)
    perp_vec = (-heading_vec[1], heading_vec[0])
    norm = math.hypot(perp_vec[0], perp_vec[1]) or 1.0
    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    half_len_deg = scale * max(lon_range, lat_range)
    dx = perp_vec[0] / norm * half_len_deg
    dy = perp_vec[1] / norm * half_len_deg
    return (lat1 - dy, lon1 - dx), (lat1 + dy, lon1 + dx)


def plot_lap_gps(lat, lon, color_channels, width=800, height=800, title=None):
    """
    Plot interactive GPS track with multi-layer color-coded data and perpendicular start line using Plotly.

    Parameters
    ----------
    lat : pandas.Series or array-like
        Latitude values along the track.
    lon : pandas.Series or array-like
        Longitude values along the track.
    color_channels : list of tuples
        List of (color_values, label, colorscale) tuples. Each tuple defines a layer:
        - color_values: pandas.Series or array-like with values for color mapping
        - label: str, label for the colorbar
        - colorscale: str, Plotly colorscale name (e.g., 'Viridis', 'Plasma', 'Jet')
        Layers are drawn bottom-to-top (last in list = bottom with largest markers,
        first in list = top with smallest markers).
    width : int, default=800
        Figure width in pixels.
    height : int, default=800
        Figure height in pixels.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objs.Figure
        Interactive Plotly figure object.

    Examples
    --------
    Single color channel:
    >>> fig = plot_lap_gps(
    ...     lat=lap_data['GPS Latitude'],
    ...     lon=lap_data['GPS Longitude'],
    ...     color_channels=[
    ...         (lap_data['speed_kmh'], 'Speed (km/h)', 'Viridis')
    ...     ]
    ... )

    Multiple color channels:
    >>> fig = plot_lap_gps(
    ...     lat=lap_data['GPS Latitude'],
    ...     lon=lap_data['GPS Longitude'],
    ...     color_channels=[
    ...         (lap_data['BrakePress'], 'Brake Pressure', 'Reds'),
    ...         (lap_data['speed_kmh'], 'Speed (km/h)', 'Viridis')
    ...     ]
    ... )
    """
    # Compute start line
    (lat_a, lon_a), (lat_b, lon_b) = compute_start_line(lat, lon)

    # Create interactive Plotly figure
    fig = go.Figure()

    # Add color channels in reverse order (bottom to top)
    # Bottom layers have larger markers, top layers have smaller markers
    num_channels = len(color_channels)
    for idx, (color_values, label, colorscale) in enumerate(reversed(color_channels)):
        # Calculate marker size: bottom layers are larger
        layer_position = num_channels - idx - 1  # 0 is top, num_channels-1 is bottom
        marker_size = 4 + (layer_position * 2)  # Bottom layers get +2, +4, etc.

        fig.add_trace(
            go.Scattergl(
                x=lon,
                y=lat,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color_values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        title=label,
                        x=1.02 + (layer_position * 0.15),  # Offset colorbars horizontally
                        len=0.8,
                    ),
                ),
                name=label,
                showlegend=False,
                hovertemplate=f"{label}: %{{marker.color:.2f}}<br>Lon: %{{x:.6f}}<br>Lat: %{{y:.6f}}<extra></extra>",
            )
        )

    # Add track outline
    fig.add_trace(
        go.Scattergl(
            x=lon,
            y=lat,
            mode="lines",
            line=dict(color="black", width=1),
            opacity=0.3,
            name="Track Outline",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add start/finish line
    fig.add_trace(
        go.Scattergl(
            x=[lon_a, lon_b],
            y=[lat_a, lat_b],
            mode="lines",
            line=dict(color="black", width=3),
            name="Start/Finish",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title=title or "GPS Path",
        width=width,
        height=height,
        xaxis=dict(
            scaleanchor="y", scaleratio=1, showticklabels=False, showgrid=False, zeroline=False
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        hovermode="closest",
    )

    return fig


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


class FileUpload:
    """Interactive file upload widget for Jupyter notebooks.

    Provides a file upload interface with status feedback.
    Falls back to a default file if no file is uploaded.

    Parameters
    ----------
    default_file : str
        Path to the default file to use if no file is uploaded.

    Examples
    --------
    >>> file_upload = FileUpload("sample_data.xrz")
    >>> file_upload.display()  # Shows upload widget with status
    >>> log = aim_xrk(file_upload.get_file_data())  # Load the file
    """

    def __init__(self, default_file: str) -> None:
        import ipywidgets as widgets

        self._default_file = default_file
        self._uploaded_data: bytes | None = None
        self._uploaded_filename: str | None = None
        self._widgets = widgets

        # Instruction label
        self._instruction = widgets.HTML(
            value="<b>📁 Upload your own .xrk/.xrz file:</b> (or skip to use the sample data)"
        )

        # Create the file upload widget
        self._upload_widget = widgets.FileUpload(
            accept=".xrk,.xrz",
            multiple=False,
            description="Choose File",
            button_style="primary",
        )

        self._status_label = widgets.HTML(
            value=f"<span style='color: #666;'>Using: {default_file} (default)</span>"
        )

        # Set up callback for upload changes
        self._upload_widget.observe(self._on_upload, names="value")

        # Container for layout
        self._container = widgets.VBox([self._instruction, self._upload_widget, self._status_label])

    def _on_upload(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle file upload event."""
        if self._upload_widget.value:
            uploaded = self._upload_widget.value[0]
            self._uploaded_filename = uploaded["name"]
            self._uploaded_data = uploaded["content"].tobytes()
            self._status_label.value = (
                f"<span style='color: green;'><b>✓ Using:</b> {self._uploaded_filename}</span>"
            )

    def display(self) -> None:
        """Display the upload widget and status label."""
        display(self._container)

    def get_file_data(self) -> Union[str, bytes]:
        """Get the file data to pass to aim_xrk.

        Returns
        -------
        str or bytes
            If a file was uploaded, returns the file content as bytes.
            Otherwise, returns the default filename as a string.
        """
        if self._uploaded_data is not None:
            return self._uploaded_data
        return self._default_file
