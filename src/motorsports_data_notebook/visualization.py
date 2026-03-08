"""Visualization functions for motorsports data.

This module provides functions for creating interactive visualizations
of lap data and GPS track data.
"""

import math
import sys
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow as pa
from plotly.subplots import make_subplots

from ._util import get_channel_unit as _get_channel_unit
from ._util import infer_channel_scale as _infer_channel_scale
from .channels import get_best_lap

if TYPE_CHECKING:
    from ._types import LogFile
    from .corners import Corner
    from .suspension import VelocityHistogramResult
    from .tire_grip import TireGripResult
    from .zones import TrackSegment


__all__ = [
    "compute_start_line",
    "format_lap_time",
    "plot_corner_inputs",
    "plot_gps_channels",
    "plot_lap_gps",
    "plot_suspension_velocity_histogram",
    "plot_tire_grip_scatter",
    "plot_tire_thermography",
    "plot_track_segments",
    "save_corner_comparison",
    "save_corner_map",
    "show_fig",
    "visualize_throttle_acceptance",
]


def format_lap_time(lap_time: pd.Timedelta) -> str:
    """Format a lap time as 'M:SS.mmm'.

    Parameters
    ----------
    lap_time : pd.Timedelta
        The lap time to format.

    Returns
    -------
    str
        Formatted lap time string.

    Examples
    --------
    >>> format_lap_time(pd.Timedelta(minutes=1, seconds=23, milliseconds=456))
    '1:23.456'
    """
    total_seconds = lap_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:06.3f}"


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


# Type alias for array-like inputs (using Any for flexibility with .values return types)
ArrayLike = Any


def plot_corner_inputs(
    distance: ArrayLike,
    corner: "Corner",
    *,
    throttle: ArrayLike | None = None,
    brake: ArrayLike | None = None,
    steering: ArrayLike | None = None,
    title: str | None = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """Plot driver inputs (throttle, brake, steering) over distance for a corner.

    Creates a subplot with shared x-axis (distance) showing the provided input
    channels. Each channel has its own y-axis row.

    Parameters
    ----------
    distance : array-like
        Distance values in meters (x-axis).
    corner : Corner
        The corner object with start_dist, end_dist, apex_dist, and name.
    throttle : array-like, optional
        Throttle values (e.g., PPS channel). Omit to skip throttle row.
    brake : array-like, optional
        Brake pressure values. Omit to skip brake row.
    steering : array-like, optional
        Steering angle values. Omit to skip steering row.
    title : str, optional
        Plot title. Defaults to "Driver Inputs - {corner.name}".
    width : int, default=1000
        Figure width in pixels.
    height : int, default=600
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with the corner inputs visualization.

    Raises
    ------
    ValueError
        If no input channels are provided.

    Examples
    --------
    >>> from motorsports_data_notebook.zones import get_corner_data
    >>> corner_data = get_corner_data(channels, laps, segments, corner_id=1, lap_num=3, margin=50)
    >>> fig = plot_corner_inputs(
    ...     corner_data["distance_m"],
    ...     corners[0],
    ...     throttle=corner_data["PPS"],
    ...     brake=corner_data["BrakePress"],
    ...     steering=corner_data["Steering"],
    ... )
    >>> show_fig(fig)
    """
    # Build list of channels to plot: (data, display_name, color)
    channel_specs: list[tuple[ArrayLike, str, str]] = []
    if throttle is not None:
        channel_specs.append((throttle, "Throttle (%)", "green"))
    if brake is not None:
        channel_specs.append((brake, "Brake Pressure", "red"))
    if steering is not None:
        channel_specs.append((steering, "Steering (°)", "blue"))

    if not channel_specs:
        raise ValueError(
            "At least one input channel (throttle, brake, or steering) must be provided"
        )

    n_rows = len(channel_specs)

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[display for _, display, _ in channel_specs],
    )

    # Add traces for each channel
    for i, (data, display_name, color) in enumerate(channel_specs, start=1):
        fig.add_trace(
            go.Scatter(
                x=distance,
                y=data,
                mode="lines",
                name=display_name,
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        # Add corner boundary shading to each subplot
        fig.add_vrect(
            x0=corner.start_dist,
            x1=corner.end_dist,
            fillcolor="gray",
            opacity=0.1,
            line_width=0,
            row=i,
            col=1,
        )

        # Add apex line to each subplot
        fig.add_vline(
            x=corner.apex_dist,
            line_dash="dot",
            line_color="orange",
            opacity=0.7,
            row=i,
            col=1,
        )

    # Update layout
    plot_title = title if title is not None else f"Driver Inputs - {corner.name}"
    fig.update_layout(
        title=plot_title,
        width=width,
        height=height,
    )

    # Label x-axis only on bottom subplot
    fig.update_xaxes(title_text="Distance (m)", row=n_rows, col=1)

    # Label y-axes
    for i, (_, display_name, _) in enumerate(channel_specs, start=1):
        fig.update_yaxes(title_text=display_name, row=i, col=1)

    return fig


def visualize_throttle_acceptance(
    distance: ArrayLike,
    throttle: ArrayLike,
    lateral_g: ArrayLike,
    corner: "Corner",
    throttle_acceptance_result: dict[str, float],
    *,
    brake: ArrayLike | None = None,
    steering: ArrayLike | None = None,
    title: str | None = None,
    width: int = 1000,
    height: int = 500,
) -> go.Figure:
    """Visualize throttle acceptance for a corner.

    Creates a visualization showing throttle, lateral G, and optionally brake
    and steering over distance, with reference lines for peak G, throttle
    acceptance point, and corner boundaries.

    Parameters
    ----------
    distance : array-like
        Distance values in meters (x-axis).
    throttle : array-like
        Throttle values (0-100%).
    lateral_g : array-like
        Lateral G values (already smoothed if desired).
    corner : Corner
        The corner object with start_dist, end_dist, apex_dist, and name.
    throttle_acceptance_result : dict
        Result from find_throttle_acceptance() containing:
        - peak_lateral_g: Peak lateral G in the corner
        - lateral_g_at_throttle: Lateral G when full throttle is reached
        - throttle_acceptance_pct: Percentage of peak G at full throttle
        - full_throttle_dist: Distance where full throttle is reached
    brake : array-like, optional
        Brake pressure values. Omit to skip brake trace.
    steering : array-like, optional
        Steering angle values. Omit to skip steering trace.
    title : str, optional
        Plot title. Defaults to "Throttle Acceptance - {corner.name}".
    width : int, default=1000
        Figure width in pixels.
    height : int, default=500
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with the throttle acceptance visualization.

    Examples
    --------
    >>> from motorsports_data_notebook.zones import get_corner_data
    >>> corner_data = get_corner_data(channels, laps, corners[0], best_lap["num"])
    >>> lateral_g_smooth = corner_data["LateralAcc"].abs().rolling(25, center=True, min_periods=1).mean()
    >>> fig = visualize_throttle_acceptance(
    ...     distance=corner_data["distance_m"],
    ...     throttle=corner_data["PPS"],
    ...     lateral_g=lateral_g_smooth,
    ...     corner=corners[0],
    ...     throttle_acceptance_result=result,
    ...     brake=corner_data.get("BrakePress"),
    ...     steering=corner_data.get("SteerAngle"),
    ... )
    """
    fig = go.Figure()

    throttle_arr = np.asarray(throttle)
    lateral_g_arr = np.asarray(lateral_g)

    # Lateral G trace (yaxis1 — left side)
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=lateral_g_arr,
            mode="lines",
            name="Lateral G",
            yaxis="y",
            line=dict(color="blue", width=2),
            hovertemplate="Distance: %{x:.0f}m<br>Lateral G: %{y:.2f}G<extra></extra>",
        )
    )

    # Throttle trace (yaxis2 — right side)
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=throttle_arr,
            mode="lines",
            name="Throttle",
            yaxis="y2",
            line=dict(color="green", width=2),
            hovertemplate="Distance: %{x:.0f}m<br>Throttle: %{y:.1f}<extra></extra>",
        )
    )

    # Brake trace (yaxis3 — right side, offset)
    axis_count = 2
    if brake is not None:
        brake_arr = np.asarray(brake)
        if brake_arr.max() > 0:
            axis_count += 1
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=brake_arr,
                    mode="lines",
                    name="Brake",
                    yaxis=f"y{axis_count}",
                    line=dict(color="red", width=2),
                    hovertemplate="Distance: %{x:.0f}m<br>Brake: %{y:.1f}<extra></extra>",
                )
            )

    # Steering trace (next available axis — right side, offset)
    if steering is not None:
        steering_arr = np.asarray(steering)
        if np.abs(steering_arr).max() > 0:
            axis_count += 1
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=steering_arr,
                    mode="lines",
                    name="Steering",
                    yaxis=f"y{axis_count}",
                    line=dict(color="purple", width=2),
                    hovertemplate="Distance: %{x:.0f}m<br>Steering: %{y:.1f}°<extra></extra>",
                )
            )

    # Peak lateral G reference line (as a shape on yaxis1)
    peak_g = throttle_acceptance_result["peak_lateral_g"]
    g_at_throttle = throttle_acceptance_result["lateral_g_at_throttle"]
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=peak_g,
        y1=peak_g,
        line=dict(dash="dash", color="blue", width=1),
        opacity=0.7,
    )
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=1,
        y=peak_g,
        text=f"Peak G: {peak_g:.2f}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(color="blue", size=10),
    )

    # Lateral G at throttle acceptance reference line
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=g_at_throttle,
        y1=g_at_throttle,
        line=dict(dash="dash", color="red", width=1),
        opacity=0.7,
    )
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=1,
        y=g_at_throttle,
        text=f"G at Full Throttle: {g_at_throttle:.2f} ({throttle_acceptance_result['throttle_acceptance_pct']:.0f}%)",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color="red", size=10),
    )

    # Full throttle point vertical line
    fig.add_vline(
        x=throttle_acceptance_result["full_throttle_dist"],
        line_dash="dash",
        line_color="green",
        opacity=0.7,
        annotation_text="Full Throttle Point",
        annotation_position="top",
    )

    # Corner boundaries
    fig.add_vrect(
        x0=corner.start_dist,
        x1=corner.end_dist,
        fillcolor="gray",
        opacity=0.1,
        line_width=0,
        annotation_text=corner.name,
        annotation_position="top left",
    )

    # Apex line
    fig.add_vline(
        x=corner.apex_dist,
        line_dash="dot",
        line_color="orange",
        opacity=0.5,
        annotation_text="Apex",
    )

    # Build layout with overlaid y-axes
    # Shrink x-domain to make room for right-side axes, positions must be in [0, 1]
    right_positions: list[float | None]
    if axis_count == 2:
        x_domain = [0, 1.0]
        right_positions = [None]  # yaxis2 anchors to "x" directly
    elif axis_count == 3:
        x_domain = [0, 0.88]
        right_positions = [0.88, 0.94]
    else:
        x_domain = [0, 0.82]
        right_positions = [0.82, 0.88, 0.94]

    layout_args: dict[str, Any] = dict(
        title=title if title is not None else f"Throttle Acceptance - {corner.name}",
        xaxis=dict(title="Distance (m)", domain=x_domain),
        yaxis=dict(
            title=dict(text="Lateral G", font=dict(color="blue")),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=dict(text="Throttle", font=dict(color="green")),
            tickfont=dict(color="green"),
            anchor="free" if axis_count > 2 else "x",
            overlaying="y",
            side="right",
            position=right_positions[0],
        ),
        width=width,
        height=height,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    if axis_count >= 3:
        layout_args["yaxis3"] = dict(
            title=dict(text="Brake", font=dict(color="red")),
            tickfont=dict(color="red"),
            anchor="free",
            overlaying="y",
            side="right",
            position=right_positions[1],
        )
    if axis_count >= 4:
        layout_args["yaxis4"] = dict(
            title=dict(text="Steering", font=dict(color="purple")),
            tickfont=dict(color="purple"),
            anchor="free",
            overlaying="y",
            side="right",
            position=right_positions[2],
        )

    fig.update_layout(**layout_args)

    return fig


def plot_gps_channels(
    channels: dict[str, pa.Table],
    lat_channel: str,
    lon_channel: str,
    color_channels: list[tuple[str, str, str]],
    width: int = 800,
    height: int = 800,
    title: str | None = None,
) -> go.Figure:
    """Plot GPS track with color-coded data from channel tables.

    This is a convenience wrapper around plot_lap_gps() that accepts channel
    tables directly. All channels must be pre-resampled to a common timebase
    by the caller using log.resample_to_channel().

    Parameters
    ----------
    channels : dict[str, pa.Table]
        Dictionary of channel tables, pre-resampled to a common timebase.
    lat_channel : str
        Name of latitude channel (e.g., "GPS Latitude").
    lon_channel : str
        Name of longitude channel (e.g., "GPS Longitude").
    color_channels : list[tuple[str, str, str]]
        List of (channel_name, label, colorscale) tuples. Each tuple defines a layer:
        - channel_name: str, name of channel in channels dict
        - label: str, label for the colorbar
        - colorscale: str, Plotly colorscale name (e.g., 'Viridis', 'Plasma', 'Jet')
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
    >>> lap_log = log.filter_by_lap(best_lap_num)
    >>> channels = lap_log.select_channels(
    ...     ["GPS Latitude", "GPS Longitude", "speed_kmh", "BrakePress"]
    ... ).resample_to_channel("GPS Latitude").channels
    >>> fig = plot_gps_channels(
    ...     channels,
    ...     lat_channel="GPS Latitude",
    ...     lon_channel="GPS Longitude",
    ...     color_channels=[("speed_kmh", "Speed (km/h)", "Viridis")],
    ... )
    """
    # Extract lat/lon arrays
    lat = channels[lat_channel].column(lat_channel).to_numpy()
    lon = channels[lon_channel].column(lon_channel).to_numpy()

    # Build color_channels list with actual arrays
    color_data = []
    for channel_name, label, colorscale in color_channels:
        values = channels[channel_name].column(channel_name).to_numpy()
        color_data.append((values, label, colorscale))

    return plot_lap_gps(lat, lon, color_data, width=width, height=height, title=title)


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
    lat_series = pd.Series(lat)
    lon_series = pd.Series(lon)
    (lat_a, lon_a), (lat_b, lon_b) = compute_start_line(lat_series, lon_series)

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


def _discover_tire_temp_channels(
    channel_names: dict[str, str],
    available_channels: set[str] | None = None,
) -> dict[str, list[str]]:
    """Discover tire temperature channel names per corner from channel_names dict.

    Parameters
    ----------
    channel_names : dict[str, str]
        Channel name mapping containing ``tire_temp_{corner}_{n}`` keys.
    available_channels : set[str] or None
        If provided, only include channels that exist in this set.
        Stops adding channels for a corner at the first missing one.

    Returns
    -------
    dict[str, list[str]]
        Mapping from corner abbreviation (``"fl"``, ``"fr"``, ``"rl"``, ``"rr"``)
        to list of actual channel names in sensor order.
    """
    result: dict[str, list[str]] = {}
    for corner in ("fl", "fr", "rl", "rr"):
        channels_list: list[str] = []
        n = 1
        while f"tire_temp_{corner}_{n}" in channel_names:
            ch_name = channel_names[f"tire_temp_{corner}_{n}"]
            if available_channels is not None and ch_name not in available_channels:
                break
            channels_list.append(ch_name)
            n += 1
        if channels_list:
            result[corner] = channels_list
    return result


def plot_tire_thermography(
    channels: dict[str, pa.Table],
    channel_names: dict[str, str] | None = None,
    title: str = "Tire Temperatures",
    width: int = 900,
    height: int = 900,
) -> go.Figure:
    """Create tire temperature heatmap with speed, G-force, and driver inputs.

    Creates a 6-row subplot figure showing:
    - Rows 1-4: Tire temperature heatmaps (FL, FR, RL, RR)
    - Row 5: Speed and combined G-force
    - Row 6: Driver inputs (throttle, brake, steering)

    All channels must be pre-resampled to a common timebase by the caller
    using log.resample_to_channel().

    Parameters
    ----------
    channels : dict[str, pa.Table]
        Channel tables for a single lap, pre-resampled to a common timebase.
    channel_names : dict[str, str], optional
        Mapping from canonical keys to actual channel names. Must contain
        ``tire_temp_{corner}_{n}`` keys for tire temps, plus ``lateral_g``,
        ``inline_g``, ``brake``, ``throttle``, and ``steering``.
        If None, falls back to legacy hardcoded AIM channel names for
        backwards compatibility.
    title : str, default="Tire Temperatures"
        Plot title.
    width : int, default=900
        Figure width in pixels.
    height : int, default=900
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If required tire temperature channels are missing.

    Examples
    --------
    >>> lap_log = log.filter_by_lap(best_lap_num)
    >>> channels = lap_log.select_channels(tire_channels + other_channels)
    ...     .resample_to_channel("distance_m").channels
    >>> fig = plot_tire_thermography(channels, CHANNEL_NAMES, title="Lap Tire Temps")
    >>> show_fig(fig)
    """
    # Build legacy channel_names if not provided (backwards compatibility)
    if channel_names is None:
        channel_names = {
            "lateral_g": "LateralAcc",
            "inline_g": "InlineAcc",
            "brake": "BrakePress",
            "throttle": "PPS",
            "steering": "SteerAngle",
        }
        for corner_key, corner_prefix in [("fl", "FL"), ("fr", "FR"), ("rl", "RL"), ("rr", "RR")]:
            for i in range(1, 9):
                channel_names[f"tire_temp_{corner_key}_{i}"] = f"{corner_prefix}_Ch{i}"

    # Discover tire temp channels per corner (only those present in data)
    tire_channels = _discover_tire_temp_channels(channel_names, set(channels.keys()))
    if not tire_channels:
        raise ValueError(
            "No tire temperature channels found. "
            "channel_names must contain keys like tire_temp_fl_1, tire_temp_fr_1, etc."
        )

    # Resolve non-tire channel names
    lateral_g_ch = channel_names.get("lateral_g", "LateralAcc")
    inline_g_ch = channel_names.get("inline_g", "InlineAcc")
    brake_ch = channel_names.get("brake", "BrakePress")
    throttle_ch = channel_names.get("throttle", "PPS")
    steering_ch = channel_names.get("steering", "SteerAngle")

    # Validate required channels exist
    required_channels = [
        "distance_m",
        "speed_kmh",
        lateral_g_ch,
        inline_g_ch,
        brake_ch,
        throttle_ch,
        steering_ch,
    ]
    for corner_channels in tire_channels.values():
        required_channels.extend(corner_channels)

    missing = [ch for ch in required_channels if ch not in channels]
    if missing:
        raise ValueError(
            f"Missing channels: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
            f"Ensure all tire temperature and input channels are present."
        )

    # Extract distance array (reference)
    distance_m = channels["distance_m"].column("distance_m").to_numpy()

    # Extract tire temperature data as 2D arrays per corner
    corner_order = ["fl", "fr", "rl", "rr"]
    corner_labels = {"fl": "Front Left", "fr": "Front Right", "rl": "Rear Left", "rr": "Rear Right"}
    corner_temps: dict[str, np.ndarray] = {}
    corner_sensor_counts: dict[str, int] = {}

    for corner in corner_order:
        ch_names = tire_channels.get(corner, [])
        if ch_names:
            corner_temps[corner] = np.vstack(
                [channels[ch].column(ch).to_numpy() for ch in ch_names]
            )
            corner_sensor_counts[corner] = len(ch_names)

    # Calculate Sum of G
    lateral_acc = channels[lateral_g_ch].column(lateral_g_ch).to_numpy()
    inline_acc = channels[inline_g_ch].column(inline_g_ch).to_numpy()
    sum_of_g = np.sqrt(lateral_acc**2 + inline_acc**2)

    # Get color scale range across all tires
    all_temps = [t for t in corner_temps.values()]
    vmin = float(min(t.min() for t in all_temps))
    vmax = float(max(t.max() for t in all_temps))

    # Read accel unit from PyArrow field metadata if available
    accel_unit = _get_channel_unit(channels[inline_g_ch], inline_g_ch) or "G"

    # Create subplots
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            corner_labels.get("fl", "Front Left"),
            corner_labels.get("fr", "Front Right"),
            corner_labels.get("rl", "Rear Left"),
            corner_labels.get("rr", "Rear Right"),
            "Speed & Sum of G",
            "Driver Inputs",
        ),
        row_heights=[0.17, 0.17, 0.17, 0.17, 0.16, 0.16],
        specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # Extract arrays for speed/brake/throttle/steering
    speed_kmh = channels["speed_kmh"].column("speed_kmh").to_numpy()
    brake_press = channels[brake_ch].column(brake_ch).to_numpy()
    throttle_pps = channels[throttle_ch].column(throttle_ch).to_numpy()
    steer_angle = channels[steering_ch].column(steering_ch).to_numpy()

    # Add heatmaps for each corner
    for row_idx, corner in enumerate(corner_order, start=1):
        if corner not in corner_temps:
            continue
        temps = corner_temps[corner]
        n_sensors = corner_sensor_counts[corner]
        y_labels = [str(i) for i in range(1, n_sensors + 1)]

        show_colorbar = row_idx == 4  # Only show on last heatmap
        fig.add_trace(
            go.Heatmap(
                z=temps,
                x=distance_m,
                y=y_labels,
                colorscale="Inferno",
                zmin=vmin,
                zmax=vmax,
                showscale=show_colorbar,
                **({"colorbar": dict(title="Temp (°C)")} if show_colorbar else {}),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(autorange="reversed", row=row_idx, col=1)

    # Speed trace
    fig.add_trace(
        go.Scatter(
            x=distance_m,
            y=speed_kmh,
            mode="lines",
            name="Speed",
            line=dict(color="black", width=1),
        ),
        row=5,
        col=1,
        secondary_y=False,
    )

    # Sum of G trace
    fig.add_trace(
        go.Scatter(
            x=distance_m,
            y=sum_of_g,
            mode="lines",
            name="Sum of G",
            line=dict(color="red", width=1),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )

    # Brake trace
    fig.add_trace(
        go.Scatter(
            x=distance_m,
            y=brake_press,
            mode="lines",
            name="Brake",
            line=dict(color="red", width=1),
        ),
        row=6,
        col=1,
        secondary_y=False,
    )

    # Throttle trace
    fig.add_trace(
        go.Scatter(
            x=distance_m,
            y=throttle_pps,
            mode="lines",
            name="Throttle",
            line=dict(color="green", width=1),
        ),
        row=6,
        col=1,
        secondary_y=False,
    )

    # Steering trace
    fig.add_trace(
        go.Scatter(
            x=distance_m,
            y=steer_angle,
            mode="lines",
            name="Steering",
            line=dict(color="black", width=1),
        ),
        row=6,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        xaxis6_title="Distance (m)",
        yaxis_title="FL",
        yaxis2_title="FR",
        yaxis3_title="RL",
        yaxis4_title="RR",
        width=width,
        height=height,
        showlegend=False,
    )

    # Set y-axis titles for speed/G subplot
    fig.update_yaxes(title_text="km/h", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text=accel_unit, row=5, col=1, secondary_y=True)

    # Set y-axis titles for driver inputs subplot
    fig.update_yaxes(title_text="%", row=6, col=1, secondary_y=False)
    fig.update_yaxes(title_text="deg", row=6, col=1, secondary_y=True)

    # Hide tick labels on heatmap y-axes
    for row in range(1, 5):
        fig.update_yaxes(showticklabels=False, row=row, col=1)

    return fig


def plot_track_segments(
    lap_channels: pd.DataFrame,
    segments: list["TrackSegment"],
    title: str = "Track Segments",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """Plot GPS map with color-coded braking, corner, and acceleration zones.

    Parameters
    ----------
    lap_channels : pd.DataFrame
        Channel data for a single lap. Must contain:
        - distance_m: Distance along lap (meters)
        - GPS Latitude: Latitude values
        - GPS Longitude: Longitude values
    segments : list[TrackSegment]
        List of TrackSegment objects defining track zones.
    title : str, default="Track Segments"
        Plot title.
    width : int, default=900
        Figure width in pixels.
    height : int, default=700
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure object with mapbox layout.

    Examples
    --------
    >>> fig = plot_track_segments(lap_channels, segments)
    >>> show_fig(fig)
    """
    distance_arr = np.asarray(lap_channels["distance_m"])
    lat_arr = np.asarray(lap_channels["GPS Latitude"])
    lon_arr = np.asarray(lap_channels["GPS Longitude"])

    def get_indices_for_range(start_dist: float, end_dist: float) -> np.ndarray:
        mask = (distance_arr >= start_dist) & (distance_arr <= end_dist)
        return np.where(mask)[0]

    segment_colors = {"braking": "red", "corner": "orange", "acceleration": "green"}

    fig = go.Figure()

    # Plot base track (gray)
    fig.add_trace(
        go.Scattermapbox(
            lat=lat_arr,
            lon=lon_arr,
            mode="lines",
            line=dict(width=3, color="lightgray"),
            name="Track",
            showlegend=True,
        )
    )

    # Track which segment types we've added to legend
    legend_added = {"braking": False, "corner": False, "acceleration": False}

    legend_names = {
        "braking": "Braking Zone",
        "corner": "Corner",
        "acceleration": "Acceleration Zone",
    }

    # Plot all segments
    for seg in segments:
        indices = get_indices_for_range(seg.start_dist, seg.end_dist)
        if len(indices) > 0:
            color = segment_colors.get(seg.segment_type, "gray")
            show_in_legend = not legend_added[seg.segment_type]
            legend_added[seg.segment_type] = True

            fig.add_trace(
                go.Scattermapbox(
                    lat=lat_arr[indices],
                    lon=lon_arr[indices],
                    mode="lines",
                    line=dict(width=6, color=color),
                    name=legend_names.get(seg.segment_type) if show_in_legend else None,
                    showlegend=show_in_legend,
                    legendgroup=seg.segment_type,
                )
            )

    # Add corner apex markers with labels
    for seg in segments:
        if seg.segment_type == "corner" and seg.apex_dist is not None:
            apex_idx = int(np.argmin(np.abs(distance_arr - seg.apex_dist)))
            fig.add_trace(
                go.Scattermapbox(
                    lat=[lat_arr[apex_idx]],
                    lon=[lon_arr[apex_idx]],
                    mode="markers+text",
                    marker=dict(size=12, color="darkred", symbol="circle"),
                    text=[seg.name],
                    textposition="top right",
                    textfont=dict(size=11, color="darkred"),
                    name=None,
                    showlegend=False,
                )
            )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=np.mean(lat_arr), lon=np.mean(lon_arr)),
            zoom=14,
        ),
        title=title,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        width=width,
        height=height,
    )

    return fig


def plot_suspension_velocity_histogram(
    result: "VelocityHistogramResult",
    title: str = "Suspension Velocity Distribution",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """Create 4-quadrant suspension velocity histogram with velocity range shading.

    Creates a 2x2 subplot grid showing velocity histograms for all four corners
    (FL, FR, RL, RR) with background shading indicating velocity ranges:
    - Gray: Friction range (< 5mm/s)
    - Light blue: Slow range (5-25mm/s)
    - Light green: Fast range (25-200mm/s)
    - Light coral: Curb range (> 200mm/s)

    Parameters
    ----------
    result : VelocityHistogramResult
        Analysis results from analyze_suspension_velocity().
    title : str, default="Suspension Velocity Distribution"
        Plot title.
    width : int, default=1000
        Figure width in pixels.
    height : int, default=800
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with 4 velocity histogram subplots.

    Examples
    --------
    >>> from motorsports_data_notebook.suspension import analyze_suspension_velocity
    >>> result = analyze_suspension_velocity(log, lap_start, lap_end)
    >>> fig = plot_suspension_velocity_histogram(result)
    >>> show_fig(fig)
    """
    # Get velocity ranges for shading
    ranges = result.velocity_ranges

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=("Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)"),
    )

    # Define corner data and positions
    corners = [
        (result.front_left, 1, 1),
        (result.front_right, 1, 2),
        (result.rear_left, 2, 1),
        (result.rear_right, 2, 2),
    ]

    # Get max y value across all histograms for consistent scaling
    max_y = max(
        corner_data.histogram.max()
        for corner_data, _, _ in corners
        if len(corner_data.histogram) > 0
    )

    # Add velocity range shading to each subplot
    for corner_data, row, col in corners:
        # Add background shading for velocity ranges
        # Determine axis references for this subplot
        subplot_idx = (row - 1) * 2 + col
        xref = "x" if subplot_idx == 1 else f"x{subplot_idx}"
        yref = "y" if subplot_idx == 1 else f"y{subplot_idx}"

        # Friction range (gray) - center
        fig.add_shape(
            type="rect",
            x0=-ranges.friction,
            x1=ranges.friction,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="gray",
            opacity=0.15,
            line_width=0,
            layer="below",
        )

        # Slow range (light blue) - positive
        fig.add_shape(
            type="rect",
            x0=ranges.friction,
            x1=ranges.slow,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightblue",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Slow range (light blue) - negative
        fig.add_shape(
            type="rect",
            x0=-ranges.slow,
            x1=-ranges.friction,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightblue",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Fast range (light green) - positive
        fig.add_shape(
            type="rect",
            x0=ranges.slow,
            x1=ranges.fast,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightgreen",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Fast range (light green) - negative
        fig.add_shape(
            type="rect",
            x0=-ranges.fast,
            x1=-ranges.slow,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightgreen",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Curb range (light coral) - positive
        fig.add_shape(
            type="rect",
            x0=ranges.fast,
            x1=300,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightcoral",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Curb range (light coral) - negative
        fig.add_shape(
            type="rect",
            x0=-300,
            x1=-ranges.fast,
            y0=0,
            y1=1,
            xref=xref,
            yref=f"{yref} domain",
            fillcolor="lightcoral",
            opacity=0.2,
            line_width=0,
            layer="below",
        )

        # Add histogram bars
        # Color bars: positive (bump) = blue, negative (rebound) = red
        bar_colors = [
            "steelblue" if center >= 0 else "indianred" for center in corner_data.bin_centers
        ]

        fig.add_trace(
            go.Bar(
                x=corner_data.bin_centers,
                y=corner_data.histogram,
                marker_color=bar_colors,
                name=corner_data.corner_name,
                showlegend=False,
                hovertemplate=("Velocity: %{x:.0f} mm/s<br>" "Time: %{y:.1f}%<extra></extra>"),
            ),
            row=row,
            col=col,
        )

        # Add zero reference line
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            line_width=1,
            opacity=0.5,
            row=row,
            col=col,
        )

        # Add statistics annotation
        stats_text = f"Skew: {corner_data.skew:.2f}<br>" f"Std: {corner_data.std:.0f} mm/s"
        # Add annotation at top right of subplot
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref=f"x{row * 2 - 2 + col} domain" if row > 1 or col > 1 else "x domain",
            yref=f"y{row * 2 - 2 + col} domain" if row > 1 or col > 1 else "y domain",
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255,255,255,0.7)",
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        bargap=0.05,
    )

    # Update x-axes labels (only bottom row)
    fig.update_xaxes(title_text="Velocity (mm/s)", row=2, col=1)
    fig.update_xaxes(title_text="Velocity (mm/s)", row=2, col=2)

    # Update y-axes labels (only left column)
    fig.update_yaxes(title_text="Time (%)", row=1, col=1)
    fig.update_yaxes(title_text="Time (%)", row=2, col=1)

    # Set consistent y-axis range
    fig.update_yaxes(range=[0, max_y * 1.1])

    return fig


def plot_tire_grip_scatter(
    result: "TireGripResult",
    title: str = "Tire Grip Analysis",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """Create 2x2 scatter plot of total G vs tire pressure/temperature.

    Each subplot shows one corner (FL, FR, RL, RR) with a bucketed percentile
    line plot of total acceleration vs the selected tire metric.

    Parameters
    ----------
    result : TireGripResult
        Analysis results from analyze_tire_grip().
    title : str, default="Tire Grip Analysis"
        Plot title.
    width : int, default=1000
        Figure width in pixels.
    height : int, default=800
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with 4 scatter subplots.

    Examples
    --------
    >>> result = analyze_tire_grip(lap_log, channel_names, metric_mode="pressure")
    >>> fig = plot_tire_grip_scatter(result, title="Tire Grip - Lap 3")
    >>> show_fig(fig)
    """
    metric_label = "Pressure" if result.metric_mode == "pressure" else "Temperature"
    metric_unit = result.metric_unit
    accel_unit = result.accel_unit
    accel_label = f"Total Accel ({accel_unit})" if accel_unit else "Total Accel"

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=("Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)"),
    )

    corners = [
        (result.front_left, 1, 1),
        (result.front_right, 1, 2),
        (result.rear_left, 2, 1),
        (result.rear_right, 2, 2),
    ]

    for corner_data, row, col in corners:
        fig.add_trace(
            go.Scatter(
                x=corner_data.bucket_centers,
                y=corner_data.bucket_values,
                mode="lines+markers",
                marker=dict(size=6, color="steelblue"),
                line=dict(width=2, color="steelblue"),
                name=corner_data.corner_name,
                showlegend=False,
                customdata=corner_data.bucket_counts,
                hovertemplate=(
                    f"{metric_label}: %{{x:.1f}} {metric_unit}<br>"
                    f"P{corner_data.percentile:g} Accel: %{{y:.2f}} {accel_unit}<br>"
                    "n = %{customdata}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Sample count annotation
        subplot_idx = (row - 1) * 2 + col
        xref = "x domain" if subplot_idx == 1 else f"x{subplot_idx} domain"
        yref = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
        total_n = int(corner_data.bucket_counts.sum()) if len(corner_data.bucket_counts) > 0 else 0
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref=xref,
            yref=yref,
            text=f"n = {total_n}",
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255,255,255,0.7)",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )

    # X-axis labels (bottom row only)
    x_label = f"{metric_label} ({metric_unit})" if metric_unit else metric_label
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_xaxes(title_text=x_label, row=2, col=2)

    # Y-axis labels (left column only)
    fig.update_yaxes(title_text=accel_label, row=1, col=1)
    fig.update_yaxes(title_text=accel_label, row=2, col=1)

    return fig


# ── Track map image (matplotlib) ─────────────────────────────────────────────


def save_track_map(
    lat: np.ndarray,
    lon: np.ndarray,
    distance: np.ndarray,
    segments: list["TrackSegment"],
    output_path: str,
    title: str = "Track Map",
    dpi: int = 150,
) -> None:
    """Save a track map image with labeled corners using matplotlib.

    Parameters
    ----------
    lat : np.ndarray
        GPS latitude values for the reference lap.
    lon : np.ndarray
        GPS longitude values for the reference lap.
    distance : np.ndarray
        Distance along track in meters for the reference lap.
    segments : list[TrackSegment]
        Track segments to color-code.
    output_path : str
        Path to save the image (PNG, JPG, etc.).
    title : str
        Chart title.
    dpi : int
        Image resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from motorsports_data_notebook.corners import gps_to_local_xy

    x, y = gps_to_local_xy(lat, lon)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")
    ax.set_title(title, fontsize=12, color="white", pad=10)

    # Base track (gray)
    ax.plot(x, y, color="#444444", linewidth=2, zorder=1)

    # Color-coded segments
    segment_colors = {"braking": "#FF4444", "corner": "#FF8800", "acceleration": "#44BB44"}
    legend_added: dict[str, bool] = {}
    legend_names = {"braking": "Braking", "corner": "Corner", "acceleration": "Acceleration"}

    for seg in segments:
        mask = (distance >= seg.start_dist) & (distance <= seg.end_dist)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        color = segment_colors.get(seg.segment_type, "gray")
        label = None
        if seg.segment_type not in legend_added:
            legend_added[seg.segment_type] = True
            label = legend_names.get(seg.segment_type, seg.segment_type)

        ax.plot(x[indices], y[indices], color=color, linewidth=4, zorder=2, label=label)

    # Corner apex markers with labels
    for seg in segments:
        if seg.segment_type == "corner" and seg.apex_dist is not None:
            apex_idx = int(np.argmin(np.abs(distance - seg.apex_dist)))
            ax.plot(x[apex_idx], y[apex_idx], "o", color="darkred", markersize=6, zorder=3)
            ax.annotate(
                seg.name,
                (x[apex_idx], y[apex_idx]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=4,
            )

    ax.set_aspect("equal")
    ax.legend(
        fontsize=9, loc="upper left", facecolor="#2a2a4e", edgecolor="gray", labelcolor="white"
    )
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_corner_comparison(
    per_lap_data: dict[int, dict[str, np.ndarray]],
    best_lap_num: int,
    corner_name: str,
    x_start: float,
    x_end: float,
    output_path: str,
    braking_start: float | None = None,
    corner_start: float | None = None,
    corner_end: float | None = None,
    apex_dist: float | None = None,
    dpi: int = 150,
) -> None:
    """Save a stacked input comparison plot for a single corner.

    Shows speed, throttle, brake, and optionally steering and total G
    as subplots sharing a distance x-axis. Best lap is bold colored, others gray.

    Parameters
    ----------
    per_lap_data : dict[int, dict[str, np.ndarray]]
        Per-lap channel arrays keyed by lap number. Each inner dict must
        have "distance_m", "speed", "throttle", "brake".
        Optional: "steering", "total_g".
    best_lap_num : int
        Lap number to highlight as best execution.
    corner_name : str
        Corner label for the title.
    x_start : float
        Distance start for the x-axis range.
    x_end : float
        Distance end for the x-axis range.
    output_path : str
        Path to save the image.
    braking_start : float, optional
        Distance of braking start for reference line.
    corner_start : float, optional
        Distance of corner entry for reference line.
    corner_end : float, optional
        Distance of corner exit for reference line.
    apex_dist : float, optional
        Distance of apex for reference line.
    dpi : int
        Image resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Determine which optional channels are available
    has_steering = any("steering" in d for d in per_lap_data.values())
    has_total_g = any("total_g" in d for d in per_lap_data.values())
    subplot_keys = ["speed", "throttle", "brake"]
    subplot_labels = ["Speed (km/h)", "Throttle (%)", "Brake (bar)"]
    subplot_colors = ["#00BFFF", "#00FF88", "#FF4444"]
    if has_steering:
        subplot_keys.append("steering")
        subplot_labels.append("Steering (°)")
        subplot_colors.append("#FFD700")
    if has_total_g:
        subplot_keys.append("total_g")
        subplot_labels.append("Total G")
        subplot_colors.append("#CC66FF")
    n_subplots = len(subplot_keys)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 2.5 * n_subplots), sharex=True)
    fig.set_facecolor("#1a1a2e")
    fig.suptitle(f"Corner Comparison — {corner_name}", fontsize=13, color="white", y=0.98)

    for ax_idx, (ax, key, label, color) in enumerate(
        zip(axes, subplot_keys, subplot_labels, subplot_colors)
    ):
        ax.set_facecolor("#1a1a2e")
        ax.set_ylabel(label, fontsize=9, color="white")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")

        # Plot other laps first (gray)
        for lap_num, data in per_lap_data.items():
            if lap_num == best_lap_num:
                continue
            if key not in data:
                continue
            dist = data["distance_m"]
            mask = (dist >= x_start) & (dist <= x_end)
            if not np.any(mask):
                continue
            ax.plot(dist[mask], data[key][mask], color="#666666", linewidth=0.8, alpha=0.4)

        # Plot best lap (bold)
        if best_lap_num in per_lap_data:
            best = per_lap_data[best_lap_num]
            if key in best:
                dist = best["distance_m"]
                mask = (dist >= x_start) & (dist <= x_end)
                if np.any(mask):
                    ax.plot(
                        dist[mask],
                        best[key][mask],
                        color=color,
                        linewidth=2.5,
                        label=f"Lap {best_lap_num} (best)",
                    )

        # Reference lines
        ref_lines = [
            (braking_start, "cyan", "dotted", "Brake start"),
            (corner_start, "#FFD700", "dashed", "Corner start"),
            (corner_end, "#FFD700", "dashed", "Corner end"),
            (apex_dist, "#FF4444", "solid", "Apex"),
        ]
        for ref_dist, ref_color, ref_style, ref_label in ref_lines:
            if ref_dist is not None and x_start <= ref_dist <= x_end:
                ax.axvline(
                    ref_dist,
                    color=ref_color,
                    linestyle=ref_style,
                    linewidth=1,
                    alpha=0.7,
                    label=ref_label if ax_idx == 0 else None,
                )

        if ax_idx == 0:
            ax.legend(
                fontsize=8,
                loc="upper right",
                facecolor="#2a2a4e",
                edgecolor="gray",
                labelcolor="white",
            )

    axes[-1].set_xlabel("Distance (m)", fontsize=9, color="white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_corner_map(
    per_lap_gps: dict[int, tuple[np.ndarray, np.ndarray]],
    best_lap_num: int,
    corner_name: str,
    output_path: str,
    apex_xy: tuple[float, float] | None = None,
    dpi: int = 150,
) -> None:
    """Save a zoomed GPS trace of a corner region.

    Best lap is bold orange, other laps are gray.

    Parameters
    ----------
    per_lap_gps : dict[int, tuple[np.ndarray, np.ndarray]]
        Per-lap (x, y) local coordinates for the corner region.
    best_lap_num : int
        Lap number to highlight.
    corner_name : str
        Corner label for the title.
    output_path : str
        Path to save the image.
    apex_xy : tuple[float, float], optional
        (x, y) coordinates of the apex on the best lap trace.
    dpi : int
        Image resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")
    ax.set_title(f"Corner Map — {corner_name}", fontsize=12, color="white", pad=10)

    # Plot other laps (gray)
    for lap_num, (lx, ly) in per_lap_gps.items():
        if lap_num == best_lap_num:
            continue
        ax.plot(lx, ly, color="#666666", linewidth=1, alpha=0.4)

    # Plot best lap (bold orange)
    if best_lap_num in per_lap_gps:
        bx, by = per_lap_gps[best_lap_num]
        ax.plot(bx, by, color="#FF8800", linewidth=3, label=f"Lap {best_lap_num} (best)")

    # Apex marker
    if apex_xy is not None:
        ax.plot(apex_xy[0], apex_xy[1], "o", color="#FF4444", markersize=8, zorder=5)
        ax.annotate(
            "Apex",
            apex_xy,
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    ax.set_aspect("equal")
    ax.legend(
        fontsize=9, loc="upper left", facecolor="#2a2a4e", edgecolor="gray", labelcolor="white"
    )
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
