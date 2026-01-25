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
import pyarrow.compute as pc
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from libxrk.base import LogFile

    from .corners import Corner
    from .zones import TrackSegment


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

    # Throttle trace (scaled to fit on same axis as G)
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=np.asarray(throttle) / 100 * 2,  # Scale 0-100% to 0-2 for visibility
            mode="lines",
            name="Throttle (scaled)",
            line=dict(color="green", width=2),
            hovertemplate="Distance: %{x:.0f}m<br>Throttle: %{customdata:.0f}%<extra></extra>",
            customdata=throttle,
        )
    )

    # Brake trace (scaled to fit on same axis as G)
    if brake is not None:
        brake_arr = np.asarray(brake)
        brake_max = brake_arr.max()
        if brake_max > 0:
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=brake_arr / brake_max * 2,  # Scale to 0-2 for visibility
                    mode="lines",
                    name="Brake (scaled)",
                    line=dict(color="red", width=2),
                    hovertemplate="Distance: %{x:.0f}m<br>Brake: %{customdata:.0f}<extra></extra>",
                    customdata=brake,
                )
            )

    # Steering trace (scaled to fit on same axis as G)
    if steering is not None:
        steering_arr = np.asarray(steering)
        steer_max = np.abs(steering_arr).max()
        if steer_max > 0:
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=steering_arr / steer_max,  # Scale to -1 to 1
                    mode="lines",
                    name="Steering (scaled)",
                    line=dict(color="purple", width=2),
                    hovertemplate="Distance: %{x:.0f}m<br>Steering: %{customdata:.1f}°<extra></extra>",
                    customdata=steering,
                )
            )

    # Lateral G trace
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=lateral_g,
            mode="lines",
            name="Lateral G",
            line=dict(color="blue", width=2),
            hovertemplate="Distance: %{x:.0f}m<br>Lateral G: %{y:.2f}G<extra></extra>",
        )
    )

    # Peak lateral G reference line
    fig.add_hline(
        y=throttle_acceptance_result["peak_lateral_g"],
        line_dash="dash",
        line_color="blue",
        opacity=0.7,
        annotation_text=f"Peak Lateral G: {throttle_acceptance_result['peak_lateral_g']:.2f}G",
        annotation_position="top right",
    )

    # Lateral G at throttle acceptance reference line
    fig.add_hline(
        y=throttle_acceptance_result["lateral_g_at_throttle"],
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text=f"Lateral G at Full Throttle: {throttle_acceptance_result['lateral_g_at_throttle']:.2f}G ({throttle_acceptance_result['throttle_acceptance_pct']:.0f}%)",
        annotation_position="bottom right",
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

    fig.update_layout(
        title=title if title is not None else f"Throttle Acceptance - {corner.name}",
        xaxis_title="Distance (m)",
        yaxis_title="Lateral G / Inputs (scaled)",
        width=width,
        height=height,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


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


def get_best_lap_data(channels: pd.DataFrame, laps: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Extract best lap info and corresponding channel data.

    Parameters
    ----------
    channels : pd.DataFrame
        Channel data with 'timecodes' column.
    laps : pd.DataFrame
        Laps table with 'start_time', 'end_time' columns.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (best_lap, lap_channels) - best lap info and channel data for that lap.

    Examples
    --------
    >>> best_lap, lap_channels = get_best_lap_data(channels, laps)
    >>> print(f"Best lap: {best_lap['num']}")
    """
    best_lap = get_best_lap(laps)
    start_ts = best_lap["start_time"]
    end_ts = best_lap["end_time"]
    # Use < for end_ts to exclude the first sample of the next lap
    lap_channels = channels.query(f"timecodes >= @start_ts and timecodes < @end_ts").copy()
    return best_lap, lap_channels


def get_lap_channels(
    log: "LogFile",
    channel_names: list[str],
    start_time: int,
    end_time: int,
) -> dict[str, pa.Table]:
    """Extract channels filtered to a time range, each at their native timebase.

    This is a fast alternative to get_channels_as_table() when you only need
    specific channels and don't need them merged onto a common timebase.

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    channel_names : list[str]
        Names of channels to extract (e.g., ["GPS Latitude", "GPS Longitude", "speed_kmh"]).
    start_time : int
        Start timestamp in milliseconds (inclusive).
    end_time : int
        End timestamp in milliseconds (exclusive).

    Returns
    -------
    dict[str, pa.Table]
        Dictionary mapping channel names to PyArrow tables, same format as log.channels.
        Each table has columns: 'timecodes' (int64) and the channel value column.

    Examples
    --------
    >>> channels = get_lap_channels(log, ["GPS Latitude", "GPS Longitude"], 60000, 120000)
    >>> lat = channels["GPS Latitude"].column("GPS Latitude").to_numpy()
    """
    result = {}
    for name in channel_names:
        if name not in log.channels:
            raise KeyError(f"Channel '{name}' not found in log. Available: {list(log.channels.keys())}")

        table = log.channels[name]
        timecodes = table.column("timecodes")

        # Create filter mask: start_time <= timecodes < end_time
        mask = pc.and_(
            pc.greater_equal(timecodes, start_time),
            pc.less(timecodes, end_time),
        )

        # Filter the table
        result[name] = table.filter(mask)

    return result


def get_best_lap_channels(
    log: "LogFile",
    laps: pd.DataFrame,
    channel_names: list[str],
) -> tuple[pd.Series, dict[str, pa.Table]]:
    """Extract best lap info and channels at their native timebases.

    This is a fast alternative to get_best_lap_data() that avoids the expensive
    get_channels_as_table() merge operation.

    Parameters
    ----------
    log : LogFile
        The loaded log file with channels dict.
    laps : pd.DataFrame
        Laps table with 'start_time', 'end_time' columns.
    channel_names : list[str]
        Names of channels to extract.

    Returns
    -------
    tuple[pd.Series, dict[str, pa.Table]]
        (best_lap, channels) - best lap info and channel tables for that lap.
        Each channel table has columns: 'timecodes' (int64) and the channel value.

    Examples
    --------
    >>> best_lap, channels = get_best_lap_channels(log, laps, ["GPS Latitude", "GPS Longitude", "speed_kmh"])
    >>> lat = channels["GPS Latitude"].column("GPS Latitude").to_numpy()
    """
    best_lap = get_best_lap(laps)
    start_ts = best_lap["start_time"]
    end_ts = best_lap["end_time"]

    channels = get_lap_channels(log, channel_names, start_ts, end_ts)
    return best_lap, channels


def interpolate_channels(
    channels: dict[str, pa.Table],
    reference_channel: str,
) -> dict[str, pa.Table]:
    """Interpolate all channels to a reference channel's timebase.

    Use this to align channels with different sample rates to a common timebase.
    For example, to align brake pressure (sampled at 100Hz) to GPS timestamps (10Hz).

    The reference channel is copied unchanged. All other channels are linearly
    interpolated to match the reference channel's timecodes.

    Parameters
    ----------
    channels : dict[str, pa.Table]
        Dictionary of channel tables (as returned by get_lap_channels).
    reference_channel : str
        Name of channel whose timebase to use as the target.

    Returns
    -------
    dict[str, pa.Table]
        New dictionary with all channels interpolated to the reference timebase.
        Each table has columns: 'timecodes' (int64) and the channel value.

    Examples
    --------
    >>> best_lap, channels = get_best_lap_channels(log, laps,
    ...     ["GPS Latitude", "GPS Longitude", "speed_kmh", "BrakePress"])
    >>> aligned = interpolate_channels(channels, reference_channel="GPS Latitude")
    >>> # Now all channels share the same timecodes as GPS Latitude
    """
    if reference_channel not in channels:
        raise KeyError(
            f"Reference channel '{reference_channel}' not found. "
            f"Available: {list(channels.keys())}"
        )

    # Get reference timebase
    ref_table = channels[reference_channel]
    ref_times = ref_table.column("timecodes").to_numpy()

    result: dict[str, pa.Table] = {}

    for name, table in channels.items():
        if name == reference_channel:
            # Copy reference channel unchanged
            result[name] = table
        else:
            # Interpolate to reference timebase
            source_times = table.column("timecodes").to_numpy()
            source_values = table.column(name).to_numpy()
            interpolated = np.interp(ref_times, source_times, source_values)

            # Create new table with reference timecodes
            result[name] = pa.table({
                "timecodes": pa.array(ref_times, type=pa.int64()),
                name: pa.array(interpolated),
            })

    return result


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
    tables directly (as returned by get_lap_channels or get_best_lap_channels).

    All color channels are automatically interpolated to the lat_channel's
    timebase to ensure correct spatial alignment. Only the channels needed
    for the plot are interpolated.

    Parameters
    ----------
    channels : dict[str, pa.Table]
        Dictionary of channel tables (as returned by get_lap_channels).
    lat_channel : str
        Name of latitude channel (e.g., "GPS Latitude"). Also used as the
        reference timebase for interpolation.
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
    >>> best_lap, channels = get_best_lap_channels(log, laps,
    ...     ["GPS Latitude", "GPS Longitude", "speed_kmh", "BrakePress"])
    >>> fig = plot_gps_channels(
    ...     channels,
    ...     lat_channel="GPS Latitude",
    ...     lon_channel="GPS Longitude",
    ...     color_channels=[("speed_kmh", "Speed (km/h)", "Viridis")],
    ... )
    """
    # Determine which channels we need
    needed_channels = {lat_channel, lon_channel}
    for channel_name, _, _ in color_channels:
        needed_channels.add(channel_name)

    # Filter to only needed channels and interpolate to lat_channel timebase
    subset = {name: channels[name] for name in needed_channels}
    aligned = interpolate_channels(subset, reference_channel=lat_channel)

    # Extract lat/lon arrays
    lat = aligned[lat_channel].column(lat_channel).to_numpy()
    lon = aligned[lon_channel].column(lon_channel).to_numpy()

    # Build color_channels list with actual arrays
    color_data = []
    for channel_name, label, colorscale in color_channels:
        values = aligned[channel_name].column(channel_name).to_numpy()
        color_data.append((values, label, colorscale))

    return plot_lap_gps(lat, lon, color_data, width=width, height=height, title=title)


def get_top_laps(laps: pd.DataFrame, threshold_pct: float = 1.03) -> pd.DataFrame:
    """Get laps within threshold percentage of best lap time.

    Excludes first and last laps, and laps with zero or negative duration.

    Parameters
    ----------
    laps : pd.DataFrame
        Laps table with 'lap_time' column (as Timedelta).
    threshold_pct : float, default=1.03
        Threshold as multiplier (e.g., 1.03 for within 103% of best).

    Returns
    -------
    pd.DataFrame
        DataFrame of qualifying laps.

    Examples
    --------
    >>> top_laps = get_top_laps(laps, threshold_pct=1.03)
    >>> print(f"Using {len(top_laps)} laps for analysis")
    """
    if "lap_time" not in laps.columns:
        raise ValueError("Expected lap_time column in laps table")

    # Exclude first/last laps and zero-duration laps
    valid_laps: pd.DataFrame = laps[laps["lap_time"] > pd.Timedelta(0)][1:-1].copy()

    if len(valid_laps) == 0:
        return valid_laps

    best_lap_time = valid_laps["lap_time"].min()
    threshold_time = best_lap_time * threshold_pct
    top_laps: pd.DataFrame = valid_laps[valid_laps["lap_time"] <= threshold_time]

    return top_laps


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


def plot_tire_thermography(
    lap_channels: pd.DataFrame,
    title: str = "Tire Temperatures",
    width: int = 900,
    height: int = 900,
) -> go.Figure:
    """Create tire temperature heatmap with speed, G-force, and driver inputs.

    Creates a 6-row subplot figure showing:
    - Rows 1-4: Tire temperature heatmaps (FL, FR, RL, RR)
    - Row 5: Speed and combined G-force
    - Row 6: Driver inputs (throttle, brake, steering)

    Parameters
    ----------
    lap_channels : pd.DataFrame
        Channel data for a single lap. Must contain:
        - distance_m: Distance along lap (meters)
        - FL_Ch1 through FL_Ch8: Front left tire temps (Ch1=outside, Ch8=inside)
        - FR_Ch1 through FR_Ch8: Front right tire temps (Ch1=inside, Ch8=outside)
        - RL_Ch1 through RL_Ch8: Rear left tire temps (Ch1=outside, Ch8=inside)
        - RR_Ch1 through RR_Ch8: Rear right tire temps (Ch1=inside, Ch8=outside)
        - speed_kmh: Speed in km/h
        - LateralAcc: Lateral acceleration (G)
        - InlineAcc: Longitudinal acceleration (G)
        - BrakePress: Brake pressure (%)
        - PPS: Throttle position (%)
        - SteerAngle: Steering angle (degrees)
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
    >>> fig = plot_tire_thermography(lap_channels, title="Best Lap Tire Temps")
    >>> show_fig(fig)
    """
    # Validate required channels
    tire_positions = ["FL", "FR", "RL", "RR"]
    required_channels = []
    for pos in tire_positions:
        required_channels.extend([f"{pos}_Ch{i}" for i in range(1, 9)])

    missing = [ch for ch in required_channels if ch not in lap_channels.columns]
    if missing:
        raise ValueError(
            f"Missing tire temperature channels: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
            f"Expected channels like FL_Ch1 through FL_Ch8 for each tire position."
        )

    # Extract tire temperature data
    fl_channel_names = [f"FL_Ch{i}" for i in range(1, 9)]
    fr_channel_names = [f"FR_Ch{i}" for i in range(1, 9)]
    rl_channel_names = [f"RL_Ch{i}" for i in range(1, 9)]
    rr_channel_names = [f"RR_Ch{i}" for i in range(1, 9)]

    fl_temps = lap_channels[fl_channel_names].values.T  # Shape: (8, n_samples)
    fr_temps = lap_channels[fr_channel_names].values.T
    rl_temps = lap_channels[rl_channel_names].values.T
    rr_temps = lap_channels[rr_channel_names].values.T

    distance_m = np.asarray(lap_channels["distance_m"])

    # Calculate Sum of G
    sum_of_g = np.sqrt(
        np.asarray(lap_channels["LateralAcc"]) ** 2 + np.asarray(lap_channels["InlineAcc"]) ** 2
    )

    # Get color scale range across all tires
    vmin = float(min(fl_temps.min(), fr_temps.min(), rl_temps.min(), rr_temps.min()))
    vmax = float(max(fl_temps.max(), fr_temps.max(), rl_temps.max(), rr_temps.max()))

    # Create subplots
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Front Left (Outside at top)",
            "Front Right (Outside at bottom)",
            "Rear Left (Outside at top)",
            "Rear Right (Outside at bottom)",
            "Speed & Sum of G",
            "Driver Inputs",
        ),
        row_heights=[0.17, 0.17, 0.17, 0.17, 0.16, 0.16],
        specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    y_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]

    # Extract arrays for speed/brake/throttle/steering
    speed_kmh = np.asarray(lap_channels["speed_kmh"])
    brake_press = np.asarray(lap_channels["BrakePress"])
    throttle_pps = np.asarray(lap_channels["PPS"])
    steer_angle = np.asarray(lap_channels["SteerAngle"])

    # Front Left heatmap
    fig.add_trace(
        go.Heatmap(
            z=fl_temps,
            x=distance_m,
            y=y_labels,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    # Front Right heatmap
    fig.add_trace(
        go.Heatmap(
            z=fr_temps,
            x=distance_m,
            y=y_labels,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            showscale=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    # Rear Left heatmap
    fig.add_trace(
        go.Heatmap(
            z=rl_temps,
            x=distance_m,
            y=y_labels,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            showscale=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    # Rear Right heatmap
    fig.add_trace(
        go.Heatmap(
            z=rr_temps,
            x=distance_m,
            y=y_labels,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Temp (°C)"),
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=4, col=1)

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
    fig.update_yaxes(title_text="G", row=5, col=1, secondary_y=True)

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
