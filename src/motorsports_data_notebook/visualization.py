"""Visualization functions for motorsports data.

This module provides functions for creating interactive visualizations
of lap data and GPS track data.
"""

import math
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

if TYPE_CHECKING:
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
