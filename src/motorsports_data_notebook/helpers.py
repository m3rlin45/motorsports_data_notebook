"""Helper functions for motorsports data analysis.

This module provides utility functions for working with lap data and creating
GPS track visualizations.
"""

import math
import sys

import plotly.graph_objects as go
import plotly.io as pio


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
    if 'pyodide' in sys.modules:
        # In Pyodide/JupyterLite, use HTML display
        from IPython.display import HTML, display
        html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        display(HTML(html))
    else:
        # Standard environment
        fig.show()


def get_best_lap(laps_df):
    """
    Find the best (fastest) lap by duration.
    
    Parameters
    ----------
    laps_df : pandas.DataFrame
        Laps table with 'start_time', 'end_time' columns.
        
    Returns
    -------
    pandas.Series
        The row corresponding to the best lap.
    """
    if not {'start_time','end_time'}.issubset(laps_df.columns):
        raise ValueError("Expected start_time and end_time columns in laps table")
    
    laps_with_duration = laps_df.copy()
    laps_with_duration['lap_duration_ms'] = laps_with_duration['end_time'] - laps_with_duration['start_time']
    best_idx = laps_with_duration['lap_duration_ms'].idxmin()
    return laps_with_duration.loc[best_idx]


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
    lat1 = lat.iloc[0]; lon1 = lon.iloc[0]
    idx2 = min(ahead_points, len(lat)-1)
    lat2 = lat.iloc[idx2]; lon2 = lon.iloc[idx2]
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
        
        fig.add_trace(go.Scattergl(
            x=lon,
            y=lat,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color_values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=label,
                    x=1.02 + (layer_position * 0.15),  # Offset colorbars horizontally
                    len=0.8
                )
            ),
            name=label,
            showlegend=False,
            hovertemplate=f'{label}: %{{marker.color:.2f}}<br>Lon: %{{x:.6f}}<br>Lat: %{{y:.6f}}<extra></extra>'
        ))
    
    # Add track outline
    fig.add_trace(go.Scattergl(
        x=lon,
        y=lat,
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.3,
        name='Track Outline',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add start/finish line
    fig.add_trace(go.Scattergl(
        x=[lon_a, lon_b],
        y=[lat_a, lat_b],
        mode='lines',
        line=dict(color='black', width=3),
        name='Start/Finish',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=title or 'GPS Path',
        width=width,
        height=height,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return fig
