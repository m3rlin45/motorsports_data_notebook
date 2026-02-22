"""Comparison visualization for suspension velocity histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from motorsports_data_notebook.suspension import VelocityHistogramResult


def plot_suspension_velocity_histogram_comparison(
    result_a: "VelocityHistogramResult",
    result_b: "VelocityHistogramResult",
    label_a: str = "Session A",
    label_b: str = "Session B",
    title: str = "Suspension Velocity Comparison",
    width: int = 1100,
    height: int = 850,
) -> go.Figure:
    """Plot two sessions with grouped bars for side-by-side comparison.

    Creates a 2x2 subplot grid showing velocity histograms for all four corners
    (FL, FR, RL, RR) with grouped bars where each velocity bin shows both
    sessions side-by-side.

    Parameters
    ----------
    result_a : VelocityHistogramResult
        Analysis results for session A.
    result_b : VelocityHistogramResult
        Analysis results for session B.
    label_a : str, default="Session A"
        Label for session A in legend.
    label_b : str, default="Session B"
        Label for session B in legend.
    title : str, default="Suspension Velocity Comparison"
        Plot title.
    width : int, default=1100
        Figure width in pixels.
    height : int, default=850
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with grouped bar histograms for comparison.

    Examples
    --------
    >>> fig = plot_suspension_velocity_histogram_comparison(
    ...     result_a, result_b,
    ...     label_a="Setup 1", label_b="Setup 2",
    ... )
    """
    # Use ranges from session A (assumes both use same ranges)
    ranges = result_a.velocity_ranges

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Front Left (FL)",
            "Front Right (FR)",
            "Rear Left (RL)",
            "Rear Right (RR)",
        ),
    )

    # Define corner data and positions
    corners = [
        (result_a.front_left, result_b.front_left, 1, 1),
        (result_a.front_right, result_b.front_right, 1, 2),
        (result_a.rear_left, result_b.rear_left, 2, 1),
        (result_a.rear_right, result_b.rear_right, 2, 2),
    ]

    # Get max y value across all histograms for consistent scaling
    max_y = 0
    for corner_a, corner_b, _, _ in corners:
        if len(corner_a.histogram) > 0:
            max_y = max(max_y, corner_a.histogram.max())
        if len(corner_b.histogram) > 0:
            max_y = max(max_y, corner_b.histogram.max())

    # Define colors for sessions
    color_a = "steelblue"
    color_b = "darkorange"

    # Track if we've added legend entries
    legend_added = False

    # Add traces for each corner
    for corner_a, corner_b, row, col in corners:
        # Add velocity range shading
        subplot_idx = (row - 1) * 2 + col
        xref = "x" if subplot_idx == 1 else f"x{subplot_idx}"
        yref = "y" if subplot_idx == 1 else f"y{subplot_idx}"

        # Add background shading (same as single histogram)
        _add_velocity_range_shading(fig, ranges, xref, yref)

        # Add Session A bars
        fig.add_trace(
            go.Bar(
                x=corner_a.bin_centers,
                y=corner_a.histogram,
                marker_color=color_a,
                name=label_a,
                legendgroup="A",
                showlegend=not legend_added,
                hovertemplate=(
                    f"{label_a}<br>" "Velocity: %{x:.0f} mm/s<br>" "Time: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Add Session B bars
        fig.add_trace(
            go.Bar(
                x=corner_b.bin_centers,
                y=corner_b.histogram,
                marker_color=color_b,
                name=label_b,
                legendgroup="B",
                showlegend=not legend_added,
                hovertemplate=(
                    f"{label_b}<br>" "Velocity: %{x:.0f} mm/s<br>" "Time: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        legend_added = True

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

        # Add statistics annotation comparing both sessions
        stats_text = (
            f"<b>{label_a[:8]}</b>: Skew {corner_a.skew:.2f}, Std {corner_a.std:.0f}<br>"
            f"<b>{label_b[:8]}</b>: Skew {corner_b.skew:.2f}, Std {corner_b.std:.0f}"
        )

        xref_ann = f"x{row * 2 - 2 + col} domain" if row > 1 or col > 1 else "x domain"
        yref_ann = f"y{row * 2 - 2 + col} domain" if row > 1 or col > 1 else "y domain"

        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref=xref_ann,
            yref=yref_ann,
            text=stats_text,
            showarrow=False,
            font=dict(size=9),
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
        )

    # Update layout for grouped bars
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        bargap=0.1,
        bargroupgap=0.0,
        barmode="group",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    # Update x-axes labels (only bottom row)
    fig.update_xaxes(title_text="Velocity (mm/s)", row=2, col=1)
    fig.update_xaxes(title_text="Velocity (mm/s)", row=2, col=2)

    # Update y-axes labels (only left column)
    fig.update_yaxes(title_text="Time (%)", row=1, col=1)
    fig.update_yaxes(title_text="Time (%)", row=2, col=1)

    # Set consistent y-axis range
    fig.update_yaxes(range=[0, max_y * 1.15])

    return fig


def _add_velocity_range_shading(
    fig: go.Figure,
    ranges,
    xref: str,
    yref: str,
) -> None:
    """Add velocity range background shading to a subplot.

    Parameters
    ----------
    fig : go.Figure
        The figure to add shapes to.
    ranges : VelocityRanges
        Velocity thresholds for shading.
    xref : str
        X-axis reference string for the subplot.
    yref : str
        Y-axis reference string for the subplot.
    """
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
