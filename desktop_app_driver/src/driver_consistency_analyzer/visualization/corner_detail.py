"""Detail view: multi-lap overlay for a single corner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from driver_consistency_analyzer.analysis.driver_consistency import CornerConsistencyData


def draw_detail(
    fig: Figure,
    corner_data: CornerConsistencyData,
    title: str = "",
) -> None:
    """Draw detail multi-lap overlay for a single corner.

    Three-row subplot with x-axis = distance:
    1. Throttle - all selected laps overlaid (color-mapped)
    2. Brake Pressure - all laps overlaid
    3. Lateral G - all laps overlaid
    Vertical lines for corner start, apex, corner end.
    Annotation for throttle acceptance point.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw on.
    corner_data : CornerConsistencyData
        Data for the selected corner.
    title : str
        Chart title override.
    """
    fig.clear()

    corner = corner_data.corner
    traces = corner_data.lap_traces

    if not traces:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5,
            0.5,
            "No lap trace data available for this corner",
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    chart_title = title or f"{corner.name} ({corner.direction}) - Multi-Lap Overlay"
    fig.suptitle(chart_title, fontsize=12, color="white")

    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

    # Color map for laps
    n_laps = len(traces)
    colors = cm.viridis([i / max(n_laps - 1, 1) for i in range(n_laps)])  # type: ignore[attr-defined]

    for i, trace in enumerate(traces):
        color = colors[i]
        label = f"Lap {trace.lap_num}"

        # Throttle
        ax1.plot(trace.distance, trace.throttle, color=color, alpha=0.8, linewidth=1, label=label)

        # Brake
        ax2.plot(trace.distance, trace.brake, color=color, alpha=0.8, linewidth=1, label=label)

        # Lateral G
        ax3.plot(trace.distance, trace.lateral_g, color=color, alpha=0.8, linewidth=1, label=label)

    # Style axes
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=9)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Braking zone start line
        if corner_data.braking_start is not None:
            ax.axvline(
                x=corner_data.braking_start, color="cyan", linestyle=":", linewidth=0.8, alpha=0.6
            )

        # Corner boundary lines
        ax.axvline(x=corner.start_dist, color="yellow", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(x=corner.apex_dist, color="red", linestyle="-", linewidth=1, alpha=0.7)
        ax.axvline(x=corner.end_dist, color="yellow", linestyle="--", linewidth=0.8, alpha=0.6)

    # Labels
    ax1.set_ylabel("Throttle (%)", fontsize=10, color="white")
    ax1.set_title("Throttle Position", fontsize=11, color="white")

    ax2.set_ylabel("Brake Pressure", fontsize=10, color="white")
    ax2.set_title("Brake Pressure", fontsize=11, color="white")

    ax3.set_ylabel("Lateral G", fontsize=10, color="white")
    ax3.set_title("Lateral G", fontsize=11, color="white")
    ax3.set_xlabel("Distance (m)", fontsize=10, color="white")

    # TA annotation
    if corner_data.ta_values:
        ax1.text(
            0.02,
            0.95,
            f"TA: {corner_data.ta_mean:.1f}% (std: {corner_data.ta_std:.1f}%)",
            transform=ax1.transAxes,
            fontsize=9,
            color="white",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#333333", alpha=0.8),
        )

    # Legend on first subplot (compact)
    if n_laps <= 10:
        ax1.legend(fontsize=7, loc="upper right", ncol=2)

    # Boundary annotations on top subplot, placed just above the axes
    if corner_data.braking_start is not None:
        ax1.text(
            corner_data.braking_start,
            ax1.get_ylim()[1],
            " Brake",
            fontsize=9,
            color="cyan",
            va="bottom",
            alpha=0.8,
            clip_on=False,
        )
    ax1.text(
        corner.start_dist,
        ax1.get_ylim()[1],
        " Entry",
        fontsize=9,
        color="yellow",
        va="bottom",
        alpha=0.8,
        clip_on=False,
    )
    ax1.text(
        corner.apex_dist,
        ax1.get_ylim()[1],
        " Apex",
        fontsize=9,
        color="red",
        va="bottom",
        alpha=0.8,
        clip_on=False,
    )
    ax1.text(
        corner.end_dist,
        ax1.get_ylim()[1],
        " Exit",
        fontsize=9,
        color="yellow",
        va="bottom",
        alpha=0.8,
        clip_on=False,
    )

    fig.tight_layout()
