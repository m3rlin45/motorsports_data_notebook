"""Track map view: GPS track with color-coded corners and segments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from motorsports_data_notebook.corners import gps_to_local_xy

if TYPE_CHECKING:
    from motorsports_data_notebook.zones import TrackSegment


SEGMENT_COLORS = {"braking": "#FF4444", "corner": "#FF8800", "acceleration": "#44BB44"}


def draw_track_map(
    fig: Figure,
    lat: np.ndarray,
    lon: np.ndarray,
    distance: np.ndarray,
    segments: list[TrackSegment],
    title: str = "Track Map",
    opportunity_scores: dict[int, float] | None = None,
) -> None:
    """Draw a track map with color-coded segments.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw on.
    lat : np.ndarray
        GPS latitude values for the reference lap.
    lon : np.ndarray
        GPS longitude values for the reference lap.
    distance : np.ndarray
        Distance along track in meters for the reference lap.
    segments : list[TrackSegment]
        Track segments to color-code.
    title : str
        Chart title.
    opportunity_scores : dict[int, float], optional
        Mapping of corner_id to opportunity score. When provided, acceleration
        segment linewidths scale with score and gold stars mark top-3 corners.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, fontsize=12, color="white")

    # Convert GPS to local XY
    x, y = gps_to_local_xy(lat, lon)

    # Draw base track (gray)
    ax.plot(x, y, color="#444444", linewidth=2, zorder=1)

    # Precompute max opportunity score for linewidth scaling
    max_opp = max(opportunity_scores.values()) if opportunity_scores else 0.0

    # Draw segments
    legend_added = {}
    for seg in segments:
        mask = (distance >= seg.start_dist) & (distance <= seg.end_dist)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        color = SEGMENT_COLORS.get(seg.segment_type, "gray")
        label = None
        if seg.segment_type not in legend_added:
            legend_added[seg.segment_type] = True
            label = {
                "braking": "Braking",
                "corner": "Corner",
                "acceleration": "Acceleration",
            }.get(seg.segment_type, seg.segment_type)

        # Scale acceleration linewidth by opportunity score
        lw = 4
        if (
            seg.segment_type == "acceleration"
            and opportunity_scores
            and seg.corner_id in opportunity_scores
            and max_opp > 0
        ):
            norm = opportunity_scores[seg.corner_id] / max_opp
            lw = 4 + 6 * norm  # 4 (baseline) to 10 (highest)

        ax.plot(x[indices], y[indices], color=color, linewidth=lw, zorder=2, label=label)

    # Draw corner apex markers with labels
    for seg in segments:
        if seg.segment_type == "corner" and seg.apex_dist is not None:
            apex_idx = int(np.argmin(np.abs(distance - seg.apex_dist)))
            ax.plot(x[apex_idx], y[apex_idx], "o", color="darkred", markersize=6, zorder=3)
            ax.annotate(
                seg.name,
                (x[apex_idx], y[apex_idx]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color="white",
                zorder=4,
            )

    # Gold stars on top-3 opportunity corners
    if opportunity_scores:
        top_ids = sorted(opportunity_scores, key=opportunity_scores.get, reverse=True)[:3]
        for seg in segments:
            if seg.segment_type == "corner" and seg.corner_id in top_ids and seg.apex_dist is not None:
                apex_idx = int(np.argmin(np.abs(distance - seg.apex_dist)))
                ax.plot(
                    x[apex_idx],
                    y[apex_idx],
                    marker="*",
                    color="#FFD700",
                    markersize=14,
                    zorder=5,
                    markeredgecolor="#B8860B",
                    markeredgewidth=0.5,
                )

    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("X (m)", fontsize=9, color="white")
    ax.set_ylabel("Y (m)", fontsize=9, color="white")

    fig.tight_layout()
