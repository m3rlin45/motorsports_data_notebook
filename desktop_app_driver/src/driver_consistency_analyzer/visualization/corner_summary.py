"""Summary view: per-corner box-and-whisker plots for throttle acceptance and braking points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from driver_consistency_analyzer.analysis.driver_consistency import DriverConsistencyResult


# Shared box plot style for dark theme
_BOX_STYLE = dict(
    showmeans=True,
    meanline=True,
    widths=0.5,
    patch_artist=True,
    meanprops=dict(color="white", linewidth=1.5, linestyle="--"),
    medianprops=dict(color="yellow", linewidth=1.5),
    whiskerprops=dict(color="white", linewidth=1),
    capprops=dict(color="white", linewidth=1),
    flierprops=dict(marker="o", markersize=4, markeredgecolor="white", alpha=0.7),
)


def _style_axis(ax) -> None:
    """Apply common dark-theme styling to an axis."""
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_summary(
    fig: Figure,
    result_a: DriverConsistencyResult,
    result_b: DriverConsistencyResult | None = None,
    label_a: str = "Session A",
    label_b: str = "Session B",
) -> None:
    """Draw summary box-and-whisker plots on the given figure.

    Two stacked subplots:
    1. Throttle Acceptance - distribution of TA% per corner
    2. Braking Points - distribution of braking distance per corner

    In comparison mode, paired boxes (Session A / Session B) per corner.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw on.
    result_a : DriverConsistencyResult
        Primary session results.
    result_b : DriverConsistencyResult, optional
        Comparison session results.
    label_a : str
        Label for primary session.
    label_b : str
        Label for comparison session.
    """
    fig.clear()

    corner_names = [f"{cd.corner.name} ({cd.corner.direction})" for cd in result_a.corner_data]

    if result_b is not None:
        _draw_comparison(fig, result_a, result_b, corner_names, label_a, label_b)
    else:
        _draw_single(fig, result_a, corner_names, label_a)

    fig.tight_layout()


def _draw_single(
    fig: Figure,
    result: DriverConsistencyResult,
    corner_names: list[str],
    label: str,
) -> None:
    """Draw single-session box plots."""
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"Driver Consistency: {label}", fontsize=12, color="white")

    positions = np.arange(1, len(corner_names) + 1)

    # Throttle Acceptance
    ta_data = [cd.ta_values if cd.ta_values else [] for cd in result.corner_data]

    _style_axis(ax1)
    if any(ta_data):
        bp1 = ax1.boxplot(
            ta_data,
            positions=positions,
            **_BOX_STYLE,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
    ax1.set_ylabel("Throttle Acceptance (%)", fontsize=10, color="white")
    ax1.set_title("Throttle Acceptance", fontsize=11, color="white")

    # Braking Points (centered on per-corner mean)
    bp_data = [
        [v - np.mean(cd.bp_values) for v in cd.bp_values] if cd.bp_values else []
        for cd in result.corner_data
    ]

    _style_axis(ax2)
    if any(bp_data):
        bp2 = ax2.boxplot(
            bp_data,
            positions=positions,
            **_BOX_STYLE,
        )
        for patch in bp2["boxes"]:
            patch.set_facecolor("darkorange")
            patch.set_alpha(0.7)
    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Braking Point Offset (m)", fontsize=10, color="white")
    ax2.set_title("Braking Point Consistency (centered on mean)", fontsize=11, color="white")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(corner_names, rotation=45, ha="right", fontsize=9)


def _draw_comparison(
    fig: Figure,
    result_a: DriverConsistencyResult,
    result_b: DriverConsistencyResult,
    corner_names: list[str],
    label_a: str,
    label_b: str,
) -> None:
    """Draw comparison box plots with paired boxes per corner."""
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=12, color="white")

    n = len(corner_names)
    # Position pairs: A slightly left, B slightly right of each tick
    offset = 0.2
    positions_a = np.arange(1, n + 1) - offset
    positions_b = np.arange(1, n + 1) + offset
    box_width = 0.3

    style_a = {**_BOX_STYLE, "widths": box_width}
    style_b = {**_BOX_STYLE, "widths": box_width}

    # Throttle Acceptance
    ta_data_a = [cd.ta_values if cd.ta_values else [] for cd in result_a.corner_data]
    ta_data_b = _get_matching_data(result_a, result_b, lambda cd: cd.ta_values)

    _style_axis(ax1)
    if any(ta_data_a):
        bp1a = ax1.boxplot(ta_data_a, positions=positions_a, **style_a)
        for patch in bp1a["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
    if any(ta_data_b):
        bp1b = ax1.boxplot(ta_data_b, positions=positions_b, **style_b)
        for patch in bp1b["boxes"]:
            patch.set_facecolor("darkorange")
            patch.set_alpha(0.7)

    ax1.set_ylabel("Throttle Acceptance (%)", fontsize=10, color="white")
    ax1.set_title("Throttle Acceptance", fontsize=11, color="white")
    # Legend via invisible patches
    from matplotlib.patches import Patch

    ax1.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.7, label=label_a),
            Patch(facecolor="darkorange", alpha=0.7, label=label_b),
        ],
        fontsize=9,
    )

    # Braking Points (centered on per-corner mean for each session)
    bp_data_a = [
        [v - np.mean(cd.bp_values) for v in cd.bp_values] if cd.bp_values else []
        for cd in result_a.corner_data
    ]
    bp_raw_b = _get_matching_data(result_a, result_b, lambda cd: cd.bp_values)
    bp_data_b = [[v - np.mean(vals) for v in vals] if vals else [] for vals in bp_raw_b]

    _style_axis(ax2)
    if any(bp_data_a):
        bp2a = ax2.boxplot(bp_data_a, positions=positions_a, **style_a)
        for patch in bp2a["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
    if any(bp_data_b):
        bp2b = ax2.boxplot(bp_data_b, positions=positions_b, **style_b)
        for patch in bp2b["boxes"]:
            patch.set_facecolor("darkorange")
            patch.set_alpha(0.7)

    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Braking Point Offset (m)", fontsize=10, color="white")
    ax2.set_title("Braking Point Consistency (centered on mean)", fontsize=11, color="white")
    ax2.set_xticks(np.arange(1, n + 1))
    ax2.set_xticklabels(corner_names, rotation=45, ha="right", fontsize=9)
    ax2.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.7, label=label_a),
            Patch(facecolor="darkorange", alpha=0.7, label=label_b),
        ],
        fontsize=9,
    )


def _get_matching_data(result_a, result_b, data_fn):
    """Get per-corner data lists from result_b matching result_a corner order."""
    data = []
    for cd_a in result_a.corner_data:
        found = False
        for cd_b in result_b.corner_data:
            if cd_b.corner.id == cd_a.corner.id:
                vals = data_fn(cd_b)
                data.append(vals if vals else [])
                found = True
                break
        if not found:
            data.append([])
    return data
