"""Chart view for roll gradient: track map + three linked scatter plots.

Panels: track map (top-left), front roll vs lateral G (top-right), rear roll
vs lateral G (bottom-left) and front roll vs rear roll whose slope is the
roll ratio (bottom-right). Hovering over the track map highlights the samples
recorded near the cursor on all three scatter plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from motorsports_data_notebook.desktop.dpi import get_screen_dpi

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from motorsports_data_notebook.roll_gradient import AxleRollData, RollGradientResult

_MAX_SCATTER_POINTS = 12000
_COLOR_A = "steelblue"
_COLOR_B = "darkorange"
_HIGHLIGHT_A = "#ffe066"
_HIGHLIGHT_B = "#ff6ec7"
_BG = "#1a1a2e"


@dataclass
class _Series:
    """One session's downsampled samples and the artists that highlight them."""

    label: str
    lat: np.ndarray
    roll_f: np.ndarray
    roll_r: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    highlights: list[tuple[PathCollection, np.ndarray, np.ndarray]] = field(default_factory=list)
    map_highlight: PathCollection | None = None
    snap_marker: PathCollection | None = None


class ChartView(ctk.CTkFrame):
    """Roll gradient scatter plots with a hover-linked track map."""

    def __init__(
        self,
        parent: ctk.CTk | ctk.CTkFrame,
        on_min_g_changed: Callable[[], None] | None = None,
        on_stats_click: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the chart view.

        Parameters
        ----------
        parent : ctk.CTk | ctk.CTkFrame
            Parent widget.
        on_min_g_changed : Callable[[], None], optional
            Callback when the minimum |lateral G| entry changes.
        on_stats_click : Callable[[], None], optional
            Callback when the statistics button is clicked.
        """
        super().__init__(parent)
        self._toolbar: NavigationToolbar2Tk | None = None
        self._on_min_g_changed = on_min_g_changed
        self._on_stats_click = on_stats_click
        self._dpi = get_screen_dpi()
        self._series: list[_Series] = []
        self._ax_map: Axes | None = None
        self._hover_radius_m = 10.0  # samples within this distance of the snap point light up
        self._snap_radius_m = 1e9  # how far from the track line the cursor may be to snap
        self._highlight_active = False
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the matplotlib canvas and toolbar row."""
        plt.style.use("dark_background")
        self._figure = Figure(figsize=(11, 7), dpi=self._dpi)
        self._figure.set_facecolor(_BG)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.draw()
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("figure_leave_event", lambda e: self._clear_highlight())

        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(side="top", fill="x")

        self._stats_btn = ctk.CTkButton(
            toolbar_frame,
            text="Statistics",
            width=90,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._on_stats_btn_click,
        )
        self._stats_btn.pack(side="right", padx=5, pady=2)

        # Minimum |lateral G| for the fits (drops the straight-line cluster)
        self._min_g_entry = ctk.CTkEntry(
            toolbar_frame, width=50, height=24, font=ctk.CTkFont(size=11)
        )
        self._min_g_entry.insert(0, "0.0")
        self._min_g_entry.pack(side="right", padx=(0, 5), pady=2)
        self._min_g_entry.bind("<Return>", lambda e: self._fire_min_g_changed())
        self._min_g_entry.bind("<FocusOut>", lambda e: self._fire_min_g_changed())
        ctk.CTkLabel(
            toolbar_frame, text="Min |G| for fit:", font=ctk.CTkFont(size=11), height=24
        ).pack(side="right", pady=2)

        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Toolbar accessors
    # ------------------------------------------------------------------

    def get_min_lateral_g(self) -> float:
        """Minimum |lateral G| used for the fits (0 if the entry is invalid)."""
        try:
            return max(0.0, float(self._min_g_entry.get()))
        except ValueError:
            return 0.0

    def _fire_min_g_changed(self) -> None:
        if self._on_min_g_changed:
            self._on_min_g_changed()

    def _on_stats_btn_click(self) -> None:
        if self._on_stats_click:
            self._on_stats_click()

    def set_stats_button_text(self, text: str) -> None:
        """Update the statistics button label."""
        self._stats_btn.configure(text=text)

    # ------------------------------------------------------------------
    # Hover linking
    # ------------------------------------------------------------------

    def _on_motion(self, event) -> None:
        """Snap the cursor to the nearest track point and highlight samples around it."""
        if (
            self._ax_map is None
            or event.inaxes is not self._ax_map
            or event.xdata is None
            or event.ydata is None
        ):
            self._clear_highlight()
            return

        # Snap point: nearest track sample (any series) to the cursor
        best: tuple[float, float] | None = None
        best_d2 = self._snap_radius_m**2
        for series in self._series:
            if series.x is None or series.y is None or len(series.x) == 0:
                continue
            d2 = (series.x - event.xdata) ** 2 + (series.y - event.ydata) ** 2
            i = int(np.argmin(d2))
            if d2[i] <= best_d2:
                best_d2 = float(d2[i])
                best = (float(series.x[i]), float(series.y[i]))
        if best is None:
            self._clear_highlight()
            return

        sx, sy = best
        any_points = False
        for series in self._series:
            if series.x is None or series.y is None:
                continue
            d2 = (series.x - sx) ** 2 + (series.y - sy) ** 2
            mask = d2 <= self._hover_radius_m**2
            if mask.any():
                any_points = True
            for artist, xs, ys in series.highlights:
                artist.set_offsets(np.column_stack([xs[mask], ys[mask]]))
            if series.map_highlight is not None:
                series.map_highlight.set_offsets(np.column_stack([series.x[mask], series.y[mask]]))
            if series.snap_marker is not None:
                series.snap_marker.set_offsets(np.array([[sx, sy]]))
        self._highlight_active = any_points or self._highlight_active
        self._canvas.draw_idle()

    def _clear_highlight(self) -> None:
        if not self._highlight_active:
            return
        empty = np.empty((0, 2))
        for series in self._series:
            for artist, _, _ in series.highlights:
                artist.set_offsets(empty)
            if series.map_highlight is not None:
                series.map_highlight.set_offsets(empty)
            if series.snap_marker is not None:
                series.snap_marker.set_offsets(empty)
        self._highlight_active = False
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Show placeholder text."""
        self._figure.clear()
        self._series = []
        self._ax_map = None
        ax = self._figure.add_subplot(111)
        ax.set_facecolor(_BG)
        ax.text(
            0.5,
            0.5,
            "Load a session, select laps and enter track widths\nto view roll gradient",
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self._canvas.draw()

    @staticmethod
    def _style_axis(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(_BG)
        ax.set_title(title, fontsize=10, color="white")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=9, color="white")
        ax.set_ylabel(ylabel, fontsize=9, color="white")

    @staticmethod
    def _stats_box(ax: Axes, text: str) -> None:
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#333333", alpha=0.85),
        )

    @staticmethod
    def _line(ax: Axes, x: np.ndarray, slope: float, intercept: float, color: str, label=None):
        if len(x) == 0 or not np.isfinite(slope):
            return
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2, label=label)

    @staticmethod
    def _binned(ax: Axes, centers: np.ndarray, means: np.ndarray, color: str) -> None:
        if len(centers) == 0:
            return
        ax.plot(
            centers,
            means,
            "o",
            color=color,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=4,
        )

    def _scatter_panel(
        self,
        ax: Axes,
        series: _Series,
        xs: np.ndarray,
        ys: np.ndarray,
        color: str,
        highlight_color: str,
        label: str | None = None,
    ) -> None:
        """Draw a downsampled scatter plus an (initially empty) highlight layer."""
        ax.scatter(xs, ys, s=3, color=color, alpha=0.15, linewidths=0, rasterized=True, label=label)
        hl = ax.scatter(
            [], [], s=14, color=highlight_color, edgecolors="black", linewidths=0.4, zorder=6
        )
        series.highlights.append((hl, xs, ys))

    @staticmethod
    def _downsample(result: RollGradientResult, label: str) -> _Series:
        n = len(result.front.lateral_g)
        step = max(1, n // _MAX_SCATTER_POINTS)
        sl = slice(None, None, step)
        x = y = None
        if result.x_m is not None and result.y_m is not None and len(result.x_m) > 0:
            x, y = result.x_m[sl], result.y_m[sl]
        return _Series(
            label=label,
            lat=result.front.lateral_g[sl],
            roll_f=result.front.roll_deg[sl],
            roll_r=result.rear.roll_deg[sl],
            x=x,
            y=y,
        )

    def _draw_map(self, ax: Axes, series_list: list[_Series], highlight_colors: list[str]) -> None:
        ax.set_facecolor(_BG)
        ax.set_title(
            "Track map (hover near the line to highlight samples)", fontsize=10, color="white"
        )
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        any_map = False
        for series, hl_color, color in zip(series_list, highlight_colors, (_COLOR_A, _COLOR_B)):
            if series.x is None or series.y is None:
                continue
            any_map = True
            ax.plot(series.x, series.y, color=color, linewidth=1.0, alpha=0.7, label=series.label)
            series.map_highlight = ax.scatter(
                [], [], s=18, color=hl_color, edgecolors="black", linewidths=0.4, zorder=6
            )
            # Snap marker: white ring on the track at the point nearest the cursor
            series.snap_marker = ax.scatter(
                [], [], s=90, facecolors="none", edgecolors="white", linewidths=1.5, zorder=7
            )
        if not any_map:
            ax.text(
                0.5,
                0.5,
                "No GPS channels for track map",
                ha="center",
                va="center",
                fontsize=10,
                color="#888888",
                transform=ax.transAxes,
            )
            self._ax_map = None
            return
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)", fontsize=8, color="white")
        ax.set_ylabel("y (m)", fontsize=8, color="white")
        # Highlight radius around the snap point: ~1.5 % of the track extent, at least 5 m.
        # The cursor may be anywhere within ~15 % of the extent of the track line to snap.
        xs = np.concatenate([s.x for s in series_list if s.x is not None])
        ys = np.concatenate([s.y for s in series_list if s.y is not None])
        extent = max(np.ptp(xs), np.ptp(ys)) if len(xs) else 0.0
        self._hover_radius_m = max(5.0, 0.015 * extent)
        self._snap_radius_m = max(50.0, 0.15 * extent)
        self._ax_map = ax

    def _render(self, results: list[tuple[RollGradientResult, str]], title: str) -> None:
        self._figure.clear()
        self._series = []
        self._highlight_active = False
        axes = self._figure.subplots(2, 2)
        ax_map, ax_front, ax_rear, ax_ratio = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        self._figure.suptitle(title, fontsize=11, color="white")

        colors = [_COLOR_A, _COLOR_B]
        highlights = [_HIGHLIGHT_A, _HIGHLIGHT_B]
        self._style_axis(ax_front, "Front roll vs lateral G", "Lateral G (g)", "Front roll (deg)")
        self._style_axis(ax_rear, "Rear roll vs lateral G", "Lateral G (g)", "Rear roll (deg)")
        self._style_axis(
            ax_ratio, "Roll ratio: front roll vs rear roll", "Rear roll (deg)", "Front roll (deg)"
        )

        for i, (result, label) in enumerate(results):
            series = self._downsample(result, label)
            self._series.append(series)
            color, hl = colors[i % 2], highlights[i % 2]
            lbl = label if len(results) > 1 else None
            self._scatter_panel(ax_front, series, series.lat, series.roll_f, color, hl, lbl)
            self._scatter_panel(ax_rear, series, series.lat, series.roll_r, color, hl)
            self._scatter_panel(ax_ratio, series, series.roll_r, series.roll_f, color, hl)
            bin_color = "white" if len(results) == 1 else color
            self._binned(ax_front, result.front.bin_centers_g, result.front.bin_mean_deg, bin_color)
            self._binned(ax_rear, result.rear.bin_centers_g, result.rear.bin_mean_deg, bin_color)
            self._binned(
                ax_ratio, result.ratio_bin_centers_deg, result.ratio_bin_mean_deg, bin_color
            )
            f, r = result.front, result.rear
            self._line(ax_front, series.lat, f.gradient_deg_per_g, f.intercept_deg, color)
            self._line(ax_rear, series.lat, r.gradient_deg_per_g, r.intercept_deg, color)
            self._line(
                ax_ratio, series.roll_r, result.roll_ratio, result.roll_ratio_intercept_deg, color
            )

        if len(results) == 1:
            result = results[0][0]
            f, r = result.front, result.rear
            self._stats_box(
                ax_front,
                f"gradient = {f.gradient_deg_per_g:+.3f} deg/g\n"
                f"R² = {f.r2:.3f}   n = {f.n_samples}",
            )
            self._stats_box(
                ax_rear,
                f"gradient = {r.gradient_deg_per_g:+.3f} deg/g\n"
                f"R² = {r.r2:.3f}   n = {r.n_samples}",
            )
            self._stats_box(
                ax_ratio,
                f"roll ratio F/R = {result.roll_ratio:.3f}\n" f"R² = {result.roll_ratio_r2:.3f}",
            )
            if not result.signs_consistent:
                ax_rear.text(
                    0.97,
                    0.03,
                    "front/rear roll in opposite directions:\ncheck left/right channel assignment",
                    transform=ax_rear.transAxes,
                    fontsize=8,
                    color="#ff8080",
                    ha="right",
                    va="bottom",
                )
        else:
            (ra, la), (rb, lb) = results[0], results[1]
            self._stats_box(
                ax_front,
                f"A {ra.front.gradient_deg_per_g:+.3f}  B {rb.front.gradient_deg_per_g:+.3f} deg/g\n"
                f"Δ|grad| = {rb.front.abs_gradient_deg_per_g - ra.front.abs_gradient_deg_per_g:+.3f}",
            )
            self._stats_box(
                ax_rear,
                f"A {ra.rear.gradient_deg_per_g:+.3f}  B {rb.rear.gradient_deg_per_g:+.3f} deg/g\n"
                f"Δ|grad| = {rb.rear.abs_gradient_deg_per_g - ra.rear.abs_gradient_deg_per_g:+.3f}",
            )
            self._stats_box(
                ax_ratio,
                f"roll ratio A {ra.roll_ratio:.3f}  B {rb.roll_ratio:.3f}\n"
                f"Δ = {rb.roll_ratio - ra.roll_ratio:+.3f}",
            )
            ax_front.legend(loc="lower right", fontsize=8, markerscale=4)

        self._draw_map(ax_map, self._series, highlights)
        self._figure.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_chart(self, result: RollGradientResult, title: str = "Roll Gradient") -> None:
        """Draw a single session.

        Parameters
        ----------
        result : RollGradientResult
            Analysis results to display.
        title : str
            Chart title.
        """
        full_title = (
            f"{title}  |  front {result.front.abs_gradient_deg_per_g:.2f} deg/g, "
            f"rear {result.rear.abs_gradient_deg_per_g:.2f} deg/g, "
            f"roll ratio {result.roll_ratio:.2f}"
        )
        self._render([(result, "Session A")], full_title)

    def update_comparison_chart(
        self,
        result_a: RollGradientResult,
        result_b: RollGradientResult,
        label_a: str,
        label_b: str,
    ) -> None:
        """Overlay two sessions' scatters, binned means and fit lines.

        Parameters
        ----------
        result_a, result_b : RollGradientResult
            Session results.
        label_a, label_b : str
            Legend labels.
        """
        self._render(
            [(result_a, label_a), (result_b, label_b)], f"Roll Gradient: {label_a} vs {label_b}"
        )

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._show_placeholder()
