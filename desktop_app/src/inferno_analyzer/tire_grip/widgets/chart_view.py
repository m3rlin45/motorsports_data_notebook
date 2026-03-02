"""Chart view widget using matplotlib for tire grip scatter plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from motorsports_data_notebook.desktop.dpi import get_screen_dpi

if TYPE_CHECKING:
    from motorsports_data_notebook.tire_grip import TireGripResult


class ChartView(ctk.CTkFrame):
    """Widget for displaying tire grip scatter plots embedded in the main window."""

    def __init__(
        self,
        parent: ctk.CTk | ctk.CTkFrame,
        on_metric_changed: Callable[[str], None] | None = None,
        on_percentile_changed: Callable[[], None] | None = None,
        on_stats_click: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the chart view.

        Parameters
        ----------
        parent : ctk.CTk | ctk.CTkFrame
            Parent widget.
        on_metric_changed : Callable[[str], None], optional
            Callback when metric mode toggle changes. Receives "pressure"
            or "temperature".
        on_percentile_changed : Callable[[], None], optional
            Callback when percentile value changes.
        on_stats_click : Callable[[], None], optional
            Callback when statistics button is clicked.
        """
        super().__init__(parent)

        self._toolbar: NavigationToolbar2Tk | None = None
        self._on_metric_changed = on_metric_changed
        self._on_percentile_changed = on_percentile_changed
        self._on_stats_click = on_stats_click

        # Get screen DPI for HiDPI support
        self._dpi = get_screen_dpi()

        # Hover tooltip support
        self._annotation: Annotation | None = None
        self._line_data: list[tuple] = []  # (line, x_data, y_data, counts, ax, label)

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the matplotlib canvas."""
        # Set dark style
        plt.style.use("dark_background")

        # Create figure with HiDPI-aware DPI
        self._figure = Figure(figsize=(10, 7), dpi=self._dpi)
        self._figure.set_facecolor("#1a1a2e")

        # Create canvas
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.draw()

        # Toolbar frame
        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(side="top", fill="x")

        # Statistics button
        self._stats_btn = ctk.CTkButton(
            toolbar_frame,
            text="Statistics",
            width=90,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._on_stats_btn_click,
        )
        self._stats_btn.pack(side="right", padx=5, pady=2)

        # Percentile entry
        self._percentile_entry = ctk.CTkEntry(
            toolbar_frame,
            width=50,
            height=24,
            font=ctk.CTkFont(size=11),
        )
        self._percentile_entry.insert(0, "99.9")
        self._percentile_entry.pack(side="right", padx=(0, 5), pady=2)
        self._percentile_entry.bind("<Return>", lambda e: self._fire_percentile_changed())
        self._percentile_entry.bind("<FocusOut>", lambda e: self._fire_percentile_changed())

        percentile_label = ctk.CTkLabel(
            toolbar_frame,
            text="P:",
            font=ctk.CTkFont(size=11),
            height=24,
        )
        percentile_label.pack(side="right", pady=2)

        # Pressure/Temperature toggle
        self._metric_btn = ctk.CTkSegmentedButton(
            toolbar_frame,
            values=["Pressure", "Temperature"],
            command=self._on_metric_btn_changed,
            font=ctk.CTkFont(size=11),
            height=24,
        )
        self._metric_btn.set("Pressure")
        self._metric_btn.pack(side="right", padx=5, pady=2)

        # Matplotlib toolbar on the left
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        # Pack canvas
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Connect hover event
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)

        # Show placeholder
        self._show_placeholder()

    def get_metric_mode(self) -> str:
        """Return the current metric mode ('pressure' or 'temperature')."""
        return str(self._metric_btn.get()).lower()

    def get_percentile(self) -> float:
        """Return the current percentile value."""
        try:
            return float(self._percentile_entry.get())
        except ValueError:
            return 99.9

    def _fire_percentile_changed(self) -> None:
        """Validate and fire percentile changed callback."""
        if self._on_percentile_changed:
            self._on_percentile_changed()

    def _on_metric_btn_changed(self, value: str) -> None:
        """Handle metric mode toggle."""
        if self._on_metric_changed:
            self._on_metric_changed(value.lower())

    def _on_stats_btn_click(self) -> None:
        """Handle statistics button click."""
        if self._on_stats_click:
            self._on_stats_click()

    def set_stats_button_text(self, text: str) -> None:
        """Update the statistics button text."""
        self._stats_btn.configure(text=text)

    def _show_placeholder(self) -> None:
        """Show placeholder text."""
        self._figure.clear()
        self._line_data = []
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5,
            0.5,
            "Load a session and select laps\nto view tire grip scatter",
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self._canvas.draw()

    def _on_hover(self, event) -> None:
        """Handle mouse hover to show line point values."""
        if event.inaxes is None or not self._line_data:
            if self._annotation and self._annotation.get_visible():
                self._annotation.set_visible(False)
                self._canvas.draw_idle()
            return

        import numpy as np

        # Find closest point on any line in the active axes
        for line, x_data, y_data, counts, ax, label in self._line_data:
            if event.inaxes != ax or len(x_data) == 0:
                continue

            contains, info = line.contains(event)
            if contains and info["ind"] is not None and len(info["ind"]) > 0:
                idx = info["ind"][0]
                x_val = x_data[idx]
                y_val = y_data[idx]
                count = counts[idx] if idx < len(counts) else 0

                # Create or update annotation
                if self._annotation is None:
                    self._annotation = ax.annotate(
                        "",
                        xy=(0, 0),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="#333333",
                            edgecolor="white",
                            alpha=0.9,
                        ),
                        fontsize=9,
                        color="white",
                        zorder=100,
                    )

                assert self._annotation is not None
                text = f"{label}\nMetric: {x_val:.1f}\nG: {y_val:.2f}\nn = {count}"
                self._annotation.set_text(text)
                self._annotation.xy = (x_val, y_val)

                # Move annotation to correct axes
                if self._annotation.axes != ax:
                    self._annotation.remove()
                    self._annotation = ax.annotate(
                        text,
                        xy=(x_val, y_val),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="#333333",
                            edgecolor="white",
                            alpha=0.9,
                        ),
                        fontsize=9,
                        color="white",
                        zorder=100,
                    )

                self._annotation.set_visible(True)
                self._canvas.draw_idle()
                return

        # No point found, hide annotation
        if self._annotation and self._annotation.get_visible():
            self._annotation.set_visible(False)
            self._canvas.draw_idle()

    def update_chart(self, result: TireGripResult, title: str = "Tire Grip Analysis") -> None:
        """Update chart with tire grip bucketed percentile line plots.

        Parameters
        ----------
        result : TireGripResult
            Analysis results to display.
        title : str
            Chart title.
        """
        self._figure.clear()
        self._line_data = []
        self._annotation = None

        # Create 2x2 subplots
        axes = self._figure.subplots(2, 2)
        self._figure.suptitle(title, fontsize=12, color="white")

        metric_label = "Pressure" if result.metric_mode == "pressure" else "Temperature"
        x_label = f"{metric_label} ({result.metric_unit})"

        corners = [
            (result.front_left, axes[0, 0], "Front Left (FL)"),
            (result.front_right, axes[0, 1], "Front Right (FR)"),
            (result.rear_left, axes[1, 0], "Rear Left (RL)"),
            (result.rear_right, axes[1, 1], "Rear Right (RR)"),
        ]

        for corner_data, ax, corner_title in corners:
            ax.set_facecolor("#1a1a2e")
            ax.set_title(corner_title, fontsize=10, color="white")

            # Line plot with markers
            (line,) = ax.plot(
                corner_data.bucket_centers,
                corner_data.bucket_values,
                "-o",
                color="steelblue",
                markersize=6,
                linewidth=2,
                picker=5,
            )

            # Store line data for hover
            self._line_data.append(
                (
                    line,
                    corner_data.bucket_centers,
                    corner_data.bucket_values,
                    corner_data.bucket_counts,
                    ax,
                    corner_title,
                )
            )

            # Stats annotation
            total_n = (
                int(corner_data.bucket_counts.sum()) if len(corner_data.bucket_counts) > 0 else 0
            )
            stats_text = f"n = {total_n}"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="#333333", alpha=0.8),
            )

            ax.tick_params(colors="white", labelsize=8)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add axis labels
        axes[1, 0].set_xlabel(x_label, fontsize=9, color="white")
        axes[1, 1].set_xlabel(x_label, fontsize=9, color="white")
        axes[0, 0].set_ylabel("Total G", fontsize=9, color="white")
        axes[1, 0].set_ylabel("Total G", fontsize=9, color="white")

        self._figure.tight_layout()
        self._canvas.draw()

    def update_comparison_chart(
        self,
        result_a: TireGripResult,
        result_b: TireGripResult,
        label_a: str,
        label_b: str,
    ) -> None:
        """Update chart with comparison of two sessions.

        Parameters
        ----------
        result_a : TireGripResult
            First session results.
        result_b : TireGripResult
            Second session results.
        label_a : str
            Label for first session.
        label_b : str
            Label for second session.
        """
        self._figure.clear()
        self._line_data = []
        self._annotation = None

        axes = self._figure.subplots(2, 2)
        self._figure.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=12, color="white")

        metric_label = "Pressure" if result_a.metric_mode == "pressure" else "Temperature"
        x_label = f"{metric_label} ({result_a.metric_unit})"

        corners = [
            (result_a.front_left, result_b.front_left, axes[0, 0], "Front Left (FL)"),
            (result_a.front_right, result_b.front_right, axes[0, 1], "Front Right (FR)"),
            (result_a.rear_left, result_b.rear_left, axes[1, 0], "Rear Left (RL)"),
            (result_a.rear_right, result_b.rear_right, axes[1, 1], "Rear Right (RR)"),
        ]

        for corner_a, corner_b, ax, corner_title in corners:
            ax.set_facecolor("#1a1a2e")
            ax.set_title(corner_title, fontsize=10, color="white")

            # Line A
            (line_a,) = ax.plot(
                corner_a.bucket_centers,
                corner_a.bucket_values,
                "-o",
                color="steelblue",
                markersize=6,
                linewidth=2,
                label=label_a,
                picker=5,
            )

            # Line B
            (line_b,) = ax.plot(
                corner_b.bucket_centers,
                corner_b.bucket_values,
                "-o",
                color="darkorange",
                markersize=6,
                linewidth=2,
                label=label_b,
                picker=5,
            )

            # Store line data for hover
            self._line_data.append(
                (
                    line_a,
                    corner_a.bucket_centers,
                    corner_a.bucket_values,
                    corner_a.bucket_counts,
                    ax,
                    f"{corner_title} - {label_a}",
                )
            )
            self._line_data.append(
                (
                    line_b,
                    corner_b.bucket_centers,
                    corner_b.bucket_values,
                    corner_b.bucket_counts,
                    ax,
                    f"{corner_title} - {label_b}",
                )
            )

            ax.tick_params(colors="white", labelsize=8)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Legend on first subplot
        axes[0, 0].legend(loc="upper left", fontsize=8)

        axes[1, 0].set_xlabel(x_label, fontsize=9, color="white")
        axes[1, 1].set_xlabel(x_label, fontsize=9, color="white")
        axes[0, 0].set_ylabel("Total G", fontsize=9, color="white")
        axes[1, 0].set_ylabel("Total G", fontsize=9, color="white")

        self._figure.tight_layout()
        self._canvas.draw()

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._show_placeholder()
