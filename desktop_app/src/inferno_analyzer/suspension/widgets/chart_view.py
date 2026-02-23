"""Chart view widget using matplotlib for embedded display."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.text import Annotation
import numpy as np

from motorsports_data_notebook.desktop.dpi import get_screen_dpi


class ChartView(ctk.CTkFrame):
    """Widget for displaying suspension velocity histograms embedded in the main window."""

    def __init__(
        self,
        parent: ctk.CTk | ctk.CTkFrame,
        on_maximize_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        """Initialize the chart view.

        Parameters
        ----------
        parent : ctk.CTk | ctk.CTkFrame
            Parent widget.
        on_maximize_toggle : Callable[[bool], None], optional
            Callback when maximize button is toggled. Receives True when
            maximizing, False when restoring.
        """
        super().__init__(parent)

        self._toolbar: NavigationToolbar2Tk | None = None
        self._on_maximize_toggle = on_maximize_toggle
        self._is_maximized = False

        # Get screen DPI for HiDPI support
        self._dpi = get_screen_dpi()

        # Hover tooltip support
        self._annotation: Annotation | None = None
        self._bar_data: list[tuple] = []  # List of (bar_container, bin_centers, histogram, ax)

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

        # Add toolbar frame with maximize button
        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(side="top", fill="x")

        # Maximize button on the right
        self._maximize_btn = ctk.CTkButton(
            toolbar_frame,
            text="⛶ Maximize",
            width=100,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._toggle_maximize,
        )
        self._maximize_btn.pack(side="right", padx=5, pady=2)

        # Matplotlib toolbar on the left
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        # Pack canvas
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Connect hover event
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)

        # Show placeholder
        self._show_placeholder()

    def _toggle_maximize(self) -> None:
        """Toggle maximize state."""
        self._is_maximized = not self._is_maximized
        if self._is_maximized:
            self._maximize_btn.configure(text="⛶ Restore")
        else:
            self._maximize_btn.configure(text="⛶ Maximize")

        if self._on_maximize_toggle:
            self._on_maximize_toggle(self._is_maximized)

    def _show_placeholder(self) -> None:
        """Show placeholder text."""
        self._figure.clear()
        self._bar_data = []
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5,
            0.5,
            "Load a session and select laps\nto view velocity histogram",
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self._canvas.draw()

    def _on_hover(self, event) -> None:
        """Handle mouse hover to show bar values."""
        if event.inaxes is None or not self._bar_data:
            if self._annotation and self._annotation.get_visible():
                self._annotation.set_visible(False)
                self._canvas.draw_idle()
            return

        # Find which bar we're hovering over
        for bar_container, bin_centers, histogram, ax, label in self._bar_data:
            if event.inaxes != ax:
                continue

            for i, bar in enumerate(bar_container):
                if bar.contains(event)[0]:
                    # Get bar info
                    velocity = bin_centers[i]
                    percentage = histogram[i]

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

                    # Update annotation (always set after the block above)
                    assert self._annotation is not None
                    text = f"{label}\nVelocity: {velocity:.0f} mm/s\nTime: {percentage:.2f}%"
                    self._annotation.set_text(text)
                    self._annotation.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())

                    # Move annotation to correct axes
                    if self._annotation.axes != ax:
                        self._annotation.remove()
                        self._annotation = ax.annotate(
                            text,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
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

        # No bar found, hide annotation
        if self._annotation and self._annotation.get_visible():
            self._annotation.set_visible(False)
            self._canvas.draw_idle()

    def load_html(self, html: str) -> None:
        """This method is called but we ignore HTML and wait for direct result."""
        pass

    def update_chart(self, result, title: str = "Suspension Velocity Distribution") -> None:
        """Update chart with velocity histogram result.

        Parameters
        ----------
        result : VelocityHistogramResult
            Analysis results to display.
        title : str
            Chart title.
        """
        self._figure.clear()
        self._bar_data = []
        self._annotation = None

        # Create 2x2 subplots
        axes = self._figure.subplots(2, 2)
        self._figure.suptitle(title, fontsize=12, color="white")

        corners = [
            (result.front_left, axes[0, 0], "Front Left (FL)"),
            (result.front_right, axes[0, 1], "Front Right (FR)"),
            (result.rear_left, axes[1, 0], "Rear Left (RL)"),
            (result.rear_right, axes[1, 1], "Rear Right (RR)"),
        ]

        ranges = result.velocity_ranges

        for corner_data, ax, corner_title in corners:
            ax.set_facecolor("#1a1a2e")
            ax.set_title(corner_title, fontsize=10, color="white")

            # Add velocity range background shading
            ax.axvspan(
                -ranges.friction, ranges.friction, alpha=0.15, color="gray", label="Friction"
            )
            ax.axvspan(ranges.friction, ranges.slow, alpha=0.2, color="lightblue")
            ax.axvspan(-ranges.slow, -ranges.friction, alpha=0.2, color="lightblue")
            ax.axvspan(ranges.slow, ranges.fast, alpha=0.2, color="lightgreen")
            ax.axvspan(-ranges.fast, -ranges.slow, alpha=0.2, color="lightgreen")
            ax.axvspan(ranges.fast, 300, alpha=0.2, color="lightcoral")
            ax.axvspan(-300, -ranges.fast, alpha=0.2, color="lightcoral")

            # Plot histogram bars
            colors = ["steelblue" if c >= 0 else "indianred" for c in corner_data.bin_centers]
            bars = ax.bar(
                corner_data.bin_centers,
                corner_data.histogram,
                width=np.diff(corner_data.bin_edges[:2])[0] * 0.9,
                color=colors,
                edgecolor="none",
            )

            # Store bar data for hover
            self._bar_data.append(
                (bars, corner_data.bin_centers, corner_data.histogram, ax, corner_title)
            )

            # Zero line
            ax.axvline(x=0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

            # Stats annotation
            stats_text = f"Skew: {corner_data.skew:.2f}\nStd: {corner_data.std:.0f} mm/s"
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

            ax.set_xlim(-300, 300)
            ax.tick_params(colors="white", labelsize=8)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add axis labels
        axes[1, 0].set_xlabel("Velocity (mm/s)", fontsize=9, color="white")
        axes[1, 1].set_xlabel("Velocity (mm/s)", fontsize=9, color="white")
        axes[0, 0].set_ylabel("Time (%)", fontsize=9, color="white")
        axes[1, 0].set_ylabel("Time (%)", fontsize=9, color="white")

        self._figure.tight_layout()
        self._canvas.draw()

    def update_comparison_chart(self, result_a, result_b, label_a: str, label_b: str) -> None:
        """Update chart with comparison of two sessions.

        Parameters
        ----------
        result_a : VelocityHistogramResult
            First session results.
        result_b : VelocityHistogramResult
            Second session results.
        label_a : str
            Label for first session.
        label_b : str
            Label for second session.
        """
        self._figure.clear()
        self._bar_data = []
        self._annotation = None

        axes = self._figure.subplots(2, 2)
        self._figure.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=12, color="white")

        corners = [
            (result_a.front_left, result_b.front_left, axes[0, 0], "Front Left (FL)"),
            (result_a.front_right, result_b.front_right, axes[0, 1], "Front Right (FR)"),
            (result_a.rear_left, result_b.rear_left, axes[1, 0], "Rear Left (RL)"),
            (result_a.rear_right, result_b.rear_right, axes[1, 1], "Rear Right (RR)"),
        ]

        ranges = result_a.velocity_ranges
        bar_width = np.diff(result_a.front_left.bin_edges[:2])[0] * 0.4

        for corner_a, corner_b, ax, corner_title in corners:
            ax.set_facecolor("#1a1a2e")
            ax.set_title(corner_title, fontsize=10, color="white")

            # Background shading
            ax.axvspan(-ranges.friction, ranges.friction, alpha=0.15, color="gray")
            ax.axvspan(ranges.friction, ranges.slow, alpha=0.2, color="lightblue")
            ax.axvspan(-ranges.slow, -ranges.friction, alpha=0.2, color="lightblue")
            ax.axvspan(ranges.slow, ranges.fast, alpha=0.2, color="lightgreen")
            ax.axvspan(-ranges.fast, -ranges.slow, alpha=0.2, color="lightgreen")
            ax.axvspan(ranges.fast, 300, alpha=0.2, color="lightcoral")
            ax.axvspan(-300, -ranges.fast, alpha=0.2, color="lightcoral")

            # Grouped bars
            bars_a = ax.bar(
                corner_a.bin_centers - bar_width / 2,
                corner_a.histogram,
                width=bar_width,
                color="steelblue",
                label=label_a,
                alpha=0.8,
            )
            bars_b = ax.bar(
                corner_b.bin_centers + bar_width / 2,
                corner_b.histogram,
                width=bar_width,
                color="darkorange",
                label=label_b,
                alpha=0.8,
            )

            # Store bar data for hover
            self._bar_data.append(
                (
                    bars_a,
                    corner_a.bin_centers,
                    corner_a.histogram,
                    ax,
                    f"{corner_title} - {label_a}",
                )
            )
            self._bar_data.append(
                (
                    bars_b,
                    corner_b.bin_centers,
                    corner_b.histogram,
                    ax,
                    f"{corner_title} - {label_b}",
                )
            )

            ax.axvline(x=0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_xlim(-300, 300)
            ax.tick_params(colors="white", labelsize=8)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Legend on first subplot
        axes[0, 0].legend(loc="upper left", fontsize=8)

        axes[1, 0].set_xlabel("Velocity (mm/s)", fontsize=9, color="white")
        axes[1, 1].set_xlabel("Velocity (mm/s)", fontsize=9, color="white")
        axes[0, 0].set_ylabel("Time (%)", fontsize=9, color="white")
        axes[1, 0].set_ylabel("Time (%)", fontsize=9, color="white")

        self._figure.tight_layout()
        self._canvas.draw()

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._show_placeholder()
