"""Chart view widget using matplotlib for embedded display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from motorsports_data_notebook.desktop.dpi import get_screen_dpi

if TYPE_CHECKING:
    from inferno_analyzer.driver.analysis.driver_consistency import DriverConsistencyResult


class ChartView(ctk.CTkFrame):
    """Widget for displaying driver consistency charts."""

    def __init__(
        self,
        parent: ctk.CTk | ctk.CTkFrame,
        on_maximize_toggle: Callable[[bool], None] | None = None,
        on_map_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        """Initialize the chart view.

        Parameters
        ----------
        parent : ctk.CTk | ctk.CTkFrame
            Parent widget.
        on_maximize_toggle : Callable[[bool], None], optional
            Callback when maximize button is toggled.
        on_map_toggle : Callable[[bool], None], optional
            Callback when map button is toggled.
        """
        super().__init__(parent)

        self._figure: Figure | None = None
        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._on_maximize_toggle = on_maximize_toggle
        self._on_map_toggle = on_map_toggle
        self._is_maximized = False
        self._is_map = False
        self._dpi = get_screen_dpi()

        # Overlay state (drawn on tkinter canvas, not matplotlib)
        self._crosshair_axes: list[Axes] = []
        self._tooltip_axes: list[Axes] = []
        self._tk_items: list[int] = []
        self._hover_bound = False

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the matplotlib canvas."""
        plt.style.use("dark_background")

        self._figure = Figure(figsize=(10, 7), dpi=self._dpi)
        self._figure.set_facecolor("#1a1a2e")

        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.draw()

        # Toolbar frame with maximize button
        toolbar_frame = ctk.CTkFrame(self)
        toolbar_frame.pack(side="top", fill="x")

        self._maximize_btn = ctk.CTkButton(
            toolbar_frame,
            text="Maximize",
            width=100,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._toggle_maximize,
        )
        self._maximize_btn.pack(side="right", padx=5, pady=2)

        self._map_btn = ctk.CTkButton(
            toolbar_frame,
            text="Track Map",
            width=100,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._toggle_map,
        )
        self._map_btn.pack(side="right", padx=5, pady=2)

        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self._show_placeholder()

    def _toggle_maximize(self) -> None:
        """Toggle maximize state."""
        self._is_maximized = not self._is_maximized
        if self._is_maximized:
            self._maximize_btn.configure(text="Restore")
        else:
            self._maximize_btn.configure(text="Maximize")

        if self._on_maximize_toggle:
            self._on_maximize_toggle(self._is_maximized)

    def _toggle_map(self) -> None:
        """Toggle track map view."""
        self._is_map = not self._is_map
        if self._is_map:
            self._map_btn.configure(text="Charts")
        else:
            self._map_btn.configure(text="Track Map")

        if self._on_map_toggle:
            self._on_map_toggle(self._is_map)

    @property
    def is_map(self) -> bool:
        """Whether track map mode is active."""
        return self._is_map

    def _show_placeholder(self) -> None:
        """Show placeholder text."""
        assert self._figure is not None
        assert self._canvas is not None
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5,
            0.5,
            "Load a session and select laps\nto view driver consistency analysis",
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
            transform=ax.transAxes,
        )
        ax.axis("off")
        self._canvas.draw()

    def get_figure(self) -> Figure:
        """Get the matplotlib figure for external drawing."""
        assert self._figure is not None
        return self._figure

    def redraw(self) -> None:
        """Redraw the canvas after figure updates."""
        assert self._canvas is not None
        self._canvas.draw()
        self._setup_hover()

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._teardown_hover()
        self._show_placeholder()

    def _setup_hover(self) -> None:
        """Set up hover overlays based on figure layout.

        - 3 axes (detail view): crosshair cursor with distance label
        - 2 axes (summary view): tooltip with box plot stats

        Draws directly on the tkinter canvas to avoid interfering
        with matplotlib layout and rendering.
        """
        self._teardown_hover()
        assert self._figure is not None
        assert self._canvas is not None

        axes = self._figure.get_axes()
        if len(axes) == 3:
            self._crosshair_axes = list(axes)
        elif len(axes) == 2:
            # Summary box plot — check for tooltip data
            if any(hasattr(ax, "_tooltip_data") for ax in axes):
                self._tooltip_axes = list(axes)
        else:
            return

        tk_widget = self._canvas.get_tk_widget()
        tk_widget.bind("<Motion>", self._on_tk_motion, add="+")
        tk_widget.bind("<Leave>", self._on_tk_leave, add="+")
        self._hover_bound = True

    def _teardown_hover(self) -> None:
        """Remove hover overlays and disconnect events."""
        if self._hover_bound and self._canvas is not None:
            tk_widget = self._canvas.get_tk_widget()
            tk_widget.unbind("<Motion>")
            tk_widget.unbind("<Leave>")
            self._hover_bound = False

        self._clear_tk_items()
        self._crosshair_axes.clear()
        self._tooltip_axes.clear()

    def _clear_tk_items(self) -> None:
        """Delete all overlay items from the tkinter canvas."""
        if self._canvas is None or not self._tk_items:
            return
        tk_widget = self._canvas.get_tk_widget()
        for item_id in self._tk_items:
            tk_widget.delete(item_id)
        self._tk_items.clear()

    def _on_tk_motion(self, event) -> None:
        """Handle mouse motion on tk canvas."""
        self._clear_tk_items()

        if self._crosshair_axes:
            self._draw_crosshair(event)
        elif self._tooltip_axes:
            self._draw_tooltip(event)

    def _draw_crosshair(self, event) -> None:
        """Draw crosshair cursor for detail view (3-axes)."""
        if self._figure is None or self._canvas is None:
            return

        tk_widget = self._canvas.get_tk_widget()

        # Convert tk coords to matplotlib display coords
        # tk: y=0 at top; matplotlib: y=0 at bottom
        fig_h = self._figure.bbox.height
        display_x = event.x
        display_y = fig_h - event.y

        # Find which axes the mouse is in
        target_ax = None
        for ax in self._crosshair_axes:
            if ax.bbox.contains(display_x, display_y):
                target_ax = ax
                break

        if target_ax is None:
            return

        # Get data x coordinate
        data_x, _ = target_ax.transData.inverted().transform((display_x, display_y))

        # Check x is within axes limits
        xlim = target_ax.get_xlim()
        if data_x < xlim[0] or data_x > xlim[1]:
            return

        # Draw vertical line across each axes
        for ax in self._crosshair_axes:
            pixel_x = ax.transData.transform((data_x, 0))[0]
            y_top = fig_h - ax.bbox.y1
            y_bot = fig_h - ax.bbox.y0

            item = tk_widget.create_line(
                pixel_x,
                y_top,
                pixel_x,
                y_bot,
                fill="white",
                width=1,
                dash=(4, 2),
            )
            self._tk_items.append(item)

        # Distance label above the top axes
        top_ax = self._crosshair_axes[0]
        label_x = top_ax.transData.transform((data_x, 0))[0]
        label_y = fig_h - top_ax.bbox.y1 - 2

        item = tk_widget.create_text(
            label_x,
            label_y,
            text=f"{data_x:.1f} m",
            fill="white",
            font=("TkDefaultFont", max(9, int(9 * self._dpi / 96))),
            anchor="s",
        )
        self._tk_items.append(item)

    def _draw_tooltip(self, event) -> None:
        """Draw tooltip for summary box plot view (2-axes)."""
        if self._figure is None or self._canvas is None:
            return

        tk_widget = self._canvas.get_tk_widget()
        fig_h = self._figure.bbox.height
        display_x = event.x
        display_y = fig_h - event.y

        # Find which axes the mouse is in
        target_ax = None
        for ax in self._tooltip_axes:
            if ax.bbox.contains(display_x, display_y):
                target_ax = ax
                break

        if target_ax is None:
            return

        tooltip_data: dict[float, str] = getattr(target_ax, "_tooltip_data", {})
        if not tooltip_data:
            return

        # Get data x coordinate
        data_x, _ = target_ax.transData.inverted().transform((display_x, display_y))

        # Find nearest box position within snap distance
        snap_threshold = 0.4  # data units
        best_pos = None
        best_dist = float("inf")
        for pos in tooltip_data:
            d = abs(data_x - pos)
            if d < best_dist:
                best_dist = d
                best_pos = pos

        if best_pos is None or best_dist > snap_threshold:
            return

        text = tooltip_data[best_pos]
        font_size = max(10, int(10 * self._dpi / 96))

        # Position tooltip near the mouse, offset to the right
        tip_x = event.x + 15
        tip_y = event.y - 10

        # Draw background rectangle and text
        item_text = tk_widget.create_text(
            tip_x,
            tip_y,
            text=text,
            fill="white",
            font=("TkDefaultFont", font_size),
            anchor="nw",
            justify="left",
        )
        self._tk_items.append(item_text)

        # Get text bounding box and draw background behind it
        bbox = tk_widget.bbox(item_text)
        if bbox:
            pad = 4
            item_bg = tk_widget.create_rectangle(
                bbox[0] - pad,
                bbox[1] - pad,
                bbox[2] + pad,
                bbox[3] + pad,
                fill="#2b2b2b",
                outline="#555555",
            )
            # Move background behind text
            tk_widget.tag_lower(item_bg, item_text)
            self._tk_items.append(item_bg)

    def _on_tk_leave(self, event) -> None:
        """Handle mouse leaving the canvas."""
        self._clear_tk_items()
