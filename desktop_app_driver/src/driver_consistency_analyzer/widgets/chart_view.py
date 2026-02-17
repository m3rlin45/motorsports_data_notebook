"""Chart view widget using matplotlib for embedded display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from driver_consistency_analyzer.analysis.driver_consistency import DriverConsistencyResult


def get_screen_dpi() -> int:
    """Get the screen DPI for HiDPI support."""
    import sys

    # Try Windows API first
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return dpi
        except Exception:
            pass

    # Try tkinter method
    try:
        import tkinter as tk

        root = tk._default_root  # type: ignore[attr-defined]
        if root is None:
            root = tk.Tk()
            root.withdraw()
        scaling = root.tk.call("tk", "scaling")
        dpi = int(float(scaling) * 72)
        if dpi > 72:
            return dpi
    except Exception:
        pass

    # Try environment variable
    import os

    gdk_scale = os.environ.get("GDK_SCALE", "1")
    try:
        scale = float(gdk_scale)
        if scale > 1:
            return int(96 * scale)
    except ValueError:
        pass

    return 96


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

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._show_placeholder()
