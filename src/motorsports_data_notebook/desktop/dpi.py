"""HiDPI detection and scaling utilities for desktop apps."""

from __future__ import annotations

import customtkinter as ctk


def get_screen_dpi() -> int:
    """Get the screen DPI for HiDPI support.

    Tries platform-specific methods in order:
    1. Windows API (SetProcessDpiAwareness + GetDeviceCaps)
    2. Tkinter scaling factor
    3. GDK_SCALE environment variable

    Returns
    -------
    int
        Detected screen DPI, or 96 as default.
    """
    import sys

    # Try Windows API first
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return dpi
        except Exception:
            pass

    # Try tkinter method (works on Linux/WSLg)
    try:
        import tkinter as tk

        root = tk._default_root  # type: ignore[attr-defined]
        if root is None:
            root = tk.Tk()
            root.withdraw()
        # Get scaling factor - tkinter returns pixels per point (1 point = 1/72 inch)
        # scaling = pixels per point, so DPI = scaling * 72
        scaling = root.tk.call("tk", "scaling")
        dpi = int(float(scaling) * 72)
        if dpi > 72:  # Sanity check
            return dpi
    except Exception:
        pass

    # Try environment variable (common in HiDPI setups)
    import os

    gdk_scale = os.environ.get("GDK_SCALE", "1")
    try:
        scale = float(gdk_scale)
        if scale > 1:
            return int(96 * scale)
    except ValueError:
        pass

    return 96  # Default DPI


def setup_hidpi_scaling(app: ctk.CTk) -> None:
    """Configure HiDPI scaling for Linux/WSLg.

    On non-Windows platforms, detects the appropriate scale factor from
    GDK_SCALE or screen resolution and applies CustomTkinter + Tk scaling.

    Parameters
    ----------
    app : ctk.CTk
        The application window to configure scaling on.
    """
    import os
    import sys

    if sys.platform != "win32":
        scale_factor = 1.0

        # Check GDK_SCALE environment variable first (user override)
        gdk_scale = os.environ.get("GDK_SCALE")
        if gdk_scale:
            try:
                scale_factor = float(gdk_scale)
            except ValueError:
                pass

        # If no GDK_SCALE, detect from screen resolution
        if scale_factor == 1.0:
            try:
                screen_width = app.winfo_screenwidth()
                # HiDPI detection: if screen width > 2500, likely 2x scaling
                if screen_width > 2500:
                    scale_factor = 2.0
                elif screen_width > 1920:
                    scale_factor = 1.5
            except Exception:
                pass

        # Apply scaling if needed
        if scale_factor > 1.0:
            ctk.set_widget_scaling(scale_factor)
            ctk.set_window_scaling(scale_factor)
            # Also set tk scaling for consistent font rendering
            app.tk.call("tk", "scaling", scale_factor * 1.33)
