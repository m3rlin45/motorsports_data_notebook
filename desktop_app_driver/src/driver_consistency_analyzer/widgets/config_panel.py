"""Configuration panel for channel names and analysis thresholds."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel

# Default channel name mappings
DEFAULT_CHANNEL_NAMES = {
    "throttle": "PPS",
    "brake": "BrakePress",
    "lateral_g": "LateralAcc",
    "gps_lat": "GPS Latitude",
    "gps_lon": "GPS Longitude",
    "gps_speed": "GPS Speed",
}

# Default analysis thresholds
DEFAULT_CORNER_THRESHOLD = 0.006
DEFAULT_THROTTLE_THRESHOLD = 98.0
DEFAULT_SUSTAIN_TIME_MS = 500.0

_CHANNEL_DISPLAY_NAMES = {
    "throttle": "Throttle:",
    "brake": "Brake:",
    "lateral_g": "Lat G:",
    "gps_lat": "GPS Lat:",
    "gps_lon": "GPS Lon:",
    "gps_speed": "GPS Spd:",
}


class ConfigPanel(BaseConfigPanel):
    """Panel for configuring channel names and analysis thresholds."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_stats_click: Callable[[], None] | None = None,
        on_config_changed: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the config panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        on_stats_click : callable, optional
            Callback when statistics button is clicked.
        on_config_changed : callable, optional
            Callback when any configuration value changes.
        """
        super().__init__(
            parent,
            channel_defaults=DEFAULT_CHANNEL_NAMES.copy(),
            channel_display_names=_CHANNEL_DISPLAY_NAMES,
            on_stats_click=on_stats_click,
            on_config_changed=on_config_changed,
        )
        self._create_threshold_widgets()
        self._layout_widgets()

    def _create_threshold_widgets(self) -> None:
        """Create threshold input widgets."""
        self.thresholds_frame = ctk.CTkFrame(self)
        self.thresholds_label = ctk.CTkLabel(
            self.thresholds_frame,
            text="Analysis Thresholds:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        self.threshold_entries: dict[str, ctk.CTkEntry] = {}
        self.threshold_labels: dict[str, ctk.CTkLabel] = {}

        threshold_config = {
            "corner_threshold": ("Corner Det.:", str(DEFAULT_CORNER_THRESHOLD)),
            "throttle_threshold": ("Throttle %:", str(DEFAULT_THROTTLE_THRESHOLD)),
            "sustain_time": ("Sustain ms:", str(DEFAULT_SUSTAIN_TIME_MS)),
        }

        for key, (display_name, default_val) in threshold_config.items():
            self.threshold_labels[key] = ctk.CTkLabel(
                self.thresholds_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self.threshold_entries[key] = ctk.CTkEntry(
                self.thresholds_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            self.threshold_entries[key].insert(0, default_val)
            self.threshold_entries[key].bind("<KeyRelease>", self._on_entry_changed)

    def _layout_widgets(self) -> None:
        """Arrange all widgets in the panel."""
        self.title_label.pack(anchor="w", padx=5, pady=(2, 5))

        # Channels
        self._pack_channels()

        # Thresholds
        self.thresholds_frame.pack(fill="x", padx=5, pady=2)
        self.thresholds_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        threshold_keys = list(self.threshold_entries.keys())
        for i, key in enumerate(threshold_keys):
            row = i + 1
            self.threshold_labels[key].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
            self.threshold_entries[key].grid(row=row, column=1, padx=1, pady=1, sticky="w")

        # Status
        self._pack_status()

    def get_corner_threshold(self) -> float:
        """Get the corner detection threshold."""
        try:
            return float(self.threshold_entries["corner_threshold"].get())
        except ValueError:
            return DEFAULT_CORNER_THRESHOLD

    def get_throttle_threshold(self) -> float:
        """Get the throttle acceptance threshold."""
        try:
            return float(self.threshold_entries["throttle_threshold"].get())
        except ValueError:
            return DEFAULT_THROTTLE_THRESHOLD

    def set_throttle_threshold(self, value: float) -> None:
        """Programmatically update the throttle threshold entry field."""
        entry = self.threshold_entries["throttle_threshold"]
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")

    def get_sustain_time_ms(self) -> float:
        """Get the sustain time in milliseconds."""
        try:
            return float(self.threshold_entries["sustain_time"].get())
        except ValueError:
            return DEFAULT_SUSTAIN_TIME_MS
