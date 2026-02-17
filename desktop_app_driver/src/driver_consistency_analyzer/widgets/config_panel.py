"""Configuration panel for channel names and analysis thresholds."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

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


class ConfigPanel(ctk.CTkFrame):
    """Panel for configuring channel names and analysis thresholds.

    Provides editable fields for channel name mappings and analysis
    parameters, plus status display and statistics button.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_stats_click: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the config panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        on_stats_click : callable, optional
            Callback when statistics button is clicked.
        """
        super().__init__(parent)

        self._on_stats_click = on_stats_click
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self) -> None:
        """Create all widgets."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="CONFIGURATION",
            font=ctk.CTkFont(size=14, weight="bold"),
        )

        # Channel names frame
        self.channels_frame = ctk.CTkFrame(self)
        self.channels_label = ctk.CTkLabel(
            self.channels_frame,
            text="Channel Names:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        # Channel name entries
        self.channel_entries: dict[str, ctk.CTkEntry] = {}
        self.channel_labels: dict[str, ctk.CTkLabel] = {}

        channel_display = {
            "throttle": "Throttle:",
            "brake": "Brake:",
            "lateral_g": "Lat G:",
            "gps_lat": "GPS Lat:",
            "gps_lon": "GPS Lon:",
            "gps_speed": "GPS Spd:",
        }

        for key, display_name in channel_display.items():
            self.channel_labels[key] = ctk.CTkLabel(
                self.channels_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self.channel_entries[key] = ctk.CTkEntry(
                self.channels_frame,
                width=110,
                font=ctk.CTkFont(size=11),
            )
            self.channel_entries[key].insert(0, DEFAULT_CHANNEL_NAMES[key])

        # Reset channels button
        self.reset_channels_btn = ctk.CTkButton(
            self.channels_frame,
            text="Reset to Default",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_channels,
        )

        # Thresholds frame
        self.thresholds_frame = ctk.CTkFrame(self)
        self.thresholds_label = ctk.CTkLabel(
            self.thresholds_frame,
            text="Analysis Thresholds:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        # Threshold entries
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

        # Status and actions frame
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")

        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )

        # Statistics button
        self.stats_btn = ctk.CTkButton(
            self.status_frame,
            text="Show Statistics",
            width=110,
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._on_stats_btn_click,
        )

    def _layout_widgets(self) -> None:
        """Arrange widgets in the panel."""
        # Title
        self.title_label.pack(anchor="w", padx=5, pady=(2, 5))

        # Channel names frame (2-column grid layout)
        self.channels_frame.pack(fill="x", padx=5, pady=2)
        self.channels_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        # Channel entries in 2-column grid
        keys = list(self.channel_entries.keys())
        for i, key in enumerate(keys):
            row = (i // 2) + 1
            col_offset = (i % 2) * 2
            self.channel_labels[key].grid(
                row=row, column=col_offset, padx=(2, 1), pady=1, sticky="e"
            )
            self.channel_entries[key].grid(
                row=row, column=col_offset + 1, padx=1, pady=1, sticky="w"
            )

        reset_row = (len(keys) // 2) + 2
        self.reset_channels_btn.grid(row=reset_row, column=0, columnspan=4, pady=2)

        # Thresholds frame
        self.thresholds_frame.pack(fill="x", padx=5, pady=2)
        self.thresholds_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        threshold_keys = list(self.threshold_entries.keys())
        for i, key in enumerate(threshold_keys):
            row = i + 1
            self.threshold_labels[key].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
            self.threshold_entries[key].grid(row=row, column=1, padx=1, pady=1, sticky="w")

        # Status and actions at bottom
        self.status_frame.pack(fill="x", padx=5, pady=(5, 2))
        self.status_label.pack(side="left", fill="x", expand=True, padx=2)
        self.stats_btn.pack(side="right", padx=2)

    def _on_stats_btn_click(self) -> None:
        """Handle statistics button click."""
        if self._on_stats_click:
            self._on_stats_click()

    def set_status(self, message: str) -> None:
        """Update the status label text."""
        self.status_label.configure(text=message)

    def set_stats_button_text(self, text: str) -> None:
        """Update the statistics button text."""
        self.stats_btn.configure(text=text)

    def _reset_channels(self) -> None:
        """Reset channel names to defaults."""
        for key, val in DEFAULT_CHANNEL_NAMES.items():
            self.channel_entries[key].delete(0, "end")
            self.channel_entries[key].insert(0, val)

    def get_channel_names(self) -> dict[str, str]:
        """Get current channel name mappings.

        Returns
        -------
        dict[str, str]
            Channel name mapping dictionary.
        """
        return {
            key: self.channel_entries[key].get() or DEFAULT_CHANNEL_NAMES[key]
            for key in DEFAULT_CHANNEL_NAMES
        }

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

    def get_sustain_time_ms(self) -> float:
        """Get the sustain time in milliseconds."""
        try:
            return float(self.threshold_entries["sustain_time"].get())
        except ValueError:
            return DEFAULT_SUSTAIN_TIME_MS
