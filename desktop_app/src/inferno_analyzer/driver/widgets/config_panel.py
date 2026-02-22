"""Configuration panel for channel names and analysis thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel

if TYPE_CHECKING:
    from motorsports_data_notebook.profiles import VehicleProfile

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


# Mapping from shared profile canonical keys to this app's canonical keys.
# The driver app uses shorter keys for GPS channels.
_PROFILE_KEY_TO_LOCAL: dict[str, str] = {
    "gps_latitude": "gps_lat",
    "gps_longitude": "gps_lon",
}


class ConfigPanel(BaseConfigPanel):
    """Panel for configuring channel names and analysis thresholds."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_stats_click: Callable[[], None] | None = None,
        on_config_changed: Callable[[], None] | None = None,
        on_save_profile: Callable[[], None] | None = None,
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
        on_save_profile : callable, optional
            Callback when Save Profile button is clicked.
        """
        self._on_save_profile = on_save_profile
        super().__init__(
            parent,
            channel_defaults=DEFAULT_CHANNEL_NAMES.copy(),
            channel_display_names=_CHANNEL_DISPLAY_NAMES,
            on_stats_click=on_stats_click,
            on_config_changed=on_config_changed,
        )
        self._create_threshold_widgets()
        self._create_save_profile_btn()
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

    def _create_save_profile_btn(self) -> None:
        """Create the Save Profile button."""
        self.save_profile_btn = ctk.CTkButton(
            self.channels_frame,
            text="Save Profile",
            width=100,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._on_save_profile_click,
        )

    def _layout_widgets(self) -> None:
        """Arrange all widgets in the panel."""
        self.title_label.pack(anchor="w", padx=5, pady=(2, 5))

        # Channels
        self._pack_channels()

        # Add save profile button next to reset channels button
        num_channel_rows = (len(self.channel_entries) + 1) // 2
        self.reset_channels_btn.grid_configure(columnspan=2)
        self.save_profile_btn.grid(row=num_channel_rows + 1, column=2, columnspan=2, pady=2)

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

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate channel entries from a VehicleProfile.

        Maps from the shared profile's canonical keys (gps_latitude,
        gps_longitude) to this app's local keys (gps_lat, gps_lon).

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate from.
        """
        for profile_key, channel_value in profile.channel_names.items():
            # Map profile key to local key (e.g. gps_latitude -> gps_lat)
            local_key = _PROFILE_KEY_TO_LOCAL.get(profile_key, profile_key)
            if local_key in self.channel_entries:
                self.channel_entries[local_key].delete(0, "end")
                self.channel_entries[local_key].insert(0, channel_value)

    def get_vehicle_profile(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from current field values.

        Maps local keys back to the shared profile's canonical keys.

        Parameters
        ----------
        name : str
            Name for the profile.

        Returns
        -------
        VehicleProfile
            Profile with current field values.
        """
        from motorsports_data_notebook.profiles import VehicleProfile
        from motorsports_data_notebook.suspension import MotionRatios

        # Map local keys back to profile keys
        local_to_profile = {v: k for k, v in _PROFILE_KEY_TO_LOCAL.items()}
        channel_names: dict[str, str] = {}
        for local_key, entry in self.channel_entries.items():
            profile_key = local_to_profile.get(local_key, local_key)
            channel_names[profile_key] = entry.get() or self._default_channels[local_key]

        return VehicleProfile(
            name=name,
            channel_names=channel_names,
            motion_ratios=MotionRatios(),
        )
