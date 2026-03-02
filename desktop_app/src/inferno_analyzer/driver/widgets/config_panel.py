"""Configuration panel for channel names and analysis thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel

if TYPE_CHECKING:
    from motorsports_data_notebook.desktop.autocomplete_entry import AutocompleteEntry
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
        on_config_changed: Callable[[], None] | None = None,
        on_save_profile: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the config panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
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
            on_config_changed=on_config_changed,
        )
        self._create_threshold_widgets()
        self._create_save_profile_btn()
        self._layout_widgets()

    _THRESHOLD_CONFIG = {
        "corner_threshold": ("Corner Det.:", str(DEFAULT_CORNER_THRESHOLD)),
        "throttle_threshold": ("Throttle %:", str(DEFAULT_THROTTLE_THRESHOLD)),
        "sustain_time": ("Sustain ms:", str(DEFAULT_SUSTAIN_TIME_MS)),
    }

    def _create_threshold_widgets(self) -> None:
        """Create threshold input widgets (Session A)."""
        self.thresholds_frame = ctk.CTkFrame(self)
        self.thresholds_label = ctk.CTkLabel(
            self.thresholds_frame,
            text="Analysis Thresholds:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        self._threshold_entries_a: dict[str, ctk.CTkEntry] = {}
        self._threshold_labels_a: dict[str, ctk.CTkLabel] = {}
        self._threshold_entries_b: dict[str, ctk.CTkEntry] | None = None
        self._threshold_labels_b: dict[str, ctk.CTkLabel] | None = None

        for key, (display_name, default_val) in self._THRESHOLD_CONFIG.items():
            self._threshold_labels_a[key] = ctk.CTkLabel(
                self.thresholds_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self._threshold_entries_a[key] = ctk.CTkEntry(
                self.thresholds_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            self._threshold_entries_a[key].insert(0, default_val)
            self._threshold_entries_a[key].bind("<KeyRelease>", self._on_entry_changed)

    @property
    def threshold_entries(self) -> dict[str, ctk.CTkEntry]:
        """Return the active session's threshold entries."""
        if self._active_session == "B" and self._threshold_entries_b is not None:
            return self._threshold_entries_b
        return self._threshold_entries_a

    @property
    def threshold_labels(self) -> dict[str, ctk.CTkLabel]:
        """Return the active session's threshold labels."""
        if self._active_session == "B" and self._threshold_labels_b is not None:
            return self._threshold_labels_b
        return self._threshold_labels_a

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
        self.title_label.pack(anchor="w", padx=5, pady=(2, 2))

        # Channels
        self._pack_channels()

        # Add save profile button next to reset channels button
        self.reset_channels_btn.grid_configure(columnspan=2)
        self.save_profile_btn.grid(row=self._channel_button_row, column=2, columnspan=2, pady=2)

        # Thresholds
        self.thresholds_frame.pack(fill="x", padx=5, pady=2)
        self._repack_thresholds_grid()

        # Status
        self._pack_status()

    def _repack_thresholds_grid(self) -> None:
        """Re-grid the active session's threshold labels/entries."""
        for widget in self.thresholds_frame.winfo_children():
            widget.grid_forget()

        self.thresholds_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        labels = self.threshold_labels
        entries = self.threshold_entries

        for i, key in enumerate(entries):
            row = i + 1
            labels[key].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
            entries[key].grid(row=row, column=1, padx=1, pady=1, sticky="w")

    def _get_threshold(self, key: str, default: float, entries: dict[str, ctk.CTkEntry]) -> float:
        """Read a float threshold from the given entry dict."""
        try:
            return float(entries[key].get())
        except ValueError:
            return default

    def get_corner_threshold(self) -> float:
        """Get Session A corner detection threshold (backward compat)."""
        return self.get_corner_threshold_a()

    def get_corner_threshold_a(self) -> float:
        """Get Session A corner detection threshold."""
        return self._get_threshold(
            "corner_threshold", DEFAULT_CORNER_THRESHOLD, self._threshold_entries_a
        )

    def get_corner_threshold_b(self) -> float:
        """Get Session B corner detection threshold."""
        if self._sync_with_a or self._threshold_entries_b is None:
            return self.get_corner_threshold_a()
        return self._get_threshold(
            "corner_threshold", DEFAULT_CORNER_THRESHOLD, self._threshold_entries_b
        )

    def get_throttle_threshold(self) -> float:
        """Get Session A throttle acceptance threshold (backward compat)."""
        return self.get_throttle_threshold_a()

    def get_throttle_threshold_a(self) -> float:
        """Get Session A throttle acceptance threshold."""
        return self._get_threshold(
            "throttle_threshold", DEFAULT_THROTTLE_THRESHOLD, self._threshold_entries_a
        )

    def get_throttle_threshold_b(self) -> float:
        """Get Session B throttle acceptance threshold."""
        if self._sync_with_a or self._threshold_entries_b is None:
            return self.get_throttle_threshold_a()
        return self._get_threshold(
            "throttle_threshold", DEFAULT_THROTTLE_THRESHOLD, self._threshold_entries_b
        )

    def set_throttle_threshold(self, value: float) -> None:
        """Programmatically update the Session A throttle threshold entry field."""
        entry = self._threshold_entries_a["throttle_threshold"]
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")

    def set_throttle_threshold_b(self, value: float) -> None:
        """Programmatically update the Session B throttle threshold entry field."""
        if self._threshold_entries_b is None:
            return
        entry = self._threshold_entries_b["throttle_threshold"]
        entry.delete(0, "end")
        entry.insert(0, f"{value:.1f}")

    def get_sustain_time_ms(self) -> float:
        """Get Session A sustain time in milliseconds (backward compat)."""
        return self.get_sustain_time_ms_a()

    def get_sustain_time_ms_a(self) -> float:
        """Get Session A sustain time in milliseconds."""
        return self._get_threshold(
            "sustain_time", DEFAULT_SUSTAIN_TIME_MS, self._threshold_entries_a
        )

    def get_sustain_time_ms_b(self) -> float:
        """Get Session B sustain time in milliseconds."""
        if self._sync_with_a or self._threshold_entries_b is None:
            return self.get_sustain_time_ms_a()
        return self._get_threshold(
            "sustain_time", DEFAULT_SUSTAIN_TIME_MS, self._threshold_entries_b
        )

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate Session A channel entries from a VehicleProfile.

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
            if local_key in self._channel_entries_a:
                self._channel_entries_a[local_key].delete(0, "end")
                self._channel_entries_a[local_key].insert(0, channel_value)

    def set_from_profile_b(self, profile: VehicleProfile) -> None:
        """Populate Session B channel entries from a VehicleProfile.

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate B entries from.
        """
        self._ensure_b_entries()
        assert self._channel_entries_b is not None
        for profile_key, channel_value in profile.channel_names.items():
            local_key = _PROFILE_KEY_TO_LOCAL.get(profile_key, profile_key)
            if local_key in self._channel_entries_b:
                self._channel_entries_b[local_key].delete(0, "end")
                self._channel_entries_b[local_key].insert(0, channel_value)

    def get_vehicle_profile(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from Session A's current field values.

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
        return self._build_profile_from_entries(name, self._channel_entries_a)

    def get_vehicle_profile_b(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from Session B's current field values.

        Parameters
        ----------
        name : str
            Name for the profile.

        Returns
        -------
        VehicleProfile
            Profile with B's channel names.
        """
        if self._sync_with_a or self._channel_entries_b is None:
            return self._build_profile_from_entries(name, self._channel_entries_a)
        return self._build_profile_from_entries(name, self._channel_entries_b)

    def _build_profile_from_entries(
        self, name: str, entries: dict[str, AutocompleteEntry]
    ) -> VehicleProfile:
        """Build a VehicleProfile from a set of channel entries."""
        from motorsports_data_notebook.profiles import VehicleProfile
        from motorsports_data_notebook.suspension import MotionRatios

        # Map local keys back to profile keys
        local_to_profile = {v: k for k, v in _PROFILE_KEY_TO_LOCAL.items()}
        channel_names: dict[str, str] = {}
        for local_key, entry in entries.items():
            profile_key = local_to_profile.get(local_key, local_key)
            channel_names[profile_key] = entry.get() or self._default_channels[local_key]

        return VehicleProfile(
            name=name,
            channel_names=channel_names,
            motion_ratios=MotionRatios(),
        )

    # ------------------------------------------------------------------
    # BaseConfigPanel hooks for per-session thresholds
    # ------------------------------------------------------------------

    def _create_b_extra(self) -> None:
        """Create Session B threshold entries."""
        self._threshold_entries_b = {}
        self._threshold_labels_b = {}
        for key, (display_name, _) in self._THRESHOLD_CONFIG.items():
            self._threshold_labels_b[key] = ctk.CTkLabel(
                self.thresholds_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self._threshold_entries_b[key] = ctk.CTkEntry(
                self.thresholds_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            # Initialize from A
            self._threshold_entries_b[key].insert(0, self._threshold_entries_a[key].get())
            self._threshold_entries_b[key].bind("<KeyRelease>", self._on_entry_changed)

    def _repack_extra_widgets(self) -> None:
        """Swap threshold entries when session changes."""
        self._repack_thresholds_grid()

    def _copy_extra_a_to_b(self) -> None:
        """Copy A thresholds to B."""
        if self._threshold_entries_b is None:
            return
        for key in self._threshold_entries_a:
            val = self._threshold_entries_a[key].get()
            self._threshold_entries_b[key].delete(0, "end")
            self._threshold_entries_b[key].insert(0, val)

    def _apply_extra_sync_state(self, state: str) -> None:
        """Enable/disable B threshold entries."""
        if self._threshold_entries_b is None:
            return
        for entry in self._threshold_entries_b.values():
            entry.configure(state=state)
