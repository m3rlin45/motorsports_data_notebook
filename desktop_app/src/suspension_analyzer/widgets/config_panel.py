"""Configuration panel for motion ratios and channel names."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.profiles import DEFAULT_CHANNEL_NAMES
from motorsports_data_notebook.suspension import MotionRatios

if TYPE_CHECKING:
    from motorsports_data_notebook.profiles import VehicleProfile


# Channel entries organized by group with display labels
_SHOCK_CHANNELS = {
    "shock_fl": "FL Shock:",
    "shock_fr": "FR Shock:",
    "shock_rl": "RL Shock:",
    "shock_rr": "RR Shock:",
}

_GPS_CHANNELS = {
    "gps_speed": "GPS Speed:",
    "gps_latitude": "GPS Lat:",
    "gps_longitude": "GPS Lon:",
}

_DYNAMICS_CHANNELS = {
    "throttle": "Throttle:",
    "brake": "Brake:",
    "lateral_g": "Lateral G:",
    "steering": "Steering:",
}


class ConfigPanel(ctk.CTkFrame):
    """Panel for configuring analysis parameters.

    Provides inputs for motion ratios and channel name mappings,
    plus status display and statistics button.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_stats_click: Callable[[], None] | None = None,
        on_save_profile: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the config panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        on_stats_click : callable, optional
            Callback when statistics button is clicked.
        on_save_profile : callable, optional
            Callback when Save Profile button is clicked.
        """
        super().__init__(parent)

        self._default_ratios = MotionRatios.toyota_86_zn6()
        self._default_channels = DEFAULT_CHANNEL_NAMES.copy()
        self._on_stats_click = on_stats_click
        self._on_save_profile = on_save_profile

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

        # Motion ratios frame
        self.ratios_frame = ctk.CTkFrame(self)
        self.ratios_label = ctk.CTkLabel(
            self.ratios_frame,
            text="Motion Ratios:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        # Motion ratio entries (2x2 grid)
        self.ratio_entries: dict[str, ctk.CTkEntry] = {}
        self.ratio_labels: dict[str, ctk.CTkLabel] = {}

        for corner, default_val in [
            ("FL", self._default_ratios.front_left),
            ("FR", self._default_ratios.front_right),
            ("RL", self._default_ratios.rear_left),
            ("RR", self._default_ratios.rear_right),
        ]:
            self.ratio_labels[corner] = ctk.CTkLabel(
                self.ratios_frame,
                text=f"{corner}:",
                font=ctk.CTkFont(size=11),
            )
            self.ratio_entries[corner] = ctk.CTkEntry(
                self.ratios_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            self.ratio_entries[corner].insert(0, f"{default_val:.3f}")

        # Reset ratios button
        self.reset_ratios_btn = ctk.CTkButton(
            self.ratios_frame,
            text="Reset to Toyota 86",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_ratios,
        )

        # Channel names frame
        self.channels_frame = ctk.CTkFrame(self)
        self.channels_label = ctk.CTkLabel(
            self.channels_frame,
            text="Channel Names:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        # Channel name entries — all groups
        self.channel_entries: dict[str, ctk.CTkEntry] = {}
        self.channel_labels: dict[str, ctk.CTkLabel] = {}

        all_channel_display = {}
        all_channel_display.update(_SHOCK_CHANNELS)
        all_channel_display.update(_GPS_CHANNELS)
        all_channel_display.update(_DYNAMICS_CHANNELS)

        for key, display_name in all_channel_display.items():
            self.channel_labels[key] = ctk.CTkLabel(
                self.channels_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self.channel_entries[key] = ctk.CTkEntry(
                self.channels_frame,
                width=120,
                font=ctk.CTkFont(size=11),
            )
            self.channel_entries[key].insert(0, self._default_channels.get(key, ""))

        # Reset channels button
        self.reset_channels_btn = ctk.CTkButton(
            self.channels_frame,
            text="Reset to Default",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_channels,
        )

        # Save Profile button
        self.save_profile_btn = ctk.CTkButton(
            self.channels_frame,
            text="Save Profile",
            width=100,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._on_save_profile_click,
        )

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

        # Motion ratios frame (compact)
        self.ratios_frame.pack(fill="x", padx=5, pady=2)
        self.ratios_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        # Motion ratio grid (2x2, compact)
        self.ratio_labels["FL"].grid(row=1, column=0, padx=(2, 1), pady=1, sticky="e")
        self.ratio_entries["FL"].grid(row=1, column=1, padx=1, pady=1)
        self.ratio_labels["FR"].grid(row=1, column=2, padx=(5, 1), pady=1, sticky="e")
        self.ratio_entries["FR"].grid(row=1, column=3, padx=1, pady=1)

        self.ratio_labels["RL"].grid(row=2, column=0, padx=(2, 1), pady=1, sticky="e")
        self.ratio_entries["RL"].grid(row=2, column=1, padx=1, pady=1)
        self.ratio_labels["RR"].grid(row=2, column=2, padx=(5, 1), pady=1, sticky="e")
        self.ratio_entries["RR"].grid(row=2, column=3, padx=1, pady=1)

        self.reset_ratios_btn.grid(row=3, column=0, columnspan=4, pady=2)

        # Channel names frame (compact, 2-column grid by group)
        self.channels_frame.pack(fill="x", padx=5, pady=2)
        self.channels_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        # Shock Pots (rows 1-2)
        row = 1
        shock_keys = list(_SHOCK_CHANNELS.keys())
        self.channel_labels[shock_keys[0]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[shock_keys[0]].grid(row=row, column=1, padx=1, pady=1, sticky="w")
        self.channel_labels[shock_keys[1]].grid(row=row, column=2, padx=(5, 1), pady=1, sticky="e")
        self.channel_entries[shock_keys[1]].grid(row=row, column=3, padx=1, pady=1, sticky="w")
        row += 1
        self.channel_labels[shock_keys[2]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[shock_keys[2]].grid(row=row, column=1, padx=1, pady=1, sticky="w")
        self.channel_labels[shock_keys[3]].grid(row=row, column=2, padx=(5, 1), pady=1, sticky="e")
        self.channel_entries[shock_keys[3]].grid(row=row, column=3, padx=1, pady=1, sticky="w")

        # GPS (row 3-4)
        row += 1
        gps_keys = list(_GPS_CHANNELS.keys())
        self.channel_labels[gps_keys[0]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[gps_keys[0]].grid(row=row, column=1, padx=1, pady=1, sticky="w")
        self.channel_labels[gps_keys[1]].grid(row=row, column=2, padx=(5, 1), pady=1, sticky="e")
        self.channel_entries[gps_keys[1]].grid(row=row, column=3, padx=1, pady=1, sticky="w")
        row += 1
        self.channel_labels[gps_keys[2]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[gps_keys[2]].grid(row=row, column=1, padx=1, pady=1, sticky="w")

        # Pedals & Dynamics (rows 5-6)
        row += 1
        dyn_keys = list(_DYNAMICS_CHANNELS.keys())
        self.channel_labels[dyn_keys[0]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[dyn_keys[0]].grid(row=row, column=1, padx=1, pady=1, sticky="w")
        self.channel_labels[dyn_keys[1]].grid(row=row, column=2, padx=(5, 1), pady=1, sticky="e")
        self.channel_entries[dyn_keys[1]].grid(row=row, column=3, padx=1, pady=1, sticky="w")
        row += 1
        self.channel_labels[dyn_keys[2]].grid(row=row, column=0, padx=(2, 1), pady=1, sticky="e")
        self.channel_entries[dyn_keys[2]].grid(row=row, column=1, padx=1, pady=1, sticky="w")
        self.channel_labels[dyn_keys[3]].grid(row=row, column=2, padx=(5, 1), pady=1, sticky="e")
        self.channel_entries[dyn_keys[3]].grid(row=row, column=3, padx=1, pady=1, sticky="w")

        # Buttons row
        row += 1
        self.reset_channels_btn.grid(row=row, column=0, columnspan=2, pady=2)
        self.save_profile_btn.grid(row=row, column=2, columnspan=2, pady=2)

        # Status and actions at bottom
        self.status_frame.pack(fill="x", padx=5, pady=(5, 2))
        self.status_label.pack(side="left", fill="x", expand=True, padx=2)
        self.stats_btn.pack(side="right", padx=2)

    def _on_stats_btn_click(self) -> None:
        """Handle statistics button click."""
        if self._on_stats_click:
            self._on_stats_click()

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    def set_status(self, message: str) -> None:
        """Update the status label text."""
        self.status_label.configure(text=message)

    def set_stats_button_text(self, text: str) -> None:
        """Update the statistics button text."""
        self.stats_btn.configure(text=text)

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate all fields from a VehicleProfile.

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate from.
        """
        # Set motion ratios
        ratios = {
            "FL": profile.motion_ratios.front_left,
            "FR": profile.motion_ratios.front_right,
            "RL": profile.motion_ratios.rear_left,
            "RR": profile.motion_ratios.rear_right,
        }
        for corner, val in ratios.items():
            self.ratio_entries[corner].delete(0, "end")
            self.ratio_entries[corner].insert(0, f"{val:.3f}")

        # Set channel names
        for key, entry in self.channel_entries.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])

    def get_vehicle_profile(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from current field values.

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

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names(),
            motion_ratios=self.get_motion_ratios(),
        )

    def _reset_ratios(self) -> None:
        """Reset motion ratios to Toyota 86 defaults."""
        defaults = MotionRatios.toyota_86_zn6()
        values = {
            "FL": defaults.front_left,
            "FR": defaults.front_right,
            "RL": defaults.rear_left,
            "RR": defaults.rear_right,
        }
        for corner, val in values.items():
            self.ratio_entries[corner].delete(0, "end")
            self.ratio_entries[corner].insert(0, f"{val:.3f}")

    def _reset_channels(self) -> None:
        """Reset channel names to defaults."""
        for key, entry in self.channel_entries.items():
            entry.delete(0, "end")
            entry.insert(0, self._default_channels.get(key, ""))

    def get_motion_ratios(self) -> MotionRatios:
        """Get current motion ratio values.

        Returns
        -------
        MotionRatios
            Motion ratios from the input fields.
        """
        try:
            return MotionRatios(
                front_left=float(self.ratio_entries["FL"].get()),
                front_right=float(self.ratio_entries["FR"].get()),
                rear_left=float(self.ratio_entries["RL"].get()),
                rear_right=float(self.ratio_entries["RR"].get()),
            )
        except ValueError:
            return MotionRatios.toyota_86_zn6()

    def get_channel_names(self) -> dict[str, str]:
        """Get current channel name mappings.

        Returns
        -------
        dict[str, str]
            Channel name mapping dictionary with all configured channels.
        """
        result = {}
        for key, entry in self.channel_entries.items():
            value = entry.get().strip()
            if value:
                result[key] = value
            elif key in self._default_channels:
                result[key] = self._default_channels[key]
        return result
