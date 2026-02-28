"""Configuration panel for tire grip channel names."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel
from motorsports_data_notebook.profiles import DEFAULT_CHANNEL_NAMES

if TYPE_CHECKING:
    from motorsports_data_notebook.profiles import VehicleProfile

# Display labels for channels used by the tire grip analysis
_CHANNEL_DISPLAY_NAMES = {
    "lateral_g": "Lat G:",
    "inline_g": "Inline G:",
    "tpms_press_fl": "Press FL:",
    "tpms_press_fr": "Press FR:",
    "tpms_press_rl": "Press RL:",
    "tpms_press_rr": "Press RR:",
    "tpms_temp_fl": "Temp FL:",
    "tpms_temp_fr": "Temp FR:",
    "tpms_temp_rl": "Temp RL:",
    "tpms_temp_rr": "Temp RR:",
}


class ConfigPanel(BaseConfigPanel):
    """Panel for configuring tire grip channel names."""

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
            channel_defaults={k: DEFAULT_CHANNEL_NAMES[k] for k in _CHANNEL_DISPLAY_NAMES},
            channel_display_names=_CHANNEL_DISPLAY_NAMES,
            on_stats_click=on_stats_click,
            on_config_changed=on_config_changed,
        )
        self._create_save_profile_btn()
        self._layout_widgets()

    def _create_save_profile_btn(self) -> None:
        """Create the Save Profile button (placed in channels frame)."""
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

        # Channels + status (from base class)
        self._pack_channels()

        # Add save profile button next to reset channels button
        self.reset_channels_btn.grid_configure(columnspan=2)
        self.save_profile_btn.grid(row=self._channel_button_row, column=2, columnspan=2, pady=2)

        self._pack_status()

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    # ------------------------------------------------------------------
    # Profile support
    # ------------------------------------------------------------------

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate Session A channel entries from a VehicleProfile.

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate from.
        """
        for key, entry in self._channel_entries_a.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])

    def set_from_profile_b(self, profile: VehicleProfile) -> None:
        """Populate Session B channel entries from a VehicleProfile.

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate B entries from.
        """
        self._ensure_b_entries()
        assert self._channel_entries_b is not None

        for key, entry in self._channel_entries_b.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])

    def get_vehicle_profile(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from Session A's current field values.

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

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_a(),
            motion_ratios=MotionRatios(),
        )

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
        from motorsports_data_notebook.profiles import VehicleProfile
        from motorsports_data_notebook.suspension import MotionRatios

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_b(),
            motion_ratios=MotionRatios(),
        )
