"""Configuration panel for motion ratios and channel names."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel
from motorsports_data_notebook.profiles import DEFAULT_CHANNEL_NAMES
from motorsports_data_notebook.suspension import MotionRatios

if TYPE_CHECKING:
    from motorsports_data_notebook.profiles import VehicleProfile

# Display labels for channels used by the suspension analysis
_CHANNEL_DISPLAY_NAMES = {
    "shock_fl": "FL Shock:",
    "shock_fr": "FR Shock:",
    "shock_rl": "RL Shock:",
    "shock_rr": "RR Shock:",
}


class ConfigPanel(BaseConfigPanel):
    """Panel for configuring motion ratios and channel name mappings."""

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
        self._default_ratios = MotionRatios.toyota_86_zn6()
        self._create_ratio_widgets()
        self._create_save_profile_btn()
        self._layout_widgets()

    _RATIO_CORNERS = [
        ("FL", "front_left"),
        ("FR", "front_right"),
        ("RL", "rear_left"),
        ("RR", "rear_right"),
    ]

    def _create_ratio_widgets(self) -> None:
        """Create motion ratio input widgets (Session A)."""
        self.ratios_frame = ctk.CTkFrame(self)
        self.ratios_label = ctk.CTkLabel(
            self.ratios_frame,
            text="Motion Ratios:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        self._ratio_entries_a: dict[str, ctk.CTkEntry] = {}
        self._ratio_labels_a: dict[str, ctk.CTkLabel] = {}
        self._ratio_entries_b: dict[str, ctk.CTkEntry] | None = None
        self._ratio_labels_b: dict[str, ctk.CTkLabel] | None = None

        for corner, attr in self._RATIO_CORNERS:
            default_val = getattr(self._default_ratios, attr)
            self._ratio_labels_a[corner] = ctk.CTkLabel(
                self.ratios_frame,
                text=f"{corner}:",
                font=ctk.CTkFont(size=11),
            )
            self._ratio_entries_a[corner] = ctk.CTkEntry(
                self.ratios_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            self._ratio_entries_a[corner].insert(0, f"{default_val:.3f}")
            self._ratio_entries_a[corner].bind("<KeyRelease>", self._on_entry_changed)

        self.reset_ratios_btn = ctk.CTkButton(
            self.ratios_frame,
            text="Reset to Toyota 86",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_ratios,
        )

    @property
    def ratio_entries(self) -> dict[str, ctk.CTkEntry]:
        """Return the active session's ratio entries."""
        if self._active_session == "B" and self._ratio_entries_b is not None:
            return self._ratio_entries_b
        return self._ratio_entries_a

    @property
    def ratio_labels(self) -> dict[str, ctk.CTkLabel]:
        """Return the active session's ratio labels."""
        if self._active_session == "B" and self._ratio_labels_b is not None:
            return self._ratio_labels_b
        return self._ratio_labels_a

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

        # Motion ratios frame
        self.ratios_frame.pack(fill="x", padx=5, pady=2)
        self._repack_ratios_grid()

        # Channels + status (from base class)
        self._pack_channels()

        # Add save profile button next to reset channels button
        self.reset_channels_btn.grid_configure(columnspan=2)
        self.save_profile_btn.grid(row=self._channel_button_row, column=2, columnspan=2, pady=2)

        self._pack_status()

    def _repack_ratios_grid(self) -> None:
        """Re-grid the active session's ratio labels/entries."""
        for widget in self.ratios_frame.winfo_children():
            widget.grid_forget()

        self.ratios_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        labels = self.ratio_labels
        entries = self.ratio_entries

        labels["FL"].grid(row=1, column=0, padx=(2, 1), pady=1, sticky="e")
        entries["FL"].grid(row=1, column=1, padx=1, pady=1)
        labels["FR"].grid(row=1, column=2, padx=(5, 1), pady=1, sticky="e")
        entries["FR"].grid(row=1, column=3, padx=1, pady=1)

        labels["RL"].grid(row=2, column=0, padx=(2, 1), pady=1, sticky="e")
        entries["RL"].grid(row=2, column=1, padx=1, pady=1)
        labels["RR"].grid(row=2, column=2, padx=(5, 1), pady=1, sticky="e")
        entries["RR"].grid(row=2, column=3, padx=1, pady=1)

        self.reset_ratios_btn.grid(row=3, column=0, columnspan=4, pady=2)

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate all fields from a VehicleProfile (Session A).

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate from.
        """
        # Set motion ratios (A)
        for corner, attr in self._RATIO_CORNERS:
            val = getattr(profile.motion_ratios, attr)
            self._ratio_entries_a[corner].delete(0, "end")
            self._ratio_entries_a[corner].insert(0, f"{val:.3f}")

        # Set channel names (explicitly target A entries)
        for key, entry in self._channel_entries_a.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])

    def set_from_profile_b(self, profile: VehicleProfile) -> None:
        """Populate Session B entries from a VehicleProfile.

        Parameters
        ----------
        profile : VehicleProfile
            The profile to populate B entries from.
        """
        self._ensure_b_entries()
        assert self._channel_entries_b is not None
        assert self._ratio_entries_b is not None

        for key, entry in self._channel_entries_b.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])

        for corner, attr in self._RATIO_CORNERS:
            val = getattr(profile.motion_ratios, attr)
            self._ratio_entries_b[corner].delete(0, "end")
            self._ratio_entries_b[corner].insert(0, f"{val:.3f}")

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

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_a(),
            motion_ratios=self.get_motion_ratios_a(),
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
            Profile with B's channel names and motion ratios.
        """
        from motorsports_data_notebook.profiles import VehicleProfile

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_b(),
            motion_ratios=self.get_motion_ratios_b(),
        )

    def _reset_ratios(self) -> None:
        """Reset motion ratios to Toyota 86 defaults for the active session."""
        defaults = MotionRatios.toyota_86_zn6()
        entries = self.ratio_entries
        for corner, attr in self._RATIO_CORNERS:
            val = getattr(defaults, attr)
            entries[corner].delete(0, "end")
            entries[corner].insert(0, f"{val:.3f}")
        if self._active_session == "A" and self._sync_with_a:
            self._copy_extra_a_to_b()
        self._on_entry_changed()

    def _ratios_from_entries(self, entries: dict[str, ctk.CTkEntry]) -> MotionRatios:
        """Parse MotionRatios from a set of entry widgets."""
        try:
            return MotionRatios(
                front_left=float(entries["FL"].get()),
                front_right=float(entries["FR"].get()),
                rear_left=float(entries["RL"].get()),
                rear_right=float(entries["RR"].get()),
            )
        except ValueError:
            return MotionRatios.toyota_86_zn6()

    def get_motion_ratios(self) -> MotionRatios:
        """Get motion ratios for Session A (backward compat)."""
        return self.get_motion_ratios_a()

    def get_motion_ratios_a(self) -> MotionRatios:
        """Get Session A motion ratio values."""
        return self._ratios_from_entries(self._ratio_entries_a)

    def get_motion_ratios_b(self) -> MotionRatios:
        """Get Session B motion ratio values.

        Returns A's values if synced or B entries don't exist.
        """
        if self._sync_with_a or self._ratio_entries_b is None:
            return self.get_motion_ratios_a()
        return self._ratios_from_entries(self._ratio_entries_b)

    # ------------------------------------------------------------------
    # BaseConfigPanel hooks for per-session motion ratios
    # ------------------------------------------------------------------

    def _create_b_extra(self) -> None:
        """Create Session B motion ratio entries."""
        self._ratio_entries_b = {}
        self._ratio_labels_b = {}
        for corner, _ in self._RATIO_CORNERS:
            self._ratio_labels_b[corner] = ctk.CTkLabel(
                self.ratios_frame,
                text=f"{corner}:",
                font=ctk.CTkFont(size=11),
            )
            self._ratio_entries_b[corner] = ctk.CTkEntry(
                self.ratios_frame,
                width=80,
                font=ctk.CTkFont(size=11),
            )
            # Initialize from A
            self._ratio_entries_b[corner].insert(0, self._ratio_entries_a[corner].get())
            self._ratio_entries_b[corner].bind("<KeyRelease>", self._on_entry_changed)

    def _repack_extra_widgets(self) -> None:
        """Swap ratio entries when session changes."""
        self._repack_ratios_grid()

    def _copy_extra_a_to_b(self) -> None:
        """Copy A motion ratios to B."""
        if self._ratio_entries_b is None:
            return
        for corner in self._ratio_entries_a:
            val = self._ratio_entries_a[corner].get()
            self._ratio_entries_b[corner].delete(0, "end")
            self._ratio_entries_b[corner].insert(0, val)

    def _apply_extra_sync_state(self, state: str) -> None:
        """Enable/disable B motion ratio entries."""
        if self._ratio_entries_b is None:
            return
        for entry in self._ratio_entries_b.values():
            entry.configure(state=state)
