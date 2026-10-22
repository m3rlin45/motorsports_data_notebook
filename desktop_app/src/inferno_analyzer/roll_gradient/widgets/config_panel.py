"""Configuration panel for roll gradient channel names and track widths.

Pot/wheel motion ratios are taken from the Suspension tab's config panel so
there is a single place to set them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel
from motorsports_data_notebook.profiles import DEFAULT_CHANNEL_NAMES
from motorsports_data_notebook.roll_gradient import TrackWidths

if TYPE_CHECKING:
    from motorsports_data_notebook.profiles import VehicleProfile

# Display labels for channels used by the roll gradient analysis
_CHANNEL_DISPLAY_NAMES = {
    "shock_fl": "FL Shock:",
    "shock_fr": "FR Shock:",
    "shock_rl": "RL Shock:",
    "shock_rr": "RR Shock:",
    "lateral_g": "Lat G:",
    "gps_latitude": "GPS Lat:",
    "gps_longitude": "GPS Lon:",
}

_TRACK_AXLES = [("Front", "front_mm"), ("Rear", "rear_mm")]


class ConfigPanel(BaseConfigPanel):
    """Panel for configuring roll gradient channels and front/rear track widths."""

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
            channel_defaults={k: DEFAULT_CHANNEL_NAMES[k] for k in _CHANNEL_DISPLAY_NAMES},
            channel_display_names=_CHANNEL_DISPLAY_NAMES,
            on_config_changed=on_config_changed,
        )
        self._create_track_widgets()
        self._create_save_profile_btn()
        self._layout_widgets()

    # ------------------------------------------------------------------
    # Track width widgets
    # ------------------------------------------------------------------

    def _create_track_widgets(self) -> None:
        """Create track width entries (Session A) and the motion ratio note."""
        self.track_frame = ctk.CTkFrame(self)
        self.track_label = ctk.CTkLabel(
            self.track_frame,
            text="Track Width (mm, hub to hub):",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.ratio_note = ctk.CTkLabel(
            self.track_frame,
            text="Pot/wheel ratios: Suspension tab",
            font=ctk.CTkFont(size=10),
            text_color="#aaaaaa",
        )

        self._track_entries_a: dict[str, ctk.CTkEntry] = {}
        self._track_labels_a: dict[str, ctk.CTkLabel] = {}
        self._track_entries_b: dict[str, ctk.CTkEntry] | None = None
        self._track_labels_b: dict[str, ctk.CTkLabel] | None = None

        for axle, _ in _TRACK_AXLES:
            self._track_labels_a[axle] = ctk.CTkLabel(
                self.track_frame, text=f"{axle}:", font=ctk.CTkFont(size=11)
            )
            entry = ctk.CTkEntry(self.track_frame, width=80, font=ctk.CTkFont(size=11))
            entry.bind("<KeyRelease>", self._on_entry_changed)
            self._track_entries_a[axle] = entry

    @property
    def track_entries(self) -> dict[str, ctk.CTkEntry]:
        """Return the active session's track width entries."""
        if self._active_session == "B" and self._track_entries_b is not None:
            return self._track_entries_b
        return self._track_entries_a

    @property
    def track_labels(self) -> dict[str, ctk.CTkLabel]:
        """Return the active session's track width labels."""
        if self._active_session == "B" and self._track_labels_b is not None:
            return self._track_labels_b
        return self._track_labels_a

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
        self.title_label.pack(anchor="w", padx=5, pady=(2, 2))

        self.track_frame.pack(fill="x", padx=5, pady=2)
        self._repack_track_grid()

        self._pack_channels()
        self.reset_channels_btn.grid_configure(columnspan=2)
        self.save_profile_btn.grid(row=self._channel_button_row, column=2, columnspan=2, pady=2)

        self._pack_status()

    def _repack_track_grid(self) -> None:
        """Re-grid the active session's track width labels/entries."""
        for widget in self.track_frame.winfo_children():
            widget.grid_forget()

        self.track_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)
        labels = self.track_labels
        entries = self.track_entries
        labels["Front"].grid(row=1, column=0, padx=(2, 1), pady=1, sticky="e")
        entries["Front"].grid(row=1, column=1, padx=1, pady=1)
        labels["Rear"].grid(row=1, column=2, padx=(5, 1), pady=1, sticky="e")
        entries["Rear"].grid(row=1, column=3, padx=1, pady=1)
        self.ratio_note.grid(row=2, column=0, columnspan=4, sticky="w", padx=2, pady=(1, 2))

    def _on_save_profile_click(self) -> None:
        """Handle Save Profile button click."""
        if self._on_save_profile:
            self._on_save_profile()

    # ------------------------------------------------------------------
    # Track width getters
    # ------------------------------------------------------------------

    @staticmethod
    def _widths_from_entries(entries: dict[str, ctk.CTkEntry]) -> TrackWidths | None:
        """Parse TrackWidths from entries; None if empty or invalid."""
        try:
            return TrackWidths(
                front_mm=float(entries["Front"].get()),
                rear_mm=float(entries["Rear"].get()),
            )
        except ValueError:
            return None

    def get_track_widths_a(self) -> TrackWidths | None:
        """Session A track widths, or None if not entered/invalid."""
        return self._widths_from_entries(self._track_entries_a)

    def get_track_widths_b(self) -> TrackWidths | None:
        """Session B track widths (A's values if synced or B entries don't exist)."""
        if self._sync_with_a or self._track_entries_b is None:
            return self.get_track_widths_a()
        return self._widths_from_entries(self._track_entries_b)

    @staticmethod
    def _set_entries(entries: dict[str, ctk.CTkEntry], widths: TrackWidths | None) -> None:
        for axle, attr in _TRACK_AXLES:
            entries[axle].delete(0, "end")
            if widths is not None:
                entries[axle].insert(0, f"{getattr(widths, attr):.0f}")

    # ------------------------------------------------------------------
    # Profile support
    # ------------------------------------------------------------------

    def set_from_profile(self, profile: VehicleProfile) -> None:
        """Populate Session A fields from a VehicleProfile."""
        for key, entry in self._channel_entries_a.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])
        self._set_entries(self._track_entries_a, profile.track_widths)
        if self._sync_with_a:
            self._copy_extra_a_to_b()

    def set_from_profile_b(self, profile: VehicleProfile) -> None:
        """Populate Session B fields from a VehicleProfile."""
        self._ensure_b_entries()
        assert self._channel_entries_b is not None
        assert self._track_entries_b is not None

        for key, entry in self._channel_entries_b.items():
            if key in profile.channel_names:
                entry.delete(0, "end")
                entry.insert(0, profile.channel_names[key])
        self._set_entries(self._track_entries_b, profile.track_widths)

    def get_vehicle_profile(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from Session A's channel names and track widths."""
        from motorsports_data_notebook.profiles import VehicleProfile
        from motorsports_data_notebook.suspension import MotionRatios

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_a(),
            motion_ratios=MotionRatios(),
            track_widths=self.get_track_widths_a(),
        )

    def get_vehicle_profile_b(self, name: str) -> VehicleProfile:
        """Build a VehicleProfile from Session B's channel names and track widths."""
        from motorsports_data_notebook.profiles import VehicleProfile
        from motorsports_data_notebook.suspension import MotionRatios

        return VehicleProfile(
            name=name,
            channel_names=self.get_channel_names_b(),
            motion_ratios=MotionRatios(),
            track_widths=self.get_track_widths_b(),
        )

    # ------------------------------------------------------------------
    # BaseConfigPanel hooks for per-session track widths
    # ------------------------------------------------------------------

    def _create_b_extra(self) -> None:
        """Create Session B track width entries, initialised from A."""
        self._track_entries_b = {}
        self._track_labels_b = {}
        for axle, _ in _TRACK_AXLES:
            self._track_labels_b[axle] = ctk.CTkLabel(
                self.track_frame, text=f"{axle}:", font=ctk.CTkFont(size=11)
            )
            entry = ctk.CTkEntry(self.track_frame, width=80, font=ctk.CTkFont(size=11))
            entry.insert(0, self._track_entries_a[axle].get())
            entry.bind("<KeyRelease>", self._on_entry_changed)
            self._track_entries_b[axle] = entry

    def _repack_extra_widgets(self) -> None:
        """Swap track width entries when the session selector changes."""
        self._repack_track_grid()

    def _copy_extra_a_to_b(self) -> None:
        """Copy A track widths to B."""
        if self._track_entries_b is None:
            return
        for axle in self._track_entries_a:
            val = self._track_entries_a[axle].get()
            self._track_entries_b[axle].delete(0, "end")
            self._track_entries_b[axle].insert(0, val)

    def _apply_extra_sync_state(self, state: str) -> None:
        """Enable/disable B track width entries."""
        if self._track_entries_b is None:
            return
        for entry in self._track_entries_b.values():
            entry.configure(state=state)
