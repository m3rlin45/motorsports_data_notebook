"""Base configuration panel widget with shared channel entry and status widgets."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.autocomplete_entry import AutocompleteEntry


class BaseConfigPanel(ctk.CTkFrame):
    """Base panel for configuration with channel entries and status display.

    Provides title label, channel name entries with autocomplete,
    reset button, and status/statistics controls. Subclasses add
    domain-specific sections and call layout methods.

    Supports per-session (A/B) channel entries for cross-vehicle comparison.
    Session B entries are created lazily when needed. A sync checkbox lets
    users keep B in sync with A (default) or edit independently.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        channel_defaults: dict[str, str],
        channel_display_names: dict[str, str],
        on_stats_click: Callable[[], None] | None = None,
        on_config_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_channels = channel_defaults
        self._channel_display_names = channel_display_names
        self._on_stats_click = on_stats_click
        self._on_config_changed = on_config_changed

        # Session state
        self._active_session: str = "A"
        self._sync_with_a: bool = True
        self._sync_var = tk.BooleanVar(value=True)

        # Suggestion lists per session
        self._suggestions_a: list[str] = []
        self._suggestions_b: list[str] = []

        # B entries (lazily created)
        self._channel_entries_b: dict[str, AutocompleteEntry] | None = None
        self._channel_labels_b: dict[str, ctk.CTkLabel] | None = None

        self._create_base_widgets(channel_display_names)

    def _create_base_widgets(self, channel_display_names: dict[str, str]) -> None:
        """Create title, channel entries, and status widgets."""
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

        # Session A channel name entries
        self._channel_entries_a: dict[str, AutocompleteEntry] = {}
        self._channel_labels_a: dict[str, ctk.CTkLabel] = {}

        for key, display_name in channel_display_names.items():
            self._channel_labels_a[key] = ctk.CTkLabel(
                self.channels_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self._channel_entries_a[key] = AutocompleteEntry(
                self.channels_frame,
                width=120,
                font=ctk.CTkFont(size=11),
                on_change=self._on_entry_changed,
            )
            self._channel_entries_a[key].insert(0, self._default_channels[key])

        # Reset channels button
        self.reset_channels_btn = ctk.CTkButton(
            self.channels_frame,
            text="Reset to Default",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_channels,
        )

        # Session selector (hidden initially)
        self._session_selector = ctk.CTkSegmentedButton(
            self.channels_frame,
            values=["Session A", "Session B"],
            command=self._on_session_selector_changed,
            font=ctk.CTkFont(size=10),
            height=24,
        )
        self._session_selector.set("Session A")

        # Sync checkbox (hidden initially)
        self._sync_checkbox = ctk.CTkCheckBox(
            self.channels_frame,
            text="Sync with A",
            variable=self._sync_var,
            command=self._on_sync_toggled,
            font=ctk.CTkFont(size=10),
            height=20,
            checkbox_width=16,
            checkbox_height=16,
        )

        # Status and actions frame
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.stats_btn = ctk.CTkButton(
            self.status_frame,
            text="Show Statistics",
            width=110,
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._on_stats_btn_click,
        )

    # ------------------------------------------------------------------
    # Backward-compatible properties: channel_entries / channel_labels
    # return the active session's entries so subclass layout code works.
    # ------------------------------------------------------------------

    @property
    def channel_entries(self) -> dict[str, AutocompleteEntry]:
        """Return the active session's channel entries."""
        if self._active_session == "B" and self._channel_entries_b is not None:
            return self._channel_entries_b
        return self._channel_entries_a

    @property
    def channel_labels(self) -> dict[str, ctk.CTkLabel]:
        """Return the active session's channel labels."""
        if self._active_session == "B" and self._channel_labels_b is not None:
            return self._channel_labels_b
        return self._channel_labels_a

    # ------------------------------------------------------------------
    # Channel grid layout
    # ------------------------------------------------------------------

    def _pack_channels(self) -> None:
        """Pack the channels frame and lay out the channel grid."""
        self.channels_frame.pack(fill="x", padx=5, pady=2)
        self._repack_channel_grid()

    def _repack_channel_grid(self) -> None:
        """Re-grid channel labels/entries for the active session.

        Layout:
          Row 0: "Channel Names:" label
          Row 1: session selector + sync checkbox (only when B entries exist)
          Row 2+: channel label/entry pairs in 2-column layout
          Last row: reset button

        Sets ``self._channel_button_row`` for subclass save-button placement.
        """
        # Ungrid everything in channels_frame
        for widget in self.channels_frame.winfo_children():
            widget.grid_forget()

        row = 0
        # Row 0: section header
        self.channels_label.grid(row=row, column=0, columnspan=4, sticky="w", padx=2, pady=1)
        row += 1

        # Row 1: session selector + sync (only when B entries exist)
        if self._channel_entries_b is not None:
            self._session_selector.grid(row=row, column=0, columnspan=3, padx=2, pady=2, sticky="w")
            if self._active_session == "B":
                self._sync_checkbox.grid(row=row, column=3, padx=2, pady=2, sticky="e")
            row += 1

        # Active entries
        entries = self.channel_entries
        labels = self.channel_labels

        keys = list(entries.keys())
        for i, key in enumerate(keys):
            grid_row = (i // 2) + row
            col_offset = (i % 2) * 2
            padx_label = (2, 1) if col_offset == 0 else (5, 1)
            labels[key].grid(row=grid_row, column=col_offset, padx=padx_label, pady=1, sticky="e")
            entries[key].grid(row=grid_row, column=col_offset + 1, padx=1, pady=1, sticky="w")

        num_entry_rows = (len(keys) + 1) // 2
        button_row = row + num_entry_rows
        self.reset_channels_btn.grid(row=button_row, column=0, columnspan=4, pady=2)

        # Expose for subclass save-button placement
        self._channel_button_row = button_row

    # ------------------------------------------------------------------
    # Lazy B entry creation
    # ------------------------------------------------------------------

    def _ensure_b_entries(self) -> None:
        """Create Session B channel entries if they don't exist yet."""
        if self._channel_entries_b is not None:
            return

        self._channel_entries_b = {}
        self._channel_labels_b = {}

        for key, display_name in self._channel_display_names.items():
            self._channel_labels_b[key] = ctk.CTkLabel(
                self.channels_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self._channel_entries_b[key] = AutocompleteEntry(
                self.channels_frame,
                width=120,
                font=ctk.CTkFont(size=11),
                on_change=self._on_entry_changed,
            )
            # Initialize from A's current values
            self._channel_entries_b[key].insert(0, self._channel_entries_a[key].get())

        # Apply current suggestions for B
        if self._suggestions_b:
            for entry in self._channel_entries_b.values():
                entry.set_suggestions(self._suggestions_b)

        # Let subclasses create their B widgets
        self._create_b_extra()

        # Apply sync state (disable B entries if synced)
        self._apply_sync_state()

    # ------------------------------------------------------------------
    # Session switching
    # ------------------------------------------------------------------

    def show_session_b_controls(self) -> None:
        """Show the session A/B selector and create B entries if needed."""
        self._ensure_b_entries()
        self._repack_channel_grid()

    def _on_session_selector_changed(self, value: str) -> None:
        """Handle session selector toggle."""
        new_session = "A" if value == "Session A" else "B"
        if new_session == self._active_session:
            return
        self._active_session = new_session
        self._repack_channel_grid()
        self._repack_extra_widgets()

    # ------------------------------------------------------------------
    # Sync logic
    # ------------------------------------------------------------------

    def _on_sync_toggled(self) -> None:
        """Handle sync checkbox toggle."""
        self._sync_with_a = self._sync_var.get()
        if self._sync_with_a:
            self._copy_a_to_b()
        self._apply_sync_state()

    def _copy_a_to_b(self) -> None:
        """Copy all A entry values to B entries."""
        if self._channel_entries_b is None:
            return
        for key in self._channel_entries_a:
            val = self._channel_entries_a[key].get()
            self._channel_entries_b[key].delete(0, "end")
            self._channel_entries_b[key].insert(0, val)
        self._copy_extra_a_to_b()

    def _apply_sync_state(self) -> None:
        """Enable or disable B entries based on sync state."""
        if self._channel_entries_b is None:
            return
        state = "disabled" if self._sync_with_a else "normal"
        for entry in self._channel_entries_b.values():
            entry.configure(state=state)
        self._apply_extra_sync_state(state)

    def set_sync(self, synced: bool) -> None:
        """Programmatically set the sync state."""
        self._sync_with_a = synced
        self._sync_var.set(synced)
        if synced:
            self._copy_a_to_b()
        self._apply_sync_state()

    # ------------------------------------------------------------------
    # Entry change handler
    # ------------------------------------------------------------------

    def _on_entry_changed(self, event=None) -> None:
        """Handle any config entry value change."""
        # If editing A while synced, propagate to B
        if self._active_session == "A" and self._sync_with_a:
            self._copy_a_to_b()
        if self._on_config_changed:
            self._on_config_changed()

    # ------------------------------------------------------------------
    # Channel reset
    # ------------------------------------------------------------------

    def _reset_channels(self) -> None:
        """Reset channel names to defaults for the active session."""
        entries = self.channel_entries
        for key, val in self._default_channels.items():
            entries[key].delete(0, "end")
            entries[key].insert(0, val)
        # If we reset A and sync is on, copy to B
        if self._active_session == "A" and self._sync_with_a:
            self._copy_a_to_b()
        self._on_entry_changed()

    # ------------------------------------------------------------------
    # Channel name getters
    # ------------------------------------------------------------------

    def get_channel_names(self) -> dict[str, str]:
        """Get channel names for Session A (backward compat)."""
        return self.get_channel_names_a()

    def get_channel_names_a(self) -> dict[str, str]:
        """Get current channel name mappings for Session A."""
        return {
            key: self._channel_entries_a[key].get() or self._default_channels[key]
            for key in self._default_channels
        }

    def get_channel_names_b(self) -> dict[str, str]:
        """Get current channel name mappings for Session B.

        Returns A's values if synced or if B entries don't exist.
        """
        if self._sync_with_a or self._channel_entries_b is None:
            return self.get_channel_names_a()
        return {
            key: self._channel_entries_b[key].get() or self._default_channels[key]
            for key in self._default_channels
        }

    # ------------------------------------------------------------------
    # Autocomplete suggestions
    # ------------------------------------------------------------------

    def update_available_channels(self, channel_names: list[str]) -> None:
        """Update autocomplete suggestions for Session A (backward compat)."""
        self.update_available_channels_a(channel_names)

    def update_available_channels_a(self, channels: list[str]) -> None:
        """Update autocomplete suggestions for Session A entries."""
        self._suggestions_a = channels
        for entry in self._channel_entries_a.values():
            entry.set_suggestions(channels)

    def update_available_channels_b(self, channels: list[str]) -> None:
        """Update autocomplete suggestions for Session B entries."""
        self._suggestions_b = channels
        if self._channel_entries_b is not None:
            for entry in self._channel_entries_b.values():
                entry.set_suggestions(channels)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_session(self) -> str:
        """Return the currently selected session ('A' or 'B')."""
        return self._active_session

    @property
    def is_synced(self) -> bool:
        """Return whether B entries are synced with A."""
        return self._sync_with_a

    # ------------------------------------------------------------------
    # Subclass hooks for extra A/B widgets (motion ratios, thresholds)
    # ------------------------------------------------------------------

    def _create_b_extra(self) -> None:
        """Create Session B versions of subclass-specific widgets.

        Called once from ``_ensure_b_entries()``. Override in subclasses
        that have additional per-session parameters.
        """

    def _repack_extra_widgets(self) -> None:
        """Re-grid subclass-specific widgets for the active session.

        Called after session selector changes. Override in subclasses.
        """

    def _copy_extra_a_to_b(self) -> None:
        """Copy subclass-specific A values to B entries.

        Called when sync copies A→B. Override in subclasses.
        """

    def _apply_extra_sync_state(self, state: str) -> None:
        """Enable or disable subclass-specific B entries.

        Parameters
        ----------
        state : str
            "disabled" when synced, "normal" when independent.
        """

    # ------------------------------------------------------------------
    # Status / stats
    # ------------------------------------------------------------------

    def _pack_status(self) -> None:
        """Pack the status frame with label and stats button."""
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
