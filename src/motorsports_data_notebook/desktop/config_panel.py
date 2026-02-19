"""Base configuration panel widget with shared channel entry and status widgets."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.autocomplete_entry import AutocompleteEntry


class BaseConfigPanel(ctk.CTkFrame):
    """Base panel for configuration with channel entries and status display.

    Provides title label, channel name entries with autocomplete,
    reset button, and status/statistics controls. Subclasses add
    domain-specific sections and call layout methods.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        channel_defaults: dict[str, str],
        channel_display_names: dict[str, str],
        on_stats_click: Callable[[], None] | None = None,
        on_config_changed: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the config panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        channel_defaults : dict[str, str]
            Default channel name mappings (key -> default value).
        channel_display_names : dict[str, str]
            Display labels for channel entries (key -> label text).
        on_stats_click : callable, optional
            Callback when statistics button is clicked.
        on_config_changed : callable, optional
            Callback when any configuration value changes.
        """
        super().__init__(parent)
        self._default_channels = channel_defaults
        self._on_stats_click = on_stats_click
        self._on_config_changed = on_config_changed
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

        # Channel name entries
        self.channel_entries: dict[str, AutocompleteEntry] = {}
        self.channel_labels: dict[str, ctk.CTkLabel] = {}

        for key, display_name in channel_display_names.items():
            self.channel_labels[key] = ctk.CTkLabel(
                self.channels_frame,
                text=display_name,
                font=ctk.CTkFont(size=11),
            )
            self.channel_entries[key] = AutocompleteEntry(
                self.channels_frame,
                width=120,
                font=ctk.CTkFont(size=11),
                on_change=self._on_entry_changed,
            )
            self.channel_entries[key].insert(0, self._default_channels[key])

        # Reset channels button
        self.reset_channels_btn = ctk.CTkButton(
            self.channels_frame,
            text="Reset to Default",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_channels,
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

    def _pack_channels(self) -> None:
        """Pack the channels frame with dynamic 2-column grid layout."""
        self.channels_frame.pack(fill="x", padx=5, pady=2)
        self.channels_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        keys = list(self.channel_entries.keys())
        for i, key in enumerate(keys):
            row = (i // 2) + 1
            col_offset = (i % 2) * 2
            padx_label = (2, 1) if col_offset == 0 else (5, 1)
            self.channel_labels[key].grid(
                row=row, column=col_offset, padx=padx_label, pady=1, sticky="e"
            )
            self.channel_entries[key].grid(
                row=row, column=col_offset + 1, padx=1, pady=1, sticky="w"
            )

        num_rows = (len(keys) + 1) // 2
        self.reset_channels_btn.grid(row=num_rows + 1, column=0, columnspan=4, pady=2)

    def _pack_status(self) -> None:
        """Pack the status frame with label and stats button."""
        self.status_frame.pack(fill="x", padx=5, pady=(5, 2))
        self.status_label.pack(side="left", fill="x", expand=True, padx=2)
        self.stats_btn.pack(side="right", padx=2)

    def _on_entry_changed(self, event=None) -> None:
        """Handle any config entry value change."""
        if self._on_config_changed:
            self._on_config_changed()

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

    def update_available_channels(self, channel_names: list[str]) -> None:
        """Update autocomplete suggestions for all channel entries."""
        for entry in self.channel_entries.values():
            entry.set_suggestions(channel_names)

    def _reset_channels(self) -> None:
        """Reset channel names to defaults."""
        for key, val in self._default_channels.items():
            self.channel_entries[key].delete(0, "end")
            self.channel_entries[key].insert(0, val)
        self._on_entry_changed()

    def get_channel_names(self) -> dict[str, str]:
        """Get current channel name mappings."""
        return {
            key: self.channel_entries[key].get() or self._default_channels[key]
            for key in self._default_channels
        }
