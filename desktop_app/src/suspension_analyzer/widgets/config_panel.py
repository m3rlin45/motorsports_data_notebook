"""Configuration panel for motion ratios and channel names."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from motorsports_data_notebook.desktop.config_panel import BaseConfigPanel
from motorsports_data_notebook.suspension import (
    MotionRatios,
    SUSPENSION_CHANNEL_NAMES,
)

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
            channel_defaults=SUSPENSION_CHANNEL_NAMES.copy(),
            channel_display_names=_CHANNEL_DISPLAY_NAMES,
            on_stats_click=on_stats_click,
            on_config_changed=on_config_changed,
        )
        self._default_ratios = MotionRatios.toyota_86_zn6()
        self._create_ratio_widgets()
        self._layout_widgets()

    def _create_ratio_widgets(self) -> None:
        """Create motion ratio input widgets."""
        self.ratios_frame = ctk.CTkFrame(self)
        self.ratios_label = ctk.CTkLabel(
            self.ratios_frame,
            text="Motion Ratios:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

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
            self.ratio_entries[corner].bind("<KeyRelease>", self._on_entry_changed)

        self.reset_ratios_btn = ctk.CTkButton(
            self.ratios_frame,
            text="Reset to Toyota 86",
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._reset_ratios,
        )

    def _layout_widgets(self) -> None:
        """Arrange all widgets in the panel."""
        self.title_label.pack(anchor="w", padx=5, pady=(2, 5))

        # Motion ratios frame
        self.ratios_frame.pack(fill="x", padx=5, pady=2)
        self.ratios_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=2, pady=1)

        self.ratio_labels["FL"].grid(row=1, column=0, padx=(2, 1), pady=1, sticky="e")
        self.ratio_entries["FL"].grid(row=1, column=1, padx=1, pady=1)
        self.ratio_labels["FR"].grid(row=1, column=2, padx=(5, 1), pady=1, sticky="e")
        self.ratio_entries["FR"].grid(row=1, column=3, padx=1, pady=1)

        self.ratio_labels["RL"].grid(row=2, column=0, padx=(2, 1), pady=1, sticky="e")
        self.ratio_entries["RL"].grid(row=2, column=1, padx=1, pady=1)
        self.ratio_labels["RR"].grid(row=2, column=2, padx=(5, 1), pady=1, sticky="e")
        self.ratio_entries["RR"].grid(row=2, column=3, padx=1, pady=1)

        self.reset_ratios_btn.grid(row=3, column=0, columnspan=4, pady=2)

        # Channels + status
        self._pack_channels()
        self._pack_status()

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
        self._on_entry_changed()

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
            # Return defaults if parsing fails
            return MotionRatios.toyota_86_zn6()
