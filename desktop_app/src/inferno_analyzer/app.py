"""Main application window for Inferno Analyzer.

Merges Suspension Analyzer and Driver Consistency Analyzer into a single
tabbed application with shared session panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

from motorsports_data_notebook.desktop.dpi import setup_hidpi_scaling
from motorsports_data_notebook.desktop.session_panel import SessionPanel
from motorsports_data_notebook.profiles import (
    get_profile_for_logger,
    save_profile_for_logger,
)

from inferno_analyzer.driver.widgets.config_panel import ConfigPanel as DriverConfigPanel
from inferno_analyzer.suspension.widgets.config_panel import ConfigPanel as SuspensionConfigPanel
from inferno_analyzer.tabs.driver_tab import DriverTab
from inferno_analyzer.tabs.suspension_tab import SuspensionTab
from inferno_analyzer.tabs.tire_grip_tab import TireGripTab
from inferno_analyzer.tire_grip.widgets.config_panel import ConfigPanel as TireGripConfigPanel

if TYPE_CHECKING:
    from motorsports_data_notebook._types import LogFile

    from inferno_analyzer.tabs.base_tab import BaseAnalysisTab


class InfernoAnalyzerApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window combining suspension, driver, and tire grip analysis.

    Layout:
    - Top row: Session A | Session B | Config Panel (swaps per tab)
    - Tab bar: [Suspension] [Driver Consistency] [Tire Grip]
    - Analysis area: fills remaining space (tab-specific content)
    """

    def __init__(self) -> None:
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        # Window setup
        self.title("Inferno Analyzer")
        self.geometry("1300x900")
        self.minsize(1000, 700)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # HiDPI scaling
        setup_hidpi_scaling(self)

        # Track active tab
        self._active_tab_name = "Suspension"

        # Collapse state for top panel
        self._top_collapsed = False

        # Build UI
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Top frame for session panels and config (all 3 side by side)
        self.top_frame = ctk.CTkFrame(self)

        # Session A panel (left)
        self.session_a_panel = SessionPanel(
            self.top_frame,
            title="SESSION A (Primary)",
            on_file_loaded=self._on_session_a_loaded,
            on_selection_changed=self._on_selection_changed,
            auto_select="top_103",
        )

        # Session B panel (middle)
        self.session_b_panel = SessionPanel(
            self.top_frame,
            title="SESSION B (Compare)",
            on_file_loaded=self._on_session_b_loaded,
            on_selection_changed=self._on_selection_changed,
            auto_select="top_103",
        )

        # Config panels — both created upfront, only one visible at a time
        # Config panel container (column 2 of top_frame)
        self._config_container = ctk.CTkFrame(self.top_frame, fg_color="transparent")

        self.suspension_config_panel = SuspensionConfigPanel(
            self._config_container,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        self.driver_config_panel = DriverConfigPanel(
            self._config_container,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        self.tire_grip_config_panel = TireGripConfigPanel(
            self._config_container,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        # Collapse button (top-right of top_frame)
        self._collapse_btn = ctk.CTkButton(
            self.top_frame,
            text="\u25b2",
            width=28,
            height=28,
            font=ctk.CTkFont(size=12),
            command=self._toggle_top_panel,
        )

        # Collapsed summary bar (hidden by default)
        self._collapsed_frame = ctk.CTkFrame(self, height=30)
        self._collapsed_label = ctk.CTkLabel(
            self._collapsed_frame,
            text="A: No file  |  B: No file",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self._expand_btn = ctk.CTkButton(
            self._collapsed_frame,
            text="\u25bc Expand",
            width=80,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._toggle_top_panel,
        )
        # Layout collapsed frame internals (frame itself not packed until collapse)
        self._collapsed_label.pack(side="left", fill="x", expand=True, padx=10)
        self._expand_btn.pack(side="right", padx=10)

        # Tab view for analysis area
        self.tabview = ctk.CTkTabview(
            self,
            command=self._on_tab_changed,
        )
        self.tabview.add("Suspension")
        self.tabview.add("Driver Consistency")
        self.tabview.add("Tire Grip")

        # Create tab instances
        self.suspension_tab = SuspensionTab(self, self.tabview.tab("Suspension"))
        self.driver_tab = DriverTab(self, self.tabview.tab("Driver Consistency"))
        self.tire_grip_tab = TireGripTab(self, self.tabview.tab("Tire Grip"))

    def _layout_widgets(self) -> None:
        """Arrange widgets in the window."""
        # Top frame
        self.top_frame.pack(fill="x", padx=10, pady=3)

        # Session panels + config container side by side
        self.session_a_panel.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        self.session_b_panel.grid(row=0, column=1, padx=3, pady=3, sticky="nsew")
        self._config_container.grid(row=0, column=2, padx=3, pady=3, sticky="nsew")
        self._collapse_btn.grid(row=0, column=3, padx=(0, 3), pady=3, sticky="ne")

        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_columnconfigure(2, weight=1)

        # Show suspension config by default
        self.suspension_config_panel.pack(fill="both", expand=True)

        # Tab view fills remaining space
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)

    # ------------------------------------------------------------------
    # Config panel swapping
    # ------------------------------------------------------------------

    def _swap_config_panel(self) -> None:
        """Show/hide config panels based on active tab."""
        panels = {
            "Suspension": self.suspension_config_panel,
            "Driver Consistency": self.driver_config_panel,
            "Tire Grip": self.tire_grip_config_panel,
        }
        for panel in panels.values():
            panel.pack_forget()
        active = panels.get(self._active_tab_name)
        if active:
            active.pack(fill="both", expand=True)

    def get_config_panel_for_tab(self, tab: BaseAnalysisTab):
        """Return the config panel associated with a tab."""
        if isinstance(tab, SuspensionTab):
            return self.suspension_config_panel
        if isinstance(tab, TireGripTab):
            return self.tire_grip_config_panel
        return self.driver_config_panel

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    def _on_tab_changed(self) -> None:
        """Handle tab switch."""
        new_tab_name = self.tabview.get()
        if new_tab_name == self._active_tab_name:
            return

        self._active_tab_name = new_tab_name
        self._swap_config_panel()

        # Notify newly active tab
        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab.on_tab_activated()

    def _get_active_tab(self) -> BaseAnalysisTab | None:
        """Return the currently active tab instance."""
        if self._active_tab_name == "Suspension":
            return self.suspension_tab
        elif self._active_tab_name == "Driver Consistency":
            return self.driver_tab
        elif self._active_tab_name == "Tire Grip":
            return self.tire_grip_tab
        return None

    # ------------------------------------------------------------------
    # Session callbacks — broadcast to tabs
    # ------------------------------------------------------------------

    def _on_session_a_loaded(self, log: LogFile, file_path: Path) -> None:
        """Handle Session A file loaded."""
        channels = sorted(log.channels.keys())
        self._channels_a = channels
        self._update_available_channels()

        # Auto-populate config from vehicle profile if logger ID is known
        logger_id = self.session_a_panel.logger_id
        profile = None
        if logger_id:
            profile = get_profile_for_logger(logger_id)
            if profile:
                self.suspension_config_panel.set_from_profile(profile)
                self.driver_config_panel.set_from_profile(profile)
                self.tire_grip_config_panel.set_from_profile(profile)

        # Notify all tabs (driver tab auto-sets throttle threshold after
        # profile has already set channel names)
        self.suspension_tab.on_session_loaded()
        self.driver_tab.on_session_a_loaded()
        self.tire_grip_tab.on_session_loaded()

        # Update status on active config panel
        active_tab = self._get_active_tab()
        if active_tab is not None:
            if profile:
                active_tab._update_status(f"Loaded: {file_path.name} (profile: {profile.name})")
            elif logger_id:
                active_tab._update_status(
                    f"Loaded: {file_path.name} (logger {logger_id}, no profile)"
                )
            else:
                active_tab._update_status(f"Loaded: {file_path.name}")

    def _on_session_b_loaded(self, log: LogFile, file_path: Path) -> None:
        """Handle Session B file loaded."""
        channels = sorted(log.channels.keys())
        self._channels_b = channels
        self._update_available_channels()

        # Show A/B session controls on all config panels
        self.suspension_config_panel.show_session_b_controls()
        self.driver_config_panel.show_session_b_controls()
        self.tire_grip_config_panel.show_session_b_controls()

        # Check if Session B has a different vehicle profile
        logger_id_b = self.session_b_panel.logger_id
        logger_id_a = self.session_a_panel.logger_id
        profile_b = None
        if logger_id_b:
            profile_b = get_profile_for_logger(logger_id_b)
            if profile_b and logger_id_b != logger_id_a:
                # Different vehicle — unsync and populate B from its profile
                self.suspension_config_panel.set_sync(False)
                self.suspension_config_panel.set_from_profile_b(profile_b)
                self.driver_config_panel.set_sync(False)
                self.driver_config_panel.set_from_profile_b(profile_b)
                self.tire_grip_config_panel.set_sync(False)
                self.tire_grip_config_panel.set_from_profile_b(profile_b)

        # Mark all tabs stale; driver tab also auto-sets B's throttle threshold
        self.suspension_tab.on_session_loaded()
        self.driver_tab.on_session_b_loaded()
        self.tire_grip_tab.on_session_loaded()

        active_tab = self._get_active_tab()
        if active_tab is not None:
            status = f"Loaded comparison: {file_path.name}"
            if profile_b and logger_id_b != logger_id_a:
                status += f" (profile: {profile_b.name})"
            active_tab._update_status(status)

    def _update_available_channels(self) -> None:
        """Update all config panels with per-session channel suggestions."""
        channels_a = getattr(self, "_channels_a", [])
        channels_b = getattr(self, "_channels_b", [])
        self.suspension_config_panel.update_available_channels_a(channels_a)
        self.suspension_config_panel.update_available_channels_b(channels_b)
        self.driver_config_panel.update_available_channels_a(channels_a)
        self.driver_config_panel.update_available_channels_b(channels_b)
        self.tire_grip_config_panel.update_available_channels_a(channels_a)
        self.tire_grip_config_panel.update_available_channels_b(channels_b)

    def _on_selection_changed(self) -> None:
        """Handle lap selection or config change — route to active tab."""
        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab.on_selection_changed()

    def _on_save_profile(self) -> None:
        """Save a merged vehicle profile from both config panels.

        Saves for whichever session (A or B) is active in the current
        config panel's segmented button.
        """
        active_tab = self._get_active_tab()
        active_config = self.get_config_panel_for_tab(active_tab) if active_tab else None

        # Determine which session to save for
        saving_b = active_config is not None and active_config.active_session == "B"

        if saving_b:
            logger_id = self.session_b_panel.logger_id
            session_panel = self.session_b_panel
        else:
            logger_id = self.session_a_panel.logger_id
            session_panel = self.session_a_panel

        if not logger_id:
            if active_tab is not None:
                active_tab._update_status("No logger ID — load a session first")
            return

        session_label = session_panel.get_session_label()

        if saving_b:
            profile = self.suspension_config_panel.get_vehicle_profile_b(name=session_label)
            driver_profile = self.driver_config_panel.get_vehicle_profile_b(name=session_label)
            tire_grip_profile = self.tire_grip_config_panel.get_vehicle_profile_b(
                name=session_label
            )
        else:
            profile = self.suspension_config_panel.get_vehicle_profile(name=session_label)
            driver_profile = self.driver_config_panel.get_vehicle_profile(name=session_label)
            tire_grip_profile = self.tire_grip_config_panel.get_vehicle_profile(name=session_label)

        profile.channel_names.update(driver_profile.channel_names)
        profile.channel_names.update(tire_grip_profile.channel_names)
        save_profile_for_logger(logger_id, profile)

        if active_tab is not None:
            active_tab._update_status(f"Profile saved for logger {logger_id}")

    # ------------------------------------------------------------------
    # Collapsible top panel
    # ------------------------------------------------------------------

    def _toggle_top_panel(self) -> None:
        """Toggle the top panel between expanded and collapsed states."""
        if self._top_collapsed:
            # Expand: swap collapsed bar -> full top frame
            self._collapsed_frame.pack_forget()
            self.tabview.pack_forget()
            self.top_frame.pack(fill="x", padx=10, pady=3)
            self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
            self._top_collapsed = False
        else:
            # Collapse: swap full top frame -> collapsed bar
            self._update_collapsed_summary()
            self.top_frame.pack_forget()
            self.tabview.pack_forget()
            self._collapsed_frame.pack(fill="x", padx=10, pady=2)
            self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
            self._top_collapsed = True

    def _update_collapsed_summary(self) -> None:
        """Update the collapsed bar text with current session info."""
        # Session A
        if self.session_a_panel.log is not None:
            a_name = self.session_a_panel._file_path.name if self.session_a_panel._file_path else "?"
            a_selected = len(self.session_a_panel.get_selected_laps())
            a_total = len(self.session_a_panel._lap_vars)
            a_text = f"A: {a_name} ({a_selected}/{a_total} laps)"
        else:
            a_text = "A: No file"

        # Session B
        if self.session_b_panel.log is not None:
            b_name = self.session_b_panel._file_path.name if self.session_b_panel._file_path else "?"
            b_selected = len(self.session_b_panel.get_selected_laps())
            b_total = len(self.session_b_panel._lap_vars)
            b_text = f"B: {b_name} ({b_selected}/{b_total} laps)"
        else:
            b_text = "B: No file"

        self._collapsed_label.configure(text=f"{a_text}  |  {b_text}")

