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

if TYPE_CHECKING:
    from inferno_analyzer.tabs.base_tab import BaseAnalysisTab
    from libxrk.base import LogFile


class InfernoAnalyzerApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window combining suspension and driver analysis.

    Layout:
    - Top row: Session A | Session B | Config Panel (swaps per tab)
    - Tab bar: [Suspension] [Driver Consistency]
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
            on_stats_click=self._on_stats_click,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        self.driver_config_panel = DriverConfigPanel(
            self._config_container,
            on_stats_click=self._on_stats_click,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        # Tab view for analysis area
        self.tabview = ctk.CTkTabview(
            self,
            command=self._on_tab_changed,
        )
        self.tabview.add("Suspension")
        self.tabview.add("Driver Consistency")

        # Create tab instances
        self.suspension_tab = SuspensionTab(self, self.tabview.tab("Suspension"))
        self.driver_tab = DriverTab(self, self.tabview.tab("Driver Consistency"))

    def _layout_widgets(self) -> None:
        """Arrange widgets in the window."""
        # Top frame
        self.top_frame.pack(fill="x", padx=10, pady=5)

        # Session panels + config container side by side
        self.session_a_panel.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        self.session_b_panel.grid(row=0, column=1, padx=3, pady=3, sticky="nsew")
        self._config_container.grid(row=0, column=2, padx=3, pady=3, sticky="nsew")

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
        if self._active_tab_name == "Suspension":
            self.driver_config_panel.pack_forget()
            self.suspension_config_panel.pack(fill="both", expand=True)
        else:
            self.suspension_config_panel.pack_forget()
            self.driver_config_panel.pack(fill="both", expand=True)

    def get_config_panel_for_tab(self, tab: BaseAnalysisTab):
        """Return the config panel associated with a tab."""
        if isinstance(tab, SuspensionTab):
            return self.suspension_config_panel
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

        # Notify all tabs (driver tab auto-sets throttle threshold after
        # profile has already set channel names)
        self.suspension_tab.on_session_loaded()
        self.driver_tab.on_session_a_loaded()

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

        # Mark both tabs stale
        self.suspension_tab.on_session_loaded()
        self.driver_tab.on_session_loaded()

        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab._update_status(f"Loaded comparison: {file_path.name}")

    def _update_available_channels(self) -> None:
        """Update both config panels with channels from all loaded sessions."""
        channels_a = getattr(self, "_channels_a", [])
        channels_b = getattr(self, "_channels_b", [])
        merged = sorted(set(channels_a) | set(channels_b))
        self.suspension_config_panel.update_available_channels(merged)
        self.driver_config_panel.update_available_channels(merged)

    def _on_selection_changed(self) -> None:
        """Handle lap selection or config change — route to active tab."""
        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab.on_selection_changed()

    def _on_stats_click(self) -> None:
        """Handle stats button click — route to active tab."""
        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab.toggle_stats_window()

    def _on_save_profile(self) -> None:
        """Save a merged vehicle profile from both config panels."""
        logger_id = self.session_a_panel.logger_id
        if not logger_id:
            active_tab = self._get_active_tab()
            if active_tab is not None:
                active_tab._update_status("No logger ID — load a session first")
            return

        # Build profile from suspension config panel (has motion ratios + shock channels)
        session_label = self.session_a_panel.get_session_label()
        profile = self.suspension_config_panel.get_vehicle_profile(name=session_label)

        # Merge in driver config panel channels (throttle, brake, GPS, etc.)
        driver_profile = self.driver_config_panel.get_vehicle_profile(name=session_label)
        profile.channel_names.update(driver_profile.channel_names)

        save_profile_for_logger(logger_id, profile)

        active_tab = self._get_active_tab()
        if active_tab is not None:
            active_tab._update_status(f"Profile saved for logger {logger_id}")

    # ------------------------------------------------------------------
    # Chart maximize (shared by both tabs)
    # ------------------------------------------------------------------

    def on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle (called by tabs)."""
        if maximized:
            self.top_frame.pack_forget()
            self.tabview._segmented_button.pack_forget()  # Hide tab bar
        else:
            # Restore normal layout
            self.tabview.pack_forget()
            self.top_frame.pack(fill="x", padx=10, pady=5)
            self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
            self.tabview._segmented_button.pack(fill="x", padx=10, pady=0)
