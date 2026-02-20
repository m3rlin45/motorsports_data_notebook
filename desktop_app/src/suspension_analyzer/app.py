"""Main application window for Suspension Analyzer."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

from motorsports_data_notebook.desktop.dpi import setup_hidpi_scaling
from motorsports_data_notebook.profiles import (
    get_profile_for_logger,
    save_profile_for_logger,
)
from motorsports_data_notebook.suspension import (
    MotionRatios,
    VelocityHistogramResult,
)

from motorsports_data_notebook.desktop.session_panel import SessionPanel

from suspension_analyzer.analysis.multi_lap import analyze_suspension_velocity_multi_lap
from suspension_analyzer.widgets.chart_view import ChartView
from suspension_analyzer.widgets.config_panel import ConfigPanel
from suspension_analyzer.widgets.stats_panel import StatsPanel

if TYPE_CHECKING:
    from libxrk.base import LogFile


class SuspensionAnalyzerApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window for Suspension Analyzer.

    Provides a GUI for analyzing suspension velocity histograms from
    XRK/XRZ telemetry files with support for multi-lap analysis and
    session comparison.
    """

    def __init__(self) -> None:
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        # Window setup
        self.title("Suspension Analyzer")
        self.geometry("1200x900")
        self.minsize(900, 700)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # HiDPI scaling - detect and apply for Linux/WSLg
        setup_hidpi_scaling(self)

        # Analysis results storage
        self._result_a: VelocityHistogramResult | None = None
        self._result_b: VelocityHistogramResult | None = None

        # Debounce timer for auto-analysis
        self._analysis_timer: str | None = None
        self._analyzing = False

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
        )

        # Session B panel (middle, always visible, comparison auto-enables on file load)
        self.session_b_panel = SessionPanel(
            self.top_frame,
            title="SESSION B (Compare)",
            on_file_loaded=self._on_session_b_loaded,
            on_selection_changed=self._on_selection_changed,
        )

        # Config panel (right) with stats and save profile callbacks
        self._stats_window: ctk.CTkToplevel | None = None
        self.config_panel = ConfigPanel(
            self.top_frame,
            on_stats_click=self._toggle_stats_window,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        # Chart view (middle) with maximize callback
        self.chart_view = ChartView(self, on_maximize_toggle=self._on_chart_maximize)

        # Stats panel (created in popup window when needed)
        self.stats_panel: StatsPanel | None = None

    def _layout_widgets(self) -> None:
        """Arrange widgets in the window."""
        # Top frame
        self.top_frame.pack(fill="x", padx=10, pady=5)

        # All 3 panels side by side in row 0
        self.session_a_panel.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        self.session_b_panel.grid(row=0, column=1, padx=3, pady=3, sticky="nsew")
        self.config_panel.grid(row=0, column=2, padx=3, pady=3, sticky="nsew")

        # Configure top frame grid weights (equal width for all 3)
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_columnconfigure(2, weight=1)

        # Chart view - middle
        self.chart_view.pack(fill="both", expand=True, padx=10, pady=5)

        # Stats panel - hidden initially (toggled via button)

    def _toggle_stats_window(self) -> None:
        """Toggle stats window visibility."""
        # Check if window exists and is valid
        try:
            if self._stats_window is not None and self._stats_window.winfo_exists():
                # Window exists, close it
                self._stats_window.destroy()
                self._stats_window = None
                self.stats_panel = None
                self.config_panel.set_stats_button_text("Show Statistics")
                return
        except Exception:
            # Window was destroyed or invalid, reset state
            self._stats_window = None
            self.stats_panel = None

        # Create new stats window
        self._stats_window = ctk.CTkToplevel(self)
        self._stats_window.title("Statistics")
        self._stats_window.geometry("700x600")
        self._stats_window.minsize(500, 400)

        # Create stats panel in the popup
        self.stats_panel = StatsPanel(self._stats_window)
        self.stats_panel.pack(fill="both", expand=True, padx=10, pady=10)

        # Update stats if we have results
        if self._result_a is not None:
            self._display_stats()

        # Handle window close
        self._stats_window.protocol("WM_DELETE_WINDOW", self._on_stats_window_close)
        self.config_panel.set_stats_button_text("Hide Statistics")

        # Bring window to front after a short delay (needed for CTkToplevel)
        self._stats_window.after(100, self._bring_stats_to_front)

    def _bring_stats_to_front(self) -> None:
        """Bring stats window to front."""
        if self._stats_window is not None:
            try:
                self._stats_window.lift()
                self._stats_window.focus_force()
            except Exception:
                pass

    def _on_stats_window_close(self) -> None:
        """Handle stats window being closed."""
        if self._stats_window is not None:
            self._stats_window.destroy()
            self._stats_window = None
            self.stats_panel = None
            self.config_panel.set_stats_button_text("Show Statistics")

    def _on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle."""
        if maximized:
            # Hide top frame
            self.top_frame.pack_forget()
            # Re-pack chart to fill window
            self.chart_view.pack_forget()
            self.chart_view.pack(fill="both", expand=True, padx=5, pady=5)
        else:
            # Restore normal layout
            self.chart_view.pack_forget()
            self.top_frame.pack(fill="x", padx=10, pady=5)
            self.chart_view.pack(fill="both", expand=True, padx=10, pady=5)

    def _on_session_a_loaded(self, log: LogFile, file_path: Path) -> None:
        """Handle Session A file loaded."""
        self._result_a = None
        self._channels_a = sorted(log.channels.keys())
        self._update_available_channels()

        # Auto-populate config from profile if logger ID is known
        logger_id = self.session_a_panel.logger_id
        if logger_id:
            profile = get_profile_for_logger(logger_id)
            if profile:
                self.config_panel.set_from_profile(profile)
                self._update_status(f"Loaded: {file_path.name} (profile: {profile.name})")
                return

        self._update_status(f"Loaded: {file_path.name}")

    def _on_session_b_loaded(self, log: LogFile, file_path: Path) -> None:
        """Handle Session B file loaded."""
        self._result_b = None
        self._channels_b = sorted(log.channels.keys())
        self._update_available_channels()
        self._update_status(f"Loaded comparison: {file_path.name}")

    def _update_available_channels(self) -> None:
        """Update config panel with channels from all loaded sessions."""
        channels_a = getattr(self, "_channels_a", [])
        channels_b = getattr(self, "_channels_b", [])
        merged = sorted(set(channels_a) | set(channels_b))
        self.config_panel.update_available_channels(merged)

    def _update_status(self, message: str) -> None:
        """Update the status label in config panel."""
        self.config_panel.set_status(message)

    def _on_selection_changed(self) -> None:
        """Handle lap selection change - trigger auto-analysis with debounce."""
        # Cancel any pending analysis
        if self._analysis_timer:
            self.after_cancel(self._analysis_timer)

        # Schedule analysis after 300ms debounce
        self._analysis_timer = self.after(300, self._run_analysis)

    def _run_analysis(self) -> None:
        """Run suspension analysis in background thread."""
        self._analysis_timer = None

        # Don't run if already analyzing
        if self._analyzing:
            return

        session_a = self.session_a_panel.log
        if session_a is None:
            return  # No file loaded yet, silently skip

        laps_a = self.session_a_panel.get_selected_laps()
        if not laps_a:
            self._update_status("Select laps to analyze")
            return

        # Session B auto-enables when file is loaded
        session_b = self.session_b_panel.log
        laps_b: list[int] = []
        if session_b is not None:
            laps_b = self.session_b_panel.get_selected_laps()
            if not laps_b:
                self._update_status("Select laps from Session B")
                return

        # Get config
        motion_ratios = self.config_panel.get_motion_ratios()
        channel_names = self.config_panel.get_channel_names()

        self._analyzing = True
        self._update_status("Analyzing...")

        # Run analysis in background thread
        thread = threading.Thread(
            target=self._analysis_worker,
            args=(session_a, laps_a, session_b, laps_b, motion_ratios, channel_names),
            daemon=True,
        )
        thread.start()

    def _analysis_worker(
        self,
        session_a: "LogFile",
        laps_a: list[int],
        session_b: "LogFile | None",
        laps_b: list[int],
        motion_ratios: MotionRatios,
        channel_names: dict,
    ) -> None:
        """Worker function for analysis (runs in background thread)."""
        try:
            # Analyze Session A
            self._result_a = analyze_suspension_velocity_multi_lap(
                session_a,
                laps_a,
                channel_names=channel_names,
                motion_ratios=motion_ratios,
            )

            # Analyze Session B if enabled
            if laps_b and session_b is not None:
                self._result_b = analyze_suspension_velocity_multi_lap(
                    session_b,
                    laps_b,
                    channel_names=channel_names,
                    motion_ratios=motion_ratios,
                )

            # Update UI on main thread
            self.after(0, self._display_results)

        except Exception as e:
            self.after(0, lambda: self._update_status(f"Error: {e}"))
            self.after(0, self._analysis_done)

    def _analysis_done(self) -> None:
        """Mark analysis as complete."""
        self._analyzing = False

    def _display_results(self) -> None:
        """Display analysis results (called on main thread)."""
        self._analyzing = False

        if self._result_a is None:
            self._update_status("Error: No analysis results")
            return

        # Update chart
        if self._result_b is not None:
            # Comparison mode
            label_a = self.session_a_panel.get_session_label()
            label_b = self.session_b_panel.get_session_label()
            self.chart_view.update_comparison_chart(
                self._result_a,
                self._result_b,
                label_a=label_a,
                label_b=label_b,
            )
        else:
            # Single session mode
            self.chart_view.update_chart(
                self._result_a,
                title="Suspension Velocity Distribution",
            )

        # Display stats table
        self._display_stats()

        self._update_status("Analysis complete")

    def _display_stats(self) -> None:
        """Display statistics using the stats panel (if open)."""
        if self._result_a is None:
            return

        # Only update if stats window is open
        if self.stats_panel is None:
            return

        label_a = self.session_a_panel.get_session_label()

        if self._result_b is not None:
            label_b = self.session_b_panel.get_session_label()
            self.stats_panel.update_stats(
                self._result_a,
                self._result_b,
                label_a=label_a,
                label_b=label_b,
            )
        else:
            self.stats_panel.update_stats(self._result_a, label_a=label_a)

    def _on_save_profile(self) -> None:
        """Handle Save Profile button click."""
        logger_id = self.session_a_panel.logger_id
        if not logger_id:
            self._update_status("No logger ID — load a session first")
            return

        session_label = self.session_a_panel.get_session_label()
        profile = self.config_panel.get_vehicle_profile(name=session_label)
        save_profile_for_logger(logger_id, profile)
        self._update_status(f"Profile saved for logger {logger_id}")
