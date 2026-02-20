"""Main application window for Driver Consistency Analyzer."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk
import numpy as np
from tkinterdnd2 import TkinterDnD

from motorsports_data_notebook.desktop.dpi import setup_hidpi_scaling
from motorsports_data_notebook.desktop.session_panel import SessionPanel
from motorsports_data_notebook.profiles import (
    get_profile_for_logger,
    save_profile_for_logger,
)

from driver_consistency_analyzer.analysis.driver_consistency import (
    DriverConsistencyResult,
    analyze_driver_consistency,
)
from driver_consistency_analyzer.visualization.corner_detail import draw_detail
from driver_consistency_analyzer.visualization.corner_summary import draw_summary
from driver_consistency_analyzer.visualization.track_map import draw_track_map
from driver_consistency_analyzer.widgets.chart_view import ChartView
from driver_consistency_analyzer.widgets.config_panel import ConfigPanel
from driver_consistency_analyzer.widgets.corner_selector import CornerSelector
from driver_consistency_analyzer.widgets.stats_panel import StatsPanel

if TYPE_CHECKING:
    from libxrk.base import LogFile


class DriverConsistencyApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window for Driver Consistency Analyzer.

    Provides a GUI for analyzing throttle acceptance and braking point
    consistency across corners from XRK/XRZ telemetry files.
    """

    def __init__(self) -> None:
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        # Window setup
        self.title("Driver Consistency Analyzer")
        self.geometry("1300x900")
        self.minsize(1000, 700)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # HiDPI scaling
        setup_hidpi_scaling(self)

        # Analysis results storage
        self._result_a: DriverConsistencyResult | None = None
        self._result_b: DriverConsistencyResult | None = None

        # Debounce timer for auto-analysis
        self._analysis_timer: str | None = None
        self._analyzing = False
        self._last_throttle_channel: str | None = None

        # Build UI
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Top frame for session panels and config
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

        # Config panel (right)
        self._stats_window: ctk.CTkToplevel | None = None
        self.config_panel = ConfigPanel(
            self.top_frame,
            on_stats_click=self._toggle_stats_window,
            on_config_changed=self._on_selection_changed,
            on_save_profile=self._on_save_profile,
        )

        # Bottom frame (corner selector + chart)
        self.bottom_frame = ctk.CTkFrame(self)

        # Corner selector (left sidebar)
        self.corner_selector = CornerSelector(
            self.bottom_frame,
            on_mode_changed=self._on_view_mode_changed,
            on_corner_selected=self._on_corner_selected,
        )

        # Chart view (right, main area)
        self.chart_view = ChartView(
            self.bottom_frame,
            on_maximize_toggle=self._on_chart_maximize,
            on_map_toggle=self._on_map_toggle,
        )

        # Stats panel (created in popup window when needed)
        self.stats_panel: StatsPanel | None = None

    def _layout_widgets(self) -> None:
        """Arrange widgets in the window."""
        # Top frame
        self.top_frame.pack(fill="x", padx=10, pady=5)

        # All 3 panels side by side
        self.session_a_panel.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        self.session_b_panel.grid(row=0, column=1, padx=3, pady=3, sticky="nsew")
        self.config_panel.grid(row=0, column=2, padx=3, pady=3, sticky="nsew")

        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_columnconfigure(2, weight=1)

        # Bottom frame
        self.bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Corner selector on left
        self.corner_selector.pack(side="left", fill="y", padx=(0, 5))

        # Chart view fills remaining space
        self.chart_view.pack(side="left", fill="both", expand=True)

    def _toggle_stats_window(self) -> None:
        """Toggle stats window visibility."""
        try:
            if self._stats_window is not None and self._stats_window.winfo_exists():
                self._stats_window.destroy()
                self._stats_window = None
                self.stats_panel = None
                self.config_panel.set_stats_button_text("Show Statistics")
                return
        except Exception:
            self._stats_window = None
            self.stats_panel = None

        # Create new stats window
        self._stats_window = ctk.CTkToplevel(self)
        self._stats_window.title("Statistics")
        self._stats_window.geometry("700x600")
        self._stats_window.minsize(500, 400)

        self.stats_panel = StatsPanel(self._stats_window)
        self.stats_panel.pack(fill="both", expand=True, padx=10, pady=10)

        if self._result_a is not None:
            self._display_stats()

        self._stats_window.protocol("WM_DELETE_WINDOW", self._on_stats_window_close)
        self.config_panel.set_stats_button_text("Hide Statistics")

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

    def _on_map_toggle(self, showing_map: bool) -> None:
        """Handle track map toggle."""
        self._update_chart()

    def _on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle."""
        if maximized:
            self.top_frame.pack_forget()
            self.corner_selector.pack_forget()
        else:
            # Re-insert top_frame before bottom_frame
            self.bottom_frame.pack_forget()
            self.top_frame.pack(fill="x", padx=10, pady=5)
            self.bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)
            self.corner_selector.pack(side="left", fill="y", padx=(0, 5), before=self.chart_view)

    def _on_session_a_loaded(self, log: LogFile, file_path: Path) -> None:
        """Handle Session A file loaded."""
        self._result_a = None
        self._result_b = None
        self._channels_a = sorted(log.channels.keys())
        self._update_available_channels()

        # Auto-populate config from profile if logger ID is known
        logger_id = self.session_a_panel.logger_id
        profile = None
        if logger_id:
            profile = get_profile_for_logger(logger_id)
            if profile:
                self.config_panel.set_from_profile(profile)

        # Auto-set throttle threshold from channel data (after profile sets channel names)
        self._auto_set_throttle_threshold()
        self._last_throttle_channel = self.config_panel.get_channel_names()["throttle"]

        if profile:
            self._update_status(f"Loaded: {file_path.name} (profile: {profile.name})")
        elif logger_id:
            self._update_status(f"Loaded: {file_path.name} (logger {logger_id}, no profile)")
        else:
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

    def _auto_set_throttle_threshold(self) -> bool:
        """Set throttle threshold to 95% of peak throttle channel value.

        Returns True if the threshold was successfully computed and set.
        """
        log = self.session_a_panel.log
        if log is None:
            return False

        throttle_name = self.config_panel.get_channel_names()["throttle"]
        if throttle_name not in log.channels:
            return False

        try:
            data = log.channels[throttle_name].column(throttle_name).to_numpy()
            # Use 95th percentile to ignore sensor spikes/overshoot.
            # The top 5% of session data is excluded, which handles noisy
            # sensors that overshoot above the true full-throttle value.
            peak = float(np.nanpercentile(data, 95))
            if peak <= 0:
                return False
            threshold = round(peak * 0.95, 1)
            self.config_panel.set_throttle_threshold(threshold)
            return True
        except Exception:
            return False

    def _diagnose_empty_ta(self) -> str:
        """Suggest why TA values are empty."""
        log = self.session_a_panel.log
        if log is None:
            return "try lower threshold"

        lateral_g_name = self.config_panel.get_channel_names()["lateral_g"]
        if lateral_g_name not in log.channels:
            return f"Lat G channel '{lateral_g_name}' not found"

        try:
            data = log.channels[lateral_g_name].column(lateral_g_name).to_numpy()
            peak = float(np.nanmax(np.abs(data)))
            if peak < 0.1:
                return f"Lat G values near zero — check channel name"
        except Exception:
            pass

        threshold = self.config_panel.get_throttle_threshold()
        return f"try lower threshold (currently {threshold})"

    def _on_selection_changed(self) -> None:
        """Handle lap selection change - trigger auto-analysis with debounce."""
        current_throttle = self.config_panel.get_channel_names()["throttle"]
        if (
            self._last_throttle_channel is not None
            and current_throttle != self._last_throttle_channel
        ):
            # Only update tracking when the channel actually exists in the log
            # (avoids tracking intermediate partial text while user types)
            if self._auto_set_throttle_threshold():
                self._last_throttle_channel = current_throttle

        if self._analysis_timer:
            self.after_cancel(self._analysis_timer)
        self._analysis_timer = self.after(300, self._run_analysis)

    def _on_view_mode_changed(self, mode: str) -> None:
        """Handle view mode toggle between summary and detail."""
        self._update_chart()

    def _on_corner_selected(self, corner_index: int) -> None:
        """Handle corner selection for detail view."""
        self._update_chart()

    def _run_analysis(self) -> None:
        """Run driver consistency analysis in background thread."""
        self._analysis_timer = None

        if self._analyzing:
            # Re-schedule so config changes made during analysis aren't lost
            self._analysis_timer = self.after(300, self._run_analysis)
            return

        session_a = self.session_a_panel.log
        if session_a is None:
            return

        laps_a = self.session_a_panel.get_selected_laps()
        if not laps_a:
            self._update_status("Select laps to analyze")
            return

        session_b = self.session_b_panel.log
        laps_b: list[int] = []
        if session_b is not None:
            laps_b = self.session_b_panel.get_selected_laps()
            if not laps_b:
                self._update_status("Select laps from Session B")
                return

        channel_names = self.config_panel.get_channel_names()
        corner_threshold = self.config_panel.get_corner_threshold()
        throttle_threshold = self.config_panel.get_throttle_threshold()
        sustain_time_ms = self.config_panel.get_sustain_time_ms()

        self._analyzing = True
        self._update_status("Analyzing...")

        thread = threading.Thread(
            target=self._analysis_worker,
            args=(
                session_a,
                laps_a,
                session_b,
                laps_b,
                channel_names,
                corner_threshold,
                throttle_threshold,
                sustain_time_ms,
            ),
            daemon=True,
        )
        thread.start()

    def _analysis_worker(
        self,
        session_a: "LogFile",
        laps_a: list[int],
        session_b: "LogFile | None",
        laps_b: list[int],
        channel_names: dict[str, str],
        corner_threshold: float,
        throttle_threshold: float,
        sustain_time_ms: float,
    ) -> None:
        """Worker function for analysis (runs in background thread)."""
        try:
            self._result_a = analyze_driver_consistency(
                session_a,
                laps_a,
                channel_names,
                corner_threshold=corner_threshold,
                throttle_threshold=throttle_threshold,
                sustain_time_ms=sustain_time_ms,
            )

            if laps_b and session_b is not None:
                self._result_b = analyze_driver_consistency(
                    session_b,
                    laps_b,
                    channel_names,
                    corner_threshold=corner_threshold,
                    throttle_threshold=throttle_threshold,
                    sustain_time_ms=sustain_time_ms,
                )
            else:
                self._result_b = None

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

        # Update corner selector
        self.corner_selector.update_corners(self._result_a.corners)

        # Update chart
        self._update_chart()

        # Update stats
        self._display_stats()

        n_corners = len(self._result_a.corners)
        total_ta = sum(len(cd.ta_values) for cd in self._result_a.corner_data)
        threshold = self.config_panel.get_throttle_threshold()
        if total_ta > 0:
            self._update_status(f"Done - {n_corners} corners, {total_ta} TA pts (thr: {threshold})")
        else:
            hint = self._diagnose_empty_ta()
            self._update_status(f"Done - {n_corners} corners, 0 TA - {hint}")

    def _update_chart(self) -> None:
        """Update the chart based on current view mode and selection."""
        if self._result_a is None:
            return

        fig = self.chart_view.get_figure()

        if self.chart_view.is_map:
            draw_track_map(
                fig,
                self._result_a.ref_lat,
                self._result_a.ref_lon,
                self._result_a.ref_distance,
                self._result_a.segments,
            )
            self.chart_view.redraw()
            return

        mode = self.corner_selector.get_mode()

        if mode == "summary":
            label_a = self.session_a_panel.get_session_label()
            if self._result_b is not None:
                label_b = self.session_b_panel.get_session_label()
                draw_summary(fig, self._result_a, self._result_b, label_a, label_b)
            else:
                draw_summary(fig, self._result_a, label_a=label_a)
        else:
            # Detail mode
            corner_idx = self.corner_selector.get_selected_corner_index()
            if corner_idx < len(self._result_a.corner_data):
                corner_data = self._result_a.corner_data[corner_idx]
                draw_detail(fig, corner_data)

        self.chart_view.redraw()

    def _display_stats(self) -> None:
        """Display statistics using the stats panel (if open)."""
        if self._result_a is None or self.stats_panel is None:
            return

        label_a = self.session_a_panel.get_session_label()
        if self._result_b is not None:
            label_b = self.session_b_panel.get_session_label()
            self.stats_panel.update_stats(
                self._result_a, self._result_b, label_a=label_a, label_b=label_b
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
