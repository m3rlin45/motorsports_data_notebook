"""Driver consistency analysis tab — throttle acceptance and braking point analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import customtkinter as ctk
import numpy as np

from inferno_analyzer.driver.analysis.driver_consistency import (
    DriverConsistencyResult,
    analyze_driver_consistency,
)
from inferno_analyzer.driver.visualization.corner_detail import draw_detail
from inferno_analyzer.driver.visualization.corner_summary import draw_summary
from inferno_analyzer.driver.visualization.track_map import draw_track_map
from inferno_analyzer.driver.widgets.chart_view import ChartView
from inferno_analyzer.driver.widgets.corner_selector import CornerSelector
from inferno_analyzer.driver.widgets.stats_panel import StatsPanel
from inferno_analyzer.tabs.base_tab import BaseAnalysisTab

if TYPE_CHECKING:
    from inferno_analyzer.app import InfernoAnalyzerApp


class DriverTab(BaseAnalysisTab):
    """Tab for driver consistency analysis (throttle acceptance + braking points)."""

    def __init__(self, app: InfernoAnalyzerApp, tab_frame: ctk.CTkFrame) -> None:
        self._last_throttle_channel: str | None = None
        super().__init__(app, tab_frame)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _create_tab_widgets(self) -> None:
        # Corner selector (left sidebar)
        self.corner_selector = CornerSelector(
            self.tab_frame,
            on_mode_changed=self._on_view_mode_changed,
            on_corner_selected=self._on_corner_selected,
        )

        # Chart view (right, main area)
        self.chart_view = ChartView(
            self.tab_frame,
            on_maximize_toggle=self._on_chart_maximize,
            on_map_toggle=self._on_map_toggle,
        )

    def _layout_tab_widgets(self) -> None:
        self.corner_selector.pack(side="left", fill="y", padx=(0, 5))
        self.chart_view.pack(side="left", fill="both", expand=True)

    def _get_analysis_params(self) -> dict | None:
        session_a = self.app.session_a_panel.log
        if session_a is None:
            return None

        laps_a = self.app.session_a_panel.get_selected_laps()
        if not laps_a:
            self._update_status("Select laps to analyze")
            return None

        session_b = self.app.session_b_panel.log
        laps_b: list[int] = []
        if session_b is not None:
            laps_b = self.app.session_b_panel.get_selected_laps()
            if not laps_b:
                self._update_status("Select laps from Session B")
                return None

        config_panel = self.app.driver_config_panel
        channel_names = config_panel.get_channel_names()
        corner_threshold = config_panel.get_corner_threshold()
        throttle_threshold = config_panel.get_throttle_threshold()
        sustain_time_ms = config_panel.get_sustain_time_ms()

        return {
            "session_a": session_a,
            "laps_a": laps_a,
            "session_b": session_b,
            "laps_b": laps_b,
            "channel_names": channel_names,
            "corner_threshold": corner_threshold,
            "throttle_threshold": throttle_threshold,
            "sustain_time_ms": sustain_time_ms,
        }

    def _analysis_worker(self, params: dict) -> None:
        self._result_a = analyze_driver_consistency(
            params["session_a"],
            params["laps_a"],
            params["channel_names"],
            corner_threshold=params["corner_threshold"],
            throttle_threshold=params["throttle_threshold"],
            sustain_time_ms=params["sustain_time_ms"],
        )

        if params["laps_b"] and params["session_b"] is not None:
            self._result_b = analyze_driver_consistency(
                params["session_b"],
                params["laps_b"],
                params["channel_names"],
                corner_threshold=params["corner_threshold"],
                throttle_threshold=params["throttle_threshold"],
                sustain_time_ms=params["sustain_time_ms"],
            )
        else:
            self._result_b = None

        self.app.after(0, self._on_analysis_complete)

    def _display_results(self) -> None:
        if self._result_a is None:
            self._update_status("Error: No analysis results")
            return

        # Update corner selector
        self.corner_selector.update_corners(self._result_a.corners)

        # Update chart
        self._update_chart()

        n_corners = len(self._result_a.corners)
        total_ta = sum(len(cd.ta_values) for cd in self._result_a.corner_data)
        threshold = self.app.driver_config_panel.get_throttle_threshold()
        if total_ta > 0:
            self._update_status(f"Done - {n_corners} corners, {total_ta} TA pts (thr: {threshold})")
        else:
            hint = self._diagnose_empty_ta()
            self._update_status(f"Done - {n_corners} corners, 0 TA - {hint}")

    def _display_stats(self) -> None:
        if self._result_a is None or self.stats_panel is None:
            return

        label_a = self.app.session_a_panel.get_session_label()
        if self._result_b is not None:
            label_b = self.app.session_b_panel.get_session_label()
            self.stats_panel.update_stats(
                self._result_a, self._result_b, label_a=label_a, label_b=label_b
            )
        else:
            self.stats_panel.update_stats(self._result_a, label_a=label_a)

    def _create_stats_panel(self, parent: ctk.CTkFrame) -> Any:
        return StatsPanel(parent)

    # ------------------------------------------------------------------
    # Session lifecycle overrides
    # ------------------------------------------------------------------

    def on_session_a_loaded(self) -> None:
        """Called when Session A file is loaded — auto-set throttle threshold."""
        self._result_a = None
        self._result_b = None
        self._stale = True
        self._auto_set_throttle_threshold()
        self._last_throttle_channel = self.app.driver_config_panel.get_channel_names()["throttle"]

    def on_selection_changed(self) -> None:
        """Override to handle throttle channel change → auto-threshold."""
        current_throttle = self.app.driver_config_panel.get_channel_names()["throttle"]
        if (
            self._last_throttle_channel is not None
            and current_throttle != self._last_throttle_channel
        ):
            if self._auto_set_throttle_threshold():
                self._last_throttle_channel = current_throttle

        super().on_selection_changed()

    # ------------------------------------------------------------------
    # Driver-specific logic
    # ------------------------------------------------------------------

    def _auto_set_throttle_threshold(self) -> bool:
        """Set throttle threshold to 95% of peak throttle channel value."""
        log = self.app.session_a_panel.log
        if log is None:
            return False

        throttle_name = self.app.driver_config_panel.get_channel_names()["throttle"]
        if throttle_name not in log.channels:
            return False

        try:
            data = log.channels[throttle_name].column(throttle_name).to_numpy()
            peak = float(np.nanpercentile(data, 95))
            if peak <= 0:
                return False
            threshold = round(peak * 0.95, 1)
            self.app.driver_config_panel.set_throttle_threshold(threshold)
            return True
        except Exception:
            return False

    def _diagnose_empty_ta(self) -> str:
        """Suggest why TA values are empty."""
        log = self.app.session_a_panel.log
        if log is None:
            return "try lower threshold"

        lateral_g_name = self.app.driver_config_panel.get_channel_names()["lateral_g"]
        if lateral_g_name not in log.channels:
            return f"Lat G channel '{lateral_g_name}' not found"

        try:
            data = log.channels[lateral_g_name].column(lateral_g_name).to_numpy()
            peak = float(np.nanmax(np.abs(data)))
            if peak < 0.1:
                return "Lat G values near zero — check channel name"
        except Exception:
            pass

        threshold = self.app.driver_config_panel.get_throttle_threshold()
        return f"try lower threshold (currently {threshold})"

    def _on_view_mode_changed(self, mode: str) -> None:
        """Handle view mode toggle between summary and detail."""
        self._update_chart()

    def _on_corner_selected(self, corner_index: int) -> None:
        """Handle corner selection for detail view."""
        self._update_chart()

    def _on_map_toggle(self, showing_map: bool) -> None:
        """Handle track map toggle."""
        self._update_chart()

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
            label_a = self.app.session_a_panel.get_session_label()
            if self._result_b is not None:
                label_b = self.app.session_b_panel.get_session_label()
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

    def _on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle."""
        if maximized:
            self.corner_selector.pack_forget()
        else:
            self.corner_selector.pack(side="left", fill="y", padx=(0, 5), before=self.chart_view)
        self.app.on_chart_maximize(maximized)
