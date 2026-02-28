"""Tire grip analysis tab — scatter plots of total G vs tire pressure/temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from inferno_analyzer.tire_grip.analysis.multi_lap import analyze_tire_grip_multi_lap
from inferno_analyzer.tire_grip.widgets.chart_view import ChartView
from inferno_analyzer.tire_grip.widgets.stats_panel import StatsPanel
from inferno_analyzer.tabs.base_tab import BaseAnalysisTab

if TYPE_CHECKING:
    from inferno_analyzer.app import InfernoAnalyzerApp
    from libxrk.base import LogFile
    from motorsports_data_notebook.tire_grip import TireGripResult


@dataclass
class TireGripParams:
    session_a: LogFile
    laps_a: list[int]
    session_b: LogFile | None
    laps_b: list[int]
    channel_names_a: dict[str, str]
    channel_names_b: dict[str, str]
    metric_mode: str
    percentile: float


class TireGripTab(BaseAnalysisTab):
    """Tab for tire grip analysis (total G vs tire pressure/temperature)."""

    def __init__(self, app: InfernoAnalyzerApp, tab_frame: ctk.CTkFrame) -> None:
        super().__init__(app, tab_frame)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _create_tab_widgets(self) -> None:
        self.chart_view = ChartView(
            self.tab_frame,
            on_maximize_toggle=self._on_chart_maximize,
            on_metric_changed=self._on_metric_changed,
            on_percentile_changed=self._on_percentile_changed,
        )

    def _layout_tab_widgets(self) -> None:
        self.chart_view.pack(fill="both", expand=True)

    def _get_analysis_params(self) -> TireGripParams | None:
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

        config_panel = self.app.tire_grip_config_panel
        return TireGripParams(
            session_a=session_a,
            laps_a=laps_a,
            session_b=session_b,
            laps_b=laps_b,
            channel_names_a=config_panel.get_channel_names_a(),
            channel_names_b=config_panel.get_channel_names_b(),
            metric_mode=self.chart_view.get_metric_mode(),
            percentile=self.chart_view.get_percentile(),
        )

    def _analysis_worker(self, params: TireGripParams) -> None:
        self._result_a = analyze_tire_grip_multi_lap(
            params.session_a,
            params.laps_a,
            channel_names=params.channel_names_a,
            metric_mode=params.metric_mode,
            percentile=params.percentile,
        )

        if params.laps_b and params.session_b is not None:
            self._result_b = analyze_tire_grip_multi_lap(
                params.session_b,
                params.laps_b,
                channel_names=params.channel_names_b,
                metric_mode=params.metric_mode,
                percentile=params.percentile,
            )
        else:
            self._result_b = None

        self.app.after(0, self._on_analysis_complete)

    def _display_results(self) -> None:
        if self._result_a is None:
            self._update_status("Error: No analysis results")
            return

        if self._result_b is not None:
            label_a = self.app.session_a_panel.get_session_label()
            label_b = self.app.session_b_panel.get_session_label()
            self.chart_view.update_comparison_chart(
                self._result_a,
                self._result_b,
                label_a=label_a,
                label_b=label_b,
            )
        else:
            self.chart_view.update_chart(
                self._result_a,
                title="Tire Grip Analysis",
            )

        self._update_status("Analysis complete")

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
    # Metric mode toggle + chart maximize
    # ------------------------------------------------------------------

    def _on_percentile_changed(self) -> None:
        """Handle percentile value change from chart toolbar."""
        self.on_selection_changed()

    def _on_metric_changed(self, mode: str) -> None:
        """Handle metric mode toggle from chart toolbar."""
        self.on_selection_changed()

    def _on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle."""
        self.app.on_chart_maximize(maximized)
