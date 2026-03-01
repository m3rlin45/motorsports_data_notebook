"""Suspension analysis tab — velocity histogram analysis for 4 shock corners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from motorsports_data_notebook.suspension import analyze_suspension_velocity_multi_lap
from inferno_analyzer.suspension.widgets.chart_view import ChartView
from inferno_analyzer.suspension.widgets.stats_panel import StatsPanel
from inferno_analyzer.tabs.base_tab import BaseAnalysisTab

if TYPE_CHECKING:
    from typing import Union

    from libxrk.base import LogFile as AimLogFile
    from libibt.base import LogFile as IbtLogFile

    LogFile = Union[AimLogFile, IbtLogFile]

    from inferno_analyzer.app import InfernoAnalyzerApp
    from motorsports_data_notebook.suspension import MotionRatios, VelocityHistogramResult


@dataclass
class SuspensionParams:
    session_a: LogFile
    laps_a: list[int]
    session_b: LogFile | None
    laps_b: list[int]
    channel_names_a: dict[str, str]
    channel_names_b: dict[str, str]
    motion_ratios_a: MotionRatios
    motion_ratios_b: MotionRatios


class SuspensionTab(BaseAnalysisTab):
    """Tab for suspension velocity histogram analysis."""

    def __init__(self, app: InfernoAnalyzerApp, tab_frame: ctk.CTkFrame) -> None:
        super().__init__(app, tab_frame)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _create_tab_widgets(self) -> None:
        self.chart_view = ChartView(
            self.tab_frame,
            on_maximize_toggle=self._on_chart_maximize,
        )

    def _layout_tab_widgets(self) -> None:
        self.chart_view.pack(fill="both", expand=True)

    def _get_analysis_params(self) -> SuspensionParams | None:
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

        config_panel = self.app.suspension_config_panel
        return SuspensionParams(
            session_a=session_a,
            laps_a=laps_a,
            session_b=session_b,
            laps_b=laps_b,
            channel_names_a=config_panel.get_channel_names_a(),
            channel_names_b=config_panel.get_channel_names_b(),
            motion_ratios_a=config_panel.get_motion_ratios_a(),
            motion_ratios_b=config_panel.get_motion_ratios_b(),
        )

    def _analysis_worker(self, params: SuspensionParams) -> None:
        self._result_a = analyze_suspension_velocity_multi_lap(
            params.session_a,
            params.laps_a,
            channel_names=params.channel_names_a,
            motion_ratios=params.motion_ratios_a,
        )

        if params.laps_b and params.session_b is not None:
            self._result_b = analyze_suspension_velocity_multi_lap(
                params.session_b,
                params.laps_b,
                channel_names=params.channel_names_b,
                motion_ratios=params.motion_ratios_b,
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
                title="Suspension Velocity Distribution",
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
    # Chart maximize
    # ------------------------------------------------------------------

    def _on_chart_maximize(self, maximized: bool) -> None:
        """Handle chart maximize/restore toggle."""
        self.app.on_chart_maximize(maximized)
