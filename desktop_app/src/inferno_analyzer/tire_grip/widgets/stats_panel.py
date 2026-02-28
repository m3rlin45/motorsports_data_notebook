"""Statistics panel widget for displaying tire grip analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

from motorsports_data_notebook.desktop.stats_panel import BaseStatsPanel

if TYPE_CHECKING:
    from motorsports_data_notebook.tire_grip import TireGripResult


class StatsPanel(BaseStatsPanel):
    """Panel for displaying tire grip statistics with styled tables."""

    def update_stats(
        self,
        result_a: TireGripResult,
        result_b: TireGripResult | None = None,
        label_a: str = "Session A",
        label_b: str = "Session B",
    ) -> None:
        """Update statistics display with styled tables."""
        self._clear_content()

        metric_label = "Pressure" if result_a.metric_mode == "pressure" else "Temperature"
        unit = result_a.metric_unit

        corners = [
            ("Front Left", result_a.front_left, result_b.front_left if result_b else None),
            ("Front Right", result_a.front_right, result_b.front_right if result_b else None),
            ("Rear Left", result_a.rear_left, result_b.rear_left if result_b else None),
            ("Rear Right", result_a.rear_right, result_b.rear_right if result_b else None),
        ]

        # === Session Legend (for comparisons) ===
        if result_b is not None:
            self._create_session_legend(self._scrollable, label_a, label_b)

        # === Summary Statistics Table ===
        self._create_section_header(self._scrollable, "Summary Statistics")

        if result_b is not None:
            headers = [
                "Corner",
                "A Mean G",
                "B Mean G",
                "\u0394 Mean G",
            ]
            rows = []
            for corner_name, corner_a, corner_b in corners:
                assert corner_b is not None
                delta_g = corner_b.mean_g - corner_a.mean_g
                rows.append(
                    [
                        corner_name,
                        f"{corner_a.mean_g:.3f}",
                        f"{corner_b.mean_g:.3f}",
                        f"{delta_g:+.3f}",
                    ]
                )
            self._create_table(self._scrollable, headers, rows, highlight_cols=[3])
        else:
            headers = [
                "Corner",
                "Mean G",
                "Std G",
                f"Mean {metric_label} ({unit})",
                f"Std {metric_label} ({unit})",
            ]
            rows = []
            for corner_name, corner_a, _ in corners:
                rows.append(
                    [
                        corner_name,
                        f"{corner_a.mean_g:.3f}",
                        f"{corner_a.std_g:.3f}",
                        f"{corner_a.mean_metric:.1f}",
                        f"{corner_a.std_metric:.1f}",
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # === Detailed Metric Table (for comparisons) ===
        if result_b is not None:
            self._create_section_header(self._scrollable, f"{metric_label} Details")
            detail_headers = [
                "Corner",
                f"A Mean ({unit})",
                f"B Mean ({unit})",
                f"\u0394 Mean",
                f"A Std ({unit})",
                f"B Std ({unit})",
                f"\u0394 Std",
            ]
            detail_rows = []
            for corner_name, corner_a, corner_b in corners:
                assert corner_b is not None
                delta_mean = corner_b.mean_metric - corner_a.mean_metric
                delta_std = corner_b.std_metric - corner_a.std_metric
                detail_rows.append(
                    [
                        corner_name,
                        f"{corner_a.mean_metric:.1f}",
                        f"{corner_b.mean_metric:.1f}",
                        f"{delta_mean:+.1f}",
                        f"{corner_a.std_metric:.1f}",
                        f"{corner_b.std_metric:.1f}",
                        f"{delta_std:+.1f}",
                    ]
                )
            self._create_table(self._scrollable, detail_headers, detail_rows, highlight_cols=[3, 6])
