"""Statistics panel widget for displaying per-corner consistency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

from motorsports_data_notebook.desktop.stats_panel import BaseStatsPanel

if TYPE_CHECKING:
    from inferno_analyzer.driver.analysis.driver_consistency import DriverConsistencyResult


class StatsPanel(BaseStatsPanel):
    """Panel for displaying per-corner driver consistency statistics."""

    def update_stats(
        self,
        result_a: DriverConsistencyResult,
        result_b: DriverConsistencyResult | None = None,
        label_a: str = "Session A",
        label_b: str = "Session B",
    ) -> None:
        """Update statistics display with per-corner consistency tables.

        Parameters
        ----------
        result_a : DriverConsistencyResult
            Primary session results.
        result_b : DriverConsistencyResult, optional
            Comparison session results.
        label_a : str
            Label for primary session.
        label_b : str
            Label for comparison session.
        """
        self._clear_content()

        # Session legend for comparison mode
        if result_b is not None:
            self._create_session_legend(self._scrollable, label_a, label_b)

        # Section 1: Entry Consistency (TA + Braking Points)
        self._create_section_header(self._scrollable, "Entry Consistency")
        if result_b is not None:
            headers = ["Corner", "A TA Mean", "A TA Std", "B TA Mean", "B TA Std", "A BP Std", "B BP Std"]
            rows = []
            for cd_a in result_a.corner_data:
                cd_b = self._find_matching_corner(cd_a.corner, result_b)
                rows.append(
                    [
                        f"{cd_a.corner.name} ({cd_a.corner.direction})",
                        f"{cd_a.ta_mean:.1f}" if cd_a.ta_values else "N/A",
                        f"{cd_a.ta_std:.1f}" if cd_a.ta_values else "N/A",
                        f"{cd_b.ta_mean:.1f}" if cd_b and cd_b.ta_values else "N/A",
                        f"{cd_b.ta_std:.1f}" if cd_b and cd_b.ta_values else "N/A",
                        f"{cd_a.bp_std:.1f}" if cd_a.bp_values else "N/A",
                        f"{cd_b.bp_std:.1f}" if cd_b and cd_b.bp_values else "N/A",
                    ]
                )
            self._create_table(self._scrollable, headers, rows)
        else:
            headers = ["Corner", "TA Mean (%)", "TA Std", "BP Std (m)", "Laps"]
            rows = []
            for cd in result_a.corner_data:
                rows.append(
                    [
                        f"{cd.corner.name} ({cd.corner.direction})",
                        f"{cd.ta_mean:.1f}" if cd.ta_values else "N/A",
                        f"{cd.ta_std:.1f}" if cd.ta_values else "N/A",
                        f"{cd.bp_std:.1f}" if cd.bp_values else "N/A",
                        str(max(len(cd.ta_values), len(cd.bp_values))),
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # Section 2: Corner Speed & Exit
        self._create_section_header(self._scrollable, "Corner Speed & Exit")
        if result_b is not None:
            headers = [
                "Corner",
                "A Min", "A Std", "B Min", "B Std",
                "A Exit", "A Std", "B Exit", "B Std",
            ]
            rows = []
            for cd_a in result_a.corner_data:
                cd_b = self._find_matching_corner(cd_a.corner, result_b)
                rows.append(
                    [
                        f"{cd_a.corner.name} ({cd_a.corner.direction})",
                        f"{cd_a.speed_mean:.1f}" if cd_a.speed_values else "N/A",
                        f"{cd_a.speed_std:.1f}" if cd_a.speed_values else "N/A",
                        f"{cd_b.speed_mean:.1f}" if cd_b and cd_b.speed_values else "N/A",
                        f"{cd_b.speed_std:.1f}" if cd_b and cd_b.speed_values else "N/A",
                        f"{cd_a.exit_speed_mean:.1f}" if cd_a.exit_speed_values else "N/A",
                        f"{cd_a.exit_speed_std:.1f}" if cd_a.exit_speed_values else "N/A",
                        f"{cd_b.exit_speed_mean:.1f}" if cd_b and cd_b.exit_speed_values else "N/A",
                        f"{cd_b.exit_speed_std:.1f}" if cd_b and cd_b.exit_speed_values else "N/A",
                    ]
                )
            self._create_table(self._scrollable, headers, rows)
        else:
            headers = ["Corner", "Min Spd", "Min Std", "Exit Spd", "Exit Std", "Laps"]
            rows = []
            for cd in result_a.corner_data:
                rows.append(
                    [
                        f"{cd.corner.name} ({cd.corner.direction})",
                        f"{cd.speed_mean:.1f}" if cd.speed_values else "N/A",
                        f"{cd.speed_std:.1f}" if cd.speed_values else "N/A",
                        f"{cd.exit_speed_mean:.1f}" if cd.exit_speed_values else "N/A",
                        f"{cd.exit_speed_std:.1f}" if cd.exit_speed_values else "N/A",
                        str(max(len(cd.speed_values), len(cd.exit_speed_values))),
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # Section 3: Opportunity Ranking (single-session only)
        if result_b is None:
            ranked = sorted(
                [cd for cd in result_a.corner_data if cd.opportunity_score > 0],
                key=lambda cd: cd.opportunity_score,
                reverse=True,
            )
            if ranked:
                self._create_section_header(self._scrollable, "Opportunity Ranking")
                headers = ["Rank", "Corner", "Exit \u03c3 (km/h)", "Accel Zone (m)", "Score"]
                rows = []
                for rank, cd in enumerate(ranked, 1):
                    rows.append(
                        [
                            str(rank),
                            f"{cd.corner.name} ({cd.corner.direction})",
                            f"{cd.exit_speed_std:.2f}",
                            f"{cd.accel_zone_length:.0f}",
                            f"{cd.opportunity_score:.1f}",
                        ]
                    )
                self._create_table(self._scrollable, headers, rows)

        # Section 4: Interpretation guide
        self._create_section_header(self._scrollable, "Interpretation Guide")
        guide_frame = ctk.CTkFrame(self._scrollable, fg_color=("#F0F0F0", "#2B2B2B"))
        guide_frame.pack(fill="x", padx=5, pady=5)

        guides = [
            ("TA% near 100:", "Driver applies full throttle at high lateral G (aggressive)"),
            ("TA% near 0:", "Driver waits until corner exit to apply throttle (conservative)"),
            ("Low BP Std:", "Consistent braking points across laps"),
            ("Low Speed Std:", "Consistent corner entry/minimum speeds"),
            ("Low Exit Std:", "Consistent exit speeds across laps"),
            ("High Opportunity:", "Inconsistent exit speed + long straight \u2014 focus here first"),
        ]
        for label_text, desc_text in guides:
            row = ctk.CTkFrame(guide_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            label = ctk.CTkLabel(
                row,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                width=140,
                anchor="w",
            )
            label.pack(side="left")
            desc = ctk.CTkLabel(
                row,
                text=desc_text,
                font=ctk.CTkFont(size=11),
                anchor="w",
            )
            desc.pack(side="left", fill="x", expand=True)

    def _find_matching_corner(self, corner, result_b):
        """Find matching corner data in result_b by corner id."""
        for cd in result_b.corner_data:
            if cd.corner.id == corner.id:
                return cd
        return None
