"""Statistics panel widget for displaying suspension analysis results."""

from __future__ import annotations

import customtkinter as ctk

from motorsports_data_notebook.desktop.stats_panel import BaseStatsPanel
from motorsports_data_notebook.suspension import VelocityHistogramResult


class StatsPanel(BaseStatsPanel):
    """Panel for displaying suspension velocity statistics with styled tables."""

    def update_stats(
        self,
        result_a: VelocityHistogramResult,
        result_b: VelocityHistogramResult | None = None,
        label_a: str = "Session A",
        label_b: str = "Session B",
    ) -> None:
        """Update statistics display with styled tables."""
        self._clear_content()

        ranges = result_a.velocity_ranges
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
            headers = ["Corner", "A Skew", "B Skew", "Δ Skew", "A Std", "B Std", "Δ Std"]
            rows = []
            for corner_name, corner_a, corner_b in corners:
                assert corner_b is not None
                delta_skew = corner_b.skew - corner_a.skew
                delta_std = corner_b.std - corner_a.std
                rows.append(
                    [
                        corner_name,
                        f"{corner_a.skew:+.3f}",
                        f"{corner_b.skew:+.3f}",
                        f"{delta_skew:+.3f}",
                        f"{corner_a.std:.1f}",
                        f"{corner_b.std:.1f}",
                        f"{delta_std:+.1f}",
                    ]
                )
            self._create_table(self._scrollable, headers, rows, highlight_cols=[3, 6])
        else:
            headers = ["Corner", "Skew", "Std Dev (mm/s)", "Mean (mm/s)"]
            rows = []
            for corner_name, corner_a, _ in corners:
                rows.append(
                    [
                        corner_name,
                        f"{corner_a.skew:+.3f}",
                        f"{corner_a.std:.1f}",
                        f"{corner_a.mean:.1f}",
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # === Velocity Range Distribution Table ===
        self._create_section_header(self._scrollable, "Velocity Range Distribution (%)")

        if result_b is not None:
            headers = ["Corner", "Direction", "Slow", "Fast", "High-Speed"]
            self._create_velocity_table_comparison(self._scrollable, headers, corners, ranges)
        else:
            headers = ["Corner", "Direction", "Slow", "Fast", "High-Speed"]
            self._create_velocity_table_single(self._scrollable, headers, corners, ranges)

        # === Balance Analysis ===
        self._create_section_header(self._scrollable, "Balance Analysis")

        fl_skew = result_a.front_left.skew
        fr_skew = result_a.front_right.skew
        rl_skew = result_a.rear_left.skew
        rr_skew = result_a.rear_right.skew

        front_avg = (fl_skew + fr_skew) / 2
        rear_avg = (rl_skew + rr_skew) / 2
        left_avg = (fl_skew + rl_skew) / 2
        right_avg = (fr_skew + rr_skew) / 2

        balance_headers = ["Axis", "Side 1", "Side 2", "Difference", "Interpretation"]
        balance_rows = [
            [
                "Front/Rear",
                f"Front: {front_avg:+.3f}",
                f"Rear: {rear_avg:+.3f}",
                f"{front_avg - rear_avg:+.3f}",
                self._interpret_balance(front_avg, rear_avg, "Front", "Rear"),
            ],
            [
                "Left/Right",
                f"Left: {left_avg:+.3f}",
                f"Right: {right_avg:+.3f}",
                f"{left_avg - right_avg:+.3f}",
                self._interpret_balance(left_avg, right_avg, "Left", "Right"),
            ],
        ]
        self._create_table(self._scrollable, balance_headers, balance_rows)

        # === Interpretation Guide ===
        self._create_section_header(self._scrollable, "Interpretation Guide")

        guide_frame = ctk.CTkFrame(self._scrollable, fg_color=("#F0F0F0", "#2B2B2B"))
        guide_frame.pack(fill="x", padx=5, pady=5)

        guides = [
            ("Positive skew:", "More time in rebound (extension) - damper extending"),
            ("Negative skew:", "More time in bump (compression) - damper compressing"),
            ("Near zero skew:", "Balanced damper response"),
            ("High Std Dev:", "More aggressive suspension movement / rougher surface"),
        ]

        for label_text, desc_text in guides:
            row = ctk.CTkFrame(guide_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)

            label = ctk.CTkLabel(
                row,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                width=120,
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

    def _compute_range_percentages_split(self, corner_data, ranges) -> dict[str, dict[str, float]]:
        """Compute percentage of time in each velocity range, split by bump/rebound.

        Returns dict with 'bump' and 'rebound' keys, each containing range percentages.
        Friction range is excluded to avoid biasing bump/rebound attribution
        (near-zero velocities are noise and shouldn't count toward either direction).
        """
        histogram = corner_data.histogram
        bin_centers = corner_data.bin_centers

        bump = {"Slow": 0.0, "Fast": 0.0, "High-Speed": 0.0}
        rebound = {"Slow": 0.0, "Fast": 0.0, "High-Speed": 0.0}

        for center, pct in zip(bin_centers, histogram):
            abs_vel = abs(center)

            # Skip friction range - don't attribute to either direction
            if abs_vel <= ranges.friction:
                continue

            # Positive velocity = bump (compression), negative = rebound (extension)
            target = bump if center >= 0 else rebound

            if abs_vel <= ranges.slow:
                target["Slow"] += pct
            elif abs_vel <= ranges.fast:
                target["Fast"] += pct
            else:
                target["High-Speed"] += pct

        return {"bump": bump, "rebound": rebound}

    def _create_velocity_table_single(
        self,
        parent: ctk.CTkFrame,
        headers: list[str],
        corners: list,
        ranges,
    ) -> None:
        """Create velocity distribution table for single session with bump/rebound rows."""
        table_frame = ctk.CTkFrame(parent, fg_color=("#E0E0E0", "#2B2B2B"))
        table_frame.pack(fill="x", padx=5, pady=5)

        # Header row
        for col, header in enumerate(headers):
            cell = ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=("#C0C0C0", "#404040"),
                corner_radius=0,
                padx=10,
                pady=5,
            )
            cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            table_frame.grid_columnconfigure(col, weight=1)

        # Data rows - 2 rows per corner (bump and rebound)
        row_idx = 1
        for corner_idx, (corner_name, corner_a, _) in enumerate(corners):
            pct = self._compute_range_percentages_split(corner_a, ranges)

            bg_color_1 = ("#F5F5F5", "#333333") if corner_idx % 2 == 0 else ("#E8E8E8", "#3A3A3A")
            bg_color_2 = ("#F0F0F0", "#383838") if corner_idx % 2 == 0 else ("#E3E3E3", "#3F3F3F")

            # Corner name cell spanning 2 rows
            corner_cell = ctk.CTkLabel(
                table_frame,
                text=corner_name,
                font=ctk.CTkFont(size=11, family="Consolas", weight="bold"),
                fg_color=bg_color_1,
                corner_radius=0,
                padx=10,
                pady=4,
            )
            corner_cell.grid(row=row_idx, column=0, rowspan=2, sticky="nsew", padx=1, pady=1)

            # Rebound row (negative velocities - extension)
            for col_idx, (range_name, val) in enumerate(
                [
                    ("Direction", "Rebound ↑"),
                    ("Slow", f"{pct['rebound']['Slow']:.1f}%"),
                    ("Fast", f"{pct['rebound']['Fast']:.1f}%"),
                    ("High-Speed", f"{pct['rebound']['High-Speed']:.1f}%"),
                ]
            ):
                cell = ctk.CTkLabel(
                    table_frame,
                    text=val,
                    font=ctk.CTkFont(size=11, family="Consolas"),
                    fg_color=bg_color_1,
                    corner_radius=0,
                    padx=10,
                    pady=4,
                )
                cell.grid(row=row_idx, column=col_idx + 1, sticky="nsew", padx=1, pady=1)

            # Bump row (positive velocities - compression)
            for col_idx, (range_name, val) in enumerate(
                [
                    ("Direction", "Bump ↓"),
                    ("Slow", f"{pct['bump']['Slow']:.1f}%"),
                    ("Fast", f"{pct['bump']['Fast']:.1f}%"),
                    ("High-Speed", f"{pct['bump']['High-Speed']:.1f}%"),
                ]
            ):
                cell = ctk.CTkLabel(
                    table_frame,
                    text=val,
                    font=ctk.CTkFont(size=11, family="Consolas"),
                    fg_color=bg_color_2,
                    corner_radius=0,
                    padx=10,
                    pady=4,
                )
                cell.grid(row=row_idx + 1, column=col_idx + 1, sticky="nsew", padx=1, pady=1)

            row_idx += 2

    def _create_velocity_table_comparison(
        self,
        parent: ctk.CTkFrame,
        headers: list[str],
        corners: list,
        ranges,
    ) -> None:
        """Create velocity distribution table for session comparison with bump/rebound rows."""
        # For comparison, show A and B values with delta
        table_frame = ctk.CTkFrame(parent, fg_color=("#E0E0E0", "#2B2B2B"))
        table_frame.pack(fill="x", padx=5, pady=5)

        # Modified headers for comparison - use short labels
        comp_headers = ["Corner", "Dir", "", "Slow", "Fast", "High-Spd"]

        for col, header in enumerate(comp_headers):
            cell = ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=("#C0C0C0", "#404040"),
                corner_radius=0,
                padx=6,
                pady=5,
            )
            cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
            table_frame.grid_columnconfigure(col, weight=1)

        row_idx = 1
        for corner_idx, (corner_name, corner_a, corner_b) in enumerate(corners):
            pct_a = self._compute_range_percentages_split(corner_a, ranges)
            pct_b = self._compute_range_percentages_split(corner_b, ranges)

            base_bg = ("#F5F5F5", "#333333") if corner_idx % 2 == 0 else ("#E8E8E8", "#3A3A3A")
            alt_bg = ("#F0F0F0", "#383838") if corner_idx % 2 == 0 else ("#E3E3E3", "#3F3F3F")

            # Corner name spanning 4 rows (rebound A, rebound B, bump A, bump B)
            corner_cell = ctk.CTkLabel(
                table_frame,
                text=corner_name,
                font=ctk.CTkFont(size=10, family="Consolas", weight="bold"),
                fg_color=base_bg,
                corner_radius=0,
                padx=6,
                pady=4,
            )
            corner_cell.grid(row=row_idx, column=0, rowspan=4, sticky="nsew", padx=1, pady=1)

            # Direction cells spanning 2 rows each
            rebound_cell = ctk.CTkLabel(
                table_frame,
                text="Reb ↑",
                font=ctk.CTkFont(size=10, family="Consolas"),
                fg_color=base_bg,
                corner_radius=0,
                padx=6,
                pady=4,
            )
            rebound_cell.grid(row=row_idx, column=1, rowspan=2, sticky="nsew", padx=1, pady=1)

            bump_cell = ctk.CTkLabel(
                table_frame,
                text="Bump ↓",
                font=ctk.CTkFont(size=10, family="Consolas"),
                fg_color=alt_bg,
                corner_radius=0,
                padx=6,
                pady=4,
            )
            bump_cell.grid(row=row_idx + 2, column=1, rowspan=2, sticky="nsew", padx=1, pady=1)

            # Rebound A row
            self._add_comparison_row(table_frame, row_idx, "A", pct_a["rebound"], base_bg)
            # Rebound B row
            self._add_comparison_row(
                table_frame,
                row_idx + 1,
                "B",
                pct_b["rebound"],
                base_bg,
                delta_from=pct_a["rebound"],
            )
            # Bump A row
            self._add_comparison_row(table_frame, row_idx + 2, "A", pct_a["bump"], alt_bg)
            # Bump B row
            self._add_comparison_row(
                table_frame, row_idx + 3, "B", pct_b["bump"], alt_bg, delta_from=pct_a["bump"]
            )

            row_idx += 4

    def _add_comparison_row(
        self,
        table_frame: ctk.CTkFrame,
        row: int,
        session_label: str,
        pct: dict[str, float],
        bg_color: tuple,
        delta_from: dict[str, float] | None = None,
    ) -> None:
        """Add a row to the comparison velocity table."""
        # Session label (A or B)
        text_color: str | None = "#3B8ED0" if session_label == "A" else "#D97706"
        cell = ctk.CTkLabel(
            table_frame,
            text=session_label,
            font=ctk.CTkFont(size=10, family="Consolas", weight="bold"),
            fg_color=bg_color,
            text_color=text_color,
            corner_radius=0,
            padx=6,
            pady=3,
        )
        cell.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)

        # Value cells (excluding Friction to avoid bias)
        for col_idx, range_name in enumerate(["Slow", "Fast", "High-Speed"]):
            val = pct[range_name]
            text = f"{val:.1f}%"

            text_color = None
            if delta_from is not None:
                delta = val - delta_from[range_name]
                if abs(delta) > 0.5:
                    text_color = "#22AA22" if delta > 0 else "#DD4444"

            cell = ctk.CTkLabel(
                table_frame,
                text=text,
                font=ctk.CTkFont(size=10, family="Consolas"),
                fg_color=bg_color,
                text_color=text_color,
                corner_radius=0,
                padx=6,
                pady=3,
            )
            cell.grid(row=row, column=col_idx + 3, sticky="nsew", padx=1, pady=1)

    def _interpret_balance(self, val1: float, val2: float, name1: str, name2: str) -> str:
        """Generate interpretation text for balance comparison."""
        diff = abs(val1 - val2)
        if diff < 0.05:
            return "Balanced"
        elif val1 > val2:
            return f"{name1} biased to rebound"
        else:
            return f"{name2} biased to rebound"
