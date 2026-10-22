"""Statistics panel widget for roll gradient results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from motorsports_data_notebook.desktop.stats_panel import BaseStatsPanel

if TYPE_CHECKING:
    from motorsports_data_notebook.roll_gradient import RollGradientResult


class StatsPanel(BaseStatsPanel):
    """Panel for displaying roll gradient statistics with styled tables."""

    def update_stats(
        self,
        result_a: RollGradientResult,
        result_b: RollGradientResult | None = None,
        label_a: str = "Session A",
        label_b: str = "Session B",
    ) -> None:
        """Update statistics display with styled tables."""
        self._clear_content()

        if result_b is not None:
            self._create_session_legend(self._scrollable, label_a, label_b)

        self._create_section_header(self._scrollable, "Roll Gradient (deg per g)")

        axles = [
            ("Front", result_a.front, result_b.front if result_b else None),
            ("Rear", result_a.rear, result_b.rear if result_b else None),
        ]

        if result_b is not None:
            headers = ["Axle", "A |gradient|", "B |gradient|", "Δ", "A R²", "B R²"]
            rows = []
            for name, axle_a, axle_b in axles:
                assert axle_b is not None
                rows.append(
                    [
                        name,
                        f"{axle_a.abs_gradient_deg_per_g:.3f}",
                        f"{axle_b.abs_gradient_deg_per_g:.3f}",
                        f"{axle_b.abs_gradient_deg_per_g - axle_a.abs_gradient_deg_per_g:+.3f}",
                        f"{axle_a.r2:.3f}",
                        f"{axle_b.r2:.3f}",
                    ]
                )
            rows.append(
                [
                    "Roll ratio (F/R)",
                    f"{result_a.roll_ratio:.3f}",
                    f"{result_b.roll_ratio:.3f}",
                    f"{result_b.roll_ratio - result_a.roll_ratio:+.3f}",
                    f"{result_a.roll_ratio_r2:.3f}",
                    f"{result_b.roll_ratio_r2:.3f}",
                ]
            )
            self._create_table(self._scrollable, headers, rows, highlight_cols=[3])
        else:
            headers = ["Axle", "Gradient (deg/g)", "Intercept (deg)", "R²", "Samples"]
            rows = []
            for name, axle_a, _ in axles:
                rows.append(
                    [
                        name,
                        f"{axle_a.gradient_deg_per_g:+.3f}",
                        f"{axle_a.intercept_deg:+.3f}",
                        f"{axle_a.r2:.3f}",
                        f"{axle_a.n_samples}",
                    ]
                )
            rows.append(
                [
                    "Roll ratio (F/R)",
                    f"{result_a.roll_ratio:.3f}",
                    f"{result_a.roll_ratio_intercept_deg:+.3f}",
                    f"{result_a.roll_ratio_r2:.3f}",
                    f"{min(result_a.front.n_samples, result_a.rear.n_samples)}",
                ]
            )
            self._create_table(self._scrollable, headers, rows)
            self._create_section_header(self._scrollable, "Derived")
            self._create_table(
                self._scrollable,
                ["Item", "Value"],
                [
                    ["Mean |gradient| (deg/g)", f"{result_a.mean_gradient_deg_per_g:.3f}"],
                    ["Front − Rear (deg/g)", f"{result_a.front_minus_rear_deg_per_g:+.3f}"],
                    ["Gradient ratio F/R", f"{result_a.gradient_ratio:.3f}"],
                ],
            )

        self._create_section_header(self._scrollable, "Setup")
        setup_rows = [
            [
                "Track width (mm)",
                f"F {result_a.track_widths.front_mm:.0f} / R {result_a.track_widths.rear_mm:.0f}",
            ],
            ["Lateral G channel unit", result_a.lateral_g_unit or "g"],
            [
                "Front/rear roll direction",
                "consistent" if result_a.signs_consistent else "OPPOSITE — check L/R channels",
            ],
        ]
        self._create_table(self._scrollable, ["Item", "Value"], setup_rows)
