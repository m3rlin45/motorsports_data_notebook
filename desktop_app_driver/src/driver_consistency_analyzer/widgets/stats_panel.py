"""Statistics panel widget for displaying per-corner consistency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from driver_consistency_analyzer.analysis.driver_consistency import DriverConsistencyResult


class StatsPanel(ctk.CTkFrame):
    """Panel for displaying per-corner driver consistency statistics."""

    def __init__(self, parent: ctk.CTk | ctk.CTkFrame) -> None:
        """Initialize the stats panel."""
        super().__init__(parent)
        self._scrollable: ctk.CTkScrollableFrame | None = None
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all widgets."""
        self._scrollable = ctk.CTkScrollableFrame(self)
        self._scrollable.pack(fill="both", expand=True, padx=5, pady=5)

        self._placeholder = ctk.CTkLabel(
            self._scrollable,
            text="Load a session and select laps to view statistics",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._placeholder.pack(pady=20)

    def _clear_content(self) -> None:
        """Clear all content from scrollable frame."""
        if self._scrollable is None:
            return
        for widget in self._scrollable.winfo_children():
            widget.destroy()

    def _create_section_header(self, parent: ctk.CTkFrame, text: str) -> None:
        """Create a styled section header."""
        header = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#3B8ED0",
        )
        header.pack(anchor="w", pady=(10, 5), padx=5)

    def _create_table(
        self,
        parent: ctk.CTkFrame,
        headers: list[str],
        rows: list[list[str]],
        highlight_cols: list[int] | None = None,
    ) -> None:
        """Create a styled table with headers and rows.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        headers : list[str]
            Column headers.
        rows : list[list[str]]
            Table rows.
        highlight_cols : list[int], optional
            Column indices to color-code by value.
        """
        table_frame = ctk.CTkFrame(parent, fg_color=("#E0E0E0", "#2B2B2B"))
        table_frame.pack(fill="x", padx=5, pady=5)

        highlight_cols = highlight_cols or []

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

        # Data rows
        for row_idx, row_data in enumerate(rows):
            bg_color = ("#F5F5F5", "#333333") if row_idx % 2 == 0 else ("#E8E8E8", "#3A3A3A")

            for col_idx, cell_value in enumerate(row_data):
                text_color = None
                if col_idx in highlight_cols and cell_value not in ("", "-", "N/A"):
                    try:
                        val = float(cell_value.replace("%", "").replace("+", ""))
                        if val > 0:
                            text_color = "#22AA22"
                        elif val < 0:
                            text_color = "#DD4444"
                    except ValueError:
                        pass

                cell = ctk.CTkLabel(
                    table_frame,
                    text=cell_value,
                    font=ctk.CTkFont(size=11, family="Consolas"),
                    fg_color=bg_color,
                    text_color=text_color,
                    corner_radius=0,
                    padx=10,
                    pady=4,
                )
                cell.grid(row=row_idx + 1, column=col_idx, sticky="nsew", padx=1, pady=1)

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

        # Throttle Acceptance table
        self._create_section_header(self._scrollable, "Throttle Acceptance (%)")
        if result_b is not None:
            headers = ["Corner", "A Mean", "A Std", "B Mean", "B Std"]
            rows = []
            for cd_a in result_a.corner_data:
                cd_b = self._find_matching_corner(cd_a.corner, result_b)
                b_mean = f"{cd_b.ta_mean:.1f}" if cd_b and cd_b.ta_values else "N/A"
                b_std = f"{cd_b.ta_std:.1f}" if cd_b and cd_b.ta_values else "N/A"
                rows.append(
                    [
                        f"{cd_a.corner.name} ({cd_a.corner.direction})",
                        f"{cd_a.ta_mean:.1f}" if cd_a.ta_values else "N/A",
                        f"{cd_a.ta_std:.1f}" if cd_a.ta_values else "N/A",
                        b_mean,
                        b_std,
                    ]
                )
            self._create_table(self._scrollable, headers, rows)
        else:
            headers = ["Corner", "Mean TA%", "Std Dev", "Laps"]
            rows = []
            for cd in result_a.corner_data:
                rows.append(
                    [
                        f"{cd.corner.name} ({cd.corner.direction})",
                        f"{cd.ta_mean:.1f}" if cd.ta_values else "N/A",
                        f"{cd.ta_std:.1f}" if cd.ta_values else "N/A",
                        str(len(cd.ta_values)),
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # Braking Consistency table
        self._create_section_header(self._scrollable, "Braking Point Consistency")
        if result_b is not None:
            headers = ["Corner", "A Std (m)", "B Std (m)"]
            rows = []
            for cd_a in result_a.corner_data:
                cd_b = self._find_matching_corner(cd_a.corner, result_b)
                b_val = f"{cd_b.bp_std:.1f}" if cd_b and cd_b.bp_values else "N/A"
                rows.append(
                    [
                        f"{cd_a.corner.name} ({cd_a.corner.direction})",
                        f"{cd_a.bp_std:.1f}" if cd_a.bp_values else "N/A",
                        b_val,
                    ]
                )
            self._create_table(self._scrollable, headers, rows)
        else:
            headers = ["Corner", "Std Dev (m)", "Laps"]
            rows = []
            for cd in result_a.corner_data:
                rows.append(
                    [
                        f"{cd.corner.name} ({cd.corner.direction})",
                        f"{cd.bp_std:.1f}" if cd.bp_values else "N/A",
                        str(len(cd.bp_values)),
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # Corner Speed table
        self._create_section_header(self._scrollable, "Minimum Corner Speed")
        if result_b is not None:
            headers = ["Corner", "A Mean", "A Std", "B Mean", "B Std"]
            rows = []
            for cd_a in result_a.corner_data:
                cd_b = self._find_matching_corner(cd_a.corner, result_b)
                b_mean = f"{cd_b.speed_mean:.1f}" if cd_b and cd_b.speed_values else "N/A"
                b_std = f"{cd_b.speed_std:.1f}" if cd_b and cd_b.speed_values else "N/A"
                rows.append(
                    [
                        f"{cd_a.corner.name} ({cd_a.corner.direction})",
                        f"{cd_a.speed_mean:.1f}" if cd_a.speed_values else "N/A",
                        f"{cd_a.speed_std:.1f}" if cd_a.speed_values else "N/A",
                        b_mean,
                        b_std,
                    ]
                )
            self._create_table(self._scrollable, headers, rows)
        else:
            headers = ["Corner", "Mean (km/h)", "Std Dev", "Laps"]
            rows = []
            for cd in result_a.corner_data:
                rows.append(
                    [
                        f"{cd.corner.name} ({cd.corner.direction})",
                        f"{cd.speed_mean:.1f}" if cd.speed_values else "N/A",
                        f"{cd.speed_std:.1f}" if cd.speed_values else "N/A",
                        str(len(cd.speed_values)),
                    ]
                )
            self._create_table(self._scrollable, headers, rows)

        # Interpretation guide
        self._create_section_header(self._scrollable, "Interpretation Guide")
        guide_frame = ctk.CTkFrame(self._scrollable, fg_color=("#F0F0F0", "#2B2B2B"))
        guide_frame.pack(fill="x", padx=5, pady=5)

        guides = [
            ("TA% near 100:", "Driver applies full throttle at high lateral G (aggressive)"),
            ("TA% near 0:", "Driver waits until corner exit to apply throttle (conservative)"),
            ("Low BP Std:", "Consistent braking points across laps"),
            ("Low Speed Std:", "Consistent corner entry/minimum speeds"),
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

    def _create_session_legend(self, parent: ctk.CTkFrame, label_a: str, label_b: str) -> None:
        """Create a legend showing session A and B full names."""
        legend_frame = ctk.CTkFrame(parent, fg_color=("#E8F4FC", "#1E3A4C"))
        legend_frame.pack(fill="x", padx=5, pady=(10, 5))

        ctk.CTkLabel(
            legend_frame,
            text=f"A: {label_a}",
            font=ctk.CTkFont(size=11),
            text_color="#3B8ED0",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            legend_frame,
            text=f"B: {label_b}",
            font=ctk.CTkFont(size=11),
            text_color="#D97706",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(2, 8))

    def _find_matching_corner(self, corner, result_b):
        """Find matching corner data in result_b by corner id."""
        for cd in result_b.corner_data:
            if cd.corner.id == corner.id:
                return cd
        return None

    def clear(self) -> None:
        """Clear and show placeholder."""
        self._clear_content()
        self._placeholder = ctk.CTkLabel(
            self._scrollable,
            text="Load a session and select laps to view statistics",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._placeholder.pack(pady=20)
