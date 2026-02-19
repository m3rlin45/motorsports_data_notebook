"""Base statistics panel widget with shared table and section rendering."""

from __future__ import annotations

import customtkinter as ctk


class BaseStatsPanel(ctk.CTkFrame):
    """Base panel for displaying statistics with styled tables.

    Provides scrollable container, section headers, table rendering,
    and session legend. Subclasses implement ``update_stats()`` with
    domain-specific logic.
    """

    def __init__(self, parent: ctk.CTk | ctk.CTkFrame) -> None:
        """Initialize the stats panel."""
        super().__init__(parent)
        self._scrollable: ctk.CTkScrollableFrame | None = None
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create scrollable container and placeholder."""
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
            Table rows (list of lists).
        highlight_cols : list[int], optional
            Column indices to highlight (e.g., delta columns).
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
                # Determine text color for delta columns
                text_color = None
                if col_idx in highlight_cols and cell_value not in ("", "-", "N/A"):
                    try:
                        val = float(cell_value.replace("%", "").replace("+", ""))
                        if val > 0:
                            text_color = "#22AA22"  # Green for positive
                        elif val < 0:
                            text_color = "#DD4444"  # Red for negative
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
