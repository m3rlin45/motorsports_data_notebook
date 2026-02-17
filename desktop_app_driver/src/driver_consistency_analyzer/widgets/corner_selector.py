"""Corner selector widget with view mode toggle."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from motorsports_data_notebook.corners import Corner


class CornerSelector(ctk.CTkFrame):
    """Sidebar widget for selecting corners and toggling view mode.

    Provides a Summary/Detail toggle and a radio-button list of detected
    corners for the detail view.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        on_mode_changed: Callable[[str], None] | None = None,
        on_corner_selected: Callable[[int], None] | None = None,
    ) -> None:
        """Initialize the corner selector.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        on_mode_changed : Callable, optional
            Callback when view mode changes. Receives "summary" or "detail".
        on_corner_selected : Callable, optional
            Callback when a corner is selected. Receives corner index.
        """
        super().__init__(parent, width=160)

        self._on_mode_changed = on_mode_changed
        self._on_corner_selected = on_corner_selected
        self._corners: list[Corner] = []
        self._mode_var = ctk.StringVar(value="summary")
        self._corner_var = ctk.IntVar(value=0)

        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self) -> None:
        """Create all widgets."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="VIEW",
            font=ctk.CTkFont(size=14, weight="bold"),
        )

        # Mode toggle frame
        self.mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.summary_btn = ctk.CTkRadioButton(
            self.mode_frame,
            text="Summary",
            variable=self._mode_var,
            value="summary",
            font=ctk.CTkFont(size=12),
            command=self._on_mode_toggle,
        )
        self.detail_btn = ctk.CTkRadioButton(
            self.mode_frame,
            text="Detail",
            variable=self._mode_var,
            value="detail",
            font=ctk.CTkFont(size=12),
            command=self._on_mode_toggle,
        )

        # Corner list label
        self.corners_label = ctk.CTkLabel(
            self,
            text="Corners:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )

        # Scrollable frame for corner radio buttons
        self.corners_scroll = ctk.CTkScrollableFrame(self, width=140)

        # Placeholder
        self._placeholder = ctk.CTkLabel(
            self.corners_scroll,
            text="Load a session\nto detect corners",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )

    def _layout_widgets(self) -> None:
        """Arrange widgets in the panel."""
        self.title_label.pack(anchor="w", padx=5, pady=(5, 3))

        self.mode_frame.pack(fill="x", padx=5, pady=3)
        self.summary_btn.pack(anchor="w", padx=5, pady=2)
        self.detail_btn.pack(anchor="w", padx=5, pady=2)

        self.corners_label.pack(anchor="w", padx=5, pady=(8, 2))
        self.corners_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self._placeholder.pack(pady=10)

    def _on_mode_toggle(self) -> None:
        """Handle view mode toggle."""
        if self._on_mode_changed:
            self._on_mode_changed(self._mode_var.get())

    def _on_corner_radio(self) -> None:
        """Handle corner radio button selection."""
        # Auto-switch to detail mode when selecting a corner
        if self._mode_var.get() != "detail":
            self._mode_var.set("detail")
            if self._on_mode_changed:
                self._on_mode_changed("detail")

        if self._on_corner_selected:
            self._on_corner_selected(self._corner_var.get())

    def update_corners(self, corners: list[Corner]) -> None:
        """Update the corner list with detected corners.

        Parameters
        ----------
        corners : list[Corner]
            Detected corners to display.
        """
        self._corners = corners

        # Clear existing
        for widget in self.corners_scroll.winfo_children():
            widget.destroy()

        if not corners:
            self._placeholder = ctk.CTkLabel(
                self.corners_scroll,
                text="No corners detected",
                font=ctk.CTkFont(size=11),
                text_color="gray",
            )
            self._placeholder.pack(pady=10)
            return

        self._corner_var.set(0)

        for i, corner in enumerate(corners):
            radio = ctk.CTkRadioButton(
                self.corners_scroll,
                text=f"{corner.name} ({corner.direction})",
                variable=self._corner_var,
                value=i,
                font=ctk.CTkFont(size=11),
                command=self._on_corner_radio,
            )
            radio.pack(anchor="w", padx=5, pady=2)

    def get_mode(self) -> str:
        """Get the current view mode.

        Returns
        -------
        str
            "summary" or "detail".
        """
        return str(self._mode_var.get())

    def get_selected_corner_index(self) -> int:
        """Get the index of the selected corner.

        Returns
        -------
        int
            Index into the corners list.
        """
        return int(self._corner_var.get())

    def clear(self) -> None:
        """Clear the corner list."""
        self._corners = []
        for widget in self.corners_scroll.winfo_children():
            widget.destroy()
        self._placeholder = ctk.CTkLabel(
            self.corners_scroll,
            text="Load a session\nto detect corners",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._placeholder.pack(pady=10)
