"""Autocomplete entry widget for channel name selection."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk


class AutocompleteEntry(ctk.CTkFrame):
    """Entry widget with filtered autocomplete dropdown.

    Wraps a CTkEntry and shows a dropdown of matching suggestions
    as the user types. Suggestions are filtered by case-insensitive
    substring match.

    Parameters
    ----------
    parent : ctk.CTkBaseClass
        Parent widget.
    width : int
        Width of the entry field.
    font : ctk.CTkFont | None
        Font for the entry field.
    **kwargs
        Additional keyword arguments passed to CTkFrame.
    """

    # Colors for validation state
    _COLOR_VALID = "#2d8a4e"
    _COLOR_INVALID = "#b33a3a"

    # Dropdown styling
    _DROPDOWN_BG = "#2b2b2b"
    _DROPDOWN_FG = "#dce4ee"
    _DROPDOWN_SELECT_BG = "#1f6aa5"
    _MAX_VISIBLE = 8

    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        width: int = 120,
        font: ctk.CTkFont | None = None,
        on_change: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._suggestions: list[str] = []
        self._filtered: list[str] = []
        self._dropdown: tk.Toplevel | None = None
        self._listbox: tk.Listbox | None = None
        self._suppress_filter = False
        self._on_change = on_change

        # Create the entry
        self._entry = ctk.CTkEntry(self, width=width, font=font)
        self._entry.pack(fill="x")

        # Store default border color for restoring neutral state
        self._default_border_color = self._entry.cget("border_color")

        # Bind events
        self._entry.bind("<KeyRelease>", self._on_key_release)
        self._entry.bind("<FocusOut>", self._on_focus_out)
        self._entry.bind("<Escape>", self._on_escape)

    def set_suggestions(self, suggestions: list[str]) -> None:
        """Update the list of available suggestions.

        Parameters
        ----------
        suggestions : list[str]
            Available channel names for autocomplete.
        """
        self._suggestions = list(suggestions)
        self._update_validation()

    def get(self) -> str:
        """Get the current entry text."""
        result: str = self._entry.get()
        return result

    def insert(self, index: int | str, text: str) -> None:
        """Insert text into the entry."""
        self._entry.insert(index, text)
        self._update_validation()

    def delete(self, first: int | str, last: int | str | None = None) -> None:
        """Delete text from the entry."""
        if last is None:
            last = "end"
        self._entry.delete(first, last)
        self._update_validation()

    def configure(self, **kwargs) -> None:
        """Configure the entry widget."""
        # Pass entry-specific kwargs to the inner entry
        entry_keys = {"state", "placeholder_text", "textvariable"}
        entry_kwargs = {k: v for k, v in kwargs.items() if k in entry_keys}
        frame_kwargs = {k: v for k, v in kwargs.items() if k not in entry_keys}
        if entry_kwargs:
            self._entry.configure(**entry_kwargs)
        if frame_kwargs:
            super().configure(**frame_kwargs)

    def _update_validation(self) -> None:
        """Update border color based on whether text matches a suggestion."""
        if not self._suggestions:
            # No suggestions loaded yet — neutral state
            self._entry.configure(border_color=self._default_border_color)
            return

        text = self._entry.get()
        if not text:
            self._entry.configure(border_color=self._default_border_color)
            return

        if text in self._suggestions:
            self._entry.configure(border_color=self._COLOR_VALID)
        else:
            self._entry.configure(border_color=self._COLOR_INVALID)

    def _on_key_release(self, event: tk.Event) -> None:
        """Handle key release in the entry field."""
        if self._suppress_filter:
            self._suppress_filter = False
            return

        # Navigation keys handled separately
        if event.keysym in ("Up", "Down"):
            self._navigate_dropdown(event.keysym)
            return
        if event.keysym in ("Return", "Tab"):
            self._select_from_dropdown()
            return
        if event.keysym == "Escape":
            return  # Handled by _on_escape binding

        self._update_validation()
        self._filter_and_show()
        if self._on_change:
            self._on_change()

    def _filter_and_show(self) -> None:
        """Filter suggestions and show/update the dropdown."""
        text = self._entry.get().lower()
        if not text or not self._suggestions:
            self._hide_dropdown()
            return

        self._filtered = [s for s in self._suggestions if text in s.lower()]

        if not self._filtered:
            self._hide_dropdown()
            return

        # Don't show dropdown if text exactly matches the only suggestion
        if len(self._filtered) == 1 and self._filtered[0] == self._entry.get():
            self._hide_dropdown()
            return

        self._show_dropdown()

    def _show_dropdown(self) -> None:
        """Show or update the dropdown list."""
        if not self._filtered:
            self._hide_dropdown()
            return

        # Get position below the entry widget
        try:
            x = self._entry.winfo_rootx()
            y = self._entry.winfo_rooty() + self._entry.winfo_height()
            width = self._entry.winfo_width()
        except tk.TclError:
            return

        if self._dropdown is None or not self._dropdown.winfo_exists():
            self._dropdown = tk.Toplevel(self)
            self._dropdown.wm_overrideredirect(True)
            self._dropdown.wm_attributes("-topmost", True)

            self._listbox = tk.Listbox(
                self._dropdown,
                bg=self._DROPDOWN_BG,
                fg=self._DROPDOWN_FG,
                selectbackground=self._DROPDOWN_SELECT_BG,
                selectforeground=self._DROPDOWN_FG,
                borderwidth=1,
                relief="solid",
                font=("TkDefaultFont", 11),
                activestyle="none",
                exportselection=False,
            )
            self._listbox.pack(fill="both", expand=True)
            self._listbox.bind("<ButtonRelease-1>", self._on_listbox_click)
            self._listbox.bind("<Motion>", self._on_listbox_motion)

        # Update listbox contents
        assert self._listbox is not None
        self._listbox.delete(0, "end")
        for item in self._filtered:
            self._listbox.insert("end", item)

        # Size and position
        visible = min(len(self._filtered), self._MAX_VISIBLE)
        row_height = 20
        height = visible * row_height + 4
        self._dropdown.wm_geometry(f"{width}x{height}+{x}+{y}")
        self._dropdown.deiconify()

    def _hide_dropdown(self) -> None:
        """Hide the dropdown list."""
        if self._dropdown is not None:
            try:
                self._dropdown.destroy()
            except tk.TclError:
                pass
            self._dropdown = None
            self._listbox = None

    def _navigate_dropdown(self, direction: str) -> None:
        """Navigate up/down in the dropdown list."""
        if self._listbox is None or not self._filtered:
            return

        selection = self._listbox.curselection()
        if not selection:
            idx = 0 if direction == "Down" else len(self._filtered) - 1
        else:
            current = selection[0]
            if direction == "Down":
                idx = min(current + 1, len(self._filtered) - 1)
            else:
                idx = max(current - 1, 0)

        self._listbox.selection_clear(0, "end")
        self._listbox.selection_set(idx)
        self._listbox.see(idx)

    def _select_from_dropdown(self) -> None:
        """Select the currently highlighted dropdown item."""
        if self._listbox is None:
            return

        selection = self._listbox.curselection()
        if not selection:
            return

        value = self._listbox.get(selection[0])
        self._set_value(value)

    def _on_listbox_click(self, event: tk.Event) -> None:
        """Handle mouse click on a dropdown item."""
        if self._listbox is None:
            return

        idx = self._listbox.nearest(event.y)
        if 0 <= idx < len(self._filtered):
            value = self._listbox.get(idx)
            self._set_value(value)

    def _on_listbox_motion(self, event: tk.Event) -> None:
        """Highlight item under mouse cursor."""
        if self._listbox is None:
            return
        idx = self._listbox.nearest(event.y)
        self._listbox.selection_clear(0, "end")
        self._listbox.selection_set(idx)

    def _set_value(self, value: str) -> None:
        """Set the entry value and close the dropdown."""
        self._suppress_filter = True
        self._entry.delete(0, "end")
        self._entry.insert(0, value)
        self._hide_dropdown()
        self._update_validation()
        self._entry.icursor("end")
        if self._on_change:
            self._on_change()

    def _on_escape(self, event: tk.Event) -> None:
        """Handle Escape key — dismiss dropdown."""
        self._hide_dropdown()

    def _on_focus_out(self, event: tk.Event) -> None:
        """Handle focus leaving the entry."""
        # Delay hiding so click on listbox can register first
        self.after(150, self._hide_dropdown)
        self._update_validation()
