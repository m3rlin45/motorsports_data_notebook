"""Base analysis tab with shared debounce, threading, and stats window logic."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

if TYPE_CHECKING:
    from inferno_analyzer.app import InfernoAnalyzerApp


class BaseAnalysisTab(ABC):
    """Abstract base class for analysis tabs.

    Extracts the identical boilerplate from both the Suspension Analyzer
    and Driver Consistency Analyzer apps: debounce timers, background
    threading, stats window lifecycle, and stale-flag logic.

    Subclasses implement the abstract methods to provide tab-specific
    analysis, display, and stats behavior.
    """

    def __init__(self, app: InfernoAnalyzerApp, tab_frame: ctk.CTkFrame) -> None:
        self.app = app
        self.tab_frame = tab_frame

        # Analysis results storage (typed by subclass)
        self._result_a: Any = None
        self._result_b: Any = None

        # Debounce timer for auto-analysis
        self._analysis_timer: str | None = None
        self._analyzing = False
        self._stale = False

        # Stats window
        self._stats_window: ctk.CTkToplevel | None = None
        self.stats_panel: Any = None

        # Build tab-specific widgets
        self._create_tab_widgets()
        self._layout_tab_widgets()

    # ------------------------------------------------------------------
    # Abstract methods — implemented by each tab
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_tab_widgets(self) -> None:
        """Create tab-specific widgets inside self.tab_frame."""

    @abstractmethod
    def _layout_tab_widgets(self) -> None:
        """Arrange tab-specific widgets."""

    @abstractmethod
    def _get_analysis_params(self) -> Any:
        """Read config from the tab's config panel.

        Returns None if analysis cannot proceed (e.g., no session loaded,
        no laps selected). Subclasses return a TypedDict specific to their
        analysis type.
        """

    @abstractmethod
    def _analysis_worker(self, params: Any) -> None:
        """Run the analysis in a background thread.

        Must store results in self._result_a / self._result_b and call
        self.app.after(0, self._on_analysis_complete) when done.
        """

    @abstractmethod
    def _display_results(self) -> None:
        """Update chart and status (called on main thread after analysis)."""

    @abstractmethod
    def _display_stats(self) -> None:
        """Update the stats panel if open (called on main thread)."""

    @abstractmethod
    def _create_stats_panel(self, parent: ctk.CTkFrame) -> Any:
        """Create and return the tab-specific StatsPanel instance."""

    # ------------------------------------------------------------------
    # Session & tab lifecycle
    # ------------------------------------------------------------------

    def on_session_loaded(self) -> None:
        """Called when a new session file is loaded."""
        self._result_a = None
        self._result_b = None
        self._stale = True

    def on_tab_activated(self) -> None:
        """Called when this tab becomes visible."""
        if self._stale:
            self._stale = False
            self.on_selection_changed()

    # ------------------------------------------------------------------
    # Debounce → analysis pipeline
    # ------------------------------------------------------------------

    def on_selection_changed(self) -> None:
        """Handle lap/config change — trigger analysis with 300ms debounce."""
        if self._analysis_timer:
            self.app.after_cancel(self._analysis_timer)
        self._analysis_timer = self.app.after(300, self._run_analysis)

    def _run_analysis(self) -> None:
        """Validate params and spawn analysis in background thread."""
        self._analysis_timer = None

        if self._analyzing:
            # Re-schedule so config changes during analysis aren't lost
            self._analysis_timer = self.app.after(300, self._run_analysis)
            return

        params = self._get_analysis_params()
        if params is None:
            return

        self._analyzing = True
        self._update_status("Analyzing...")

        thread = threading.Thread(
            target=self._analysis_worker_wrapper,
            args=(params,),
            daemon=True,
        )
        thread.start()

    def _analysis_worker_wrapper(self, params: dict) -> None:
        """Wrap the analysis worker with error handling."""
        try:
            self._analysis_worker(params)
        except Exception as e:
            self.app.after(0, lambda: self._update_status(f"Error: {e}"))
            self.app.after(0, self._analysis_done)

    def _on_analysis_complete(self) -> None:
        """Called on main thread after analysis finishes successfully."""
        self._analyzing = False
        self._display_results()
        self._display_stats()

    def _analysis_done(self) -> None:
        """Mark analysis as complete (for error paths)."""
        self._analyzing = False

    # ------------------------------------------------------------------
    # Stats window lifecycle
    # ------------------------------------------------------------------

    def toggle_stats_window(self) -> None:
        """Toggle stats window visibility."""
        try:
            if self._stats_window is not None and self._stats_window.winfo_exists():
                self._stats_window.destroy()
                self._stats_window = None
                self.stats_panel = None
                self.chart_view.set_stats_button_text("Statistics")
                return
        except Exception:
            self._stats_window = None
            self.stats_panel = None

        # Create new stats window
        self._stats_window = ctk.CTkToplevel(self.app)
        self._stats_window.title("Statistics")
        self._stats_window.geometry("700x600")
        self._stats_window.minsize(500, 400)

        self.stats_panel = self._create_stats_panel(self._stats_window)
        self.stats_panel.pack(fill="both", expand=True, padx=10, pady=10)

        if self._result_a is not None:
            self._display_stats()

        self._stats_window.protocol("WM_DELETE_WINDOW", self._on_stats_window_close)
        self.chart_view.set_stats_button_text("Hide Stats")

        self._stats_window.after(100, self._bring_stats_to_front)

    def _bring_stats_to_front(self) -> None:
        """Bring stats window to front."""
        if self._stats_window is not None:
            try:
                self._stats_window.lift()
                self._stats_window.focus_force()
            except Exception:
                pass

    def _on_stats_window_close(self) -> None:
        """Handle stats window being closed."""
        if self._stats_window is not None:
            self._stats_window.destroy()
            self._stats_window = None
            self.stats_panel = None
            self.chart_view.set_stats_button_text("Statistics")

    def close_stats_window(self) -> None:
        """Close the stats window if open (used when switching tabs)."""
        if self._stats_window is not None:
            try:
                if self._stats_window.winfo_exists():
                    self._stats_window.destroy()
            except Exception:
                pass
            self._stats_window = None
            self.stats_panel = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_config_panel(self):
        """Get this tab's config panel from the app."""
        return self.app.get_config_panel_for_tab(self)

    def _update_status(self, message: str) -> None:
        """Update the status label in the active config panel."""
        self._get_config_panel().set_status(message)
