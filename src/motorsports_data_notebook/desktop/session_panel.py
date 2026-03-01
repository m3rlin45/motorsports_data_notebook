"""Session panel widget combining file drop and lap selector."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import customtkinter as ctk
from tkinterdnd2 import DND_FILES

from motorsports_data_notebook.desktop.loader import load_session
from motorsports_data_notebook.profiles import get_logger_id

if TYPE_CHECKING:
    from motorsports_data_notebook._types import LogFile


class SessionPanel(ctk.CTkFrame):
    """Panel for loading a session file and selecting laps.

    Provides a drag-and-drop zone for XRK/XRZ files and a scrollable
    checkbox list for lap selection.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        title: str = "Session",
        on_file_loaded: Callable[[LogFile, Path], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
        auto_select: Literal["best", "top_103"] = "best",
    ) -> None:
        """Initialize the session panel.

        Parameters
        ----------
        parent : ctk.CTkFrame
            Parent widget.
        title : str
            Panel title displayed at the top.
        on_file_loaded : Callable, optional
            Callback when a file is successfully loaded.
            Receives (log: LogFile, file_path: Path).
        on_selection_changed : Callable, optional
            Callback when lap selection changes.
        auto_select : {"best", "top_103"}
            Which laps to auto-select when a file is loaded.
            "best" selects only the fastest lap; "top_103" selects all
            laps within 103% of the best.
        """
        super().__init__(parent)

        self._title = title
        self._on_file_loaded = on_file_loaded
        self._on_selection_changed = on_selection_changed
        self._auto_select = auto_select
        self._log: LogFile | None = None
        self._file_path: Path | None = None
        self._logger_id: str | None = None
        self._lap_vars: dict[int, ctk.BooleanVar] = {}

        self._create_widgets()
        self._layout_widgets()
        self._setup_dnd()

    def _create_widgets(self) -> None:
        """Create all widgets."""
        # Title label
        self.title_label = ctk.CTkLabel(
            self,
            text=self._title,
            font=ctk.CTkFont(size=14, weight="bold"),
        )

        # Drop zone frame (compact)
        self.drop_frame = ctk.CTkFrame(self, height=50, fg_color=("gray85", "gray25"))
        self.drop_label = ctk.CTkLabel(
            self.drop_frame,
            text="Drop XRK/XRZ/IBT file here or click to browse",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60"),
        )

        # File info label
        self.file_label = ctk.CTkLabel(
            self,
            text="No file loaded",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )

        # Lap selector frame
        self.laps_frame = ctk.CTkFrame(self)
        self.laps_label = ctk.CTkLabel(
            self.laps_frame,
            text="Select Laps:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )

        # Scrollable frame for lap checkboxes (compact height)
        self.laps_scroll = ctk.CTkScrollableFrame(self.laps_frame, height=80)

        # Selection buttons
        self.btn_frame = ctk.CTkFrame(self.laps_frame, fg_color="transparent")
        self.select_best_btn = ctk.CTkButton(
            self.btn_frame,
            text="Best",
            width=50,
            height=24,
            command=self._select_best,
        )
        self.select_top_btn = ctk.CTkButton(
            self.btn_frame,
            text="Top 103%",
            width=70,
            height=24,
            command=self._select_top_103,
        )
        self.select_all_btn = ctk.CTkButton(
            self.btn_frame,
            text="All",
            width=50,
            height=24,
            command=self._select_all,
        )
        self.select_none_btn = ctk.CTkButton(
            self.btn_frame,
            text="None",
            width=50,
            height=24,
            command=self._select_none,
        )

    def _layout_widgets(self) -> None:
        """Arrange widgets in the panel."""
        # Title at top
        self.title_label.pack(anchor="w", padx=5, pady=(5, 0))

        # Drop zone (compact padding)
        self.drop_frame.pack(fill="x", padx=5, pady=3)
        self.drop_label.pack(expand=True, pady=5)

        # File info
        self.file_label.pack(fill="x", padx=5, pady=2)

        # Lap selector frame
        self.laps_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.laps_label.pack(anchor="w", padx=5, pady=2)

        # Button frame
        self.btn_frame.pack(fill="x", padx=5, pady=2)
        self.select_best_btn.pack(side="left", padx=2)
        self.select_top_btn.pack(side="left", padx=2)
        self.select_all_btn.pack(side="left", padx=2)
        self.select_none_btn.pack(side="left", padx=2)

        # Scrollable lap list
        self.laps_scroll.pack(fill="both", expand=True, padx=5, pady=5)

    def _setup_dnd(self) -> None:
        """Set up drag-and-drop handling."""
        # Register drop zone for file drops
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)
        self.drop_frame.dnd_bind("<<DragEnter>>", self._on_drag_enter)
        self.drop_frame.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        # Also allow clicking to browse
        self.drop_frame.bind("<Button-1>", self._on_click_browse)
        self.drop_label.bind("<Button-1>", self._on_click_browse)

    def _on_drag_enter(self, event) -> None:
        """Highlight drop zone on drag enter."""
        self.drop_frame.configure(fg_color=("gray75", "gray35"))

    def _on_drag_leave(self, event) -> None:
        """Remove highlight on drag leave."""
        self.drop_frame.configure(fg_color=("gray85", "gray25"))

    def _on_drop(self, event) -> None:
        """Handle file drop."""
        self.drop_frame.configure(fg_color=("gray85", "gray25"))

        # Parse dropped file path(s)
        files = self._parse_dropped_files(event.data)
        if files:
            self._load_file(files[0])

    def _parse_dropped_files(self, data: str) -> list[Path]:
        """Parse dropped file paths from DnD data."""
        # Handle Windows paths with curly braces for paths with spaces
        files = []
        if data.startswith("{"):
            # Format: {path with spaces} {another path}
            import re

            for match in re.finditer(r"\{([^}]+)\}", data):
                files.append(Path(match.group(1)))
        else:
            # Simple space-separated paths
            for path_str in data.split():
                files.append(Path(path_str))

        # Filter for supported telemetry files
        return [f for f in files if f.suffix.lower() in (".xrk", ".xrz", ".ibt")]

    def _on_click_browse(self, event) -> None:
        """Open file browser dialog."""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Select telemetry file",
            filetypes=[
                ("Telemetry Files", "*.xrk *.xrz *.ibt"),
                ("AIM Telemetry", "*.xrk *.xrz"),
                ("iRacing Telemetry", "*.ibt"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self._load_file(Path(file_path))

    def _load_file(self, file_path: Path) -> None:
        """Load a telemetry file."""
        try:
            # Read file contents
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Load session using motorsports_data_notebook
            self._log = load_session(file_data)
            self._file_path = file_path
            self._logger_id = get_logger_id(self._log)

            # Build file info with metadata
            info_parts = [f"File: {file_path.name}"]
            if self._logger_id:
                info_parts.append(f"Logger: {self._logger_id}")
            metadata = getattr(self._log, "metadata", {}) or {}
            vehicle = metadata.get("Vehicle Name") or metadata.get("vehicle_name")
            driver = metadata.get("Driver Name") or metadata.get("driver_name")
            if vehicle:
                info_parts.append(f"Vehicle: {vehicle}")
            if driver:
                info_parts.append(f"Driver: {driver}")
            self.file_label.configure(text=" | ".join(info_parts))
            self._populate_laps()

            # Notify callback
            if self._on_file_loaded:
                self._on_file_loaded(self._log, file_path)

        except Exception as e:
            self.file_label.configure(text=f"Error: {e}")
            self._log = None
            self._file_path = None
            self._logger_id = None

    def _populate_laps(self) -> None:
        """Populate the lap checkbox list."""
        # Clear existing checkboxes
        for widget in self.laps_scroll.winfo_children():
            widget.destroy()
        self._lap_vars.clear()

        if self._log is None:
            return

        # Get laps from the log
        laps_df = self._log.laps.to_pandas()

        for _, lap in laps_df.iterrows():
            lap_num = int(lap["num"])
            lap_time = lap.get("lap_time")

            # Format lap time
            if lap_time is not None and hasattr(lap_time, "total_seconds"):
                total_seconds = lap_time.total_seconds()
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                time_str = f"{minutes}:{seconds:06.3f}"
            else:
                time_str = "N/A"

            # Create checkbox with change callback
            var = ctk.BooleanVar(value=False)
            self._lap_vars[lap_num] = var

            checkbox = ctk.CTkCheckBox(
                self.laps_scroll,
                text=f"Lap {lap_num}: {time_str}",
                variable=var,
                font=ctk.CTkFont(size=11),
                command=self._on_lap_toggled,
            )
            checkbox.pack(anchor="w", padx=5, pady=2)

        # Auto-select based on configured mode
        if self._auto_select == "top_103":
            self._select_top_103()
        else:
            self._select_best()

    def _on_lap_toggled(self) -> None:
        """Handle lap checkbox toggle."""
        if self._on_selection_changed:
            self._on_selection_changed()

    def _select_all(self) -> None:
        """Select all laps."""
        for var in self._lap_vars.values():
            var.set(True)
        if self._on_selection_changed:
            self._on_selection_changed()

    def _select_none(self) -> None:
        """Deselect all laps."""
        for var in self._lap_vars.values():
            var.set(False)
        if self._on_selection_changed:
            self._on_selection_changed()

    def _select_best(self) -> None:
        """Select only the best (fastest) lap."""
        if self._log is None:
            return

        laps_df = self._log.laps.to_pandas()
        best_lap_num = self._get_best_lap_num(laps_df)

        if best_lap_num is not None:
            # Deselect all, then select only the best
            for lap_num, var in self._lap_vars.items():
                var.set(lap_num == best_lap_num)
            if self._on_selection_changed:
                self._on_selection_changed()

    def _select_top_103(self) -> None:
        """Select all laps within 103% of the best lap time."""
        if self._log is None:
            return

        laps_df = self._log.laps.to_pandas()
        best_lap_num = self._get_best_lap_num(laps_df)

        if best_lap_num is None:
            return

        # Get best lap time
        best_row = laps_df[laps_df["num"] == best_lap_num].iloc[0]
        best_time = best_row.get("lap_time")

        if best_time is None or not hasattr(best_time, "total_seconds"):
            return

        best_seconds = best_time.total_seconds()
        threshold = best_seconds * 1.03

        # Select all laps within threshold
        for _, lap in laps_df.iterrows():
            lap_num = int(lap["num"])
            lap_time = lap.get("lap_time")

            if lap_num in self._lap_vars:
                if lap_time is not None and hasattr(lap_time, "total_seconds"):
                    within_threshold = lap_time.total_seconds() <= threshold
                    self._lap_vars[lap_num].set(within_threshold)
                else:
                    self._lap_vars[lap_num].set(False)

        if self._on_selection_changed:
            self._on_selection_changed()

    def _get_best_lap_num(self, laps_df) -> int | None:
        """Get the lap number with the fastest time.

        Parameters
        ----------
        laps_df : pd.DataFrame
            DataFrame with lap data.

        Returns
        -------
        int | None
            Lap number of the fastest lap, or None if no valid laps.
        """
        best_lap_num = None
        best_time = None

        for _, lap in laps_df.iterrows():
            lap_time = lap.get("lap_time")
            if lap_time is not None and hasattr(lap_time, "total_seconds"):
                lap_seconds = lap_time.total_seconds()
                if lap_seconds > 0 and (best_time is None or lap_seconds < best_time):
                    best_time = lap_seconds
                    best_lap_num = int(lap["num"])

        return best_lap_num

    def get_selected_laps(self) -> list[int]:
        """Get list of selected lap numbers.

        Returns
        -------
        list[int]
            Lap numbers that are currently selected.
        """
        return [lap_num for lap_num, var in self._lap_vars.items() if var.get()]

    def get_session_label(self) -> str:
        """Get a label for this session (filename without extension).

        Returns
        -------
        str
            Session label for display.
        """
        if self._file_path:
            return self._file_path.stem
        return "Session"

    @property
    def log(self) -> LogFile | None:
        """Get the loaded LogFile."""
        return self._log

    @property
    def logger_id(self) -> str | None:
        """Get the logger serial number from the loaded file."""
        return self._logger_id
