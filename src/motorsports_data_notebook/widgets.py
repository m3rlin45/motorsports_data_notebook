"""Interactive widgets for motorsports data notebooks.

This module provides Jupyter widgets for interactive data loading and analysis.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from IPython.display import display

from .corners import compute_lap_distance
from .visualization import format_lap_time

if TYPE_CHECKING:
    import ipywidgets as widgets
    from libxrk.base import LogFile


def load_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from XRK/XRZ file.

    Loads the file using libxrk and adds derived columns:
    - speed_kmh: Speed in km/h (from GPS Speed * 3.6)
    - distance_m: Per-lap cumulative distance in meters
    - lap_time: Lap duration as timedelta (added to laps table)

    Parameters
    ----------
    file_data : str or bytes
        Path to the XRK/XRZ file, or bytes containing file data.

    Returns
    -------
    LogFile
        The enriched LogFile object with derived channels and lap_time column.

    Examples
    --------
    >>> from motorsports_data_notebook.widgets import load_session, FileUpload
    >>> file_upload = FileUpload("sample.xrz")
    >>> file_upload.display()
    >>> log = load_session(file_upload.get_file_data())
    >>> channels = log.get_channels_as_table().to_pandas()
    >>> laps = log.laps.to_pandas()
    """
    from libxrk import aim_xrk

    log = aim_xrk(file_data)

    # Check if GPS Speed channel exists
    has_gps_speed = "GPS Speed" in log.channels

    if has_gps_speed:
        gps_speed_table = log.channels["GPS Speed"]
        timecodes = gps_speed_table.column("timecodes")
        gps_speed = gps_speed_table.column("GPS Speed")

        # Add speed_kmh channel
        speed_kmh = pc.multiply(gps_speed, 3.6)
        speed_kmh_table = pa.table({"timecodes": timecodes, "speed_kmh": speed_kmh})
        log.channels["speed_kmh"] = speed_kmh_table

    # Compute lap_time for laps table (end_time - start_time in ms -> timedelta)
    laps_table = log.laps
    start_times = laps_table.column("start_time")
    end_times = laps_table.column("end_time")
    lap_time_ms = pc.subtract(end_times, start_times)
    # Convert to duration in milliseconds
    lap_time_duration = pc.multiply(lap_time_ms, 1000000)  # ms to nanoseconds
    lap_time_duration = lap_time_duration.cast(pa.duration("ns"))
    log.laps = laps_table.append_column("lap_time", lap_time_duration)

    # Compute distance_m for each lap
    if has_gps_speed:
        timecodes_np = timecodes.to_numpy()
        gps_speed_np = gps_speed.to_numpy()
        start_times_np = start_times.to_numpy()
        end_times_np = end_times.to_numpy()

        distance_m = np.zeros(len(timecodes_np))

        for i in range(len(start_times_np)):
            start_time = start_times_np[i]
            end_time = end_times_np[i]

            lap_mask = (timecodes_np >= start_time) & (timecodes_np <= end_time)
            lap_indices = np.where(lap_mask)[0]

            if len(lap_indices) > 0:
                lap_timecodes = timecodes_np[lap_indices]
                lap_speed = gps_speed_np[lap_indices]
                distance_values = compute_lap_distance(lap_timecodes, lap_speed)
                distance_m[lap_indices] = distance_values

        # Add distance_m channel
        distance_table = pa.table({"timecodes": timecodes, "distance_m": distance_m})
        log.channels["distance_m"] = distance_table

    return log


class FileUpload:
    """Interactive file upload widget for Jupyter notebooks.

    Provides a file upload interface with status feedback.
    Falls back to a default file if no file is uploaded.

    Parameters
    ----------
    default_file : str
        Path to the default file to use if no file is uploaded.

    Examples
    --------
    >>> file_upload = FileUpload("sample_data.xrz")
    >>> file_upload.display()  # Shows upload widget with status
    >>> log = aim_xrk(file_upload.get_file_data())  # Load the file
    """

    def __init__(self, default_file: str) -> None:
        import ipywidgets as widgets

        self._default_file = default_file
        self._uploaded_data: bytes | None = None
        self._uploaded_filename: str | None = None
        self._widgets = widgets

        # Instruction label
        self._instruction = widgets.HTML(
            value="<b>📁 Upload your own .xrk/.xrz file:</b> (or skip to use the sample data)"
        )

        # Create the file upload widget
        self._upload_widget = widgets.FileUpload(
            accept=".xrk,.xrz",
            multiple=False,
            description="Choose File",
            button_style="primary",
        )

        self._status_label = widgets.HTML(
            value=f"<span style='color: #666;'>Using: {default_file} (default)</span>"
        )

        # Set up callback for upload changes
        self._upload_widget.observe(self._on_upload, names="value")

        # Container for layout
        self._container = widgets.VBox([self._instruction, self._upload_widget, self._status_label])

    def _on_upload(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle file upload event."""
        if self._upload_widget.value:
            uploaded = self._upload_widget.value[0]
            self._uploaded_filename = uploaded["name"]
            self._uploaded_data = uploaded["content"].tobytes()
            self._status_label.value = (
                f"<span style='color: green;'><b>✓ Using:</b> {self._uploaded_filename}</span>"
            )

    def display(self) -> None:
        """Display the upload widget and status label."""
        display(self._container)

    def get_file_data(self) -> Union[str, bytes]:
        """Get the file data to pass to aim_xrk.

        Returns
        -------
        str or bytes
            If a file was uploaded, returns the file content as bytes.
            Otherwise, returns the default filename as a string.
        """
        if self._uploaded_data is not None:
            return self._uploaded_data
        return self._default_file


class LapPicker:
    """Dropdown widget for selecting a lap to analyze.

    Provides a dropdown interface for lap selection with formatted lap times.
    The fastest lap (excluding first/last pit laps) is pre-selected by default.

    Parameters
    ----------
    laps : pd.DataFrame
        Laps table with columns: 'num', 'start_time', 'end_time', 'lap_time'.
        The 'lap_time' column should be a timedelta.

    Examples
    --------
    >>> from motorsports_data_notebook.widgets import LapPicker, load_session
    >>> log = load_session("sample.xrz")
    >>> laps = log.laps.to_pandas()
    >>> lap_picker = LapPicker(laps)
    >>> lap_picker.display()
    >>> selected_lap = lap_picker.get_selected_lap()
    """

    def __init__(self, laps: pd.DataFrame) -> None:
        import ipywidgets as widgets

        self._validate_laps(laps)
        self._laps = laps.copy()
        self._widgets = widgets

        # Build dropdown options
        options = self._build_options()

        # Find fastest middle lap for default selection
        default_value = self._get_default_lap_index()

        # Create dropdown widget
        self._dropdown = widgets.Dropdown(
            options=options,
            value=default_value,
            description="Lap:",
            style={"description_width": "auto"},
        )

        # Status label showing current selection
        self._status_label = widgets.HTML(value=self._get_status_html())

        # Update status when selection changes
        self._dropdown.observe(self._on_selection_change, names="value")

        # Container for layout
        self._container = widgets.VBox([self._dropdown, self._status_label])

    def _validate_laps(self, laps: pd.DataFrame) -> None:
        """Validate required columns exist in laps DataFrame."""
        required_columns = {"num", "start_time", "end_time", "lap_time"}
        missing = required_columns - set(laps.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(laps) == 0:
            raise ValueError("Laps DataFrame is empty")

    def _build_options(self) -> list[tuple[str, Any]]:
        """Build dropdown options with formatted labels."""
        options: list[tuple[str, Any]] = []
        n_laps = len(self._laps)
        fastest_idx = self._find_fastest_middle_lap_index()

        for idx, row in self._laps.iterrows():
            lap_num = int(row["num"])
            lap_time_str = format_lap_time(row["lap_time"])

            # Build label with annotations
            label = f"Lap {lap_num}: {lap_time_str}"

            # Add annotations for special laps
            annotations = []
            if idx == self._laps.index[0]:
                annotations.append("out lap")
            if idx == self._laps.index[-1]:
                annotations.append("in lap")
            if n_laps > 2 and idx == fastest_idx:
                annotations.append("fastest")

            if annotations:
                label += f" ({', '.join(annotations)})"

            options.append((label, idx))

        return options

    def _find_fastest_middle_lap_index(self) -> Any:
        """Find the index of the fastest lap, excluding first and last."""
        if len(self._laps) <= 2:
            return None

        # Exclude first and last laps
        middle_laps = self._laps.iloc[1:-1]
        fastest_idx = middle_laps["lap_time"].idxmin()
        return fastest_idx

    def _get_default_lap_index(self) -> Any:
        """Get the default lap index to select."""
        fastest_idx = self._find_fastest_middle_lap_index()
        if fastest_idx is not None:
            return fastest_idx
        # Fall back to first lap if not enough laps
        return self._laps.index[0]

    def _get_status_html(self) -> str:
        """Generate status HTML for current selection."""
        idx = self._dropdown.value
        row = self._laps.loc[idx]
        lap_num = int(row["num"])
        lap_time_str = format_lap_time(row["lap_time"])
        return f"<span style='color: #666;'>Selected: Lap {lap_num} ({lap_time_str})</span>"

    def _on_selection_change(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle dropdown selection change."""
        self._status_label.value = self._get_status_html()

    def display(self) -> None:
        """Display the lap picker widget."""
        display(self._container)

    def get_selected_lap(self) -> "pd.Series[Any]":
        """Return the selected lap row.

        Returns
        -------
        pd.Series
            The row from the laps DataFrame corresponding to the selected lap.
            Contains columns: 'num', 'start_time', 'end_time', 'lap_time'.
        """
        idx = self._dropdown.value
        result: pd.Series[Any] = self._laps.loc[idx]
        return result

    def update_laps(self, laps: pd.DataFrame) -> None:
        """Update the lap picker with new laps data.

        Parameters
        ----------
        laps : pd.DataFrame
            New laps table with columns: 'num', 'start_time', 'end_time', 'lap_time'.
        """
        self._validate_laps(laps)
        self._laps = laps.copy()

        # Rebuild dropdown options
        options = self._build_options()
        default_value = self._get_default_lap_index()

        # Update dropdown
        self._dropdown.options = options
        self._dropdown.value = default_value

        # Update status
        self._status_label.value = self._get_status_html()


class SessionPicker:
    """Combined file upload and lap picker widget.

    Provides a unified interface for uploading telemetry files and selecting
    laps to analyze. The lap picker automatically updates when a new file
    is uploaded, without requiring additional cell execution.

    Parameters
    ----------
    default_file : str
        Path to the default file to use if no file is uploaded.

    Examples
    --------
    >>> from motorsports_data_notebook.widgets import SessionPicker
    >>> session = SessionPicker("sample.xrz")
    >>> session.display()  # Shows file upload and lap picker in one widget
    >>> # After selecting a lap:
    >>> log = session.get_log()
    >>> selected_lap = session.get_selected_lap()
    """

    def __init__(self, default_file: str) -> None:
        import ipywidgets as widgets

        self._default_file = default_file
        self._widgets = widgets
        self._log: "LogFile | None" = None
        self._laps: pd.DataFrame | None = None

        # File upload section
        self._instruction = widgets.HTML(
            value="<b>Upload your own .xrk/.xrz file:</b> (or use the sample data)"
        )

        self._upload_widget = widgets.FileUpload(
            accept=".xrk,.xrz",
            multiple=False,
            description="Choose File",
            button_style="primary",
        )

        self._file_status = widgets.HTML(
            value=f"<span style='color: #666;'>Using: {default_file} (default)</span>"
        )

        # Loading indicator
        self._loading_label = widgets.HTML(value="")

        # Lap picker section (initially hidden until file loads)
        self._lap_label = widgets.HTML(value="<b>Select lap to analyze:</b>")

        self._lap_dropdown = widgets.Dropdown(
            options=[("Loading...", 0)],
            description="Lap:",
            style={"description_width": "auto"},
            disabled=True,
        )

        self._lap_status = widgets.HTML(value="")

        # Set up callbacks
        self._upload_widget.observe(self._on_upload, names="value")
        self._lap_dropdown.observe(self._on_lap_change, names="value")

        # Layout
        self._container = widgets.VBox(
            [
                self._instruction,
                self._upload_widget,
                self._file_status,
                self._loading_label,
                widgets.HTML(value="<hr style='margin: 10px 0;'>"),
                self._lap_label,
                self._lap_dropdown,
                self._lap_status,
            ]
        )

        # Load default file
        self._load_file(default_file)

    def _load_file(self, file_data: Union[str, bytes]) -> None:
        """Load a file and update the lap picker."""
        self._loading_label.value = "<span style='color: #888;'>Loading session data...</span>"
        self._lap_dropdown.disabled = True

        try:
            self._log = load_session(file_data)
            self._laps = self._log.laps.to_pandas()
            self._update_lap_dropdown()
            self._loading_label.value = ""
        except Exception as e:
            self._loading_label.value = f"<span style='color: red;'>Error loading file: {e}</span>"
            self._lap_dropdown.disabled = True

    def _update_lap_dropdown(self) -> None:
        """Update the lap dropdown with current laps data."""
        if self._laps is None or len(self._laps) == 0:
            self._lap_dropdown.options = [("No laps found", 0)]
            self._lap_dropdown.disabled = True
            return

        # Build options
        options: list[tuple[str, Any]] = []
        n_laps = len(self._laps)
        fastest_idx = self._find_fastest_middle_lap_index()

        for idx, row in self._laps.iterrows():
            lap_num = int(row["num"])
            lap_time_str = format_lap_time(row["lap_time"])

            label = f"Lap {lap_num}: {lap_time_str}"

            annotations = []
            if idx == self._laps.index[0]:
                annotations.append("out lap")
            if idx == self._laps.index[-1]:
                annotations.append("in lap")
            if n_laps > 2 and idx == fastest_idx:
                annotations.append("fastest")

            if annotations:
                label += f" ({', '.join(annotations)})"

            options.append((label, idx))

        # Update dropdown
        self._lap_dropdown.options = options
        self._lap_dropdown.value = fastest_idx if fastest_idx is not None else self._laps.index[0]
        self._lap_dropdown.disabled = False

        # Update status
        self._update_lap_status()

    def _find_fastest_middle_lap_index(self) -> Any:
        """Find the index of the fastest lap, excluding first and last."""
        if self._laps is None or len(self._laps) <= 2:
            return None
        middle_laps = self._laps.iloc[1:-1]
        return middle_laps["lap_time"].idxmin()

    def _update_lap_status(self) -> None:
        """Update the lap status label."""
        if self._laps is None:
            return
        idx = self._lap_dropdown.value
        row = self._laps.loc[idx]
        lap_num = int(row["num"])
        lap_time_str = format_lap_time(row["lap_time"])
        self._lap_status.value = (
            f"<span style='color: #666;'>Selected: Lap {lap_num} ({lap_time_str})</span>"
        )

    def _on_upload(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle file upload event."""
        if self._upload_widget.value:
            uploaded = self._upload_widget.value[0]
            filename = uploaded["name"]
            file_data = uploaded["content"].tobytes()

            self._file_status.value = (
                f"<span style='color: green;'><b>✓ Using:</b> {filename}</span>"
            )

            # Load the new file and update lap picker
            self._load_file(file_data)

    def _on_lap_change(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle lap selection change."""
        self._update_lap_status()

    def display(self) -> None:
        """Display the combined session picker widget."""
        display(self._container)

    def get_log(self) -> "LogFile":
        """Return the loaded LogFile.

        Returns
        -------
        LogFile
            The loaded and enriched LogFile object.

        Raises
        ------
        RuntimeError
            If no file has been loaded successfully.
        """
        if self._log is None:
            raise RuntimeError("No session loaded. Upload a file first.")
        return self._log

    def get_laps(self) -> pd.DataFrame:
        """Return the laps DataFrame.

        Returns
        -------
        pd.DataFrame
            The laps table with columns: 'num', 'start_time', 'end_time', 'lap_time'.

        Raises
        ------
        RuntimeError
            If no file has been loaded successfully.
        """
        if self._laps is None:
            raise RuntimeError("No session loaded. Upload a file first.")
        return self._laps.copy()

    def get_selected_lap(self) -> "pd.Series[Any]":
        """Return the selected lap row.

        Returns
        -------
        pd.Series
            The row from the laps DataFrame corresponding to the selected lap.

        Raises
        ------
        RuntimeError
            If no file has been loaded successfully.
        """
        if self._laps is None:
            raise RuntimeError("No session loaded. Upload a file first.")
        idx = self._lap_dropdown.value
        result: pd.Series[Any] = self._laps.loc[idx]
        return result
