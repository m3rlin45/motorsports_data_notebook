"""Interactive widgets for motorsports data notebooks.

This module provides Jupyter widgets for interactive data loading and analysis.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from IPython.display import display

from ._util import clean_laps_table
from .corners import compute_lap_distance
from .visualization import format_lap_time

if TYPE_CHECKING:
    from typing import Union

    import ipywidgets as widgets
    from libxrk.base import LogFile as AimLogFile
    from libibt.base import LogFile as IbtLogFile

    LogFile = Union[AimLogFile, IbtLogFile]


def load_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from a telemetry file.

    Supports AIM (XRK/XRZ) and iRacing (IBT) file formats. Automatically
    detects the file type and dispatches to the appropriate loader.

    Adds derived columns:
    - speed_kmh: Speed in km/h
    - distance_m: Per-lap cumulative distance in meters
    - lap_time: Lap duration as timedelta (added to laps table)

    Parameters
    ----------
    file_data : str or bytes
        Path to the telemetry file, or bytes containing file data.

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
    from ._util import detect_file_type

    if detect_file_type(file_data) == "ibt":
        return _load_ibt_session(file_data)
    return _load_aim_session(file_data)


def _load_aim_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from an AIM XRK/XRZ file."""
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

    log.laps = clean_laps_table(log.laps, log.channels)
    return log


def _load_ibt_session(file_data: Union[str, bytes]) -> "LogFile":
    """Load and prepare session data from an iRacing IBT file."""
    from libibt import ibt

    log = ibt(file_data)

    # iRacing Speed channel is in m/s — add speed_kmh
    has_speed = "Speed" in log.channels
    if has_speed:
        speed_table = log.channels["Speed"]
        timecodes = speed_table.column("timecodes")
        speed_ms = speed_table.column("Speed")

        speed_kmh = pc.multiply(speed_ms, 3.6)
        speed_kmh_table = pa.table({"timecodes": timecodes, "speed_kmh": speed_kmh})
        log.channels["speed_kmh"] = speed_kmh_table

    # Compute per-lap distance_m from Speed (same approach as AIM).
    # iRacing's LapDist wraps at the S/F line, causing discontinuities
    # when lap boundaries don't perfectly align with the wrap point.
    if has_speed:
        timecodes_np = timecodes.to_numpy()
        speed_np = speed_ms.to_numpy(zero_copy_only=False).astype(np.float64)

        laps_table_tmp = log.laps
        start_times_np = laps_table_tmp.column("start_time").to_numpy()
        end_times_np = laps_table_tmp.column("end_time").to_numpy()

        distance_m = np.zeros(len(timecodes_np))
        for i in range(len(start_times_np)):
            lap_mask = (timecodes_np >= start_times_np[i]) & (timecodes_np < end_times_np[i])
            lap_indices = np.where(lap_mask)[0]
            if len(lap_indices) > 0:
                distance_m[lap_indices] = compute_lap_distance(
                    timecodes_np[lap_indices], speed_np[lap_indices]
                )

        distance_table = pa.table({"timecodes": timecodes, "distance_m": distance_m})
        log.channels["distance_m"] = distance_table

    # Compute lap_time for laps table (same logic as AIM)
    laps_table = log.laps
    start_times = laps_table.column("start_time")
    end_times = laps_table.column("end_time")
    lap_time_ms = pc.subtract(end_times, start_times)
    lap_time_duration = pc.multiply(lap_time_ms, 1000000)  # ms to nanoseconds
    lap_time_duration = lap_time_duration.cast(pa.duration("ns"))
    log.laps = laps_table.append_column("lap_time", lap_time_duration)

    log.laps = clean_laps_table(log.laps, log.channels)
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
            value="<b>📁 Upload your own .xrk/.xrz/.ibt file:</b> (or skip to use the sample data)"
        )

        # Create the file upload widget
        self._upload_widget = widgets.FileUpload(
            accept=".xrk,.xrz,.ibt",
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
        fastest_idx = self._find_fastest_lap_index()

        for idx, row in self._laps.iterrows():
            lap_num = int(row["num"])
            lap_time_str = format_lap_time(row["lap_time"])

            # Build label with annotations
            label = f"Lap {lap_num}: {lap_time_str}"

            # Add annotations for special laps
            annotations = []
            if len(self._laps) > 1 and idx == fastest_idx:
                annotations.append("fastest")

            if annotations:
                label += f" ({', '.join(annotations)})"

            options.append((label, idx))

        return options

    def _find_fastest_lap_index(self) -> Any:
        """Find the index of the fastest lap.

        The laps table is pre-cleaned by ``clean_laps_table`` which removes
        incomplete/fractional laps, so all remaining laps are valid candidates.
        """
        if len(self._laps) == 0:
            return None

        fastest_idx = self._laps["lap_time"].idxmin()
        return fastest_idx

    def _get_default_lap_index(self) -> Any:
        """Get the default lap index to select."""
        fastest_idx = self._find_fastest_lap_index()
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
    """Combined file upload, lap picker, and optional channel picker widget.

    Provides a unified interface for uploading telemetry files, selecting
    laps to analyze, and optionally configuring channel name mappings.
    All components automatically update when a new file is uploaded.

    Parameters
    ----------
    default_file : str
        Path to the default file to use if no file is uploaded.
    channel_mapping : dict[str, str], optional
        Default channel name mapping. If provided, a channel picker section
        is displayed allowing users to configure channel names with typeahead
        autocomplete and validation. Keys are logical names (e.g., "throttle"),
        values are default channel names (e.g., "PPS").
    show_lap_picker : bool, default=True
        Whether to show the lap selection dropdown. Set to False when lap
        selection is handled externally (e.g., via get_top_laps).

    Examples
    --------
    >>> from motorsports_data_notebook.widgets import SessionPicker
    >>> session = SessionPicker(
    ...     "sample.xrz",
    ...     channel_mapping={
    ...         "gps_latitude": "GPS Latitude",
    ...         "throttle": "PPS",
    ...         "brake": "BrakePress",
    ...     }
    ... )
    >>> session.display()  # Shows file upload, lap picker, and channel picker
    >>> log = session.get_log()
    >>> selected_lap = session.get_selected_lap()
    >>> CHANNEL_NAMES = session.get_channel_names()
    """

    def __init__(
        self,
        default_file: str,
        channel_mapping: dict[str, str] | None = None,
        show_lap_picker: bool = True,
        tire_channel_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        import ipywidgets as widgets

        self._default_file = default_file
        self._widgets = widgets
        self._log: "LogFile | None" = None
        self._laps: pd.DataFrame | None = None
        self._channel_mapping = channel_mapping
        self._show_lap_picker = show_lap_picker
        self._tire_channel_mapping = tire_channel_mapping

        # File upload section
        self._instruction = widgets.HTML(
            value="<b>Upload your own .xrk/.xrz/.ibt file:</b> (or use the sample data)"
        )

        self._upload_widget = widgets.FileUpload(
            accept=".xrk,.xrz,.ibt",
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

        # Channel picker section (only if channel_mapping provided)
        self._channel_picker: ChannelPicker | None = None
        self._channel_section: "widgets.VBox | None" = None

        if channel_mapping is not None:
            # Create channel picker with empty available channels initially
            # Will be updated when file loads
            self._channel_picker = ChannelPicker(channel_mapping, [])
            self._channel_section = widgets.VBox(
                [
                    widgets.HTML(value="<hr style='margin: 10px 0;'>"),
                    self._channel_picker._container,
                ]
            )

        # Tire channel picker section (only if tire_channel_mapping provided)
        self._tire_picker: TireChannelPicker | None = None
        self._tire_section: "widgets.VBox | None" = None

        if tire_channel_mapping is not None:
            self._tire_picker = TireChannelPicker(tire_channel_mapping, [])
            self._tire_section = widgets.VBox(
                [
                    widgets.HTML(value="<hr style='margin: 10px 0;'>"),
                    self._tire_picker._container,
                ]
            )

        # Set up callbacks
        self._upload_widget.observe(self._on_upload, names="value")
        self._lap_dropdown.observe(self._on_lap_change, names="value")

        # Build layout
        layout_items = [
            self._instruction,
            self._upload_widget,
            self._file_status,
            self._loading_label,
        ]

        if self._show_lap_picker:
            layout_items += [
                widgets.HTML(value="<hr style='margin: 10px 0;'>"),
                self._lap_label,
                self._lap_dropdown,
                self._lap_status,
            ]

        if self._channel_section is not None:
            layout_items.append(self._channel_section)

        if self._tire_section is not None:
            layout_items.append(self._tire_section)

        self._container = widgets.VBox(layout_items)

        # Load default file
        self._load_file(default_file)

    def _load_file(self, file_data: Union[str, bytes]) -> None:
        """Load a file and update the lap picker and channel picker."""
        self._loading_label.value = "<span style='color: #888;'>Loading session data...</span>"
        self._lap_dropdown.disabled = True

        try:
            self._log = load_session(file_data)
            self._laps = self._log.laps.to_pandas()
            self._update_lap_dropdown()
            self._update_channel_picker()

            # Auto-populate channel names from profile if logger ID is known
            self._apply_profile_channels()

            self._loading_label.value = ""
        except Exception as e:
            self._loading_label.value = f"<span style='color: red;'>Error loading file: {e}</span>"
            self._lap_dropdown.disabled = True

    def _apply_profile_channels(self) -> None:
        """Auto-populate channel names from a vehicle profile."""
        if self._channel_picker is None and self._tire_picker is None:
            return
        if self._log is None:
            return

        from .profiles import get_logger_id, get_profile_for_logger

        logger_id = get_logger_id(self._log)
        if not logger_id:
            return

        profile = get_profile_for_logger(logger_id)
        if profile:
            if self._channel_picker is not None:
                self._channel_picker.set_channel_values(profile.channel_names)
            if self._tire_picker is not None:
                self._tire_picker.set_channel_values(profile.channel_names)
            self._file_status.value += (
                f"<br><span style='color: green;'><b>✓</b> Profile: {profile.name} "
                f"(logger {logger_id})</span>"
            )

    def _update_channel_picker(self) -> None:
        """Update the channel picker with available channels from loaded log."""
        if self._log is not None:
            available_channels = list(self._log.channels.keys())
            if self._channel_picker is not None:
                self._channel_picker.update_available_channels(available_channels)
            if self._tire_picker is not None:
                self._tire_picker.update_available_channels(available_channels)

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

    def get_channel_names(self) -> dict[str, str]:
        """Return the configured channel name mapping.

        Merges the regular channel picker and tire channel picker mappings.

        Returns
        -------
        dict[str, str]
            Dictionary mapping logical channel names to configured channel names.

        Raises
        ------
        RuntimeError
            If no channel_mapping was provided during initialization.
        """
        if self._channel_picker is None and self._tire_picker is None:
            raise RuntimeError(
                "No channel_mapping provided. "
                "Pass channel_mapping parameter to SessionPicker to enable channel configuration."
            )
        result: dict[str, str] = {}
        if self._channel_picker is not None:
            result.update(self._channel_picker.get_channel_names())
        if self._tire_picker is not None:
            result.update(self._tire_picker.get_channel_names())
        return result

    def get_file_type(self) -> str:
        """Return the file type of the loaded session.

        Returns
        -------
        str
            ``"aim"`` for AIM XRK/XRZ files, ``"ibt"`` for iRacing IBT files.

        Raises
        ------
        RuntimeError
            If no file has been loaded successfully.
        """
        if self._log is None:
            raise RuntimeError("No session loaded. Upload a file first.")
        from .profiles import is_iracing_session

        return "ibt" if is_iracing_session(self._log) else "aim"


class ChannelPicker:
    """Interactive widget for configuring channel name mappings with validation.

    Provides a form-based interface for mapping logical channel names (e.g., "throttle")
    to actual channel names in the telemetry data (e.g., "PPS"). Features typeahead
    autocomplete and visual validation feedback for unmatched channels.

    Parameters
    ----------
    default_mapping : dict[str, str]
        Dictionary mapping logical channel names to default channel names.
        Keys are logical names (e.g., "throttle"), values are default channel
        names (e.g., "PPS").
    available_channels : list[str]
        List of available channel names in the current log file. Used for
        autocomplete suggestions and validation.

    Examples
    --------
    >>> from motorsports_data_notebook.widgets import SessionPicker, ChannelPicker
    >>> session = SessionPicker("sample.xrz")
    >>> session.display()
    >>> log = session.get_log()
    >>> channel_picker = ChannelPicker(
    ...     default_mapping={
    ...         "gps_latitude": "GPS Latitude",
    ...         "gps_longitude": "GPS Longitude",
    ...         "throttle": "PPS",
    ...         "brake": "BrakePress",
    ...     },
    ...     available_channels=list(log.channels.keys()),
    ... )
    >>> channel_picker.display()
    >>> CHANNEL_NAMES = channel_picker.get_channel_names()
    """

    def __init__(self, default_mapping: dict[str, str], available_channels: list[str]) -> None:
        import ipywidgets as widgets

        self._widgets = widgets
        self._default_mapping = default_mapping.copy()
        self._available_channels = sorted(available_channels)

        # Create header
        self._header = widgets.HTML(value="<b>Channel Configuration</b>")

        # Create a row for each channel mapping
        self._comboboxes: dict[str, "widgets.Combobox"] = {}
        self._status_labels: dict[str, "widgets.HTML"] = {}
        self._rows: list["widgets.HBox"] = []

        for logical_name, default_value in default_mapping.items():
            # Label for the channel
            label = widgets.HTML(
                value=f"<span style='font-family: monospace; width: 150px; display: inline-block;'>{logical_name}:</span>"
            )

            # Combobox with autocomplete
            combobox = widgets.Combobox(
                value=default_value,
                options=self._available_channels,
                ensure_option=False,  # Allow custom values
                placeholder="Type to search...",
                layout=widgets.Layout(width="200px"),
            )
            self._comboboxes[logical_name] = combobox

            # Status indicator
            status = widgets.HTML(value="")
            self._status_labels[logical_name] = status

            # Observe changes
            combobox.observe(self._on_change, names="value")

            # Create row
            row = widgets.HBox([label, combobox, status])
            self._rows.append(row)

        # Summary label
        self._summary = widgets.HTML(value="")

        # Container
        self._container = widgets.VBox([self._header] + self._rows + [self._summary])

        # Initial validation
        self._update_validation()

    def _on_change(self, change: dict) -> None:  # type: ignore[type-arg]
        """Handle combobox value changes."""
        self._update_validation()

    def _update_validation(self) -> None:
        """Update validation status for all channels."""
        unmatched = self.get_unmatched_channels()

        for logical_name, combobox in self._comboboxes.items():
            value = combobox.value
            status_label = self._status_labels[logical_name]

            if value in self._available_channels:
                status_label.value = (
                    "<span style='color: green; margin-left: 10px;'>&#10003; Found</span>"
                )
            else:
                status_label.value = (
                    "<span style='color: red; margin-left: 10px;'>&#10007; Not found</span>"
                )

        # Update summary
        if len(unmatched) == 0:
            self._summary.value = (
                "<span style='color: green; margin-top: 10px; display: block;'>"
                "&#10003; All channels configured correctly</span>"
            )
        else:
            self._summary.value = (
                f"<span style='color: red; margin-top: 10px; display: block;'>"
                f"&#10007; {len(unmatched)} channel(s) not found</span>"
            )

    def display(self) -> None:
        """Display the channel picker widget."""
        display(self._container)

    def get_channel_names(self) -> dict[str, str]:
        """Get the current channel name mapping.

        Returns
        -------
        dict[str, str]
            Dictionary mapping logical channel names to configured channel names.
        """
        return {logical_name: combobox.value for logical_name, combobox in self._comboboxes.items()}

    def get_unmatched_channels(self) -> list[str]:
        """Get list of logical channel names with unmatched values.

        Returns
        -------
        list[str]
            List of logical channel names whose values are not in the
            available channels list.
        """
        unmatched = []
        for logical_name, combobox in self._comboboxes.items():
            if combobox.value not in self._available_channels:
                unmatched.append(logical_name)
        return unmatched

    def set_channel_values(self, channel_names: dict[str, str]) -> None:
        """Set channel values from a mapping (e.g. from a vehicle profile).

        Only updates channels that exist in the picker; ignores unknown keys.

        Parameters
        ----------
        channel_names : dict[str, str]
            Mapping of logical channel names to channel values.
        """
        for logical_name, value in channel_names.items():
            if logical_name in self._comboboxes:
                self._comboboxes[logical_name].value = value
        self._update_validation()

    def update_available_channels(self, available_channels: list[str]) -> None:
        """Update the list of available channels.

        Parameters
        ----------
        available_channels : list[str]
            New list of available channel names.
        """
        self._available_channels = sorted(available_channels)
        for combobox in self._comboboxes.values():
            combobox.options = self._available_channels
        self._update_validation()

    def is_valid(self) -> bool:
        """Check if all channels are matched.

        Returns
        -------
        bool
            True if all configured channel names are found in available channels.
        """
        return len(self.get_unmatched_channels()) == 0


class TireChannelPicker:
    """Interactive 2x2 grid widget for configuring tire temperature channel mappings.

    Renders tire temp channels in a 4-quadrant layout (FL top-left, FR top-right,
    RL bottom-left, RR bottom-right), with a configurable number of sensors per corner.

    Parameters
    ----------
    default_mapping : dict[str, list[str]]
        Default channel names per corner. Keys are "FL", "FR", "RL", "RR",
        values are lists of channel names (e.g. ``["FL_Ch1", ..., "FL_Ch8"]``).
    available_channels : list[str]
        List of available channel names for autocomplete and validation.

    Examples
    --------
    >>> picker = TireChannelPicker(
    ...     {"FL": ["FL_Ch1", "FL_Ch2"], "FR": ["FR_Ch1", "FR_Ch2"],
    ...      "RL": ["RL_Ch1", "RL_Ch2"], "RR": ["RR_Ch1", "RR_Ch2"]},
    ...     available_channels=["FL_Ch1", "FL_Ch2", "FR_Ch1", "FR_Ch2",
    ...                         "RL_Ch1", "RL_Ch2", "RR_Ch1", "RR_Ch2"],
    ... )
    >>> picker.display()
    >>> channel_names = picker.get_channel_names()
    """

    _CORNERS = ["FL", "FR", "RL", "RR"]
    _CORNER_LABELS = {
        "FL": "Front Left",
        "FR": "Front Right",
        "RL": "Rear Left",
        "RR": "Rear Right",
    }
    _CORNER_KEYS = {"FL": "fl", "FR": "fr", "RL": "rl", "RR": "rr"}

    def __init__(
        self, default_mapping: dict[str, list[str]], available_channels: list[str]
    ) -> None:
        import ipywidgets as widgets

        self._widgets = widgets
        self._available_channels = sorted(available_channels)

        # Per-corner comboboxes: {"FL": [combo1, combo2, ...], ...}
        self._corner_combos: dict[str, list["widgets.Combobox"]] = {}
        self._corner_statuses: dict[str, list["widgets.HTML"]] = {}

        # Build per-corner widgets
        corner_boxes: dict[str, "widgets.VBox"] = {}
        for corner in self._CORNERS:
            channel_list = default_mapping.get(corner, [])
            combos: list["widgets.Combobox"] = []
            statuses: list["widgets.HTML"] = []
            rows: list["widgets.HBox"] = []

            header = widgets.HTML(value=f"<b>{self._CORNER_LABELS[corner]}</b>")
            rows.append(header)

            for i, ch_name in enumerate(channel_list):
                label = widgets.HTML(
                    value=f"<span style='font-family: monospace; width: 30px; display: inline-block;'>{i + 1}:</span>"
                )
                combo = widgets.Combobox(
                    value=ch_name,
                    options=self._available_channels,
                    ensure_option=False,
                    placeholder="Type to search...",
                    layout=widgets.Layout(width="160px"),
                )
                combo.observe(self._on_change, names="value")
                combos.append(combo)

                status = widgets.HTML(value="")
                statuses.append(status)

                rows.append(widgets.HBox([label, combo, status]))

            self._corner_combos[corner] = combos
            self._corner_statuses[corner] = statuses
            corner_boxes[corner] = widgets.VBox(rows)

        # Store corner VBox references for later replacement
        self._corner_boxes = corner_boxes

        # Arrange in 2x2 grid
        self._top_row = widgets.HBox(
            [corner_boxes["FL"], corner_boxes["FR"]],
            layout=widgets.Layout(gap="20px"),
        )
        self._bottom_row = widgets.HBox(
            [corner_boxes["RL"], corner_boxes["RR"]],
            layout=widgets.Layout(gap="20px"),
        )

        self._header = widgets.HTML(value="<b>Tire Temperature Channels</b>")
        self._summary = widgets.HTML(value="")
        self._container = widgets.VBox(
            [self._header, self._top_row, self._bottom_row, self._summary]
        )

        self._update_validation()

    def _on_change(self, change: dict) -> None:  # type: ignore[type-arg]
        self._update_validation()

    def _update_validation(self) -> None:
        unmatched_count = 0
        total_count = 0
        for corner in self._CORNERS:
            for combo, status in zip(self._corner_combos[corner], self._corner_statuses[corner]):
                total_count += 1
                if combo.value in self._available_channels:
                    status.value = "<span style='color: green; margin-left: 5px;'>&#10003;</span>"
                else:
                    status.value = "<span style='color: red; margin-left: 5px;'>&#10007;</span>"
                    unmatched_count += 1

        if unmatched_count == 0:
            self._summary.value = (
                "<span style='color: green; margin-top: 5px; display: block;'>"
                "&#10003; All tire channels configured</span>"
            )
        else:
            self._summary.value = (
                f"<span style='color: red; margin-top: 5px; display: block;'>"
                f"&#10007; {unmatched_count} tire channel(s) not found</span>"
            )

    def display(self) -> None:
        """Display the tire channel picker widget."""
        display(self._container)

    def get_channel_names(self) -> dict[str, str]:
        """Get the current tire channel mapping.

        Returns
        -------
        dict[str, str]
            Mapping from canonical keys (``tire_temp_fl_1``, etc.) to channel names.
        """
        result: dict[str, str] = {}
        for corner in self._CORNERS:
            key_prefix = f"tire_temp_{self._CORNER_KEYS[corner]}"
            for i, combo in enumerate(self._corner_combos[corner]):
                result[f"{key_prefix}_{i + 1}"] = combo.value
        return result

    def set_channel_values(self, channel_names: dict[str, str]) -> None:
        """Set channel values from a profile mapping, adjusting sensor count.

        Dynamically adds or removes sensor rows per corner to match the
        number of ``tire_temp_{corner}_{n}`` keys in the profile.

        Parameters
        ----------
        channel_names : dict[str, str]
            Mapping with ``tire_temp_fl_1``, ``tire_temp_fr_2``, etc. keys.
        """
        import ipywidgets as widgets

        for corner in self._CORNERS:
            key_prefix = f"tire_temp_{self._CORNER_KEYS[corner]}"
            # Collect profile values for this corner
            profile_channels: list[str] = []
            n = 1
            while f"{key_prefix}_{n}" in channel_names:
                profile_channels.append(channel_names[f"{key_prefix}_{n}"])
                n += 1

            if not profile_channels:
                continue

            current_count = len(self._corner_combos[corner])
            target_count = len(profile_channels)

            if target_count != current_count:
                # Rebuild this corner's widgets
                combos: list["widgets.Combobox"] = []
                statuses: list["widgets.HTML"] = []
                rows: list["widgets.HBox | widgets.HTML"] = []

                header = widgets.HTML(value=f"<b>{self._CORNER_LABELS[corner]}</b>")
                rows.append(header)

                for i, ch_name in enumerate(profile_channels):
                    label = widgets.HTML(
                        value=f"<span style='font-family: monospace; width: 30px; display: inline-block;'>{i + 1}:</span>"
                    )
                    combo = widgets.Combobox(
                        value=ch_name,
                        options=self._available_channels,
                        ensure_option=False,
                        placeholder="Type to search...",
                        layout=widgets.Layout(width="160px"),
                    )
                    combo.observe(self._on_change, names="value")
                    combos.append(combo)

                    status = widgets.HTML(value="")
                    statuses.append(status)

                    rows.append(widgets.HBox([label, combo, status]))

                self._corner_combos[corner] = combos
                self._corner_statuses[corner] = statuses

                # Replace the corner VBox in the grid row
                new_box = widgets.VBox(rows)
                self._corner_boxes[corner] = new_box
                corner_idx = self._CORNERS.index(corner)
                grid_row = self._top_row if corner_idx < 2 else self._bottom_row
                col_idx = corner_idx % 2
                children = list(grid_row.children)
                children[col_idx] = new_box
                grid_row.children = tuple(children)
            else:
                # Same count — just update values
                for i, ch_name in enumerate(profile_channels):
                    self._corner_combos[corner][i].value = ch_name

        self._update_validation()

    def update_available_channels(self, available_channels: list[str]) -> None:
        """Update the list of available channels for autocomplete.

        Parameters
        ----------
        available_channels : list[str]
            New list of available channel names.
        """
        self._available_channels = sorted(available_channels)
        for corner in self._CORNERS:
            for combo in self._corner_combos[corner]:
                combo.options = self._available_channels
        self._update_validation()
