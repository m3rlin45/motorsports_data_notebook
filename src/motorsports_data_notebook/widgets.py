"""Interactive widgets for motorsports data notebooks.

This module provides Jupyter widgets for interactive data loading and analysis.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from IPython.display import display

from .corners import compute_lap_distance

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
