"""Tests for widgets module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.widgets import (
    load_session,
    LapPicker,
    SessionPicker,
    ChannelPicker,
    TireChannelPicker,
)

# Path to patch aim_xrk - it's imported inside the function
AIM_XRK_PATCH_PATH = "libxrk.aim_xrk"
IBT_PATCH_PATH = "libibt.ibt"


# ============================================================================
# Fixtures for mocking
# ============================================================================


@pytest.fixture
def mock_log_file():
    """Create a mock LogFile object."""
    # Create sample channel data
    n_samples = 300
    timecodes = np.linspace(0, 180000, n_samples).astype(np.int64)  # 3 minutes of data
    gps_speed_values = np.random.uniform(20, 50, n_samples)  # m/s

    # GPS Speed channel table
    gps_speed_table = pa.table(
        {
            "timecodes": timecodes,
            "GPS Speed": gps_speed_values,
        }
    )

    # Laps table (3 laps)
    laps_table = pa.table(
        {
            "num": [1, 2, 3],
            "start_time": [0, 60000, 120000],
            "end_time": [60000, 120000, 180000],
        }
    )

    mock_log = MagicMock()
    mock_log.channels = {"GPS Speed": gps_speed_table}
    mock_log.laps = laps_table
    mock_log.metadata = {}
    mock_log.file_name = "test.xrz"

    return mock_log


# ============================================================================
# Tests for load_session
# ============================================================================


class TestLoadSession:
    """Tests for load_session function."""

    def test_load_session_returns_log_file(self, mock_log_file):
        """Test that load_session returns a LogFile object."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        # Should return the same mock object (enriched)
        assert result is mock_log_file

    def test_load_session_adds_speed_kmh_channel(self, mock_log_file):
        """Test that speed_kmh channel is added."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        assert "speed_kmh" in result.channels

    def test_load_session_speed_kmh_is_correct(self, mock_log_file):
        """Test that speed_kmh is GPS Speed * 3.6."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        speed_kmh_table = result.channels["speed_kmh"]
        speed_kmh_arr = speed_kmh_table.column("speed_kmh").to_numpy()

        # Get original GPS Speed from channels dict
        gps_speed_arr = mock_log_file.channels["GPS Speed"].column("GPS Speed").to_numpy()

        # speed_kmh should be GPS Speed * 3.6
        expected = gps_speed_arr * 3.6
        np.testing.assert_array_almost_equal(speed_kmh_arr, expected, decimal=5)

    def test_load_session_adds_distance_m_channel(self, mock_log_file):
        """Test that distance_m channel is added."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        assert "distance_m" in result.channels

    def test_load_session_distance_resets_per_lap(self, mock_log_file):
        """Test that distance resets at the start of each lap."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        distance_table = result.channels["distance_m"]
        distance_df = distance_table.to_pandas()

        # Get timecodes
        timecodes = distance_df["timecodes"].values
        distance_m = distance_df["distance_m"].values

        # Find lap boundaries and check distance resets
        laps_df = result.laps.to_pandas()
        for _, lap in laps_df.iterrows():
            lap_mask = (timecodes >= lap["start_time"]) & (timecodes <= lap["end_time"])
            lap_distances = distance_m[lap_mask]

            if len(lap_distances) > 0:
                # Distance should start near 0 at each lap
                assert lap_distances[0] == pytest.approx(0, abs=1)

    def test_load_session_adds_lap_time_column(self, mock_log_file):
        """Test that lap_time column is added to laps."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        laps_df = result.laps.to_pandas()
        assert "lap_time" in laps_df.columns

    def test_load_session_lap_time_is_timedelta(self, mock_log_file):
        """Test that lap_time column is a timedelta."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        laps_df = result.laps.to_pandas()
        assert pd.api.types.is_timedelta64_dtype(laps_df["lap_time"])

    def test_load_session_lap_time_is_correct(self, mock_log_file):
        """Test that lap_time values are correct."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            result = load_session(b"fake_data")

        laps_df = result.laps.to_pandas()

        # Each lap should be 60 seconds
        for _, lap in laps_df.iterrows():
            expected_duration = lap["end_time"] - lap["start_time"]
            actual_duration = lap["lap_time"].total_seconds() * 1000
            assert actual_duration == pytest.approx(expected_duration, abs=1)

    def test_load_session_accepts_string_path(self, mock_log_file):
        """Test that load_session accepts string file paths."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file) as mock_aim_xrk:
            load_session("path/to/file.xrz")

        mock_aim_xrk.assert_called_once_with("path/to/file.xrz")

    def test_load_session_accepts_bytes(self, mock_log_file):
        """Test that load_session accepts bytes."""
        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file) as mock_aim_xrk:
            load_session(b"file_content")

        mock_aim_xrk.assert_called_once_with(b"file_content")


class TestLoadSessionEdgeCases:
    """Edge case tests for load_session."""

    def test_load_session_handles_missing_gps_speed(self):
        """Test handling when GPS Speed channel is missing."""
        # Create mock with no GPS Speed
        mock_log = MagicMock()
        mock_log.channels = {}
        mock_log.laps = pa.table(
            {
                "num": [1],
                "start_time": [0],
                "end_time": [60000],
            }
        )

        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log):
            result = load_session(b"fake_data")

        # Should still work, just without speed_kmh and distance_m
        assert "speed_kmh" not in result.channels
        assert "distance_m" not in result.channels

    def test_load_session_handles_empty_laps(self):
        """Test handling when laps table is empty."""
        mock_log = MagicMock()
        gps_speed_table = pa.table(
            {
                "timecodes": np.array([0, 1000, 2000], dtype=np.int64),
                "GPS Speed": np.array([30.0, 35.0, 40.0]),
            }
        )
        mock_log.channels = {"GPS Speed": gps_speed_table}
        mock_log.laps = pa.table(
            {
                "num": pa.array([], type=pa.int64()),
                "start_time": pa.array([], type=pa.int64()),
                "end_time": pa.array([], type=pa.int64()),
            }
        )

        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log):
            result = load_session(b"fake_data")

        # Should complete without error
        assert result is mock_log


# ============================================================================
# Fixtures for LapPicker
# ============================================================================


@pytest.fixture
def sample_laps_df():
    """Create a sample laps DataFrame with 5 laps."""
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5],
            "start_time": [0, 60000, 120000, 180000, 240000],
            "end_time": [60000, 120000, 180000, 240000, 300000],
            "lap_time": pd.to_timedelta([65, 62, 58, 61, 70], unit="s"),
        }
    )


@pytest.fixture
def few_laps_df():
    """Create a laps DataFrame with only 2 laps."""
    return pd.DataFrame(
        {
            "num": [1, 2],
            "start_time": [0, 60000],
            "end_time": [60000, 120000],
            "lap_time": pd.to_timedelta([65, 62], unit="s"),
        }
    )


@pytest.fixture
def mock_ipywidgets():
    """Create mock ipywidgets module."""
    mock_widgets = MagicMock()

    # Track the value passed to Dropdown constructor
    def create_dropdown(**kwargs):
        mock_dropdown = MagicMock()
        mock_dropdown.value = kwargs.get("value")
        mock_dropdown.observe = MagicMock()
        return mock_dropdown

    mock_widgets.Dropdown.side_effect = create_dropdown
    mock_widgets.HTML.return_value = MagicMock()
    mock_widgets.VBox.return_value = MagicMock()
    return mock_widgets


# ============================================================================
# Tests for LapPicker
# ============================================================================


class TestLapPicker:
    """Tests for LapPicker widget."""

    def test_preselects_fastest_lap(self, sample_laps_df, mock_ipywidgets):
        """Test that the fastest lap is pre-selected."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            # Lap 3 (index 2) has the fastest time (58s)
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 2

    def test_selects_fastest_lap_even_if_first(self, sample_laps_df, mock_ipywidgets):
        """Test that first lap is selected if it's the fastest (laps are pre-cleaned)."""
        modified_df = sample_laps_df.copy()
        modified_df.loc[0, "lap_time"] = pd.Timedelta(seconds=50)

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(modified_df)

            # Lap 1 (index 0) is fastest — should be selected since
            # incomplete laps are already removed by clean_laps_table
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 0

    def test_marks_fastest_lap_in_labels(self, sample_laps_df, mock_ipywidgets):
        """Test that fastest lap is marked in label."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            options = call_kwargs["options"]

            # Lap 3 (index 2) should have "fastest" in label
            assert "fastest" in options[2][0]
            # Other laps should not have "fastest"
            assert "fastest" not in options[0][0]
            assert "fastest" not in options[1][0]
            assert "fastest" not in options[3][0]

    def test_validates_required_columns(self, mock_ipywidgets):
        """Test that validation fails for missing columns."""
        invalid_df = pd.DataFrame({"num": [1, 2, 3]})

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            with pytest.raises(ValueError, match="Missing required columns"):
                LapPicker(invalid_df)

    def test_validates_empty_dataframe(self, mock_ipywidgets):
        """Test that validation fails for empty DataFrame."""
        empty_df = pd.DataFrame(columns=["num", "start_time", "end_time", "lap_time"])

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            with pytest.raises(ValueError, match="empty"):
                LapPicker(empty_df)

    def test_handles_few_laps(self, few_laps_df, mock_ipywidgets):
        """Test handling when there are only 2 laps."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(few_laps_df)

            # Should select the fastest lap (lap 2 at 62s, index 1)
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 1

            # Fastest lap should be marked
            options = call_kwargs["options"]
            assert "fastest" in options[1][0]
            assert "fastest" not in options[0][0]

    def test_get_selected_lap_returns_correct_row(self, sample_laps_df, mock_ipywidgets):
        """Test that get_selected_lap returns the correct lap row."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)
            # The picker defaults to lap 3 (index 2) which is the fastest middle lap
            selected = picker.get_selected_lap()

            assert selected["num"] == 3
            assert selected["start_time"] == 120000
            assert selected["end_time"] == 180000

    def test_lap_time_format_in_options(self, sample_laps_df, mock_ipywidgets):
        """Test that lap times are formatted correctly in dropdown options."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            options = call_kwargs["options"]

            # Lap 3 has 58 seconds = 0:58.000
            assert "0:58.000" in options[2][0]

    def test_update_laps_changes_options(self, sample_laps_df, mock_ipywidgets):
        """Test that update_laps updates the dropdown options."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            # Create new laps data with different times
            new_laps = pd.DataFrame(
                {
                    "num": [1, 2, 3],
                    "start_time": [0, 60000, 120000],
                    "end_time": [60000, 120000, 180000],
                    "lap_time": pd.to_timedelta([70, 55, 80], unit="s"),
                }
            )

            picker.update_laps(new_laps)

            # Check that dropdown options were updated (3 laps now)
            assert len(picker._dropdown.options) == 3
            # Check that value was updated to new fastest (lap 2, index 1)
            assert picker._dropdown.value == 1


# ============================================================================
# Tests for SessionPicker
# ============================================================================


@pytest.fixture
def mock_ipywidgets_session():
    """Create mock ipywidgets module for SessionPicker tests."""
    mock_widgets = MagicMock()

    def create_dropdown(**kwargs):
        mock_dropdown = MagicMock()
        mock_dropdown.value = kwargs.get("value", 0)
        mock_dropdown.options = kwargs.get("options", [])
        mock_dropdown.disabled = kwargs.get("disabled", False)
        mock_dropdown.observe = MagicMock()
        return mock_dropdown

    def create_combobox(**kwargs):
        mock_combo = MagicMock()
        mock_combo.value = kwargs.get("value", "")
        mock_combo.options = kwargs.get("options", [])
        mock_combo.observe = MagicMock()
        return mock_combo

    mock_widgets.Dropdown.side_effect = create_dropdown
    mock_widgets.Combobox.side_effect = create_combobox
    mock_widgets.HTML.return_value = MagicMock()
    mock_widgets.HBox.return_value = MagicMock()
    mock_widgets.VBox.return_value = MagicMock()
    mock_widgets.Layout.return_value = MagicMock()
    mock_widgets.FileUpload.return_value = MagicMock()
    mock_widgets.FileUpload.return_value.observe = MagicMock()
    mock_widgets.FileUpload.return_value.value = None
    return mock_widgets


class TestSessionPicker:
    """Tests for SessionPicker widget."""

    def test_loads_default_file_on_init(self, mock_log_file, mock_ipywidgets_session):
        """Test that SessionPicker loads the default file on initialization."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")

            # Should have loaded the log
            assert picker._log is not None
            assert picker._laps is not None

    def test_get_log_returns_loaded_log(self, mock_log_file, mock_ipywidgets_session):
        """Test that get_log returns the loaded LogFile."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")
            log = picker.get_log()

            assert log is mock_log_file

    def test_get_laps_returns_dataframe(self, mock_log_file, mock_ipywidgets_session):
        """Test that get_laps returns a DataFrame copy."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")
            laps = picker.get_laps()

            assert isinstance(laps, pd.DataFrame)
            assert "num" in laps.columns
            assert "lap_time" in laps.columns

    def test_get_selected_lap_returns_series(self, mock_log_file, mock_ipywidgets_session):
        """Test that get_selected_lap returns a Series."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")
            selected = picker.get_selected_lap()

            assert isinstance(selected, pd.Series)
            assert "num" in selected.index
            assert "start_time" in selected.index
            assert "end_time" in selected.index

    def test_raises_error_when_no_session_loaded(self, mock_ipywidgets_session):
        """Test that methods raise RuntimeError when no session is loaded."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, side_effect=Exception("File not found")),
        ):
            picker = SessionPicker("nonexistent.xrz")

            with pytest.raises(RuntimeError, match="No session loaded"):
                picker.get_log()

            with pytest.raises(RuntimeError, match="No session loaded"):
                picker.get_laps()

            with pytest.raises(RuntimeError, match="No session loaded"):
                picker.get_selected_lap()

    def test_with_channel_mapping_creates_channel_picker(
        self, mock_log_file, mock_ipywidgets_session
    ):
        """Test that providing channel_mapping creates a channel picker."""
        channel_mapping = {"throttle": "PPS", "brake": "BrakePress"}

        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz", channel_mapping=channel_mapping)

            assert picker._channel_picker is not None
            assert picker._channel_section is not None

    def test_without_channel_mapping_no_channel_picker(
        self, mock_log_file, mock_ipywidgets_session
    ):
        """Test that not providing channel_mapping doesn't create a channel picker."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")

            assert picker._channel_picker is None
            assert picker._channel_section is None

    def test_get_channel_names_returns_mapping(self, mock_log_file, mock_ipywidgets_session):
        """Test that get_channel_names returns the channel mapping."""
        channel_mapping = {"throttle": "PPS", "brake": "BrakePress"}

        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz", channel_mapping=channel_mapping)
            result = picker.get_channel_names()

            assert result == channel_mapping

    def test_get_channel_names_raises_without_mapping(self, mock_log_file, mock_ipywidgets_session):
        """Test that get_channel_names raises error when no mapping provided."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")

            with pytest.raises(RuntimeError, match="No channel_mapping provided"):
                picker.get_channel_names()

    def test_channel_picker_updates_on_file_load(self, mock_log_file, mock_ipywidgets_session):
        """Test that channel picker is updated with available channels on file load."""
        channel_mapping = {"throttle": "PPS"}

        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz", channel_mapping=channel_mapping)

            # The channel picker should have been updated with available channels
            # from the log file (which has "GPS Speed" channel)
            assert picker._channel_picker is not None
            assert "GPS Speed" in picker._channel_picker._available_channels


# ============================================================================
# Fixtures for ChannelPicker
# ============================================================================


@pytest.fixture
def sample_available_channels():
    """Sample list of available channels."""
    return [
        "GPS Latitude",
        "GPS Longitude",
        "GPS Speed",
        "PPS",
        "BrakePress",
        "LateralAcc",
        "SteerAngle",
        "speed_kmh",
        "distance_m",
    ]


@pytest.fixture
def sample_channel_mapping():
    """Sample default channel mapping."""
    return {
        "gps_latitude": "GPS Latitude",
        "gps_longitude": "GPS Longitude",
        "throttle": "PPS",
        "brake": "BrakePress",
    }


@pytest.fixture
def mock_ipywidgets_channel():
    """Create mock ipywidgets module for ChannelPicker tests."""
    mock_widgets = MagicMock()

    def create_combobox(**kwargs):
        mock_combo = MagicMock()
        mock_combo.value = kwargs.get("value", "")
        mock_combo.options = kwargs.get("options", [])
        mock_combo.observe = MagicMock()
        return mock_combo

    mock_widgets.Combobox.side_effect = create_combobox
    mock_widgets.HTML.return_value = MagicMock()
    mock_widgets.HBox.return_value = MagicMock()
    mock_widgets.VBox.return_value = MagicMock()
    mock_widgets.Layout.return_value = MagicMock()
    return mock_widgets


# ============================================================================
# Tests for ChannelPicker
# ============================================================================


class TestChannelPicker:
    """Tests for ChannelPicker widget."""

    def test_initializes_with_default_mapping(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that ChannelPicker initializes with the provided default mapping."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            # Should create comboboxes for each channel
            assert len(picker._comboboxes) == len(sample_channel_mapping)
            assert "gps_latitude" in picker._comboboxes
            assert "throttle" in picker._comboboxes

    def test_get_channel_names_returns_current_values(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that get_channel_names returns the current combobox values."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            result = picker.get_channel_names()

            assert result == sample_channel_mapping

    def test_get_unmatched_channels_returns_empty_when_all_valid(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that get_unmatched_channels returns empty list when all channels are valid."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            unmatched = picker.get_unmatched_channels()

            assert unmatched == []

    def test_get_unmatched_channels_returns_invalid_channels(
        self, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that get_unmatched_channels returns channels not in available list."""
        mapping_with_invalid = {
            "gps_latitude": "GPS Latitude",
            "throttle": "InvalidChannel",  # Not in available_channels
        }

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(mapping_with_invalid, sample_available_channels)

            unmatched = picker.get_unmatched_channels()

            assert "throttle" in unmatched
            assert "gps_latitude" not in unmatched

    def test_is_valid_returns_true_when_all_matched(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that is_valid returns True when all channels are matched."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            assert picker.is_valid() is True

    def test_is_valid_returns_false_when_unmatched(
        self, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that is_valid returns False when channels are unmatched."""
        mapping_with_invalid = {
            "gps_latitude": "GPS Latitude",
            "throttle": "InvalidChannel",
        }

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(mapping_with_invalid, sample_available_channels)

            assert picker.is_valid() is False

    def test_update_available_channels_updates_options(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that update_available_channels updates combobox options."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            new_channels = ["NewChannel1", "NewChannel2"]
            picker.update_available_channels(new_channels)

            assert picker._available_channels == sorted(new_channels)
            # All comboboxes should have updated options
            for combobox in picker._comboboxes.values():
                assert combobox.options == sorted(new_channels)

    def test_update_available_channels_updates_validation(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that update_available_channels triggers validation update."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            # Initially all valid
            assert picker.is_valid() is True

            # Update to channels that don't include the defaults
            new_channels = ["NewChannel1", "NewChannel2"]
            picker.update_available_channels(new_channels)

            # Now all channels should be unmatched
            assert picker.is_valid() is False
            assert len(picker.get_unmatched_channels()) == len(sample_channel_mapping)

    def test_combobox_values_sorted_alphabetically(
        self, sample_channel_mapping, sample_available_channels, mock_ipywidgets_channel
    ):
        """Test that available channels are sorted alphabetically."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, sample_available_channels)

            assert picker._available_channels == sorted(sample_available_channels)

    def test_empty_available_channels(self, sample_channel_mapping, mock_ipywidgets_channel):
        """Test handling of empty available channels list."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_channel}):
            picker = ChannelPicker(sample_channel_mapping, [])

            assert picker._available_channels == []
            assert picker.is_valid() is False
            assert len(picker.get_unmatched_channels()) == len(sample_channel_mapping)


# ============================================================================
# Fixtures for IBT loading
# ============================================================================


@pytest.fixture
def mock_ibt_log_file():
    """Create a mock iRacing IBT LogFile object."""
    n_samples = 300
    timecodes = np.linspace(0, 180000, n_samples).astype(np.int64)
    speed_values = np.random.uniform(20, 50, n_samples)  # m/s
    lapdist_values = np.linspace(0, 4000, n_samples)  # meters

    speed_table = pa.table({"timecodes": timecodes, "Speed": speed_values})
    lapdist_table = pa.table({"timecodes": timecodes, "LapDist": lapdist_values})

    laps_table = pa.table(
        {
            "num": [1, 2, 3],
            "start_time": [0, 60000, 120000],
            "end_time": [60000, 120000, 180000],
        }
    )

    mock_log = MagicMock()
    mock_log.channels = {"Speed": speed_table, "LapDist": lapdist_table}
    mock_log.laps = laps_table
    mock_log.metadata = {"session_info_yaml": "some_yaml_content"}
    mock_log.file_name = "test.ibt"

    return mock_log


# ============================================================================
# Tests for IBT load_session
# ============================================================================


class TestLoadIbtSession:
    """Tests for load_session with iRacing IBT files."""

    def test_dispatches_to_libibt_for_ibt_bytes(self, mock_ibt_log_file):
        """load_session should call libibt.ibt for IBT magic bytes."""
        # IBT magic bytes header
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file) as mock_ibt:
            load_session(ibt_bytes)

        mock_ibt.assert_called_once_with(ibt_bytes)

    def test_dispatches_to_libibt_for_ibt_path(self, mock_ibt_log_file):
        """load_session should call libibt.ibt for .ibt file paths."""
        with patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file) as mock_ibt:
            load_session("path/to/file.ibt")

        mock_ibt.assert_called_once_with("path/to/file.ibt")

    def test_dispatches_to_libxrk_for_aim_bytes(self, mock_log_file):
        """load_session should call libxrk.aim_xrk for non-IBT bytes."""
        aim_bytes = b"\xff\x00\x00\x00" + b"\x00" * 100

        with patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file):
            load_session(aim_bytes)

    def test_adds_speed_kmh_channel(self, mock_ibt_log_file):
        """IBT load_session should add speed_kmh channel from Speed * 3.6."""
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file):
            result = load_session(ibt_bytes)

        assert "speed_kmh" in result.channels
        speed_kmh = result.channels["speed_kmh"].column("speed_kmh").to_numpy()
        speed_ms = mock_ibt_log_file.channels["Speed"].column("Speed").to_numpy()
        np.testing.assert_array_almost_equal(speed_kmh, speed_ms * 3.6, decimal=5)

    def test_adds_distance_m_from_speed(self, mock_ibt_log_file):
        """IBT load_session should compute distance_m from Speed integration."""
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file):
            result = load_session(ibt_bytes)

        assert "distance_m" in result.channels
        distance_m = result.channels["distance_m"].column("distance_m").to_numpy()
        # Should start at 0 and have positive values (speed-integrated distance)
        assert distance_m[0] == 0.0
        assert np.max(distance_m) > 0.0

    def test_adds_lap_time_column(self, mock_ibt_log_file):
        """IBT load_session should add lap_time column to laps table."""
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file):
            result = load_session(ibt_bytes)

        laps_df = result.laps.to_pandas()
        assert "lap_time" in laps_df.columns
        assert pd.api.types.is_timedelta64_dtype(laps_df["lap_time"])

    def test_handles_missing_speed_channel(self):
        """IBT load_session should work without Speed channel."""
        mock_log = MagicMock()
        mock_log.channels = {}
        mock_log.laps = pa.table({"num": [1], "start_time": [0], "end_time": [60000]})
        mock_log.metadata = {"session_info_yaml": "yaml"}

        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with patch(IBT_PATCH_PATH, return_value=mock_log):
            result = load_session(ibt_bytes)

        assert "speed_kmh" not in result.channels
        assert "distance_m" not in result.channels


# ============================================================================
# Tests for SessionPicker.get_file_type
# ============================================================================


class TestSessionPickerGetFileType:
    """Tests for SessionPicker.get_file_type method."""

    def test_returns_aim_for_aim_files(self, mock_log_file, mock_ipywidgets_session):
        """get_file_type should return 'aim' for AIM files."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, return_value=mock_log_file),
        ):
            picker = SessionPicker("test.xrz")
            assert picker.get_file_type() == "aim"

    def test_returns_ibt_for_ibt_files(self, mock_ibt_log_file, mock_ipywidgets_session):
        """get_file_type should return 'ibt' for iRacing files."""
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100

        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(IBT_PATCH_PATH, return_value=mock_ibt_log_file),
        ):
            # Load with IBT bytes by simulating upload
            picker = SessionPicker.__new__(SessionPicker)
            picker._log = mock_ibt_log_file
            picker._laps = None
            picker._channel_picker = None

            assert picker.get_file_type() == "ibt"

    def test_raises_when_no_session(self, mock_ipywidgets_session):
        """get_file_type should raise RuntimeError when no session loaded."""
        with (
            patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_session}),
            patch(AIM_XRK_PATCH_PATH, side_effect=Exception("File not found")),
        ):
            picker = SessionPicker("nonexistent.xrz")

            with pytest.raises(RuntimeError, match="No session loaded"):
                picker.get_file_type()


# ============================================================================
# Fixtures for TireChannelPicker
# ============================================================================


@pytest.fixture
def tire_available_channels():
    """Available channels for tire picker tests."""
    channels = []
    for pos in ["FL", "FR", "RL", "RR"]:
        for ch in range(1, 9):
            channels.append(f"{pos}_Ch{ch}")
    return channels


@pytest.fixture
def tire_mapping_8():
    """Default 8-sensor tire mapping."""
    return {
        "FL": [f"FL_Ch{i}" for i in range(1, 9)],
        "FR": [f"FR_Ch{i}" for i in range(1, 9)],
        "RL": [f"RL_Ch{i}" for i in range(1, 9)],
        "RR": [f"RR_Ch{i}" for i in range(1, 9)],
    }


@pytest.fixture
def tire_mapping_3():
    """Default 3-zone tire mapping (iRacing)."""
    return {
        "FL": ["LFtempL", "LFtempM", "LFtempR"],
        "FR": ["RFtempL", "RFtempM", "RFtempR"],
        "RL": ["LRtempL", "LRtempM", "LRtempR"],
        "RR": ["RRtempL", "RRtempM", "RRtempR"],
    }


@pytest.fixture
def mock_ipywidgets_tire():
    """Create mock ipywidgets module for TireChannelPicker tests."""
    mock_widgets = MagicMock()

    def create_combobox(**kwargs):
        mock_combo = MagicMock()
        mock_combo.value = kwargs.get("value", "")
        mock_combo.options = kwargs.get("options", [])
        mock_combo.observe = MagicMock()
        return mock_combo

    class MockBox:
        """Mock HBox/VBox that supports .children attribute."""

        def __init__(self, children=None, **kwargs):
            self.children = tuple(children) if children else ()

    mock_widgets.Combobox.side_effect = create_combobox
    mock_widgets.HTML.return_value = MagicMock()
    mock_widgets.HBox.side_effect = MockBox
    mock_widgets.VBox.side_effect = MockBox
    mock_widgets.Layout.return_value = MagicMock()
    return mock_widgets


# ============================================================================
# Tests for TireChannelPicker
# ============================================================================


class TestTireChannelPicker:
    """Tests for TireChannelPicker widget."""

    def test_creates_with_8_sensors(
        self, tire_mapping_8, tire_available_channels, mock_ipywidgets_tire
    ):
        """Test creation with 8-sensor default mapping."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_8, tire_available_channels)

            # Should have 8 combos per corner
            for corner in ["FL", "FR", "RL", "RR"]:
                assert len(picker._corner_combos[corner]) == 8

    def test_creates_with_3_sensors(self, tire_mapping_3, mock_ipywidgets_tire):
        """Test creation with 3-zone default mapping."""
        iracing_channels = [
            f"{pos}temp{z}" for pos in ["LF", "RF", "LR", "RR"] for z in ["L", "M", "R"]
        ]
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_3, iracing_channels)

            for corner in ["FL", "FR", "RL", "RR"]:
                assert len(picker._corner_combos[corner]) == 3

    def test_get_channel_names_returns_tire_temp_keys(
        self, tire_mapping_8, tire_available_channels, mock_ipywidgets_tire
    ):
        """Test that get_channel_names returns correct tire_temp_* keys."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_8, tire_available_channels)

            names = picker.get_channel_names()

            assert names["tire_temp_fl_1"] == "FL_Ch1"
            assert names["tire_temp_fl_8"] == "FL_Ch8"
            assert names["tire_temp_fr_1"] == "FR_Ch1"
            assert names["tire_temp_rr_8"] == "RR_Ch8"
            assert len(names) == 32  # 4 corners x 8 sensors

    def test_get_channel_names_3_zone(self, tire_mapping_3, mock_ipywidgets_tire):
        """Test get_channel_names with 3-zone mapping."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_3, [])

            names = picker.get_channel_names()

            assert names["tire_temp_fl_1"] == "LFtempL"
            assert names["tire_temp_fl_2"] == "LFtempM"
            assert names["tire_temp_fl_3"] == "LFtempR"
            assert len(names) == 12  # 4 corners x 3 zones

    def test_set_channel_values_adjusts_sensor_count(
        self, tire_mapping_8, tire_available_channels, mock_ipywidgets_tire
    ):
        """Test that set_channel_values adjusts sensor count from profile."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_8, tire_available_channels)

            # Start with 8 sensors
            assert len(picker._corner_combos["FL"]) == 8

            # Apply iRacing 3-zone profile
            profile_names = {
                "tire_temp_fl_1": "LFtempL",
                "tire_temp_fl_2": "LFtempM",
                "tire_temp_fl_3": "LFtempR",
                "tire_temp_fr_1": "RFtempL",
                "tire_temp_fr_2": "RFtempM",
                "tire_temp_fr_3": "RFtempR",
                "tire_temp_rl_1": "LRtempL",
                "tire_temp_rl_2": "LRtempM",
                "tire_temp_rl_3": "LRtempR",
                "tire_temp_rr_1": "RRtempL",
                "tire_temp_rr_2": "RRtempM",
                "tire_temp_rr_3": "RRtempR",
            }
            picker.set_channel_values(profile_names)

            # Should now have 3 sensors per corner
            for corner in ["FL", "FR", "RL", "RR"]:
                assert len(picker._corner_combos[corner]) == 3

            # Values should match
            names = picker.get_channel_names()
            assert names["tire_temp_fl_1"] == "LFtempL"
            assert len(names) == 12

    def test_update_available_channels(
        self, tire_mapping_8, tire_available_channels, mock_ipywidgets_tire
    ):
        """Test that update_available_channels updates all combos."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets_tire}):
            picker = TireChannelPicker(tire_mapping_8, tire_available_channels)

            new_channels = ["NewCh1", "NewCh2"]
            picker.update_available_channels(new_channels)

            assert picker._available_channels == sorted(new_channels)
            for corner in ["FL", "FR", "RL", "RR"]:
                for combo in picker._corner_combos[corner]:
                    assert combo.options == sorted(new_channels)
