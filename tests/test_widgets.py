"""Tests for widgets module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.widgets import load_session, LapPicker, SessionPicker


# Path to patch aim_xrk - it's imported inside the function
AIM_XRK_PATCH_PATH = "libxrk.aim_xrk"


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

    def test_preselects_fastest_middle_lap(self, sample_laps_df, mock_ipywidgets):
        """Test that the fastest lap (excluding first/last) is pre-selected."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            # Lap 3 (index 2) has the fastest time (58s) among middle laps
            # The Dropdown should be initialized with value=2
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 2

    def test_excludes_first_last_from_fastest_selection(
        self, sample_laps_df, mock_ipywidgets
    ):
        """Test that first and last laps are excluded from fastest selection."""
        # Modify so first lap is fastest overall
        modified_df = sample_laps_df.copy()
        modified_df.loc[0, "lap_time"] = pd.Timedelta(seconds=50)

        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(modified_df)

            # Even though lap 1 (index 0) is fastest, it should select lap 3 (index 2)
            # which is fastest among middle laps
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 2

    def test_marks_pit_laps_in_labels(self, sample_laps_df, mock_ipywidgets):
        """Test that first lap is marked as out lap and last as in lap."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            options = call_kwargs["options"]

            # First option should have "out lap"
            assert "out lap" in options[0][0]
            # Last option should have "in lap"
            assert "in lap" in options[-1][0]

    def test_marks_fastest_lap_in_labels(self, sample_laps_df, mock_ipywidgets):
        """Test that fastest middle lap is marked in label."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(sample_laps_df)

            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            options = call_kwargs["options"]

            # Lap 3 (index 2) should have "fastest" in label
            assert "fastest" in options[2][0]
            # Other middle laps should not have "fastest"
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
        """Test handling when there are only 2 laps (no middle laps)."""
        with patch.dict("sys.modules", {"ipywidgets": mock_ipywidgets}):
            picker = LapPicker(few_laps_df)

            # Should fall back to first lap
            call_kwargs = mock_ipywidgets.Dropdown.call_args[1]
            assert call_kwargs["value"] == 0

            # No lap should be marked as "fastest" since there are no middle laps
            options = call_kwargs["options"]
            for label, _ in options:
                assert "fastest" not in label

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

    mock_widgets.Dropdown.side_effect = create_dropdown
    mock_widgets.HTML.return_value = MagicMock()
    mock_widgets.VBox.return_value = MagicMock()
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

    def test_get_selected_lap_returns_series(
        self, mock_log_file, mock_ipywidgets_session
    ):
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
