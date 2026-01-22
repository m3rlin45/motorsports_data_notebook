"""Tests for widgets module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from motorsports_data_notebook.widgets import load_session


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
