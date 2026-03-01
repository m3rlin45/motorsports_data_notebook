"""Tests for channels module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from unittest.mock import MagicMock

from motorsports_data_notebook.channels import (
    get_best_lap,
    get_best_lap_channels,
    get_top_laps,
)


class TestGetBestLap:
    """Tests for get_best_lap function."""

    def test_finds_fastest_full_lap(self):
        """Should return the fastest full lap when lap_type column is present."""
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000, 120000, 178000, 238000],
                "end_time": [60000, 120000, 178000, 238000, 298000],
                "lap_type": ["out", "full", "full", "full", "in"],
            }
        )

        best = get_best_lap(laps)
        # Lap at index 2 has duration 58000ms, the fastest among full laps
        assert best["start_time"] == 120000
        assert best["end_time"] == 178000

    def test_excludes_non_full_laps(self):
        """Non-full laps should be excluded when lap_type is present."""
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000, 120000, 180000],
                "end_time": [50000, 120000, 180000, 230000],  # First lap is fastest
                "lap_type": ["out", "full", "full", "in"],
            }
        )

        best = get_best_lap(laps)
        # Should not be out/in laps, should be one of the full laps
        assert best["start_time"] in [60000, 120000]

    def test_raises_no_full_laps(self):
        """Should raise ValueError when no full laps exist."""
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000],
                "end_time": [60000, 120000],
                "lap_type": ["out", "in"],
            }
        )

        with pytest.raises(ValueError, match="No full laps"):
            get_best_lap(laps)

    def test_fallback_excludes_first_and_last(self):
        """Without lap_type column, should fall back to excluding first/last laps."""
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000, 120000, 180000],
                "end_time": [50000, 120000, 180000, 230000],
            }
        )

        best = get_best_lap(laps)
        assert best["start_time"] in [60000, 120000]

    def test_fallback_raises_with_too_few_laps(self):
        """Without lap_type, should raise ValueError with fewer than 3 laps."""
        laps = pd.DataFrame({"start_time": [0, 60000], "end_time": [60000, 120000]})

        with pytest.raises(ValueError, match="at least 3 laps"):
            get_best_lap(laps)


class TestGetBestLapChannels:
    """Tests for get_best_lap_channels function."""

    def test_returns_best_lap_and_channels(self):
        """Should return best lap info and filtered channels."""
        # Create mock channels
        channels = {
            "speed": pa.table(
                {
                    "timecodes": pa.array([100, 200, 300], type=pa.int64()),
                    "speed": pa.array([10.0, 20.0, 30.0]),
                }
            )
        }

        # Create laps with lap 3 being fastest (58s vs 60s)
        laps = pd.DataFrame(
            {
                "num": [1, 2, 3, 4, 5],
                "start_time": [0, 60000, 120000, 178000, 238000],
                "end_time": [60000, 120000, 178000, 238000, 298000],
                "lap_type": ["out", "full", "full", "full", "in"],
            }
        )

        # Create mock log with filter_by_lap and select_channels methods
        mock_log = MagicMock()
        mock_log.filter_by_lap.return_value = mock_log
        mock_log.select_channels.return_value = mock_log
        mock_log.channels = channels

        best_lap, result_channels = get_best_lap_channels(mock_log, laps, ["speed"])

        # Best lap should be lap 3 (index 2, fastest middle lap)
        assert best_lap["start_time"] == 120000
        assert best_lap["num"] == 3

        # Should have called filter_by_lap with lap number
        mock_log.filter_by_lap.assert_called_once_with(3)
        mock_log.select_channels.assert_called_once_with(["speed"])

        # Should return channels dict
        assert "speed" in result_channels


class TestGetTopLaps:
    """Tests for get_top_laps function."""

    def test_filters_by_threshold_with_lap_type(self):
        """Should return full laps within threshold of best."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([60, 61, 62, 63, 70, 65], unit="s"),
                "lap_type": ["out", "full", "full", "full", "full", "in"],
            }
        )

        # Full laps: 61, 62, 63, 70. Best is 61s.
        # 103% of 61 = 62.83, so 61, 62 pass
        result = get_top_laps(laps, threshold_pct=1.03)
        assert len(result) == 2

    def test_excludes_non_full_laps(self):
        """Non-full laps should be excluded when lap_type is present."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([50, 60, 60, 60, 50], unit="s"),
                "lap_type": ["out", "full", "full", "full", "in"],
            }
        )

        result = get_top_laps(laps, threshold_pct=1.5)
        # Only the 3 full laps should be included
        assert len(result) == 3

    def test_fallback_excludes_first_and_last(self):
        """Without lap_type, should fall back to excluding first/last laps."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([50, 60, 60, 60, 50], unit="s"),
            }
        )

        result = get_top_laps(laps, threshold_pct=1.5)
        # Excludes first and last, leaves middle 3
        assert len(result) == 3

    def test_fallback_filters_by_threshold(self):
        """Without lap_type, should filter middle laps by threshold."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([60, 61, 62, 63, 70, 65], unit="s"),
            }
        )

        # Middle laps [61, 62, 63, 70], best is 61s
        # 103% of 61 = 62.83, so 61, 62 pass
        result = get_top_laps(laps, threshold_pct=1.03)
        assert len(result) == 2
