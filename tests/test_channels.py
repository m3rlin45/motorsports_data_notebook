"""Tests for channels module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from dataclasses import dataclass
from typing import Dict

from motorsports_data_notebook.channels import (
    get_best_lap,
    get_lap_channels,
    get_top_laps,
    interpolate_channels,
)


@dataclass
class MockLogFile:
    """Mock LogFile for testing."""

    channels: Dict[str, pa.Table]
    laps: pa.Table = None
    metadata: Dict[str, str] = None
    file_name: str = "test.xrk"

    def __post_init__(self):
        if self.laps is None:
            self.laps = pa.table({"num": [], "start_time": [], "end_time": []})
        if self.metadata is None:
            self.metadata = {}


class TestGetLapChannels:
    """Tests for get_lap_channels function."""

    def test_filters_to_time_range(self):
        """Channels should be filtered to the specified time range."""
        timecodes = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={
                "speed": pa.table(
                    {
                        "timecodes": pa.array(timecodes, type=pa.int64()),
                        "speed": pa.array(values),
                    }
                )
            }
        )

        result = get_lap_channels(log, ["speed"], start_time=150, end_time=450)
        result_times = result["speed"].column("timecodes").to_numpy()
        np.testing.assert_array_equal(result_times, [200, 300, 400])

    def test_missing_channel_raises(self):
        """Requesting a missing channel should raise KeyError."""
        log = MockLogFile(channels={})

        with pytest.raises(KeyError, match="not found"):
            get_lap_channels(log, ["nonexistent"], start_time=0, end_time=100)


class TestGetBestLap:
    """Tests for get_best_lap function."""

    def test_finds_fastest_lap(self):
        """Should return the fastest lap excluding first and last."""
        # Lap durations: 60s, 60s, 58s (fastest), 60s, 60s
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000, 120000, 178000, 238000],
                "end_time": [60000, 120000, 178000, 238000, 298000],
            }
        )

        best = get_best_lap(laps)
        # Lap 2 (index 2) has duration 58000ms, the fastest among middle laps
        assert best["start_time"] == 120000
        assert best["end_time"] == 178000

    def test_excludes_first_and_last(self):
        """First and last laps should be excluded."""
        laps = pd.DataFrame(
            {
                "start_time": [0, 60000, 120000, 180000],
                "end_time": [50000, 120000, 180000, 230000],  # First lap is fastest
            }
        )

        best = get_best_lap(laps)
        # Should not be lap 0, should be lap 1 or 2
        assert best["start_time"] in [60000, 120000]

    def test_raises_with_too_few_laps(self):
        """Should raise ValueError with fewer than 3 laps."""
        laps = pd.DataFrame({"start_time": [0, 60000], "end_time": [60000, 120000]})

        with pytest.raises(ValueError, match="at least 3 laps"):
            get_best_lap(laps)


class TestInterpolateChannels:
    """Tests for interpolate_channels function."""

    def test_interpolates_to_reference(self):
        """Channels should be interpolated to reference timebase."""
        channels = {
            "ref": pa.table(
                {
                    "timecodes": pa.array([0, 100, 200], type=pa.int64()),
                    "ref": pa.array([1.0, 2.0, 3.0]),
                }
            ),
            "other": pa.table(
                {
                    "timecodes": pa.array([0, 50, 100, 150, 200], type=pa.int64()),
                    "other": pa.array([0.0, 5.0, 10.0, 15.0, 20.0]),
                }
            ),
        }

        result = interpolate_channels(channels, "ref")

        # Reference should be unchanged
        np.testing.assert_array_equal(result["ref"].column("timecodes").to_numpy(), [0, 100, 200])

        # Other should be interpolated to ref's timebase
        np.testing.assert_array_equal(result["other"].column("timecodes").to_numpy(), [0, 100, 200])
        np.testing.assert_array_almost_equal(
            result["other"].column("other").to_numpy(), [0.0, 10.0, 20.0]
        )

    def test_missing_reference_raises(self):
        """Missing reference channel should raise KeyError."""
        channels = {"a": pa.table({"timecodes": pa.array([0]), "a": pa.array([1.0])})}

        with pytest.raises(KeyError, match="not found"):
            interpolate_channels(channels, "nonexistent")


class TestGetTopLaps:
    """Tests for get_top_laps function."""

    def test_filters_by_threshold(self):
        """Should return laps within threshold of best."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([60, 61, 62, 63, 70, 65], unit="s"),
            }
        )

        # 103% of 61s = 62.83s, so laps 1-3 (61, 62, 63s) should pass
        result = get_top_laps(laps, threshold_pct=1.03)

        # Excludes first (index 0) and last (index 5)
        # From middle laps [61, 62, 63, 70], best is 61s
        # 103% of 61 = 62.83, so 61, 62 pass, 63 and 70 fail
        assert len(result) == 2

    def test_excludes_first_and_last(self):
        """First and last laps should be excluded."""
        laps = pd.DataFrame(
            {
                "lap_time": pd.to_timedelta([50, 60, 60, 60, 50], unit="s"),
            }
        )

        result = get_top_laps(laps, threshold_pct=1.5)
        # Even though first and last are fastest, they should be excluded
        assert len(result) == 3
