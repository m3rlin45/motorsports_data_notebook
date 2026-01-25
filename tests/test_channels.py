"""Tests for channels module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from dataclasses import dataclass
from typing import Dict

from motorsports_data_notebook.channels import (
    GPS_CHANNEL_NAMES,
    fix_gps_timing_gaps,
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


def _create_gps_table(name: str, timecodes: np.ndarray, values: np.ndarray) -> pa.Table:
    """Create a GPS channel table with proper metadata."""
    table = pa.table(
        {
            "timecodes": pa.array(timecodes, type=pa.int64()),
            name: pa.array(values, type=pa.float64()),
        }
    )
    # Add metadata like real GPS channels have
    field = table.schema.field(name).with_metadata(
        {b"units": b"m/s", b"dec_pts": b"1", b"interpolate": b"True"}
    )
    new_schema = pa.schema([table.schema.field("timecodes"), field])
    return table.cast(new_schema)


class TestFixGpsTimingGaps:
    """Tests for fix_gps_timing_gaps function."""

    def test_no_gps_channels_returns_unchanged(self):
        """Log without GPS channels should be returned unchanged."""
        log = MockLogFile(
            channels={
                "BRK": pa.table(
                    {"timecodes": pa.array([0, 20, 40]), "BRK": pa.array([0.0, 1.0, 2.0])}
                )
            }
        )
        result = fix_gps_timing_gaps(log)
        assert result is log
        assert "BRK" in result.channels

    def test_no_gaps_returns_unchanged(self):
        """GPS channels with no large gaps should be unchanged."""
        timecodes = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        np.testing.assert_array_equal(result_times, timecodes)

    def test_single_large_gap_is_corrected(self):
        """A single large gap should be corrected."""
        # 65533ms gap at index 2 (should be 40ms)
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        expected = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        np.testing.assert_array_equal(result_times, expected)

    def test_multiple_gaps_are_corrected(self):
        """Multiple large gaps should all be corrected."""
        # Two gaps: 1000ms at index 1 and 500ms at index 3
        timecodes = np.array([0, 1000, 1040, 1540, 1580], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        # First gap: 1000ms -> 40ms (correction: 960ms)
        # Second gap: 500ms -> 40ms (correction: 460ms)
        expected = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        np.testing.assert_array_equal(result_times, expected)

    def test_all_gps_channels_are_corrected(self):
        """All GPS channels should have their timecodes corrected."""
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        channels = {}
        for name in GPS_CHANNEL_NAMES:
            channels[name] = _create_gps_table(name, timecodes.copy(), values.copy())

        log = MockLogFile(channels=channels)
        result = fix_gps_timing_gaps(log)

        expected = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        for name in GPS_CHANNEL_NAMES:
            result_times = result.channels[name].column("timecodes").to_numpy()
            np.testing.assert_array_equal(result_times, expected, err_msg=f"{name} failed")

    def test_metadata_is_preserved(self):
        """Channel metadata should be preserved after correction."""
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        field = result.channels["GPS Speed"].schema.field("GPS Speed")
        assert field.metadata is not None
        assert field.metadata[b"units"] == b"m/s"
        assert field.metadata[b"interpolate"] == b"True"

    def test_values_are_unchanged(self):
        """Channel values should not be modified."""
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_values = result.channels["GPS Speed"].column("GPS Speed").to_numpy()
        np.testing.assert_array_equal(result_values, values)

    def test_custom_expected_dt(self):
        """Custom expected_dt_ms should be respected."""
        # Gap of 500ms, expected 20ms, threshold 200ms -> should be corrected
        timecodes = np.array([0, 20, 520, 540], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log, expected_dt_ms=20.0)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        expected = np.array([0, 20, 40, 60], dtype=np.int64)
        np.testing.assert_array_equal(result_times, expected)

    def test_gap_below_threshold_not_corrected(self):
        """Gaps below 10x expected_dt should not be corrected."""
        # Gap of 200ms with expected 40ms -> threshold is 400ms -> not corrected
        timecodes = np.array([0, 40, 240, 280], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        np.testing.assert_array_equal(result_times, timecodes)

    def test_single_sample_unchanged(self):
        """Log with single GPS sample should be unchanged."""
        timecodes = np.array([0], dtype=np.int64)
        values = np.array([1.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)}
        )
        result = fix_gps_timing_gaps(log)

        result_times = result.channels["GPS Speed"].column("timecodes").to_numpy()
        np.testing.assert_array_equal(result_times, timecodes)

    def test_partial_gps_channels(self):
        """Should work when only some GPS channels are present."""
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Only GPS Speed and GPS Latitude present
        log = MockLogFile(
            channels={
                "GPS Speed": _create_gps_table("GPS Speed", timecodes.copy(), values),
                "GPS Latitude": _create_gps_table(
                    "GPS Latitude", timecodes.copy(), values
                ),
            }
        )
        result = fix_gps_timing_gaps(log)

        expected = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        for name in ["GPS Speed", "GPS Latitude"]:
            result_times = result.channels[name].column("timecodes").to_numpy()
            np.testing.assert_array_equal(result_times, expected)

    def test_lap_times_corrected_with_gap(self):
        """Lap start_time and end_time should be corrected when gap is present."""
        # Gap of 65533ms at timestamp 80
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Lap starts before gap, ends after gap
        laps = pa.table(
            {
                "num": pa.array([1, 2], type=pa.int64()),
                "start_time": pa.array([0, 65613], type=pa.int64()),
                "end_time": pa.array([65613, 130000], type=pa.int64()),
            }
        )

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)},
            laps=laps,
        )
        result = fix_gps_timing_gaps(log)

        # Gap correction: 65533ms - 40ms = 65493ms
        # Lap 1: start=0 (before gap, unchanged), end=65613 -> 120
        # Lap 2: start=65613 -> 120, end=130000 -> 64507
        result_starts = result.laps.column("start_time").to_numpy()
        result_ends = result.laps.column("end_time").to_numpy()

        np.testing.assert_array_equal(result_starts, [0, 120])
        np.testing.assert_array_equal(result_ends, [120, 64507])

    def test_lap_times_unchanged_without_gap(self):
        """Lap times should be unchanged when no gap is present."""
        timecodes = np.array([0, 40, 80, 120, 160], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        laps = pa.table(
            {
                "num": pa.array([1], type=pa.int64()),
                "start_time": pa.array([0], type=pa.int64()),
                "end_time": pa.array([160], type=pa.int64()),
            }
        )

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)},
            laps=laps,
        )
        result = fix_gps_timing_gaps(log)

        result_starts = result.laps.column("start_time").to_numpy()
        result_ends = result.laps.column("end_time").to_numpy()

        np.testing.assert_array_equal(result_starts, [0])
        np.testing.assert_array_equal(result_ends, [160])

    def test_empty_laps_table_handled(self):
        """Empty laps table should not cause errors."""
        timecodes = np.array([0, 40, 80, 65613, 65653], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)},
            laps=pa.table(
                {
                    "num": pa.array([], type=pa.int64()),
                    "start_time": pa.array([], type=pa.int64()),
                    "end_time": pa.array([], type=pa.int64()),
                }
            ),
        )
        result = fix_gps_timing_gaps(log)

        # Should not raise, laps table should remain empty
        assert len(result.laps) == 0

    def test_lap_times_before_gap_unchanged(self):
        """Lap times entirely before the gap should be unchanged."""
        # Normal samples until 160ms, then a huge gap to 65693ms
        # Gap occurs at index 4 (between 160 and 65693)
        timecodes = np.array([0, 40, 80, 120, 160, 65693], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Lap entirely before the gap (both start=0 and end=100 are <= gap_time=160)
        laps = pa.table(
            {
                "num": pa.array([1], type=pa.int64()),
                "start_time": pa.array([0], type=pa.int64()),
                "end_time": pa.array([100], type=pa.int64()),
            }
        )

        log = MockLogFile(
            channels={"GPS Speed": _create_gps_table("GPS Speed", timecodes, values)},
            laps=laps,
        )
        result = fix_gps_timing_gaps(log)

        result_starts = result.laps.column("start_time").to_numpy()
        result_ends = result.laps.column("end_time").to_numpy()

        # Both times are <= gap_time (160), so unchanged
        np.testing.assert_array_equal(result_starts, [0])
        np.testing.assert_array_equal(result_ends, [100])


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
        np.testing.assert_array_equal(
            result["ref"].column("timecodes").to_numpy(), [0, 100, 200]
        )

        # Other should be interpolated to ref's timebase
        np.testing.assert_array_equal(
            result["other"].column("timecodes").to_numpy(), [0, 100, 200]
        )
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
