"""Tests for _util module."""

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook._util import (
    clean_ibt_laps,
    clean_laps,
    detect_file_type,
    infer_channel_scale,
)


class TestInferChannelScale:
    """Tests for infer_channel_scale function."""

    def test_zero_to_one_scale(self):
        """Data in 0-1 range should return 1.0."""
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert infer_channel_scale(data) == 1.0

    def test_zero_to_hundred_scale(self):
        """Data in 0-100 range should return 100.0."""
        data = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        assert infer_channel_scale(data) == 100.0

    def test_partial_zero_to_one(self):
        """Partial throttle in 0-1 range (max < 1.0) should return 1.0."""
        data = np.array([0.0, 0.1, 0.3, 0.5])
        assert infer_channel_scale(data) == 1.0

    def test_partial_zero_to_hundred(self):
        """Partial throttle in 0-100 range (e.g. max=60) should return 100.0."""
        data = np.array([0.0, 10.0, 30.0, 60.0])
        assert infer_channel_scale(data) == 100.0

    def test_boundary_at_1_5(self):
        """Max value of exactly 1.5 should return 1.0 (boundary case)."""
        data = np.array([0.0, 1.5])
        assert infer_channel_scale(data) == 1.0

    def test_just_above_boundary(self):
        """Max value just above 1.5 should return 100.0."""
        data = np.array([0.0, 1.6])
        assert infer_channel_scale(data) == 100.0

    def test_negative_values_use_abs(self):
        """Negative values (e.g. lateral G) should use absolute value."""
        data = np.array([-0.5, 0.0, 0.5])
        assert infer_channel_scale(data) == 1.0

    def test_all_zeros(self):
        """All zeros should return 1.0 (no data to distinguish)."""
        data = np.zeros(10)
        assert infer_channel_scale(data) == 1.0


class TestDetectFileType:
    """Tests for detect_file_type function."""

    def test_detect_aim_from_xrk_extension(self):
        """String path ending in .xrk should return 'aim'."""
        assert detect_file_type("path/to/file.xrk") == "aim"

    def test_detect_aim_from_xrz_extension(self):
        """String path ending in .xrz should return 'aim'."""
        assert detect_file_type("path/to/file.xrz") == "aim"

    def test_detect_ibt_from_extension(self):
        """String path ending in .ibt should return 'ibt'."""
        assert detect_file_type("path/to/file.ibt") == "ibt"

    def test_detect_ibt_from_extension_case_insensitive(self):
        """String path ending in .IBT (uppercase) should return 'ibt'."""
        assert detect_file_type("path/to/file.IBT") == "ibt"

    def test_detect_aim_from_xrk_extension_case_insensitive(self):
        """String path ending in .XRK (uppercase) should return 'aim'."""
        assert detect_file_type("path/to/file.XRK") == "aim"

    def test_detect_ibt_from_magic_bytes(self):
        """Bytes starting with IBT magic header should return 'ibt'."""
        ibt_bytes = b"\x02\x00\x00\x00" + b"\x00" * 100
        assert detect_file_type(ibt_bytes) == "ibt"

    def test_detect_aim_from_other_bytes(self):
        """Bytes without IBT magic header should return 'aim' (fallback)."""
        aim_bytes = b"\xff\x00\x00\x00" + b"\x00" * 100
        assert detect_file_type(aim_bytes) == "aim"

    def test_detect_aim_from_empty_bytes(self):
        """Empty bytes should return 'aim' (fallback)."""
        assert detect_file_type(b"") == "aim"

    def test_detect_aim_from_short_bytes(self):
        """Bytes shorter than 4 should return 'aim' (fallback)."""
        assert detect_file_type(b"\x02\x00") == "aim"

    def test_detect_aim_from_unknown_extension(self):
        """String path with unknown extension should return 'aim'."""
        assert detect_file_type("path/to/file.txt") == "aim"

    def test_detect_bytearray_ibt(self):
        """Bytearray with IBT magic header should return 'ibt'."""
        ibt_data = bytearray(b"\x02\x00\x00\x00" + b"\x00" * 100)
        assert detect_file_type(ibt_data) == "ibt"


class TestCleanLaps:
    """Tests for clean_laps function."""

    def _make_laps(self, nums, start_times, end_times):
        """Helper to create a laps PyArrow table."""
        return pa.table(
            {
                "num": pa.array(nums, type=pa.int64()),
                "start_time": pa.array(start_times, type=pa.int64()),
                "end_time": pa.array(end_times, type=pa.int64()),
            }
        )

    def test_removes_lap_zero(self):
        """Lap 0 (out-lap) should be removed."""
        laps = self._make_laps(
            nums=[0, 1, 2, 3],
            start_times=[0, 60000, 120000, 180000],
            end_times=[60000, 120000, 180000, 240000],
        )
        result = clean_laps(laps)
        assert result.column("num").to_pylist() == [1, 2, 3]

    def test_deduplicates_lap_numbers(self):
        """When duplicate lap numbers exist, keep the longest."""
        laps = self._make_laps(
            nums=[1, 1, 2],
            start_times=[0, 30000, 90000],
            end_times=[30000, 90000, 150000],
        )
        result = clean_laps(laps)
        nums = result.column("num").to_pylist()
        assert nums == [1, 2]
        # The kept lap 1 should be the longer one (30000→90000 = 60s, not 0→30000 = 30s)
        assert result.column("start_time").to_pylist() == [30000, 90000]

    def test_empty_table_returns_unchanged(self):
        """Empty table should be returned as-is."""
        laps = self._make_laps([], [], [])
        result = clean_laps(laps)
        assert len(result) == 0

    def test_no_valid_laps_returns_original(self):
        """If filtering would remove all laps, return original table."""
        laps = self._make_laps(
            nums=[0],
            start_times=[0],
            end_times=[60000],
        )
        result = clean_laps(laps)
        # Only lap 0 exists, would be removed — fallback returns original
        assert len(result) == 1


class TestCleanIbtLaps:
    """Tests for clean_ibt_laps function."""

    def _make_laps(self, nums, start_times, end_times):
        """Helper to create a laps PyArrow table."""
        return pa.table(
            {
                "num": pa.array(nums, type=pa.int64()),
                "start_time": pa.array(start_times, type=pa.int64()),
                "end_time": pa.array(end_times, type=pa.int64()),
            }
        )

    def test_removes_pit_markers(self):
        """iRacing pit markers (< 10s) should be removed."""
        laps = self._make_laps(
            nums=[1, 1, 2, 2, 3],
            start_times=[0, 60000, 60016, 120000, 180000],
            end_times=[60000, 60016, 120000, 180000, 240000],
        )
        ldp_table = pa.table(
            {
                "timecodes": pa.array(list(range(0, 250000, 1000)), type=pa.int64()),
                "LapDistPct": pa.array(
                    [(t % 60000) / 60000.0 for t in range(0, 250000, 1000)],
                    type=pa.float64(),
                ),
            }
        )
        result = clean_ibt_laps(laps, {"LapDistPct": ldp_table})
        result = clean_laps(result)
        nums = result.column("num").to_pylist()
        assert 0 not in nums
        assert len(nums) == len(set(nums))

    def test_removes_partial_laps(self):
        """Partial laps (didn't start/end near S/F) should be removed."""
        laps = self._make_laps(
            nums=[1, 2, 3],
            start_times=[0, 60000, 120000],
            end_times=[60000, 120000, 180000],
        )
        tc = list(range(0, 180000, 100))
        ldp = []
        for t in tc:
            if t < 60000:
                ldp.append((t % 60000) / 60000.0)
            elif t < 120000:
                ldp.append(0.5 + ((t - 60000) / 60000.0) * 0.5)
            else:
                ldp.append(((t - 120000) % 60000) / 60000.0)
        ldp_table = pa.table(
            {
                "timecodes": pa.array(tc, type=pa.int64()),
                "LapDistPct": pa.array(ldp, type=pa.float64()),
            }
        )
        result = clean_ibt_laps(laps, {"LapDistPct": ldp_table})
        nums = result.column("num").to_pylist()
        assert 2 not in nums
        assert 1 in nums
        assert 3 in nums

    def test_no_lapdistpct_returns_unchanged(self):
        """Without LapDistPct channel, returns table unchanged."""
        laps = self._make_laps(
            nums=[0, 1, 2, 3],
            start_times=[0, 60000, 120000, 180000],
            end_times=[60000, 120000, 180000, 240000],
        )
        result = clean_ibt_laps(laps, {})
        assert len(result) == len(laps)
