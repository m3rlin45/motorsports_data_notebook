"""Tests for wall-clock-aware split-session merging.

Covers the two halves of the fix for falsely-merged AIM sessions:
- ``split_group_by_wallclock`` — consecutive run numbers more than
  MERGE_MAX_GAP_S apart are separate sessions, not restart splits.
- ``MergedLogFile(offsets_ms=...)`` — genuinely merged files get their lap
  times shifted onto the first file's clock so the merged timeline is
  monotonic and stint detection sees the real gaps.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from motorsports_data_notebook._util import MergedLogFile
from motorsports_data_notebook.tire_etl.extract import (
    MERGE_MAX_GAP_S,
    split_group_by_wallclock,
)
from motorsports_data_notebook.tire_etl.stints import assign_stint_ids


def _mk_log(time_str: str | None, *, lap_ms: int = 120_000, n_laps: int = 3) -> MagicMock:
    """Fake LogFile: laps on a zero-based clock + AIM metadata start time."""
    log = MagicMock()
    starts = [i * lap_ms for i in range(n_laps)]
    log.laps = pa.table(
        {
            "num": list(range(1, n_laps + 1)),
            "start_time": starts,
            "end_time": [s + lap_ms for s in starts],
        }
    )
    meta = {"Driver": "CMD", "Venue": "Suzuka"}
    if time_str is not None:
        meta["Log Date"] = "08/09/2026"
        meta["Log Time"] = time_str
    log.metadata = meta
    log.channels = {}
    log.file_name = "fake.xrk"
    return log


PATHS = [Path("a_0213.xrk"), Path("b_0214.xrk")]


class TestSplitGroupByWallclock:
    def test_far_apart_files_split(self):
        # 09:32 run of 3×2min laps ends ~09:38; next file starts 11:30.
        logs = [_mk_log("09:32:00"), _mk_log("11:30:00")]
        groups = split_group_by_wallclock(PATHS, logs, track_canonical=None)
        assert len(groups) == 2
        assert groups[0][0] == [PATHS[0]]
        assert groups[1][0] == [PATHS[1]]
        assert groups[0][2] == [0]
        assert groups[1][2] == [0]

    def test_adjacent_files_merge_with_offsets(self):
        # Second file starts 10 minutes after the first: the 6-minute first
        # file leaves an end-to-start gap of 4 min, within MERGE_MAX_GAP_S.
        assert 4 * 60 < MERGE_MAX_GAP_S
        logs = [_mk_log("09:00:00"), _mk_log("09:10:00")]
        groups = split_group_by_wallclock(PATHS, logs, track_canonical=None)
        assert len(groups) == 1
        sub_paths, _, offsets = groups[0]
        assert sub_paths == PATHS
        assert offsets == [0, 600_000]  # 10 min in ms

    def test_gap_measured_end_to_start_not_start_to_start(self):
        # A 40-minute first file followed immediately by a restart must
        # still merge even though the start-to-start gap exceeds the cap.
        long_first = _mk_log("09:00:00", lap_ms=120_000, n_laps=20)  # ends 09:40
        second = _mk_log("09:43:00")
        assert 43 * 60 > MERGE_MAX_GAP_S
        groups = split_group_by_wallclock(PATHS, [long_first, second], track_canonical=None)
        assert len(groups) == 1

    def test_missing_metadata_keeps_group_with_sequential_offsets(self):
        logs = [_mk_log(None, lap_ms=120_000, n_laps=3), _mk_log(None)]
        groups = split_group_by_wallclock(PATHS, logs, track_canonical=None)
        assert len(groups) == 1
        _, _, offsets = groups[0]
        # Second file stitched right after the first file's last lap end.
        assert offsets == [0, 360_000]

    def test_single_file_passthrough(self):
        groups = split_group_by_wallclock([PATHS[0]], [_mk_log("09:00:00")], track_canonical=None)
        assert len(groups) == 1
        assert groups[0][2] == [0]

    def test_three_files_split_into_two(self):
        paths = [Path("a_01.xrk"), Path("a_02.xrk"), Path("a_03.xrk")]
        logs = [_mk_log("09:00:00"), _mk_log("09:08:00"), _mk_log("13:00:00")]
        groups = split_group_by_wallclock(paths, logs, track_canonical=None)
        assert [len(g[0]) for g in groups] == [2, 1]
        assert groups[0][2] == [0, 480_000]


class TestMergedLogFileOffsets:
    def test_offsets_shift_lap_times(self):
        logs = [_mk_log("09:00:00"), _mk_log("09:10:00")]
        merged = MergedLogFile(logs, offsets_ms=[0, 600_000])
        starts = merged.laps.column("start_time").to_pylist()
        # First file: 0, 120k, 240k. Second file shifted by 600k.
        assert starts == [0, 120_000, 240_000, 600_000, 720_000, 840_000]
        # Monotonic timeline — the pre-fix behavior overlapped here.
        assert starts == sorted(starts)

    def test_no_offsets_keeps_legacy_behavior(self):
        logs = [_mk_log(None), _mk_log(None)]
        merged = MergedLogFile(logs)
        starts = merged.laps.column("start_time").to_pylist()
        assert starts == [0, 120_000, 240_000, 0, 120_000, 240_000]

    def test_offsets_length_mismatch_raises(self):
        logs = [_mk_log(None), _mk_log(None)]
        with pytest.raises(ValueError):
            MergedLogFile(logs, offsets_ms=[0])

    def test_stint_detection_sees_real_gap(self):
        # Two runs ~2h apart merged (as the legacy pipeline did) but with
        # correct offsets now yield two stints instead of one.
        logs = [_mk_log("09:32:00"), _mk_log("11:30:00")]
        offset_ms = int((2 * 3600 - 2 * 60) * 1000)  # 11:30 - 09:32
        merged = MergedLogFile(logs, offsets_ms=[0, offset_ms])
        stints = assign_stint_ids(merged.laps).to_pylist()
        assert stints == [1, 1, 1, 2, 2, 2]
