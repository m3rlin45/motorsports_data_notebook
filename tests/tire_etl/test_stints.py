"""Tests for stint ID assignment."""

from __future__ import annotations

import pyarrow as pa

from motorsports_data_notebook.tire_etl.stints import assign_stint_ids


def _laps(starts_end_ms: list[tuple[int, int]]) -> pa.Table:
    starts = [s for s, _ in starts_end_ms]
    ends = [e for _, e in starts_end_ms]
    return pa.table(
        {
            "num": pa.array(list(range(1, len(starts) + 1)), type=pa.int32()),
            "start_time": pa.array(starts, type=pa.int64()),
            "end_time": pa.array(ends, type=pa.int64()),
        }
    )


def test_single_stint_close_gaps() -> None:
    # Three laps with < 30s gaps -> single stint
    t = _laps([(0, 60_000), (60_500, 120_000), (120_500, 180_000)])
    ids = assign_stint_ids(t).to_pylist()
    assert ids == [1, 1, 1]


def test_new_stint_after_600s_gap() -> None:
    t = _laps(
        [
            (0, 60_000),
            (60_500, 120_000),
            # 10-minute pit gap
            (720_500, 780_000),
            (780_500, 840_000),
        ]
    )
    ids = assign_stint_ids(t).to_pylist()
    assert ids == [1, 1, 2, 2]


def test_empty_laps_returns_empty() -> None:
    t = _laps([])
    assert len(assign_stint_ids(t)) == 0
