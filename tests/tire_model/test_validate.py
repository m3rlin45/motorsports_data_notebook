"""Unit tests for the held-out validation helpers.

The full end-to-end ``run_holdout_validation`` requires the committed
dataset and is exercised by ``just tire-predict-holdout``. These tests
cover the pure pandas logic in ``_pick_holdout_sessions``.
"""

from __future__ import annotations

import pandas as pd

from motorsports_data_notebook.tire_model.validate import _pick_holdout_sessions


def _make_sessions(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal sessions DataFrame matching the columns the picker reads."""
    defaults = {"status": "ok", "has_tpms": True}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_laps_with_usable_per_session(counts: dict[str, int]) -> pd.DataFrame:
    """Build a laps DataFrame where each session_id has ``counts[sid]`` usable laps."""
    rows = []
    for sid, n in counts.items():
        for lap in range(n):
            rows.append({"session_id": sid, "lap_num": lap, "tire_usable": True})
    return pd.DataFrame(rows)


def test_picker_returns_empty_when_no_bucket_meets_threshold() -> None:
    sessions = _make_sessions(
        [
            {"session_id": "s1", "track_canonical": "tsukuba_2000", "car": "CarA"},
            {"session_id": "s2", "track_canonical": "tsukuba_2000", "car": "CarA"},
        ]
    )
    laps = _make_laps_with_usable_per_session({"s1": 5, "s2": 5})
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    assert out == []


def test_picker_returns_first_n_session_ids_sorted_within_eligible_bucket() -> None:
    # 12 sessions in one (track, car) bucket → eligible at min_bucket_size=10
    sids = [f"sess_{i:02d}" for i in range(12)]
    sessions = _make_sessions(
        [{"session_id": sid, "track_canonical": "tsukuba_2000", "car": "CarA"} for sid in sids]
    )
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sids})
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    # Picks first 2 by sort order
    assert out == sorted(sids)[:2]


def test_picker_skips_sessions_with_too_few_usable_laps() -> None:
    """A session needs at least 3 usable laps to be a valid held-out candidate."""
    sids = [f"sess_{i:02d}" for i in range(15)]
    sessions = _make_sessions(
        [{"session_id": sid, "track_canonical": "tsukuba_2000", "car": "CarA"} for sid in sids]
    )
    # Most sessions have 8 usable laps; first 2 (by sort order) have only 2
    counts = {sid: 8 for sid in sids}
    counts["sess_00"] = 2  # would be picked first if usable, but only 2 laps
    counts["sess_01"] = 2
    laps = _make_laps_with_usable_per_session(counts)
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    # Picker skips sess_00 and sess_01, picks sess_02 and sess_03
    assert out == ["sess_02", "sess_03"]


def test_picker_picks_from_multiple_buckets_independently() -> None:
    # Two eligible buckets: Tsukuba/CarA and Fuji/CarA
    sids_tsukuba = [f"tsk_{i:02d}" for i in range(12)]
    sids_fuji = [f"fji_{i:02d}" for i in range(11)]
    sessions = _make_sessions(
        [
            {"session_id": sid, "track_canonical": "tsukuba_2000", "car": "CarA"}
            for sid in sids_tsukuba
        ]
        + [{"session_id": sid, "track_canonical": "fuji", "car": "CarA"} for sid in sids_fuji]
    )
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sids_tsukuba + sids_fuji})
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    # 2 from each bucket = 4 total
    assert len(out) == 4
    assert sorted(sids_tsukuba)[:2] == [s for s in out if s.startswith("tsk")]
    assert sorted(sids_fuji)[:2] == [s for s in out if s.startswith("fji")]


def test_picker_drops_non_ok_or_no_tpms_sessions() -> None:
    rows: list[dict] = [
        {"session_id": "ok1", "track_canonical": "tsukuba_2000", "car": "CarA"},
        {"session_id": "ok2", "track_canonical": "tsukuba_2000", "car": "CarA"},
        {
            "session_id": "err",
            "track_canonical": "tsukuba_2000",
            "car": "CarA",
            "status": "error",
        },
        {
            "session_id": "noTPMS",
            "track_canonical": "tsukuba_2000",
            "car": "CarA",
            "has_tpms": False,
        },
    ]
    rows.extend(
        {"session_id": f"filler_{i}", "track_canonical": "tsukuba_2000", "car": "CarA"}
        for i in range(10)
    )
    sessions = _make_sessions(rows)
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sessions["session_id"].tolist()})
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    assert "err" not in out
    assert "noTPMS" not in out


def test_picker_skips_sessions_with_missing_track_or_car() -> None:
    sessions = _make_sessions(
        [{"session_id": "s_bad_track", "track_canonical": None, "car": "CarA"}]
        + [{"session_id": "s_bad_car", "track_canonical": "tsukuba_2000", "car": None}]
        + [
            {"session_id": f"ok_{i}", "track_canonical": "tsukuba_2000", "car": "CarA"}
            for i in range(12)
        ]
    )
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sessions["session_id"].tolist()})
    out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    assert "s_bad_track" not in out
    assert "s_bad_car" not in out


def test_picker_fold_index_returns_disjoint_slices() -> None:
    # A bucket of 10 sessions, n_per_bucket=2 → 5 disjoint folds of size 2.
    sessions = pd.DataFrame(
        [
            {"session_id": f"s_{i:02d}", "track_canonical": "tsukuba_2000", "car": "CarA"}
            for i in range(10)
        ]
    ).assign(status="ok", has_tpms=True)
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sessions["session_id"].tolist()})

    seen: list[list[str]] = []
    for fold in range(5):
        out = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10, fold=fold)
        assert len(out) == 2
        seen.append(out)

    # Every session is held out exactly once across all folds.
    flat = [sid for fold_sids in seen for sid in fold_sids]
    assert sorted(flat) == [f"s_{i:02d}" for i in range(10)]

    # Asking for fold 5 (past the end) returns nothing — bucket exhausted.
    assert _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10, fold=5) == []
