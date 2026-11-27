"""Unit tests for the held-out validation helpers.

The full end-to-end ``run_holdout_validation`` requires the committed
dataset and is exercised by ``just tire-predict-holdout``. These tests
cover the pure pandas logic in ``_pick_holdout_sessions``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.validate import (
    FoldLaps,
    _fold_constants,
    _score_fold_laps,
    _calculator_lap_inputs,
    _pick_holdout_sessions,
)


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


def test_picker_stratifies_by_condition_with_a_smaller_rain_threshold() -> None:
    """With session conditions, buckets are (track, car, condition): the 12
    dry sessions need min_bucket_size, the 4 damp sessions only need the
    rain threshold, and unknown-condition sessions are never held out."""
    dry = [f"dry_{i:02d}" for i in range(12)]
    damp = [f"damp_{i:02d}" for i in range(4)]
    unk = ["unk_00", "unk_01", "unk_02"]
    sids = dry + damp + unk
    sessions = _make_sessions(
        [{"session_id": sid, "track_canonical": "tsukuba_2000", "car": "CarA"} for sid in sids]
    )
    laps = _make_laps_with_usable_per_session({sid: 8 for sid in sids})
    cond = {**{s: "dry" for s in dry}, **{s: "damp" for s in damp}, **{s: "unknown" for s in unk}}
    out = _pick_holdout_sessions(
        sessions, laps, n_per_bucket=2, min_bucket_size=10, session_condition=cond
    )
    assert out == sorted(damp)[:2] + sorted(dry)[:2]
    # fold 1 takes the next slice of each bucket; fold 2 exhausts the damp bucket
    out1 = _pick_holdout_sessions(
        sessions, laps, n_per_bucket=2, min_bucket_size=10, fold=1, session_condition=cond
    )
    assert out1 == sorted(damp)[2:4] + sorted(dry)[2:4]
    out2 = _pick_holdout_sessions(
        sessions, laps, n_per_bucket=2, min_bucket_size=10, fold=2, session_condition=cond
    )
    assert out2 == sorted(dry)[4:6]
    # Without conditions the legacy (track, car) behaviour is unchanged.
    legacy = _pick_holdout_sessions(sessions, laps, n_per_bucket=2, min_bucket_size=10)
    assert legacy == sorted(sids)[:2]


def test_calculator_lap_inputs_use_bucket_g2_pace_scaled_and_lap_clock() -> None:
    """The calculator never sees the lap's own g² or clock: it gets the
    bucket ⟨g²⟩ scaled along the pace curve at the entered target lap time,
    t = N × target for flying lap N, and the bucket's typical out-lap when
    the stint is anchored at pit exit."""
    model = {
        "g2_typ_by_track_car_cond": [
            {
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "g2_typ": 0.8,
                "n_laps_used": 50,
                "g2_vs_lap_time": {"lap_time_s": [60.0, 70.0], "g2": [1.0, 0.5]},
            },
        ],
        "lap_time_typ_by_track_car_cond": [
            {
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "lap_time_typ_s": 65.0,
                "n_laps_used": 50,
            },
        ],
        "outlap_typ_by_track_car_cond": [
            {
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "outlap_moving_s": 80.0,
                "outlap_g2": 0.4,
                "n_laps_used": 20,
            },
        ],
        "g2_lap_time_model": {
            "default_exponent": 3.0,
            "multiplier_clamp": {"min": 0.4, "max": 2.5},
        },
    }
    # At the typical lap time the scale is 1 and the flying clock is N laps long.
    g2, t, scale, out_s, out_g2 = _calculator_lap_inputs(
        model, "t", "c", "dry", 5, 65.0, with_outlap=True
    )
    assert scale == 1.0 and g2 == 0.8 and t == 5 * 65.0
    assert out_s == 80.0 and out_g2 == 0.4
    # A faster session scales g² up along the curve (0.75 at 65 s → 1.0 at 60 s).
    g2_fast, t_fast, scale_fast, _, _ = _calculator_lap_inputs(
        model, "t", "c", "dry", 1, 60.0, with_outlap=True
    )
    assert scale_fast == 1.0 / 0.75 and t_fast == 60.0
    assert g2_fast > g2
    # No pit-exit anchor: no out-lap segment.
    _, _, _, out0, g0 = _calculator_lap_inputs(model, "t", "c", "dry", 1, 65.0, with_outlap=False)
    assert out0 == 0.0 and g0 == 0.0
    # Rain condition without its own bucket falls back to dry.
    g2_wet, _, _, out_w, _ = _calculator_lap_inputs(
        model, "t", "c", "wet", 1, 65.0, with_outlap=True
    )
    assert g2_wet == 0.8 and out_w == 80.0


# ---------- one fit, several heat sources (#208's harness) ----------


def _harness_model() -> dict:
    """A fold model with one car's constants and one cell's entries."""
    return {
        "K_buckets": [
            {"key": {"car": "c", "corner": k, "condition": "dry"}, "value_kelvin_per_g2": 5.0}
            for k in ("fl", "fr", "rl", "rr")
        ],
        "K_by_car_compound_corner_cond": [
            {
                "car": "c",
                "compound": "soft",
                "corner": "fl",
                "condition": "dry",
                "value_kelvin_per_g2": 7.0,
            }
        ],
        "tau_sec_by_car_corner_cond": [
            {"car": "c", "corner": k, "condition": "dry", "value_seconds": 300.0}
            for k in ("fl", "fr", "rl", "rr")
        ],
        "g2_typ_by_track_car_cond": [
            {
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "g2_typ": 4.0,
                "n_laps_used": 50,
                "q_typ_by_corner": {"fl": 4.0, "fr": 4.0, "rl": 4.0, "rr": 4.0},
            }
        ],
        "lap_time_typ_by_track_car_cond": [
            {
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "lap_time_typ_s": 60.0,
                "n_laps_used": 50,
            }
        ],
        "outlap_typ_by_track_car_cond": [],
        "g2_lap_time_model": {
            "default_exponent": 3.0,
            "multiplier_clamp": {"min": 0.4, "max": 2.5},
        },
    }


def _harness_laps(sessions: tuple[str, ...] = ("s1", "s2")) -> FoldLaps:
    """Two flying laps per session on a first-lap anchor, front left only."""
    rows = []
    for sid in sessions:
        for n in (1, 2):
            rows.append(
                {
                    "session_id": sid,
                    "track_canonical": "t",
                    "car": "c",
                    "condition": "dry",
                    "lap_num": n,
                    "stint_id": 0,
                    "lap_within_stint": n,
                    "t_cum_s": 60.0 * n,
                    "on_track_s": 60.0,
                    "t_eff_c": 20.0,
                    "heat_proxy": float("nan"),
                    "q_lap_fl": 4.0,
                    "t_gas_fl_end": 30.0,
                    "tpms_temp_fl_end": 30.0,
                    "t_anchor_fl": 0.0,
                    "t_start_fl": 20.0,
                    "anchor_kind_fl": "first_lap",
                    "p_start_fl": 1.5,
                    "tpms_press_fl_end": 1.6,
                }
            )
    flying = pd.DataFrame(rows)
    return FoldLaps(
        flying=flying,
        full=flying,
        label_by_session={},
        session_lap_time={sid: 60.0 for sid in sessions},
        stint_outlap={},
        kappa_by_car={},
    )


def test_fold_constants_take_the_compound_k_and_fall_back_to_dry() -> None:
    pick = _fold_constants(_harness_model())
    assert pick("c", "fl", "dry", None) == (5.0, 300.0)
    assert pick("c", "fl", "dry", "soft") == (7.0, 300.0)
    # a compound without its own K on that corner keeps the pooled one
    assert pick("c", "fr", "dry", "soft") == (5.0, 300.0)
    # a condition without constants reads the dry ones
    assert pick("c", "fl", "wet", None) == (5.0, 300.0)
    assert pick("other", "fl", "dry", None) == (None, None)


def test_score_fold_laps_scores_one_preparation_under_several_entry_tables() -> None:
    """The laps are prepared once and scored against any number of models:
    another entry in the cell's table moves the prediction, the rows stay
    the same rows."""
    prep, model = _harness_laps(), _harness_model()
    base = _score_fold_laps(prep, model)
    assert list(base["corner"].unique()) == ["fl"] and len(base) == 4
    hotter = _harness_model()
    hotter["g2_typ_by_track_car_cond"][0]["q_typ_by_corner"]["fl"] = 8.0
    moved = _score_fold_laps(prep, hotter)
    keys = ["session_id", "lap_num", "corner", "T_hot_obs_c"]
    pd.testing.assert_frame_equal(base[keys], moved[keys])
    assert (moved["T_hot_pred_c"] > base["T_hot_pred_c"]).all()
    assert (moved["g2_used"] == 8.0).all() and (base["g2_used"] == 4.0).all()


def test_score_fold_laps_can_take_the_laps_of_some_sessions() -> None:
    prep, model = _harness_laps(), _harness_model()
    every = _score_fold_laps(prep, model)
    one = _score_fold_laps(prep, model, session_ids=["s2"])
    want = every[every["session_id"] == "s2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(one, want)


def test_score_fold_laps_feeds_the_curve_from_a_given_function() -> None:
    """A heat source that is not a table of entries is a function with the
    signature of ``_calculator_lap_inputs``."""
    seen: list[tuple] = []

    def source(model, track, car, cond, n, lap_time_s, *, with_outlap, corner=None):  # type: ignore[no-untyped-def]
        seen.append((track, car, cond, n, lap_time_s, with_outlap, corner))
        return 6.0, n * lap_time_s, 1.5, 0.0, 0.0

    got = _score_fold_laps(_harness_laps(("s1",)), _harness_model(), lap_inputs_fn=source)
    assert seen == [
        ("t", "c", "dry", 1, 60.0, False, "fl"),
        ("t", "c", "dry", 2, 60.0, False, "fl"),
    ]
    assert (got["g2_used"] == 6.0).all() and (got["g2_scale"] == 1.5).all()


def test_evaluate_fold_does_not_fit_when_given_the_fold_model(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """With ``model=`` the fold is not fitted again: the model given is the
    one the laps are prepared and scored with."""
    from motorsports_data_notebook.tire_model import validate

    def no_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the fold was fitted again")

    prep, model = _harness_laps(), _harness_model()
    asked: list[dict] = []

    def prepare(root, holdout_ids, fold_model, *, inferred_labels=None):  # type: ignore[no-untyped-def]
        asked.append(fold_model)
        return prep

    monkeypatch.setattr(validate, "build_warmup_table", no_fit)
    monkeypatch.setattr(validate, "_prepare_fold_laps", prepare)
    got = validate._evaluate_fold(tmp_path, ["s1", "s2"], model=model)
    assert asked == [model]
    pd.testing.assert_frame_equal(got, _score_fold_laps(prep, model))


# ---------- the calculator's N on a first-lap anchor (#374) ----------


def _stint(t_anchor: float, kind: str = "first_lap", *, out_lap: bool = False) -> FoldLaps:
    """One stint of four 60 s laps (lap 0 is the stint's first lap: an
    out-lap from the pits when ``out_lap``, else a flying lap), front left
    anchored at ``t_anchor`` on the stint clock; the scored laps are 1–3."""
    rows = []
    for n in range(4):
        rows.append(
            {
                "session_id": "s1",
                "track_canonical": "t",
                "car": "c",
                "condition": "dry",
                "lap_num": 10 + n,
                "stint_id": 0,
                "lap_within_stint": n,
                "is_outlap": out_lap and n == 0,
                "_is_out": out_lap and n == 0,
                "t_cum_s": 60.0 * (n + 1),
                "on_track_s": 60.0,
                "moving_s": 60.0,
                "t_eff_c": 20.0,
                "heat_proxy": float("nan"),
                "q_lap_fl": 4.0,
                "t_gas_fl_end": 30.0,
                "tpms_temp_fl_end": 30.0,
                "t_anchor_fl": t_anchor,
                "t_start_fl": 20.0,
                "anchor_kind_fl": kind,
                "p_start_fl": 1.5,
                "tpms_press_fl_end": 1.6,
            }
        )
    full = pd.DataFrame(rows)
    return FoldLaps(
        flying=full[full["lap_within_stint"] > 0].reset_index(drop=True),
        full=full,
        label_by_session={},
        session_lap_time={"s1": 60.0},
        stint_outlap={},
        kappa_by_car={},
    )


def _fed(prep: FoldLaps) -> tuple[list[tuple[int, int, bool]], pd.DataFrame]:
    """``[(lap_within_stint, N fed, with_outlap)]`` per scored lap, and the rows."""
    seen: list[tuple[int, bool]] = []

    def source(model, track, car, cond, n, lap_time_s, *, with_outlap, corner=None):  # type: ignore[no-untyped-def]
        seen.append((n, with_outlap))
        return 4.0, n * lap_time_s, 1.0, 0.0, 0.0

    got = _score_fold_laps(prep, _harness_model(), lap_inputs_fn=source)
    laps = got["lap_within_stint"].tolist()
    return [(lap, n, out) for lap, (n, out) in zip(laps, seen)], got


def test_a_stint_anchored_at_its_first_laps_start_is_fed_that_lap_too() -> None:
    """No pit-exit reading and the first lap's start reading as the anchor:
    by the end of lap N the tyre has run N + 1 laps from the reading, and
    the calculator, whose clock starts at the reading, is fed N + 1."""
    fed, rows = _fed(_stint(0.0))
    assert fed == [(1, 2, False), (2, 3, False), (3, 4, False)]
    assert rows["n_flying"].tolist() == [2, 3, 4]
    # the calculator's clock is then the stint's: N laps of the typed 60 s
    assert (rows["t_used_s"] == rows["t_cum_s"] - rows["t_anchor_s"]).all()
    # the same when the first lap is an out-lap that did not leave the pits
    fed_out, _ = _fed(_stint(0.0, out_lap=True))
    assert fed_out == fed


def test_a_stint_anchored_at_its_first_laps_end_keeps_its_lap_number() -> None:
    """The first lap's end reading (or the next lap's start reading, whose
    time is a few hundredths of a second off the stint clock) leaves that
    lap behind the anchor: N is the lap number."""
    want = [(1, 1, False), (2, 2, False), (3, 3, False)]
    assert _fed(_stint(60.0))[0] == want
    assert _fed(_stint(59.98))[0] == want


def test_a_lap_ending_on_a_later_anchor_is_not_scored() -> None:
    """An anchor at lap 2's start: lap 1 ends on it (the anchor's own lap)
    and is not scored, in either inputs mode; the laps after count from it."""
    fed, _ = _fed(_stint(119.97))
    assert fed == [(2, 1, False), (3, 2, False)]
    oracle = _score_fold_laps(_stint(119.97), _harness_model(), inputs="oracle")
    assert oracle["lap_within_stint"].tolist() == [2, 3]


def test_a_pit_exit_anchor_is_fed_the_flying_laps_after_the_out_lap() -> None:
    """A pit-exit reading is inside the out-lap, which the calculator
    integrates by itself: N is the flying-lap number, as before."""
    fed, _ = _fed(_stint(12.0, "pit_exit", out_lap=True))
    assert fed == [(1, 1, True), (2, 2, True), (3, 3, True)]


def test_score_fold_laps_reads_the_raw_cars_own_entry() -> None:
    """A held-out lap is answered by its raw car's own entry where the
    model has one (#383) and by the pooled car's otherwise; the constants
    are the pooled car's either way."""
    model = _harness_model()
    model["car_aliases"] = {"r1": "c", "r2": "c"}
    model["g2_typ_by_track_raw_car_cond"] = [
        dict(model["g2_typ_by_track_car_cond"][0], car="r1", q_typ_by_corner={"fl": 8.0})
    ]
    model["lap_time_typ_by_track_raw_car_cond"] = [
        dict(model["lap_time_typ_by_track_car_cond"][0], car="r1")
    ]
    prep = _harness_laps()
    prep.flying["raw_car"] = prep.flying["session_id"].map({"s1": "r1", "s2": "r2"})
    got = _score_fold_laps(prep, model)
    used = got.groupby("session_id")["g2_used"].unique().to_dict()
    assert list(used["s1"]) == [8.0] and list(used["s2"]) == [4.0]
    assert (got["car"] == "c").all()
    # without the raw car's tables every session reads the pooled entry, as before
    bare = {k: v for k, v in model.items() if not k.endswith("_by_track_raw_car_cond")}
    pd.testing.assert_frame_equal(
        _score_fold_laps(prep, bare), _score_fold_laps(prep, _harness_model())
    )


def test_scoring_reads_the_pressure_at_speed_with_the_sessions_kappa_and_the_centrifuge() -> None:
    """The predicted pressure at rest is brought to the lap-end speed with
    the κ of the session where it carries one, else the car's; a computed
    centrifuge is added at the gas temperature of each reading (#417)."""
    import dataclasses

    from motorsports_data_notebook.tire_model.energy_balance import P_ATM_BAR, T_ZERO_C_TO_K

    base = _harness_laps()
    flying = base.flying.assign(speed_end_ms=50.0, speed_anchor_fl_ms=10.0)
    model = _harness_model()

    def score(**kw):  # type: ignore[no-untyped-def]
        prep = dataclasses.replace(base, flying=flying, full=flying, **kw)
        return _score_fold_laps(prep, model).set_index(["session_id", "lap_num"])

    rest = score()  # no κ: the pressure at rest
    car = score(kappa_by_car={"c": 2e-6})
    own = score(kappa_by_car={"c": 2e-6}, kappa_by_session={"s2": 5e-6})
    p = lambda df: df["P_hot_pred_bar"] + P_ATM_BAR  # noqa: E731

    def factor(k_a: float, k_end: float) -> float:
        return (1 + k_a * 10.0**2) / (1 + k_end * 50.0**2)

    assert (p(car) / p(rest)).to_numpy() == pytest.approx(factor(2e-6, 2e-6))
    assert (p(own) / p(rest)).loc["s1"].to_numpy() == pytest.approx(factor(2e-6, 2e-6))
    assert (p(own) / p(rest)).loc["s2"].to_numpy() == pytest.approx(factor(5e-6, 5e-6))
    stack = score(kappa_by_car={"c": 1e-6}, centrifuge={"c": {"front": 1.6e-6, "rear": 9e-6}})
    t_hot_k = stack["T_hot_pred_c"].to_numpy() + T_ZERO_C_TO_K
    want = [
        factor(1e-6 + 1.6e-6 * 320.0 / (20.0 + T_ZERO_C_TO_K), 1e-6 + 1.6e-6 * 320.0 / t)
        for t in t_hot_k
    ]
    assert (p(stack) / p(rest)).to_numpy() == pytest.approx(want)


def test_a_held_out_session_takes_the_kappa_of_its_pinned_compound(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from motorsports_data_notebook.tire_model import validate, warmup_table

    laps = pd.DataFrame({"session_id": ["a", "b", "c"], "car": ["X", "X", "Y"]})
    monkeypatch.setattr(
        warmup_table, "pinned_compounds", lambda root, df: {"a": "SOFT", "c": "SOFT"}
    )
    rows = [
        {"car": "X", "compound": "SOFT", "kappa": 7e-6, "held": True},
        {"car": "X", "compound": "HARD", "kappa": 12e-6, "held": True},
        {"car": "Y", "compound": "SOFT", "kappa": 3e-6, "held": False},  # too little behind it
    ]
    model = {"energy_balance": {"speed_pressure": {"by_car_compound": rows}}}
    # b has no pinned label, c's compound is not held: both keep their car's κ
    assert validate._held_out_kappa(Path("."), laps, model) == {"a": 7e-6}
    assert validate._held_out_kappa(Path("."), laps, {"energy_balance": {}}) == {}
