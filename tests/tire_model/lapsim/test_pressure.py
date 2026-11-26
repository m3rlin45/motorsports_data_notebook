"""Tests for the mechanics of gate F (#208): entries, the recommended
cold pressure, logged and simulated entries."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import heat, pace
from motorsports_data_notebook.tire_model.lapsim import pressure as pr
from motorsports_data_notebook.tire_model.validate import FoldLaps, _score_fold_laps

from .test_car_gate import _model, _oval_track
from .test_holdout import SHARE
from .test_pace import _ranged

TYRES = ("fl", "fr", "rl", "rr")


def _fold_model() -> dict:
    """A fold's model: one car's constants, and entries at two tracks."""

    def heat_row(track, q, curve=True):
        row = {
            "track_canonical": track,
            "car": "c",
            "condition": "dry",
            "g2_typ": float(np.mean(q)),
            "n_laps_used": 40,
            "q_typ_by_corner": dict(zip(TYRES, q)),
        }
        if curve:
            row["g2_vs_lap_time"] = {
                "lap_time_s": [50.0, 60.0, 70.0],
                "g2": [1.5, 1.0, 0.7],
                "g2_by_corner": {c: [1.6, 1.0, 0.6] for c in TYRES},
                "n_laps": 15,
                "method": "uniform_scaled_lap_shares",
            }
        return row

    return {
        "K_buckets": [
            {"key": {"car": "c", "corner": k, "condition": "dry"}, "value_kelvin_per_g2": 5.0}
            for k in TYRES
        ],
        "K_by_car_compound_corner_cond": [],
        "tau_sec_by_car_corner_cond": [
            {"car": "c", "corner": k, "condition": "dry", "value_seconds": 300.0} for k in TYRES
        ],
        "g2_typ_by_track_car_cond": [
            heat_row("a", [4.0, 4.5, 5.0, 5.5]),
            heat_row("b", [6.0, 6.0, 7.0, 7.0], curve=False),
        ],
        "lap_time_typ_by_track_car_cond": [
            {
                "track_canonical": t,
                "car": "c",
                "condition": "dry",
                "lap_time_typ_s": s,
                "n_laps_used": 40,
            }
            for t, s in (("a", 60.0), ("b", 90.0))
        ],
        "outlap_typ_by_track_car_cond": [
            {
                "track_canonical": "a",
                "car": "c",
                "condition": "dry",
                "outlap_moving_s": 80.0,
                "outlap_g2": 2.0,
                "n_laps_used": 12,
                "outlap_q_by_corner": {c: 2.0 for c in TYRES},
            }
        ],
        "g2_lap_time_model": {
            "default_exponent": 3.0,
            "multiplier_clamp": {"min": 0.4, "max": 2.5},
        },
    }


def _stints(track="a", **over) -> pd.DataFrame:
    row = {
        "session_id": "s1",
        "stint_id": 0,
        "track_canonical": track,
        "car": "c",
        "condition": "dry",
        "t_eff_c": 22.0,
        "compound": None,
        "target_hot_bar": 1.8,
        **{f"t_start_{c}": 20.0 for c in TYRES},
        **over,
    }
    return pd.DataFrame([row])


class TestEntries:
    def test_an_entry_read_and_placed_again_is_the_rows_it_was_read_from(self):
        model = _fold_model()
        for track in ("a", "b"):
            entry = pr.entry_of(model, track, "c", "dry")
            placed = pr.place(model, track, "c", "dry", entry)
            for name in (
                "g2_typ_by_track_car_cond",
                "lap_time_typ_by_track_car_cond",
                "outlap_typ_by_track_car_cond",
            ):
                assert placed[name] == model[name]
        assert pr.entry_of(model, "nowhere", "c", "dry") is None

    def test_what_an_entry_holds(self):
        entry = pr.entry_of(_fold_model(), "a", "c", "dry")
        assert entry.lap_time_s == 60.0 and list(entry.q) == [4.0, 4.5, 5.0, 5.5]
        assert list(entry.curve_t) == [50.0, 60.0, 70.0] and entry.curve_q.shape == (3, 4)
        assert entry.out_s == 80.0 and list(entry.out_q) == [2.0] * 4
        bare = pr.entry_of(_fold_model(), "b", "c", "dry")
        assert bare.curve_t is None and bare.out_s is None

    def test_placing_changes_the_cell_and_nothing_else(self):
        model = _fold_model()
        new = pr.Entry(lap_time_s=58.0, q=np.array([1.0, 2.0, 3.0, 4.0]), source="sim")
        placed = pr.place(model, "a", "c", "dry", new)
        got = pr.entry_of(placed, "a", "c", "dry")
        assert got.lap_time_s == 58.0 and list(got.q) == [1.0, 2.0, 3.0, 4.0]
        # the entry has no out-lap: the cell's row is gone, not left from the old entry
        assert got.out_s is None and got.curve_t is None
        assert pr.entry_of(placed, "b", "c", "dry").lap_time_s == 90.0
        assert placed["K_buckets"] is model["K_buckets"]  # the constants are the model's own
        assert pr.entry_of(model, "a", "c", "dry").lap_time_s == 60.0  # and it is not changed
        # a new cell goes in beside the others, in the tables' order
        third = pr.place(model, "aa", "c", "dry", new)
        assert [r["track_canonical"] for r in third["g2_typ_by_track_car_cond"]] == ["a", "aa", "b"]

    def test_without_an_entry_the_fallback_answers(self):
        from motorsports_data_notebook.tire_model.predict import _lookup_lap_time

        bare = pr.without_entry(_fold_model(), "a", "c", "dry")
        assert pr.entry_of(bare, "a", "c", "dry") is None
        assert _lookup_lap_time(bare, "a", "c", "dry")[2] == "global"

    def test_a_placed_entry_answers_the_cell_for_the_raw_cars_too(self):
        """The raw cars' own rows at the cell (#383) are taken out with the
        cell's, so a session scored by its raw car reads the placed entry;
        ``raw=False`` leaves them, and every other cell's are kept."""
        from motorsports_data_notebook.tire_model.validate import _calculator_lap_inputs

        model = _fold_model()
        model["car_aliases"] = {"r": "c"}
        own = dict(model["g2_typ_by_track_car_cond"][0], car="r", g2_typ=9.0)
        own["q_typ_by_corner"] = {c: 9.0 for c in TYRES}
        model["g2_typ_by_track_raw_car_cond"] = [own, dict(own, track_canonical="b")]
        model["lap_time_typ_by_track_raw_car_cond"] = [
            dict(model["lap_time_typ_by_track_car_cond"][0], car="r")
        ]
        model["outlap_typ_by_track_raw_car_cond"] = []
        assert _calculator_lap_inputs(model, "a", "r", "dry", 1, 60.0, with_outlap=False)[0] == 9.0
        new = pr.Entry(lap_time_s=60.0, q=np.array([1.0, 2.0, 3.0, 4.0]), source="sim")
        placed = pr.place(model, "a", "c", "dry", new)
        g2 = _calculator_lap_inputs(
            placed, "a", "r", "dry", 1, 60.0, with_outlap=False, corner="fl"
        )
        assert g2[0] == 1.0
        assert [r["track_canonical"] for r in placed["g2_typ_by_track_raw_car_cond"]] == ["b"]
        assert placed["lap_time_typ_by_track_raw_car_cond"] == []
        kept = pr.place(model, "a", "c", "dry", new, raw=False)
        assert kept["g2_typ_by_track_raw_car_cond"] == model["g2_typ_by_track_raw_car_cond"]
        bare = pr.without_entry(model, "a", "c", "dry")
        assert [r["track_canonical"] for r in bare["g2_typ_by_track_raw_car_cond"]] == ["b"]
        # a model without raw-car tables gains none
        assert "g2_typ_by_track_raw_car_cond" not in pr.place(_fold_model(), "a", "c", "dry", new)

    def test_scaled_and_fed_at_another_lap_time(self):
        entry = pr.entry_of(_fold_model(), "a", "c", "dry")
        up = entry.scaled(1.05)
        assert up.q == pytest.approx(entry.q * 1.05) and up.out_q == pytest.approx([2.1] * 4)
        assert up.lap_time_s == entry.lap_time_s
        moved = entry.at_lap_time(55.0)
        assert moved.lap_time_s == 55.0 and moved.curve_t is None
        assert list(moved.q) == list(entry.q)


class TestRecommend:
    def test_a_cold_pressure_per_pit_exit_tyre(self):
        got = pr.recommend(_fold_model(), _stints(t_start_rr=float("nan")))
        assert list(got["corner"]) == ["fl", "fr", "rl"]  # no pit-exit reading on the rear right
        # the tyre warms, so the cold pressure is under the hot target; more heat, lower still
        assert (got["cold_bar"] < 1.8).all() and got["cold_bar"].is_monotonic_decreasing
        assert (got["lap_time_s"] == 60.0).all()  # no lap time typed: the entry's own
        assert list(got["q"]) == [4.0, 4.5, 5.0]

    def test_the_arithmetic(self):
        from motorsports_data_notebook.tire_model.energy_balance import (
            gay_lussac_cold_pressure_bar,
            warmup_two_stage_c,
        )

        got = pr.recommend(_fold_model(), _stints(), n=5).set_index("corner")
        _, t_hot = warmup_two_stage_c(
            t_outlap_s=80.0,
            g2_outlap=2.0,
            t_flying_s=5 * 60.0,
            g2_flying=4.0,
            t_eff_c=22.0,
            k_kelvin_per_g2=5.0,
            tau_sec=300.0,
            t_start_c=20.0,
        )
        assert got.loc["fl", "t_hot_c"] == t_hot
        assert got.loc["fl", "cold_bar"] == gay_lussac_cold_pressure_bar(
            target_hot_pressure_bar=1.8, t_hot_c=t_hot, t_cold_c=20.0
        )

    def test_a_typed_lap_time_moves_the_heat_along_the_curve_and_the_clock(self):
        own = pr.recommend(_fold_model(), _stints()).set_index("corner")
        fast = pr.recommend(_fold_model(), _stints(), lap_time_s=55.0).set_index("corner")
        assert fast.loc["fl", "q"] == pytest.approx(4.0 * 1.3)  # 1.6 at 50 s, 1.0 at 60 s
        assert fast.loc["fl", "lap_time_s"] == 55.0
        assert fast.loc["fl", "t_hot_c"] != own.loc["fl", "t_hot_c"]
        later = pr.recommend(_fold_model(), _stints(), n=10).set_index("corner")
        assert later.loc["fl", "t_hot_c"] > own.loc["fl", "t_hot_c"]

    def test_the_compounds_k_and_a_stint_without_weather(self):
        model = _fold_model()
        model["K_by_car_compound_corner_cond"] = [
            {
                "car": "c",
                "compound": "soft",
                "corner": c,
                "condition": "dry",
                "value_kelvin_per_g2": 8.0,
            }
            for c in TYRES
        ]
        plain = pr.recommend(model, _stints())
        soft = pr.recommend(model, _stints(compound="soft"))
        assert (soft["t_hot_c"].to_numpy() > plain["t_hot_c"].to_numpy()).all()
        assert pr.recommend(model, _stints(t_eff_c=float("nan"))).empty
        assert pr.recommend(model, _stints(target_hot_bar=float("nan"))).empty

    def test_the_target_is_the_teams_for_the_compound_or_the_cars_first(self):
        targets = {("86", "A052"): 1.8, ("86", "RE-71RS"): 2.0, ("FJ", "DRY"): 1.5}
        assert pr.target_hot(targets, "86", "RE-71RS") == 2.0
        assert pr.target_hot(targets, "86", None) == 1.8
        assert pr.target_hot(targets, "86", "unlisted") == 1.8
        assert math.isnan(pr.target_hot(targets, "other", None))


def _laps(times, *, out=True, q=4.0, stint=0, session="s1", first_lap=0) -> pd.DataFrame:
    """A stint as the holdout prepares it: an out-lap, then flying laps."""
    rows = []
    if out:
        rows.append(
            {
                "lap_num": first_lap,
                "lap_time_s": 140.0,
                "on_track_s": 140.0,
                "moving_s": 100.0,
                "is_outlap": True,
                "outlap_from_pit": True,
                **{f"q_lap_{c}": 2.0 for c in TYRES},
            }
        )
    for k, t in enumerate(times):
        rows.append(
            {
                "lap_num": first_lap + int(out) + k,
                "lap_time_s": t,
                "on_track_s": t,
                "moving_s": t,
                "is_outlap": False,
                "outlap_from_pit": False,
                **{f"q_lap_{c}": q + 0.1 * k for c in TYRES},
            }
        )
    df = pd.DataFrame(rows)
    df["session_id"], df["stint_id"] = session, stint
    df["track_canonical"], df["car"], df["condition"] = "a", "c", "dry"
    df["t_eff_c"], df["heat_proxy"] = 22.0, 1.0
    df["lap_within_stint"] = np.arange(len(df))
    df["_is_out"] = df["is_outlap"]
    for c in TYRES:
        df[f"anchor_kind_{c}"] = "pit_exit" if out else "first_lap"
        df[f"t_start_{c}"] = 21.0
    return df


def _prep(full: pd.DataFrame, labels=None) -> FoldLaps:
    return FoldLaps(
        flying=full[full["lap_within_stint"] > 0].reset_index(drop=True),
        full=full,
        label_by_session=labels or {},
        session_lap_time={},
        stint_outlap={},
        kappa_by_car={},
    )


class TestLoggedLaps:
    def test_the_ordinary_laps_of_a_stint(self):
        full = _laps([62.0, 61.0, 60.0, 60.5, 60.2, 75.0, 60.4])
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        got = pr.mark_laps(_prep(full), raw).set_index("lap_num")
        assert got["stint_best_s"].eq(60.0).all()
        assert list(got.index[got["cool_down"]]) == [6]  # 75 s is over 1.12 × 60 s
        assert list(got.index[got["after_cool_down"]]) == [6, 7]
        # from flying lap 4 on, the cool-down lap apart
        assert list(got.index[got["ordinary"]]) == [4, 5, 7]
        assert not got["after_dropped"].any()

    def test_the_stints_best_is_on_the_full_lap_list(self):
        full = _laps([70.0, 70.0, 70.0, 70.0, 70.0])
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        # a lap cleaning dropped (not in the prepared laps) was the stint's best
        dropped = pd.DataFrame(
            [
                {
                    "session_id": "s1",
                    "stint_id": 0,
                    "lap_num": 99,
                    "lap_time_s": 60.0,
                    "is_outlap": False,
                }
            ]
        )
        got = pr.mark_laps(_prep(full), pd.concat([raw, dropped]))
        assert got["stint_best_s"].eq(60.0).all() and got["cool_down"].sum() == 5

    def test_rows_after_a_lap_cleaning_dropped(self):
        full = _laps([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        full = full[full["lap_num"] != 3].reset_index(drop=True)
        full["lap_within_stint"] = np.arange(len(full))
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        got = pr.mark_laps(_prep(full), raw).set_index("lap_num")
        assert list(got.index[got["after_dropped"]]) == [4, 5, 6]

    def test_a_lap_without_a_heat_on_every_tyre_is_not_ordinary(self):
        full = _laps([60.0] * 6)
        full.loc[full["lap_num"] == 5, "q_lap_rl"] = np.nan
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        got = pr.mark_laps(_prep(full), raw).set_index("lap_num")
        assert list(got.index[got["ordinary"]]) == [4, 6]

    def test_a_logged_entry_is_the_builders_medians(self, tmp_path):
        full = _laps([62.0, 61.0, 60.0, 60.5, 60.2])
        entry = pr.logged_entry(tmp_path, full, SHARE, curve=False)
        assert entry.lap_time_s == 60.5 and entry.q == pytest.approx([4.2] * 4)
        assert entry.out_s == 100.0 and list(entry.out_q) == [2.0] * 4
        assert entry.curve_t is None and entry.n_laps == 5
        # without an out-lap from the pit the entry has none
        no_out = pr.logged_entry(tmp_path, _laps([60.0] * 5, out=False), SHARE, curve=False)
        assert no_out.out_s is None and no_out.lap_time_s == 60.0
        assert pr.logged_entry(tmp_path, full.iloc[:0], SHARE, curve=False) is None
        assert (
            pr.logged_entry(tmp_path, full.iloc[:1], SHARE, curve=False) is None
        )  # an out-lap alone

    def test_a_day_is_its_ordinary_laps_at_their_median_lap_time(self, tmp_path):
        full = _laps([70.0, 65.0, 62.0, 60.5, 60.0, 61.0, 80.0])
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        marked = pr.mark_laps(_prep(full), raw)
        day = pr.day_entry(tmp_path, marked, SHARE)
        # flying laps 4, 5 and 6 (the seventh is a cool-down lap)
        assert day.lap_time_s == 60.5 and day.q == pytest.approx([4.4] * 4)
        assert day.out_s == 100.0 and day.n_laps == 3 and day.source == "day"
        short = pr.mark_laps(_prep(full.iloc[:6]), raw)
        assert pr.day_entry(tmp_path, short, SHARE) is None  # two ordinary laps

    def test_another_days_heat_at_a_given_lap_time(self):
        full = _laps([70.0, 65.0, 62.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0])
        raw = full[["session_id", "stint_id", "lap_num", "lap_time_s", "is_outlap"]]
        marked = pr.mark_laps(_prep(full), raw)
        q, lap_time = pr.near_heat(marked, 60.0)
        # its five ordinary laps nearest 60 s: flying laps 4 to 8
        assert lap_time == 62.0 and q == pytest.approx([4.5] * 4)
        assert pr.near_heat(marked.iloc[:6], 60.0) is None


class TestStints:
    def test_one_row_per_stint_with_its_pit_exit_readings(self):
        full = pd.concat(
            [_laps([60.0] * 3), _laps([60.0] * 3, out=False, stint=1, first_lap=10)],
            ignore_index=True,
        )
        full.loc[full["stint_id"] == 0, "t_start_rr"] = np.nan
        targets = {("c", "soft"): 2.0, ("c", "hard"): 1.7}
        got = pr.stints_of(_prep(full, {"s1": "hard"}), targets)
        assert list(got["stint_id"]) == [0, 1]
        first = got.iloc[0]
        assert first["t_start_fl"] == 21.0 and math.isnan(first["t_start_rr"])
        assert first["compound"] == "hard" and first["target_hot_bar"] == 1.7
        # the second stint has no out-lap: it is not anchored at a pit exit
        assert all(math.isnan(got.iloc[1][f"t_start_{c}"]) for c in TYRES)
        assert pr.stints_of(_prep(full), targets).iloc[0]["target_hot_bar"] == 2.0


def _bins(scale=1.0, seconds=60):
    t = np.arange(0.0, seconds, 0.02)
    return heat.bin_1s(
        t,
        30.0 + 5.0 * np.sin(t / 7.0),
        scale * 1.1 * np.sin(t / 3.0),
        scale * (0.5 * np.cos(t / 2.0) - 0.1),
        0.0,
        float(seconds),
    )


class TestFrameFactor:
    def _frames(self, n_imu=10, n_gps=10):
        keys, sensor, tyres = [], {}, {}
        for k in range(n_imu + n_gps):
            source = "imu" if k < n_imu else "gps"
            lift = 1.03 if source == "imu" else 1.08
            keys.append({"session_id": f"s{k}", "lap_num": 3, "layout_id": "other"})
            sensor[(f"s{k}", 3)] = (_bins(lift), source, False)
            tyres[(f"s{k}", 3)] = _bins(1.0)
        return pd.DataFrame(keys), sensor, tyres

    def test_per_kind_of_source_from_the_laps_bins(self):
        keys, sensor, tyres = self._frames()
        heats = pr.heats_frame(keys, sensor, tyres, SHARE)
        assert len(heats) == 20 and set(heats["layout_id"]) == {"other"}
        factors = pr.frame_factors(heats)
        want = heat.lap_heat(_bins(1.03), SHARE) / heat.lap_heat(_bins(1.0), SHARE)
        assert factors["imu"] == pytest.approx(want)
        assert (factors["gps"] > factors["imu"]).all()

    def test_the_factor_is_of_the_shares(self):
        keys, sensor, tyres = self._frames()
        other = replace(SHARE, p_f=0.60, g_transfer_front=1.0, g_transfer_rear=1.0)
        a = pr.frame_factors(pr.heats_frame(keys, sensor, tyres, SHARE))["imu"]
        b = pr.frame_factors(pr.heats_frame(keys, sensor, tyres, other))["imu"]
        assert not np.allclose(a, b, rtol=1e-6)

    def test_too_few_laps_holes_and_mixed_sources_give_no_factor(self):
        keys, sensor, tyres = self._frames(n_imu=10, n_gps=pr.FACTOR_MIN_LAPS - 1)
        assert set(pr.frame_factors(pr.heats_frame(keys, sensor, tyres, SHARE))) == {"imu"}
        for k in range(4):
            bins, _, _ = sensor[(f"s{k}", 3)]
            sensor[(f"s{k}", 3)] = (bins, "mixed" if k % 2 else "imu", k % 2 == 0)
        assert pr.frame_factors(pr.heats_frame(keys, sensor, tyres, SHARE)) == {}
        assert pr.frame_factors(pd.DataFrame()) == {}

    def test_a_lap_without_bins_is_left_out(self):
        keys, sensor, tyres = self._frames(2, 0)
        del tyres[("s1", 3)]
        del sensor[("s0", 3)]
        heats = pr.heats_frame(keys, sensor, tyres, SHARE)
        assert list(heats["session_id"]) == ["s1"] and heats["t_fl"].isna().all()


@pytest.fixture(scope="module")
def source():
    track = _oval_track()
    track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
    table = pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track)
    return table, pr.SimSource.of(table.laps_export(), SHARE)


class TestSimSource:
    FACTOR = np.array([1.02, 1.03, 1.04, 1.05])

    def test_an_entry_at_the_fed_use_supplies_its_own_lap_time(self, source):
        table, src = source
        entry = src.entry(self.FACTOR, use=0.9)
        assert table.use_at(entry.lap_time_s) == pytest.approx(0.9, rel=1e-9)
        want = table.heat(entry.lap_time_s, SHARE) * self.FACTOR
        assert entry.q == pytest.approx(want, rel=1e-12)
        assert entry.source == "sim" and entry.status == pr.OK
        # the curve is the table's knots, slowest lap last
        assert (np.diff(entry.curve_t) > 0).all() and len(entry.curve_t) == len(table.use)
        np.testing.assert_array_equal(entry.curve_q[0], table.heat_knots(SHARE)[-1] * self.FACTOR)

    def test_the_out_lap_is_the_tracks_rolling_time_and_the_part_from_the_join(self, source):
        table, src = source
        entry = src.entry(self.FACTOR, use=0.9)
        part, hot = table.read_out(entry.lap_time_s, SHARE)
        assert entry.out_s == pytest.approx(20.0 + part)
        # no heat is counted before the join: the part's heat over the whole rolling time
        assert entry.out_q == pytest.approx(hot * self.FACTOR * part / (20.0 + part))

    def test_placed_and_read_as_the_calculators_read_it(self, source):
        """Fed at its own lap time the entry gives its own heat; at
        another, the linear read of its knots is near the table's own."""
        from motorsports_data_notebook.tire_model.validate import _calculator_lap_inputs

        table, src = source
        model = _fold_model()
        entry = src.entry(self.FACTOR, use=0.9)
        placed = pr.place(model, "a", "c", "dry", entry)
        g2, t_fly, scale, out_s, out_g2 = _calculator_lap_inputs(
            placed, "a", "c", "dry", 5, entry.lap_time_s, with_outlap=True, corner="rl"
        )
        assert scale == pytest.approx(1.0) and g2 == pytest.approx(entry.q[2])
        assert t_fly == 5 * entry.lap_time_s and out_s == entry.out_s
        slower = entry.lap_time_s * 1.03
        g2_slow = _calculator_lap_inputs(
            placed, "a", "c", "dry", 5, slower, with_outlap=True, corner="rl"
        )[0]
        want = table.heat(slower, SHARE)[2] * self.FACTOR[2]
        assert g2_slow == pytest.approx(want, rel=0.02)
        direct = src.direct(self.FACTOR, slower)
        assert direct.q[2] == pytest.approx(want, rel=1e-12) and direct.curve_t is None
        assert direct.lap_time_s == slower and direct.source == "sim_direct"

    def test_nine_nodes_on_the_logged_curves_grid(self, source):
        _, src = source
        entry = src.entry(self.FACTOR, use=0.9, nodes=9)
        assert entry.curve_t == pytest.approx(entry.lap_time_s * np.linspace(0.75, 1.5, 9))
        assert entry.curve_q.shape == (9, 4)
        # the curve is read as a ratio to its own value at the typical lap time
        table, _ = source
        slower = entry.lap_time_s * 1.03
        ratio = np.array(
            [
                np.interp(slower, entry.curve_t, entry.curve_q[:, k])
                / np.interp(entry.lap_time_s, entry.curve_t, entry.curve_q[:, k])
                for k in range(4)
            ]
        )
        want = table.heat(slower, SHARE) / table.heat(entry.lap_time_s, SHARE)
        # a coarse grid: the heat bends between two nodes 9 % of a lap time apart
        assert ratio == pytest.approx(want, rel=0.05)
        assert np.abs(np.log(ratio / want)).max() > 0.005

    def test_a_lap_time_given_and_how_it_is_answered(self, source):
        table, src = source
        ok = src.entry(self.FACTOR, use=0.9).lap_time_s
        assert src.status(ok) == pr.OK
        assert src.status(src.entry(self.FACTOR, use=0.5).lap_time_s) == pr.OUTSIDE_SUPPORT
        assert src.status(table.fastest_lap_s * 0.99) == pr.NO_SOLUTION
        assert src.status(float("nan")) == pr.NO_SOLUTION
        # faster than the table: the fastest knot's heat stands in, flagged
        fast = src.entry(self.FACTOR, lap_time_s=table.fastest_lap_s * 0.99)
        assert fast.status == pr.NO_SOLUTION
        assert fast.q == pytest.approx(table.heat_knots(SHARE)[-1] * self.FACTOR)

    def test_no_entry_without_a_factor_or_a_lap_for_the_use(self, source):
        _, src = source
        assert src.entry(np.full(4, np.nan), use=0.9) is None
        assert src.entry(self.FACTOR, use=0.1) is None  # slower than the table's slow end
        assert src.entry(self.FACTOR) is None

    def test_a_table_without_a_pit_exit_writes_no_out_lap(self):
        table = pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), _oval_track())
        entry = pr.SimSource.of(table.laps_export(), SHARE).entry(self.FACTOR, use=0.9)
        assert entry.out_s is None and entry.out_q is None


class TestOtherLayouts:
    def _model(self):
        model = _fold_model()
        # another car has run "new", and shares track "a" with the car
        for track, lap_time in (("new", 100.0), ("a", 66.0)):
            model["lap_time_typ_by_track_car_cond"].append(
                {
                    "track_canonical": track,
                    "car": "other",
                    "condition": "dry",
                    "lap_time_typ_s": lap_time,
                    "n_laps_used": 30,
                }
            )
        return model

    def test_the_cars_heat_at_its_other_layouts_unscaled(self):
        entry = pr.other_layouts_entry(self._model(), "new", "c")
        assert entry.q == pytest.approx([5.0, 5.25, 6.0, 6.25])  # the mean of its two entries
        assert entry.source == "other_layouts" and entry.curve_t is None
        # the other car's lap time there, times the two cars' ratio at the track they share
        assert entry.lap_time_s == pytest.approx(100.0 * 60.0 / 66.0)
        # the car's own out-lap as a ratio of a flying lap, where it has one
        assert entry.out_s == pytest.approx(entry.lap_time_s * 80.0 / 60.0)
        assert entry.out_q == pytest.approx(
            entry.q * np.array([2 / 4.0, 2 / 4.5, 2 / 5.0, 2 / 5.5])
        )

    def test_the_held_out_tracks_own_entry_is_not_among_them(self):
        entry = pr.other_layouts_entry(self._model(), "a", "c")
        assert list(entry.q) == [6.0, 6.0, 7.0, 7.0]

    def test_no_other_car_at_the_track_and_no_lap_time(self):
        entry = pr.other_layouts_entry(_fold_model(), "new", "c")
        assert math.isnan(entry.lap_time_s) and entry.out_s is None
        assert pr.other_layouts_entry(_fold_model(), "new", "unknown car") is None


def test_an_entry_placed_scores_the_holdout_as_its_model_does():
    """The identity source: the holdout on a model with its own entries
    read and placed again is the holdout, bit for bit."""
    full = _laps([60.0, 60.5, 61.0])
    for c in TYRES:
        full[f"t_gas_{c}_end"] = 40.0
        full[f"tpms_temp_{c}_end"] = 40.0
        full[f"t_anchor_{c}"] = 5.0
        full[f"p_start_{c}"] = 1.5
        full[f"tpms_press_{c}_end"] = 1.7
    full["t_cum_s"] = full["moving_s"].cumsum()
    prep = _prep(full)
    prep.session_lap_time["s1"] = 60.0
    model = _fold_model()
    placed = pr.place(model, "a", "c", "dry", pr.entry_of(model, "a", "c", "dry"))
    before, after = _score_fold_laps(prep, model), _score_fold_laps(prep, placed)
    assert len(before) == 12
    pd.testing.assert_frame_equal(before, after, check_exact=True)


class TestPooledEntry:
    """The one entry the calculators read for cars pooled under one name."""

    def _entry(self, lap_time, q, out_s=None):
        nodes = lap_time * np.array([0.9, 1.0, 1.2])
        return pr.Entry(
            lap_time_s=lap_time,
            q=np.full(4, q),
            curve_t=nodes,
            curve_q=np.outer([1.3, 1.0, 0.7], np.full(4, q)),
            out_s=out_s,
            out_q=None if out_s is None else np.full(4, 0.5 * q),
            source="sim",
        )

    def test_the_weighted_geometric_mean_per_tyre(self):
        a, b = self._entry(60.0, 4.0, 80.0), self._entry(60.0, 9.0, 90.0)
        even = pr.pooled_entry([a, b], [1.0, 1.0])
        assert even.q == pytest.approx([6.0] * 4) and even.lap_time_s == pytest.approx(60.0)
        assert even.curve_q[1] == pytest.approx([6.0] * 4)
        assert even.out_s == pytest.approx(math.sqrt(80.0 * 90.0))
        assert even.out_q == pytest.approx([3.0] * 4) and even.source == "sim_pooled"
        # three laps of one car to one of the other
        heavy = pr.pooled_entry([a, b], [3.0, 1.0])
        assert heavy.q == pytest.approx([4.0**0.75 * 9.0**0.25] * 4)

    def test_one_car_alone_is_its_own_entry(self):
        a, b = self._entry(60.0, 4.0), self._entry(60.0, 9.0)
        alone = pr.pooled_entry([a, b], [1.0, 0.0])
        assert alone.q == pytest.approx(a.q) and alone.curve_t == pytest.approx(a.curve_t)

    def test_no_out_lap_unless_every_car_has_one_and_the_worst_status(self):
        a, b = self._entry(60.0, 4.0, 80.0), replace(
            self._entry(60.0, 9.0), status=pr.OUTSIDE_SUPPORT
        )
        got = pr.pooled_entry([a, b], [1.0, 1.0])
        assert got.out_s is None and got.status == pr.OUTSIDE_SUPPORT
        assert pr.pooled_entry([], []) is None and pr.pooled_entry([a], [0.0]) is None


def test_a_simulated_entry_on_nodes_shared_between_cars(source):
    table, src = source
    grid = np.array([0.9, 1.0, 1.1, 1.3])
    typ = float(table.lap_time_s[8])
    entry = src.entry(TestSimSource.FACTOR, lap_time_s=typ, nodes=grid)
    assert entry.curve_t == pytest.approx(typ * grid)
    assert entry.curve_q[1] == pytest.approx(entry.q, rel=1e-12)
