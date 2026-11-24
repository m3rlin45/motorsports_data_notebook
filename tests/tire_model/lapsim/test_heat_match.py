"""Tests for gate E1 and arm B of gate E2 (#207): heat per tyre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import heat_match as hm
from motorsports_data_notebook.tire_model.lapsim.heat import CORNERS

BASE = np.array([100.0, 110.0, 120.0, 130.0])  # a lap's heat per tyre in the tyres' frame


def _obs(rng, layout, date, n, source="imu", lift=1.03, level=1.0, start=0):
    """Logged laps: the tyres'-frame heat and production's ``lift`` above it."""
    rows = []
    for k in range(n):
        t = BASE * level * np.exp(rng.normal(0.0, 0.004, 4))
        row = {
            "car": "car",
            "layout_id": layout,
            "date": date,
            "session_id": f"s_{layout}_{date}",
            "lap_num": start + k,
            "source": source,
            "hole": False,
        }
        for i, c in enumerate(CORNERS):
            row[f"t_{c}"] = t[i]
            row[f"q_{c}"] = t[i] * lift
        rows.append(row)
    return pd.DataFrame(rows)


def _laps(obs, kind, miss=1.0, low=1.2):
    """Gate C3's per-lap frame: the fitted car simulates the tyres'-frame
    heat times ``miss``, a planted car ``low`` times it."""
    out = obs[["car", "layout_id", "session_id", "lap_num", "date"]].copy()
    out.insert(1, "kind", kind)
    for c in CORNERS:
        out[f"sim_q_{c}"] = obs[f"t_{c}"] * miss
        out[f"sim_q_{c}_low_power"] = obs[f"t_{c}"] * low
    return out


class TestErrors:
    def test_a_day_folds_factor_is_the_layouts_other_days(self):
        rng = np.random.default_rng(1)
        obs = pd.concat(
            [_obs(rng, "a", "d1", 12, lift=1.03), _obs(rng, "a", "d2", 12, lift=1.05)],
            ignore_index=True,
        )
        errs = hm.errors(_laps(obs, "day"), obs)
        fit = errs[errs["model"] == "fit"]
        assert len(fit) == 24 and set(errs["model"]) == {"fit", "low_power"}
        # in the tyres' frame the simulated lap is the logged one
        assert fit[list(hm.G_MEASURES)].abs().to_numpy().max() < 1e-12
        # through the other day's factor, a day reads its own sensor's lift against it
        d1 = fit[fit["date"] == "d1"]["e_fl"].median()
        assert d1 == pytest.approx(np.log(1.05 / 1.03), abs=1e-9)
        low = errs[errs["model"] == "low_power"]
        assert low["g_rr"].median() == pytest.approx(np.log(1.2), abs=1e-9)

    def test_a_layout_folds_factor_is_carried_from_the_other_layouts(self):
        rng = np.random.default_rng(2)
        obs = pd.concat(
            [
                _obs(rng, "a", "d1", 12, lift=1.03),
                _obs(rng, "b", "d2", 12, lift=0.94, level=1.3),  # another layout: more heat
            ],
            ignore_index=True,
        )
        errs = hm.errors(_laps(obs, "layout"), obs)
        fit = errs[(errs["model"] == "fit") & (errs["layout_id"] == "b")]
        # layout b's own factor is 0.94 and the carried one is a's 1.03
        assert fit["e_fl"].median() == pytest.approx(np.log(1.03 / 0.94), abs=1e-9)
        assert fit["g_fl"].abs().max() < 1e-12
        other = errs[(errs["model"] == hm.OTHER_LAYOUTS) & (errs["layout_id"] == "b")]
        # the car's other layouts, unscaled: a's heat, 1.3 times too little
        assert other["g_rl"].median() == pytest.approx(np.log(1 / 1.3), abs=0.01)
        assert other["e_rl"].median() == pytest.approx(np.log(1.03 / (1.3 * 0.94)), abs=0.01)

    def test_sources_keep_their_own_factor_and_unusable_laps_are_left_out(self):
        rng = np.random.default_rng(3)
        obs = pd.concat(
            [
                _obs(rng, "a", "d1", 10, source="imu", lift=1.02),
                _obs(rng, "a", "d2", 10, source="gps", lift=1.09),
                _obs(rng, "a", "d3", 10, source="imu", lift=1.02),
                _obs(rng, "a", "d4", 10, source="gps", lift=1.09),
            ],
            ignore_index=True,
        )
        obs.loc[0, "hole"] = True
        obs.loc[1, "source"] = "mixed"
        errs = hm.errors(_laps(obs, "day"), obs)
        fit = errs[errs["model"] == "fit"]
        assert len(fit) == 38
        assert fit[list(hm.E_MEASURES)].abs().to_numpy().max() < 1e-9  # each source by its own
        thin = hm.errors(_laps(obs[obs["date"].isin(["d1", "d2"])], "day"), obs.iloc[:20])
        assert thin[thin["model"] == "fit"]["e_fl"].isna().all()  # no other day of the source
        assert hm.errors(_laps(obs, "day").drop(columns=["sim_q_fl"]), obs).empty


class TestYardsticks:
    def test_a_pair_is_two_laps_of_one_source(self):
        rng = np.random.default_rng(4)
        cell = pd.concat(
            [
                _obs(rng, "a", "d1", 6, source="imu", lift=1.00),
                _obs(rng, "a", "d2", 6, source="imu", lift=1.02),
                _obs(rng, "a", "d3", 6, source="gps", lift=1.09),
            ],
            ignore_index=True,
        )
        cell["lap_time_s"] = 60.0 + 0.01 * np.arange(len(cell))
        yard = hm.yardsticks(cell, levels=(0.0,))
        d1 = yard[yard["date"] == "d1"]
        # an accelerometer lap is set against the other accelerometer day, not the GPS one
        assert d1["e_fl_other_day"].median() == pytest.approx(np.log(1.02), abs=0.01)
        assert d1["g_fl_other_day"].median() < 0.02 and d1["e_fl_same_day"].median() < 0.02
        # the GPS day has no other day of its source
        assert yard[yard["date"] == "d3"]["e_fl_other_day"].isna().all()
        assert yard[yard["date"] == "d3"]["e_fl_same_day"].notna().all()


class TestScore:
    def test_cells_per_tyre_and_whole_groups(self):
        rng = np.random.default_rng(5)
        obs = pd.concat(
            [_obs(rng, "a", f"d{k}", 10, lift=1.03 + 0.01 * (k % 2)) for k in range(4)],
            ignore_index=True,
        )
        obs["lap_num"] = np.arange(len(obs))
        laps = _laps(obs, "day", low=1.2)
        lap_time = pd.Series(
            60.0 + 0.1 * (np.arange(len(obs)) % 10) + 0.001 * (np.arange(len(obs)) // 10),
            index=pd.MultiIndex.from_frame(obs[["session_id", "lap_num"]]),
        )
        errs = hm.errors(laps, obs)
        yard = hm.cell_yardsticks(laps, obs, lap_time)
        fit = hm.score(errs[errs["model"] == "fit"], yard, hm.G_MEASURES)
        low = hm.score(errs[errs["model"] == "low_power"], yard, hm.G_MEASURES)
        assert len(fit) == 3 * 4 and fit["scored"].all()
        got, wrong = hm.tally(fit), hm.tally(low)
        assert got == {"passed": 12, "scored": 12, "whole": 3, "groups": 3}
        assert wrong["passed"] == 0 and wrong["whole"] == 0  # 20 % too much heat is seen
        table = hm.signed(errs).set_index("model")
        assert table.loc["low_power", "g_fl"] == pytest.approx(np.log(1.2), abs=1e-6)
        assert table.loc["fit", "days"] == 4 and table.loc["fit", "laps"] == 40
        assert hm.tally(pd.DataFrame()) == {"passed": 0, "scored": 0, "whole": 0, "groups": 0}
        assert hm.score(errs.iloc[:0], yard, hm.G_MEASURES).empty


def _arm_laps(obs, lifts):
    """Laps of the four arms: arm ``a`` simulates the tyres'-frame heat
    times ``lifts[a]`` (a number or one per tyre)."""
    parts = []
    for arm, lift in lifts.items():
        part = obs[["car", "layout_id", "session_id", "lap_num", "date"]].copy()
        part["arm"] = arm
        for k, c in enumerate(CORNERS):
            part[f"sim_q_{c}"] = obs[f"t_{c}"] * np.broadcast_to(lift, 4)[k]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


class TestTransfers:
    def test_transfers_between_the_arms_and_whether_they_add(self):
        rng = np.random.default_rng(6)
        obs = pd.concat(
            [_obs(rng, "a", f"d{k}", 8, start=10 * k) for k in range(3)], ignore_index=True
        )
        track = np.array([1.02, 1.0, 1.01, 1.0])  # the other cars' line: more heat on the lefts
        lifts = {"A": 1.0, "B": 1.01, "C": track, "D": 1.01 * track * 1.001}
        per_lap, by_cell = hm.transfers(_arm_laps(obs, lifts), obs)
        assert len(per_lap) == 4 * 24
        got = by_cell.set_index("what")
        assert got.loc["A", "g_fl"] == pytest.approx(0.0, abs=1e-12)
        assert got.loc["B-A", "g_rr"] == pytest.approx(np.log(1.01), abs=1e-12)
        assert got.loc["C-A", "g_fl"] == pytest.approx(np.log(1.02), abs=1e-12)
        assert got.loc["C-A", "g_fr"] == pytest.approx(0.0, abs=1e-12)
        assert got.loc["D-A", "g_fl"] == pytest.approx(np.log(1.01 * 1.02 * 1.001), abs=1e-12)
        assert got.loc["D-B-C+A", "g_fl"] == pytest.approx(np.log(1.001), abs=1e-12)
        assert got.loc["D", "days"] == 3 and got.loc["D", "laps"] == 24
        assert hm.transfers(pd.DataFrame(), obs)[1].empty

    def test_the_report_item_reads_the_cells_and_scores_arm_d(self):
        rng = np.random.default_rng(7)
        here = pd.concat(
            [_obs(rng, "a", f"d{k}", 10, start=10 * k) for k in range(3)], ignore_index=True
        )
        there = pd.concat(
            [_obs(rng, "b", f"e{k}", 10, start=100 + 10 * k, level=1.3) for k in range(2)],
            ignore_index=True,
        )
        obs = pd.concat([here, there], ignore_index=True)
        laps = _laps(obs, "layout")  # gate C3's laps: where the car's other layouts are
        lap_time = pd.Series(
            60.0 + 0.1 * (np.arange(len(obs)) % 10),
            index=pd.MultiIndex.from_frame(obs[["session_id", "lap_num"]]),
        )
        yard = hm.cell_yardsticks(laps, obs, lap_time)
        arms = _arm_laps(here, {"A": 1.0, "B": 1.01, "C": 1.015, "D": 1.01 * 1.015})
        donors = pd.DataFrame(
            [{"car": "car", "layout_id": "a", "donor_laps": 40, "donor_days": 4, "donor_cars": "x"}]
        )
        (item,) = hm.transfer_item(arms, obs, yard, laps, donors)
        assert "arms C and D" in item.name and item.passed is None
        assert "worst tyre within 0.05 in 1 of 1 cells (1 of 1 with a verdict)" in item.measured
        assert "C-A up to 0.015" in item.measured and "within 0.0000" in item.measured
        assert "other_layouts" in item.detail and "| car | layout_id | days |" in item.detail
        thin = donors.assign(donor_days=2)
        (listed,) = hm.transfer_item(arms, obs, yard, laps, thin)
        assert "(0 of 0 with a verdict)" in listed.measured
        assert hm.transfer_item(arms.iloc[:0], obs, yard, laps, donors) == []

    def test_the_lap_time_not_given(self):
        rng = np.random.default_rng(8)
        obs = pd.concat(
            [_obs(rng, "a", f"d{k}", 4, start=10 * k) for k in range(3)], ignore_index=True
        )
        best = obs.groupby("date").head(1)  # one lap a day stands as the day's best
        arms = _arm_laps(best, {"B*": 1.04, "D*": 1.06})
        arms["lap_time_s"] = 60.0
        arms["lap_time_sim_s"] = np.where(arms["arm"] == "B*", 60.6, [59.4, 60.9, 61.2] * 2)
        got = hm.pace_carried(arms, obs).set_index("arm")
        assert got.loc["B*", "days"] == 3
        assert got.loc["B*", "lap_time"] == pytest.approx(np.log(60.6 / 60.0))
        assert got.loc["B*", "g_fl"] == pytest.approx(np.log(1.04), abs=1e-12)
        # a lap time now fast and now slow: the size is not the signed median
        assert got.loc["D*", "lap_time"] == pytest.approx(np.log(60.9 / 60.0))
        assert got.loc["D*", "lap_time_size"] == pytest.approx(np.log(60.9 / 60.0))
        assert hm.pace_carried(_arm_laps(obs, {"A": 1.0}), obs).empty

    def test_the_track_from_other_cars_is_laid_on_the_own_tracks_nodes(self):
        from .test_car_gate import _oval_track
        from .test_wet_gate import _drawn

        own = _drawn(_oval_track())
        own.bank = 0.03 * np.sign(own.kappa_heat)
        other = hm.replace(own, kappa_heat=own.kappa_heat * 1.04, bank=np.zeros(0))
        laid = hm.laid_track(own, other)
        np.testing.assert_allclose(laid.kappa_heat, own.kappa_heat * 1.04, atol=1e-9)
        np.testing.assert_array_equal(laid.bank, own.bank)  # the road's bank stays
        assert hm.fold_track_name("KK-SII", "tsukuba_2000") == "tsukuba_2000__without_FJ__rung5"
        assert hm.fold_track_name("Inferno 86", "fuji_gp") == "fuji_gp__without_Inferno-86__rung5"


class TestTransferLaps:
    def test_four_arms_and_the_lap_time_not_given_on_simulated_laps(self, monkeypatch):
        from dataclasses import replace

        from motorsports_data_notebook.tire_model.lapsim import car as cm
        from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
        from motorsports_data_notebook.tire_model.lapsim import car_holdout as ch
        from motorsports_data_notebook.tire_model.lapsim import heat
        from motorsports_data_notebook.tire_model.lapsim import solver as sv

        from .test_car_gate import CAR
        from .test_car_holdout import _sim_dataset
        from .test_holdout import SHARE

        tables, rows, tracks = _sim_dataset(monkeypatch)
        # the other cars' line at layout a: every corner 4 % tighter (no geometry to lay it by)
        monkeypatch.setattr(hm, "laid_track", lambda own, other: other)
        others = {"a": replace(tracks["a"], kappa_heat=tracks["a"].kappa_heat * 1.04)}
        own = ch.car_only(tables, "sim")
        lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
        best = cg.day_best_laps(rows, "sim")
        full = ch.convene(
            cm.fit_car(own, "sim")[0],
            tracks,
            best,
            laps=cg.car_use_laps(own, lap_time, "sim", tracks),
        )
        got = hm.transfer_laps(tables, rows, tracks, others, "sim", full, SHARE)
        assert set(got["arm"]) == set(hm.ARMS) | set(hm.CARRIED)
        assert set(got["layout_id"]) == {"a"} and got.groupby("arm").size().eq(3).all()
        by = got.set_index(["arm", "session_id"])
        # the laps are the car's own simulated laps: on its own fit and track their heat comes back
        truth = heat.sim_heat(sv.simulate_lap(tracks["a"], CAR, 1.0), SHARE)
        assert by.loc["A", "sim_q_fl"].median() == pytest.approx(truth[0], rel=0.01)
        assert by.loc["B", "sim_q_fl"].median() == pytest.approx(truth[0], rel=0.03)
        # a tighter line at the same lap time is another lap
        assert abs(by.loc["C", "sim_q_fl"].median() / by.loc["A", "sim_q_fl"].median() - 1) > 0.003
        # nothing given: a lap time comes with the heat, near the lap's own on the car's own fit
        own_fit = got[got["arm"] == "A*"]
        assert (own_fit["lap_time_sim_s"] / own_fit["lap_time_s"] - 1).abs().max() < 0.01
        assert got[got["arm"] == "D*"]["lap_time_sim_s"].notna().all()
        # a layout with no track from other cars is not driven
        assert hm.transfer_laps(tables, rows, tracks, {}, "sim", full, SHARE).empty
