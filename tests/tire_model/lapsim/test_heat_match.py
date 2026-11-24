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
