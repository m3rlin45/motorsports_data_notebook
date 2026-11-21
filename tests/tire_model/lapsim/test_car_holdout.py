"""Gate C3: the car on held-out days and layouts (lap simulation C3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car as cm
from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import car_holdout as ch
from motorsports_data_notebook.tire_model.lapsim import solver as sv

from .test_car_gate import CAR, _model, _oval_track
from .test_holdout import SHARE


def _sim_dataset(monkeypatch, car=CAR):
    """Tables and a lap index of ``car`` simulated on two ovals, three
    "days" each, a lap a day at nearly one pace."""
    tracks = {"a": _oval_track("a"), "b": _oval_track("b", straight=700.0, radius=45.0)}
    monkeypatch.setattr(cg, "RECOVERY_U", (0.998, 1.0, 1.002))
    monkeypatch.setattr(ch, "MIN_FOLD_LAPS", 1)
    tables = cg.simulated_tables(car, tracks)
    speed, speed_laps, rows = {}, [], []
    for layout, track in tracks.items():
        traces = []
        for k, u in enumerate(cg.RECOVERY_U):
            sim = sv.simulate_lap(track, car, u)
            traces.append(sim.v_node_ms)
            key = {
                "session_id": f"sim_{layout}_{k}",
                "car": "sim",
                "date": f"2000-01-{k + 1:02d}",
                "layout_id": layout,
                "basis": "dry",
                "lap_num": 1,
            }
            speed_laps.append({**key, "row": k})
            rows.append({**key, "lap_time_s": sim.lap_time_s})
        speed[layout] = np.vstack(traces)
    tables.speed = speed
    tables.speed_laps = pd.DataFrame(speed_laps)
    return tables, pd.DataFrame(rows), tracks


class TestLayout:
    def test_apexes_and_stretches_cover_the_lap(self):
        track = _oval_track()
        layout = ch.Layout.of(track)
        assert len(layout.apex_nodes) == len(layout.stretches) == 2
        # each stretch runs from one apex to the next; together, once round
        assert sum(len(s) - 1 for s in layout.stretches) == track.n
        assert layout.stretches[1][-1] == layout.stretches[0][0]  # the second wraps the line


class TestLapStats:
    def _lap(self):
        track = _oval_track()
        return ch.Layout.of(track), sv.simulate_lap(track, CAR).v_node_ms

    def test_the_same_lap(self):
        layout, v = self._lap()
        got = ch.lap_stats(layout, v, v)
        assert got["S1"] == 0.0 and got["S2"] == 0.0 and got["S3"] == 0.0 and got["top"] == 0.0
        assert set(got) == {"S1", "S2", "S3", "top"}

    def test_a_faster_straight_and_a_slower_apex(self):
        layout, v = self._lap()
        sim = v.copy()
        sim[v > v.max() - 3.0] += 2.0  # the ends of the straights
        sim[v < v.min() + 1.0] -= 1.5  # the apexes
        got = ch.lap_stats(layout, sim, v)
        assert got["S2"] == pytest.approx(2.0, abs=1e-9)
        assert got["S3"] == pytest.approx(-1.5, abs=1e-9)
        assert got["S1"] > 0 and got["top"] > got["S1"]

    def test_no_straight_no_s2(self):
        layout, v = self._lap()
        flat = np.full_like(v, 30.0)
        assert np.isnan(ch.lap_stats(layout, flat, flat)["S2"])


class TestScoreLaps:
    def test_the_car_itself_and_a_car_with_less_power(self):
        track = _oval_track()
        layout = ch.Layout.of(track)
        sim = sv.simulate_lap(track, CAR, 0.97)
        laps = pd.DataFrame({"session_id": ["s"], "lap_num": [1], "lap_time_s": [sim.lap_time_s]})
        own = ch.score_laps(CAR, layout, laps, sim.v_node_ms[None, :]).iloc[0]
        assert own["u"] == pytest.approx(0.97, abs=1e-5)
        assert own["S1"] < 1e-3 and abs(own["S2"]) < 1e-3 and abs(own["S3"]) < 1e-3
        weak = ch.score_laps(
            replace(CAR, p_specific=85.0), layout, laps, sim.v_node_ms[None, :]
        ).iloc[0]
        # the lap time is matched with more grip: slower down the straight, faster at the apex
        assert weak["u"] > 0.97 and weak["S2"] < -1.0 and weak["S3"] > 0.3

    def test_a_lap_the_car_cannot_do(self):
        track = _oval_track()
        laps = pd.DataFrame({"session_id": ["s"], "lap_num": [1], "lap_time_s": [10.0]})
        got = ch.score_laps(CAR, ch.Layout.of(track), laps, np.full((1, track.n), 30.0)).iloc[0]
        assert np.isnan(got["u"]) and np.isnan(got["S1"])


class TestFolds:
    def test_without_and_car_only(self, monkeypatch):
        tables, rows, _ = _sim_dataset(monkeypatch)
        left = ch.without(tables, "a", "2000-01-02")
        gone = (tables.throttle["layout_id"] == "a") & (tables.throttle["date"] == "2000-01-02")
        assert len(left.throttle) == len(tables.throttle) - int(gone.sum()) > 0
        assert set(ch.without(tables, "a").apex["layout_id"]) == {"b"}
        assert ch.car_only(tables, "nobody").throttle.empty
        assert len(ch.car_only(tables, "sim").brake) == len(tables.brake)

    def test_convene_plants_a_power_and_raises_the_grip(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        best = cg.day_best_laps(rows, "sim")
        raw = cm.fit_car(tables, "sim")[0]
        fitted = ch.convene(raw, tracks, best)
        low = ch.convene(raw, tracks, best, power_scale=0.85)
        assert fitted.model.grip["mu_y"] == pytest.approx(CAR.mu_y, rel=0.03)
        assert low.power_scale == 0.85 and fitted.power_scale == 1.0
        assert low.car().power_at(30.0) == pytest.approx(0.85 * fitted.car().power_at(30.0))
        assert low.model.grip["mu_y"] > 1.03 * fitted.model.grip["mu_y"]
        assert raw.power["p_specific"] == pytest.approx(CAR.p_specific, rel=0.01)  # untouched
        assert ch.convene(raw, tracks, best.iloc[:0]) is None

    def test_the_gate_passes_the_car_and_catches_both_controls(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        other = _model(sv.SolverCar(mu_y=1.0, mu_b=0.8, p_specific=60.0, c_drag=2.0e-4))
        folds, laps = ch.run_folds(tables, rows, tracks, "sim", other)
        assert set(folds["model"]) == set(ch.MODELS)
        assert set(folds["kind"]) == {"day", "layout"}
        assert len(folds[folds["model"] == "fit"]) == 2 + 6  # two layouts, three days at each
        wide, by = ch.verdict(folds)
        assert wide["inside"].all()
        assert wide["powered_low_power"].all() and wide["powered_other_car"].all()
        assert set(by["result"]) == {"pass"}
        # the layout folds carry the variants
        lay = laps[laps["kind"] == "layout"]
        assert {"S1_other", "top_other", "top_0.0", "top_1.0", "top_1.5"} <= set(lay)
        # no drag at all is worse at the top of the speed range than the model's
        assert lay["top_0.0"].median() > lay["top_1.0"].median() + 0.3
        assert "S1_flat" not in lay  # no track has a bank
        assert "h_fl" not in lay  # and no force shares were given

    def test_the_layout_folds_carry_the_flat_road_and_the_heat(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        # the laps were driven on a flat road; one of the tracks is then said to lean
        tracks["a"].bank = 0.06 * np.sign(tracks["a"].kappa_heat)
        folds, laps = ch.run_folds(tables, rows, tracks, "sim", None, shares={"sim": SHARE})
        lay = laps[laps["kind"] == "layout"].set_index("layout_id")
        assert {"S1_flat", "S3_flat", "top_flat", "u_flat"} <= set(lay.columns)
        # held out, the leaning layout wants less of the grip than the flat one gave the car
        assert lay.loc["a", "u"].median() < lay.loc["a", "u_flat"].median() - 0.02
        assert lay.loc["a", "S1"].median() > lay.loc["a", "S1_flat"].median()
        # and a car set on the leaning layout is short of grip on the flat one
        assert lay.loc["b", "u"].median() > lay.loc["b", "u_flat"].median() + 0.02
        day = laps[laps["kind"] == "day"]
        assert "S1_flat" not in day or day["S1_flat"].isna().all()
        # heat: on the road the laps were driven on, the car set elsewhere gives it back,
        # at the lap's own time and with the pace carried; on the wrong road it does not
        cell = laps[(laps["kind"] == "layout") & (laps["layout_id"] == "a")]
        assert ch.worst_tyre(cell, "h", "_flat") < 0.01
        assert ch.worst_tyre(cell, "hc", "_flat") < 0.03
        assert ch.worst_tyre(cell, "hc") > ch.worst_tyre(cell, "hc", "_flat") + 0.03
        assert np.isnan(ch.worst_tyre(cell, "h", "_nothing"))

    def test_without_shares_for_the_car_its_heat_is_not_scored(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        _, laps = ch.run_folds(tables, rows, tracks, "sim", None, shares={"another": SHARE})
        assert "h_fl" not in laps.columns


def _folds(fit, low, other, floor=1.0, kinds=("day", "layout")):
    """A hand-made folds frame: per kind one fold; each model's (S1, S2, S3)."""
    rows = []
    for kind in kinds:
        for name, (s1, s2, s3) in (("fit", fit), ("low_power", low), ("other_car", other)):
            rows.append(
                {
                    "car": "car",
                    "kind": kind,
                    "layout_id": "a",
                    "date": "d" if kind == "day" else "",
                    "laps": 5,
                    "floor": floor,
                    "model": name,
                    "S1": s1,
                    "S2": s2,
                    "S3": s3,
                    "u": 1.0,
                }
            )
    return pd.DataFrame(rows)


class TestVerdict:
    def test_inside_is_all_three(self):
        f = _folds((1.3, 0.5, -0.5), (1.5, 0.0, 0.0), (1.0, 1.2, 0.0))
        assert list(ch.inside(f)[:3]) == [True, False, False]

    def test_pass(self):
        _, by = ch.verdict(_folds((1.2, 0.2, 0.1), (1.2, -1.6, 0.4), (2.5, 0.0, 0.0)))
        assert set(by["result"]) == {"pass"}

    def test_a_fitted_car_outside_on_a_powered_fold_fails(self):
        _, by = ch.verdict(_folds((1.2, 1.4, 0.1), (1.2, -1.6, 0.4), (2.5, 0.0, 0.0)))
        assert set(by["result"]) == {"fail"}

    def test_a_control_that_stays_inside_leaves_the_gate_unscored(self):
        wide, by = ch.verdict(_folds((1.2, 0.2, 0.1), (1.2, -0.6, 0.4), (2.5, 0.0, 0.0)))
        assert not wide["powered_low_power"].any()
        assert set(by["result"]) == {"not scored"}

    def test_outside_on_a_powered_fold_fails_even_when_a_control_has_no_power(self):
        # the low-power car is inside everywhere, the other car is not, and
        # the fitted car is outside too: that is a failure, not a gap
        _, by = ch.verdict(_folds((1.2, 1.4, 0.1), (1.2, -0.6, 0.4), (2.5, 0.0, 0.0)))
        assert set(by["result"]) == {"fail"}

    def test_no_layout_fold_no_score(self):
        _, by = ch.verdict(
            _folds((1.2, 0.2, 0.1), (1.2, -1.6, 0.4), (2.5, 0.0, 0.0), kinds=("day",))
        )
        assert set(by["result"]) == {"not scored"}

    def test_a_fold_without_a_floor_has_no_power(self):
        wide, by = ch.verdict(
            _folds((1.2, 0.2, 0.1), (1.2, -1.6, 0.4), (2.5, 0.0, 0.0), floor=np.nan)
        )
        assert not wide["powered"].any() and set(by["result"]) == {"not scored"}
