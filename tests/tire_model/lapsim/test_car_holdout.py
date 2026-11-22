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
        assert got["brake_g"] == 0.0
        assert set(got) == {"S1", "S2", "S3", "brake_g", "top"}

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


def _stop_lap(n=600, ds=2.0, at=150, level=0.8, top=50.0, low=25.0):
    """A closed lap that cruises at ``top``, brakes from node ``at`` at a
    constant ``level`` (g) down to ``low``, and speeds up again at 0.3 g."""
    g = 9.80665
    v2 = np.full(n, top**2)
    k = at
    while v2[k - 1] - 2 * level * g * ds > low**2:
        v2[k] = v2[k - 1] - 2 * level * g * ds
        k += 1
    v2[k:] = np.minimum(low**2 + 2 * 0.3 * g * ds * np.arange(n - k), top**2)
    return np.sqrt(v2)


class TestBrakingMatch:
    DS = 2.0

    def test_deceleration_is_the_drop_of_v_squared_over_distance(self):
        v = _stop_lap(level=0.8)
        a = ch.deceleration_g(v, self.DS)
        assert a[170] == pytest.approx(0.8, abs=1e-9)  # well inside the stop
        assert a[60] == pytest.approx(0.0, abs=1e-12)  # cruising
        assert a[320] == pytest.approx(-0.3, abs=1e-9)  # gaining speed
        # every lap of an array at once, along the last axis
        both = ch.deceleration_g(np.vstack([v, _stop_lap(level=1.0)]), self.DS)
        assert both.shape == (2, 600) and both[1, 170] == pytest.approx(1.0, abs=1e-9)

    def test_the_same_braking_is_no_distance_and_a_harder_one_is_some(self):
        logged = _stop_lap(level=0.8)
        assert ch.braking_match(logged, logged, self.DS) == 0.0
        # the same speed shed, harder and so later: 0.2 g over where both brake, 1 g where one does
        harder = _stop_lap(level=1.0, at=150 + 30)
        little = _stop_lap(level=0.85, at=150 + 9)
        assert ch.braking_match(harder, logged, self.DS) > 0.2
        assert (
            0.0
            < ch.braking_match(little, logged, self.DS)
            < ch.braking_match(harder, logged, self.DS)
        )

    def test_braking_at_another_place_counts(self):
        logged = _stop_lap()
        moved = [ch.braking_match(np.roll(logged, k), logged, self.DS) for k in (1, 2, 4)]
        assert 0.0 < moved[0] < moved[1] < moved[2]  # 2, 4 and 8 m later

    def test_a_lap_without_braking_has_no_score_and_gaps_are_filled(self):
        flat = np.full(600, 40.0)
        assert np.isnan(ch.braking_match(flat, flat, self.DS))
        logged = _stop_lap()
        holed = logged.copy()
        holed[[10, 11, 400]] = np.nan
        assert ch.braking_match(logged, holed, self.DS) == pytest.approx(0.0, abs=1e-9)

    def test_fast_laps_are_those_above_a_percentile_of_pace(self):
        t = np.arange(60.0, 80.0)  # twenty laps a second apart
        assert np.flatnonzero(ch.fast_laps(t, 90.0)).tolist() == [0, 1]
        assert np.flatnonzero(ch.fast_laps(t, 75.0)).tolist() == [0, 1, 2, 3, 4]
        assert ch.fast_laps(t[::-1], 90.0).tolist() == ch.fast_laps(t, 90.0).tolist()[::-1]
        # rounded up: three laps keep one, eleven keep two, and a tie with the last is in
        assert ch.fast_laps(np.array([61.0, 60.0, 62.0]), 90.0).tolist() == [False, True, False]
        assert ch.fast_laps(np.arange(11.0), 90.0).sum() == 2
        assert ch.fast_laps(np.array([60.0, 60.0, 61.0]), 95.0).tolist() == [True, True, False]
        assert ch.fast_laps(np.zeros(0), 90.0).size == 0

    def test_the_yardstick_is_other_logged_laps_of_the_same_level(self):
        track = _oval_track()
        layout = ch.Layout.of(track)
        a = sv.simulate_lap(track, CAR).v_node_ms
        laps = pd.DataFrame(
            {
                "date": ["d1", "d1", "d2", "d3", "d3", "d4"],
                "lap_time_s": [60.0, 60.6, 60.3, 62.0, 62.1, 65.0],
            }
        )
        v = np.vstack([a, a, np.roll(a, 2), np.roll(a, 3), np.roll(a, 5), np.roll(a, 9)])
        got = ch.pace_yardsticks(layout, laps, v, levels=(50.0, 80.0))
        # the fastest three of six at the 50th percentile, the fastest two at the 80th
        assert got[got["level"] == 50.0]["lap"].tolist() == [0, 1, 2]
        assert got[got["level"] == 80.0]["lap"].tolist() == [0, 2]
        half = got[got["level"] == 50.0].set_index("lap")
        # a lap is set against the other laps of its set whatever their lap times
        assert half.loc[0, "brake_g_same_day"] == 0.0 and half.loc[0, "S1_same_day"] == 0.0
        moved = ch.lap_stats(layout, np.roll(a, 2), a)
        assert half.loc[0, "brake_g_other_day"] == pytest.approx(moved["brake_g"])
        assert half.loc[0, "S1_other_day"] == pytest.approx(moved["S1"])
        assert half.loc[0, "S3_other_day"] == pytest.approx(abs(moved["S3"]))  # a size
        assert np.isnan(half.loc[2, "brake_g_same_day"])  # the only lap of its day in the set
        # at the higher level the two laps left are of different days
        top = got[got["level"] == 80.0].set_index("lap")
        assert np.isnan(top.loc[0, "S1_same_day"]) and top.loc[0, "S1_other_day"] > 0.0

    def test_by_pace_is_the_median_over_days_of_the_days_median(self):
        keys = {"car": "c", "layout_id": "a", "lap_num": 1}
        laps = pd.DataFrame(
            [
                {**keys, "session_id": s, "date": d, "S1": s1, "S2": s2, "S3": 0.2, "brake_g": 0.3}
                for s, d, s1, s2 in (
                    ("s1", "d1", 1.0, -0.4),
                    ("s2", "d1", 2.0, 0.6),
                    ("s3", "d2", 5.0, 0.1),
                )
            ]
        )
        yard = laps[["car", "layout_id", "session_id", "lap_num", "date"]].assign(level=90.0)
        for k in ch.YARDSTICK_MEASURES:
            yard[f"{k}_other_day"], yard[f"{k}_same_day"] = 1.5, 0.5
        got = ch.by_pace(laps, yard).iloc[0]
        assert (got["laps"], got["days"]) == (3, 2)
        assert got["S1"] == pytest.approx(np.median([1.5, 5.0]))  # d1's median, then d2's
        assert got["S2"] == pytest.approx(np.median([0.5, 0.1]))  # sizes: |−0.4| and |0.6|
        assert got["S1_other_day"] == 1.5 and got["brake_g_same_day"] == 0.5
        assert ch.by_pace(laps.iloc[:0], yard).empty

    def test_attainment_counts_the_cells_no_further_than_a_second_logged_lap(self):
        pace = pd.DataFrame(
            [
                # nearer than two logged laps on speed and braking, further at the apex,
                # and nothing to set the end of the straights against
                {"car": "c", "layout_id": "a", "level": 90.0, "laps": 8, "days": 3,
                 "S1": 0.9, "S1_other_day": 1.0, "S1_same_day": 0.7,
                 "S2": 0.3, "S2_other_day": np.nan, "S2_same_day": 0.2,
                 "S3": 0.4, "S3_other_day": 0.3, "S3_same_day": 0.2,
                 "brake_g": 0.2, "brake_g_other_day": 0.2, "brake_g_same_day": 0.15},
            ]
        )  # fmt: skip
        got = ch.attainment(pace).set_index("measure")
        assert got["scored"].to_dict() == {"S1": True, "S2": False, "S3": True, "brake_g": True}
        assert got["passed"].to_dict() == {"S1": True, "S2": False, "S3": False, "brake_g": True}
        assert got.loc["S3", "simulated"] == 0.4 and got.loc["S3", "other_day"] == 0.3
        assert (got["laps"] == 8).all() and (got["days"] == 3).all()


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


class TestCorners:
    def test_the_corners_are_in_the_order_of_the_apex_windows(self):
        track = _oval_track(radius=60.0)
        layout = ch.Layout.of(track)
        frame = ch.corner_frame(layout)
        assert list(frame["corner"]) == [1, 2]
        assert frame["s_apex_m"].is_monotonic_increasing
        assert frame["radius_m"].to_numpy() == pytest.approx([60.0, 60.0], rel=1e-6)
        # each window holds its corner's apex
        ds = track.ds_m
        for nodes, s_apex in zip(layout.apex_nodes, frame["s_apex_m"]):
            assert int(round(s_apex / ds)) % track.n in set(nodes.tolist())

    def test_the_apex_errors_average_to_s3(self):
        track = _oval_track()
        layout = ch.Layout.of(track)
        v = sv.simulate_lap(track, CAR).v_node_ms
        sim = v.copy()
        sim[layout.apex_nodes[0]] -= 1.0  # slow at the first corner only
        err, low = ch.apex_errors(layout, sim, v)
        assert err == pytest.approx([-1.0, 0.0], abs=1e-9)
        assert low == pytest.approx([v[w].min() for w in layout.apex_nodes])
        assert float(np.mean(err)) == pytest.approx(ch.lap_stats(layout, sim, v)["S3"])

    def test_score_laps_carries_the_corners_on_request(self):
        track = _oval_track()
        layout = ch.Layout.of(track)
        sim = sv.simulate_lap(track, CAR, 0.97)
        laps = pd.DataFrame({"session_id": ["s"], "lap_num": [1], "lap_time_s": [sim.lap_time_s]})
        plain = ch.score_laps(CAR, layout, laps, sim.v_node_ms[None, :])
        assert not [c for c in plain.columns if c.startswith("apex_")]
        got = ch.score_laps(CAR, layout, laps, sim.v_node_ms[None, :], corners=True).iloc[0]
        assert abs(got["apex_1"]) < 1e-3 and abs(got["apex_2"]) < 1e-3
        assert got["apex_v_1"] == pytest.approx(sim.v_node_ms[layout.apex_nodes[0]].min())

    def test_the_table_is_the_median_over_days_of_the_days_median(self):
        layout = ch.Layout.of(_oval_track())
        laps = pd.DataFrame(
            {
                "car": "A",
                "kind": ["day"] * 5 + ["layout"],
                "layout_id": "oval",
                "date": ["d1", "d1", "d1", "d2", "d3", "d1"],
                "apex_1": [0.0, 1.0, 5.0, 3.0, -1.0, 99.0],
                "apex_v_1": [20.0, 20.0, 20.0, 22.0, 24.0, 0.0],
                "apex_2": [0.5] * 5 + [99.0],
                "apex_v_2": [30.0] * 6,
            }
        )
        table = ch.corner_table(laps, {"oval": layout})
        first = table[table["corner"] == 1].iloc[0]
        # days read 1.0, 3.0 and -1.0: the layout fold's lap is not in it
        assert first["error_ms"] == pytest.approx(1.0) and first["days"] == 3
        assert first["laps"] == 5 and first["v_apex_ms"] == pytest.approx(22.0)
        assert table[table["corner"] == 2].iloc[0]["error_ms"] == pytest.approx(0.5)
        assert ch.corner_table(laps.iloc[:0], {"oval": layout}).empty

    def test_cars_agree_where_their_errors_repeat(self):
        rows = []
        for car, sign in (("A", 1.0), ("B", 1.0), ("C", -1.0)):
            for k, e in enumerate([0.5, -0.2, 1.0, -0.7, 0.1]):
                rows.append({"car": car, "layout_id": "x", "corner": k + 1, "error_ms": sign * e})
        rows.append({"car": "A", "layout_id": "y", "corner": 1, "error_ms": 0.3})
        rows.append({"car": "B", "layout_id": "y", "corner": 1, "error_ms": 0.3})
        got = ch.corner_agreement(pd.DataFrame(rows)).set_index(["car_a", "car_b"])
        assert got.loc[("A", "B"), "r"] == pytest.approx(1.0)
        assert got.loc[("A", "C"), "r"] == pytest.approx(-1.0)
        assert set(got["layout_id"]) == {"x"}  # one shared corner is no agreement


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

    def test_convene_reads_the_drivers_braking_or_keeps_one(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        best = cg.day_best_laps(rows, "sim")
        raw = cm.fit_car(tables, "sim")[0]
        lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
        laps = cg.car_use_laps(tables, lap_time, "sim", tracks)
        assert set(laps) == {"a", "b"} and len(laps["a"].frame) == 3
        # three laps a layout are too few nodes for a reading: braked at the limit
        plain = ch.convene(raw, tracks, best)
        assert ch.convene(raw, tracks, best, laps=laps).model.driver == {}
        # a driver's use that was read elsewhere is kept, and the softer braking wants more grip
        use = {"brake_use": 0.8, "brake_use_ellipse": 0.4}
        kept = ch.convene(raw, tracks, best, use=use)
        assert kept.model.driver == use and kept.car().brake_use == 0.8
        assert kept.model.grip["mu_y"] > plain.model.grip["mu_y"] * 1.005
        # a fold's reading is from the laps that are left
        left = ch.laps_left(laps, "a", "2000-01-02")
        assert len(left["a"].frame) == 2 and len(left["b"].frame) == 3
        assert set(ch.laps_left(laps, "a")) == {"b"}

    def test_the_gate_passes_the_car_and_catches_both_controls(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        other = _model(sv.SolverCar(mu_y=1.0, mu_b=0.8, p_specific=60.0, c_drag=2.0e-4))
        folds, laps, yard = ch.run_folds(tables, rows, tracks, "sim", other)
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
        # every held-out lap has its braking beside what two logged laps differ by
        held = laps[laps["kind"] == "day"]
        # and, braked at the limit, the same again: no driver's use could be read of three laps
        assert folds["brake_use"].isna().all()
        assert np.array_equal(held["S1_limit"].to_numpy(), held["S1"].to_numpy())
        assert np.array_equal(held["brake_g_limit"].to_numpy(), held["brake_g"].to_numpy())
        assert held["brake_g"].notna().all() and held["brake_g"].median() < 0.05
        # and the yardsticks: at each level the fastest of a layout's three laps
        assert set(yard["level"]) == set(ch.PACE_LEVELS)
        assert yard.groupby(["layout_id", "level"]).size().tolist() == [1] * 6
        assert {"car", "session_id", "lap_num", "date", "S1_other_day", "brake_g_same_day"} <= set(
            yard.columns
        )
        pace = ch.by_pace(held, yard)
        assert len(pace) == 6 and pace["brake_g"].notna().all()  # two layouts, three levels
        assert pace["brake_g_other_day"].isna().all()  # one lap a level: nothing to set it against
        assert not ch.attainment(pace)["scored"].any()
        assert "h_fl" not in lay  # and no force shares were given

    def test_the_layout_folds_carry_the_flat_road_and_the_heat(self, monkeypatch):
        tables, rows, tracks = _sim_dataset(monkeypatch)
        # the laps were driven on a flat road; one of the tracks is then said to lean
        tracks["a"].bank = 0.06 * np.sign(tracks["a"].kappa_heat)
        folds, laps, yard = ch.run_folds(tables, rows, tracks, "sim", None, shares={"sim": SHARE})
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
        _, laps, _ = ch.run_folds(tables, rows, tracks, "sim", None, shares={"another": SHARE})
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
                    "brake_g": 0.2,
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
