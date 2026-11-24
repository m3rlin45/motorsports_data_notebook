"""Tests for the pace table of gate D1 (#206)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import heat
from motorsports_data_notebook.tire_model.lapsim import pace
from motorsports_data_notebook.tire_model.lapsim import solver as sv

from .test_car_gate import CAR, _model, _oval_track, _two_engine_model
from .test_holdout import SHARE


def _ranged(model, **layouts):
    model.meta["use_range"] = {
        k: {
            "min": lo,
            "p05": lo,
            "median": 0.5 * (lo + hi),
            "p95": hi,
            "max": hi,
            "laps": 40,
            "days": 4,
        }
        for k, (lo, hi) in layouts.items()
    }
    return model


@pytest.fixture(scope="module")
def table():
    return pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), _oval_track())


class TestTable:
    def test_knots_from_the_slow_end_to_a_use_of_one(self, table):
        assert len(table.use) == pace.KNOTS and table.use[-1] == 1.0
        assert table.use[0] == pytest.approx(pace.USE_SLOW)
        assert np.allclose(np.diff(table.q), table.q[1] - table.q[0])  # even in q
        assert (np.diff(table.lap_time_s) < 0).all()
        track = _oval_track()
        fastest = sv.simulate_lap(track, CAR, 1.0)
        assert table.fastest_lap_s == pytest.approx(fastest.lap_time_s, rel=1e-12)
        np.testing.assert_array_equal(table.v_node_ms[-1], fastest.v_node_ms)

    def test_a_knots_lap_is_the_lap_the_solver_drives(self, table):
        track = _oval_track()
        for k in (0, 5, pace.KNOTS - 1):
            want = sv.simulate_lap(track, CAR, float(table.use[k]))
            got = table.lap(k)
            for name in ("dt_s", "v_ms", "lat_g", "long_sf_g", "long_tyre_g"):
                np.testing.assert_array_equal(getattr(got, name), getattr(want, name))

    def test_heat_and_use_between_the_knots_against_a_direct_solve(self, table):
        times = np.linspace(table.lap_time_s[-1], table.lap_time_s[0], 22)[1:-1]
        use, laps = pace.direct(table, times)
        other = replace(SHARE, p_f=0.55, g_transfer_front=1.5)
        for share in (SHARE, other):
            want = np.vstack([heat.sim_heat(lap, share) for lap in laps])
            got = table.heat(times, share)
            assert np.abs(np.log(got / want)).max() < 0.01
        read = np.array([table.use_at(float(t)) for t in times])
        assert np.abs(np.log(read / use)).max() < 0.01
        # at a knot the table is the lap itself
        np.testing.assert_allclose(
            table.heat(float(table.lap_time_s[4]), SHARE), heat.sim_heat(table.lap(4), SHARE)
        )

    def test_the_shares_are_the_callers(self, table):
        t = float(np.median(table.lap_time_s))
        front = table.heat(t, replace(SHARE, p_f=0.60))
        rear = table.heat(t, replace(SHARE, p_f=0.40))
        assert front[:2].sum() > rear[:2].sum() and front[2:].sum() < rear[2:].sum()

    def test_another_chain(self, table):
        def halved(sim, share):
            return 0.5 * heat.sim_heat(sim, share)

        t = float(np.median(table.lap_time_s))
        np.testing.assert_allclose(table.heat(t, SHARE, chain=halved), 0.5 * table.heat(t, SHARE))


class TestStatus:
    def test_inside_and_outside_the_logged_range(self, table):
        def at(use):
            return float(sv.lap_time(_oval_track(), CAR, use))

        assert table.status(at(0.90)) == pace.OK
        slow = table.read(at(0.70), SHARE)
        assert slow.status == pace.OUTSIDE_SUPPORT  # answered and flagged
        assert np.isfinite(slow.heat).all() and slow.use == pytest.approx(0.70, rel=0.01)
        assert slow.q == pytest.approx(-math.log(slow.use))
        assert table.status(at(0.99)) == pace.OUTSIDE_SUPPORT  # faster than any logged lap

    def test_faster_than_a_use_of_one_has_no_solution(self, table):
        got = table.read(table.fastest_lap_s - 0.5, SHARE)
        assert got.status == pace.NO_SOLUTION
        assert np.isnan(got.use) and np.isnan(got.heat).all()
        assert got.fastest_lap_s == table.fastest_lap_s
        # a car with no driver entry brakes at its limit: the max driver's lap is the fast end
        assert got.max_driver_lap_s == table.fastest_lap_s
        assert np.isnan(table.heat(table.fastest_lap_s - 0.5, SHARE)).all()

    def test_the_max_driver_is_faster_than_the_knobs_lap_at_a_use_of_one(self):
        model = _ranged(_model(), oval=(0.80, 0.97))
        model.driver = {"brake_use": 0.85, "brake_use_ellipse": 0.4, "use_ref": 0.9}
        got = pace.build_table(model, _oval_track())
        assert got.max_driver_lap_s < got.fastest_lap_s
        limit = sv.SolverCar.from_model(model, brake_use=False)
        assert got.max_driver_lap_s == pytest.approx(sv.lap_time(_oval_track(), limit, 1.0))
        between = got.read(0.5 * (got.max_driver_lap_s + got.fastest_lap_s), SHARE)
        assert between.status == pace.NO_SOLUTION  # until the knob has its cap
        assert between.max_driver_lap_s < between.lap_time_s < between.fastest_lap_s

    def test_beyond_the_slow_end_is_read_off_the_last_knots_and_flagged(self, table):
        beyond = float(table.lap_time_s[0]) * 1.1
        got = table.read(beyond, SHARE)
        assert got.status == pace.OUTSIDE_SUPPORT
        assert got.use < pace.USE_SLOW and (got.heat < table.heat_knots(SHARE)[0]).all()

    def test_a_layout_the_car_has_not_run_takes_the_cars_range(self):
        model = _ranged(_model(), a=(0.80, 0.95), b=(0.75, 1.004))
        got = pace.build_table(model, _oval_track())
        assert (got.support["use_min"], got.support["use_max"]) == (0.75, 1.0)
        assert "has not run" in got.support["from"] and got.support["laps"] == 80
        bare = pace.build_table(_model(), _oval_track())
        assert bare.support == {} and bare.status(float(bare.lap_time_s[3])) == pace.OUTSIDE_SUPPORT


class TestAboveAUseOfOne:
    def _capped(self):
        model = _ranged(_model(), oval=(0.80, 0.97))
        model.grip["mu_y_limit"] = model.grip["mu_y"]
        model.driver = {"brake_use": 0.85, "brake_use_ellipse": 0.0, "use_ref": 0.9}
        return model, pace.build_table(model, _oval_track())

    def test_knots_to_the_knobs_end_lap(self):
        model, table = self._capped()
        assert len(table.use) == pace.KNOTS + pace.KNOTS_ABOVE
        assert table.use[pace.KNOTS - 1] == 1.0 and table.use[-1] == pytest.approx(1.0 / 0.85)
        assert (np.diff(table.lap_time_s) < 0).all()
        # the end lap: braked at the car's level; the max driver's is that lap here
        # (the driver holds the car's own shape in this model)
        assert table.fastest_lap_s == pytest.approx(table.max_driver_lap_s, rel=1e-9)

    def test_between_one_and_the_end_is_answered_and_flagged(self):
        _, table = self._capped()
        t = 0.5 * (float(table.lap_time_s[pace.KNOTS - 1]) + table.fastest_lap_s)
        got = table.read(t, SHARE)
        assert got.status == pace.OUTSIDE_SUPPORT and np.isfinite(got.heat).all()
        assert got.use > 1.0 and got.q < 0.0
        assert got.cornering_use == 1.0 and 0.85 < got.braking_use < 1.0
        under = table.read(float(table.lap_time_s[5]), SHARE)
        assert under.cornering_use == pytest.approx(table.use[5])
        assert under.braking_use == pytest.approx(0.85 * table.use[5])
        assert table.read(table.fastest_lap_s - 0.3, SHARE).status == pace.NO_SOLUTION

    def test_the_table_holds_to_a_direct_solve_above_one(self):
        _, table = self._capped()
        times = np.linspace(table.fastest_lap_s, float(table.lap_time_s[pace.KNOTS - 1]), 8)[1:-1]
        use, laps = pace.direct(table, times)
        assert (use > 1.0).all() and all(lap is not None for lap in laps)
        want = np.vstack([heat.sim_heat(lap, SHARE) for lap in laps])
        assert np.abs(np.log(table.heat(times, SHARE) / want)).max() < 0.01


class TestWetInTheSupport:
    def test_the_slow_end_is_the_wettest_logged_lap(self):
        model = _ranged(_model(), oval=(0.80, 0.97))
        wet_range = {"min": 0.55, "p05": 0.56, "median": 0.62, "p95": 0.70, "max": 0.72}
        model.meta["use_range_wet"] = {"oval": {**wet_range, "laps": 20, "days": 2}}
        got = pace.build_table(model, _oval_track())
        assert got.support["use_min"] == 0.55 and got.support["use_min_dry"] == 0.80
        assert got.support["wet_laps"] == 20 and "wettest logged day" in got.support["from"]

        def at(use):
            return float(sv.lap_time(_oval_track(), CAR, use))

        assert got.status(at(0.60)) == pace.OK  # a wet lap, read off the same table
        assert got.status(at(0.50)) == pace.OUTSIDE_SUPPORT  # wetter than the wettest logged day

    def test_a_layout_with_no_wet_laps_takes_the_cars_wet_range(self):
        model = _ranged(_model(), oval=(0.80, 0.97))
        model.meta["use_range_wet"] = {
            "other": {
                "min": 0.60,
                "p05": 0.6,
                "median": 0.65,
                "p95": 0.7,
                "max": 0.7,
                "laps": 9,
                "days": 1,
            }
        }
        got = pace.build_table(model, _oval_track())
        assert got.support["use_min"] == 0.60
        assert "at the layouts it has run in the wet" in got.support["from"]


class TestAssumes:
    def test_the_latest_powertrain_unless_one_is_named(self):
        model = _two_engine_model()
        track = _oval_track()
        latest = pace.build_table(model, track)
        assert latest.assumes["powertrain"] == "second engine"
        assert latest.assumes["air"].startswith("standard")
        first = pace.build_table(model, track, powertrain="first engine")
        assert first.assumes["powertrain"] == "first engine"
        assert first.assumes["shift_rpm"] == 6900.0
        assert first.fastest_lap_s != latest.fastest_lap_s
        with pytest.raises(KeyError):
            pace.build_table(model, track, powertrain="third engine")

    def test_per_circuit_gearing_is_said(self):
        model = _two_engine_model()
        spec = model.power["engines"]["second engine"]
        spec["gear_sets"] = [{"gears": {"3": 5.0}}, {"gears": {"3": 5.4}}]
        _, logged = pace.assumed_car(model)
        assert "the set at this layout is assumed" in logged["gears_note"]
        spec["gears_inferred"] = True
        _, inferred = pace.assumed_car(model)
        assert "changes nothing in the lap" in inferred["gears_note"]

    def test_saved_and_read_back(self, table, tmp_path):
        path = tmp_path / "tables" / "car__oval.json"
        table.save(path)
        model = _ranged(_model(), oval=(0.80, 0.97))
        back = pace.from_dict(json.loads(path.read_text()), model, _oval_track())
        np.testing.assert_array_equal(back.v_node_ms, table.v_node_ms)
        assert back.support == table.support and back.assumes == table.assumes
        t = float(table.lap_time_s[3]) + 0.2
        np.testing.assert_array_equal(back.heat(t, SHARE), table.heat(t, SHARE))


class TestOutLap:
    def _piped(self, s_join=400.0, v_join=15.0):
        track = _oval_track()
        track.meta["pit"] = {"s_join_m": s_join, "v_join_ms": v_join, "rolling_s": 20.0, "n": 12}
        return track, pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track)

    def test_from_the_join_to_the_line_at_the_knots_use(self):
        track, table = self._piped()
        n, ds = track.n, track.ds_m
        assert table.v_out_ms is not None and table.v_out_ms.shape == (pace.KNOTS, n - 200 + 1)
        for k in (0, pace.KNOTS - 1):
            lap, flying = table.out_lap(k), table.lap(k)
            assert lap.v_node_ms[0] == pytest.approx(15.0)
            assert lap.s_m[-1] + ds == pytest.approx((n - 200) * ds)
            # begun slowly on the straight, it is the flying lap again by the next corner
            same = np.concatenate([flying.v_node_ms[-299:], flying.v_node_ms[:1]])
            np.testing.assert_allclose(lap.v_node_ms[-300:], same, rtol=1e-9)
            assert lap.lap_time_s > flying.dt_s[200:].sum()
            # and it arrives at the line as a lap that goes on, at the flying lap's speed there
            assert lap.v_node_ms[-1] == pytest.approx(flying.v_node_ms[0], abs=1e-9)

    def test_its_time_and_heat_at_a_lap_time(self):
        _, table = self._piped()
        part, hot = table.out_knots(SHARE)
        assert part.shape == (pace.KNOTS,) and hot.shape == (pace.KNOTS, 4)
        assert (np.diff(part) < 0).all()
        np.testing.assert_allclose(hot[3], heat.sim_heat(table.out_lap(3), SHARE))
        got = table.read_out(float(table.lap_time_s[3]), SHARE)
        assert got[0] == pytest.approx(part[3]) and got[1] == pytest.approx(hot[3])
        none = table.read_out(table.fastest_lap_s - 1.0, SHARE)
        assert np.isnan(none[0]) and np.isnan(none[1]).all()

    def test_a_track_without_a_pit_exit_has_none(self, table, tmp_path):
        assert table.v_out_ms is None and table.out_lap(0) is None
        assert table.out_knots(SHARE) is None and table.read_out(60.0, SHARE) is None
        track, piped = self._piped()
        piped.save(tmp_path / "t.json")
        model = _ranged(_model(), oval=(0.80, 0.97))
        back = pace.from_dict(json.loads((tmp_path / "t.json").read_text()), model, track)
        np.testing.assert_array_equal(back.v_out_ms, piped.v_out_ms)
        assert back.pit == piped.pit
        no_speed = _oval_track()
        no_speed.meta["pit"] = {"s_join_m": 400.0, "rolling_s": 20.0, "n": 12}
        assert pace.build_table(_model(), no_speed).v_out_ms is None


class TestExport:
    def test_numbers_for_a_consumer_without_a_solver(self, table):
        got = table.export(SHARE)
        assert len(got["knots"]) == pace.KNOTS and got["fastest_lap_s"] == table.fastest_lap_s
        knot = got["knots"][4]
        want = heat.sim_heat(table.lap(4), SHARE)
        assert [knot[f"q_{c}"] for c in ("fl", "fr", "rl", "rr")] == pytest.approx(list(want))
        assert knot["use"] == pytest.approx(table.use[4]) and knot["q"] == pytest.approx(table.q[4])
        assert knot["cornering_use"] == knot["braking_use"] == knot["use"]  # no cap, no driver
        assert "out_part_s" not in knot and got["support"] == table.support
        # read log-log between two knots, it is the table's own answer
        a, b = got["knots"][4], got["knots"][5]
        t = math.sqrt(a["lap_time_s"] * b["lap_time_s"])
        assert math.sqrt(a["q_fl"] * b["q_fl"]) == pytest.approx(table.heat(t, SHARE)[0])
        json.dumps(got)  # plain numbers

    def test_with_the_out_lap_and_other_shares(self):
        track = _oval_track()
        track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
        table = pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track)
        got = table.export(SHARE)
        part, hot = table.out_knots(SHARE)
        assert got["knots"][3]["out_part_s"] == pytest.approx(part[3])
        assert got["knots"][3]["out_q_rr"] == pytest.approx(hot[3, 3])
        other = table.export(replace(SHARE, p_f=0.60))
        assert other["knots"][3]["q_fl"] > got["knots"][3]["q_fl"]
        assert other["knots"][3]["lap_time_s"] == got["knots"][3]["lap_time_s"]


class TestUseRange:
    def test_the_range_of_use_over_a_cars_build_laps(self):
        track = _oval_track()
        uses = np.array([0.82, 0.86, 0.90, 0.93, 0.97])
        rows = pd.DataFrame(
            {
                "car": "car",
                "layout_id": "oval",
                "basis": "dry",
                "date": ["d1", "d1", "d2", "d2", "d3"],
                "lap_time_s": [sv.lap_time(track, CAR, float(u)) for u in uses],
            }
        )
        wet = rows.iloc[:1].assign(basis="wet", lap_time_s=rows["lap_time_s"].iloc[0] * 1.2)
        got = cg.use_range(_model(), {"oval": track}, pd.concat([rows, wet]), "car")["oval"]
        assert (got["laps"], got["days"]) == (5, 3)
        assert got["min"] == pytest.approx(0.82, abs=1e-4)
        assert got["max"] == pytest.approx(0.97, abs=1e-4)
        assert got["median"] == pytest.approx(0.90, abs=1e-4)
        assert cg.use_range(_model(), {}, rows, "car") == {}
        # the wet laps, as a lower use of the same car on the dry track
        wet_only = cg.use_range(_model(), {"oval": track}, pd.concat([rows, wet]), "car", "wet")
        assert wet_only["oval"]["laps"] == 1 and wet_only["oval"]["max"] < got["min"]


@pytest.fixture(scope="module")
def piped():
    track = _oval_track()
    track.meta["pit"] = {"s_join_m": 400.0, "v_join_ms": 15.0, "rolling_s": 20.0, "n": 12}
    return pace.build_table(_ranged(_model(), oval=(0.80, 0.97)), track)


class TestLapsExport:
    """The table as the laps the heat chain reads (#208): any fit's shares
    split it, without a solver or a track."""

    def test_any_shares_split_the_export_as_they_split_the_table(self, piped):
        export = piped.laps_export({"commit": "abc", "date": "2026-11-24"})
        other = replace(SHARE, p_f=0.55, g_transfer_front=1.5)
        for share in (SHARE, other):
            np.testing.assert_array_equal(export.heat_knots(share), piped.heat_knots(share))
            part, hot = export.out_knots(share)
            want_part, want_hot = piped.out_knots(share)
            np.testing.assert_array_equal(hot, want_hot)
            assert part == pytest.approx(want_part, rel=1e-12)
        assert export.rolling_s == 20.0 and export.made["commit"] == "abc"
        np.testing.assert_array_equal(export.lap_time_s, piped.lap_time_s)

    def test_read_between_the_knots_as_the_table_is(self, piped):
        export = piped.laps_export()
        t = float(np.sqrt(piped.lap_time_s[3] * piped.lap_time_s[4]))
        np.testing.assert_allclose(
            export.at(t, export.heat_knots(SHARE)), piped.heat(t, SHARE), rtol=1e-12
        )
        assert export.use_at(t) == pytest.approx(piped.use_at(t), rel=1e-12)
        assert math.isnan(export.use_at(piped.fastest_lap_s * 0.99))

    def test_the_lap_time_of_a_use_is_the_inverse(self, piped):
        export = piped.laps_export()
        for use in (0.45, 0.8, 0.93, 1.0):
            t = export.lap_time_at(use)
            assert export.use_at(t) == pytest.approx(use, rel=1e-9)
        assert export.lap_time_at(1.0) == pytest.approx(piped.lap_time_s[pace.KNOTS - 1])
        assert math.isnan(export.lap_time_at(0.2)) and math.isnan(export.lap_time_at(3.0))

    def test_round_trip_and_no_pit_exit(self, piped, table, tmp_path):
        export = piped.laps_export({"commit": "abc"})
        export.save(tmp_path / "export.json")
        back = type(export).load(tmp_path / "export.json")
        np.testing.assert_array_equal(back.heat_knots(SHARE), export.heat_knots(SHARE))
        np.testing.assert_array_equal(back.out_knots(SHARE)[1], export.out_knots(SHARE)[1])
        assert back.support == export.support and back.assumes == export.assumes
        assert back.made == {"commit": "abc"} and back.rolling_s == 20.0
        bare = table.laps_export()
        assert bare.out is None and bare.out_knots(SHARE) is None and bare.rolling_s == 0.0
        assert "out" not in bare.to_dict()["knots"][0]


class TestPlantedCar:
    """A planted wrong car (#208's controls): its knob is a free scale."""

    def test_no_cap_and_knots_past_one(self):
        model, track = _ranged(_model(), oval=(0.80, 0.97)), _oval_track()
        plain = pace.build_table(model, track)
        free = pace.build_table(model, track, use_cap=False, use_fast=1.5)
        assert len(free.use) == pace.KNOTS + pace.KNOTS_FREE
        assert free.use[-1] == 1.5 and free.use[pace.KNOTS - 1] == 1.0
        assert (np.diff(free.lap_time_s) < 0).all()
        # up to a use of 1 the laps are the capped table's
        np.testing.assert_array_equal(free.lap_time_s[: pace.KNOTS], plain.lap_time_s[: pace.KNOTS])
        assert free.assumes["planted"] == {"use_cap": False, "power_scale": 1.0}
        assert "planted" not in plain.assumes
        # a lap faster than the plain table's fastest has a use above 1
        assert free.use_at(float(free.lap_time_s[-1])) == pytest.approx(1.5)

    def test_less_power_is_a_slower_lap_at_the_same_use(self, tmp_path):
        model, track = _ranged(_model(), oval=(0.80, 0.97)), _oval_track()
        plain = pace.build_table(model, track)
        low = pace.build_table(model, track, power_scale=0.85, use_cap=False, use_fast=1.3)
        assert (low.lap_time_s[: pace.KNOTS] > plain.lap_time_s[: pace.KNOTS]).all()
        assert low.driven.power_scale == pytest.approx(0.85 * plain.driven.power_scale)
        low.save(tmp_path / "low.json")
        back = pace.from_dict(json.loads((tmp_path / "low.json").read_text()), model, track)
        assert back.driven.power_scale == low.driven.power_scale
        np.testing.assert_array_equal(back.heat_knots(SHARE), low.heat_knots(SHARE))
