"""Gate C2 through the solver (lap simulation C2)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car as cm
from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import limits as lim
from motorsports_data_notebook.tire_model.lapsim import samples as smp
from motorsports_data_notebook.tire_model.lapsim import solver as sv
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import write_dataset, write_model_json

CAR = sv.SolverCar(mu_y=1.2, mu_b=1.0, p_specific=100.0, c_drag=3.36e-4)


def _oval_track(key="oval", straight=900.0, radius=60.0, ramp=40.0, ds=2.0):
    """A stadium as a track: long enough straights to pass 50 m/s."""
    arc = math.pi * radius - ramp
    piece = np.concatenate(
        [
            np.zeros(int(straight / ds)),
            np.linspace(0.0, 1.0, int(ramp / ds), endpoint=False),
            np.ones(int(round(arc / ds))),
            np.linspace(1.0, 0.0, int(ramp / ds), endpoint=False),
        ]
    )
    kappa = np.concatenate([piece, piece]) / radius
    n = len(kappa)
    return trk.TrackModel(
        layout_key=key,
        track_canonical=key,
        length_m=n * ds,
        ds_m=ds,
        kappa_heat=kappa,
        x_m=np.zeros(n),
        y_m=np.zeros(n),
        alt_m=np.zeros(n),
        grade=np.zeros(n),
        n_laps=np.ones(n, dtype=int),
        kappa_se=np.zeros(n),
        corners=trk.corner_table(kappa, ds),
        kappa_v=np.zeros(n),
    )


def _model(car=CAR, **grip):
    return cm.CarModel(
        car="car",
        raw_car="car",
        mass_kg=None,
        grip={"mu_y": car.mu_y, "mu_b": car.mu_b, "n_brake": 1.0, **grip},
        power={"p_specific": car.p_specific, "c_drag": car.c_drag, "k_corner_drag": 0.0},
        wet={"grip_scale": None},
    )


GEARED = replace(
    CAR,
    curve_rpm=(4125.0, 5125.0, 6125.0, 7125.0),
    curve_p=(70.0, 85.0, 100.0, 100.0),
    gears=(5.0, 6.5, 8.0, 9.5),
    gear_numbers=(3, 4, 5, 6),
    shift_rpm=7000.0,
)


def _two_engine_model():
    model = _model()
    model.power.update(
        rho_ref=cm.RHO_REF,
        default_engine="second engine",
        engines={
            "first engine": {
                "p_specific": 90.0,
                "curve": [[5000.0, 80.0], [6000.0, 90.0], [7000.0, 90.0]],
                "gears": {"3": 5.0, "4": 6.5},
                "shift_rpm": 6900.0,
            },
            "second engine": {
                "p_specific": 100.0,
                "curve": [[5000.0, 88.0], [6000.0, 99.0], [7000.0, 99.0]],
                "gears": {"3": 5.3, "4": 6.9},
                "shift_rpm": 7300.0,
            },
        },
    )
    return model


class TestDrive:
    def test_a_lap_gets_its_days_powertrain_shift_point_and_air(self):
        air = pd.DataFrame({"rho": [1.25], "cf": [0.95]}, index=["cold"])
        drive = cg.Drive(_two_engine_model(), air=air, shifts={"2025-08-01": 7050.0})
        # the Inferno 86's engines are told apart by the date
        old = drive.car("Inferno 86", "2025-08-01", "cold")
        assert old.gears == (5.0, 6.5) and old.shift_rpm == 7050.0
        assert old.power_scale == pytest.approx(1 / 0.95)
        assert old.c_drag == pytest.approx(CAR.c_drag * 1.25 / cm.RHO_REF)
        new = drive.car("Inferno 86", "2026-08-01", "no weather")
        assert new.gears == (5.3, 6.9) and new.shift_rpm == 7300.0  # the powertrain's own
        assert new.power_scale == 1.0 and new.c_drag == pytest.approx(CAR.c_drag)
        assert drive.car("Inferno 86", "2025-08-01", "cold") is old  # kept

    def test_a_day_is_driven_through_the_gear_set_it_ran(self):
        drive = cg.Drive(_two_engine_model(), gears={"2026-08-01": (4.5, 5.5, 6.5)})
        assert drive.car("Inferno 86", "2026-08-01").gears == (4.5, 5.5, 6.5)
        assert drive.car("Inferno 86", "2026-08-02").gears == (5.3, 6.9)  # the powertrain's own
        work = {"engines": {"e": {"shifts": {}, "gear_sets": {"d": {2: 6.5, 1: 5.0}}}}}
        assert cg.day_gears(work) == {"d": (5.0, 6.5)}

    def test_a_car_the_model_has_no_powertrain_for_gets_the_default(self):
        drive = cg.Drive(_two_engine_model())
        assert drive.car("KK-SII", "2026-01-01").gears == (5.3, 6.9)
        assert drive.car().gears == (5.3, 6.9)

    def test_the_tracks_bank_is_a_setting_of_the_lap(self):
        track = _oval_track()
        track.bank = 0.05 * np.sign(track.kappa_heat)
        drive = cg.Drive(_two_engine_model())
        assert cg.profile_of(drive, track).bank == pytest.approx(track.bank)
        assert cg.profile_of(CAR, track).bank is not None  # a bare car too
        level = drive.but(bank=False)
        assert level.bank is False and level.but(power_scale=0.9).bank is False
        assert cg.profile_of(level, track).bank is None
        # the reference laps are driven on it: the same lap times want less of the grip
        best = pd.DataFrame({"layout_id": ["oval"], "date": ["d"], "lap_time_s": [60.0]})
        flat = cg.reference_u(level, {"oval": track}, best)["oval"]["u_ref"]
        assert cg.reference_u(drive, {"oval": track}, best)["oval"]["u_ref"] < flat - 0.01

    def test_planted_power_and_drag(self):
        drive = cg.Drive(_two_engine_model())
        low = drive.but(power_scale=0.85, drag_scale=1.5)
        assert low.car().power_at(30.0) == pytest.approx(0.85 * drive.car().power_at(30.0))
        assert low.car().c_drag == pytest.approx(1.5 * drive.car().c_drag)
        assert low.model is drive.model and drive.power_scale == 1.0

    def test_a_bare_car_or_model_is_taken_as_it_is(self):
        assert cg.as_drive(CAR) is CAR
        assert isinstance(cg.as_drive(_model()), cg.Drive)


class TestPaceCurve:
    def test_solves_like_the_bisection(self):
        track = _oval_track()
        curve = sv.PaceCurve.build(track, CAR)
        assert (np.diff(curve.t) < 0).all()
        want = sv.lap_time(track, CAR, 0.93)
        u, status = curve.solve(want)
        assert status == "ok" and u == pytest.approx(0.93, abs=1e-6)
        assert curve.solve(want / 3.0)[1] == "no_root"


class TestConvention:
    def test_mu_y_goes_where_the_reference_laps_u_is_one(self):
        tracks = {"a": _oval_track("a"), "b": _oval_track("b", straight=500.0, radius=40.0)}
        rows = []
        for layout, track in tracks.items():
            for day, u in (("2026-01-10", 0.98), ("2026-01-11", 1.0), ("2026-01-12", 1.02)):
                rows.append(
                    {"layout_id": layout, "date": day, "lap_time_s": sv.lap_time(track, CAR, u)}
                )
        best = pd.DataFrame(rows)
        model, refs = cg.set_convention(_model(mu_y=1.35), tracks, best)
        assert model.grip["mu_y"] == pytest.approx(1.2, abs=0.005)
        assert model.grip["mu_b"] == 1.0  # a measurement: it stays
        assert model.meta["mu_y_apex_table"] == 1.35
        for layout in tracks:
            assert refs[layout]["u_ref"] == pytest.approx(1.0, abs=0.005)
            assert model.ref_u[layout][0] == pytest.approx(1.0, abs=0.005)
            assert model.ref_u[layout][1] == pytest.approx(0.02, abs=0.003)
            assert set(refs[layout]["by_day"]) == {"2026-01-10", "2026-01-11", "2026-01-12"}

    def test_each_day_is_driven_as_it_was(self):
        # a day driven with later shifts is a faster lap at the same grip:
        # read with its own shift point its u is the same as the other day's
        track = _oval_track()
        early, late = replace(GEARED, shift_rpm=6400.0), GEARED
        best = pd.DataFrame(
            {
                "layout_id": "a",
                "date": ["d1", "d2"],
                "lap_time_s": [sv.lap_time(track, early, 1.0), sv.lap_time(track, late, 1.0)],
                "raw_car": "car",
                "session_id": ["s1", "s2"],
            }
        )
        model = _model(GEARED)
        model.power.update(
            default_engine="engine",
            engines={
                "engine": {
                    "p_specific": 100.0,
                    "curve": [[r - 125.0, p] for r, p in zip(GEARED.curve_rpm, GEARED.curve_p)],
                    "gears": {str(n): k for n, k in zip(GEARED.gear_numbers, GEARED.gears)},
                    "shift_rpm": 7000.0,
                }
            },
        )
        with_shifts = cg.Drive(model, shifts={"d1": 6400.0, "d2": 7000.0})
        us = cg.reference_u(with_shifts, {"a": track}, best)["a"]["by_day"]
        assert us["d1"] == pytest.approx(1.0, abs=1e-4) and us["d2"] == pytest.approx(1.0, abs=1e-4)
        blind = cg.reference_u(cg.Drive(model), {"a": track}, best)["a"]["by_day"]
        assert blind["d1"] < 0.995  # the early shifter's lap read as less grip

    def test_day_best_laps(self):
        rows = pd.DataFrame(
            {
                "car": ["car"] * 4 + ["other"],
                "basis": ["dry", "dry", "wet", "dry", "dry"],
                "layout_id": "a",
                "date": ["d1", "d1", "d1", "d2", "d1"],
                "lap_time_s": [61.0, 60.0, 55.0, 62.0, 50.0],
            }
        )
        best = cg.day_best_laps(rows, "car")
        assert dict(zip(best["date"], best["lap_time_s"])) == {"d1": 60.0, "d2": 62.0}
        keyed = rows.assign(session_id=list("abcde"), raw_car="car")
        assert list(cg.day_best_laps(keyed, "car")["session_id"]) == ["b", "d"]


class TestSetLimit:
    """``mu_y`` at the car's limit, the knob the driver's use of it (#312)."""

    USE = {"brake_use": 0.8, "brake_use_ellipse": 0.5}

    def _best(self, tracks, car, us=(0.98, 1.0, 1.02)):
        rows = []
        for layout, track in tracks.items():
            for k, u in enumerate(us):
                rows.append(
                    {"layout_id": layout, "date": f"d{k}", "lap_time_s": sv.lap_time(track, car, u)}
                )
        return pd.DataFrame(rows)

    def test_mu_y_goes_to_the_limit_and_the_knob_is_the_use(self):
        tracks = {"a": _oval_track("a"), "b": _oval_track("b", straight=500.0, radius=40.0)}
        best = self._best(tracks, CAR)  # driven with 1.2 g of cornering at the reference
        model = _model(mu_y=1.35)
        model.grip["mu_y_limit"] = 1.5
        got, refs = cg.set_limit(model, tracks, best)
        assert got.grip["mu_y"] == 1.5 and got.grip["mu_b"] == 1.0
        assert got.meta["mu_y_apex_table"] == 1.35 and "limit" in got.meta["mu_y_source"]
        for layout in tracks:
            # about 1.2 of 1.5; not exactly, for mu_b did not rise with mu_y and the
            # knob lowers both
            assert 0.79 < refs[layout]["u_ref"] < 0.83
            assert got.ref_u[layout][0] == pytest.approx(refs[layout]["u_ref"], abs=1e-4)
        assert got.meta["use_ref"] == pytest.approx(
            np.median([r["u_ref"] for r in refs.values()]), abs=1e-5
        )
        assert cg.typical_use(got) == cg.typical_use(cg.Drive(got)) == got.meta["use_ref"]
        assert cg.typical_use(CAR) == 1.0 and cg.typical_use(_model()) == 1.0

    def test_every_lap_is_the_lap_it_was(self):
        track = _oval_track()
        model = _model()
        model.driver = dict(self.USE)
        car = sv.SolverCar.from_model(model)
        best = self._best({"a": track}, car)
        times = [sv.lap_time(track, car, u) for u in (0.93, 1.0, 1.04)]
        was, _ = cg.set_convention(model, {"a": track}, best)
        base = sv.solve_laps(track, sv.SolverCar.from_model(was), times)
        for limit in (1.3, 1.5, 1.65):
            raised = _model()
            raised.driver = dict(self.USE)
            raised.grip["mu_y_limit"] = limit
            got, _ = cg.set_limit(raised, {"a": track}, best)
            # the share of mu_b rises as mu_y does: the same braking in g at the same lap
            assert got.driver["brake_use"] == pytest.approx(0.8 * limit / 1.2)
            assert got.driver["brake_use_ellipse"] == 0.5
            solved = sv.solve_laps(track, sv.SolverCar.from_model(got), times)
            assert solved.v_node_ms == pytest.approx(base.v_node_ms, rel=1e-6)
            assert solved.u * limit == pytest.approx(base.u * was.grip["mu_y"], rel=1e-6)
            assert (solved.u < 1.0).all() and got.driver["use_ref"] == got.meta["use_ref"]

    def test_a_wrong_car_keeps_the_fitted_drivers_braking_level(self):
        track = _oval_track()
        model = _model()
        model.driver = dict(self.USE)
        model.grip["mu_y_limit"] = 1.5
        best = self._best({"a": track}, sv.SolverCar.from_model(model), us=(0.78, 0.8, 0.82))
        fitted, _ = cg.set_limit(model, {"a": track}, best)
        bare = _model()
        bare.grip["mu_y_limit"] = 1.5
        wrong, _ = cg.set_limit(
            cg.Drive(bare, power_scale=0.85), {"a": track}, best, hold=fitted.driver
        )
        # less power: the laps want more of the limit, and the share of mu_b falls so
        # that the braking in g at the typical use is the fitted driver's
        assert wrong.meta["use_ref"] > fitted.meta["use_ref"] * 1.01
        assert wrong.driver["brake_use"] * wrong.meta["use_ref"] == pytest.approx(
            fitted.driver["brake_use"] * fitted.meta["use_ref"], rel=3e-3
        )
        assert wrong.driver["brake_use_ellipse"] == 0.5

    def test_a_model_without_a_limit_is_set_by_the_convention(self):
        track = _oval_track()
        best = self._best({"a": track}, CAR)
        got, refs = cg.set_limit(_model(mu_y=1.35), {"a": track}, best)
        assert got.grip["mu_y"] == pytest.approx(1.2, abs=0.005)
        assert refs["a"]["u_ref"] == pytest.approx(1.0, abs=0.005) and cg.typical_use(got) == 1.0
        used = _model(mu_y=1.35)
        held, _ = cg.set_limit(used, {"a": track}, best, hold={**self.USE, "use_ref": 0.8})
        assert held.driver == self.USE  # the convention's own way of holding a use


class TestAtLimitAndMaxDriver:
    def test_the_use_is_read_as_a_share_of_the_limit(self):
        model = _model(mu_y=1.35)
        assert cg.at_limit(model) is model  # no limit: nothing to move to
        model.grip["mu_y_limit"] = 1.5
        raised = cg.at_limit(model)
        assert raised.grip["mu_y"] == 1.5 and raised.meta["mu_y_apex_table"] == 1.35
        track = _oval_track()
        best = pd.DataFrame(
            {"layout_id": "a", "date": ["d1"], "lap_time_s": [sv.lap_time(track, CAR, 1.0)]}
        )
        got, _ = cg.set_limit(raised, {"a": track}, best)
        assert got.grip["mu_y"] == 1.5 and got.meta["mu_y_apex_table"] == 1.35  # still the table's

    def test_the_max_driver_against_the_best_lap(self):
        track = _oval_track()
        model = _model()
        model.driver = {"brake_use": 0.8, "brake_use_ellipse": 0.5}
        drive = cg.Drive(model)
        car = sv.SolverCar.from_model(model)
        laps = pd.DataFrame(
            {
                "layout_id": ["a", "a", "b"],
                "date": ["d1", "d2", "d1"],
                "lap_time_s": [sv.lap_time(track, car, u) for u in (0.9, 0.95, 0.9)],
            }
        )
        levels = {"m20": 1.2, "low": 1.0, "none": float("nan")}
        rows = cg.max_driver_rows("car", drive, {"a": track}, laps, levels, ("b",))
        assert len(rows) == 1
        row = rows[0]
        assert row["best_lap_s"] == pytest.approx(laps["lap_time_s"][1]) and row["date"] == "d2"
        assert row["borrowed"] and "none" not in row
        # every use at 1 and the brakes at the car's limit: faster than a lap at 0.95 of it
        at_limit = sv.lap_time(track, sv.SolverCar.from_model(model, brake_use=False), 1.0)
        assert row["m20"] == pytest.approx(at_limit / row["best_lap_s"] - 1.0, abs=1e-9)
        assert row["m20"] < 0.0 < row["low"]  # a limit under what the lap used is slower
        assert (
            cg.max_driver_rows(
                "car", drive, {"a": track}, laps[laps["layout_id"] == "b"], levels, ()
            )
            == []
        )


class TestBestSegments:
    def test_a_lap_stitched_from_the_expected_best_of_its_parts(self):
        track = _oval_track()
        fast = sv.simulate_lap(track, CAR, 1.0).v_node_ms
        nodes = np.arange(track.n)
        apexes = [int(round(c["s_apex"] / track.ds_m)) for c in track.corners]
        laps = []
        for k in range(40):  # every lap loses time round one apex or round the other
            near = np.abs(nodes - apexes[k % 2]) <= 40
            laps.append(np.where(near, 0.9 * fast, fast))
        stitched, fastest = cg.best_segments(track, np.vstack(laps), 5)
        whole = float(np.sum(track.ds_m / (0.5 * (fast + np.roll(fast, -1)))))
        assert fastest > whole * 1.002  # no lap was fast everywhere
        assert whole <= stitched < fastest  # its best parts together nearly are
        assert stitched == pytest.approx(whole, rel=2e-3)
        same = np.vstack([fast] * 10)
        both = cg.best_segments(track, same, 5)
        assert both[0] == pytest.approx(both[1], rel=1e-12)
        assert np.isnan(cg.best_segments(track, same[:3], 5)[0])

    def test_beside_the_max_driver(self):
        track = _oval_track()
        model = _model()
        car = sv.SolverCar.from_model(model)
        laps = pd.DataFrame(
            {"layout_id": "a", "date": ["d1"], "lap_time_s": [sv.lap_time(track, car, 0.95)]}
        )
        traces = {"a": np.vstack([sv.simulate_lap(track, car, 0.95).v_node_ms] * 25)}
        rows = cg.max_driver_rows(
            "car", cg.Drive(model), {"a": track}, laps, {"m20": 1.2}, ("a",), traces=traces
        )
        assert rows[0]["stitched"] == pytest.approx(0.0, abs=1e-9)  # every lap the same lap
        assert rows[0]["m20"] < 0.0 and not rows[0]["borrowed"]


class TestBrakingUse:
    """How the driver brakes, read from speed traces at each lap's own u."""

    USE = {"brake_use": 0.8, "brake_use_ellipse": 0.4}

    def _driven(self):
        """Two ovals whose corners come on slowly (braking under every share
        of lateral), a model with a braking use, and laps it drove."""
        tracks = {
            "a": _oval_track("a", straight=600.0, radius=100.0, ramp=200.0),
            "b": _oval_track("b", straight=500.0, radius=70.0, ramp=160.0),
        }
        model = _model(brake_ellipse=0.7)
        model.driver = dict(self.USE)
        car = cg.Drive(model).car()
        laps = {}
        for key, track in tracks.items():
            got = sv.drive_laps(track, car, np.linspace(0.94, 1.06, 13))
            frame = pd.DataFrame(
                {
                    "session_id": [f"s{k}" for k in range(13)],
                    "lap_num": 1,
                    "date": [f"d{k % 3}" for k in range(13)],
                    "lap_time_s": got.lap_time_s,
                }
            )
            laps[key] = cg.use_laps(track, frame, got.v_node_ms)
        return tracks, model, laps

    def test_time_into_a_stop_that_passes_the_start_line(self):
        slowing = np.zeros((1, 10), bool)
        slowing[0, [0, 1, 4, 5, 6, 9]] = True
        t = cg.time_into_stop(slowing, np.full((1, 10), 20.0), 2.0)  # 0.1 s a node
        assert t[0, [4, 5, 6]] == pytest.approx([0.0, 0.1, 0.2])
        assert t[0, [9, 0, 1]] == pytest.approx([0.0, 0.1, 0.2])  # began before the line
        assert np.isnan(t[0, [2, 3, 7, 8]]).all()
        assert np.isnan(
            cg.time_into_stop(np.zeros((1, 10), bool), np.full((1, 10), 20.0), 2.0)
        ).all()

    def test_the_reading_gives_a_level_and_a_shape_back(self, monkeypatch):
        monkeypatch.setattr(cg, "USE_BAND_MIN_NODES", 5)
        x = np.tile(np.linspace(0.02, 0.88, 44), 5)
        y = 0.8 * (0.6 * (1.0 - x) + 0.4 * np.sqrt(1.0 - x * x))
        level, shape, bands = cg.read_use(y, x)
        assert level == pytest.approx(0.8, abs=1e-3) and shape == pytest.approx(0.4, abs=1e-3)
        assert len(bands) == cg.USE_BANDS and bands["nodes"].sum() == len(x)
        assert np.isnan(cg.read_use(y[:4], x[:4])[0])  # one band is no reading

    def test_the_nodes_are_past_the_build_and_in_the_solvers_terms(self):
        tracks, model, laps = self._driven()
        own = laps["a"]
        assert len(own.frame) == 13 and set(own.lap) == set(range(13))
        # braking force over the load: what slows the car less the drag, never over the level
        car = cg.Drive(model).car()
        force = own.slowing - car.c_drag * own.v2
        u = np.linspace(0.94, 1.06, 13)[own.lap]
        assert np.nanmax(force / (car.mu_b * u)) < 0.8 * 1.03
        assert (own.lat >= 0).all() and np.nanmax(own.lat / (car.mu_y * u)) <= 1.02
        some = own.keep(own.frame["date"].to_numpy() != "d0")
        assert len(some.frame) == 8 and set(some.lap) == set(range(8))
        assert len(some.lap) == int((own.frame["date"].to_numpy()[own.lap] != "d0").sum())

    def test_a_use_planted_in_simulated_laps_comes_back(self, monkeypatch):
        monkeypatch.setattr(cg, "USE_BAND_MIN_NODES", 20)
        tracks, model, laps = self._driven()
        back, work = cg.set_use(cg.Drive(_model(brake_ellipse=0.7)), tracks, laps)
        assert back.driver["brake_use"] == pytest.approx(0.8, rel=0.03)
        assert back.driver["brake_use_ellipse"] == pytest.approx(0.4, abs=0.08)
        assert back.grip == model.grip  # the car's limits are not touched
        assert len(work["rounds"]) >= 2 and work["laps"] == 26 and work["nodes"] > 500
        # the first reading, from laps braked at the limit, is not yet the one that holds
        assert abs(work["rounds"][0][0] - 0.8) > abs(work["rounds"][-1][0] - 0.8)
        again = cg.recover_use(cg.Drive(model), tracks, laps)
        assert again["level_back"] == pytest.approx(again["level"], rel=0.03)
        assert again["shape_back"] == pytest.approx(again["shape"], abs=0.08)

    def test_too_few_nodes_and_the_laps_are_braked_at_the_limit(self):
        tracks, model, laps = self._driven()
        one = {k: v.keep(np.arange(len(v.frame)) == 0) for k, v in laps.items()}  # a lap each
        back, work = cg.set_use(cg.Drive(model), tracks, one)
        assert back.driver == {} and work["rounds"] == []
        assert cg.Drive(back).car().brake_use == 1.0

    def test_the_convention_moves_the_share_and_leaves_the_laps(self):
        track = _oval_track()
        model = _model()
        model.driver = dict(self.USE)
        moved = cg._with_mu_y(model, 1.26)  # mu_y × 1.05
        assert moved.driver["brake_use"] == pytest.approx(0.84)
        a = sv.simulate_lap(track, sv.SolverCar.from_model(model), 1.05)
        b = sv.simulate_lap(track, sv.SolverCar.from_model(moved), 1.0)
        assert a.v_node_ms == pytest.approx(b.v_node_ms, rel=1e-9)
        assert cg._with_mu_y(model, 1.26, hold_use=True).driver["brake_use"] == 0.8
        car = sv.SolverCar.from_model(model)
        best = pd.DataFrame(
            {
                "layout_id": "a",
                "date": ["d1", "d2", "d3"],
                "lap_time_s": [sv.lap_time(track, car, u) for u in (0.98, 1.0, 1.02)],
            }
        )
        start = cg._with_mu_y(model, 1.35)
        got, _ = cg.set_convention(start, {"a": track}, best)
        assert got.grip["mu_y"] == pytest.approx(1.2, abs=0.005)
        assert got.driver["brake_use"] == pytest.approx(0.8, abs=0.005)
        held, _ = cg.set_convention(start, {"a": track}, best, hold_use=True)
        assert held.driver["brake_use"] == start.driver["brake_use"]
        assert held.grip["mu_y"] < got.grip["mu_y"]  # braked harder, the laps want less grip

    def test_a_drive_can_brake_at_the_limit(self):
        model = _model(brake_ellipse=0.7)
        model.driver = dict(self.USE)
        drive = cg.Drive(model)
        assert (drive.car().brake_use, drive.car().brake_ellipse) == (0.8, 0.4)
        limit = drive.but(brake_use=False)
        assert (limit.car().brake_use, limit.car().brake_ellipse) == (1.0, 0.7)
        assert not limit.but(bank=False).brake_use  # and it is kept


class TestEraLadder:
    def test_a_cars_days_in_order_with_what_changed(self):
        best = pd.DataFrame(
            {
                "layout_id": ["b", "a", "a"],
                "date": ["2026-07-15", "2025-05-05", "2025-03-08"],
                "lap_time_s": [126.4, 78.8, 77.2],
                "raw_car": ["Inferno 86"] * 3,
            }
        )
        refs = {
            "a": {"by_day": {"2025-03-08": 1.008, "2025-05-05": 0.969}},
            "b": {"by_day": {"2026-07-15": 1.003}},
        }
        days = pd.DataFrame(
            {
                "date": ["2025-03-08", "2026-07-15"],
                "layout_id": ["a", "b"],
                "level": [0.991, 0.982],
                "shift_rpm": [7286.0, 7356.0],
            }
        )
        compounds = {("Inferno 86", "a", "2025-05-05"): "RE-71RS"}
        got = cg.era_ladder("Inferno 86", best, refs, days, compounds)
        assert list(got["date"]) == ["2025-03-08", "2025-05-05", "2026-07-15"]
        assert list(got["engine"]) == ["first engine", "first engine", "second engine"]
        assert list(got["compound"]) == ["", "RE-71RS", ""]
        assert list(got["u"]) == [1.008, 0.969, 1.003]
        assert got["level"].iloc[0] == 0.991 and np.isnan(got["level"].iloc[1])
        assert got["shift_rpm"].iloc[2] == 7356.0

    def test_a_day_without_a_solved_lap_is_left_out(self):
        best = pd.DataFrame({"layout_id": ["a"], "date": ["d"], "lap_time_s": [70.0]})
        assert cg.era_ladder("X", best, {"a": {"by_day": {}}}, pd.DataFrame(), {}).empty

    def test_a_dataset_without_labels_has_no_compounds(self, tmp_path):
        rows = pd.DataFrame({"car": ["X"], "layout_id": ["a"], "date": ["d"], "session_id": ["s"]})
        assert cg.day_compounds(tmp_path, rows) == {}


class TestClassLap:
    def test_the_other_drivers_lap_beside_the_cars_own(self, monkeypatch):
        monkeypatch.setitem(
            cm.CLASS_LAPS,
            "X",
            {
                "layout_id": "a",
                "what": "another",
                "lap_s": 59.9,
                "lat_g": (1.5, 1.6),
                "brake_g": 1.2,
            },
        )
        apex = pd.DataFrame(
            {
                "car": "X",
                "layout_id": "a",
                "basis": "dry",
                "corner": [1] * 10 + [2] * 10,
                "sign": 1,
                "s_apex_m": [100.0] * 10 + [300.0] * 10,
                "date": "d",
                "v_apex_ms": 20.0,
                "lat_doppler_g": [1.4] * 10 + [1.6] * 10,
                "lat_track_g": 1.4,
                "lat_accel_g": 1.4,
            }
        )
        brake = pd.DataFrame(
            {
                "car": "X",
                "layout_id": "a",
                "basis": ["dry"] * 4 + ["wet"],
                "decel_p95_g": [1.0, 1.1, 1.2, 1.3, 9.0],
            }
        )

        class Tables:
            pass

        tables = Tables()
        tables.apex, tables.brake = apex, brake
        rows = pd.DataFrame(
            {
                "car": "X",
                "layout_id": "a",
                "basis": ["dry", "dry", "wet"],
                "lap_time_s": [61.0, 60.2, 50.0],
            }
        )
        got = cg.class_lap_rows(tables, rows, {"X": _model()}).iloc[0]
        assert got["best_lap_s"] == 60.2 and got["class_lap_s"] == 59.9
        assert got["apex_level_g"] == pytest.approx(1.5)  # the median of the two corners' p90
        assert got["brake_p90_g"] == pytest.approx(np.quantile([1.0, 1.1, 1.2, 1.3], 0.9))
        assert got["mu_b"] == _model().grip["mu_b"]

    def test_a_car_without_such_a_lap_or_without_laps_there(self):
        class Tables:
            apex = pd.DataFrame()
            brake = pd.DataFrame()

        none = pd.DataFrame(columns=["car", "layout_id", "basis", "lap_time_s"])
        assert cg.class_lap_rows(Tables(), none, {"KK-SII": _model()}).empty
        assert cg.class_lap_rows(Tables(), none, {}).empty


class TestInSample:
    def test_a_lap_the_model_drove_scores_zero_at_its_own_u(self):
        track = _oval_track()
        sims = [sv.simulate_lap(track, CAR, u) for u in (0.93, 1.0)]
        laps = pd.DataFrame({"lap_time_s": [s.lap_time_s for s in sims], "date": ["d1", "d2"]})
        got = cg.lap_scores(CAR, track, laps, np.vstack([s.v_node_ms for s in sims]))
        assert got["u"].to_numpy() == pytest.approx([0.93, 1.0], abs=1e-5)
        assert got["speed_rmse"].max() < 1e-3
        assert list(got["status"]) == ["ok", "ok"]

    def test_a_wrong_car_does_not(self):
        track = _oval_track()
        sim = sv.simulate_lap(track, CAR, 1.0)
        laps = pd.DataFrame({"lap_time_s": [sim.lap_time_s], "date": ["d1"]})
        wrong = replace(CAR, p_specific=85.0)
        got = cg.lap_scores(wrong, track, laps, sim.v_node_ms[None, :])
        assert got["u"].iloc[0] > 1.0 and got["speed_rmse"].iloc[0] > 0.5

    def test_day_floor_pairs_days_at_the_same_pace(self):
        base = np.full(100, 30.0)
        v = np.vstack([base, base + 1.0, base + 3.0, base + 2.0])
        laps = pd.DataFrame(
            {"lap_time_s": [60.0, 60.1, 60.05, 70.0], "date": ["d1", "d2", "d1", "d2"]}
        )
        got = cg.day_floor(laps, v)
        # pairs across days within 0.5 %: (0, 1) and (1, 2); the 70 s lap has no partner
        assert got["pairs"] == 2 and got["floor"] == pytest.approx(1.5)
        one_day = cg.day_floor(laps.assign(date="d1"), v)
        assert one_day["pairs"] == 0 and np.isnan(one_day["floor"])


class TestRecovery:
    def test_a_simulated_lap_as_a_session(self):
        track = _oval_track()
        profile = sv.Profile.from_track(track)
        sim = sv.simulate_lap(profile, CAR, 1.0)
        sig, s, grade, lap_of = cg.signals_from_sim(sim, profile, CAR)
        in_lap = lap_of == 1
        assert in_lap.sum() == pytest.approx(sim.lap_time_s * 25.0, abs=2)
        assert (grade == 0).all() and s.min() >= 0 and s.max() < track.length_m
        assert sig.geom.speed[in_lap].max() == pytest.approx(sim.v_node_ms.max(), abs=0.1)
        assert sig.lat_g[in_lap].max() == pytest.approx(1.2, abs=0.03)
        assert sig.long_g[in_lap].min() < -0.9
        share = sig.throttle[in_lap].mean()
        assert 0.3 < share < 0.9
        assert not (sig.brake_on & (sig.throttle > 0.5)).any()

    def test_the_estimator_gives_the_car_back(self):
        tracks = {"a": _oval_track("a"), "b": _oval_track("b", straight=500.0, radius=40.0)}
        got = cg.recover(CAR, tracks)
        assert got["drag_layouts"] == ["a"]  # the short one does not reach 50 m/s
        assert got["p_ratio"] == pytest.approx(1.0, abs=0.03)
        assert got["c_ratio"] == pytest.approx(1.0, abs=0.2)
        assert got["mu_b_ratio"] == pytest.approx(1.0, abs=0.05)
        assert 0.9 < got["mu_y_ratio"] < 1.05

    def test_a_car_with_cornering_drag_comes_back_with_it(self):
        # a slow corner and a fast one with a long exit: full throttle at
        # lateral g at the speeds the straights are also driven at
        ds = 2.0
        pieces = []
        for radius, arc, ramp in ((60.0, 150.0, 40.0), (150.0, 200.0, 240.0)):
            pieces += [
                np.zeros(int(900.0 / ds)),
                np.linspace(0.0, 1.0, int(40.0 / ds), endpoint=False) / radius,
                np.full(int(arc / ds), 1.0 / radius),
                np.linspace(1.0, 0.0, int(ramp / ds), endpoint=False) / radius,
            ]
        kappa = np.concatenate(pieces)
        track = _oval_track("a")
        n = len(kappa)
        track = replace(
            track,
            length_m=n * ds,
            kappa_heat=kappa,
            x_m=np.zeros(n),
            y_m=np.zeros(n),
            alt_m=np.zeros(n),
            grade=np.zeros(n),
            n_laps=np.ones(n, dtype=int),
            kappa_se=np.zeros(n),
            kappa_v=np.zeros(n),
            corners=trk.corner_table(kappa, ds),
        )
        car = replace(CAR, k_corner_drag=0.04)
        got = cg.recover(car, {"a": track})
        assert got["k_true"] == 0.04
        assert got["k_corner_drag"] == pytest.approx(0.04, abs=0.015)
        assert got["p_ratio"] == pytest.approx(1.0, abs=0.03)
        assert got["c_ratio"] == pytest.approx(1.0, abs=0.2)

    def test_a_geared_car_comes_back_with_its_curve_gears_and_shift_point(self):
        track = _oval_track("a")
        sim = sv.simulate_lap(track, GEARED, 1.0)
        sig, *_ = cg.signals_from_sim(sim, sv.Profile.from_track(track), GEARED)
        assert set(np.unique(sig.gear)) <= {3.0, 4.0, 5.0, 6.0} and np.nanmax(sig.rpm) < 7300.0
        got = cg.recover(GEARED, {"a": track})
        assert got["p_ratio"] == pytest.approx(1.0, abs=0.03)
        assert got["gear_off"] < 0.005
        assert got["shift_true"] == 7000.0 and 6600.0 < got["shift_rpm"] <= 7000.0

    def test_tables_of_simulated_laps(self):
        tables = cg.simulated_tables(CAR, {"a": _oval_track("a")})
        assert tables.throttle["date"].nunique() == len(cg.RECOVERY_U)
        assert set(tables.apex["corner"]) == {0, 1}
        assert len(tables.brake) == 2 * len(cg.RECOVERY_U)
        assert (tables.throttle["a_g"] > 0).all()


class TestPlantedLimit:
    """The cornering limit's statistic on a car whose limit is known."""

    def _plant(self, **kw):
        track = _oval_track(straight=300.0)
        args = dict(use=0.9, sd_pass=0.04, noise_g=0.0, days=6, laps=10, m=10)
        args.update(kw)
        return cg.plant_limit(CAR, track, **args)

    def test_without_noise_the_statistic_reads_the_planted_passes(self):
        got = self._plant()
        assert got["limit"] == CAR.mu_y and got["corners"] == 2 and got["passes"] == 120
        assert got["truth"] < got["limit"]  # a typical use of 0.9 and a spread of 4 %
        # the table builder alone: a corner's largest lateral over the load at its apex
        assert got["clean"] == pytest.approx(got["truth"], rel=0.01)
        assert got["level"] == pytest.approx(got["clean"], rel=1e-6)
        assert got["cross"] == pytest.approx(got["clean"], rel=0.005)
        assert got["sd_noise"] == pytest.approx(0.0, abs=1e-9)

    def test_noise_inside_the_window_raises_a_sensors_own_peak(self):
        got = self._plant(noise_g=0.04, seed=1)
        assert got["raw"] > got["clean"] * 1.01  # the largest of noisy fixes
        assert got["level"] < got["raw"]  # the correction takes the spread's part out
        assert got["level"] > got["truth"] * 1.01  # and leaves what the maximum adds
        # read where the other sensor peaks, the noise chooses nothing for itself
        assert abs(got["cross"] / got["truth"] - 1.0) < abs(got["level"] / got["truth"] - 1.0)
        assert got["sd_noise"] > 0.005

    def test_a_capped_top_and_mistakes(self):
        capped = self._plant(top="capped")
        assert capped["truth"] == pytest.approx(
            CAR.mu_y, rel=2e-3
        )  # the passes pile up at the limit
        slow = self._plant(top="mistakes", seed=2)
        assert slow["truth"] < CAR.mu_y
        with pytest.raises(ValueError):
            self._plant(top="flat")

    def test_an_error_the_readings_share_is_not_removed(self):
        apart = self._plant(noise_g=0.02, seed=3)
        shared = self._plant(noise_g=0.02, sd_shared=0.03, seed=3)
        assert shared["level"] > apart["level"] and shared["cross"] > apart["cross"]

    def test_the_gates_rows(self):
        track = _oval_track(straight=300.0)
        model = _model()
        got = lim.Limit(
            1.2,
            1.21,
            20,
            corners=2,
            by_corner=pd.DataFrame(
                {
                    "layout_id": ["oval", "oval"],
                    "corner": [0, 1],
                    "n": [60, 60],
                    "sd": [0.045, 0.045],
                    "sd_noise": [0.012, 0.012],
                    "supports": [True, True],
                }
            ),
        )
        rows = pd.DataFrame(cg.plant_rows("car", model, got, {"oval": track}))
        assert list(rows["top"]) == [t for t in cg.PLANT_TOPS for _ in range(2)]
        assert set(rows["shared_sd"]) == {0.0, cg.PLANT_SHARED_SD}
        assert (rows["limit"] == 1.2).all()  # the car's own limit is what is planted
        # the noise is set so that the plant shows the statistic what the logs show it
        bell = rows[(rows["top"] == "bell") & (rows["shared_sd"] == 0.0)].iloc[0]
        assert bell["noise_found"] == pytest.approx(0.012, rel=0.25)
        assert bell["level"] > bell["cross"]
        assert cg.plant_rows("car", model, replace(got, one_sensor=True), {"oval": track}) == []
        assert cg.plant_rows("car", model, got, {}) == []


class TestGate:
    def test_report_on_a_synthetic_dataset(self, tmp_path, monkeypatch):
        geo._INDEX_CACHE.clear()
        sessions = {
            f"2026-01-1{i}_s{i}": {
                "car": "car",
                "date": f"2026-01-1{i}",
                "kwargs": {"n_laps": 4, "seed": i, "controls": True},
            }
            for i in range(3)
        }
        write_dataset(tmp_path, sessions)
        write_model_json(tmp_path)
        monkeypatch.setattr(smp, "MODEL_CARS", ("car",))
        # the stadium's laps never pass 50 m/s: the car needs a published drag
        monkeypatch.setitem(cm.PUBLISHED, "car", {"mass_kg": 500.0, "c_drag_prior": 3e-4})
        rep = cg.check_car_fit(tmp_path, tracks_dir=tmp_path / "tracks", cars_dir=tmp_path / "cars")
        assert rep.gate == "C2"
        names = [it.name for it in rep.items]
        for want in (
            "Power agrees between layouts",
            "`mu_y` is the car's limit; the driver's use of it per layout",
            "Recovery on simulated laps",
            "In-sample speed RMSE at each lap's own u",
        ):
            assert want in names
        saved = cm.CarModel.load(tmp_path / "cars" / "car.json")
        assert saved.ref_u and "median over layouts" in saved.meta["mu_y_source"]
        assert saved.driver == {}  # a dozen laps: too few braking nodes to read the driver
        assert list(saved.ref_u.values())[0][0] == pytest.approx(1.0, abs=0.01)
        # the car's logged range of use at the layout, which a pace table's support is
        (layout,) = saved.meta["use_range"]
        ranged = saved.meta["use_range"][layout]
        assert ranged["laps"] == 12 and ranged["days"] == 3
        assert ranged["min"] <= ranged["median"] <= ranged["max"]

        from motorsports_data_notebook.tire_model.lapsim import pace, pace_gate

        table_rep = pace_gate.check_pace_table(
            tmp_path,
            tracks_dir=tmp_path / "tracks",
            cars_dir=tmp_path / "cars",
            out_dir=tmp_path / "pace",
        )
        assert table_rep.gate == "D1T"
        named = {it.name: it for it in table_rep.items}
        # a 27 s lap is 27 one-second bins: the interpolation is a little over the gate's 1 %
        heat_item = named["Interpolated per-tyre heat against a direct solve"]
        assert float(heat_item.measured.split()[0]) < 0.02
        assert named["The use read off the table against a direct solve"].passed
        saved_table = json.loads((tmp_path / "pace" / f"car__{layout}.json").read_text())
        track = pace_gate.load_tracks(tmp_path / "tracks")[layout]
        back = pace.from_dict(saved_table, saved, track)
        assert back.support["use_min"] == ranged["min"] and "at this layout" in back.support["from"]
        empty = pace_gate.check_pace_table(
            tmp_path, tracks_dir=tmp_path / "none", cars_dir=tmp_path / "cars"
        )
        assert not empty.passed
