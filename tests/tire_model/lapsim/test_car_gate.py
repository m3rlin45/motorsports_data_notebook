"""Gate C2 through the solver (lap simulation C2)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import car as cm
from motorsports_data_notebook.tire_model.lapsim import car_gate as cg
from motorsports_data_notebook.tire_model.lapsim import geometry as geo
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
            "`mu_y` by convention, and `ref_u` per layout",
            "Recovery on simulated laps",
            "In-sample speed RMSE at each lap's own u",
        ):
            assert want in names
        saved = cm.CarModel.load(tmp_path / "cars" / "car.json")
        assert saved.ref_u and "median over layouts" in saved.meta["mu_y_source"]
        assert list(saved.ref_u.values())[0][0] == pytest.approx(1.0, abs=0.01)
