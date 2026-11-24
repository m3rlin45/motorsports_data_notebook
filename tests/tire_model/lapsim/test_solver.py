"""The lap solver (lap simulation D1, the part segment C needs)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.lapsim import _kernel
from motorsports_data_notebook.tire_model.lapsim import bank as bankmod
from motorsports_data_notebook.tire_model.lapsim import solver as sv
from motorsports_data_notebook.tire_model.lapsim.car import CarModel

G = 9.80665
CAR = sv.SolverCar(mu_y=1.2, mu_b=1.0, p_specific=100.0, c_drag=3.36e-4)
P_OVER_C = CAR.p_specific / CAR.c_drag


def _straight(length_m, ds=2.0, grade=0.0):
    n = int(round(length_m / ds)) + 1
    return sv.Profile(ds, np.zeros(n), np.full(n, grade), np.zeros(n), closed=False)


def _oval(ds=2.0, straight=400.0, radius=60.0, ramp=40.0):
    """A stadium with the curvature ramped in and out of each corner."""
    arc = math.pi * radius - ramp  # the ramps turn as much as ramp/2 of full curvature each
    piece = np.concatenate(
        [
            np.zeros(int(straight / ds)),
            np.linspace(0.0, 1.0, int(ramp / ds), endpoint=False),
            np.ones(int(round(arc / ds))),
            np.linspace(1.0, 0.0, int(ramp / ds), endpoint=False),
        ]
    )
    return sv.Profile.flat(np.concatenate([piece, piece]) / radius, ds)


class TestClosedForms:
    def test_constant_radius(self):
        car = replace(CAR, c_drag=0.0)
        sim = sv.simulate_lap(sv.Profile.flat(np.full(200, 1 / 50.0), 2.0), car)
        assert sim.v_node_ms == pytest.approx(math.sqrt(1.2 * G * 50.0), abs=1e-9)
        assert sim.lat_g == pytest.approx(1.2, abs=1e-9)
        assert sim.long_kin_g == pytest.approx(0.0, abs=1e-12)

    def test_constant_radius_with_drag_holds_just_under_the_limit(self):
        # what is left of the grip has to carry the drag
        sim = sv.simulate_lap(sv.Profile.flat(np.full(2000, 1 / 50.0), 2.0), CAR)
        v = float(sim.v_node_ms[1000])
        assert v < math.sqrt(1.2 * G * 50.0)
        room = math.sqrt((1.2 * G) ** 2 - (v * v / 50.0) ** 2)
        assert room == pytest.approx(CAR.c_drag * v * v, abs=1e-3)

    def test_power_and_drag_on_a_straight(self):
        prof = _straight(1000.0)
        sim = sv.simulate_lap(prof, CAR, v_start=20.0)
        s = np.arange(prof.n) * prof.ds_m
        want = np.cbrt(P_OVER_C + (20.0**3 - P_OVER_C) * np.exp(-3.0 * CAR.c_drag * s))
        assert np.abs(sim.v_node_ms - want).max() < 0.01

    def test_braking_with_drag(self):
        prof = _straight(600.0)
        sim = sv.simulate_lap(prof, CAR, v_start=55.0, v_end=20.0)
        s = np.arange(prof.n) * prof.ds_m
        k = CAR.mu_b * G / CAR.c_drag
        back = np.sqrt((20.0**2 + k) * np.exp(2.0 * CAR.c_drag * (600.0 - s)) - k)
        on = np.cbrt(P_OVER_C + (55.0**3 - P_OVER_C) * np.exp(-3.0 * CAR.c_drag * s))
        want = np.minimum(on, back)
        # away from the node where the two branches meet
        cross = int(np.argmin(np.abs(on - back)))
        away = np.abs(np.arange(prof.n) - cross) > 1
        assert np.abs(sim.v_node_ms - want)[away].max() < 0.01
        # the braking point: where the solved speed leaves the drive branch
        s_cross = np.interp(
            0.0, (back - on)[cross + 1 : cross - 2 : -1], s[cross + 1 : cross - 2 : -1]
        )
        leaves = s[np.flatnonzero(sim.long_kin_g < 0.0)[0]]
        assert abs(leaves - s_cross) < prof.ds_m

    def test_terminal_speed_on_a_grade(self):
        grade = 0.05
        lo, hi = 20.0, 80.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if CAR.p_specific / mid - CAR.c_drag * mid * mid - G * grade > 0:
                lo = mid
            else:
                hi = mid
        sim = sv.simulate_lap(_straight(8000.0, grade=grade), CAR, v_start=lo - 3.0)
        assert sim.v_node_ms[-1] == pytest.approx(lo, rel=1e-4)

    def test_a_crest_takes_grip_away(self):
        n = 200
        prof = sv.Profile(2.0, np.full(n, 1 / 50.0), np.zeros(n), np.full(n, -0.002))
        lim = sv.corner_speed_limit(prof, 1.2)
        assert lim == pytest.approx(math.sqrt(1.2 * G / (1 / 50.0 + 1.2 * 0.002)))
        flat = sv.corner_speed_limit(sv.Profile.flat(np.full(n, 1 / 50.0), 2.0), 1.2)
        assert (lim < flat).all()

    def test_a_downhill_helix_is_held_at_the_root_of_the_combined_rule(self):
        # a constant radius going down: the car brakes all the way round to hold its
        # speed, with what the cornering leaves of the grip, the drag helping
        n, kappa, grade = 400, 1 / 50.0, -0.08
        prof = sv.Profile(2.0, np.full(n, kappa), np.full(n, grade), np.zeros(n))
        load = G / math.sqrt(1.0 + grade**2)

        def net(v, shape):  # what is left to speed the car up, braked at the limit
            used = v * v * kappa / (CAR.mu_y * load)
            left = (1.0 - shape) * (1.0 - used) + shape * math.sqrt(1.0 - used * used)
            return -CAR.mu_b * load * left - CAR.c_drag * v * v - G * grade

        for shape in (0.0, 0.6):
            lo, hi = 1.0, math.sqrt(CAR.mu_y * load / kappa)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if net(mid, shape) < 0.0 else (lo, mid)
            assert lo < math.sqrt(CAR.mu_y * load / kappa) - 0.02  # under the cornering limit
            sim = sv.simulate_lap(prof, replace(CAR, brake_ellipse=shape), 1.0)
            assert sim.v_node_ms == pytest.approx(lo, rel=1e-4)

    def test_the_engines_work_is_the_tyres_work_at_full_throttle(self):
        sim = sv.simulate_lap(_straight(1000.0), CAR, v_start=20.0)
        tyres = float(np.sum(sim.long_tyre_g * G * sim.v_ms * sim.dt_s))
        engine = CAR.p_specific * float(sim.dt_s.sum())
        assert tyres == pytest.approx(engine, rel=1e-4)
        # and it went into speed and drag
        gained = 0.5 * (sim.v_node_ms[-1] ** 2 - sim.v_node_ms[0] ** 2)
        drag = float(np.sum(CAR.c_drag * sim.v_ms**3 * sim.dt_s))
        assert gained + drag == pytest.approx(engine, rel=1e-4)

    def test_a_straight_has_no_cornering_limit(self):
        assert sv.corner_speed_limit(sv.Profile.flat(np.zeros(5), 2.0), 1.2) == pytest.approx(
            sv.V_MAX_MS
        )


#: The outside reference of #206's review: two stadium ovals with arcs of
#: radius 160/π m and 400 m straights, one with a curvature step (1,120 m)
#: and one with 40 m clothoid ramps (1,200 m), lapped by a point mass with
#: μ_y 1.2, μ_x 1.0, 100 W/kg and c = 3.36e-4 /m at g = 9.81. The lap
#: times are an ODE reference's and a closed form's, not this solver's.
REF_RADIUS = 160.0 / math.pi
#: On a flat road g enters only with the grip: the reference's g against
#: the solver's is a factor on μ.
REF_G = 9.81 / G
REF_CAR = sv.SolverCar(mu_y=1.2 * REF_G, mu_b=1.0 * REF_G, p_specific=100.0, c_drag=3.36e-4)
#: "Ellipse both sides": braking and drive each within the ellipse with
#: μ_x along the car: the braking limit's shape at the ellipse, and the
#: drive at μ_x / μ_y of what the cornering leaves of the grip.
REF_ELLIPSE = replace(REF_CAR, brake_ellipse=1.0, drive_share=1.0 / 1.2)
#: "Diamond on braking only": the drive held by the engine alone.
REF_POWER_ONLY = replace(REF_CAR, drive_share=1.0e6)


def _reference_oval(ds, ramp):
    return _oval(ds, straight=400.0, radius=REF_RADIUS, ramp=ramp)


class TestReferenceOvals:
    """Gate D1's outside reference (issue #206)."""

    def test_the_ovals_are_the_reviews(self):
        assert _reference_oval(2.0, 0.0).length_m == pytest.approx(1120.0)
        clothoid = _reference_oval(2.0, 40.0)
        assert clothoid.length_m == pytest.approx(1200.0)
        assert np.sum(clothoid.kappa) * 2.0 == pytest.approx(2.0 * math.pi)  # it closes

    def test_ellipse_both_sides_on_the_clothoid_oval(self):
        # 35.7254 s; +0.032 % at 2 m and +0.0009 % at 0.25 m
        assert sv.lap_time(_reference_oval(2.0, 40.0), REF_ELLIPSE, 1.0) == pytest.approx(
            35.7254, rel=4e-4
        )
        assert sv.lap_time(_reference_oval(0.25, 40.0), REF_ELLIPSE, 1.0) == pytest.approx(
            35.7254, rel=5e-5
        )

    def test_ellipse_on_the_stepped_oval(self):
        # 35.1056 s; −0.004 % at 2 m
        assert sv.lap_time(_reference_oval(2.0, 0.0), REF_ELLIPSE, 1.0) == pytest.approx(
            35.1056, rel=1e-4
        )
        assert sv.lap_time(_reference_oval(0.25, 0.0), REF_ELLIPSE, 1.0) == pytest.approx(
            35.1056, rel=5e-5
        )

    def test_the_error_falls_with_the_step(self):
        off = [
            abs(sv.lap_time(_reference_oval(ds, 40.0), REF_ELLIPSE, 1.0) / 35.7254 - 1.0)
            for ds in (2.0, 1.0, 0.5)
        ]
        assert off[1] < 0.5 * off[0] and off[2] < 0.5 * off[1]

    def test_braking_on_the_diamond_with_the_drive_by_power_alone(self):
        # what the review's "the rule above" figures were computed with: 36.0337 s on
        # the clothoid oval and 35.1036 s on the stepped one
        assert sv.lap_time(_reference_oval(0.25, 40.0), REF_POWER_ONLY, 1.0) == pytest.approx(
            36.0337, rel=2e-4
        )
        prof = _reference_oval(0.25, 0.0)
        sim = sv.simulate_lap(prof, REF_POWER_ONLY, 1.0)
        assert sim.lap_time_s == pytest.approx(35.1036, rel=2e-4)
        # the straight after the arc, by the closed form with drag: speed 100, 200 and
        # 300 m on, the peak and where the braking begins
        v = sim.v_node_ms[: int(400.0 / 0.25)]
        at = [float(v[int(s / 0.25)]) for s in (100.0, 200.0, 300.0)]
        assert at == pytest.approx([34.7066, 40.4806, 44.5657], abs=0.03)
        assert float(v.max()) == pytest.approx(45.528, abs=0.01)
        assert 0.25 * int(np.argmax(v)) == pytest.approx(328.184, abs=0.5)

    def test_the_rule_as_written_holds_the_drive_within_the_grip(self):
        # the issue's rule (braking on the diamond, drive within the circle of the
        # grip) has no reference of its own: the review's 36.0337 s is with the drive
        # by power alone. On the stepped oval the two are the same lap; on the
        # clothoid one the grip holds the drive on the way out of each corner
        assert sv.lap_time(_reference_oval(2.0, 0.0), REF_CAR, 1.0) == pytest.approx(
            35.1036, rel=1e-4
        )
        written = sv.lap_time(_reference_oval(0.25, 40.0), REF_CAR, 1.0)
        assert 36.0337 * 1.0005 < written < 36.5213  # under "diamond both sides"
        assert written == pytest.approx(36.0787, abs=0.002)  # the solver's own figure

    def test_an_outside_solver_on_the_same_ovals(self):
        # TUM's trajectory_planning_helpers; skipped where it is absent or does not import
        tph = pytest.importorskip("trajectory_planning_helpers")
        g = 9.81
        mass = 1000.0
        for ramp in (0.0, 40.0):
            prof = _reference_oval(1.0, ramp)
            ggv = np.array([[0.0, 1.0 * g, 1.2 * g], [100.0, 1.0 * g, 1.2 * g]])
            engine = np.array([[v, 100.0 / max(v, 1.0)] for v in np.arange(0.0, 101.0, 0.5)])
            vx = tph.calc_vel_profile.calc_vel_profile(
                ggv=ggv,
                kappa=prof.kappa,
                el_lengths=np.full(prof.n, 1.0),
                closed=True,
                filt_window=None,
                dyn_model_exp=2.0,
                drag_coeff=3.36e-4 * mass,
                m_veh=mass,
                ax_max_machines=engine,
                v_max=100.0,
            )
            theirs = float(np.sum(1.0 / (0.5 * (vx + np.roll(vx, -1)))))
            assert sv.lap_time(prof, REF_ELLIPSE, 1.0) == pytest.approx(theirs, rel=1e-3)


class TestClosedLap:
    def test_distance_time_and_the_closed_lap_identity(self):
        prof = _oval()
        sim = sv.simulate_lap(prof, CAR)
        assert float(np.sum(sim.v_ms * sim.dt_s)) == pytest.approx(prof.length_m, rel=1e-12)
        assert sim.t_s[-1] + sim.dt_s[-1] == pytest.approx(sim.lap_time_s)
        work = float(np.sum(sim.v_ms * sim.long_kin_g * sim.dt_s))
        assert abs(work) < 1e-9 * prof.length_m
        gain = float(np.sum(np.maximum(sim.long_kin_g, 0) * sim.v_ms * sim.dt_s))
        loss = float(np.sum(np.maximum(-sim.long_kin_g, 0) * sim.v_ms * sim.dt_s))
        assert gain / loss == pytest.approx(1.0, abs=1e-6)

    def test_it_brakes_accelerates_and_corners(self):
        sim = sv.simulate_lap(_oval(), CAR)
        assert sim.long_kin_g.min() < -0.9 and sim.long_kin_g.max() > 0.2
        assert sim.lat_g.max() == pytest.approx(1.2, abs=0.01) and sim.lat_g.min() >= 0.0
        assert 30.0 < sim.lap_time_s < 45.0

    def test_no_step_outside_the_envelope(self):
        prof = _oval()
        sim = sv.simulate_lap(prof, CAR)
        assert sv.combined_use(sim, prof, CAR).max() <= 1.005

    def test_the_start_node_does_not_matter(self):
        prof = _oval()
        base = sv.simulate_lap(prof, CAR)
        moved = sv.simulate_lap(prof.rolled(137), CAR)
        assert moved.lap_time_s == pytest.approx(base.lap_time_s, rel=1e-9)
        assert np.roll(moved.v_node_ms, 137) == pytest.approx(base.v_node_ms, abs=1e-6)

    def test_the_mirrored_track_mirrors_the_lateral(self):
        prof = _oval()
        right, left = sv.simulate_lap(prof, CAR), sv.simulate_lap(prof.mirrored(), CAR)
        assert left.lap_time_s == pytest.approx(right.lap_time_s, rel=1e-12)
        assert left.lat_g == pytest.approx(-right.lat_g)

    def test_halving_the_step(self):
        prof = _oval()
        coarse = sv.simulate_lap(prof, CAR).lap_time_s
        fine = sv.simulate_lap(prof.refined(2), CAR).lap_time_s
        assert fine == pytest.approx(coarse, rel=5e-4)

    def test_grade_and_the_three_longitudinal_series(self):
        prof = _oval()
        n = prof.n
        hill = replace(prof, grade=0.03 * np.sin(2 * math.pi * np.arange(n) / n))
        sim = sv.simulate_lap(hill, CAR)
        grade = 0.5 * (hill.grade + np.roll(hill.grade, -1))
        assert sim.long_sf_g - sim.long_kin_g == pytest.approx(grade)
        assert (sim.long_tyre_g > sim.long_sf_g).all()
        assert sim.lap_time_s != pytest.approx(sv.simulate_lap(prof, CAR).lap_time_s, rel=1e-4)


class TestKnob:
    def test_lap_time_falls_as_u_rises(self):
        prof = _oval()
        times = [sv.lap_time(prof, CAR, u) for u in np.arange(0.5, 1.5001, 0.05)]
        assert (np.diff(times) < 0).all()

    def test_the_knob_scales_grip_and_leaves_the_engine(self):
        prof = _oval()
        slow = sv.simulate_lap(prof, CAR, 0.8)
        assert slow.lat_g.max() == pytest.approx(0.96, abs=0.01)
        assert slow.long_kin_g.min() > -0.9  # braking falls with it
        # the same top-end acceleration where the engine, not the tyre, limits
        full = sv.simulate_lap(prof, CAR, 1.0)
        i = int(np.argmax(full.v_ms))
        assert slow.long_kin_g[i - 20] * slow.v_ms[i - 20] == pytest.approx(
            full.long_kin_g[i - 20] * full.v_ms[i - 20], rel=0.05
        )

    def test_wet_is_the_same_knob(self):
        prof = _oval()
        car = replace(CAR, wet_scale=0.8)
        assert sv.lap_time(prof, car, 1.0, wet=True) == pytest.approx(sv.lap_time(prof, CAR, 0.8))
        assert sv.lap_time(prof, car, 1.0) == pytest.approx(sv.lap_time(prof, CAR, 1.0))

    def test_solve_u_round_trip(self):
        prof = _oval()
        u, status = sv.solve_u(prof, CAR, sv.lap_time(prof, CAR, 0.9))
        assert status == "ok" and u == pytest.approx(0.9, abs=1e-6)

    def test_a_lap_time_the_car_cannot_do(self):
        prof = _oval()
        u, status = sv.solve_u(prof, CAR, sv.lap_time(prof, CAR, 1.0) / 3.0)
        assert status == "no_root" and math.isnan(u)

    def test_cornering_drag_is_a_variant(self):
        prof = _oval()
        assert sv.lap_time(prof, replace(CAR, k_corner_drag=0.05), 1.0) > sv.lap_time(
            prof, CAR, 1.0
        )


class TestInputs:
    def test_car_from_a_model(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "n_brake": 1.0},
            power={"p_specific": 70.0, "c_drag": 2.4e-4, "k_corner_drag": 0.04},
            wet={"grip_scale": 0.8},
        )
        car = sv.SolverCar.from_model(model)
        assert (car.mu_y, car.mu_b, car.p_specific, car.c_drag) == (1.3, 0.9, 70.0, 2.4e-4)
        assert car.wet_scale == 0.8
        # the cost of cornering at full throttle is carried unless it is switched off
        assert sv.CORNER_DRAG_DEFAULT and car.k_corner_drag == 0.04
        assert sv.SolverCar.from_model(model, corner_drag=False).k_corner_drag == 0.0
        assert sv.SolverCar.from_model(model, corner_drag=True).k_corner_drag == 0.04

    def test_refined_profile_keeps_the_track(self):
        prof = _oval()
        fine = prof.refined(2)
        assert fine.n == 2 * prof.n and fine.length_m == pytest.approx(prof.length_m)
        assert fine.kappa[::2] == pytest.approx(prof.kappa)


GEARED = replace(
    CAR,
    curve_rpm=(5000.0, 6000.0, 7000.0),
    curve_p=(80.0, 100.0, 100.0),
    gears=(5.0, 6.5, 8.0),
    gear_numbers=(3, 4, 5),
    shift_rpm=7000.0,
)


SHIFTING = replace(GEARED, shift_s=(0.4, 0.3, 0.0))


class TestBrakeOnset:
    """The time between drive and braking: a flat top on every speed peak."""

    def _open(self, v, tau, closed=True, ds=2.0):
        out = np.empty(len(v))
        _kernel.open_peaks(np.asarray(v, float), ds, tau, closed, out)
        return out

    def test_a_peak_is_cut_flat_where_it_is_that_long(self):
        # 20 m/s rising 0.1 m/s a node to 40 and falling 0.5 a node: a sharp peak
        v = np.concatenate([np.linspace(20.0, 40.0, 201), np.linspace(39.5, 20.0, 40)])
        got = self._open(v, 0.5, closed=False)
        top = got.max()
        flat = np.flatnonzero(got == top)
        assert 36.0 < top < 40.0 and len(flat) > 3
        # the flat top is as long as the car takes to cover it at that speed
        assert (flat[-1] - flat[0]) * 2.0 / top == pytest.approx(0.5, abs=2.0 / top)
        assert flat[0] < 195 and 199 <= flat[-1] <= 203  # it leaves the drive curve early
        away = np.r_[np.arange(0, flat[0] - 1), np.arange(flat[-1] + 2, len(v))]
        assert np.array_equal(got[away], v[away])  # both curves are otherwise as they were
        assert np.all(got <= v)

    def test_minima_and_tops_already_that_flat_are_left(self):
        dip = np.concatenate([np.linspace(40.0, 20.0, 60), np.linspace(20.0, 40.0, 60)])
        assert np.array_equal(self._open(dip, 0.4, closed=False), dip)
        table = np.concatenate(
            [np.linspace(20.0, 40.0, 50), np.full(40, 40.0), np.linspace(40, 20, 50)]
        )
        assert np.array_equal(self._open(table, 0.4, closed=False), table)
        assert np.array_equal(self._open(table, 0.0, closed=False), table)

    def test_it_moves_smoothly_with_the_time_and_round_the_lap(self):
        v = 30.0 + 8.0 * np.abs(np.sin(np.linspace(0.0, 4.0 * np.pi, 400, endpoint=False)))
        tops = [self._open(v, tau).max() for tau in np.linspace(0.0, 1.0, 101)]
        assert np.all(np.diff(tops) <= 1e-12) and np.max(np.abs(np.diff(tops))) < 0.05
        # a closed lap has no end: turned round, the answer turns with it
        assert np.array_equal(np.roll(self._open(np.roll(v, 137), 0.4), -137), self._open(v, 0.4))

    def test_the_lap_stops_gaining_speed_earlier_and_takes_longer(self):
        prof = _oval(straight=500.0)
        plain = sv.simulate_lap(prof, CAR)
        slow = sv.simulate_lap(prof, replace(CAR, onset_s=0.4))
        assert slow.lap_time_s > plain.lap_time_s and slow.lap_time_s < plain.lap_time_s + 0.4
        a, b = int(np.argmax(plain.v_node_ms[:260])), int(np.argmax(slow.v_node_ms[:260]))
        assert 3 <= a - b <= 12  # the speed peak comes 6 to 24 m earlier
        assert slow.v_node_ms.min() == plain.v_node_ms.min()  # the apexes are the same
        assert np.all(slow.v_node_ms <= plain.v_node_ms)
        times = sv.drive_laps(
            prof, replace(CAR, onset_s=0.4), np.linspace(0.6, 1.5, 181)
        ).lap_time_s
        assert np.all(np.diff(times) < 0)  # and the lap time still falls as u rises

    def test_from_a_model_and_the_days_own(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "onset_s": 0.42},
            power={"c_drag": 2.0e-4, "p_specific": 70.0},
            wet={"grip_scale": None},
        )
        default = sv.SolverCar.from_model(model).onset_s
        assert default == (0.42 if sv.BRAKE_ONSET_DEFAULT else 0.0)
        assert sv.SolverCar.from_model(model, brake_onset=True).onset_s == 0.42
        assert sv.SolverCar.from_model(model, brake_onset=True, onset_s=0.3).onset_s == 0.3
        assert (
            sv.SolverCar.from_model(model, brake_onset=True, onset_s=float("nan")).onset_s == 0.42
        )
        assert sv.SolverCar.from_model(model, brake_onset=False, onset_s=0.3).onset_s == 0.0
        model.grip["onset_s"] = None
        assert sv.SolverCar.from_model(model, brake_onset=True).onset_s == 0.0


class TestBrakingShape:
    """The braking limit between the diamond and the ellipse."""

    def test_in_a_straight_line_the_shape_changes_nothing(self):
        prof = _straight(600.0)
        for share in (0.5, 1.0):
            a = sv.simulate_lap(prof, CAR, v_start=55.0, v_end=20.0)
            b = sv.simulate_lap(prof, replace(CAR, brake_ellipse=share), v_start=55.0, v_end=20.0)
            assert np.array_equal(a.v_node_ms, b.v_node_ms)

    def test_into_a_corner_the_car_brakes_later_and_no_step_is_over_its_limit(self):
        prof = _oval(straight=500.0)
        laps = {s: sv.simulate_lap(prof, replace(CAR, brake_ellipse=s)) for s in (0.0, 0.5, 1.0)}
        times = [laps[s].lap_time_s for s in (0.0, 0.5, 1.0)]
        assert times[0] > times[1] > times[2]  # more of the grip is used on the way in
        tops = [float(np.argmax(laps[s].v_node_ms[:260])) for s in (0.0, 0.5, 1.0)]
        assert tops[0] < tops[1] < tops[2]  # the braking point moves down the straight
        for share, sim in laps.items():
            use = sv.combined_use(sim, prof, replace(CAR, brake_ellipse=share))
            assert use.max() <= 1.005
        # and the lap under the diamond is inside the ellipse with room to spare
        assert sv.combined_use(laps[0.0], prof, replace(CAR, brake_ellipse=1.0)).max() <= 1.005
        assert sv.combined_use(laps[1.0], prof, CAR).max() > 1.2

    def test_the_lap_time_still_falls_as_u_rises(self):
        prof = _oval(straight=500.0)
        times = sv.drive_laps(prof, replace(CAR, brake_ellipse=0.7), np.linspace(0.6, 1.5, 91))
        assert np.all(np.diff(times.lap_time_s) < 0)

    def test_from_a_model_unless_switched_off(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "n_brake": 1.0, "brake_ellipse": 0.7},
            power={"c_drag": 2.0e-4, "p_specific": 70.0},
            wet={"grip_scale": None},
        )
        assert sv.BRAKE_SHAPE_DEFAULT and sv.SolverCar.from_model(model).brake_ellipse == 0.7
        assert sv.SolverCar.from_model(model, brake_shape=False).brake_ellipse == 0.0
        model.grip["brake_ellipse"] = None  # a car whose zones do not say: the diamond
        assert sv.SolverCar.from_model(model).brake_ellipse == 0.0


class TestBrakingUse:
    """The level a lap is braked at, as a share of the car's limit: the
    driver's, where the limit is the car's."""

    USED = replace(CAR, brake_use=0.8, brake_ellipse=0.5)

    def test_a_share_of_the_limit_is_that_limit(self):
        prof = _oval(straight=500.0)
        used = sv.simulate_lap(prof, self.USED)
        lower = sv.simulate_lap(prof, replace(CAR, mu_b=CAR.mu_b * 0.8, brake_ellipse=0.5))
        assert np.array_equal(used.v_node_ms, lower.v_node_ms)
        at_limit = sv.simulate_lap(prof, replace(CAR, brake_ellipse=0.5))
        assert used.lap_time_s > at_limit.lap_time_s + 0.05
        # the car that brakes less begins sooner
        assert np.argmax(used.v_node_ms[:260]) < np.argmax(at_limit.v_node_ms[:260])
        # 1 is the level the lap is braked at, and the car's limit is not reached
        assert sv.combined_use(used, prof, self.USED).max() <= 1.005
        braking = used.long_tyre_g < -0.3
        assert sv.combined_use(used, prof, replace(self.USED, brake_use=1.0))[braking].max() < 0.85

    def test_the_knob_multiplies_it_and_the_lap_time_still_falls(self):
        prof = _oval(straight=500.0)
        times = sv.drive_laps(prof, self.USED, np.linspace(0.6, 1.5, 91))
        assert np.all(np.diff(times.lap_time_s) < 0)
        same = sv.drive_laps(prof, replace(CAR, mu_b=0.8, brake_ellipse=0.5), [0.9, 1.1])
        assert np.array_equal(sv.drive_laps(prof, self.USED, [0.9, 1.1]).v_node_ms, same.v_node_ms)

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    def test_the_two_forms_agree_to_the_last_bit(self, monkeypatch):
        prof = _oval(straight=500.0)
        times = [sv.lap_time(prof, self.USED, u) for u in (0.95, 1.0, 1.05)]
        got = {}
        for compiled in (True, False):
            monkeypatch.setattr(sv, "USE_COMPILED", compiled)
            got[compiled] = sv.solve_laps(prof, self.USED, times)
        assert np.array_equal(got[True].u, got[False].u)
        assert np.array_equal(got[True].v_node_ms, got[False].v_node_ms)

    def test_from_a_models_driver_unless_switched_off(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "n_brake": 1.0, "brake_ellipse": 0.7},
            power={"c_drag": 2.0e-4, "p_specific": 70.0},
            wet={"grip_scale": None},
            driver={"brake_use": 0.8, "brake_use_ellipse": 0.4},
        )
        car = sv.SolverCar.from_model(model)
        # the limit stays the car's; the lap is braked at the driver's level and shape
        assert (car.mu_b, car.brake_use, car.brake_ellipse) == (0.9, 0.8, 0.4)
        limit = sv.SolverCar.from_model(model, brake_use=False)
        assert (limit.mu_b, limit.brake_use, limit.brake_ellipse) == (0.9, 1.0, 0.7)
        diamond = sv.SolverCar.from_model(model, brake_shape=False)
        assert (diamond.brake_use, diamond.brake_ellipse) == (0.8, 0.0)
        assert CarModel.from_json(model.to_json()).driver == model.driver
        model.driver = {}  # a car whose driver was not read: braked at the limit
        assert sv.SolverCar.from_model(model).brake_use == 1.0
        assert sv.SolverCar.from_model(model).brake_ellipse == 0.7
        old = {k: v for k, v in model.to_json().items() if k != "driver"}
        assert CarModel.from_json(old).driver == {}  # a JSON from before the entry


class TestShiftTime:
    """The time an upshift takes, taken as a share of the power over a
    band of speed above the shift."""

    def test_a_band_above_each_upshift_and_none_above_the_top_gear(self):
        bands = SHIFTING.shift_bands()
        assert [b[0] for b in bands] == pytest.approx([35.0, 45.5])
        eta = sv.SHIFT_BAND_SHARE
        for (v_s, dv), lost, ratio in zip(bands, (0.4, 0.3), (6.5, 8.0)):
            p = float(np.interp(v_s / ratio * 1000.0, GEARED.curve_rpm, GEARED.curve_p))
            assert dv == pytest.approx(lost * ((1 - eta) * p / v_s - CAR.c_drag * v_s**2) / eta)
        v = np.array([34.9, 35.2, 35.0 + bands[0][1] + 0.1, 45.6, 60.0])
        ratio = SHIFTING.power_at(v) / GEARED.power_at(v)
        assert ratio == pytest.approx([1.0, 1 - eta, 1.0, 1 - eta, 1.0])
        assert GEARED.shift_bands() == [] and CAR.shift_bands() == []
        assert replace(GEARED, shift_s=(0.0, 0.0, 0.0)).shift_bands() == [(35.0, 0.0), (45.5, 0.0)]

    def test_a_band_stops_at_the_next_shift_and_where_the_car_cannot_pull(self):
        slow = replace(GEARED, shift_s=(60.0, 0.3, 0.0)).shift_bands()
        assert slow[0][1] == pytest.approx(45.5 - 35.0)
        dragging = replace(GEARED, c_drag=3e-3, shift_s=(0.4, 0.3, 0.0)).shift_bands()
        assert dragging[1][1] == 0.0  # 75 W/kg cannot overcome that drag at 45 m/s

    def test_the_table_holds_the_band_at_its_width(self):
        grid = np.arange(0.0, sv.V_MAX_MS + 10.0, sv.POWER_TABLE_DV)
        for shift_rpm in (7000.0, 7013.0, 6951.0):  # the edges fall anywhere between points
            car = replace(SHIFTING, shift_rpm=shift_rpm)
            # the table's last entry is the driven axle's share, not power
            plain = sv._power_table(replace(car, shift_s=()))[:-1]
            share = 1.0 - sv._power_table(car)[1:-1] / plain[1:]
            width = np.sum(share) * sv.POWER_TABLE_DV / sv.SHIFT_BAND_SHARE
            assert width == pytest.approx(sum(dv for _, dv in car.shift_bands()), rel=1e-9)
            assert share.max() <= sv.SHIFT_BAND_SHARE + 1e-12 and grid.size == plain.size

    def test_on_the_flat_the_car_is_later_by_the_shift_time(self):
        # no drag and a flat curve: the engine's work is all there is
        car = replace(
            SHIFTING, c_drag=0.0, mu_y=5.0, curve_p=(100.0, 100.0, 100.0), shift_s=(0.4, 0.0, 0.0)
        )
        prof = _straight(1200.0)
        late = sv.simulate_lap(prof, car, v_start=25.0)
        soon = sv.simulate_lap(prof, replace(car, shift_s=()), v_start=25.0)

        def time_at(sim, speed):
            return float(np.interp(speed, sim.v_node_ms[:-1], sim.t_s))

        (v_s, dv), _ = car.shift_bands()
        assert time_at(late, 34.0) == pytest.approx(time_at(soon, 34.0), abs=1e-6)
        lost = time_at(late, 42.0) - time_at(soon, 42.0)
        assert lost == pytest.approx(0.4 * (1 + dv / (2 * v_s)), rel=0.03)
        # and it is the engine's work that is short: P · shift_s, and a little
        # more because the car is longer in the band at its upper end
        t = 0.9 * late.t_s[-1]
        e_late = np.interp(t, late.t_s, late.v_node_ms[:-1]) ** 2 / 2
        e_soon = np.interp(t, soon.t_s, soon.v_node_ms[:-1]) ** 2 / 2
        assert e_soon - e_late == pytest.approx(100.0 * 0.4 * (1 + dv / (2 * v_s)), rel=0.03)

    def test_the_lap_time_still_falls_as_u_rises(self):
        prof = _oval(straight=700.0)
        us = np.linspace(0.6, 1.5, 181)
        times = sv.drive_laps(prof, SHIFTING, us).lap_time_s
        assert np.all(np.diff(times) < 0)
        assert times[90] > sv.lap_time(prof, GEARED, float(us[90]))  # and shifting costs time

    def test_from_a_model_by_gear_number_by_order_or_by_ratio(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "n_brake": 1.0},
            power={
                "c_drag": 2.0e-4,
                "default_engine": "e",
                "p_specific": 70.0,
                "engines": {
                    "e": {
                        "p_specific": 70.0,
                        "curve": [[5000.0, 60.0], [6000.0, 70.0]],
                        "gears": {"2": 3.5, "3": 5.0, "4": 6.5},
                        "shift_rpm": 7000.0,
                        "shift_s": {"2": 0.3, "3": 0.4},
                    }
                },
            },
            wet={"grip_scale": None},
        )
        assert sv.SolverCar.from_model(model, shift_loss=False).shift_s == ()
        assert sv.SHIFT_LOSS_DEFAULT
        on = sv.SolverCar.from_model(model)
        assert on.shift_s == (0.3, 0.4, 0.0)  # nothing out of the top gear
        # the day's own set: as long as the powertrain's, in order
        day = sv.SolverCar.from_model(model, shift_loss=True, gears=[3.6, 5.2, 6.8])
        assert day.shift_s == (0.3, 0.4, 0.0) and day.gears == (3.6, 5.2, 6.8)
        # a shorter one by the nearest ratio; a gear without a time takes the median
        two = sv.SolverCar.from_model(model, shift_loss=True, gears=[5.05, 6.5])
        assert two.shift_s == (0.4, 0.0)
        odd = sv.SolverCar.from_model(model, shift_loss=True, gears=[4.2, 6.5])
        assert odd.shift_s == (0.35, 0.0)
        model.power["engines"]["e"]["shift_s"] = None
        assert sv.SolverCar.from_model(model, shift_loss=True).shift_s == ()


class TestDrivenAxle:
    """Drive out of a corner goes through the driven axle (#312)."""

    @pytest.fixture(params=[True, False], ids=["compiled", "numpy"])
    def form(self, request, monkeypatch):
        if request.param and not sv._kernel.HAVE_NUMBA:
            pytest.skip("numba is not installed")
        monkeypatch.setattr(sv, "USE_COMPILED", request.param)
        return request.param

    def test_a_share_of_one_is_the_lap_as_it_was(self, form):
        prof = _oval()
        whole = sv.drive_laps(prof, replace(CAR, drive_share=1.0), np.array([0.9, 1.0]))
        plain = sv.drive_laps(prof, CAR, np.array([0.9, 1.0]))
        assert np.array_equal(whole.v_node_ms, plain.v_node_ms)
        assert sv._power_table(CAR)[-1] == 1.0

    def test_the_rear_axle_alone_cannot_take_all_the_engine_gives(self):
        # more power than half the grip can put down: on the straights the car pulls
        # at the driven axle's share of the grip, not at what the engine gives
        car = replace(CAR, p_specific=400.0, c_drag=0.0, drive_share=0.5)
        prof = _oval()
        sim = sv.simulate_lap(prof, car, 1.0)
        pulling = (np.abs(sim.lat_g) < 1e-9) & (sim.long_tyre_g > 0.0)
        assert pulling.sum() > 50
        assert sim.long_tyre_g[pulling] == pytest.approx(0.5 * car.mu_y, rel=1e-6)
        use = sv.combined_use(sim, prof, car)
        assert use[pulling] == pytest.approx(1.0, rel=1e-6)  # at its limit, by the axle's share
        whole = sv.simulate_lap(prof, replace(car, drive_share=1.0), 1.0)
        assert whole.long_tyre_g.max() > 1.1  # the whole car's grip takes it

    def test_out_of_a_corner_the_car_is_slower(self, form):
        prof = _oval()
        car = replace(CAR, p_specific=200.0)
        whole = sv.simulate_lap(prof, car, 1.0)
        rear = sv.simulate_lap(prof, replace(car, drive_share=0.5), 1.0)
        assert rear.lap_time_s > whole.lap_time_s
        # the corner is the same but for the drag, which the rear alone now carries
        assert rear.v_node_ms.min() == pytest.approx(whole.v_node_ms.min(), rel=1e-3)
        assert (rear.v_node_ms <= whole.v_node_ms + 1e-9).all()
        assert (sv.combined_use(rear, prof, replace(car, drive_share=0.5)) <= 1.005).all()

    def test_the_two_forms_agree_to_the_last_bit(self, monkeypatch):
        if not sv._kernel.HAVE_NUMBA:
            pytest.skip("numba is not installed")
        prof = _oval()
        cars = [replace(CAR, p_specific=200.0, drive_share=r) for r in (0.47, 0.58, 1.0)]
        times = [sv.lap_time(prof, c, 0.95) for c in cars]
        out = {}
        for compiled in (True, False):
            monkeypatch.setattr(sv, "USE_COMPILED", compiled)
            out[compiled] = sv.solve_laps(prof, cars, times)
        assert np.array_equal(out[True].v_node_ms, out[False].v_node_ms)
        assert np.array_equal(out[True].u, out[False].u)

    def test_the_lap_time_still_falls_as_u_rises(self):
        prof = _oval()
        car = replace(CAR, p_specific=200.0, drive_share=0.5)
        times = [sv.lap_time(prof, car, u) for u in np.arange(0.5, 1.5001, 0.05)]
        assert (np.diff(times) < 0).all()

    def test_from_a_model_unless_switched_off(self):
        model = CarModel(
            car="car",
            raw_car="car",
            mass_kg=None,
            grip={"mu_y": 1.2, "mu_b": 1.0, "drive_share": 0.47},
            power={"p_specific": 100.0, "c_drag": 3e-4},
        )
        assert sv.SolverCar.from_model(model).drive_share == 0.47
        assert sv.SolverCar.from_model(model, driven_axle=False).drive_share == 1.0
        del model.grip["drive_share"]
        assert sv.SolverCar.from_model(model).drive_share == 1.0


class TestGears:
    def test_the_lowest_gear_under_the_shift_point(self):
        gear, rpm = GEARED.gear_and_rpm(np.array([30.0, 35.0, 36.0, 50.0, 70.0]))
        assert list(gear) == [0, 0, 1, 2, 2]  # third to 35 m/s, fourth to 45.5, then top
        assert rpm == pytest.approx([6000.0, 7000.0, 36 / 6.5 * 1000, 6250.0, 8750.0])

    def test_power_follows_the_curve_in_the_gear(self):
        p = GEARED.power_at(np.array([25.0, 27.5, 30.0, 36.0, 60.0, 20.0]))
        assert p[0] == pytest.approx(80.0)  # 5,000 rpm in third
        assert p[1] == pytest.approx(90.0)
        assert p[2] == pytest.approx(100.0)
        assert p[3] == pytest.approx(80.0 + 20.0 * (36 / 6.5 * 1000 - 5000) / 1000)
        assert p[4] == pytest.approx(100.0)  # past the curve in top: held
        assert p[5] == pytest.approx(80.0 * 4000 / 5000)  # under it: the torque is held

    def test_without_gears_power_is_one_number(self):
        assert not CAR.geared
        assert CAR.power_at(np.array([10.0, 60.0])) == pytest.approx(100.0)
        assert replace(CAR, power_scale=0.9).power_at(30.0) == pytest.approx(90.0)

    def test_holding_a_gear_longer_is_faster_where_the_curve_is_flat(self):
        prof = _oval(straight=900.0)
        early = sv.lap_time(prof, replace(GEARED, shift_rpm=6200.0), 1.0)
        late = sv.lap_time(prof, GEARED, 1.0)
        assert late < early
        sim = sv.simulate_lap(prof, GEARED, 1.0)
        assert float(np.sum(sim.v_ms * sim.dt_s)) == pytest.approx(prof.length_m, rel=1e-12)
        assert sv.combined_use(sim, prof, GEARED).max() <= 1.005

    def test_car_from_a_model_with_powertrains(self):
        model = CarModel(
            car="c",
            raw_car="c",
            mass_kg=None,
            grip={"mu_y": 1.3, "mu_b": 0.9, "n_brake": 1.0},
            power={
                "c_drag": 2.0e-4,
                "rho_ref": 1.157,
                "k_corner_drag": 0.04,
                "default_engine": "new",
                "p_specific": 80.0,
                "engines": {
                    "old": {
                        "p_specific": 70.0,
                        "curve": [[5000.0, 60.0], [6000.0, 70.0]],
                        "gears": {"3": 5.0, "4": 6.5},
                        "shift_rpm": 7000.0,
                    },
                    "new": {"p_specific": 80.0, "curve": None, "gears": None, "shift_rpm": None},
                },
            },
            wet={"grip_scale": None},
        )
        new = sv.SolverCar.from_model(model)
        assert not new.geared and new.p_specific == 80.0
        old = sv.SolverCar.from_model(model, engine="old", shift_rpm=7300.0, cf=1.05, rho=1.25)
        assert old.geared and old.shift_rpm == 7300.0 and old.gear_numbers == (3, 4)
        assert old.curve_rpm == (5125.0, 6125.0)  # the bins' centres
        assert old.power_scale == pytest.approx(1 / 1.05)
        assert old.c_drag == pytest.approx(2.0e-4 * 1.25 / 1.157)
        assert sv.SolverCar.from_model(model, engine="old").shift_rpm == 7000.0
        # the gear set fitted that day replaces the powertrain's own
        day = sv.SolverCar.from_model(model, engine="old", gears=[6.0, 4.6, 7.4])
        assert day.gears == (4.6, 6.0, 7.4) and day.gear_numbers == (1, 2, 3)
        # a gear made from speed over rpm gives the gear set, not the drive
        model.power["engines"]["old"]["gears_inferred"] = True
        level = sv.SolverCar.from_model(model, engine="old", gears=[6.0, 4.6, 7.4])
        assert not level.geared and level.p_specific == 70.0
        assert sv.SolverCar.from_model(model, engine="old", through_gears=True).geared
        model.power["engines"]["old"]["gears_inferred"] = False
        assert not sv.SolverCar.from_model(model, engine="old", through_gears=False).geared


def _hilly(prof):
    n = prof.n
    x = 2 * math.pi * np.arange(n) / n
    return replace(prof, grade=0.04 * np.sin(x), kappa_v=0.0006 * np.cos(3 * x))


def _rough(prof):
    """A profile with the texture of a measured one."""
    rng = np.random.default_rng(3)
    return replace(_hilly(prof), kappa=prof.kappa + rng.normal(0.0, 4e-4, prof.n))


PARITY_CASES = {
    "oval": (lambda: _oval(), CAR, 1.0, False, None, None),
    "slow lap": (lambda: _oval(), CAR, 0.7, False, None, None),
    "fast lap": (lambda: _oval(), CAR, 1.4, False, None, None),
    "hills": (lambda: _hilly(_oval()), CAR, 1.0, False, None, None),
    "rough left-hander": (lambda: _rough(_oval()).mirrored(), CAR, 0.95, False, None, None),
    "cornering drag": (lambda: _oval(), replace(CAR, k_corner_drag=0.05), 1.0, False, None, None),
    "gears": (lambda: _rough(_oval(straight=900.0)), None, 1.0, False, None, None),
    "shift time": (
        lambda: _rough(_oval(straight=900.0)),
        replace(GEARED, shift_s=(0.4, 0.3, 0.0)),
        1.0,
        False,
        None,
        None,
    ),
    "braking shape": (
        lambda: _rough(_oval()),
        replace(CAR, brake_ellipse=0.7),
        1.0,
        False,
        None,
        None,
    ),
    "onset": (
        lambda: _rough(_oval()),
        replace(CAR, onset_s=0.44, brake_ellipse=0.7),
        1.0,
        False,
        None,
        None,
    ),
    "onset through gears": (
        lambda: _rough(_oval(straight=900.0)),
        replace(GEARED, onset_s=0.26, shift_s=(0.4, 0.3, 0.0)),
        0.9,
        False,
        None,
        None,
    ),
    "onset, open": (
        lambda: _straight(600.0),
        replace(CAR, onset_s=0.5),
        1.0,
        False,
        55.0,
        20.0,
    ),
    "braking on the ellipse, open": (
        lambda: _straight(600.0),
        replace(CAR, brake_ellipse=1.0),
        1.0,
        False,
        55.0,
        20.0,
    ),
    "wet": (lambda: _oval(), replace(CAR, wet_scale=0.8), 1.0, True, None, None),
    "no drag": (
        lambda: sv.Profile.flat(np.full(400, 1 / 50.0), 2.0),
        replace(CAR, c_drag=0.0),
        1.0,
        False,
        None,
        None,
    ),
    "no apex under the power limit": (
        lambda: sv.Profile.flat(np.full(400, 1 / 900.0), 2.0),
        CAR,
        1.0,
        False,
        None,
        None,
    ),
    "open straight": (lambda: _straight(1000.0), CAR, 1.0, False, 20.0, None),
    "open, braked to an end speed": (lambda: _straight(600.0), CAR, 1.0, False, 55.0, 20.0),
    "open uphill": (lambda: _straight(3000.0, grade=0.05), CAR, 1.0, False, 30.0, None),
}


class TestCompiled:
    """The compiled lanes against the numpy laps they were written from."""

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    @pytest.mark.parametrize("name", list(PARITY_CASES))
    def test_same_node_speeds(self, name, monkeypatch):
        make, car, u, wet, v_start, v_end = PARITY_CASES[name]
        car = GEARED if car is None else car
        prof = make()
        monkeypatch.setattr(sv, "USE_COMPILED", True)
        fast = sv._passes(prof, car, u, wet, v_start, v_end)
        monkeypatch.setattr(sv, "USE_COMPILED", False)
        ref = sv._passes(prof, car, u, wet, v_start, v_end)
        assert np.isfinite(fast).all()
        assert np.array_equal(fast, ref)  # to the last bit

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    def test_same_lap_time_and_same_u(self, monkeypatch):
        prof = _rough(_oval(straight=900.0))
        got = {}
        for compiled in (True, False):
            monkeypatch.setattr(sv, "USE_COMPILED", compiled)
            t = sv.lap_time(prof, GEARED, 0.93)
            got[compiled] = (t, sv.solve_u_near(prof, GEARED, t * 1.01, 0.9)[0])
        assert got[True][0] == pytest.approx(got[False][0], rel=1e-12)
        assert got[True][1] == pytest.approx(got[False][1], abs=1e-7)

    def test_without_numba_the_numpy_laps_run(self, monkeypatch):
        monkeypatch.setattr(sv, "USE_COMPILED", False)
        sim = sv.simulate_lap(_oval(), CAR)
        assert 30.0 < sim.lap_time_s < 45.0

    def test_lap_time_is_the_simulated_laps(self):
        for prof in (_oval(), _straight(800.0)):
            sim = sv.simulate_lap(prof, CAR, 0.9)
            assert sv.lap_time(prof, CAR, 0.9) == pytest.approx(sim.lap_time_s, rel=1e-13)

    def test_the_power_table_is_kept_per_car(self):
        assert sv._power_table(GEARED) is sv._power_table(replace(GEARED))
        assert sv._power_table(GEARED) is not sv._power_table(replace(GEARED, shift_rpm=6500.0))


def _mixed(n):
    """``n`` laps of one track by different cars at different paces."""
    rng = np.random.default_rng(11)
    cars = [
        replace(
            GEARED if k % 3 == 0 else CAR,
            c_drag=CAR.c_drag * f,
            mu_b=CAR.mu_b * b,
            brake_ellipse=(0.0, 0.6, 1.0)[k % 3 if k % 2 else 0],
            onset_s=(0.0, 0.3, 0.45)[(k // 2) % 3],
            shift_s=(0.3, 0.2, 0.0) if k % 6 == 0 else (),
        )
        for k, (f, b) in enumerate(zip(rng.uniform(0.8, 1.2, n), rng.uniform(0.9, 1.1, n)))
    ]
    return cars, rng.uniform(0.85, 1.15, n)


def _short():
    """A small oval: the numpy laps cost about 200 µs a node whatever the
    batch, so the tests that run them keep the track short."""
    return _oval(straight=160.0, radius=35.0, ramp=30.0)


@pytest.fixture(params=[True, False], ids=["compiled", "numpy"])
def either(request, monkeypatch):
    """The solver with its passes compiled, and without."""
    if request.param and not _kernel.HAVE_NUMBA:
        pytest.skip("numba is not installed")
    monkeypatch.setattr(sv, "USE_COMPILED", request.param)
    return request.param


class TestBatch:
    """Many laps at once against the same laps one at a time."""

    def test_a_lap_of_a_batch_is_the_lap_alone(self, either):
        prof = _rough(_short())
        cars, us = _mixed(12)
        times = sv.drive_laps(prof, cars, us).lap_time_s.tolist()
        got = sv.solve_laps(prof, cars, times, guess=0.97)
        assert got.status == ["ok"] * 12
        assert got.u == pytest.approx(us, abs=2e-7)
        assert got.lap_time_s == pytest.approx(times, rel=1e-15)
        for k in range(0, 12, 1 if either else 5):  # one lap alone is slow in numpy
            u, status = sv.solve_u_near(prof, cars[k], times[k], 0.97)
            assert status == "ok" and got.u[k] == u
            alone = sv.simulate_lap(prof, cars[k], float(got.u[k])).v_node_ms
            assert np.array_equal(got.v_node_ms[k], alone)

    def test_one_car_and_one_guess_serve_every_lap(self, either):
        prof = _short()
        times = [sv.lap_time(prof, CAR, u) for u in (0.9, 1.0, 1.1)]
        one = sv.solve_laps(prof, CAR, times)
        each = sv.solve_laps(prof, [CAR] * 3, times, guess=[1.0, 1.0, 1.0])
        assert np.array_equal(one.u, each.u) and np.array_equal(one.v_node_ms, each.v_node_ms)
        assert one.u == pytest.approx([0.9, 1.0, 1.1], abs=2e-7)

    def test_a_lap_the_car_cannot_do_has_no_root_and_no_speeds(self, either):
        prof = _short()
        t = sv.lap_time(prof, CAR, 1.0)
        got = sv.solve_laps(prof, CAR, [t * 0.2, t, t * 30.0, float("nan")])
        assert got.status == ["no_root", "ok", "no_root", "no_root"]
        assert np.isnan(got.u[[0, 2, 3]]).all() and np.isnan(got.lap_time_s[[0, 2, 3]]).all()
        assert np.isnan(got.v_node_ms[[0, 2, 3]]).all() and np.isfinite(got.v_node_ms[1]).all()

    def test_a_guess_far_from_the_root_finds_the_same_root(self, either):
        prof = _rough(_short())
        t = sv.lap_time(prof, GEARED, 0.8)
        got = sv.solve_laps(prof, GEARED, [t] * 5, guess=[0.31, 0.8, 1.0, 1.7, 2.5])
        assert got.status == ["ok"] * 5
        assert got.u == pytest.approx(0.8, abs=2e-7)

    def test_in_the_wet(self, either):
        prof = _short()
        car = replace(CAR, wet_scale=0.8)
        t = sv.lap_time(prof, car, 1.0, wet=True)
        got = sv.solve_laps(prof, car, [t], wet=True)
        assert got.u[0] == pytest.approx(1.0, abs=2e-7)
        assert got.u[0] == pytest.approx(sv.solve_u_near(prof, car, t, wet=True)[0], abs=1e-12)
        assert sv.solve_laps(prof, car, [t]).u[0] == pytest.approx(0.8, abs=2e-7)

    def test_laps_driven_at_their_own_u(self, either):
        prof = _rough(_short())
        cars, us = _mixed(9)
        got = sv.drive_laps(prof, cars, us)
        assert got.ok.all() and np.array_equal(got.u, us)
        for k, car in enumerate(cars):
            sim = sv.simulate_lap(prof, car, float(us[k]))
            assert np.abs(got.v_node_ms[k] - sim.v_node_ms).max() < 1e-12
            assert got.lap_time_s[k] == pytest.approx(sim.lap_time_s, rel=1e-12)

    def test_an_open_profile_goes_lap_by_lap(self, either):
        prof = _straight(800.0)
        got = sv.drive_laps(prof, CAR, [0.9, 1.0])
        for k, u in enumerate((0.9, 1.0)):
            sim = sv.simulate_lap(prof, CAR, u)
            assert np.array_equal(got.v_node_ms[k], sim.v_node_ms)
            assert got.lap_time_s[k] == pytest.approx(sim.lap_time_s, rel=1e-12)

    def test_no_laps_and_the_wrong_number_of_cars(self, either):
        prof = _short()
        none = sv.solve_laps(prof, CAR, [])
        assert len(none.u) == 0 and none.v_node_ms.shape == (0, prof.n) and none.status == []
        assert sv.drive_laps(prof, [], []).v_node_ms.shape == (0, prof.n)
        with pytest.raises(ValueError, match="2 cars for 3 laps"):
            sv.solve_laps(prof, [CAR, CAR], [40.0, 41.0, 42.0])

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    def test_spread_over_the_cores_it_is_the_same(self, monkeypatch):
        prof = _rough(_short())
        cars, us = _mixed(24)
        times = sv.drive_laps(prof, cars, us).lap_time_s.tolist()
        got = {}
        for name, least in (("spread", 0), ("in turn", 10**12)):
            monkeypatch.setattr(sv, "PARALLEL_MIN_NODES", least)
            got[name] = (sv.solve_laps(prof, cars, times), sv.drive_laps(prof, cars, us))
        for a, b in zip(got["spread"], got["in turn"]):
            assert np.array_equal(a.u, b.u) and np.array_equal(a.ok, b.ok)
            assert np.array_equal(a.v_node_ms, b.v_node_ms)
            assert np.array_equal(a.lap_time_s, b.lap_time_s)

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    def test_the_compiled_lanes_are_the_numpy_laps(self, monkeypatch):
        prof = _rough(_short())
        cars, us = _mixed(13)
        times = sv.drive_laps(prof, cars, us).lap_time_s.tolist()
        # a lap too fast for the car, one too slow, one without a time; a
        # guess at each end of the range and one that is not a number
        times[10:] = [times[0] * 0.2, times[1] * 40.0, float("nan")]
        guesses = [1.0, 0.5, 2.2, 0.97, 1.3, 0.31, 2.5, float("nan"), 0.8, 1.2, 1.0, 1.0, 1.0]
        got = {}
        for compiled in (True, False):
            monkeypatch.setattr(sv, "USE_COMPILED", compiled)
            got[compiled] = (
                sv.solve_laps(prof, cars, times, guess=guesses),
                sv.drive_laps(prof, cars, us),
                [sv.solve_u_near(prof, cars[k], times[k], guesses[k]) for k in (0, 5, 7, 11)],
            )
        for a, b in zip(got[True][:2], got[False][:2]):  # to the last bit
            assert np.array_equal(a.u, b.u, equal_nan=True) and np.array_equal(a.ok, b.ok)
            assert np.array_equal(a.v_node_ms, b.v_node_ms, equal_nan=True)
            assert np.array_equal(a.lap_time_s, b.lap_time_s, equal_nan=True)
        assert got[True][0].status == ["ok"] * 10 + ["no_root"] * 3
        assert got[True][0].u[:10] == pytest.approx(us[:10], abs=2e-7)
        assert [s for _, s in got[True][2]] == [s for _, s in got[False][2]]
        assert np.array_equal(
            [u for u, _ in got[True][2]], [u for u, _ in got[False][2]], equal_nan=True
        )

    def test_a_long_batch_in_numpy_goes_through_in_pieces(self, monkeypatch):
        monkeypatch.setattr(sv, "USE_COMPILED", False)
        prof = _short()
        cars, us = _mixed(7)
        times = sv.drive_laps(prof, cars, us).lap_time_s.tolist()
        whole = (sv.solve_laps(prof, cars, times), sv.drive_laps(prof, cars, us))
        monkeypatch.setattr(sv, "TENSOR_MAX_LAPS", 3)
        pieces = (sv.solve_laps(prof, cars, times), sv.drive_laps(prof, cars, us))
        for a, b in zip(whole, pieces):
            assert np.array_equal(a.u, b.u) and np.array_equal(a.v_node_ms, b.v_node_ms)
            assert np.array_equal(a.lap_time_s, b.lap_time_s)


def _banked(prof, into=0.06, across=0.02):
    """``prof`` on a road that leans into its corners by ``into`` (rad) at
    full curvature and has a cross-slope of ``across`` on its straights."""
    top = np.abs(prof.kappa).max()
    lean = across + (into - across) * np.abs(prof.kappa) / top
    # the bank is stored left side higher: the outside of a right turn
    return replace(prof, bank=lean * np.where(prof.kappa >= 0.0, 1.0, -1.0))


class _Track:
    """What :meth:`Profile.from_track` reads of a track."""

    def __init__(self, n, bank):
        self.n, self.ds_m = n, 2.0
        self.kappa_heat = np.full(n, 1 / 50.0)
        self.grade, self.kappa_v, self.bank = np.zeros(n), np.zeros(n), bank


class TestBank:
    """The road's bank: the tyres hold less of the cornering and carry more
    load where the road leans into the turn."""

    FREE = replace(CAR, c_drag=0.0)

    @pytest.mark.parametrize("side", [1.0, -1.0])
    @pytest.mark.parametrize("lean", [0.08, -0.05])
    def test_constant_radius(self, side, lean):
        # V² (cos b − mu·sin b) / R = g (mu·cos b + sin b), turning either way
        n, mu, radius = 200, 1.2, 50.0
        prof = replace(
            sv.Profile.flat(np.full(n, side / radius), 2.0), bank=np.full(n, side * lean)
        )
        sim = sv.simulate_lap(prof, self.FREE)
        want = math.sqrt(
            G * radius * (mu * math.cos(lean) + math.sin(lean))
            / (math.cos(lean) - mu * math.sin(lean))
        )  # fmt: skip
        assert sim.v_node_ms == pytest.approx(want, abs=1e-9)
        flat = math.sqrt(mu * G * radius)
        assert (want > flat) == (lean > 0.0)
        # the ground-frame lateral is the cornering; the tyres hold the grip times the load
        assert np.abs(sim.lat_g) == pytest.approx(want**2 / radius / G, abs=1e-9)
        load = math.cos(lean) + want**2 / radius * math.sin(lean) / G
        assert sim.lat_tyre_g * side == pytest.approx(mu * load, abs=1e-9)
        assert sv.combined_use(sim, prof, self.FREE) == pytest.approx(1.0, abs=1e-9)

    def test_no_bank_and_a_bank_of_nothing_are_the_flat_road(self, either):
        prof = _rough(_short())
        cars, us = _mixed(6)
        flat = sv.drive_laps(prof, cars, us)
        zero = sv.drive_laps(replace(prof, bank=np.zeros(prof.n)), cars, us)
        assert np.array_equal(flat.v_node_ms, zero.v_node_ms)  # to the last bit
        assert np.array_equal(flat.lap_time_s, zero.lap_time_s)
        sim = sv.simulate_lap(prof, CAR, 0.95)
        assert sim.lat_tyre_g is sim.lat_g

    def test_into_the_turn_is_faster_and_out_of_it_slower(self, either):
        prof = _short()
        flat = sv.lap_time(prof, CAR, 1.0)
        assert sv.lap_time(_banked(prof, 0.06, 0.0), CAR, 1.0) < flat - 0.1
        assert sv.lap_time(_banked(prof, -0.06, 0.0), CAR, 1.0) > flat + 0.1

    def test_a_cross_slope_on_a_straight_takes_from_the_braking(self):
        # the tyres hold the car against the slope: g·sin b of the grip, taken linearly
        lean = 0.1
        prof = replace(_straight(600.0), bank=np.full(_straight(600.0).n, lean))
        sim = sv.simulate_lap(prof, self.FREE, v_start=55.0, v_end=20.0)
        used = math.sin(lean) / (CAR.mu_y * math.cos(lean))
        assert sim.long_tyre_g.min() == pytest.approx(
            -CAR.mu_b * math.cos(lean) * (1.0 - used), abs=1e-6
        )
        # either way the road leans
        other = sv.simulate_lap(replace(prof, bank=-prof.bank), self.FREE, v_start=55.0, v_end=20.0)
        assert np.array_equal(other.v_node_ms, sim.v_node_ms)

    def test_mirrored_rolled_and_refined_it_is_the_same_road(self, either):
        prof = _banked(_rough(_short()))
        lap = sv.simulate_lap(prof, CAR, 0.97)
        mirror = sv.simulate_lap(prof.mirrored(), CAR, 0.97)
        assert np.array_equal(mirror.v_node_ms, lap.v_node_ms)
        assert np.array_equal(mirror.lat_tyre_g, -lap.lat_tyre_g)
        # a step's is the mean of its two nodes', by the one formula the heat chain uses
        nodes = bankmod.tyre_lateral_g(prof.kappa, lap.v_node_ms, prof.bank)
        assert lap.lat_tyre_g == pytest.approx(0.5 * (nodes + np.roll(nodes, -1)), abs=1e-12)
        rolled = sv.simulate_lap(prof.rolled(37), CAR, 0.97)
        assert np.roll(rolled.v_node_ms, 37) == pytest.approx(lap.v_node_ms, abs=1e-6)
        assert sv.lap_time(prof.refined(2), CAR, 0.97) == pytest.approx(lap.lap_time_s, rel=2e-3)

    def test_no_step_is_over_the_limit_and_the_lap_time_falls_as_u_rises(self, either):
        prof = _banked(_rough(_short()))
        car = replace(CAR, brake_ellipse=0.6, k_corner_drag=0.02)
        sim = sv.simulate_lap(prof, car, 0.95)
        assert sv.combined_use(sim, prof, car).max() <= 1.005
        # in the banked corners the tyres hold less than the cornering
        apex = int(np.argmax(np.abs(prof.kappa)))
        assert abs(sim.lat_tyre_g[apex]) < abs(sim.lat_g[apex])
        times = [sv.lap_time(prof, car, u) for u in np.linspace(0.6, 1.4, 17)]
        assert np.all(np.diff(times) < 0.0)

    @pytest.mark.skipif(not _kernel.HAVE_NUMBA, reason="numba is not installed")
    def test_the_compiled_lanes_are_the_numpy_laps(self, monkeypatch):
        prof = _banked(_rough(_short()), 0.09, -0.03)
        cars, us = _mixed(9)
        times = sv.drive_laps(prof, cars, us).lap_time_s.tolist()
        got = {}
        for compiled in (True, False):
            monkeypatch.setattr(sv, "USE_COMPILED", compiled)
            got[compiled] = (
                sv.solve_laps(prof, cars, times, guess=0.9, wet=True),
                sv.drive_laps(prof, cars, us),
                sv._passes(prof, cars[1], 0.93, False, None, None),
            )
        for a, b in zip(got[True][:2], got[False][:2]):
            assert np.array_equal(a.u, b.u) and np.array_equal(a.ok, b.ok)
            assert np.array_equal(a.v_node_ms, b.v_node_ms)  # to the last bit
            assert np.array_equal(a.lap_time_s, b.lap_time_s)
        assert np.array_equal(got[True][2], got[False][2])

    def test_a_road_leaning_out_by_more_than_the_grip_holds_leaves_the_floor(self, either):
        n = 100
        prof = replace(sv.Profile.flat(np.full(n, 1 / 50.0), 2.0), bank=np.full(n, -0.6))
        got = sv.drive_laps(prof, replace(CAR, mu_y=0.5), [1.0])
        assert got.v_node_ms[0] == pytest.approx(sv.V_FLOOR_MS)

    def test_a_tracks_bank_is_taken_unless_the_road_is_asked_flat(self):
        n = 60
        banked, bare = _Track(n, np.full(n, 0.05)), _Track(n, np.zeros(0))
        got = sv.Profile.from_track(banked)
        assert got.bank == pytest.approx(0.05) and got.lean == pytest.approx(0.05)
        assert sv.Profile.from_track(banked, bank=False).bank is None
        # a track without one is a flat road
        assert sv.Profile.from_track(bare).bank is None
        assert sv.Profile.from_track(bare).lean == pytest.approx(0.0)
        # and a lap of the track itself is driven on its bank
        flat = sv.lap_time(sv.Profile.from_track(banked, bank=False), self.FREE, 1.0)
        assert sv.lap_time(banked, self.FREE, 1.0) < flat - 0.05


class TestUseCap:
    """A use is capped at 1 on a car at its limit (#206; the driver
    committee's clause 3.1): above it the cornering stays and the braking
    level rises to the car's own."""

    CAPPED = replace(REF_CAR, brake_use=0.85, use_cap=1.0)

    def _track(self):
        return _reference_oval(2.0, 40.0)

    def test_at_or_under_the_cap_nothing_changes(self):
        track = self._track()
        free = replace(self.CAPPED, use_cap=math.inf)
        for u in (0.8, 1.0):
            a = sv.simulate_lap(track, self.CAPPED, u)
            b = sv.simulate_lap(track, free, u)
            np.testing.assert_array_equal(a.v_node_ms, b.v_node_ms)
        both = sv.drive_laps(track, [self.CAPPED, free], [0.93, 0.93])
        np.testing.assert_array_equal(both.v_node_ms[0], both.v_node_ms[1])

    def test_above_it_the_cornering_stays_and_the_braking_rises(self):
        track = self._track()
        at_cap = sv.simulate_lap(track, self.CAPPED, 1.0)
        above = sv.simulate_lap(track, self.CAPPED, 1.1)
        free = sv.simulate_lap(track, replace(self.CAPPED, use_cap=math.inf), 1.1)
        # the slowest point of the lap is the corner's, at the cornering limit: unchanged
        assert above.v_node_ms.min() == pytest.approx(at_cap.v_node_ms.min(), rel=1e-12)
        assert free.v_node_ms.min() > above.v_node_ms.min() * 1.02
        # braking is harder than at the cap, by the knob
        assert -above.long_tyre_g.min() == pytest.approx(-at_cap.long_tyre_g.min() * 1.1, rel=0.01)
        assert at_cap.lap_time_s > above.lap_time_s > free.lap_time_s
        # it is the car at its cap with the driver braking that much harder
        same = replace(self.CAPPED, brake_use=0.85 * 1.1, use_cap=math.inf)
        np.testing.assert_array_equal(above.v_node_ms, sv.simulate_lap(track, same, 1.0).v_node_ms)

    def test_lap_time_is_continuous_through_the_cap_and_flat_past_the_end(self):
        track = self._track()
        end = sv.knob_end(self.CAPPED)
        assert end == pytest.approx(1.0 / 0.85)
        t = [sv.lap_time(track, self.CAPPED, u) for u in (1.0 - 1e-6, 1.0, 1.0 + 1e-6)]
        # no step at the cap; past it a knob buys less lap time (braking only)
        assert 0.0 < t[1] - t[2] < t[0] - t[1] < 1e-4
        assert sv.lap_time(track, self.CAPPED, end) == sv.lap_time(track, self.CAPPED, end * 1.3)
        # the end lap is the car braked at its own level, on the shape it holds
        own_level = replace(self.CAPPED, brake_use=1.0, use_cap=math.inf)
        assert sv.lap_time(track, self.CAPPED, end) == pytest.approx(
            sv.lap_time(track, own_level, 1.0), rel=1e-12
        )
        assert sv.knob_end(REF_CAR) == math.inf

    @pytest.mark.parametrize("compiled", [True, False])
    def test_every_way_of_solving_agrees_above_the_cap(self, compiled, monkeypatch):
        monkeypatch.setattr(sv, "USE_COMPILED", compiled and sv.USE_COMPILED)
        track = self._track()
        target = sv.lap_time(track, self.CAPPED, 1.08)
        one, status = sv.solve_u(track, self.CAPPED, target)
        near, near_status = sv.solve_u_near(track, self.CAPPED, target, guess=0.95)
        batch = sv.solve_laps(track, self.CAPPED, [target, sv.lap_time(track, self.CAPPED, 0.9)])
        curve, curve_status = sv.PaceCurve.build(track, self.CAPPED).solve(target)
        assert status == near_status == curve_status == "ok" and batch.ok.all()
        for got in (one, near, batch.u[0], curve):
            assert got == pytest.approx(1.08, abs=2e-5)
        assert batch.u[1] == pytest.approx(0.9, abs=2e-5)
        np.testing.assert_allclose(
            batch.v_node_ms[0], sv.simulate_lap(track, self.CAPPED, 1.08).v_node_ms, atol=2e-3
        )

    @pytest.mark.parametrize("compiled", [True, False])
    def test_faster_than_the_end_lap_has_no_root(self, compiled, monkeypatch):
        monkeypatch.setattr(sv, "USE_COMPILED", compiled and sv.USE_COMPILED)
        track = self._track()
        target = sv.lap_time(track, self.CAPPED, sv.knob_end(self.CAPPED)) - 0.05
        assert sv.solve_u(track, self.CAPPED, target)[1] == "no_root"
        assert sv.solve_u_near(track, self.CAPPED, target, guess=1.0)[1] == "no_root"
        batch = sv.solve_laps(track, self.CAPPED, [target])
        assert not batch.ok[0] and np.isnan(batch.u[0]) and np.isnan(batch.v_node_ms[0]).all()
        assert sv.PaceCurve.build(track, self.CAPPED).solve(target)[1] == "no_root"
        # the same lap time has a root without the cap
        assert sv.solve_laps(track, replace(self.CAPPED, use_cap=math.inf), [target]).ok[0]

    def test_the_compiled_and_the_numpy_form_agree_to_the_last_bit(self, monkeypatch):
        if not sv.USE_COMPILED:
            pytest.skip("numba is not installed")
        track = self._track()
        fast = sv.drive_laps(track, self.CAPPED, [0.9, 1.05, 1.3])
        monkeypatch.setattr(sv, "USE_COMPILED", False)
        slow = sv.drive_laps(track, self.CAPPED, [0.9, 1.05, 1.3])
        np.testing.assert_array_equal(fast.v_node_ms, slow.v_node_ms)
        np.testing.assert_array_equal(fast.lap_time_s, slow.lap_time_s)

    def test_how_much_of_the_envelope_a_capped_lap_uses(self):
        track = self._track()
        sim = sv.simulate_lap(track, self.CAPPED, 1.1)
        used = sv.combined_use(sim, track, self.CAPPED)
        assert used.max() == pytest.approx(1.0, abs=0.02)
