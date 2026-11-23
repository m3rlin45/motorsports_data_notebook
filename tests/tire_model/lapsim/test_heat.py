"""Production's heat chain on one lap (lap simulation, #205 / #206)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.heat_input import ShareParams, share_heat_rate
from motorsports_data_notebook.tire_model.lapsim import heat
from motorsports_data_notebook.tire_model.lapsim import solver as sv

SHARE = ShareParams(
    p_f=0.45,
    g_transfer_front=2.5,
    g_transfer_rear=2.5,
    beta0=math.log(0.6 / 0.4),
    beta1=0.0,
    eps_drive=1.0,
    eps_brake=1.0,
)


def _series(duration=10.4, hz=50.0, lat=0.0, long=0.0, v=30.0):
    t = np.arange(0.0, duration, 1.0 / hz) + 100.0
    n = t.size
    return t, np.full(n, v), np.full(n, lat), np.full(n, long)


class TestBins:
    def test_seconds_from_the_lap_start_and_a_short_last_bin(self):
        t, v, lat, lng = _series(lat=0.8)
        bins = heat.bin_1s(t, v, lat, lng, 100.0, 110.4)
        assert len(bins.dur_s) == 11
        assert bins.dur_s[:10] == pytest.approx(1.0)
        assert bins.dur_s[-1] == pytest.approx(0.4)
        assert bins.coverage == pytest.approx(1.0)
        assert bins.lat_pos == pytest.approx(0.8)
        assert bins.lat_neg == pytest.approx(0.0)
        assert bins.v_ms == pytest.approx(30.0)

    def test_positive_parts_are_taken_before_the_mean(self):
        # a second that holds as much drive as braking has both, not neither
        t, v, lat, lng = _series()
        lng = np.where(np.arange(t.size) % 2 == 0, 0.5, -0.5)
        bins = heat.bin_1s(t, v, lat, lng, 100.0, 110.0)
        assert bins.long_pos == pytest.approx(0.25)
        assert bins.long_neg == pytest.approx(0.25)

    def test_only_the_laps_own_samples_and_the_clip(self):
        t, v, lat, lng = _series(duration=30.0, lat=5.0)
        bins = heat.bin_1s(t, v, lat, lng, 105.0, 115.0)
        assert len(bins.dur_s) == 10
        assert bins.dur_s == pytest.approx(1.0)
        assert bins.lat_pos == pytest.approx(3.0)

    def test_a_non_finite_sample_is_left_out_not_counted_as_zero(self):
        t, v, lat, lng = _series(lat=1.0)
        lat[::2] = np.nan
        bins = heat.bin_1s(t, v, lat, lng, 100.0, 110.4)
        assert bins.lat_pos == pytest.approx(1.0)


class TestLapHeat:
    def test_equals_the_share_model_on_a_steady_series(self):
        t, v, lat, lng = _series(lat=1.0, long=-0.3)
        q = heat.series_heat(t, v, lat, lng, SHARE, 100.0, 110.4)
        one = np.array([1.0])
        want = [
            float(share_heat_rate(SHARE, one, 0 * one, 0 * one, 0.3 * one, 30 * one, c)[0])
            for c in heat.CORNERS
        ]
        assert q == pytest.approx(want)

    def test_a_right_turn_heats_the_left_tyres_and_the_mirror_swaps_them(self):
        t, v, lat, lng = _series(lat=1.2)
        right = heat.series_heat(t, v, lat, lng, SHARE, 100.0, 110.4)
        left = heat.series_heat(t, v, -lat, lng, SHARE, 100.0, 110.4)
        fl, fr, rl, rr = right
        assert fl > fr and rl > rr
        assert left == pytest.approx(right[[1, 0, 3, 2]])

    def test_a_lap_with_a_hole_has_no_heat(self):
        t, v, lat, lng = _series(duration=20.0, lat=1.0)
        hole = (t >= 105.0) & (t < 110.0)
        q = heat.series_heat(t[~hole], v[~hole], lat[~hole], lng[~hole], SHARE, 100.0, 120.0)
        assert np.isnan(q).all()

    def test_the_rate_of_the_series_does_not_matter(self):
        rng = np.random.default_rng(0)
        t = np.arange(0.0, 60.0, 0.01)
        lat = np.sin(t / 3.0)
        lng = 0.4 * np.cos(t / 2.0)
        v = 30.0 + 5.0 * np.sin(t / 7.0)
        fine = heat.series_heat(t, v, lat, lng, SHARE, 0.0, 60.0)
        pick = np.sort(rng.choice(t.size, t.size // 4, replace=False))
        coarse = heat.series_heat(t[pick], v[pick], lat[pick], lng[pick], SHARE, 0.0, 60.0)
        assert coarse == pytest.approx(fine, rel=0.003)


class TestPhase:
    def test_the_bins_edges_moved_back_keep_the_laps_time_and_a_steady_heat(self):
        t, v, lat, lng = _series(lat=1.0, long=-0.3)
        plain = heat.bin_1s(t, v, lat, lng, 100.0, 110.4)
        moved = heat.bin_1s(t, v, lat, lng, 100.0, 110.4, phase_s=0.5)
        assert moved.dur_s.sum() == pytest.approx(10.4) and moved.dur_s[0] == pytest.approx(0.5)
        assert len(moved.dur_s) == 11 and moved.dur_s[-1] == pytest.approx(0.9)
        assert heat.lap_heat(moved, SHARE) == pytest.approx(heat.lap_heat(plain, SHARE))

    def test_where_speed_and_braking_change_within_a_second_the_phase_moves_the_heat(self):
        # heat is the bin's mean speed times its mean force: a second that holds the
        # braking at 40 m/s and the drive at 20 is not the two seconds apart
        t = np.arange(0.0, 40.0, 0.02)
        even = np.floor(t) % 2 == 0
        lng = np.where(even, 0.5, -0.9)
        v = np.where(even, 20.0, 40.0)
        apart = heat.lap_heat(heat.bin_1s(t, v, 0 * t, lng, 0.0, 40.0), SHARE)
        mixed = heat.lap_heat(heat.bin_1s(t, v, 0 * t, lng, 0.0, 40.0, phase_s=0.5), SHARE)
        assert (np.abs(mixed / apart - 1.0) > 0.01).any()


def _oval(ds=2.0, straight=400.0, radius=60.0, ramp=40.0):
    arc = math.pi * radius - ramp
    piece = np.concatenate(
        [
            np.zeros(int(straight / ds)),
            np.linspace(0.0, 1.0, int(ramp / ds), endpoint=False),
            np.ones(int(round(arc / ds))),
            np.linspace(1.0, 0.0, int(ramp / ds), endpoint=False),
        ]
    )
    return sv.Profile.flat(np.concatenate([piece, piece]) / radius, ds)


CAR = sv.SolverCar(mu_y=1.2, mu_b=1.0, p_specific=100.0, c_drag=3.36e-4)


class TestSimulatedLap:
    """Production's chain on a simulated lap (gate D1 of #206)."""

    def test_a_steady_circle_is_the_share_model(self):
        prof = sv.Profile.flat(np.full(300, 1 / 60.0), 2.0)
        sim = sv.simulate_lap(prof, CAR, 1.0)
        v = float(sim.v_ms.mean())
        lat = v * v / 60.0 / sv.G_MS2
        one = np.array([1.0])
        want = [
            float(share_heat_rate(SHARE, lat * one, 0 * one, 0 * one, 0 * one, v * one, c)[0])
            for c in heat.CORNERS
        ]
        assert heat.sim_heat(sim, SHARE) == pytest.approx(want, rel=1e-6)

    def test_a_right_hand_lap_heats_the_left_tyres_and_the_mirror_swaps_them(self):
        prof = _oval()
        right = heat.sim_heat(sv.simulate_lap(prof, CAR, 1.0), SHARE)
        left = heat.sim_heat(sv.simulate_lap(prof.mirrored(), CAR, 1.0), SHARE)
        fl, fr, rl, rr = right
        assert fl > fr and rl > rr  # κ is positive in a right turn
        assert left == pytest.approx(right[[1, 0, 3, 2]], rel=1e-12)

    def test_halving_the_step_moves_the_heat_by_under_a_thousandth(self):
        prof = _oval()
        coarse = heat.sim_heat(sv.simulate_lap(prof, CAR, 0.95), SHARE)
        fine = heat.sim_heat(sv.simulate_lap(prof.refined(2), CAR, 0.95), SHARE)
        assert fine == pytest.approx(coarse, rel=1e-3)

    def test_the_clock_it_is_put_on_does_not_matter(self):
        sim = sv.simulate_lap(_oval(), CAR, 1.0)
        assert heat.sim_heat(sim, SHARE, hz=25.0) == pytest.approx(
            heat.sim_heat(sim, SHARE), rel=5e-3
        )

    def test_its_series_is_time_weighted_and_as_an_accelerometer_reads(self):
        n = 400
        prof = sv.Profile(
            2.0, np.full(n, 1 / 80.0), np.full(n, 0.04), np.zeros(n), bank=np.full(n, 0.06)
        )
        sim = sv.simulate_lap(prof, CAR, 1.0)
        t, v, lat, lng = heat.sim_series(sim)
        assert t[-1] < sim.lap_time_s and np.diff(t) == pytest.approx(0.01)
        assert lng == pytest.approx(0.04, abs=1e-3)  # a steady climb: no dV/dt, the grade
        assert lat == pytest.approx(sim.lat_tyre_g[0], rel=1e-6)  # what the tyres hold
        assert sim.lat_tyre_g[0] < sim.lat_g[0]  # the bank carries the rest
        # a lap that is slow in one half: the mean over time, not over distance
        both = sv.simulate_lap(_oval(), CAR, 1.0)
        _, v, _, _ = heat.sim_series(both)
        by_time = float(np.sum(both.v_ms * both.dt_s) / both.dt_s.sum())
        assert v.mean() == pytest.approx(by_time, rel=1e-3)
        assert abs(v.mean() - both.v_ms.mean()) > 0.5
