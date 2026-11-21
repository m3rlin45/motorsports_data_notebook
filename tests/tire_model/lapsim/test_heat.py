"""Production's heat chain on one lap (lap simulation, #205 / #206)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.heat_input import ShareParams, share_heat_rate
from motorsports_data_notebook.tire_model.lapsim import heat

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
