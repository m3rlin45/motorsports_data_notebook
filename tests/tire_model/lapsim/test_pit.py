"""Pit-exit join and vertical curvature (asked of the track by #206)."""

from __future__ import annotations

import numpy as np
import pytest

from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from motorsports_data_notebook.tire_model.lapsim import pit
from motorsports_data_notebook.tire_model.lapsim import track as trk

from ._synthetic import LAYOUT_ID, LENGTH_M, stadium, write_dataset


@pytest.fixture()
def build(tmp_path):
    geo._INDEX_CACHE.clear()
    write_dataset(
        tmp_path,
        {
            f"2026-01-1{i}_s{i}": {
                "car": "car",
                "date": f"2026-01-1{i}",
                "kwargs": {"n_laps": 5, "seed": i, "noise_m": 0.02},
            }
            for i in range(3)
        },
    )
    return tmp_path, trk.build_track_model(tmp_path, LAYOUT_ID, inertial=False, n_boot=0)


def _outlap(join_s: float, side_m: float = 9.0):
    """An out-lap that runs ``side_m`` to the right of the track from 250 m
    before the line, crosses the line's position and merges at ``join_s``
    over 40 m, then follows the track to the line."""
    s = np.concatenate([np.arange(-250.0, 0.0, 1.0), np.arange(0.0, LENGTH_M, 1.0)])
    x, y, h = stadium(s)
    blend = np.clip((join_s - s) / 40.0, 0.0, 1.0)  # 1 before the merge, 0 after it
    x = x + side_m * blend * np.cos(h)  # right of a compass heading h is (cos h, -sin h)
    y = y - side_m * blend * np.sin(h)
    speed = np.where(s < join_s, 12.0, 25.0)
    t = np.concatenate([[0.0], np.cumsum(np.diff(s) / speed[:-1])])
    return t, x, y, speed


class TestPitJoin:
    def test_join_is_where_the_out_lap_meets_the_driven_line(self, build):
        _, b = build
        t, x, y, speed = _outlap(join_s=180.0)
        hit = pit.join_of_outlap(b, t, x, y, speed)
        assert hit is not None
        s_join, rolling = hit
        # inside the build laps' band (about 1 m) near the end of the 40 m merge
        assert 140.0 < s_join < 185.0
        assert rolling == pytest.approx((250.0 + s_join) / 12.0, rel=0.05)

    def test_lap_that_never_leaves_the_track_has_no_join(self, build):
        _, b = build
        t, x, y, speed = _outlap(join_s=180.0, side_m=0.0)
        assert pit.join_of_outlap(b, t, x, y, speed) is None

    def test_too_short_to_read(self, build):
        _, b = build
        t, x, y, speed = _outlap(join_s=180.0)
        assert pit.join_of_outlap(b, t[-50:], x[-50:], y[-50:], speed[-50:]) is None

    def test_the_band_is_the_builds_and_can_be_handed_in(self, build):
        _, b = build
        lo, hi = pit.build_band(b)
        assert lo.shape == hi.shape == (b.model.n,) and np.all(hi - lo >= 2 * pit.BAND_MARGIN_M)
        t, x, y, speed = _outlap(join_s=180.0)
        assert pit.join_of_outlap(b, t, x, y, speed, (lo, hi)) == pit.join_of_outlap(
            b, t, x, y, speed
        )

    def test_layout_without_pit_out_laps(self, build):
        root, b = build
        # the synthetic out-laps start on the track itself
        assert pit.pit_exit(root, b) == {"n": 0}


class TestVerticalCurvature:
    def test_second_difference_of_the_altitude(self):
        n, ds = 600, 2.0
        s = np.arange(n) * ds
        alt = 10.0 * np.sin(2 * np.pi * s / (n * ds))
        grade = (np.roll(alt, -12) - np.roll(alt, 12)) / (24 * ds)
        kv = trk.vertical_curvature(grade, ds)
        want = -10.0 * (2 * np.pi / (n * ds)) ** 2 * np.sin(2 * np.pi * s / (n * ds))
        assert kv == pytest.approx(want, abs=3e-6)

    def test_flat_track_has_none_and_round_trips(self, build, tmp_path):
        _, b = build
        assert b.model.kappa_v.shape == b.model.kappa_heat.shape
        assert np.abs(b.model.kappa_v).max() < 1e-7
        back = trk.TrackModel.load(b.model.save(tmp_path / "t.json"))
        assert back.kappa_v == pytest.approx(b.model.kappa_v, abs=1e-7)

    def test_older_json_without_it_still_loads(self, build):
        _, b = build
        data = b.model.to_json()
        del data["kappa_v_1_per_m"]
        assert trk.TrackModel.from_json(data).kappa_v.size == 0
