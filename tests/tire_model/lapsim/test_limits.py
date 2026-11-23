"""The car's cornering limit as its best moments (lap simulation, #312)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from motorsports_data_notebook.tire_model.lapsim import limits as lim


def _apex(
    values: np.ndarray,
    *,
    layout: str = "a",
    corner: int = 0,
    days: int = 10,
    noise: tuple[float, float] = (0.0, 0.0),
    accel_scale: float = 0.93,
    load: float = 1.05,
    seed: int = 0,
    accel: bool = True,
) -> pd.DataFrame:
    """An apex table of one corner: ``values`` are the passes' true lateral
    over load, spread over ``days`` days (one session a day)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    day = np.arange(n) % days
    x1 = values * np.exp(rng.normal(0.0, noise[0], n)) if noise[0] else values
    x2 = values * np.exp(rng.normal(0.0, noise[1], n)) if noise[1] else values
    tyre = x1 * load
    return pd.DataFrame(
        {
            "session_id": [f"s{layout}{d}" for d in day],
            "car": "car",
            "date": [f"2026-01-{d + 1:02d}" for d in day],
            "layout_id": layout,
            "basis": "dry",
            "lap_num": np.arange(n),
            "corner": corner,
            "sign": 1,
            "s_apex_m": 100.0 * (corner + 1),
            "v_apex_ms": 30.0,
            "v_min_ms": 30.0,
            "lat_track_g": tyre,
            "lat_doppler_g": tyre * 1.02,  # the ground frame reads the bank's help too
            "lat_accel_g": x2 * load * accel_scale if accel else np.nan,
            "lat_tyre_g": tyre,
            "load": load,
            # where the other sensor peaks, each reads 2 % under its own peak
            "lat_tyre_cross_g": tyre * 0.98,
            "lat_accel_cross_g": x2 * load * accel_scale * 0.98 if accel else np.nan,
        }
    )


class TestExpectedBest:
    def test_one_pass_is_the_mean_and_all_of_them_the_largest(self):
        x = np.array([3.0, 1.0, 2.0, 5.0])
        assert lim.expected_best(x, 1) == pytest.approx(x.mean())
        assert lim.expected_best(x, 4) == pytest.approx(5.0)
        assert np.isnan(lim.expected_best(x, 5))

    @pytest.mark.parametrize("m", [2, 3, 6])
    def test_it_is_the_mean_of_the_largest_over_every_draw(self, m):
        x = np.random.default_rng(1).normal(1.3, 0.05, 9)
        want = np.mean([max(c) for c in combinations(x, m)])
        assert lim.expected_best(x, m) == pytest.approx(want, rel=1e-12)

    def test_many_passes_do_not_overflow(self):
        x = np.linspace(1.0, 1.4, 2000)
        got = lim.expected_best(x, 50)
        assert 1.38 < got < 1.4

    def test_more_passes_of_the_same_kind_do_not_raise_it(self):
        rng = np.random.default_rng(2)
        few = np.mean([lim.expected_best(rng.normal(0, 1, 40), 20) for _ in range(400)])
        many = np.mean([lim.expected_best(rng.normal(0, 1, 400), 20) for _ in range(400)])
        assert few == pytest.approx(many, abs=0.05)  # where a maximum would rise by 0.8


class TestBestOfNormal:
    def test_the_expected_largest_of_m_standard_normals(self):
        assert lim.best_of_normal(1) == pytest.approx(0.0, abs=1e-6)
        assert lim.best_of_normal(2) == pytest.approx(1.0 / np.sqrt(np.pi), rel=1e-3)
        draws = np.random.default_rng(0).normal(size=(200_000, 20)).max(axis=1).mean()
        assert lim.best_of_normal(20) == pytest.approx(draws, rel=5e-3)


class TestPasses:
    def test_the_accelerometer_is_put_on_the_dopplers_scale_per_session(self):
        values = np.linspace(1.1, 1.3, 60)
        got = lim.passes(_apex(values))
        assert len(got) == 60
        assert np.allclose(got["x1"], np.log(values))
        assert np.allclose(got["x2"], got["x1"])  # 0.93 of it, in every session
        assert np.allclose(got["y"], np.log(values))

    def test_a_thin_day_a_slow_pass_and_a_wet_lap_are_left_out(self):
        apex = _apex(np.full(60, 1.2))
        apex.loc[apex["date"] == "2026-01-03", "date"] = [f"2026-02-{k:02d}" for k in range(1, 7)]
        apex.loc[0, "lat_doppler_g"] = 0.05
        apex.loc[1, "basis"] = "wet"
        got = lim.passes(apex)
        assert len(got) == 60 - 6 - 2

    def test_only_the_grip_limited_corners(self):
        hard = _apex(np.full(60, 1.2))
        easy = _apex(np.full(60, 0.6), corner=1)
        got = lim.passes(pd.concat([hard, easy], ignore_index=True))
        assert set(got["corner"]) == {0}

    def test_without_an_accelerometer_there_are_none_unless_one_sensor_is_asked(self):
        apex = _apex(np.full(60, 1.2), accel=False)
        assert lim.passes(apex).empty
        alone = lim.passes(apex, one_sensor=True)
        assert len(alone) == 60 and alone["x2"].isna().all()
        assert np.allclose(alone["y"], alone["x1"])

    def test_read_where_the_other_sensor_peaks(self):
        values = np.linspace(1.1, 1.3, 60)
        own = lim.passes(_apex(values))
        cross = lim.passes(_apex(values), cross=True)
        # both readings 2 % lower, on the scale the sessions' own peaks set
        assert np.allclose(cross["x1"], own["x1"] + np.log(0.98))
        assert np.allclose(cross["x2"], cross["x1"])
        assert lim.passes(_apex(values), cross=True, one_sensor=True).empty
        assert lim.passes(_apex(values).drop(columns=["lat_tyre_cross_g"]), cross=True).empty

    def test_a_table_without_the_tyres_frame_has_none(self):
        assert lim.passes(_apex(np.full(60, 1.2)).drop(columns=["load"])).empty


class TestCornerLevels:
    def test_without_noise_the_level_is_the_expected_best_as_read(self):
        values = np.random.default_rng(3).normal(1.2, 0.05, 200)
        got = lim.corner_levels(lim.passes(_apex(values)), 20).iloc[0]
        assert got["n"] == 200 and got["supports"]
        assert got["raw"] == pytest.approx(lim.expected_best(values, 20))
        assert got["level"] == pytest.approx(got["raw"], rel=1e-9)
        assert got["sd_noise"] == pytest.approx(0.0, abs=1e-12)

    def test_the_sensors_noise_raises_the_reading_and_is_taken_out(self):
        rng = np.random.default_rng(4)
        truth, raws, levels, noises = [], [], [], []
        for seed in range(40):
            values = np.exp(rng.normal(np.log(1.2), 0.04, 300))
            got = lim.corner_levels(
                lim.passes(_apex(values, noise=(0.03, 0.03), seed=seed)), 20
            ).iloc[0]
            truth.append(lim.expected_best(values, 20))
            raws.append(got["raw"])
            levels.append(got["level"])
            noises.append(got["sd_noise"])
        # what is left in the mean of two readings of 3 % each
        assert np.mean(noises) == pytest.approx(0.03 / np.sqrt(2.0), rel=0.05)
        assert np.mean(raws) / np.mean(truth) - 1.0 > 0.004
        assert np.mean(levels) / np.mean(truth) - 1.0 == pytest.approx(0.0, abs=0.002)

    def test_support_is_twice_m(self):
        got = lim.corner_levels(lim.passes(_apex(np.full(60, 1.2))), 20)
        assert bool(got.iloc[0]["supports"])
        assert not bool(
            lim.corner_levels(lim.passes(_apex(np.full(60, 1.2))), 50).iloc[0]["supports"]
        )


class TestCarLimit:
    def _car(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        parts = []
        for layout, corner, level, n in (
            ("a", 0, 1.20, 200),
            ("a", 1, 1.24, 200),
            ("b", 0, 1.30, 200),
        ):
            values = np.exp(rng.normal(np.log(level), 0.03, n))
            parts.append(_apex(values, layout=layout, corner=corner, seed=corner))
        parts.append(_apex(np.full(30, 1.5), layout="c"))  # too few passes to count
        return pd.concat(parts, ignore_index=True)

    def test_the_median_over_the_supporting_corners(self):
        apex = self._car()
        got = lim.car_limit(apex, 20)
        corners = got.by_corner[got.by_corner["supports"]]
        assert got.found and not got.one_sensor
        assert got.corners == 3 and got.passes == 600 and got.layouts == ("a", "b")
        assert got.level == pytest.approx(float(corners["level"].median()))
        assert 1.24 < got.level < 1.36  # above every corner's typical pass
        assert np.isnan(got.lo) and got.to_json()["lo"] is None

    def test_a_car_without_two_readings_is_read_on_the_doppler_alone(self):
        apex = self._car()
        apex["lat_accel_g"] = np.nan
        got = lim.car_limit(apex, 20)
        assert got.found and got.one_sensor and got.corners == 3
        assert got.level == got.raw  # nothing to take the noise out with

    def test_the_cross_reading_has_no_one_sensor_form(self):
        apex = self._car()
        got = lim.car_limit(apex, 20, cross=True)
        assert got.found and got.level == pytest.approx(0.98 * lim.car_limit(apex, 20).level)
        apex["lat_accel_g"] = np.nan
        apex["lat_accel_cross_g"] = np.nan
        assert not lim.car_limit(apex, 20, cross=True).found

    def test_no_supporting_corner_at_all(self):
        got = lim.car_limit(_apex(np.full(30, 1.2)), 20)
        assert not got.found and got.corners == 0 and got.to_json()["level"] is None
        assert not lim.car_limit(pd.DataFrame(), 20).found

    def test_a_day_bootstrap_brackets_the_level(self):
        got = lim.car_limit(self._car(), 20, boot=60)
        assert got.lo < got.level < got.hi
        assert got.hi - got.lo < 0.05
        again = lim.car_limit(self._car(), 20, boot=60)
        assert (got.lo, got.hi) == (again.lo, again.hi)  # seeded
