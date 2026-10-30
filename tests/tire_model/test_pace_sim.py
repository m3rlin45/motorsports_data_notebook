"""Tests for the simulated-lap pace model (tire_model.pace_sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from motorsports_data_notebook.tire_model.heat_input import CarFacts, ShareParams
from motorsports_data_notebook.tire_model.pace_sim import (
    CORNERS,
    GRID_FACTORS,
    LapSamples,
    build_pace_model,
    corner_heat_means,
    fast_lap_check,
    lap_heat_ratios,
    load_lap_samples,
    scaled_lap,
    summarize_fast_lap_check,
)


def _share(g_transfer: float = 2.0) -> ShareParams:
    cf = CarFacts(p_f=0.5, brake_bias_front=0.6, driven="rear")
    return ShareParams(
        p_f=cf.p_f,
        g_transfer_front=g_transfer,
        g_transfer_rear=g_transfer,
        beta0=cf.beta0,
        beta1=0.0,
        eps_drive=1.0,
        eps_brake=1.0,
        driven="rear",
    )


def _synthetic_lap(
    lap_time_s: float = 60.0, g_level: float = 1.0, hz: float = 10.0, right_turns_only: bool = False
) -> LapSamples:
    """Half the lap cornering at |lat| = g_level (alternating direction unless
    ``right_turns_only``), braking into each corner, accelerating out."""
    n = int(lap_time_s * hz)
    t = np.arange(n) / hz
    phase = (t / lap_time_s * 4) % 1.0  # four corner/straight cycles
    lat = np.where(phase < 0.5, g_level, 0.0)
    if not right_turns_only:
        sign = np.where(((t / lap_time_s * 4).astype(int) % 2) == 0, 1.0, -1.0)
        lat = lat * sign
    lng = np.where((phase >= 0.5) & (phase < 0.65), -0.8, np.where(phase >= 0.65, 0.3, 0.0))
    return LapSamples(np.full(n, 1.0 / hz), np.full(n, 40.0), lat, lng)


def _write_session_ts(root: Path, session_id: str, laps: dict[int, LapSamples]) -> None:
    rows = {k: [] for k in ["lap_num", "t_lap_s", "speed_ms", "lat_g", "long_g"]}
    for lap_num, s in laps.items():
        t = np.cumsum(s.dt_s) - s.dt_s[0]
        rows["lap_num"].extend([lap_num] * len(t))
        rows["t_lap_s"].extend(t.tolist())
        rows["speed_ms"].extend(s.speed_ms.tolist())
        rows["lat_g"].extend(s.lat_g.tolist())
        rows["long_g"].extend(s.long_g.tolist())
    out = root / "timeseries" / "2026-01"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / f"{session_id}.parquet")


def _laps_meta(session_id: str, lap_times: list[float], car: str = "ToyCar") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": session_id,
            "lap_num": list(range(1, len(lap_times) + 1)),
            "track_canonical": "track_x",
            "car": car,
            "condition": "dry",
            "on_track_s": lap_times,
        }
    )


class TestScaling:
    def test_scaled_lap_is_uniform(self) -> None:
        lap = _synthetic_lap()
        faster = scaled_lap(lap, 1.05)
        assert faster.lap_time_s == pytest_approx(lap.lap_time_s / 1.05)
        np.testing.assert_allclose(faster.speed_ms, lap.speed_ms * 1.05)
        np.testing.assert_allclose(faster.lat_g, lap.lat_g * 1.05**2)
        np.testing.assert_array_equal(faster.long_g, lap.long_g)

    def test_pure_cornering_heat_scales_as_speed_cubed_without_load_transfer(self) -> None:
        # |F| ∝ lat ∝ m², V ∝ m → q ∝ m³ when the axle split does not move
        # (g_transfer → ∞ keeps the inner/outer shares at ½).
        lap = _synthetic_lap()
        lap = LapSamples(lap.dt_s, lap.speed_ms, lap.lat_g, np.zeros_like(lap.long_g))
        ratios = lap_heat_ratios(_share(g_transfer=1e9), lap, [lap.lap_time_s / 1.1])
        assert ratios is not None
        np.testing.assert_allclose(ratios[0], 1.1**3, rtol=1e-9)

    def test_load_transfer_loads_the_outer_tyres_more(self) -> None:
        # Right turns only: lat_pos loads the LEFT tyres (share_heat_rate
        # convention). The loaded side's share grows with |lat|, so its
        # heat rises faster than m³; the unloaded side's slower.
        lap = _synthetic_lap(right_turns_only=True)
        lap = LapSamples(lap.dt_s, lap.speed_ms, lap.lat_g, np.zeros_like(lap.long_g))
        ratios = lap_heat_ratios(_share(g_transfer=1.0), lap, [lap.lap_time_s / 1.1])
        assert ratios is not None
        fl, fr, rl, rr = ratios[0]
        assert fl > 1.1**3 > fr
        assert rl > 1.1**3 > rr

    def test_held_longitudinal_force_damps_every_corner_below_m_cubed(self) -> None:
        # Braking and traction g are held at their limits, so the part of
        # each corner's force they supply does not grow with pace.
        lap = _synthetic_lap()
        ratios = lap_heat_ratios(_share(g_transfer=1e9), lap, [lap.lap_time_s / 1.1])
        assert ratios is not None
        assert np.all(ratios[0] < 1.1**3) and np.all(ratios[0] > 1.1)

    def test_ratio_is_one_at_own_lap_time_and_zero_heat_lap_is_rejected(self) -> None:
        lap = _synthetic_lap()
        ratios = lap_heat_ratios(_share(), lap, [lap.lap_time_s])
        assert ratios is not None
        np.testing.assert_allclose(ratios[0], 1.0, rtol=1e-12)
        still = LapSamples(
            lap.dt_s, lap.speed_ms, np.zeros_like(lap.lat_g), np.zeros_like(lap.long_g)
        )
        assert lap_heat_ratios(_share(), still, [50.0]) is None
        assert np.all(corner_heat_means(_share(), still) == 0.0)


class TestBuildPaceModel:
    def test_curve_shape_and_direction(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "s0", {i: _synthetic_lap(60.0 + 0.1 * i) for i in range(1, 9)})
        laps = _laps_meta("s0", [60.0 + 0.1 * i for i in range(1, 9)])
        curves = build_pace_model(tmp_path, laps, {"ToyCar": _share()})
        curve = curves[("track_x", "ToyCar", "dry")]
        assert len(curve["lap_time_s"]) == len(GRID_FACTORS)
        assert curve["lap_time_s"][0] < curve["lap_time_s"][-1]
        assert curve["g2"][0] > 1.0 > curve["g2"][-1]  # faster laps carry more heat
        assert set(curve["g2_by_corner"]) == set(CORNERS)
        assert all(len(v) == len(GRID_FACTORS) for v in curve["g2_by_corner"].values())
        mean = np.mean([curve["g2_by_corner"][c] for c in CORNERS], axis=0)
        np.testing.assert_allclose(curve["g2"], mean, atol=2e-5)
        assert curve["n_laps"] == 8 and curve["method"] == "uniform_scaled_lap_shares"

    def test_excluded_sessions_do_not_leak(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "s0", {i: _synthetic_lap(60.0) for i in range(1, 7)})
        _write_session_ts(
            tmp_path, "s1", {i: _synthetic_lap(45.0, g_level=2.0) for i in range(1, 7)}
        )
        laps = _laps_meta("s0", [60.0] * 6)
        curve = build_pace_model(tmp_path, laps, {"ToyCar": _share()})[("track_x", "ToyCar", "dry")]
        assert curve["lap_time_s"][0] == pytest_approx(
            60.0 * GRID_FACTORS[0]
        )  # s1's 45 s laps absent
        assert curve["n_laps"] == 6

    def test_too_few_laps_or_no_shares_gives_no_curve(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "s0", {i: _synthetic_lap(60.0) for i in range(1, 4)})
        laps = _laps_meta("s0", [60.0] * 3)
        assert build_pace_model(tmp_path, laps, {"ToyCar": _share()}) == {}
        _write_session_ts(tmp_path, "s1", {i: _synthetic_lap(60.0) for i in range(1, 9)})
        assert (
            build_pace_model(tmp_path, _laps_meta("s1", [60.0] * 8), {"OtherCar": _share()}) == {}
        )

    def test_no_timeseries(self, tmp_path: Path) -> None:
        (tmp_path / "timeseries").mkdir()
        assert build_pace_model(tmp_path, _laps_meta("s0", [60.0] * 8), {"ToyCar": _share()}) == {}
        assert load_lap_samples(tmp_path) == {}


class TestFastLapCheck:
    def test_simulation_consistent_data_scores_unbiased(self, tmp_path: Path) -> None:
        # A session whose fast laps ARE uniformly scaled median laps: the
        # observed per-lap heat ratios equal the simulated ones.
        share = _share()
        base = _synthetic_lap(60.0)
        laps_ts = {i: base for i in range(1, 9)}
        for i in (9, 10):
            laps_ts[i] = scaled_lap(base, 60.0 / 58.0)
        _write_session_ts(tmp_path, "s0", laps_ts)
        samples = load_lap_samples(tmp_path)
        # on_track_s as the loaded samples see it (the first sample's dt is 0)
        meta = _laps_meta("s0", [samples[("s0", n)].lap_time_s for n in range(1, 11)])
        for c in CORNERS:
            meta[f"q_lap_{c}"] = [
                corner_heat_means(share, samples[("s0", int(n))])[CORNERS.index(c)]
                for n in meta["lap_num"]
            ]
        frame = fast_lap_check(meta, samples, {"ToyCar": share})
        assert len(frame) == 1
        summary = summarize_fast_lap_check(frame)
        assert set(summary["corner"]) == set(CORNERS)
        np.testing.assert_allclose(summary["bias_log"], 0.0, atol=1e-9)
        assert (summary["mean_log_obs"] > 0).all()

    def test_sessions_without_fast_laps_are_skipped(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "s0", {i: _synthetic_lap(60.0) for i in range(1, 9)})
        meta = _laps_meta("s0", [60.0] * 8)
        for c in CORNERS:
            meta[f"q_lap_{c}"] = 1.0
        assert fast_lap_check(meta, load_lap_samples(tmp_path), {"ToyCar": _share()}).empty
        assert summarize_fast_lap_check(pd.DataFrame()).empty


def pytest_approx(v: float, rel: float = 1e-9):
    import pytest

    return pytest.approx(v, rel=rel)
