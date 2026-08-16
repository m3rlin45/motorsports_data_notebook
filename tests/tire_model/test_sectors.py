"""Tests for the per-sector pace→energy model (tire_model.sectors)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from motorsports_data_notebook.tire_model import sectors
from motorsports_data_notebook.tire_model.sectors import (
    N_SECTORS,
    build_pace_model,
    compute_sector_table,
)


def _write_session_ts(
    root: Path,
    session_id: str,
    *,
    n_laps: int,
    lap_time_s: float,
    g_level: float,
    hz: float = 10.0,
    bad_sector_lap: int | None = None,
    jitter_s: float = 0.0,
) -> None:
    """Synthetic timeseries: constant speed, constant |g| per lap.

    ``bad_sector_lap`` (1-based) gets its middle time-third wrecked: near-zero
    g at crawl speed — the 'one bad turn on a good lap' scenario.
    ``jitter_s`` ramps lap time by that much per lap so the session has real
    pace spread (g scales down as laps slow, mimicking real driving).
    """
    rows = {k: [] for k in ["lap_num", "t_lap_s", "speed_ms", "lat_g", "long_g"]}
    for lap in range(1, n_laps + 1):
        this_lap_time = lap_time_s + (lap - 1) * jitter_s
        this_g = g_level * (lap_time_s / this_lap_time) ** 2
        n = int(this_lap_time * hz)
        t = np.arange(n) / hz
        speed = np.full(n, 40.0)
        g = np.full(n, this_g / np.sqrt(2.0))  # split evenly lat/long
        if lap == bad_sector_lap:
            third = n // 3
            g[third : 2 * third] = 0.05  # crawled through the middle sector
            speed[third : 2 * third] = 15.0
        rows["lap_num"].extend([lap] * n)
        rows["t_lap_s"].extend(t.tolist())
        rows["speed_ms"].extend(speed.tolist())
        rows["lat_g"].extend(g.tolist())
        rows["long_g"].extend(g.tolist())
    out = root / "timeseries" / "2026-01"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out / f"{session_id}.parquet")


def _laps_meta(session_id: str, n_laps: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": [session_id] * n_laps,
            "lap_num": list(range(1, n_laps + 1)),
            "track_canonical": ["track_x"] * n_laps,
            "car": ["ToyCar"] * n_laps,
            "condition": ["dry"] * n_laps,
        }
    )


@pytest.fixture(autouse=True)
def _clear_sector_cache():
    sectors._SECTOR_CACHE.clear()
    yield
    sectors._SECTOR_CACHE.clear()


class TestComputeSectorTable:
    def test_splits_laps_into_sectors_with_consistent_g2(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "sess1", n_laps=3, lap_time_s=60.0, g_level=1.0)
        tbl = compute_sector_table(tmp_path)
        assert len(tbl) == 3 * N_SECTORS
        # Constant speed + constant g: each sector ≈ a third of the lap,
        # g² ≈ g_level² everywhere.
        assert tbl["t_s"].sum() == pytest.approx(3 * 60.0, rel=0.05)
        assert tbl["g2_s"].to_numpy() == pytest.approx(np.full(9, 1.0), rel=0.05)

    def test_bad_sector_isolated_from_good_sectors(self, tmp_path: Path) -> None:
        _write_session_ts(
            tmp_path, "sess1", n_laps=1, lap_time_s=60.0, g_level=1.0, bad_sector_lap=1
        )
        tbl = compute_sector_table(tmp_path)
        by_sector = tbl.set_index("sector")["g2_s"]
        # The wrecked span drags only its own (distance) sector down while
        # the neighbors keep the lap's true intensity.
        assert by_sector.min() < 0.5
        assert by_sector.max() > 0.9

    def test_cache_reused(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "sess1", n_laps=1, lap_time_s=60.0, g_level=1.0)
        first = compute_sector_table(tmp_path)
        assert compute_sector_table(tmp_path) is first


class TestBuildPaceModel:
    def test_curve_decreases_with_lap_time(self, tmp_path: Path) -> None:
        # Two pace populations: fast laps at higher g, slow laps at lower g.
        metas = []
        for i, (lap_time, g) in enumerate([(58.0, 1.1), (60.0, 1.0), (63.0, 0.85)]):
            sid = f"sess{i}"
            _write_session_ts(tmp_path, sid, n_laps=12, lap_time_s=lap_time, g_level=g)
            metas.append(_laps_meta(sid, 12))
        laps = pd.concat(metas, ignore_index=True)
        curves, default_exp = build_pace_model(tmp_path, laps)
        curve = curves[("track_x", "ToyCar", "dry")]
        g2 = curve["g2"]
        assert g2[0] > g2[-1]  # faster laps carry more energy
        assert curve["lap_time_s"][0] < curve["lap_time_s"][-1]
        assert 0.0 <= default_exp <= 6.0

    def test_too_few_laps_no_curve(self, tmp_path: Path) -> None:
        _write_session_ts(tmp_path, "sess1", n_laps=5, lap_time_s=60.0, g_level=1.0)
        curves, _ = build_pace_model(tmp_path, _laps_meta("sess1", 5))
        assert curves == {}

    def test_excluded_sessions_do_not_leak(self, tmp_path: Path) -> None:
        # Only sess0 in the laps frame: sess1's laps must not shape the curve.
        _write_session_ts(tmp_path, "sess0", n_laps=30, lap_time_s=60.0, g_level=1.0, jitter_s=0.1)
        _write_session_ts(tmp_path, "sess1", n_laps=30, lap_time_s=45.0, g_level=2.0, jitter_s=0.1)
        curves, _ = build_pace_model(tmp_path, _laps_meta("sess0", 30))
        curve = curves[("track_x", "ToyCar", "dry")]
        assert curve["lap_time_s"][0] > 50.0  # sess1's 45 s laps absent
        assert max(curve["g2"]) < 1.5

    def test_empty_timeseries_dir(self, tmp_path: Path) -> None:
        (tmp_path / "timeseries").mkdir()
        curves, default_exp = build_pace_model(tmp_path, _laps_meta("sess0", 3))
        assert curves == {}
        assert default_exp == 3.0
