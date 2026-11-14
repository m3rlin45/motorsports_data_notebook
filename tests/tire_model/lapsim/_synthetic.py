"""A synthetic session on a stadium-shaped circuit with a known GPS lag.

The circuit is two 200 m straights joined by two 50 m semicircles, driven
clockwise (right turns), with the start/finish gate in the middle of the
west straight at the registry's Tsukuba gate point. Position samples are
exact; the Doppler quantities (speed, the heading whose one-sample
difference is ``gps_yaw_rate_dps``) are those of ``lag_s`` earlier, as a
receiver whose velocity solution trails its position solution reports them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from motorsports_data_notebook.tire_model.lapsim import geometry as geo
from tests._timeseries_fixture import write_session_timeseries

STRAIGHT_M = 200.0
RADIUS_M = 50.0
LENGTH_M = 2 * STRAIGHT_M + 2 * math.pi * RADIUS_M
LAYOUT_ID = "tsukuba_2000"
DT_S = 0.04
START_S_M = -150.0  # the recording starts this far before the gate


def stadium(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, heading)`` at lap distance ``s`` (heading unwrapped, rad)."""
    s = np.asarray(s, float)
    lap = np.floor(s / LENGTH_M)
    u = s - lap * LENGTH_M
    a, r = STRAIGHT_M, RADIUS_M
    arc = math.pi * r
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    h = np.zeros_like(u)
    m = u < a / 2
    y[m] = u[m]
    m = (u >= a / 2) & (u < a / 2 + arc)
    phi = (u[m] - a / 2) / r
    x[m], y[m], h[m] = r - r * np.cos(phi), a / 2 + r * np.sin(phi), phi
    m = (u >= a / 2 + arc) & (u < 3 * a / 2 + arc)
    x[m], y[m], h[m] = 2 * r, a / 2 - (u[m] - a / 2 - arc), math.pi
    m = (u >= 3 * a / 2 + arc) & (u < 3 * a / 2 + 2 * arc)
    phi = (u[m] - 3 * a / 2 - arc) / r
    x[m], y[m], h[m] = r + r * np.cos(phi), -a / 2 - r * np.sin(phi), math.pi + phi
    m = u >= 3 * a / 2 + 2 * arc
    y[m], h[m] = -a / 2 + (u[m] - 3 * a / 2 - 2 * arc), 2 * math.pi
    return x, y, h + 2 * math.pi * lap


def speed_at(s: np.ndarray) -> np.ndarray:
    """36 m/s in the middle of the straights, 20 m/s in the middle of the corners."""
    return 28.0 + 8.0 * np.cos(4 * math.pi * np.asarray(s, float) / LENGTH_M)


def _to_lat_lon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate = geo.layout_gate(LAYOUT_ID)
    unit = geo.local_xy(np.array([gate.lat + 1.0]), np.array([gate.lon + 1.0]), gate.lat, gate.lon)
    return gate.lat + y / unit[0, 1], gate.lon + x / unit[0, 0]


def synthetic_session(
    *,
    n_laps: int = 5,
    lag_s: float = 0.1,
    mirror: bool = False,
    gyro_scale: float = 1.0,
    gyro_offset_dps: float = 0.0,
    noise_m: float = 0.0,
    slew: tuple[int, float] | None = None,
    accel_delay_s: float | None = None,
    pace: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """One recording: ``START_S_M`` before the gate, ``n_laps`` full laps and
    100 m of an in-lap. ``mirror`` drives it anticlockwise. ``slew`` is
    ``(lap_num, metres)``: a position excursion of that size, 0.6 s long, in
    the middle of the lap's second straight. ``accel_delay_s`` adds an
    accelerometer (``lat_g``, ``long_g``) that reports the true ground-frame
    accelerations that much late. ``pace`` multiplies the lap time (1.1: a
    wet session). Returns the per-sample frame
    (GPS and gyro columns on one 25 Hz clock) and the truth."""
    fine = 0.001
    total_m = n_laps * LENGTH_M + 100.0 - START_S_M
    steps = int(total_m / 20.0 / fine) + 1
    s_fine = np.empty(steps)
    s_fine[0] = START_S_M
    k = 4 * math.pi / LENGTH_M
    pos = START_S_M
    for i in range(1, steps):
        pos += (28.0 + 8.0 * math.cos(k * pos)) / pace * fine
        s_fine[i] = pos
    s_fine = s_fine[s_fine <= n_laps * LENGTH_M + 100.0]
    t_fine = np.arange(s_fine.size) * fine
    t = np.arange(0.0, t_fine[-1], DT_S)
    s = np.interp(t, t_fine, s_fine)
    s_late = np.interp(np.maximum(t - lag_s, 0.0), t_fine, s_fine)
    x, y, h = stadium(s)
    _, _, h_late = stadium(s_late)
    v_late = speed_at(s_late) / pace
    sign = -1.0 if mirror else 1.0
    x, h, h_late = sign * x, sign * h, sign * h_late
    rng = np.random.default_rng(seed)
    if noise_m:
        x = x + rng.normal(0.0, noise_m, x.size)
        y = y + rng.normal(0.0, noise_m, y.size)
    lap_num = np.where(s >= 0.0, np.floor(s / LENGTH_M) + 1, 0).astype(int)
    if slew is not None:
        slew_lap, size = slew
        at = (slew_lap - 1) * LENGTH_M + STRAIGHT_M + math.pi * RADIUS_M
        t0 = float(np.interp(at, s_fine, t_fine))
        x = x + size * np.exp(-(((t - t0) / 0.3) ** 2))
    lat, lon = _to_lat_lon(x, y)
    yaw = np.zeros_like(t)
    yaw[1:] = np.degrees(np.diff(h_late)) / DT_S
    t_lap = np.empty_like(t)
    for num in np.unique(lap_num):
        m = lap_num == num
        t_lap[m] = t[m] - t[m][0]
    true_yaw = sign * np.degrees(np.gradient(stadium(s)[2], t))
    frame = pd.DataFrame(
        {
            "lap_num": lap_num,
            "t_session_s": t,
            "t_lap_s": t_lap,
            "gps_lat": lat,
            "gps_lon": lon,
            "gps_alt_m": np.full(t.size, 20.0),
            "speed_ms": v_late,
            "gps_yaw_rate_dps": yaw,
            "gps_lat_g": v_late * np.radians(yaw) / 9.81,
            "yaw_rate_dps": gyro_scale * true_yaw + gyro_offset_dps,
        }
    )
    if accel_delay_s is not None:
        s_acc = np.interp(np.maximum(t - accel_delay_s, 0.0), t_fine, s_fine)
        v_acc = speed_at(s_acc) / pace
        h_acc = stadium(s_acc)[2]
        frame["lat_g"] = sign * v_acc * np.gradient(h_acc, t) / 9.80665
        frame["long_g"] = np.gradient(v_acc, t) / 9.80665
    crossings = np.interp(np.arange(n_laps + 1) * LENGTH_M, s_fine, t_fine)
    truth = {
        "t": t,
        "s": s,
        "heading": h,
        "speed": speed_at(s) / pace,
        "crossings": crossings,
        "lag_s": lag_s,
        "n_laps": n_laps,
    }
    return frame, truth


def lap_rows(session_id: str, truth: dict) -> pd.DataFrame:
    """The dataset's lap rows for a synthetic session: an out-lap, the full
    laps and an in-lap."""
    c = truth["crossings"]
    n = truth["n_laps"]
    times = [float(c[0])] + [float(d) for d in np.diff(c)] + [float(truth["t"][-1] - c[-1])]
    return pd.DataFrame(
        {
            "session_id": session_id,
            "lap_num": np.arange(n + 2, dtype=np.int16),
            "stint_id": np.ones(n + 2, dtype=np.int16),
            "lap_time_s": np.asarray(times, dtype=np.float32),
            "lap_type": ["out"] + ["full"] * n + ["in"],
            "on_track_s": np.asarray(times, dtype=np.float32),
            "outlap_from_pit": [True] + [False] * (n + 1),
        }
    )


def write_dataset(root: Path, sessions: dict[str, dict]) -> None:
    """A dataset root holding synthetic sessions. ``sessions`` maps a
    session id to ``{"car", "date", "kwargs"}`` (``kwargs`` for
    :func:`synthetic_session`)."""
    session_rows = []
    laps = []
    for sid, spec in sessions.items():
        frame, truth = synthetic_session(**spec.get("kwargs", {}))
        write_session_timeseries(root, sid, frame, gyro_source="imu")
        laps.append(lap_rows(sid, truth))
        session_rows.append(
            {
                "session_id": sid,
                "date": pd.Timestamp(spec["date"]).date(),
                "session_start_utc": pd.Timestamp(spec["date"], tz="UTC"),
                "car": spec["car"],
                "driver": "driver",
                "logger_id": spec.get("logger_id", "1"),
                "layout_id": LAYOUT_ID,
                "track_canonical": LAYOUT_ID,
                "lap_length_m": LENGTH_M,
                "lap_source": "logger",
                "accel_source": "imu",
                "kin_imu_lat_slope": 1.0,
                "gyro_source": "imu",
                "status": "ok",
            }
        )
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "laps").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(session_rows)), root / "sessions/2026-01.parquet"
    )
    pq.write_table(
        pa.Table.from_pandas(pd.concat(laps, ignore_index=True)), root / "laps/2026-01.parquet"
    )
