"""Write a schema-7 session timeseries file from a flat per-sample frame.

Model-side tests build synthetic sessions as one wide frame (``lap_num``,
``t_session_s`` or ``t_lap_s``, ``speed_ms``, ``lat_g``, ``long_g``, TPMS
columns, ...). This splits it into one timebase per sensor family, all on the
frame's clock, and writes it the way the extractor does, with a verdict that
trusts the IMU, so the production loaders (``load_group``,
``load_kinematics``) read it back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from motorsports_data_notebook.tire_etl.dataset import (
    FAMILIES,
    build_timeseries_table,
    channel_family,
    timeseries_summary,
    write_timeseries,
)
from motorsports_data_notebook.tire_etl.kinematics import KinematicsVerdict


def write_session_timeseries(
    root: Path,
    session_id: str,
    frame: pd.DataFrame | pa.Table,
    *,
    partition: str = "2026-01",
    accel_source: str = "imu",
    gyro_source: str = "none",
) -> Path:
    df = frame.to_pandas() if isinstance(frame, pa.Table) else frame.copy()
    df = df.reset_index(drop=True)
    n = len(df)
    lap = (
        df["lap_num"].to_numpy(dtype=np.int64)
        if "lap_num" in df.columns
        else np.ones(n, dtype=np.int64)
    )
    if "t_session_s" in df.columns:
        t_sess = df["t_session_s"].to_numpy(dtype=np.float64)
    else:
        # Stack the laps end to end on a session clock.
        t_lap = df["t_lap_s"].to_numpy(dtype=np.float64)
        t_sess = np.empty(n)
        offset = 0.0
        for ln in sorted(set(lap.tolist())):
            m = lap == ln
            t_sess[m] = t_lap[m] + offset
            offset += float(np.nanmax(t_lap[m])) + 1.0
    if "t_lap_s" in df.columns:
        t_lap_s = df["t_lap_s"].to_numpy(dtype=np.float64)
    else:
        t_lap_s = np.empty(n)
        for ln in set(lap.tolist()):
            m = lap == ln
            t_lap_s[m] = t_sess[m] - t_sess[m].min()
    stint = df["stint_id"].to_numpy(dtype=np.int64) if "stint_id" in df.columns else np.ones(n, int)
    order = np.lexsort((t_sess, lap))
    sample_idx = np.empty(n, dtype=np.int64)
    for ln in set(lap.tolist()):
        m = np.flatnonzero(lap[order] == ln)
        sample_idx[order[m]] = np.arange(len(m))
    keys = {
        "lap_num": pa.array(lap.astype(np.int16)),
        "stint_id": pa.array(stint.astype(np.int16)),
        "sample_idx": pa.array(sample_idx.astype(np.int32)),
        "t_session_s": pa.array(t_sess, type=pa.float64()),
        "t_lap_s": pa.array(t_lap_s.astype(np.float32)),
    }
    parts = []
    for fam in FAMILIES:
        cols = [c for c in df.columns if channel_family(c) == fam]
        if not cols:
            continue
        tbl = dict(keys)
        for c in cols:
            typ = pa.float64() if c in ("gps_lat", "gps_lon") else pa.float32()
            tbl[c] = pa.array(df[c].to_numpy(dtype=np.float64), type=typ)
        parts.append((fam, fam, pa.table(tbl)))
    table = build_timeseries_table(session_id, parts)
    out = root / "timeseries" / partition
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{session_id}.parquet"
    verdict = KinematicsVerdict(accel_source=accel_source, gyro_source=gyro_source, ref_lap=None)
    write_timeseries(table, path, summary=timeseries_summary(table), kinematics=verdict.to_json())
    return path
