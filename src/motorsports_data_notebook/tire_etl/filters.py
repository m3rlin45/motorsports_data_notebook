"""Lap-level filters for tire modeling.

Adds ``tire_usable`` and ``exclude_reason`` columns to the per-lap summary
table. Uses only columns that already exist on that table — so this is a
post-aggregation pass, cheap to re-run in notebooks without touching the
timeseries data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class FilterConfig:
    # Out-laps are tire-model data: the pit-exit TPMS reading is the stint's
    # initial condition and the lap's own g²/moving time make it an ordinary
    # warm-up observation. They are judged on moving time, not lap time
    # (the lap time includes the grid / pit-lane wait).
    exclude_outlap: bool = False
    min_outlap_moving_s: float = 60.0
    exclude_inlap: bool = True
    min_lap_time_s: float = 30.0
    max_lap_time_vs_best_pct: float = 1.40
    require_tpms: bool = True
    tpms_stuck_tolerance_bar: float = 0.003
    min_on_track_s: float = 25.0
    min_speed_kmh_mean: float = 40.0


def apply_filters(laps: pa.Table, cfg: FilterConfig) -> pa.Table:
    """Return ``laps`` with ``tire_usable`` and ``exclude_reason`` columns filled.

    ``laps`` must already contain: ``lap_num, stint_id, lap_time_s, is_outlap,
    is_inlap, speed_kmh_mean, on_track_s``, plus per-corner
    ``tpms_press_{c}_min`` / ``_max``; ``moving_s`` when out-laps are kept.
    """
    n = len(laps)
    if n == 0:
        return laps.append_column("tire_usable", pa.array([], type=pa.bool_())).append_column(
            "exclude_reason", pa.array([], type=pa.string())
        )

    reasons: list[str | None] = [None] * n
    lap_time = laps.column("lap_time_s").to_numpy().astype(np.float64)
    is_outlap = laps.column("is_outlap").to_pylist()
    is_inlap = laps.column("is_inlap").to_pylist()
    on_track = laps.column("on_track_s").to_numpy().astype(np.float64)
    speed_mean = laps.column("speed_kmh_mean").to_numpy().astype(np.float64)
    moving = (
        laps.column("moving_s").to_numpy().astype(np.float64)
        if "moving_s" in laps.schema.names
        else on_track
    )

    # Per-stint best lap time (ignoring NaN).
    stint_ids = laps.column("stint_id").to_numpy()
    best_by_stint: dict[int, float] = {}
    for i in range(n):
        t = lap_time[i]
        if is_outlap[i] or np.isnan(t) or t < cfg.min_lap_time_s:
            continue
        sid = int(stint_ids[i])
        if sid not in best_by_stint or t < best_by_stint[sid]:
            best_by_stint[sid] = t

    # TPMS stuck / missing check: need at least one corner to span > tol.
    def _corner_range(i: int, corner: str) -> float:
        col_min = f"tpms_press_{corner}_min"
        col_max = f"tpms_press_{corner}_max"
        if col_min not in laps.schema.names or col_max not in laps.schema.names:
            return float("nan")
        lo = laps.column(col_min)[i].as_py()
        hi = laps.column(col_max)[i].as_py()
        if lo is None or hi is None:
            return float("nan")
        if np.isnan(lo) or np.isnan(hi):
            return float("nan")
        return float(hi - lo)

    for i in range(n):
        if is_outlap[i]:
            if cfg.exclude_outlap:
                reasons[i] = "outlap"
                continue
            # An out-lap is judged on rolling time only: its lap time and
            # mean speed include the wait before the car moves.
            if np.isnan(moving[i]) or moving[i] < cfg.min_outlap_moving_s:
                reasons[i] = "outlap_too_short"
                continue
        else:
            if cfg.exclude_inlap and is_inlap[i]:
                reasons[i] = "inlap"
                continue
            t = lap_time[i]
            if np.isnan(t) or t < cfg.min_lap_time_s:
                reasons[i] = "lap_too_short"
                continue
            sid = int(stint_ids[i])
            best = best_by_stint.get(sid)
            if best is not None and t > best * cfg.max_lap_time_vs_best_pct:
                reasons[i] = "lap_too_slow"
                continue
            ot = on_track[i]
            if not np.isnan(ot) and ot < cfg.min_on_track_s:
                reasons[i] = "on_track_too_short"
                continue
            sm = speed_mean[i]
            if not np.isnan(sm) and sm < cfg.min_speed_kmh_mean:
                reasons[i] = "speed_too_low"
                continue
        if cfg.require_tpms:
            ranges = [_corner_range(i, c) for c in ("fl", "fr", "rl", "rr")]
            finite = [r for r in ranges if not np.isnan(r)]
            if not finite:
                reasons[i] = "no_tpms"
                continue
            if max(finite) < cfg.tpms_stuck_tolerance_bar:
                reasons[i] = "tpms_stuck"
                continue

    usable = [r is None for r in reasons]
    return laps.append_column("tire_usable", pa.array(usable, type=pa.bool_())).append_column(
        "exclude_reason", pa.array(reasons, type=pa.string())
    )
