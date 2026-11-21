"""Where the pit exit joins the track (asked of the track by #206).

An out-lap begins in the pit lane and ends at the start/finish line. Laid
on the track's reference line by counting distance back from the line it
ends on, it sits beside the track (in the pit lane and on the exit road)
until it merges with where the build laps drive. The **join** is the
distance along the lap at which an out-lap comes inside the band the
build laps occupy and stays inside it; the **rolling time** is how long
the car had been moving by then. A layout's figures are the medians over
its out-laps from the pit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import MOVING_MS
from . import geometry as geo
from . import track as trk

#: The band the build laps occupy at a bin is their 1st–99th percentile of
#: lateral offset, widened by this much either side.
BAND_MARGIN_M = 1.0
#: An out-lap has joined once it stays inside the band for this distance.
JOIN_STAY_M = 150.0
MIN_OUTLAP_SAMPLES = 200


def build_band(build: trk.TrackBuild) -> tuple[np.ndarray, np.ndarray]:
    """The band the build laps occupy at each bin: their 1st and 99th
    percentile of lateral offset, widened by :data:`BAND_MARGIN_M`."""
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(build.proj.offset[:, :-1], 1, axis=0) - BAND_MARGIN_M
        hi = np.nanpercentile(build.proj.offset[:, :-1], 99, axis=0) + BAND_MARGIN_M
    return np.asarray(lo), np.asarray(hi)


def join_of_outlap(
    build: trk.TrackBuild,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    band: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float] | None:
    """``(s_join_m, rolling_s)`` of one out-lap given as samples up to the
    line it ends on; None when it never leaves the build laps' band (it
    did not start in the pit lane) or never settles inside it. ``band`` is
    :func:`build_band` of the build, for a caller with many out-laps."""
    model = build.model
    length, ds, n = model.length_m, model.ds_m, model.n
    path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    guess = length - (path[-1] - path)
    rolling = np.isfinite(speed) & (speed >= MOVING_MS)
    if not rolling.any():
        return None
    t_first = float(t[int(np.flatnonzero(rolling)[0])])  # of the whole out-lap
    keep = guess > -trk.PROJECT_WINDOW_M  # the lap before the line, not the pit lane behind it
    if keep.sum() < MIN_OUTLAP_SAMPLES:
        return None
    t, x, y, speed, guess = t[keep], x[keep], y[keep], speed[keep], guess[keep]
    s, off = trk.project_onto(build.ref_x, build.ref_y, x, y, guess, closed=True)
    lo, hi = build_band(build) if band is None else band
    b = np.clip(np.rint(s / ds).astype(int), 0, n - 1)
    inside = (off >= lo[b]) & (off <= hi[b])
    if inside.all():
        return None
    # the first sample from which the lap is inside the band for JOIN_STAY_M
    outside = np.flatnonzero(~inside)
    for k, i in enumerate(outside):
        nxt = outside[k + 1] if k + 1 < len(outside) else len(s) - 1
        if (
            i + 1 < len(s)
            and s[nxt] - s[i + 1] >= JOIN_STAY_M
            or (k + 1 == len(outside) and s[-1] - s[i] >= JOIN_STAY_M)
        ):
            j = i + 1
            if j >= len(s) or t[j] <= t_first:
                return None
            return float(s[j] % length), float(t[j] - t_first)
    return None


def pit_exit(root: Path, build: trk.TrackBuild, *, max_outlaps: int = 60) -> dict:
    """The layout's pit-exit join: ``{"s_join_m", "s_join_p10_m",
    "s_join_p90_m", "rolling_s", "rolling_p10_s", "rolling_p90_s", "n"}``
    over its out-laps from the pit (the dataset's ``outlap_from_pit``), at
    most ``max_outlaps`` of them, the latest first; ``n`` is 0 when none
    could be read."""
    layout = build.model.layout_key
    laps = geo.load_laps(root)
    if "outlap_from_pit" not in laps.columns:
        return {"n": 0}
    sessions = geo.load_sessions(root)
    here = set(sessions.loc[sessions["layout_id"] == layout, "session_id"].astype(str))
    out = laps[
        (laps["lap_type"] == "out")
        & laps["outlap_from_pit"].fillna(False).astype(bool)
        & laps["session_id"].astype(str).isin(here)
    ].sort_values("session_id", ascending=False)
    joins: list[tuple[float, float]] = []
    band = build_band(build)  # the same for every out-lap
    for sid, grp in out.groupby("session_id", sort=False):
        if len(joins) >= max_outlaps:
            break
        geom = geo.load_session_geometry(root, str(sid), layout)
        if geom is None or geom.crossings.size == 0:
            continue
        raw_speed = np.where(np.isfinite(geom.speed), geom.speed, 0.0)
        for num in grp["lap_num"]:
            span = geom.lap_spans.get(int(num))
            if span is None:
                continue
            # the out-lap ends on the first crossing at or after its own end
            after = geom.crossings[geom.crossings >= span[1] - 1.0]
            if after.size == 0 or after[0] - span[1] > 2.0:
                continue
            i0, i1 = np.searchsorted(geom.t, [span[0], after[0]])
            if i1 - i0 < MIN_OUTLAP_SAMPLES or len(np.unique(geom.seg[i0:i1])) != 1:
                continue
            sl = slice(i0, i1)
            hit = join_of_outlap(build, geom.t[sl], geom.x[sl], geom.y[sl], raw_speed[sl], band)
            if hit is not None:
                joins.append(hit)
    if not joins:
        return {"n": 0}
    s_join = np.array([j[0] for j in joins])
    roll = np.array([j[1] for j in joins])
    return {
        "s_join_m": float(np.median(s_join)),
        "s_join_p10_m": float(np.percentile(s_join, 10)),
        "s_join_p90_m": float(np.percentile(s_join, 90)),
        "rolling_s": float(np.median(roll)),
        "rolling_p10_s": float(np.percentile(roll, 10)),
        "rolling_p90_s": float(np.percentile(roll, 90)),
        "n": int(len(joins)),
    }


def pit_table(rows: list[dict]) -> pd.DataFrame:
    """Per-layout pit-exit rows as one frame (for the gate report)."""
    return pd.DataFrame(rows)
