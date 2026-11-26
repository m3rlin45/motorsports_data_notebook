"""A session's heat input in the tyres' frame, from GPS and the track (#376).

The pressure model's heat is production's chain on a lateral and a
longitudinal acceleration. By default those are the series the sensor
verdict chose (the accelerometer, or the GPS-derived fallback,
:func:`~...tire_etl.kinematics.load_kinematics`). This module rebuilds them
in the frame a simulated lap is in, so that a fit on them and a simulated
lap served to them need no frame factor and no sensor verdict:

- lateral: what the tyres hold on the road's bank, ``V·Δψ/Δt / g`` from the
  Doppler course laid on the track (``L·cos b − sin b``,
  :func:`.bank.to_tyres_frame`);
- longitudinal: ``ΔV/Δt / g`` plus the track's grade;

both on the position clock (:func:`.frame.ground_series`), over the whole
session: out-lap, flying laps, in-lap and pit lane. A stretch of at most
:data:`FILL_MAX_S` without a value is bridged; a longer one, and one over a
fix the extractor discarded, is a hole. A fix more than
:data:`OFF_LINE_M` from the track's reference line (the pit lane, the
paddock), and every fix of a layout without a committed track, is in the
ground frame with bank and grade 0 (``on_line`` false). Where the car stands the course is
undefined and the car's own acceleration is 0 (on the line the tyres still
hold the road's slope; the heat, the speed times the force, is 0 there).

Nothing here touches the extractor: it is computed at query time from the
stored GPS family and the committed track JSON, and handed to the fit on
the kinematics clock (:func:`tyres_kinematics`) with the same columns
``load_kinematics`` has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from . import bank as bankmod
from . import frame as frm
from . import geometry as geo
from . import track as trk

if TYPE_CHECKING:
    from .track import TrackModel

#: A fix further than this from the reference line is off the track (m):
#: the pit lane and the paddock. The line is the one driven, and a car
#: runs up to about 10 m to the side of it on a wide straight.
OFF_LINE_M = 15.0
#: Below this speed the car stands: the course is undefined and the heat
#: input is 0 (m/s; the fit's own threshold, ``statespace.MOVING_SPEED_MS``).
STILL_SPEED_MS = 5.0 / 3.6
#: The longest stretch without a usable value that is bridged (s), one bin
#: of production's chain: a kinematics row between two usable values at most
#: this far apart is linear between them (the Doppler course is undefined
#: for a few fixes now and then, and each of its 0.5 s differences then
#: lacks a value for about 0.3 s, at speed), and a row before the first or
#: after the last usable value of the session by at most this much holds it
#: (a file that starts or ends at speed: the differences and the velocity
#: lag leave its first and last fixes without one). Anything longer, and
#: any stretch over a fix the extractor marked as discarded, is a hole.
FILL_MAX_S = 1.0

#: Source labels of the rows :func:`tyres_kinematics` hands out: one source
#: for every row with a value (the ``on_line`` column says which of them are
#: laid on the track's bank and grade), and the holes.
TYRES = "tyres"
DISCARD = "discard"  # moving with no usable fix: a hole in the heat input


@dataclass
class TyresSeries:
    """A session's accelerations in the tyres' frame on the position clock."""

    t: np.ndarray
    seg: np.ndarray
    lat_g: np.ndarray
    long_g: np.ndarray
    on_line: np.ndarray  # bool: laid on the track's line (bank and grade applied)
    #: bool: a fix the extractor marked as discarded (never bridged)
    hard: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def project_session(
    track: "TrackModel", x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(s, offset)`` of every fix on ``track``'s closed reference line,
    with no lap needed: the nearest node is the guess and
    :func:`.track.project_onto` drops the fix onto the line near it.
    ``offset`` is positive to the right of the direction of travel; both
    are NaN where the fix is not finite."""
    from scipy.spatial import cKDTree

    n = len(x)
    s = np.full(n, np.nan)
    off = np.full(n, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any() or track.n < 3:
        return s, off
    tree = cKDTree(np.c_[track.x_m, track.y_m])
    _, near = tree.query(np.c_[x[ok], y[ok]])
    guess = np.asarray(near, float) * track.ds_m
    xr = np.concatenate([track.x_m, track.x_m[:1]])
    yr = np.concatenate([track.y_m, track.y_m[:1]])
    along, lateral = trk.project_onto(xr, yr, x[ok], y[ok], guess, closed=True)
    s[ok] = np.mod(along, track.length_m)
    off[ok] = lateral
    return s, off


def session_series(geom: geo.SessionGeometry, track: "TrackModel | None") -> TyresSeries:
    """:class:`TyresSeries` of one session's geometry, laid on ``track``
    (the ground frame everywhere when it is None)."""
    lat, lng = frm.ground_series(geom)
    lat = np.asarray(lat, float).copy()
    lng = np.asarray(lng, float).copy()
    still = np.nan_to_num(geom.speed, nan=0.0) < STILL_SPEED_MS
    lat[still & ~np.isfinite(lat)] = 0.0
    lng[still & ~np.isfinite(lng)] = 0.0
    on_line = np.zeros(len(geom), dtype=bool)
    if track is not None and len(geom):
        s, off = project_session(track, geom.x, geom.y)
        on_line = np.isfinite(s) & (np.abs(np.nan_to_num(off, nan=np.inf)) <= OFF_LINE_M)
        if on_line.any():
            nodes = track.s_m
            left = bankmod.road_arrays(track)[0]
            grade = np.nan_to_num(np.asarray(track.grade, float))
            at = np.interp(s[on_line], nodes, left, period=track.length_m)
            up = np.interp(s[on_line], nodes, grade, period=track.length_m)
            lat[on_line] = bankmod.to_tyres_frame(lat[on_line], at)
            lng[on_line] = lng[on_line] + up
    disc = (
        np.asarray(geom.discard, bool)
        if len(geom.discard) == len(geom)
        else np.zeros(len(geom), dtype=bool)
    )
    lat[disc] = np.nan
    lng[disc] = np.nan
    return TyresSeries(t=geom.t, seg=geom.seg, lat_g=lat, long_g=lng, on_line=on_line, hard=disc)


def on_clock(series: TyresSeries, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lat_g, long_g, on_line)`` of ``series`` at the times ``t``, from its
    usable fixes (both accelerations finite): linear between the two either
    side when they are at most :data:`FILL_MAX_S` apart and no discarded fix
    lies between them; the first (last) usable value for a time before (after)
    it by at most :data:`FILL_MAX_S`; NaN elsewhere. ``on_line`` is the nearer
    usable fix's."""
    n = len(t)
    lat = np.full(n, np.nan)
    lng = np.full(n, np.nan)
    on = np.zeros(n, dtype=bool)
    ok = np.isfinite(series.lat_g) & np.isfinite(series.long_g)
    idx = np.flatnonzero(ok)
    if len(idx) == 0 or n == 0:
        return lat, lng, on
    ts = series.t[idx]
    la, lo_, ln_ = series.lat_g[idx], series.long_g[idx], series.on_line[idx]
    hard = series.hard if len(series.hard) == len(series.t) else np.zeros(len(series.t), bool)
    n_hard = np.concatenate([[0], np.cumsum(hard)])  # discarded fixes before each index
    m = len(ts)
    hi = np.searchsorted(ts, t, side="left")  # first usable fix at or after t
    exact = (hi < m) & (ts[np.minimum(hi, m - 1)] == t)
    # inside: between two usable fixes close enough, with no discarded fix between
    a = np.clip(hi - 1, 0, m - 1)
    b = np.clip(hi, 0, m - 1)
    between = (hi > 0) & (hi < m)
    span = ts[b] - ts[a]
    clean = n_hard[idx[b]] - n_hard[idx[a] + 1] == 0
    inside = between & (span <= FILL_MAX_S) & clean
    w = np.where(span > 0, (t - ts[a]) / np.where(span > 0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    for src, dst in ((la, lat), (lo_, lng)):
        v = (1.0 - w) * src[a] + w * src[b]
        dst[inside] = v[inside]
    on[inside] = np.where(w < 0.5, ln_[a], ln_[b])[inside]
    # at a usable fix itself
    at = exact
    lat[at], lng[at], on[at] = la[hi[at]], lo_[hi[at]], ln_[hi[at]]
    # before the first and after the last usable value, by a little
    head = (hi == 0) & ~exact & (ts[0] - t <= FILL_MAX_S)
    tail = (hi >= m) & (t - ts[-1] <= FILL_MAX_S)
    for sel, j in ((head, 0), (tail, m - 1)):
        lat[sel], lng[sel], on[sel] = la[j], lo_[j], ln_[j]
    return lat, lng, on


def tyres_kinematics(
    root: Path,
    session_id: str,
    kin: pa.Table,
    layout_id: str,
    track: "TrackModel | None",
) -> pa.Table | None:
    """``kin`` (a :func:`~...tire_etl.kinematics.load_kinematics` table:
    its clock, laps, stints and speed) with ``lat_g`` and ``long_g`` in the
    tyres' frame, ``lat_source`` / ``long_source`` :data:`TYRES` or (moving
    with no usable fix) :data:`DISCARD`, and ``on_line`` true where the row
    is laid on the track's bank and grade. ``sensor_source`` keeps the
    sensor verdict's ``lat_source`` as a label (what the session's logger
    measured with: reports and the pairing by kind of source read it; no
    heat does). None when the session has no GPS geometry."""
    geom = geo.load_session_geometry(root, session_id, layout_id)
    if geom is None or len(geom) < 2:
        return None
    series = session_series(geom, track)
    t = np.asarray(kin.column("t_session_s").to_numpy(zero_copy_only=False), float)
    lat, lng, on = on_clock(series, t)
    names = kin.schema.names
    speed = (
        np.asarray(kin.column("speed_ms").to_numpy(zero_copy_only=False), float)
        if "speed_ms" in names
        else np.full(len(t), np.nan)
    )
    hole = ~(np.isfinite(lat) & np.isfinite(lng))
    still = np.nan_to_num(speed, nan=0.0) < STILL_SPEED_MS
    lat = np.where(hole & still, 0.0, lat)
    lng = np.where(hole & still, 0.0, lng)
    source = np.full(len(t), TYRES, dtype=object)
    source[hole & ~still] = DISCARD
    lat[hole & ~still] = np.nan
    lng[hole & ~still] = np.nan
    out = kin
    for name, values in (("lat_g", lat), ("long_g", lng)):
        col = pa.array(values, type=pa.float64())
        if name in names:
            out = out.set_column(out.schema.names.index(name), name, col)
        else:
            out = out.append_column(name, col)
    labels = pa.array(source.tolist(), type=pa.string())
    sensor = (
        kin.column("lat_source")
        if "lat_source" in names
        else pa.array([None] * len(t), type=pa.string())
    )
    for name, col in (
        ("sensor_source", sensor),
        ("lat_source", labels),
        ("long_source", labels),
        ("on_line", pa.array(on & ~hole, type=pa.bool_())),
    ):
        if name in out.schema.names:
            out = out.set_column(out.schema.names.index(name), name, col)
        else:
            out = out.append_column(name, col)
    return out
