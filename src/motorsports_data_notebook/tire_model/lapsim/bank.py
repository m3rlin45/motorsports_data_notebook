"""The road's bank from the ground elevation model (#299, after #204).

A track's line says where the road is; the elevation model says how the
ground lies across it. At each node of a track:

1. the line's direction over ± :data:`TANGENT_M`, and across it the
   ground's height at offsets of :data:`STEP_M` out to
   :data:`HALF_WIDTH_M` either side (:meth:`.dem.GsiDem.elevation`,
   bilinear in the 5 m mesh);
2. a least-squares line through those heights: its slope is the
   cross-slope, positive where the **left side of the direction of travel
   is higher**;
3. nodes that cannot be read are flagged (:data:`FLAGS`) and filled along
   the track from their neighbours: no data; a bridge or a structure,
   found as gate B2 finds it (the bare-earth model leaves the logged
   altitude, :func:`notch_zones`); and a section that is not a plane
   (the line's residual over :data:`RESIDUAL_MAX_M`, or a slope over
   :data:`SLOPE_MAX`: a wall, an earth bank beside the road);
4. a moving mean of ± :data:`SMOOTH_M` along the track, and the angle is
   the slope's arctangent.

The bank leans **into** a turn where its sign is the curvature's
(:func:`into_turn`; curvature is positive in a right turn, where the left
side is the outside).

Measured in the review of the car model (docs/tire_model.md §2.12 C5,
item 11): on straights the angle scatters by 0.003–0.004 rad at Fuji,
Sodegaura and Tsukuba, 0.010 at Motegi and 0.019 at Suzuka; where the
bank is large it agrees with what the cars' accelerometers read of it
(slope 0.92–0.99). A 5 m mesh cannot see a crown or a kerb, a bridge
deck reads the ground beneath it, and the tiles cover Japan only: a
track without them has no bank and is driven flat.

Like everything read from the tiles the bank is kept with the track in
``workspace/lapsim/tracks/`` and is not committed (see :mod:`.dem`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ...tire_etl.kinematics import G_MS2
from . import geometry as geo
from . import track as trk

if TYPE_CHECKING:
    from .track import TrackModel

#: The line's direction at a node is taken over this either side of it.
TANGENT_M = 6.0
#: The section across the line: heights this far either side, this far apart.
HALF_WIDTH_M = 5.0
STEP_M = 1.0
#: Moving mean along the track, either side of a node.
SMOOTH_M = 10.0
#: A section is not a plane where the fitted line leaves the heights by
#: more than this (rms), or is steeper than this.
RESIDUAL_MAX_M = 0.15
SLOPE_MAX = 0.12
#: A bridge in the bare-earth model: it leaves the logged altitude by more
#: than this against the difference's median over the base length.
NOTCH_M = 2.0
NOTCH_BASE_M = 300.0
NOTCH_WIDEN_M = 30.0
#: The ground line is compared with the logged altitude only where the
#: model covers this share of the track.
COVER_MIN = 0.9
#: What a node's flag means.
FLAGS = {0: "read", 1: "no data", 2: "bridge", 3: "not a plane"}
#: The scatter of the bank is read on straights: stretches with ``|κ|``
#: under the first value for at least the second.
STRAIGHT_KAPPA = 0.001
STRAIGHT_MIN_M = 100.0
SOURCE = "GSI elevation tiles (Geospatial Information Authority of Japan)"


class Ground(Protocol):
    """An elevation model: :class:`.dem.GsiDem`, or a stand-in."""

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class RoadBank:
    """A track's bank at its nodes."""

    #: Radians, positive where the left side of the direction of travel is
    #: higher; flagged nodes are filled from their neighbours.
    left_rad: np.ndarray
    #: Per node, a key of :data:`FLAGS`.
    flag: np.ndarray
    #: The numbers of the report (see :func:`road_bank`).
    info: dict = field(default_factory=dict)


def notch_zones(
    ground: np.ndarray, alt_m: np.ndarray, ds: float
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Where the bare-earth model steps away from the logged altitude and
    back within a few tens of metres (a bridge, an underpass):
    ``(zones, departure)``, the zones as ``(first, last)`` nodes
    (:func:`.track.find_zones`) and the departure as the model less the
    logged altitude, less that difference's running median over
    :data:`NOTCH_BASE_M`. ``ground`` is the model along the track's line,
    already filled and smoothed as the altitude is."""
    n = len(ground)
    diff = ground - alt_m
    reach = int(round(NOTCH_BASE_M / 2.0 / ds))
    ring = np.concatenate([diff[-reach:], diff, diff[:reach]])
    base = np.array([np.median(ring[i : i + 2 * reach + 1]) for i in range(n)])
    departure = diff - base
    return trk.find_zones(np.abs(departure) > NOTCH_M, ds, NOTCH_WIDEN_M), departure


def zone_nodes(zone: tuple[int, int], n: int) -> np.ndarray:
    """The nodes of a zone of a closed track (one across the seam has
    ``first > last``)."""
    a, b = zone
    return np.arange(a, b + 1) if a <= b else np.arange(a, b + 1 + n) % n


def cross_sections(
    track: "TrackModel", ground: Ground, *, half_width_m: float = HALF_WIDTH_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ground across the track's line at every node: ``(offsets,
    heights, layer)``. ``offsets`` (m) are positive to the left of the
    direction of travel, ``heights`` is offsets × nodes (NaN where the
    model has no data) and ``layer`` names the model's layer under the
    line itself."""
    x, y = np.asarray(track.x_m, float), np.asarray(track.y_m, float)
    k = max(int(round(TANGENT_M / track.ds_m)), 1)
    tx, ty = np.roll(x, -k) - np.roll(x, k), np.roll(y, -k) - np.roll(y, k)
    norm = np.hypot(tx, ty)
    norm = np.where(norm > 0.0, norm, 1.0)
    left_x, left_y = -ty / norm, tx / norm  # x is east and y north: travel turned a quarter left
    offsets = np.arange(-half_width_m, half_width_m + 0.5 * STEP_M, STEP_M)
    heights = np.full((len(offsets), len(x)), np.nan)
    layer = np.full(len(x), "", dtype=object)
    for j, off in enumerate(offsets):
        lat, lon = geo.local_lat_lon(
            x + off * left_x, y + off * left_y, track.gate["lat"], track.gate["lon"]
        )
        heights[j], used = ground.elevation(lat, lon)
        if j == len(offsets) // 2:
            layer = np.asarray(used, dtype=object)
    return offsets, heights, layer


def _slope(offsets: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Per node, the slope of the least-squares line through ``heights``
    (offsets × nodes) against ``offsets``."""
    o = offsets - offsets.mean()
    return np.asarray((o[:, None] * (heights - heights.mean(axis=0))).sum(axis=0) / (o * o).sum())


def road_bank(track: "TrackModel", ground: Ground) -> RoadBank | None:
    """The bank of ``track`` from ``ground`` (see the module docstring);
    None where the model reads no node of it.

    ``info`` holds what gate B2 reports: the nodes by flag
    (``no_data``, ``bridge``, ``not_plane``), the layers under the line
    (``layers``, the share of nodes each), the median residual of the
    sections (``residual_med_m``), how far the two halves of a section
    differ in slope at a tenth of the nodes (``crown_p90``: a crown or an
    edge the mean slope hides), the scatter of the angle along the
    straights about each straight's own mean (``straight_sd_rad``, an
    upper bound of the noise, over ``straights`` of them), the mean and
    the sd of the bank into the turn over the corners
    (``into_corners_mean_rad``, ``into_corners_sd_rad``) and the largest
    angle (``max_abs_rad``).
    """
    n, ds = track.n, float(track.ds_m)
    offsets, heights, layer = cross_sections(track, ground)
    have = np.isfinite(heights).all(axis=0)
    if not have.any():
        return None
    with np.errstate(invalid="ignore"):
        slope = _slope(offsets, heights)
        fitted = heights.mean(axis=0) + (offsets - offsets.mean())[:, None] * slope
        residual = np.sqrt(np.mean((heights - fitted) ** 2, axis=0))
        crown = np.abs(
            _slope(offsets[offsets >= 0], heights[offsets >= 0])
            - _slope(offsets[offsets <= 0], heights[offsets <= 0])
        )
    flag = np.where(have, 0, 1)
    centre = heights[len(offsets) // 2]
    alt = np.asarray(track.alt_m, float)
    if np.isfinite(centre).mean() > COVER_MIN and np.isfinite(alt).mean() > COVER_MIN:
        line = trk._periodic_mean(
            trk._fill_nan_periodic(centre), int(round(trk.ALT_SMOOTH_M / 2.0 / ds))
        )
        for zone in notch_zones(line, trk._fill_nan_periodic(alt), ds)[0]:
            idx = zone_nodes(zone, n)
            flag[idx] = np.where(flag[idx] == 0, 2, flag[idx])
    with np.errstate(invalid="ignore"):
        rough = (residual > RESIDUAL_MAX_M) | (np.abs(slope) > SLOPE_MAX)
    flag[(flag == 0) & rough] = 3
    good = np.flatnonzero(flag == 0)
    if not good.size:
        return None
    filled = np.interp(np.arange(n), good, slope[good], period=n)
    left = np.arctan(trk._periodic_mean(filled, max(int(round(SMOOTH_M / ds)), 1)))

    kappa = np.nan_to_num(np.asarray(track.kappa_heat, float))
    into = into_turn(left, kappa)
    parts = []
    for zone in trk.find_zones(np.abs(kappa) < STRAIGHT_KAPPA, ds):
        idx = zone_nodes(zone, n)
        idx = idx[flag[idx] == 0]
        if len(idx) * ds >= STRAIGHT_MIN_M:
            parts.append(left[idx] - left[idx].mean())
    corner = np.abs(kappa) >= trk.CORNER_APEX_KAPPA
    names, counts = np.unique(layer[layer != ""].astype(str), return_counts=True)
    info = {
        "source": SOURCE,
        "half_width_m": HALF_WIDTH_M,
        "smooth_m": SMOOTH_M,
        "nodes": int(n),
        "no_data": int((flag == 1).sum()),
        "bridge": int((flag == 2).sum()),
        "not_plane": int((flag == 3).sum()),
        "layers": {str(k): round(float(v) / n, 4) for k, v in zip(names, counts)},
        "residual_med_m": float(np.nanmedian(residual)),
        "crown_p90": float(np.nanpercentile(crown[flag == 0], 90)),
        "straights": len(parts),
        "straight_sd_rad": (
            float(np.sqrt(np.mean(np.concatenate(parts) ** 2))) if parts else float("nan")
        ),
        "into_corners_mean_rad": float(into[corner].mean()) if corner.any() else float("nan"),
        "into_corners_sd_rad": float(into[corner].std()) if corner.any() else float("nan"),
        "max_abs_rad": float(np.abs(left).max()),
    }
    return RoadBank(left_rad=left, flag=flag.astype(np.int64), info=info)


def into_turn(left_rad: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """The bank as the lean into the turn (rad): the left-side-higher
    angle with the sign of the curvature, which is positive in a right
    turn. Where the road is straight the sign is the right turn's; only
    the size of the lean matters there."""
    return np.asarray(left_rad * np.where(np.asarray(kappa, float) >= 0.0, 1.0, -1.0))


def tyre_lateral_g(
    kappa: np.ndarray, speed: np.ndarray, left_rad: np.ndarray | None = None
) -> np.ndarray:
    """What the tyres hold sideways (g, positive in a right turn) at
    ``speed`` on the curvature ``kappa``: ``V²κ / g`` on a flat road, and
    ``V²κ·cos b / g − sin b`` where the left side is higher by the angle
    ``b`` (``left_rad``). It is what an accelerometer fixed to the road's
    plane reads, and so what production's heat chain is given by a car
    with a sound one; the ground-frame lateral ``V²κ / g`` is more than
    the tyres hold wherever the road leans into the turn."""
    lat = np.asarray(speed, float) ** 2 * np.asarray(kappa, float) / G_MS2
    if left_rad is None:
        return np.asarray(lat)
    lean = np.asarray(left_rad, float)
    return np.asarray(lat * np.cos(lean) - np.sin(lean))


def to_tyres_frame(lat_g: np.ndarray | float, left_rad: np.ndarray | float) -> np.ndarray:
    """A measured ground-frame lateral (g, positive in a right turn) as
    what the tyres hold on a road whose left side is higher by
    ``left_rad``: ``L·cos b − sin b``, the same rule as
    :func:`tyre_lateral_g`. With the lateral taken positive into its own
    corner, hand the lean into that corner instead."""
    lean = np.asarray(left_rad, float)
    return np.asarray(np.asarray(lat_g, float) * np.cos(lean) - np.sin(lean))


def road_arrays(track: "TrackModel") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(left, cos, kv)`` at the nodes of ``track``: its bank (rad,
    positive where the left side is higher; zeros on a track without
    one), and the two arrays the load on the tyres is made of, ``cos θ·cos
    b`` and ``κ_v + |κ|·sin b`` with b the lean into the turn. The load
    over the weight at the speed V is ``cos + V²·kv / g``, as the solver's
    passes have it (:func:`~.solver._road`)."""
    n = track.n
    kappa = np.abs(np.nan_to_num(np.asarray(track.kappa_heat, float)))
    grade = np.nan_to_num(np.asarray(track.grade, float))
    kv = np.nan_to_num(np.asarray(track.kappa_v, float)) if len(track.kappa_v) == n else np.zeros(n)
    cos = 1.0 / np.sqrt(1.0 + grade**2)
    if len(track.bank) != n:
        return np.zeros(n), cos, kv
    left = np.nan_to_num(np.asarray(track.bank, float))
    lean = into_turn(left, np.nan_to_num(np.asarray(track.kappa_heat, float)))
    return left, cos * np.cos(lean), kv + kappa * np.sin(lean)


def attach(track: "TrackModel", ground: Ground | None) -> RoadBank | None:
    """Give ``track`` its bank from ``ground`` (``bank``, ``bank_flag`` and
    ``meta["bank"]``) and return it; a track the model does not read, or
    no model, is left without one."""
    got = road_bank(track, ground) if ground is not None else None
    if got is None:
        return None
    track.bank = got.left_rad
    track.bank_flag = got.flag
    track.meta["bank"] = {
        k: ((round(v, 5) if np.isfinite(v) else None) if isinstance(v, float) else v)
        for k, v in got.info.items()
    }
    return got


__all__ = [
    "FLAGS",
    "Ground",
    "RoadBank",
    "attach",
    "cross_sections",
    "into_turn",
    "notch_zones",
    "road_arrays",
    "road_bank",
    "to_tyres_frame",
    "tyre_lateral_g",
    "zone_nodes",
]
