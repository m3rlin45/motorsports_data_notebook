"""Does a car's model predict days and layouts it never saw, and can the
gate tell it from a wrong car's? (lap simulation segment C3, issue #205)

**Folds.** Leave-one-day-out per layout, and leave-one-layout-out: the
car is read from its tables without the held-out laps (:func:`fold_car`),
the driver's braking use read from the speed traces that are left
(:func:`~.car_gate.set_use`) with the car at its own cornering limit
(``mu_y``, read without the held-out laps: :func:`~.car_gate.set_limit`),
and every held-out build
lap is simulated at its own lap time, its own use of that limit, with the powertrain the
car had that day, the driver's shift point of the day and the session's
air (:class:`~.car_gate.Drive`). The shift point of a held-out day comes
from that day's own logs: it is the driver's, like the lap time.

**Statistics**, per lap, a fold reading the median over its laps:

- ``S1``: RMS difference of speed over the track's nodes;
- ``S2``: signed end-of-straight speed error, the mean over the layout's
  straights of the simulated top speed less the logged one;
- ``S3``: signed apex speed error, the mean over the corners of the
  simulated minimum less the logged one;
- ``brake_g``: how far the lap's braking is from the logged lap's
  (:func:`braking_match`): wherever the logged lap slows by 0.2 g or
  more, the RMS difference of deceleration at the same places on the
  track. Its yardstick is the same measure between two logged laps
  (:func:`pace_yardsticks`).

**Controls** through the same folds: (a) the fold's own car with
``p_specific × 0.85`` and the same limit, so that its laps use more of
it, braked as the fold's fitted car is
(its driver's level in g at the typical use); (b) another car's
parameter set, its driver's use with it. A lap
time alone cannot tell either from the car, because u is solved from it.

**Verdict.** A model is *inside* a fold when ``S1`` is within the fold's
day floor + 0.4 m/s and ``|S2|`` and ``|S3|`` within 1.0 m/s. A fold in
which a control is inside has no power against that control: it is listed
and not counted; so is a fold without a day floor (a layout the car ran
on one day). A car that is outside on any powered fold fails. One that
is not passes when it has a powered day fold and a powered layout fold
for each control, and is otherwise *not scored*, which is not a pass.

The layout's track is the one built from every car's laps, the held-out
ones included; a track without them is B3's subject.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ...tire_etl.kinematics import G_MS2
from ..heat_input import ShareParams
from . import bank as bankmod
from . import car as cm
from . import car_gate as cg
from . import geometry as geo
from . import holdout as ho
from . import samples as smp
from . import solver as sv
from . import track as trk
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS as HEAT_TYRES
from .heat import sim_heat

logger = logging.getLogger(__name__)

S1_OVER_FLOOR_MAX = cg.SPEED_OVER_FLOOR_MAX
S2_MAX = 1.0
S3_MAX = 1.0
POWER_CONTROL = 0.85
#: The car whose parameters stand in as "another car".
OTHER_CAR = {"Inferno 86": "KK-SII", "KK-SII": "Inferno 86", "KK-F": "Inferno 86"}
#: A straight is a stretch between two apexes this long on which the car
#: gains this much speed over the faster of the two apexes.
STRAIGHT_MIN_M = 200.0
STRAIGHT_MIN_GAIN_MS = 5.0
APEX_WINDOW_M = 20.0
MIN_FOLD_LAPS = 3
#: Braking is scored where the logged lap slows by this much or more (g),
#: the deceleration read from the change of V² over this distance centred
#: on the node; a lap needs this many such nodes to have a score.
BRAKE_MATCH_MIN_G = 0.2
BRAKE_MATCH_WINDOW_M = cg.DECEL_WINDOW_M
BRAKE_MATCH_MIN_NODES = 20
#: The measures are read on the laps above each of these percentiles of
#: pace among the car's laps at the layout (owner, 2026-11-21): the fast
#: laps matter most and repeat best, and three levels show whether the
#: simulation holds as the pace rises. Percentiles and not a margin of
#: lap time: the same share of the laps at every layout.
PACE_LEVELS = (75.0, 90.0, 95.0)
TOP_BAND_QUANTILE = 0.75
DRAG_GRID = (0.0, 0.5, 1.0, 1.5)
TOP_BAND_MAX = 0.3
MODELS = ("fit", "low_power", "other_car")


# -------------------------------------------------------------- the layout


@dataclass
class Layout:
    """What the statistics need of a track: the nodes around each apex and
    the stretches between consecutive apexes."""

    track: trk.TrackModel
    apex_nodes: list[np.ndarray]
    stretches: list[np.ndarray]  # node indices from one apex to the next, in driving order

    @classmethod
    def of(cls, track: trk.TrackModel) -> "Layout":
        n, ds = track.n, track.ds_m
        apex = sorted(int(round(c["s_apex"] / ds)) % n for c in track.corners)
        half = max(int(round(APEX_WINDOW_M / ds)), 1)
        windows = [np.arange(a - half, a + half + 1) % n for a in apex]
        stretches = []
        for a, b in zip(apex, apex[1:] + apex[:1]):
            length = (b - a) % n or n
            stretches.append(np.arange(a, a + length + 1) % n)
        return cls(track=track, apex_nodes=windows, stretches=stretches if apex else [])


deceleration_g = cg.deceleration_g


def braking_match(v_sim: np.ndarray, v_obs: np.ndarray, ds: float) -> float:
    """How far a lap's braking is from the logged lap's (g): the RMS
    difference of :func:`deceleration_g` between the two, at the same
    places on the track, over the nodes where the logged lap slows by
    :data:`BRAKE_MATCH_MIN_G` or more.

    The two laps are compared by position, node against node, with
    nothing moved to make them line up: braking at another place counts
    as much as braking at another strength. NaN where the logged lap has
    fewer than :data:`BRAKE_MATCH_MIN_NODES` such nodes. ``v_sim`` may be
    a simulated lap or another logged one (how closely the driver repeats
    a lap is the measure's yardstick, :func:`pace_yardsticks`)."""
    obs = deceleration_g(trk._fill_nan_periodic(np.asarray(v_obs, float)), ds)
    zone = obs >= BRAKE_MATCH_MIN_G
    if int(zone.sum()) < BRAKE_MATCH_MIN_NODES:
        return float("nan")
    sim = deceleration_g(trk._fill_nan_periodic(np.asarray(v_sim, float)), ds)
    return float(np.sqrt(np.mean((sim - obs)[zone] ** 2)))


def fast_laps(lap_time_s: np.ndarray, percentile: float) -> np.ndarray:
    """Which laps of a car at a layout are above ``percentile`` of pace:
    the fastest tenth of them at 90 (rounded up, so there is always one,
    and laps tied with the slowest of them are in)."""
    t = np.asarray(lap_time_s, float)
    if not t.size:
        return np.zeros(0, bool)
    keep = max(int(math.ceil(t.size * (1.0 - percentile / 100.0))), 1)
    return np.asarray(t <= np.sort(t)[keep - 1])


YARDSTICK_MEASURES = ("S1", "S2", "S3", "brake_g")
#: Gate D3 of #206, in the same form: lap-mean lateral heat, simulated
#: over logged, and the same for right turns less left turns (log ratios).
MOMENT_MEASURES = ("lat_heat", "side")
#: Measures that are signed and are compared by their size.
SIZE_MEASURES = ("S2", "S3", *MOMENT_MEASURES)


def side_heat(v: np.ndarray, lat_g: np.ndarray, ds: float) -> tuple[float, float]:
    """``(right, left)``: the lap-mean of ``V·lat`` (G·m/s) over the lap's
    time, from its speed and lateral at the track's nodes, for the
    lateral into right turns and into left turns apart. The lateral is
    what the tyres hold, positive in a right turn."""
    dt = ds / np.maximum(v, 1.0)
    lap = float(dt.sum())
    right = float(np.sum(v * np.maximum(lat_g, 0.0) * dt) / lap)
    left = float(np.sum(v * np.maximum(-lat_g, 0.0) * dt) / lap)
    return right, left


def moments(sim: tuple[float, float], obs: tuple[float, float]) -> dict:
    """``lat_heat`` and ``side`` of :data:`MOMENT_MEASURES` from two laps'
    :func:`side_heat`: the log of simulated over logged lateral heat, and
    that of the right turns less that of the left turns (NaN on a layout
    that turns one way only). Also ``balance``: the log of the logged
    lap's right over its left. A mirrored track or a swapped sign reads a
    ``side`` of minus twice that, so on a layout whose two sides carry
    alike ``side`` cannot see one."""
    total = math.log((sim[0] + sim[1]) / (obs[0] + obs[1])) if obs[0] + obs[1] > 0 else np.nan
    if min(*sim, *obs) <= 0.0:
        return {"lat_heat": total, "side": float("nan"), "balance": float("nan")}
    return {
        "lat_heat": total,
        "side": math.log(sim[0] / obs[0]) - math.log(sim[1] / obs[1]),
        "balance": math.log(obs[0] / obs[1]),
    }


#: A ``side`` this large is a sign or mirror defect (#206, gate D3).
MIRROR_DEFECT = 0.12


def side_signed(laps: pd.DataFrame) -> pd.DataFrame:
    """Per car and layout, from the held-out laps of the day folds: the
    signed ``lat_heat`` and ``side`` (the median over days of the day's
    median lap), ``mirror`` (what a mirrored track would read as ``side``:
    twice the size of the logged laps' ``balance``) and ``sees_mirror``
    (that is :data:`MIRROR_DEFECT` or more)."""
    cols = ["lat_heat", "side", "balance"]
    if laps.empty or any(k not in laps.columns for k in cols):
        return pd.DataFrame()
    cell = ["car", "layout_id"]
    days = laps.dropna(subset=["lat_heat"]).groupby(cell + ["date"], sort=False)[cols].median()
    out = days.groupby(cell, sort=False)[cols].median()
    out.insert(0, "days", days.groupby(cell, sort=False).size())
    out["mirror"] = 2.0 * out.pop("balance").abs()
    out["sees_mirror"] = out["mirror"] >= MIRROR_DEFECT
    return out.reset_index()


def pace_yardsticks(
    layout: "Layout",
    laps: pd.DataFrame,
    v_obs: np.ndarray,
    levels: tuple[float, ...] = PACE_LEVELS,
    lat_obs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Reality against itself, at each level of pace: for every lap of a
    car at a layout (``date, lap_time_s``; ``v_obs`` its speed at the
    nodes) that is above the level's percentile (:func:`fast_laps`), how
    far other logged laps of that set are from it.

    One row per lap and level: ``lap`` (the lap's row in ``laps``),
    ``level`` and, for each of :data:`YARDSTICK_MEASURES`,
    ``<measure>_other_day`` and ``<measure>_same_day``: the median over
    the lap's :data:`~.car_gate.FLOOR_PAIRS_PER_LAP` nearest laps in lap
    time, from other days and from its own, of :func:`lap_stats` with the
    other lap standing where the simulated one would. ``S2`` and ``S3``
    are sizes (absolute values). NaN without a pair: at a high level a
    layout's set may hold one day's laps only. With ``lat_obs`` (the
    laps' own lateral at the nodes) the :data:`MOMENT_MEASURES` have their
    yardsticks too, each logged lap with its own lateral.

    Two logged laps each carry their own lap-to-lap variation, and two
    from different days two GPS position offsets where a simulated lap
    against a logged one carries one; so a simulation that matched the
    driver's usual lap would read under ``other_day``, and ``same_day``
    is about the best any could do."""
    t = laps["lap_time_s"].to_numpy(float)
    day = laps["date"].to_numpy()
    filled = [trk._fill_nan_periodic(np.asarray(v, float)) for v in v_obs]
    measures = YARDSTICK_MEASURES + (MOMENT_MEASURES if lat_obs is not None else ())
    rows: list[dict] = []
    for level in levels:
        inset = fast_laps(t, level)
        for i in np.flatnonzero(inset):
            row: dict = {"lap": int(i), "level": float(level)}
            others = inset & (np.arange(len(t)) != i)
            for tag, pool in (("other_day", day != day[i]), ("same_day", day == day[i])):
                pick = np.flatnonzero(others & pool)
                pick = pick[np.argsort(np.abs(t[pick] - t[i]))][: cg.FLOOR_PAIRS_PER_LAP]
                stats = [
                    lap_stats(
                        layout,
                        filled[j],
                        filled[i],
                        lat_obs=None if lat_obs is None else lat_obs[i],
                        lat_sim=None if lat_obs is None else lat_obs[j],
                    )
                    for j in pick
                ]
                for k in measures:
                    vals = [abs(st[k]) for st in stats if np.isfinite(st.get(k, np.nan))]
                    row[f"{k}_{tag}"] = float(np.median(vals)) if vals else float("nan")
            rows.append(row)
    cols = ["lap", "level"] + [f"{k}_{tag}" for tag in ("other_day", "same_day") for k in measures]
    return pd.DataFrame(rows, columns=cols)


def lap_stats(
    layout: Layout,
    v_sim: np.ndarray,
    v_obs: np.ndarray,
    *,
    lat_obs: np.ndarray | None = None,
    lat_sim: np.ndarray | None = None,
) -> dict:
    """``S1, S2, S3`` and ``brake_g`` of one lap (module docstring) and
    ``top``: the RMS difference over the nodes where the logged lap is in
    its own top quarter of speed.

    With ``lat_obs``, the logged lap's own lateral at the nodes (what its
    tyres held), also :data:`MOMENT_MEASURES`: the simulated lap's
    lateral heat over the logged lap's, and that by side. The simulated
    lap's lateral is what the tyres hold on the track's curvature and
    bank at its speed, unless ``lat_sim`` gives it (another logged lap
    standing where the simulated one would)."""
    ds = layout.track.ds_m
    ok = np.isfinite(v_obs)
    out = {"S1": cg._rms((v_sim - v_obs)[ok]), "S2": np.nan, "S3": np.nan}
    out["brake_g"] = braking_match(v_sim, v_obs, ds)
    band = ok & (v_obs >= np.nanquantile(v_obs, TOP_BAND_QUANTILE))
    out["top"] = cg._rms((v_sim - v_obs)[band])
    if layout.apex_nodes:
        out["S3"] = float(
            np.mean([np.min(v_sim[w]) - np.nanmin(v_obs[w]) for w in layout.apex_nodes])
        )
    tops: list[float] = []
    for nodes in layout.stretches:
        if (len(nodes) - 1) * ds < STRAIGHT_MIN_M:
            continue
        obs = v_obs[nodes]
        if np.nanmax(obs) - max(obs[0], obs[-1]) < STRAIGHT_MIN_GAIN_MS:
            continue
        tops.append(float(np.max(v_sim[nodes]) - np.nanmax(obs)))
    if tops:
        out["S2"] = float(np.mean(tops))
    if lat_obs is not None and np.isfinite(lat_obs).any():
        track = layout.track
        held = lat_sim
        if held is None:
            banked = len(track.bank) == track.n
            held = bankmod.tyre_lateral_g(
                np.nan_to_num(np.asarray(track.kappa_heat, float)),
                v_sim,
                np.nan_to_num(np.asarray(track.bank, float)) if banked else None,
            )
        logged_v = trk._fill_nan_periodic(np.asarray(v_obs, float))
        logged_lat = trk._fill_nan_periodic(np.asarray(lat_obs, float))
        other_lat = trk._fill_nan_periodic(np.asarray(held, float))
        out.update(moments(side_heat(v_sim, other_lat, ds), side_heat(logged_v, logged_lat, ds)))
    return out


def apex_errors(
    layout: Layout, v_sim: np.ndarray, v_obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per corner of ``layout`` (in the order of its apex windows): the
    simulated lap's lowest speed in the window less the logged lap's, and
    the logged lap's lowest speed there (m/s). ``S3`` of
    :func:`lap_stats` is the mean of the first."""
    low = np.array([np.nanmin(v_obs[w]) for w in layout.apex_nodes], dtype=float)
    sim = np.array([np.min(v_sim[w]) for w in layout.apex_nodes], dtype=float)
    return sim - low, low


def corner_frame(layout: Layout) -> pd.DataFrame:
    """The layout's corners in the order of its apex windows: ``corner``
    (1 for the first apex after the start line), ``s_apex_m``,
    ``radius_m`` at the apex and ``sign`` (positive in a right turn)."""
    track = layout.track
    rows = sorted(
        (
            (
                int(round(c["s_apex"] / track.ds_m)) % track.n,
                float(c["s_apex"]),
                float(1.0 / abs(c["kappa_apex"])) if c.get("kappa_apex") else float("nan"),
                int(c.get("sign", 0)),
            )
            for c in track.corners
        ),
        key=lambda r: r[0],
    )
    return pd.DataFrame(
        [
            {"corner": k + 1, "s_apex_m": r[1], "radius_m": r[2], "sign": r[3]}
            for k, r in enumerate(rows)
        ],
        columns=["corner", "s_apex_m", "radius_m", "sign"],
    )


def score_laps(
    who: cg.Who,
    layout: Layout,
    laps: pd.DataFrame,
    v_obs: np.ndarray,
    *,
    corners: bool = False,
    lat_obs: np.ndarray | None = None,
    share: ShareParams | None = None,
) -> pd.DataFrame:
    """:func:`lap_stats` of every lap of ``laps`` (``session_id, lap_num,
    lap_time_s``; ``v_obs`` its speed at the nodes) simulated at its own
    lap time, with ``u``: by one car, or by a :class:`~.car_gate.Drive`
    with each lap's own powertrain, shift point and air. A lap whose time
    the car cannot do has NaN statistics. With ``corners`` the frame also
    has, per corner k of :func:`corner_frame`, ``apex_<k>``
    (:func:`apex_errors`) and ``apex_v_<k>`` (the logged lap's lowest
    speed there). With ``share`` it also has ``sim_q_<tyre>``: the simulated
    lap's lap-mean heat rate per tyre through production's chain
    (:func:`~.heat.sim_heat`; the tyres' frame, G·m/s), for gate E1 (#207)."""
    drive = cg.as_drive(who)
    profile = cg.profile_of(drive, layout.track)
    typical = cg.typical_use(drive)
    recs = cg._records(laps)
    cars = [cg._lap_car(drive, rec) for rec in recs]
    got = sv.solve_laps(
        profile,
        cars,
        [float(rec["lap_time_s"]) for rec in recs],
    )
    rows = []
    for i in range(len(recs)):
        row = {
            # the pace against the car's typical use, and the use itself
            "u": float(got.u[i]) / typical,
            "use": float(got.u[i]),
            "S1": np.nan,
            "S2": np.nan,
            "S3": np.nan,
            "brake_g": np.nan,
            "top": np.nan,
        }
        if got.ok[i]:
            row.update(
                lap_stats(
                    layout,
                    got.v_node_ms[i],
                    v_obs[i],
                    lat_obs=None if lat_obs is None else lat_obs[i],
                )
            )
            if share is not None:
                sim = sv.lap_from_nodes(profile, cars[i], got.v_node_ms[i], float(got.u[i]))
                for tyre, q in zip(HEAT_TYRES, sim_heat(sim, share)):
                    row[f"sim_q_{tyre}"] = float(q)
            if corners and layout.apex_nodes:
                err, low = apex_errors(layout, got.v_node_ms[i], v_obs[i])
                for k in range(len(err)):
                    row[f"apex_{k + 1}"] = float(err[k])
                    row[f"apex_v_{k + 1}"] = float(low[k])
        rows.append(row)
    return pd.DataFrame(rows)


def corner_table(laps: pd.DataFrame, layouts: dict[str, Layout]) -> pd.DataFrame:
    """The apex-speed error corner by corner, from the held-out laps of
    the day folds (``laps`` of :func:`run_folds`, with the ``apex_<k>``
    columns): per car, layout and corner, ``error_ms`` (simulated less
    logged lowest speed, the median over days of the day's median lap, as
    the gate reads a cell), ``v_apex_ms`` (the logged one, likewise),
    ``laps`` and ``days``, with the corner's place and radius."""
    cols = ["car", "layout_id", "corner", "s_apex_m", "radius_m", "sign"]
    cols += ["laps", "days", "v_apex_ms", "error_ms"]
    rows: list[dict] = []
    held = laps[laps["kind"] == "day"] if len(laps) else laps
    for (car, layout_id), cell in held.groupby(["car", "layout_id"]) if len(held) else []:
        if layout_id not in layouts:
            continue
        for c in corner_frame(layouts[str(layout_id)]).to_dict("records"):
            k = int(c["corner"])
            if f"apex_{k}" not in cell.columns:
                continue
            got = cell[["date", f"apex_{k}", f"apex_v_{k}"]].dropna()
            if got.empty:
                continue
            by_day = got.groupby("date")[[f"apex_{k}", f"apex_v_{k}"]].median()
            rows.append(
                {
                    "car": car,
                    "layout_id": layout_id,
                    **c,
                    "laps": int(len(got)),
                    "days": int(len(by_day)),
                    "v_apex_ms": float(by_day[f"apex_v_{k}"].median()),
                    "error_ms": float(by_day[f"apex_{k}"].median()),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def corner_agreement(table: pd.DataFrame, min_corners: int = 4) -> pd.DataFrame:
    """Per layout and pair of cars: the correlation over the layout's
    corners of their apex-speed errors (``r``) and the corners behind it.
    An error that repeats between unlike cars is the track's (its bank,
    its line through the corner), one that does not is the car's."""
    rows: list[dict] = []
    for layout_id, cell in table.groupby("layout_id") if len(table) else []:
        wide = cell.pivot(index="corner", columns="car", values="error_ms")
        names = list(wide.columns)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                both = wide[[a, b]].dropna()
                if len(both) < min_corners:
                    continue
                rows.append(
                    {
                        "layout_id": layout_id,
                        "car_a": a,
                        "car_b": b,
                        "corners": int(len(both)),
                        "r": float(np.corrcoef(both[a], both[b])[0, 1]),
                    }
                )
    return pd.DataFrame(rows, columns=["layout_id", "car_a", "car_b", "corners", "r"])


def heat_errors(
    who: cg.Who,
    layout: Layout,
    laps: pd.DataFrame,
    v_obs: np.ndarray,
    share: ShareParams,
    best: pd.DataFrame,
) -> pd.DataFrame:
    """Per lap of ``laps``, the log of simulated over logged lap-mean heat
    per tyre through production's share model
    (:func:`~.holdout.tyre_heat`), the lap driven two ways:

    - ``h_<tyre>``: at its own lap time (u solved from it);
    - ``hc_<tyre>``: with the pace level carried. ``best`` is the layout's
      day-best laps; their median u under ``who`` is the layout's
      reference, and the lap is driven at its own u over that. Where the
      car was set without this layout that takes the layout's reference
      lap as u = 1, which is all a car can be told of a layout it has not
      run, and leaves the lap only its pace against the reference.

    Both sides are on the track's curvature, each at its own speed, and
    in the tyres' frame where the laps are driven on the track's bank
    (the ground frame on a flat road). A lap the car cannot do is NaN."""
    drive = cg.as_drive(who)
    track = layout.track
    profile = cg.profile_of(drive, track)
    recs = cg._records(laps)
    cars = [cg._lap_car(drive, rec) for rec in recs]
    got = sv.solve_laps(profile, cars, [float(rec["lap_time_s"]) for rec in recs])
    ref = cg.reference_u(drive, {track.layout_key: track}, best).get(track.layout_key, {})
    u_ref = float(ref.get("u_ref", float("nan")))
    # the lap's pace against its layout's reference, at the car's typical use
    typical = cg.typical_use(drive)
    carried = sv.drive_laps(profile, cars, np.where(got.ok, got.u / u_ref * typical, typical))
    out = np.full((len(recs), 2 * len(ho.CORNERS)), np.nan)
    for i in range(len(recs)):
        if not got.ok[i]:
            continue
        logged = ho.tyre_heat(
            share,
            profile.kappa,
            trk._fill_nan_periodic(np.asarray(v_obs[i], float)),
            profile.ds_m,
            profile.bank,
        )
        for k, sim in enumerate((got.v_node_ms[i], carried.v_node_ms[i])):
            if np.isfinite(sim).all():
                heat = ho.tyre_heat(share, profile.kappa, sim, profile.ds_m, profile.bank)
                out[i, k * len(ho.CORNERS) : (k + 1) * len(ho.CORNERS)] = np.log(heat / logged)
    cols = [f"{kind}_{c}" for kind in ("h", "hc") for c in ho.CORNERS]
    return pd.DataFrame(out, columns=cols)


def worst_tyre(laps: pd.DataFrame, kind: str, tag: str = "") -> float:
    """The largest over the tyres of the absolute median over ``laps`` of
    :func:`heat_errors`' ``kind`` (``h`` or ``hc``; ``tag`` the variant's
    suffix); NaN where the laps carry none."""
    cols = [f"{kind}_{c}{tag}" for c in ho.CORNERS]
    if not set(cols) <= set(laps.columns):
        return float("nan")
    med = laps[cols].median().abs()
    return float(med.max()) if med.notna().any() else float("nan")


# ---------------------------------------------------------------- the folds


def car_only(tables: smp.CarTables, car: str) -> smp.CarTables:
    """``tables`` with the rows of ``car`` alone (speed traces left as
    they are: they are looked up by row)."""

    def _own(tbl: pd.DataFrame) -> pd.DataFrame:
        return tbl[tbl["car"] == car] if len(tbl) and "car" in tbl.columns else tbl

    return smp.CarTables(
        sessions=_own(tables.sessions),
        throttle=_own(tables.throttle),
        coast=_own(tables.coast),
        brake=_own(tables.brake),
        apex=_own(tables.apex),
        speed_laps=_own(tables.speed_laps),
        speed=tables.speed,
        lateral=tables.lateral,
    )


def without(tables: smp.CarTables, layout_id: str, date: str | None = None) -> smp.CarTables:
    """``tables`` without the rows of ``layout_id`` (on ``date`` only, when
    one is given)."""

    def _keep(tbl: pd.DataFrame) -> pd.DataFrame:
        if tbl.empty or "layout_id" not in tbl.columns:
            return tbl
        out = tbl["layout_id"] == layout_id
        if date is not None:
            out &= tbl["date"] == date
        return tbl[~out]

    return smp.CarTables(
        sessions=tables.sessions,
        throttle=_keep(tables.throttle),
        coast=_keep(tables.coast),
        brake=_keep(tables.brake),
        apex=_keep(tables.apex),
        speed_laps=tables.speed_laps,
        speed=tables.speed,
        lateral=tables.lateral,
    )


def fold_car(
    tables: smp.CarTables,
    car: str,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    power_scale: float = 1.0,
    laps: dict[str, cg.UseLaps] | None = None,
) -> cg.Drive | None:
    """The car read from ``tables`` (already without the held-out laps)
    with ``mu_y`` by the convention on ``best`` (the day-best laps that
    are left), as a :class:`~.car_gate.Drive` with the days' shift points
    and air; ``power_scale`` plants a wrong power before the convention is
    set, and ``laps`` are those the driver's braking use is read from
    (:func:`convene`). None when the tables give no power, drag or
    braking."""
    model = cm.fit_car(tables, car, air=air)[0]
    return convene(
        model,
        tracks,
        best,
        air=air,
        shifts=shifts,
        gears=gears,
        power_scale=power_scale,
        laps=laps,
    )


def convene(
    model: cm.CarModel,
    tracks: dict[str, trk.TrackModel],
    best: pd.DataFrame,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    power_scale: float = 1.0,
    bank: bool = True,
    laps: dict[str, cg.UseLaps] | None = None,
    use: dict | None = None,
    driven_axle: bool | None = None,
) -> cg.Drive | None:
    """``model`` (as :func:`~.car.fit_car` returns it) as a
    :class:`~.car_gate.Drive` with its power scaled, the driver's braking
    use read from ``laps`` (:func:`~.car_gate.set_use`) and ``mu_y`` then
    at the car's cornering limit, with the car's typical use of it read
    on ``best`` (:func:`~.car_gate.set_limit`; the convention for a model
    without a limit); None when the model lacks power,
    drag or braking, or no reference lap is left. Without ``laps`` the
    laps are braked at the car's limit at its typical use (``mu_b`` at
    the reference pace, the knob moving it like the rest), or with
    ``use`` (a ``driver`` as another model of the car carries it) as that
    driver brakes: the level in g at the typical use is held.
    ``bank=False`` has its laps, the reference laps among them, driven on
    a flat road, and ``driven_axle=False`` on the whole car's grip under
    drive (None: the solver's default)."""
    need = (
        model.power["p_specific"],
        model.power["c_drag"],
        model.grip["mu_b"],
        model.grip["mu_y"],
    )
    if any(v is None for v in need) or best.empty:
        return None
    drive = cg.Drive(
        model,
        air=air,
        shifts=dict(shifts or {}),
        gears=dict(gears or {}),
        power_scale=power_scale,
        bank=bank,
        driven_axle=driven_axle,
    )
    left = {k: v for k, v in tracks.items() if k in set(best["layout_id"])}
    # the car's limit first: the driver's use is read as a share of it
    drive = drive.but(model=cg.at_limit(drive.model))
    if laps:
        drive = drive.but(model=cg.set_use(drive, left, laps)[0])
    held = use if use and not laps else None
    if not laps and not use and drive.model.grip.get("mu_y_limit") is not None:
        # braked at the car's limit *at its typical use*, which is what "at
        # the limit" was while the reference lap's knob was 1: a share of 1
        # at a use of 1 would brake every logged lap below ``mu_b``
        held = {"brake_use": 1.0, "use_ref": 1.0}
    return drive.but(model=cg.set_limit(drive, left, best, hold=held)[0])


def laps_left(
    laps: dict[str, cg.UseLaps], layout_id: str, date: str | None = None
) -> dict[str, cg.UseLaps]:
    """``laps`` without those of ``layout_id`` (on ``date`` only, when
    one is given): what a fold's braking use is read from."""
    out: dict[str, cg.UseLaps] = {}
    for key, own in laps.items():
        if key != layout_id:
            out[key] = own
        elif date is not None:
            out[key] = own.keep((own.frame["date"] != date).to_numpy())
    return out


def _pair_floor(
    held: pd.DataFrame, v_held: np.ndarray, rest: pd.DataFrame, v_rest: np.ndarray
) -> float:
    """The median RMS difference of speed between each held-out lap and
    the laps of other days within :data:`~.car_gate.FLOOR_LAP_TIME_TOL` of
    its lap time (its nearest three); NaN without a pair."""
    t_rest = rest["lap_time_s"].to_numpy(float)
    vals: list[float] = []
    for i, t in enumerate(held["lap_time_s"].to_numpy(float)):
        near = np.flatnonzero(np.abs(t_rest / t - 1.0) <= cg.FLOOR_LAP_TIME_TOL)
        for j in near[np.argsort(np.abs(t_rest[near] - t))][: cg.FLOOR_PAIRS_PER_LAP]:
            vals.append(cg._rms(v_held[i] - v_rest[j]))
    return float(np.median(vals)) if vals else float("nan")


def run_folds(
    tables: smp.CarTables,
    rows: pd.DataFrame,
    tracks: dict[str, trk.TrackModel],
    car: str,
    other: cg.Drive | cm.CarModel | None,
    *,
    air: pd.DataFrame | None = None,
    shifts: dict[str, float] | None = None,
    gears: dict[str, tuple[float, ...]] | None = None,
    shares: dict[str, ShareParams] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every fold of ``car``: ``(folds, laps, yardsticks)``.

    ``tables`` and ``rows`` are by family (:func:`~.car.as_families`). A
    held-out lap is driven with its own day's powertrain, shift point and
    gear set (``shifts`` and ``gears``, read from the logs of that day:
    the driver's and the setup's, like the lap time) and air; ``other`` is the car whose parameters stand in as
    another car's.

    ``folds`` has one row per fold and model (``fit``, ``low_power``,
    ``other_car``): ``kind`` (``day`` or ``layout``), ``layout_id``,
    ``date``, ``laps``, ``floor`` and the medians ``S1, S2, S3, brake_g``.
    ``laps`` has the per-lap statistics of the fitted car with its
    apex-speed error per corner (``apex_<k>``, :func:`score_laps`), the same with
    the suffix ``_limit`` for the fold's car braked at its limit (no
    driver's use, ``mu_y`` set again: the gate as it stood before the
    use), and its variants on the layout folds: ``S1_other`` and ``top_other`` (the
    same car with its cornering drag the other way round: off when the
    car carries it, on when it does not), ``S1_flat``, ``S3_flat``,
    ``top_flat`` and ``u_flat`` (the fold's car with every road taken as
    flat, ``mu_y`` set again by the convention there, so that ``u``
    against ``u_flat`` says how the pace level carries to the held-out
    layout each way) and, on a layout that reaches 50 m/s, ``top_<k>``
    (the top-quarter RMS at k × the model's drag). ``yardsticks`` is
    :func:`pace_yardsticks` of the car's laps at each layout, keyed like
    ``laps`` (``car, layout_id, session_id, lap_num, date`` and
    ``level``): how far two logged laps are from each other on each
    measure, at each level of pace. With ``shares`` (the
    production force shares per car, :func:`~.holdout.production_shares`)
    the layout folds also carry :func:`heat_errors`, on the bank, with
    the suffix ``_flat`` without it and ``_limit`` braked at the limit.
    """
    own = car_only(tables, car)
    best_all = cg.day_best_laps(rows, car)
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
    layouts = {k: Layout.of(v) for k, v in tracks.items() if k in set(best_all["layout_id"])}
    other_car = cg.Drive(other, air=air) if isinstance(other, cm.CarModel) else other
    any_bank = any(len(t.bank) for t in tracks.values())
    # what the driver's braking use is read from, less each fold's laps
    use_tracks = {k: v.track for k, v in layouts.items()}
    use_laps = cg.car_use_laps(own, lap_time, car, use_tracks)
    use_flat = cg.car_use_laps(own, lap_time, car, use_tracks, bank=False) if any_bank else {}
    share = None
    if shares is not None:
        try:
            share = ho.share_for(car, shares)
        except KeyError:
            logger.info("holdout: no force shares for %s, its heat is not scored", car)
    fold_rows: list[dict] = []
    lap_rows: list[pd.DataFrame] = []
    yard_rows: list[pd.DataFrame] = []
    for layout_id, layout in layouts.items():
        cell, v_cell = own.lap_speeds(car, layout_id)
        lat_cell = own.lap_laterals(car, layout_id)
        if cell.empty:
            continue
        keys = pd.MultiIndex.from_frame(cell[["session_id", "lap_num"]])
        cell = cell.assign(lap_time_s=lap_time.reindex(keys).to_numpy())
        cell_floor = cg.day_floor(cell, v_cell)["floor"]
        yard = pace_yardsticks(layout, cell, v_cell, lat_obs=lat_cell)
        keyed = cell.iloc[yard["lap"].to_numpy()][["session_id", "lap_num", "date"]]
        yard = pd.concat([keyed.reset_index(drop=True), yard.drop(columns="lap")], axis=1)
        yard.insert(0, "car", car)
        yard.insert(1, "layout_id", layout_id)
        yard_rows.append(yard)
        days = sorted(cell["date"].unique())
        plans: list[tuple[str, str | None]] = [("layout", None)]
        if len(days) > 1:
            plans += [("day", d) for d in days]
        for kind, date in plans:
            held_mask = (cell["date"] == date).to_numpy() if date else np.ones(len(cell), bool)
            held = cell[held_mask].reset_index(drop=True)
            v_held = v_cell[held_mask]
            lat_held = None if lat_cell is None else lat_cell[held_mask]
            if len(held) < MIN_FOLD_LAPS:
                continue
            best = best_all[
                ~(
                    (best_all["layout_id"] == layout_id)
                    & ((best_all["date"] == date) if date else True)
                )
            ]
            raw = cm.fit_car(without(own, layout_id, date), car, air=air)[0]
            read = laps_left(use_laps, layout_id, date)
            fitted = convene(raw, tracks, best, air=air, shifts=shifts, gears=gears, laps=read)
            if fitted is None:
                logger.info("holdout: no car for %s without %s %s", car, layout_id, date or "")
                continue
            low = convene(
                raw,
                tracks,
                best,
                air=air,
                shifts=shifts,
                gears=gears,
                power_scale=POWER_CONTROL,
                use=fitted.model.driver,  # a wrong power is planted, not another driver
            )
            # the fitted car's held-out laps are answered under the cap (a use
            # cannot pass 1); a planted wrong car needs a knob above 1 by
            # construction and keeps the arithmetic without it
            models: dict[str, cg.Drive | None] = {
                "fit": fitted.but(use_cap=True),
                "low_power": low,
                "other_car": other_car,
            }
            floor = cell_floor
            if date:
                rest = cell[~held_mask].reset_index(drop=True)
                own_floor = _pair_floor(held, v_held, rest, v_cell[~held_mask])
                floor = own_floor if np.isfinite(own_floor) else cell_floor
            base = {
                "car": car,
                "kind": kind,
                "layout_id": layout_id,
                "date": date or "",
                "laps": int(len(held)),
                "floor": floor,
            }
            fit_laps: pd.DataFrame | None = None
            for name, driven in models.items():
                if driven is None:
                    continue
                scores = score_laps(
                    driven,
                    layout,
                    held,
                    v_held,
                    corners=name == "fit",
                    lat_obs=lat_held if name == "fit" else None,
                    share=share,
                )
                if name != "fit" and fit_laps is not None and "sim_q_fl" in scores.columns:
                    # a planted car's heat on the same laps, beside the fitted car's (gate E1)
                    for tyre in HEAT_TYRES:
                        fit_laps[f"sim_q_{tyre}_{name}"] = scores[f"sim_q_{tyre}"].to_numpy()
                fold_rows.append(
                    {
                        **base,
                        "model": name,
                        **{k: float(scores[k].median()) for k in ("S1", "S2", "S3", "brake_g")},
                        "u": float(scores["u"].median()),
                        "brake_use": driven.model.driver.get("brake_use", float("nan")),
                        "held_shape": driven.model.driver.get("brake_use_ellipse", float("nan")),
                        "mu_y": driven.model.grip["mu_y"],
                        "p_specific": driven.model.power["p_specific"] * driven.power_scale,
                        "c_drag": driven.model.power["c_drag"],
                    }
                )
                if name != "fit":
                    continue
                per_lap = pd.concat([held[["session_id", "lap_num", "date"]], scores], axis=1)
                fit_laps = per_lap
                per_lap.insert(0, "car", car)
                per_lap.insert(1, "kind", kind)
                per_lap.insert(2, "layout_id", layout_id)
                # the variants below are comparison arms (another drive rule, a
                # drag that is not the car's, a flat road): like the planted cars
                # they are driven without the cap, as before it
                # the fold's car with the other drive rule (the whole car's grip
                # where the default is the driven axle's share), its use read again
                whole = convene(
                    raw,
                    tracks,
                    best,
                    air=air,
                    shifts=shifts,
                    gears=gears,
                    laps=read,
                    driven_axle=not sv.DRIVEN_AXLE_DEFAULT,
                )
                if whole is not None:
                    other_drive = score_laps(whole, layout, held, v_held)
                    for col in (*YARDSTICK_MEASURES, "u"):
                        per_lap[f"{col}_axle"] = other_drive[col]
                # the fold's car braked at its limit, mu_y set again
                plain = convene(raw, tracks, best, air=air, shifts=shifts, gears=gears)
                if plain is not None:
                    at_limit = score_laps(plain, layout, held, v_held)
                    for col in (*YARDSTICK_MEASURES, "u"):
                        per_lap[f"{col}_limit"] = at_limit[col]
                if kind == "layout":
                    flipped = fitted.but(corner_drag=not sv.CORNER_DRAG_DEFAULT)
                    variant = score_laps(flipped, layout, held, v_held)
                    per_lap["S1_other"] = variant["S1"]
                    per_lap["top_other"] = variant["top"]
                    # and with the shift times the other way round
                    shifted = score_laps(
                        fitted.but(shift_loss=not sv.SHIFT_LOSS_DEFAULT), layout, held, v_held
                    )
                    per_lap["S1_shift"] = shifted["S1"]
                    per_lap["top_shift"] = shifted["top"]
                    # and with the braking limit's shape the other way round
                    braked = score_laps(
                        fitted.but(brake_shape=not sv.BRAKE_SHAPE_DEFAULT), layout, held, v_held
                    )
                    per_lap["S1_brake"] = braked["S1"]
                    per_lap["S3_brake"] = braked["S3"]
                    per_lap["brake_g_brake"] = braked["brake_g"]
                    # and with every road taken as flat, mu_y set again by
                    # the convention on the layouts that are left
                    level = (
                        convene(
                            raw,
                            tracks,
                            best,
                            air=air,
                            shifts=shifts,
                            gears=gears,
                            bank=False,
                            laps=laps_left(use_flat, layout_id, date),
                        )
                        if any_bank
                        else None
                    )
                    if level is not None:
                        on_flat = score_laps(level, layout, held, v_held)
                        for col in ("S1", "S3", "top", "u"):
                            per_lap[f"{col}_flat"] = on_flat[col]
                    # heat per tyre, at the lap's own time and with the pace carried
                    own_best = best_all[best_all["layout_id"] == layout_id]
                    for who, tag in (
                        (driven, ""),
                        (level, "_flat"),
                        (plain, "_limit"),
                        (whole, "_axle"),
                    ):
                        if share is None or who is None:
                            continue
                        heat = heat_errors(who, layout, held, v_held, share, own_best)
                        for col in heat.columns:
                            per_lap[f"{col}{tag}"] = heat[col]
                    if np.nanmax(v_held) >= cm.DRAG_REACH_MS:
                        for k in DRAG_GRID:
                            if k == 1.0:
                                per_lap["top_1.0"] = per_lap["top"]
                                continue
                            grid = fitted.but(drag_scale=k)
                            per_lap[f"top_{k}"] = score_laps(grid, layout, held, v_held)["top"]
                lap_rows.append(per_lap)
    folds = pd.DataFrame(fold_rows)
    laps = pd.concat(lap_rows, ignore_index=True) if lap_rows else pd.DataFrame()
    yardsticks = pd.concat(yard_rows, ignore_index=True) if yard_rows else pd.DataFrame()
    return folds, laps, yardsticks


def by_pace(
    laps: pd.DataFrame,
    yardsticks: pd.DataFrame,
    measures: tuple[str, ...] = YARDSTICK_MEASURES,
) -> pd.DataFrame:
    """The measures per car, layout and level of pace: held-out laps
    (``laps`` of :func:`run_folds`) joined to their yardsticks, as the
    median over days of the day's median lap. Columns: ``car, layout_id,
    level, laps, days``, then per measure of ``measures`` the simulated
    lap's (:data:`SIZE_MEASURES` as sizes), ``<measure>_other_day`` and
    ``<measure>_same_day``."""
    if laps.empty or yardsticks.empty or any(k not in laps.columns for k in measures):
        return pd.DataFrame()
    keys = ["car", "layout_id", "session_id", "lap_num"]
    sim = laps[keys + list(measures)].copy()
    sized = [k for k in measures if k in SIZE_MEASURES]
    sim[sized] = sim[sized].abs()
    both = yardsticks.merge(sim, on=keys, how="inner")
    cols = [f"{k}{tag}" for k in measures for tag in ("", "_other_day", "_same_day")]
    cell = ["car", "layout_id", "level"]
    days = both.groupby(cell + ["date"], sort=False)[cols].median()
    out = days.groupby(cell, sort=False)[cols].median()
    out.insert(0, "laps", both.groupby(cell, sort=False).size())
    out.insert(1, "days", days.groupby(cell, sort=False).size())
    return out.reset_index()


#: What the measures are called in the report.
MEASURE_NAMES = {
    "S1": "speed round the lap (m/s)",
    "S2": "end of straights (m/s)",
    "S3": "apex (m/s)",
    "brake_g": "braking (g)",
    "lat_heat": "lateral heat (log ratio)",
    "side": "lateral heat, right less left (log ratio)",
}


def attainment(pace: pd.DataFrame, measures: tuple[str, ...] = YARDSTICK_MEASURES) -> pd.DataFrame:
    """The gate as a score (owner, 2026-11-21: not a hard pass or fail;
    the goal is to raise the number of passes). One row per car, layout,
    level of pace and measure (:func:`by_pace`): ``simulated``,
    ``other_day`` and ``same_day`` (two logged laps), ``scored`` (the cell
    has both a simulated figure and logged laps of another day to set it
    against) and ``passed``: the simulated lap is no further from the
    logged lap than a logged lap of another day is."""
    rows = []
    for r in pace.to_dict("records"):
        for k in measures:
            sim, other = float(r[k]), float(r[f"{k}_other_day"])
            scored = bool(np.isfinite(sim) and np.isfinite(other))
            rows.append(
                {
                    "car": r["car"],
                    "layout_id": r["layout_id"],
                    "level": r["level"],
                    "measure": k,
                    "laps": int(r["laps"]),
                    "days": int(r["days"]),
                    "simulated": sim,
                    "other_day": other,
                    "same_day": float(r[f"{k}_same_day"]),
                    "scored": scored,
                    "passed": bool(scored and sim <= other),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the verdict


def inside(folds: pd.DataFrame) -> pd.Series:
    """Whether each row of ``folds`` is inside the limits."""
    with np.errstate(invalid="ignore"):
        return (
            (folds["S1"] <= folds["floor"] + S1_OVER_FLOOR_MAX)
            & (folds["S2"].abs() <= S2_MAX)
            & (folds["S3"].abs() <= S3_MAX)
        )


def verdict(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(wide, by_car)``: one row per fold with each model's statistics
    and ``inside`` side by side and ``powered_<control>``; and per car and
    kind the counts and ``result`` (``pass``, ``fail`` or ``not scored``)."""
    keys = ["car", "kind", "layout_id", "date"]
    f = folds.assign(inside=inside(folds))
    wide = f[f["model"] == "fit"][keys + ["laps", "floor", "S1", "S2", "S3", "brake_g", "inside"]]
    for name in MODELS[1:]:
        ctl = f[f["model"] == name][keys + ["S1", "S2", "S3", "inside"]]
        ctl = ctl.rename(columns={c: f"{c}_{name}" for c in ("S1", "S2", "S3", "inside")})
        wide = wide.merge(ctl, on=keys, how="left")
        wide[f"powered_{name}"] = wide[f"inside_{name}"].eq(False) & wide["floor"].notna()
    wide["powered"] = wide[[f"powered_{n}" for n in MODELS[1:]]].any(axis=1)
    rows = []
    for (car, kind), g in wide.groupby(["car", "kind"]):
        row = {
            "car": car,
            "kind": kind,
            "folds": int(len(g)),
            "no_floor": int(g["floor"].isna().sum()),
            "fit_inside": int(g["inside"].sum()),
        }
        for name in MODELS[1:]:
            row[f"powered_{name}"] = int(g[f"powered_{name}"].sum())
        row["fit_outside_on_powered"] = int((g["powered"] & ~g["inside"]).sum())
        rows.append(row)
    by = pd.DataFrame(rows)
    results = []
    for car, g in by.groupby("car"):
        kinds = set(g["kind"])
        scored = kinds == {"day", "layout"} and all(
            (g[f"powered_{name}"] > 0).all() for name in MODELS[1:]
        )
        if g["fit_outside_on_powered"].sum():
            results.append("fail")  # whatever else is missing, this sends the work back
        else:
            results.append("pass" if scored else "not scored")
    by["result"] = by["car"].map(dict(zip(sorted(by["car"].unique()), results)))
    return wide, by


def check_car_holdout(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    folds_dir: Path | None = None,
) -> GateReport:
    """Gate C3: a car's parameters predict its speed trace on days and at
    layouts the fit never saw, and two planted wrong cars are told apart.
    ``folds_dir`` is where gate B3 saved the tracks from other cars
    (``tracks_dir / "folds"`` when it is not given)."""
    tables = cm.as_families(smp.car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir))
    rows = cm.as_family(smp.build_laps(geo.clean_lap_index(root, cache_dir=cache_dir)))
    air = cm.air_state(root, geo.load_sessions(root))
    cars = [c for c in smp.MODEL_CARS if (tables.sessions["car"] == c).any()]
    layouts = sorted(str(k) for k in rows[rows["car"].isin(cars)]["layout_id"].unique())
    tracks = {
        k: smp.load_track(root, k, tracks_dir=tracks_dir, cache_dir=cache_dir) for k in layouts
    }
    try:
        shares: dict[str, ShareParams] | None = ho.production_shares(root)
    except FileNotFoundError:
        shares = None  # no fitted pressure model beside the dataset: heat is not scored
    full: dict[str, cg.Drive | None] = {}
    shifts: dict[str, dict[str, float]] = {}
    gears: dict[str, dict[str, tuple[float, ...]]] = {}
    for c in cars:
        best = cg.day_best_laps(rows, c)
        model, work = cm.fit_car(car_only(tables, c), c, air=air)
        shifts[c], gears[c] = cg.day_shifts(work), cg.day_gears(work)
        own = {k: v for k, v in tracks.items() if k in set(best["layout_id"])}
        lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
        full[c] = convene(
            model,
            tracks,
            best,
            air=air,
            shifts=shifts[c],
            gears=gears[c],
            laps=cg.car_use_laps(car_only(tables, c), lap_time, c, own),
        )
    all_folds, all_laps, all_yard = [], [], []
    for c in cars:
        # another car's parameters, its own default powertrain and shift point
        stand_in = full.get(OTHER_CAR.get(c, ""))
        other = stand_in.but(shifts={}, gears={}) if stand_in is not None else None
        folds, laps, yard = run_folds(
            tables,
            rows,
            tracks,
            c,
            other,
            air=air,
            shifts=shifts[c],
            gears=gears[c],
            shares=shares,
        )
        all_folds.append(folds)
        all_laps.append(laps)
        all_yard.append(yard)
    folds = pd.concat(all_folds, ignore_index=True)
    laps = pd.concat(all_laps, ignore_index=True)
    yardsticks = pd.concat(all_yard, ignore_index=True)
    wide, by = verdict(folds)
    rep = GateReport("C3", "the car on held-out days and layouts (`lapsim check-car-holdout`)")
    rep.notes.append(
        f"{len(wide)} folds of {MIN_FOLD_LAPS}+ laps ({int((wide['kind'] == 'day').sum())} by "
        f"day, {int((wide['kind'] == 'layout').sum())} by layout), "
        f"{int(laps[laps['kind'] == 'day']['S1'].notna().sum())} held-out laps by day. Controls: "
        f"`low_power` is the fold's own car with its power × {POWER_CONTROL}, the same limit "
        "and the fitted car's braking level in g at its typical use; `other_car` is "
        + ", ".join(f"{v} for the {k}" for k, v in OTHER_CAR.items() if k in cars)
        + "."
    )

    # 0. The use of the car's limit that the held-out laps solve to.
    if "use" in laps.columns and laps["use"].notna().any():
        got_use = laps[laps["use"].notna()]
        over = (
            got_use.assign(above=got_use["use"] > 1.0)
            .groupby(["car", "kind"], sort=False)
            .agg(
                laps=("use", "size"),
                above_1=("above", "sum"),
                use_p05=("use", lambda x: float(x.quantile(0.05))),
                use_median=("use", "median"),
                use_p95=("use", lambda x: float(x.quantile(0.95))),
                use_max=("use", "max"),
            )
            .reset_index()
        )
        rep.items.append(
            GateItem(
                "The driver's use of the car's limit on the held-out laps",
                None,
                "; ".join(
                    f"{c}: {int(g['above_1'].sum())} of {int(g['laps'].sum())} laps above 1, "
                    f"highest {g['use_max'].max():.3f}"
                    for c, g in over.groupby("car", sort=False)
                ),
                "reported (#312): a use above 1 is a lap the fold's limit says the car cannot "
                "corner hard enough for; the knob scales cornering and braking together, so "
                "such a lap is driven above the limit, not capped",
                "`use`: the solver's knob of a held-out lap, the share of the fold's own "
                "cornering limit (`mu_y`, read without the fold) that its lap time takes; "
                "q = −ln of it. `above_1`: the laps that need more than the limit.\n\n"
                + md_table(over, floats="{:.3f}"),
            )
        )

    # 1. The verdict per car.
    results = by.groupby("car")["result"].first()
    for c in cars:
        res = results.get(c, "not scored")
        g = by[by["car"] == c]
        rep.items.append(
            GateItem(
                f"{c}: inside on every powered fold, both controls outside",
                None,
                res
                + "; "
                + "; ".join(
                    f"{r.kind}: fitted inside {r.fit_inside}/{r.folds}, low power powered "
                    f"{r.powered_low_power}, other car powered {r.powered_other_car}, fitted "
                    f"outside on a powered fold {r.fit_outside_on_powered}"
                    for r in g.itertuples()
                ),
                f"reported: S1 ≤ day floor + {S1_OVER_FLOOR_MAX}, |S2| ≤ {S2_MAX}, |S3| ≤ {S3_MAX} "
                "m/s over every quick lap; a powered day fold and a powered layout fold per "
                "control",
                md_table(wide[wide["car"] == c].drop(columns=["car"]), floats="{:.2f}"),
            )
        )

    # 2. Against the logged lap, at three levels of pace: the gate's score.
    day_laps = laps[laps["kind"] == "day"]
    pace = by_pace(day_laps, yardsticks)

    # Gate D3 of #206: lateral heat by side, in this gate's form.
    by_side = by_pace(day_laps, yardsticks, MOMENT_MEASURES)
    if len(by_side):
        sides = attainment(by_side, MOMENT_MEASURES)
        sides["level"] = [f"P{lv:g}" for lv in sides["level"]]
        side_cells = sides[sides["scored"]]
        side_tally = "; ".join(
            f"{MEASURE_NAMES[str(k)]} {int(g['passed'].sum())} of {len(g)}"
            for k, g in side_cells.groupby("measure", sort=False)
        )
        worst = side_cells.loc[side_cells["simulated"].idxmax()] if len(side_cells) else None
        signed = side_signed(day_laps)
        # the same at a layout the car was not fitted on
        lay_laps = laps[laps["kind"] == "layout"]
        lay_sides = attainment(by_pace(lay_laps, yardsticks, MOMENT_MEASURES), MOMENT_MEASURES)
        lay_cells = lay_sides[lay_sides["scored"]] if len(lay_sides) else lay_sides
        lay_note, lay_detail = "", ""
        if len(lay_cells):
            lay_sides["level"] = [f"P{lv:g}" for lv in lay_sides["level"]]
            lay_note = (
                f"; held out by layout {int(lay_cells['passed'].sum())} of {len(lay_cells)} "
                f"(furthest {lay_cells['simulated'].max():.3f})"
            )
            lay_detail = (
                "\n\n**Held out by layout** (the car fitted without the layout; signed, then "
                "by level of pace).\n\n"
                + md_table(side_signed(lay_laps), floats="{:.3f}")
                + "\n\n"
                + md_table(
                    lay_sides[
                        ["car", "layout_id", "level", "measure", "laps", "days", "simulated"]
                        + ["other_day", "same_day", "scored", "passed"]
                    ],
                    floats="{:.3f}",
                )
            )
        blind = signed[~signed["sees_mirror"]] if len(signed) else signed
        blind_note = (
            "; a mirrored track would read under "
            f"{MIRROR_DEFECT:g} by side, and so not be seen, at "
            + ", ".join(
                f"{r['car']} {r['layout_id']} ({r['mirror']:.2f})" for r in blind.to_dict("records")
            )
            if len(blind)
            else "; a mirrored track would read "
            f"{signed['mirror'].min():.2f} or more by side at every layout"
        )
        rep.items.append(
            GateItem(
                "Lateral heat by side (gate D3 of #206)",
                None,
                f"{int(side_cells['passed'].sum())} of {len(side_cells)} cells; {side_tally}"
                + (
                    f"; furthest {worst['car']} at {worst['layout_id']} {worst['level']} "
                    f"{worst['measure']} {worst['simulated']:.3f} (another day's lap "
                    f"{worst['other_day']:.3f})"
                    if worst is not None
                    else ""
                )
                + lay_note
                + blind_note,
                "a score, in gate C3's form (owner, 2026-11-23): the simulated lap's lateral heat "
                "is no further from the logged lap's than a logged lap of another day is; not "
                "counted in the attainment above",
                "`lat_heat`: the log of the simulated lap's lap-mean `V·|lat|` over the logged "
                "lap's; `side`: the same for right turns less left turns. The simulated lap's "
                "lateral is what the tyres hold on the track's curvature and bank at its speed; "
                "the logged lap's is its own (the Doppler lateral on the road's bank, at the "
                "track's nodes), so a wrong sign, a mirrored track or a curvature that is wrong on "
                "one side shows here where a comparison on the track's curvature on both sides "
                "would not. Held out by day, the median over days of the day's median lap, sizes "
                "compared; `other_day` and `same_day` are the same two measures between two "
                "logged laps, each with its own lateral. A `side` of 0.12 or more is a sign or "
                "mirror defect; between 0.06 and 0.12 it points at the track's curvature by side "
                "(#206).\n\n**Signed, and what a mirror would read.** Every held-out lap of "
                "the day folds: `lat_heat` and `side` with their signs (above zero: the simulated "
                "lap has more lateral heat; more of it in the right turns), and `mirror`: the "
                "`side` a mirrored track or a swapped sign would read at this layout, twice the "
                "log of the logged laps' right over left. Where the two sides of a layout carry "
                "alike, `side` cannot see a mirror.\n\n"
                + md_table(signed, floats="{:.3f}")
                + "\n\n**By level of pace.**\n\n"
                + md_table(
                    sides[
                        ["car", "layout_id", "level", "measure", "laps", "days", "simulated"]
                        + ["other_day", "same_day", "scored", "passed"]
                    ],
                    floats="{:.3f}",
                )
                + lay_detail,
            )
        )
    use_note = ""
    if len(pace):
        grid = attainment(pace)
        grid["level"] = [f"P{lv:g}" for lv in grid["level"]]
        got = grid[grid["scored"]]
        # the same cells with the car braked at its limit: what the driver's use is worth
        at_limit = {f"{k}_limit": k for k in YARDSTICK_MEASURES}
        if set(at_limit) <= set(day_laps.columns):
            was = attainment(
                by_pace(
                    day_laps.drop(columns=list(YARDSTICK_MEASURES)).rename(columns=at_limit),
                    yardsticks,
                )
            )
            was["level"] = [f"P{lv:g}" for lv in was["level"]]
            cell = ["car", "layout_id", "level", "measure"]
            both = got.merge(was[cell + ["simulated", "passed"]], on=cell, suffixes=("", "_limit"))
            won = both["passed"] & ~both["passed_limit"]
            lost = ~both["passed"] & both["passed_limit"]
            tally = (
                both.assign(won=won, lost=lost)
                .groupby("measure", sort=False)
                .agg(
                    cells=("passed", "size"),
                    passed=("passed", "sum"),
                    at_the_limit=("passed_limit", "sum"),
                    won=("won", "sum"),
                    lost=("lost", "sum"),
                )
            )
            tally.index = [MEASURE_NAMES[k] for k in tally.index]
            use_note = (
                f"{int(both['passed_limit'].sum())} of {len(both)} with the car braked at its "
                f"limit: the driver's braking use wins {int(won.sum())} cells and loses "
                f"{int(lost.sum())}"
            )
            use_table = (
                "\n\n**The driver's braking use** (the car's `driver`, read in each fold from "
                "the laps that are left): the same cells with the fold's car braked at its limit "
                "instead, `mu_y` set again (`at_the_limit`).\n\n"
                + md_table(tally.reset_index(names="measure"))
                + "\n\nCells that change:\n\n"
                + md_table(
                    both[won | lost][cell + ["simulated", "simulated_limit", "other_day"]].assign(
                        change=np.where(won[won | lost], "won", "lost")
                    ),
                    floats="{:.3f}",
                )
            )
        else:
            use_table = ""
        rep.score = (
            f"{int(got['passed'].sum())} of {len(got)} cells (car × layout × level of pace × "
            f"measure) no further from the logged lap than a logged lap of another day is; "
            f"{int((~grid['scored']).sum())} cells have no such lap to be set against"
        )

        def _tally(frame: pd.DataFrame) -> str:
            return f"{int(frame['passed'].sum())} of {len(frame)}"

        count = got.groupby(["car", "layout_id", "measure"], sort=False)["passed"].agg(
            ["sum", "size"]
        )
        count["tally"] = [f"{int(a)} of {int(b)}" for a, b in zip(count["sum"], count["size"])]
        board = (
            count["tally"]
            .unstack("measure")
            .reindex(columns=list(YARDSTICK_MEASURES))
            .rename(columns=MEASURE_NAMES)
            .fillna("")
            .reset_index()
        )
        detail = (
            "A cell is one car at one layout at one level of pace on one measure. A level's "
            "laps are those above that percentile of pace among the car's laps at the layout "
            f"({', '.join(f'P{lv:g}' for lv in PACE_LEVELS)}: to see that the simulation holds "
            "as the pace rises). The car is fitted without the lap's day and the lap driven at "
            "its own lap time. `simulated`: the simulated lap against the logged one; "
            "`other_day` and `same_day`: another logged lap of the same set against it, from "
            "other days and from its own. Each figure is the median over days of the day's "
            "median lap. A cell passes when `simulated` is no more than `other_day`; one "
            "without laps of another day in its set is not scored. Speed round the lap is the "
            "RMS difference of speed; the end of the straights and the apex are sizes; braking "
            f"is the RMS difference of deceleration where the logged lap slows by "
            f"{BRAKE_MATCH_MIN_G} g or more.\n\nPasses of the levels scored, per car and "
            "layout:\n\n" + md_table(board)
        )
        by_measure = ", ".join(
            f"{MEASURE_NAMES[k].split(' (')[0]} {_tally(got[got['measure'] == k])}"
            for k in YARDSTICK_MEASURES
        )
        by_level = ", ".join(f"{lv} {_tally(g)}" for lv, g in got.groupby("level", sort=False))
        for k in YARDSTICK_MEASURES:
            one = grid[grid["measure"] == k].drop(columns=["measure", "scored"])
            one = one.assign(
                passed=np.where(
                    grid.loc[one.index, "scored"], np.where(one["passed"], "pass", "no"), ""
                )
            )
            detail += f"\n\n**{MEASURE_NAMES[k]}**\n\n" + md_table(one, floats="{:.3f}")
        detail += use_table
        rep.items.insert(
            0,
            GateItem(
                "Attainment",
                None,
                f"{_tally(got)} cells; by measure: {by_measure}; by level: {by_level}"
                + (f"; {use_note}" if use_note else ""),
                "a score to be raised, not a pass or a fail (owner, 2026-11-21): the cells in "
                "which the simulated lap is no further from the logged lap than a logged lap of "
                "another day is",
                detail,
            ),
        )

    fit = folds[folds["model"] == "fit"]

    # 3. Top-speed band on the layouts that reach 50 m/s.
    top_cols = [f"top_{k}" for k in DRAG_GRID if f"top_{k}" in laps.columns]
    lay = laps[laps["kind"] == "layout"]
    if top_cols:
        band = lay.dropna(subset=top_cols).groupby(["car", "layout_id"])[top_cols].median()
        band["best"] = band[top_cols].min(axis=1)
        band["model_over_best"] = band["top_1.0"] - band["best"]
        band = band.reset_index()
        c_used = fit[fit["kind"] == "layout"].set_index(["car", "layout_id"])["c_drag"]
        band["c_drag_e4"] = [
            1e4 * c_used.get(k, np.nan) for k in zip(band["car"], band["layout_id"])
        ]
        rep.items.append(
            GateItem(
                "Top-speed band",
                None,
                "; ".join(
                    f"{r.car} {r.layout_id} +{r.model_over_best:.2f}" for r in band.itertuples()
                )
                + " m/s over the best of the grid",
                f"reported: the held-out layout's top quarter of speeds: RMSE with the model's drag within "
                f"{TOP_BAND_MAX} m/s of the best over {DRAG_GRID} × it",
                "The drag is the one the car has without the held-out layout (`c_drag_e4`); "
                "`top_k`: RMS over the nodes where the lap is in its top quarter of speed, at "
                "k × that drag.\n\n" + md_table(band, floats="{:.2f}"),
            )
        )

    # 4. Cornering drag: does it earn its place?
    if "S1_other" in lay.columns:
        on, off = ("", "_other") if sv.CORNER_DRAG_DEFAULT else ("_other", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                top_with=(f"top{on}", "median"),
                top_without=(f"top{off}", "median"),
            )
            .reset_index()
        )
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        top_lower = int((term["top_with"] < term["top_without"]).sum())
        rep.items.append(
            GateItem(
                "Cornering drag as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells and "
                f"the top quarter of speeds in {top_lower} of {len(term)}",
                "in the default only if it lowers the error on the top quarter of speeds and "
                "does not raise the leave-one-layout-out error; "
                + ("it is on" if sv.CORNER_DRAG_DEFAULT else "it is off"),
                md_table(term, floats="{:.2f}"),
            )
        )

    # 4b. The time lost in the upshifts: the same question.
    if "S1_shift" in lay.columns:
        on, off = ("", "_shift") if sv.SHIFT_LOSS_DEFAULT else ("_shift", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                top_with=(f"top{on}", "median"),
                top_without=(f"top{off}", "median"),
            )
            .reset_index()
        )
        term = term[(term["S1_with"] != term["S1_without"])].reset_index(drop=True)
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        top_lower = int((term["top_with"] < term["top_without"]).sum())
        rep.items.append(
            GateItem(
                "Shift time as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells and "
                f"the top quarter of speeds in {top_lower} of {len(term)}",
                "in the default only if it lowers the error on the top quarter of speeds and "
                "does not raise the leave-one-layout-out error; "
                + ("it is on" if sv.SHIFT_LOSS_DEFAULT else "it is off"),
                "The cells of the cars driven through their gears with a measured shift time "
                "(a car on one power level has no shift).\n\n" + md_table(term, floats="{:.2f}"),
            )
        )

    # 4c. The braking limit's shape: the same question, and the braking.
    if "S1_brake" in lay.columns:
        on, off = ("", "_brake") if sv.BRAKE_SHAPE_DEFAULT else ("_brake", "")
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=(f"S1{on}", "median"),
                S1_without=(f"S1{off}", "median"),
                S3_with=(f"S3{on}", "median"),
                S3_without=(f"S3{off}", "median"),
                brake_g_with=(f"brake_g{on}", "median"),
                brake_g_without=(f"brake_g{off}", "median"),
            )
            .reset_index()
        )
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        nearer = int((term["brake_g_with"] < term["brake_g_without"]).sum())
        rep.items.append(
            GateItem(
                "The shape of the braking under cornering as a term of the default",
                None,
                f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells and "
                f"the braking nearer the logged lap's in {nearer} of {len(term)}",
                "in the default only if it does not raise the leave-one-layout-out error and "
                "brings the braking nearer the logged lap's (`brake_g`); "
                + ("it is on" if sv.BRAKE_SHAPE_DEFAULT else "it is off"),
                "With the shape the laps are braked on (the driver's held one where the car "
                "carries a braking use, else the limit's `brake_ellipse`) against the diamond."
                "\n\n" + md_table(term, floats="{:.3f}"),
            )
        )

    # 4c''. The driven axle: against the drive on the whole car's grip.
    if "S1_axle" in laps.columns:
        days = laps[laps["kind"] == "day"]
        tally_text, change_table = "", ""
        swap = {f"{k}_axle": k for k in YARDSTICK_MEASURES}
        if len(days) and set(swap) <= set(days.columns):
            cell = ["car", "layout_id", "level", "measure"]
            here = attainment(by_pace(days, yardsticks))
            there = attainment(
                by_pace(
                    days.drop(columns=list(YARDSTICK_MEASURES)).rename(columns=swap), yardsticks
                )
            )
            both = here[here["scored"]].merge(
                there[cell + ["simulated", "passed"]], on=cell, suffixes=("", "_other")
            )
            won = both["passed"] & ~both["passed_other"]
            lost = ~both["passed"] & both["passed_other"]
            per_car = (
                both.assign(won=won, lost=lost)
                .groupby("car", sort=False)
                .agg(
                    cells=("passed", "size"),
                    passed=("passed", "sum"),
                    other=("passed_other", "sum"),
                    won=("won", "sum"),
                    lost=("lost", "sum"),
                )
                .reset_index()
            )
            tally_text = (
                f"{int(both['passed'].sum())} of {len(both)} cells with the default, "
                f"{int(both['passed_other'].sum())} the other way: it wins {int(won.sum())} and "
                f"loses {int(lost.sum())}; "
            )
            change_table = (
                "\n\nBy car (`other`: the cells passed the other way):\n\n"
                + md_table(per_car)
                + "\n\nCells that change:\n\n"
                + md_table(
                    both[won | lost][cell + ["simulated", "simulated_other", "other_day"]].assign(
                        change=np.where(won[won | lost], "won", "lost")
                    ),
                    floats="{:.3f}",
                )
            )
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=("S1", "median"),
                S1_without=("S1_axle", "median"),
                S2_with=("S2", "median"),
                S2_without=("S2_axle", "median"),
                S3_with=("S3", "median"),
                S3_without=("S3_axle", "median"),
                u_with=("u", "median"),
                u_without=("u_axle", "median"),
            )
            .reset_index()
        )
        cells = [g for _, g in lay.groupby(["car", "layout_id"])]
        for name, kind in (("heat", "h"), ("heat_carried", "hc")):
            term[f"{name}_with"] = [worst_tyre(g, kind) for g in cells]
            term[f"{name}_without"] = [worst_tyre(g, kind, "_axle") for g in cells]
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        apex = int((term["S3_with"].abs() < term["S3_without"].abs()).sum())
        measured = tally_text + (
            f"leave-one-layout-out S1 is lower with the default in {lower} of {len(term)} cells "
            f"and the apex speed nearer in {apex}"
        )
        if term["heat_carried_with"].notna().any():
            measured += (
                f"; worst-tyre heat with the pace carried "
                f"{100 * float(term['heat_carried_with'].median()):.1f} % with the default, "
                f"{100 * float(term['heat_carried_without'].median()):.1f} % the other way, and "
                f"at the lap's own time {100 * float(term['heat_with'].median()):.1f} % against "
                f"{100 * float(term['heat_without'].median()):.1f} %"
            )
        rep.items.append(
            GateItem(
                "Drive through the driven axle as a term of the default",
                None,
                measured,
                "reported; it is "
                + ("on" if sv.DRIVEN_AXLE_DEFAULT else "off")
                + " for all three cars by the owner's decision of 2026-11-23 (issue #312; the "
                "committee's default was the Inferno 86 only)",
                "The grip left for drive is the driven axle's share of what the cornering "
                "leaves (`grip.drive_share`: the rear's share of the static weight from "
                "production's facts, 0.47 on the Inferno 86 and 0.58 on the formula cars, nothing "
                "fitted) against the whole car's. Each way the fold's driver's use is read "
                "again. `_with` is the default and `_without` the other rule."
                + change_table
                + "\n\nLeave-one-layout-out, medians over a cell's laps:\n\n"
                + md_table(term, floats="{:.3f}"),
            )
        )

    # 4c'. The driver's braking use: against the car braked at its limit.
    if "S1_limit" in lay.columns:
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=("S1", "median"),
                S1_without=("S1_limit", "median"),
                S3_with=("S3", "median"),
                S3_without=("S3_limit", "median"),
                brake_g_with=("brake_g", "median"),
                brake_g_without=("brake_g_limit", "median"),
                u_with=("u", "median"),
                u_without=("u_limit", "median"),
            )
            .reset_index()
        )
        cells = [g for _, g in lay.groupby(["car", "layout_id"])]
        for name, kind in (("heat", "h"), ("heat_carried", "hc")):
            term[f"{name}_with"] = [worst_tyre(g, kind) for g in cells]
            term[f"{name}_without"] = [worst_tyre(g, kind, "_limit") for g in cells]
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        nearer = int((term["brake_g_with"] < term["brake_g_without"]).sum())
        apex = int((term["S3_with"].abs() < term["S3_without"].abs()).sum())
        measured = (
            f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells, the "
            f"braking nearer the logged lap's in {nearer} and the apex speed nearer in {apex}"
        )
        if term["heat_carried_with"].notna().any():
            cooler = int((term["heat_carried_with"] < term["heat_carried_without"]).sum())
            measured += (
                f"; worst-tyre heat with the pace carried "
                f"{100 * float(term['heat_carried_with'].median()):.1f} % with it, "
                f"{100 * float(term['heat_carried_without'].median()):.1f} % without (lower in "
                f"{cooler}), and at the lap's own time "
                f"{100 * float(term['heat_with'].median()):.1f} % against "
                f"{100 * float(term['heat_without'].median()):.1f} %"
            )
        rep.items.append(
            GateItem(
                "The driver's braking use as a term of the default",
                None,
                measured,
                "reported; it is on by the owner's decision of 2026-11-21 (the attainment "
                "committee's step 1)",
                "With the laps braked as the driver brakes (the level and the held shape read "
                "from the car's other layouts) against the car braked at its limit, `mu_y` set "
                "by the convention each way. The day folds' cells are in the attainment item. "
                "`heat` and `heat_carried` as in the bank's item below.\n\n"
                + md_table(term, floats="{:.3f}"),
            )
        )

    # 4d. The road's bank: against every road taken as flat.
    if "S1_flat" in lay.columns:
        term = (
            lay.groupby(["car", "layout_id"])
            .agg(
                laps=("S1", "count"),
                S1_with=("S1", "median"),
                S1_without=("S1_flat", "median"),
                S3_with=("S3", "median"),
                S3_without=("S3_flat", "median"),
                u_with=("u", "median"),
                u_without=("u_flat", "median"),
            )
            .reset_index()
        )
        cells = [g for _, g in lay.groupby(["car", "layout_id"])]
        for name, kind in (("heat", "h"), ("heat_carried", "hc")):
            term[f"{name}_with"] = [worst_tyre(g, kind) for g in cells]
            term[f"{name}_without"] = [worst_tyre(g, kind, "_flat") for g in cells]
        lower = int((term["S1_with"] < term["S1_without"]).sum())
        apex = int((term["S3_with"].abs() < term["S3_without"].abs()).sum())
        nearer = int(((term["u_with"] - 1).abs() < (term["u_without"] - 1).abs()).sum())
        measured = (
            f"with it, leave-one-layout-out S1 is lower in {lower} of {len(term)} cells, the "
            f"apex speed nearer in {apex} and the held-out layout's u nearer 1 in {nearer}; "
            f"median |u − 1| {float((term['u_with'] - 1).abs().median()):.3f} with it, "
            f"{float((term['u_without'] - 1).abs().median()):.3f} without"
        )
        if term["heat_carried_with"].notna().any():
            cooler = int((term["heat_carried_with"] < term["heat_carried_without"]).sum())
            measured += (
                f"; worst-tyre heat with the pace carried "
                f"{100 * float(term['heat_carried_with'].median()):.1f} % with it, "
                f"{100 * float(term['heat_carried_without'].median()):.1f} % without (lower in "
                f"{cooler}), and at the lap's own time "
                f"{100 * float(term['heat_with'].median()):.1f} % against "
                f"{100 * float(term['heat_without'].median()):.1f} %"
            )
        rep.items.append(
            GateItem(
                "The road's bank as a term of the default",
                None,
                measured,
                "reported; it is on by the owner's decision of 2026-11-17",
                "The bank is the track's, from the elevation model at full strength; without it "
                "every road is taken as flat (`Profile.from_track(track, bank=False)`). `mu_y` "
                "is set again by the convention on the layouts that are left, each way. `u`: "
                "the median over the held-out layout's laps of the pace knob, which is 1 where "
                "the car's level carries to a layout it was not set on. `heat`: the largest "
                "over the four tyres of the absolute median over the laps of log(simulated / "
                "logged lap-mean heat), through production's share model, in the tyres' frame "
                "on the bank and the ground frame without it; `heat_carried`: the same with "
                "each lap driven at its own u over the layout's reference u, the pace level "
                "carried from the car's other layouts.\n\n" + md_table(term, floats="{:.3f}"),
            )
        )

    # 5. The apex speed corner by corner.
    by_corner = corner_table(laps, {k: Layout.of(v) for k, v in tracks.items()})
    if len(by_corner):
        agree = corner_agreement(by_corner)
        worst = by_corner.reindex(by_corner["error_ms"].abs().sort_values(ascending=False).index)
        rep.items.append(
            GateItem(
                "The apex speed corner by corner",
                None,
                "furthest: "
                + "; ".join(
                    f"{r.car} {r.layout_id} corner {r.corner} ({r.v_apex_ms:.0f} m/s) "
                    f"{r.error_ms:+.2f}"
                    for r in worst.head(5).itertuples()
                )
                + " m/s; between cars over a layout's corners: "
                + (
                    "; ".join(
                        f"{r.layout_id} {r.car_a} / {r.car_b} r = {r.r:+.2f} ({r.corners})"
                        for r in agree.itertuples()
                    )
                    if len(agree)
                    else "no layout with two cars"
                ),
                "reported (issue #205: the per-corner residual table)",
                "Held out by day: the simulated lap's lowest speed in ± "
                f"{APEX_WINDOW_M:.0f} m of each apex less the logged lap's (`error_ms`; the "
                "median over days of the day's median lap), with the logged one (`v_apex_ms`), "
                "the corner's place on the lap and its radius. `S3` is the mean of these over "
                "a lap's corners, so errors of opposite sign cancel in it and stand here. An "
                "error that repeats between unlike cars at the same corners is the track's (its "
                "bank, its line through the corner); one that does not is the car's or the "
                "driver's.\n\n"
                + md_table(by_corner, floats="{:.2f}")
                + "\n\nThe correlation over a layout's corners of two cars' errors:\n\n"
                + (md_table(agree, floats="{:.2f}") if len(agree) else "_none_"),
            )
        )

    # 6. Fold summaries.
    summary = (
        folds.groupby(["car", "kind", "model"])
        .agg(
            folds=("S1", "size"),
            S1=("S1", "median"),
            S2=("S2", "median"),
            S3=("S3", "median"),
            u=("u", "median"),
            brake_use=("brake_use", "median"),
            held_shape=("held_shape", "median"),
            mu_y=("mu_y", "median"),
            p_specific=("p_specific", "median"),
        )
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "The three models through the folds",
            None,
            "; ".join(
                f"{r.car} {r.kind} {r.model} {r.S1:.2f} / {r.S2:+.2f} / {r.S3:+.2f}"
                for r in summary[summary["kind"] == "layout"].itertuples()
            ),
            "median over folds of S1 / S2 / S3 (m/s); reported",
            md_table(summary, floats="{:.3f}"),
        )
    )
    # Gates E1 and E2 (arm B) of #207: heat per tyre on the same folds.
    if shares is not None and len(laps) and "sim_q_fl" in laps.columns:
        from . import heat_match

        rep.items.extend(
            heat_match.report_items(
                root,
                laps,
                shares,
                rows.set_index(["session_id", "lap_num"])["lap_time_s"],
                tracks_dir=tracks_dir,
                cache_dir=cache_dir,
                folds_dir=folds_dir,
                transfer={
                    "tables": tables,
                    "rows": rows,
                    "tracks": tracks,
                    "full": full,
                    "shifts": shifts,
                    "gears": gears,
                    "air": air,
                    "index": cm.as_family(geo.clean_lap_index(root, cache_dir=cache_dir)),
                },
            )
        )
    return rep


__all__ = [
    "Layout",
    "apex_errors",
    "attainment",
    "corner_agreement",
    "corner_frame",
    "corner_table",
    "laps_left",
    "by_pace",
    "braking_match",
    "car_only",
    "check_car_holdout",
    "convene",
    "deceleration_g",
    "fast_laps",
    "fold_car",
    "heat_errors",
    "inside",
    "lap_stats",
    "pace_yardsticks",
    "run_folds",
    "score_laps",
    "verdict",
    "without",
    "worst_tyre",
]
