"""Gate F (#208): the pressure a simulated lap leads to at a track the car
has not run, against paired real days (``lapsim check-pressure``).

The question, as the owner approved it on 2026-11-24: **is the cold
pressure the calculators would recommend from a simulated lap as near to
the one a day's own logged laps give as another logged day is?** The
mechanics are :mod:`.pressure`; this module is the folds, the arms and the
report. The gate is a report in the form of gates C3, D2, D3 and E: it
computes no pass or fail beyond the checks on its own harness.

**A fold** is a pooled car at a track (production pools the formula cars
and keys its tables by track): every session of the cell is left out of
the fit, whether it could be scored or not, and the fold's model (its
force shares, tau, K, kappa, compound multipliers and every other cell's
entries) is what every arm is read through. A fold is fitted once and kept
in ``workspace/lapsim/cache/pressure_folds/``.

**A day** is a dry held-out session with three ordinary flying laps
(:func:`~.pressure.mark_laps`); its recommendation is from its own heat at
its own lap time (:func:`~.pressure.day_entry`). The figure of an arm is
its distance from the day's recommendation in mbar, per pit-exit stint and
tyre; a cell is a raw car at a track on one tyre: the median over days of
the day's median stint. Two forms:

- ``heat``, at a given lap time: every side at the day's lap time, on the
  out-lap the fold model itself answers with (the same on every side).
- ``chain``, nothing given: each side at the lap time and with the
  out-lap its own entry supplies.

**The arms.** ``other_day``: the yardstick, a logged day of the same raw
car, powertrain and kind of source at the track. In ``heat`` it is the
heat of up to three sessions of other days nearest in lap time, each from
its laps nearest the day's lap time, the pair's lap-time gap kept; in
``chain`` every other day as if it alone had written the entry.
``ceiling``: the logged entry built without the day. ``sim``: the
simulated entry (:func:`sim_cell`). ``other_layouts`` and ``fallback``:
what the simulation replaces (#260). The planted controls: ``heat_up`` /
``heat_down`` (the day's own heat times 1.05 and 0.95), ``low_power`` and
``other_car`` (gate C3's wrong cars, driven through the same chain as the
simulation), ``lap_up`` / ``lap_down`` (the simulation's own lap time
moved by 1 %).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import car_gate as cg
from . import car_holdout as ch
from . import geometry as geo
from . import heat_gate as hg
from . import heat_match as hm
from . import holdout as ho
from . import pace
from . import pressure as pr
from . import samples as smp
from . import served as srv
from . import track as trk
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS

logger = logging.getLogger(__name__)

#: A raw car's cell is scored with this many dry days, as in gate E.
MIN_DAYS = 3
#: How far past 1 a planted car's table runs: its knob is a free scale.
FREE_KNOB = 2.0
#: Leave-date-out folds: the reference level on cells the fit has seen.
DATE_FOLDS = 10
#: The planted errors: a heat scaled by this either way, a lap time by this.
HEAT_PLANT = 0.05
LAP_TIME_PLANT = 0.01
#: The curve nodes two cars' simulated entries share before they are
#: pooled: multiples of the reference lap time.
POOL_GRID = np.exp(np.linspace(np.log(0.9), np.log(1.6), 14))
WET = ("wet", "damp")
FORMS = ("heat", "chain")
FORM_NAMES = {"heat": "heat, at a given lap time", "chain": "the whole chain, nothing given"}
#: The simulated arms and how each one's table is built (:func:`~.pace.build_table`).
SIM_ARMS: dict[str, dict] = {
    "sim": {},
    "low_power": {"power_scale": ch.POWER_CONTROL, "use_cap": False, "use_fast": FREE_KNOB},
    "other_car": {"use_cap": False, "use_fast": FREE_KNOB},
}
#: The source files a fold's fit is made by (beside ``tire_model/``: the
#: loaders the fit reads the timeseries through): a cached fold is of
#: their bytes.
_FIT_SOURCES = (
    "../tire_etl/kinematics.py",
    "../tire_etl/dataset.py",
    "warmup_table.py",
    "statespace.py",
    "heat_input.py",
    "energy_balance.py",
    "compound_infer.py",
    "compounds.py",
    "pace_sim.py",
    "cleaning.py",
    "wetness.py",
    "solar.py",
    "heat_frame.py",
)
#: What a fold fitted in the tyres' frame is made by beside them (#376): the
#: rebuilding of the series from GPS and the track, and the committed tracks.
_TYRES_SOURCES = (
    "lapsim/tyres_frame.py",
    "lapsim/geometry.py",
    "lapsim/frame.py",
    "lapsim/bank.py",
    "lapsim/track.py",
)
_FIT_SIDECARS = (
    "sensor_blacklist.yaml",
    "tire_compounds.yaml",
    "session_overrides.yaml",
    "condition_overrides.yaml",
)


def _shipped_aliases() -> dict[str, str]:
    from ..warmup_table import CAR_FIT_ALIASES

    return dict(CAR_FIT_ALIASES)


# ------------------------------------------------------- production's side


@dataclass
class Production:
    """The dataset as the pressure model reads it. ``sessions`` has every
    session with its raw ``car`` and the ``pooled`` name the model fits it
    under; ``raw_laps`` every lap, cleaned, usable or not; ``trained`` the
    sessions that reach a fit (a usable lap with TPMS); ``conditions``
    production's label of those; ``targets`` the team's hot pressures."""

    root: Path
    sessions: pd.DataFrame
    raw_laps: pd.DataFrame
    trained: set[str]
    conditions: dict[str, str]
    targets: dict[tuple[str, str], float]


def load_production(root: Path) -> Production:
    from ..validate import _load_sessions_and_laps, _session_conditions
    from ..warmup_table import CAR_FIT_ALIASES, _load_filtered_laps, load_hot_pressure_targets

    sessions, raw_laps = _load_sessions_and_laps(root)
    sessions = sessions.assign(pooled=sessions["car"].replace(CAR_FIT_ALIASES))
    sessions["date"] = sessions["date"].map(lambda d: str(d)[:10])
    trained = set(_load_filtered_laps(root)["session_id"].astype(str))
    return Production(
        root=Path(root),
        sessions=sessions,
        raw_laps=raw_laps,
        trained=trained,
        conditions={str(k): str(v) for k, v in _session_conditions(root).items()},
        targets=load_hot_pressure_targets(Path(root)),
    )


def cell_sessions(prod: Production, pooled: str, track: str) -> list[str]:
    """Every session of a (pooled car, track), whatever its status or
    condition: what a fold leaves out of the fit (#208, finding 9a: a
    session too short to be scored still reaches training)."""
    own = prod.sessions[
        (prod.sessions["pooled"] == pooled) & (prod.sessions["track_canonical"] == track)
    ]
    return sorted(own["session_id"].astype(str))


def cell_table(prod: Production) -> pd.DataFrame:
    """The cells with a session in a fit, per raw car: its sessions that
    reach training and, of the dry ones, the sessions and days."""
    own = prod.sessions[prod.sessions["session_id"].astype(str).isin(prod.trained)].copy()
    own["condition"] = own["session_id"].astype(str).map(prod.conditions).fillna("unknown")
    rows = []
    for (pooled, track, car), grp in own.groupby(["pooled", "track_canonical", "car"], sort=True):
        dry = grp[grp["condition"] == "dry"]
        rows.append(
            {
                "pooled": pooled,
                "track": track,
                "raw_car": car,
                "left_out": len(cell_sessions(prod, str(pooled), str(track))),
                "trained": int(len(grp)),
                "dry_sessions": int(len(dry)),
                "dry_days": int(dry["date"].nunique()),
                "wet_damp_sessions": int(grp["condition"].isin(["wet", "damp"]).sum()),
            }
        )
    return pd.DataFrame(rows)


#: The pooling the pressure model ships with (``warmup_table.CAR_FIT_ALIASES``
#: as imported), which a cached fold is of unless :func:`split_pooling` says
#: otherwise.
_SHIPPED_POOLING = dict(_shipped_aliases())


def split_pooling(own: tuple[str, ...]) -> dict[str, str]:
    """An experiment's switch (#208, the owner's question of 2026-11-25:
    does splitting the pooled formula car help?): fit the raw cars ``own``
    as a pooled car of their own in place of the one they are pooled
    under. It changes ``warmup_table.CAR_FIT_ALIASES`` for the process
    (every module reads that one dict) and gives the new car the facts of
    the car it leaves; nothing is written. Returns the pooling in force. A
    fold fitted under it is cached apart.

    Several cars are named after all of them (``"KK-F+KK-F-V2"``), never
    after the first (#420). A raw car pooled under another name keeps
    logged entries of its own (#383) and a car pooled under no other name
    has only the pooled ones, so a group named ``"KK-F"`` took the raw
    KK-F's own entries away (it read the group's, the KK-F-V2's laps among
    them) while the KK-F-V2 kept its own: the switch moved entries as well
    as the pooling of the constants. One car split out alone keeps its
    name; its pooled entries are then its own."""
    from ..heat_input import CAR_FACTS
    from ..warmup_table import CAR_FIT_ALIASES

    if not own:
        return dict(CAR_FIT_ALIASES)
    name = own[0] if len(own) == 1 else "+".join(own)
    left = CAR_FIT_ALIASES.get(own[0], own[0])
    for raw in own:
        if raw == name:
            CAR_FIT_ALIASES.pop(raw, None)
        else:
            CAR_FIT_ALIASES[raw] = name
    if left in CAR_FACTS and name not in CAR_FACTS:
        CAR_FACTS[name] = CAR_FACTS[left]
    return dict(CAR_FIT_ALIASES)


def _fit_signature(root: Path) -> str:
    """What a cached fold is of: the dataset's files, the fit's source,
    the sidecars a fit reads and, where it is not the shipped one, the
    pooling of the cars."""
    from ..warmup_table import CAR_FIT_ALIASES

    h = hashlib.sha256(geo._index_signature(root).encode())
    if dict(CAR_FIT_ALIASES) != _SHIPPED_POOLING:
        h.update(repr(sorted(CAR_FIT_ALIASES.items())).encode())
    src = Path(__file__).resolve().parent.parent
    for name in _FIT_SOURCES:
        h.update((src / name).read_bytes())
    for name in _FIT_SIDECARS:
        f = Path(root) / name
        if f.exists():
            h.update(f.read_bytes())
    weather = Path(root) / "weather_hourly"
    for f in sorted(weather.glob("*/*.parquet")) if weather.exists() else []:
        st = f.stat()
        h.update(f"{f.name}:{st.st_mtime_ns}:{st.st_size};".encode())
    return h.hexdigest()


def _tyres_signature() -> str:
    """What a fit in the tyres' frame reads beside :func:`_fit_signature`:
    the code that rebuilds the series and the committed tracks' bytes."""
    from ..heat_frame import default_tracks_dir

    h = hashlib.sha256(b"heat-frame:tyres")
    src = Path(__file__).resolve().parent.parent
    for name in _TYRES_SOURCES:
        h.update((src / name).read_bytes())
    tracks = default_tracks_dir()
    for f in sorted(tracks.glob("*.json")) if tracks.exists() else []:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def fold_model(
    root: Path, ids: list[str], cache_dir: Path | None = None, heat_frame: str | None = None
) -> dict[str, Any]:
    """The pressure model fitted without ``ids``, nothing written under
    the dataset, on heat in ``heat_frame`` (:mod:`..heat_frame`; None: the
    frame in force). With ``cache_dir`` the fit is kept there, keyed by the
    sessions left out, :func:`_fit_signature` and, in the tyres' frame,
    :func:`_tyres_signature`: a fold is fitted once however often its arms
    are scored."""
    from ..heat_frame import SENSOR, resolve
    from ..validate import _fit_fold

    heat_frame = resolve(heat_frame)
    path = None
    if cache_dir is not None:
        sig = _fit_signature(root) + ("" if heat_frame == SENSOR else _tyres_signature())
        key = hashlib.sha256(("\n".join(sorted(ids)) + sig).encode()).hexdigest()
        path = Path(cache_dir) / "pressure_folds" / f"{key[:24]}.json"
        if path.exists():
            model: dict[str, Any] = json.loads(path.read_text())
            return model
    model = _fit_fold(root, ids, heat_frame)
    if "simulation" in model:
        raise RuntimeError("a fold's fit has simulated entries placed in it: the export saw it")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model))
        # read back, so that a fold is the same numbers fitted now or found in the cache
        model = json.loads(path.read_text())
    return model


@dataclass
class Held:
    """A fold: the cell left out, the model fitted without it and the
    held-out laps as that model reads them. ``days`` has one row per dry
    session with a recommendation of its own (``session_id, raw_car,
    date, layout_id, powertrain, source, lap_time_s``); ``entries`` those
    sessions' :func:`~.pressure.day_entry`."""

    root: Path
    pooled: str
    track: str
    model: dict[str, Any]
    prep: Any
    share: ShareParams
    laps: pd.DataFrame
    stints: pd.DataFrame
    days: pd.DataFrame
    entries: dict[str, pr.Entry]
    left_out: int = 0


def _majority(values: list[str]) -> str:
    return max(sorted(set(values)), key=values.count) if values else ""


def prepare(
    prod: Production, ids: list[str], cache_dir: Path | None = None
) -> tuple[dict[str, Any], Any] | None:
    """A fold's model and its held-out laps as that model reads them;
    None when none of ``ids`` has a usable lap."""
    from ..validate import _prepare_fold_laps

    model = fold_model(prod.root, ids, cache_dir)
    prep = _prepare_fold_laps(prod.root, ids, model)
    return None if prep is None else (model, prep)


def held_of(
    prod: Production,
    model: dict[str, Any],
    prep: Any,
    pooled: str,
    track: str,
    left_out: int,
    conditions: tuple[str, ...] = ("dry",),
) -> Held:
    """The held-out laps of one (pooled car, track) of a fold
    (:class:`Held`), its days being the sessions of ``conditions``."""
    share = ho.shares_of(model)[pooled]
    meta = prod.sessions.set_index(prod.sessions["session_id"].astype(str))
    laps = pr.mark_laps(prep, prod.raw_laps)
    laps = laps[(laps["car"] == pooled) & (laps["track_canonical"] == track)].reset_index(drop=True)
    laps["raw_car"] = laps["session_id"].map(meta["car"]).astype(str)
    laps["date"] = laps["session_id"].map(meta["date"]).astype(str)
    laps["layout_id"] = laps["session_id"].map(meta["layout_id"]).astype(str)
    stints = pr.stints_of(prep, prod.targets)
    stints = stints[(stints["car"] == pooled) & (stints["track_canonical"] == track)]
    ordinary = laps[laps["ordinary"] & laps["condition"].isin(conditions)]
    keys = ordinary[["session_id", "lap_num", "car"]]
    kinds = {k: v[1] for k, v in hg.sensor_lap_bins(prod.root, keys).items()} if len(keys) else {}
    entries: dict[str, pr.Entry] = {}
    rows = []
    for sid, own in laps[laps["condition"].isin(conditions)].groupby("session_id", sort=True):
        entry = pr.day_entry(prod.root, own, share)
        if entry is None:
            continue
        sid = str(sid)
        entries[sid] = entry
        raw_car, date = str(own["raw_car"].iloc[0]), str(own["date"].iloc[0])
        seen = [kinds.get((sid, int(n)), "") for n in own.loc[own["ordinary"], "lap_num"]]
        rows.append(
            {
                "session_id": sid,
                "raw_car": raw_car,
                "date": date,
                "layout_id": str(own["layout_id"].iloc[0]),
                "powertrain": cm.powertrain_of(raw_car, date),
                "source": _majority([s for s in seen if s]),
                "lap_time_s": entry.lap_time_s,
                "ordinary_laps": entry.n_laps,
                "condition": str(own["condition"].iloc[0]),
            }
        )
    cols = ["session_id", "raw_car", "date", "layout_id", "powertrain", "source", "lap_time_s"]
    days = pd.DataFrame(rows, columns=cols + ["ordinary_laps", "condition"])
    return Held(prod.root, pooled, track, model, prep, share, laps, stints, days, entries, left_out)


def hold_out(
    prod: Production, pooled: str, track: str, cache_dir: Path | None = None
) -> Held | None:
    """The fold of a (pooled car, track); None when none of its sessions
    has a usable lap."""
    ids = cell_sessions(prod, pooled, track)
    got = prepare(prod, ids, cache_dir)
    return None if got is None else held_of(prod, got[0], got[1], pooled, track, len(ids))


def date_folds(prod: Production, k: int = DATE_FOLDS) -> list[list[str]]:
    """Leave-date-out: the dates with a session in a fit, in order, dealt
    into ``k`` folds; a fold is every session of its dates, whatever the
    car or the track (a held-out session then has no same-day sibling of
    its cell in training, #208 finding 9b)."""
    own = prod.sessions[prod.sessions["session_id"].astype(str).isin(prod.trained)]
    dates = sorted(set(own["date"].astype(str)))
    folds = []
    for i in range(k):
        mine = set(dates[i::k])
        ids = sorted(prod.sessions.loc[prod.sessions["date"].isin(mine), "session_id"].astype(str))
        if ids:
            folds.append(ids)
    return folds


# ----------------------------------------------------------- the lap sim's


@dataclass
class Sim:
    """What gate C3 fits from, read once: the car tables and build laps by
    family, the air, the tracks, each car's own fit (``full``) with its
    days' shift points and gears, the laps the fed use is read from
    (``fed``: one row per dry session, its median flying lap) and the dry
    build laps of the model cars binned in both frames."""

    root: Path
    tracks_dir: Path | None
    folds_dir: Path | None
    cache_dir: Path | None
    tables: smp.CarTables
    rows: pd.DataFrame
    index: pd.DataFrame
    air: pd.DataFrame | None
    tracks: dict[str, trk.TrackModel]
    full: dict[str, cg.Drive | None]
    shifts: dict[str, dict[str, float]]
    gears: dict[str, dict[str, tuple[float, ...]]]
    use_laps: dict[str, dict[str, cg.UseLaps]]
    fed: pd.DataFrame
    sensor: dict[tuple[str, int], tuple[Any, str, bool]]
    tyres: dict[tuple[str, int], Any]


def fed_laps(index: pd.DataFrame) -> pd.DataFrame:
    """Per dry session of the lap index, its median flying lap, as the
    pressure model takes a cell's (``cleaning``'s lap filter without the
    TPMS: full laps of at least the shortest lap time and within the
    slowest-lap limit of their stint's best), where three such laps
    stand: ``session_id, car, date, layout_id, lap_time_s, laps``."""
    from ..cleaning import FilterConfig

    cfg = FilterConfig()
    own = index[(index["condition"] == "dry") & (index["lap_type"] == "full")]
    own = own[own["lap_time_s"] >= cfg.min_lap_time_s]
    best = own.groupby(["session_id", "stint_id"])["lap_time_s"].transform("min")
    own = own[own["lap_time_s"] <= cfg.max_lap_time_vs_best_pct * best]
    got = (
        own.groupby(["session_id", "car", "date", "layout_id"], sort=True)["lap_time_s"]
        .agg(lap_time_s="median", laps="size")
        .reset_index()
    )
    return got[got["laps"] >= pr.MIN_ORDINARY_LAPS].reset_index(drop=True)


def load_sim(
    root: Path,
    *,
    cache_dir: Path | None,
    tracks_dir: Path | None,
    folds_dir: Path | None = None,
) -> Sim:
    """:class:`Sim`, as gate C3 sets its folds up. ``folds_dir`` is where
    the tracks from other cars are kept (gate B3's, and those built here
    without a scored date); ``tracks_dir / "folds"`` when it is not given."""
    if folds_dir is None and tracks_dir is not None:
        folds_dir = Path(tracks_dir) / "folds"
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    tables = cm.as_families(smp.car_tables(root, cache_dir=cache_dir, tracks_dir=tracks_dir))
    rows = cm.as_family(smp.build_laps(index))
    air = cm.air_state(root, geo.load_sessions(root))
    cars = [c for c in smp.MODEL_CARS if (tables.sessions["car"] == c).any()]
    layouts = sorted(str(k) for k in rows[rows["car"].isin(cars)]["layout_id"].unique())
    tracks = {
        k: smp.load_track(root, k, tracks_dir=tracks_dir, cache_dir=cache_dir) for k in layouts
    }
    lap_time = rows.set_index(["session_id", "lap_num"])["lap_time_s"]
    full: dict[str, cg.Drive | None] = {}
    shifts: dict[str, dict[str, float]] = {}
    gears: dict[str, dict[str, tuple[float, ...]]] = {}
    use_laps: dict[str, dict[str, cg.UseLaps]] = {}
    for c in cars:
        best = cg.day_best_laps(rows, c)
        own = ch.car_only(tables, c)
        model, work = cm.fit_car(own, c, air=air)
        shifts[c], gears[c] = cg.day_shifts(work), cg.day_gears(work)
        own_tracks = {k: v for k, v in tracks.items() if k in set(best["layout_id"])}
        use_laps[c] = cg.car_use_laps(own, lap_time, c, own_tracks)
        full[c] = ch.convene(
            model, tracks, best, air=air, shifts=shifts[c], gears=gears[c], laps=use_laps[c]
        )
    dry = rows[rows["car"].isin(cars) & (rows["basis"] == "dry")]
    keys = dry[["session_id", "lap_num", "car"]].drop_duplicates(["session_id", "lap_num"])
    return Sim(
        root=Path(root),
        tracks_dir=tracks_dir,
        folds_dir=folds_dir,
        cache_dir=cache_dir,
        tables=tables,
        rows=rows,
        index=index,
        air=air,
        tracks=tracks,
        full=full,
        shifts=shifts,
        gears=gears,
        use_laps=use_laps,
        fed=cm.as_family(fed_laps(index)),
        sensor=hg.sensor_lap_bins(root, keys),
        tyres=hg.tyres_frame_lap_bins(root, keys, tracks_dir=tracks_dir, cache_dir=cache_dir),
    )


def donor_laps(sim: Sim, raw_car: str, layout_id: str) -> pd.DataFrame:
    """The dry build laps of ``layout_id`` from cars of another class
    than ``raw_car``'s: what a track that has not seen the car is built
    from (``session_id, car, date``)."""
    held = ho.CLASS_OF.get(raw_car, raw_car)
    own = sim.index[sim.index["in_build"] & (sim.index["layout_id"] == layout_id)]
    other = own[own["car"].map(lambda c: ho.CLASS_OF.get(c, c)) != held]
    return other[["session_id", "car", "date"]].assign(date=other["date"].astype(str))


def donor_track(
    sim: Sim, raw_car: str, layout_id: str, without_date: str = ""
) -> trk.TrackModel | None:
    """The track of ``layout_id`` from other classes' laps, none of them
    of ``without_date`` (#208, finding 9c: a track holds no lap of the day
    it is scored on), its curvature laid on the own track's nodes
    (:func:`~.heat_match.laid_track`). None where the donors do not have
    gate E's three days and thirty laps behind them: no track from another
    class then exists at the layout, whatever the date. Gate B3's saved
    track is taken where it is of exactly the donor laps asked for; any
    other is built now and kept beside it."""
    laps = donor_laps(sim, raw_car, layout_id)
    if laps["date"].nunique() < ho.VOTE_DAYS or len(laps) < ho.VOTE_DONOR_LAPS:
        return None
    left = laps[laps["date"] != without_date] if without_date else laps
    if left.empty:
        return None
    stem = hm.fold_track_name(raw_car, layout_id)
    folds = sim.folds_dir
    whole = len(left) == len(laps)
    path = None
    if folds is not None:
        path = folds / (f"{stem}.json" if whole else f"{stem}__not_{without_date}.json")
    other = None
    if path is not None and path.exists():
        other = trk.TrackModel.load(path)
        if int((other.meta.get("built_from") or {}).get("n_laps") or 0) != len(left):
            other = None  # not of these laps (the dataset has moved on): built again
    if other is None:
        gone = set(laps["session_id"]) - set(left["session_id"])
        other = trk.build_track_model(
            sim.root,
            layout_id,
            cars=sorted(laps["car"].unique()),
            exclude_session_ids=gone,
            cache_dir=sim.cache_dir,
            inertial=False,
            n_boot=0,
        ).model
        if path is not None and not whole:
            other.save(path)
    laid = hm.laid_track(sim.tracks[layout_id], other)
    laid.meta = {
        **laid.meta,
        "donor": {
            "cars": sorted(left["car"].unique()),
            "laps": int(len(left)),
            "days": int(left["date"].nunique()),
            "without_date": without_date,
        },
    }
    return laid


@dataclass
class SimCell:
    """The simulation's side of a fold, for one car at the held-out
    layout: the car fitted without the layout and the two planted cars
    (``drives``), each one's fed use per raw car read at the car's other
    layouts (``fed``), the frame factor carried from them under the
    fold's shares (``factors`` by kind of source, ``factor_laps`` behind
    each) and the tracks. ``neither`` says the track too has not seen the
    car's class."""

    family: str
    layout_id: str
    share: ShareParams
    drives: dict[str, cg.Drive | None]
    fed: dict[str, dict[str, dict]]
    factors: dict[str, np.ndarray]
    factor_laps: dict[str, int]
    neither: bool
    donors: dict
    _sim: Sim
    _raw_car: str
    _tracks: dict[str, trk.TrackModel] = field(default_factory=dict)
    _sources: dict[tuple, pr.SimSource | None] = field(default_factory=dict)
    #: The one track every day of the cell is driven on, where it is given
    #: (:func:`seen_cell`: the layout's track built without a fold's dates).
    _fixed: trk.TrackModel | None = None

    def track(self, date: str) -> trk.TrackModel:
        """The track a day's lap is driven on: from other classes' laps
        without that date where such a track exists, else the own."""
        if self._fixed is not None:
            return self._fixed
        if not self.neither:
            return self._sim.tracks[self.layout_id]
        has = date in set(donor_laps(self._sim, self._raw_car, self.layout_id)["date"])
        key = date if has else ""
        if key not in self._tracks:
            got = donor_track(self._sim, self._raw_car, self.layout_id, key)
            assert got is not None
            self._tracks[key] = got
            built = got.meta.get("donor") or {}
            self.donors["fewest_laps"] = min(
                int(self.donors.get("fewest_laps", self.donors["laps"])),
                int(built.get("laps", self.donors["laps"])),
            )
        return self._tracks[key]

    def source(self, arm: str, powertrain: str, date: str) -> pr.SimSource | None:
        """The table of ``arm``'s car with the day's powertrain on the
        day's track, exported and split through the fold's shares; None
        where the car has no such powertrain away from the layout."""
        drive = self.drives.get(arm)
        if drive is None:
            return None
        track = self.track(date)
        engines = drive.model.power.get("engines") or {}
        engine = None if arm == "other_car" or not engines else powertrain
        if engine is not None and engine not in engines:
            return None  # a powertrain the car has only run at the held-out layout
        key = (arm, engine, id(track))
        if key not in self._sources:
            try:
                table = pace.build_table(drive.model, track, powertrain=engine, **SIM_ARMS[arm])
            except KeyError:
                self._sources[key] = None
            else:
                self._sources[key] = pr.SimSource.of(table.laps_export(), self.share)
        return self._sources[key]

    def direct_entry(
        self,
        raw_car: str,
        date: str,
        session_id: str,
        lap_time_s: float,
        factor: np.ndarray,
        *,
        setup: bool = False,
    ) -> pr.Entry | None:
        """The entry of a lap driven for the asking, with no table (the
        shootout's direct simulation): the fitted car with the day's
        powertrain in the session's air, the knob solved for
        ``lap_time_s``, the out-lap driven at that knob from the pit
        exit's join; its heat through the chain and the cell's shares, in
        production's frame (``factor``). The shift point and the gears are
        the powertrain's own, or with ``setup`` the day's logged ones.
        None where the car has no lap of that lap time."""
        from . import solver as sv
        from .heat import bin_1s, lap_heat, sim_series

        drive = self.drives.get("sim")
        if drive is None or not np.isfinite(factor).all():
            return None
        asked = drive.but(use_cap=True) if setup else drive.but(use_cap=True, shifts={}, gears={})
        car = asked.car(raw_car, date, session_id)
        track = self.track(date)
        profile = cg.profile_of(asked, track)
        got = sv.solve_laps(profile, [car], [float(lap_time_s)], guess=cg.typical_use(drive))
        if not bool(got.ok[0]):
            return None
        knob = float(got.u[0])

        def _heat(lap: sv.SimLap) -> np.ndarray:
            return np.asarray(lap_heat(bin_1s(*sim_series(lap), 0.0, lap.lap_time_s), self.share))

        q = _heat(sv.lap_from_nodes(profile, car, got.v_node_ms[0], knob)) * factor
        out_s = out_q = None
        pit = dict(track.meta.get("pit") or {})
        v_out = pace.out_laps(profile, car, np.array([knob]), pit)
        if v_out is not None:
            part = pace._out_profile(profile, pace._join_node(profile, pit), whole=False)
            out_lap = sv.lap_from_nodes(part, car, v_out[0], knob)
            out_s = float(pit.get("rolling_s") or 0.0) + out_lap.lap_time_s
            out_q = _heat(out_lap) * factor * out_lap.lap_time_s / out_s
        return pr.Entry(
            lap_time_s=float(lap_time_s), q=q, out_s=out_s, out_q=out_q, source="direct"
        )

    def use(self, arm: str, raw_car: str) -> float:
        """The fed use of ``raw_car`` as ``arm``'s car reads it (NaN
        where the raw car has no dry session at another layout)."""
        return float((self.fed.get(arm) or {}).get(raw_car, {}).get("use", float("nan")))


def sim_cell(sim: Sim, raw_car: str, layout_id: str, share: ShareParams) -> SimCell | None:
    """The simulation's side of a fold for ``raw_car`` at ``layout_id``
    (:class:`SimCell`): the car model fitted without the layout, as a
    layout fold of gate C3 fits it; None for a car the lap simulation has
    no model of or no track for."""
    family = cm.family_of(raw_car)
    if family not in sim.full or layout_id not in sim.tracks:
        return None
    own = ch.car_only(sim.tables, family)
    best = cg.day_best_laps(sim.rows, family)
    best = best[best["layout_id"] != layout_id]
    others = {k: v for k, v in sim.tracks.items() if k != layout_id}
    raw = cm.fit_car(ch.without(own, layout_id), family, air=sim.air)[0]
    kw: dict[str, Any] = {
        "air": sim.air,
        "shifts": sim.shifts[family],
        "gears": sim.gears[family],
    }
    unseen = ch.convene(
        raw, sim.tracks, best, laps=ch.laps_left(sim.use_laps[family], layout_id), **kw
    )
    drives: dict[str, cg.Drive | None] = {"sim": None, "low_power": None, "other_car": None}
    if unseen is not None:
        # the table's support: the car's logged range of use at its other layouts
        ranged = cm.CarModel.from_json(unseen.model.to_json())
        ranged.meta = {
            **unseen.model.meta,
            "use_range": cg.use_range(unseen, others, sim.rows, family),
        }
        drives["sim"] = unseen.but(model=ranged)
        drives["low_power"] = ch.convene(
            raw, sim.tracks, best, power_scale=ch.POWER_CONTROL, use=unseen.model.driver, **kw
        )
    stand_in = sim.full.get(ch.OTHER_CAR.get(family, ""))
    drives["other_car"] = stand_in.but(shifts={}, gears={}) if stand_in is not None else None
    fed_rows = sim.fed[(sim.fed["car"] == family) & (sim.fed["layout_id"] != layout_id)]
    fed = {
        arm: cg.fed_use(drive, others, fed_rows)
        for arm, drive in drives.items()
        if drive is not None
    }
    build = sim.rows[
        (sim.rows["car"] == family)
        & (sim.rows["basis"] == "dry")
        & (sim.rows["layout_id"] != layout_id)
    ]
    heats = pr.heats_frame(
        build[["session_id", "lap_num", "layout_id"]], sim.sensor, sim.tyres, share
    )
    factors = unit_factors(heats) if heats_in_tyres_frame() else pr.frame_factors(heats)
    usable = heats[~heats["hole"].astype(bool)] if len(heats) else heats
    donors = donor_laps(sim, raw_car, layout_id)
    neither = donor_track(sim, raw_car, layout_id) is not None
    return SimCell(
        family=family,
        layout_id=layout_id,
        share=share,
        drives=drives,
        fed=fed,
        factors=factors,
        factor_laps={
            str(k): int(v) for k, v in usable.groupby("source").size().items() if k in factors
        },
        neither=neither,
        donors={
            "cars": sorted(donors["car"].unique()),
            "laps": int(len(donors)),
            "days": int(donors["date"].nunique()),
        },
        _sim=sim,
        _raw_car=raw_car,
    )


def track_without(sim: Sim, layout_id: str, held: set[str]) -> trk.TrackModel:
    """The layout's track with its curvature from the build laps of every
    car but the sessions ``held`` (laid on the own track's nodes; the
    bank and the grade stay the road's)."""
    own = sim.index[sim.index["in_build"] & (sim.index["layout_id"] == layout_id)]
    gone = set(own.loc[own["session_id"].astype(str).isin(held), "session_id"])
    if not gone:
        return sim.tracks[layout_id]
    other = trk.build_track_model(
        sim.root,
        layout_id,
        exclude_session_ids=gone,
        cache_dir=sim.cache_dir,
        inertial=False,
        n_boot=0,
    ).model
    return hm.laid_track(sim.tracks[layout_id], other)


def without_sessions(tables: smp.CarTables, held: set[str]) -> smp.CarTables:
    """``tables`` without every sample of the sessions ``held``."""

    def _keep(tbl: pd.DataFrame) -> pd.DataFrame:
        if tbl.empty or "session_id" not in tbl.columns:
            return tbl
        return tbl[~tbl["session_id"].astype(str).isin(held)]

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


def held_out_drive(
    sim: Sim, family: str, held: set[str]
) -> tuple[cg.Drive | None, dict, pd.DataFrame]:
    """A car with the sessions ``held`` out of everything it is made of,
    at every layout: ``(drive, fed, rows)``. The car model is fitted from
    the sample tables without them, its limit, its driver's braking use
    and its typical use read on the laps that are left (as a fold of gate
    C3, with sessions in place of a day); ``fed`` is the fed use per raw
    car from the sessions that are left (:func:`~.car_gate.fed_use`) and
    ``rows`` the build laps that are left."""
    tables = without_sessions(ch.car_only(sim.tables, family), held)
    rows = sim.rows[~sim.rows["session_id"].astype(str).isin(held)]
    best = cg.day_best_laps(rows, family)
    use_laps = {
        k: own.keep((~own.frame["session_id"].astype(str).isin(held)).to_numpy())
        for k, own in sim.use_laps[family].items()
    }
    raw = cm.fit_car(tables, family, air=sim.air)[0]
    drive = ch.convene(
        raw,
        sim.tracks,
        best,
        air=sim.air,
        shifts=sim.shifts[family],
        gears=sim.gears[family],
        laps=use_laps,
    )
    if drive is None:
        return None, {}, rows
    ranged = cm.CarModel.from_json(drive.model.to_json())
    ranged.meta = {**drive.model.meta, "use_range": cg.use_range(drive, sim.tracks, rows, family)}
    drive = drive.but(model=ranged)
    left = sim.fed[(sim.fed["car"] == family) & ~sim.fed["session_id"].astype(str).isin(held)]
    return drive, cg.fed_use(drive, sim.tracks, left), rows


def seen_cell(
    sim: Sim,
    raw_car: str,
    layout_id: str,
    share: ShareParams,
    held: set[str],
    track: trk.TrackModel | None = None,
    memo: dict | None = None,
) -> SimCell | None:
    """The simulation's side of a cell the car *has* run, for a fold that
    holds the sessions ``held`` out: the simulation as the heat source
    where the calculator has logged laps. **The holdout applies to every
    model under the entry**: the car model, its limit, the driver's
    braking use and the typical use (:func:`held_out_drive`); the track's
    curvature (``track``: :func:`track_without`); the frame factor (the
    car's build laps that are left at the layout, per kind of source; its
    other layouts' where the layout has too few of that kind); and the
    fed use (the raw car's own sessions that are left at the layout; the
    median over its layouts where it has none). ``memo`` keeps a fold's
    car per family between cells."""
    family = cm.family_of(raw_car)
    if family not in sim.full or layout_id not in sim.tracks:
        return None
    memo = {} if memo is None else memo
    if family not in memo:
        memo[family] = held_out_drive(sim, family, held)
    drive, carried, rows = memo[family]
    if drive is None:
        return None
    fed = {
        car: {**own, "use": own["layouts"].get(layout_id, own)["use"]}  # the layout's own first
        for car, own in carried.items()
    }
    build = rows[(rows["car"] == family) & (rows["basis"] == "dry")]
    heats = pr.heats_frame(
        build[["session_id", "lap_num", "layout_id"]], sim.sensor, sim.tyres, share
    )
    if heats_in_tyres_frame():
        factors = unit_factors(heats)
    else:
        factors = pr.frame_factors(heats[heats["layout_id"] != layout_id]) if len(heats) else {}
        factors.update(
            pr.frame_factors(heats[heats["layout_id"] == layout_id]) if len(heats) else {}
        )
    return SimCell(
        family=family,
        layout_id=layout_id,
        share=share,
        drives={"sim": drive, "low_power": None, "other_car": None},
        fed={"sim": fed},
        factors=factors,
        factor_laps={},
        neither=False,
        donors={"cars": [], "laps": 0, "days": 0},
        _sim=sim,
        _raw_car=raw_car,
        _fixed=track if track is not None else track_without(sim, layout_id, held),
    )


# ---------------------------------------------------------------- the arms


def _bare(entry: pr.Entry) -> pr.Entry:
    """An entry with no out-lap of its own: the fold model's lookup then
    answers, the same on every side of a comparison at a given lap time."""
    return replace(entry, out_s=None, out_q=None)


def _recommend(
    held: Held,
    sid: str,
    entry: pr.Entry | None,
    lap_time_s: float | None,
    condition: str = "dry",
) -> pd.DataFrame:
    """The recommendations for a session's stints from ``entry`` placed
    in the fold's model as the cell's entry for ``condition`` (None: the
    fold model as it stands), at each flying lap of
    :data:`~.pressure.FLYING_LAPS`."""
    model = held.model
    if entry is not None:
        model = pr.place(held.model, held.track, held.pooled, condition, entry)
    stints = held.stints[held.stints["session_id"] == sid]
    parts = [
        pr.recommend(model, stints, lap_time_s=lap_time_s, n=n).assign(n=n) for n in pr.FLYING_LAPS
    ]
    return pd.concat(parts, ignore_index=True)


def _pairs(held: Held, day: dict) -> pd.DataFrame:
    """The sessions a day can be paired with: other dates of the same raw
    car, powertrain and kind of source at the track."""
    d = held.days
    same = (
        (d["raw_car"] == day["raw_car"])
        & (d["powertrain"] == day["powertrain"])
        & (d["source"] == day["source"])
        & (d["date"] != day["date"])
    )
    return pd.DataFrame(d[same])


def other_day_arms(held: Held, day: dict) -> list[dict]:
    """The yardstick's arms for one day (``other_day``): at the day's lap
    time, up to three sessions of other days nearest in lap time, each
    read from its laps nearest the day's lap time; given nothing, each
    other day as if it alone had written the entry."""
    t_day = float(day["lap_time_s"])
    laps = held.laps[held.laps["condition"] == "dry"]
    pairs = _pairs(held, day)
    out: list[dict] = []
    near = pairs.assign(off=(pairs["lap_time_s"] - t_day).abs()).sort_values(["off", "session_id"])
    for k, other in enumerate(near.head(cg.FLOOR_PAIRS_PER_LAP).to_dict("records")):
        got = pr.near_heat(laps[laps["session_id"] == other["session_id"]], t_day)
        if got is None:
            continue
        out.append(
            {
                "arm": "other_day",
                "form": "heat",
                "entry": pr.Entry(lap_time_s=t_day, q=got[0], source="other_day"),
                "lap_time_s": t_day,
                "pick": k,
                "gap": float(np.log(got[1] / t_day)),
                "paired": other["session_id"],
            }
        )
    for k, (date, group) in enumerate(pairs.groupby("date", sort=True)):
        alone = pr.logged_entry(
            held.root,
            laps[laps["session_id"].isin(set(group["session_id"]))],
            held.share,
            curve=False,
        )
        if alone is None:
            continue
        out.append(
            {
                "arm": "other_day",
                "form": "chain",
                "entry": alone,
                "lap_time_s": None,
                "pick": k,
                "gap": float(np.log(alone.lap_time_s / t_day)),
                "paired": str(date),
            }
        )
    return out


def day_arms(held: Held, day: dict, cell: SimCell | None, *, seen: bool = False) -> list[dict]:
    """Every arm's entry for one day, in both forms: dicts of ``arm,
    form, entry`` (None: the fold model as it stands), ``lap_time_s``
    (the lap time typed in; None: the entry's own), ``pick`` and, where
    they apply, ``gap`` (the log of a paired day's lap time over the
    day's), ``status`` and ``use``. The form ``tpms`` marks the entries
    pressure against the TPMS is scored with (S2: production's own
    inputs, the entry as it would be written).

    ``seen`` is for a model that has the day's cell (a fold that leaves
    the day's date out, or the model as shipped): the arms are then
    ``entry``, the model as it stands, which is the calculator as it is,
    and the yardstick."""
    root = held.root
    sid, t_day = day["session_id"], float(day["lap_time_s"])
    own = held.entries[sid]
    out: list[dict] = [
        {"arm": "day", "form": "heat", "entry": _bare(own), "lap_time_s": t_day},
        {"arm": "day", "form": "chain", "entry": own, "lap_time_s": t_day},
    ]
    if seen:
        there = pr.entry_of(held.model, held.track, held.pooled, "dry")
        if there is not None:
            gap = float(np.log(there.lap_time_s / t_day))
            out.append({"arm": "entry", "form": "heat", "entry": _bare(there), "lap_time_s": t_day})
            out.append(
                {"arm": "entry", "form": "chain", "entry": None, "lap_time_s": None, "gap": gap}
            )
            out.append({"arm": "entry", "form": "tpms", "entry": None})
        out += other_day_arms(held, day)
        return out + (sim_arms(cell, day) if cell is not None else [])
    typed = float(held.prep.session_lap_time.get(sid, t_day))
    for arm, factor in (("heat_up", 1.0 + HEAT_PLANT), ("heat_down", 1.0 - HEAT_PLANT)):
        out.append(
            {"arm": arm, "form": "heat", "entry": _bare(own.scaled(factor)), "lap_time_s": t_day}
        )
        out.append({"arm": arm, "form": "chain", "entry": own.scaled(factor), "lap_time_s": t_day})
        out.append({"arm": arm, "form": "tpms", "entry": own.scaled(factor)})
    out.append({"arm": "own_heat", "form": "tpms", "entry": own})
    laps = held.laps[held.laps["condition"] == "dry"]
    out += other_day_arms(held, day)
    # the ceiling: the raw car's logged entry at the cell, built without the day
    rest = laps[(laps["raw_car"] == day["raw_car"]) & (laps["date"] != day["date"])]
    ceiling = pr.logged_entry(root, rest, held.share) if len(rest) else None
    # the same from every raw car pooled under the model's name: what the calculators read
    pooled_rest = laps[laps["date"] != day["date"]]
    pooled = None
    if pooled_rest["raw_car"].nunique() > 1:
        pooled = pr.logged_entry(root, pooled_rest, held.share)
    for arm, entry in (("ceiling", ceiling), ("ceiling_pooled", pooled)):
        if entry is None:
            continue
        out.append({"arm": arm, "form": "heat", "entry": _bare(entry), "lap_time_s": t_day})
        out.append(
            {
                "arm": arm,
                "form": "chain",
                "entry": entry,
                "lap_time_s": None,
                "gap": float(np.log(entry.lap_time_s / t_day)),
            }
        )
        if arm == "ceiling":
            out.append({"arm": arm, "form": "tpms", "entry": entry})
    # what the simulation replaces: the car's other layouts, and the fallback as coded
    others = pr.other_layouts_entry(held.model, held.track, held.pooled)
    if others is not None:
        out.append(
            {
                "arm": "other_layouts",
                "form": "heat",
                "entry": _bare(others.at_lap_time(t_day)),
                "lap_time_s": t_day,
            }
        )
        if np.isfinite(others.lap_time_s):
            out.append(
                {
                    "arm": "other_layouts",
                    "form": "chain",
                    "entry": others,
                    "lap_time_s": None,
                    "gap": float(np.log(others.lap_time_s / t_day)),
                }
            )
        # against the TPMS the heat is unscaled at the lap time that is typed in
        out.append({"arm": "other_layouts", "form": "tpms", "entry": others.at_lap_time(typed)})
    out.append({"arm": "fallback", "form": "heat", "entry": None, "lap_time_s": t_day})
    out.append({"arm": "fallback", "form": "chain", "entry": None, "lap_time_s": None})
    out.append({"arm": "fallback", "form": "tpms", "entry": None})
    if cell is not None:
        out += sim_arms(cell, day)
    return out


def sim_arms(cell: SimCell, day: dict) -> list[dict]:
    """The simulated arms' entries for one day (see :func:`day_arms`):
    the simulation and the two planted cars in both forms, the
    simulation's lap time moved either way in ``chain``, and in ``heat``
    the simulation's table read two other ways (``sim_direct``: log-log
    at the day's lap time, no entry curve; ``sim_9``: an entry of nine
    curve nodes, as the calculators' logged curves have)."""
    t_day = float(day["lap_time_s"])
    factor = cell.factors.get(str(day["source"]))
    out: list[dict] = []
    if factor is None:
        return out
    for arm in SIM_ARMS:
        src = cell.source(arm, str(day["powertrain"]), str(day["date"]))
        use = cell.use(arm, str(day["raw_car"]))
        if src is None or not np.isfinite(use):
            continue
        entry = src.entry(factor, use=use, source=arm)
        if entry is None:
            # the car has no lap at its fed use on this track: a miss, kept
            out.append({"arm": arm, "form": "chain", "entry": None, "status": pr.NO_SOLUTION})
            out.append({"arm": arm, "form": "heat", "entry": None, "status": pr.NO_SOLUTION})
            continue
        gap = float(np.log(entry.lap_time_s / t_day))
        out.append(
            {
                "arm": arm,
                "form": "heat",
                "entry": _bare(entry),
                "lap_time_s": t_day,
                "status": src.status(t_day),
                "use": use,
            }
        )
        out.append(
            {
                "arm": arm,
                "form": "chain",
                "entry": entry,
                "lap_time_s": None,
                "status": entry.status,
                "gap": gap,
                "use": use,
            }
        )
        if arm != "sim":
            continue
        out.append({"arm": arm, "form": "tpms", "entry": entry})
        for name, moved in (("lap_up", 1.0 + LAP_TIME_PLANT), ("lap_down", 1.0 - LAP_TIME_PLANT)):
            other = src.entry(factor, lap_time_s=entry.lap_time_s * moved, source=name)
            if other is not None:
                out.append(
                    {
                        "arm": name,
                        "form": "chain",
                        "entry": other,
                        "lap_time_s": None,
                        "status": other.status,
                        "gap": float(np.log(other.lap_time_s / t_day)),
                    }
                )
        direct = src.direct(factor, t_day)
        if direct is not None:
            out.append(
                {
                    "arm": "sim_direct",
                    "form": "heat",
                    "entry": _bare(direct),
                    "lap_time_s": t_day,
                    "status": src.status(t_day),
                }
            )
        nine = src.entry(factor, use=use, nodes=9, source="sim_9")
        if nine is not None:
            out.append(
                {
                    "arm": "sim_9",
                    "form": "heat",
                    "entry": _bare(nine),
                    "lap_time_s": t_day,
                    "status": src.status(t_day),
                }
            )
    return out


def latest_powertrain(raw_car: str) -> str:
    """The powertrain a raw car has today: what its entry is of."""
    return cm.powertrain_of(raw_car, "9999-12-31")


def pool_weights(prod: Production, pooled: str, left_out: list[str]) -> dict[str, float]:
    """Each raw car's usable laps among the pooled car's sessions that
    reach a fold's fit: what its share of a pooled entry is weighed by."""
    own = prod.sessions[
        (prod.sessions["pooled"] == pooled)
        & prod.sessions["session_id"].astype(str).isin(prod.trained - set(left_out))
    ]
    laps = prod.raw_laps[prod.raw_laps["tire_usable"].astype(bool)]
    count = laps.groupby("session_id").size()
    got = own.assign(laps=own["session_id"].map(count).fillna(0)).groupby("car")["laps"].sum()
    return {str(k): float(v) for k, v in got.items() if v > 0}


def pooled_sim_arms(
    cells: dict[str, SimCell | None], weights: dict[str, float], day: dict
) -> list[dict]:
    """The pooled simulated entry for one day, where the model pools more
    than one raw car (``sim_pooled``, both forms): each raw car's table of
    its latest powertrain read at one reference lap time (the weighted
    geometric mean of the lap times of their fed uses) on shared nodes,
    and pooled (:func:`~.pressure.pooled_entry`). A raw car with no fed
    use, no table or no frame factor for the day's kind of source has no
    part in it. Nothing where the pooled entry would be the day's own raw
    car's alone (then it is the ``sim`` arm)."""
    t_day = float(day["lap_time_s"])
    parts = []
    for raw_car, weight in sorted(weights.items()):
        cell = cells.get(raw_car)
        if cell is None:
            continue
        factor = cell.factors.get(str(day["source"]))
        src = cell.source("sim", latest_powertrain(raw_car), str(day["date"]))
        use = cell.use("sim", raw_car)
        if factor is None or src is None or not np.isfinite(use):
            continue
        own = src.export.lap_time_at(use)
        if np.isfinite(own):
            parts.append((src, factor, float(own), weight, raw_car))
    if not parts or {p[4] for p in parts} == {str(day["raw_car"])}:
        return []
    w = np.array([p[3] for p in parts])
    ref = float(np.exp(np.sum(w * np.log([p[2] for p in parts])) / w.sum()))
    entries = [src.entry(factor, lap_time_s=ref, nodes=POOL_GRID) for src, factor, *_ in parts]
    pooled = pr.pooled_entry([e for e in entries if e is not None], list(w))
    if pooled is None:
        return []
    return [
        {"arm": "sim_pooled", "form": "heat", "entry": _bare(pooled), "lap_time_s": t_day},
        {
            "arm": "sim_pooled",
            "form": "chain",
            "entry": pooled,
            "lap_time_s": None,
            "status": pooled.status,
            "gap": float(np.log(pooled.lap_time_s / t_day)),
        },
    ]


def wet_arms(held: Held, day: dict, cell: SimCell | None) -> list[dict]:
    """The arms of a wet or damp day, at the day's lap time (``heat``;
    the lap time of a wet entry is typed in, #261): the day's own;
    ``other_day``, which in the wet is another session **of the same day**
    (D2's form: wet days do not repeat), same raw car, powertrain, kind of
    source and condition, up to three nearest in lap time; the simulation
    (the dry table read at the wet lap time, no wet term: gate D2); the
    planted heat; the fold model as it stands (``fallback``)."""
    sid, t_day = day["session_id"], float(day["lap_time_s"])
    own = held.entries[sid]
    out: list[dict] = [{"arm": "day", "form": "heat", "entry": _bare(own), "lap_time_s": t_day}]
    for arm, scale in (("heat_up", 1.0 + HEAT_PLANT), ("heat_down", 1.0 - HEAT_PLANT)):
        out.append(
            {"arm": arm, "form": "heat", "entry": _bare(own.scaled(scale)), "lap_time_s": t_day}
        )
    d = held.days
    same = d[
        (d["raw_car"] == day["raw_car"])
        & (d["powertrain"] == day["powertrain"])
        & (d["source"] == day["source"])
        & (d["condition"] == day["condition"])
        & (d["date"] == day["date"])
        & (d["session_id"] != sid)
    ]
    near = same.assign(off=(same["lap_time_s"] - t_day).abs()).sort_values(["off", "session_id"])
    for k, other in enumerate(near.head(cg.FLOOR_PAIRS_PER_LAP).to_dict("records")):
        got = pr.near_heat(held.laps[held.laps["session_id"] == other["session_id"]], t_day)
        if got is None:
            continue
        out.append(
            {
                "arm": "other_day",
                "form": "heat",
                "entry": pr.Entry(lap_time_s=t_day, q=got[0], source="same_day"),
                "lap_time_s": t_day,
                "pick": k,
                "gap": float(np.log(got[1] / t_day)),
            }
        )
    out.append({"arm": "fallback", "form": "heat", "entry": None, "lap_time_s": t_day})
    if cell is None:
        return out
    factor = cell.factors.get(str(day["source"]))
    src = cell.source("sim", str(day["powertrain"]), str(day["date"]))
    if factor is None or src is None:
        return out
    entry = src.entry(factor, lap_time_s=t_day)
    if entry is not None:
        out.append(
            {
                "arm": "sim",
                "form": "heat",
                "entry": _bare(entry),
                "lap_time_s": t_day,
                "status": src.status(t_day),
                "use": float(src.export.use_at(t_day)),
            }
        )
    return out


def score_cell(
    held: Held,
    cells: dict[str, SimCell | None],
    *,
    seen: bool = False,
    wet: bool = False,
    weights: dict[str, float] | None = None,
    sessions: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(rows, tpms)`` of a fold's cell.

    ``rows``: every arm's distance from the day's own recommendation, per
    day, pit-exit stint, tyre, flying lap and form: ``d_mbar`` (the arm's
    cold pressure less the day's) with ``pooled, track, raw_car, date,
    session_id, stint_id, corner, n, form, arm, pick, gap, status, use``.
    A simulated arm with no lap keeps a row with a NaN distance and the
    status ``no_solution``.

    ``tpms``: pressure against the TPMS (S2), which is reported and never
    ranks: per arm with a ``tpms`` entry, the holdout's own rows
    (production's inputs: the session's 25th-percentile lap time typed
    in), every scored lap and tyre, with ``resid_mbar``, ``flying_lap``,
    ``pit_exit`` and whether the row is ``after_dropped`` or
    ``after_cool_down`` (:func:`~.pressure.mark_laps`). :func:`at_lap`
    takes the rows S2 is read on.

    ``cells`` is the simulation's side per raw car (None: no simulated
    arm); ``seen`` as in :func:`day_arms`; ``wet`` scores the days with
    :func:`wet_arms` (the held days are then wet or damp ones);
    ``weights`` (:func:`pool_weights`) adds the pooled simulated entry;
    with ``sessions`` only those days are scored (the others are there to
    be paired with)."""
    from ..validate import _score_fold_laps

    rows: list[pd.DataFrame] = []
    tpms: list[pd.DataFrame] = []
    flags = held.laps[["session_id", "lap_num", "after_dropped", "after_cool_down"]]
    for day in held.days.to_dict("records"):
        sid = day["session_id"]
        if sessions is not None and sid not in sessions:
            continue
        cond = str(day.get("condition") or "dry")
        if wet:
            arms = wet_arms(held, day, cells.get(day["raw_car"]))
        else:
            arms = day_arms(held, day, cells.get(day["raw_car"]), seen=seen)
            if weights and not seen:
                arms += pooled_sim_arms(cells, weights, day)
        ref = {
            a["form"]: _recommend(held, sid, a["entry"], a["lap_time_s"], cond)
            for a in arms
            if a["arm"] == "day"
        }
        base_cols = {
            "pooled": held.pooled,
            "track": held.track,
            "raw_car": day["raw_car"],
            "date": day["date"],
            "session_id": sid,
        }
        for a in arms:
            if a["arm"] == "day":
                continue
            if a["form"] == "tpms":
                model = held.model
                if a["entry"] is not None:
                    model = pr.place(held.model, held.track, held.pooled, cond, a["entry"])
                got = _score_fold_laps(held.prep, model, session_ids=[sid])
                if got.empty:
                    continue
                got = got[got["resid_bar"].notna()].merge(
                    flags, on=["session_id", "lap_num"], how="left"
                )
                tpms.append(
                    pd.DataFrame(
                        {
                            **base_cols,
                            "stint_id": got["stint_id"],
                            "lap_num": got["lap_num"],
                            "corner": got["corner"],
                            "arm": a["arm"],
                            "flying_lap": got["n_flying"],
                            "pit_exit": got["anchor_kind"] == "pit_exit",
                            "resid_mbar": 1000.0 * got["resid_bar"],
                            "after_dropped": got["after_dropped"].fillna(False).astype(bool),
                            "after_cool_down": got["after_cool_down"].fillna(False).astype(bool),
                        }
                    )
                )
                continue
            base = ref[a["form"]]
            if a["entry"] is None and a.get("status") == pr.NO_SOLUTION:
                got = base.assign(cold_bar=np.nan)
            else:
                got = _recommend(held, sid, a["entry"], a["lap_time_s"], cond)
            both = base.merge(
                got, on=["session_id", "stint_id", "corner", "n"], suffixes=("_day", "")
            )
            if both.empty:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        **base_cols,
                        "stint_id": both["stint_id"],
                        "corner": both["corner"],
                        "n": both["n"],
                        "form": a["form"],
                        "arm": a["arm"],
                        "pick": int(a.get("pick", 0)),
                        "d_mbar": 1000.0 * (both["cold_bar"] - both["cold_bar_day"]),
                        "gap": float(a.get("gap", 0.0)),
                        "status": str(a.get("status", pr.OK)),
                        "use": float(a.get("use", np.nan)),
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.concat(tpms, ignore_index=True) if tpms else pd.DataFrame(),
    )


def at_lap(tpms: pd.DataFrame, n: int = pr.FLYING_LAP_N) -> pd.DataFrame:
    """The rows of :func:`score_cell`'s ``tpms`` that S2 is read on: the
    end of flying lap ``n`` of a pit-exit stint."""
    if tpms.empty or "flying_lap" not in tpms.columns:
        return tpms
    return tpms[(tpms["flying_lap"] == n) & tpms["pit_exit"].astype(bool)]


def tpms_table(tpms: pd.DataFrame) -> pd.DataFrame:
    """Pressure against the TPMS per raw car, track and arm at flying lap
    5 of pit-exit stints (:func:`at_lap`) on the clean rows (not after a
    dropped lap, not at or after a cool-down lap): ``rows``, ``mae_mbar``,
    ``bias_mbar``, with the rows set aside."""
    tpms = at_lap(tpms)
    if tpms.empty:
        return tpms
    aside = tpms["after_dropped"] | tpms["after_cool_down"]
    keys = ["pooled", "track", "raw_car", "arm"]
    clean = tpms[~aside].groupby(keys, sort=True)["resid_mbar"]
    out = pd.DataFrame(
        {
            "rows": clean.size(),
            "mae_mbar": clean.apply(lambda x: float(x.abs().mean())),
            "bias_mbar": clean.mean(),
        }
    )
    out["set_aside"] = tpms[aside].groupby(keys)["resid_mbar"].size()
    out["set_aside"] = out["set_aside"].fillna(0).astype(int)
    return out.reset_index()


# -------------------------------------------------------------- the scores

_STINT = ["pooled", "track", "raw_car", "date", "session_id", "stint_id", "corner", "n", "form"]
_CELL = ["pooled", "track", "raw_car", "corner", "n", "form"]


def by_stint(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per stint, tyre, flying lap, form and arm: ``size`` (the
    median over the arm's picks of the size of its distance from the
    day's recommendation, mbar), ``signed`` (the median of the distance),
    ``gap`` (the median of the size of the picks' lap-time gaps),
    ``picks`` and ``status``. A pick with no lap (``no_solution``) has an
    infinite size: a miss, never dropped."""
    if rows.empty:
        return rows
    df = rows.assign(
        size=np.where(rows["status"] == pr.NO_SOLUTION, np.inf, rows["d_mbar"].abs()),
        gap_size=rows["gap"].abs(),
    )
    got = (
        df.groupby(_STINT + ["arm"], sort=False)
        .agg(
            size=("size", "median"),
            signed=("d_mbar", "median"),
            gap=("gap_size", "median"),
            gap_signed=("gap", "median"),
            picks=("pick", "nunique"),
            status=("status", "first"),
            use=("use", "first"),
        )
        .reset_index()
    )
    return got


def _cells(own: pd.DataFrame) -> pd.DataFrame:
    """Stint rows as cells: per raw car, track, tyre, flying lap, form and
    arm the median over days of the day's median stint."""
    cols = ["size", "signed", "gap", "gap_signed"]
    days = own.groupby(_CELL + ["arm", "date"], sort=False)[cols].median()
    got = days.groupby(_CELL + ["arm"], sort=False)[cols].median()
    got["days"] = days.groupby(_CELL + ["arm"], sort=False).size()
    got["stints"] = own.groupby(_CELL + ["arm"], sort=False).size()
    flagged = own.assign(flag=own["status"] != pr.OK).groupby(_CELL + ["arm"], sort=False)["flag"]
    got["flagged"] = flagged.sum()
    return got.reset_index()


def by_cell(stints: pd.DataFrame, beside: str = "other_day") -> pd.DataFrame:
    """One row per raw car, track, tyre, flying lap, form and arm: the
    median over days of the day's median stint of ``size``, ``signed`` and
    ``gap``, with ``days`` and ``stints``. Only the stints that the arm
    ``beside`` also has are counted (a day with no pair is listed, not
    scored). An arm that lacks some of those stints is on fewer rows than
    ``beside`` here: two arms are set against each other by
    :func:`attain`, on the rows both have."""
    if stints.empty:
        return stints
    keys = stints.loc[stints["arm"] == beside, _STINT].drop_duplicates()
    return _cells(stints.merge(keys, on=_STINT, how="inner"))


def attain(stints: pd.DataFrame, arm: str, beside: str = "other_day") -> pd.DataFrame:
    """``arm`` beside ``beside``, cell by cell (a raw car at a track on
    one tyre), **on the stints both have**: ``size`` and ``signed`` of the
    arm, ``beside`` the size of the other, ``scored`` (:data:`MIN_DAYS`
    days) and ``passed`` (the arm is no further from the day than the
    other is). ``stints`` is :func:`by_stint`'s rows of one form."""
    if stints.empty:
        return stints
    mine = stints[stints["arm"] == arm]
    theirs = stints[stints["arm"] == beside]
    keys = mine[_STINT].merge(theirs[_STINT], on=_STINT).drop_duplicates()
    if keys.empty:
        return pd.DataFrame()
    a = _cells(mine.merge(keys, on=_STINT)).drop(columns="arm")
    b = _cells(theirs.merge(keys, on=_STINT))[_CELL + ["size", "days", "gap"]].rename(
        columns={"size": "beside", "days": "beside_days", "gap": "beside_gap"}
    )
    both = a.merge(b, on=_CELL, how="inner")
    both["scored"] = (both["days"] >= MIN_DAYS) & (both["beside_days"] >= MIN_DAYS)
    both["passed"] = both["scored"] & (both["size"] <= both["beside"])
    return both


def tally(att: pd.DataFrame) -> dict:
    """``passed`` of ``scored`` cells, and ``whole`` of ``groups``: the
    raw car × track groups with every scored tyre inside (the four tyres
    of a cell move together), with the median ``size`` and ``beside``
    over the scored cells."""
    own = att[att["scored"]] if len(att) else att
    if own.empty:
        nan = float("nan")
        return {"passed": 0, "scored": 0, "whole": 0, "groups": 0, "size": nan, "beside": nan}
    groups = own.groupby(["raw_car", "track"])["passed"].all()
    return {
        "passed": int(own["passed"].sum()),
        "scored": int(len(own)),
        "whole": int(groups.sum()),
        "groups": int(len(groups)),
        "size": float(own["size"].median()),
        "beside": float(own["beside"].median()),
    }


# ----------------------------------------------- the checks on the harness


def identity_check(model: dict[str, Any], prep: Any) -> tuple[bool, int]:
    """An entry read from the fold's model and placed again is the rows it
    was read from, and the holdout scored on that model is the holdout,
    bit for bit: ``(same, rows)``. The raw cars' own rows are left in place
    (``place(..., raw=False)``)."""
    from ..predict import RAW_CAR_TABLES
    from ..validate import _score_fold_laps

    placed = model
    for r in list(model["g2_typ_by_track_car_cond"]):
        key = (r["track_canonical"], r["car"], r["condition"])
        entry = pr.entry_of(placed, *key)
        if entry is not None:
            placed = pr.place(placed, *key, entry, raw=False)
    tables = ("g2_typ_by_track_car_cond", "lap_time_typ_by_track_car_cond")
    same = all(placed[t] == model[t] for t in tables)
    same &= all(placed.get(t) == model.get(t) for t in RAW_CAR_TABLES.values())
    same &= placed.get("outlap_typ_by_track_car_cond") == model.get("outlap_typ_by_track_car_cond")
    before = _score_fold_laps(prep, model)
    after = _score_fold_laps(prep, placed)
    try:
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    except AssertionError:
        same = False
    return bool(same), int(len(before))


def materialised_check(
    model: dict[str, Any], prep: Any, logged: set[tuple[str, str]]
) -> tuple[int, int, int]:
    """With an entry materialised in every cell that has no logged laps
    (``logged``: the (car, track) cells of the whole dataset; the entries
    here are stand-ins, the car's first entry at another track), is the
    holdout of the cells that have theirs the same holdout, row for row?
    ``(moved, moved_on_a_fallback, rows)``: the rows of cells with an
    entry in the fold model that changed (none, if an entry for another
    cell touches nothing), the rows of cells without one that did (they
    rest on the lookups' fallback, which reads other cells), and the rows
    compared."""
    from ..validate import _score_fold_laps

    cars = sorted({r["car"] for r in model["g2_typ_by_track_car_cond"]})
    tracks = sorted({r["track_canonical"] for r in model["g2_typ_by_track_car_cond"]})
    here = {(r["car"], r["track_canonical"]) for r in model["g2_typ_by_track_car_cond"]}
    placed = model
    for car in cars:
        own = [r for r in model["g2_typ_by_track_car_cond"] if r["car"] == car]
        stand_in = pr.entry_of(model, own[0]["track_canonical"], car, own[0]["condition"])
        for track in tracks:
            if (car, track) not in logged and stand_in is not None:
                placed = pr.place(placed, track, car, "dry", replace(stand_in, source="sim"))
    before = _score_fold_laps(prep, model)
    after = _score_fold_laps(prep, placed)
    if len(before) != len(after):
        return len(before), 0, len(before)
    differs = ~(
        (before["P_hot_pred_bar"] == after["P_hot_pred_bar"])
        | (before["P_hot_pred_bar"].isna() & after["P_hot_pred_bar"].isna())
    )
    with_entry = pd.Series(
        [(c, t) in here for c, t in zip(before["car"], before["track"])], index=before.index
    )
    return int((differs & with_entry).sum()), int((differs & ~with_entry).sum()), int(len(before))


def left_in(held: Held) -> int:
    """Rows of the fold model's entry tables that are the held-out
    cell's: none, when the fold left every session of its group out."""
    return sum(
        1
        for name in (
            "g2_typ_by_track_car_cond",
            "lap_time_typ_by_track_car_cond",
            "outlap_typ_by_track_car_cond",
        )
        for r in held.model.get(name) or []
        if r["track_canonical"] == held.track and r["car"] == held.pooled
    )


# ---------------------------------------------------------------- the gate


def _fmt_cells(att: pd.DataFrame) -> pd.DataFrame:
    cols = ["raw_car", "track", "corner", "days", "stints", "size", "signed", "beside"]
    cols += ["beside_gap", "scored", "passed"]
    return att[[c for c in cols if c in att.columns]]


def check_pressure(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    folds_dir: Path | None = None,
    simulate: bool = True,
    dates: bool = True,
) -> GateReport:
    """Gate F (module docstring). ``simulate=False`` stops at the
    yardstick: the folds, the cells, another day against a day, the
    ceiling and what the simulation replaces, with no simulated lap in
    it. That is the gate's first output, to be read before any simulated
    lap is scored. ``dates=False`` leaves the leave-date-out folds out
    (:data:`DATE_FOLDS` more fits)."""
    rep = GateReport("F", "the pressure a simulated lap leads to (`lapsim check-pressure`)")
    prod = load_production(root)
    table = cell_table(prod)
    folds = sorted({(str(r.pooled), str(r.track)) for r in table.itertuples()})
    sim = None
    if simulate:
        sim = load_sim(root, cache_dir=cache_dir, tracks_dir=tracks_dir, folds_dir=folds_dir)
    rep.notes.append(
        f"{len(folds)} folds, each a pooled car at a track with every one of its sessions left "
        "out of the fit; a day is a dry held-out session with "
        f"{pr.MIN_ORDINARY_LAPS} ordinary flying laps (from flying lap {pr.ORDINARY_FROM_LAP} on, "
        f"no lap over {pr.COOL_DOWN} × the stint's best) and a pit-exit stint. The recommended "
        f"cold pressure is for the end of flying lap {pr.FLYING_LAP_N} "
        f"({pr.FLYING_LAPS[1]} beside it) at the team's hot-pressure target for the car and the "
        "session's compound ("
        + ", ".join(
            f"{car} {comp} {bar:g} bar" for (car, comp), bar in sorted(prod.targets.items())
        )
        + "; the car's first listed where the session has no label), from the stint's own "
        "pit-exit temperature and the session's weather, through the fold's K, tau and shares."
    )
    from ..validate import _prepare_fold_laps

    all_rows, all_tpms, checks, day_rows, sim_rows = [], [], [], [], []
    wet_rows: list[pd.DataFrame] = []
    wet_days: list[pd.DataFrame] = []

    def _sim_cells(raw_cars: list[str], layout_of: dict[str, str], share: ShareParams) -> dict:
        out: dict[str, SimCell | None] = {}
        for raw_car in raw_cars:
            if sim is not None and raw_car in layout_of:
                out[raw_car] = sim_cell(sim, raw_car, layout_of[raw_car], share)
        return out

    for pooled, track in folds:
        ids = cell_sessions(prod, pooled, track)
        got = prepare(prod, ids, cache_dir)
        if got is None:
            continue
        model, prep = got
        held = held_of(prod, model, prep, pooled, track, len(ids))
        same, n_rows = identity_check(held.model, held.prep)
        checks.append(
            {
                "fold": f"{pooled} at {track}",
                "left_out": held.left_out,
                "identity": same,
                "holdout_rows": n_rows,
                "cell_rows_in_model": left_in(held),
            }
        )
        # the layout a raw car's laps at the track are of (the cell's most common)
        layout = _majority(list(held.laps["layout_id"])) if len(held.laps) else ""
        weights = pool_weights(prod, pooled, ids)
        raw_cars = sorted(set(held.days["raw_car"]) | set(weights))
        cells = _sim_cells(raw_cars, {r: layout for r in raw_cars}, held.share)
        rows, tpms = score_cell(held, cells, weights=weights)
        # after the scoring: a cell's tracks are built as its days ask for them
        scored = set(held.days["raw_car"])
        sim_rows += [
            _sim_row(pooled, track, k, c) for k, c in cells.items() if c is not None and k in scored
        ]
        all_rows.append(rows)
        all_tpms.append(tpms)
        day_rows.append(held.days.assign(pooled=pooled, track=track))
        # the wet and damp days of the cell: the same fold, and the fold with
        # the car's wet and damp sessions elsewhere left out too
        wet = held_of(prod, model, prep, pooled, track, len(ids), conditions=WET)
        if len(wet.days):
            wet_cells = _sim_cells(
                sorted(set(wet.days["raw_car"])),
                {r: layout for r in wet.days["raw_car"]},
                wet.share,
            )
            wet_rows.append(score_cell(wet, wet_cells, wet=True)[0].assign(fold="track new"))
            wet_days.append(wet.days.assign(pooled=pooled, track=track))
            elsewhere = prod.sessions[
                (prod.sessions["pooled"] == pooled)
                & prod.sessions["session_id"].astype(str).map(prod.conditions).isin(WET)
            ]
            both = sorted(set(ids) | set(elsewhere["session_id"].astype(str)))
            if len(both) > len(ids):
                model2 = fold_model(prod.root, both, cache_dir)
                prep2 = _prepare_fold_laps(prod.root, ids, model2)
                if prep2 is not None and pooled in ho.shares_of(model2):
                    wet2 = held_of(prod, model2, prep2, pooled, track, len(both), conditions=WET)
                    cells2 = _sim_cells(
                        sorted(set(wet2.days["raw_car"])),
                        {r: layout for r in wet2.days["raw_car"]},
                        wet2.share,
                    )
                    wet_rows.append(
                        score_cell(wet2, cells2, wet=True)[0].assign(fold="track and condition new")
                    )
        logger.info("pressure: %s at %s scored (%d sessions)", pooled, track, len(held.days))
    rows = pd.concat([r for r in all_rows if len(r)], ignore_index=True)
    tpms = pd.concat([t for t in all_tpms if len(t)], ignore_index=True)
    days = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()

    # the chronological first visit: a fit of what was logged before the car first ran the track
    visit_rows: list[pd.DataFrame] = []
    for pooled, track, date in first_visits(prod, days) if dates else []:
        after = sorted(prod.sessions.loc[prod.sessions["date"] >= date, "session_id"].astype(str))
        model = fold_model(prod.root, after, cache_dir)
        if pooled not in ho.shares_of(model):
            continue
        own = days[(days["pooled"] == pooled) & (days["track"] == track) & (days["date"] == date)]
        prep = _prepare_fold_laps(prod.root, list(own["session_id"]), model)
        if prep is None:
            continue
        held = held_of(prod, model, prep, pooled, track, len(after))
        layout = _majority(list(held.laps["layout_id"])) if len(held.laps) else ""
        cells = _sim_cells(
            sorted(set(held.days["raw_car"])), {r: layout for r in held.days["raw_car"]}, held.share
        )
        a, b = score_cell(held, cells)
        visit_rows.append(a.assign(what="first visit", tpms=False))
        b = at_lap(b)
        if len(b):
            visit_rows.append(
                b.rename(columns={"resid_mbar": "d_mbar"}).assign(
                    what="first visit", tpms=True, form="tpms", n=pr.FLYING_LAP_N
                )
            )
        logger.info("pressure: first visit of %s to %s on %s scored", pooled, track, date)

    # leave-date-out: the reference level on cells the fit has seen
    seen_rows, seen_tpms = [], []
    logged = {(str(r.pooled), str(r.track)) for r in table.itertuples()}
    for k, ids in enumerate(date_folds(prod) if dates else []):
        got = prepare(prod, ids, cache_dir)
        if got is None:
            continue
        model, prep = got
        # every session's laps through this fold's model: the held-out days are
        # scored, the others are the days they are paired with
        every = _prepare_fold_laps(prod.root, sorted(prod.trained), model)
        present = prep.full[["car", "track_canonical"]].drop_duplicates()
        out_ids, fold_memo = set(ids), {}  # type: ignore[var-annotated]
        fold_tracks: dict[str, trk.TrackModel] = {}
        for pooled, track in zip(present["car"], present["track_canonical"]):
            held = held_of(prod, model, every, str(pooled), str(track), len(ids))
            seen_cells: dict[str, SimCell | None] = {}
            if sim is not None:
                mine = held.days[held.days["session_id"].isin(ids)]
                for raw_car, own in mine.groupby("raw_car", sort=True):
                    layout_id = _majority(list(own["layout_id"]))
                    if layout_id in sim.tracks and layout_id not in fold_tracks:
                        fold_tracks[layout_id] = track_without(sim, layout_id, out_ids)
                    seen_cells[str(raw_car)] = seen_cell(
                        sim,
                        str(raw_car),
                        layout_id,
                        held.share,
                        out_ids,
                        fold_tracks.get(layout_id),
                        fold_memo,
                    )
            a, b = score_cell(held, seen_cells, seen=True, sessions=set(ids))
            seen_rows.append(a)
            seen_tpms.append(b)
        same_all, n_all = identity_check(model, prep)
        moved, on_fallback, _ = materialised_check(model, prep, logged)
        checks.append(
            {
                "fold": f"dates {k + 1} of {DATE_FOLDS}",
                "left_out": len(ids),
                "identity": same_all,
                "holdout_rows": n_all,
                "cell_rows_in_model": 0,
                "moved_by_other_cells_entries": moved,
                "moved_on_a_fallback": on_fallback,
            }
        )
        logger.info("pressure: date fold %d scored", k + 1)
    seen = pd.concat([r for r in seen_rows if len(r)], ignore_index=True) if seen_rows else None
    seen_t = pd.concat([t for t in seen_tpms if len(t)], ignore_index=True) if seen_tpms else None

    # the calculator as shipped: tire_model.json, every day in its fit
    shipped, shipped_t = None, None
    artifact = Path(root) / "tire_model.json"
    from ..heat_frame import current, frame_of

    shipped_model = json.loads(artifact.read_text()) if dates and artifact.exists() else None
    if shipped_model is not None and frame_of(shipped_model) != current():
        # its entries and the days' heat would be in two frames (#376)
        rep.notes.append(
            f"The model as shipped is not scored: it is in the {frame_of(shipped_model)} frame "
            f"and this run in the {current()} frame (`lapsim check-pressure` reads the "
            "committed model's)."
        )
        shipped_model = None
    if shipped_model is not None:
        model = shipped_model
        every = _prepare_fold_laps(prod.root, sorted(prod.trained), model)
        parts_a, parts_b = [], []
        for pooled, track in folds:
            if every is None or pooled not in ho.shares_of(model):
                continue
            a, b = score_cell(held_of(prod, model, every, pooled, track, 0), {}, seen=True)
            parts_a.append(a)
            parts_b.append(b)
        if any(len(x) for x in parts_a):
            shipped = pd.concat([x for x in parts_a if len(x)], ignore_index=True)
            shipped_t = pd.concat([x for x in parts_b if len(x)], ignore_index=True)
        logger.info("pressure: the model as shipped scored")

    stints = by_stint(rows)
    if cache_dir is not None:
        # the rows behind the report, for whoever reads a miss: not a cache
        out = Path(cache_dir) / "pressure_rows"
        out.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out / "rows.parquet")
        tpms.to_parquet(out / "tpms.parquet")
        days.to_parquet(out / "days.parquet")
        if seen is not None:
            seen.to_parquet(out / "seen_rows.parquet")
        if shipped is not None:
            shipped.to_parquet(out / "shipped_rows.parquet")
    report_cells(rep, table, days)
    report_harness(rep, pd.DataFrame(checks))
    main = stints[stints["n"] == pr.FLYING_LAP_N] if len(stints) else stints
    report_yardstick(rep, main)
    if sim is not None:
        report_simulation(rep, main, pd.DataFrame(sim_rows), stints)
    report_replaced(rep, main)
    report_seen(rep, main, seen)
    report_tpms(rep, tpms, seen_t)
    report_current(rep, main, seen, shipped, tpms, seen_t, shipped_t)
    report_nfold(rep, seen, seen_t)
    report_wet(rep, wet_rows, wet_days)
    report_first_visit(rep, visit_rows, rows, tpms)
    return rep


def first_visits(prod: Production, days: pd.DataFrame) -> list[tuple[str, str, str]]:
    """``(pooled car, track, date)`` of each cell's first scored day, where
    the pooled car had a session in a fit at another track before it:
    the visits a fit of what was logged before them can be scored on."""
    if days.empty:
        return []
    trained = prod.sessions[prod.sessions["session_id"].astype(str).isin(prod.trained)]
    out = []
    for (pooled, track), own in days.groupby(["pooled", "track"], sort=True):
        first = str(own["date"].min())
        before = trained[
            (trained["pooled"] == pooled)
            & (trained["track_canonical"] != track)
            & (trained["date"] < first)
        ]
        if len(before):
            out.append((str(pooled), str(track), first))
    return out


def report_current(
    rep: GateReport,
    main: pd.DataFrame,
    seen: pd.DataFrame | None,
    shipped: pd.DataFrame | None,
    tpms: pd.DataFrame,
    seen_t: pd.DataFrame | None,
    shipped_t: pd.DataFrame | None,
) -> None:
    """The calculator as it is today by the same measure, beside the
    simulation: at a cell it has no logged laps for (the fallback as
    coded, on the cell folds), at a cell it has (its own entry, with the
    day's date left out of the fit) and as shipped (the day in the fit)."""

    def _five(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        own = by_stint(frame)
        return own[own["n"] == pr.FLYING_LAP_N]

    def _mae(frame: pd.DataFrame | None, arm: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        t = tpms_table(frame)
        return t[t["arm"] == arm]

    sources = [
        ("the simulation, at a cell without logged laps", main, "sim", _mae(tpms, "sim")),
        (
            "the calculator today, at a cell without logged laps (the fallback as coded)",
            main,
            "fallback",
            _mae(tpms, "fallback"),
        ),
        (
            "the calculator today, at a cell with logged laps, the day's date left out",
            _five(seen),
            "entry",
            _mae(seen_t, "entry"),
        ),
        (
            "the calculator as shipped, the day in its fit",
            _five(shipped),
            "entry",
            _mae(shipped_t, "entry"),
        ),
    ]
    lines, cells = [], []
    for name, stints, arm, mae in sources:
        if stints is None or len(stints) == 0:
            continue
        row: dict[str, Any] = {"what": name}
        for form in FORMS:
            att = attain(stints[stints["form"] == form], arm)
            got = tally(att)
            row[f"{form}_cells"] = f"{got['passed']} of {got['scored']}"
            row[f"{form}_mbar"] = got["size"]
            row[f"{form}_another_day"] = got["beside"]
            if len(att):
                per = att[att["scored"]].groupby(["raw_car", "track"], sort=True)
                cells.append(
                    per.agg(size=("size", "median"), another_day=("beside", "median"))
                    .reset_index()
                    .assign(what=name, form=form)
                )
        ok = mae[mae["rows"] >= 4 * MIN_DAYS] if len(mae) else mae
        row["tpms_mae"] = float(ok["mae_mbar"].median()) if len(ok) else float("nan")
        lines.append(row)
    if not lines:
        return
    table = pd.DataFrame(lines)
    detail = (
        "Every row by the one measure: the cold pressure recommended for the end of flying lap "
        f"{pr.FLYING_LAP_N} against the day's own, with another logged day beside it on the "
        "same stints (`_another_day`), at a given lap time (`heat`) and with nothing given "
        "(`chain`: the calculator's own prefilled lap time and out-lap); `_cells`: cells (a raw "
        "car at a track on one tyre) no further from the day than another day is; `_mbar`: the "
        "distance in the median scored cell. `tpms_mae`: hot-pressure MAE against the TPMS at "
        "flying lap 5 in the median cell, which does not rank a heat source. *The calculator "
        "today* reads one entry per pooled car and track. Where it has no logged laps for the "
        "cell it falls back on other cells (the cell folds: every constant fitted without the "
        "cell). Where it has them, the entry is scored on days whose date was left out of the "
        f"fit ({DATE_FOLDS} folds), and as shipped (`tire_model.json`, which has seen the day: "
        "that row is in sample). The simulation's row is at a cell the constants, the car "
        "model and (at Tsukuba and Fuji) the track have not seen.\n\n"
        + md_table(table, floats="{:.3g}")
    )
    if cells:
        both = pd.concat(cells, ignore_index=True)
        wide = both.pivot_table(
            index=["form", "raw_car", "track"], columns="what", values="size", sort=False
        ).reset_index()
        yard = both.groupby(["form", "raw_car", "track"], sort=False)["another_day"].median()
        wide = wide.merge(yard.reset_index(), on=["form", "raw_car", "track"])
        detail += (
            "\n\n**By cell** (the four tyres' median distance from the day, mbar; "
            "`another_day` the median over the rows' yardsticks).\n\n"
            + md_table(wide, floats="{:.3g}")
        )
    rep.items.append(
        GateItem(
            "The calculator as it is today, by the same measure",
            None,
            "; ".join(
                f"{r['what']}: {r['heat_mbar']:.1f} mbar at a given lap time "
                f"({r['heat_cells']}), {r['chain_mbar']:.1f} with nothing given "
                f"({r['chain_cells']})"
                for r in lines
            ),
            "reported: the same pairing, rows and cells as the simulation's",
            detail,
        )
    )


def report_nfold(rep: GateReport, seen: pd.DataFrame | None, seen_t: pd.DataFrame | None) -> None:
    """The n-fold analysis: on the cells the calculator has logged, every
    dry day scored once with its date left out, the simulated entry in
    place of the logged one on the same rows."""
    if seen is None or seen.empty or not (seen["arm"] == "sim").any():
        return
    stints = by_stint(seen)
    stints = stints[stints["n"] == pr.FLYING_LAP_N]
    lines, cells, heads = [], [], []
    for form in FORMS:
        own = stints[stints["form"] == form]
        # the two on the stints both have, and the yardstick has
        keys = own.loc[own["arm"] == "sim", _STINT].merge(
            own.loc[own["arm"] == "entry", _STINT], on=_STINT
        )
        own = own.merge(keys.drop_duplicates(), on=_STINT)
        row: dict[str, Any] = {"form": form}
        both = None
        for arm, name in (("entry", "logged"), ("sim", "sim")):
            att = attain(own, arm)
            got = tally(att)
            row[f"{name}_cells"] = f"{got['passed']} of {got['scored']}"
            row[f"{name}_mbar"] = got["size"]
            row["another_day"] = got["beside"]
            part = att[_CELL + ["days", "stints", "size", "signed", "beside", "scored"]].rename(
                columns={"size": name, "signed": f"{name}_signed"}
            )
            both = (
                part
                if both is None
                else both.merge(part[_CELL + [name, f"{name}_signed"]], on=_CELL)
            )
        versus = attain(own, "sim", beside="entry")
        versus = versus[versus["scored"]] if len(versus) else versus
        row["sim_nearer"] = int((versus["size"] < versus["beside"]).sum()) if len(versus) else 0
        row["sim_further"] = int((versus["size"] > versus["beside"]).sum()) if len(versus) else 0
        lines.append(row)
        if both is not None:
            cells.append(both.assign(form_=form))
        heads.append(
            f"{FORM_NAMES[form]}: the logged entry {row['logged_mbar']:.1f} mbar from the day "
            f"({row['logged_cells']}), the simulated entry {row['sim_mbar']:.1f} "
            f"({row['sim_cells']}), another day {row['another_day']:.1f}; the simulation nearer "
            f"in {row['sim_nearer']} cells and further in {row['sim_further']}"
        )
    detail = (
        f"{DATE_FOLDS} folds of whole dates, so every dry day with a pit-exit stint is scored "
        "once by a fit that has not seen its date. On each day two entries are read for the "
        "day's cell through that fit: the fold model's own (the calculator today: the pooled "
        "car's logged laps at the track on the other dates) and a simulated one, in which "
        "nothing has seen the fold's dates at the layout either: the car model is fitted "
        "without them, the track's curvature built without them, the frame factor and the fed "
        "use read from the car's laps at the layout on other dates. The measure, the pairing "
        "and the cells are the gate's; both entries are on the stints both have.\n\n"
        + md_table(pd.DataFrame(lines), floats="{:.3g}")
    )
    if cells:
        table = pd.concat(cells, ignore_index=True)
        detail += (
            "\n\n**By cell** (`logged`, `sim`: the distance from the day, mbar, with the "
            "signed median; `beside`: another day's).\n\n"
            + md_table(
                table[
                    ["form_", "raw_car", "track", "corner", "days", "stints", "logged"]
                    + ["logged_signed", "sim", "sim_signed", "beside", "scored"]
                ],
                floats="{:.3g}",
            )
        )
    if seen_t is not None and len(seen_t):
        lap = ["session_id", "stint_id", "lap_num", "corner"]
        have = seen_t.loc[seen_t["arm"] == "sim", lap].merge(
            seen_t.loc[seen_t["arm"] == "entry", lap], on=lap
        )
        own_t = seen_t[seen_t["arm"].isin(["entry", "sim"])].merge(have.drop_duplicates(), on=lap)
        clean = own_t[~(own_t["after_dropped"] | own_t["after_cool_down"])]
        rows_t = []
        for label, frame in (
            ("every scored lap", own_t),
            ("every scored lap, clean rows", clean),
            (f"flying lap {pr.FLYING_LAP_N} of pit-exit stints, clean rows", at_lap(clean)),
        ):
            for pooled, grp in list(frame.groupby("pooled", sort=True)) + [("all", frame)]:
                wide = grp.groupby("arm")["resid_mbar"].agg(
                    mae=lambda x: float(x.abs().mean()), bias="mean", rows="size"
                )
                if {"entry", "sim"} <= set(wide.index):
                    rows_t.append(
                        {
                            "rows_of": label,
                            "car": pooled,
                            "rows": int(wide.loc["sim", "rows"]),
                            "logged_mae": wide.loc["entry", "mae"],
                            "sim_mae": wide.loc["sim", "mae"],
                            "logged_bias": wide.loc["entry", "bias"],
                            "sim_bias": wide.loc["sim", "bias"],
                        }
                    )
        if rows_t:
            t = pd.DataFrame(rows_t)
            top = t[(t["rows_of"] == "every scored lap") & (t["car"] == "all")].iloc[0]
            heads.append(
                f"against the TPMS over {int(top['rows'])} lap × tyre rows: the logged entry "
                f"{top['logged_mae']:.1f} mbar MAE, the simulated entry {top['sim_mae']:.1f}"
            )
            detail += (
                "\n\n**Pressure against the TPMS**, the holdout's own measure (hot-pressure "
                "MAE and signed bias, mbar; production's inputs, the session's 25th-percentile "
                "lap time typed in), on the lap × tyre rows both entries have. It does not rank "
                "a heat source (a heat too low scores better in most cells); it is here "
                "because it is the number the standing holdout reports.\n\n"
                + md_table(t, floats="{:.3g}")
            )
    rep.items.append(
        GateItem(
            "The n-fold analysis: the simulated entry in place of the logged one, on the cells "
            "the calculator has logged",
            None,
            "; ".join(heads),
            "reported: an entry with logged laps keeps its logged heat (#208)",
            detail,
        )
    )


def report_wet(rep: GateReport, wet_rows: list[pd.DataFrame], wet_days: list[pd.DataFrame]) -> None:
    """Reported beside: the wet and damp cells, against a logged run of
    the same wet day."""
    frames = [r for r in wet_rows if len(r)]
    if not frames:
        return
    rows = pd.concat(frames, ignore_index=True)
    days = pd.concat(wet_days, ignore_index=True)
    lines, parts = [], []
    for fold, own in rows.groupby("fold", sort=True):
        stints = by_stint(own)
        stints = stints[stints["n"] == pr.FLYING_LAP_N]
        att = attain(stints, "sim")
        if att.empty:
            continue
        paired = att[att["beside_days"] >= 1]
        lines.append(
            f"{fold}: the simulation no further than a run of the same wet day in "
            f"{int((paired['size'] <= paired['beside']).sum())} of {len(paired)} cells, "
            f"{paired['size'].median():.1f} mbar from the day against {paired['beside'].median():.1f}"
        )
        alone = _cells(stints[stints["arm"].isin(["sim", "fallback", "heat_up"])])
        wide = alone.pivot_table(index=_CELL, columns="arm", values="size").reset_index()
        both = att[_CELL + ["days", "stints", "size", "signed", "beside", "flagged"]].merge(
            wide.rename(columns={"sim": "sim_every_day"}), on=_CELL, how="outer"
        )
        parts.append(both.assign(fold=fold))
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    counts = (
        days.groupby(["pooled", "track", "raw_car", "condition"], sort=True)
        .agg(sessions=("session_id", "size"), days=("date", "nunique"))
        .reset_index()
    )
    rep.items.append(
        GateItem(
            "The wet and damp cells, against a logged run of the same wet day",
            None,
            "; ".join(lines),
            "reported beside (#208): nothing rests on one",
            "A wet or damp day's own recommendation (its own heat at its own lap time, the "
            "fold's wet constants where it has them) against the simulation's: the dry table "
            "read at the wet lap time, which is typed in, with no wet term (gate D2). The "
            "yardstick in the wet is another session of the *same* day (wet days do not "
            "repeat), same raw car, powertrain, kind of source and condition. `track new`: the "
            "cell's fold; `track and condition new`: the car's wet and damp sessions at every "
            "other track left out of the fit as well. `size`, `signed`, `beside`: on the stints "
            "with such a run (mbar); `sim_every_day`, `fallback`, `heat_up`: the simulation, "
            "the fold model as it stands and the day's own heat × "
            f"{1.0 + HEAT_PLANT:g} on every wet stint, with or without a pair.\n\n"
            + md_table(
                table[
                    ["fold", "raw_car", "track", "corner", "days", "stints", "size", "signed"]
                    + ["beside", "flagged", "sim_every_day", "heat_up", "fallback"]
                ],
                floats="{:.3g}",
            )
            + "\n\n**The wet and damp days.**\n\n"
            + md_table(counts),
        )
    )


def report_first_visit(
    rep: GateReport, visit_rows: list[pd.DataFrame], rows: pd.DataFrame, tpms: pd.DataFrame
) -> None:
    """Reported: the first day a car ran a track, under a fit of what was
    logged before it, beside the same day under the cell's fold."""
    frames = [r for r in visit_rows if len(r)]
    if not frames:
        return
    visit = pd.concat(frames, ignore_index=True)
    sessions = set(visit["session_id"])
    cell = rows[rows["session_id"].isin(sessions)].assign(what="cell left out", tpms=False)
    cell_t = at_lap(tpms[tpms["session_id"].isin(sessions)]).rename(
        columns={"resid_mbar": "d_mbar"}
    )
    cell_t = cell_t.assign(what="cell left out", tpms=True, form="tpms", n=pr.FLYING_LAP_N)
    both = pd.concat([visit, cell, cell_t], ignore_index=True)
    both = both[both["n"] == pr.FLYING_LAP_N]
    arms = ["sim", "ceiling", "other_layouts", "fallback", "own_heat"]
    both = both[both["arm"].isin(arms)]
    table = (
        both.assign(size=both["d_mbar"].abs())
        .groupby(["pooled", "track", "date", "form", "arm", "what"], sort=True)["size"]
        .agg(["median", "mean", "size"])
        .reset_index()
    )
    wide = table.pivot_table(
        index=["pooled", "track", "date", "form", "arm"], columns="what", values=["median", "mean"]
    )
    names: list[str] = []
    for column in wide.columns.tolist():
        stat, what = tuple(column)
        names.append(f"{what}, {stat}")
    wide = wide.set_axis(names, axis=1).reset_index()
    events = visit.drop_duplicates(["pooled", "track", "date"])
    rep.items.append(
        GateItem(
            "The chronological first visit: a fit of what was logged before it",
            None,
            f"{len(events)} first visits, {len(sessions)} sessions: "
            + "; ".join(f"{r.pooled} at {r.track} on {r.date}" for r in events.itertuples()),
            "reported (#208): a check that leaving a cell out does not flatter",
            "The first scored day of a car at a track, where the car had run elsewhere before: "
            "the pressure model fitted on the sessions before that date only (`first visit`), "
            "beside the same sessions under the cell's own fold (`cell left out`, which knows "
            "the car's later days elsewhere). The simulated side is the cell fold's in both "
            "(the car fitted without the layout, not without its later days). `heat`, `chain`: "
            "the size of an arm's distance from the day's own recommendation, mbar, over the "
            "day's stints and tyres; `tpms`: the size of prediction less the TPMS reading at "
            "flying lap 5, where `mean` is the MAE. One day has no other day beside it.\n\n"
            + md_table(wide, floats="{:.3g}"),
        )
    )


def _sim_row(pooled: str, track: str, raw_car: str, cell: SimCell) -> dict:
    fed = (cell.fed.get("sim") or {}).get(raw_car, {})
    layouts = fed.get("layouts") or {}
    return {
        "pooled": pooled,
        "track": track,
        "raw_car": raw_car,
        "layout_id": cell.layout_id,
        "track_from": ", ".join(cell.donors["cars"]) if cell.neither else "own (car unseen only)",
        "donor_laps": cell.donors["laps"] if cell.neither else 0,
        "donor_days": cell.donors["days"] if cell.neither else 0,
        "fewest_donor_laps": (
            cell.donors.get("fewest_laps", cell.donors["laps"]) if cell.neither else 0
        ),
        "fed_use": fed.get("use", float("nan")),
        "fed_from": "; ".join(
            f"{k} {v['use']:.3f} ({v['sessions']} sessions, {v['days']} days)"
            for k, v in layouts.items()
        ),
        "fed_spread": fed.get("spread", float("nan")),
        "typical_use": cg.typical_use(cell.drives["sim"]) if cell.drives["sim"] else float("nan"),
        "factor": "; ".join(
            f"{src} " + "/".join(f"{x:.3f}" for x in f) + f" ({cell.factor_laps.get(src, 0)} laps)"
            for src, f in sorted(cell.factors.items())
        ),
    }


def report_cells(rep: GateReport, table: pd.DataFrame, days: pd.DataFrame) -> None:
    """The folds and the cells, per raw car, before anything is scored."""
    if days.empty:
        rep.items.append(GateItem("The cells", False, "no day to score", "at least one"))
        return
    per = (
        days.groupby(["pooled", "track", "raw_car"], sort=True)
        .agg(
            days_scored=("date", "nunique"),
            sessions_scored=("session_id", "size"),
            sources=(
                "source",
                lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()),
            ),
            powertrains=("powertrain", lambda s: ", ".join(sorted(set(s)))),
            lap_time_s=("lap_time_s", "median"),
        )
        .reset_index()
    )
    both = table.merge(per, on=["pooled", "track", "raw_car"], how="left")
    both[["days_scored", "sessions_scored"]] = (
        both[["days_scored", "sessions_scored"]].fillna(0).astype(int)
    )
    both["verdict"] = both["days_scored"] >= MIN_DAYS
    rep.items.append(
        GateItem(
            "The folds and the cells, per raw car",
            None,
            f"{int(both['verdict'].sum())} raw-car cells with {MIN_DAYS} dry days or more: "
            + "; ".join(
                f"{r.raw_car} at {r.track} ({r.sessions_scored} sessions on {r.days_scored} days)"
                for r in both[both["verdict"]].itertuples()
            ),
            f"reported; a raw car's cell is scored with {MIN_DAYS} dry days",
            "`left_out`: every session of the pooled car at the track, left out of the fold's "
            "fit; `trained`: those of the raw car that reach a fit; `dry_sessions`, `dry_days`: "
            "of those, labelled dry; `sessions_scored`, `days_scored`: with a recommendation of "
            "their own; `sources`: the kind of sensor behind the ordinary laps (a day is paired "
            "with days of the same kind); `verdict`: scored.\n\n" + md_table(both, floats="{:.2f}"),
        )
    )


def report_harness(rep: GateReport, checks: pd.DataFrame) -> None:
    """Clause 1: the harness is what the issue says it is."""
    if checks.empty:
        return
    ok = bool(checks["identity"].all())
    rep.items.append(
        GateItem(
            "An entry read from a fold's model and placed again is the holdout, bit for bit",
            ok,
            f"{int(checks['identity'].sum())} of {len(checks)} folds identical, "
            f"{int(checks['holdout_rows'].sum())} holdout rows",
            "every fold: the same entry rows, and `validate._score_fold_laps` the same frame",
            md_table(checks),
        )
    )
    if "moved_by_other_cells_entries" in checks.columns:
        by_date = checks[checks["moved_by_other_cells_entries"].notna()]
        moved = int(by_date["moved_by_other_cells_entries"].sum())
        rep.items.append(
            GateItem(
                "Entries materialised in cells without logged laps leave the leave-date-out "
                "holdout as it was",
                moved == 0,
                f"{moved} rows of cells with an entry moved, of {int(by_date['holdout_rows'].sum())}; "
                f"{int(by_date['moved_on_a_fallback'].sum())} rows of cells the fold left without "
                "an entry moved (they rest on the lookups' fallback, which reads other cells: "
                "#260)",
                "no row of a cell that has its entry in the fold model",
            )
        )
    leak = int(checks["cell_rows_in_model"].sum())
    rep.items.append(
        GateItem(
            "A fold leaves no session of its group in training",
            leak == 0,
            f"{leak} rows of a held-out cell in a fold model's entry tables",
            "none",
        )
    )


def _arm_table(main: pd.DataFrame, arms: tuple[str, ...], form: str) -> pd.DataFrame:
    """Per arm in one form: its cells beside the yardstick's, each on the
    stints the arm and the yardstick both have."""
    own = main[main["form"] == form]
    out = []
    for arm in arms:
        att = attain(own, arm)
        if att.empty:
            continue
        got = tally(att)
        out.append(
            {
                "arm": arm,
                "cells": f"{got['passed']} of {got['scored']}",
                "whole": f"{got['whole']} of {got['groups']}",
                "from_the_day_mbar": got["size"],
                "another_day_mbar": got["beside"],
            }
        )
    return pd.DataFrame(out)


def report_yardstick(rep: GateReport, main: pd.DataFrame) -> None:
    """Clause 2: another day against a day, and the ceiling, in both forms."""
    if main.empty:
        return
    parts, lines = [], []
    for form in FORMS:
        cells = by_cell(main[main["form"] == form])
        yard = cells[cells["arm"] == "other_day"]
        if yard.empty:
            continue
        scored = yard[yard["days"] >= MIN_DAYS]
        ceiling = attain(main[main["form"] == form], "ceiling")
        got = tally(ceiling)
        lines.append(
            f"{FORM_NAMES[form]}: another day is {scored['size'].median():.1f} mbar from the "
            f"day in the median cell ({scored['size'].min():.1f} to {scored['size'].max():.1f} "
            f"over {len(scored)} cells; lap-time gap of a pair {100 * scored['gap'].median():.2f} "
            f"%), the ceiling {got['size']:.1f} mbar ({got['passed']} of {got['scored']} cells no "
            "further than another day)"
        )
        shown = yard.merge(
            ceiling[_CELL + ["size", "signed", "gap"]].rename(
                columns={"size": "ceiling", "signed": "ceiling_signed", "gap": "ceiling_gap"}
            ),
            on=_CELL,
            how="left",
        )
        shown = shown.rename(columns={"size": "another_day", "gap": "pair_gap"})
        shown.insert(0, "form_", form)
        parts.append(
            shown[
                ["form_", "raw_car", "track", "corner", "days", "stints", "another_day"]
                + ["pair_gap", "ceiling", "ceiling_signed", "ceiling_gap"]
            ]
        )
    detail = (
        "The distance of a recommendation from the day's own, in mbar of cold pressure at the "
        f"end of flying lap {pr.FLYING_LAP_N}: the median over days of the day's median pit-exit "
        "stint, per raw car, track and tyre. `another_day`: a logged day of the same raw car, "
        "powertrain and kind of source at the track. In `heat` it is the heat of up to "
        f"{cg.FLOOR_PAIRS_PER_LAP} sessions of other days nearest in lap time, each from its "
        f"{pr.NEAR_LAPS} ordinary laps nearest the day's lap time, fed at the day's lap time; "
        "`pair_gap` is the size of the log of those laps' lap time over the day's. In `chain` "
        "it is every other day as if it alone had written the entry (its own median lap time, "
        "median heat and out-lap), and `pair_gap` the size of the log of that lap time over the "
        "day's. `ceiling`: the raw car's logged entry at the cell built without the day (what a "
        "seen cell's entry gives on a new day), with its signed distance and lap-time gap.\n\n"
        + md_table(pd.concat(parts, ignore_index=True), floats="{:.3g}")
        if parts
        else ""
    )
    rep.items.append(
        GateItem(
            "The yardstick: another day against a day, and the ceiling",
            None,
            "; ".join(lines),
            "reported, before any simulated lap is scored (#208, clause 2)",
            detail,
        )
    )


def report_replaced(rep: GateReport, main: pd.DataFrame) -> None:
    """Clause 6 without the simulation: what it replaces, on the same rows."""
    if main.empty:
        return
    tables = []
    for form in FORMS:
        t = _arm_table(main, ("ceiling", "other_layouts", "fallback"), form)
        if len(t):
            t.insert(0, "form", form)
            tables.append(t)
    if not tables:
        return
    both = pd.concat(tables, ignore_index=True)
    rep.items.append(
        GateItem(
            "What stands in for a cell without logged laps today",
            None,
            "; ".join(
                f"{r['form']} {r['arm']} {r['from_the_day_mbar']:.1f} mbar ({r['cells']})"
                for r in both.to_dict("records")
            ),
            "reported: the car's other layouts (#260's last step) and the fallback as coded, "
            "beside the ceiling",
            "`cells`: the cells in which the arm is no further from the day than another day "
            "is; `whole`: raw car × track groups with all four tyres inside; "
            "`from_the_day_mbar`, `another_day_mbar`: the two distances in the median scored "
            "cell.\n\n" + md_table(both, floats="{:.3g}"),
        )
    )


def report_seen(rep: GateReport, main: pd.DataFrame, seen: pd.DataFrame | None) -> None:
    """The reference level on seen cells, and the pooled entry's cost."""
    if seen is not None and len(seen):
        own = by_stint(seen)
        own = own[own["n"] == pr.FLYING_LAP_N]
        parts = []
        for form in FORMS:
            cells = by_cell(own[own["form"] == form], beside="entry")
            cells = cells[(cells["arm"] == "entry") & (cells["days"] >= MIN_DAYS)]
            parts.append(cells.assign(form_=form))
        both = pd.concat(parts, ignore_index=True)
        if len(both):
            rep.items.append(
                GateItem(
                    "The reference level: a seen cell's entry on a day it has not seen "
                    "(leave-date-out)",
                    None,
                    "; ".join(
                        f"{form}: {grp['size'].median():.1f} mbar from the day in the median "
                        f"cell ({len(grp)} cells)"
                        for form, grp in both.groupby("form_", sort=False)
                    ),
                    "reported: entries with logged laps are untouched by the simulation",
                    f"{DATE_FOLDS} folds, each every session of its dates left out of the fit "
                    "(so a held-out session has no same-day sibling of its cell in training): "
                    "the fold model's own entry for the cell, which the constants have seen and "
                    "which is the pooled car's, against the day's own recommendation. `size`, "
                    "`signed`: mbar; `gap`: the size of the log of the entry's lap time over "
                    "the day's.\n\n"
                    + md_table(
                        both[
                            ["form_", "raw_car", "track", "corner", "days", "stints", "size"]
                            + ["signed", "gap"]
                        ],
                        floats="{:.3g}",
                    ),
                )
            )
    if main.empty:
        return
    parts = []
    for form in FORMS:
        cells = by_cell(main[main["form"] == form], beside="ceiling_pooled")
        own_c = cells[cells["arm"] == "ceiling"][_CELL + ["size", "days"]]
        pooled = cells[cells["arm"] == "ceiling_pooled"][_CELL + ["size", "signed"]]
        yard = cells[cells["arm"] == "other_day"][_CELL + ["size"]]
        both = own_c.merge(pooled, on=_CELL, suffixes=("_own", "_pooled")).merge(
            yard.rename(columns={"size": "another_day"}), on=_CELL, how="left"
        )
        parts.append(both.assign(form_=form))
    cost = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cost = cost[cost["days"] >= MIN_DAYS] if len(cost) else cost
    sims = []
    for form in FORMS:
        own = main[main["form"] == form]
        att = attain(own, "sim_pooled", beside="sim")
        if len(att):
            sims.append(att[att["days"] >= MIN_DAYS].assign(form_=form))
    pooled_sim = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame()
    if len(pooled_sim):
        rep.items.append(
            GateItem(
                "The pooled simulated entry's cost per raw car",
                None,
                "; ".join(
                    f"{form}: the pooled simulated entry {grp['size'].median():.1f} mbar from the "
                    f"day where the raw car's own simulated entry is {grp['beside'].median():.1f}"
                    for form, grp in pooled_sim.groupby("form_", sort=False)
                ),
                "reported: splitting the alias is the owner's",
                "What the calculators would read where the model pools raw cars: per tyre the "
                "geometric mean of the raw cars' simulated entries at one reference lap time, "
                "each of its latest powertrain, weighted by the raw car's share of the pooled "
                "car's usable laps in the fold's fit (a raw car with no fed use, as the KK-F-V2, "
                "or no frame factor for the day's kind of source has no part). `size`, `signed`: "
                "the pooled entry's distance from the day; `beside`: the raw car's own simulated "
                "entry's, on the same stints (mbar).\n\n"
                + md_table(
                    pooled_sim[
                        ["form_", "raw_car", "track", "corner", "days", "stints", "size"]
                        + ["signed", "beside", "gap"]
                    ],
                    floats="{:.3g}",
                ),
            )
        )
    if len(cost):
        rep.items.append(
            GateItem(
                "The pooled formula-car entry's cost per raw car",
                None,
                "; ".join(
                    f"{form}: the pooled entry {grp['size_pooled'].median():.1f} mbar from the "
                    f"day where the raw car's own is {grp['size_own'].median():.1f}"
                    for form, grp in cost.groupby("form_", sort=False)
                ),
                "reported: splitting the alias is the owner's",
                "The logged entry built without the day from every raw car pooled under the "
                "model's name (`size_pooled`, `signed`), which is what the calculators read, "
                "beside the raw car's own (`size_own`) and the distance between two days of "
                "the raw car (`another_day`), mbar.\n\n"
                + md_table(
                    cost[
                        ["form_", "raw_car", "track", "corner", "days", "size_own"]
                        + ["size_pooled", "signed", "another_day"]
                    ],
                    floats="{:.3g}",
                ),
            )
        )


def report_tpms(rep: GateReport, tpms: pd.DataFrame, seen: pd.DataFrame | None) -> None:
    """Clause 7: pressure against the TPMS, reported and never ranking."""
    if tpms.empty:
        return
    # every arm on the rows the simulation has (all of them where there is none)
    keys = ["session_id", "stint_id", "lap_num", "corner"]
    if (tpms["arm"] == "sim").any():
        tpms = tpms.merge(tpms.loc[tpms["arm"] == "sim", keys].drop_duplicates(), on=keys)
    table = tpms_table(tpms)
    cell = ["pooled", "track", "raw_car"]
    wide = table.pivot_table(index=cell, columns="arm", values=["mae_mbar", "bias_mbar"])
    names: list[str] = []
    for column in wide.columns.tolist():
        what, arm = tuple(column)
        names.append(f"{arm}_{str(what).split('_')[0]}")
    wide = wide.set_axis(names, axis=1).reset_index()
    rows = table.groupby(cell)[["rows", "set_aside"]].max().reset_index()
    wide = rows.merge(wide, on=cell)
    if seen is not None and len(seen):
        ref = tpms_table(seen)
        ref = ref[ref["arm"] == "entry"][cell + ["mae_mbar", "bias_mbar"]]
        wide = wide.merge(
            ref.rename(columns={"mae_mbar": "in_the_fit_mae", "bias_mbar": "in_the_fit_bias"}),
            on=cell,
            how="left",
        )
    wide = wide[wide["rows"] >= 4 * MIN_DAYS]
    arms = [c[:-4] for c in wide.columns if c.endswith("_mae")]
    first = ("own_heat", "heat_up", "heat_down", "sim", "ceiling")
    order: list[str] = [c for c in first if c in arms] + [c for c in arms if c not in first]
    shown = wide[cell + ["rows", "set_aside"] + [f"{a}_mae" for a in order]]
    bias = wide[cell + [f"{a}_bias" for a in order]]
    plant = ""
    if {"own_heat_mae", "heat_up_mae", "heat_down_mae"} <= set(wide.columns):
        moved = (wide["heat_down_mae"] - wide["own_heat_mae"]).median()
        up = (wide["heat_up_mae"] - wide["own_heat_mae"]).median()
        plant = (
            f"; the day's own heat scaled by ± {HEAT_PLANT:.0%} moves the MAE by {up:+.1f} and "
            f"{moved:+.1f} mbar in the median cell"
        )
    rep.items.append(
        GateItem(
            "Pressure against the TPMS at flying lap "
            f"{pr.FLYING_LAP_N} (S2: reported, it does not rank a heat source)",
            None,
            "; ".join(f"{a} {wide[f'{a}_mae'].median():.0f} mbar" for a in order)
            + " in the median cell"
            + plant,
            "reported (#208, clause 7): a heat source with too little heat scores better",
            "Hot-pressure MAE (prediction less the TPMS reading, mbar) at the end of flying lap "
            f"{pr.FLYING_LAP_N} of pit-exit stints, production's own inputs (the session's "
            "25th-percentile lap time typed in), per raw car and cell, the constants fitted "
            "without the cell, every arm on the rows the simulation has. `rows`: stint × tyre "
            "rows; `set_aside`: rows after a lap that cleaning dropped from the stint or at or "
            "after a cool-down lap, not counted. `own_heat`: the day's own heat as the entry; "
            "`heat_up`, `heat_down`: that heat scaled either way. `in_the_fit`: the same cell "
            "with only the day's date left out (leave-date-out, the pooled entry on every one "
            "of its rows), which is the level the constants reach when they have seen the "
            "cell. The floor no heat source moves is about 25 mbar on the formula cars and 45 "
            "on the Inferno 86 (#208, finding 5).\n\n"
            + md_table(shown, floats="{:.3g}")
            + "\n\n**Signed bias** (the mean of prediction less reading, mbar).\n\n"
            + md_table(bias, floats="{:.3g}"),
        )
    )


def _lap_time_detail(chain: pd.DataFrame) -> str:
    """With nothing given, each side's lap-time error in per cent and in
    mbar: the simulation's, the ceiling's and another day's, the mbar
    being the error times what 1 % of lap time is worth in the cell (half
    the distance between the simulation's lap time moved up and down)."""
    wide = chain.pivot_table(index=_STINT, columns="arm", values="signed", aggfunc="first")
    if not {"lap_up", "lap_down"} <= set(wide.columns):
        return ""
    worth = ((wide["lap_up"] - wide["lap_down"]).abs() / (2.0 * 100.0 * LAP_TIME_PLANT)).rename(
        "mbar_per_pct"
    )
    cell = ["raw_car", "track"]
    per = worth.reset_index().groupby(cell, sort=True)["mbar_per_pct"].median().reset_index()
    gaps = by_cell(chain)
    rows = []
    for rec in per.to_dict("records"):
        grp = gaps[(gaps["raw_car"] == rec["raw_car"]) & (gaps["track"] == rec["track"])]
        row: dict[str, Any] = {str(k): v for k, v in rec.items()}
        for arm in ("sim", "ceiling", "other_day"):
            own = grp[grp["arm"] == arm]
            if own.empty:
                continue
            pct = 100.0 * float(own["gap"].median())
            row[f"{arm}_pct"] = pct
            row[f"{arm}_mbar"] = pct * float(rec["mbar_per_pct"])
        rows.append(row)
    if not rows:
        return ""
    return (
        "\n\n**The lap time each side supplies.** `_pct`: the size of the log of the side's "
        "lap time over the day's, in per cent (the median over days); `mbar_per_pct`: what 1 % "
        "of lap time is worth in the cell at flying lap 5, the four tyres' median; `_mbar`: the "
        "two multiplied.\n\n" + md_table(pd.DataFrame(rows), floats="{:.3g}")
    )


def report_simulation(
    rep: GateReport, main: pd.DataFrame, sims: pd.DataFrame, stints: pd.DataFrame
) -> None:
    """Clauses 3 to 6: the simulation, its controls and what it replaces."""
    if main.empty or sims.empty:
        return
    neither = set(
        zip(sims.loc[sims["donor_laps"] > 0, "raw_car"], sims.loc[sims["donor_laps"] > 0, "track"])
    )
    rep.items.append(
        GateItem(
            "The simulated side of each cell: the track, the fed use and the frame factor",
            None,
            "; ".join(
                f"{r['raw_car']} at {r['track']}: fed use {r['fed_use']:.3f} (typical "
                f"{r['typical_use']:.3f}), track from {r['track_from']}"
                for r in sims.to_dict("records")
            ),
            "reported",
            "`track_from`: the cars whose laps the track is built from (`donor_laps` on "
            "`donor_days`; a day is driven on a track with no lap of its own date in it, "
            "`fewest_donor_laps` being the smallest such track), or the cell's own track where "
            "no other class has three days and thirty laps there (only the car is then unseen). `fed_use`: the use of a session's "
            "median flying lap, the median over the raw car's sessions at each other layout and "
            "then over those layouts (`fed_from`; `fed_spread` the largest less the smallest), "
            "read with the car fitted without the layout; `typical_use` is the use of a day's "
            "best lap. `factor`: production's heat over the tyres'-frame heat per tyre (fl / fr "
            "/ rl / rr) by kind of source, from the car's build laps at its other layouts under "
            "the fold's shares.\n\n" + md_table(sims, floats="{:.4g}"),
        )
    )
    arms = ("sim", "low_power", "other_car", "heat_up", "heat_down", "lap_up", "lap_down")
    for form, clause in (("heat", 4), ("chain", 5)):
        own = main[main["form"] == form]
        att = attain(own, "sim")
        if att.empty:
            continue
        att["neither"] = [(r, t) in neither for r, t in zip(att["raw_car"], att["track"])]
        got_n, got_c = tally(att[att["neither"]]), tally(att[~att["neither"]])
        table = _arm_table(main, arms + ("ceiling", "other_layouts", "fallback"), form)
        # cells in which the simulation is nearer the day than an arm, and further:
        # on the stints the two and the yardstick all have
        paired = own.merge(own.loc[own["arm"] == "other_day", _STINT].drop_duplicates(), on=_STINT)
        won = []
        for arm in table["arm"]:
            both = attain(paired, arm, beside="sim") if arm != "sim" else pd.DataFrame()
            both = both[both["scored"]] if len(both) else both
            won.append(
                f"{int((both['beside'] < both['size']).sum())} / "
                f"{int((both['beside'] > both['size']).sum())}"
                if len(both)
                else ""
            )
        table["sim_nearer / further"] = won
        # the rows: the yardstick's stints, and those the simulation has of them
        yard = by_cell(own)
        yard = yard[yard["arm"] == "other_day"][_CELL + ["stints", "days"]].rename(
            columns={"stints": "yardstick_stints", "days": "yardstick_days"}
        )
        att = att.merge(yard, on=_CELL, how="left")
        extra = ""
        if form == "chain":
            extra = _lap_time_detail(main[main["form"] == form])
        flagged = main[(main["form"] == form) & (main["arm"] == "sim")]
        status = flagged["status"].value_counts().to_dict()
        shown = att.copy()
        shown["neither_seen"] = shown.pop("neither")
        detail = (
            f"**{FORM_NAMES[form].capitalize()}.** A cell is a raw car at a track on one tyre; "
            "it passes when the simulation is no further from the day's recommendation than "
            "another day is. `size`, `signed`: the simulation's distance in mbar and its signed "
            "median (above zero: a higher cold pressure, so less heat); `beside`: another day's; "
            "`gap`: the size of the log of the simulation's lap time over the day's (zero where "
            "the lap time is given). Both sides are on the stints both have: `stints` of the "
            "yardstick's `yardstick_stints` (the simulation has no entry for a day whose kind "
            "of source has no frame factor at the car's other layouts).\n\n"
            + md_table(
                shown[
                    ["raw_car", "track", "corner", "neither_seen", "days", "stints"]
                    + ["yardstick_stints", "size", "signed", "beside", "gap", "gap_signed"]
                    + ["flagged", "scored", "passed"]
                ],
                floats="{:.3g}",
            )
            + "\n\n**Every arm on the same rows.** `sim_nearer / further`: the scored cells in "
            "which the simulation is nearer the day than the arm, and further.\n\n"
            + md_table(table, floats="{:.3g}")
            + extra
            + f"\n\nThe simulation's stint rows by how the table answered: {status}."
        )
        rep.items.append(
            GateItem(
                f"The simulation, {FORM_NAMES[form]} (#208, clause {clause})",
                None,
                f"neither the car nor the track seen: {got_n['passed']} of {got_n['scored']} "
                f"cells ({got_n['whole']} of {got_n['groups']} whole), {got_n['size']:.1f} mbar "
                f"from the day where another day is {got_n['beside']:.1f}; only the car unseen: "
                f"{got_c['passed']} of {got_c['scored']} ({got_c['size']:.1f} against "
                f"{got_c['beside']:.1f})",
                "a score in gate C3's form (owner, 2026-11-24): no fixed limit in bar",
                detail,
            )
        )
    # the cells with too few days for a pair of days to mean anything
    listed = []
    for form in FORMS:
        cells = by_cell(main[main["form"] == form], beside="sim")
        own = cells[cells["arm"].isin(["sim", "ceiling"]) & (cells["days"] < MIN_DAYS)]
        listed.append(own.assign(form_=form))
    few = pd.concat(listed, ignore_index=True) if listed else pd.DataFrame()
    if len(few):
        rep.items.append(
            GateItem(
                f"The listed cells: fewer than {MIN_DAYS} dry days, no yardstick",
                None,
                "; ".join(
                    f"{car} at {track}: {grp['size'].median():.1f} mbar"
                    for (car, track), grp in few[
                        (few["arm"] == "sim") & (few["form_"] == "heat")
                    ].groupby(["raw_car", "track"], sort=True)
                )
                + " from the day at a given lap time (the four tyres' median)",
                "listed, not scored",
                "The simulation's and the ceiling's distance from the day's own recommendation "
                "(mbar; `gap` the size of the log of the entry's lap time over the day's) where "
                "the raw car has too few days at the track for another day to stand beside "
                "it.\n\n"
                + md_table(
                    few[
                        ["form_", "raw_car", "track", "corner", "arm", "days", "stints", "size"]
                        + ["signed", "gap"]
                    ],
                    floats="{:.3g}",
                ),
            )
        )
    # the entry as the calculators read it, and N = 10
    heat = by_cell(main[main["form"] == "heat"], beside="sim")
    read = heat[heat["arm"].isin(["sim", "sim_9", "sim_direct"])]
    if len(read):
        wide = read.pivot_table(
            index=["raw_car", "track", "corner"], columns="arm", values="signed"
        )
        if {"sim", "sim_direct"} <= set(wide.columns):
            wide["knots_less_direct"] = wide["sim"] - wide["sim_direct"]
        if {"sim_9", "sim_direct"} <= set(wide.columns):
            wide["nine_less_direct"] = wide["sim_9"] - wide["sim_direct"]
        cols = [c for c in ("knots_less_direct", "nine_less_direct") if c in wide.columns]
        rep.items.append(
            GateItem(
                "The entry as the calculators read it: a linear read of the curve against log-log",
                None,
                "; ".join(f"{c} up to {wide[c].abs().max():.2f} mbar" for c in cols),
                "reported: whether the ports take more nodes is decided on this",
                "At the day's lap time, the simulation's signed distance read three ways: `sim` "
                "(the entry's curve is the table's knots, read linearly as the calculators do), "
                "`sim_9` (nine nodes from 0.75 to 1.5 × the entry's lap time, the grid of a "
                "logged curve) and `sim_direct` (the table read log-log, no entry curve).\n\n"
                + md_table(wide.reset_index(), floats="{:.3g}"),
            )
        )
    ten = stints[stints["n"] == pr.FLYING_LAPS[1]]
    if len(ten):
        t = pd.concat(
            [_arm_table(ten, ("sim", "ceiling"), form).assign(form=form) for form in FORMS],
            ignore_index=True,
        )
        if len(t):
            rep.items.append(
                GateItem(
                    f"Flying lap {pr.FLYING_LAPS[1]}",
                    None,
                    "; ".join(
                        f"{r['form']} {r['arm']} {r['cells']} cells, {r['from_the_day_mbar']:.1f} "
                        f"mbar against {r['another_day_mbar']:.1f}"
                        for r in t.to_dict("records")
                    ),
                    "reported beside",
                    md_table(t, floats="{:.3g}"),
                )
            )


# ------------------------------------- the standing holdout, on simulated heat


#: The calculators of the standing holdout: the current one, and the
#: simulation-based one as a precomputed pace table, as a lap driven
#: directly (with the powertrain's own shift point and gears, and with
#: the day's logged ones), and as it is served (:mod:`.served`: the
#: table's heat with the cell's logged out-lap).
HOLDOUT_ARMS = ("calculator", "table", "direct", "direct_setup", "served")
ARM_NAMES = {
    "calculator": "the current calculator",
    "table": "simulation, precomputed pace table",
    "direct": "simulation, driven directly",
    "direct_setup": "simulation, driven directly with the day's shift point and gears",
    "served": "simulation, as served (the table with the cell's logged out-lap, as shipped)",
}
#: The rules the ``served`` arm places its entry by (:mod:`.served`); the
#: committee's (``served.COMMITTEE``) or the shipping ones.
SERVED_RULES_DEFAULT = "shipping"


def heats_in_tyres_frame() -> bool:
    """Is the heat handed out now in the tyres' frame (:mod:`..heat_frame`)?
    Then a simulated lap is already in the logged laps' frame."""
    from ..heat_frame import TYRES, current

    return current() == TYRES


def unit_factors(heats: pd.DataFrame) -> dict[str, np.ndarray]:
    """A frame factor of 1 for every kind of source (and none named): the
    factors of a cell whose logged heat is in the tyres' frame (#376)."""
    kinds = {"", "imu", "gps", "mixed"}
    if len(heats) and "source" in heats.columns:
        kinds |= {str(k) for k in heats["source"].unique()}
    return {k: np.ones(len(CORNERS)) for k in sorted(kinds)}


def served_factor(
    model: dict[str, Any], factors: dict[str, np.ndarray], source: str
) -> np.ndarray | None:
    """The frame factor a simulated entry is served with under ``model``:
    1 on every tyre when the model's heat is in the tyres' frame (#376: the
    simulated lap is in that frame, and the session's kind of sensor is read
    by nothing), else the cell's factor for the session's kind of source
    (None where it has none)."""
    from ..heat_frame import TYRES, frame_of

    if frame_of(model) == TYRES:
        return np.ones(len(CORNERS))
    return factors.get(source)


def holdout_rows(
    prod: Production,
    sim: Sim,
    ids: list[str],
    labels: dict[str, str],
    cache_dir: Path | None,
    heat_frame: str | None = None,
    served_rules: str = SERVED_RULES_DEFAULT,
    served_steps: tuple[str, ...] = (),
    arms: tuple[str, ...] = HOLDOUT_ARMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One fold of production's holdout scored on one fit by every arm of
    :data:`HOLDOUT_ARMS`: ``(rows, sessions)``. ``rows`` are
    ``validate._score_fold_laps``'s (every scored lap and tyre, calculator
    inputs) with ``arm``. ``calculator`` is the fold model as it stands,
    the standing holdout's own rows. The others are the same fit with the
    session's cell and condition answered by the simulated stack, **in
    which the fold's held-out sessions are out of every model**
    (:func:`seen_cell`): ``table`` reads a pace table's knots at the lap
    time typed in, the entry's own lap being that of the fed use;
    ``direct`` drives the lap of the typed lap time in the session's air
    (:meth:`SimCell.direct_entry`), ``direct_setup`` with the day's logged
    shift point and gears as well. A wet or damp session is the same car
    at a lower use on the dry track (gate D2: no wet term). ``sessions``
    says per held-out session which arms have an entry and, where one has
    none, why.

    With ``heat_frame="tyres"`` the fold is fitted on heat in the tyres'
    frame (#376), its logged entries are of the same series, and every
    simulated entry is served with a frame factor of 1: the sensor kind of
    a session (``source``, still reported) is read by nothing. The frame in
    force must be the fold's (``sim``'s logged laps were read in it; the
    worker sets both, :func:`_fold_job`): a mixed run is an error."""
    from ..heat_frame import HeatFrameError, current, frame_of
    from ..sim_entries import answer
    from ..validate import _prepare_fold_laps, _score_fold_laps

    for name in (served_rules, *served_steps):
        srv.RULES[name]  # an unknown name fails before the fold is fitted
    model = fold_model(prod.root, ids, cache_dir, heat_frame=heat_frame)
    if frame_of(model) != current():
        raise HeatFrameError(
            f"a fold fitted in the {frame_of(model)} frame scored with heat in the {current()} "
            "frame: run it inside heat_frame.using(<the fold's frame>)"
        )
    prep = _prepare_fold_laps(prod.root, ids, model, inferred_labels=labels)
    if prep is None:
        return pd.DataFrame(), pd.DataFrame()
    shares = ho.shares_of(model)
    meta = prod.sessions.set_index(prod.sessions["session_id"].astype(str))
    held = set(ids)
    parts = [_score_fold_laps(prep, model).assign(arm="calculator")]
    kinds = {
        k: v[1]
        for k, v in hg.sensor_lap_bins(
            prod.root, prep.flying[["session_id", "lap_num", "car"]]
        ).items()
    }
    cells: dict[tuple[str, str], SimCell | None] = {}
    tracks: dict[str, trk.TrackModel] = {}
    memo: dict = {}
    notes = []
    for sid, own in prep.flying.groupby("session_id", sort=True):
        sid = str(sid)
        pooled, track = str(own["car"].iloc[0]), str(own["track_canonical"].iloc[0])
        cond = str(own["condition"].iloc[0])
        cond = "dry" if cond == "unknown" else cond
        raw_car, date = str(meta.loc[sid, "car"]), str(meta.loc[sid, "date"])
        layout_id = str(meta.loc[sid, "layout_id"])
        typed = float(prep.session_lap_time.get(sid, float("nan")))
        seen = [kinds.get((sid, int(n)), "") for n in own["lap_num"]]
        source = _majority([k for k in seen if k])
        note: dict[str, Any] = {"session_id": sid, "car": pooled, "raw_car": raw_car}
        note.update(track=track, condition=cond, source=source, why="")
        entries: dict[str, pr.Entry | None] = {}
        served_models: dict[str, dict[str, Any]] = {}
        if pooled not in shares or layout_id not in sim.tracks:
            note["why"] = "no track or no car model"
        else:
            key = (raw_car, layout_id)
            if key not in cells:
                if layout_id not in tracks:
                    tracks[layout_id] = track_without(sim, layout_id, held)
                cells[key] = seen_cell(
                    sim, raw_car, layout_id, shares[pooled], held, tracks[layout_id], memo
                )
            cell = cells[key]
            factor = None if cell is None else served_factor(model, cell.factors, source)
            if cell is None:
                note["why"] = "no car model"
            elif factor is None:
                note["why"] = f"no frame factor for a {source or 'mixed'} session"
            else:
                src = cell.source("sim", cm.powertrain_of(raw_car, date), date)
                use = cell.use("sim", raw_car)
                if src is None:
                    note["why"] = "no table for the day's powertrain"
                elif not np.isfinite(use):
                    note["why"] = "no fed use"
                else:
                    entries["table"] = src.entry(factor, use=use)
                    # what ships: the cell served in the dry and in the session's condition
                    for name in (served_rules, *served_steps):
                        got, how = srv.serve_cell(
                            model, src, factor, track, raw_car, use, ("dry", cond), srv.RULES[name]
                        )
                        # what ships for the session, served or not in its condition (rule (c))
                        note[f"served_{name}"] = how.get(cond, "")
                        if how.get(cond) == srv.SIM:
                            heat = answer(got, "heat", track, raw_car, cond)
                            if heat != srv.SIM:  # the scorer reads the session's raw car first
                                raise RuntimeError(
                                    f"{sid}: the served cell answers {heat} for {raw_car}, not sim"
                                )
                        served_models["served" if name == served_rules else f"served:{name}"] = got
                    note["use"] = use
                    note["table_status"] = src.status(typed)
                if src is not None and np.isfinite(typed) and "direct" in arms:
                    entries["direct"] = cell.direct_entry(raw_car, date, sid, typed, factor)
                if src is not None and np.isfinite(typed) and "direct_setup" in arms:
                    entries["direct_setup"] = cell.direct_entry(
                        raw_car, date, sid, typed, factor, setup=True
                    )
        for arm in HOLDOUT_ARMS[1:]:
            if arm not in arms:
                continue
            entry = entries.get(arm)
            if arm == "served":
                note[arm] = arm in served_models
                if arm in served_models:
                    parts.append(
                        _score_fold_laps(prep, served_models[arm], session_ids=[sid]).assign(
                            arm=arm
                        )
                    )
                continue
            note[arm] = entry is not None
            if entry is not None:
                placed = pr.place(model, track, pooled, cond, entry)
                parts.append(_score_fold_laps(prep, placed, session_ids=[sid]).assign(arm=arm))
        for arm, placed in served_models.items():
            if arm != "served":  # a step between the committee's rules and the shipping ones
                parts.append(_score_fold_laps(prep, placed, session_ids=[sid]).assign(arm=arm))
        notes.append(note)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(notes)


def _common(rows: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    """The rows of ``arms`` on the lap × tyre keys every one of them has."""
    keys = ["session_id", "lap_num", "corner", "fold"]
    ok = rows[rows["resid_bar"].notna() & rows["arm"].isin(arms)]
    have = None
    for arm in arms:
        mine = ok.loc[ok["arm"] == arm, keys].drop_duplicates()
        have = mine if have is None else have.merge(mine, on=keys)
    return ok.merge(have, on=keys) if have is not None else ok


def mae_table(
    rows: pd.DataFrame, by: list[str], arms: tuple[str, ...] = HOLDOUT_ARMS
) -> pd.DataFrame:
    """Hot-pressure MAE and bias (bar) per tyre for each of ``arms``
    (:func:`holdout_rows`), grouped by ``by``, on the lap × tyre rows
    every one of them has: ``<arm>_mae`` and ``<arm>_bias``."""
    ok = _common(rows, arms)
    out = []
    for key, grp in ok.groupby(by, sort=True) if by else [((), ok)]:
        key = key if isinstance(key, tuple) else (key,)
        for corner, tyre in list(grp.groupby("corner", sort=False)) + [("all", grp)]:
            stat = tyre.groupby("arm")["resid_bar"].agg(
                mae=lambda x: float(x.abs().mean()), bias="mean", rows="size"
            )
            if not set(arms) <= set(stat.index):
                continue
            row: dict[str, Any] = {**dict(zip(by, key)), "tyre": corner}
            row["rows"] = int(stat.loc[arms[0], "rows"])
            row.update({f"{arm}_mae": float(stat.loc[arm, "mae"]) for arm in arms})
            row.update({f"{arm}_bias": float(stat.loc[arm, "bias"]) for arm in arms})
            out.append(row)
    return pd.DataFrame(out)


def mae_by_pace(
    rows: pd.DataFrame,
    arms: tuple[str, ...] = HOLDOUT_ARMS,
    levels: tuple[float, ...] = (0.0, *ch.PACE_LEVELS),
    by: tuple[str, ...] = (),
) -> pd.DataFrame:
    """The holdout's MAE at levels of pace, as gate C3 reads its
    measures: per level the laps above that percentile of pace among the
    held-out laps of the same raw car, track and condition
    (:func:`~.car_holdout.fast_laps`; level 0 is every lap), and on their
    lap × tyre rows each arm's MAE and bias (bar), with the laps and rows
    counted. ``rows`` are :func:`holdout_rows`'s with ``raw_car`` and the
    lap's ``lap_time_s``; ``by`` groups further (the pooled car, say)."""
    ok = _common(rows, arms)
    ok = ok[np.isfinite(ok["lap_time_s"])]
    laps = ok.drop_duplicates(["session_id", "lap_num"])
    out = []
    for level in levels:
        fast = []
        for _, cell in laps.groupby(["raw_car", "track", "condition"], sort=False):
            keep = ch.fast_laps(cell["lap_time_s"].to_numpy(float), level) if level else None
            fast.append(cell if keep is None else cell[keep])
        chosen = pd.concat(fast)[["session_id", "lap_num"]] if fast else laps.iloc[:0]
        own = ok.merge(chosen, on=["session_id", "lap_num"])
        for key, grp in own.groupby(list(by), sort=True) if by else [((), own)]:
            key = key if isinstance(key, tuple) else (key,)
            stat = grp.groupby("arm")["resid_bar"].agg(
                mae=lambda x: float(x.abs().mean()), bias="mean"
            )
            if not set(arms) <= set(stat.index):
                continue
            row: dict[str, Any] = {**dict(zip(by, key)), "level": f"P{level:g}" if level else "all"}
            row["laps"] = int(grp.drop_duplicates(["session_id", "lap_num"]).shape[0])
            row["rows"] = int(len(grp) // len(arms))
            row.update({f"{arm}_mae": float(stat.loc[arm, "mae"]) for arm in arms})
            row.update({f"{arm}_bias": float(stat.loc[arm, "bias"]) for arm in arms})
            out.append(row)
    return pd.DataFrame(out)


def by_margin(rows: pd.DataFrame, arms: tuple[str, ...] = HOLDOUT_ARMS) -> pd.DataFrame:
    """A holdout by pace by how far under the training pace a session's
    lap time is (``rows`` with ``under_typed`` and ``under_lap``,
    :func:`fastest_sessions`): for every session, those under every lap
    left in training, and those more than 2 % and 3 % under the fastest
    typed lap time left, each over every cell and per cell, the arms' MAE
    and bias (bar) and each simulated arm's per-session difference from
    the first arm."""
    picks = (
        ("every held-out session", rows["under_typed"].notna()),
        ("under every lap left in training", rows["under_lap"] < 0),
        ("more than 2 % under the training pace", rows["under_typed"] < -0.02),
        ("more than 3 % under the training pace", rows["under_typed"] < -0.03),
    )
    out = []
    for label, mask in picks:
        own = rows[mask]
        groups = [("every cell", own)]
        if label == picks[0][0]:
            groups += [(f"{c} at {t}", g) for (c, t), g in own.groupby(["raw_car", "track"])]
        for cell, grp in groups:
            table = mae_table(grp, [], arms)
            if table.empty:
                continue
            top = table[table["tyre"] == "all"].iloc[0]
            row: dict[str, Any] = {"sessions_of": label, "cell": cell}
            row["sessions"] = int(_common(grp, arms)["session_id"].nunique())
            row["rows"] = int(top["rows"])
            row.update({f"{a}_mae": float(top[f"{a}_mae"]) for a in arms})
            row.update({f"{a}_bias": float(top[f"{a}_bias"]) for a in arms})
            for arm in arms[1:]:
                pair = paired_by_session(grp, arm, base=arms[0])
                row[f"{arm}_less_{arms[0]}"] = (
                    f"{1000 * pair['mean']:+.1f} ± {1000 * pair['se']:.1f} "
                    f"({pair['nearer']} of {pair['sessions']})"
                )
            out.append(row)
    return pd.DataFrame(out)


def pace_margins(prod: Production, fractions: tuple[float, ...]) -> pd.DataFrame:
    """What a holdout by pace asks for at each fraction, before any fit:
    per raw car and track the sessions held out and left, how far under
    the fastest typed lap time left the held-out sessions are (median and
    furthest, per cent) and how many are under every single lap left."""
    out = []
    for fraction in fractions:
        _, m = fastest_sessions(prod, fraction)
        if m.empty:
            continue
        groups = [("every cell", m)] + [
            (f"{c} at {t}", g) for (c, t), g in m.groupby(["raw_car", "track"], sort=True)
        ]
        for cell, grp in groups:
            out.append(
                {
                    "fastest": f"{fraction:.0%}",
                    "cell": cell,
                    "held_out": int(len(grp)),
                    "left": int(grp["sessions_left"].iloc[0]) if cell != "every cell" else None,
                    "under_typed_median_pct": float(-100.0 * grp["under_typed"].median()),
                    "under_typed_furthest_pct": float(-100.0 * grp["under_typed"].min()),
                    "under_every_lap_left": int((grp["under_lap"] < 0).sum()),
                }
            )
    return pd.DataFrame(out)


def check_pace_margins(
    root: Path, fractions: tuple[float, ...] = (0.25, 0.5, 0.8, 0.9)
) -> GateReport:
    """`lapsim pressure-pace-margins`: :func:`pace_margins`, to choose a
    fraction for `check-pressure-holdout --fastest` (no fit is made)."""
    rep = GateReport("FHM", "what a holdout by pace asks for (`lapsim pressure-pace-margins`)")
    table = pace_margins(load_production(root), fractions)
    whole = table[table["cell"] == "every cell"] if len(table) else table
    rep.items.append(
        GateItem(
            "How far under the training pace the held-out sessions are",
            None,
            "; ".join(
                f"{r['fastest']}: {r['held_out']} sessions, {r['under_typed_median_pct']:.1f} % "
                f"under in the median, {r['under_every_lap_left']} under every lap left"
                for r in whole.to_dict("records")
            ),
            "reported",
            "Per raw car and track with ten dry sessions in a fit, the fastest fraction of them "
            "by the lap time a driver would type is held out. `under_typed_*`: how far under "
            "the fastest typed lap time left in training the held-out sessions' own are; "
            "`under_every_lap_left`: those under every single flying lap left at the cell.\n\n"
            + md_table(table, floats="{:.2f}"),
        )
    )
    return rep


def compare_holdouts(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> GateReport:
    """Two runs of the holdout on the same folds, row for row
    (`lapsim compare-pressure-holdouts`): what a change to the fit does
    to each calculator (the cars fitted apart against pooled, say). Per
    raw car, per raw car and condition and over every car, each arm's MAE
    and bias in the two runs on the rows both have, and per raw car the
    change session by session."""
    rep = GateReport("FHC", f"two holdouts on the same rows: {names[0]} against {names[1]}")
    keys = ["session_id", "lap_num", "corner", "fold", "arm"]
    extra = [c for c in ("raw_car", "condition") if c in a.columns]
    both = a[keys + extra + ["resid_bar"]].merge(
        b[keys + ["resid_bar"]], on=keys, suffixes=("_a", "_b")
    )
    both = both.dropna(subset=["resid_bar_a", "resid_bar_b"])
    rep.notes.append(
        f"{len(both)} lap × tyre × arm rows in both runs, of {len(a)} and {len(b)}. `a` is "
        f"{names[0]}, `b` is {names[1]}; MAE and bias in bar."
    )
    if both.empty:
        rep.items.append(GateItem("Rows in both runs", False, "none", "at least one"))
        return rep
    both = both.assign(size_a=both["resid_bar_a"].abs(), size_b=both["resid_bar_b"].abs())

    def _table(by: list[str]) -> pd.DataFrame:
        got = both.groupby(by, sort=True).agg(
            rows=("size_a", "size"),
            a_mae=("size_a", "mean"),
            b_mae=("size_b", "mean"),
            a_bias=("resid_bar_a", "mean"),
            b_bias=("resid_bar_b", "mean"),
        )
        got["b_less_a_mbar"] = 1000.0 * (got["b_mae"] - got["a_mae"])
        return got.reset_index()

    paired = []
    for (car, arm), grp in both.groupby(["raw_car", "arm"], sort=True):
        per = (grp["size_b"] - grp["size_a"]).groupby(grp["session_id"]).mean()
        paired.append(
            {
                "raw_car": car,
                "arm": arm,
                "sessions": int(len(per)),
                "b_less_a_mbar": float(1000.0 * per.mean()),
                "se_mbar": (
                    float(1000.0 * per.std(ddof=1) / np.sqrt(len(per)))
                    if len(per) > 1
                    else float("nan")
                ),
                "b_better_on": int((per < 0).sum()),
            }
        )
    whole = _table(["arm"])
    rep.items.append(
        GateItem(
            f"Each calculator in the two runs: {names[1]} less {names[0]}",
            None,
            "; ".join(
                f"{r['arm']} {r['a_mae']:.4f} → {r['b_mae']:.4f} bar"
                for r in whole.to_dict("records")
            ),
            "reported",
            "**Every car.**\n\n"
            + md_table(whole, floats="{:.4f}")
            + "\n\n**Per raw car.**\n\n"
            + md_table(_table(["raw_car", "arm"]), floats="{:.4f}")
            + "\n\n**Per raw car and condition.**\n\n"
            + md_table(_table(["raw_car", "condition", "arm"]), floats="{:.4f}")
            + "\n\n**Session by session** (the mean over a raw car's sessions of the change in "
            "its mean absolute residual, and its standard error).\n\n"
            + md_table(pd.DataFrame(paired), floats="{:.2f}"),
        )
    )
    return rep


def paired_by_session(rows: pd.DataFrame, arm: str, base: str = "calculator") -> dict:
    """``arm``'s mean absolute residual less ``base``'s, per held-out
    session on the rows both have: its mean over sessions, the standard
    error of that mean and the sessions on which ``arm`` is nearer the
    TPMS (bar)."""
    keys = ["session_id", "lap_num", "corner", "fold"]
    ok = _common(rows, (arm, base)).assign(size=lambda d: d["resid_bar"].abs())
    wide = ok.pivot_table(index=keys, columns="arm", values="size", aggfunc="first").dropna()
    if wide.empty or not {arm, base} <= set(wide.columns):
        return {"sessions": 0, "mean": float("nan"), "se": float("nan"), "nearer": 0}
    per = (wide[arm] - wide[base]).groupby(level="session_id").mean()
    return {
        "sessions": int(len(per)),
        "mean": float(per.mean()),
        "se": float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan"),
        "nearer": int((per < 0).sum()),
    }


def fastest_sessions(
    prod: Production, fraction: float, min_sessions: int = 10
) -> tuple[list[tuple[str, list[str]]], pd.DataFrame]:
    """A holdout by pace: per raw car and track with ``min_sessions`` dry
    sessions in a fit, the fastest ``fraction`` of them by the lap time a
    driver would type (the session's 25th-percentile flying lap), so that
    what is asked of a model is a lap time under every session it was
    trained on at the track. ``(folds, margins)``: one fold per cell
    (``"car at track"``, the session ids), and per held-out session its
    typed lap time with ``under_typed`` and ``under_lap``: how far it is
    under the fastest typed lap time and under the fastest single flying
    lap left in training at the cell (log ratios; below zero is faster
    than anything left)."""
    from ..validate import TARGET_LAP_TIME_QUANTILE

    laps = prod.raw_laps[
        prod.raw_laps["tire_usable"].astype(bool)
        & ~prod.raw_laps["is_outlap"].fillna(False).astype(bool)
    ]
    typed = laps.groupby("session_id")["on_track_s"].quantile(TARGET_LAP_TIME_QUANTILE)
    best = laps.groupby("session_id")["lap_time_s"].min()
    own = prod.sessions[prod.sessions["session_id"].astype(str).isin(prod.trained)].copy()
    own["condition"] = own["session_id"].astype(str).map(prod.conditions)
    own = own[own["condition"] == "dry"]
    own = own.assign(typed=own["session_id"].map(typed), best=own["session_id"].map(best))
    own = own[np.isfinite(own["typed"])]
    folds, rows = [], []
    for (car, track), cell in own.groupby(["car", "track_canonical"], sort=True):
        if len(cell) < min_sessions:
            continue
        cell = cell.sort_values(["typed", "session_id"])
        n = max(int(np.ceil(fraction * len(cell))), 1)
        held, left = cell.iloc[:n], cell.iloc[n:]
        folds.append((f"{car} at {track}", [str(x) for x in held["session_id"]]))
        for r in held.itertuples():
            rows.append(
                {
                    "session_id": str(r.session_id),
                    "raw_car": str(car),
                    "track": str(track),
                    "typed_s": float(r.typed),
                    "under_typed": float(np.log(r.typed / left["typed"].min())),
                    "under_lap": float(np.log(r.typed / left["best"].min())),
                    "sessions_left": int(len(left)),
                }
            )
    return folds, pd.DataFrame(rows)


#: What a worker process of the holdout has loaded: the dataset and the
#: lap simulation's side, once, however many folds it scores.
_WORKER: dict = {}


def _fold_job(job: tuple) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """One fold of the holdout in a worker process: ``(fold, rows,
    sessions)`` of :func:`holdout_rows`, every heat (the simulation's logged
    laps among them) in the run's frame."""
    from ..heat_frame import using

    (
        root,
        cache_dir,
        tracks_dir,
        folds_dir,
        fold,
        ids,
        labels,
        split,
        heat_frame,
        rules,
        steps,
        arms,
    ) = job
    split_pooling(tuple(split))
    key = (str(root), str(cache_dir), str(tracks_dir), str(folds_dir), tuple(split), heat_frame)
    with using(heat_frame):
        if _WORKER.get("key") != key:
            _WORKER.update(
                key=key,
                prod=load_production(root),
                sim=load_sim(root, cache_dir=cache_dir, tracks_dir=tracks_dir, folds_dir=folds_dir),
            )
        rows, notes = holdout_rows(
            _WORKER["prod"],
            _WORKER["sim"],
            ids,
            labels,
            cache_dir,
            heat_frame=heat_frame,
            served_rules=rules,
            served_steps=tuple(steps),
            arms=tuple(arms),
        )
    return fold, rows, notes


def check_pressure_holdout(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    folds_dir: Path | None = None,
    n_folds: int = 20,
    n_per_bucket: int = 2,
    min_bucket_size: int = 10,
    workers: int | None = None,
    split: tuple[str, ...] = (),
    fastest: float = 0.0,
    heat_frame: str | None = None,
    served_rules: str = SERVED_RULES_DEFAULT,
    served_steps: tuple[str, ...] = (),
    arms: tuple[str, ...] = HOLDOUT_ARMS,
) -> GateReport:
    """The calculator's own evaluation, MAE over folds of held-out
    sessions (``tire-predict-holdout``: the every-session holdout,
    calculator inputs, hot pressure against the TPMS), for a calculator
    whose entries are simulated laps, beside the calculator as it is on
    the same folds, fits and rows (owner, 2026-11-25: the goal is a
    simulation-based model that replaces the current calculator).

    The folds are independent (a fit, then its scoring), so they run side
    by side: ``workers`` processes (half the machine's cores by default,
    a fit and the solver using about two each; one fold needs about 2.3
    GB), each loading the dataset and the simulation's side once.

    ``split`` names raw cars to fit as a pooled car of their own
    (:func:`split_pooling`): the same folds, with those cars' pressure
    constants, shares and entries apart from the car they are pooled
    with. The rows then go to ``holdout_rows__split.parquet``.

    ``fastest`` replaces the folds by a holdout by pace
    (:func:`fastest_sessions`): per raw car and track the fastest fraction
    of the dry sessions, one fold a cell, so that every model is asked for
    a lap time under what it was trained on there
    (``holdout_rows__fast.parquet``).

    ``heat_frame`` is the frame every fold is fitted in, and the
    calculator's logged entries are built in (None: the frame in force,
    ``tyres`` outside any ``using``): ``tyres`` (#376) serves the simulated
    laps with no frame factor; ``sensor`` fits on the sensor
    verdict's series and brings the simulated laps to it by a frame factor
    per kind of source (for comparison; ``holdout_rows__sensor.parquet``)."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from ..validate import _inferred_compound_labels, _pick_holdout_sessions

    from ..heat_frame import SENSOR, resolve

    heat_frame = resolve(heat_frame)  # concrete: the workers are other processes
    sensor = heat_frame == SENSOR
    # The compound of an unlabelled session is what a driver selects, an input:
    # the EM that stands in for it is run under the shipped pooling whatever
    # ``split`` says, so that both runs are fed the same tyre (#420: the two
    # EMs differ on the KK-F's damp sessions).
    labels = _inferred_compound_labels(root)
    pooling = split_pooling(tuple(split))
    rep = GateReport(
        ("FHP" if fastest else "FHS" if split else "FH") + ("_sensor" if sensor else ""),
        "the standing holdout on simulated heat (`lapsim check-pressure-holdout`)"
        + (f", with {', '.join(split)} fitted apart" if split else "")
        + (f", the fastest {fastest:.0%} of each cell's dry sessions held out" if fastest else "")
        + (", the heat in the sensor's frame and a frame factor" if sensor else ""),
    )
    if split:
        rep.notes.append(
            f"The pooling of the cars in this run: {pooling}. The compound labels of unlabelled "
            "sessions are the shipped pooling's (the same input as the pooled run's)."
        )
    prod = load_production(root)
    jobs = []
    margins = pd.DataFrame()
    if fastest:
        by_pace, margins = fastest_sessions(prod, fastest, min_bucket_size)
        jobs = [
            (
                root,
                cache_dir,
                tracks_dir,
                folds_dir,
                k,
                ids,
                labels,
                tuple(split),
                heat_frame,
                served_rules,
                tuple(served_steps),
                tuple(arms),
            )
            for k, (_, ids) in enumerate(by_pace)
        ]
    for fold in range(max(1, n_folds) if not fastest else 0):
        ids = _pick_holdout_sessions(
            prod.sessions,
            prod.raw_laps,
            n_per_bucket=n_per_bucket,
            min_bucket_size=min_bucket_size,
            fold=fold,
            session_condition=prod.conditions,
        )
        if not ids:
            break
        jobs.append(
            (
                root,
                cache_dir,
                tracks_dir,
                folds_dir,
                fold,
                ids,
                labels,
                tuple(split),
                heat_frame,
                served_rules,
                tuple(served_steps),
                tuple(arms),
            )
        )
    cores = os.cpu_count() or 2
    workers = max(1, min(len(jobs), workers or cores // 2))
    all_rows, all_notes = [], []
    if workers == 1:
        done = map(_fold_job, jobs)
        results = list(done)
    else:
        # fresh interpreters (the solver's compiled threads do not survive a fork),
        # each given its share of the cores
        threads = str(max(1, cores // workers))
        saved = {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")}
        saved["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ.update({k: threads for k in saved})
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                results = list(pool.map(_fold_job, jobs))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    for fold, rows, notes in results:
        if len(rows):
            all_rows.append(rows.assign(fold=fold))
            all_notes.append(notes.assign(fold=fold))
    logger.info("pressure holdout: %d folds scored on %d workers", len(results), workers)
    if not all_rows:
        rep.items.append(GateItem("Held-out sessions", False, "none", "at least one"))
        return rep
    rows = pd.concat(all_rows, ignore_index=True)
    notes = pd.concat(all_notes, ignore_index=True)
    # each row's raw car and its lap's own lap time, for the levels of pace
    lap_time = prod.raw_laps[["session_id", "lap_num", "lap_time_s"]].drop_duplicates(
        ["session_id", "lap_num"]
    )
    rows = rows.merge(lap_time, on=["session_id", "lap_num"], how="left").merge(
        notes[["session_id", "raw_car"]].drop_duplicates("session_id"), on="session_id", how="left"
    )
    if len(margins):
        rows = rows.merge(
            margins[["session_id", "under_typed", "under_lap"]], on="session_id", how="left"
        )
    if cache_dir is not None:
        out = Path(cache_dir) / "pressure_rows"
        out.mkdir(parents=True, exist_ok=True)
        tag = (
            ("__split" if split else "")
            + (f"__fast{round(100 * fastest)}" if fastest else "")
            + ("__sensor" if sensor else "")
            + ("" if served_rules == SERVED_RULES_DEFAULT else f"__served_{served_rules}")
        )
        rows.to_parquet(out / f"holdout_rows{tag}.parquet")
        notes.to_parquet(out / f"holdout_sessions{tag}.parquet")
    own = rows[rows["arm"] == "calculator"]
    rep.notes.append(
        f"{len(results)} folds (of {n_folds} asked for: the buckets run out) of {n_per_bucket} "
        f"sessions per (track, car, condition) bucket (dry buckets of {min_bucket_size} sessions "
        f"or more, rain buckets of 3): {notes['session_id'].nunique()} sessions held out, "
        f"{int(own['resid_bar'].notna().sum())} lap × tyre rows with a TPMS reading. Calculator "
        "inputs: the pit-exit temperature, N laps of the session's 25th-percentile lap time, "
        "the compound a driver would have selected. Each fold is fitted once. `calculator` is "
        "that fit as it stands (the standing holdout, row for row). The simulation-based "
        "calculator is the same fit with the cell answered by the simulated stack, and **the "
        "fold's held-out sessions are out of every model in that stack**: the pressure model's "
        "K, tau, kappa, shares and compound multipliers (the fold's fit); the car model, its "
        "limit, the driver's braking use and typical use; the track's curvature; the frame "
        "factor; the fed use. What a fold does not take out: which laps of the dataset count "
        "as build laps (the lap index judges pace against every session), the track's "
        "reference line, bank, grade and pit exit, the day's own powertrain and (for "
        "`direct_setup` only) its logged shift point and gears, which are inputs, and the "
        "compound label of an unlabelled session, which is the standing holdout's own (an EM "
        "on the full dataset). `table`: a precomputed pace table read at the typed lap time. "
        "`direct`: the lap of the typed lap time driven for the asking, in the session's air."
    )
    if len(margins):
        per = (
            margins.groupby(["raw_car", "track"], sort=True)
            .agg(
                held_out=("session_id", "size"),
                left=("sessions_left", "first"),
                typed_s=("typed_s", "median"),
                under_typed=("under_typed", "median"),
                under_lap=("under_lap", "median"),
                under_every_lap=("under_lap", lambda x: int((x < 0).sum())),
            )
            .reset_index()
        )
        rep.items.append(
            GateItem(
                "The sessions held out by pace",
                None,
                f"{len(margins)} sessions in {len(per)} cells, their typed lap time "
                f"{-100 * margins['under_typed'].median():.1f} % under the fastest typed lap time "
                f"left in training (median); {int((margins['under_lap'] < 0).sum())} of them under "
                "every single lap left in training at the cell",
                "reported: a lap time the models were not trained on",
                "`under_typed`, `under_lap`: the log of a held-out session's typed lap time over "
                "the fastest typed lap time, and over the fastest single flying lap, left in "
                "training at the cell (the cell's median; below zero is faster than anything "
                "left); `under_every_lap`: the held-out sessions whose typed lap time is under "
                "every lap left.\n\n" + md_table(per, floats="{:.4f}"),
            )
        )
    whole = float(own["resid_bar"].abs().mean())
    rep.items.append(
        GateItem(
            "The calculator's rows are the standing holdout's",
            None,
            f"{int(own['resid_bar'].notna().sum())} rows, pooled MAE {whole:.4f} bar",
            "reported: `tire-predict-holdout --n-folds` prints the same",
        )
    )
    sims = tuple(a for a in HOLDOUT_ARMS[1:] if a in arms)
    cover = {arm: int(notes[arm].sum()) for arm in sims if arm in notes.columns}
    missing = notes[~notes[[a for a in sims if a in notes.columns]].all(axis=1)]
    rep.items.append(
        GateItem(
            "Sessions each simulation-based calculator has an answer for",
            None,
            "; ".join(f"{arm} {n} of {len(notes)}" for arm, n in cover.items())
            + (
                "; without a frame factor or a car: "
                + "; ".join(f"{n} {why}" for why, n in missing["why"].value_counts().items() if why)
                if len(missing)
                else ""
            ),
            "reported: a replacement has to answer every session; the tables below are on "
            "the rows every arm has",
            (
                md_table(
                    missing[
                        ["session_id", "raw_car", "track", "condition", "source", "why", *cover]
                    ]
                )
                if len(missing)
                else ""
            ),
        )
    )
    arms = tuple(a for a in HOLDOUT_ARMS if (rows["arm"] == a).any())
    pooled = mae_table(rows, [], arms)
    by_car = mae_table(rows, ["car"], arms)
    by_cond = mae_table(rows, ["condition"], arms)
    by_cell = mae_table(rows, ["car", "track", "condition"], arms)
    by_cell = by_cell[by_cell["tyre"] == "all"] if len(by_cell) else by_cell
    if pooled.empty:
        return rep
    top = pooled[pooled["tyre"] == "all"].iloc[0]
    pairs = {arm: paired_by_session(rows, arm) for arm in arms[1:]}
    rep.items.append(
        GateItem(
            "Hot-pressure MAE over the held-out folds: the simulation-based calculator beside "
            "the current one",
            None,
            f"pooled over {int(top['rows'])} lap × tyre rows: "
            + "; ".join(f"{ARM_NAMES[a]} {top[f'{a}_mae']:.4f} bar" for a in arms)
            + ". Per session, less the current calculator: "
            + "; ".join(
                f"{a} {1000 * p['mean']:+.1f} ± {1000 * p['se']:.1f} mbar, nearer the TPMS on "
                f"{p['nearer']} of {p['sessions']} sessions"
                for a, p in pairs.items()
            ),
            "reported: the calculator's own measure of accuracy",
            "Lap-end hot-pressure residual (predicted less the TPMS reading), bar, on the rows "
            "every arm has.\n\n**Pooled.**\n\n"
            + md_table(pooled, floats="{:.4f}")
            + "\n\n**Per car.**\n\n"
            + md_table(by_car, floats="{:.4f}")
            + "\n\n**Per condition.**\n\n"
            + md_table(by_cond, floats="{:.4f}")
            + "\n\n**Per cell** (all four tyres).\n\n"
            + md_table(by_cell, floats="{:.4f}"),
        )
    )
    paced = mae_by_pace(rows, arms)
    if len(paced):
        dry = mae_by_pace(rows[rows["condition"] == "dry"], arms, by=("car",))
        rep.items.append(
            GateItem(
                "The same holdout at levels of pace",
                None,
                "; ".join(
                    f"{r['level']} ({r['laps']} laps): "
                    + ", ".join(f"{a} {r[f'{a}_mae']:.4f}" for a in arms)
                    for r in paced.to_dict("records")
                ),
                "reported: gate C3's levels (the laps above a percentile of pace)",
                "The rows of the holdout above, kept where the lap is above the level's "
                "percentile of pace among the held-out laps of the same raw car, track and "
                "condition (`P90`: the fastest tenth; `all`: every lap). MAE and bias in bar. "
                "Every arm is fed the same thing on every lap of a session, the session's "
                "25th-percentile lap time, so a fast lap is one driven harder than the plan it "
                "is predicted with.\n\n"
                + md_table(paced, floats="{:.4f}")
                + "\n\n**Dry laps, per car.**\n\n"
                + md_table(dry, floats="{:.4f}"),
            )
        )
    if "under_lap" in rows.columns:
        under = by_margin(rows, arms)
        if len(under):
            rep.items.append(
                GateItem(
                    "By how far the lap time asked for is under the training pace",
                    None,
                    "; ".join(
                        f"{r['sessions_of']} ({r['sessions']} sessions): "
                        + ", ".join(f"{a} {r[f'{a}_mae']:.4f}" for a in arms[:3])
                        for r in under[under["cell"] == "every cell"].to_dict("records")
                    ),
                    "reported: a lap time the models were not trained on at the track",
                    "The same rows, kept where the held-out session's typed lap time is under "
                    "every single lap left in training at its cell, and more than 2 % and 3 % "
                    "under the fastest typed lap time left. MAE and bias in bar on the rows "
                    "every arm has; `<arm>_less_calculator`: the arm's mean absolute residual "
                    "less the current calculator's per session (mbar, mean ± its standard "
                    "error), with the sessions on which the arm is nearer the TPMS.\n\n"
                    + md_table(under, floats="{:.4f}"),
                )
            )
    if {"table", "direct"} <= set(arms):
        keys = ["session_id", "lap_num", "corner", "fold"]
        both = _common(rows, ("table", "direct"))
        wide = both.pivot_table(index=keys, columns="arm", values="P_hot_pred_bar", aggfunc="first")
        gap = 1000.0 * (wide["table"] - wide["direct"]).dropna()
        ab = paired_by_session(rows, "direct", base="table")
        rep.items.append(
            GateItem(
                "The shootout: the precomputed pace table against the lap driven directly",
                None,
                f"their two predictions differ by {gap.abs().median():.1f} mbar in the median "
                f"row and {gap.abs().quantile(0.95):.1f} at the 95th percentile (table less "
                f"direct {gap.median():+.1f} mbar); per session the direct lap's MAE less the "
                f"table's is {1000 * ab['mean']:+.1f} ± {1000 * ab['se']:.1f} mbar, nearer the "
                f"TPMS on {ab['nearer']} of {ab['sessions']} sessions",
                "reported: the owner's call",
                "`table`: the entry's own lap is that of the fed use and the typed lap time is "
                "read along its knots, in standard air, with the powertrain's own shift point "
                "and gears. `direct`: the knob is solved for the typed lap time in the "
                "session's air. The difference between the two predictions is on the same "
                "rows, hot pressure at the lap's end.",
            )
        )
    return rep


__all__ = [
    "FORMS",
    "Held",
    "MIN_DAYS",
    "Production",
    "Sim",
    "SimCell",
    "at_lap",
    "attain",
    "by_cell",
    "by_stint",
    "cell_sessions",
    "cell_table",
    "check_pressure",
    "by_margin",
    "check_pace_margins",
    "check_pressure_holdout",
    "compare_holdouts",
    "pace_margins",
    "fastest_sessions",
    "split_pooling",
    "holdout_rows",
    "mae_by_pace",
    "mae_table",
    "paired_by_session",
    "date_folds",
    "other_day_arms",
    "day_arms",
    "donor_track",
    "fed_laps",
    "first_visits",
    "fold_model",
    "held_of",
    "hold_out",
    "load_production",
    "load_sim",
    "pool_weights",
    "pooled_sim_arms",
    "prepare",
    "score_cell",
    "seen_cell",
    "sim_arms",
    "sim_cell",
    "tally",
    "tpms_table",
    "track_without",
    "without_sessions",
    "held_out_drive",
    "HOLDOUT_ARMS",
    "wet_arms",
]
