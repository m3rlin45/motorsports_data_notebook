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
FORMS = ("heat", "chain")
FORM_NAMES = {"heat": "heat, at a given lap time", "chain": "the whole chain, nothing given"}
#: The simulated arms and how each one's table is built (:func:`~.pace.build_table`).
SIM_ARMS: dict[str, dict] = {
    "sim": {},
    "low_power": {"power_scale": ch.POWER_CONTROL, "use_cap": False, "use_fast": FREE_KNOB},
    "other_car": {"use_cap": False, "use_fast": FREE_KNOB},
}
#: The source files a fold's fit is made by: a cached fold is of their bytes.
_FIT_SOURCES = (
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
)
_FIT_SIDECARS = ("sensor_blacklist.yaml", "tire_compounds.yaml", "session_overrides.yaml")


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


def _fit_signature(root: Path) -> str:
    """What a cached fold is of: the dataset's files, the fit's source and
    the sidecars a fit reads."""
    h = hashlib.sha256(geo._index_signature(root).encode())
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


def fold_model(root: Path, ids: list[str], cache_dir: Path | None = None) -> dict[str, Any]:
    """The pressure model fitted without ``ids``, nothing written under
    the dataset. With ``cache_dir`` the fit is kept there, keyed by the
    sessions left out and :func:`_fit_signature`: a fold is fitted once
    however often its arms are scored."""
    from ..validate import _fit_fold

    path = None
    if cache_dir is not None:
        key = hashlib.sha256(("\n".join(sorted(ids)) + _fit_signature(root)).encode()).hexdigest()
        path = Path(cache_dir) / "pressure_folds" / f"{key[:24]}.json"
        if path.exists():
            model: dict[str, Any] = json.loads(path.read_text())
            return model
    model = _fit_fold(root, ids)
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
    prod: Production, model: dict[str, Any], prep: Any, pooled: str, track: str, left_out: int
) -> Held:
    """The held-out laps of one (pooled car, track) of a fold (:class:`Held`)."""
    share = ho.shares_of(model)[pooled]
    meta = prod.sessions.set_index(prod.sessions["session_id"].astype(str))
    laps = pr.mark_laps(prep, prod.raw_laps)
    laps = laps[(laps["car"] == pooled) & (laps["track_canonical"] == track)].reset_index(drop=True)
    laps["raw_car"] = laps["session_id"].map(meta["car"]).astype(str)
    laps["date"] = laps["session_id"].map(meta["date"]).astype(str)
    laps["layout_id"] = laps["session_id"].map(meta["layout_id"]).astype(str)
    stints = pr.stints_of(prep, prod.targets)
    stints = stints[(stints["car"] == pooled) & (stints["track_canonical"] == track)]
    ordinary = laps[laps["ordinary"] & (laps["condition"] == "dry")]
    keys = ordinary[["session_id", "lap_num", "car"]]
    kinds = {k: v[1] for k, v in hg.sensor_lap_bins(prod.root, keys).items()} if len(keys) else {}
    entries: dict[str, pr.Entry] = {}
    rows = []
    for sid, own in laps[laps["condition"] == "dry"].groupby("session_id", sort=True):
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
            }
        )
    cols = ["session_id", "raw_car", "date", "layout_id", "powertrain", "source", "lap_time_s"]
    days = pd.DataFrame(rows, columns=cols + ["ordinary_laps"])
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

    def track(self, date: str) -> trk.TrackModel:
        """The track a day's lap is driven on: from other classes' laps
        without that date where such a track exists, else the own."""
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
    factors = pr.frame_factors(heats)
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


# ---------------------------------------------------------------- the arms


def _bare(entry: pr.Entry) -> pr.Entry:
    """An entry with no out-lap of its own: the fold model's lookup then
    answers, the same on every side of a comparison at a given lap time."""
    return replace(entry, out_s=None, out_q=None)


def _recommend(
    held: Held, sid: str, entry: pr.Entry | None, lap_time_s: float | None
) -> pd.DataFrame:
    """The recommendations for a session's stints from ``entry`` placed
    in the fold's model (None: the fold model as it stands), at each
    flying lap of :data:`~.pressure.FLYING_LAPS`."""
    model = held.model
    if entry is not None:
        model = pr.place(held.model, held.track, held.pooled, "dry", entry)
    stints = held.stints[(held.stints["session_id"] == sid) & (held.stints["condition"] == "dry")]
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


def day_arms(held: Held, day: dict, cell: SimCell | None, *, seen: bool = False) -> list[dict]:
    """Every arm's entry for one day, in both forms: dicts of ``arm,
    form, entry`` (None: the fold model as it stands), ``lap_time_s``
    (the lap time typed in; None: the entry's own), ``pick`` and, where
    they apply, ``gap`` (the log of a paired day's lap time over the
    day's), ``status`` and ``use``. The form ``tpms`` marks the entries
    pressure against the TPMS is scored with (S2: production's own
    inputs, the entry as it would be written).

    ``seen`` is for a fold that leaves the day's date out and not its
    cell: the one arm is then ``entry``, the fold model as it stands,
    whose entry for the cell was built without the day."""
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
        return out
    typed = float(held.prep.session_lap_time.get(sid, t_day))
    for arm, factor in (("heat_up", 1.0 + HEAT_PLANT), ("heat_down", 1.0 - HEAT_PLANT)):
        out.append(
            {"arm": arm, "form": "heat", "entry": _bare(own.scaled(factor)), "lap_time_s": t_day}
        )
        out.append({"arm": arm, "form": "chain", "entry": own.scaled(factor), "lap_time_s": t_day})
    laps = held.laps[held.laps["condition"] == "dry"]
    pairs = _pairs(held, day)
    # another day, at the day's lap time: up to three sessions nearest in lap time
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
    # another day, given nothing: each other day as if it alone had written the entry
    for k, (date, group) in enumerate(pairs.groupby("date", sort=True)):
        alone = pr.logged_entry(
            root, laps[laps["session_id"].isin(set(group["session_id"]))], held.share, curve=False
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


def score_cell(
    held: Held, cells: dict[str, SimCell | None], *, seen: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(rows, tpms)`` of a fold's cell.

    ``rows``: every arm's distance from the day's own recommendation, per
    day, pit-exit stint, tyre, flying lap and form: ``d_mbar`` (the arm's
    cold pressure less the day's) with ``pooled, track, raw_car, date,
    session_id, stint_id, corner, n, form, arm, pick, gap, status, use``.
    A simulated arm with no lap keeps a row with a NaN distance and the
    status ``no_solution``.

    ``tpms``: pressure against the TPMS (S2), which is reported and never
    ranks: per arm with a ``tpms`` entry, the holdout's own rows at flying
    lap :data:`~.pressure.FLYING_LAP_N` of pit-exit stints (production's
    inputs: the session's 25th-percentile lap time typed in), with
    ``resid_mbar`` and whether the row is ``after_dropped`` or
    ``after_cool_down`` (:func:`~.pressure.mark_laps`).

    ``cells`` is the simulation's side per raw car (None: no simulated
    arm); ``seen`` as in :func:`day_arms`."""
    from ..validate import _score_fold_laps

    rows: list[pd.DataFrame] = []
    tpms: list[pd.DataFrame] = []
    flags = held.laps[["session_id", "lap_num", "after_dropped", "after_cool_down"]]
    for day in held.days.to_dict("records"):
        sid = day["session_id"]
        arms = day_arms(held, day, cells.get(day["raw_car"]), seen=seen)
        ref = {
            a["form"]: _recommend(held, sid, a["entry"], a["lap_time_s"])
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
                    model = pr.place(held.model, held.track, held.pooled, "dry", a["entry"])
                got = _score_fold_laps(held.prep, model, session_ids=[sid])
                if got.empty:
                    continue
                got = got[
                    (got["lap_within_stint"] == pr.FLYING_LAP_N)
                    & (got["anchor_kind"] == "pit_exit")
                    & got["resid_bar"].notna()
                ].merge(flags, on=["session_id", "lap_num"], how="left")
                tpms.append(
                    pd.DataFrame(
                        {
                            **base_cols,
                            "stint_id": got["stint_id"],
                            "lap_num": got["lap_num"],
                            "corner": got["corner"],
                            "arm": a["arm"],
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
                got = _recommend(held, sid, a["entry"], a["lap_time_s"])
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


def tpms_table(tpms: pd.DataFrame) -> pd.DataFrame:
    """Pressure against the TPMS per raw car, track and arm on the clean
    rows (not after a dropped lap, not at or after a cool-down lap):
    ``rows``, ``mae_mbar``, ``bias_mbar``, with the rows set aside."""
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


def by_cell(stints: pd.DataFrame, beside: str = "other_day") -> pd.DataFrame:
    """One row per raw car, track, tyre, flying lap, form and arm: the
    median over days of the day's median stint of ``size``, ``signed`` and
    ``gap``, with ``days`` and ``stints``. Only the stints that the arm
    ``beside`` also has are counted (a day with no pair is listed, not
    scored), so that an arm and the yardstick stand on the same rows."""
    if stints.empty:
        return stints
    keys = stints.loc[stints["arm"] == beside, _STINT].drop_duplicates()
    own = stints.merge(keys, on=_STINT, how="inner")
    cols = ["size", "signed", "gap", "gap_signed"]
    days = own.groupby(_CELL + ["arm", "date"], sort=False)[cols].median()
    got = days.groupby(_CELL + ["arm"], sort=False)[cols].median()
    got["days"] = days.groupby(_CELL + ["arm"], sort=False).size()
    got["stints"] = own.groupby(_CELL + ["arm"], sort=False).size()
    flagged = own.assign(flag=own["status"] != pr.OK).groupby(_CELL + ["arm"], sort=False)["flag"]
    got["flagged"] = flagged.sum()
    return got.reset_index()


def attain(cells: pd.DataFrame, arm: str, beside: str = "other_day") -> pd.DataFrame:
    """``arm`` beside ``beside``, cell by cell (a raw car at a track on
    one tyre): ``size`` and ``signed`` of the arm, ``beside`` the size of
    the other, ``scored`` (both have :data:`MIN_DAYS` days) and ``passed``
    (the arm is no further from the day than the other is)."""
    if cells.empty:
        return cells
    a = cells[cells["arm"] == arm].drop(columns="arm")
    b = cells[cells["arm"] == beside][_CELL + ["size", "days", "gap"]].rename(
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
    bit for bit: ``(same, rows)``."""
    from ..validate import _score_fold_laps

    placed = model
    for r in list(model["g2_typ_by_track_car_cond"]):
        key = (r["track_canonical"], r["car"], r["condition"])
        entry = pr.entry_of(placed, *key)
        if entry is not None:
            placed = pr.place(placed, *key, entry)
    tables = ("g2_typ_by_track_car_cond", "lap_time_typ_by_track_car_cond")
    same = all(placed[t] == model[t] for t in tables)
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
    all_rows, all_tpms, checks, day_rows, sim_rows = [], [], [], [], []
    for pooled, track in folds:
        held = hold_out(prod, pooled, track, cache_dir)
        if held is None:
            continue
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
        cells: dict[str, SimCell | None] = {}
        if sim is not None:
            for raw_car, own in held.days.groupby("raw_car", sort=True):
                layout_id = _majority(list(own["layout_id"]))
                cell = sim_cell(sim, str(raw_car), layout_id, held.share)
                cells[str(raw_car)] = cell
        rows, tpms = score_cell(held, cells)
        # after the scoring: a cell's tracks are built as its days ask for them
        sim_rows += [_sim_row(pooled, track, k, c) for k, c in cells.items() if c is not None]
        all_rows.append(rows)
        all_tpms.append(tpms)
        day_rows.append(held.days.assign(pooled=pooled, track=track))
        logger.info("pressure: %s at %s scored (%d sessions)", pooled, track, len(held.days))
    rows = pd.concat([r for r in all_rows if len(r)], ignore_index=True)
    tpms = pd.concat([t for t in all_tpms if len(t)], ignore_index=True)
    days = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()

    # leave-date-out: the reference level on cells the fit has seen
    seen_rows, seen_tpms = [], []
    logged = {(str(r.pooled), str(r.track)) for r in table.itertuples()}
    for k, ids in enumerate(date_folds(prod) if dates else []):
        got = prepare(prod, ids, cache_dir)
        if got is None:
            continue
        model, prep = got
        present = prep.full[["car", "track_canonical"]].drop_duplicates()
        for pooled, track in zip(present["car"], present["track_canonical"]):
            held = held_of(prod, model, prep, str(pooled), str(track), len(ids))
            a, b = score_cell(held, {}, seen=True)
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
    report_cells(rep, table, days)
    report_harness(rep, pd.DataFrame(checks))
    main = stints[stints["n"] == pr.FLYING_LAP_N] if len(stints) else stints
    report_yardstick(rep, main)
    if sim is not None:
        report_simulation(rep, main, pd.DataFrame(sim_rows), stints)
    report_replaced(rep, main)
    report_seen(rep, main, seen)
    report_tpms(rep, tpms, seen_t)
    return rep


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
    """Per arm in one form: its cells beside the yardstick's."""
    cells = by_cell(main[main["form"] == form])
    out = []
    for arm in arms:
        att = attain(cells, arm)
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
        ceiling = attain(cells, "ceiling")
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
    table = tpms_table(tpms)
    wide = table.pivot_table(
        index=["pooled", "track", "raw_car"], columns="arm", values=["mae_mbar", "bias_mbar"]
    )
    names: list[str] = []
    for column in wide.columns.tolist():
        what, arm = tuple(column)
        names.append(f"{arm}_{str(what).split('_')[0]}")
    wide = wide.set_axis(names, axis=1)
    wide = wide.reset_index()
    rows = table.groupby(["pooled", "track", "raw_car"])[["rows", "set_aside"]].max().reset_index()
    wide = rows.merge(wide, on=["pooled", "track", "raw_car"])
    if seen is not None and len(seen):
        ref = tpms_table(seen)
        ref = ref[ref["arm"] == "entry"][["pooled", "track", "raw_car", "mae_mbar", "bias_mbar"]]
        wide = wide.merge(
            ref.rename(columns={"mae_mbar": "in_the_fit_mae", "bias_mbar": "in_the_fit_bias"}),
            on=["pooled", "track", "raw_car"],
            how="left",
        )
    wide = wide[wide["rows"] >= 4 * MIN_DAYS]
    arms = [c[:-4] for c in wide.columns if c.endswith("_mae")]
    rep.items.append(
        GateItem(
            "Pressure against the TPMS at flying lap "
            f"{pr.FLYING_LAP_N} (S2: reported, it does not rank a heat source)",
            None,
            "; ".join(f"{a} {wide[f'{a}_mae'].median():.0f} mbar" for a in arms)
            + " in the median cell",
            "reported (#208, clause 7): a heat source with too little heat scores better",
            "Hot-pressure MAE and signed bias (prediction less the TPMS reading, mbar) at the "
            f"end of flying lap {pr.FLYING_LAP_N} of pit-exit stints, production's own inputs "
            "(the session's 25th-percentile lap time typed in), per raw car and cell, the "
            "constants fitted without the cell. `rows`: stint × tyre rows; `set_aside`: rows "
            "after a lap that cleaning dropped from the stint or at or after a cool-down lap, "
            "not counted. `in_the_fit`: the same cell with only the day's date left out "
            "(leave-date-out), which is the level the constants reach when they have seen the "
            "cell. The floor no heat source moves is about 25 mbar on the formula cars and 45 "
            "on the Inferno 86 (#208, finding 5).\n\n" + md_table(wide, floats="{:.3g}"),
        )
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
        cells = by_cell(main[main["form"] == form])
        att = attain(cells, "sim")
        if att.empty:
            continue
        att["neither"] = [(r, t) in neither for r, t in zip(att["raw_car"], att["track"])]
        got_n, got_c = tally(att[att["neither"]]), tally(att[~att["neither"]])
        table = _arm_table(main, arms + ("ceiling", "other_layouts", "fallback"), form)
        # cells an arm wins and loses against the simulation
        won = []
        for arm in table["arm"]:
            other = attain(cells, arm)
            both = att[att["scored"]].merge(
                other[_CELL + ["size", "scored"]], on=_CELL, suffixes=("", "_arm")
            )
            both = both[both["scored_arm"]]
            won.append(
                f"{int((both['size'] < both['size_arm']).sum())} / "
                f"{int((both['size'] > both['size_arm']).sum())}"
            )
        table["sim_nearer / further"] = won
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
            "the lap time is given).\n\n"
            + md_table(
                shown[
                    ["raw_car", "track", "corner", "neither_seen", "days", "stints", "size"]
                    + ["signed", "beside", "gap", "gap_signed", "flagged", "scored", "passed"]
                ],
                floats="{:.3g}",
            )
            + "\n\n**Every arm on the same rows.** `sim_nearer / further`: the scored cells in "
            "which the simulation is nearer the day than the arm, and further.\n\n"
            + md_table(table, floats="{:.3g}")
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


__all__ = [
    "FORMS",
    "Held",
    "MIN_DAYS",
    "Production",
    "Sim",
    "SimCell",
    "attain",
    "by_cell",
    "by_stint",
    "cell_sessions",
    "cell_table",
    "check_pressure",
    "date_folds",
    "day_arms",
    "donor_track",
    "fed_laps",
    "fold_model",
    "held_of",
    "hold_out",
    "load_production",
    "load_sim",
    "prepare",
    "score_cell",
    "sim_arms",
    "sim_cell",
    "tally",
    "tpms_table",
]
