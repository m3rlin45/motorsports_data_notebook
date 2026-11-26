"""Gate D1's clauses on logged laps (lap simulation, issue #206).

:mod:`.heat` is production's heat chain for one lap of any series, the
seconds counted from the lap's start. Production itself bins a whole stint
on its rolling clock (``statespace._bin_stint``) and sums a lap from the
stint's bins (``statespace.lap_heat_share``). This gate sets the one
against the other on the laps production fits, so that a heat computed
by :mod:`.heat` for a simulated lap can be read beside production's:

- **per lap and per stint**: the chain's heat over production's own
  per-lap heat, from the same samples and the same shares;
- **holes**: a lap with a window the sensor verdict discarded has no
  heat in either;
- **the phase of the bins**: what moving the bins' edges half a second
  does to a lap's heat (production's edges fall where its stint's clock
  puts them, a lap's own where the lap begins);
- **smoothing**: issue #207 has both sides of a comparison smoothed over
  0.5 s before the positive parts are taken, which this chain does not
  do; how far the two are apart, on logged laps and on simulated ones.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..heat_input import ShareParams
from . import car as cm
from . import geometry as geo
from . import holdout as ho
from . import samples as smp
from . import solver as sv
from .gates import GateItem, GateReport, md_table
from .heat import CORNERS, LapBins, bin_1s, lap_heat, sim_series
from .track import IMU_SOURCES

logger = logging.getLogger(__name__)

#: Mean over a car's laps of ``|log(chain / production)|``, per tyre.
LAP_MAX = 0.01
#: The same over its stints of the stint's mean log ratio.
STINT_MAX = 0.003
#: 90th percentile over laps of what half a second of bin phase moves.
PHASE_MAX = 0.02
PHASE_S = 0.5
#: Half the window of #207 E0's smoothing.
SMOOTH_HALF_S = 0.25


def production_lap_heat(root: Path) -> pd.DataFrame:
    """Production's own per-lap heat: the laps it fits, with ``q_lap_<tyre>``
    from its stints' bins (``warmup_table.attach_lap_heat``) and the car
    under the name its shares are fitted for."""
    from ..heat_experiment import prepare_laps_for_fit
    from ..warmup_table import attach_lap_heat

    model = json.loads((Path(root) / "tire_model.json").read_text())
    laps = attach_lap_heat(root, prepare_laps_for_fit(root), model)
    cols = ["session_id", "stint_id", "lap_num", "car"] + [f"q_lap_{c}" for c in CORNERS]
    if any(c not in laps.columns for c in cols):
        return pd.DataFrame(columns=cols)
    return laps[cols].reset_index(drop=True)


def _smooth(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(geo.nan_window_mean(x, t, SMOOTH_HALF_S))


def _sensor_laps(root: Path, laps: pd.DataFrame, cars: Iterable[str] | None = None):  # type: ignore[no-untyped-def]
    """Walk the laps of ``laps`` (``session_id, lap_num, car``) as
    production's chain sees them: per lap ``(session_id, car, lap_num,
    hole, source, t, v, lat, lng, lat_lp, lng_lp, a, b)``, the moving
    samples outside discarded windows of the sensor's series
    (``lat_lp, lng_lp`` smoothed over 0.5 s) and the lap's span. A session
    without a series, or of a car not in ``cars`` when it is given, is
    skipped."""
    from ..statespace import MOVING_SPEED_MS, _session_timeseries

    known = None if cars is None else set(cars)
    for sid, grp in laps.groupby("session_id", sort=True):
        series = _session_timeseries(str(root), str(sid))
        car = str(grp["car"].iloc[0])
        if series is None or (known is not None and car not in known):
            continue
        kin = series.kin.sort_values("t_session_s")
        t_all = kin["t_session_s"].to_numpy(float)
        lap_all = kin["lap_num"].to_numpy()
        v_all = kin["speed_ms"].to_numpy(float)
        lat_all = kin["lat_g"].to_numpy(float)
        lng_all = kin["long_g"].to_numpy(float)
        has_source = "lat_source" in kin.columns
        src_all = kin["lat_source"].to_numpy() if has_source else np.full(len(kin), "imu")
        # the kind of sensor the session logged with, as a label: in the tyres'
        # frame (#376) the heat is GPS-derived and the verdict is carried beside
        kind_all = (
            kin["sensor_source"].fillna("").to_numpy()
            if "sensor_source" in kin.columns
            else src_all
        )
        step = float(np.median(np.diff(t_all))) if len(t_all) > 1 else 0.02
        lat_lp, lng_lp = _smooth(lat_all, t_all), _smooth(lng_all, t_all)
        for lap_num in sorted(int(x) for x in grp["lap_num"].unique()):
            on = lap_all == lap_num
            if on.sum() < 10:
                continue
            t, v = t_all[on], v_all[on]
            kinds = {str(s) for s in src_all[on]}
            hole = "discard" in kinds
            moving_kinds = {str(s) for s in kind_all[on]} - {"discard", ""}
            # a second through a discarded window has no known heat: its
            # samples are left out, and the lap then lacks coverage
            keep = (np.nan_to_num(v, nan=0.0) > MOVING_SPEED_MS) & (src_all[on] != "discard")
            source = (
                "imu"
                if moving_kinds <= set(IMU_SOURCES)
                else "mixed" if moving_kinds & set(IMU_SOURCES) else "gps"
            )
            yield (
                str(sid),
                car,
                lap_num,
                hole,
                source,
                t[keep],
                v[keep],
                lat_all[on][keep],
                lng_all[on][keep],
                lat_lp[on][keep],
                lng_lp[on][keep],
                float(t[0]),
                float(t[-1]) + step,
            )


def chain_lap_heat(root: Path, laps: pd.DataFrame, shares: dict[str, ShareParams]) -> pd.DataFrame:
    """The chain of :mod:`.heat` on each lap of ``laps``
    (:func:`production_lap_heat`), from the samples production bins:
    ``q_<tyre>`` with the bins from the lap's start, ``phase_<tyre>``
    with their edges :data:`PHASE_S` back, ``smooth_<tyre>`` with both
    accelerations smoothed over 0.5 s first; ``hole`` says the lap holds
    a discarded window (its heats are then NaN) and ``source`` what its
    samples are (``imu``, ``gps`` or ``mixed``)."""
    rows: list[dict] = []
    for sid, car, lap_num, hole, source, t, v, lat, lng, lat_lp, lng_lp, a, b in _sensor_laps(
        root, laps, shares
    ):
        row: dict = {"session_id": sid, "lap_num": lap_num, "hole": hole, "source": source}
        for tag, lat_, lng_, phase in (
            ("q", lat, lng, 0.0),
            ("phase", lat, lng, PHASE_S),
            ("smooth", lat_lp, lng_lp, 0.0),
        ):
            bins = bin_1s(t, v, lat_, lng_, a, b, phase_s=phase)
            for c, q in zip(CORNERS, lap_heat(bins, shares[car])):
                row[f"{tag}_{c}"] = float(q)
        rows.append(row)
    return pd.DataFrame(rows)


def sensor_lap_bins(
    root: Path, laps: pd.DataFrame
) -> dict[tuple[str, int], tuple[LapBins, str, bool]]:
    """Each lap of ``laps`` (``session_id, lap_num, car``) as production's
    chain bins it, before any force shares: ``(session_id, lap_num) ->
    (bins, source, hole)``. :func:`~.heat.lap_heat` of the bins is
    :func:`chain_lap_heat`'s ``q_<tyre>`` to the last bit, so one read of
    the timeseries serves every fit's shares (#208: a fold's heat is of
    the fold's own shares)."""
    return {
        (sid, lap_num): (bin_1s(t, v, lat, lng, a, b), source, hole)
        for sid, _car, lap_num, hole, source, t, v, lat, lng, _lp, _lg, a, b in _sensor_laps(
            root, laps
        )
    }


def simulated_smoothing(
    root: Path,
    cars_dir: Path,
    shares: dict[str, ShareParams],
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Per model car and layout: the heat of the car's lap at its typical
    use, as the chain has it, over the same lap with both accelerations
    smoothed over 0.5 s first (worst tyre and the four-tyre mean, as log
    ratios). Empty without the car models gate C2 saves."""
    rows = []
    for car_name in smp.MODEL_CARS:
        path = Path(cars_dir) / f"{car_name.replace(' ', '_')}.json"
        if not path.exists():
            continue
        model = cm.CarModel.load(path)
        try:
            share = ho.share_for(car_name, shares)
        except KeyError:
            continue
        car = sv.SolverCar.from_model(model)
        for layout in sorted(model.ref_u):
            use = model.ref_u[layout][0]
            if use is None:
                continue
            track = smp.load_track(root, layout, tracks_dir=tracks_dir, cache_dir=cache_dir)
            sim = sv.simulate_lap(sv.Profile.from_track(track), car, float(use))
            t, v, lat, lng = sim_series(sim)
            plain = lap_heat(bin_1s(t, v, lat, lng, 0.0, sim.lap_time_s), share)
            smooth = lap_heat(
                bin_1s(t, v, _smooth(lat, t), _smooth(lng, t), 0.0, sim.lap_time_s), share
            )
            ratio = np.log(plain / smooth)
            rows.append(
                {
                    "car": car_name,
                    "layout_id": layout,
                    "lap_s": sim.lap_time_s,
                    "mean": float(ratio.mean()),
                    "worst": float(ratio[np.argmax(np.abs(ratio))]),
                }
            )
    return pd.DataFrame(rows)


def _tyres_frame_laps(  # type: ignore[no-untyped-def]
    root: Path,
    laps: pd.DataFrame,
    cars: Iterable[str] | None = None,
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
):
    """Walk the laps of ``laps`` in the frame a simulated lap is in: per
    lap ``(session_id, car, lap_num, t, v, lat, along, a, b)``, the lap's
    samples on the position clock and its span. Laps without geometry, a
    track or a place in the lap index, or of a car not in ``cars`` when it
    is given, are skipped."""
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    spans = index.set_index(["session_id", "lap_num"])[["t_start_s", "t_end_s", "layout_id"]]
    known = None if cars is None else set(cars)
    tracks: dict[str, object] = {}
    for sid, grp in laps.groupby("session_id", sort=True):
        car = str(grp["car"].iloc[0])
        keys = [(sid, int(n)) for n in grp["lap_num"] if (sid, int(n)) in spans.index]
        if not keys or (known is not None and car not in known):
            continue
        own = spans.loc[keys].reset_index()
        own = own[np.isfinite(own["t_start_s"]) & np.isfinite(own["t_end_s"])]
        if own.empty:
            continue
        layout = str(own["layout_id"].iloc[0])
        if layout not in tracks:
            try:
                tracks[layout] = smp.load_track(
                    root, layout, tracks_dir=tracks_dir, cache_dir=cache_dir
                )
            except (ValueError, FileNotFoundError, KeyError):
                tracks[layout] = None
        track = tracks[layout]
        sig = smp.session_signals(root, str(sid), layout) if track is not None else None
        if sig is None or track is None:
            continue
        s, grade, _ = smp.track_position(sig, track, own)  # type: ignore[arg-type]
        lat = smp.in_tyres_frame(sig, track, s)  # type: ignore[arg-type]
        along = sig.long_g + np.nan_to_num(grade)
        t, v = sig.geom.t, np.nan_to_num(sig.geom.speed, nan=0.0)
        for n, a, b in zip(own["lap_num"], own["t_start_s"], own["t_end_s"]):
            i0, i1 = np.searchsorted(t, [a, b])
            yield sid, car, int(n), t[i0:i1], v[i0:i1], lat[i0:i1], along[i0:i1], a, b


def tyres_frame_lap_heat(
    root: Path,
    laps: pd.DataFrame,
    shares: dict[str, ShareParams],
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """The heat of each logged lap of ``laps`` (``session_id, lap_num,
    car``) **in the frame a simulated lap is in**: the lap's own Doppler
    lateral as the tyres hold it on the road's bank, and ``dV/dt`` with the
    grade (what an accelerometer reads along the road), on the position
    clock, through the chain. ``t_<tyre>`` per lap; laps without geometry,
    a track or a place in the lap index are left out."""
    from .heat import series_heat

    rows: list[dict] = []
    for sid, car, n, t, v, lat, along, a, b in _tyres_frame_laps(
        root, laps, shares, tracks_dir=tracks_dir, cache_dir=cache_dir
    ):
        q = series_heat(t, v, lat, along, shares[car], a, b)
        rows.append(
            {"session_id": sid, "lap_num": n, **{f"t_{c}": q[k] for k, c in enumerate(CORNERS)}}
        )
    return pd.DataFrame(rows, columns=["session_id", "lap_num"] + [f"t_{c}" for c in CORNERS])


def tyres_frame_lap_bins(
    root: Path,
    laps: pd.DataFrame,
    *,
    tracks_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[tuple[str, int], LapBins]:
    """Each lap of ``laps`` binned in the frame a simulated lap is in,
    before any force shares: ``(session_id, lap_num) -> bins``, of which
    :func:`~.heat.lap_heat` is :func:`tyres_frame_lap_heat`'s ``t_<tyre>``
    to the last bit."""
    return {
        (str(sid), n): bin_1s(t, v, lat, along, a, b)
        for sid, _car, n, t, v, lat, along, a, b in _tyres_frame_laps(
            root, laps, tracks_dir=tracks_dir, cache_dir=cache_dir
        )
    }


def frame_factor(both: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(factors, left)``: what brings a heat in the simulated lap's frame
    to production's (#206's frame clause). ``both`` has, per logged lap,
    production's chain on the sensor's series (``q_<tyre>``, with
    ``source``) and the same lap in the simulated lap's frame
    (``t_<tyre>``), with ``car, layout_id, session_id``.

    ``factors``: per car, layout and source, the per-tyre median over the
    laps of sensor-frame over tyres'-frame heat (``f_<tyre>``), with
    ``laps`` and ``sessions``. ``left``: per car and source, what the
    factor leaves session by session: the median and the 90th percentile
    over sessions of the size of the session's mean log ratio less its
    cell's, on the worst tyre."""
    cols = [f"r_{c}" for c in CORNERS]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pd.DataFrame(
            {f"r_{c}": np.log(both[f"q_{c}"] / both[f"t_{c}"]) for c in CORNERS}, index=both.index
        )
    laps = pd.concat([both[["car", "layout_id", "source", "session_id"]], ratio], axis=1)
    laps = laps.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    cell = ["car", "layout_id", "source"]
    if laps.empty:
        return pd.DataFrame(), pd.DataFrame()
    med = laps.groupby(cell, sort=True)[cols].median()
    factors = pd.DataFrame(
        np.exp(med.to_numpy()), index=med.index, columns=[f"f_{c}" for c in CORNERS]
    )
    factors.insert(0, "laps", laps.groupby(cell).size())
    factors.insert(1, "sessions", laps.groupby(cell)["session_id"].nunique())
    by_session = laps.groupby(cell + ["session_id"], sort=True)[cols].mean().reset_index()
    centre = by_session[cell].merge(med.reset_index(), on=cell, how="left")[cols].to_numpy()
    off = by_session[cell + ["session_id"]].assign(
        worst=np.abs(by_session[cols].to_numpy() - centre).max(axis=1)
    )
    left = (
        off.groupby(["car", "source"])["worst"]
        .agg(sessions="size", median="median", p90=lambda x: float(np.percentile(x, 90)))
        .reset_index()
    )
    return factors.reset_index(), left


def check_heat_chain(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    cars_dir: Path | None = None,
) -> GateReport:
    """Gate D1 on logged laps: the chain against production's own per-lap
    heat, holes, the phase of the bins, and the 0.5 s smoothing."""
    rep = GateReport("D1", "production's heat chain on a lap (`lapsim check-heat-chain`)")
    own = production_lap_heat(root)
    shares = ho.production_shares(root)
    got = chain_lap_heat(root, own, shares)
    if own.empty or got.empty:
        rep.items.append(GateItem("Laps production fits", False, "none", "at least one"))
        return rep
    both = own.merge(got, on=["session_id", "lap_num"])
    for c in CORNERS:
        both[f"r_{c}"] = np.log(both[f"q_{c}"] / both[f"q_lap_{c}"])
        both[f"p_{c}"] = np.log(both[f"phase_{c}"] / both[f"q_{c}"])
        both[f"s_{c}"] = np.log(both[f"q_{c}"] / both[f"smooth_{c}"])
    ratio = [f"r_{c}" for c in CORNERS]
    scored = both[both[ratio].notna().all(axis=1)]
    rep.notes.append(
        f"{len(own)} laps in {own.groupby(['session_id', 'stint_id']).ngroups} stints that "
        f"production fits; {len(scored)} have a heat from both sides. The chain is given the "
        "samples production bins (`kinematics.load_kinematics`, moving, outside discarded "
        "windows) and production's fitted shares."
    )

    # 1. Per lap.
    per_lap = (
        scored.groupby("car")
        .agg(
            laps=("lap_num", "size"),
            **{f"mean_{c}": (f"r_{c}", "mean") for c in CORNERS},
            **{f"abs_{c}": (f"r_{c}", lambda x: float(x.abs().mean())) for c in CORNERS},
            **{f"p90_{c}": (f"r_{c}", lambda x: float(x.abs().quantile(0.9))) for c in CORNERS},
        )
        .reset_index()
    )
    worst_lap = float(per_lap[[f"abs_{c}" for c in CORNERS]].to_numpy().max())
    rep.items.append(
        GateItem(
            "A lap's heat against production's own",
            bool(worst_lap <= LAP_MAX),
            "; ".join(
                f"{r.car} {max(getattr(r, f'abs_{c}') for c in CORNERS):.4f} on the worst tyre"
                for r in per_lap.itertuples()
            ),
            f"mean over a car's laps of |log(chain / production)| ≤ {LAP_MAX} on every tyre",
            "`mean_<tyre>`: the mean log ratio of the chain's lap heat (bins from the lap's "
            "start) over production's (the stint's bins summed over the lap); `abs_`: the mean "
            "of its size; `p90_`: the 90th percentile of its size.\n\n"
            + md_table(per_lap, floats="{:.4f}"),
        )
    )

    # 2. Per stint.
    stints = scored.groupby(["car", "session_id", "stint_id"])[ratio].mean().reset_index()
    per_stint = (
        stints.groupby("car")
        .agg(
            stints=("stint_id", "size"),
            **{f"abs_{c}": (f"r_{c}", lambda x: float(x.abs().mean())) for c in CORNERS},
            **{f"p90_{c}": (f"r_{c}", lambda x: float(x.abs().quantile(0.9))) for c in CORNERS},
        )
        .reset_index()
    )
    worst_stint = float(per_stint[[f"abs_{c}" for c in CORNERS]].to_numpy().max())
    rep.items.append(
        GateItem(
            "A stint's heat against production's own",
            bool(worst_stint <= STINT_MAX),
            "; ".join(
                f"{r.car} {max(getattr(r, f'abs_{c}') for c in CORNERS):.4f} on the worst tyre"
                for r in per_stint.itertuples()
            ),
            f"mean over a car's stints of |the stint's mean log ratio| ≤ {STINT_MAX} on every "
            "tyre",
            md_table(per_stint, floats="{:.4f}"),
        )
    )

    # 3. Holes.
    holed = both[both["hole"]]
    no_heat = int(holed[[f"q_{c}" for c in CORNERS]].isna().all(axis=1).sum())
    missing = own.merge(got[["session_id", "lap_num", "hole"]], on=["session_id", "lap_num"])
    prod_none = missing[missing[[f"q_lap_{c}" for c in CORNERS]].isna().all(axis=1)]
    rep.items.append(
        GateItem(
            "A lap with a hole has no heat",
            bool(no_heat == len(holed)),
            f"{len(holed)} laps hold a discarded window; the chain gives {no_heat} of them no "
            f"heat; production gives no heat to {len(prod_none)} laps, {int(prod_none['hole'].sum())} "
            "of them with a hole",
            "every lap with a discarded window comes back without a heat, not with a smaller one",
        )
    )

    # 4. The phase of the bins.
    phase = [f"p_{c}" for c in CORNERS]
    moved = scored[scored[phase].notna().all(axis=1)]
    per_phase = (
        moved.groupby("car")
        .agg(
            laps=("lap_num", "size"),
            **{f"abs_{c}": (f"p_{c}", lambda x: float(x.abs().mean())) for c in CORNERS},
            **{f"p90_{c}": (f"p_{c}", lambda x: float(x.abs().quantile(0.9))) for c in CORNERS},
        )
        .reset_index()
    )
    worst_phase = float(per_phase[[f"p90_{c}" for c in CORNERS]].to_numpy().max())
    rep.items.append(
        GateItem(
            "The phase of the bins",
            bool(worst_phase <= PHASE_MAX),
            "; ".join(
                f"{r.car} {max(getattr(r, f'p90_{c}') for c in CORNERS):.4f} at the 90th "
                "percentile on the worst tyre"
                for r in per_phase.itertuples()
            ),
            f"the bins' edges {PHASE_S} s back move a lap's heat by under {PHASE_MAX} at the "
            "90th percentile, on every tyre and car",
            "The log ratio of the lap's heat with the bins' edges half a second back over its "
            "heat with the bins from the lap's start: `abs_` its mean size, `p90_` the 90th "
            "percentile of its size. This is what the difference between production's bins "
            "(on the stint's clock) and a lap's own is made of.\n\n"
            + md_table(per_phase, floats="{:.4f}"),
        )
    )

    # 5. The 0.5 s smoothing of #207 E0.
    smooth = [f"s_{c}" for c in CORNERS]
    lp = scored[scored[smooth].notna().all(axis=1)]
    per_smooth = (
        lp.groupby(["car", "source"])
        .agg(laps=("lap_num", "size"), **{f"mean_{c}": (f"s_{c}", "mean") for c in CORNERS})
        .reset_index()
    )
    sims = (
        simulated_smoothing(root, cars_dir, shares, tracks_dir=tracks_dir, cache_dir=cache_dir)
        if cars_dir is not None
        else pd.DataFrame()
    )
    logged_range = per_smooth[[f"mean_{c}" for c in CORNERS]].to_numpy()
    text = f"logged laps {np.nanmin(logged_range):+.4f} to {np.nanmax(logged_range):+.4f}"
    if len(sims):
        text += (
            f"; simulated laps {sims['mean'].min():+.4f} to {sims['mean'].max():+.4f} in the "
            f"four-tyre mean, {sims['worst'].abs().max():.4f} at most on a tyre"
        )
    rep.items.append(
        GateItem(
            "Smoothing over 0.5 s before the positive parts (issue #207, E0)",
            None,
            text,
            "reported: a comparison is in one chain on both sides",
            "The log of a lap's heat as this chain has it over its heat with the lateral and the "
            "longitudinal smoothed over 0.5 s first, as #207 E0 would have both sides of a "
            "comparison. On logged laps by the samples' source (a GPS-derived series is noisier "
            "than an accelerometer's):\n\n"
            + md_table(per_smooth, floats="{:.4f}")
            + (
                "\n\nOn each model car's lap at its typical use, simulated (`mean`: the "
                "four-tyre mean; `worst`: the tyre furthest from 0):\n\n"
                + md_table(sims, floats="{:.4f}")
                if len(sims)
                else ""
            ),
        )
    )

    # What brings a simulated lap's heat to production's (#206's frame clause).
    in_frame = tyres_frame_lap_heat(root, own, shares, tracks_dir=tracks_dir, cache_dir=cache_dir)
    if len(in_frame):
        index = geo.clean_lap_index(root, cache_dir=cache_dir)
        where = index[["session_id", "lap_num", "layout_id"]].drop_duplicates()
        framed = both.merge(in_frame, on=["session_id", "lap_num"]).merge(
            where, on=["session_id", "lap_num"]
        )
        factors, left = frame_factor(framed)
        if len(factors):
            f_cols = [f"f_{c}" for c in CORNERS]
            imu = factors[factors["source"] == "imu"]
            span = imu[f_cols].to_numpy() if len(imu) else factors[f_cols].to_numpy()
            rep.items.append(
                GateItem(
                    "The frame factor: production's heat over the same lap in a simulated lap's "
                    "frame",
                    None,
                    f"accelerometer laps {np.nanmin(span):.3f} to {np.nanmax(span):.3f} by car, "
                    f"layout and tyre; after it a session is off by "
                    + "; ".join(
                        f"{r['car']} {r['source']} {r['median']:.3f} median, {r['p90']:.3f} P90"
                        for r in left.to_dict("records")
                    ),
                    "reported",
                    "A simulated lap is in the tyres' frame on the position clock: the lateral "
                    "the tyres hold on the road's bank, and `dV/dt` with the grade. Production's "
                    "tau and K were fitted on the heat of the sensor's series. Where a table's "
                    "heat has to meet them it is multiplied by this: per production car, layout "
                    "and source, the per-tyre median over the logged laps of production's chain "
                    "on the sensor's series over the same lap's Doppler lateral in the tyres' "
                    "frame and its `dV/dt` with the grade, through the same chain. No simulated "
                    "lap is in it. It goes away if production is refitted on the tyres'-frame "
                    "series (#255).\n\n"
                    + md_table(factors, floats="{:.3f}")
                    + "\n\n**What it leaves, session by session** (the size of a session's mean "
                    "log ratio less its cell's, on the worst tyre).\n\n"
                    + md_table(left, floats="{:.4f}"),
                )
            )
    return rep


__all__ = [
    "LAP_MAX",
    "PHASE_MAX",
    "STINT_MAX",
    "chain_lap_heat",
    "check_heat_chain",
    "frame_factor",
    "production_lap_heat",
    "sensor_lap_bins",
    "simulated_smoothing",
    "tyres_frame_lap_bins",
    "tyres_frame_lap_heat",
]
