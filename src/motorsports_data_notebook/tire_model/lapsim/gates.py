"""Gate reports of the lap simulation's track segment (issue #204).

A gate is a list of :class:`GateItem`: a statement with a threshold, the
number measured on the dataset and whether it holds. ``passed`` is None for
an item that is reported and not gated. :func:`check_geometry` is gate B1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo

#: Operators' published lap lengths (m); sources on issue #204.
OFFICIAL_LENGTH_M = {
    "tsukuba_2000": 2045.0,
    "fuji_gp": 4563.0,
    "suzuka_full": 5807.0,
    "motegi_full": 4801.379,
    "sodegaura": 2436.0,
}

CLOSURE_MAX_TURNS = 0.002
CLOSURE_MIN_SHARE = 0.99
PATH_VS_SPEED_MAX = 0.003
SESSION_LENGTH_MAX = 0.005
LENGTH_SPREAD_MAX = 0.01
OFFICIAL_LENGTH_MAX = 0.02
PEAK_G_MAX = 3.0
FLAGGED_OVERALL_MAX = 0.02
FLAGGED_CELL_MAX = 0.05
GYRO_SPREAD_MAX = 0.01
MIN_VOTE_DAYS = 3
#: An unmatched lap this much longer than the session's median lap holds a
#: stop or the start of the recording; this much shorter, it is a fragment.
NOT_FLYING_FACTOR = 1.3
FRAGMENT_FACTOR = 0.5
PATH_PROFILE_BIN_M = 250.0
PATH_PROFILE_SHOW_M = 1.0
PATH_PROFILE_MIN_LAPS = 5


@dataclass
class GateItem:
    name: str
    passed: bool | None
    measured: str
    threshold: str = ""
    detail: str = ""  # markdown shown under the summary table


@dataclass
class GateReport:
    gate: str
    title: str
    items: list[GateItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(i.passed is not False for i in self.items)

    def to_markdown(self) -> str:
        mark = {True: "pass", False: "**FAIL**", None: "reported"}
        out = [f"# {self.gate}: {self.title}", ""]
        out.append(f"Overall: {'pass' if self.passed else '**FAIL**'}")
        out += ["", "| Check | Threshold | Measured | Result |", "|---|---|---|---|"]
        for it in self.items:
            out.append(f"| {it.name} | {it.threshold} | {it.measured} | {mark[it.passed]} |")
        for note in self.notes:
            out += ["", note]
        for it in self.items:
            if it.detail:
                out += ["", f"## {it.name}", "", it.detail]
        return "\n".join(out) + "\n"


def md_table(df: pd.DataFrame, *, floats: str = "{:.4g}") -> str:
    """A GitHub-flavoured markdown table of ``df`` (index dropped)."""
    if df.empty:
        return "_none_"

    def _cell(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floats.format(v)
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def _q(p: float):
    return lambda s: s.quantile(p)


def expected_turns(index: pd.DataFrame) -> pd.Series:
    """Whole turns of a lap per layout: the commonest value on its build
    laps (+1 clockwise, −1 anticlockwise, 0 for a figure of eight)."""
    b = index[index["in_build"] & index["turns"].notna()]
    return b.groupby("layout_id")["turns"].agg(lambda s: float(s.round().mode().iloc[0]))


def unmatched_causes(index: pd.DataFrame) -> pd.DataFrame:
    """Full laps on a scored layout that match no crossing interval, with
    the cause: ``not_flying`` (well over the session's median lap: a stop or
    the start of the recording inside it), ``fragment`` (well under),
    ``no_crossings`` (the session crosses the gate fewer than twice) or
    ``unexplained``."""
    full = index[
        (index["lap_type"] == "full")
        & index["layout_id"].notna()
        & ~index["layout_id"].isin(geo.EXCLUDED_LAYOUTS)
    ]
    median = full.groupby("session_id")["lap_time_s"].transform("median")
    um = full[~full["matched"]].copy()
    rel = um["lap_time_s"] / median[um.index]
    cause = pd.Series("unexplained", index=um.index, dtype=object)
    cause[um["n_crossings"] < 2] = "no_crossings"
    cause[(cause == "unexplained") & (rel > NOT_FLYING_FACTOR)] = "not_flying"
    cause[(cause == "unexplained") & (rel < FRAGMENT_FACTOR)] = "fragment"
    um["cause"] = cause
    um["vs_median_lap"] = rel
    return um[["session_id", "lap_num", "lap_time_s", "vs_median_lap", "lap_source", "cause"]]


def path_excess_profile(
    root: Path,
    build: pd.DataFrame,
    layout_id: str,
    *,
    bin_m: float | None = None,
    max_sessions: int = 6,
    laps_per_session: int = 5,
) -> pd.DataFrame:
    """Where along a lap the XY path gains on the integral of speed.

    For a sample of the layout's build laps, ``Σ(Δs − V·Δt)`` in bins of
    lap distance, median over the laps: ``s_from_m, excess_m``. A position
    error that sits at one place every lap (an underpass) shows as metres
    of extra path in its bin; sample noise spreads over the lap.
    """
    bin_m = PATH_PROFILE_BIN_M if bin_m is None else bin_m
    rows = build[build["layout_id"] == layout_id]
    top = rows.groupby("session_id").size().sort_values(ascending=False).index[:max_sessions]
    per_lap: list[pd.Series] = []
    for sid in top:
        geom = geo.load_session_geometry(root, str(sid), layout_id)
        if geom is None:
            continue
        for lap in rows[rows["session_id"] == sid]["lap_num"].iloc[:laps_per_session]:
            tr = geo.lap_trace(geom, int(lap))
            if tr is None:
                continue
            ds = np.diff(tr.s)
            dv = 0.5 * (tr.speed[1:] + tr.speed[:-1]) * np.diff(tr.t)
            ok = np.isfinite(dv)
            bins = (tr.s[:-1][ok] // bin_m).astype(int)
            per_lap.append(pd.Series(ds[ok] - dv[ok]).groupby(bins).sum())
    if not per_lap:
        return pd.DataFrame({"s_from_m": [], "excess_m": []})
    med = pd.concat(per_lap, axis=1).median(axis=1)
    out = pd.DataFrame({"s_from_m": med.index.to_numpy() * bin_m, "excess_m": med.to_numpy()})
    out.attrs["n_laps"] = len(per_lap)
    return out


def check_geometry(root: Path, *, cache_dir: Path | None = None) -> GateReport:
    """Gate B1: every lap has a position, a course and a speed on one
    clock, and the laps that build a track are driven at pace on a sound
    receiver."""
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    rep = GateReport("B1", "lap geometry on one clock (`lapsim check-geometry`)")
    clean = index[index["clean"]]
    build = index[index["in_build"]]
    rep.notes.append(
        f"{len(index)} laps in {index['session_id'].nunique()} sessions; "
        f"{len(clean)} clean laps in {clean['session_id'].nunique()} sessions; "
        f"{len(build)} build laps in {build['session_id'].nunique()} sessions on "
        f"{build['date'].nunique()} days."
    )

    # 1. The Doppler course closes.
    turns = expected_turns(index)
    want = build["layout_id"].map(turns)
    closes = (build["closure"].abs() <= CLOSURE_MAX_TURNS) & (build["turns"] == want)
    share = float(closes.mean()) if len(build) else float("nan")
    q = build["closure"].quantile([0.1, 0.9])
    rep.items.append(
        GateItem(
            "Doppler course closes",
            bool(share >= CLOSURE_MIN_SHARE),
            f"{share:.2%} of build laps; closure P10–P90 {q.iloc[0]:+.4f} to {q.iloc[1]:+.4f} turns",
            f"|turns − n| ≤ {CLOSURE_MAX_TURNS} on ≥ {CLOSURE_MIN_SHARE:.0%} of build laps",
            "Whole turns per layout: "
            + ", ".join(f"{k} {v:+.0f}" for k, v in turns.items())
            + ".\n\nBuild laps outside:\n\n"
            + md_table(build[~closes][["session_id", "lap_num", "turns", "closure"]])
            + "\n\nClean laps outside (not in a build):\n\n"
            + md_table(
                clean[
                    ~clean["in_build"]
                    & ~(
                        (clean["closure"].abs() <= CLOSURE_MAX_TURNS)
                        & (clean["turns"] == clean["layout_id"].map(turns))
                    )
                ][["session_id", "lap_num", "turns", "closure", "build_reason"]]
            ),
        )
    )

    # 2. Lag by logger.
    sess = (
        index.dropna(subset=["lag_s"])
        .groupby("session_id")
        .agg(
            car=("car", "first"),
            logger_id=("logger_id", "first"),
            lag_ms=("lag_s", lambda s: 1e3 * s.iloc[0]),
            raw_deg=("lag_scatter_raw_deg", "first"),
            scatter_deg=("lag_scatter_deg", "first"),
            clean_laps=("clean", "sum"),
        )
        .reset_index()
    )
    by_logger = (
        sess.groupby(["logger_id", "car"])
        .agg(
            sessions=("lag_ms", "size"),
            lag_p10_ms=("lag_ms", _q(0.1)),
            lag_median_ms=("lag_ms", "median"),
            lag_p90_ms=("lag_ms", _q(0.9)),
            scatter_as_logged_deg=("raw_deg", "median"),
            scatter_median_deg=("scatter_deg", "median"),
            scatter_max_deg=("scatter_deg", "max"),
        )
        .reset_index()
    )
    with_laps = sess[sess["clean_laps"] > 0]
    over = with_laps[~(with_laps["scatter_deg"] <= geo.LAG_SCATTER_MAX_DEG)]
    n_out = int((index["build_reason"] == "receiver").sum())
    rep.items.append(
        GateItem(
            "Lag removed",
            None,
            f"{len(over)} of {len(with_laps)} sessions with clean laps over "
            f"{geo.LAG_SCATTER_MAX_DEG}°; they are left out of the build ({n_out} laps)",
            f"scatter of course − chord heading ≤ {geo.LAG_SCATTER_MAX_DEG}° after the shift",
            md_table(by_logger, floats="{:.3g}")
            + "\n\nSessions over the bound (left out of the build as `receiver`):\n\n"
            + md_table(over.sort_values("scatter_deg", ascending=False), floats="{:.3g}"),
        )
    )

    # 3. Path length against the integral of speed.
    rel = build["length_m"] / build["v_integral_m"] - 1.0
    inside = rel.abs() <= PATH_VS_SPEED_MAX
    by_cell = (
        build.assign(rel_pct=100.0 * rel, inside=inside)
        .groupby(["layout_id", "car"])
        .agg(
            laps=("rel_pct", "size"),
            median_pct=("rel_pct", "median"),
            min_pct=("rel_pct", "min"),
            max_pct=("rel_pct", "max"),
            inside=("inside", "mean"),
        )
        .reset_index()
    )
    detail = md_table(by_cell, floats="{:.3g}")
    outside_by_layout = build[~inside].groupby("layout_id").size()
    for layout_id in outside_by_layout[outside_by_layout >= PATH_PROFILE_MIN_LAPS].index:
        prof = path_excess_profile(root, build, str(layout_id))
        big = prof[prof["excess_m"].abs() >= PATH_PROFILE_SHOW_M]
        detail += (
            f"\n\n{layout_id}: {int(outside_by_layout[layout_id])} laps outside. Where the "
            f"path gains on ∫V dt (median of {int(prof.attrs.get('n_laps', 0))} build laps, "
            f"{PATH_PROFILE_BIN_M:g} m bins holding {PATH_PROFILE_SHOW_M:g} m or more; the "
            f"whole lap gains {prof['excess_m'].sum():+.1f} m):\n\n"
            + md_table(big, floats="{:.3g}")
        )
    rep.items.append(
        GateItem(
            "Path length = ∫V dt",
            bool(inside.all()),
            f"{inside.mean():.2%} of build laps inside; median {100 * rel.median():+.3f} %, "
            f"range {100 * rel.min():+.2f} to {100 * rel.max():+.2f} %",
            f"within {PATH_VS_SPEED_MAX:.1%} on every build lap",
            detail,
        )
    )

    # 4. Lap length against the session row and its spread.
    # The session row's figure is the median over every crossing interval,
    # so it is compared with the median of every matched full lap.
    matched_full = index[index["matched"] & (index["lap_type"] == "full")]
    matched_full = matched_full[matched_full["layout_id"].isin(clean["layout_id"].unique())]
    per_session = (
        matched_full.groupby("session_id")
        .agg(
            layout_id=("layout_id", "first"),
            median_m=("length_m", "median"),
            row_m=("lap_length_m", "first"),
            laps=("length_m", "size"),
        )
        .reset_index()
    )
    per_session["rel"] = per_session["median_m"] / per_session["row_m"] - 1.0
    off = per_session[~(per_session["rel"].abs() <= SESSION_LENGTH_MAX)]
    spread = (
        clean.groupby("layout_id")["length_m"]
        .agg(laps="size", median_m="median", p10_m=_q(0.1), p90_m=_q(0.9))
        .reset_index()
    )
    spread["spread_pct"] = 100.0 * (spread["p90_m"] - spread["p10_m"]) / spread["median_m"]
    spread["official_m"] = spread["layout_id"].map(OFFICIAL_LENGTH_M)
    spread["vs_official_pct"] = 100.0 * (spread["median_m"] / spread["official_m"] - 1.0)
    rep.items.append(
        GateItem(
            "Lap length = session row",
            bool(off.empty),
            f"{len(per_session) - len(off)} of {len(per_session)} sessions inside",
            f"median of the session's full laps within {SESSION_LENGTH_MAX:.1%} of `lap_length_m`",
            md_table(off, floats="{:.5g}"),
        )
    )
    rep.items.append(
        GateItem(
            "Lap length spread",
            bool((spread["spread_pct"] <= 100.0 * LENGTH_SPREAD_MAX).all()),
            "largest " + f"{spread['spread_pct'].max():.2f} %",
            f"P90 − P10 ≤ {LENGTH_SPREAD_MAX:.0%} per layout",
            md_table(spread, floats="{:.5g}"),
        )
    )
    known = spread.dropna(subset=["official_m"])
    rep.items.append(
        GateItem(
            "Layout is the declared one",
            bool((known["vs_official_pct"].abs() <= 100.0 * OFFICIAL_LENGTH_MAX).all()),
            ", ".join(f"{r.layout_id} {r.vs_official_pct:+.2f} %" for r in known.itertuples()),
            f"median within {OFFICIAL_LENGTH_MAX:.0%} of the official length",
            f"{int((index['clean_reason'] == 'length').sum())} full laps fail the 2 % length "
            "test against their layout's median and are not clean.",
        )
    )

    # 5. No unflagged clean lap over 3 g.
    over_g = clean[
        ~clean["flagged"]
        & ((clean["peak_position_g"] > PEAK_G_MAX) | (clean["peak_doppler_g"] > PEAK_G_MAX))
    ]
    flagged = clean[clean["flagged"]]
    cells = (
        clean.groupby(["layout_id", "car"])
        .agg(clean_laps=("flagged", "size"), flagged=("flagged", "sum"))
        .reset_index()
    )
    cells["share_pct"] = 100.0 * cells["flagged"] / cells["clean_laps"]
    cols = ["session_id", "lap_num", "spike_g", "spike_s_m", "peak_position_g", "peak_doppler_g"]
    rep.items.append(
        GateItem(
            "No unflagged clean lap over 3 g",
            bool(over_g.empty),
            f"{len(over_g)} laps",
            f"|V²κ| ≤ {PEAK_G_MAX:g} g over {geo.SMOOTH_S} s, position and Doppler",
            md_table(over_g[cols], floats="{:.3g}"),
        )
    )
    share_flagged = len(flagged) / max(len(clean), 1)
    over_cells = cells[cells["share_pct"] > 100.0 * FLAGGED_CELL_MAX]
    rep.items.append(
        GateItem(
            "Flagged laps",
            bool(share_flagged <= FLAGGED_OVERALL_MAX),
            f"{len(flagged)} of {len(clean)} clean laps ({share_flagged:.2%}); "
            f"{len(over_cells)} cells over {FLAGGED_CELL_MAX:.0%}"
            + (
                " ("
                + ", ".join(
                    f"{car} at {lay} {int(n_flag)} of {int(n_clean)}"
                    for car, lay, n_flag, n_clean in zip(
                        over_cells["car"],
                        over_cells["layout_id"],
                        over_cells["flagged"],
                        over_cells["clean_laps"],
                    )
                )
                + ")"
                if len(over_cells)
                else ""
            ),
            f"≤ {FLAGGED_OVERALL_MAX:.0%} overall; a car and layout over "
            f"{FLAGGED_CELL_MAX:.0%} is listed for a cause",
            md_table(cells[cells["flagged"] > 0], floats="{:.3g}")
            + "\n\n"
            + md_table(flagged[cols], floats="{:.3g}"),
        )
    )

    # 6. Every lap maps to a crossing interval or has a cause.
    causes = unmatched_causes(index)
    unexplained = causes[causes["cause"] == "unexplained"]
    n_edge = int((clean["edge"]).sum())
    rep.items.append(
        GateItem(
            "Laps map to crossing intervals",
            bool(unexplained.empty),
            f"{len(causes)} full laps without an interval, {len(unexplained)} unexplained; "
            f"{n_edge} clean laps use an extrapolated crossing",
            "every full lap matches one interval or has a cause",
            "Causes: "
            + ", ".join(f"{k} {v}" for k, v in causes["cause"].value_counts().items())
            + ".\n\n"
            + md_table(unexplained, floats="{:.4g}"),
        )
    )

    # 7. Gyro scale.
    gyro = clean.dropna(subset=["gyro_scale"])
    by_era = (
        gyro.groupby(["car", "logger_id"])
        .agg(
            laps=("gyro_scale", "size"),
            first=("date", "min"),
            last=("date", "max"),
            p10=("gyro_scale", _q(0.1)),
            median=("gyro_scale", "median"),
            p90=("gyro_scale", _q(0.9)),
        )
        .reset_index()
    )
    by_era["spread_pct"] = 100.0 * (by_era["p90"] - by_era["p10"])
    rep.items.append(
        GateItem(
            "Gyro scale spread",
            bool((by_era["spread_pct"] <= 100.0 * GYRO_SPREAD_MAX).all()),
            f"largest {by_era['spread_pct'].max():.2f} % (per car and logger)",
            f"P10–P90 ≤ {GYRO_SPREAD_MAX:.0%} per car and era",
            md_table(by_era, floats="{:.4g}"),
        )
    )

    # 8. Build population.
    pop = (
        build.groupby(["layout_id", "car"])
        .agg(
            laps=("lap_num", "size"),
            sessions=("session_id", "nunique"),
            days=("date", "nunique"),
            reference_pace_s=("reference_pace_s", "first"),
        )
        .reset_index()
    )
    pop["votes_in_B3"] = np.where(pop["days"] >= MIN_VOTE_DAYS, "yes", "no")
    reasons = index["build_reason"].replace("", "in_build").value_counts()
    rep.items.append(
        GateItem(
            "Build population",
            None,
            f"{len(build)} laps, {build['session_id'].nunique()} sessions; "
            f"{int((pop['votes_in_B3'] == 'yes').sum())} of {len(pop)} cells have "
            f"{MIN_VOTE_DAYS} days or more",
            "reported",
            md_table(pop, floats="{:.5g}")
            + "\n\nEvery lap by outcome: "
            + ", ".join(f"{k} {v}" for k, v in reasons.items())
            + ".\n\nSessions by condition: "
            + ", ".join(
                f"{k} {v}" for k, v in index.groupby("condition")["session_id"].nunique().items()
            )
            + ".",
        )
    )
    return rep


def write_report(report: GateReport, out_dir: Path) -> Path:
    """Write the report as ``<out_dir>/<gate>_<slug>.md``; returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = {
        "B1": "geometry",
        "B2": "track_truth",
        "B3": "track_holdout",
        "C0": "frame",
        "C1": "envelope_samples",
        "C2": "car_fit",
        "C3": "car_holdout",
    }.get(report.gate, "report")
    path = out_dir / f"{report.gate}_{slug}.md"
    path.write_text(report.to_markdown())
    return path


# ------------------------------------------------------------------ gate B2

#: Whole turns of a lap: +1 clockwise, 0 for Suzuka's figure of eight.
LAYOUT_TURNS = {
    "tsukuba_2000": 1,
    "fuji_gp": 1,
    "motegi_full": 1,
    "sodegaura": 1,
    "suzuka_full": 0,
}
TRACK_LAYOUTS = tuple(LAYOUT_TURNS)
PROFILE_CLOSURE_MAX = 0.002
LINE_GAP_MAX_M = 5.0
REPROJECT_MAX_G = 0.01
INERTIAL_RMS_MAX_G = 0.06
APEX_SMOOTH_MEAN_MAX = 0.03
APEX_SMOOTH_ANY_MAX = 0.06
APEX_INERTIAL_MEAN_MAX = 0.05
LOW_KAPPA = 0.004
SIDE_MIN_KAPPA = 0.0015
SIDE_MAX = 0.02
GRADE_BETWEEN_CARS_MAX = 0.01
MIN_CAR_LAPS = 10
#: A car's own accelerometer profile is a median over its laps; under this
#: many it is too rough to gate the Doppler profile against.
INERTIAL_MIN_LAPS = 30
#: A layout gets a wet track from this many wet-basis laps on this many days.
WET_MIN_LAPS = 30
WET_MIN_DAYS = 3
DEM_GRADE_RMS_MAX = 0.01
#: Net turning between long straights: a car crossing a 150 m straight from
#: one side to the other is 3° off the track's axis.
DRAWING_TURN_MAX_DEG = 5.0
DRAWING_EDGE_MIN = 0.95
DEM_RELIEF_MAX_M = 3.0
#: The bank is held against a car's accelerometer where the car has this
#: many unflagged nodes with both.
BANK_SENSOR_MIN_NODES = 200


def _bank_row(info: dict) -> dict:
    """The line of gate B2's bank table from :func:`.bank.road_bank`'s numbers."""
    return {
        "nodes": info["nodes"],
        "filled": info["no_data"] + info["bridge"] + info["not_plane"],
        "bridge": info["bridge"],
        "not_plane": info["not_plane"],
        "layers": ", ".join(f"{k} {v:.0%}" for k, v in info["layers"].items()),
        "crown_p90": info["crown_p90"],
        "straight_sd": info["straight_sd_rad"],
        "into_corners": info["into_corners_mean_rad"],
        "into_corners_sd": info["into_corners_sd_rad"],
        "max_abs": info["max_abs_rad"],
    }


def bank_against_sensor(bank: np.ndarray, proxy_g: np.ndarray, lat_ms2: np.ndarray) -> dict | None:
    """The bank against what an accelerometer reads of it.

    Node by node, ``proxy_g`` is the ground-frame lateral less the
    accelerometer's (g, right-positive): on a road whose left side is
    higher by the angle ``bank`` the accelerometer reads that much less.
    The accelerometer has been divided by its session's audit slope,
    which takes out whatever goes with the lateral (``lat_ms2``): the
    body's roll, and with it the bank's own habit of leaning into the
    turns. So the proxy is fitted as ``slope × bank + per_g × lateral +
    constant`` and the bank is read where it does not follow the lateral:
    ``slope`` (1 where the bank is read in full), ``per_g``, the partial
    correlation of the proxy with the bank (``partial_r``), and the rms of
    the proxy with the lateral and the constant taken out, before and
    after the bank is (``rms_g``, ``rms_g_fitted``). None under
    :data:`BANK_SENSOR_MIN_NODES` nodes."""
    if len(bank) < BANK_SENSOR_MIN_NODES:
        return None
    lat = np.asarray(lat_ms2, float) / 9.80665
    rest = np.column_stack([lat, np.ones(len(bank))])
    left_p = proxy_g - rest @ np.linalg.lstsq(rest, proxy_g, rcond=None)[0]
    left_b = bank - rest @ np.linalg.lstsq(rest, bank, rcond=None)[0]
    full = np.column_stack([bank, lat, np.ones(len(bank))])
    coef = np.linalg.lstsq(full, proxy_g, rcond=None)[0]
    spread = float(np.std(left_p) * np.std(left_b))
    return {
        "nodes": int(len(bank)),
        "slope": float(coef[0]),
        "per_g": float(coef[1]),
        "partial_r": float(np.mean(left_p * left_b) / spread) if spread > 0.0 else float("nan"),
        "rms_g": float(np.std(left_p)),
        "rms_g_fitted": float(np.std(proxy_g - full @ coef)),
    }


def _corner_bins(corner: dict, ds: float, n: int) -> np.ndarray:
    a, b = int(round(corner["s_in"] / ds)) % n, int(round(corner["s_out"] / ds)) % n
    return np.arange(a, b) if a < b else np.arange(a, b + n) % n


def corner_apexes(kappa: np.ndarray, corners: list[dict], ds: float) -> np.ndarray:
    """The largest curvature in each corner's own direction (1/m, ≥ 0);
    ``kappa`` is one profile or laps × bins."""
    k = np.atleast_2d(kappa)
    n = k.shape[1]
    out = np.full((k.shape[0], len(corners)), np.nan)
    for j, c in enumerate(corners):
        with np.errstate(invalid="ignore"):
            piece = c["sign"] * k[:, _corner_bins(c, ds, n)]
            ok = np.isfinite(piece).mean(axis=1) > 0.9
            out[ok, j] = np.nanmax(piece[ok], axis=1)
    return out[0] if np.ndim(kappa) == 1 else out


def lap_curvature(course: np.ndarray, ds: float, window_m: float) -> np.ndarray:
    """``dψ/ds`` of every lap (laps × bins, bins 0..n) over ``window_m``,
    at the first n bins; NaN within half a window of the lap's ends."""
    half = window_m / 2.0 / ds
    lo, frac = int(math.floor(half)), half - math.floor(half)
    n = course.shape[1] - 1
    idx = np.arange(n)

    def _at(shift: int) -> np.ndarray:
        j = idx + shift
        ok = (j >= 0) & (j <= n)
        out = np.full((course.shape[0], n), np.nan)
        out[:, ok] = course[:, j[ok]]
        return out

    ahead = (1 - frac) * _at(lo) + frac * _at(lo + 1)
    behind = (1 - frac) * _at(-lo) + frac * _at(-lo - 1)
    return np.asarray((ahead - behind) / window_m)


def _low_share(kappa: np.ndarray, speed: np.ndarray) -> float:
    w = speed * speed * np.abs(kappa)
    return float(np.sum(w[np.abs(kappa) < LOW_KAPPA]) / np.sum(w))


def _rms(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def against_drawing(model, drawing: dict) -> tuple[dict, str]:  # type: ignore[no-untyped-def]
    """A track (``TrackModel``) against the operator's transcribed
    geometry: a summary row and the tables behind it (markdown).

    ``sequence_matches``: the corner table alternates between right and
    left as the drawing does. ``turn_worst_deg``: the largest difference
    in net turning between long straights (NaN without arc lengths).
    ``apex_vs_inside_edge_min``: the smallest ratio of a group's tightest
    driven radius to the inside-edge radius ``R − w/2`` of its tightest
    printed element (NaN when the orders differ)."""
    from . import drawings as dr

    ds, n = model.ds_m, model.n
    mine = dr.turn_sequence([c["sign"] for c in model.corners])
    theirs = dr.turn_sequence([e["sign"] for e in drawing["elements"]])
    row: dict = {
        "sequence_matches": mine == theirs,
        "groups_drawing": len(theirs),
        "groups_profile": len(mine),
        "disclaimed": bool(drawing.get("dimensions_disclaimed", False)),
        "turn_pieces": 0,
        "turn_worst_deg": float("nan"),
        "apex_vs_inside_edge_min": float("nan"),
    }
    parts: list[str] = []
    pieces = dr.straight_to_straight(drawing)
    if pieces:
        scale = model.length_m / drawing["length_m"]
        rows = []
        for piece in pieces:
            a = int(round(piece["station_from_m"] * scale / ds))
            b = int(round(piece["station_to_m"] * scale / ds))
            got = float(np.degrees(np.sum(model.kappa_heat[np.arange(a, b) % n]) * ds))
            rows.append(
                {
                    "station_from_m": piece["station_from_m"],
                    "station_to_m": piece["station_to_m"],
                    "drawing_deg": piece["turn_deg"],
                    "profile_deg": got,
                    "difference_deg": got - piece["turn_deg"],
                }
            )
        frame = pd.DataFrame(rows)
        row["turn_pieces"] = len(frame)
        row["turn_worst_deg"] = float(
            frame["difference_deg"].iloc[frame["difference_deg"].abs().argmax()]
        )
        parts.append("Net turning between long straights:\n\n" + md_table(frame, floats="{:.4g}"))
    if mine == theirs:
        ours = dr.groups(
            [{"sign": c["sign"], "r": 1.0 / abs(c["kappa_apex"])} for c in model.corners]
        )
        rows = []
        for got_group, drawn in zip(ours, dr.groups(drawing["elements"])):
            edge = dr.inside_edge_radius(drawn, drawing["width_m"][1])
            tight = min(g["r"] for g in got_group)
            rows.append(
                {
                    "direction": "R" if drawn[0]["sign"] > 0 else "L",
                    "printed_radii_m": ", ".join(
                        "?" if e["radius_m"] is None else f"{e['radius_m']:g}" for e in drawn
                    ),
                    "driven_tightest_m": tight,
                    "inside_edge_m": edge if edge is not None else float("nan"),
                    "ratio": tight / edge if edge else float("nan"),
                }
            )
        frame = pd.DataFrame(rows)
        if frame["ratio"].notna().any():
            row["apex_vs_inside_edge_min"] = float(frame["ratio"].min())
        parts.append(
            "Tightest driven apex per group of corners:\n\n" + md_table(frame, floats="{:.3g}")
        )
    return row, "\n\n".join(parts)


def wet_against_dry(wet, dry, shares: dict) -> dict:  # type: ignore[no-untyped-def]
    """A layout's wet track beside its dry one (two ``TrackBuild``): how
    the wet one closes, how far its line lies from the dry line, how its
    apexes compare in the dry track's corners, and the heat difference at
    the dry speed trace (the largest per tyre, through the force shares
    of the dry build's commonest car where production has them)."""
    from . import holdout
    from . import track as trk

    dm, wm = dry.model, wet.model
    n, ds = dm.n, dm.ds_m
    frac = np.arange(n) / n
    at = np.arange(wm.n) / wm.n
    # one lap of the wet track, stretched onto the dry track's length
    k_wet = np.interp(frac, at, wm.kappa_heat, period=1.0) * (wm.length_m / dm.length_m)
    wx = np.interp(frac, at, wm.x_m, period=1.0)
    wy = np.interp(frac, at, wm.y_m, period=1.0)
    dx, dy = np.concatenate([dm.x_m, dm.x_m[:1]]), np.concatenate([dm.y_m, dm.y_m[:1]])
    _, off = trk.project_onto(dx, dy, wx, wy, np.arange(n) * ds, closed=True)
    dlog = np.log(
        corner_apexes(k_wet, dm.corners, ds) / corner_apexes(dm.kappa_heat, dm.corners, ds)
    )
    out = {
        "length_m": wm.length_m,
        "turns": wm.turns,
        "closure_turns": wet.closure_turns,
        "line_gap_m": wet.line_gap_m,
        "line_shift_rms_m": float(np.sqrt(np.mean(off**2))),
        "line_shift_max_m": float(np.max(np.abs(off))),
        "apex_kappa_mean": float(np.mean(dlog)) if len(dlog) else float("nan"),
        "apex_kappa_min": float(np.min(dlog)) if len(dlog) else float("nan"),
        "apex_kappa_max": float(np.max(dlog)) if len(dlog) else float("nan"),
        "heat_same_speed": float("nan"),
    }
    cars = [lap.car for lap in dry.laps]
    try:
        lens = holdout.share_for(max(set(cars), key=cars.count), shares)
    except KeyError:
        return out
    res = holdout.compare_profiles(k_wet, dm.kappa_heat, dry.speed_median, ds, lens)
    out["heat_same_speed"] = res["dq_max"]
    return out


def check_track_truth(
    root: Path,
    *,
    cache_dir: Path | None = None,
    tracks_dir: Path | None = None,
    layouts: tuple[str, ...] = TRACK_LAYOUTS,
    dem: object | None = None,
    drawings: dict[str, dict] | None = None,
) -> GateReport:
    """Gate B2: the profile is right in scale, apex curvature and gentle
    curvature, against sensors it was not built from. ``dem`` (an object
    with ``elevation(lat, lon) -> (height, layer)``, see :mod:`.dem`) adds
    the altitude check against the ground elevation model and gives each
    track its bank (:mod:`.bank`); ``drawings`` (layout -> the operator's
    transcribed geometry, see :mod:`.drawings`) adds the check against the
    published corners."""
    from ...tire_etl.kinematics import G_MS2
    from . import bank as bankmod
    from . import track as trk

    rep = GateReport("B2", "the track profile (`lapsim check-track-truth`)")
    assembly, inertial, apex, low, side, grade_rows, fills, by_year = [], [], [], [], [], [], [], []
    dem_rows: list[dict] = []
    notches: list[dict] = []
    bank_rows: list[dict] = []
    bank_sensor: list[dict] = []
    wet_rows: list[dict] = []
    pit_rows: list[dict] = []
    draw_rows: list[dict] = []
    draw_detail: list[str] = []
    index = geo.clean_lap_index(root, cache_dir=cache_dir)
    try:
        from . import holdout

        force_shares = holdout.production_shares(root)
    except FileNotFoundError:
        force_shares = {}
    for layout in layouts:
        build = trk.build_track_model(root, layout, cache_dir=cache_dir, measure_convergence=True)
        model, proj = build.model, build.proj
        from . import pit as pitmod

        model.meta["pit"] = pitmod.pit_exit(root, build)
        pit_rows.append({"layout": layout, **model.meta["pit"]})
        # The road's bank, from the elevation model across the track's line.
        banked = bankmod.attach(model, dem)  # type: ignore[arg-type]
        if banked is not None:
            bank_rows.append({"layout": layout, "basis": "dry", **_bank_row(banked.info)})
        if tracks_dir is not None:
            model.save(Path(tracks_dir) / f"{layout}.json")
        ds, n = model.ds_m, model.n
        cars = np.array([lap.car for lap in build.laps])
        days = np.array([lap.date for lap in build.laps])
        speed = build.speed_median
        made = model.meta["built_from"]
        assembly.append(
            {
                "layout": layout,
                "laps": made["n_laps"],
                "sessions": made["n_sessions"],
                "days": made["n_days"],
                "length_m": model.length_m,
                "turns": model.turns,
                "closure_turns": build.closure_turns,
                "line_gap_m": build.line_gap_m,
                "reproject_g": build.reproject_rms_g,
                "corners": len(model.corners),
            }
        )
        for f in model.fills:
            fills.append({"layout": layout, **f})

        # Against the inertial profile: every car's laps, then car by car.
        in_fill = np.zeros(n, dtype=bool)
        for f in model.fills:
            a, b = int(round(f["s_from"] / ds)), int(round(f["s_to"] / ds))
            in_fill[(np.arange(a, b + 1) if a <= b else np.arange(a, b + 1 + n) % n)] = True
        groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(cars), dtype=bool))]
        groups += [(c, cars == c) for c in np.unique(cars)]
        k_in_all = None
        for name, rows in groups:
            k_in, count = trk.inertial_kappa(build, rows)
            if name == "all":
                k_in_all = k_in
            n_imu = int(np.isfinite(proj.lat_g[rows]).any(axis=1).sum())
            if n_imu < trk.MIN_BIN_LAPS or not np.isfinite(k_in).any():
                continue
            with np.errstate(invalid="ignore"):
                v = trk._fill_nan_periodic(geo.nan_median(proj.speed[rows][:, :-1], 0))
            # A car is compared on its own line: its laps' Doppler profile
            # against its laps' accelerometer. Against the pooled track the
            # difference would also hold the car's line, which is B3's.
            own = model.kappa_heat
            if name != "all":
                own = trk._kappa_from_course(proj.course[rows], ds)[0]
            d = v * v * (own - k_in) / G_MS2
            if banked is not None and name != "all":
                # what the accelerometer reads of the bank, on sensors it was not made from
                keep = (banked.flag == 0) & ~in_fill & np.isfinite(d)
                row_b = bank_against_sensor(banked.left_rad[keep], d[keep], (v * v * own)[keep])
                if row_b is not None:
                    bank_sensor.append({"layout": layout, "car": name, "imu_laps": n_imu, **row_b})
            stretch = trk.lateral_difference_g(own, k_in, v, ds, trk.FILL_STRETCH_M)
            stretch = np.where(in_fill | ~np.isfinite(k_in), 0.0, stretch)
            worst = int(np.argmax(np.abs(stretch)))
            inertial.append(
                {
                    "layout": layout,
                    "car": name,
                    "imu_laps": n_imu,
                    "gated": "yes" if n_imu >= INERTIAL_MIN_LAPS else "no",
                    "rms_g": _rms(d),
                    "worst_20m_g": float(stretch[worst]),
                    "at_s_m": worst * ds,
                }
            )

        # Apexes: against the unsmoothed course, the accelerometer and the laps' own.
        # What the 10 m window takes off an apex. A boxcar of width w lowers
        # a peak by an amount that goes with w², so the profile over twice
        # the window loses four times as much and a third of the difference
        # between the two is the loss of the first. (The peak of the course
        # differenced over one bin is no reference: the largest of a
        # corner's noisy bins reads 2–5 % high.)
        wide = trk.curvature(build.psi, ds, model.turns, 2.0 * trk.KAPPA_WINDOW_M)
        a_track = corner_apexes(build.kappa_doppler, model.corners, ds)
        a_wide = corner_apexes(wide, model.corners, ds)
        a_in = np.full(len(model.corners), np.nan)
        if k_in_all is not None:
            # the accelerometer profile at the track's own bandwidth
            smooth = trk._periodic_mean(
                trk._fill_nan_periodic(k_in_all), int(round(trk.KAPPA_WINDOW_M / 2.0 / ds))
            )
            a_in = corner_apexes(smooth, model.corners, ds)
        k_laps = lap_curvature(proj.course, ds, trk.KAPPA_WINDOW_M)
        with np.errstate(invalid="ignore"):
            a_laps = geo.nan_median(corner_apexes(k_laps, model.corners, ds), 0)
        for j, c in enumerate(model.corners):
            apex.append(
                {
                    "layout": layout,
                    "s_apex_m": c["s_apex"],
                    "sign": c["sign"],
                    "turn_deg": c["turn_angle"],
                    "radius_m": 1.0 / a_track[j],
                    "apex_loss": math.log(a_track[j] / a_wide[j]) / 3.0,
                    "dlog_vs_accelerometer": (
                        math.log(a_track[j] / a_in[j]) if a_in[j] > 0 else float("nan")
                    ),
                    "peak_retention": a_track[j] / a_laps[j],
                }
            )

        # Gentle curvature and its day-split floor.
        uniq = np.unique(days)
        halves = [np.isin(days, uniq[i::2]) for i in (0, 1)]
        shares = [
            _low_share(trk._kappa_from_course(proj.course[h], ds)[0], speed)
            for h in halves
            if h.sum() >= trk.MIN_BIN_LAPS
        ]
        low.append(
            {
                "layout": layout,
                "share_below_0.004": _low_share(model.kappa_heat, speed),
                "day_split_floor": abs(shares[0] - shares[1]) if len(shares) == 2 else np.nan,
            }
        )

        # Right against left, per car, at the laps' own speed.
        v_laps = proj.speed[:, :-1]
        keep = np.abs(model.kappa_heat) >= SIDE_MIN_KAPPA
        for car in np.unique(cars):
            rows = cars == car
            if rows.sum() < MIN_CAR_LAPS:
                continue
            vals = []
            for kl, vl in zip(k_laps[rows], v_laps[rows]):
                ok = keep & np.isfinite(kl) & np.isfinite(vl)
                if ok.mean() < 0.3:
                    continue
                w = vl[ok] ** 2
                kt, ko = model.kappa_heat[ok], kl[ok]
                right = np.sum(w * np.maximum(kt, 0)) / np.sum(w * np.maximum(ko, 0))
                left = np.sum(w * np.maximum(-kt, 0)) / np.sum(w * np.maximum(-ko, 0))
                vals.append(math.log(right) - math.log(left))
            if vals:
                side.append(
                    {
                        "layout": layout,
                        "car": car,
                        "laps": len(vals),
                        "right_minus_left": float(np.median(vals)),
                    }
                )

        # Grade between cars.
        grades = {}
        for car in np.unique(cars):
            rows = np.flatnonzero(cars == car)
            if rows.size < MIN_CAR_LAPS:
                continue
            sub = trk.Projection(
                grid=proj.grid,
                **{
                    k: getattr(proj, k)[rows]
                    for k in ("course", "speed", "offset", "alt", "time", "lat_g")
                },
            )
            grades[car] = trk._altitude(sub, [build.laps[i] for i in rows], ds)[1]
        names = sorted(grades)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                grade_rows.append(
                    {
                        "layout": layout,
                        "cars": f"{a} / {b}",
                        "grade_rms": _rms(grades[a] - grades[b]),
                    }
                )

        # Altitude against the ground elevation model, along the track's own line.
        if dem is not None:
            lat, lon = geo.local_lat_lon(model.x_m, model.y_m, model.gate["lat"], model.gate["lon"])
            ground, layer = dem.elevation(lat, lon)  # type: ignore[attr-defined]
            if np.isfinite(ground).mean() > 0.9:
                ground = trk._periodic_mean(
                    trk._fill_nan_periodic(ground), int(round(trk.ALT_SMOOTH_M / 2.0 / ds))
                )
                half = max(int(round(trk.GRADE_WINDOW_M / 2.0 / ds)), 1)
                g_dem = (np.roll(ground, -half) - np.roll(ground, half)) / (2 * half * ds)
                # A bridge: the bare-earth model steps away from the logged
                # altitude and back within a few tens of metres.
                zones, departure = bankmod.notch_zones(ground, model.alt_m, ds)
                masked = np.zeros(n, dtype=bool)
                for a, b in zones:
                    idx = bankmod.zone_nodes((a, b), n)
                    masked[idx] = True
                    notches.append(
                        {
                            "layout": layout,
                            "s_from_m": a * ds,
                            "s_to_m": b * ds,
                            "dem_minus_logged_m": float(
                                departure[idx][np.argmax(np.abs(departure[idx]))]
                            ),
                        }
                    )
                keep_alt = ~masked
                dem_rows.append(
                    {
                        "layout": layout,
                        "dem": ", ".join(sorted({str(v) for v in layer if v})),
                        "grade_rms": _rms((model.grade - g_dem)[keep_alt]),
                        "relief_logged_m": float(np.ptp(model.alt_m[keep_alt])),
                        "relief_dem_m": float(np.ptp(ground[keep_alt])),
                        "grade_max_logged": float(np.max(model.grade)),
                        "grade_max_dem": float(np.max(g_dem[keep_alt])),
                        "grade_min_logged": float(np.min(model.grade)),
                        "grade_min_dem": float(np.min(g_dem[keep_alt])),
                        "vertical_g_min": float(
                            np.min((speed**2 * model.kappa_v / G_MS2)[keep_alt])
                        ),
                        "vertical_g_max": float(
                            np.max((speed**2 * model.kappa_v / G_MS2)[keep_alt])
                        ),
                        "vertical_g_rms_vs_dem": _rms(
                            (
                                speed**2
                                * (model.kappa_v - trk.vertical_curvature(g_dem, ds))
                                / G_MS2
                            )[keep_alt]
                        ),
                        "notches": len(zones),
                    }
                )

        # Against the operator's drawing.
        if drawings and layout in drawings:
            row_d, text = against_drawing(model, drawings[layout])
            draw_rows.append({"layout": layout, **row_d})
            draw_detail.append(f"**{layout}** ({drawings[layout]['source']})\n\n{text}")

        # The wet version of the track, where the layout has the laps for one.
        # Every car's wet laps are pooled, as the dry track pools its cars.
        wet_laps = index[index["in_build_wet"] & (index["layout_id"] == layout)]
        if len(wet_laps):
            row: dict = {
                "layout": layout,
                "laps": int(len(wet_laps)),
                "sessions": int(wet_laps["session_id"].nunique()),
                "days": int(wet_laps["date"].nunique()),
                "cars": ", ".join(sorted(str(c) for c in wet_laps["car"].unique())),
                "built": "no",
            }
            if len(wet_laps) >= WET_MIN_LAPS and row["days"] >= WET_MIN_DAYS:
                wet = trk.build_track_model(
                    root, layout, cache_dir=cache_dir, basis="wet", inertial=False, n_boot=0
                )
                wet_bank = bankmod.attach(wet.model, dem)  # type: ignore[arg-type]
                if wet_bank is not None:
                    bank_rows.append({"layout": layout, "basis": "wet", **_bank_row(wet_bank.info)})
                if tracks_dir is not None:
                    wet.model.save(Path(tracks_dir) / f"{layout}__wet.json")
                row.update(built="yes", **wet_against_dry(wet, build, force_shares))
            wet_rows.append(row)

        # Motegi was resurfaced for 2026: the profile per year before pooling.
        years = np.array([d[:4] for d in days])
        if layout == "motegi_full" and len(np.unique(years)) > 1:
            per = {
                y: trk._kappa_from_course(proj.course[years == y], ds)[0]
                for y in np.unique(years)
                if (years == y).sum() >= trk.MIN_BIN_LAPS
            }
            ys = sorted(per)
            for i, a in enumerate(ys):
                for b in ys[i + 1 :]:
                    by_year.append(
                        {
                            "layout": layout,
                            "years": f"{a} / {b}",
                            "laps": f"{int((years == a).sum())} / {int((years == b).sum())}",
                            "v2_dkappa_rms_g": _rms(speed * speed * (per[a] - per[b]) / G_MS2),
                        }
                    )

    asm = pd.DataFrame(assembly)
    closes = (
        (asm["closure_turns"].abs() <= PROFILE_CLOSURE_MAX)
        & (asm["line_gap_m"] <= LINE_GAP_MAX_M)
        & (asm["turns"] == asm["layout"].map(LAYOUT_TURNS))
    )
    rep.items.append(
        GateItem(
            "Profile closes",
            bool(closes.all()),
            f"closure up to {asm['closure_turns'].abs().max():.5f} turns, line gap "
            f"{asm['line_gap_m'].min():.1f}–{asm['line_gap_m'].max():.1f} m",
            f"∮κ ds within {PROFILE_CLOSURE_MAX} turns of n; line closes within "
            f"{LINE_GAP_MAX_M:g} m; n as expected",
            md_table(asm, floats="{:.5g}"),
        )
    )
    rep.items.append(
        GateItem(
            "Re-referencing converged",
            bool((asm["reproject_g"] <= REPROJECT_MAX_G).all()),
            f"one more re-projection moves V²κ by {asm['reproject_g'].min():.4f}–"
            f"{asm['reproject_g'].max():.4f} g rms",
            f"≤ {REPROJECT_MAX_G} g rms",
        )
    )
    ine = pd.DataFrame(inertial)
    if ine.empty:
        rep.items.append(
            GateItem("Against the inertial profile", None, "no IMU-sourced laps", "see #204")
        )
    else:
        ok_in = (ine["rms_g"] <= INERTIAL_RMS_MAX_G) & (
            ine["worst_20m_g"].abs() <= trk.FILL_THRESHOLD_G
        )
        ok_in |= ine["gated"] == "no"
        whole = ine[ine["car"] == "all"]
        rep.items.append(
            GateItem(
                "Against the inertial profile",
                bool(ok_in.all()),
                f"per layout rms {whole['rms_g'].min():.3f}–{whole['rms_g'].max():.3f} g; "
                f"{int((~ok_in).sum())} of {int((ine['gated'] == 'yes').sum())} gated rows outside",
                f"rms ≤ {INERTIAL_RMS_MAX_G} g per layout, and per car on its own laps where it "
                f"has {INERTIAL_MIN_LAPS} IMU-sourced laps; no {trk.FILL_STRETCH_M:g} m over "
                f"{trk.FILL_THRESHOLD_G} g outside a fill",
                md_table(ine, floats="{:.3g}"),
            )
        )
    fl = pd.DataFrame(fills)
    rep.items.append(
        GateItem(
            "Fills",
            None,
            f"{len(fl)} stretches replaced from the accelerometer profile",
            "reported",
            md_table(fl, floats="{:.4g}") if len(fl) else "",
        )
    )
    apx = pd.DataFrame(apex)
    by_lay = apx.groupby("layout").agg(
        corners=("s_apex_m", "size"),
        smooth_mean_abs=("apex_loss", lambda s: s.abs().mean()),
        smooth_worst=("apex_loss", lambda s: s.abs().max()),
        accel_mean=("dlog_vs_accelerometer", "mean"),
        retention_min=("peak_retention", "min"),
        retention_median=("peak_retention", "median"),
    )
    rep.items.append(
        GateItem(
            "Apexes not flattened",
            bool(
                (by_lay["smooth_mean_abs"] <= APEX_SMOOTH_MEAN_MAX).all()
                and (by_lay["smooth_worst"] <= APEX_SMOOTH_ANY_MAX).all()
            ),
            f"mean |Δlog κ| {by_lay['smooth_mean_abs'].min():.3f}–"
            f"{by_lay['smooth_mean_abs'].max():.3f}, worst corner "
            f"{by_lay['smooth_worst'].max():.3f}",
            f"apex κ lost to the 10 m window: mean ≤ {APEX_SMOOTH_MEAN_MAX}, any corner "
            f"≤ {APEX_SMOOTH_ANY_MAX}",
            md_table(by_lay.reset_index(), floats="{:.3g}"),
        )
    )
    rep.items.append(
        GateItem(
            "Apex κ against the accelerometer",
            (
                None
                if by_lay["accel_mean"].isna().all()
                else bool((by_lay["accel_mean"].dropna().abs() <= APEX_INERTIAL_MEAN_MAX).all())
            ),
            "mean log ratio per layout "
            + ", ".join(f"{k} {v:+.3f}" for k, v in by_lay["accel_mean"].items()),
            f"mean within ± {APEX_INERTIAL_MEAN_MAX}",
            md_table(apx, floats="{:.3g}"),
        )
    )
    rep.items.append(
        GateItem(
            "Peak retention",
            None,
            f"profile apex over the laps' own, median per layout "
            f"{by_lay['retention_median'].min():.3f}–{by_lay['retention_median'].max():.3f}, "
            f"lowest corner {by_lay['retention_min'].min():.3f}",
            "reported (for the car model, #205)",
        )
    )
    rep.items.append(
        GateItem(
            "Share of lateral heat under 0.004 /m",
            None,
            ", ".join(f"{r['layout']} {r['share_below_0.004']:.3f}" for r in low),
            "reported, with the day-split floor",
            md_table(pd.DataFrame(low), floats="{:.3g}"),
        )
    )
    sd = pd.DataFrame(side)
    rep.items.append(
        GateItem(
            "Right against left at the laps' own speed",
            None,
            f"{int((sd['right_minus_left'].abs() > SIDE_MAX).sum())} of {len(sd)} layout × car "
            f"rows over {SIDE_MAX}; largest {sd['right_minus_left'].abs().max():.3f}",
            f"reported against {SIDE_MAX} (asked by #206)",
            md_table(sd, floats="{:.3g}"),
        )
    )
    gr = pd.DataFrame(grade_rows)
    rep.items.append(
        GateItem(
            "Grade between cars",
            bool((gr["grade_rms"] <= GRADE_BETWEEN_CARS_MAX).all()) if len(gr) else None,
            (
                f"{100 * gr['grade_rms'].min():.2f}–{100 * gr['grade_rms'].max():.2f} % rms"
                if len(gr)
                else "no layout with two cars"
            ),
            f"≤ {GRADE_BETWEEN_CARS_MAX:.0%} rms",
            md_table(gr, floats="{:.3g}"),
        )
    )
    if by_year:
        yr = pd.DataFrame(by_year)
        rep.items.append(
            GateItem(
                "Motegi by year",
                None,
                ", ".join(f"{r.years}: {r.v2_dkappa_rms_g:.3f} g" for r in yr.itertuples()),
                "reported (resurfaced for 2026)",
                md_table(yr, floats="{:.3g}"),
            )
        )
    if dem_rows:
        dm = pd.DataFrame(dem_rows)
        relief_gap = (dm["relief_logged_m"] - dm["relief_dem_m"]).abs()
        rep.items.append(
            GateItem(
                "Altitude against the GSI DEM",
                bool(
                    (dm["grade_rms"] <= DEM_GRADE_RMS_MAX).all()
                    and (relief_gap <= DEM_RELIEF_MAX_M).all()
                ),
                f"grade {100 * dm['grade_rms'].min():.2f}–{100 * dm['grade_rms'].max():.2f} % rms; "
                f"relief within {relief_gap.max():.1f} m; {len(notches)} bridge notches masked",
                f"grade over {trk.GRADE_WINDOW_M:g} m within {DEM_GRADE_RMS_MAX:.0%} rms; relief "
                f"within {DEM_RELIEF_MAX_M:g} m",
                md_table(dm, floats="{:.3g}")
                + "\n\nStretches masked (the bare-earth model against the logged altitude):\n\n"
                + md_table(pd.DataFrame(notches), floats="{:.4g}"),
            )
        )
    else:
        rep.items.append(GateItem("Altitude against the GSI DEM", None, "not measured", "see #204"))
    if bank_rows:
        bk = pd.DataFrame(bank_rows)
        dry = bk[bk["basis"] == "dry"]
        detail = (
            "The cross-slope of the ground over "
            f"± {bankmod.HALF_WIDTH_M:g} m across the track's line, ± {bankmod.SMOOTH_M:g} m mean "
            "along it; positive in `into_corners` leans into the turn. `filled`: nodes read from "
            "their neighbours (no data, a bridge, a section that is not a plane). `crown_p90`: "
            "how far the two halves of a section differ in slope at a tenth of the nodes. "
            "`straight_sd`: the scatter along the straights about each straight's own mean, an "
            "upper bound of the noise.\n\n" + md_table(bk, floats="{:.3g}")
        )
        enough = [r for r in bank_sensor if r["imu_laps"] >= INERTIAL_MIN_LAPS]
        if bank_sensor:
            detail += (
                "\n\nAgainst each car's accelerometer on its own laps: ground-frame lateral less "
                "the accelerometer's, node by node, fitted as slope × bank + `per_g` × lateral + a "
                "constant. A slope of 1 is the bank read in full. The accelerometer is already "
                "divided by its session's audit slope, which takes out what goes with the "
                "lateral, so the bank is read where it does not follow the lateral; `rms_g` is "
                "the proxy with the lateral and the constant out, `rms_g_fitted` with the bank "
                f"out too. A car under {INERTIAL_MIN_LAPS} IMU laps is listed and not quoted."
                "\n\n" + md_table(pd.DataFrame(bank_sensor), floats="{:.3g}")
            )
        rep.items.append(
            GateItem(
                "Bank from the GSI DEM",
                None,
                f"mean into the corners {dry['into_corners'].min():.3f}–"
                f"{dry['into_corners'].max():.3f} rad; scatter on the straights "
                f"{dry['straight_sd'].min():.3f}–{dry['straight_sd'].max():.3f} rad; "
                f"{int(dry['filled'].sum())} nodes filled"
                + (
                    "; accelerometer slope "
                    + ", ".join(f"{r['car']} {r['layout']} {r['slope']:.2f}" for r in enough)
                    if enough
                    else ""
                ),
                "reported (#299): the bank kept with each track; a slope of 1 against a sensor "
                "is the bank read in full",
                detail,
            )
        )
    else:
        rep.items.append(GateItem("Bank from the GSI DEM", None, "not measured", "see #299"))
    pt = pd.DataFrame(pit_rows)
    if "s_join_m" in pt.columns:
        rep.items.append(
            GateItem(
                "Pit exit",
                None,
                ", ".join(
                    f"{lay} {s_join:.0f} m after {roll:.0f} s"
                    for lay, s_join, roll in zip(pt["layout"], pt["s_join_m"], pt["rolling_s"])
                    if np.isfinite(s_join)
                ),
                "reported (for the out-lap, #206): where an out-lap joins the driven line, "
                "and how long it had been rolling",
                md_table(pt, floats="{:.4g}"),
            )
        )
    if wet_rows:
        wt = pd.DataFrame(wet_rows)
        built = wt[wt["built"] == "yes"]
        wet_ok: bool | None = None
        summary = ""
        if len(built):
            wet_ok = bool(
                (
                    (built["closure_turns"].abs() <= PROFILE_CLOSURE_MAX)
                    & (built["line_gap_m"] <= LINE_GAP_MAX_M)
                    & (built["turns"] == built["layout"].map(LAYOUT_TURNS))
                ).all()
            )
            summary = "; against the dry track: " + "; ".join(
                f"{lay} line {rms:.1f} m rms, apex κ {100 * apx_mean:+.0f} %, heat at equal "
                f"speed {heat:.3f}"
                for lay, rms, apx_mean, heat in zip(
                    built["layout"],
                    built["line_shift_rms_m"],
                    built["apex_kappa_mean"],
                    built["heat_same_speed"],
                )
            )
        rep.items.append(
            GateItem(
                "Wet tracks",
                wet_ok,
                f"{len(built)} of {len(wt)} layouts with wet laps have enough for a wet track"
                + summary,
                f"built from {WET_MIN_LAPS} wet-basis laps on {WET_MIN_DAYS} days, every car "
                "pooled; a built one closes like a dry one",
                md_table(wt, floats="{:.3g}"),
            )
        )
    if draw_rows:
        dw = pd.DataFrame(draw_rows)
        gated = dw[~dw["disclaimed"]]
        ok = (
            dw["sequence_matches"].all()
            and (dw["turn_worst_deg"].fillna(0.0).abs() <= DRAWING_TURN_MAX_DEG).all()
            and (gated["apex_vs_inside_edge_min"].fillna(np.inf) >= DRAWING_EDGE_MIN).all()
        )
        rep.items.append(
            GateItem(
                "Against the operators' drawings",
                bool(ok),
                f"left/right order matches at {int(dw['sequence_matches'].sum())} of {len(dw)} "
                f"layouts; turning between long straights within "
                f"{dw['turn_worst_deg'].abs().max():.1f}°; tightest driven apex at "
                f"{gated['apex_vs_inside_edge_min'].min():.2f} × the inside-edge radius or wider",
                f"the order of lefts and rights exact; net turning between long straights "
                f"within {DRAWING_TURN_MAX_DEG:g}° where arc lengths are published; no apex "
                f"tighter than {DRAWING_EDGE_MIN} × the inside edge, R − w/2 (not gated where "
                "the operator disclaims its dimensions)",
                md_table(dw, floats="{:.3g}") + "\n\n" + "\n\n".join(draw_detail),
            )
        )
    else:
        rep.items.append(
            GateItem("Against the operators' drawings", None, "not measured", "see #204")
        )
    return rep
