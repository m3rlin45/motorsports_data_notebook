"""Gate reports of the lap simulation's track segment (issue #204).

A gate is a list of :class:`GateItem`: a statement with a threshold, the
number measured on the dataset and whether it holds. ``passed`` is None for
an item that is reported and not gated. :func:`check_geometry` is gate B1.
"""

from __future__ import annotations

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
    slug = {"B1": "geometry"}.get(report.gate, "report")
    path = out_dir / f"{report.gate}_{slug}.md"
    path.write_text(report.to_markdown())
    return path
