"""Training-side data cleaning for the tire dataset.

The extractor (:mod:`..tire_etl`) turns the AIM files into in-repo data with
the minimum processing needed for a stable, manageable dataset: it splits
laps and stints, resolves the circuit from GPS, selects and renames channels
(stored verbatim at their native rates, dataset schema 7), validates the IMU
against GPS, and writes a raw per-lap summary. Nothing under ``data/tire_dataset/`` is masked, filtered or judged.
Every decision about which samples and laps the model may learn from is made
here, at training time, so a cleaning rule can change without re-extracting.

Four steps, in the order the model applies them:

1. :func:`mask_stale_tpms` — a TPMS module that is asleep at roll-out
   reports its stored value until it wakes; that stale prefix is masked per
   corner (:func:`mask_stale_session_prefix`), and so is the stale packet a
   stint starts with after a stop in the session
   (:func:`mask_stale_stint_starts`). Both consumers of the streams read
   them through it: the lap aggregates below and the per-second fit
   (``statespace._session_timeseries``).
2. :func:`recompute_tpms_lap_aggregates` — the per-lap TPMS summary columns
   (start / end / min / max / mean, rise rate, first-valid time) are rebuilt
   from the masked timeseries, so an out-lap's ``tpms_*_start`` — the stint
   anchor — is the first *live* reading.
3. :func:`apply_filters` — ``tire_usable`` / ``exclude_reason`` per lap.
4. :func:`drifting_tpms` — a corner whose amount of gas falls against its
   siblings' through a session (a leak, or a transducer that drifts) is
   flagged from :func:`gas_drift_screen` and its TPMS columns are masked
   for that session, at training and scoring alike.

:func:`clean_laps` runs all four over a laps frame and is what every model
entry point (fit, holdout, notes validation, sensor audit, compound
inference) goes through. ``tire-model clean-laps`` writes the result to a
parquet file for ad-hoc queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..tire_etl.aggregates import compute_corner_aggregates
from ..tire_etl.dataset import load_group
from ..tire_etl.paths import laps_dir, timeseries_dir

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
TPMS_STATS = ("start", "end", "min", "max", "mean")
TPMS_TS_COLUMNS = [f"tpms_press_{c}_bar" for c in CORNERS] + [f"tpms_temp_{c}_c" for c in CORNERS]

#: One count of the TPMS pressure reading: 31.3 ± 0.2 mbar, read from the
#: streams of every TPMS kit in the dataset (logged to 0.01 bar, so a count
#: shows as a step of 0.03 or 0.04 bar; #371).
TPMS_PRESSURE_STEP_BAR = 0.0313
#: :func:`mask_stale_stint_starts`: a stint's first pressure change of this
#: many counts or more ...
STALE_RESTART_MIN_STEPS = 3
#: ... within this many seconds of the stint's first reading shows that the
#: readings before it were a held packet.
STALE_RESTART_WINDOW_S = 60.0


def tpms_lap_columns() -> list[str]:
    """The per-lap TPMS summary columns that :func:`clean_laps` rebuilds."""
    cols: list[str] = []
    for c in CORNERS:
        for stat in TPMS_STATS:
            cols += [f"tpms_press_{c}_{stat}", f"tpms_temp_{c}_{stat}"]
        cols += [
            f"tpms_press_{c}_rise_bar_per_min",
            f"tpms_temp_{c}_first_valid_s",
            f"tpms_press_{c}_first_valid_s",
        ]
    return cols


# ------------------------------------------------------------ 1. stale TPMS


def _stale_prefix_len(arr: np.ndarray) -> int:
    """Return the count of leading stale-sensor samples in ``arr``.

    A TPMS module that's still asleep reports its last pre-sleep ADC value
    bit-for-bit identical sample to sample; once it wakes up, ±1-LSB
    sensor noise gives every sample a different float value. So the
    sleeping signature is a *sustained* run of strictly-equal values at
    the head of the series.

    Walk forward from the first finite sample to the first one whose
    value differs bit-exactly. If that gap is 0 or 1 sample, the sensor
    was awake from the start — return 0. Otherwise return the length of
    the constant prefix. An entirely flat array returns its full length
    (sensor never woke up for the whole session).
    """
    if arr.size <= 1:
        return 0
    finite_mask = ~np.isnan(arr)
    if not finite_mask.any():
        return 0
    first_idx = int(np.argmax(finite_mask))
    baseline = arr[first_idx]
    for i in range(first_idx + 1, arr.size):
        if np.isnan(arr[i]):
            continue
        if arr[i] != baseline:
            return i if i - first_idx >= 2 else 0
    return arr.size  # flat from first_idx to end → all-stale


def mask_stale_session_prefix(ts: pa.Table) -> pa.Table:
    """Mask the leading stale-sensor samples in each TPMS channel as NaN.

    Operates session-wide (across every lap in this timeseries), because
    the sensor wake-up only happens once per physical run, not on every
    lap. The decision is made **per corner**: a corner's pressure and
    temperature come from one module in one radio packet, so the first
    change on *either* stream proves the module is awake, and from that
    sample on the other stream's (possibly still constant — the values are
    quantised to 1 °C / 0.03 bar) reading is a real reading. Both streams
    are therefore unmasked from the earlier of their two first-change
    points. Different corners still wake independently. The samples are
    the module's own packets at their native rate, so the first unmasked
    reading is the first live one.
    """
    new_table = ts
    for c in CORNERS:
        names = [
            n for n in (f"tpms_press_{c}_bar", f"tpms_temp_{c}_c") if n in new_table.schema.names
        ]
        if not names:
            continue
        arrays = {
            n: new_table.column(new_table.schema.get_field_index(n))
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
            for n in names
        }
        stale = {n: _stale_prefix_len(a) for n, a in arrays.items()}
        # A stream that never changes (stale_len == its length) cannot
        # vouch for the module; take the earliest *real* change.
        proven = [v for n, v in stale.items() if v < arrays[n].size]
        corner_stale = min(proven) if proven else max(stale.values())
        if corner_stale == 0:
            continue
        for n in names:
            arr = arrays[n]
            idx = new_table.schema.get_field_index(n)
            field = new_table.schema.field(idx)
            arr[:corner_stale] = np.nan
            new_col = pa.array(arr.astype(field.type.to_pandas_dtype()), type=field.type)
            new_table = new_table.set_column(idx, n, new_col)
    return new_table


def mask_stale_stint_starts(
    ts: pa.Table,
    *,
    min_steps: int = STALE_RESTART_MIN_STEPS,
    window_s: float = STALE_RESTART_WINDOW_S,
) -> pa.Table:
    """Mask the held packet a stint starts with after a stop (#371, rule A).

    The modules go silent while the car stands and wake once it rolls at
    20–30 km/h; until then the logger keeps re-sending the last packet it
    had, the hot pressure of the pit entry. :func:`mask_stale_session_prefix`
    catches that once, at the start of the recording; a stop in the middle
    of a session starts a new stint on such a packet.

    Per stint and corner, on the stint's first lap (the lowest ``lap_num``
    with its ``stint_id``): take the corner's finite pressure samples in
    time order and find the first one that differs from the first. If it
    differs by ``min_steps`` counts of :data:`TPMS_PRESSURE_STEP_BAR` or more
    (more than half a count short of it, so the threshold sits between two
    quantised step sizes) within ``window_s`` of the first sample, every
    earlier sample of that lap is masked on both of the corner's streams
    (the stint's anchor pairs them). Three counts need a true change of more than
    two, about 65 mbar or 8 K of gas, which the tyre cannot make in a minute;
    two counts a hot restart can. Pressure is unmasked at its own first
    change, not at the temperature's, so the stint's anchor is the first
    fresh pressure, read at its own time.

    ``ts`` needs ``stint_id``, ``lap_num`` and ``t_session_s`` beside the
    TPMS columns; it is returned with the same columns.
    """
    missing = [c for c in ("stint_id", "lap_num", "t_session_s") if c not in ts.schema.names]
    if missing:
        raise ValueError(f"the TPMS table lacks {missing}")
    stint = ts.column("stint_id").to_numpy(zero_copy_only=False)
    lap = ts.column("lap_num").to_numpy(zero_copy_only=False)
    t = ts.column("t_session_s").to_numpy(zero_copy_only=False).astype(np.float64)
    keyed = pd.DataFrame({"stint": stint, "lap": lap}).dropna()
    first_laps = keyed.groupby("stint")["lap"].min().to_numpy()
    in_lap = {ln: lap == ln for ln in first_laps}
    threshold = (min_steps - 0.5) * TPMS_PRESSURE_STEP_BAR
    new_table = ts
    for c in CORNERS:
        p_name, t_name = f"tpms_press_{c}_bar", f"tpms_temp_{c}_c"
        if p_name not in new_table.schema.names:
            continue
        p = new_table.column(p_name).to_numpy(zero_copy_only=False).astype(np.float64)
        stale = np.zeros(p.size, dtype=bool)
        for ln, rows in in_lap.items():
            idx = np.flatnonzero(rows & np.isfinite(p))
            if idx.size < 2:
                continue
            idx = idx[np.argsort(t[idx], kind="stable")]
            changed = np.flatnonzero(p[idx] != p[idx[0]])
            if not changed.size:
                continue
            first, fresh = idx[0], idx[changed[0]]
            if abs(p[fresh] - p[first]) >= threshold and t[fresh] - t[first] <= window_s:
                stale |= rows & (t < t[fresh])
        if not stale.any():
            continue
        for n in (p_name, t_name):
            if n not in new_table.schema.names:
                continue
            idx_col = new_table.schema.get_field_index(n)
            field = new_table.schema.field(idx_col)
            arr = new_table.column(idx_col).to_numpy(zero_copy_only=False).astype(np.float64)
            arr[stale] = np.nan
            new_col = pa.array(arr.astype(field.type.to_pandas_dtype()), type=field.type)
            new_table = new_table.set_column(idx_col, n, new_col)
    return new_table


def mask_stale_tpms(ts: pa.Table) -> pa.Table:
    """Every stale-TPMS rule, in order: the recording's wake-up prefix
    (:func:`mask_stale_session_prefix`), then the held packet at the start
    of each stint (:func:`mask_stale_stint_starts`). The one entry point for
    every consumer of the masked streams; ``ts`` as the latter needs it."""
    return mask_stale_stint_starts(mask_stale_session_prefix(ts))


# --------------------------------------------- 2. per-lap TPMS aggregates


def load_tpms_timeseries(root: Path, session_id: str) -> pa.Table | None:
    """The session's TPMS streams (plus ``lap_num`` / ``t_lap_s``), masked.

    The ``tpms`` family of the native-rate file: each corner's pressure and
    temperature are valid on the rows of that corner's own timebase (null
    elsewhere), which the per-corner masking and the lap aggregates read
    through :func:`~..tire_etl.aggregates.channel_series`. ``None`` when the
    session has no timeseries file or no TPMS channels.
    """
    ts = load_group(root, session_id, "tpms", columns=TPMS_TS_COLUMNS)
    if ts is None:
        return None
    tpms = [c for c in TPMS_TS_COLUMNS if c in ts.schema.names]
    if not tpms or "lap_num" not in ts.schema.names:
        return None
    cols = ["lap_num"] + (["t_lap_s"] if "t_lap_s" in ts.schema.names else []) + tpms
    return mask_stale_tpms(ts.select(["stint_id", "t_session_s"] + cols)).select(cols)


def recompute_tpms_lap_aggregates(laps: pd.DataFrame, ts: pa.Table) -> pd.DataFrame:
    """Rebuild one session's per-lap TPMS summary columns from ``ts``.

    ``laps`` holds that session's rows (any other columns are kept as they
    are); ``ts`` is its masked TPMS timeseries from
    :func:`load_tpms_timeseries`. Mirrors the extractor's lap summary, so
    the only difference from the committed columns is the masking.
    """
    out = laps.copy()
    lap_col = ts.column("lap_num").to_numpy()
    values: dict[str, list[float]] = {col: [] for col in tpms_lap_columns()}
    for ln in out["lap_num"].to_numpy():
        lap_ts = ts.filter(pa.array(lap_col == ln))
        for c in CORNERS:
            agg = compute_corner_aggregates(lap_ts, c)
            values[f"tpms_press_{c}_start"].append(agg.press_start)
            values[f"tpms_press_{c}_end"].append(agg.press_end)
            values[f"tpms_press_{c}_min"].append(agg.press_min)
            values[f"tpms_press_{c}_max"].append(agg.press_max)
            values[f"tpms_press_{c}_mean"].append(agg.press_mean)
            values[f"tpms_press_{c}_rise_bar_per_min"].append(agg.press_rise_bar_per_min)
            values[f"tpms_temp_{c}_start"].append(agg.temp_start)
            values[f"tpms_temp_{c}_end"].append(agg.temp_end)
            values[f"tpms_temp_{c}_min"].append(agg.temp_min)
            values[f"tpms_temp_{c}_max"].append(agg.temp_max)
            values[f"tpms_temp_{c}_mean"].append(agg.temp_mean)
            values[f"tpms_temp_{c}_first_valid_s"].append(agg.temp_first_valid_s)
            values[f"tpms_press_{c}_first_valid_s"].append(agg.press_first_valid_s)
    for col, vals in values.items():
        out[col] = np.asarray(vals, dtype=np.float64)
    return out


def _dataset_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    files = sorted(laps_dir(root).glob("*.parquet")) + sorted(
        timeseries_dir(root).glob("*/*.parquet")
    )
    sig = []
    for f in files:
        st = f.stat()
        sig.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(sig)


@lru_cache(maxsize=4)
def _cleaned_tpms_aggregates(
    root_str: str, signature: tuple[tuple[str, int, int], ...]
) -> pd.DataFrame:
    """Cleaned per-lap TPMS columns for every session with a TPMS timeseries,
    keyed by ``(session_id, lap_num)``. Cached per dataset state: the fit and
    the k-fold holdout reload the laps many times per run."""
    root = Path(root_str)
    files = sorted(laps_dir(root).glob("*.parquet"))
    key_cols = ["session_id", "lap_num"]
    if not files:
        return pd.DataFrame(columns=key_cols + tpms_lap_columns())
    laps = pa.concat_tables(
        [pq.read_table(f) for f in files], promote_options="default"
    ).to_pandas()
    parts: list[pd.DataFrame] = []
    for sid, grp in laps.groupby("session_id", sort=False):
        ts = load_tpms_timeseries(root, str(sid))
        if ts is None:
            continue
        parts.append(recompute_tpms_lap_aggregates(grp[key_cols], ts))
    if not parts:
        return pd.DataFrame(columns=key_cols + tpms_lap_columns())
    out = pd.concat(parts, ignore_index=True)
    logger.info("cleaned TPMS aggregates: %d laps in %d sessions", len(out), len(parts))
    return out


# ------------------------------------------------------- 3. lap filters


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


FILTER_INPUT_COLUMNS = (
    "lap_num",
    "stint_id",
    "lap_time_s",
    "is_outlap",
    "is_inlap",
    "on_track_s",
    "speed_kmh_mean",
)


def apply_filters(
    laps: pd.DataFrame | pa.Table, cfg: FilterConfig | None = None
) -> pd.DataFrame | pa.Table:
    """Return ``laps`` with ``tire_usable`` and ``exclude_reason`` columns filled.

    ``laps`` must contain :data:`FILTER_INPUT_COLUMNS` plus per-corner
    ``tpms_press_{c}_min`` / ``_max``; ``moving_s`` when out-laps are kept.
    The best lap used for the "too slow" rule is per ``(session_id, stint_id)``
    when a ``session_id`` column is present, else per ``stint_id``. A pyarrow
    table in gives a pyarrow table back.
    """
    cfg = cfg or FilterConfig()
    is_table = isinstance(laps, pa.Table)
    df = laps.to_pandas() if isinstance(laps, pa.Table) else laps.copy()
    n = len(df)
    reasons: list[str | None] = [None] * n
    if n:
        lap_time = df["lap_time_s"].to_numpy(dtype=np.float64)
        is_outlap = df["is_outlap"].fillna(False).to_numpy(dtype=bool)
        is_inlap = df["is_inlap"].fillna(False).to_numpy(dtype=bool)
        on_track = df["on_track_s"].to_numpy(dtype=np.float64)
        speed_mean = df["speed_kmh_mean"].to_numpy(dtype=np.float64)
        moving = df["moving_s"].to_numpy(dtype=np.float64) if "moving_s" in df.columns else on_track
        stint_ids = df["stint_id"].to_numpy()
        session_ids = (
            df["session_id"].to_numpy() if "session_id" in df.columns else np.zeros(n, dtype=int)
        )
        keys = list(zip(session_ids.tolist(), stint_ids.tolist()))

        # Per-stint best lap time (ignoring NaN and out-laps).
        best_by_stint: dict[tuple, float] = {}
        for i in range(n):
            t = lap_time[i]
            if is_outlap[i] or np.isnan(t) or t < cfg.min_lap_time_s:
                continue
            if keys[i] not in best_by_stint or t < best_by_stint[keys[i]]:
                best_by_stint[keys[i]] = t

        # TPMS stuck / missing check: need at least one corner to span > tol.
        ranges = np.full((n, len(CORNERS)), np.nan)
        for j, c in enumerate(CORNERS):
            lo_col, hi_col = f"tpms_press_{c}_min", f"tpms_press_{c}_max"
            if lo_col in df.columns and hi_col in df.columns:
                ranges[:, j] = df[hi_col].to_numpy(dtype=np.float64) - df[lo_col].to_numpy(
                    dtype=np.float64
                )

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
                best = best_by_stint.get(keys[i])
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
                finite = ranges[i][~np.isnan(ranges[i])]
                if finite.size == 0:
                    reasons[i] = "no_tpms"
                    continue
                if float(finite.max()) < cfg.tpms_stuck_tolerance_bar:
                    reasons[i] = "tpms_stuck"
                    continue

    df["tire_usable"] = np.array([r is None for r in reasons], dtype=bool)
    df["exclude_reason"] = pd.Series(reasons, index=df.index, dtype=object)
    if is_table:
        table = pa.Table.from_pandas(df, preserve_index=False)
        idx = table.schema.get_field_index("exclude_reason")
        if table.schema.field(idx).type != pa.string():
            table = table.set_column(idx, "exclude_reason", pa.array(reasons, type=pa.string()))
        return table
    return df


# --------------------------------------------- 4. a corner losing gas (#373)

#: :func:`gas_drift_screen`: a steady lap's TPMS temperature moves by at most
#: this much within the lap on every tyre (one count of the 1 °C reading) ...
DRIFT_STEADY_TEMP_RANGE_C = 1.0
#: ... and a session is judged on this many steady laps or more ...
DRIFT_MIN_STEADY_LAPS = 8
#: ... spanning this much on-track time or more.
DRIFT_MIN_SPAN_S = 900.0
#: :func:`drifting_tpms`: the chance that a sound tyre anywhere in the dataset
#: is flagged (a one-sided Bonferroni bound over the screened tyre-sessions).
DRIFT_FALSE_FLAG_RATE = 0.05
#: :func:`drifting_tpms` flags nothing on fewer screened tyre-sessions than
#: this: the sound tyres' spread is read from them.
DRIFT_MIN_SCREENED = 20
#: Gauge to absolute pressure (bar) and °C to K, for the amount of gas.
_DRIFT_P_ATM_BAR = 1.013
_DRIFT_ZERO_C_K = 273.15


def gas_drift_screen(laps: pd.DataFrame) -> pd.DataFrame:
    """Each tyre's change in its amount of gas against its siblings over a
    session's steady laps (#373).

    The amount of gas in a tyre goes as ``(P + 1.013) / (T + 273.15)`` (gauge
    bar, the sensor's °C); on steady laps it changes only by a leak or by a
    transducer that drifts, and the session's other tyres share whatever the
    sensors have in common. Per session, over its usable laps on which all
    four tyres read a finite mean pressure and temperature and a temperature
    range within the lap of :data:`DRIFT_STEADY_TEMP_RANGE_C` or less: each
    tyre's log amount of gas of the lap means, less the mean of the other
    three tyres', is fitted linearly against the session's on-track clock
    (each lap's end on the cumulative ``on_track_s`` of all its laps, in lap
    order), and the slope times the span is the tyre's change over the
    session, in %. A session is judged on :data:`DRIFT_MIN_STEADY_LAPS`
    steady laps over :data:`DRIFT_MIN_SPAN_S` or more.

    ``laps`` is a cleaned laps frame (:func:`clean_laps`' columns, every lap
    of the sessions to judge). Returns one row per judged session and tyre:
    ``session_id, corner, change_pct``, ``sibling_changes_pct`` (each of the
    other three tyres' change against the remaining two, so that a flag can
    ask for a sound reference), ``n_steady`` and ``span_s``.
    """
    cols = ["session_id", "corner", "change_pct", "sibling_changes_pct", "n_steady", "span_s"]
    need = {"session_id", "lap_num", "on_track_s", "tire_usable"}
    for c in CORNERS:
        need |= {f"tpms_press_{c}_mean", f"tpms_temp_{c}_mean"}
        need |= {f"tpms_temp_{c}_min", f"tpms_temp_{c}_max"}
    if laps.empty or not need.issubset(laps.columns):
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for sid, grp in laps.sort_values(["session_id", "lap_num"]).groupby("session_id", sort=True):
        clock = grp["on_track_s"].fillna(0.0).to_numpy(dtype=np.float64).cumsum()
        steady = grp["tire_usable"].fillna(False).to_numpy(dtype=bool).copy()
        for c in CORNERS:
            lo = grp[f"tpms_temp_{c}_min"].to_numpy(dtype=np.float64)
            hi = grp[f"tpms_temp_{c}_max"].to_numpy(dtype=np.float64)
            steady &= np.isfinite(grp[f"tpms_press_{c}_mean"].to_numpy(dtype=np.float64))
            steady &= np.isfinite(grp[f"tpms_temp_{c}_mean"].to_numpy(dtype=np.float64))
            with np.errstate(invalid="ignore"):
                steady &= hi - lo <= DRIFT_STEADY_TEMP_RANGE_C
        t = clock[steady]
        if t.size < DRIFT_MIN_STEADY_LAPS or float(t.max() - t.min()) < DRIFT_MIN_SPAN_S:
            continue
        span = float(t.max() - t.min())
        gas = {
            c: np.log(
                (grp[f"tpms_press_{c}_mean"].to_numpy(dtype=np.float64)[steady] + _DRIFT_P_ATM_BAR)
                / (grp[f"tpms_temp_{c}_mean"].to_numpy(dtype=np.float64)[steady] + _DRIFT_ZERO_C_K)
            )
            for c in CORNERS
        }

        def change(c: str, among: tuple[str, ...]) -> float:
            rel = gas[c] - np.mean([gas[o] for o in among if o != c], axis=0)
            return float(100.0 * np.polyfit(t, rel, 1)[0] * span)

        for c in CORNERS:
            others = tuple(o for o in CORNERS if o != c)
            rows.append(
                {
                    "session_id": str(sid),
                    "corner": c,
                    "change_pct": change(c, CORNERS),
                    "sibling_changes_pct": tuple(change(o, others) for o in others),
                    "n_steady": int(t.size),
                    "span_s": span,
                }
            )
    return pd.DataFrame(rows, columns=cols)


def drifting_tpms(
    screen: pd.DataFrame, false_flag_rate: float = DRIFT_FALSE_FLAG_RATE
) -> pd.DataFrame:
    """Flag the tyres that lose gas through a session (#373).

    The threshold is read from the sound tyres, not set by hand: over every
    tyre-session of ``screen`` (:func:`gas_drift_screen`) the centre ``m``
    is the median change and the scale ``s`` is 1.4826 × the median absolute
    deviation from it (robust, so a leaking tyre does not widen it). A tyre
    is flagged when its change is below ``m − z·s``, ``z`` being the
    one-sided quantile at which a sound tyre anywhere among the ``N``
    screened tyre-sessions is flagged by chance with probability
    ``false_flag_rate`` at most (Bonferroni: ``z = Φ⁻¹(1 − rate / N)``), and
    when its reference is sound: each of the other three tyres' change
    against the remaining two lies within ``z·s`` of ``m``. Only a loss is
    flagged; a tyre does not gain gas in a session. Nothing is flagged on
    fewer than :data:`DRIFT_MIN_SCREENED` tyre-sessions.

    Returns ``screen`` with ``flagged`` and the ``threshold_pct`` and
    ``reference_band_pct`` used.
    """
    out = screen.copy()
    n = len(out)
    if n < DRIFT_MIN_SCREENED:
        return out.assign(flagged=False, threshold_pct=np.nan, reference_band_pct=np.nan)
    change = out["change_pct"].to_numpy(dtype=np.float64)
    m = float(np.median(change))
    s = 1.4826 * float(np.median(np.abs(change - m)))
    z = NormalDist().inv_cdf(1.0 - false_flag_rate / n)
    band = z * s
    sound_reference = np.array(
        [max(abs(x - m) for x in sib) <= band for sib in out["sibling_changes_pct"]], dtype=bool
    )
    out["flagged"] = (change < m - band) & sound_reference
    out["threshold_pct"] = m - band
    out["reference_band_pct"] = band
    return out


@lru_cache(maxsize=4)
def _drifting_tpms_flags(
    root_str: str, signature: tuple[tuple[str, int, int], ...]
) -> frozenset[tuple[str, str]]:
    """The ``(session_id, corner)`` pairs :func:`drifting_tpms` flags over
    every session of the dataset at ``root``, cleaned by steps 1–3 with the
    default filters. Cached per dataset state."""
    root = Path(root_str)
    files = sorted(laps_dir(root).glob("*.parquet"))
    if not files:
        return frozenset()
    laps = pa.concat_tables(
        [pq.read_table(f) for f in files], promote_options="default"
    ).to_pandas()
    flags = drifting_tpms(gas_drift_screen(_clean_steps_1_to_3(root, laps, None)))
    hit = flags[flags["flagged"]]
    for sid, corner, change, threshold in zip(
        hit["session_id"], hit["corner"], hit["change_pct"], hit["threshold_pct"]
    ):
        logger.info(
            "TPMS drift: %s %s loses %.1f %% of its gas against its siblings (threshold %.1f %%)",
            sid,
            str(corner).upper(),
            -float(change),
            -float(threshold),
        )
    return frozenset(zip(hit["session_id"].astype(str), hit["corner"].astype(str)))


def mask_corner_sessions(
    laps: pd.DataFrame, pairs: frozenset[tuple[str, str]] | set
) -> pd.DataFrame:
    """``laps`` with every TPMS summary column of each ``(session_id,
    corner)`` in ``pairs`` set to NaN: the corner gives no anchor and no
    observation in that session."""
    if not pairs or laps.empty:
        return laps
    out = laps.copy()
    sid = out["session_id"].astype(str).to_numpy()
    for session_id, corner in pairs:
        rows = sid == session_id
        if not rows.any():
            continue
        for col in tpms_lap_columns():
            if col in out.columns and f"_{corner}_" in col:
                out.loc[rows, col] = np.nan
    return out


# ------------------------------------------------------------- all four


def _clean_steps_1_to_3(root: Path, laps: pd.DataFrame, cfg: FilterConfig | None) -> pd.DataFrame:
    df = laps.copy()
    if not df.empty and laps_dir(root).exists() and timeseries_dir(root).exists():
        agg = _cleaned_tpms_aggregates(str(root), _dataset_signature(root))
        if not agg.empty:
            cols = [c for c in tpms_lap_columns() if c in agg.columns]
            flagged = agg[["session_id", "lap_num"] + cols].assign(_has_clean=True)
            merged = df[["session_id", "lap_num"]].merge(
                flagged, on=["session_id", "lap_num"], how="left"
            )
            has_clean = merged["_has_clean"].fillna(False).to_numpy(dtype=bool)
            for col in cols:
                cleaned = merged[col].to_numpy(dtype=np.float64)
                current = (
                    df[col].to_numpy(dtype=np.float64)
                    if col in df.columns
                    else np.full(len(df), np.nan)
                )
                df[col] = np.where(has_clean, cleaned, current)
    out = apply_filters(df, cfg)
    assert isinstance(out, pd.DataFrame)
    return out


def clean_laps(root: Path, laps: pd.DataFrame, cfg: FilterConfig | None = None) -> pd.DataFrame:
    """Masked TPMS aggregates, usability flags and the drift flag for a laps
    frame read from ``root``.

    Rows whose session has a TPMS timeseries under ``root`` get their TPMS
    summary columns replaced by the cleaned values; the others keep what the
    frame had. A corner that :func:`drifting_tpms` flags over the whole
    dataset has its TPMS columns masked for that session. A frame that lacks
    the filter inputs (synthetic test data, frames that were cleaned
    already) is returned unchanged when it carries a ``tire_usable`` column.
    """
    if not set(FILTER_INPUT_COLUMNS).issubset(laps.columns):
        if "tire_usable" in laps.columns:
            return laps.copy()
        missing = sorted(set(FILTER_INPUT_COLUMNS) - set(laps.columns))
        raise ValueError(f"laps frame lacks the filter inputs {missing}")
    out = _clean_steps_1_to_3(root, laps, cfg)
    if not out.empty and laps_dir(root).exists() and timeseries_dir(root).exists():
        out = mask_corner_sessions(out, _drifting_tpms_flags(str(root), _dataset_signature(root)))
    return out
