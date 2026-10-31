"""Training-side data cleaning for the tire dataset.

The extractor (:mod:`..tire_etl`) turns the AIM files into in-repo data with
the minimum processing needed for a stable, manageable dataset: it splits
laps and stints, resolves the circuit from GPS, selects and renames channels
(stored verbatim at their native rates, dataset schema 7), validates the IMU
against GPS, and writes a raw per-lap summary. Nothing under ``data/tire_dataset/`` is masked, filtered or judged.
Every decision about which samples and laps the model may learn from is made
here, at training time, so a cleaning rule can change without re-extracting.

Three steps, in the order the model applies them:

1. :func:`mask_stale_session_prefix` — a TPMS module that is asleep at
   roll-out reports its stored value until it wakes; that stale prefix is
   masked per corner.
2. :func:`recompute_tpms_lap_aggregates` — the per-lap TPMS summary columns
   (start / end / min / max / mean, rise rate, first-valid time) are rebuilt
   from the masked timeseries, so an out-lap's ``tpms_*_start`` — the stint
   anchor — is the first *live* reading.
3. :func:`apply_filters` — ``tire_usable`` / ``exclude_reason`` per lap.

:func:`clean_laps` runs all three over a laps frame and is what every model
entry point (fit, holdout, notes validation, sensor audit, compound
inference) goes through. ``tire-model clean-laps`` writes the result to a
parquet file for ad-hoc queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
    return mask_stale_session_prefix(ts.select(cols))


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


# ------------------------------------------------------------- all three


def clean_laps(root: Path, laps: pd.DataFrame, cfg: FilterConfig | None = None) -> pd.DataFrame:
    """Masked TPMS aggregates + usability flags for a laps frame read from ``root``.

    Rows whose session has a TPMS timeseries under ``root`` get their TPMS
    summary columns replaced by the cleaned values; the others keep what the
    frame had. A frame that lacks the filter inputs (synthetic test data,
    frames that were cleaned already) is returned unchanged when it carries a
    ``tire_usable`` column.
    """
    df = laps.copy()
    if not set(FILTER_INPUT_COLUMNS).issubset(df.columns):
        if "tire_usable" in df.columns:
            return df
        missing = sorted(set(FILTER_INPUT_COLUMNS) - set(df.columns))
        raise ValueError(f"laps frame lacks the filter inputs {missing}")
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
