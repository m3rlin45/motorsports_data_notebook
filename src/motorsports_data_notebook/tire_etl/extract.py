"""Extract a single AIM session into per-sample timeseries + per-lap summaries.

This is the heart of the ETL: one function reads one xrk/xrz and returns three
PyArrow tables (session row, laps, timeseries). Caller is responsible for
upserting them into the dataset via :mod:`dataset`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from .aggregates import compute_corner_aggregates, compute_lap_dynamics
from .discovery import SessionCandidate, parse_filename, scan_aim_tree
from .filters import FilterConfig, apply_filters
from .stints import assign_stint_ids

if TYPE_CHECKING:
    from .._types import LogFile

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
CORNER_AIM_CODES = {"fl": "LF", "fr": "RF", "rl": "LR", "rr": "RR"}

# Canonical profile keys used as the reference for each channel family.
# We pull actual AIM names from the profile's channel_names dict.
TIRE_PRESS_KEYS = [f"tpms_press_{c}" for c in CORNERS]
TIRE_TEMP_KEYS = [f"tpms_temp_{c}" for c in CORNERS]
SURF_TEMP_KEYS = {c: [f"tire_temp_{c}_{i}" for i in range(1, 9)] for c in CORNERS}
DYNAMICS_KEYS = [
    "throttle",
    "brake",
    "gps_speed",
    "gps_latitude",
    "gps_longitude",
    "lateral_g",
    "inline_g",
    "steering",
]


@dataclass(frozen=True)
class ExtractResult:
    session_row: pa.Table  # 1 row
    laps_rows: pa.Table
    timeseries: pa.Table  # one row per (lap, sample); session_id is included
    status: str  # "ok" | "partial" | "error"
    error_msg: str | None


def _session_id_for(path: Path, mtime_ns: int) -> str:
    h = hashlib.sha1(f"{path.resolve()}|{mtime_ns}".encode()).hexdigest()
    return h[:16]


def _resolve_channels(profile_channel_names: dict[str, str], keys: list[str]) -> dict[str, str]:
    """Return {canonical_key: actual_aim_name} for keys present in the profile."""
    out: dict[str, str] = {}
    for k in keys:
        actual = profile_channel_names.get(k)
        if actual:
            out[k] = actual
    return out


def _extract_session_datetime_utc(
    log: "LogFile",
    fallback_date: str,
    track_canonical: str | None,
) -> datetime:
    """Derive a UTC timestamp for session start.

    Prefers AIM metadata if it exposes a parseable "Date" field, else falls
    back to midnight of the parsed filename date in the track's local tz.
    """
    from .tracks import get_track

    meta = getattr(log, "metadata", {}) or {}
    raw = None
    for key in ("Session date", "session date", "Date", "date", "Start time", "start_time"):
        if key in meta and meta[key]:
            raw = str(meta[key])
            break

    # Try several common AIM formats: "YYYY-MM-DD HH:MM:SS", ISO, etc.
    if raw:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                dt_local = datetime.strptime(raw, fmt)
                tz_name = None
                if track_canonical:
                    ti = get_track(track_canonical)
                    if ti:
                        tz_name = ti.timezone
                if tz_name:
                    from zoneinfo import ZoneInfo

                    return dt_local.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
                return dt_local.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    # Fallback: midnight local.
    y, m, d = (int(x) for x in fallback_date.split("-"))
    try:
        if track_canonical:
            ti = get_track(track_canonical)
            if ti:
                from zoneinfo import ZoneInfo

                return datetime(y, m, d, tzinfo=ZoneInfo(ti.timezone)).astimezone(timezone.utc)
    except Exception:
        pass
    return datetime(y, m, d, tzinfo=timezone.utc)


def _build_lap_timeseries(
    log: "LogFile",
    lap_num: int,
    lap_start_ms: int,
    lap_end_ms: int,
    stint_id: int,
    selected_channels: list[str],
    ref_channel: str,
) -> pa.Table:
    """Align all selected channels for one lap to ``ref_channel``'s timebase."""
    lap_log = log.filter_by_lap(lap_num)
    # Only request channels that exist on this log
    available = [c for c in selected_channels if c in lap_log.channels]
    if ref_channel not in available:
        # Can't align — return empty
        return pa.table({})
    aligned = lap_log.select_channels(available).resample_to_channel(ref_channel).channels
    ref_tbl = aligned.get(ref_channel)
    if ref_tbl is None or len(ref_tbl) == 0:
        return pa.table({})
    timecodes = ref_tbl.column("timecodes").to_numpy().astype(np.int64)
    t_lap_s = (timecodes - lap_start_ms).astype(np.float64) / 1000.0
    cols: dict[str, pa.Array] = {
        "lap_num": pa.array(np.full(len(timecodes), lap_num, dtype=np.int16)),
        "stint_id": pa.array(np.full(len(timecodes), stint_id, dtype=np.int16)),
        "sample_idx": pa.array(np.arange(len(timecodes), dtype=np.int32)),
        "t_session_s": pa.array(timecodes.astype(np.float64) / 1000.0),
        "t_lap_s": pa.array(t_lap_s.astype(np.float32)),
    }
    for ch, tbl in aligned.items():
        if ch == ref_channel:
            arr = tbl.column(ref_channel).to_numpy(zero_copy_only=False)
        else:
            arr = tbl.column(ch).to_numpy(zero_copy_only=False)
        cols[ch] = pa.array(np.asarray(arr, dtype=np.float64))
    return pa.table(cols)


def _rename_to_canonical(
    ts: pa.Table,
    *,
    profile_names: dict[str, str],
) -> pa.Table:
    """Rename AIM channel columns to canonical snake_case names.

    Also collapses 8 surface-temp sensors per corner into ``surf_temp_{c}_mean_c``
    (and keeps the 8 per-sensor columns as ``surf_temp_{c}_ch{i}_c``).
    """
    rename_map: dict[str, str] = {}

    def _rename_if(key: str, out_name: str) -> None:
        src = profile_names.get(key)
        if src and src in ts.schema.names:
            rename_map[src] = out_name

    for c in CORNERS:
        _rename_if(f"tpms_press_{c}", f"tpms_press_{c}_bar")
        _rename_if(f"tpms_temp_{c}", f"tpms_temp_{c}_c")
        for i in range(1, 9):
            _rename_if(f"tire_temp_{c}_{i}", f"surf_temp_{c}_ch{i}_c")

    _rename_if("throttle", "throttle_pct")
    _rename_if("brake", "brake_bar")
    _rename_if("gps_speed", "speed_ms")
    _rename_if("gps_latitude", "gps_lat")
    _rename_if("gps_longitude", "gps_lon")
    _rename_if("lateral_g", "lat_g")
    _rename_if("inline_g", "long_g")
    _rename_if("steering", "steer_deg")

    if rename_map:
        new_names = [rename_map.get(name, name) for name in ts.schema.names]
        ts = ts.rename_columns(new_names)

    # Derive speed_kmh from speed_ms if available (mirrors _util.py behavior).
    if "speed_ms" in ts.schema.names and "speed_kmh" not in ts.schema.names:
        speed_ms = ts.column("speed_ms").to_numpy(zero_copy_only=False)
        ts = ts.append_column("speed_kmh", pa.array(np.asarray(speed_ms, dtype=np.float64) * 3.6))

    # Normalize TPMS pressures to bar. AIM usually already reports bar (values
    # 1-3). If we see kPa values (typically 100-300), divide by 100.
    for c in CORNERS:
        col = f"tpms_press_{c}_bar"
        if col in ts.schema.names:
            arr = np.asarray(ts.column(col).to_numpy(zero_copy_only=False), dtype=np.float64)
            finite = arr[~np.isnan(arr)]
            if finite.size and float(np.nanmax(np.abs(finite))) > 10.0:
                arr = arr / 100.0
                idx = ts.schema.get_field_index(col)
                ts = ts.set_column(idx, col, pa.array(arr))

    # Collapse per-sensor surface temps into a mean column per corner.
    for c in CORNERS:
        ch_cols = [f"surf_temp_{c}_ch{i}_c" for i in range(1, 9)]
        present = [col for col in ch_cols if col in ts.schema.names]
        if present:
            arrs = [ts.column(col).to_numpy(zero_copy_only=False) for col in present]
            stacked = np.vstack([np.asarray(a, dtype=np.float64) for a in arrs])
            with np.errstate(all="ignore"):
                mean = np.nanmean(stacked, axis=0)
            ts = ts.append_column(f"surf_temp_{c}_mean_c", pa.array(mean))

    return ts


def extract_session(
    path: Path,
    *,
    fallback_date_str: str | None = None,
    extractor_version: str | None = None,
) -> ExtractResult:
    """Extract one AIM session into session/laps/timeseries tables.

    On failure returns an ExtractResult with ``status="error"`` and empty
    data tables — callers still record the attempt in the manifest so we
    don't retry forever.
    """
    from .._util import load_session
    from ..profiles import (
        DEFAULT_CHANNEL_NAMES,
        get_logger_id,
        get_profile_for_logger,
    )
    from . import EXTRACTOR_VERSION

    ev = extractor_version or EXTRACTOR_VERSION
    stat = path.stat()
    mtime_ns = stat.st_mtime_ns
    file_size = stat.st_size
    session_id = _session_id_for(path, mtime_ns)
    # Use filename date if parent is YYYY-MM-DD
    cand = parse_filename(path)
    date_str = f"{cand.date.year:04d}-{cand.date.month:02d}-{cand.date.day:02d}"
    if fallback_date_str is None:
        fallback_date_str = date_str

    empty_session = _build_empty_session_row(
        session_id=session_id,
        path=path,
        mtime_ns=mtime_ns,
        file_size=file_size,
        cand=cand,
        extractor_version=ev,
    )

    try:
        log = load_session(str(path))
    except Exception as e:
        logger.exception("load_session failed for %s", path)
        row = empty_session.set_column(
            empty_session.schema.get_field_index("status"),
            "status",
            pa.array(["error"]),
        ).set_column(
            empty_session.schema.get_field_index("error_msg"),
            "error_msg",
            pa.array([str(e)[:500]]),
        )
        return ExtractResult(
            session_row=row,
            laps_rows=pa.table({}),
            timeseries=pa.table({}),
            status="error",
            error_msg=str(e),
        )

    # Resolve profile + channel names.
    logger_id = get_logger_id(log) or ""
    profile = get_profile_for_logger(logger_id) if logger_id else None
    channel_names = dict(DEFAULT_CHANNEL_NAMES)
    profile_name = "default"
    if profile is not None:
        channel_names.update(profile.channel_names)
        profile_name = profile.name

    # Build list of AIM channel names we want in the timeseries.
    wanted_keys = (
        TIRE_PRESS_KEYS
        + TIRE_TEMP_KEYS
        + [f"tire_temp_{c}_{i}" for c in CORNERS for i in range(1, 9)]
        + DYNAMICS_KEYS
    )
    wanted_aim: dict[str, str] = _resolve_channels(channel_names, wanted_keys)
    selected_channels = sorted(set(wanted_aim.values()))

    # Pick a reference channel — prefer the first available TPMS pressure.
    ref_channel: str | None = None
    for k in TIRE_PRESS_KEYS:
        if k in wanted_aim and wanted_aim[k] in log.channels:
            ref_channel = wanted_aim[k]
            break
    if ref_channel is None:
        # Fall back to gps_speed or first available dynamics channel
        for k in ("gps_speed", "lateral_g", "inline_g"):
            if k in wanted_aim and wanted_aim[k] in log.channels:
                ref_channel = wanted_aim[k]
                break

    laps_table = log.laps
    if ref_channel is None or len(laps_table) == 0:
        row = empty_session.set_column(
            empty_session.schema.get_field_index("status"),
            "status",
            pa.array(["error"]),
        ).set_column(
            empty_session.schema.get_field_index("error_msg"),
            "error_msg",
            pa.array(["no usable reference channel or no laps"]),
        )
        return ExtractResult(
            session_row=row,
            laps_rows=pa.table({}),
            timeseries=pa.table({}),
            status="error",
            error_msg="no_ref_channel",
        )

    # Per-lap iteration: build full-session timeseries table.
    lap_nums = laps_table.column("num").to_pylist()
    starts = laps_table.column("start_time").to_numpy().astype(np.int64)
    ends = laps_table.column("end_time").to_numpy().astype(np.int64)
    stint_ids = assign_stint_ids(laps_table).to_numpy().astype(np.int16)

    ts_parts: list[pa.Table] = []
    for i, ln in enumerate(lap_nums):
        part = _build_lap_timeseries(
            log,
            lap_num=int(ln),
            lap_start_ms=int(starts[i]),
            lap_end_ms=int(ends[i]),
            stint_id=int(stint_ids[i]),
            selected_channels=selected_channels,
            ref_channel=ref_channel,
        )
        if len(part) == 0:
            continue
        ts_parts.append(part)

    if not ts_parts:
        row = empty_session.set_column(
            empty_session.schema.get_field_index("status"),
            "status",
            pa.array(["error"]),
        ).set_column(
            empty_session.schema.get_field_index("error_msg"),
            "error_msg",
            pa.array(["no samples after alignment"]),
        )
        return ExtractResult(
            session_row=row,
            laps_rows=pa.table({}),
            timeseries=pa.table({}),
            status="error",
            error_msg="no_samples",
        )

    timeseries = pa.concat_tables(ts_parts, promote_options="default")

    # Rename AIM channel names to canonical snake_case (and derive surf means).
    timeseries = _rename_to_canonical(timeseries, profile_names=channel_names)

    # Prepend session_id column.
    session_id_col = pa.array([session_id] * len(timeseries))
    timeseries = timeseries.add_column(0, "session_id", session_id_col)

    # Build per-lap summary.
    laps_rows = _build_laps_table(
        session_id=session_id,
        lap_nums=[int(x) for x in lap_nums],
        lap_starts_ms=starts.tolist(),
        lap_ends_ms=ends.tolist(),
        stint_ids=stint_ids.tolist(),
        timeseries=timeseries,
    )

    # Apply filters.
    laps_rows = apply_filters(laps_rows, FilterConfig())

    has_tpms = any(f"tpms_press_{c}_bar" in timeseries.schema.names for c in CORNERS)
    has_surface_temp = any(f"surf_temp_{c}_mean_c" in timeseries.schema.names for c in CORNERS)
    status = "ok" if has_tpms else "partial"
    err = None if status == "ok" else "tpms channels missing"

    session_row = _build_session_row(
        session_id=session_id,
        path=path,
        mtime_ns=mtime_ns,
        file_size=file_size,
        cand=cand,
        extractor_version=ev,
        logger_id=logger_id,
        profile_name=profile_name,
        n_laps=len(laps_rows),
        n_tire_usable=int(sum(laps_rows.column("tire_usable").to_pylist())),
        has_tpms=has_tpms,
        has_surface_temp=has_surface_temp,
        has_ambient=False,  # AIM logs rarely carry a dedicated ambient channel in this fleet
        has_track=False,
        ts_rate_hz=_estimate_ts_rate(timeseries),
        session_start_utc=_extract_session_datetime_utc(log, date_str, cand.track_canonical),
        status=status,
        error_msg=err,
    )

    return ExtractResult(
        session_row=session_row,
        laps_rows=laps_rows,
        timeseries=timeseries,
        status=status,
        error_msg=err,
    )


def _estimate_ts_rate(ts: pa.Table) -> float:
    """Estimate the timeseries sample rate in Hz from ``t_lap_s`` spacing."""
    if "t_lap_s" not in ts.schema.names or len(ts) < 2:
        return 0.0
    t = ts.column("t_lap_s").to_numpy()
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    median = float(np.median(dt))
    return 1.0 / median if median > 0 else 0.0


def _build_empty_session_row(
    *,
    session_id: str,
    path: Path,
    mtime_ns: int,
    file_size: int,
    cand: SessionCandidate,
    extractor_version: str,
) -> pa.Table:
    return _build_session_row(
        session_id=session_id,
        path=path,
        mtime_ns=mtime_ns,
        file_size=file_size,
        cand=cand,
        extractor_version=extractor_version,
        logger_id="",
        profile_name="",
        n_laps=0,
        n_tire_usable=0,
        has_tpms=False,
        has_surface_temp=False,
        has_ambient=False,
        has_track=False,
        ts_rate_hz=0.0,
        session_start_utc=datetime(
            cand.date.year, cand.date.month, cand.date.day, tzinfo=timezone.utc
        ),
        status="pending",
        error_msg=None,
    )


def _build_session_row(
    *,
    session_id: str,
    path: Path,
    mtime_ns: int,
    file_size: int,
    cand: SessionCandidate,
    extractor_version: str,
    logger_id: str,
    profile_name: str,
    n_laps: int,
    n_tire_usable: int,
    has_tpms: bool,
    has_surface_temp: bool,
    has_ambient: bool,
    has_track: bool,
    ts_rate_hz: float,
    session_start_utc: datetime,
    status: str = "ok",
    error_msg: str | None = None,
) -> pa.Table:
    return pa.table(
        {
            "session_id": [session_id],
            "xrk_path": [str(path)],
            "xrk_mtime_ns": pa.array([mtime_ns], type=pa.int64()),
            "file_size": pa.array([file_size], type=pa.int64()),
            "date": pa.array([cand.date], type=pa.date32()),
            "session_start_utc": pa.array([session_start_utc], type=pa.timestamp("us", tz="UTC")),
            "driver": [cand.driver],
            "car": [cand.car],
            "track": [cand.track_raw],
            "track_canonical": [cand.track_canonical],
            "session_type": [cand.session_type],
            "run_num": pa.array([cand.run_num], type=pa.int32()),
            "logger_id": [logger_id],
            "profile_name": [profile_name],
            "extractor_version": [extractor_version],
            "extracted_at": pa.array(
                [datetime.now(timezone.utc)], type=pa.timestamp("us", tz="UTC")
            ),
            "status": [status],
            "error_msg": pa.array([error_msg], type=pa.string()),
            "n_laps": pa.array([n_laps], type=pa.int32()),
            "n_tire_usable_laps": pa.array([n_tire_usable], type=pa.int32()),
            "has_tpms": [has_tpms],
            "has_surface_temp": [has_surface_temp],
            "has_ambient_temp": [has_ambient],
            "has_track_temp": [has_track],
            "ts_rate_hz": pa.array([ts_rate_hz], type=pa.float32()),
        }
    )


def _build_laps_table(
    *,
    session_id: str,
    lap_nums: list[int],
    lap_starts_ms: list[int],
    lap_ends_ms: list[int],
    stint_ids: list[int],
    timeseries: pa.Table,
) -> pa.Table:
    rows: dict[str, list] = {
        "session_id": [],
        "lap_num": [],
        "stint_id": [],
        "lap_time_s": [],
        "is_outlap": [],
        "is_inlap": [],
        "speed_kmh_mean": [],
        "speed_kmh_max": [],
        "brake_mean": [],
        "brake_max": [],
        "throttle_mean": [],
        "lat_g_peak": [],
        "long_g_peak_brake": [],
        "heat_proxy": [],
        "on_track_s": [],
        "distance_m": [],
    }
    for c in CORNERS:
        for stat in ("start", "end", "min", "max", "mean"):
            rows[f"tpms_press_{c}_{stat}"] = []
            rows[f"tpms_temp_{c}_{stat}"] = []
        rows[f"tpms_press_{c}_rise_bar_per_min"] = []
        for stat in ("mean", "min", "max"):
            rows[f"surf_temp_{c}_{stat}"] = []

    lap_num_col = timeseries.column("lap_num").to_numpy()

    # Per-stint first-lap detection (out-laps).
    stint_first_lap: dict[int, int] = {}
    for i, s in enumerate(stint_ids):
        if s not in stint_first_lap or lap_nums[i] < stint_first_lap[s]:
            stint_first_lap[s] = lap_nums[i]
    # Per-stint last-lap detection (potential in-lap).
    stint_last_lap: dict[int, int] = {}
    for i, s in enumerate(stint_ids):
        if s not in stint_last_lap or lap_nums[i] > stint_last_lap[s]:
            stint_last_lap[s] = lap_nums[i]

    # Per-stint median lap time (for in-lap detection: > 1.2x median).
    lap_times_s = [(lap_ends_ms[i] - lap_starts_ms[i]) / 1000.0 for i in range(len(lap_nums))]
    stint_laps: dict[int, list[float]] = {}
    for i, s in enumerate(stint_ids):
        stint_laps.setdefault(s, []).append(lap_times_s[i])
    stint_median: dict[int, float] = {s: float(np.median(ts)) for s, ts in stint_laps.items() if ts}

    for i, ln in enumerate(lap_nums):
        mask = lap_num_col == ln
        lap_ts = timeseries.filter(pa.array(mask))

        rows["session_id"].append(session_id)
        rows["lap_num"].append(ln)
        rows["stint_id"].append(stint_ids[i])
        rows["lap_time_s"].append(lap_times_s[i])
        rows["is_outlap"].append(ln == stint_first_lap.get(stint_ids[i]))
        # In-lap: last lap of stint AND lap_time > 1.2 * stint median
        is_inlap = ln == stint_last_lap.get(stint_ids[i]) and lap_times_s[
            i
        ] > 1.2 * stint_median.get(stint_ids[i], float("inf"))
        rows["is_inlap"].append(is_inlap)

        dyn = compute_lap_dynamics(lap_ts)
        for k in (
            "speed_kmh_mean",
            "speed_kmh_max",
            "brake_mean",
            "brake_max",
            "throttle_mean",
            "lat_g_peak",
            "long_g_peak_brake",
            "heat_proxy",
            "on_track_s",
            "distance_m",
        ):
            rows[k].append(dyn[k])

        for c in CORNERS:
            agg = compute_corner_aggregates(lap_ts, c)
            rows[f"tpms_press_{c}_start"].append(agg.press_start)
            rows[f"tpms_press_{c}_end"].append(agg.press_end)
            rows[f"tpms_press_{c}_min"].append(agg.press_min)
            rows[f"tpms_press_{c}_max"].append(agg.press_max)
            rows[f"tpms_press_{c}_mean"].append(agg.press_mean)
            rows[f"tpms_press_{c}_rise_bar_per_min"].append(agg.press_rise_bar_per_min)
            rows[f"tpms_temp_{c}_start"].append(agg.temp_start)
            rows[f"tpms_temp_{c}_end"].append(agg.temp_end)
            rows[f"tpms_temp_{c}_min"].append(agg.temp_min)
            rows[f"tpms_temp_{c}_max"].append(agg.temp_max)
            rows[f"tpms_temp_{c}_mean"].append(agg.temp_mean)
            rows[f"surf_temp_{c}_mean"].append(agg.surf_mean)
            rows[f"surf_temp_{c}_min"].append(agg.surf_min)
            rows[f"surf_temp_{c}_max"].append(agg.surf_max)

    # Explicit typing: all the float columns go to float32 for size,
    # identifiers stay int; lap flags are booleans.
    schema_fields: list[pa.Field] = [
        pa.field("session_id", pa.string()),
        pa.field("lap_num", pa.int16()),
        pa.field("stint_id", pa.int16()),
        pa.field("lap_time_s", pa.float32()),
        pa.field("is_outlap", pa.bool_()),
        pa.field("is_inlap", pa.bool_()),
        pa.field("speed_kmh_mean", pa.float32()),
        pa.field("speed_kmh_max", pa.float32()),
        pa.field("brake_mean", pa.float32()),
        pa.field("brake_max", pa.float32()),
        pa.field("throttle_mean", pa.float32()),
        pa.field("lat_g_peak", pa.float32()),
        pa.field("long_g_peak_brake", pa.float32()),
        pa.field("heat_proxy", pa.float32()),
        pa.field("on_track_s", pa.float32()),
        pa.field("distance_m", pa.float32()),
    ]
    for c in CORNERS:
        for stat in ("start", "end", "min", "max", "mean"):
            schema_fields.append(pa.field(f"tpms_press_{c}_{stat}", pa.float32()))
            schema_fields.append(pa.field(f"tpms_temp_{c}_{stat}", pa.float32()))
        schema_fields.append(pa.field(f"tpms_press_{c}_rise_bar_per_min", pa.float32()))
        for stat in ("mean", "min", "max"):
            schema_fields.append(pa.field(f"surf_temp_{c}_{stat}", pa.float32()))
    schema = pa.schema(schema_fields)

    # Build table with explicit schema — pa.table handles type coercion via
    # construction, but we cast after to be safe.
    return pa.table(rows).cast(schema)


def run_extract(
    *,
    aim_root: Path,
    dataset_root: Path,
    since=None,
    only_car: str | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> dict[str, int]:
    """Extract all new sessions and upsert them into the dataset."""
    from .dataset import load_manifest, upsert_session

    existing = load_manifest(dataset_root)
    counts = {"scanned": 0, "skipped": 0, "extracted": 0, "errors": 0}

    for cand in scan_aim_tree(aim_root, since=since, only_car=only_car):
        counts["scanned"] += 1
        stat = cand.path.stat()
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
        from . import EXTRACTOR_VERSION

        key = (str(cand.path), mtime_ns, size, EXTRACTOR_VERSION)
        prior = existing.get(str(cand.path))
        if (
            prior is not None
            and not force
            and prior.get("xrk_mtime_ns") == mtime_ns
            and prior.get("file_size") == size
            and prior.get("extractor_version") == EXTRACTOR_VERSION
            and (prior.get("status") != "error" or not retry_errors)
        ):
            counts["skipped"] += 1
            continue

        t0 = time.perf_counter()
        result = extract_session(cand.path)
        elapsed = time.perf_counter() - t0
        logger.info(
            "extracted %s status=%s elapsed=%.2fs",
            cand.path.name,
            result.status,
            elapsed,
        )

        upsert_session(dataset_root, result)
        if result.status == "error":
            counts["errors"] += 1
        else:
            counts["extracted"] += 1
        existing[str(cand.path)] = {
            "xrk_mtime_ns": mtime_ns,
            "file_size": size,
            "extractor_version": EXTRACTOR_VERSION,
            "status": result.status,
        }
    return counts
