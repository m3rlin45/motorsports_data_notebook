"""Extract a single AIM session into per-sample timeseries + per-lap summaries.

This is the heart of the ETL: one function reads one xrk/xrz and returns three
PyArrow tables (session row, laps, timeseries). Caller is responsible for
upserting them into the dataset via :mod:`dataset`.

The timeseries keeps every selected channel **verbatim at its native rate**
(dataset schema 7): channels the logger sampled on one timecode array form a
*timebase*, each timebase is a block of rows with its own ``t_session_s``,
and a channel column is null outside its timebase. Nothing is resampled or
aligned here — see :mod:`.dataset` for the layout and the loaders that merge
at query time, and :mod:`.kinematics` for the IMU validation whose verdict
is recorded on the session row.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from .aggregates import compute_corner_aggregates, compute_lap_dynamics
from .dataset import build_timeseries_table, narrowest_float, timeseries_summary
from .discovery import SessionCandidate, parse_filename, scan_aim_tree
from .kinematics import (
    KIN_ROW_COLUMNS,
    KinematicsVerdict,
    choose_reference_lap,
    empty_verdict_row,
    session_kinematics,
    validate_session,
)
from .layouts import LayoutDetection, apply_layout_correction, detect_layout
from .stints import assign_stint_ids

if TYPE_CHECKING:
    from .._types import LogFile

logger = logging.getLogger(__name__)

CORNERS = ("fl", "fr", "rl", "rr")
CORNER_AIM_CODES = {"fl": "LF", "fr": "RF", "rl": "LR", "rr": "RR"}

# Channel families stored at their native rate: profile key -> canonical
# column, per family. The AIM names come from the profile's channel_names.
# ``imu_lateral_g`` / ``imu_inline_g`` name the raw IMU regardless of what
# the analysis-side ``lateral_g`` / ``inline_g`` keys point at for a vehicle
# (the KK profiles alias those to the logger's GPS-derived channels).
FAMILY_KEYS: dict[str, dict[str, str]] = {
    "gps": {
        "gps_latitude": "gps_lat",
        "gps_longitude": "gps_lon",
        "gps_altitude": "gps_alt_m",
        "gps_speed": "speed_ms",
        "gps_lateral_g": "gps_lat_g",
        "gps_inline_g": "gps_long_g",
        "gps_yaw_rate": "gps_yaw_rate_dps",
    },
    "accel": {"imu_lateral_g": "lat_g", "imu_inline_g": "long_g", "vertical_g": "vert_g"},
    "gyro": {
        "yaw_rate": "yaw_rate_dps",
        "pitch_rate": "pitch_rate_dps",
        "roll_rate": "roll_rate_dps",
    },
    "tpms": {
        **{f"tpms_press_{c}": f"tpms_press_{c}_bar" for c in CORNERS},
        **{f"tpms_temp_{c}": f"tpms_temp_{c}_c" for c in CORNERS},
    },
    "surface": {
        f"tire_temp_{c}_{i}": f"surf_temp_{c}_ch{i}_c" for c in CORNERS for i in range(1, 9)
    },
    "controls": {"throttle": "throttle_pct", "brake": "brake_bar", "steering": "steer_deg"},
}
# Position must keep the file's double precision: float32 quantises latitude
# to ~0.5 m and longitude to ~1.5 m at 36°N / 139°E, which destroys the
# curvature the IMU validation is referenced to.
FLOAT64_CHANNELS = {"gps_lat", "gps_lon"}
TIRE_PRESS_KEYS = [f"tpms_press_{c}" for c in CORNERS]
TIRE_TEMP_KEYS = [f"tpms_temp_{c}" for c in CORNERS]
SURF_TEMP_KEYS = {c: [f"tire_temp_{c}_{i}" for i in range(1, 9)] for c in CORNERS}


@dataclass(frozen=True)
class ExtractResult:
    session_row: pa.Table  # 1 row
    laps_rows: pa.Table
    timeseries: pa.Table  # one row per raw sample; session_id is included
    status: str  # "ok" | "partial" | "error"
    error_msg: str | None
    timebases: list[dict] | None = None  # per-timebase summary (dataset.timeseries_summary)
    kinematics: dict | None = None  # KinematicsVerdict.to_json()


_ID_UNSAFE = re.compile(r"[^A-Za-z0-9.+-]+")


def _id_slug(value: str | None) -> str | None:
    """``value`` with every run of characters outside ``[A-Za-z0-9.+-]``
    collapsed to one ``-`` (``"Inferno 86"`` → ``Inferno-86``); None when
    nothing is left."""
    if value is None:
        return None
    out = _ID_UNSAFE.sub("-", value.strip()).strip("-")
    return out or None


def _run_token(run_nums: list[int | None]) -> str | None:
    """``a0370`` for one run, ``a0370-0372`` for a consecutive range (a
    merged restart session), ``a0370+0375`` otherwise; None without run
    numbers."""
    runs = sorted({r for r in run_nums if r is not None})
    if not runs:
        return None
    if len(runs) == 1:
        return f"a{runs[0]:04d}"
    if runs == list(range(runs[0], runs[-1] + 1)):
        return f"a{runs[0]:04d}-{runs[-1]:04d}"
    return "a" + "+".join(f"{r:04d}" for r in runs)


def session_id_for(
    date_str: str,
    start_local: datetime | None,
    car: str | None,
    track: str | None,
    run_nums: list[int | None],
) -> str:
    """The session's id, readable and sortable:
    ``2026-10-24_122053_KK-F_tsukuba_a0370-0372`` — the date, the logger's
    wall-clock start (``000000`` when unknown), the car, the canonical track
    (or the filename's token when unresolved) and the run number(s) of the
    constituent files. It names the timeseries file and keys every table.
    It changes when the session's identity changes (another car tag, a
    different grouping of files), not when a file is merely rewritten: the
    manifest's mtime / size check decides re-extraction, and a re-extract
    replaces the rows under the same id.
    """
    parts = [
        date_str,
        start_local.strftime("%H%M%S") if start_local is not None else "000000",
        _id_slug(car) or "unknown-car",
        _id_slug(track) or "unknown-track",
    ]
    runs = _run_token(run_nums)
    if runs:
        parts.append(runs)
    return "_".join(parts)


def _local_wallclock(start_utc: datetime | None, track_canonical: str | None) -> datetime | None:
    """The logger's own wall-clock reading of ``start_utc`` in the track's
    timezone (Japan when the track is unresolved — every circuit in the
    dataset is there, and a GPS-derived start is true UTC whatever the
    track), so the id shows the time Race Studio shows."""
    if start_utc is None:
        return None
    from zoneinfo import ZoneInfo

    from .tracks import get_track

    tz: Any = ZoneInfo("Asia/Tokyo")
    if track_canonical:
        ti = get_track(track_canonical)
        if ti:
            tz = ZoneInfo(ti.timezone)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    return start_utc.astimezone(tz)


def _resolve_channels(profile_channel_names: dict[str, str], keys: list[str]) -> dict[str, str]:
    """Return {canonical_key: actual_aim_name} for keys present in the profile."""
    out: dict[str, str] = {}
    for k in keys:
        actual = profile_channel_names.get(k)
        if actual:
            out[k] = actual
    return out


def _localize_to_utc(dt_local: datetime, track_canonical: str | None) -> datetime:
    """Attach the track's local timezone (UTC if unknown) and convert to UTC."""
    from .tracks import get_track

    tz_name = None
    if track_canonical:
        ti = get_track(track_canonical)
        if ti:
            tz_name = ti.timezone
    if tz_name:
        from zoneinfo import ZoneInfo

        return dt_local.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
    return dt_local.replace(tzinfo=timezone.utc)


def _try_parse_log_start_utc(log: "LogFile", track_canonical: str | None) -> datetime | None:
    """Parse the wall-clock session start from AIM metadata, or None.

    Unlike :func:`_extract_session_datetime_utc` this has NO fallback —
    callers that need to reason about wall-clock adjacency (split-session
    grouping) must be able to tell "parsed" from "guessed".
    """
    meta = getattr(log, "metadata", {}) or {}

    # AIM RaceStudio splits the start timestamp into two fields:
    #   "Log Date": "MM/DD/YYYY" (US format, confirmed empirically)
    #   "Log Time": "HH:MM:SS"   (track-local, 24-hour)
    # Combine and parse in the track's local tz.
    log_date = meta.get("Log Date")
    log_time = meta.get("Log Time")
    if log_date and log_time:
        try:
            dt_local = datetime.strptime(f"{log_date} {log_time}", "%m/%d/%Y %H:%M:%S")
            return _localize_to_utc(dt_local, track_canonical)
        except ValueError:
            pass

    # Legacy / alternate metadata spellings we've seen on other loggers.
    raw = None
    for key in ("Session date", "session date", "Date", "date", "Start time", "start_time"):
        if key in meta and meta[key]:
            raw = str(meta[key])
            break
    if raw:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                dt_local = datetime.strptime(raw, fmt)
                return _localize_to_utc(dt_local, track_canonical)
            except ValueError:
                continue
    return None


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

    parsed = _try_parse_log_start_utc(log, track_canonical)
    if parsed is not None:
        return parsed

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


# Max wall-clock gap between the END of one constituent file and the START
# of the next for the two to still count as one physical session. Genuine
# logger restarts (spin, power cycle) resume within a couple of minutes;
# anything longer is a separate track session even when AIM numbers the
# files consecutively.
MERGE_MAX_GAP_S = 300.0


def file_comment(log: "LogFile") -> str | None:
    """The free-text comment typed into RaceStudio for this file, or None.

    AIM stores it as the ``Long Comment`` metadata field. The driver uses it
    for the tire fitted ("re-71rs", "qually, slick tires, damp track"), setup
    changes and race results; the model's compound labelling reads it.
    """
    try:
        raw = (log.metadata or {}).get("Long Comment")
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return None
    if not isinstance(raw, str):
        return None
    text = "\n".join(line.strip() for line in raw.replace("\r\n", "\n").split("\n"))
    text = text.strip()
    return text or None


def session_comment(logs: list) -> str | None:
    """Join the distinct file comments of a (possibly merged) session, in file order."""
    seen: list[str] = []
    for lg in logs:
        c = file_comment(lg)
        if c and c not in seen:
            seen.append(c)
    return "\n".join(seen) if seen else None


def load_and_correct_session(path: Path) -> tuple["LogFile", LayoutDetection]:
    """Load one AIM file and reconcile its laps/track with the GPS trace.

    The logger's track *variant* selection decides where the lap beacon sits
    and what the filename/``Venue`` say; both can be wrong (full Suzuka laps
    logged as "Suzuka West"). :func:`~.layouts.detect_layout` resolves the
    layout actually driven from the GPS trace and, when the beacon is not on
    that layout's start/finish, :func:`~.layouts.apply_layout_correction`
    rebuilds ``log.laps`` from GPS crossings of the correct line. The
    detection travels with the log so the session row can record what
    happened.
    """
    from .._util import load_session

    # Keep the out-lap (pit exit to the first start/finish crossing) and the
    # in-lap: the out-lap's first valid TPMS reading is the stint's starting
    # temperature for the tire model. Notebook analysis keeps the default
    # (full laps only).
    log = load_session(str(path), lap_types=("full", "out", "in"))
    cand = parse_filename(path)
    venue_meta = None
    try:
        venue_meta = (log.metadata or {}).get("Venue") or None
    except Exception:  # noqa: BLE001 — metadata is best-effort
        venue_meta = None
    det = detect_layout(log, declared_track_raw=cand.track_raw, venue_meta=venue_meta)
    apply_layout_correction(log, det)
    return log, det


# Channels whose last sample marks the end of the recording (GPS timebase).
_END_OF_DATA_CHANNELS = ("GPS Speed", "GPS Latitude", "GPS Longitude")


def _log_end_ms(log: "LogFile") -> int:
    """End of the recording in ms on the file's own clock; 0 if nothing usable.

    Uses the last GPS sample when available and falls back to the last lap
    end. libxrk (>= 0.13) types the first/last laps as out/in; the tire ETL
    keeps them, but older extractions and the notebook loader drop them, so
    the last lap in the table can end minutes before the logger stopped.
    Merge decisions in
    :func:`split_group_by_wallclock` must see the real end, or a restart
    pair whose gap is measured from the in-lap would wrongly split.
    """
    end = 0
    laps = log.laps
    if len(laps) > 0:
        end = int(np.max(laps.column("end_time").to_numpy()))
    channels = getattr(log, "channels", None)
    if isinstance(channels, dict):
        for name in _END_OF_DATA_CHANNELS:
            tbl = channels.get(name)
            if tbl is not None and len(tbl) > 0:
                end = max(end, int(tbl.column("timecodes")[-1].as_py()))
                break
    return end


def split_group_by_wallclock(
    paths: list[Path],
    logs: list,
    *,
    track_canonical: str | None,
    max_gap_s: float = MERGE_MAX_GAP_S,
) -> list[tuple[list[Path], list, list[int]]]:
    """Split a filename-grouped candidate list into wall-clock-contiguous runs.

    Filename grouping (consecutive ``run_num``) cannot tell a mid-run logger
    restart from two separate track sessions that happen to be numbered
    consecutively. This uses each file's metadata wall-clock start plus its
    lap-table duration: whenever the next file starts more than ``max_gap_s``
    after the previous file ends, the group is split.

    Returns ``[(paths, logs, offsets_ms), ...]`` — one tuple per physical
    session. ``offsets_ms[i]`` places file *i* on its subgroup's first file's
    clock (for :class:`~motorsports_data_notebook._util.MergedLogFile`).

    If any file's wall-clock start can't be parsed, the group is kept whole
    (legacy behavior) but each file is stitched immediately after the
    previous one so the merged timeline is at least monotonic.
    """
    if len(logs) == 1:
        return [(list(paths), list(logs), [0])]

    maybe_starts = [_try_parse_log_start_utc(log, track_canonical) for log in logs]
    if any(s is None for s in maybe_starts):
        offsets = [0]
        for i in range(1, len(logs)):
            offsets.append(offsets[i - 1] + _log_end_ms(logs[i - 1]))
        return [(list(paths), list(logs), offsets)]
    starts = [s for s in maybe_starts if s is not None]

    groups: list[tuple[list[Path], list, list[int]]] = []
    sub_start = 0
    for i in range(1, len(logs) + 1):
        if i < len(logs):
            prev_end = starts[i - 1] + timedelta(milliseconds=_log_end_ms(logs[i - 1]))
            gap_s = (starts[i] - prev_end).total_seconds()
            if gap_s <= max_gap_s:
                continue
        base = starts[sub_start]
        offsets = [int((starts[j] - base).total_seconds() * 1000.0) for j in range(sub_start, i)]
        groups.append((list(paths[sub_start:i]), list(logs[sub_start:i]), offsets))
        sub_start = i
    return groups


@dataclass
class RawTimebase:
    """Channels the logger sampled on one timecode array (one file, or the
    same channel set across the files of a merged session)."""

    family: str
    channels: dict[str, np.ndarray]  # canonical column -> float64 values
    t_ms: np.ndarray  # int64, on the session clock


def _log_parts(log: Any) -> list[tuple[Any, int]]:
    """``[(log, offset_ms)]``: a merged session's files with the offsets
    that put their raw timecodes on the merged clock; a single file at 0."""
    parts = getattr(log, "parts", None)
    return list(parts()) if callable(parts) else [(log, 0)]


def _wanted_channels(channel_names: dict[str, str]) -> dict[str, tuple[str, str]]:
    """``{aim_channel_name: (family, canonical_column)}`` for the profile."""
    out: dict[str, tuple[str, str]] = {}
    for family, keys in FAMILY_KEYS.items():
        for key, canonical in keys.items():
            aim = channel_names.get(key)
            if aim and aim not in out:
                out[aim] = (family, canonical)
    return out


def _raw_timebases(log: Any, wanted: dict[str, tuple[str, str]]) -> list[RawTimebase]:
    """Group the wanted channels by identical timecode array (within a file)
    and family; the same channel set across a merged session's files is one
    timebase (their clocks are disjoint once offset)."""
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}
    for part, offset in _log_parts(log):
        chans = getattr(part, "channels", None) or {}
        by_hash: dict[tuple[str, str], list[tuple[str, np.ndarray, np.ndarray]]] = {}
        for aim, (family, canonical) in wanted.items():
            tbl = chans.get(aim)
            if tbl is None or len(tbl) == 0 or aim not in tbl.column_names:
                continue
            t = tbl.column("timecodes").to_numpy().astype(np.int64)
            v = np.asarray(tbl.column(aim).to_numpy(zero_copy_only=False), dtype=np.float64)
            if v.shape != t.shape:
                continue
            h = hashlib.blake2b(t.tobytes(), digest_size=8).hexdigest()
            by_hash.setdefault((family, h), []).append((canonical, v, t))
        for (family, _), items in by_hash.items():
            names = tuple(sorted(c for c, _, _ in items))
            t = items[0][2] + int(offset)
            groups.setdefault((family, names), []).append((t, {c: v for c, v, _ in items}))
    out: list[RawTimebase] = []
    for (family, names), pieces in groups.items():
        if len(pieces) == 1:
            t, ch = pieces[0]
        else:
            t = np.concatenate([pt for pt, _ in pieces])
            ch = {c: np.concatenate([pch[c] for _, pch in pieces]) for c in names}
        out.append(RawTimebase(family=family, channels=ch, t_ms=t))
    return out


def _timebase_names(tbs: list[RawTimebase]) -> list[str]:
    """``family`` when a family has one timebase, else ``family/<tag>`` with
    the tag naming the channels (``tpms/fl``, ``surface/fl_ch1-4``,
    ``accel/long_g+vert_g``)."""
    counts = Counter(tb.family for tb in tbs)
    names: list[str] = []
    for tb in tbs:
        if counts[tb.family] == 1:
            names.append(tb.family)
            continue
        cols = sorted(tb.channels)
        if tb.family == "tpms":
            tag = "+".join(sorted({c.split("_")[2] for c in cols}))
        elif tb.family == "surface":
            tags = []
            for corner in CORNERS:
                idx = sorted(
                    int(c.split("_ch")[1].split("_")[0])
                    for c in cols
                    if c.startswith(f"surf_temp_{corner}_")
                )
                if idx:
                    tags.append(
                        f"{corner}_ch{idx[0]}-{idx[-1]}" if len(idx) > 1 else f"{corner}_ch{idx[0]}"
                    )
            tag = "+".join(tags)
        else:
            tag = "+".join(cols)
        names.append(f"{tb.family}/{tag}")
    seen: Counter = Counter()
    final: list[str] = []
    for n in names:
        seen[n] += 1
        final.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return final


def _pressure_to_bar(arr: np.ndarray) -> np.ndarray:
    """AIM usually reports TPMS pressure in bar (values 1-3); kPa values
    (typically 100-300) are divided by 100."""
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.max(np.abs(finite))) > 10.0:
        return arr / 100.0
    return arr


def build_session_timeseries(
    log: Any,
    channel_names: dict[str, str],
    *,
    session_id: str,
    lap_nums: list[int],
    lap_starts_ms: list[int],
    lap_ends_ms: list[int],
    stint_ids: list[int],
) -> tuple[pa.Table, list[dict]]:
    """The session's native-rate table and its per-timebase summary.

    Every wanted channel that exists on the log is stored verbatim: its raw
    timecodes (on the merged clock for a multi-file session) become
    ``t_session_s``, each sample is assigned the lap whose
    ``[start, end)`` window holds it (samples outside every lap are
    dropped, as the per-lap extraction always did), and ``t_lap_s`` /
    ``sample_idx`` count from that lap's start. Channel values are stored
    at the narrowest float width that holds them exactly (float16 for the
    channels the logger logged as float16, see
    :func:`~.dataset.narrowest_float`; float64 for positions); TPMS
    pressures are normalised to bar.
    """
    tbs = _raw_timebases(log, _wanted_channels(channel_names))
    if not tbs or not lap_nums:
        return pa.table({}), []
    order = np.argsort(np.asarray(lap_starts_ms, dtype=np.int64), kind="stable")
    starts = np.asarray(lap_starts_ms, dtype=np.int64)[order]
    ends = np.asarray(lap_ends_ms, dtype=np.int64)[order]
    nums = np.asarray(lap_nums, dtype=np.int64)[order]
    stints = np.asarray(stint_ids, dtype=np.int64)[order]
    parts: list[tuple[str, str, pa.Table]] = []
    for name, tb in zip(_timebase_names(tbs), tbs):
        t = tb.t_ms
        sort = np.argsort(t, kind="stable")
        t = t[sort]
        idx = np.searchsorted(starts, t, side="right") - 1
        idx_c = np.clip(idx, 0, len(starts) - 1)
        valid = (idx >= 0) & (t < ends[idx_c])
        if not valid.any():
            continue
        t_v = t[valid]
        lap_i = idx_c[valid]
        n = int(t_v.size)
        change = np.concatenate([[True], lap_i[1:] != lap_i[:-1]])
        lap_start_pos = np.maximum.accumulate(np.where(change, np.arange(n), 0))
        cols: dict[str, pa.Array] = {
            "lap_num": pa.array(nums[lap_i].astype(np.int16)),
            "stint_id": pa.array(stints[lap_i].astype(np.int16)),
            "sample_idx": pa.array((np.arange(n) - lap_start_pos).astype(np.int32)),
            "t_session_s": pa.array(t_v.astype(np.float64) / 1000.0, type=pa.float64()),
            "t_lap_s": pa.array(((t_v - starts[lap_i]) / 1000.0).astype(np.float32)),
        }
        for canonical, values in tb.channels.items():
            v = values[sort][valid]
            if canonical.startswith("tpms_press_"):
                v = _pressure_to_bar(v)
            cols[canonical] = narrowest_float(v, float64=canonical in FLOAT64_CHANNELS)
        parts.append((name, tb.family, pa.table(cols)))
    table = build_timeseries_table(session_id, parts)
    return table, timeseries_summary(table)


def extract_session(
    paths: Path | list[Path],
    *,
    fallback_date_str: str | None = None,
    extractor_version: str | None = None,
    _preloaded_logs: list | None = None,
    _offsets_ms: list[int] | None = None,
    _preloaded_detections: list[LayoutDetection] | None = None,
) -> ExtractResult:
    """Extract one AIM session into session/laps/timeseries tables.

    Accepts either a single path or a list of paths. When given multiple
    paths, they're loaded and stitched together with
    :class:`~motorsports_data_notebook._util.MergedLogFile` so the AIM
    logger's "new file per restart" splits become one logical session
    with sequentially-renumbered laps and one continuous timeseries on the
    first file's clock (later files' lap times are shifted by the wall-clock
    delta between file starts). Boundary partial laps (truncated mid-spin)
    are dropped automatically by ``MergedLogFile``.

    ``_preloaded_logs``/``_offsets_ms``/``_preloaded_detections`` let
    :func:`run_extract` reuse the logs (and their layout detections) it
    already loaded for wall-clock split grouping; external callers should
    pass paths only.

    Track identity and lap boundaries come from
    :func:`load_and_correct_session`: the GPS trace decides the layout, the
    filename/``Venue`` only break ties, and laps are re-split when the
    logger's beacon sat on the wrong start/finish line.

    On failure returns an ExtractResult with ``status="error"`` and empty
    data tables — callers still record the attempt in the manifest so we
    don't retry forever.
    """
    from .._util import MergedLogFile
    from ..profiles import (
        DEFAULT_CHANNEL_NAMES,
        get_logger_id,
        get_profile_for_logger,
    )
    from . import EXTRACTOR_VERSION

    if isinstance(paths, Path):
        paths = [paths]
    if not paths:
        raise ValueError("extract_session requires at least one path")

    ev = extractor_version or EXTRACTOR_VERSION
    primary_path = paths[0]
    stats = [(p, p.stat().st_mtime_ns, p.stat().st_size) for p in paths]
    primary_mtime_ns = stats[0][1]
    total_file_size = sum(s for _, _, s in stats)
    # Use filename date if parent is YYYY-MM-DD
    cand = parse_filename(primary_path)
    date_str = f"{cand.date.year:04d}-{cand.date.month:02d}-{cand.date.day:02d}"
    if fallback_date_str is None:
        fallback_date_str = date_str
    run_nums = [parse_filename(p).run_num for p in paths]
    # Provisional id from the filename alone (a file that fails to load
    # keeps it); the final one below adds the logger's start time and the
    # resolved car / track.
    session_id = session_id_for(
        date_str, None, cand.car, cand.track_canonical or cand.track_raw, run_nums
    )

    empty_session = _build_empty_session_row(
        session_id=session_id,
        path=primary_path,
        mtime_ns=primary_mtime_ns,
        file_size=total_file_size,
        cand=cand,
        extractor_version=ev,
        xrk_paths=[str(p) for p in paths],
    )

    try:
        if _preloaded_logs is not None:
            logs = list(_preloaded_logs)
            dets = list(_preloaded_detections or [])
            if len(dets) != len(logs):
                dets = [
                    detect_layout(lg, declared_track_raw=cand.track_raw, venue_meta=None)
                    for lg in logs
                ]
        else:
            loaded = [load_and_correct_session(p) for p in paths]
            logs = [lg for lg, _ in loaded]
            dets = [d for _, d in loaded]
    except Exception as e:
        logger.exception("load_session failed for %s", primary_path)
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
    # GPS-resolved track identity wins over the filename token. run_extract
    # only merges files whose detected layouts agree, so the file with the
    # most logger laps stands for the session; disagreement (direct callers)
    # is logged. ``n_laps_logger`` is the group total.
    det = max(dets, key=lambda d: d.n_laps_logger)
    for other in dets:
        if other.track_canonical != det.track_canonical or other.layout_id != det.layout_id:
            logger.warning(
                "%s: merged files disagree on GPS layout (%s vs %s); using the one with most laps",
                primary_path.name,
                det.layout_id,
                other.layout_id,
            )
    # A merged row reports the strongest reconciliation any file needed.
    from .layouts import LAP_SOURCE_RESPLIT, TRACK_RES_GPS_OVERRIDE

    det = replace(
        det,
        n_laps_logger=sum(d.n_laps_logger for d in dets),
        lap_source=(
            LAP_SOURCE_RESPLIT
            if any(d.lap_source == LAP_SOURCE_RESPLIT for d in dets)
            else det.lap_source
        ),
        track_resolution=(
            TRACK_RES_GPS_OVERRIDE
            if any(d.track_resolution == TRACK_RES_GPS_OVERRIDE for d in dets)
            else det.track_resolution
        ),
    )
    cand = replace(
        cand,
        track_canonical=det.track_canonical,
        car=_car_name(cand.car, _vehicle_meta(logs[0])),
    )

    if len(logs) == 1:
        log = logs[0]
    else:
        offsets_ms = _offsets_ms
        if offsets_ms is None:
            # Compute wall-clock offsets without splitting (max_gap_s=inf):
            # the caller already decided these files are one session.
            _, _, offsets_ms = split_group_by_wallclock(
                paths, logs, track_canonical=cand.track_canonical, max_gap_s=float("inf")
            )[0]
        log = MergedLogFile(logs, offsets_ms=offsets_ms)

    session_start_utc = _extract_session_datetime_utc(log, date_str, cand.track_canonical)
    session_id = session_id_for(
        date_str,
        _local_wallclock(session_start_utc, cand.track_canonical),
        cand.car,
        cand.track_canonical or cand.track_raw,
        run_nums,
    )
    empty_session = _build_empty_session_row(
        session_id=session_id,
        path=primary_path,
        mtime_ns=primary_mtime_ns,
        file_size=total_file_size,
        cand=cand,
        extractor_version=ev,
        xrk_paths=[str(p) for p in paths],
        layout=det,
    )

    # Resolve profile + channel names.
    logger_id = get_logger_id(log) or ""
    profile = get_profile_for_logger(logger_id) if logger_id else None
    channel_names = dict(DEFAULT_CHANNEL_NAMES)
    profile_name = "default"
    if profile is not None:
        channel_names.update(profile.channel_names)
        profile_name = profile.name

    # The channels to keep, per family, in this profile's AIM names.
    wanted = _wanted_channels(channel_names)
    present = [aim for aim in wanted if aim in log.channels]

    laps_table = log.laps
    if not present or len(laps_table) == 0:
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

    lap_nums = [int(x) for x in laps_table.column("num").to_pylist()]
    starts = laps_table.column("start_time").to_numpy().astype(np.int64)
    ends = laps_table.column("end_time").to_numpy().astype(np.int64)
    lap_types = (
        [str(x) for x in laps_table.column("lap_type").to_pylist()]
        if "lap_type" in laps_table.column_names
        else ["full"] * len(lap_nums)
    )
    stint_ids = assign_stint_ids(laps_table).to_numpy().astype(np.int16)

    timeseries, tb_summary = build_session_timeseries(
        log,
        channel_names,
        session_id=session_id,
        lap_nums=lap_nums,
        lap_starts_ms=starts.tolist(),
        lap_ends_ms=ends.tolist(),
        stint_ids=[int(x) for x in stint_ids],
    )

    if len(timeseries) == 0:
        row = empty_session.set_column(
            empty_session.schema.get_field_index("status"),
            "status",
            pa.array(["error"]),
        ).set_column(
            empty_session.schema.get_field_index("error_msg"),
            "error_msg",
            pa.array(["no samples inside any lap"]),
        )
        return ExtractResult(
            session_row=row,
            laps_rows=pa.table({}),
            timeseries=pa.table({}),
            status="error",
            error_msg="no_samples",
        )

    # Sensor verdict: validate the IMU against GPS on the fastest full lap
    # and assemble the chosen series on one clock for the lap dynamics.
    lap_times_s = ((ends - starts) / 1000.0).tolist()
    ref_lap = choose_reference_lap(lap_nums, lap_times_s, lap_types)
    verdict = validate_session(timeseries, ref_lap)
    kin = session_kinematics(timeseries, verdict)

    # Build per-lap summary.
    laps_rows = _build_laps_table(
        session_id=session_id,
        lap_nums=lap_nums,
        lap_starts_ms=starts.tolist(),
        lap_ends_ms=ends.tolist(),
        stint_ids=stint_ids.tolist(),
        timeseries=timeseries,
        lap_types=lap_types,
        kinematics=kin,
    )

    # A pit exit is a new stint whatever the gap: the tire's starting
    # temperature is re-measured there. Re-split stints at every from-pit
    # out-lap and carry the new ids onto the timeseries.
    laps_rows, timeseries = _restint_at_pit_exits(laps_rows, timeseries)

    names = timeseries.schema.names
    has_tpms = any(f"tpms_press_{c}_bar" in names for c in CORNERS)
    has_surface_temp = any(n.startswith("surf_temp_") for n in names)
    status = "ok" if has_tpms else "partial"
    err = None if status == "ok" else "tpms channels missing"

    session_row = _build_session_row(
        session_id=session_id,
        path=primary_path,
        mtime_ns=primary_mtime_ns,
        file_size=total_file_size,
        cand=cand,
        extractor_version=ev,
        logger_id=logger_id,
        profile_name=profile_name,
        n_laps=len(laps_rows),
        has_tpms=has_tpms,
        has_surface_temp=has_surface_temp,
        has_ambient=False,  # AIM logs rarely carry a dedicated ambient channel in this fleet
        has_track=False,
        timebases=tb_summary,
        kinematics=verdict,
        session_start_utc=session_start_utc,
        status=status,
        error_msg=err,
        xrk_paths=[str(p) for p in paths],
        layout=det,
        comment=session_comment(logs),
    )

    return ExtractResult(
        session_row=session_row,
        laps_rows=laps_rows,
        timeseries=timeseries,
        status=status,
        error_msg=err,
        timebases=tb_summary,
        kinematics=verdict.to_json(),
    )


def _restint_at_pit_exits(laps_rows: pa.Table, timeseries: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Start a new stint at each out-lap that left the pits.

    :func:`assign_stint_ids` splits stints on long gaps only; a short pit
    visit (tyre pressures, a quick check) keeps the stint id although the
    tire cooled and its temperature was read again at pit exit. Stint ids
    stay 1-based and monotonic; the timeseries ``stint_id`` column follows
    the lap table.
    """
    if len(laps_rows) == 0 or "outlap_from_pit" not in laps_rows.column_names:
        return laps_rows, timeseries
    lap_nums = laps_rows.column("lap_num").to_pylist()
    old = laps_rows.column("stint_id").to_pylist()
    from_pit = laps_rows.column("outlap_from_pit").to_pylist()
    order = sorted(range(len(lap_nums)), key=lambda i: lap_nums[i])
    new = [0] * len(lap_nums)
    cur = 0
    prev_old = None
    for i in order:
        if prev_old is None or old[i] != prev_old or from_pit[i]:
            if prev_old is not None or cur == 0:
                cur += 1
        new[i] = cur
        prev_old = old[i]
    if new == old:
        return laps_rows, timeseries
    idx = laps_rows.schema.get_field_index("stint_id")
    laps_rows = laps_rows.set_column(idx, "stint_id", pa.array(new, type=pa.int16()))
    # Vectorised: the session table has millions of rows at native rates.
    keys = np.asarray(lap_nums, dtype=np.int64)
    vals = np.asarray(new, dtype=np.int16)
    ts_laps = timeseries.column("lap_num").to_numpy().astype(np.int64)
    order_k = np.argsort(keys, kind="stable")
    keys_sorted, vals_sorted = keys[order_k], vals[order_k]
    pos = np.clip(np.searchsorted(keys_sorted, ts_laps), 0, len(keys_sorted) - 1)
    hit = keys_sorted[pos] == ts_laps
    mapped = np.where(hit, vals_sorted[pos], np.int16(1)).astype(np.int16)
    ts_idx = timeseries.schema.get_field_index("stint_id")
    timeseries = timeseries.set_column(ts_idx, "stint_id", pa.array(mapped, type=pa.int16()))
    return laps_rows, timeseries


def _vehicle_meta(log) -> str | None:
    """The Race Studio ``Vehicle`` metadata of a log, or None when absent/blank."""
    try:
        value = (log.metadata or {}).get("Vehicle")
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return None
    value = (value or "").strip()
    return value or None


def _car_name(filename_car: str | None, vehicle_meta: str | None) -> str | None:
    """Car name for the session row: the filename token in the logger's spelling.

    Race Studio names the file after the ``Vehicle`` metadata, so the two
    normally agree. When they differ only in letter case (a recording that
    reached the AIM tree as ``cmd_kk-f_...xrk`` next to the ``CMD_KK-F_...``
    files of the same day) the metadata spelling wins, otherwise the model
    would see a separate ``kk-f`` car. A token that differs in substance
    stays as written in the filename — the filename is the operator's
    classification, the metadata only its spelling.
    """
    if filename_car is None:
        return vehicle_meta
    if vehicle_meta and vehicle_meta.casefold() == filename_car.casefold():
        return vehicle_meta
    return filename_car


def _build_empty_session_row(
    *,
    session_id: str,
    path: Path,
    mtime_ns: int,
    file_size: int,
    cand: SessionCandidate,
    extractor_version: str,
    xrk_paths: list[str] | None = None,
    layout: LayoutDetection | None = None,
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
        has_tpms=False,
        has_surface_temp=False,
        has_ambient=False,
        has_track=False,
        timebases=None,
        kinematics=None,
        session_start_utc=datetime(
            cand.date.year, cand.date.month, cand.date.day, tzinfo=timezone.utc
        ),
        status="pending",
        error_msg=None,
        xrk_paths=xrk_paths,
        layout=layout,
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
    has_tpms: bool,
    has_surface_temp: bool,
    has_ambient: bool,
    has_track: bool,
    timebases: list[dict] | None,
    kinematics: KinematicsVerdict | None,
    session_start_utc: datetime,
    status: str = "ok",
    error_msg: str | None = None,
    xrk_paths: list[str] | None = None,
    layout: LayoutDetection | None = None,
    comment: str | None = None,
) -> pa.Table:
    paths_list = xrk_paths if xrk_paths is not None else [str(path)]
    lay = layout
    tb_text = (
        ";".join(f"{e['timebase']}:{e['rate_hz']:.2f}Hz:{e['n']}" for e in timebases)
        if timebases
        else None
    )
    kin_row = kinematics.to_row() if kinematics is not None else empty_verdict_row()
    kin_cols = {name: pa.array([kin_row.get(name)], type=typ) for name, typ in KIN_ROW_COLUMNS}
    return pa.table(
        {
            "session_id": [session_id],
            "xrk_path": [str(path)],
            "xrk_paths": pa.array([paths_list], type=pa.list_(pa.string())),
            "xrk_mtime_ns": pa.array([mtime_ns], type=pa.int64()),
            "file_size": pa.array([file_size], type=pa.int64()),
            "date": pa.array([cand.date], type=pa.date32()),
            "session_start_utc": pa.array([session_start_utc], type=pa.timestamp("us", tz="UTC")),
            "driver": [cand.driver],
            "car": [cand.car],
            "track": [cand.track_raw],
            "track_canonical": [cand.track_canonical],
            # GPS layout reconciliation (see layouts.py). ``track_canonical``
            # above is already the GPS-resolved value; these columns record
            # how it was reached so misidentified variants can be audited.
            "venue_meta": pa.array([lay.venue_meta if lay else None], type=pa.string()),
            "layout_id": pa.array([lay.layout_id if lay else None], type=pa.string()),
            "track_declared_canonical": pa.array(
                [lay.declared_track_canonical if lay else None], type=pa.string()
            ),
            "track_resolution": pa.array([lay.track_resolution if lay else None], type=pa.string()),
            "lap_source": pa.array([lay.lap_source if lay else None], type=pa.string()),
            "beacon_gate": pa.array([lay.beacon_gate if lay else None], type=pa.string()),
            "n_laps_logger": pa.array([lay.n_laps_logger if lay else None], type=pa.int32()),
            "lap_length_m": pa.array([lay.lap_length_m if lay else None], type=pa.float64()),
            "session_type": [cand.session_type],
            "run_num": pa.array([cand.run_num], type=pa.int32()),
            # RaceStudio "Long Comment" of the file(s): the driver's own note
            # on tires / setup / result (see file_comment).
            "comment": pa.array([comment], type=pa.string()),
            "logger_id": [logger_id],
            "profile_name": [profile_name],
            "extractor_version": [extractor_version],
            "extracted_at": pa.array(
                [datetime.now(timezone.utc)], type=pa.timestamp("us", tz="UTC")
            ),
            "status": [status],
            "error_msg": pa.array([error_msg], type=pa.string()),
            "n_laps": pa.array([n_laps], type=pa.int32()),
            "has_tpms": [has_tpms],
            "has_surface_temp": [has_surface_temp],
            "has_ambient_temp": [has_ambient],
            "has_track_temp": [has_track],
            # Native-rate storage (schema 7): the raw timebases the file
            # carries, "name:rate:n" per timebase, and the sensor verdict.
            "n_timebases": pa.array([len(timebases) if timebases else 0], type=pa.int32()),
            "timebases": pa.array([tb_text], type=pa.string()),
            **kin_cols,
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
    lap_types: list[str] | None = None,
    kinematics: pa.Table | None = None,
) -> pa.Table:
    """Per-lap summary rows.

    ``kinematics`` (:func:`~.kinematics.session_kinematics`) supplies the
    accelerations the dynamics columns are computed from — the session's
    validated sensor on its own clock; without it the raw ``lat_g`` /
    ``long_g`` columns of the timeseries are used.

    ``lap_types`` (libxrk's ``full`` / ``out`` / ``in``) sets ``is_outlap`` /
    ``is_inlap`` directly. Without it (legacy callers) the last lap of a
    stint is an in-lap when it is > 1.2× the stint median; no lap is an
    out-lap — the first *full* lap of a stint is a flying lap and used to
    be mis-flagged as the out-lap before the real out-lap was retained.
    """
    if lap_types is None:
        lap_types = ["full"] * len(lap_nums)
    rows: dict[str, list] = {
        "session_id": [],
        "lap_num": [],
        "stint_id": [],
        "lap_time_s": [],
        "lap_type": [],
        "is_outlap": [],
        "is_inlap": [],
        "outlap_from_pit": [],
        "speed_kmh_mean": [],
        "speed_kmh_max": [],
        "brake_mean": [],
        "brake_max": [],
        "throttle_mean": [],
        "lat_g_peak": [],
        "long_g_peak_brake": [],
        "heat_proxy": [],
        "heat_proxy_fl": [],
        "heat_proxy_fr": [],
        "heat_proxy_rl": [],
        "heat_proxy_rr": [],
        "on_track_s": [],
        "moving_s": [],
        "speed_kmh_first": [],
        "distance_m": [],
    }
    for c in CORNERS:
        for stat in ("start", "end", "min", "max", "mean"):
            rows[f"tpms_press_{c}_{stat}"] = []
            rows[f"tpms_temp_{c}_{stat}"] = []
        rows[f"tpms_press_{c}_rise_bar_per_min"] = []
        rows[f"tpms_temp_{c}_first_valid_s"] = []
        rows[f"tpms_press_{c}_first_valid_s"] = []
        for stat in ("mean", "min", "max"):
            rows[f"surf_temp_{c}_{stat}"] = []

    lap_num_col = timeseries.column("lap_num").to_numpy()
    kin_lap_col = (
        kinematics.column("lap_num").to_numpy()
        if kinematics is not None and "lap_num" in kinematics.schema.names
        else None
    )

    # Per-stint last-lap detection (potential in-lap, legacy heuristic).
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
        rows["lap_type"].append(lap_types[i])
        is_outlap = lap_types[i] == "out"
        rows["is_outlap"].append(is_outlap)
        # In-lap: typed by the logger, else the legacy heuristic — last lap
        # of the stint AND lap_time > 1.2 × stint median.
        is_inlap = lap_types[i] == "in" or (
            ln == stint_last_lap.get(stint_ids[i])
            and lap_times_s[i] > 1.2 * stint_median.get(stint_ids[i], float("inf"))
        )
        rows["is_inlap"].append(is_inlap)

        kin_lap = (
            kinematics.filter(pa.array(kin_lap_col == ln))
            if kinematics is not None and kin_lap_col is not None
            else None
        )
        dyn = compute_lap_dynamics(lap_ts, kin_lap)
        # An out-lap that begins below pit-lane speed started from standstill
        # (a real pit exit); one that begins at speed is a file that started
        # mid-track and must not be used as a stint's starting temperature.
        first_speed = dyn.get("speed_kmh_first", float("nan"))
        rows["outlap_from_pit"].append(
            bool(is_outlap and not np.isnan(first_speed) and first_speed < 30.0)
        )
        for k in (
            "speed_kmh_mean",
            "speed_kmh_max",
            "brake_mean",
            "brake_max",
            "throttle_mean",
            "lat_g_peak",
            "long_g_peak_brake",
            "heat_proxy",
            "heat_proxy_fl",
            "heat_proxy_fr",
            "heat_proxy_rl",
            "heat_proxy_rr",
            "on_track_s",
            "moving_s",
            "speed_kmh_first",
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
            rows[f"tpms_temp_{c}_first_valid_s"].append(agg.temp_first_valid_s)
            rows[f"tpms_press_{c}_first_valid_s"].append(agg.press_first_valid_s)
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
        pa.field("lap_type", pa.string()),
        pa.field("is_outlap", pa.bool_()),
        pa.field("is_inlap", pa.bool_()),
        pa.field("outlap_from_pit", pa.bool_()),
        pa.field("speed_kmh_mean", pa.float32()),
        pa.field("speed_kmh_max", pa.float32()),
        pa.field("brake_mean", pa.float32()),
        pa.field("brake_max", pa.float32()),
        pa.field("throttle_mean", pa.float32()),
        pa.field("lat_g_peak", pa.float32()),
        pa.field("long_g_peak_brake", pa.float32()),
        pa.field("heat_proxy", pa.float32()),
        pa.field("heat_proxy_fl", pa.float32()),
        pa.field("heat_proxy_fr", pa.float32()),
        pa.field("heat_proxy_rl", pa.float32()),
        pa.field("heat_proxy_rr", pa.float32()),
        pa.field("on_track_s", pa.float32()),
        pa.field("moving_s", pa.float32()),
        pa.field("speed_kmh_first", pa.float32()),
        pa.field("distance_m", pa.float32()),
    ]
    for c in CORNERS:
        for stat in ("start", "end", "min", "max", "mean"):
            schema_fields.append(pa.field(f"tpms_press_{c}_{stat}", pa.float32()))
            schema_fields.append(pa.field(f"tpms_temp_{c}_{stat}", pa.float32()))
        schema_fields.append(pa.field(f"tpms_press_{c}_rise_bar_per_min", pa.float32()))
        schema_fields.append(pa.field(f"tpms_temp_{c}_first_valid_s", pa.float32()))
        schema_fields.append(pa.field(f"tpms_press_{c}_first_valid_s", pa.float32()))
        for stat in ("mean", "min", "max"):
            schema_fields.append(pa.field(f"surf_temp_{c}_{stat}", pa.float32()))
    schema = pa.schema(schema_fields)

    # Build table with explicit schema — pa.table handles type coercion via
    # construction, but we cast after to be safe.
    return pa.table(rows).cast(schema)


def _split_on_layout(
    split: list[tuple[list[Path], list, list[int]]],
    det_by_path: dict[str, LayoutDetection],
) -> list[tuple[list[Path], list, list[int]]]:
    """Break wall-clock subgroups further wherever the detected layout changes.

    Filename grouping pools dash variants of one venue ("Suzuka West" +
    "Suzuka Car" both resolve to ``suzuka``), so a one-lap wrong-variant
    fragment can sit next to a proper session with consecutive run numbers.
    Their GPS-detected layouts differ, and merging them would stamp the
    fragment's layout and beacon on the real session.
    """
    out: list[tuple[list[Path], list, list[int]]] = []
    for sp, sl, so in split:
        start = 0
        for i in range(1, len(sp) + 1):
            if i < len(sp):
                a = det_by_path[str(sp[i - 1])]
                b = det_by_path[str(sp[i])]
                same = a.layout_id == b.layout_id and a.track_canonical == b.track_canonical
                if same:
                    continue
                logger.info(
                    "layout split: %s (%s) | %s (%s)",
                    sp[i - 1].name,
                    a.layout_id,
                    sp[i].name,
                    b.layout_id,
                )
            base = so[start]
            out.append((sp[start:i], sl[start:i], [o - base for o in so[start:i]]))
            start = i
    return out


def _extract_group(paths: list[Path], dataset_root: Path) -> dict[str, str]:
    """Extract one file group (split on wall-clock gaps and layout, one
    session per subgroup) and upsert the results; ``{path: status}``."""
    from .dataset import upsert_session

    t0 = time.perf_counter()
    # Filename grouping can fuse two separate track sessions that AIM
    # happened to number consecutively; verify wall-clock adjacency
    # before merging. Load failures fall through to extract_session,
    # which retries the load and records the error result uniformly.
    subgroups: list[tuple[list[Path], list | None, list[int] | None, list | None]]
    if len(paths) == 1:
        subgroups = [(paths, None, None, None)]
    else:
        try:
            loaded = [load_and_correct_session(p) for p in paths]
        except Exception:
            subgroups = [(paths, None, None, None)]
        else:
            logs = [lg for lg, _ in loaded]
            det_by_path = {str(p): d for p, (_, d) in zip(paths, loaded)}
            split = split_group_by_wallclock(
                paths, logs, track_canonical=loaded[0][1].track_canonical
            )
            if len(split) > 1:
                logger.info(
                    "wall-clock gap split %d-file group (primary=%s) into %d sessions",
                    len(paths),
                    paths[0].name,
                    len(split),
                )
            split = _split_on_layout(split, det_by_path)
            subgroups = [(sp, sl, so, [det_by_path[str(p)] for p in sp]) for sp, sl, so in split]

    status_by_path: dict[str, str] = {}
    for sub_paths, sub_logs, sub_offsets, sub_dets in subgroups:
        result = extract_session(
            sub_paths,
            _preloaded_logs=sub_logs,
            _offsets_ms=sub_offsets,
            _preloaded_detections=sub_dets,
        )
        if len(sub_paths) == 1:
            logger.info(
                "extracted %s status=%s elapsed=%.2fs",
                sub_paths[0].name,
                result.status,
                time.perf_counter() - t0,
            )
        else:
            logger.info(
                "extracted %d-file merged session %s (primary=%s) status=%s elapsed=%.2fs",
                len(sub_paths),
                result.session_row.column("session_id")[0].as_py(),
                sub_paths[0].name,
                result.status,
                time.perf_counter() - t0,
            )

        upsert_session(dataset_root, result, all_xrk_paths=sub_paths)
        for p in sub_paths:
            status_by_path[str(p)] = result.status
    return status_by_path


def run_extract(
    *,
    aim_root: Path,
    dataset_root: Path,
    since=None,
    only_car: str | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> dict[str, int]:
    """Extract all new sessions and upsert them into the dataset.

    Files are parsed by libxrk's Rust backend (``LIBXRK_BACKEND=rust``, set
    by the CLI): the Cython parser leaks ~30 MB of native memory per file
    (m3rlin45/libxrk#111), which took a full re-extract past 14 GB; the Rust
    parser is flat and its output matches the Cython one on every logger
    channel, timecode and lap (its derived GPS latitude / altitude / lateral
    g differ by 1e-14° / 3e-9 m / one float32 ulp).

    Files that the AIM logger split across a single physical run (matching
    ``date / driver / car / track / session_type`` and with consecutive
    ``run_num`` values) are grouped via
    :func:`~motorsports_data_notebook.tire_etl.discovery.group_split_sessions`,
    then verified against wall-clock metadata with
    :func:`split_group_by_wallclock` — consecutive run numbers more than
    ``MERGE_MAX_GAP_S`` apart are separate track sessions, not restart
    splits, and are extracted individually. Each constituent file still gets
    its own manifest row (so re-runs detect when any file changed) pointing
    at its (sub)group's session_id.
    """
    from .dataset import load_manifest, prune_orphan_sessions, remove_manifest_entries
    from .discovery import group_split_sessions, stem_key

    existing = load_manifest(dataset_root)
    counts = {"scanned": 0, "skipped": 0, "extracted": 0, "errors": 0}

    candidates = scan_aim_tree(aim_root, since=since, only_car=only_car)
    # A recording the scan now lists under another spelling (the lowercase
    # .xrk that appeared next to the .xrz an earlier run extracted) supersedes
    # the manifest row of the old spelling; left in place, both spellings
    # would stay in the dataset and the session would be counted twice. Only
    # a same-stem replacement is removed here — a file that merely vanished
    # from the tree keeps its row, so an unmounted or half-synced AIM drive
    # cannot empty the dataset.
    scanned_by_stem = {stem_key(c.path): str(c.path) for c in candidates}
    scanned_paths = set(scanned_by_stem.values())
    superseded = [
        old
        for old in existing
        if old not in scanned_paths and stem_key(Path(old)) in scanned_by_stem
    ]
    for old in superseded:
        logger.info(
            "manifest: %s superseded by %s",
            Path(old).name,
            Path(scanned_by_stem[stem_key(Path(old))]).name,
        )
        del existing[old]
    counts["superseded"] = remove_manifest_entries(dataset_root, superseded)
    groups = group_split_sessions(candidates)
    from . import EXTRACTOR_VERSION

    for group in groups:
        counts["scanned"] += len(group)
        stats = [(c.path, c.path.stat()) for c in group]
        mtimes = [s.st_mtime_ns for _, s in stats]
        sizes = [s.st_size for _, s in stats]

        # Skip only if EVERY constituent file is cache-clean — any change
        # to any file in the group invalidates the merged session.
        all_clean = True
        for cand, mtime_ns, size in zip(group, mtimes, sizes):
            prior = existing.get(str(cand.path))
            if (
                prior is None
                or prior.get("xrk_mtime_ns") != mtime_ns
                or prior.get("file_size") != size
                or prior.get("extractor_version") != EXTRACTOR_VERSION
                or (prior.get("status") == "error" and retry_errors)
            ):
                all_clean = False
                break
        if all_clean and not force:
            counts["skipped"] += len(group)
            continue

        paths = [c.path for c in group]
        status_by_path = _extract_group(paths, dataset_root)
        for p in paths:
            if status_by_path.get(str(p), "error") == "error":
                counts["errors"] += 1
            else:
                counts["extracted"] += 1

        for cand, mtime_ns, size in zip(group, mtimes, sizes):
            existing[str(cand.path)] = {
                "xrk_mtime_ns": mtime_ns,
                "file_size": size,
                "extractor_version": EXTRACTOR_VERSION,
                "status": status_by_path.get(str(cand.path), "error"),
            }
    # Grouping changes leave the previous grouping's rows behind (they are
    # keyed by a session_id nothing writes any more); drop them.
    pruned = prune_orphan_sessions(dataset_root)
    counts["pruned_sessions"] = pruned["session_rows"]
    return counts
