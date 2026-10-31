"""Parquet I/O, partitioning, and MANIFEST ledger for the tire dataset.

Design principles
-----------------
- **Deterministic output.** Partition files are always rewritten sorted by
  ``(session_id, lap_num, sample_idx)``. No row ordering surprises in diffs.
- **Atomic writes.** Write to ``{target}.tmp``, fsync, rename.
- **MANIFEST.jsonl is canonical.** Each line is ``{session_id, xrk_path,
  xrk_mtime_ns, file_size, extractor_version, status, n_laps, n_samples,
  extracted_at}``, sorted by ``(date, session_id)``.

Timeseries layout (dataset schema 9)
------------------------------------
One parquet per session, ``timeseries/YYYY-MM/<session_id>.parquet``, holding
every selected channel **verbatim at its native rate**: one row per raw
sample, one row group per raw *timebase* (a set of channels the logger
sampled on one timecode array), the ``timebase`` and ``family`` columns
naming it, and every channel column an Arrow null outside the rows of its
own timebase. Nothing is resampled or aligned here; merging onto one clock
is done at query time by :func:`load_group` consumers,
:func:`~.kinematics.load_kinematics` and the per-lap aggregates. The file's
schema metadata carries the per-timebase summary and the session's sensor
verdict (``tire_etl.timebases`` / ``tire_etl.kinematics``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .aggregates import channel_series
from .paths import (
    laps_dir,
    manifest_path,
    sessions_dir,
    timeseries_dir,
)

__all__ = ["channel_series"]

if TYPE_CHECKING:
    from .extract import ExtractResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    session_id: str
    xrk_path: str
    xrk_mtime_ns: int
    file_size: int
    extractor_version: str
    status: str
    n_laps: int
    n_samples: int
    date: str  # ISO YYYY-MM-DD
    extracted_at: str  # ISO UTC


def load_manifest(dataset_root: Path) -> dict[str, dict]:
    """Load MANIFEST.jsonl keyed by ``xrk_path`` for fast lookup."""
    path = manifest_path(dataset_root)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "xrk_path" in entry:
            out[entry["xrk_path"]] = entry
    return out


def _write_manifest(dataset_root: Path, entries: list[dict]) -> None:
    path = manifest_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries_sorted = sorted(entries, key=lambda e: (e.get("date", ""), e.get("session_id", "")))
    lines = [json.dumps(e, sort_keys=True, ensure_ascii=False) for e in entries_sorted]
    content = "\n".join(lines) + ("\n" if lines else "")
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _update_manifest(dataset_root: Path, new_entry: dict) -> None:
    """Insert-or-replace a manifest entry keyed by xrk_path."""
    existing = load_manifest(dataset_root)
    existing[new_entry["xrk_path"]] = new_entry
    _write_manifest(dataset_root, list(existing.values()))


def remove_manifest_entries(dataset_root: Path, xrk_paths: list[str]) -> int:
    """Drop the manifest rows of ``xrk_paths``; returns how many were removed.

    The session rows those entries pointed at become orphans for
    :func:`prune_orphan_sessions` unless another manifest row still
    references the same ``session_id``.
    """
    existing = load_manifest(dataset_root)
    removed = 0
    for p in xrk_paths:
        if existing.pop(p, None) is not None:
            removed += 1
    if removed:
        _write_manifest(dataset_root, list(existing.values()))
    return removed


def _partition_for_date(date_str: str) -> str:
    return date_str[:7]  # YYYY-MM


def _read_partition(path: Path) -> pa.Table | None:
    if not path.exists():
        return None
    return pq.read_table(path)


def _atomic_write_parquet(table: pa.Table, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, target)


TIMESERIES_SCHEMA_VERSION = 9
# An interval at least this long inside a timebase is a dropout, not a
# sample period: it is excluded from the rate and counted as gap time.
RATE_GAP_S = 1.0
# Largest magnitude float16 represents; a channel with a finite value beyond
# it stays float32 whatever its source.
_FLOAT16_MAX = 65504.0
TIMESERIES_KEY_COLUMNS = [
    "session_id",
    "timebase",
    "family",
    "lap_num",
    "stint_id",
    "sample_idx",
    "t_session_s",
    "t_lap_s",
]
FAMILIES = ("gps", "accel", "gyro", "tpms", "surface", "controls")
_META_TIMEBASES = b"tire_etl.timebases"
_META_KINEMATICS = b"tire_etl.kinematics"
_META_SCHEMA = b"tire_etl.schema"


def channel_family(name: str) -> str | None:
    """The sensor family a canonical channel column belongs to, or None for
    a key column / unknown name."""
    if name.startswith("tpms_"):
        return "tpms"
    if name.startswith("surf_temp_"):
        return "surface"
    if name.startswith("gps_") or name == "speed_ms":
        return "gps"
    if name in ("lat_g", "long_g", "vert_g"):
        return "accel"
    if name in ("yaw_rate_dps", "pitch_rate_dps", "roll_rate_dps"):
        return "gyro"
    if name in ("throttle_pct", "brake_bar", "steer_deg"):
        return "controls"
    return None


def family_columns(names: Iterable[str], family: str) -> list[str]:
    """The channel columns of ``family`` among ``names``, in their order."""
    return [n for n in names if channel_family(n) == family]


def narrowest_float(values: np.ndarray, *, float64: bool = False) -> pa.Array:
    """``values`` as the narrowest Arrow float array that holds them exactly.

    libxrk returns every logger channel as float32 although the AIM file
    stores the IMU, the gyro and the CAN channels as float16 (CHS decoder
    type 20; libxrk issue #115). A channel whose finite samples are all
    float16-representable is stored as float16: the cast is exact, the
    column halves, and a channel the logger really logged as float32 (an
    analog input, an IR zone, a TPMS pressure) never satisfies the test on
    real data. ``float64`` keeps the column at float64 (positions).
    """
    v = np.asarray(values)
    if float64:
        return pa.array(v.astype(np.float64), type=pa.float64())
    v32 = v.astype(np.float32)
    finite = v32[np.isfinite(v32)]
    if (
        finite.size
        and float(np.max(np.abs(finite))) <= _FLOAT16_MAX
        and np.array_equal(finite.astype(np.float16).astype(np.float32), finite)
    ):
        return pa.array(v32.astype(np.float16), type=pa.float16())
    return pa.array(v32, type=pa.float32())


def build_timeseries_table(session_id: str, parts: list[tuple[str, str, pa.Table]]) -> pa.Table:
    """Assemble the session table from per-timebase parts.

    Each part is ``(timebase, family, table)`` where ``table`` has the key
    columns ``lap_num, stint_id, sample_idx, t_session_s, t_lap_s`` and the
    timebase's channel columns. The parts are stacked with the channels of
    the other timebases as nulls, key columns first, then the channels
    ordered by family and name.
    """
    tables: list[pa.Table] = []
    for timebase, family, tbl in parts:
        if len(tbl) == 0:
            continue
        n = len(tbl)
        tbl = tbl.add_column(0, "family", pa.array([family] * n, type=pa.string()))
        tbl = tbl.add_column(0, "timebase", pa.array([timebase] * n, type=pa.string()))
        tbl = tbl.add_column(0, "session_id", pa.array([session_id] * n, type=pa.string()))
        tables.append(tbl)
    if not tables:
        return pa.table({})
    merged = pa.concat_tables(tables, promote_options="default")
    channels = [c for c in merged.schema.names if c not in TIMESERIES_KEY_COLUMNS]
    order = [c for c in TIMESERIES_KEY_COLUMNS if c in merged.schema.names]
    for fam in FAMILIES:
        order += sorted(family_columns(channels, fam))
    order += sorted(c for c in channels if channel_family(c) is None)
    return merged.select(order)


def timeseries_summary(table: pa.Table) -> list[dict]:
    """One entry per timebase: name, family, row count, rate (Hz) and the
    channels it carries.

    The rate is the number of samples over the time they cover, counting
    only the intervals shorter than :data:`RATE_GAP_S`: a median interval
    would call the KK-SII's accelerometer, logged as pairs 2 ms apart on
    50 Hz frames, 500 Hz, and a span-based mean under-reports a stream
    with dropouts (an IMU that stops for minutes mid-session). The gap
    time and count are reported alongside (``gap_s``, ``n_gaps``)."""
    if len(table) == 0 or "timebase" not in table.schema.names:
        return []
    out: list[dict] = []
    tb_col = table.column("timebase")
    for name in pc.unique(tb_col).to_pylist():
        sub = table.filter(pc.equal(tb_col, name))
        t = np.sort(
            np.asarray(sub.column("t_session_s").to_numpy(zero_copy_only=False), dtype=float)
        )
        dt = np.diff(t)
        live = dt[(dt > 0) & (dt < RATE_GAP_S)]
        rate = float(live.size / live.sum()) if live.size and live.sum() > 0 else float("nan")
        gaps = dt[dt >= RATE_GAP_S]
        channels = [
            c
            for c in sub.schema.names
            if c not in TIMESERIES_KEY_COLUMNS and sub.column(c).null_count < len(sub)
        ]
        out.append(
            {
                "timebase": name,
                "family": sub.column("family")[0].as_py(),
                "n": len(sub),
                "rate_hz": rate,
                "gap_s": float(gaps.sum()) if gaps.size else 0.0,
                "n_gaps": int(gaps.size),
                "channels": channels,
            }
        )
    out.sort(
        key=lambda e: (
            FAMILIES.index(e["family"]) if e["family"] in FAMILIES else 99,
            e["timebase"],
        )
    )
    return out


def write_timeseries(
    table: pa.Table,
    target: Path,
    *,
    summary: list[dict] | None = None,
    kinematics: dict | None = None,
) -> None:
    """Atomically write a session table with one row group per timebase.

    Rows are sorted by ``(timebase, t_session_s)``; readers filter on
    ``family`` / ``timebase`` and parquet's row-group statistics skip the
    other timebases. ``summary`` (see :func:`timeseries_summary`) and the
    sensor ``kinematics`` verdict go into the schema metadata so a loader
    needs only the file.

    Encoding: zstd level 9 over byte-stream-split for the float32/float64
    columns, PLAIN for float16 (DuckDB cannot read byte-stream-split
    float16), delta packing for ``sample_idx``, dictionary encoding only on
    the string keys. Parquet's default dictionary attempt on a float
    column of mostly distinct values falls back to PLAIN, which zstd
    barely compresses: the same 20 MB session file is 8.7 MB this way,
    identical values.
    """
    if "timebase" in table.schema.names and len(table) > 1:
        idx = pc.sort_indices(
            table, sort_keys=[("timebase", "ascending"), ("t_session_s", "ascending")]
        )
        table = table.take(idx)
    meta = dict(table.schema.metadata or {})
    meta[_META_SCHEMA] = str(TIMESERIES_SCHEMA_VERSION).encode()
    if summary is not None:
        meta[_META_TIMEBASES] = json.dumps(summary).encode()
    if kinematics is not None:
        meta[_META_KINEMATICS] = json.dumps(kinematics).encode()
    table = table.replace_schema_metadata(meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    wide_floats = [f.name for f in table.schema if f.type in (pa.float32(), pa.float64())]
    strings = [f.name for f in table.schema if pa.types.is_string(f.type)]
    encoding = {"sample_idx": "DELTA_BINARY_PACKED"} if "sample_idx" in table.schema.names else {}
    writer_opts: dict = dict(
        compression="zstd",
        compression_level=9,
        use_dictionary=strings,
        use_byte_stream_split=wide_floats,
        column_encoding=encoding,
    )
    if "timebase" in table.schema.names and len(table):
        tb = table.column("timebase").to_numpy(zero_copy_only=False)
        bounds = np.flatnonzero(np.concatenate([[True], tb[1:] != tb[:-1], [True]]))
        with pq.ParquetWriter(tmp, table.schema, **writer_opts) as w:
            for a, b in zip(bounds[:-1], bounds[1:]):
                piece = table.slice(int(a), int(b - a))
                w.write_table(piece, row_group_size=max(len(piece), 1))
    else:
        pq.write_table(table, tmp, **writer_opts)
    os.replace(tmp, target)


def timeseries_path(dataset_root: Path, session_id: str) -> Path | None:
    """The session's timeseries file, or None."""
    files = sorted(timeseries_dir(dataset_root).glob(f"*/{session_id}.parquet"))
    return files[0] if files else None


def timeseries_metadata(path: Path) -> dict:
    """``{"schema": int | None, "timebases": [...], "kinematics": {...} | None}``
    from the file's schema metadata."""
    meta = pq.read_schema(path).metadata or {}
    schema = meta.get(_META_SCHEMA)
    tbs = meta.get(_META_TIMEBASES)
    kin = meta.get(_META_KINEMATICS)
    return {
        "schema": int(schema) if schema else None,
        "timebases": json.loads(tbs) if tbs else [],
        "kinematics": json.loads(kin) if kin else None,
    }


def _require_current_schema(path: Path, metadata: dict | None) -> None:
    """Refuse a session file the current loaders cannot read correctly.

    Only files written by this schema carry the ``timebase`` / ``family``
    row groups the loaders filter on; an older file (every channel
    resampled onto one clock) would be read as one timebase with the wrong
    rates. Re-extract (``just tire-etl``) instead of reading it.
    """
    raw = (metadata or {}).get(_META_SCHEMA)
    found = int(raw) if raw else None
    if found != TIMESERIES_SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: timeseries schema {found if found is not None else 'missing'}, "
            f"this code reads schema {TIMESERIES_SCHEMA_VERSION} only — re-extract the dataset"
        )


def read_timeseries(
    path: Path,
    *,
    columns: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    timebases: Iterable[str] | None = None,
) -> pa.Table:
    """Read (part of) a session file: the key columns plus ``columns`` that
    exist, restricted to the given families / timebases (row groups the
    statistics rule out are skipped)."""
    file_schema = pq.read_schema(path)
    _require_current_schema(path, file_schema.metadata)
    schema_names = file_schema.names
    cols: list[str] | None = None
    if columns is not None:
        wanted = list(columns)
        cols = [c for c in TIMESERIES_KEY_COLUMNS if c in schema_names]
        cols += [c for c in wanted if c in schema_names and c not in cols]
    filters: list[tuple] = []
    if families is not None and "family" in schema_names:
        filters.append(("family", "in", list(families)))
    if timebases is not None and "timebase" in schema_names:
        filters.append(("timebase", "in", list(timebases)))
    return pq.read_table(path, columns=cols, filters=filters or None)


def load_group(
    dataset_root: Path, session_id: str, family: str, columns: Iterable[str] | None = None
) -> pa.Table | None:
    """The rows of one sensor family (``gps`` / ``accel`` / ``gyro`` /
    ``tpms`` / ``surface`` / ``controls``) with the key columns and the
    family's channels (all of them, or the subset ``columns``); None when
    the session has no file or no such family. Each channel column is
    valid only on its own timebase's rows (:func:`channel_series`)."""
    path = timeseries_path(dataset_root, session_id)
    if path is None:
        return None
    names = pq.read_schema(path).names
    if "family" not in names:
        return None
    chans = family_columns(names, family)
    if columns is not None:
        chans = [c for c in columns if c in chans]
    tbl = read_timeseries(path, columns=chans, families=[family])
    return tbl if len(tbl) else None


def _anti_join_and_concat(
    existing: pa.Table | None,
    new_rows: pa.Table,
    *,
    key: str = "session_id",
    drop_ids: set[str] | None = None,
) -> pa.Table:
    """Return ``existing`` rows with matching session_id removed, then concat ``new_rows``.

    ``drop_ids`` are removed from ``existing`` even when ``new_rows`` is empty:
    a session whose re-extraction produced no lap rows (it became a zero-lap
    error) must not keep the rows of an earlier run.
    """
    if existing is None or len(existing) == 0:
        return new_rows
    incoming_ids = set(new_rows.column(key).to_pylist()) if len(new_rows) > 0 else set()
    incoming_ids |= set(drop_ids or ())
    if not incoming_ids:
        if len(new_rows) == 0:
            return existing
        return pa.concat_tables([existing, new_rows], promote_options="default")
    # Filter out rows from existing whose key is in the incoming set.
    mask = pc.invert(pc.is_in(existing.column(key), pa.array(list(incoming_ids))))
    keep = existing.filter(mask)
    if len(new_rows) == 0:
        return keep
    # Align schemas before concat (missing columns become null), then put the
    # columns in the incoming rows' order so every partition shares one
    # column order — pa.concat_tables without promotion (used by readers)
    # rejects partitions whose schemas differ only in order.
    merged = pa.concat_tables([keep, new_rows], promote_options="default")
    order = [c for c in new_rows.schema.names] + [
        c
        for c in merged.schema.names
        # A column the incoming rows no longer write (dropped by a schema
        # bump) is kept only while an older row still carries a value.
        if c not in new_rows.schema.names and merged.column(c).null_count < len(merged)
    ]
    return merged.select(order)


def _sort_sessions(table: pa.Table) -> pa.Table:
    if len(table) == 0:
        return table
    sort_keys = [("date", "ascending"), ("session_id", "ascending")]
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    return table.take(indices)


def _sort_laps(table: pa.Table) -> pa.Table:
    if len(table) == 0:
        return table
    indices = pc.sort_indices(
        table, sort_keys=[("session_id", "ascending"), ("lap_num", "ascending")]
    )
    return table.take(indices)


def upsert_session(
    dataset_root: Path,
    result: "ExtractResult",
    *,
    all_xrk_paths: list[Path] | None = None,
) -> None:
    """Write one session's tables into the dataset, replacing prior rows.

    The timeseries table gets its own file; sessions and laps partitions are
    rewritten from the union of their prior contents (minus this session) and
    the new rows. When ``all_xrk_paths`` is given (a merged multi-file
    session), one manifest entry is written per constituent file — they all
    point at the same ``session_id`` so any file changing invalidates the
    whole group's cache on the next run.
    """
    if len(result.session_row) != 1:
        raise ValueError("ExtractResult.session_row must have exactly one row")

    session_id = result.session_row.column("session_id")[0].as_py()
    date_str = str(result.session_row.column("date")[0].as_py())
    partition = _partition_for_date(date_str)
    extracted_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # Timeseries: one file per session, one row group per raw timebase. A
    # session that now extracts without samples (an error) loses the file
    # an earlier run wrote for it, so no stale layout survives a re-extract.
    if len(result.timeseries) > 0:
        ts_path = timeseries_dir(dataset_root) / partition / f"{session_id}.parquet"
        write_timeseries(
            result.timeseries,
            ts_path,
            summary=getattr(result, "timebases", None),
            kinematics=getattr(result, "kinematics", None),
        )
    else:
        stale = timeseries_path(dataset_root, str(session_id))
        if stale is not None:
            stale.unlink()

    # Sessions partition.
    sess_path = sessions_dir(dataset_root) / f"{partition}.parquet"
    existing_sessions = _read_partition(sess_path)
    merged_sessions = _anti_join_and_concat(existing_sessions, result.session_row)
    merged_sessions = _sort_sessions(merged_sessions)
    _atomic_write_parquet(merged_sessions, sess_path)

    # Laps partition. Rewritten even when this extraction produced no laps,
    # so a session that turned into a zero-lap error drops its earlier rows.
    laps_path = laps_dir(dataset_root) / f"{partition}.parquet"
    existing_laps = _read_partition(laps_path)
    if len(result.laps_rows) > 0 or existing_laps is not None:
        merged_laps = _anti_join_and_concat(
            existing_laps, result.laps_rows, drop_ids={str(session_id)}
        )
        if len(merged_laps) > 0:
            merged_laps = _sort_laps(merged_laps)
        _atomic_write_parquet(merged_laps, laps_path)

    extractor_version = result.session_row.column("extractor_version")[0].as_py()
    n_laps = result.session_row.column("n_laps")[0].as_py()
    # Manifest: one row per constituent file pointing at the merged session_id.
    if all_xrk_paths is None:
        all_xrk_paths = [Path(result.session_row.column("xrk_path")[0].as_py())]
    for p in all_xrk_paths:
        try:
            stat = p.stat()
            mtime_ns = stat.st_mtime_ns
            size = stat.st_size
        except OSError:
            # Best-effort: fall back to whatever the session row carries.
            mtime_ns = result.session_row.column("xrk_mtime_ns")[0].as_py()
            size = result.session_row.column("file_size")[0].as_py()
        entry = {
            "session_id": session_id,
            "xrk_path": str(p),
            "xrk_mtime_ns": mtime_ns,
            "file_size": size,
            "extractor_version": extractor_version,
            "status": result.status,
            "n_laps": n_laps,
            "n_samples": len(result.timeseries),
            "date": date_str,
            "extracted_at": extracted_at,
        }
        _update_manifest(dataset_root, entry)


def prune_orphan_sessions(dataset_root: Path) -> dict[str, int]:
    """Drop session/lap rows and timeseries files whose session_id the manifest no longer lists.

    ``upsert_session`` replaces rows by ``session_id`` only. When the grouping
    of split files changes between runs (a restart pair that used to be two
    sessions is now one, or vice versa) the old grouping's session_id is
    never written again, so its rows would linger next to the new ones and
    double-count laps. The manifest is keyed by file and always reflects the
    latest grouping, so anything it does not reference is an orphan.
    """
    live = {e["session_id"] for e in load_manifest(dataset_root).values() if "session_id" in e}
    counts = {"session_rows": 0, "lap_rows": 0, "timeseries_files": 0}
    if not live:
        return counts
    live_arr = pa.array(sorted(live))
    for sub, counter in (
        (sessions_dir(dataset_root), "session_rows"),
        (laps_dir(dataset_root), "lap_rows"),
    ):
        if not sub.exists():
            continue
        for part in sorted(sub.glob("*.parquet")):
            table = pq.read_table(part)
            if len(table) == 0 or "session_id" not in table.schema.names:
                continue
            mask = pc.is_in(table.column("session_id"), live_arr)
            kept = table.filter(mask)
            dropped = len(table) - len(kept)
            if dropped:
                counts[counter] += dropped
                if len(kept) == 0:
                    part.unlink()
                else:
                    _atomic_write_parquet(kept, part)
    ts_root = timeseries_dir(dataset_root)
    if ts_root.exists():
        for f in sorted(ts_root.glob("*/*.parquet")):
            if f.stem not in live:
                f.unlink()
                counts["timeseries_files"] += 1
                if not any(f.parent.iterdir()):
                    f.parent.rmdir()
    if any(counts.values()):
        logger.info("pruned orphans: %s", counts)
    return counts


def manifest_groups(entries: list[dict]) -> dict[tuple[str, ...], dict]:
    """Group manifest entries by session, keyed by the sorted tuple of file paths.

    ``session_id`` changes whenever merge grouping changes, so comparing two
    manifests by id over-reports; the file set is the stable identity.
    """
    by_sid: dict[str, dict] = {}
    for e in entries:
        g = by_sid.setdefault(
            e["session_id"], {"paths": [], "status": e.get("status"), "n_laps": e.get("n_laps")}
        )
        g["paths"].append(e["xrk_path"])
    return {tuple(sorted(g["paths"])): g for g in by_sid.values()}


def compare_manifests(base: list[dict], current: list[dict]) -> dict[str, list]:
    """Describe how ``current`` differs from ``base`` at the file-group level.

    Returns ``{"regrouped": [...], "added": [...], "removed": [...],
    "changed": [...]}`` where regrouped lists file sets whose grouping changed
    (a base group whose files now span other groups, or vice versa), added /
    removed list file sets only present on one side with no overlap, and
    changed lists identical file sets whose status or ``n_laps`` differ.
    """
    gb = manifest_groups(base)
    gc = manifest_groups(current)
    base_files = {p for key in gb for p in key}
    cur_files = {p for key in gc for p in key}
    out: dict[str, list] = {"regrouped": [], "added": [], "removed": [], "changed": []}
    for key, g in gc.items():
        if key in gb:
            b = gb[key]
            if b["status"] != g["status"] or b["n_laps"] != g["n_laps"]:
                out["changed"].append(
                    {
                        "paths": list(key),
                        "status": (b["status"], g["status"]),
                        "n_laps": (b["n_laps"], g["n_laps"]),
                    }
                )
        elif any(p in base_files for p in key):
            out["regrouped"].append(
                {"paths": list(key), "status": g["status"], "n_laps": g["n_laps"]}
            )
        else:
            out["added"].append({"paths": list(key), "status": g["status"], "n_laps": g["n_laps"]})
    for key, g in gb.items():
        if key in gc:
            continue
        if not any(p in cur_files for p in key):
            out["removed"].append(
                {"paths": list(key), "status": g["status"], "n_laps": g["n_laps"]}
            )
    return out
