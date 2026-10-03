"""Parquet I/O, partitioning, and MANIFEST ledger for the tire dataset.

Design principles
-----------------
- **Deterministic output.** Partition files are always rewritten sorted by
  ``(session_id, lap_num, sample_idx)``. No row ordering surprises in diffs.
- **Atomic writes.** Write to ``{target}.tmp``, fsync, rename.
- **MANIFEST.jsonl is canonical.** Each line is ``{session_id, xrk_path,
  xrk_mtime_ns, file_size, extractor_version, status, n_laps, n_samples,
  extracted_at}``, sorted by ``(date, session_id)``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .paths import (
    laps_dir,
    manifest_path,
    sessions_dir,
    timeseries_dir,
)

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


def _anti_join_and_concat(
    existing: pa.Table | None,
    new_rows: pa.Table,
    *,
    key: str = "session_id",
) -> pa.Table:
    """Return ``existing`` rows with matching session_id removed, then concat ``new_rows``."""
    if existing is None or len(existing) == 0:
        return new_rows
    if len(new_rows) == 0:
        return existing
    incoming_ids = set(new_rows.column(key).to_pylist())
    if not incoming_ids:
        return pa.concat_tables([existing, new_rows], promote_options="default")
    # Filter out rows from existing whose key is in the incoming set.
    mask = pc.invert(pc.is_in(existing.column(key), pa.array(list(incoming_ids))))
    keep = existing.filter(mask)
    # Align schemas before concat (missing columns become null), then put the
    # columns in the incoming rows' order so every partition shares one
    # column order — pa.concat_tables without promotion (used by readers)
    # rejects partitions whose schemas differ only in order.
    merged = pa.concat_tables([keep, new_rows], promote_options="default")
    order = [c for c in new_rows.schema.names] + [
        c for c in merged.schema.names if c not in new_rows.schema.names
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

    # Timeseries: one file per session.
    if len(result.timeseries) > 0:
        ts_path = timeseries_dir(dataset_root) / partition / f"{session_id}.parquet"
        _atomic_write_parquet(result.timeseries, ts_path)

    # Sessions partition.
    sess_path = sessions_dir(dataset_root) / f"{partition}.parquet"
    existing_sessions = _read_partition(sess_path)
    merged_sessions = _anti_join_and_concat(existing_sessions, result.session_row)
    merged_sessions = _sort_sessions(merged_sessions)
    _atomic_write_parquet(merged_sessions, sess_path)

    # Laps partition.
    if len(result.laps_rows) > 0:
        laps_path = laps_dir(dataset_root) / f"{partition}.parquet"
        existing_laps = _read_partition(laps_path)
        merged_laps = _anti_join_and_concat(existing_laps, result.laps_rows)
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
