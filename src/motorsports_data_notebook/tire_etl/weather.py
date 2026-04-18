"""Open-Meteo Historical Weather Archive client with per-(track, year) cache.

Free API, no key required:
  https://archive-api.open-meteo.com/v1/archive
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .paths import default_dataset_root, weather_dir
from .tracks import get_track, known_canonicals

logger = logging.getLogger(__name__)

_API_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]


@dataclass(frozen=True)
class WeatherRange:
    track_canonical: str
    start: _date
    end: _date


def _fetch_open_meteo(lat: float, lon: float, start: _date, end: _date) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(_HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = httpx.get(_API_URL, params=params, timeout=60.0)
    resp.raise_for_status()
    payload = resp.json()
    assert isinstance(payload, dict)
    return payload


def _parse_response_to_table(payload: dict) -> pa.Table:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    rows: dict[str, list] = {"ts_utc": times}
    for var in _HOURLY_VARS:
        rows[var] = hourly.get(var, [None] * len(times))
    return pa.table(
        {
            "ts_utc": pa.array(rows["ts_utc"], type=pa.string()),
            "temperature_2m": pa.array(rows["temperature_2m"], type=pa.float32()),
            "relative_humidity_2m": pa.array(rows["relative_humidity_2m"], type=pa.float32()),
            "surface_pressure": pa.array(rows["surface_pressure"], type=pa.float32()),
            "wind_speed_10m": pa.array(rows["wind_speed_10m"], type=pa.float32()),
            "wind_direction_10m": pa.array(rows["wind_direction_10m"], type=pa.float32()),
            "precipitation": pa.array(rows["precipitation"], type=pa.float32()),
            "cloud_cover": pa.array(rows["cloud_cover"], type=pa.float32()),
        }
    )


def _cache_path(dataset_root: Path, track_canonical: str, year: int) -> Path:
    return weather_dir(dataset_root) / track_canonical / f"{year}.parquet"


def _load_cache(path: Path) -> pa.Table | None:
    if not path.exists():
        return None
    return pq.read_table(path)


def _dates_already_covered(table: pa.Table | None) -> set[str]:
    if table is None or len(table) == 0:
        return set()
    ts = table.column("ts_utc").to_pylist()
    return {t[:10] for t in ts}  # "2026-04-18T00:00"[:10] -> "2026-04-18"


def _merge_and_write(path: Path, existing: pa.Table | None, new: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is None or len(existing) == 0:
        combined = new
    else:
        existing_ts = set(existing.column("ts_utc").to_pylist())
        new_ts_list = new.column("ts_utc").to_pylist()
        keep_mask = [t not in existing_ts for t in new_ts_list]
        filtered_new = new.filter(pa.array(keep_mask))
        combined = pa.concat_tables([existing, filtered_new], promote_options="default")
    indices = pc.sort_indices(combined, sort_keys=[("ts_utc", "ascending")])
    combined = combined.take(indices)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp, compression="zstd", compression_level=3)
    tmp.replace(path)


def fetch_for_track(
    track_canonical: str,
    dates: list[_date],
    *,
    dataset_root: Path | None = None,
) -> None:
    """Fetch + cache weather for the union of ``dates`` for one track."""
    if dataset_root is None:
        dataset_root = default_dataset_root()
    if not dates:
        return
    ti = get_track(track_canonical)
    if ti is None:
        logger.warning("unknown track %s — skipping weather fetch", track_canonical)
        return

    # Group dates by year so we write one partition at a time.
    by_year: dict[int, list[_date]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    for year, ds in by_year.items():
        path = _cache_path(dataset_root, track_canonical, year)
        existing = _load_cache(path)
        covered = _dates_already_covered(existing)
        missing = sorted({d for d in ds if d.isoformat() not in covered})
        if not missing:
            continue
        # Fetch a single contiguous range spanning missing dates; the API is
        # cheap and this minimizes request count.
        try:
            payload = _fetch_open_meteo(ti.lat, ti.lon, missing[0], missing[-1])
        except httpx.HTTPError as e:
            logger.warning(
                "Open-Meteo fetch failed for %s %s..%s: %s",
                track_canonical,
                missing[0],
                missing[-1],
                e,
            )
            continue
        new_tbl = _parse_response_to_table(payload)
        _merge_and_write(path, existing, new_tbl)
        logger.info(
            "weather: %s %s..%s -> %d rows", track_canonical, missing[0], missing[-1], len(new_tbl)
        )


def run_enrich_weather(
    sessions: pa.Table | None = None,
    *,
    dataset_root: Path | None = None,
) -> dict[str, int]:
    """Fetch weather for every (track_canonical, date) in ``sessions``.

    If ``sessions`` is None, reads sessions from the dataset parquet partitions.
    """
    from .paths import sessions_dir

    if dataset_root is None:
        dataset_root = default_dataset_root()
    if sessions is None:
        sess_root = sessions_dir(dataset_root)
        if not sess_root.exists():
            return {"tracks": 0, "dates": 0}
        tables = []
        for p in sorted(sess_root.glob("*.parquet")):
            tables.append(pq.read_table(p))
        if not tables:
            return {"tracks": 0, "dates": 0}
        sessions = pa.concat_tables(tables, promote_options="default")

    n_tracks = 0
    n_dates = 0
    seen: dict[str, set[_date]] = {}
    for track_can, d in zip(
        sessions.column("track_canonical").to_pylist(),
        sessions.column("date").to_pylist(),
    ):
        if track_can is None or d is None:
            continue
        if track_can not in known_canonicals():
            continue
        seen.setdefault(track_can, set()).add(d)
    for track_canonical, dates in seen.items():
        fetch_for_track(track_canonical, sorted(dates), dataset_root=dataset_root)
        n_tracks += 1
        n_dates += len(dates)
    return {"tracks": n_tracks, "dates": n_dates}
