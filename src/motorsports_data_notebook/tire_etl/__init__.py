"""Tire ETL pipeline.

Batch-processes AIM telemetry sessions plus hand-written run notes plus
historical weather into a queryable Parquet dataset under ``data/tire_dataset/``.

Public API
----------
- :func:`run_extract` — walk the AIM data tree and extract new sessions
- :func:`run_enrich_notes` — parse run-note .txt files via ``claude -p``
- :func:`run_enrich_weather` — fetch Open-Meteo historical weather
- :data:`EXTRACTOR_VERSION` — bumped to force re-extraction of all sessions
"""

from __future__ import annotations

# 0.8.0: wall-clock-aware split-session merging — filename groups are split
# when the gap between one file's end and the next file's start exceeds
# MERGE_MAX_GAP_S, and genuinely merged files get their lap times shifted
# onto the first file's clock (fixes overlapping timelines, single-stint
# collapse, and warmup-time corruption in merged sessions).
# 0.9.0: GPS layout reconciliation — the track is resolved from the GPS trace
# (gate crossings per venue layout) instead of the filename token alone, and
# laps are re-split at the correct start/finish when the logger's beacon sat
# on a different variant's line (full Suzuka logged as "Suzuka West", Motegi
# full logged as "Motegi East"). Sessions whose filename carried no track
# ("Generic testing", "Race") now resolve via Venue metadata + GPS. New
# session columns: venue_meta, layout_id, track_declared_canonical,
# track_resolution, lap_source, beacon_gate, n_laps_logger.
EXTRACTOR_VERSION = "0.9.0"

from .extract import extract_session, run_extract  # noqa: E402
from .notes_parser import run_enrich_notes  # noqa: E402
from .weather import run_enrich_weather  # noqa: E402

__all__ = [
    "EXTRACTOR_VERSION",
    "extract_session",
    "run_extract",
    "run_enrich_notes",
    "run_enrich_weather",
]
