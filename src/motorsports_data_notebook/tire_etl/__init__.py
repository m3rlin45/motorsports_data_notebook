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
EXTRACTOR_VERSION = "0.8.0"

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
