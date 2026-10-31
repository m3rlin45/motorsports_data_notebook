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
# 0.11.0: the extractor writes raw data only. The stale-TPMS prefix mask
# and the per-lap usability filters (tire_usable / exclude_reason,
# n_tire_usable_laps) moved to the model package (tire_model/cleaning.py),
# so a cleaning rule can change without re-extracting. Dataset schema 4.
# 0.12.0: layouts that cross the same gates (Fuji GP vs Short) are told
# apart by the measured GPS lap length against the registry's nominal
# lap_length_m instead of the dash variant name; every "Fuji GP Sh" session
# so far was a GP-course session (#209). New session column: lap_length_m
# (median GPS path length per lap at the resolved S/F). Dataset schema 5.
# 0.13.0: the session row carries the file's RaceStudio "Long Comment"
# (``comment``; the distinct comments of a merged session joined in file
# order). The driver records the tire fitted there, and the model's
# compound labelling reads it. Dataset schema 6.
# 0.14.0: native-rate storage. Every selected channel is stored verbatim on
# its own raw timecode array (one row group per timebase in the session's
# parquet, ``timebase`` / ``family`` columns, channel columns null outside
# their timebase) — nothing is resampled or aligned in the extractor; the
# loaders in dataset.py / kinematics.py merge at query time. New channels:
# IMU vertical g and the gyro rates, the logger's GPS-derived g / yaw rate
# and altitude. The IMU is validated against GPS position + speed and the
# session row records one accelerometer and one gyro source verdict
# (``accel_source`` / ``gyro_source`` with the regression numbers); the
# lap table's dynamics come from the chosen series. Merged sessions' later
# files are now on the merged clock (t_lap_s used to go negative there).
# Dataset schema 7.
# 0.15.0: channels are stored at the narrowest float width that holds them
# exactly — float16 where the AIM file logged float16 (IMU, gyro, CAN
# brake/steer/throttle; libxrk widens them, issue #115), float32 for the
# analog, IR and TPMS pressure channels, float64 for positions — and the
# parquet floats are byte-stream-split without dictionary encoding (a 20 MB
# session file becomes 8.6 MB with identical values). Dataset schema 8.
# 0.16.0: readable, sortable session ids —
# ``2026-10-24_122053_KK-F_tsukuba_a0370-0372`` (date, logger wall-clock
# start, car, canonical track, run numbers) — replace the path+mtime hash as
# the key of every table and the timeseries file name; a rewritten file
# re-extracts under the same id. Dataset schema 9 (file contents unchanged).
EXTRACTOR_VERSION = "0.16.0"

from .extract import extract_session, run_extract  # noqa: E402
from .session_notes import run_enrich_notes  # noqa: E402
from .weather import run_enrich_weather  # noqa: E402

__all__ = [
    "EXTRACTOR_VERSION",
    "extract_session",
    "run_extract",
    "run_enrich_notes",
    "run_enrich_weather",
]
