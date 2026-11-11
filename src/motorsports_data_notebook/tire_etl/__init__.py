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
# 0.17.0: the IMU checks compare the candidate at the reference's
# bandwidth (0.5 s window) and the gyro must read within 0.9-1.1× the
# position-derived yaw rate (the accelerometer keeps 0.85-1.15×): healthy
# longitudinal axes no longer fail on their own vibration, and the 0.87×
# gyros (Inferno 86 since 2025-06-09, SFJ) fall back to position yaw.
# Verdicts and the laps' dynamics columns change; file layout does not
# (schema stays 9).
# 0.18.0: ``imu_swapped`` verdict — an IMU whose two horizontal channels
# are exchanged in the file (the KK-SII dash since 2026-03-18) is kept, the
# loader reading the columns exchanged; those sessions used the 25 Hz GPS
# channels before. Session rows gain kin_imu_swapped_* columns. Schema 9.
# 0.20.0: the per-timebase ``rate_hz`` in the file metadata is samples
# over the time they cover, counting only intervals under 1 s, with the
# dropout time and count reported as ``gap_s`` / ``n_gaps`` — not
# 1 / median interval, which called the KK-SII's accelerometer (pairs 2 ms
# apart on 50 Hz frames, 100 Hz) 500 Hz, nor a span mean, which a stream
# that stops mid-session drags down. Metadata only.
# 0.21.0: a per-window sensor source next to the session verdict. The session
# is cut into 20 s windows and each is ``imu`` (the session's IMU is sound and
# has samples there), ``gps`` (it does not; the GPS fallback — the logger's
# GPS channels when they validated, else the position-derived series — has no
# spike over 3 g) or ``discard`` (neither). An IMU stream that stops part-way
# through a run is used while it lasts (it used to leave the rest of the run
# without dynamics when it stopped after the reference lap, and to put the
# whole session on GPS when it stopped before it: the IMU is now scored on
# the fastest full lap it covers); a frozen IMU window goes to GPS; a GPS
# spike with no IMU behind it is discarded instead of entering the heat
# input. More than three windows where a live IMU does not follow the GPS
# reference and the IMU is not trusted for the session. The windows are in
# the file metadata (``kinematics.windows``); session rows gain
# ``kin_gps_fallback``, ``kin_imu_ref_lap`` and ``kin_win_{imu,gps,discard,
# mismatch}``. A lap with a discarded window has NaN ``heat_proxy*`` /
# ``lat_g_peak`` / ``long_g_peak_brake``. Raw channels unchanged. Schema 9.
# 0.22.0: the discard mark is a column. Every timeseries file gains the
# sparse boolean key column ``discard``: true on each row, of any timebase,
# whose session time falls in a discarded window, null elsewhere (all null
# in a session without one). The samples of those rows are stored as
# before; the column makes the mark filterable in ``just tire-query``
# (``WHERE discard``), where the window string in the file metadata was
# not. Nothing else changes. Schema 9 (an added column).
EXTRACTOR_VERSION = "0.22.0"

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
