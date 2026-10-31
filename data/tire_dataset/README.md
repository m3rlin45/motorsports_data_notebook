# Tire dataset

Derived telemetry + enrichment artifacts for tire warmup/pressure modeling.
All files here are **produced by the `tire_etl` pipeline** and checked in so
deltas show up in PR diffs.

## Layout

- `MANIFEST.jsonl` — line-per-session ledger. Canonical diff summary: git diff
  this file to see which sessions were added/changed in a commit.
- `schema_version.txt` — single integer bumped when the parquet schema changes.
- `sessions/YYYY-MM.parquet` — one row per extracted session (metadata + flags).
  `track_canonical` is resolved from the GPS trace against the circuit registry in
  `tire_etl/layouts.py`, not from the filename: the dash's track *variant* can be
  wrong ("Suzuka West" while lapping the full circuit, "Motegi East" on the full
  course) or absent ("Generic testing", "Race"). The reconciliation is recorded in
  `track_resolution` (`declared` / `gps` / `gps_override`), `layout_id`,
  `lap_source` (`logger` / `gps_resplit` when laps were rebuilt at the correct
  start/finish / `logger_unverified`), `beacon_gate`, `n_laps_logger`,
  `venue_meta`, `track_declared_canonical`, and (schema v5) `lap_length_m`, the
  median GPS path length per lap at the resolved start/finish: layouts that
  share every gate (Fuji GP vs Short) are separated by it, not by the dash
  variant name. Review with `just tire-track-audit`.
  (`n_tire_usable_laps` was dropped in schema v4; usability is a training-time
  decision.) `comment` (schema v6, extractor 0.13.0) is the RaceStudio
  "Long Comment" the driver typed for the file(s) — tire fitted, setup, race
  result; the per-day notes parse below reads it. Schema v7 (extractor
  0.14.0) adds `timebases` / `n_timebases` (the raw timebases the file
  carries, `name:rate:n`) and the **sensor verdict**: `accel_source`
  (`imu` / `imu_swapped` — the IMU's two horizontal channels exchanged in
  the file, read back exchanged by the loader / `gps_channel` /
  `gps_position`) and `gyro_source` (`imu` /
  `gps_position`) with every regression number behind them
  (`kin_{imu,gps}_{lat,long,yaw}_{slope,corr,offset,coverage,std,cross,ok}`,
  `kin_ref_lap`, `kin_note`): the IMU is checked against GPS position + speed
  on the fastest full lap and the loaders use the chosen series
  (`tire_etl/kinematics.py`; review with `just tire-kinematics-audit`).
- `laps/YYYY-MM.parquet` — one row per lap (summary stats derived from
  timeseries), including the **out-lap** (`lap_type = "out"`, `lap_num = 0`:
  pit exit, or recording start, to the first start/finish crossing) and any
  in-lap (`lap_type = "in"`). Schema v4 (extractor 0.11.0): the table is
  **raw** — no TPMS masking, no usability flags. `lap_type`, `outlap_from_pit`
  (the out-lap began below pit-lane speed, i.e. from standstill — mid-track
  file starts are not pit exits), `moving_s` (rolling time above 5 km/h; an
  out-lap's `lap_time_s` and `on_track_s` include the grid / pit-lane wait),
  `speed_kmh_first`, and per-corner TPMS / surface summaries straight from the
  timeseries (`tpms_*_start` is the first finite sample of the lap, stale or
  not; `tpms_*_{corner}_first_valid_s` its lap-relative time). A new
  `stint_id` starts at every from-pit out-lap, not only after a ≥ 10 min gap.
  Cleaning happens when the model loads the data
  (`src/motorsports_data_notebook/tire_model/cleaning.py`): the stale-TPMS
  prefix is masked, the TPMS summaries are rebuilt from the masked streams
  (so an out-lap's start is the pit-exit reading the driver actually set),
  and `tire_usable` / `exclude_reason` are assigned. From schema v7 the
  dynamics columns (`lat_g_peak`, `long_g_peak_brake`, `heat_proxy*`) are
  computed from the session's **validated** accelerations (the sensor the
  verdict chose, at its native rate), and `speed_kmh_max` / the per-corner
  summaries from each channel's own samples. `just tire-clean-laps` writes that cleaned
  table to a parquet file for queries. Before v4 (schemas 1–3) the extractor
  masked and filtered, so the committed columns depended on the cleaning
  rules and any rule change meant a full re-extract.
- `timeseries/YYYY-MM/{session_id}.parquet` — per-sample telemetry for one session,
  the source of truth; the per-lap aggregates are rebuildable from it. The
  `session_id` (schema v9, extractor 0.16.0) is readable and sorts
  chronologically: `2026-10-24_122053_KK-F_tsukuba_a0370-0372` — the date,
  the logger's wall-clock start, the car, the canonical track and the run
  number(s) of the file(s) (`a0370-0372` is a three-file merged session). It
  keys every table and names the file; it changes when the session's
  identity changes (car tag, file grouping), not when a file is rewritten —
  the manifest's mtime / size check decides re-extraction, which replaces
  the rows under the same id. Schema v7
  stores every selected channel **verbatim at its native rate**: one row per raw
  sample, one row group per raw *timebase* (channels the logger sampled on one
  timecode array — `gps` 25 Hz with `gps_lat`/`gps_lon` (float64),
  `gps_alt_m`, `speed_ms` and the logger's `gps_lat_g`/`gps_long_g`/
  `gps_yaw_rate_dps`; `accel` 50 Hz — the KK-SII dash since 2026-03 logs it as pairs 2 ms
  apart on 50 Hz frames, 100 Hz in bursts — `lat_g`/`long_g`/`vert_g`; `gyro` `yaw_rate_dps`/`pitch_rate_dps`/`roll_rate_dps`; `tpms/<corner>`
  1.25 Hz; `surface/<corner>_ch1-4` and `_ch5-8` 10 Hz; `controls/...`), named
  by the `timebase` and `family` columns; every channel column is null outside
  its timebase, and `t_session_s`, `lap_num`, `stint_id`, `sample_idx`,
  `t_lap_s` are per row. Each channel is stored at the narrowest float
  width that holds its samples exactly (schema v8, extractor 0.15.0):
  float16 where the AIM file logged float16 — the IMU, the gyro, the CAN
  brake / steer / throttle (libxrk widens them to float32, libxrk #115) —
  float32 for the analog inputs, the IR zones and the TPMS pressures,
  float64 for latitude / longitude; the parquet floats are byte-stream-split
  under zstd 9 (float16 PLAIN, DuckDB's limit) and read back as `FLOAT` in
  `just tire-query`. Nothing is resampled or aligned in the extractor —
  merging onto one clock happens at query time (`dataset.load_group`,
  `dataset.channel_series`, `kinematics.load_kinematics`, which puts speed and
  the verdict's lateral/longitudinal/yaw series on the chosen sensor's clock).
  The file's schema metadata carries the per-timebase summary (rows, rate
  over the covered time with dropouts over 1 s reported as `gap_s` /
  `n_gaps`, channels) and the sensor verdict. `just tire-query` exposes one DuckDB view per family (`gps`,
  `accel`, `gyro`, `tpms`, `surface`, `controls`) next to `timeseries`.
  The loaders read this schema only (the file's `tire_etl.schema` metadata
  must say 9); a file from an earlier extractor is refused, not reinterpreted.
- `session_notes/YYYY-MM-DD.json` — one `claude -p` (Sonnet) call per track day
  over the day's run-notes `.txt` file(s) from the OneDrive notes folder and
  every logged session's `comment`; the model matches notes blocks to sessions
  and returns per-session cold pressures, tire compound, track condition, setup
  changes and incidents. Cached by prompt/model/input hash (`just tire-notes`
  re-parses only days whose inputs changed). `notes_matches.parquet` is rebuilt
  from these caches every run: one row per session the inputs said anything
  about (`source_file` / `note_session_index` / `match_confidence` when a notes
  block matched; the session's `comment` is carried along).
- `tire_compounds.yaml` — compound exceptions for sessions the notes / comment
  parse cannot label (plus the wheel-set map and the KK-SII weather seeds).
- `hot_pressure_targets.yaml` — team target hot pressure per (car, compound),
  bar; the calculators prefill each corner's target from it, copied into
  `tire_model.json` (`target_hot_pressure_by_car_compound`) by the model build.
- `weather_hourly/{track}/{YYYY}.parquet` — Open-Meteo Historical Weather Archive
  cache (temp, humidity, wind, precip, cloud cover) keyed by track+year.

## Updating (delta workflow)

The `/tire-model-update` skill (`.claude/skills/tire-model-update/SKILL.md`) walks
the full refresh — extract, track audit, notes, weather, retrain, validate, parity
fixture, commit. The core is:

```bash
just tire-refresh              # runs extract + notes + weather
just tire-track-audit          # sessions whose track/laps were reconciled from GPS
just tire-build-warmup-table   # retrain
uv run scripts/regen_tire_predict_fixture.py
sl status                      # inspect diff
sl commit -m "tire dataset: extend through YYYY-MM-DD"
```

Idempotent: running `tire-refresh` with no new input files produces zero diff
(except `extracted_at` in `MANIFEST.jsonl` for re-extracted sessions).
