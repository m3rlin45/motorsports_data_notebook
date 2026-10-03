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
  `venue_meta`, `track_declared_canonical`. Review with `just tire-track-audit`.
- `laps/YYYY-MM.parquet` — one row per *full* lap (summary stats derived from
  timeseries). libxrk ≥ 0.13 types the first/last lap of a recording as out/in
  and the loader drops them; before schema v2 most sessions were extracted with
  an older libxrk and still carried their in-lap.
- `timeseries/YYYY-MM/{session_id}.parquet` — per-sample telemetry for one session.
  This is the source of truth; the per-lap aggregates are rebuildable from it.
- `notes_extracted/*.json` — structured JSON extracted from run-note `.txt` files
  via `claude -p` (Opus 4.7). The committed JSON is the cache; claude only re-runs
  on changed notes.
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
