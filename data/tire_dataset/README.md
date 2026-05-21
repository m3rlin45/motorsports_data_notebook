# Tire dataset

Derived telemetry + enrichment artifacts for tire warmup/pressure modeling.
All files here are **produced by the `tire_etl` pipeline** and checked in so
deltas show up in PR diffs.

## Layout

- `MANIFEST.jsonl` — line-per-session ledger. Canonical diff summary: git diff
  this file to see which sessions were added/changed in a commit.
- `schema_version.txt` — single integer bumped when the parquet schema changes.
- `sessions/YYYY-MM.parquet` — one row per extracted session (metadata + flags).
- `laps/YYYY-MM.parquet` — one row per lap (summary stats derived from timeseries).
- `timeseries/YYYY-MM/{session_id}.parquet` — per-sample telemetry for one session.
  This is the source of truth; the per-lap aggregates are rebuildable from it.
- `notes_extracted/*.json` — structured JSON extracted from run-note `.txt` files
  via `claude -p` (Opus 4.7). The committed JSON is the cache; claude only re-runs
  on changed notes.
- `weather_hourly/{track}/{YYYY}.parquet` — Open-Meteo Historical Weather Archive
  cache (temp, humidity, wind, precip, cloud cover) keyed by track+year.

## Updating (delta workflow)

```bash
just tire-refresh              # runs extract + notes + weather
sl status                      # inspect diff
sl commit -m "tire dataset: extend through YYYY-MM"
sl pr submit
```

Idempotent: running `tire-refresh` with no new input files produces zero diff.
