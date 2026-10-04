---
name: tire-model-update
description: Refresh the tire dataset from the AIM RaceStudio3 folder (new sessions, notes, weather), reconcile misidentified circuit variants, retrain the cold tire pressure model, validate, regenerate the parity fixture, and commit.
---

# Tire Model Update

End-to-end refresh of `data/tire_dataset/` and the fitted warmup model, ending in a
single Sapling commit. Every step is a `just` recipe or a committed script — do not
write ad-hoc data-processing code. Use `sl`, never `git`.

```
/tire-model-update [--aim-root PATH]
```

Default AIM root is `/mnt/c/AIM_SPORT/RaceStudio3/user/data` (one `YYYY-MM-DD/`
folder per track day). Run notes live in the OneDrive `Car Running Notes` folder
and are picked up automatically by `just tire-notes`.

## 0. Preflight

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && sl status && sl log -l 3 --template '{node|short} {desc|firstline}\n'
```

- The working copy must be clean apart from pending changes under
  `data/tire_dataset/` (a deliberately staged schema bump or dropped weather
  cache is fine and gets folded into this refresh's commit). Anything pending
  under `src/`, `tests/` or elsewhere: stop and report.
- Note the current `MANIFEST.jsonl` line count and the last `tire dataset:` commit
  date — the commit message needs the date range of what you add.

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && wc -l data/tire_dataset/MANIFEST.jsonl && ls /mnt/c/AIM_SPORT/RaceStudio3/user/data | tail -15
```

## 1. Extract new sessions

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && LOG="${TMPDIR:-/tmp}/tire-etl.log" && just tire-etl > "$LOG" 2>&1; echo "exit=$?"; grep -E "^extract:|re-split|layout split|wall-clock|pruned" "$LOG" | tail -40
```

(Use your session scratch directory for `LOG` if `/tmp` is not writable. A full
re-extract takes 15–25 minutes: either give the command a 30-minute timeout, or
run it in the background and read the log when the `extract:` line appears.)

Idempotent: only files that are new, changed, or extracted with an older
`EXTRACTOR_VERSION` are processed. After a loader/merge/registry code fix
*without* a version bump, nothing would be re-processed: use
`just tire-etl --force` (everything) or `--force --since YYYY-MM-DD`. After an extractor bump **every** session is
re-extracted (~360 files, 15–25 minutes): run the command with a 30-minute
timeout, or in the background and poll the `extracted=` counter in its log.
Output ends with
`extract: scanned=N skipped=N extracted=N errors=N pruned_sessions=N`. The
exit code is 1 whenever *any* file errored (never hide it behind a pipe), and
that is normal: every file with no full lap is an error, and after a full
re-extract that is ~70 of ~390 files. List them and confirm each has
`n_laps = 0` — the message lives in the sessions table, not the manifest:

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-query "SELECT date, car, track, n_laps, error_msg FROM sessions WHERE status = 'error' AND date >= '<first new date>' ORDER BY date"
```

After an extractor bump run that query without the date filter (expect ~25
error sessions, all with `n_laps = 0`). An error with `n_laps > 0`, or a
message other than `no usable reference channel or no laps` / `tpms channels
missing` (that one is `partial`, not an error), is a real failure: stop and
report it.

**Lap semantics:** `n_laps` counts *full* laps only. libxrk (≥ 0.13) types the
first and last lap of a recording as out/in and the loader drops them, so a
one-lap file (an out-lap only) is a benign `no laps` error, a single-file
session has one lap fewer than the dash shows, a merged restart pair loses
one per file, and a run recorded as many one-lap fragments keeps nothing.
The model never used those laps (`tire_usable` excludes out/in laps), but
`n_laps` in commit messages before 2026-10 predates this typing. The
`pruned_sessions=` counter at the end of the extract output is the number of
stale rows removed because merge grouping changed; non-zero is normal after
an extractor bump and means nothing by itself.

**What the extractor now does for you:** the track is resolved from the GPS
trace, not the filename. The AIM dash can be set to the wrong circuit *variant*
("Suzuka West" while running the full circuit, "Tsukuba_Bike" in a car session,
"Motegi East" on the full course) or to no track at all ("Generic testing",
"Race"). `tire_etl/layouts.py` holds the canonical circuit registry (per-venue
start/finish gates and the layouts that cross them); the extractor picks the
layout actually driven, coerces `track_canonical` to it, and when the logger's
lap beacon sat on the wrong line it re-splits laps at the correct start/finish.
Every session row records how it was resolved (`track_resolution`, `lap_source`,
`layout_id`, `beacon_gate`, `n_laps_logger`).

## 2. Audit the reconciliation

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-track-audit --since <first new date>
```

After an extractor bump the run **without** `--since` is mandatory —
historical sessions may have been reconciled for the first time (expect roughly
40–45 rows: ~13 re-split, ~13 variant overrides, ~8 GPS-supplied tracks; the
exact counts drift by a few as fragments gain or lose their single lap). Read every
row. Expected and fine:

| `track_resolution` | `lap_source` | Meaning |
|---|---|---|
| `gps_override` | `gps_resplit` | Wrong dash variant; laps rebuilt at the real S/F. Check `n_laps` ≈ `n_laps_logger` (−1 per file is normal: the offset beacon loses a partial lap at each end; `n_laps_logger` sums all files of a merged session). `n_laps = 0` with `n_laps_logger ≥ 1` is a fragment whose only lap was split on the wrong line and was dropped on purpose. |
| `gps_override` | `logger` | Wrong variant *label* but the beacon already sat on the real S/F (e.g. "Motegi East" selected, main-line beacon). Nothing to do. |
| `gps_override` or `declared` | `logger_unverified` with laps | Beacon not matched to any gate (no GPS fix at the lap starts) and no re-split possible; logger laps kept. Sanity-check lap times against the venue's normal. |
| `declared` | `gps_resplit` | Variant was right but the beacon was not on any gate (GPS gaps / odd beacon placement); laps rebuilt. Check lap times. |
| `gps` | `logger` | Filename/Venue had no track; GPS supplied it. |
| `declared` | `logger_unverified`, `n_laps = 0` | A no-lap file (out-lap only), possibly with a variant layout like `suzuka_west` / `motegi_east` from a single gate crossing. Harmless. |
| any | any, `track_canonical` null, `n_laps = 0` | Road drive or an unregistered venue with no laps. Harmless. |

Exit codes: 0 when every lapped session has a venue; 1 when a lapped session
has `track_canonical` null (see "Needs action").

Needs action:

- **`track_canonical` is null with `n_laps > 0`** (exit code 1): the car ran at a
  venue not in `layouts.VENUES`. Measure the start/finish point (median GPS
  position at the logger's lap starts — see the module docstring), add a `Venue`
  with gates + layouts to `src/motorsports_data_notebook/tire_etl/layouts.py`,
  add the name to `tire_etl/tracks.py` (weather coordinates + aliases), add a
  registry test in `tests/tire_etl/test_layouts.py`, then re-run
  `just tire-etl --force --since <date>`. Road drives (no laps) are fine to leave.
- **A re-split session whose lap times look wrong** (not ≈ the venue's normal lap
  time for that car): inspect with
  `just tire-query "SELECT lap_num, lap_time_s, is_outlap, is_inlap FROM laps WHERE session_id='<id>'"`.
  If the gate geometry is off, fix it in `layouts.py` and re-extract that date
  with `--force`.
- **A genuine new layout** (e.g. a real Suzuka West day): it will be classified
  as its own `layout_id` but pooled into the venue's `track_canonical`, matching
  the Fuji GP/Short and Motegi/East convention. Mention it in the summary; do not
  change the pooling in this skill.

Also check merges. Files merge only when they share filename date/driver/car/
track/session-type, have consecutive run numbers, start within 5 minutes of
the previous file's end of recording, and resolve to the same GPS layout. A
group that newly merged or split versus the previous commit shows up as a
changed `session_id` in the manifest diff (step 5); spot-check one of each by
confirming the files' lap times look like one continuous run.

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-query "SELECT date, car, track_canonical, layout_id, lap_source, n_laps, status, len(xrk_paths) n_files FROM sessions WHERE date >= '<first new date>' ORDER BY date, session_id"
```

## 3. Notes and weather

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-notes 2>&1 | tail -5 && just tire-weather 2>&1 | tail -3
```

`tire-notes` parses only new/changed `.txt` notes via `claude -p` (cached JSON is
committed) and re-matches all notes to sessions. `tire-weather` fetches
Open-Meteo history only for uncached (track, date) pairs — network required; if it
fails, retry once, otherwise report and continue (the model falls back to track
medians for missing weather).

## 4. Sensor audit

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-sensor-audit 2>&1 | tail -30
```

Lists (session, corner) TPMS channels that look stuck. The audit's own
threshold produces ~10 candidates with std 0.25–1.0 °C every run; those are
live sensors. Only add a row to `data/tire_dataset/sensor_blacklist.yaml`
when std < 0.1 °C over ≥ 10 laps while the other three corners vary normally;
otherwise leave the file alone (it has needed no entry so far) and list any
*new* candidate session in the summary.

## 5. Retrain and validate

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-build-warmup-table 2>&1 | grep -v "^DEBUG" | tail -25
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-predict-holdout --n-folds 5 2>&1 | tail -40
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-predict-holdout --n-folds 5 --inputs oracle 2>&1 | tail -40
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-predict-validate 2>&1 | tail -20
```

**Out-laps (dataset schema v3).** The extractor keeps the out-lap
(`lap_type = "out"`, lap 0) and the first valid TPMS reading in it is the
stint's starting temperature. After a refresh, sanity-check the new sessions'
out-laps: `just tire-query "SELECT session_id, lap_num, outlap_from_pit,
moving_s, tpms_temp_fl_start FROM read_parquet('laps/*.parquet') WHERE
is_outlap AND tire_usable ORDER BY session_id DESC LIMIT 20"` —
`outlap_from_pit` should be true for real pit exits (file starts mid-track are
not), and `moving_s` should be a plausible pit-exit-to-line time for the track.

**Rain buckets.** Damp and wet τ/K are fitted per condition with
τ_rain ≤ τ_dry bounded inside the fit, and only when a (car, track) rain
bucket has ≥ 3 sessions; thinner buckets fall back to dry at prediction.
The holdout prints a per-condition table — quote the wet row alongside dry,
and say when a rain bucket newly crosses the 3-session threshold (its
predictions switch from dry-inherited to fitted).

The first holdout scores what the calculator would have told the driver
(bucket ⟨g²⟩ at the session's pace, N × lap time, typed-in start temperature);
the `--inputs oracle` run scores the thermal model with the lap's real g² and
clock. Quote the calculator numbers as the headline and the oracle numbers as
the ceiling; a growing gap between them means the inputs (pace curve,
typical lap time) are drifting, not the physics.

Record for the commit message: per-car hot-temp MAE from the 5-fold CV, the
notes-validation MAE in bar, and any bucket that moved from prior to fitted
(or back). Compare against the previous dataset commit message
(`sl log -r 'last(desc("tire dataset:"))' --template '{desc}\n'`). A jump of more
than ~1 °C MAE on a car with lots of data is a red flag — look at which new
sessions dominate the new laps before committing. When the extractor version
changed, also diff the manifest (`sl cat -r . data/tire_dataset/MANIFEST.jsonl`
vs the working copy). `session_id` changes whenever merge grouping changes, so
compare by **file group** with the committed revision:

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-manifest-diff            # vs the current commit
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-manifest-diff --base 'last(desc("tire dataset:"))'
```

It prints added / removed / regrouped file groups and status / `n_laps`
changes inside identical groups. Explain every change for cars that got no new data — a
systematic shift there is an extractor/loader change, not new data, and belongs
in the commit message. Expect single-file groups to lose one lap and multi-file
groups one per file after the lap-typing harmonization; a group of one-lap
fragments can lose everything (four such 2025 groups exist).

## 6. Keep the calculator ports in sync

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && uv run scripts/regen_tire_predict_fixture.py
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-web-test 2>&1 | tail -5
cd /home/m3rlin45/code/motorsports_data_notebook && uv run pytest tests/tire_etl tests/tire_model -q --no-cov 2>&1 | tail -3
cd /home/m3rlin45/code/motorsports_data_notebook && just lint && just typecheck
```

`just lint` (black) and `just typecheck` (mypy) must pass; if you edited anything
under `src/` in step 2 or step 5, run `just format` first.

The fixture pins the C# and JS predictors to the Python model; it must be
regenerated every time `tire_model.json` changes. Parity tests compare against
the fixture and must pass unchanged. A few web/.NET tests assert *model facts*
(which compound runs cooler, whether a car's wet bucket is fitted or falls
back); when one of those fails, verify the new fact in
`data/tire_dataset/tire_model.json`, relax the assertion only to the invariant
that still holds, and call the behavior change out explicitly in the commit
message — it is a model change the reviewer must see, not a test bug.
If `dotnet` is available, also run
`cd /home/m3rlin45/code/motorsports_data_notebook/tire_pressure_calculator && dotnet test Tests 2>&1 | tail -5`;
if it is not installed or takes more than a few minutes, skip it and say so.

## 7. Review the diff and commit

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && sl status && sl diff data/tire_dataset/MANIFEST.jsonl | grep '^[+-]{' | head -40
```

Expected changes: new `MANIFEST.jsonl` lines, new/updated monthly `sessions/` and
`laps/` partitions, new `timeseries/YYYY-MM/<id>.parquet` files, weather parquet
for touched (track, year), `tire_model.json`, `warmup_table.parquet`, the parity
fixture, and possibly `notes_extracted/*.json`. Anything under `src/` changing
means you edited code in step 2; calculator test edits come from step 6 — keep
both in the same commit and describe them.

Count what you are adding with a query, not by hand (sessions, laps and
files per car and track — these are the commit message's first line):

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && just tire-query "SELECT car, track_canonical, count(*) sessions, sum(n_laps) laps, sum(len(xrk_paths)) files FROM sessions WHERE date >= '<first new date>' GROUP BY ALL ORDER BY 1, 2"
```

Calculator test files relaxed in step 6 belong in this commit too (they are
data-dependent assertions, not code); say so in the message.

New timeseries files are untracked and dropped weather caches show as missing
(`!`): `sl addremove data/tire_dataset` records both. Then commit on the current
draft (no version bump — versions are bumped only at release time):

```bash
cd /home/m3rlin45/code/motorsports_data_notebook && sl addremove data/tire_dataset && sl commit -m "$(cat <<'EOF'
tire dataset: extend through YYYY-MM-DD

<N> new sessions (<car> <track> ...), <merged-file notes>, <notes/weather notes>.
<Variant reconciliation: which sessions were coerced / re-split and why.>
<Model effect: bucket lap counts, tau/K movements, 5-fold CV MAE per car, notes
validation MAE.> Parity fixture regenerated.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

Do not push or open a PR unless asked. Finish with a summary that stands alone:
sessions added (dates, cars, tracks), reconciliation rows and what they meant,
model metrics before/after, anything skipped or unresolved.
