---
name: session-analysis
description: Analyze a motorsports telemetry session (.xrz/.xrk), identify improvement areas, and set KPIs.
---

# Session Analysis Skill

Analyze motorsports telemetry sessions, identify improvement areas, set KPIs, and track progress.

## Usage

```
/session-analysis <file_path> [file_path2 ...]
```

## Workflow

### 1. Generate Report Data

For each provided file path, run:

```bash
uv run python scripts/analyze_session.py "<file_path>" --output .session-analysis/report.json --track-map .session-analysis/track_map.png --comparison-dir .session-analysis/comparisons/ --session-num <N>
```

Where `<N>` is the 1-based session number for the day (default: 1). This populates `metadata.session_id` in the JSON.

Read the JSON output with the Read tool. After reading the JSON, use `metadata.session_id` for naming output files when called from the day-review skill (e.g., `session_2026-01-12_1300_s01_report.md`). For standalone use, the default names are fine.

**No ad-hoc code:** The `analyze_session.py` script is the ONLY command you run via Bash. Do NOT run inline Python scripts, `python -c` one-liners, or any other data processing commands. All telemetry processing, metric computation, and image generation is handled by `analyze_session.py`. After it produces JSON + images, use only the Read tool to read the data and the Write tool to produce markdown reports. Everything you need is in the JSON output.

The `--track-map` flag generates a track map image with labeled corners. The `--comparison-dir` flag generates per-corner input comparison plots and zoomed GPS maps for the top 5 opportunity corners.

### 2. Check for Previous Reports

If previous session report(s) are provided (e.g., as file paths in the prompt), read them for **cross-session comparison**. For standalone use, check if `.session-analysis/report.md` exists from a prior run.

Previous reports contain KPI tables and improvement areas in markdown — use these directly for comparison rather than parsing JSON state.

### 3. Identify Improvement Areas

Rank corners by impact and select the **top 3** for the main report. All remaining corners go to a separate appendix file (see step 7).

**Priority ranking:**

| Priority | Area | Metric | Threshold |
|----------|------|--------|-----------|
| High | Corner exit speed consistency | `opportunity_score` (= `exit_speed_std` × `accel_zone_length`) | Highest scores first |
| High | Braking point consistency | `bp_std` | > 3m |
| High | Braking consistency | `peak_brake_std` / `peak_brake_mean` | > 5% CoV |
| High | Brake release consistency | `brake_release_std` | > 3m |
| High | Corner speed consistency | `min_speed_std` / `min_speed_mean` | > 3% CoV |
| High | Throttle acceptance level | `ta_mean` vs `best_lap.throttle_acceptance_pct` | Gap > 10% — driver is consistently hesitating |
| High | G utilization consistency | `g_utilization_std` | > 5% |
| Medium | Entry speed consistency | `entry_speed_std` | > 3 km/h |
| Medium | Throttle acceptance consistency | `ta_std` | > 5% |
| Medium | Lap time consistency | `std_top_lap_time_s` | > 0.5s |
| Medium | G utilization level | `g_utilization_mean` | < 70% — driver has grip holes in transitions |
| Setup | Suspension balance issues | `pct_friction` > 30%, L/R symmetry diffs > 5% | Flag for engineer |
| Setup | Tire grip anomalies | Large per-corner `std_g` differences | Flag for engineer |
| Setup | Brake balance anomalies | Per-corner `balance_pct` deviation from overall mean | > 3% — flag for engineer |

**Important:** Consistency (low std) and absolute level (high mean) are both required. A driver who is consistently slow needs different coaching than one who is fast but erratic. Always compare `ta_mean` against the best-lap value — a large gap means the driver has proven capability but isn't using it. Track both `ta_mean` and `ta_std` as separate KPIs when relevant.

### 4. Show Best Execution with Root Cause

For each improvement area, use the `best_lap` data in the corner's consistency entry:

**Turn N — Corner Exit Speed**
Your best execution was on **Lap X**: you braked Ym later than average, carried Z km/h more through the apex, and exited W km/h faster.
**Target**: Replicate this on 80%+ of laps.

When braking metrics are available, include them in the best-lap description:

**Turn N — Braking Zone**
Your best execution was on **Lap X**: you braked W bar harder, released Y m earlier, and carried Z km/h more entry speed than average.
**Target**: Replicate this on 80%+ of laps.

Reference `best_lap.vs_mean` for the specific deltas and `best_lap.selection_reason` for context.

**Exit speed root cause analysis**: When the best lap has better exit speed (`vs_mean.exit_speed > 0`), analyze the `vs_mean` deltas to explain WHY it was faster. Check two causal chains:

- **Braking chain**: `entry_speed` → `min_speed` → `exit_speed` (carried more speed through)
- **Throttle chain**: `throttle_point` / `throttle_acceptance_pct` → `exit_speed` (got on power earlier)

Format the explanation as a ranked list of contributing factors:

**Why was Lap X faster at Turn N?**
1. **Got on throttle 8m earlier** — the biggest factor
2. **Carried 2.4 km/h more through the apex**

Only include factors with meaningful deltas (speed > 1 km/h, distance > 2m, TA > 5%). Rank by impact magnitude. If no clear cause can be determined from the data, state that rather than guessing.

**G utilization technique notes**: For corners where `g_utilization_mean < 70%`, add an actionable technique note based on `total_g_min_phase`:

| Phase | Note |
|-------|------|
| entry | "Trail brake deeper — releasing brake before turn-in wastes lateral G" |
| exit | "Get on throttle earlier — pause between turning and accelerating costs grip" |
| mid | "Commit through the apex — lifting mid-corner breaks momentum" |
| braking | "Brake later — you finished braking before the corner and coasted into the turn" |

When `total_g_min_phase = "braking"`, always check `early_braking_coast_m` (on the best lap or the corner-level mean `early_braking_coast_mean`). This measures the distance from where braking G faded to the corner start — the "dead zone" where the driver is neither braking nor turning. Include the distance in the note: "You coasted Xm before the corner — brake that much later to flow directly into turn-in."

Only mention G utilization for corners where it's below 70%. Don't add a standalone G utilization table — fold it into the improvement area notes.

### 5. Set KPIs

Define **at most 3 KPIs**, all tied to the highlighted corners. Pick the metrics that would most directly address the root causes identified in step 4. Not every corner needs a KPI — if two corners share the same underlying issue, one KPI may cover both.

Realistic targets:
- `exit_speed_std`: reduce by 40–50%
- `bp_std`: reduce to < 2m
- `min_speed_std`: reduce by 30–40%
- `ta_mean`: close gap to best-lap TA by 50% (e.g., mean 70%, best 90% → target 80%)
- `ta_std`: reduce to < 3%
- `std_top_lap_time_s`: reduce by 30%
- `peak_brake_std`: reduce CoV to < 3%
- `brake_release_std`: reduce to < 2m
- `entry_speed_std`: reduce by 30–40%

### 6. Generate Report

Output the **main report** (focused, actionable) and an **appendix** (full data for reference).

#### Main report

1. **Track Map** — embed the track map image: `![Track Map](track_map.png)` (using the image generated in step 1)
2. **Quick Reference** — extremely brief, memorizable list of the top 3 technique focuses. Format:
   ```
   ## Quick Reference
   1. **T2**: Trail brake deeper, commit to the apex
   2. **T9**: Get on throttle earlier at exit
   3. **T4**: Full throttle sooner — you proved it on Lap 9
   ```
   Each item should be one line with the corner ID and the specific action. Reference a lap number where the driver proved they can do it. This must be concise enough to remember while driving.
3. **Session Overview** — Include the session date/time (`metadata.log_date`, `metadata.log_time`), driver (`metadata.driver`), vehicle (`metadata.vehicle`), venue (`metadata.venue`), weather conditions (`weather`), lap times, and track info. If `metadata.session_notes` is present, include it as a blockquote — these are engineer/driver notes recorded at the track (setup changes, driver feedback, tire compound, etc.). Use these notes to add context to the analysis (e.g., if notes say "lots of oversteer," correlate that with rear grip data). If the notes contain abbreviations or references you don't understand, ask the user for clarification rather than guessing. Format example:
   ```
   **Date:** 2026-01-12 13:00 | **Driver:** CMD | **Vehicle:** Inferno 86
   **Venue:** Sodegaura | **Weather:** Clear sky, 8.7°C, 37% humidity, wind 4.2 km/h WNW

   > **Session Notes:** Rear 23, still lots of oversteer, lots of pickup
   ```
   Weather comes from the Open-Meteo historical API (fetched automatically using circuit GPS center). Include `weather_description`, `temperature_c`, `relative_humidity_pct`, `wind_speed_kmh`, and `wind_direction_deg` (convert degrees to compass direction). Omit the weather line if `weather` is null.
4. **Top 3 Improvement Areas** — the 3 highest-impact corners only. For each corner, use this format:
   ```
   ### N. Turn X — Short Description (Opportunity: NNNN)

   | Metric | Value | Consistency | Best Lap |
   |--------|-------|-------------|----------|
   | Min Speed | 93.5 km/h | 4.2 km/h | 95.3 km/h (Lap 11) |
   | Exit Speed | 120.8 km/h | 2.7 km/h | 124.9 km/h (Lap 11) |
   | Throttle Acceptance | 78.4% | 17.8% | 98.8% (Lap 11) |
   | Braking Point | 450m | 4.1m | 452.6m (Lap 11) |
   | G Utilization | 40.3% | — | — |

   (Only include rows for metrics that are relevant/available for this corner.)

   **Best Execution: Lap 11**
   ...root cause analysis...

   **Technique:** ...G utilization note if < 70%...

   **Target:** ...

   ![Turn X Inputs](comparisons/comparison_t{id}_inputs.png)
   ![Turn X Map](comparisons/comparison_t{id}_map.png)
   ```
   Include: metrics table, root cause analysis for exit speed gains, G utilization technique notes where applicable (< 70%), comparison images
5. **Tire Conditions** — if `tire_conditions` is available in the report, show per-lap max pressure and temperature for the best lap (and optionally a few other top laps for comparison):
   ```
   ## Tire Conditions

   Best lap (Lap 9):
   | Wheel | Max Pressure | Max Temperature |
   |-------|-------------|-----------------|
   | FL | 2.29 bar | 55 C |
   | FR | 2.20 bar | 50 C |
   | RL | 2.20 bar | 48 C |
   | RR | 2.16 bar | 44 C |
   ```
   Include units from `tire_conditions.pressure_unit` and `tire_conditions.temperature_unit`. Show whichever metrics are available (both pressure and temperature when present). Omit this section if `tire_conditions` is null.

   **Important:** Present tire data as-is without making qualitative judgments (e.g., "low" or "high"). Normal pressure and temperature ranges vary by car, tire compound, and target setup — the analysis tool has no reference values to judge against. Only note objective cross-wheel differences (e.g., "RR pressure 0.10 bar lower than other corners") and trends across laps or sessions. Leave interpretation to the driver/engineer.
6. **KPIs** — at most 3 targets for next session, all tied to the highlighted corners
7. **Skipped Analyses** — any analyses that couldn't run and why (omit if none)

The track map and comparison images are already in `.session-analysis/` from step 1, so the relative image references in the markdown report work without copying.

#### Appendix

Write a separate appendix file alongside the main report (e.g., `report_appendix.md` or `step<NN>_appendix.md` when called from day-review). This contains:

1. **Corner-by-Corner Summary** — table of all corners with key consistency metrics
2. **Remaining Corner Analyses** — same format as the top 3 but for corners ranked 4+
3. **Brake Balance** — filtered summary (low-brake corners are excluded automatically by the report generator)
4. **Setup Notes** — suspension and tire grip findings (if available)

Link to the appendix from the main report: `See [full corner analysis](report_appendix.md) for all corners, brake balance, and setup notes.`

### File Naming

When `metadata.session_id` is available, use it for output file naming to make files self-describing:

- Report: `session_<session_id>_report.md` (e.g., `session_2026-01-12_1300_s01_report.md`)
- Appendix: `session_<session_id>_appendix.md`
- JSON: `session_<session_id>_data.json`

Include the driver name and vehicle in the report's H1 heading:
```
# Session Report — CMD / Inferno 86 — Sodegaura S1
```
Where "CMD" is `metadata.driver`, "Inferno 86" is `metadata.vehicle`, "Sodegaura" is `metadata.venue`, and "S1" is the session number. Omit any field that is null.

When called from the day-review skill with step-based naming, the step prefix takes precedence but the session_id should still appear in the report heading.

### Driver & Vehicle Awareness

Always display driver and vehicle prominently:
- **Report H1 heading**: Include driver, vehicle, venue, and session number (see File Naming above)
- **Session Overview**: Always show `metadata.driver` and `metadata.vehicle` even if null (display "Unknown" as placeholder)

When comparing across sessions, check `metadata.driver` and `metadata.vehicle`:

- **Same driver + same car**: Full comparison is valid — compare all metrics
- **Different driver + same car**: Compare **best-lap values** across drivers as proof of car capability. Any metric where the other driver achieved a better best-lap value is valid evidence that the car can do it:
  - Speeds: `best_lap.exit_speed`, `best_lap.min_speed`, `best_lap.entry_speed`
  - Technique: `best_lap.throttle_acceptance_pct`, `best_lap.braking_point`, `best_lap.brake_release_point`, `best_lap.g_utilization_pct`, `best_lap.early_braking_coast_m`
  - Use phrasing like: "Sobu San achieved 95% throttle acceptance at T5 in S2 — this proves the car allows aggressive throttle application here. Your best was 84% — room to improve."
  - Do **NOT** compare consistency/progression metrics across drivers: all `_std` metrics, `_mean` metrics, `opportunity_score`. These reflect driver skill level, not car capability.
  - Do **NOT** compare driver progression (e.g., "Driver A improved more than Driver B"). Each driver's progression is tracked independently.
- **Different car**: Skip cross-session corner comparison entirely (different car = different physics)

### Cross-Session Comparison

When previous session report(s) are available:

1. **Verify track match**: corner count and track layout should match (same turn IDs)
2. **Check weather stability** before comparing per-corner data:

   | Condition | Threshold |
   |-----------|-----------|
   | Temperature | Within 5°C |
   | Rain transition | No dry↔wet (WMO codes 0–3 = dry, 51+ = precipitation) |
   | Wind speed | Within 15 km/h |

   If weather changed significantly, note it in the report and skip per-corner lap-level comparisons for that session pair. KPI trend comparison (mean/std) is still valid.

3. **Compare KPIs**: read the KPI table from the previous report and calculate progress:
   - Status: **MET** (reached target), **IMPROVED** (moved toward target), **REGRESSED** (moved away), **UNCHANGED** (< 5% change)
4. **Per-corner lap-level comparison** (new): When previous session JSON is available and weather is stable:
   - Read `corner_consistency[*].per_lap_metrics` from the previous JSON
   - Compare current session's per-corner best values against previous session's per-lap data at matching corners (same corner ID)
   - Report improvements: "Your best exit at T3 was 124.9 km/h (Lap 11), up from 122.1 km/h best in Session 1 (Lap 7)"
   - **Causality**: only reference older sessions, never newer ones
5. **Generate comparison section** in the report showing previous → current → target for each KPI
6. **Update KPIs** — keep unmet KPIs, adjust targets for met ones, add new areas

### Important Notes

- **Prefer `.xrk` over `.xrz`:** When both formats exist, use `.xrk`. XRK files have correct metadata (driver, vehicle, etc.) while XRZ compressed archives can have stale/incorrect metadata fields.
- The report JSON uses canonical channel key names, not raw AIM channel names
- `opportunity_score = exit_speed_std × accel_zone_length` — higher means more lap time to gain
- All speeds are in km/h, distances in meters, times in seconds
- **Lap time display**: Always use the `_fmt` fields (e.g., `best_lap_time_fmt`, `mean_top_lap_time_fmt`, per-lap `lap_time_fmt`) which are pre-formatted as `M:SS.mmm` (e.g., `1:57.566`). Never display raw seconds for lap times in the report.
- `best_lap.vs_mean` values are deltas: positive = better than average
- `excluded_laps` lists laps that went off-track at each corner (GPS deviation > 10m from median path). These laps are automatically excluded from all consistency stats and best-lap selection. If a corner has excluded laps, mention them briefly (e.g., "Laps 5, 11 excluded — off-track excursion detected").
- If `skipped_analyses` is non-empty, mention which analyses weren't available and why
- Suspension and tire grip findings are **setup** recommendations, not driver technique

### G Utilization & G Hole Interpretation

G utilization measures how continuously the driver uses available tire grip through the braking → turn-in → mid-corner → exit → acceleration sequence. A "G hole" is the valley in total G between the braking peak and the cornering peak — it's the transition where the driver has released the brake but hasn't yet built up lateral G through turn-in.

**G hole detection** finds the minimum total G between the braking G peak and cornering G peak for each corner:
- `total_g_min_mean` — average depth of the G hole across laps
- `total_g_min_phase` — which phase the G hole occurs in (braking/entry/mid/exit)
- `g_utilization_pct` — G hole depth relative to the lower of the two surrounding peaks (higher = smoother)

**Best lap G data** (on `best_lap`): `g_utilization_pct`, `total_g_min`, `total_g_min_dist`, `total_g_min_phase` — use these to describe the specific execution: "On your best lap at Turn 1, total G dropped to 0.15G at 712m during the entry phase"

**Phase interpretation:**
- `total_g_min_phase = "entry"` → trail braking gap — brakes released before turn-in builds lateral G. Fix: maintain brake pressure deeper into the turn, release gradually as steering input increases
- `total_g_min_phase = "exit"` → exit hesitation — gap between turning and accelerating. Fix: begin throttle application earlier while still unwinding steering
- `total_g_min_phase = "mid"` → mid-corner lift — driver pauses or lifts at apex. Fix: carry more commitment through the apex
- `total_g_min_phase = "braking"` → braking too early — driver finished braking and coasted before the corner started. Check `early_braking_coast_m` for the distance. Fix: brake later so braking flows directly into turn-in via trail braking

**Severity thresholds:**
- `g_utilization_mean` < 30% → HIGH — large G holes, major coaching priority
- `g_utilization_mean` < 50% → MEDIUM — significant grip wasted in transitions
- `g_utilization_mean` < 70% → LOW — some room for improvement
- `g_utilization_mean` >= 70% → OK — smooth transitions

Compare per-phase G means (`braking_g_mean_val`, `entry_g_mean_val`, `mid_g_mean_val`, `exit_g_mean_val`): the lowest phase is where the driver is leaving grip on the table.

### Report Formatting Conventions

- **Lap references**: Always write "Lap 1", "Lap 2", etc. — never abbreviate to "L1", "L2"
- **Throttle acceptance**: Always write "throttle acceptance" in full — never abbreviate to "TA"
- **Consistency, not std**: When describing variation to the driver, say "consistency" not "std" or "standard deviation". E.g., "exit speed consistency: 2.7 km/h" not "exit speed std: 2.7 km/h". The underlying metric names in code/tables can still use `_std` but narrative text should say consistency.
- **No blockquotes for analysis details**: Use normal formatting (bold headers, paragraphs, lists) for best execution descriptions, technique notes, and targets. Do not wrap them in blockquotes (`>`). Blockquotes are only for session notes from the logger metadata.
- **Table headers**: Use "Consistency" instead of "Std" as the column header in metrics tables

### Brake Balance Interpretation

When `braking_balance` is available in the report (requires front + rear brake channels):

- The report automatically filters out low-brake corners (< 20% of session peak) to remove noise from lift-and-turn situations. The `min_brake_threshold` field shows the cutoff used.
- Consistent balance across corners = good setup baseline
- Per-corner balance shifts > 3% from session mean suggest the driver is modulating balance with pedal technique (flag for discussion with driver)
- If `overall_balance_pct` is far from typical (e.g., < 55% or > 75% front bias), suggest engineer review
- Large `overall_balance_std` indicates the driver's pedal technique creates variable balance — may be intentional or a pedal feel issue
