# Session Analysis Skill

Analyze motorsports telemetry sessions, identify improvement areas, set KPIs, and track progress.

## Usage

```
/session-analysis <file_path> [file_path2 ...]
```

## Workflow

### 1. Generate Report

For each provided file path, run:

```bash
uv run python scripts/analyze_session.py "<file_path>" --output /tmp/session_report.json --track-map /tmp/track_map.png --comparison-dir /tmp/comparisons/
```

Read the JSON output with the Read tool. The `--track-map` flag generates a track map image with labeled corners. The `--comparison-dir` flag generates per-corner input comparison plots and zoomed GPS maps for the top 5 opportunity corners.

### 2. Check Previous State

Look for `.session-analysis/latest.json` in the project root. If it exists, this is a **cross-session comparison**. If not, this is a **first analysis**.

### 3. Identify Improvement Areas

Rank the top 3–5 improvement areas by impact using corner consistency data from the report:

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

> **Turn N — Corner Exit Speed**
> Your best execution was on **Lap X**: you braked Ym later than average, carried Z km/h more through the apex, and exited W km/h faster.
> **Target**: Replicate this on 80%+ of laps.

When braking metrics are available, include them in the best-lap description:

> **Turn N — Braking Zone**
> Your best execution was on **Lap X**: you braked W bar harder, released Y m earlier, and carried Z km/h more entry speed than average.
> **Target**: Replicate this on 80%+ of laps.

Reference `best_lap.vs_mean` for the specific deltas and `best_lap.selection_reason` for context.

**Exit speed root cause analysis**: When the best lap has better exit speed (`vs_mean.exit_speed > 0`), analyze the `vs_mean` deltas to explain WHY it was faster. Check two causal chains:

- **Braking chain**: `entry_speed` → `min_speed` → `exit_speed` (carried more speed through)
- **Throttle chain**: `throttle_point` / `throttle_acceptance_pct` → `exit_speed` (got on power earlier)

Format the explanation as a ranked list of contributing factors:

> **Why was Lap X faster at Turn N?**
> 1. **Got on throttle 8m earlier** — the biggest factor
> 2. **Carried 2.4 km/h more through the apex**

Only include factors with meaningful deltas (speed > 1 km/h, distance > 2m, TA > 5%). Rank by impact magnitude. If no clear cause can be determined from the data, state that rather than guessing.

**G utilization technique notes**: For corners where `g_utilization_mean < 70%`, add an actionable technique note based on `total_g_min_phase`:

| Phase | Note |
|-------|------|
| entry | "Trail brake deeper — releasing brake before turn-in wastes lateral G" |
| exit | "Get on throttle earlier — pause between turning and accelerating costs grip" |
| mid | "Commit through the apex — lifting mid-corner breaks momentum" |
| braking | "Apply brakes more progressively — abrupt application creates a grip gap" |

Only mention G utilization for corners where it's below 70%. Don't add a standalone G utilization table — fold it into the improvement area notes.

### 5. Set KPIs

For each improvement area, define a measurable KPI:

```json
{
  "name": "Turn 3 exit speed consistency",
  "area": "corner_exit",
  "corner_id": 3,
  "metric_key": "exit_speed_std",
  "current_value": 4.2,
  "target_value": 2.0,
  "unit": "km/h"
}
```

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

Output a structured markdown report with:

1. **Track Map** — embed the track map image: `![Track Map](track_map.png)` (using the image generated in step 1)
2. **Quick Reference** — extremely brief, memorizable list of the top 3 technique focuses. Format:
   ```
   ## Quick Reference
   1. **T2**: Trail brake deeper, commit to the apex
   2. **T9**: Get on throttle earlier at exit
   3. **T4**: Full throttle sooner — you proved it on Lap 9
   ```
   Each item should be one line with the corner ID and the specific action. Reference a lap number where the driver proved they can do it. This must be concise enough to remember while driving.
3. **Session Overview** — metadata, lap times, track info
4. **Top Improvement Areas** — ranked list. For each corner, use this format:
   ```
   ### N. Turn X — Short Description (Opportunity: NNNN)

   | Metric | Value | Std | Best Lap |
   |--------|-------|-----|----------|
   | Min Speed | 93.5 km/h | 4.2 km/h | 95.3 km/h (L11) |
   | Exit Speed | 120.8 km/h | 2.7 km/h | 124.9 km/h (L11) |
   | Throttle Acceptance | 78.4% | 17.8% | 98.8% (L11) |
   | Braking Point | 450m | 4.1m | 452.6m (L11) |
   | G Utilization | 40.3% | — | — |

   (Only include rows for metrics that are relevant/available for this corner.)

   > **Best Execution: Lap 11**
   > ...root cause analysis...
   >
   > **Technique:** ...G utilization note if < 70%...
   >
   > **Target:** ...

   ![Turn X Inputs](comparisons/comparison_t{id}_inputs.png)
   ![Turn X Map](comparisons/comparison_t{id}_map.png)
   ```
   Include: metrics table, root cause analysis for exit speed gains, G utilization technique notes where applicable (< 70%), comparison images
5. **Corner-by-Corner Summary** — table of all corners with key consistency metrics
6. **Brake Balance** — filtered summary (low-brake corners are excluded automatically by the report generator)
7. **Setup Notes** — suspension and tire grip findings (if available)
8. **KPIs** — table of targets for next session
9. **Skipped Analyses** — any analyses that couldn't run and why

When saving the report, copy the track map image and comparison images to `.session-analysis/` alongside the markdown report so the relative image references work.

### 7. Save State

Save the full state for future comparison:

```bash
mkdir -p .session-analysis/history
```

Write `.session-analysis/latest.json`:

```json
{
  "analysis_date": "YYYY-MM-DDTHH:MM:SS",
  "session_file": "<file_path>",
  "track_info": {
    "track_length_m": 4523.0,
    "corner_count": 12
  },
  "report": { "/* SessionReport.to_dict() */" : "..." },
  "kpis": [ "/* KPI objects */" ]
}
```

Archive to `.session-analysis/history/YYYY-MM-DD_<session_name>.json` (same content).

### Cross-Session Comparison

When `.session-analysis/latest.json` exists:

1. **Verify track match**: corner count must match, `track_length_m` within 5%
2. **Calculate KPI progress** for each previous KPI:
   - `% change` = (current - previous) / previous × 100
   - Status: **MET** (reached target), **IMPROVED** (moved toward target), **REGRESSED** (moved away), **UNCHANGED** (< 5% change)
3. **Generate comparison table** showing previous → current → target for each KPI
4. **Update KPIs** — keep unmet KPIs, adjust targets for met ones, add new areas
5. **Save updated state** to `latest.json` and archive

### Important Notes

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
- `total_g_min_phase = "braking"` → late/abrupt brake application creating a gap before peak braking force

**Severity thresholds:**
- `g_utilization_mean` < 30% → HIGH — large G holes, major coaching priority
- `g_utilization_mean` < 50% → MEDIUM — significant grip wasted in transitions
- `g_utilization_mean` < 70% → LOW — some room for improvement
- `g_utilization_mean` >= 70% → OK — smooth transitions

Compare per-phase G means (`braking_g_mean_val`, `entry_g_mean_val`, `mid_g_mean_val`, `exit_g_mean_val`): the lowest phase is where the driver is leaving grip on the table.

### Brake Balance Interpretation

When `braking_balance` is available in the report (requires front + rear brake channels):

- The report automatically filters out low-brake corners (< 20% of session peak) to remove noise from lift-and-turn situations. The `min_brake_threshold` field shows the cutoff used.
- Consistent balance across corners = good setup baseline
- Per-corner balance shifts > 3% from session mean suggest the driver is modulating balance with pedal technique (flag for discussion with driver)
- If `overall_balance_pct` is far from typical (e.g., < 55% or > 75% front bias), suggest engineer review
- Large `overall_balance_std` indicates the driver's pedal technique creates variable balance — may be intentional or a pedal feel issue
