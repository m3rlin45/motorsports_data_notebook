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
uv run python scripts/analyze_session.py "<file_path>" --output /tmp/session_report.json
```

Read the JSON output with the Read tool.

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

### 4. Show Best Execution

For each improvement area, use the `best_lap` data in the corner's consistency entry:

> **Turn N — Corner Exit Speed**
> Your best execution was on **Lap X**: you braked Ym later than average, carried Z km/h more through the apex, and exited W km/h faster.
> **Target**: Replicate this on 80%+ of laps.

When braking metrics are available, include them in the best-lap description:

> **Turn N — Braking Zone**
> Your best execution was on **Lap X**: you braked W bar harder, released Y m earlier, and carried Z km/h more entry speed than average.
> **Target**: Replicate this on 80%+ of laps.

Reference `best_lap.vs_mean` for the specific deltas and `best_lap.selection_reason` for context.

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

1. **Session Overview** — metadata, lap times, track info
2. **Top Improvement Areas** — ranked list with current metrics and best-lap references
3. **Corner-by-Corner Summary** — table of all corners with key consistency metrics
4. **Setup Notes** — suspension and tire grip findings (if available)
5. **KPIs** — table of targets for next session
6. **Skipped Analyses** — any analyses that couldn't run and why

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
- `best_lap.vs_mean` values are deltas: positive = better than average
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

- Consistent balance across corners = good setup baseline
- Per-corner balance shifts > 3% from session mean suggest the driver is modulating balance with pedal technique (flag for discussion with driver)
- If `overall_balance_pct` is far from typical (e.g., < 55% or > 75% front bias), suggest engineer review
- Large `overall_balance_std` indicates the driver's pedal technique creates variable balance — may be intentional or a pedal feel issue
