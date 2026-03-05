# Session Day Review Skill

Analyze all sessions from a track day, track progress across sessions, and generate a day summary.

## Usage

```
/session-day-review <directory_path>
```

## Workflow

### 1. Discover Sessions

List all `.xrz` and `.xrk` files in the specified directory, sorted by filename (AIM filenames encode session number, e.g., `CMD_KK-SII_Fuji GP_Generic testing_a_0035.xrz`).

```bash
ls -1 "<directory_path>"/*.xrz "<directory_path>"/*.xrk 2>/dev/null | sort
```

Print the list of files found and confirm with the user before proceeding.

### 2. Analyze Each Session

For each session file **sequentially**, spawn a sub-agent that:

1. Runs `/session-analysis <file_path>`
2. The sub-agent writes its analysis to `.session-analysis/latest.json` and `.session-analysis/history/`
3. Each subsequent session picks up the previous state automatically via the cross-session comparison workflow in `/session-analysis`

**Key design:** Each sub-agent writes to disk (`.session-analysis/` files). The orchestrating agent reads results from disk rather than receiving them in-context. This keeps the main context clean regardless of how many sessions are analyzed.

The sub-agent should be spawned with:
```
Agent tool with:
  - subagent_type: general-purpose
  - prompt: "Run /session-analysis on <file_path>. After the analysis completes, copy .session-analysis/latest.json to .session-analysis/steps/step<NN>_<session_name>.json where <NN> is the zero-padded step number and <session_name> is the base filename without extension."
```

Create the steps directory first:
```bash
mkdir -p .session-analysis/steps
```

### 3. Read Results

After all sessions are processed:

1. Read `.session-analysis/latest.json` for the final cumulative state
2. Read each `.session-analysis/steps/step*.json` for individual session snapshots
3. Extract from each step file:
   - `session_file` — which file was analyzed
   - `report.lap_times` — best and mean lap times
   - `kpis` — KPI values at that point in the day

### 4. Generate Day Summary

Output a structured markdown report covering:

#### Lap Time Progression
- Table: Session # | Best Lap | Mean Top Lap | Lap Count | Improvement vs Previous
- Trend: improving, plateauing, or regressing across the day
- Note any session where times got significantly worse (potential setup change, tire degradation, or fatigue)

#### KPI Evolution
- Table: KPI Name | Session 1 | Session 2 | ... | Session N | Target | Status
- Status: **MET**, **IMPROVING**, **REGRESSING**, **UNCHANGED**
- Highlight KPIs that were met during the day

#### Corner-by-Corner Progression
- For the top 3 opportunity corners (by `opportunity_score`):
  - How did consistency change across sessions?
  - Did the driver's best execution improve?
  - Which metrics improved vs regressed?

#### Top 3 Takeaways

Distill the day into 3 actionable takeaways for the driver:

1. **Biggest improvement** — what got better and by how much (reference specific corners, sessions, and metrics)
2. **Biggest remaining gap** — the highest-impact area still needing work, with specific targets
3. **Consistency trend** — whether the driver maintained improvements or regressed through the day (fatigue, tire deg, etc.)

#### Recommendations for Next Track Day

Based on the final KPI state and day-long trends:
- Which KPIs to prioritize in the next session
- Specific technique focuses (e.g., "Trail braking into Turn 3 — you improved brake release point by 2m but still have 3m to gain")
- Setup suggestions if flagged (suspension, brake balance)
- Session planning advice (e.g., "Focus first session on Turn 3 braking, use later sessions for full-lap consistency")

### 5. Save Day Summary

Write the day summary markdown to:

```bash
.session-analysis/day_summary_YYYY-MM-DD.md
```

Where `YYYY-MM-DD` is today's date.

### Important Notes

- Sessions must be from the same track (verify corner count and track length match across step files)
- Sequential processing is required because each `/session-analysis` invocation reads `.session-analysis/latest.json` for cross-session comparison — session N must complete before session N+1 starts
- If a session fails to analyze, log the error and continue with remaining sessions
- The day summary should reference specific session numbers and lap numbers for actionable feedback
- All speeds in km/h, distances in meters, times in seconds
- Each sub-agent's context is isolated — the orchestrating agent reads all results from disk after all sessions complete
- All sub-agents share the same `.session-analysis/latest.json` — this is intentional, as each session builds on the previous one's KPIs
