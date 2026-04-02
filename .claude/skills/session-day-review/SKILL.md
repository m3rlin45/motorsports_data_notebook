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

Process sessions **sequentially** — each one builds on the previous session's KPIs via `.session-analysis/latest.json`.

Create the steps directory first:
```bash
rm -rf .session-analysis && mkdir -p .session-analysis/steps .session-analysis/history
```

For each session file, spawn a **background** sub-agent:

```
Agent tool with:
  - subagent_type: general-purpose
  - mode: bypassPermissions
  - run_in_background: true
  - prompt: |
      Follow the session-analysis skill at .claude/skills/session-analysis/SKILL.md.

      Session file: <file_path>
      Step number: <NN> (zero-padded, e.g., 01, 02)

      1. Run: uv run python scripts/analyze_session.py "<file_path>" --output /tmp/session_report.json --track-map /tmp/track_map.png --comparison-dir /tmp/comparisons/
      2. Read the JSON output with the Read tool
      3. Check if .session-analysis/latest.json exists for cross-session comparison
      4. Follow the SKILL.md: identify improvements, set KPIs, generate markdown report
      5. Save state to .session-analysis/latest.json and .session-analysis/history/
      6. Copy .session-analysis/latest.json to .session-analysis/steps/step<NN>_<session_basename>.json
      7. Copy /tmp/track_map.png to .session-analysis/step<NN>_track_map.png
      8. Copy /tmp/comparisons/ to .session-analysis/step<NN>_comparisons/ (if it exists)
      9. Write the markdown report to .session-analysis/step<NN>_report.md (reference the track map as ![Track Map](step<NN>_track_map.png) and comparison images as ![Turn N Inputs](step<NN>_comparisons/comparison_t{id}_inputs.png))
```

**CRITICAL — context isolation:**
- Each sub-agent MUST run with `run_in_background: true` so its results stay out of the main conversation context
- Wait for each sub-agent to complete (via the automatic completion notification) before spawning the next one
- Do NOT read the sub-agent's output back into the main context — the results are on disk
- Give the user a brief status update after each session completes (e.g., "Session 2/4 done") but do NOT summarize the sub-agent's findings

### 3. Generate Day Summary

After all sessions are processed, spawn one final **background** sub-agent to read the step files from disk and generate the day summary:

```
Agent tool with:
  - subagent_type: general-purpose
  - mode: bypassPermissions
  - run_in_background: true
  - prompt: |
      Read the step JSON files and session reports from .session-analysis/steps/ and
      .session-analysis/step*_report.md. Follow the day summary format in
      .claude/skills/session-day-review/SKILL.md section "Day Summary Format".
      Write the result to .session-analysis/day_summary_YYYY-MM-DD.md.
```

When the day summary agent completes, tell the user the file path and offer to show it.

### Day Summary Format

The day summary markdown should cover:

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

### 4. Save Day Summary

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
