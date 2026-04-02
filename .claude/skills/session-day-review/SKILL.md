---
name: session-day-review
description: Analyze all sessions from a track day, track progress across sessions, and generate a day summary.
---

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

### 2. Set Up Team and Analyze Sessions

Process sessions **sequentially** — each one builds on the previous session's KPIs.

#### Create output directory

```bash
rm -rf .session-analysis && mkdir -p .session-analysis
```

#### Create a team for background agents

Standalone background agents (`run_in_background: true`) **cannot** prompt the user for tool permissions and will fail on Write/Bash calls. Agent teams solve this — team members inherit permissions from the controller's session.

```
TeamCreate with:
  - team_name: "session-analysis"
  - description: "Analyze telemetry sessions from a track day"
  - agent_type: "controller"
```

#### Spawn session analyzers sequentially

For each session file, spawn a **team member** agent. Sessions must be sequential (each reads the previous session's markdown report for cross-session comparison), so wait for each teammate's completion message before spawning the next.

For **session 1** (no previous report):

```
Agent tool with:
  - subagent_type: general-purpose
  - mode: bypassPermissions
  - run_in_background: true
  - name: "session-01"
  - team_name: "session-analysis"
  - prompt: |
      Follow the session-analysis skill at .claude/skills/session-analysis/SKILL.md.

      Session file: <file_path>
      Step number: 01

      1. Run: uv run python scripts/analyze_session.py "<file_path>" --output .session-analysis/step01_data.json --track-map .session-analysis/step01_track_map.png --comparison-dir .session-analysis/step01_comparisons/
      2. Read the JSON output with the Read tool
      3. No previous reports — this is the first session
      4. Follow the SKILL.md: identify improvements, set KPIs, generate markdown report
      5. Write the markdown report to .session-analysis/step01_report.md
         - Reference track map as ![Track Map](step01_track_map.png)
         - Reference comparisons as ![Turn N Inputs](step01_comparisons/comparison_t{id}_inputs.png)
      6. Send a message to the team lead with a brief status
```

For **session N** (N > 1, with previous reports):

```
Agent tool with:
  - subagent_type: general-purpose
  - mode: bypassPermissions
  - run_in_background: true
  - name: "session-<NN>"
  - team_name: "session-analysis"
  - prompt: |
      Follow the session-analysis skill at .claude/skills/session-analysis/SKILL.md.

      Session file: <file_path>
      Step number: <NN>

      1. Run: uv run python scripts/analyze_session.py "<file_path>" --output .session-analysis/step<NN>_data.json --track-map .session-analysis/step<NN>_track_map.png --comparison-dir .session-analysis/step<NN>_comparisons/
      2. Read the JSON output with the Read tool
      3. Read the previous session report for cross-session comparison:
         .session-analysis/step<NN-1>_report.md
      4. Follow the SKILL.md: identify improvements, set KPIs, compare against previous KPIs
      5. Write the markdown report to .session-analysis/step<NN>_report.md
         - Reference track map as ![Track Map](step<NN>_track_map.png)
         - Reference comparisons as ![Turn N Inputs](step<NN>_comparisons/comparison_t{id}_inputs.png)
         - Include a KPI progress section comparing current values to previous report's targets
      6. Send a message to the team lead with a brief status
```

**Context isolation:**
- Each teammate runs in the background — only their short status message enters the main context
- Wait for each teammate's message before spawning the next one
- Give the user a brief status update after each session completes (e.g., "Session 2/4 done")
- Send a `shutdown_request` to each teammate after receiving their status message

### 3. Generate Day Summary

After all sessions are processed, spawn one final team member to read all the markdown reports and generate the day summary:

```
Agent tool with:
  - subagent_type: general-purpose
  - mode: bypassPermissions
  - run_in_background: true
  - name: "day-summary"
  - team_name: "session-analysis"
  - prompt: |
      Read all session reports from .session-analysis/step*_report.md.
      Follow the day summary format in .claude/skills/session-day-review/SKILL.md section "Day Summary Format".
      Write the result to .session-analysis/day_summary_YYYY-MM-DD.md.
      When done, send a message to the team lead with "Day summary written to .session-analysis/day_summary_YYYY-MM-DD.md".
```

When the day summary agent completes, shut it down, delete the team, tell the user the file path and offer to show it.

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

- Sessions must be from the same track (verify corner count and track layout match across reports)
- Sequential processing is required because each session reads the previous session's markdown report for cross-session comparison
- If a session fails to analyze, log the error and continue with remaining sessions
- The day summary should reference specific session numbers and lap numbers for actionable feedback
- All speeds in km/h, distances in meters, times in seconds
- Each teammate's context is isolated — the controller reads only their short status messages
- Cross-session state is carried entirely through the markdown reports — no separate JSON state files
