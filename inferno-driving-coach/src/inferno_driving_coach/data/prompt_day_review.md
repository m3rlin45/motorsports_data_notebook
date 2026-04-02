# Session Day Review

Analyze all sessions from a track day, track progress across sessions, and generate a day summary.

## Directory Path

{{directory_path}}

## Workflow

### 1. Discover Sessions

List all `.xrz` and `.xrk` files in the specified directory, sorted by filename (AIM filenames encode session number, e.g., `CMD_KK-SII_Fuji GP_Generic testing_a_0035.xrz`).

**Prefer `.xrk` over `.xrz`:** When both formats exist for the same session (same base name), use the `.xrk` file. XRK files contain the raw data with correct metadata (driver name, etc.), while XRZ compressed archives can have stale/incorrect metadata fields. Deduplicate by base name and pick `.xrk` when both are present.

Print the list of files found and confirm with the user before proceeding.

#### Detect split sessions

AIM loggers create a new file when the car is restarted (e.g., after a spin). These split files are really one session and should be merged for analysis. After deduplicating to `.xrk` files, call the `group_sessions` tool:

```
group_sessions(files=["<file1>", "<file2>", ...], max_gap_minutes=30.0)
```

This returns a JSON array of groups. Each group is one logical session (one or more files).

Parse the JSON and present the grouping to the user, showing start times, lap counts, and gaps for merged files:

```
Detected 3 logical sessions from 5 files:
  Session 1: file_0095.xrk (08:51, 9 laps) + file_0096.xrk (09:11, 4 laps) — merged, 20 min gap
  Session 2: file_0098.xrk (10:10, 14 laps)
  Session 3: file_0100.xrk (11:32, 3 laps) + file_0101.xrk (11:49, 7 laps) — merged, 17 min gap
```

Confirm with the user before proceeding. When analyzing sessions, pass all files in a group to the `analyze_session` tool. The tool handles merging internally (renumbers laps, excludes partial boundary laps). For example:

```
analyze_session(
    session_files=["file_0095.xrk", "file_0096.xrk"],
    output_dir=".session-analysis/session_01",
    session_num=1
)
```

### 2. Analyze Sessions Sequentially

Process sessions **sequentially** — each one builds on the previous session's KPIs.

#### Create output directory

Create `.session-analysis/` for outputs.

#### Analyze each session

For each session, call the `analyze_session` tool. Sessions must be sequential (each reads the previous session's report for cross-session comparison).

For **session 1** (no previous report):

```
analyze_session(
    session_files=["<file_path>", ...],
    output_dir=".session-analysis/session_01",
    session_num=1
)
```

Then follow the session analysis workflow:
1. Parse the returned JSON report
2. No previous reports — this is the first session
3. Note the driver name and vehicle
4. Identify improvements, set KPIs, generate markdown report
5. Write the markdown report to `.session-analysis/step01_report.md`
6. Write the appendix to `.session-analysis/step01_appendix.md`

For **session N** (N > 1, with previous reports):

```
analyze_session(
    session_files=["<file_path>", ...],
    output_dir=".session-analysis/session_<NN>",
    session_num=<N>
)
```

Then:
1. Parse the returned JSON report
2. Read the previous session's markdown report for KPI comparison
3. Check driver/vehicle: compare metadata between current and previous
   - Same driver + same car: full comparison valid
   - Different driver + same car: only compare car capability metrics
   - Different car: skip cross-session corner comparison
4. Check weather stability between sessions (temperature within 5C, no dry/wet transition, wind within 15 km/h)
5. Identify improvements, set KPIs, compare against previous KPIs
6. Write the markdown report with KPI progress section
7. Write the appendix

### 3. Generate Day Summary

After all sessions are processed, read all the markdown reports and JSON data, then generate the day summary.

### Day Summary Format

The day summary heading should include driver, vehicle, venue, and date:
```
# Day Summary — CMD / Inferno 86 — Sodegaura — 2026-01-12
```

The day summary markdown should cover:

#### Driver & Vehicle
- List the driver(s) and vehicle for each session
- If the driver changed between sessions, clearly note which sessions had which driver
- Flag that cross-session comparisons between different drivers used car-capability-only mode

#### Lap Time Progression
- Table: Session # | Driver | Best Lap | Mean Top Lap | Lap Count | Improvement vs Previous
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

```
.session-analysis/day_summary_YYYY-MM-DD.md
```

Where `YYYY-MM-DD` is the date from the session metadata.

### Important Notes

- Sessions must be from the same track (verify corner count and track layout match across reports)
- Sequential processing is required because each session reads the previous session's markdown report for cross-session comparison
- If a session fails to analyze, log the error and continue with remaining sessions
- The day summary should reference specific session numbers and lap numbers for actionable feedback
- All speeds in km/h, distances in meters, times in seconds
- Cross-session state is carried through both markdown reports (KPIs, narrative) and JSON data (per-lap corner data, metadata)
- **Driver changes**: Check `metadata.driver` from each session. If drivers differ between consecutive sessions, switch to car-capability-only comparison mode. The day summary should clearly note driver changes.
- **Weather stability**: Compare `weather` objects between consecutive sessions. If temperature differs by >5C, wind by >15 km/h, or a dry/wet transition occurred, per-corner lap-level comparisons are skipped for that pair.

### Report Formatting Conventions

- **Lap references**: Always write "Lap 1", "Lap 2", etc. — never abbreviate to "L1", "L2"
- **Throttle acceptance**: Always write "throttle acceptance" in full — never abbreviate to "TA"
- **Consistency, not std**: When describing variation to the driver, say "consistency" not "std" or "standard deviation"
- **Lap time display**: Always use the `_fmt` fields which are pre-formatted as `M:SS.mmm`
