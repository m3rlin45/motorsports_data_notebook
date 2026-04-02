# inferno-driving-coach

MCP server for motorsports telemetry coaching. Analyzes AIM (XRK/XRZ) and iRacing (IBT) telemetry data to produce structured session reports with corner consistency analysis, braking analysis, G utilization, and actionable improvement areas.

## Installation

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "inferno-driving-coach": {
      "command": "uvx",
      "args": ["inferno-driving-coach"]
    }
  }
}
```

### Claude Code

Add to your project or global settings:

```json
{
  "mcpServers": {
    "inferno-driving-coach": {
      "command": "uvx",
      "args": ["inferno-driving-coach"]
    }
  }
}
```

## Tools

### `analyze_session`

Run full session analysis on telemetry files. Produces a structured report with:

- Lap times and consistency metrics
- Corner-by-corner consistency analysis (braking, apex speed, exit speed, throttle acceptance)
- G utilization and "G hole" detection
- Braking balance analysis
- Suspension velocity histograms
- Tire grip and condition data
- Track map and corner comparison images

Accepts multiple files from the same session (e.g., split by a car restart) and merges them automatically.

### `group_sessions`

Detect and group split session files into logical sessions. AIM loggers create a new file when the car is restarted — this tool detects those splits by comparing metadata timestamps and groups files that belong together.

### `list_profiles`

List available vehicle profiles and their channel name mappings. Profiles map canonical channel names (like "throttle", "brake", "gps_speed") to the actual AIM channel names for each vehicle/logger combination.

## Prompts

### `session_analysis`

Full coaching workflow: analyzes a session, identifies the top 3 improvement areas, shows best execution with root cause analysis, sets KPIs, and generates a markdown report with track map and comparison images.

### `session_day_review`

Multi-session workflow: discovers all telemetry files in a directory, groups split sessions, analyzes each session sequentially with cross-session KPI tracking, and generates a day summary with progression analysis.

## License

MIT
