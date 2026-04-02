# inferno-driving-coach — MCP Server Design

## Overview

Standalone PyPI package providing an MCP server for motorsports telemetry coaching. Bundles session analysis tools and coaching prompts into a single `uvx`-installable package.

Install: `uvx inferno-driving-coach`

Claude Code config:
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

## Architecture

Workspace member in the `motorsports-data-notebook` repo (like `desktop_app/`). Depends on `motorsports-data-notebook` as a regular PyPI dependency.

```
motorsports_data_notebook-worktree/
  src/motorsports_data_notebook/          # core library (existing)
    session_runner.py                      # NEW — shared logic extracted from scripts/
  desktop_app/                             # existing workspace member
  inferno-driving-coach/                   # NEW workspace member
    pyproject.toml
    src/inferno_driving_coach/
      __init__.py                          # FastMCP instance + main()
      __main__.py                          # python -m entry point
      tools.py                             # @mcp.tool() definitions
      prompts.py                           # @mcp.prompt() definitions
      data/
        prompt_session_analysis.md         # Adapted from session-analysis SKILL.md
        prompt_day_review.md               # Adapted from session-day-review SKILL.md
```

## Implementation Steps

### Step 1: Extract shared logic into `session_runner.py`

Create `src/motorsports_data_notebook/session_runner.py` with functions extracted from the CLI scripts. This lets both the scripts and the MCP tools share the same logic.

**`run_session_analysis()`** — Logic from `scripts/analyze_session.py:main()` minus argparse:
- Accepts: `session_files: list[str]`, `profile: str | None`, `threshold: float`, `session_num: int`, `output_dir: str | None`
- Returns: `dict` with keys `report` (SessionReport), `image_paths` (list[str])
- Reuses: `load_session()`, `MergedLogFile` from `_util.py`; `generate_session_report()` from `report.py`; profile resolution from `profiles.py`; image gen from `visualization.py`
- Handles: file loading, merging, profile resolution, report generation, track map + comparison images

**`run_group_sessions()`** — Wrapper around `group_session_files()` from `scripts/group_sessions.py`:
- Accepts: `files: list[str]`, `max_gap_minutes: float = 30.0`
- Returns: `list[dict]` (group metadata)
- Imports `group_session_files` from `scripts/group_sessions.py` or moves the logic inline

Refactor both scripts to call these functions (scripts keep working for non-MCP users).

**Key files:**
- `src/motorsports_data_notebook/session_runner.py` (new)
- `scripts/analyze_session.py` (refactor: call `run_session_analysis()`, keep argparse + CLI args)
- `scripts/group_sessions.py` (refactor: call `run_group_sessions()`, keep argparse)

### Step 2: Create `inferno-driving-coach/` package

**`inferno-driving-coach/pyproject.toml`:**
```toml
[project]
name = "inferno-driving-coach"
version = "0.1.0"
description = "MCP server for motorsports telemetry coaching — session analysis, day reviews, and driving KPIs"
authors = [
    {name = "Christopher Dewan", email = "chris.dewan@m3rlin.net"}
]
requires-python = ">=3.12,<4.0"
dependencies = [
    "mcp>=1.2.0",
    "motorsports-data-notebook",
]

[project.scripts]
inferno-driving-coach = "inferno_driving_coach:main"

[build-system]
requires = ["setuptools>=75.0.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
inferno_driving_coach = ["data/*.md"]

[tool.uv.sources]
motorsports-data-notebook = { workspace = true }

[tool.black]
line-length = 100
target-version = ['py312']

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
check_untyped_defs = true

[[tool.mypy.overrides]]
module = ["mcp.*", "motorsports_data_notebook.*"]
ignore_missing_imports = true
```

### Step 3: Add workspace member to root `pyproject.toml`

```toml
[tool.uv.workspace]
members = ["desktop_app", "inferno-driving-coach"]
```

Also add `"mcp.*"` to the root mypy overrides `ignore_missing_imports` list.

### Step 4: Create MCP server module

**`__init__.py`:**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inferno-driving-coach")

from . import tools, prompts  # noqa: F401, E402

def main():
    mcp.run()
```

**`__main__.py`:**
```python
from . import main
main()
```

### Step 5: Implement tools in `tools.py`

Three `@mcp.tool()` functions:

**`analyze_session`** — Run full session analysis
- Params: `session_files: list[str]`, `profile: str | None = None`, `threshold: float = 1.03`, `session_num: int = 1`, `output_dir: str | None = None`
- Calls `session_runner.run_session_analysis()`
- Returns: JSON string of the SessionReport, plus image paths if output_dir specified

**`group_sessions`** — Detect split session files
- Params: `files: list[str]`, `max_gap_minutes: float = 30.0`
- Calls `session_runner.run_group_sessions()`
- Returns: JSON string of session groups with metadata

**`list_profiles`** — List available vehicle profiles
- No params
- Calls `load_builtin_profiles()` + `load_user_profiles()` from `profiles.py`
- Returns: JSON with profile names and channel mappings

### Step 6: Implement prompts in `prompts.py`

Two `@mcp.prompt()` functions, loading text from `data/prompt_*.md` via `importlib.resources`:

**`session_analysis(file_paths: str)`** — Adapted from `.claude/skills/session-analysis/SKILL.md`
- Replace `uv run python scripts/analyze_session.py ...` → `call the analyze_session tool`
- Replace `uv run python scripts/group_sessions.py ...` → `call the group_sessions tool`
- All coaching logic, report format, KPI setting, G utilization interpretation unchanged

**`session_day_review(directory_path: str)`** — Adapted from `.claude/skills/session-day-review/SKILL.md`
- Same tool reference changes
- Orchestration instructions (teams, sequential agents) stay the same — those are Claude Code features

### Step 7: Create prompt data files

Copy and adapt the two SKILL.md files into `inferno-driving-coach/src/inferno_driving_coach/data/`:
- `prompt_session_analysis.md` — session-analysis skill with MCP tool references
- `prompt_day_review.md` — day-review skill with MCP tool references

Only change: CLI script invocations → MCP tool calls. All coaching knowledge stays identical.

### Step 8: Update existing skills

Add note to top of each SKILL.md:
```markdown
> **MCP Server available:** If the `inferno-driving-coach` MCP server is configured,
> the prompts and tools are available directly — no need to use this skill file.
```

Skills remain as fallback.

## Verification

1. `just check` — lint, typecheck, tests all pass
2. `uv sync` in root — workspace resolves with new member
3. `cd inferno-driving-coach && uv sync` — MCP dependency installs
4. `uv run inferno-driving-coach` — server starts (stdio transport)
5. Existing scripts still work: `uv run python scripts/analyze_session.py <file>`
6. Manual test: add MCP config to Claude Code settings, verify prompts appear, run against test .xrk file
