# CLAUDE.md

## Project Overview

**motorsports-data-notebook** is a Python toolkit for analyzing AIM motorsports telemetry data (XRK/XRZ files). It provides interactive Jupyter notebooks that run entirely in the browser via JupyterLite/Pyodide (Python compiled to WebAssembly). No server-side computation — all analysis happens locally.

Live site: https://analysis-preview.inferno.racing/

## Tech Stack

- **Python 3.12+** with **Poetry 2.x** for dependency management
- **PyArrow** for efficient columnar data filtering (prefer over pandas where possible)
- **Plotly** for interactive visualizations (primary charting library)
- **libxrk** for parsing AIM telemetry files (XRK/XRZ)
- **JupyterLite + Pyodide** for browser-based deployment
- **poethepoet (poe)** as task runner

## Repository Structure

```
src/motorsports_data_notebook/   # Main package
  channels.py      # Channel extraction, lap filtering, interpolation
  corners.py       # Corner detection from GPS via curvature analysis
  zones.py         # Braking/acceleration zone detection, track segments
  visualization.py # Plotly-based interactive visualizations (largest module)
  driver_analysis.py # Driver performance metrics (throttle acceptance)
  widgets.py       # Jupyter file upload widget, session loading
tests/             # Unit tests (pytest), one file per module
workspace_template/ # Example notebooks + sample .xrz data file
scripts/           # Build utilities (lite wheel builder, notebook executor)
```

## Common Commands

```bash
# Install dependencies
poetry install
poetry install --extras dev          # dev tools (black, mypy, pytest)
poetry install --extras jupyterlite  # JupyterLite build tools

# Run all checks (lint + typecheck + tests)
poetry run poe check

# Individual checks
poetry run poe lint                  # black --check src/
poetry run poe typecheck             # mypy src/
poetry run poe test                  # pytest with coverage
poetry run poe lint-notebooks        # black --check on notebooks
poetry run poe typecheck-notebooks   # mypy on notebooks

# Format code
poetry run poe format               # black src/
poetry run poe format-notebooks      # black on notebooks

# Run locally
poetry run poe run_clean             # Reset workspace + start JupyterLab

# Build JupyterLite site
poetry run poe build_lite            # Build site (uses pre-built wheels in pypi/)
poetry run poe build_lite_full       # Rebuild project wheel + build site
poetry run poe build_and_serve_lite  # build_lite_full + serve locally

# Emscripten SDK setup (one-time, needed for rebuilding libxrk Pyodide wheel)
poetry run poe setup_emsdk           # Installs emsdk to ~/emsdk
poetry run poe build_libxrk_pyodide  # Rebuild libxrk for Pyodide (requires setup_emsdk)
```

## Code Style & Conventions

- **Formatter:** Black, 100-char line length, target py312
- **Type checker:** mypy (check_untyped_defs=true, disallow_untyped_defs=false)
- **Docstrings:** NumPy-style with Parameters/Returns/Raises/Examples sections
- **Type hints:** Use `from __future__ import annotations`, PEP 604 unions (`X | None`), `TYPE_CHECKING` blocks for import-only types
- **Data handling:** Prefer PyArrow tables and `pyarrow.compute` (pc) operations for filtering/transforms over pandas. Convert to pandas/numpy only for analysis steps that require it.
- **Channel names:** Passed as dictionaries (`channel_names["throttle"]` -> `"PPS"`) to support different AIM telemetry formats. Use `_validate_channel_names()` for validation.
- **Visualization:** All notebook visualizations use Plotly. Use `show_fig()` for environment-aware display (handles JupyterLite vs standard Jupyter).
- **Tests:** pytest with `pytest-cov`. Use mock `MockLogFile` dataclass objects for isolation. Float comparisons with `pytest.approx`. Test classes named `TestClassName`.

## Architecture Notes

### Data Flow
1. Load file (XRK/XRZ) via `widgets.load_session()` -> LogFile with derived channels
2. Extract laps via `channels.get_best_lap()` / `get_top_laps()`
3. Detect corners via `corners.identify_corners()` (GPS -> local XY -> curvature -> corners)
4. Detect zones via `zones.detect_zones_averaged()` (braking/acceleration across multiple laps)
5. Create segments via `zones.create_track_segments()` (combine corners + zones)
6. Compute stats via `zones.compute_segment_stats()` (per-lap metrics)
7. Visualize via `visualization.*` functions

### Key Design Decisions
- **PyArrow over pandas merges:** `channels.get_lap_channels()` avoids expensive merge operations by keeping native timebases and filtering with PyArrow compute
- **Grid-based zone averaging:** `zones.average_zones_across_laps()` uses a uniform grid with voting across laps rather than point-by-point comparison
- **GPS smoothing:** Rolling averages on position (15 pts) and curvature (30 pts) to reduce GPS noise
- **Time-based accel zone merging:** Merges acceleration zones separated by short time gaps (gear changes)

## CI/CD

GitHub Actions workflows:
- **ci.yml** — Runs on all pushes/PRs to `master`: lint, typecheck, tests, coverage upload
- **deploy-preview.yml** — On push to `master`: builds Pyodide wheels + JupyterLite site, deploys to preview
- **deploy-release.yml** — On GitHub Release: same build, deploys to production

## Branches

- **master** — Main branch, target for PRs
- Development branches are feature-specific
