# CLAUDE.md

## Project Overview

**motorsports-data-notebook** is a Python toolkit for analyzing AIM motorsports telemetry data (XRK/XRZ files). It provides interactive Jupyter notebooks that run entirely in the browser via JupyterLite/Pyodide (Python compiled to WebAssembly). No server-side computation — all analysis happens locally.

Live site: https://analysis-preview.inferno.racing/

## Tech Stack

- **Python 3.12+** with **uv** for dependency management
- **just** as task runner
- **PyArrow** for efficient columnar data filtering (prefer over pandas where possible)
- **Plotly** for interactive visualizations (primary charting library)
- **libxrk** for parsing AIM telemetry files (XRK/XRZ)
- **JupyterLite + Pyodide** for browser-based deployment

## Repository Structure

```
src/motorsports_data_notebook/   # Main package
  channels.py      # Lap selection (get_best_lap, get_top_laps, get_best_lap_channels)
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
uv sync --extra app                        # app environment (jupyterlab, etc.)
uv sync --extra app --group dev            # + dev tools (black, mypy, pytest)
uv sync --extra app --extra jupyterlite    # + JupyterLite build tools

# Run all checks (lint + typecheck + tests)
just check

# Individual checks
just lint                  # black --check src/
just typecheck             # mypy src/
just test                  # pytest with coverage
just lint-notebooks        # black --check on notebooks
just typecheck-notebooks   # mypy on notebooks

# Format code
just format               # black src/
just format-notebooks      # black on notebooks

# Run locally
just run-clean             # Reset workspace + start JupyterLab

# Build JupyterLite site
just build-lite-full       # Rebuild project wheel + build site
just build-and-serve-lite  # build-lite-full + serve locally

# Emscripten SDK setup (one-time, needed for rebuilding libxrk Pyodide wheel)
just setup-emsdk           # Installs emsdk to ~/emsdk
just build-libxrk-pyodide  # Rebuild libxrk for Pyodide (requires setup-emsdk)
```

## Bash Command Conventions

When running shell commands:
- **Always use absolute paths** or combine `cd` with the command in the same invocation
- Bad: `./build_wine.sh` (assumes correct working directory)
- Good: `cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app && ./build_wine.sh`
- Good: `/home/m3rlin45/code/motorsports_data_notebook/desktop_app/build_wine.sh`

## Desktop App (Inferno Analyzer)

The `desktop_app/` directory contains a standalone CustomTkinter GUI application that combines suspension velocity histogram analysis and driver consistency analysis (throttle acceptance & braking points) into a single tabbed interface.

```
desktop_app/
  src/inferno_analyzer/
    app.py                      # InfernoAnalyzerApp (shared sessions + CTkTabview)
    tabs/
      base_tab.py               # BaseAnalysisTab (debounce, threading, stats window)
      suspension_tab.py          # SuspensionTab
      driver_tab.py              # DriverTab
    suspension/                  # Suspension velocity histogram analysis
      analysis/multi_lap.py
      widgets/{chart_view,config_panel,stats_panel}.py
      visualization/comparison.py
    driver/                      # Driver consistency analysis
      analysis/driver_consistency.py
      widgets/{chart_view,config_panel,corner_selector,stats_panel}.py
      visualization/{corner_detail,corner_summary,track_map}.py
```

### Running locally

```bash
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app && uv sync && python -m inferno_analyzer
```

### Building Windows exe (via Wine)

```bash
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app && ./build_wine.sh
```

Run on Windows from WSL:
```bash
powershell.exe -Command "Start-Process '$(wslpath -w /home/m3rlin45/code/motorsports_data_notebook/desktop_app/dist/InfernoAnalyzer.exe)'"
```

### Copying exe to Windows Desktop

Use PowerShell's `GetFolderPath` macro to resolve the desktop (handles OneDrive redirection):
```bash
DESKTOP=$(powershell.exe -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r') && cp /path/to/App.exe "$(wslpath "$DESKTOP")/"
```

### Building Linux Native Binary

Requires `python3-tk` system package:
```bash
sudo apt install python3-tk  # Ubuntu/Debian
```

Build:
```bash
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app && uv sync --extra build && uv run pyinstaller --clean --noconfirm inferno_analyzer.spec
```

Run (displays via WSLg on WSL2):
```bash
/home/m3rlin45/code/motorsports_data_notebook/desktop_app/dist/InfernoAnalyzer
```

## Rust Desktop App (desktop_app_rs/)

Rust rewrite of the Inferno Analyzer using egui. Three workspace crates:

```
desktop_app_rs/
  crates/
    inferno-core/   # Data loading, analysis, profiles (no GUI)
    inferno-ui/     # egui widgets, charts, theme
    inferno-app/    # eframe application (InfernoApp), main entry point
      src/lib.rs    # InfernoApp struct + eframe::App impl
      src/main.rs   # Thin entry point
      tests/
        snapshot_tests.rs       # Visual regression tests
        snapshots/*.png         # Golden snapshot images
```

### Running and building

```bash
# Run locally
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app_rs && cargo run

# Run all tests (including snapshot tests)
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app_rs && cargo test

# Clippy
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app_rs && cargo clippy
```

### Snapshot tests (egui_kittest)

Visual regression tests render the app headlessly via `egui_kittest` and compare against golden PNG snapshots in `crates/inferno-app/tests/snapshots/`. Tests load real XRZ data, run analysis synchronously via `InfernoApp::load_and_analyze()`, then snapshot the summary, detail, and track map views in both dark mode (app default) and light mode.

```bash
# Run snapshot tests
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app_rs && cargo test --test snapshot_tests

# Update golden snapshots after intentional UI changes
cd /home/m3rlin45/code/motorsports_data_notebook/desktop_app_rs && UPDATE_SNAPSHOTS=1 cargo test --test snapshot_tests
```

**When modifying UI code in inferno-ui or inferno-app:**
1. Run `cargo test --test snapshot_tests` — if tests fail, the UI rendering changed
2. If the change is intentional, regenerate snapshots with `UPDATE_SNAPSHOTS=1`
3. Visually verify the new snapshots look correct (`wslview crates/inferno-app/tests/snapshots/*.png`)
4. Commit the updated `.png` files alongside the code changes

**Adding new snapshot tests:** Add test functions in `crates/inferno-app/tests/snapshot_tests.rs`. Set `corner_selector.view_mode` to control which view is rendered. Use `harness.try_snapshot("name")` — the golden file is saved to `tests/snapshots/{name}.png`. Use `create_app_with_data()` for dark mode and `create_app_light()` for light mode. Always add tests for both themes when adding a new view.

## Code Style & Conventions

- **Formatter:** Black, 100-char line length, target py312
- **Type checker:** mypy (check_untyped_defs=true, disallow_untyped_defs=false)
- **Docstrings:** NumPy-style with Parameters/Returns/Raises/Examples sections
- **Type hints:** Use `from __future__ import annotations`, PEP 604 unions (`X | None`), `TYPE_CHECKING` blocks for import-only types
- **Data handling:** Prefer PyArrow tables and `pyarrow.compute` (pc) operations for filtering/transforms over pandas. Convert to pandas/numpy only for analysis steps that require it.
- **Channel names:** Passed as dictionaries (`channel_names["throttle"]` -> `"PPS"`) to support different AIM telemetry formats. Use `_validate_channel_names()` for validation. When adding new channel keys to `DEFAULT_CHANNEL_NAMES` in `profiles.py`, also add the corresponding entries to each vehicle profile in `src/motorsports_data_notebook/data/builtin_profiles.yaml` (channel names vary per vehicle/logger).
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
- **libxrk 0.5.0 method chaining:** Use `log.filter_by_lap(n).select_channels([...]).resample_to_channel(ref).channels` for efficient lap filtering and channel alignment
- **Single-lap functions receive pre-filtered LogFile:** Functions like `get_corner_data()` and `analyze_suspension_velocity()` expect the caller to filter via `log.filter_by_lap(n)`
- **Multi-lap functions filter internally:** Functions like `detect_zones_averaged()` and `compute_segment_stats()` loop over laps and call `log.filter_by_lap()` internally
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

## Notebook Internationalization

### Directory Structure

Notebooks are organized by language under `workspace_template/`:
```
workspace_template/
├── data/           # Shared sample data files (.xrz)
├── en/             # English notebooks (source of truth)
└── ja/             # Japanese translations
```

### Translation Workflow

English notebooks in `en/` are the source of truth. When English notebooks change:

1. **Identify changes**: Compare the modified English notebook with its Japanese counterpart
2. **Update Japanese version**: Translate only the changed content:
   - Markdown cells (headers, explanations, bullet points, tables)
   - Code comments (lines starting with `#`)
   - Print statements and f-string user-facing text
   - Plot titles and axis labels
3. **Do NOT translate**:
   - Variable/function/class names
   - Channel names (`GPS Latitude`, `BrakePress`, `PPS`, etc.)
   - File paths
   - Python keywords
   - DataFrame column names used in code (keep English for code compatibility)

### Translation Glossary

Consistent terminology mapping for motorsports and technical terms:

| English | Japanese | Notes |
|---------|----------|-------|
| lap | ラップ | |
| lap time | ラップタイム | |
| best lap | ベストラップ | |
| top laps / fastest laps | 上位ラップ | Plural, not single fastest |
| corner | コーナー | |
| apex | エイペックス | Keep katakana |
| braking zone | ブレーキングゾーン | |
| acceleration zone | 加速ゾーン | |
| throttle | スロットル | |
| brake pressure | ブレーキ圧 | |
| steering | ステアリング | |
| suspension | サスペンション | |
| damper | ダンパー | |
| shock pot | ショックポット | |
| bump (compression) | バンプ（圧縮） | |
| rebound (extension) | リバウンド（伸長） | |
| motion ratio | モーションレシオ | |
| spring rate | スプリングレート | |
| camber | キャンバー | |
| tire temperature | タイヤ温度 | |
| lateral G | 横G | |
| inline G / longitudinal G | 縦G | |
| throttle acceptance | アクセル踏み込み点 | Include English/katakana reference: (英語: Throttle Acceptance / スロットルアクセプタンス) |
| consistency | 一貫性 | |
| deviation | 偏差 | |
| standard deviation | 標準偏差 | |
| histogram | ヒストグラム | |
| heatmap | ヒートマップ | |
| overlay | オーバーレイ | |
| segment | セグメント | |
| channel | チャンネル | |
| upload | アップロード | |
| widget | ウィジェット | |
| timebase | 時間軸 | Data alignment reference |
| friction (suspension) | 摩擦 | Use native Japanese, not フリクション |

### User Correction Workflow

When users provide translation corrections:

1. **Update the glossary** above if the correction affects a standard term
2. **Apply the correction** to all Japanese notebooks consistently
3. **Document the change** in commit message with rationale

Example correction flow:
- User says: "スロットル開度 is more natural than スロットル for throttle position"
- Update glossary: `throttle (position)` → `スロットル開度`
- Search and replace in all `ja/*.ipynb` files
- Commit: "Update throttle translation to スロットル開度 per user feedback"
