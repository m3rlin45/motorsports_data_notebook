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
tire_pressure_calculator/ # C# Avalonia cold tire pressure calculator (.NET 10)
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
```

Note: libxrk/libibt wasm wheels come from PyPI (resolved by micropip in the
browser) — there is no local cross-compilation step.

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

## C# Desktop Analyzer (desktop_app_cs/)

The `desktop_app_cs/` directory is the .NET 10 / Avalonia 12 rebuild of the desktop analyzer (plan and status in `desktop_app_cs/PLAN.md` — read it first). It reads XRK/XRZ/IBT files through C-ABI builds of the Rust parsers (`libxrk-ffi`, `libibt-ffi`, Arrow C Data Interface) and ports the Python kernels with per-module parity fixtures.

```
desktop_app_cs/
  Core/              # InfernoAnalyzer.Core — Io (FFI readers, LogFile), Profiles, Tracks, Analysis kernels; no UI deps
  Charting/          # ChartSpec records, ChartTheme + ChartColors tokens, IChartSurface, CursorLinkGroup; no chart-lib deps
  Charting.ScottPlot/ # the ScottPlot 5 backend: ScottPlotSurface<TSpec> bases, PlotStyle/MapStyle, one surface per spec kind
  App/               # Avalonia MVVM app: ViewModels (one per tab), Views, Controls (ChartHost, custom-drawn split table + track editor), Styles/AppStyles.axaml, Settings
  Tests/             # xunit v3 (AOT flavour) + Avalonia.Headless; Fixtures/*.json dumped by tools/extract_*_fixture.py
  tools/             # fixture extractors (uv run), build_native.py (builds the parser FFI libs at the libxrk/libibt versions `uv.lock` pins), make_icon.py
  native/runtimes/   # built FFI libraries (git-ignored), copied into Core's output
  native/static/     # the same as static libraries (host builds only), linked into a NativeAOT publish
```

```bash
just cs-native                 # build libxrk-ffi/libibt-ffi for the host (needs cargo); add --rids win-x64 for the cross-build
just cs-build / cs-test / cs-format
just cs-run FILE               # run the app, optionally opening a session file
just cs-kernel-fixture MODULE  # regenerate a parity fixture from the Python library
just cs-test-aot               # the tests as a NativeAOT executable (the shipping mode; CI runs this on Windows and Linux)
just cs-publish-linux          # NativeAOT binary;  just cs-publish-win = trimmed single-file exe for a quick look
```

The app ships as one NativeAOT executable: the parsers are linked in statically (`Directory.Build.targets`), and with `-p:InfernoSelfExtract=true` the graphics libraries that exist as shared libraries only (SkiaSharp, HarfBuzzSharp, ANGLE on Windows) are embedded and unpacked at start-up to a per-user cache folder (`App/NativeBundle.cs`; `%LOCALAPPDATA%\InfernoAnalyzer\native\<hash>`). The Windows one can only be built on Windows, so CI builds it (`build-analyzer-cs.yml`: AOT-compiled tests, publish, `tools/check_publish.py`, a start-up check). On a published GitHub release the same workflow attaches the two executables as `InfernoAnalyzer-CS.exe` and `InfernoAnalyzer-CS-Linux` (the release's `InfernoAnalyzer.exe` and `InfernoAnalyzer-Linux` are the Python desktop app's, from `build-desktop.yml`). The test project is itself NativeAOT-publishable, which rules out reflection-based test extensions: UI tests are plain `[Fact]`s whose body runs through `Ui.Run(...)` (the headless Avalonia app's UI thread, a fresh application per test), not `[AvaloniaFact]`. Tests that drive view models go through `Ui.Run` too, so their continuations run on one thread as in the app. Save captured frames with `RenderSmokeTests.SaveRender`.

Threads: on Windows Avalonia composes frames on its own thread, and ScottPlot's stock control (`AvaPlot`) runs ScottPlot's whole render there. Charts use `Charting.ScottPlot/UiThreadPlot` instead, never `AvaPlot` directly: ScottPlot renders on the UI thread into a recorded picture and the compositor only replays it, so plots are touched by one thread and ScottPlot's render events (`RenderManager.AxisLimitsChanged` and the like) arrive on the UI thread. A handler of such an event still runs inside a draw, so it posts its work (`ScottPlotStripChart.QueueLimitsChanged`) rather than changing plots or invalidating controls there. The headless tests draw everything on the UI thread and cannot see a threading mistake by themselves; `StripChartInteractionTests` covers it by counting ScottPlot renders across a composed frame and by rendering a plot from another thread.

No swallowed exceptions: Avalonia reports failures inside it (an exception while a frame is drawn, for one) only to its logger and carries on. `App/AvaloniaLog.cs` is that logger's sink: an error that carries an exception is thrown again from where Avalonia caught it (`AvaloniaCaughtException`), which on the UI thread reaches the dispatcher's handler and anywhere else ends the process with the exception in `analyzer_crash.log` and a message box; other errors and warnings are written to the log. The one exception is rendering-backend selection (the `OpenGL`, `Vulkan` and `*Platform` log areas): Avalonia tries GLX/EGL/ANGLE/swap-chain kinds in turn and catches the unavailable ones, so those go to `analyzer_platform.log` (rewritten at each start) and are not thrown, or the app could not start on a machine without a usable GPU (WSLg, for one). `Ui.Run` fails a test on any error logged while it runs. Do not add catch-and-continue code of our own; `CrashLog` is the one place that decides what an unhandled exception does. `App/HangWatch.cs` logs the UI thread's native stack when it stops responding (Windows; decode with the `InfernoAnalyzer-Windows-symbols` CI artifact and `llvm-symbolizer --relative-address`).

Conventions:

- **Kernels** match numpy operation order before tolerances are loosened (pairwise sums, `x * (π/180)`, numpy `_lerp`, float32 round-trips). Multi-lap kernels take an optional `CancellationToken` checked between laps.
- **Charting** code never leaks ScottPlot types above `Charting.ScottPlot`. Specs name colours by `ChartColors` token, never a hex literal, so they draw in light, dark and high-contrast themes; a backend derives from `ScottPlotSurface<TSpec>` (or `ScottPlotGridSurface<TSpec>`) and styles plots through `PlotStyle` / `MapStyle` only. Re-run the dataviz palette validator after touching the lap hues in `ChartColors`.
- **Tabs** derive from `AnalysisTabViewModel<TParameters, TResult>`: `GetParameters` snapshots everything (labels included) on the UI thread or returns `Hint("why not")`, `Analyze` computes from the snapshot alone on a worker, `Display` shows it; `ClearDisplay` drops the specs so the chart area shows the reason instead of a stale chart. Numbers typed by the user are `NumericFieldViewModel`s: an invalid one is a `Hint`, never a silent default. Read laps with `session.SelectedIn(Scope)`. The track regions ticked in the sidebar (`RegionsViewModel`, one choice for every tab, hosted by `RegionPanelView` under the lap lists) reach a tab through its base class: it overrides `RegionUse` (`Filter`: the tab shows only what lies inside the ticked regions — the statistical tabs cut the laps, the Driver tab lists their corners; `Focus`: Time/Distance keeps whole laps, shades the regions and brings them into view; `None`: the selector is dimmed on that tab), puts `RegionSnapshot()` in its parameters and `selection.Windows(log, laps, slot)` into the kernel's `windowsByLap`, sets `RegionLabel` in `Display` (cleared in `ClearDisplay`) and shows `RegionTagView` beside its title; the chart's `ChartHost.Caption` is bound to `ImageCaption` so a copied image says what it covers. A kernel cuts to the windows after any smoothing or differentiation over the whole lap. Corners are detected once for the region list and the Driver tab (`DriverConsistencyCache`).
- **Views** use the classes and tokens in `App/Styles/AppStyles.axaml` (no inline font sizes, paddings or colours) and host charts in `ChartHost`. A new tab adds a case to `TabViewLocator` (and `ConfigViewLocator`, `ChartSurfaces.CreateFor`) and lays its toolbar out in the common order: title, parameters, view switch, Channels…, Statistics.
- `builtin_profiles.yaml` is embedded from `src/` so profile changes need no C# edits.
- PR screenshots of a view are the headless render tests' PNGs (`Tests/bin/Debug/net10.0/renders/`), taken at the commit that adds the view and published as described under PR Screenshots.

## Tire Pressure Calculator

The `tire_pressure_calculator/` directory contains the cold tire pressure calculator: a manual Gay-Lussac mode (four FL/FR/RL/RR quadrants: current temp °C + target hot temp °C + target hot pressure bar → cold gauge pressure to set) and a Circuit Prediction mode that predicts per-corner hot temps from the fitted tire warmup model (`data/tire_dataset/tire_model.json`), with inputs for track, car, tire compound (forced per-car choice — one tire on all four corners), condition, lap, ambient, cloud cover, and target lap time (m:ss.s text), plus prefill of the corner-card targets (hot temp from the model's per-car medians; hot pressure from the team table `data/tire_dataset/hot_pressure_targets.yaml` per car + compound, copied into `tire_model.json`). Circuit Prediction is the default mode. In that mode a Preview button beside the mode toggle (Setup while previewing) swaps the four cards' fields for pressure-over-stint charts (`web/js/chart.js`, data from `predictCornerTrajectory` in `model.js`): time from pit exit on x with a tick per lap, gauge pressure on y, the target as a dashed line, the viewport showing 1.5 × the target lap and scrolling on to five times it; the web app only so far. Corner targets are remembered per configuration (car + compound + condition, `TargetMemory` in the settings JSON): a configuration is prefilled the first time it is opened and after Reset, otherwise the targets the user left for it come back. Two implementations share the same behavior, strings, and settings JSON shape:

```
tire_pressure_calculator/
  Core/           # C# Avalonia shared UI + modeling (.NET 10, Avalonia 11.3)
    Services/Modeling/   # EnergyBalance, TireModel, loader — C# port of tire_model/predict.py
    Services/CircuitPredictor.cs
    ViewModels/ Views/ Localization/
  Desktop/        # Windows/Linux desktop head
  Android/        # Android head
  Tests/          # xunit tests incl. Python-parity fixture (Fixtures/python_predictions.json)
  web/            # Static web app (plain HTML/CSS/JS, no build step) — deployed to tires.inferno.racing
    js/model.js   # JS port of the modeling layer (pinned to the same parity fixture)
    js/strings.js # Must stay in sync with Core/Localization/strings.json (tested)
    js/app.js     # DOM wiring, localStorage settings (same key/shape as C# AppSettings)
    sw.js         # Service worker (installable PWA, offline); cache version stamped at deploy
    tests/        # node --test suites
```

The web app fetches `tire_model.json` at runtime; deploy workflows copy it from `data/tire_dataset/` next to `index.html`. Retraining the model (`just tire-build-warmup-table`) + redeploy is enough to update it.

### Web app

```bash
just tire-web-test    # node --test parity + unit tests
just tire-web-serve   # serve locally, open http://localhost:8080/tire_pressure_calculator/web/
```

### Building the .NET heads

```bash
# Dev run (requires display - WSLg or X11):
cd /home/m3rlin45/code/motorsports_data_notebook/tire_pressure_calculator && dotnet run --project Desktop

# Windows single-file exe (cross-compiled from Linux, trimmed, not AOT):
cd /home/m3rlin45/code/motorsports_data_notebook/tire_pressure_calculator && \
dotnet publish Desktop/TirePressureCalculator.Desktop.csproj -c Release -r win-x64 --self-contained \
  -p:PublishAot=false -p:PublishTrimmed=true -p:TrimMode=full \
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true

# Linux AOT native binary (requires clang, zlib1g-dev):
cd /home/m3rlin45/code/motorsports_data_notebook/tire_pressure_calculator && \
dotnet publish Desktop/TirePressureCalculator.Desktop.csproj -c Release -r linux-x64 --self-contained
```

Note: Cross-OS NativeAOT is not supported. Windows builds from Linux use trimmed single-file instead. NativeAOT works for linux-x64 when building on Linux.

## Tire Dataset (tire_etl / tire_model)

`src/motorsports_data_notebook/tire_etl/` extracts the AIM sessions into `data/tire_dataset/` (sessions, laps, timeseries, notes, weather; see `data/tire_dataset/README.md` for the layout and the delta workflow); `src/motorsports_data_notebook/tire_model/` trains and validates the cold tire pressure model from it. Design rules, in priority order:

- **The extractor does the minimum.** Lap/stint splitting, GPS track and layout resolution, channel selection and renaming, notes/weather enrichment. Nothing else. An extractor change means a version bump and a full re-extract (20+ min and a rewrite of every timeseries file), so anything that can be a code change plus a retrain must not live here.
- **Raw samples are stored verbatim at their native rate.** Every selected channel keeps its own raw timecode array from the file: no downsampling, no alignment onto another channel's clock, no interpolation, no dropped or filled samples. Values are stored at the narrowest float width that holds them exactly (float16 where the logger logged float16, float32 otherwise, float64 for positions); the cast is exact by construction, never a rounding. The raw files do not share timebases even within a sensor family (each TPMS corner is its own timebase; the KK-SII dash logs its accelerometer as pairs of samples 2 ms apart on 50 Hz frames, 100 Hz in bursts; the KK-F's `YawRate` is separate from `RollRate`/`PitchRate`), so a storage rule must never be allowed to shift a sample to make channels line up.
- **Resampling and merging happen only at query time.** Putting channels on one clock (lateral g, longitudinal g and speed for the heat rate; IMU candidates onto the GPS clock for validation; TPMS onto the 1 s rolling clock) is the job of the loaders in `tire_etl/dataset.py` and of the consumer, never of the extractor.
- **Cleaning happens at training time.** Stale-TPMS masking, usability filters and any future data-quality rule live in `tire_model/cleaning.py` (tests in `tests/tire_model/test_cleaning.py`) and run when the model loads the data. Fix a data problem there and retrain; do not re-extract. `just tire-clean-laps` materialises the cleaned laps table for queries.
- **The car is what the file says, unless `session_overrides.yaml` says otherwise.** The KK-SII, the KK-F and the KK-F-V2 share one SW4 wheel and the vehicle name on it is changed by hand, so a recording can name the wrong car (its channel set is then the other car's). Such files are listed in `data/tire_dataset/session_overrides.yaml` with the car really recorded and the evidence; the extractor applies it before grouping (`tire_etl/overrides.py`), so the session id, the vehicle profile and `sessions.car` follow the correction and `sessions.car_declared` keeps what the file says. Editing the file re-extracts the affected recordings; never fix a car label anywhere else.
- **The FJ's constants are pooled; its logged entries are per raw car** (#383, on by the owner on 2026-11-26). `CAR_FIT_ALIASES` pools the KK-SII, the KK-F and the KK-F-V2 for K, tau, kappa, the shares, the compound K and the corner defaults. The fit also writes each entry (heat per tyre with its pace curve, typical lap time, out-lap) per raw car in `*_by_track_raw_car_cond` beside the pooled tables (`warmup_table.RAW_CAR_ENTRIES`), only where the raw car has at least two training sessions and a pace curve there (the guard, rule (b), `RAW_CAR_MIN_SESSIONS`, `RAW_CAR_NEEDS_CURVE`); elsewhere it reads the pooled entry. Every predictor (`predict._entry_row`, `model.js` `entryRow`, `TireModel.EntryRow`) reads a raw car's own row before the pooled car's at each step of the condition chain; the track and global fallbacks average the pooled rows only. A simulated entry placed for a pooled car answers its raw cars too (`pressure.place`). With the tables in the artifact the calculators' car picker lists the raw cars in place of the FJ (a saved FJ keeps working); the constants and prefills resolve a raw name to the FJ. Measured in docs §2.5a.
- **Sensor selection is the one ETL verdict.** Which accelerometer/gyro series to trust (IMU vs GPS-derived, issue #203) is decided by the extractor and recorded as metadata: one verdict per session for whether the sensor is sound (including `imu_swapped`, where the loader hands the IMU's two horizontal channels out exchanged), plus a source for every 20 s window of the run — `imu`, `gps` (the IMU has no samples or is frozen there) or `discard` (no usable sensor: the GPS fallback has a spike and there is no IMU behind it). The raw channels are still stored unmodified — a discarded window keeps every sample, marked by the sparse `discard` column of the timeseries (true on its rows, null elsewhere) — and the loader (`kinematics.load_kinematics`) applies both. Consumers train on windows, not whole sessions: a discarded window is a hole in the heat input, never a value — per-lap summaries of a lap with a hole are NaN, and the per-second fit restarts from the measured state after it.

## Lap Simulation (tire_model/lapsim)

Exploration code for the simulated-lap heat source (issues #202–#208); the production pressure model does not read it. Reports and caches go to `workspace/lapsim/` (not committed), so every gate's numbers are also written into `docs/tire_model.md` §2.12. The tracks are committed, in `data/lapsim/tracks/` (below). Run the gates with `just tire-lapsim <command>` under `ulimit -v 10000000`.

- **Nothing in the lap simulation goes into the extractor.** Geometry is computed per session at query time (`lapsim/geometry.py`); `tire_etl/kinematics.position_kinematics` and `layouts._local_xy` are the reference the IMU verdicts were scored against and are used read-only.
- **Course comes from the Doppler velocity, not from position.** The stored `gps_yaw_rate_dps` is libxrk's one-sample difference of the GPS course, so its running sum is the course; curvature is one derivative of that. Position gives the frame, the lap boundaries and the check.
- **Everything is on the position clock.** GPS velocity trails GPS position by 65–440 ms depending on the logger; `geometry.session_geometry` estimates the lag per session and hands out speed and course at `t + lag`. Take V from there, not from `speed_ms` directly, whenever it is paired with position.
- **Metric frame.** `geometry.local_xy` (WGS84 ellipsoid, origin at the layout's start/finish gate) for anything whose scale matters; the extractor's sphere projection is 0.2 % anisotropic.
- **The accelerometer is late too.** It trails GPS position by about 120 ms on the SW4, 50–60 ms on the Inferno 86's loggers and 30 ms on the others (`track.accel_delay`). Shift it onto the position clock before comparing it with anything from GPS at better than a few hundredths of a g.
- **A track is `track.build_track_model(root, layout_key)`**: laps projected onto a reference line within an 80 m window (Suzuka crosses itself), κ from the per-bin median of the Doppler course over 10 m, positive in a right turn. A lap does not end where it began (up to 10 m to the side on a wide straight), so the first reference line is open at the start line.
- **Compare tracks profile against profile, through the heat chain** (`holdout.compare_profiles`): two curvature profiles at one speed trace through `share_heat_rate`, as the log ratio of lap-mean heat per tyre. The day-to-day floor of that number is 0.003–0.012 depending on the layout; a difference between cars means something only beside it, and only with three days and thirty laps behind each side.
- **Laps keep the dataset's `lap_num`**; gate crossings only place `s = 0`. `geometry.clean_lap_index(root)` is the one place the clean-lap and build-lap verdicts are computed. The dry build takes laps on pace alone and does not use the wet label: the road is the same wet or dry, and a wet session is slow.
- **Wet days are a separate basis, and a layout keeps a wet track beside the dry one where it has the laps** (`build_track_model(..., basis="wet")`, 30 laps on 3 days, every car pooled). The line driven in the wet is 1–3 m from the dry line with apexes 4–9 % tighter. "Fast" in the wet is judged among wet sessions only: labelled wet, off the dry pace, then the same 1.05 × and 1.02 × rules against the car's wet reference.
- **The car model is in the ground frame** (`frame.ground_series`: `V·Δψ/Δt` and `ΔV/Δt` over 0.5 s on the position clock). What a session's accelerometer reads of it is a per-session measurement (`frame.session_frame`: lateral slope and zero, longitudinal zero), never a per-car constant: the slope belongs to the layout and the zero steps between days. Heat for any comparison is production's chain on one lap, `heat.bin_1s` + `heat.lap_heat`, the seconds counted from the lap's start. A simulated lap goes through it by `heat.sim_heat` (an even 100 Hz clock, the tyres' lateral, the longitudinal an accelerometer reads). Gate D1 (`check-heat-chain`, `heat_gate.py`) holds it to production's own per-lap heat on the laps production fits: 0.02 % in the mean, 0.4–0.5 % lap by lap, which is the phase of the bins. Gate D3 (lateral heat simulated over logged, and right turns less left) is two more measures in gate C3's report in the same attainment form, the logged lap read by its own lateral at the track's nodes (`CarTables.lateral`, tables version 8): 22 of 38 cells held out by day and 21 of 38 by layout; no sign or mirror defect at the four layouts where one would show (a mirror reads 0.7–1.7 by side there, 0.043 at most is read), and nearly blind at Suzuka, whose two sides carry alike (a mirror reads 0.15–0.17); the simulated lap has 0.5–1.6 % less lateral heat than the logged one, most of which is the logged lateral on the straights (noise through the absolute value: 0.6–1.0 % of a lap's lateral heat, which two logged laps share and a simulated lap lacks), and the right turns carry 1–3 % less of the simulated heat than of the logged in seven of nine cells, a lean common to the layouts that is in the simulated speed and not in the track or the bank (the same with every road flat): the simulated apex speed is 1.4 % lower in right turns than left ones against the logged car, which on the formula cars is consistent with the logged cars holding more lateral in right turns (the limit's statistic by side: +4.5 % on the KK-SII, +2.7 % on the KK-F) while the simulation has one limit both ways; the Inferno 86's lean (its sides differ by 0.6 %) is not accounted for. The owner (2026-11-24): the formula cars are symmetrical, so it is the corners; nothing is fitted by side, and the higher-side reading of the limit (+2.7 % on the KK-SII) is the owner's to decide. **Where a table's heat has to meet production's τ and K it is multiplied by the frame factor** (since #376 only in the sensor's frame, `--heat-frame sensor`: production is fitted in the tyres' frame and the factor is 1; `heat_gate.frame_factor`, in `check-heat-chain`'s report: per production car, layout and source, production's chain on the sensor's series over the same logged lap in the tyres' frame; about 1 on the formula cars' accelerometer at Suzuka and Tsukuba (their rears at Fuji read 0.93–0.94, unexplained), 1.02–1.04 on the Inferno 86's, 1.06–1.10 on a GPS-derived series; it leaves 1–2 % per session in the median). **One chain on both sides of a comparison**: smoothing the series over 0.5 s first (as #207's E0 would) lowers a simulated or an accelerometer lap's heat by 0.6–1.3 % and a GPS-derived one's by 3–5 %, so a table made by one chain cannot be read against laps put through the other.
- **A car's numbers come from its sample tables** (`samples.car_tables`: full throttle, coast, braking zones, apexes, on the dry and the wet build laps), with the track's grade taken out. Throttle and brake are read relative to the session's own range (full = top 5 % of closed-to-open, brake on = 5 % of the hardest application), so no threshold is written per car, logger or unit; channels logged on change hold their last value.
- **`mu_y` is the car's cornering limit, and the solver's knob is the driver's use of it** (#312, built 2026-11-23; `limits.car_limit`, `car_gate.set_limit`; docs §2.12 C9). The limit is the best the car has been shown to do: per corner the expected best of m = 20 passes of the lateral the tyres hold over the load on them (two readings a pass, tables' `lat_tyre_g` and `load`), the median over the corners with 2m passes; it is `grip.mu_y_limit`, read without the fold in every fold, and no pace moves it. The knob u is the share of it a lap uses, in cornering and braking alike (q = −ln u; under 1 on every logged lap); a car's *typical use* (`meta.use_ref`, the median over its layouts of the use of a day's best lap: 0.918 / 0.878 / 0.936 with every term of #312 in) is what is carried to a layout it has not run. Reports keep `u` as the pace against the typical use and add `use`. The step changed no simulated lap (1,102 held-out laps within 0.00004 m/s at every node; 56 of 76 cell for cell): do not describe u as grip. Three things to know: (1) a driver's braking share is of `mu_b` at a use of 1 and rose as `mu_y` rose (0.87 / 0.97 / 0.90: the KK-SII has little room left; the best braking held, `limits.braking_best`, reads 0.944 / 1.210 / 1.099, at or under `mu_b`, so it is reported in gate C2 and `mu_b` stays the stops' peak level by the owner's decision of 2026-11-24); (2) a car with no driver entry brakes at `mu_b` times the use, so "braked at the limit" as the gate's comparison arm means `mu_b` at the typical use (`car_holdout.convene` holds it there); (3) **the limit's reading is biased upward by an amount the logs bound**: read as each sensor's own peak in the corner (the committee's reading, the default) the largest of noisy fixes is raised by the noise; read where the *other* sensor peaks (`lat_*_cross_g`, `car_limit(..., cross=True)`) it is lowered by however far apart the peaks are, so the truth lies between (1.283 / 1.541 / 1.493 against 1.249 / 1.524 / 1.466: 2.7 / 1.1 / 1.8 % apart). On a planted car (`car_gate.plant_limit`) the own-peak reading fails the rule written for it (+1.8 to +3.6 %) and the cross reading is within about 1 %, but the plant overstates: it predicts a gap between the readings 1.3–3 times the logs' (a point mass holds its lateral through the whole corner). The owner kept the own peaks (2026-11-24: the reading on which the max driver is faster than every best lap; it may read up to that gap high); no simulated lap depends on it, only the max driver's margins (faster than the best lap in ten of ten cells on own peaks, thinnest the Inferno 86 at Fuji at 0.60 %; nine of ten on the cross reading, where that cell is 0.05 % slower). **Drive out of a corner goes through the driven axle** (`SolverCar.drive_share`, `solver.DRIVEN_AXLE_DEFAULT`; on for all three cars by the owner's decision of 2026-11-23): the grip left for drive is the rear's share of the static weight (production's `CAR_FACTS`: 0.47 / 0.58, stated, not fitted) times what the cornering leaves. The share rides in the last entry of the car's power table (no speed reaches it), so neither solver form gained an argument and a share of 1 is the old arithmetic to the last bit. With it 59 of 76 (56 without; 5 cells won, 2 lost, all on the formula cars: inside chance) and heat at an unseen layout 2.6 % against 3.0 %; the typical use rises to 0.916 / 0.877 / 0.935. **The fit reads its laterals in the tyres' frame** (`car.TYRES_FRAME_READS`: `k_corner_drag`, the straight-line stops and the braking shape from the `*_tyre_g` columns; 60 of 76). A model without a limit (no corner supports the statistic) is still set by the old convention (`car_gate.set_convention`): the value at which the median over a car's layouts of the reference lap's u is 1. That convention's value is limit × typical use, and it sits 13–21 % under the apex table's p90 with the laps driven on the road's bank and braked as the driver brakes (16–24 % braked at the limit, 11–14 % on flat roads, 9–10 % before the braking limit had its shape): the table is in the ground frame, where the bank's help counts as the car's. `mu_b`, power and drag are measurements and are not moved by it.
- **A car is a chassis with powertrains; how it is driven is not the car** (owner's decision, 2026-11-15). Drag, cornering drag, braking and lateral level are shared by a car's raw cars; an engine change is a new powertrain of the same car (`car.FAMILY`, `car.POWERTRAINS`: the KK-F-V2 under the KK-F; the Inferno 86's second engine from 2026-06-01, with its 4.1 final drive), each with its power against rpm, its gears and its `scale` against the car's first. The driver's shift point is read per day from the logs (`car.shift_rpm_by_day`) and goes to the solver beside the pace knob; a new shift light is not a new car. Never split a car for a change in how it is driven.
- **A gear is made from speed over rpm where the logger has none, at query time** (`car.with_gears`: a day's speed/rpm values fall into one cluster per gear; never in the extractor, never replacing a logged gear). It is numbered among the gears seen, not as the gearbox numbers them. **The gear set is the day's setup** (`car.gear_sets_by_day`, `car.complete_sets`): the KK-SII has per-circuit gearing (two sets; its first Tsukuba day ran the Suzuka gears), so a day's set goes to the solver beside the shift point and is not a new car. A powertrain whose gear is inferred is still driven on its one `p_specific` (`solver.INFERRED_GEARS_DRIVE`): on the KK-SII the curve through the gears was worse on 19 of 22 held-out folds (candidate cause: the solver shifts in no time; the Inferno 86 loses 0.3–0.9 s of drive per upshift, the KK-SII 0.1 s, the KK-F none — measured in §2.12 D1, not yet a term).
- **The solver works on a batch of laps, in two forms that must stay identical** (`solver.solve_laps`, `solver.drive_laps`; a lap's own car and u per row). `lapsim/_tensor.py` is numpy with the laps as an axis (the reference, and what runs without numba); `lapsim/_kernel.py` is the same statements compiled with numba, one lane per lap across the cores. Change the two together: the tests hold node speeds, lap times, u and statuses to the last bit, and a speed-up is accepted only when the C2 and C3 reports come out unchanged. Give the gates batches (all the laps of a layout in one call), never a loop of single laps: one lap alone costs 0.5 s in numpy.
- **Do a thing once at the level it belongs to, and every line of an array at once.** What depends on the session (a sorted, filtered series; a speed on another clock) or on the track (the build laps' band) is computed there and a lap takes a slice of it; a per-row or per-column statistic goes through one call (`geometry.nan_median`, a single pandas grouped call), never a Python loop over groups. A rewrite of this kind is accepted only with the outputs identical: the gate reports, the track JSON, the cached tables and the fit's unrounded workings (§2.12 has the method and the timings).
- **Power is compared at equal rpm and in standard air.** A level read per speed bin mixes the engine with the gear the car was in and the air of the day. `car.air_state` gives each session its moist-air density and SAE J1349 factor; samples are brought to standard air before pooling and `car_gate.Drive` hands each lap its day's powertrain, shift point and air. Drag comes from the same rpm in different gears (`car.fit_drag_rpm`), not from an assumed constant power at the top of a straight.
- **A car is judged at each lap's own u, against a floor and against wrong cars** (`car_holdout`): a lap time alone cannot tell a car from a wrong one, because u is solved from it. A speed-trace error means something only beside the day floor (two logged laps of like lap time from different days), and a fold counts only if a planted wrong car (power × 0.85 with grip raised; another car's parameters) is outside the limits on it.
- **The engine gives `P/V − k·lat²`** (`k_corner_drag`, read from full-throttle samples above 0.5 g; `solver.CORNER_DRAG_DEFAULT`). It entered the default by the rule for any new term: lower error on the top quarter of speeds and no higher leave-one-layout-out error, here in ten of ten cells for both. Without it every car is 0.3–0.9 m/s fast at the end of the straights.
- **The braking limit has a measured shape and an upshift takes a measured time** (docs §2.12 C4; on by the owner's decision of 2026-11-16, `solver.BRAKE_SHAPE_DEFAULT`, `solver.SHIFT_LOSS_DEFAULT`). `grip.brake_ellipse` (`car.braking_shape`) is how far the limit under cornering lies from #206's diamond towards the ellipse (0.69 / 0.52 / 0.63); `mu_b` is the level of the zones braked in a straight line (`car.braking_level`), because that is the pair that gives a simulated car back. A powertrain's `shift_s` (`car.shift_times`) is the drive lost per upshift (0.25–0.6 s on the Inferno 86, under 0.1 s on the formula cars). With them the Inferno 86 is inside on 29 of 31 days and the braking point is within 4 m on two cars; the KK-SII is worse (S1 1.14 → 1.29, braking 11 m late) and neither term met the entry rule in every cell. Speed peak to speed peak the simulation is still 11–14 m late on all three: it goes from full drive to full braking at once, where the logs have 0.16–0.32 s from lift to brake and 0.2–0.36 s for the pressure to build (`grip.onset_s` from `car.brake_onset`; the solver cuts each speed peak flat for that long, `_kernel.open_peaks`; built and measured, `solver.BRAKE_ONSET_DEFAULT` off pending the owner: with it the Inferno 86 passes gate C3 on 31 of 31 days and the KK-F loses the KK-F-V2's day). Do not tune either to the gate: a lower share scored better and was not taken. In the solver a term must be arithmetic and square roots only (a blend, not a power): that is what keeps the numpy and the compiled form identical to the last bit, and a term must leave the lap time continuous in u (the shift time is a band of speed, not a place on the road).
- **The road's bank is the track's, read from the elevation model, and every lap is driven on it** (owner's decisions, 2026-11-17; docs §2.12 B2b). `bank.road_bank` reads the cross-slope of the GSI 5 m mesh over ± 5 m across the line, ± 10 m mean along it, bridges and sections that are not a plane filled from their neighbours; it is kept as `TrackModel.bank`, positive where the left side is higher, attached and reported by gate B2. With b the lean into the turn the tyres hold `V²|κ|·cos b − g·sin b` under the load `g·cos θ·cos b + V²(κ_v + |κ|·sin b)`, folded into the track's arrays by `solver._road` so that a flat road is the old arithmetic to the last bit. Full strength and nothing fitted: half and one-and-a-half strength both score worse. There is no default to set: a track with a bank is driven on it and a track without one (no tiles, `--no-dem`) is flat; `Profile.from_track(track, bank=False)` / `Drive.bank = False` exist for the comparison gate C3 reports. With the bank the heat error at a layout the car has not run is 3.7 % against 5.9 %; S1 is no better and Tsukuba is worse on every car, which is the car or the driver and not the road. Never carry a learned per-corner grip factor between cars: it mixes the track's bank with a line that belongs to the car class. **The tracks are committed with their bank, in `data/lapsim/tracks/`, with attribution to GSI** (owner, 2026-11-24: "commit the bank track info, add attribution, no flat stuff"; #208): gate B2 reads and writes them there, `data/lapsim/README.md` carries the source and the processed-content note GSI's terms ask for (出典：国土地理院ウェブサイト; 「地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成」), and anything shipped that is computed from a lap driven on them (a simulated entry in the calculators) carries the same attribution. The raw tiles stay in `workspace/lapsim/gsi/` and the tracks of gate B3's folds, which have no bank, in `workspace/lapsim/tracks/folds/`. No flat-road arm is run.
- **Heat is of what the tyres hold** (`bank.tyre_lateral_g`: `V²κ·cos b / g − sin b`, b the angle by which the left side is higher; `holdout.tyre_heat(..., bank)`, `SimLap.lat_tyre_g`). That is what an accelerometer on the road's plane reads and so what production's chain is given; `V²κ` is the ground-frame lateral and is more than the tyres hold wherever the road leans into the turn. Compare simulated with logged heat in one frame on both sides. Since version 5 the car's sample tables keep every lateral in both frames (`*_tyre_g` beside the ground-frame column, `bank.to_tyres_frame`) and an apex has the `load` on the tyres there, so that lateral over load is the share of the grip a pass used (#312). Since stage 9 of #312 the fit reads `k_corner_drag`, the straight-line stops and the braking shape from them (`car.TYRES_FRAME_READS`).
- **Braking is judged by how far the simulated deceleration is from the logged one** (owner's decisions, 2026-11-21; docs §2.12 C6): `car_holdout.braking_match`, the RMS difference in g, at the same places on the track, wherever the logged lap slows by 0.2 g or more, held out by day. Its yardstick is the same measure between two logged laps (`pace_yardsticks`). **Measures are read at three levels of pace, P75, P90 and P95** (`car_holdout.PACE_LEVELS`: the laps above that percentile of pace among the car's laps at the layout), so that a solution is seen to scale with pace; a lap is set against the other laps of its set. **Gate C3 is an attainment score, not a pass or a fail** (owner, 2026-11-21: the goal is to raise the number of passes): a cell is one car at one layout at one level on one measure (speed round the lap, the sizes of the end-of-straight and apex errors, braking), and it passes when the simulated lap is no further from the logged lap than a logged lap of another day is (`car_holdout.attainment`, `GateReport.score`). Today 60 of 76 cells (56 before the driven axle and the tyres'-frame reads of #312, 34 before the driver's braking use; speed 17 of 19, end of straights 16, apex 9, braking 18); 32 cells hold one day's laps and cannot be scored. The count moves by about 2 cells by chance: a gain under about 5 is not told from noise, and about 66 is its ceiling (a layout's median logged lap of other days scores 59–66). Judge a change by the cells it wins and loses. Keep a measure this plain: a number for the match to reality, with reality against itself beside it; the owner set aside a four-part set as "designing with metrics". Use percentiles of pace, not a margin of lap time, and do not score above P95: the fastest laps of a layout come from one day and have nothing to be set against. Laps are compared by position with nothing moved, so 2 m of position error reads 0.04–0.05 g; compare two versions of the simulation on the same laps, where it cancels. It does not say why a lap differs, and heat cannot (braking heat is the energy shed whatever the level). The braking point (`brake_m`, ± 5 m) that stood in the gate before is removed: it answered to power and to the driver's onset more than to the braking level, and the braking points quoted above and in docs C3/C4 are what it read while it stood. By the new measure the braking limit's shape put the braking further from the logged lap's in nine of ten cells; the shape the driver holds, which the laps are now braked on, brings it nearer in nine of ten.
- **The car's braking limit is the car's; the laps are braked as the driver brakes** (owner, 2026-11-21: step 1 of the attainment committee's plan, issue #312's split; docs §2.12 C7). A model's `driver` entry holds `brake_use` (the level as a share of `mu_b·u`: 0.80 / 0.85 / 0.84) and `brake_use_ellipse` (the shape held under cornering: 0.59 / 0.28 / 0.47, less than the limit's), read by `car_gate.set_use` from the logged deceleration of the car's laps, each at its own u: the braking force over `mu_b·u` against the lateral over `mu_y·u`, the median per band of lateral, then the solver's own form by least squares. The pace knob multiplies the level like the limit (fixed in g it loses 8 cells). Since 2026-11-23 the car is put at its cornering limit first and the use is read there (`car_gate.at_limit`, then `set_use`), as a share of `mu_b` at a use of 1: 0.87 / 0.97 / 0.90 with the driven axle in (0.886 / 0.990 / 0.914 before it), which at the cars' typical use is about the 0.80 / 0.85 / 0.84 quoted above. `_with_mu_y` moves the share with `mu_y` so that the laps stay the same laps; while `mu_y` was a convention the use had to be read before it was set (applied after, it left u 2–3 % high and the heat at an unseen layout worse, 5.3 % against 2.9 %). The reading must give a planted use back (gate C2: within 2 %); a median at low lateral did not. `Drive.brake_use = False` and `SolverCar.from_model(..., brake_use=False)` brake at the limit, for the comparison and for the recovery of a known car. A planted wrong car keeps the fitted car's driver (`car_holdout.convene(..., use=)`). Do not choose the level by the count: a plain scale of `mu_b` scores as well at 0.77 as at 0.84. What it leaves: the simulated car slow through the braking zones, u up 3 % and the Inferno 86's apexes over at every level; the brake's onset still off (it loses 4 cells on top); apex speed at a given lap time is the open step, and grip against speed, the car's own line and a split knob were tried for it and not taken.
- **The pace knob is to be the driver's, not a scale on the grip** (owner's decision, 2026-11-22; docs §2.12 C7; issue #313). u multiplies `mu_y` and `mu_b` (the kernel is unchanged), and since 2026-11-23 `mu_y` is a limit that pace does not move, so u is the share of it used; a wet lap still differs from a dry one by `wet.grip_scale` on the same two numbers and nothing else (the wet as a condition on the limit is #206's). The decision: grip belongs to the car, the day and the condition; the lap's pace is how much of it the driver uses and for how much of the lap. Do not describe u as grip, and do not add terms that lean on it being one. **The car's limit is its best moments put together** (owner, 2026-11-22: "while we cannot find the true limit, we can certainly use momentary max performances to derive a limit that is higher than has ever been attained so far"): the true limit cannot be read from one driver's laps, so the car model holds the best shown in each respect, each where it was best, labelled as a floor under the true limit; the driver model is every way a lap falls short of it. **A committee's design for both is in `docs/lapsim_driver_model.md`; the owner ruled on its seven decisions on 2026-11-23, and its first step (the limit, the knob as the use) is built (#312; the bullet on `mu_y` above).** The rulings: the uniform knob is accepted now, and a direction by phase or corner is added only if a stint-level test finds one; m = 20, with 10 and 50 reported beside it; a dry day's level (about 2 %) is the driver's and goes into the knob; drive out of a corner goes through the driven axle on all three cars (the committee's default was the Inferno 86 only, the formula cars' reading being confounded); the car is checked by the max driver against the best logged lap, with the sum of the best segments as a diagnostic; a wet day's water level is solved from the target lap time, with no wetness input; a target outside the logged range of pace is answered and flagged, not refused. In short: the cornering limit is, per corner, the expected best of m passes (m = 20 by the owner's decision) of the lateral over the tyres' load, read on the mean of the Doppler and accelerometer readings with their noise taken out, the median over corners (Inferno 86 1.278, KK-SII 1.539, KK-F 1.492, against the convention's 1.160 / 1.318 / 1.372); a maximum or a 99th percentile rises as days are added and reads the sensor, a 90th percentile reads the driver. The knob it offers is one use of that limit, the same everywhere and capped at 1: the arithmetic of today's u under a new meaning, so "less time at the max" is nearly empty at logged paces (about 2 corner passes in 100 reach the cap). **Do not read a slope against lap time within a day as how pace moves the lap**: laps simulated with no knob, only corner-to-corner scatter, give slopes of the logged size (the faster laps of a day are those whose corners happened to go well), and a direction planted in the knob is not read back; braking is the exception (the Inferno 86's peak braking hardly rises with pace). A lap time or a speed trace fixes only the product of limit and use. One lap time is made in no distinct ways: a continuum of gives and takes corner by corner, with a heavy slow tail of mistakes (about 2 % of corner passes), which moves the mean heat by under 0.3 %. In the wet the formula cars lose cornering and braking about alike and the Inferno 86 holds its braking better (0.92 against 0.83–0.88 of dry, on few wet days), and a wet day's water level is a condition, not the driver's.
- **A use cannot pass 1, where a lap is answered** (`solver._knob`, `SolverCar.use_cap`; docs §2.12 "The cap"; built 2026-11-24, the driver committee's clause 3.1). With the knob above 1 the cornering stays at the limit and the braking level rises, `min(mu_b·brake_use·u, mu_b)`, to the car's own at u = 1/`brake_use` (the knob's end lap); past it the lap time is flat and a faster target has no root. It is built above the two solver forms (a capped lane is handed its limits with a knob of 1; neither `_kernel` nor `_tensor` changed) and is asked for explicitly, `from_model(..., use_cap=True)` / `Drive.use_cap`: by the pace table and by gate C3 for the fitted car's held-out laps, and only for a car whose `mu_y` is its limit. **Do not turn it on in the fit's workings, a recovery, a planted wrong car or a comparison arm**: they use the knob as a free scale, and with it on gate C2's recovery and gate C3's controls moved. Gates C2 and D2 are byte for byte with it and gate C3 stays at 60 of 76, differing only on the Inferno 86's folds that hold one of the five laps above 1 (the highest needs a knob of 1.102 capped, 1.009 before).
- **A pace table holds laps, not heat, on the axis of use** (`pace.build_table(model, track)`, `lapsim check-pace-table`; docs §2.12 "Gate D1, the pace table"; built 2026-11-24). Twelve laps evenly in q = −ln(use) from a use of 0.40 to 1, each kept as its speed at the track's nodes; the lap at a use of 1 is braked as the driver brakes and is not the max driver's lap (habits off, 0.3–0.9 s faster, kept beside the table as `max_driver_lap_s`); `PaceTable.heat(lap_time, shares)` splits a lap with the caller's force shares through gate D1's chain (or the caller's), log-log between the knots: within 0.9 % of a direct solve on all fifteen car × layout tables (limit 1 %; nine knots would fail at 1.2 %, and more than twelve buy little: a tyre's heat is jagged by 0.2–0.4 % as the one-second bins slide along the lap with the lap time). Never store per-tyre heat columns in it: heat is defined by the shares of the fit in hand (#208). A table states what it assumes of the car (`assumes`: the latest powertrain unless one is named, its own shift point and gears, standard air, braked as the driver brakes, dry), and the powertrain is worth up to 2.8 % of a tyre's heat at one lap time, so read a lap against the table of its own powertrain. A target lap time is `ok`, `outside_support` (answered and flagged: outside the car's logged range of use at the layout, `meta.use_range` saved by gate C2, or the car's range over its layouts where it has not run this one) or `no_solution` (faster than the knob's end lap, given beside it with the max driver's). Since the cap, a table of a car at its limit has two more knots from a use of 1 to the end lap; a target there is answered and flagged, with the cornering use (1) and the braking use apart. The out-lap of each knot runs from the pit exit's join (`meta.pit`, with the logged out-laps' speed there) to the line. The wet entry waits for D2.
- **The wet is the same car at a lower use on the dry track, and no wet term is in the default** (gate D2, `lapsim check-wet`, `wet.py`, `wet_gate.py`; docs §2.12 "Gate D2, the wet"; built 2026-11-24). A wet lap is solved to its lap time, so the wet grip scale, the day's water level and the driver's use are one product no simulated lap depends on; only braking falling less than cornering (`SolverCar.wet_brake`, from `wet.wet_terms`: 1.07 / 1.02 / 0.83, Inferno 86 / KK-SII / KK-F) and the wet line (Tsukuba only) are not absorbed, and neither met the entry rule written before the run (nearer than the null on 9 of 13 car-days where 10 were needed; 6 of 9 where 8 were): `solver.WET_BRAKE_DEFAULT` is off and the wet line is not driven. **D2 is scored against a logged lap of the same wet day** (owner's ruling of 2026-11-24, for the wet only: "wet is harder, we know that"; gate C3 and D3 keep the other-day form; 135 wet build laps give 4 cells per measure against another day's lap and 13 against a same-day one): the null reaches 14 of 52, to be read beside the *dry* gate's 31 of 104 against same-day laps, not beside 60 of 76. Do not re-cut the braking term by car to make it pass, and do not build a wet pace table: the dry table read at the target lap time is the wet entry until a wet term enters, its support running down to the car's slowest wet build lap (`meta.use_range_wet`, saved by gate C2; without it every wet lap was flagged `outside_support`). A D2 fold holds out the wet term only; the car is gate C2's dry car.
- **Segment E (#207) compares the simulation's heat with production's heat as it is today, as paired runs against real laps, with no fixed limits** (owner's rulings, 2026-11-24). Production is not retrained for it: E0 (#255's refit on GPS-derived heat) is set aside, so "observed heat" in E and F is production's own per-lap heat (its chain on the sensor's series) and a simulated lap's heat is brought to it by the frame factor; one chain, production's and unsmoothed, on both sides. A cell is a car at a layout at a level of pace (P75, P90, P95) on one tyre and passes when the simulated lap's heat is no further from the logged lap's than a logged lap of another day is. **Do not report E against a fixed percentage** ("get rid of the 5 % all together, use the vs real life paired runs like we do elsewhere"): #207's 0.03 / 0.02 / 0.05 are gone from code and reports, after a car with 0.85 of the power was found inside 5 % in every cell. The simulation is adopted at gate F's last clause, for entries with no logged laps only: an entry with logged laps keeps its logged heat and the pressure model's fit does not stop.
- **Gate E1 and gate E2 ride on gate C3's folds** (`heat_match.py`; three items in `check-car-holdout`'s report and one in `check-wet`; docs §2.12; run 2026-11-24). A held-out lap's own solved lap goes through production's chain (`score_laps(..., share=)` gives `sim_q_<tyre>`; not `h_<tyre>`, which is gate C3's heat *error* and is written into the same frame on layout folds) and is brought to production's heat by the fold's frame factor, read only from laps the fold does not hold out; a yardstick pair is two logged laps of the same kind of source. **With the lap time given the simulated lap is about as far from the logged lap as another day's logged lap is** (1.4–1.5 % against 1.3 % in the median cell; 40 of 76 cells on held-out days, 41 at a layout the car was not fitted on), and not having seen the cell costs nothing that can be told: on a track from other cars with the car unseen (`transfer_laps`, arm D) 25 of 40 cells against 22 for the car's own fit on its own track; the car moves a tyre's heat by 0.2 % or less and the track by up to 1.6 %. The car's logged heat at its other layouts, unscaled, passes 4 of 76 and 1 of 40. **The count does not tell the fitted car from one with 0.85 of the power** (30 and 27 of 76; another car's parameters 19 and 13): heat at a given lap time is set by the track and the lap time. **With no lap time given the lap time is supplied about as well as days repeat** (0.8 % from the day's best in the median cell, another day's best 1.0 %) **and the heat of that lap is not the logged lap's** (4 of 40 cells; 4.6 % against 1.2 %): heat is three to five times as steep as lap time. The typical use is also a best-lap pace: a session's 25th-percentile lap, which production feeds, is 0.8–2.2 % slower, so a simulated lap time prefilled as it stands would add 3–6 % of heat; a typical use for the lap that is fed is not built (#208). The wet clause (`wet_gate.wet_heat`): 27 of 48 cells against a same-day lap with no wet term. **The pace response is measured in bands of pace, not as a slope** (`heat_match.band_heat`: every clean dry lap within 12 % of its session's best; the gates' build laps are within 2 %): the simulated heat stays nearer the logged lap than another logged lap of the same band does at every pace (27 of 36, 27 of 36 and 20 of 28 cells; 1.1 → 2.0 % from the logged lap where two logged laps go 1.6 → 3.1 %), drifting 0.7 → 1.4 % low in the mean, with the lean to the left turns growing on the formula cars at Tsukuba (right less left 3.4 → 7.3 % on the KK-SII).
- **Segment F (#208) compares the pressure a simulated lap leads to with paired real days, and the switchover is the owner's call on that report** (the plan approved on 2026-11-24, not built; docs §2.12 "Segment F"). The figure is the cold pressure the calculators would recommend at flying lap 5, through the fold model, with no TPMS reading in it (pressure against the TPMS cannot rank heat sources and is reported only): the simulation's distance in mbar from the one a logged day's own heat gives, with another logged day of the same raw car, powertrain and kind of source at the same track beside it. A cell is a raw car at a track on one tyre. **No fixed limit in bar and no levels of pace**; with nothing given, the real run beside the simulation is given nothing either (another single day's own lap time and heat), with the logged entry built without the day as the ceiling. Per-tyre heat always goes through the fold model's own shares. Sodegaura and Suzuka are scored with only the car unseen and tallied apart (a track from another class of car exists at Tsukuba and Fuji only). The count is about three groups, so read the distances, the planted controls and the simulation against the car's other layouts; do not add measures to enlarge it. **A simulated entry must mean what a logged one means** (the cell's median flying lap and median heat): the typical use is a best-lap pace, so a *fed use* is built first, per raw car from its own laps and never read off the pooled FJ entry. **Adoption** is a committed export of the simulated laps (the series the heat chain reads, per knot) that every retrain splits through its shares, since an entry's heat is of one fit; and the owner ruled that **the tracks are committed with their bank, with attribution to GSI, and no flat-road arm is run** (done: the bullet on the road's bank above).
- **The goal, as the owner restated it on 2026-11-25: the simulation-based model replaces the current calculator model, and runs as a simulation in the browser** ("the goal is to delete the current calculator model and replace it with one that does a simulation IN THE BROWSER"; "the simulation based model is everything built in 202 ... a track model, a car model, a driver model, a heat model, and a tire pressure model, all of which stack together to predict the tire pressure growth"; docs §2.12 "The goal restated"). This replaces the plan of 2026-11-24 where it said an entry with logged laps keeps its logged heat and that the simulation reaches the calculators only as precomputed entries. **It is not built**: the solver is Python with no port, the pressure model is still the current one (K, τ, κ fitted on logged heat, in the tyres' frame since #376, so that a simulated lap meets it with no factor), and gate F read the simulation through a pace table. **The calculator is judged by MAE over folds of held-out data** (owner, same day): the standing holdout with the simulation-based calculator beside the current one on the same folds, fits and rows is `just tire-lapsim check-pressure-holdout`; when the owner says "the n-fold analysis" this is what is meant, not gate F's paired days (which are reported beside). Read that MAE knowing a heat 5 % too low scores better in three of five cells. **Next is a shootout between a precomputed pace table (A: no solver in the client) and a direct simulation (B: the solver ported, driving the lap asked for)**: accuracy by the holdout's MAE and by gate F's measure, A against B directly, coverage, and cost (bytes shipped, compute per prediction, the port and its parity with the Python solver); B is scored with the Python solver before any port is written, and the call is the owner's. Do not port the solver, delete the current model or materialise simulated entries before that call.
- **The standing holdout on simulated heat, as run on 2026-11-25** (`just tire-lapsim check-pressure-holdout`; 15 folds, 128 sessions, 5,751 rows; the folds run side by side on half the cores as worker processes of two threads each, `--workers`): the current calculator 0.0437 bar, the simulation-based one 0.0446 as a pace table and 0.0447 driven directly (per session +0.7 ± 0.3 and +0.8 ± 0.3 mbar); FJ 0.0353 against 0.0366, Inferno 86 0.0551 against 0.0555 and 0.0558; the gap does not close at P75, P90 or P95 of pace; the table and the direct lap are 0.8 mbar apart in the median row and not told apart. **A fold's held-out sessions, and only those, are out of every model under a simulated entry** (`pressure_gate.held_out_drive`, `track_without`, `seen_cell`): the pressure fit, the car model, the track's curvature, the frame factor, the fed use. Do not hold out more (a whole car, a whole track) in this evaluation, and do not leave a held-out session in any of them. By gate F's paired-day measure on the same question (leave-date-out) the simulated entry is nearer the day than the pooled logged entry (3.2 against 3.5 mbar, 6.0 against 7.0; 15 cells of 20): the two measures disagree by about 1 mbar each way.
- **How the calculators are compared, and how to do it again: `docs/calculator_evaluation.md`; `just tire-sim-compare` runs every comparison** (gate F, the standing holdout with the simulation-based calculator beside the current one, the KK-F fitted apart and `compare-pressure-holdouts`, `pressure-pace-margins` and the holdout by pace at four fractions). The simulation-based calculator is landed for comparison only: nothing is cut over, and a change to a track, a car model, the heat chain or the pressure fit is judged by running that recipe again.
- **Two experiments on that holdout, neither adopted** (owner's questions of 2026-11-25; docs §2.12). *The KK-F fitted apart from the KK-SII* (`check-pressure-holdout --split KK-F KK-F-V2`; `pressure_gate.split_pooling` changes `CAR_FIT_ALIASES` for the process only): level in total (0.0437 → 0.0436 bar); better on the KK-SII (0.0318 → 0.0302, bias +7.6 → +2.7 mbar) and in the dry and the wet on the KK-F, worse by 22 mbar on the KK-F's four damp sessions, which is all of its loss; with the cars apart the simulation-based calculator is level with the current one on the KK-F in the dry. *A lap time under anything trained on* (`--fastest 0.25`: each cell's fastest quarter of dry sessions out of every model): the simulation is not better there (+1.2 ± 0.7 mbar, as in the standing folds; nearer only at the Inferno 86's Fuji), and at a quarter the data ask for only 0.5 % under the training pace. With 80 or 90 % of a cell's fastest sessions held out the lap times asked for are 2 to 6 % under anything trained on at the track (62 of 85 and 80 of 95 sessions under every lap left) and the simulation is still 1 to 1.5 mbar behind (0.0459 against 0.0481 and 0.0485 bar at 80 %), level on the sessions more than 3 % under, nearer at the Inferno 86's Fuji (−2.5 to −4 mbar) and further at Sodegaura and the FJ's Tsukuba. The current calculator's pace curve is already a simulated faster lap, so do not expect the simulation to win on extrapolating pace without data that extrapolates.
- **Gate F is built and run; adoption is not, and waits for the owner** (`just tire-lapsim check-pressure`; `lapsim/pressure.py` the mechanics, `lapsim/pressure_gate.py` the folds, the arms and the report; docs §2.12 "Gate F"; 2026-11-24). As measured: five raw-car cells have three dry days (20 cells with the tyres). Two logged days are 4.1 mbar apart at a given lap time and 7.0 with nothing given; the simulation is 3.4 and 6.9 from the day (13 and 10 of 20 cells; another day on the simulation's own stints 4.4 and 7.6), the cell's own logged entry without the day 3.8 and 5.4, the car's other layouts 20.1 and 18.4, the fallback as coded 52.6 and 65.0. A heat 5 % off, another car's parameters and a lap time 1 % slow are seen; **a car with 0.85 of the power is not, in either form**, so do not quote this gate as a check of the car's power. Rules for whoever touches it:
  - **The calculator as it is today, by the same measure** (owner's question of 2026-11-25; `pressure_gate.report_current`): at a cell without logged laps its fallback is 52.6 and 65.0 mbar from the day (1 and 3 of 20 cells) where the simulation is 3.4 and 6.9; at a cell it has logged, with the day's date left out, its own entry is 3.5 and 7.0 (7 of 20 in both forms; in sample 3.4 and 6.6). So a simulated entry at an unlogged cell recommends about as near the day as a logged entry does at a logged one; what the logged cell has beside is constants that have seen it (33 against 43 mbar against the TPMS).
  - **A fold is a pooled car at a track with every one of its sessions left out**, scored or not, any condition (`pressure_gate.cell_sessions`); it is fitted once (33–65 s) and kept in `workspace/lapsim/cache/pressure_folds/`, keyed by the sessions left out, the dataset's files and the bytes of the fit's source and of the loaders it reads the timeseries through (`pressure_gate._FIT_SOURCES`: add a module there when the fit starts to depend on it), so a change to the fit refits and a change to the scoring does not. Always `write_artifacts=False`: the gate writes nothing under `data/`.
  - **Every heat is of the fold's own shares**: `holdout.shares_of(model)` on the fold's model, never `production_shares(root)` inside a fold. A logged lap is binned once (`heat_gate.sensor_lap_bins`, `tyres_frame_lap_bins`) and split per fold; a simulated lap is split from the table's exported bins (`PaceTable.laps_export`, `lapsim/export.py`); the frame factor is read under the fold's shares from the car's laps at its other layouts and is not a stored number (a refit moves it by up to 1.2 %).
  - **A heat source is the entry it would write** (`pressure.Entry`), placed in a copy of the fold's model (`pressure.place`) and read by `validate._calculator_lap_inputs`. `validate._evaluate_fold` is `_fit_fold` + `_prepare_fold_laps` + `_score_fold_laps` and takes `model=` and `lap_inputs_fn=`; a change there has to leave the standing holdout bit for bit (the gate's first check does it on every fold).
  - **The figure is the recommended cold pressure at flying lap 5 against the day's own, with another day beside it; no TPMS pressure is in it.** Pressure against the TPMS (S2) is in the report and never ranks: the day's own heat scaled down by 5 % scores better against it in three of five cells.
  - **The pairing was fixed before the simulation was scored and is not to be moved to make a cell pass**: a day is a dry session with three ordinary flying laps (from flying lap 4 on, none over 1.12 × the stint's best) and a pit-exit stint; another day is of the same raw car, powertrain and kind of source; at a given lap time up to three sessions nearest in lap time, each from its five laps nearest the day's; with nothing given every other day as if it alone had written the entry; a cell needs three days. **Two arms are compared on the stints both have** (`pressure_gate.attain`): the simulation has no entry for a day whose kind of source has no frame factor at the car's other layouts (the KK-F's accelerometer days at Tsukuba), and a yardstick over more stints than the arm beside it is not the same rows.
  - **A day is driven on a track with no lap of its own date in it** (`pressure_gate.donor_track`), and a track from another class exists only where the donors have three days and thirty laps (Tsukuba and Fuji, and Motegi, whose cell has two days); elsewhere only the car is unseen and the cell is tallied apart.
  - **A simulated entry supplies its lap time from the fed use, per raw car** (`car_gate.fed_use`: the use of a session's median flying lap, median over sessions and then layouts; 0.05–0.07 under the typical use). Never read it off the pooled FJ entry, and never prefill a simulated lap time at the typical use.
  - **A simulated entry's curve is the table's knots** (the ports read any number of nodes linearly: within 0.24 mbar of the table read log-log; nine nodes on a logged curve's grid are up to 4.3 mbar off).
  - **Reported beside, in the same report, and nothing rests on one**: the wet and damp cells (the dry table read at the wet lap time against a run of the *same* wet day: 9 of 12 cells, 2.4 mbar against 6.0, on one or two days a cell), the chronological first visit (a fit of what was logged before a car first ran a track gives the simulation's distance within 0.5 mbar of the cell fold's on four visits: leaving a cell out does not flatter), the pooled simulated entry (its cost is the pooled lap time, 2 to 4 mbar on the KK-F with nothing given).
  - **What of the cell the simulated side still sees**: the track's pit exit (`meta.pit` is the median over every car's out-laps, the held-out car's among them) and the own track's reference line, bank and grade.
  - Not built: adoption (the committed export, entries with a `source` flag in `tire_model.json`, the calculators' line and attribution), which is the owner's call on the report; and #260's changes to `predict.py`, which #208 lists as not in F.
- **The pressure model's heat is in the tyres' frame, so that no frame factor exists** (#376; default since the owner's promotion of 2026-11-26; docs §2.12 "Heat in the tyres' frame"). Every session's heat input is rebuilt from its GPS course laid on the committed track (`lapsim/tyres_frame.py`: what the tyres hold on the bank, `dV/dt` plus the grade, the whole session; the ground frame, flat, off the line), so that the fit, its logged entries and a simulated lap meet with a factor of 1 and no IMU-or-GPS verdict is read by any heat (the verdict is kept beside as the label `sensor_source`, for reports and the pairing by kind of source). `heat_frame` is the frame in force (`tyres` outside any `heat_frame.using(...)`; a `heat_frame=None` argument means it), and the fit, `tire-predict-holdout`, `tire-build-warmup-table` and `tire-lapsim check-pressure-holdout` default to it; `--heat-frame sensor` / `using("sensor")` is the sensor verdict's series, kept for comparison, and **only there does a frame factor exist**: the frame-factor figures quoted in the bullets above (D1, E, F) are sensor-frame runs from before 2026-11-26. The model records `heat_input.frame` (absent: `sensor`) and `heat_input.frame_fallbacks`; `attach_lap_heat` reads the frame from the model, but `pressure.day_entry`, `logged_entry` and `heat_gate.sensor_lap_bins` read the frame in force, so wrap a scoring script in `using(frame_of(model))` (`holdout_rows` refuses a fold whose frame is not the one in force). The other lap-sim gates (`check-heat-chain`, `check-car-holdout`, `check-wet`, `check-pressure`, ...) compare with production and read heat in the frame of the committed `tire_model.json` (`heat_frame.model_frame`): the sensor's until it is retrained on this. Where the frame cannot be built it says so: a layout with no committed track (today `motegi_east`, `fuji_short`, `suzuka_west`, `minami`) is the ground frame, bank and grade 0, with a warning and the layout listed in `frame_fallbacks`; a session with no layout or no GPS geometry has no heat in this frame and is left out of the fit, listed the same way; a dataset with no sessions table is a `HeatFrameError`. A short stretch without a GPS value (≤ 1 s) is bridged and the session's edges hold their value; without that, 149 training laps (a file's last lap at speed among them) lost their heat. Measured on the 15 standing folds against A-master (master 47426819c324 in the sensor's frame; paired, ± session / date SE): the calculator 43.79 → 43.60 mbar (−0.18 ± 0.16 / 0.19), the table as served 44.73 → 43.76 (−0.97 ± 0.27 / 0.35; FJ −1.54, Inferno 86 −0.21), direct 44.83 → 43.84; the table's gap to the calculator +0.94 → +0.16 (KK-SII +1.36 → −0.14, KK-F +1.29 → +0.22, Inferno 86 +0.42 → +0.39), and with the logged out-lap (an out-lap swap on the rows, not yet an arm) −0.04 ± 0.19: parity. The pre-registered gate, written against the standing holdout of 2026-11-25 (before #388: calculator 43.71, table 44.64, direct 44.74), passed 2 of 4 (it failed on κ_FJ's spread, which did not shrink, and by 0.02 mbar on the good side of the table's ±0.30 band). On the unseen KK-F-V2 Fuji day of 2026-11-25 GPS and IMU heat agree within 1 % (7–10 % before). Never compare a tyres'-frame fold with a sensor-frame one through a frame factor: the factor does not exist in that frame.
- **Segment D's gates on logged laps take gate C3's form** (owner's decision, 2026-11-23; #206). D2 (the wet, once it is rewritten against the driver committee's design) and D3 (lap-mean lateral heat simulated over logged, and the same for right turns minus left) are scored as an attainment: the number with a logged lap of another day beside it, at P75, P90 and P95 of pace, a cell passing when the simulated lap is no further from the logged lap than a logged lap of another day is. The fixed limits written in #206 (0.03 and ± 0.06 on laps within 1.02 × the session best; 0.3 on the wet elasticities) are superseded as what passes. The unit tests of gate D1 (closed forms, the reference ovals, parity with production's heat) keep their tolerances: they are not a match to reality.
- **A further term is held to the committee's test** (docs §2.12 C7): the rule written before the run and its value read from the logs; day folds and layout folds; a gain above zero in 95 % of day resamples, or its own statistic named beforehand where the count cannot decide; a 40 m blur of the trace moved 4 m earlier (a placebo worth 10 cells on the old default) beaten, and no gain from blurring the candidate's own traces; planted wrong cars still 10 and 20 cells below; heat at an unseen layout no worse; nothing indexed by layout, corner or node, nothing lent by another car.
- **A gate is re-measured, not assumed.** A failed check is fixed or recorded with its cause in §2.12 before the next piece; a threshold the measurements show was wrong is re-baselined on the issue with the original quoted.

## Code Style & Conventions

- **Formatter:** Black, 100-char line length, target py312
- **Type checker:** mypy (check_untyped_defs=true, disallow_untyped_defs=false)
- **Docstrings:** NumPy-style with Parameters/Returns/Raises/Examples sections
- **Type hints:** Use `from __future__ import annotations`, PEP 604 unions (`X | None`), `TYPE_CHECKING` blocks for import-only types
- **Data handling:** Prefer PyArrow tables and `pyarrow.compute` (pc) operations for filtering/transforms over pandas. Convert to pandas/numpy only for analysis steps that require it.
- **Channel names:** Passed as dictionaries (`channel_names["throttle"]` -> `"PPS"`) to support different AIM telemetry formats. Use `_validate_channel_names()` for validation. When adding new channel keys to `DEFAULT_CHANNEL_NAMES` in `profiles.py`, also add the corresponding entries to each vehicle profile in `src/motorsports_data_notebook/data/builtin_profiles.yaml` (channel names vary per vehicle/logger).
- **Vehicle profiles:** Resolve with `profiles.get_profile_for_log(log)` (logger ID + vehicle name), not by logger ID alone — the KK-SII, the KK-F and the KK-F-V2 (the KK-F with a new intake, from 2026-11-13; its profile is a copy of the KK-F's) share SW4 logger 8401203 and are told apart by the Race Studio vehicle name (`Vehicle` metadata in .xrk, file-name token `CMD_<Vehicle>_…` for .xrz). `logger_map` entries may be a profile id or `{default, by_vehicle}`; a logger lent between cars (6520348: the KK-F in 2026-06/07 under `kk_f_dl`, and the 718 Cayman) has `by_vehicle` only, so another car on it gets no profile instead of the wrong one. The tire ETL resolves the same way (`tire_etl.extract.session_channel_names`, #254). The suspension analysis uses the **pot/wheel** motion ratio only (pot travel / wheel travel, dividing the logged shock pot channel); every profile stores a `shock_calibration` block (`wheel_per_sensor`, `damper_per_sensor` per axle, fitted with `scripts/fit_shock_calibration.py`) from which `motion_ratios` derive as `1 / wheel_per_sensor`; `damper_per_wheel` is stored as a directly fitted slope for other modeling, never recomputed as a ratio of slopes.
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
- **deploy-preview.yml** — On push to `master`: builds the JupyterLite site, deploys to preview
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
