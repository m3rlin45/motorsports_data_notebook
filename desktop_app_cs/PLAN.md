# Inferno Analyzer v3 — C# desktop rebuild plan

Plan recreated 2026-10-22 from the 2026-08-19 brainstorm and the 2026-09-20 spike
(the original plan file was lost; this copy lives in the repo so it cannot be).
Re-baselined against the repo at v1.2.0 and the .NET/Avalonia ecosystem as of
October 2026.

## Why

Three implementations of the telemetry analysis app exist and none is right:

- **Python/CustomTkinter** (`desktop_app/`, 6.8k lines): works, four tabs, but a
  clunky UI with matplotlib embedded in Tk.
- **Rust/egui** (GitHub PRs #57–#82, never merged): reached feature parity, but
  `egui_plot` immediate-mode charts look bad.
- **JupyterLite**: the most powerful, poor UX for "run a known analysis on new data".

Rebuild in **C# (.NET 10 LTS / Avalonia 12)**, which already works well for the
tire pressure calculator. Risk is concentrated in charting; everything else has
been de-risked twice (the Rust port transliterated every kernel, and the tire
calculator proved the Avalonia tooling).

## Decisions (user)

| Decision | Date | Note |
|---|---|---|
| OSS-only libraries | 2026-09-20 | Avalonia Charts, SciChart, LightningChart, TreeDataGrid 12 (all Pro/commercial) ruled out |
| Native plots: ScottPlot 5, no WebView | 2026-09-20 | The `IChartSurface` seam keeps a Plotly-in-WebView adapter possible later |
| Visualization is a swappable module | 2026-09-20 | Viewmodels emit semantic specs; backends adapt |
| v1 features: delta-time trace, user-defined sector gates, YAML track store + editor | 2026-08-19 | |
| Execution starts by rebuilding the spike in-repo | 2026-10-22 | So the ScottPlot verdict comes from a tracked build |
| Split-times table is custom-drawn | 2026-10-22 | TreeDataGrid 12.x is Avalonia Pro-only; DataGrid (MIT) stays for plain lists |

## Ecosystem baseline (October 2026)

| Component | Version | Notes |
|---|---|---|
| .NET | 10 LTS (10.0.1xx SDK) | .NET 11 (STS) ships 2026-11-10; ignore it |
| Avalonia | 12.1.4 | Tire calculator stays on 11.3.x; do not block on migrating it |
| ScottPlot.Avalonia | 5.1.59 | First release with Avalonia 12 support; slow cadence (~3/yr) |
| Avalonia.Headless.XUnit | 12.1.4 | Built on **xunit v3**, so this solution uses `xunit.v3` (the tire calculator uses v2) |
| Apache.Arrow | 23.0.0 | C Data Interface for the libxrk FFI (Phase 1) |
| DuckDB.NET | 1.5.5 | Arrow result streaming; optional, Phase 3+ |
| libxrk / libibt | v0.13.0 / v0.0.5 | Pure-Rust core crates with an `arrow` feature + PyO3 wrappers; **no C-ABI crate yet** |

## Architecture

```
desktop_app_cs/
  InfernoAnalyzer.slnx
  Directory.Build.props     # net10.0, pinned Avalonia/ScottPlot versions, version from ../pyproject.toml
  Charting/                 # spec records + IChartSurface + CursorLinkGroup — no chart-library dependency
  Charting.ScottPlot/       # the one live backend (only project referencing ScottPlot)
  App/                      # Avalonia views; chooses the backend in ChartSurfaces.cs
  Core/                     # (Phase 1+) Arrow data model, profiles, analysis kernels — no UI deps
  Tests/                    # xunit v3 + Avalonia.Headless; Fixtures/ holds Python-parity JSON
  tools/extract_fixture.py  # dumps lap data with the Python library (uv run)
```

**The seam.** Viewmodels build a closed set of semantic chart specs
(`StripChartSpec` now; `BoxPlotSpec`, `HistogramGridSpec`, `ScatterSpec`,
`TrackMapSpec` to come). A backend implements `IChartSurface`: `Render(spec)`,
`SetCursor(x)`, and `CursorMoved`/`Clicked` events, all in data coordinates.
The linked crosshair lives above the seam in `CursorLinkGroup`, so the hardest
interaction is backend-independent by construction. Specs carry
`ReadOnlyMemory<double>` slices, so Arrow buffers flow through without copies.

**Data path (Phase 1).** `.xrk/.xrz` → `libxrk-ffi` (new cdylib crate in the
libxrk repo exporting the Arrow C Data Interface) → P/Invoke → `Apache.Arrow`
RecordBatches → kernels and specs. Until then the app reads JSON fixtures
produced by `tools/extract_fixture.py`.

**Scope to port (4 existing tabs + new views).** The Python app's tabs are
Suspension (2×2 velocity histograms, ~1.6k lines), Driver Consistency (corner
summary box plots, detail overlay, track map, ~2.5k), Tire Grip (G-vs-pressure
scatters, ~0.9k) and Roll Gradient (3 scatters with fit lines + track-map hover
highlight, ~1.0k, added 2026-10-22; matplotlib only, no Plotly version exists).
Plus the new Time/Distance view (strip chart + delta-time + track-map cursor
dot). Then the split-times table (custom-drawn) and the track store/editor
(custom-drawn, gates stored as GPS crossing chords). Shared Tk widgets to
re-create: session A/B panel, config panel base, autocomplete channel entry,
stats panel (`src/motorsports_data_notebook/desktop/`, 1.4k lines) and the
debounce/worker pattern in `desktop_app/.../tabs/base_tab.py`.

**Kernels to port** (Python is the parity oracle; the Rust branches are the
structural crib): `_util` (distance, start line, resampling) → `corners` →
`zones` → `suspension` → `roll_gradient` → `driver_analysis` → `tire_grip`.
Profiles: the C# loader is the fourth consumer of `builtin_profiles.yaml` and
must resolve logger ID **plus** vehicle name (`get_profile_for_log`; vehicle
from `.xrk` metadata or the `CMD_<Vehicle>_` file-name token) and derive motion
ratios from `shock_calibration` exactly as `profiles.py` does. Only
`toyota_86_zn6`, `kk_s2` and `kk_f` carry `shock_calibration`/`track_width_mm`;
`honda_ek9`, `super_fj` and `iracing` fall back to default motion ratios and
have no roll-gradient track widths. Known Python quirk not to copy: the speed
channel for `distance_m` is resolved by logger ID alone, ignoring the vehicle.

Rust crib (GitHub, open and stale since 2026-04, `desktop_app_rs/`, egui 0.33 +
arrow 57, no Roll Gradient tab): #57 workspace → #58 data model + XRK/IBT
loading → #59 profiles YAML → #60 analysis (math, corners, zones, throttle
acceptance) → #75 UI/charts → #80 Suspension → #81 Tire Grip → #82 track-map
hover (top of stack). Fetch read-only with `gh` into the scratchpad.

## Phases

0. **Spike in the repo** (this PR) — scaffold, strip chart with linked crosshair,
   delta-time panel, Distance/Time toggle, two fixtures (25 Hz and a 500 Hz
   stress set), headless render test, `just cs-*` recipes, CI, Windows and
   Linux publishes. Verdict recorded below.
1. **Data layer** — `libxrk-ffi`/`libibt-ffi` crates; `Core/Io` P/Invoke +
   Arrow import; `Core/Profiles` YAML loader.
2. **Kernels** — one module at a time, each with a fixture dumped by
   `tools/extract_fixture.py --module X` and an xunit parity test.
3. **Views** — Time/Distance tab, then Suspension, Roll Gradient, Driver;
   split-times table; track store + editor.
4. **Packaging** — Linux NativeAOT, Windows trimmed single-file, release
   workflow, README; retire the CustomTkinter app at parity.

Dropped from the original brainstorm for v1: JupyterLite-in-WebView, CSnakes,
Roslyn UDFs, DuckDB SQL (all remain possible later).

## Developer workflow

```bash
just cs-build            # dotnet build
just cs-test             # unit + headless render tests (PNGs in Tests/bin/.../renders/)
just cs-run              # launch the spike window (WSLg / X11 / Windows)
just cs-run path.json    # ... with another fixture, e.g. App/fixtures/laps_dense.json
just cs-fixture          # regenerate App/fixtures/laps.json (laps 14 & 12 + neighbours, 25 Hz)
just cs-fixture-dense    # 13 top laps on the 500 Hz shock-pot timebase (git-ignored)
just cs-profiles-fixture # regenerate Tests/Fixtures/profiles.json from the Python profiles module
just cs-session-fixture  # regenerate Tests/Fixtures/session_86.json (Rust backend)
just cs-native           # build the libxrk/libibt FFI libraries into native/runtimes/<rid>/native/
just cs-publish-win      # trimmed single-file win-x64 exe (cross-compiled)
just cs-publish-linux    # linux-x64 NativeAOT binary
```

## Status

### Phase 0 — 2026-10-22

Built: `Charting` (specs, `IChartSurface`, `CursorLinkGroup`), `Charting.ScottPlot`
(stacked `AvaPlot`s with pixel-aligned fixed layout, x-limits synced via
`AxisLimitsChanged`, dashed crosshair per panel, legend on the top panel only),
`App` (fixture loader, `StripChartBuilder` with the delta-time panel,
Distance/Time toggle), tests (builder, cursor group, headless render and
hover), fixtures, `just` recipes, CI workflow.

Verified 2026-10-22 (WSL2, .NET SDK 10.0.113):

| Check | Result |
|---|---|
| `dotnet build` / `dotnet test` / `dotnet format --verify-no-changes` | green, 34 tests |
| Headless render, 2 laps @ 25 Hz (`Tests/bin/.../renders/strip_chart_{distance,time,context}.png`) | 4 panels, both laps overlaid, delta-time panel shows lap 12 losing 0.4 s; crosshair + per-lap value labels at 2158 m / 60 s; context view shows lap 13/11 faded before the start line |
| Headless render, 13 laps on the 500 Hz shock-pot timebase (3.3 M samples) | load + spec build 0.7 s, first render 0.7 s |
| `just cs-run` on WSLg, both fixtures | window up, no exceptions |
| `just cs-publish-win` | `App/bin/Release/net10.0/win-x64/publish/InfernoAnalyzer.exe`, 40 MB |
| `just cs-publish-linux` (NativeAOT) | `App/bin/Release/net10.0/linux-x64/publish/InfernoAnalyzer`, 23 MB, runs |

Gotcha recorded: ScottPlot's `PixelPadding` positional order is
(left, right, **bottom, top**); always use named arguments.

First by-eye pass (2026-10-23) and what changed in response:

- The crosshair was invisible: a white dashed line on ScottPlot's default white
  plot. The panels now use a dark theme matching the window and the cursor is
  tested through Avalonia's real input pipeline (`HeadlessWindowExtensions.MouseMove`),
  not only via `CursorLinkGroup.SetCursor`.
- Values live beside the cursor, MoTeC-style: each panel stacks one label per
  lap (value + units, in the lap's colour) next to the crosshair with a marker
  on the trace, flipping to the left in the right quarter of the view; the
  x position ("2158 m" / "60.00 s") is a label on the bottom axis. The bottom
  readout bar is gone. Known limit: labels stack 16 px apart, so more than
  ~8 overlaid laps overflow a panel at 4 panels × 860 px.
- The delta-time panel only appears in Distance mode; against time the laps
  are not aligned, so "time lost here" has no meaning.
- The view scrolls past the lap: the fixture carries each lap's predecessor
  and successor (`before`/`after`), the builder stitches them onto the lap at
  negative x / past the lap end, the backend draws the context at 30 % alpha
  and dotted lap-boundary lines on every panel with "Lap N" labels along the
  bottom axis (one row per overlaid lap). The initial view is the lap itself
  (`StripChartSpec.InitialView`); pan or zoom out from there.
- Dragging or wheeling over the data area moves x only; the y-axis is panned
  or zoomed by dragging/wheeling over the axis strip itself
  (`Charting.ScottPlot/DataAreaInput.cs`). ScottPlot's own `LockY` flags were
  not usable because they also lock the axis drag; the lock is decided per
  gesture from where it starts. Gesture tests drive Avalonia's headless
  mouse into the control and check which axis ranges moved.

**Spike verdict (2026-10-23): ScottPlot cleared the bar.** After the pass
above the user judged look, hover and pan/zoom "good". ScottPlot 5 stays the
charting backend; the `IChartSurface` seam remains the escape hatch.
Next: Phase 1 (libxrk FFI + profiles loader).

### Phase 1 — in progress

**Profiles (done 2026-10-23).** `Core/Profiles/` ports `profiles.py`:
`ChannelKeys` (defaults), `VehicleProfile`, `MotionRatios`, `TrackWidths`,
`ShockCalibration`, `ProfileKey`, `ProfilesYaml` (parse + write through
YamlDotNet's representation model, so NativeAOT-safe), `ProfileStore`
(embedded `builtin_profiles.yaml`, linked from the Python package so there is
one source of truth; user `profiles.yaml` at the same XDG/APPDATA path Python
uses, so both apps share edits) and `LogIdentity` (logger ID + vehicle name
from metadata / Race Studio file name). Parity: `tools/extract_profiles_fixture.py`
dumps Python's resolution of every builtin key plus lookup and vehicle-name
cases to `Tests/Fixtures/profiles.json`; `ProfilesParityTests` replays them.
One quirk reproduced on purpose: an empty `Logger ID` is "absent" for the
logger lookup but "present" for the file-name vehicle fallback.

**Data layer (done 2026-10-23, pending the parser PRs).**

- `crates/libxrk-ffi` (lib `xrk_ffi`) and `crates/libibt-ffi` (lib `ibt_ffi`)
  in the parser repositories: a session is parsed once into one Arrow
  `RecordBatch` per channel plus the laps batch behind an opaque handle;
  `*_channel_stream` / `*_laps_stream` export through the Arrow C Data
  Interface (`ArrowArrayStream`), `*_metadata_json` gives the Python
  `LogFile.metadata` dict as JSON, every entry point is `catch_unwind`-safe
  and reports errors through an owned string. Rust round-trip tests drive the
  ABI as a foreign caller on each repo's sample. Both build as cdylib +
  staticlib for linux-x64 and (cross-compiled with mingw) win-x64. The crates
  sit in local commits in scratchpad clones (`libxrk` 015ad78a, `libibt`
  6e8fb20c) — **opening those PRs needs the user** (blocked by the agent's
  permission policy); `tools/build_native.py` pins the same commit hashes, so
  the pins hold once the branches are pushed.
- `Core/Io`: `NativeLoader` (resolves `runtimes/<rid>/native/` or the exe
  dir), source-generated `LibraryImport` bindings (`XrkNative`, `IbtNative`),
  `NativeSessionReader` (Arrow C stream → `Channel`/`Lap`, metadata JSON
  flattened to strings), `XrkReader` / `IbtReader`, `LogFile` with the Python
  `select_channels` / `filter_by_time_range` / `filter_by_lap` /
  `resample_to_timecodes` / `resample_to_channel` semantics (numpy.interp for
  `interpolate == "True"` channels, forward-fill otherwise, interpolated
  float32 channels rounded back to single precision as Python's Arrow cast
  does), and `SessionLoader` = `load_session` (speed channel from the profile
  **by logger ID only**, `speed_kmh`, per-lap `distance_m`, lap-type filter,
  `clean_laps`). Values are widened to `double[]` on import (one copy; the
  zero-copy slice idea is deferred until a kernel needs it).
- Parity: `tools/extract_session_fixture.py` dumps per-channel summaries,
  raw laps, scalar metadata and the `load_session` outcome for the Fuji 86
  sample (Rust backend) and for libibt's `test.ibt`; `SessionParityTests`
  checks 100 channels, laps, metadata, derived channels, and replays the lap
  pipeline against the spike's `laps.json` to its rounding. The IBT test is
  opt-in (`INFERNO_IBT_SAMPLE`, the 70 MB sample is not in this repo).
  Tolerances: 1e-9 relative on GPS-derived channels, because libxrk master
  (what the FFI builds) reorders the GPS float arithmetic vs the 0.13.0 wheel.
- Build: `just cs-native` (host) or `--rids linux-x64 win-x64`; CI installs
  Rust and builds the libraries before the C# test/publish jobs. The App does
  not reference `Core` yet (Phase 3 wires it in), so publishes are unchanged.

Phase 1 is complete once the two parser PRs are merged (then re-pin to the
tagged releases).

