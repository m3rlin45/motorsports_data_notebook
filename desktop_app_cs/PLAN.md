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
have no roll-gradient track widths.

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
   `tools/extract_kernel_fixture.py --module X` and an xunit parity test.
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
just cs-run              # launch the app (WSLg / X11 / Windows)
just cs-run FILE         # ... loading a session (.xrk/.xrz/.ibt) as Session A, or a fixture JSON as the Time/Distance demo
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
  plot. The panels now take a `ChartTheme` (Dark / Light, following the OS
  like the window chrome) and the cursor is
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

**Data layer (done 2026-10-23; libxrk-ffi merged 2026-11-11, libibt-ffi released in libibt 0.2.0 on 2026-11-11).**

- `crates/libxrk-ffi` (lib `xrk_ffi`) and `crates/libibt-ffi` (lib `ibt_ffi`)
  in the parser repositories: a session is parsed once into one Arrow
  `RecordBatch` per channel plus the laps batch behind an opaque handle;
  `*_channel_stream` / `*_laps_stream` export through the Arrow C Data
  Interface (`ArrowArrayStream`), `*_metadata_json` gives the Python
  `LogFile.metadata` dict as JSON (same keys, same order, nested values
  included: for iRacing the parsed session info and the entries derived from
  it), every entry point is `catch_unwind`-safe, opening reports errors
  through an owned string and the stream exports through the shared status
  codes (0 ok, 1 index, 2 null, 3 panic; `NativeStatus` in C#). Rust
  round-trip tests drive the ABI as a foreign caller on each repo's sample.
  Both build as cdylib + staticlib for linux-x64 and (cross-compiled with
  mingw) win-x64. `tools/build_native.py` pins both by commit: `libxrk` at
  master 33f311f4 (the commit that landed `libxrk-ffi`; no release carries it
  yet), `libibt` at 1c880537, its v0.2.0 tag, which is also the version the
  Python project depends on.
- `Core/Io`: `NativeLoader` (resolves `runtimes/<rid>/native/` or the exe
  dir), source-generated `LibraryImport` bindings (`XrkNative`, `IbtNative`),
  `NativeSessionReader` (Arrow C stream → `Channel`/`Lap`, metadata JSON
  flattened to strings), `XrkReader` / `IbtReader`, `LogFile` with the Python
  `select_channels` / `filter_by_time_range` / `filter_by_lap` /
  `resample_to_timecodes` / `resample_to_channel` semantics (numpy.interp for
  `interpolate == "True"` channels, forward-fill otherwise, interpolated
  float32 channels rounded back to single precision as Python's Arrow cast
  does), and `SessionLoader` = `load_session` (speed channel from the profile
  by logger ID plus vehicle, `speed_kmh`, per-lap `distance_m`, lap-type
  filter, `clean_laps`). Values are widened to `double[]` on import (one copy; the
  zero-copy slice idea is deferred until a kernel needs it). libibt's array
  variables: a time-subsample array (`SteeringWheelTorque_ST`, 360 Hz) is an
  ordinary channel at its own rate; a vector-valued one (per-car `CarIdx*`,
  a FixedSizeList per tick) has no scalar series and is left out.
- Parity: `tools/extract_session_fixture.py` dumps per-channel summaries,
  raw laps, the metadata (every key, scalars as text, nested values as JSON)
  and the `load_session` outcome for the Fuji 86 sample (Rust backend) and
  for libibt's `test.ibt`; `SessionParityTests` checks 102 channels, laps,
  metadata, derived channels, and replays the lap pipeline against the
  spike's `laps.json` to its rounding; `IbtParityTests` checks the same for
  the 274 iRacing channels. The IBT tests need `INFERNO_IBT_SAMPLE` (the
  70 MB sample is not in this repo); CI points it at the copy in the libibt
  source that `build_native.py` fetches. The IBT fixture is dumped with the
  libibt wheel the project installs (0.2.0, the release the FFI library is
  built from). The AIM fixture is dumped with libxrk's Python package built from the same
  pinned commit (`just cs-session-fixture` does this), not the 0.13.0 wheel
  the project installs: master has since fixed unit codes, added the
  `GPS_ITOW` / `GPS_Week` channels and a `session` lap column, and orders
  channels by index. Channel order and each channel's first/last/min/max
  match exactly; sums to 1e-9 relative (numpy sums pairwise).
- Build: `just cs-native` (host) or `--rids linux-x64 win-x64`; CI installs
  Rust and builds the libraries before the C# test and publish jobs; the
  publish jobs fail if the parser libraries are missing from their output.

Phase 1 is complete. Left for the next libxrk release (0.14.0): re-pin
libxrk to its tag and drop the source build from `cs-session-fixture`, as
libibt already is.

### Phase 2 — done 2026-10-29

Kernels live in `Core/Analysis/`, one file per Python module, driven by
`tools/extract_kernel_fixture.py --module X` (`just cs-kernel-fixture X`),
which dumps the Python inputs verbatim and the outputs (full where small,
summaries where large) from the Fuji 86 sample, laps 14 and 12.

| Module | State | Notes |
|---|---|---|
| `Numerics` | done | numpy/pandas primitives: pairwise `Sum`/`Mean` (bit-identical to numpy), `Gradient`, centred `RollingMeanCentered` with pandas' even-window offset, `CumulativeSum`, `ArgMax` |
| `corners` | done 2026-10-29 | `GpsToLocalXy`, `ComputeCurvature`, `IdentifyCorners(FromCurvature)`; corner lists identical to Python on both laps, curvature arrays to 1e-7 relative (pandas' rolling mean uses a compensated running sum, so the last bits differ) |
| `zones` | done 2026-10-29 | single-lap and averaged zones, accel merging, `CreateTrackSegments`, segment stats, G utilization (both inline-G paths), `LapSelection` (`get_top_laps` / `get_best_lap`); every output identical to Python on laps 14 & 12, lap arrays reproduced end to end from the C# loader |
| `suspension` | done 2026-10-29 | shock velocity (numpy.gradient with coordinates, edge-padded box smoothing), wheel velocity, zero-centred histogram, moments and range shares, single- and multi-lap analysis, symmetry and group-difference tables; velocity derivation bit-identical on a dumped slice, histograms within one sample per bin and moments to 1e-6 end to end from the C# loader (the 500 Hz inputs are not dumped) |
| `roll_gradient` | done 2026-10-29 | roll angle, least-squares gradient with R², lateral-g binning (numpy digitize/bincount), asymmetric edge-padded smoothing, single/multi-lap over the C# LogFile with GPS track positions; fits and ratios to 1e-6 end to end, bin counts within 1 % (smoothed samples on 0.1 g edges flip with summation order), synthetic fit/bin/smooth cases to 1e-10 |
| `driver_analysis` | done 2026-10-29 | `PrepareThrottleAcceptance` / `FindThrottleAcceptance`; identical to Python per corner and lap on dumped inputs. `DriverConsistency` ports the desktop app's `analyze_driver_consistency` pipeline (corners from the fastest selected lap, zones, segments, stats, per-corner throttle acceptance, braking/speed spreads, opportunity score, traces) and reproduces the fixture's acceptance values end to end |
| `tire_grip` | done 2026-10-29 | bucketed-percentile grip analysis, single and multi-lap, pressure and temperature modes. Python runs this in float32 (the G and TPMS channels are float32 arrays), so parity is 1e-6 with an edge-sample allowance; the synthetic float64 case is exact, after reproducing numpy's percentile `_lerp` |

Phase 2 kernels are complete; what remains of the Python analysis surface
(`report.py`, `session_runner.py`, the tire model and ETL) is out of the
desktop app's scope.

### Phase 3 — in progress

The App is MVVM on CommunityToolkit.Mvvm (source-generated, AOT-safe) with
compiled bindings; views are thin and the ScottPlot surface is created by the
view and fed the view model's spec.

| Piece | State | Notes |
|---|---|---|
| App shell | done 2026-10-29 | `SessionViewModel` (drop zone / file picker, profile resolution by logger + vehicle, lap checkboxes, Best / Top 103 % / All / None, auto top-103 on load), `AnalysisTabViewModel` (300 ms debounce, one run at a time with queued rerun, stale-when-hidden, status), `MainWindowViewModel` (two slots, tab routing, collapsible top panel), `cs-run file.xrz` loads Session A, `cs-run file.json` shows a fixture |
| Time/Distance tab | done 2026-10-29 | `LapSeriesBuilder` extracts the chosen channels per selected lap (with neighbours) from the `LogFile`; channel picker (speed/throttle/brake preselected from the profile); fastest lap is the delta reference. The lap picker doubles as the legend (`StripChartSpec.Legend` off), so overlaid laps never hide the data. Known: with 13 overlaid laps the per-lap cursor value labels overflow a panel |
| Suspension tab | done 2026-10-29 | `HistogramGridSpec` + `ScottPlotHistogramGrid` (2×2 bars, range bands, side-by-side A/B, hover label); `ChannelConfigViewModel` (A/B entries with completion, sync, profile fill, reset) reusable by every tab; motion ratios; stats pane with the statistics, symmetry and group-difference tables |
| Roll Gradient tab | done 2026-10-29 | `ScatterGridSpec` (track map + scatter panels with fit lines, binned means, annotations) + `ScottPlotScatterGrid` (map hover snaps to the track and highlights the index-aligned samples on every panel, equal-units map); config with GPS/lateral/shock channels and track widths, motion ratios shared with the Suspension config as in Python; min |G| entry; roll table |
| Driver tab | done 2026-10-29 | `BoxPlotSpec` + `ScottPlotBoxPlot` (stacked boxes with numpy quartiles, 1.5 IQR whiskers, fliers, mean line, gold opportunity bands, hover stats), `TrackMapSpec` + `ScottPlotTrackMap` (segments coloured by type, accel width by opportunity, apex labels, gold stars), the detail overlay reuses the strip chart (viridis laps, brake/entry/apex/exit markers); config with thresholds per session and the 95th-percentile auto throttle threshold; Summary/Detail/Map views, corner selector, stats tables |
| Tire Grip tab | done 2026-10-29 | `LineGridSpec` + `ScottPlotLineGrid` (bucket-percentile lines with markers, hover showing the bucket's metric, G and count); pressure/temperature switch and percentile entry; A/B overlay; stats tables (per-corner means and the comparison/detail tables) |
| Save Profile | done 2026-10-29 | merges every tab's channel names (first tab owns a shared key, so Suspension's shock entries win over Roll Gradient's — Python lets the last panel win), the Suspension motion ratios and the Roll Gradient track widths into a `VehicleProfile` saved under logger ID + vehicle for the session the active config edits; the loaded shock calibration is kept only while the entered ratios still derive from it |
| Packaging check | 2026-10-29 | Linux NativeAOT publish: no trimming warnings, 25 MB, parser libraries under `runtimes/linux-x64/native/`, runs with the sample. Windows trimmed single-file publish: 42 MB exe; the parser DLLs must be `ExcludeFromSingleFile` (the bundler swallowed them and extracted them flat where `NativeLoader` could not find them — "Dll was not found" on load); verified on Windows loading a KK-F `.xrk` with a Japanese file name. CI now fails the publish jobs if the libraries are missing from the output |
| Track store + sector kernel | done 2026-10-29 | `Core/Tracks`: `TrackDefinition` (anchor + gate chords in lap order), `TrackYaml`, `TrackStore` (embedded `builtin_tracks.yaml` if present + user `tracks.yaml` beside `profiles.yaml`, nearest-anchor lookup), `SectorTimes` (gate crossings by segment intersection in a local frame, interpolated in time; split table with best sectors, theoretical best and best lap; `GateAcrossPath` for the editor). Verified on a synthetic square lap and on Fuji lap 14 with gates placed at known samples |
| Sectors tab (split-times table + track editor) | done 2026-10-29 | `SectorsViewModel` (track picker, "New track from session" anchored at the first GPS fix and named from the Venue metadata (the track display name for iRacing), nearest-track auto-select on load, gates kept in lap order, dirty/save); `SplitTableControl` custom-drawn (lap rows, sector + lap-time columns, best purple / second-best green, theoretical-best row, row click highlights); `TrackEditorControl` custom-drawn (reference lap fitted to the control, click the line to add a gate, click to select, drag to move along the line) |
| Phase 3 remaining | next | Time/Distance delta-time against sector gates? (not planned); otherwise Phase 4 packaging polish and retiring the Python app |

Lesson from corners: match numpy's operation order exactly (`x * (π/180)`,
not `x * π / 180`; pairwise sums) before loosening a tolerance.

