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
| xunit.v3 (AOT flavour) | 4.0.3 | `xunit.v3.aot.mtp-off`: tests found by a source generator, so the suite also runs NativeAOT-compiled (the tire calculator uses v2). Avalonia.Headless.XUnit is reflection-based and is not used; UI tests call Avalonia.Headless through `Tests/Ui.cs` |
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
`toyota_86_zn6`, `kk_s2`, `kk_f` and `kk_f_v2` carry `shock_calibration`/`track_width_mm`;
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
4. **Packaging** — NativeAOT on Windows and Linux with the parsers linked
   in (the Windows build on a Windows CI runner), release workflow, README;
   retire the CustomTkinter app at parity.

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
just cs-test-aot         # the tests compiled with NativeAOT and run natively (linux-x64)
just cs-publish-win      # trimmed single-file win-x64 exe (cross-compiled, for a quick look; CI builds the NativeAOT one)
just cs-publish-linux    # linux-x64 NativeAOT binary, parsers linked in
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
  mingw) win-x64. `tools/build_native.py` builds both at the versions the Python
  project is locked to: it reads `libxrk` and `libibt` from `uv.lock` and
  fetches each release's tag (`v<version>`), so a dependency bump in
  `pyproject.toml` moves both sides together (since 2026-11-15; before that
  the script pinned commits of its own and the two sides had drifted apart
  by eleven libxrk commits).
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
  built from). The AIM fixture is dumped with the libxrk the project installs
  (`just cs-session-fixture`), the same release the FFI library is built
  from. libxrk hands a channel over at the width the logger wrote it:
  float16 for accelerations and shock pots (`ChannelValueType.Float16`),
  which widens exactly and, as in Python, becomes a float32 channel when
  resampled. Channel order and each channel's first/last/min/max
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
compiled bindings; views are thin: a `ChartHost` binds the view model's spec
to a surface from `ChartSurfaces` and keeps it on the app's theme.

| Piece | State | Notes |
|---|---|---|
| App shell | done 2026-10-29, reworked 2026-11-13 | `SessionViewModel` (open / close, profile resolution by logger + vehicle, lap list with gap to best, Best / Top 103 % / All / None, one lap selection per `LapScope`: top 103 % for the statistical tabs, the best two for the trace overlay), `AnalysisTabViewModel<TParameters, TResult>` (300 ms debounce, one run at a time, a superseded run is cancelled and never displayed, stale-when-hidden, status with a kind, an empty state instead of a stale chart), `MainWindowViewModel` (two slots, tab routing, open/recent/swap, settings), `MainWindow` (menu, session sidebar, status bar with busy indicator, key bindings, window-wide file drop), `cs-run file.xrz [file2.xrz]` loads Session A (and B), `cs-run file.json` shows a fixture |
| Time/Distance tab | done 2026-10-29, reworked 2026-11-13 | `LapSeriesBuilder` extracts the chosen channels per selected lap (with neighbours) from the `LogFile`; channel picker (speed/throttle/brake preselected from the profile, the ticked set remembered per vehicle and kept across loads, panels reorderable); fastest lap is the delta reference. The lap list is the legend (`StripChartSpec.Legend` off): each lap keeps one `LapStyle` (hue, dashed past eight) on every tab. The view keeps its zoom while the same session is redrawn (`StripChartSpec.ViewKey`); the cursor can be pinned and stepped from the keyboard; read-outs wrap into columns when there are more laps than fit a panel |
| Suspension tab | done 2026-10-29, reworked 2026-11-13 | `HistogramGridSpec` + `ScottPlotHistogramGrid` (2×2 bars, range bands, side-by-side A/B, hover label); `ChannelConfigViewModel` (A/B entries with completion, sync, profile fill, reset) reusable by every tab and edited in the vehicle profile window; motion ratios as validated `NumericFieldViewModel`s (a bad entry is reported, never replaced by a default); stats pane with the statistics, symmetry and group-difference tables, each copyable as TSV |
| Roll Gradient tab | done 2026-10-29, reworked 2026-11-13 | `ScatterGridSpec` (track map + scatter panels with fit lines, binned means, annotations) + `ScottPlotScatterGrid` (map hover snaps to the track and highlights the index-aligned samples on every panel through preallocated highlight plottables, equal-units map in the shared `MapStyle`: no axes, a scale bar); config with GPS/lateral/shock channels and track widths, motion ratios shared with the Suspension config as in Python; min |G| entry; roll table |
| Driver tab | done 2026-10-29, reworked 2026-11-13 | `BoxPlotSpec` + `ScottPlotBoxPlot` (stacked boxes with numpy quartiles, 1.5 IQR whiskers, fliers, mean line, gold opportunity bands, hover stats), `TrackMapSpec` + `ScottPlotTrackMap` (neutral outline, braking and acceleration zones coloured over it, accel width by opportunity, apex labels, gold stars), the detail overlay reuses the strip chart (laps in their lap-list styles, brake/entry/apex/exit markers); config with thresholds per session and the 95th-percentile auto throttle threshold derived on the worker thread; Summary/Detail/Map as one switch, a click on a summary column, a map apex or a list row opens that corner's detail (Esc returns); stats tables |
| Tire Grip tab | done 2026-10-29, reworked 2026-11-13 | `LineGridSpec` + `ScottPlotLineGrid` (bucket-percentile lines with markers, hover showing the bucket's metric, G and count); pressure/temperature switch and validated percentile entry, both remembered between runs; A/B overlay; stats tables (per-corner means and the comparison/detail tables) |
| Save Profile | done 2026-10-29 | the vehicle profile window's Save button: merges every tab's channel names (first tab owns a shared key, so Suspension's shock entries win over Roll Gradient's — Python lets the last panel win), the Suspension motion ratios and the Roll Gradient track widths into a `VehicleProfile` saved under logger ID + vehicle for the session the active config edits; the loaded shock calibration is kept only while the entered ratios still derive from it |
| Packaging check | 2026-10-29 | Linux NativeAOT publish: no trimming warnings, 25 MB, parser libraries under `runtimes/linux-x64/native/`, runs with the sample. Windows trimmed single-file publish: 42 MB exe; the parser DLLs must be `ExcludeFromSingleFile` (the bundler swallowed them and extracted them flat where `NativeLoader` could not find them — "Dll was not found" on load); verified on Windows loading a KK-F `.xrk` with a Japanese file name. CI now fails the publish jobs if the libraries are missing from the output |
| NativeAOT packaging + AOT-compiled tests | 2026-11-14 | The parsers are linked into a NativeAOT publish as static libraries (`tools/build_native.py` keeps the host build's `.a`/`.lib` under `native/static/<rid>/`; `Directory.Build.targets` adds `DirectPInvoke` + `NativeLibrary` and drops `runtimes/` from the publish). Avalonia's graphics stack exists as shared libraries only (SkiaSharp, HarfBuzzSharp and, on Windows, ANGLE), and an operating system loads a shared library from a file: with `-p:InfernoSelfExtract=true` they are embedded in the executable and `App/NativeBundle.cs` unpacks them at start-up to a per-user cache folder named after their hashes (checked against the hashes on every start), loads them and answers the importing assemblies' lookups with them, so the publish is one file. Linking them in statically instead would need static builds of Skia, HarfBuzz and ANGLE (a third-party prebuilt package exists; building them ourselves is a Chromium-style build per library) and is left for later. The test project publishes the same way and CI runs it as a native executable on Windows and Linux before publishing the app, then checks the publish contents (`tools/check_publish.py`) and, on Windows, that the app starts on the sample session. Linux: one 42 MB executable, 322 tests pass AOT-compiled. Found by running the tests this way: saving a captured frame could crash in the PNG encoder (Avalonia.Skia encodes from a pixmap that does not keep its bitmap alive; optimised code let the finaliser free it), fixed in the tests with `SaveRender`; the app's own Save Image holds the bitmap in a `using`. View-model tests now run on the headless UI thread like the UI tests, which removes a race between a debounced run and the test thread (an intermittent index-out-of-range in CI) From 1.3.1 the workflow also runs on a published GitHub release and attaches the two executables to it as `InfernoAnalyzer-CS.exe` and `InfernoAnalyzer-CS-Linux`, beside the Python app's `InfernoAnalyzer.exe` / `InfernoAnalyzer-Linux`. |
| Windows: window stopped drawing after Time/Distance → Sectors | fixed 2026-11-15 | Found on the first Windows desktop run of the NativeAOT build: after a tab switch away from Time/Distance (or selecting a single lap there) the window stopped updating while the app carried on. The strip chart reacted to ScottPlot's `AxisLimitsChanged` by invalidating its overlay, and ScottPlot raises that from inside its render, which on Windows is Avalonia's compositor thread: "The calling thread cannot access this object", thrown in the render loop, where Avalonia logs it and drops the frame. Two changes: charts no longer run ScottPlot on the compositor's thread at all (`UiThreadPlot` renders on the UI thread into a recorded picture that the compositor replays; before, the UI thread also changed plots while the compositor could be drawing them, without ScottPlot's lock), and the reaction to the event is posted rather than run inside the draw. Nothing showed it before because the headless tests render on the UI thread and Avalonia's log went to a trace nobody reads. Avalonia's logger now ends in `AvaloniaLog`: an exception Avalonia caught is thrown again (on the render thread that ends the process, with the crash log and a message box; not for rendering-backend selection, where Avalonia catches the backends that are unavailable and moves to the next, which goes to `analyzer_platform.log`), other errors are logged and shown, any of them fails a test, and tests count ScottPlot renders across a composed frame and render a plot from another thread |
| Driver tab: small track map | 2026-11-15 | Under the corner list, a compact map (`TrackMapSpec.Compact`: no legend or scale bar, no pan or zoom) with a dot at every apex and, per corner, the stretch from where its braking starts to where its acceleration ends as a `TrackMapHighlightSpec`. It shows the corner under the pointer in the box plots, otherwise the one selected in the list; a click on a dot opens that corner. Charts report the item under the pointer (`IChartSurface.ItemHovered`, raised by the box plot per column) and can emphasise one (`HighlightItem`; the map shows or hides prebuilt plottables, so following the pointer is a redraw, not a rebuild) |
| Track regions (Suspension, Roll Gradient, Tire Grip) | 2026-11-15 | A region column on the three statistical tabs: tick corners and sectors and the analysis covers only the parts of the selected laps inside them (their union; nothing ticked is whole laps), with the whole-lap result kept as a reference (an outline over the Suspension histograms, a grey fit line on Roll Gradient, a grey curve on Tire Grip; "Show whole lap behind" turns it off; one session only — a two-session comparison draws no reference, the sessions are compared with each other). `Core/Tracks/TrackRegion`: a region is the stretch between two geographic gates — a corner's are placed across the reference lap where its braking starts and its acceleration ends, a sector's are the saved track's — and every lap of either session is cut where its own GPS path crosses them (`TrackRegions.WindowsByLap`), so nothing depends on lap distances agreeing. The three multi-lap kernels take `windowsByLap` and cut after smoothing/differentiating over the whole lap (not in Python, no parity fixture). `RegionsViewModel` (one, on the main window, shared like the lap selection) lists them: corners detected once per Session A from its top laps with the Driver tab's channels, sectors from the saved track, refreshed when a track is saved (ticked corners kept). `RegionPanelView` is the column, with a compact map (ticked stretches coloured, row hover highlighted, a click on a corner's dot ticks it). Not done: regions on the Driver tab (it is per corner already) and on Time/Distance (zooming does that). |
| Region selector in the sidebar | 2026-11-15 | The region list moved out of the three statistical tabs into a card under the lap lists (file → laps → which part of the track), one selection for every tab. A row per corner and per sector, like the laps' rows (tick box, name, and for a corner its direction and its speed at the slowest point; a double click ticks that one alone; "Whole lap" is lit when nothing is ticked), over a map that shows where they are (numbered corners, ticked stretches in `ChartColors.Region`, the amber of the sector gates; always shown). The card is as tall as its rows need while every loaded session keeps its header and about six laps (`MainWindow.ArrangeSidebar`); on a shorter sidebar the card gives way and its rows scroll. A tab says what it does with the ticks through `AnalysisTabViewModel.RegionUse` (None / Filter / Focus) and reads them with `RegionSnapshot()`; on a `None` tab the rows are dimmed and a line says the tab shows whole laps. What a displayed result covers is a tag with a clear button beside the tab title (`RegionTagView`, from `RegionLabel`: the computed selection, not the live ticks) and a caption line composed into copied/saved chart images (`ChartHost.Caption`), since the toolbar is not part of them. "Whole-lap reference" is a checkbox in the three statistical toolbars, enabled for one session with regions ticked. Corner identity: the list and the Driver tab share one detection (`DriverConsistencyCache`, the tab's Session A channels and thresholds, the session's default laps); a re-listing keeps ticks by place (same region, or a corner within 30 m of the ticked apex). Driver Consistency is a `Filter` tab too: its box plots, tables and corner list show the ticked corners and the corners inside a ticked sector (matched by apex position, `RegionSelection.CoversCorner`: a corner's stretch starts with its braking and can reach back past the previous apex, so containment would pull the neighbour in); the list stays as the detail picker and the maps keep every corner. Time/Distance is the `Focus` tab: the traces are never cut; the ticked regions are shaded behind every panel (`StripChartSpec.Bands`) where the reference lap (Session A's fastest ticked) drives through them, and the initial view runs from the first to the last of them with an 8 % margin (`StripChartBuilder.Build(…, focus)`; the view key carries the region names, so another choice is another view and Fit returns to it, while `DataKey` lets a pinned cursor that is inside the new view stay). Against time the bands are the reference lap's seconds; other laps drift from them by their delta, as they already do on that axis. Sectors stays unaffected (it defines the sectors). The sidebar map also shows where the pointer on the active tab is on the track: a tab raises `TrackPositionChanged` with the fraction of the lap (the Time/Distance cursor, the Driver detail cursor, the apex of the corner whose column is hovered in the Driver summary) and the map moves one marker along its outline (`TrackMapSpec.Cursor`, `IChartSurface.SetCursor`), without rebuilding. Not yet: Roll Gradient's hovered samples and the Sectors table. |
| Track store + sector kernel | done 2026-10-29 | `Core/Tracks`: `TrackDefinition` (anchor + gate chords in lap order), `TrackYaml`, `TrackStore` (embedded `builtin_tracks.yaml` if present + user `tracks.yaml` beside `profiles.yaml`, nearest-anchor lookup), `SectorTimes` (gate crossings by segment intersection in a local frame, interpolated in time; split table with best sectors, theoretical best and best lap; `GateAcrossPath` for the editor). Verified on a synthetic square lap and on Fuji lap 14 with gates placed at known samples |
| Sectors tab (split-times table + track editor) | done 2026-10-29, reworked 2026-11-13 | `SectorsViewModel` (track picker, "New track" anchored at the first GPS fix and named from the Venue metadata (the track display name for iRacing), nearest-track auto-select on load, gates kept in lap order, rename and gate names as edits of their own, undo, dirty/save with a prompt before unsaved edits are lost); `SplitTableControl` custom-drawn (lap rows, sector + lap-time columns right-aligned in tabular figures, best purple and bold / second-best green and semibold, theoretical-best row, row click or Up/Down highlights, copy as TSV); `TrackEditorControl` custom-drawn (reference lap fitted to the control, click the line to add a gate, click to select, drag or arrow keys to move along the line, Delete removes). Both take their colours and font from the theme and expose an automation peer |
| Phase 3 remaining | next | Time/Distance delta-time against sector gates? (not planned); otherwise Phase 4 packaging polish and retiring the Python app |

Lesson from corners: match numpy's operation order exactly (`x * (π/180)`,
not `x * π / 180`; pairwise sums) before loosening a tolerance.

### UI review pass — 2026-11-13

A four-lens review (C#/Avalonia idiom, interaction, visual design, a
pragmatist's triage) of the Phase 3 app; its findings were folded back into
the commits that introduced each piece, so the stack reads as built this way.
What changed, by layer:

- **Charting.** `ChartTheme` carries the surfaces, ink, type ramp and the
  values of the `ChartColors` tokens; specs name colours by token
  (`@session-a`, `@lap-3`, `@braking`) so one spec draws correctly in light,
  dark and high-contrast themes. The lap hues are the dataviz reference
  palette stepped per surface (every slot ≥ 3:1 on the data area; re-run its
  validator after any edit). `IChartSurface.ResetView`; `ChartPoint.ItemId`
  for clickable chart items.
- **Charting.ScottPlot.** `ScottPlotSurface<TSpec>` / `ScottPlotGridSurface<TSpec>`
  own the spec guard, theme re-render, events, plot grid and reset;
  `PlotStyle` is the one chart style (left/bottom frame in the grid colour,
  no minor ticks, flat label boxes, the UI font via `ScottPlotFonts`,
  `Font.Automatic` so non-Latin session names draw); `MapStyle` is the one
  map style. The strip chart's crosshair, markers, read-outs and title tags
  are an Avalonia overlay (`StripCursorOverlay`), so moving the cursor never
  re-rasterises the traces. ScottPlot's Avalonia control renders in
  device-independent pixels with `DisplayScale` fixed at 1, which is what
  lets the overlay map data to pixels from the axis limits and the fixed
  padding alone. `DataAreaInput` also removes ScottPlot's double-click
  benchmark, middle-click autoscale, keyboard pan and context menu.
- **App.** `Styles/AppStyles.axaml` holds the tokens (page / card / plot
  surfaces, text, timing colours), the type ramp (12 body, 12 semibold
  section, 14 semibold title) and the classes views use instead of inline
  attributes; `FluentTheme` runs in Compact density. Windows and macOS use
  the system UI font, Linux the bundled Inter (the headless tests pin Inter).
  Tab views are created once and kept (`TabViewLocator`); a tab contributes
  its own View-menu commands and shortcuts (`TabCommand`). Settings live in
  `analyzer_settings.json` beside `profiles.yaml` (window size, sidebar,
  theme, last tab and folder, recent files, per-tab view state, ticked
  channels per vehicle). Unhandled exceptions go to `analyzer_crash.log`
  and, on the UI thread, a dialog. On Windows the window asks for Mica and
  puts the menu in the title bar (View > Menu in Title Bar turns that off).

Not verified from Linux, check on a Windows run: the Segoe UI rendering,
Mica, and the menu-in-title-bar layout (drag area, caption-button spacing).
The app icon (`App/Assets`, drawn by `tools/make_icon.py`) is a placeholder.

Left out on purpose: converting the stacked strip-chart panels to a single
ScottPlot `Multiplot` (the overlay removed the cost it was meant to address);
`[ObservableProperty]` on partial properties (cosmetic).

