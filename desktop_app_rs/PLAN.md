# Inferno Analyzer — Rust Rewrite Plan

Rewriting the Driver Consistency tab of the Python desktop app in Rust using egui.

## Architecture

```
desktop_app_rs/
  crates/
    inferno-core/   # Data model + analysis (zero UI deps)
    inferno-ui/     # egui widgets + chart views
    inferno-app/    # eframe entry point (binary)
```

**Dependencies:** libxrk + libibt from GitHub (arrow feature), eframe 0.33, egui_plot 0.33, arrow 57, rayon.

**Migration escape hatch:** inferno-core has no UI deps. If egui_plot proves insufficient, swap to Tauri + Plotly.js by replacing only inferno-ui and inferno-app.

## Progress

### Done

- [x] **Phase 0: Workspace scaffold** — 3-crate workspace, dark-themed egui window (1300×900), libxrk+libibt compile
- [x] **Phase 1: Core data model** — Session/Channel/Lap types, XRK+IBT loading via Arrow, derived channels (speed_kmh, distance_m), lap filtering, resampling, LapData extraction
- [x] **Phase 1c: Vehicle profiles** — YAML persistence compatible with Python app, logger ID mapping
- [x] **Phase 2: Analysis algorithms** — math utils, corner detection (GPS→XY→curvature), zone detection (grid voting), throttle acceptance, full driver consistency pipeline with rayon parallelism
- [x] **Phase 3: Widgets** — theme colors, session panel (file dialog + drag-drop + lap checkboxes), config panel (channel names + thresholds + profile save), corner selector (summary/detail radio + corner list), stats window (grid tables)
- [x] **Phase 4: Charts** — viridis colormap + colors, summary box plots (BP/TA/exit speed + A/B comparison + opportunity bands), detail line plots (throttle/brake/lat_g + viridis per-lap + VLine markers + TA annotation), track map (segment coloring + apex markers + opportunity diamonds)
- [x] **Phase 5: App integration** — InfernoApp state, background analysis (std::thread + mpsc + 300ms debounce + panic catch), layout (collapsible top panel + corner sidebar + chart area + status bar), widget/chart wiring, profile auto-load, auto-throttle threshold (95% of peak), save profile, Statistics popup, Show/Hide toggle
- [x] **Phase 5.5: Integration tests** — 20 integration tests against real XRZ data (86 only, tracked in repo), 4 tiers: regression (CAN channels, lap-relative distance), data integrity (monotonic distance, corner data completeness), invariants (corner distances, segment refs, TA bounds), edge cases (single lap, missing channels, high threshold)
- [x] **Best/Top 103% lap selection** — quick-select buttons matching Python app behavior
- [x] **Snapshot tests** — 6 egui_kittest visual regression tests (summary, detail, track map × dark/light themes)
- [x] **CI/CD** — `ci-rust.yml` (clippy + fmt + unit/integration tests + snapshot tests), `build-desktop-rs.yml` (Linux + Windows release builds + release attachment), Python CI path-ignore for `desktop_app_rs/`

**Test count:** 34 unit + 20 integration + 6 snapshot = 60 passing, clippy clean

- [x] **Suspension Velocity tab** — Full histogram analysis (velocity from shock pot displacement, motion ratios, zero-centered binning, skewness/kurtosis), 2×2 histogram chart with velocity range shading, suspension config panel (shock channels, motion ratios, velocity thresholds), suspension stats window (balance analysis), A/B comparison, tab switching in InfernoApp

- [x] **Tire Grip tab** — Bucketed percentile analysis (total G vs tire pressure/temperature), 2×2 line+marker chart, pressure/temperature mode toggle, tire grip config panel (TPMS channels, percentile), tire grip stats window, integration tests with real TPMS data

**Test count:** 53 unit + 33 integration + 10 snapshot = 96 passing

### Future

## Key Design Decisions

- **Arrow as data backbone** — channels stored as RecordBatch, zero-copy `&[f64]` slices for algorithms
- **Git deps** — libxrk v0.11.1, libibt v0.0.4 from GitHub (not local paths)
- **egui_plot 0.33** — native BoxPlot/BoxElem, gradient_color, link_axis/link_cursor
- **rayon for per-lap parallelism** — most expensive step in analysis pipeline
- **std::thread for background analysis** — no async needed, mpsc for results

## Verification

Each commit must pass:
```bash
cargo build && cargo clippy && cargo test
```

End-to-end test checklist:
1. `cargo run --release` → dark window opens
2. Drop .xrz file → corners detected, summary box plots render
3. Click corner → detail overlay with lap traces
4. Toggle track map → colored segments
5. Open stats → tables populated
6. Load Session B → A/B comparison boxes appear
7. Adjust thresholds → analysis re-runs after debounce
8. Save Profile → writes to ~/.config/motorsports_data_notebook/profiles.yaml
