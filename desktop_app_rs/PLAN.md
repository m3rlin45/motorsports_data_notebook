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

**Test count:** 34 passing, clippy clean

### Remaining — Parallel Work Plan

Three independent workstreams, all writing to separate files:

#### Agent: charts (inferno-ui)
Files: `src/charts/{mod,colors,summary,detail,track_map}.rs`

- [ ] `colors.rs` — viridis colormap, opportunity gradient (steelblue→gold), segment colors
- [ ] `summary.rs` — 3 stacked box plots (BP centered, TA%, exit speed) using BoxPlot/BoxElem/BoxSpread, A/B comparison, opportunity highlight bands
- [ ] `detail.rs` — 3 linked line plots (throttle, brake, lat G) with viridis per-lap coloring, VLine markers (brake/entry/apex/exit), TA annotation
- [ ] `track_map.rs` — Plot::data_aspect(1.0), segment coloring (red/orange/green), apex markers, top-3 opportunity stars

#### Agent: widgets (inferno-ui)
Files: `src/{theme,widgets/{mod,session_panel,config_panel,corner_selector,stats_window}}.rs`

- [ ] `theme.rs` — color constants (#1a1a2e bg, steelblue, darkorange, gold accents)
- [ ] `session_panel.rs` — file dialog (rfd), drag-drop, lap checkboxes, file name display
- [ ] `config_panel.rs` — channel name text inputs, threshold sliders (corner 0.006, throttle 98%, sustain 500ms), A/B session toggle, profile load/save
- [ ] `corner_selector.rs` — Summary/Detail radio, scrollable corner radio buttons, auto-switch to detail on corner select
- [ ] `stats_window.rs` — egui::Window popup, scrollable Grid tables (entry consistency, corner speed, opportunity ranking, interpretation guide)

#### Me: app integration (inferno-app + inferno-ui wiring)
Files: `inferno-ui/src/lib.rs`, `inferno-app/src/main.rs`

- [ ] App state struct (sessions, analysis state, UI state, config)
- [ ] Background analysis thread (std::thread + mpsc, 300ms debounce)
- [ ] Layout: collapsible top panel → session panels + config | corner selector sidebar | chart area | status bar
- [ ] Wire widgets + charts together, view mode switching
- [ ] Profile persistence on session load
- [ ] Error handling UX (status bar messages)
- [ ] Auto-throttle threshold (95% of peak on load)

### Future (not in this PR)

- [ ] Suspension Velocity tab
- [ ] Tire Grip tab
- [ ] `AnalysisTab` trait when adding second tab

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

End-to-end test (after charts + widgets + integration):
1. `cargo run --release` → dark window opens
2. Drop .xrz file → corners detected, summary box plots render
3. Click corner → detail overlay with lap traces
4. Toggle track map → colored segments
5. Open stats → tables populated
