# Inferno Racing Notebook Analysis

Analyze AiM motorsports telemetry data in your browser — no software installation required.

## 🏁 Get Started Now

**[Launch the Analysis Tool →](https://analysis-preview.inferno.racing/)**

Upload your `.xrk` or `.xrz` files and start analyzing immediately.

**[Tire Pressure Calculator →](https://tires.inferno.racing/)**

Cold tire pressures from target hot pressure and temperature — manually via Gay-Lussac, or predicted per corner by the fitted tire warmup model (track, car, tire compound, condition, target lap time).

---

## How It Works

This tool runs entirely in your browser using [JupyterLite](https://jupyterlite.readthedocs.io/) and [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly).

### Storage & Privacy

- **Your data stays local**: All files and computations happen in your browser — nothing is uploaded to any server
- **Browser storage**: Your notebooks and data files are saved in your browser's IndexedDB storage
- **Persistent between sessions**: Your work persists until you clear browser data

### Troubleshooting

If you experience issues (notebooks not loading, kernel crashes, etc.), try clearing your browser storage:

1. In JupyterLite, go to **Help** → **Clear Browser Data**
2. Refresh the page

---

## Included Notebooks

### 📊 Basics
Load telemetry data, view lap times, and visualize speed and driver inputs on GPS track maps.

### 🛞 Tire Analysis
Visualize tire temperature distribution across all four tires with heatmaps, correlated with speed, G-forces, and driver inputs.

### 🏎️ Track Analysis
Automatically detect corners, braking zones, and acceleration zones from GPS data. View color-coded track segmentation.

### 📈 Consistency Analysis
Measure lap-to-lap variation in braking points, corner speeds, and throttle application with box plots and summary statistics.

---

## Local Development

For contributors or advanced users who want to run locally or modify the code.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [just](https://github.com/casey/just#installation)

### Setup

```bash
# Install dependencies
uv sync --extra app

# Install dev dependencies (linting, type checking)
uv sync --extra app --group dev

# Start JupyterLab with the workspace
just run-clean
```

### Code Quality

This project uses [Black](https://black.readthedocs.io/) for formatting and [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
# Run all checks (linting + type checking for source and notebooks)
just check

# Format code
just format            # Format source files
just format-notebooks  # Format notebooks

# Individual checks
just lint              # Check source formatting
just typecheck         # Type check source files
just lint-notebooks    # Check notebook formatting
just typecheck-notebooks  # Type check notebooks
```

These checks run automatically on all PRs and pushes to `master` via GitHub Actions.

### Building the JupyterLite Site

```bash
# Install JupyterLite dependencies
uv sync --extra app --extra jupyterlite

# Build and serve locally
just build-and-serve-lite
```

### Building Pyodide Wheels

To rebuild `libxrk` for Pyodide (after updating the C++ bindings):

```bash
# Install Emscripten SDK (one time)
uv sync --extra app --extra jupyterlite
just setup-emsdk

# Build all wheels and site
just build-lite-full
```

---

Licensed under the terms in `LICENSE`.
