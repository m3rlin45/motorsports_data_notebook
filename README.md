# Inferno Racing Notebook Analysis

Analyze AiM motorsports telemetry data in your browser — no software installation required.

## 🏁 Get Started Now

**[Launch the Analysis Tool →](https://analysis-preview.inferno.racing/)**

Upload your `.xrk` or `.xrz` files and start analyzing immediately.

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
- [Poetry 2.x](https://python-poetry.org/docs/#installation)

### Setup

```bash
# Install dependencies
poetry install

# Install dev dependencies (linting, type checking)
poetry install --extras dev

# Start JupyterLab with the workspace
poetry run poe run_clean
```

### Code Quality

This project uses [Black](https://black.readthedocs.io/) for formatting and [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
# Run all checks (linting + type checking for source and notebooks)
poetry run poe check

# Format code
poetry run poe format            # Format source files
poetry run poe format-notebooks  # Format notebooks

# Individual checks
poetry run poe lint              # Check source formatting
poetry run poe typecheck         # Type check source files
poetry run poe lint-notebooks    # Check notebook formatting
poetry run poe typecheck-notebooks  # Type check notebooks
```

These checks run automatically on all PRs and pushes to `master` via GitHub Actions.

### Building the JupyterLite Site

```bash
# Install JupyterLite dependencies
poetry install --extras jupyterlite

# Build and serve locally
poetry run poe build_and_serve_lite
```

### Building Pyodide Wheels

To rebuild `libxrk` for Pyodide (after updating the C++ bindings):

```bash
# Install Emscripten SDK (one time)
pip install pyodide-build
pyodide xbuildenv install 0.27.6

git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk
EMSCRIPTEN_VERSION=$(pyodide config get emscripten_version)
python3 emsdk.py install $EMSCRIPTEN_VERSION
python3 emsdk.py activate $EMSCRIPTEN_VERSION

# Build all wheels and site
poetry run poe build_lite_full
```

---

Licensed under the terms in `LICENSE`.
