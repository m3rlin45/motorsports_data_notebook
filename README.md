# Inferno Racing Notebook Analysis

An easy toolbox to slice and dice data from AiM motorsports loggers in a comfortable Jupyter-based environment. This project uses **libxrk** for reading AiM `.xrk` files directly.

## Quick start

### Prerequisites

- Python 3.12+
- Poetry 2.x (https://python-poetry.org/docs/#installation)

### Setup (one time)

1. Install dependencies and create the virtual environment:

   ```bash
   poetry install
   ```

2. (If needed) Register the Jupyter kernel for this environment:

   ```bash
   poetry run python -m ipykernel install --user --name motorsports-data-notebook --display-name "Python (motorsports-data-notebook)"
   ```

## Use it

### For Notebook Users (Quick Start) 🏁

**Start here!** Populate the `workspace/` folder - this is your personal sandbox:

```bash
# empty the workspace folder and copy in the starter notebook, then start the kernel
poetry run poe run_clean
```


The `workspace/` directory contains:
- `starter_notebook.ipynb` - Example workflows and tutorials
- Your analysis notebooks (create as many as you need!)
- Your data files (`.xrk`, `.csv`, `.parquet`, etc.)
- **Everything here is git-ignored** - it's yours!

### JupyterLite (Static Web Deployment) 🌐

You can also build and serve this project as a static JupyterLite site that runs entirely in the browser.

#### GitHub Pages Deployment

This project automatically deploys to GitHub Pages:

- **Preview builds**: Every push to `master` deploys to the `github-pages-preview` environment
- **Release builds**: Published GitHub releases deploy to the production `github-pages` environment

To create a release, use the GitHub UI (Releases → Create a new release) or the CLI:
```bash
gh release create v1.0.0 --title "v1.0.0" --notes "Release notes here"
```

#### Local Development

```bash
# Install JupyterLite dependencies (one time)
poetry install --extras jupyterlite

# Build the static site (uses pre-built wheels from pypi/)
poetry run poe build_lite

# Serve locally for testing
poetry run poe serve_lite

# Or do both at once
poetry run poe build_and_serve_lite
```

#### Building Pyodide Wheels (Advanced)

To rebuild all wheels including `libxrk` for Pyodide (required after updating dependencies):

1. Install the Emscripten SDK (one time):
   ```bash
   # Install pyodide-build and get the required Emscripten version
   pip install pyodide-build
   pyodide xbuildenv install 0.27.6
   
   # Clone and install emsdk
   git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
   cd ~/emsdk
   EMSCRIPTEN_VERSION=$(pyodide config get emscripten_version)
   python3 emsdk.py install $EMSCRIPTEN_VERSION
   python3 emsdk.py activate $EMSCRIPTEN_VERSION
   ```

2. Build all wheels and the site:
   ```bash
   poetry run poe build_lite_full
   ```

The built site will be in the `dist/` directory and can be deployed to any static hosting service (GitHub Pages, Netlify, Vercel, etc.). The site includes:
- All notebooks from `workspace_template/`
- All helper functions from the `motorsports_data_notebook` package  
- Pre-built `libxrk` wheel compiled for Pyodide/WebAssembly
- A Python runtime that runs entirely in the browser (via Pyodide)

**Note:** JupyterLite uses Pyodide (Python in WebAssembly) which has some limitations compared to native Python. Not all packages may be available.

## Helper Functions 🛠️

The package includes useful helper functions for common motorsports data analysis tasks:

### `get_best_lap(laps_df)`
Finds the fastest lap from a laps DataFrame by duration.

```python
from motorsports_data_notebook import get_best_lap

best_lap = get_best_lap(laps)
print(f"Best lap time: {best_lap['lap_duration_ms']} ms")
```

### `compute_start_line(lat, lon, ahead_points=100, scale=0.02)`
Computes endpoints for a perpendicular start/finish line at the beginning of a GPS track.

```python
from motorsports_data_notebook import compute_start_line

(lat_a, lon_a), (lat_b, lon_b) = compute_start_line(
    lap_data['GPS Latitude'],
    lap_data['GPS Longitude']
)
```

### `plot_lap_gps(lat, lon, color_channels, width=800, height=800, title=None)`
Creates an interactive Plotly visualization of GPS track data with multi-layer color-coded channels.

**Single channel example:**
```python
from motorsports_data_notebook import plot_lap_gps

fig = plot_lap_gps(
    lat=lap_data['GPS Latitude'],
    lon=lap_data['GPS Longitude'],
    color_channels=[
        (lap_data['speed_kmh'], 'Speed (km/h)', 'Viridis')
    ],
    title='Lap Speed Analysis'
)
fig.show()
```

**Multi-channel example (overlaid layers):**
```python
fig = plot_lap_gps(
    lat=lap_data['GPS Latitude'],
    lon=lap_data['GPS Longitude'],
    color_channels=[
        (lap_data['BrakePress'], 'Brake Pressure', 'Reds'),
        (lap_data['speed_kmh'], 'Speed (km/h)', 'Viridis')
    ],
    title='Speed and Braking Analysis'
)
fig.show()
```

Color channels are layered with the first in the list on top (smallest markers) and the last on bottom (largest markers). Each channel displays its own colorbar.

Licensed under the terms in `LICENSE`.
