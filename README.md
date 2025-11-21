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
