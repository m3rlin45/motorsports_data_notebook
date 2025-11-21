# Motorsports Data Notebook

An easy toolbox to slice and dice data from AiM motorsports loggers in a comfortable Jupyter-based environment. This project uses Poetry for dependency management and ships with a curated data stack: JupyterLab, pandas, NumPy, PyArrow, matplotlib, seaborn, and Plotly, plus **libxrk** for reading AiM `.xrk` files.

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

**Start here!** Navigate to the `workspace/` folder - this is your personal sandbox:

```bash
cd workspace
poetry run jupyter lab
```

The `workspace/` directory contains:
- `starter_notebook.ipynb` - Example workflows and tutorials
- Your analysis notebooks (create as many as you need!)
- Your data files (`.xrk`, `.csv`, `.parquet`, etc.)
- **Everything here is git-ignored** - it's yours!

### For Developers

- Launch JupyterLab from project root:

  ```bash
  poetry run jupyter lab
  ```

- Or launch the classic Notebook:

  ```bash
  poetry run jupyter notebook
  ```

- In VS Code: open any `.ipynb` file, then pick the kernel "Python (motorsports-data-notebook)".

## What's included

Core packages pinned by Poetry:

- **libxrk** (AiM XRK/XRZ file reader)
- JupyterLab & Notebook (interactive environment)
- ipykernel (dedicated kernel)
- pandas (tabular data)
- numpy (numeric computing)
- pyarrow (Arrow & Parquet I/O)
- matplotlib & seaborn (static visualization)
- plotly (interactive visualization)

## Example: Loading AiM data with libxrk

```python
from libxrk import AIMXRK
import pandas as pd

# Load an AiM .xrk file
xrk = AIMXRK('path/to/your/file.xrk')

# Access data as pandas DataFrames
# (See starter_notebook.ipynb for detailed examples)
```

## Tips for AiM logger data

- Use `libxrk` to directly read `.xrk` and `.xrz` files
- For large datasets, save processed data as `.parquet` for fast reloading
- Keep your raw AiM files and notebooks in `workspace/` - they won't be committed to git

## Common commands

```bash
# Sync environment to lockfile
poetry install

# Add a new package
poetry add <package>

# Update packages
poetry update

# Update libxrk specifically
poetry update libxrk

# Run Python / Jupyter from workspace
cd workspace
poetry run jupyter lab
```

## Troubleshooting

- Kernel not showing? Re-run the ipykernel install command above.
- On WSL, run all commands inside the Linux filesystem (avoid /mnt/c paths for notebooks if possible for performance).
- Plots not displaying? Confirm you're using the correct kernel and that the cell executed successfully.

---
Licensed under the terms in `LICENSE`.
