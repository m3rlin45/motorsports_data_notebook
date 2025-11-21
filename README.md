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


Licensed under the terms in `LICENSE`.
