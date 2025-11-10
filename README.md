# Motorsports Data Notebook

An easy toolbox to slice and dice data from AiM motorsports loggers in a comfortable Jupyter-based environment. This project uses Poetry for dependency management and ships with a curated data stack: JupyterLab, pandas, NumPy, PyArrow, matplotlib, seaborn, and Plotly.

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

- Launch JupyterLab:

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

- JupyterLab & Notebook (interactive environment)
- ipykernel (dedicated kernel)
- pandas (tabular data)
- numpy (numeric computing)
- pyarrow (Arrow & Parquet I/O)
- matplotlib & seaborn (static visualization)
- plotly (interactive visualization)

## Example: quick data wrangling & plot

Create a new notebook and try:

```python
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

# Fake lap data example
np.random.seed(42)
df = pd.DataFrame({
    "lap": np.arange(1, 11),
    "lap_time_s": np.random.normal(loc=90, scale=2.5, size=10).round(3),
    "max_speed_kph": np.random.normal(loc=180, scale=5, size=10).round(1),
})

# Save/read with Parquet
table = pa.Table.from_pandas(df)
pq.write_table(table, "laps.parquet")
df2 = pq.read_table("laps.parquet").to_pandas()

print(df2.head())

# Simple visualization
plt.figure(figsize=(6, 3))
sns.lineplot(data=df2, x="lap", y="lap_time_s", marker="o")
plt.title("Lap Times")
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.show()
```

## Tips for AiM logger data

- AiM exports can be converted to CSV or directly to Parquet via pandas & PyArrow. For large logs, prefer Parquet for speed & size.
- Consider organizing raw logs under `data/raw/` and processed tables under `data/processed/` with notebooks in `notebooks/`.

## Common commands

```bash
# Sync environment to lockfile
poetry install

# Add a new package
poetry add <package>

# Update packages (watch reproducibility)
poetry update

# Run Python / Jupyter
poetry run python -V
poetry run jupyter lab
```

## Troubleshooting

- Kernel not showing? Re-run the ipykernel install command above.
- On WSL, run all commands inside the Linux filesystem (avoid /mnt/c paths for notebooks if possible for performance).
- Plots not displaying? Confirm you're using the correct kernel and that the cell executed successfully.

---
Licensed under the terms in `LICENSE`.
