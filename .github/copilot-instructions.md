# Copilot Instructions for motorsports_data_notebook

## Package Manager

This project uses **Poetry** for dependency management. Always use `poetry run` to execute Python commands:

```bash
# Correct
poetry run python script.py
poetry run poe <task>
poetry run pytest

# Incorrect - do not use
python script.py
pip install <package>
```

## Common Tasks (via poethepoet)

Use `poetry run poe <task>` for all project tasks:

- `poe check` - Run all linting and type checking
- `poe build_lite_full` - Build wheels and JupyterLite site (executes notebooks during build)
- `poe clean` - Reset workspace with fresh notebooks

## Project Structure

- `src/motorsports_data_notebook/` - Main package source
- `workspace_template/` - Template Jupyter notebooks (executed during CI)
- `workspace/` - Local working directory for notebooks (gitignored)
- `scripts/` - Build and utility scripts
- `build/libxrk/` - Pyodide wheel build directory

## Dependencies

- Add runtime deps to `[project.dependencies]` in pyproject.toml
- Add dev deps to `[project.optional-dependencies.dev]`
- Add JupyterLite/CI deps to `[project.optional-dependencies.jupyterlite]`
- After modifying deps, run `poetry lock && poetry install`

## Notebooks

Template notebooks in `workspace_template/` are:
- Executed during CI to populate outputs before deployment
- Renamed with version suffix (e.g., `basics_v0.3.0.ipynb`) when packaged
- Must be able to run with the sample `.xrz` data file in the same directory
