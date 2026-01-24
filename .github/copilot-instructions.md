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

## CI/CD Debugging Tips

When checking GitHub Actions CI failures:

1. **List recent runs** (include limit 2+ to see both CI and Deploy jobs):
   ```bash
   gh run list --branch <branch> --limit 3 --json databaseId,conclusion,name
   ```

2. **Get failed job logs**:
   ```bash
   gh run view <run-id> --log-failed 2>&1 | tail -80
   ```

3. **Common issues**:
   - Poetry uses `--extras <name>` for `[project.optional-dependencies]` (PEP 621 style)
   - Poetry uses `--with <name>` for `[tool.poetry.group.<name>.dependencies]` (Poetry groups)
   - CI installs via `poetry install --extras dev` to use locked versions from `poetry.lock`
   - If CI uses `pip install .[dev]`, it ignores `poetry.lock` and gets latest allowed versions

## Long-Running Commands

**Be patient with build commands!** The following commands take significant time:

- `poe build_lite_full` - **3-5 minutes**: Compiles C++ (libxrk), builds wheels, executes all notebooks
- `poe build_and_serve_lite` - Same as above plus starts a server
- Notebook execution during builds - Each notebook may take 30-60 seconds

**Do NOT**:
- Assume the command failed just because output is still streaming
- Interrupt commands prematurely - wait for the actual exit code
- Confuse compiler warnings (like `-Wsign-compare`) with errors

**DO**:
- Wait for the command to fully complete before analyzing output
- Look for explicit error messages like `Error:` or non-zero exit codes
- If a build seems stuck, wait at least 2-3 minutes before investigating

**CRITICAL - Avoiding Self-Cancellation**:
- Run the command ONCE with `isBackground: false` and let it complete
- Do NOT run any other terminal commands while waiting - this cancels the running command
- Do NOT use `isBackground: true` for builds where you need to see the output
- Do NOT try to "check" on the command or run `sleep` commands - just wait
- The tool will return all output when the command finishes - be patient