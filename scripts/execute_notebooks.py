#!/usr/bin/env python3
"""Execute all template notebooks to populate their outputs.

This script clears existing outputs and re-executes all notebooks in the
workspace_template directory, saving them in-place with populated outputs.
Used during CI/CD to make notebooks visually interesting on first load.

Notebooks are executed in parallel using ProcessPoolExecutor for faster builds.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor, ExecutePreprocessor


# 5 minute timeout per notebook
EXECUTION_TIMEOUT = 300
KERNEL_NAME = "python3"


def execute_notebook(notebook_path: Path) -> tuple[Path, str | None]:
    """Clear outputs and execute a notebook in-place.

    Args:
        notebook_path: Path to the notebook file to execute.

    Returns:
        Tuple of (notebook_path, error_message). error_message is None on success.
    """
    print(f"Executing: {notebook_path.name}", flush=True)

    try:
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Clear existing outputs
        clear_preprocessor = ClearOutputPreprocessor()
        nb, _ = clear_preprocessor.preprocess(nb, {})

        # Execute the notebook
        execute_preprocessor = ExecutePreprocessor(
            timeout=EXECUTION_TIMEOUT,
            kernel_name=KERNEL_NAME,
        )

        # Execute with cwd set to the notebook's directory (for data file access)
        nb, _ = execute_preprocessor.preprocess(
            nb, {"metadata": {"path": str(notebook_path.parent)}}
        )

        # Write the executed notebook back
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"  ✓ Completed: {notebook_path.name}", flush=True)
        return (notebook_path, None)
    except Exception as e:
        print(f"  ✗ Failed: {notebook_path.name}", flush=True)
        return (notebook_path, str(e))


def main() -> int:
    """Execute all template notebooks in parallel.

    Returns:
        0 on success, 1 on failure.
    """
    # Script is run with cwd=workspace_template
    template_dir = Path.cwd()

    # Find all notebooks
    notebooks = sorted(template_dir.glob("*.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {template_dir}")
        return 1

    print(f"Found {len(notebooks)} notebook(s) to execute in parallel:")
    for nb in notebooks:
        print(f"  - {nb.name}")
    print(flush=True)

    # Execute notebooks in parallel
    failures: list[tuple[Path, str]] = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(execute_notebook, nb): nb for nb in notebooks}

        for future in as_completed(futures):
            notebook_path, error = future.result()
            if error is not None:
                failures.append((notebook_path, error))

    # Report results
    if failures:
        print(f"\n✗ {len(failures)} notebook(s) failed:")
        for notebook_path, error in failures:
            print(f"  - {notebook_path.name}: {error}")
        return 1

    print(f"\n✓ Successfully executed all {len(notebooks)} notebook(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
