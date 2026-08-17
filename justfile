# Linting and formatting
format:
    uv run black src/ tests/ scripts/ inferno-driving-coach/src/

lint:
    uv run black --check src/ tests/ scripts/ inferno-driving-coach/src/

typecheck:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run mypy src/ scripts/
    rc1=$?
    cd inferno-driving-coach && uv run mypy src/
    rc2=$?
    exit $(( rc1 > rc2 ? rc1 : rc2 ))

format-notebooks:
    uv run nbqa black workspace_template/en/ workspace_template/ja/

lint-notebooks:
    uv run nbqa black --check workspace_template/en/ workspace_template/ja/

typecheck-notebooks:
    uv run nbqa mypy workspace_template/en/ workspace_template/ja/

# Testing
test:
    uv run pytest tests/ --cov=motorsports_data_notebook --cov-report=term-missing --cov-report=html:coverage_html

# Run all checks
check: lint typecheck lint-notebooks typecheck-notebooks test

# Workspace management
clean:
    rm -rf workspace/* && mkdir -p workspace/en workspace/ja workspace/data && cp workspace_template/en/*.ipynb workspace/en/ && cp workspace_template/ja/*.ipynb workspace/ja/ && cp workspace_template/data/*.xrz workspace/data/

run:
    uv run jupyter lab workspace/

run-clean: clean run

# Tire ETL pipeline — extract per-sample telemetry into data/tire_dataset/
tire-etl *ARGS:
    uv run python -m motorsports_data_notebook.tire_etl.cli extract {{ARGS}}

# Parse run-notes into structured JSON via `claude -p`, then re-match them
# to sessions (matching must re-run even on cached parses — session IDs
# change when the extractor is bumped).
tire-notes *ARGS:
    uv run python -m motorsports_data_notebook.tire_etl.cli enrich-notes {{ARGS}}
    uv run python -m motorsports_data_notebook.tire_etl.cli match-notes

# Fetch historical weather for sessions
tire-weather *ARGS:
    uv run python -m motorsports_data_notebook.tire_etl.cli enrich-weather {{ARGS}}

# Ad-hoc SQL query over the dataset via DuckDB
tire-query SQL:
    uv run python -m motorsports_data_notebook.tire_etl.cli query "{{SQL}}"

# Run all three phases in order (full refresh)
tire-refresh: tire-etl tire-notes tire-weather

# Detect candidate broken TPMS sensors (low-variance channels) for human review
tire-sensor-audit *ARGS:
    uv run python -m motorsports_data_notebook.tire_model.cli audit-sensors {{ARGS}}

# Fit the energy-balance tire warmup model from the committed dataset
tire-build-warmup-table *ARGS:
    uv run python -m motorsports_data_notebook.tire_model.cli build-warmup-table {{ARGS}}

# Predict per-corner cold pressures (see `tire-predict --help`)
tire-predict *ARGS:
    uv run python -m motorsports_data_notebook.tire_model.cli predict {{ARGS}}

# Per-corner MAE report vs. notes-recorded cold pressures (uses production model)
tire-predict-validate *ARGS:
    uv run python -m motorsports_data_notebook.tire_model.cli validate {{ARGS}}

# Held-out validation: train without N sessions, predict per-lap T_hot, report residuals
tire-predict-holdout *ARGS:
    uv run python -m motorsports_data_notebook.tire_model.cli holdout {{ARGS}}

# Run the tire pressure calculator web app tests (parity vs. Python fixture)
tire-web-test:
    node --test 'tire_pressure_calculator/web/tests/*.test.mjs'

# Serve the tire pressure calculator web app locally
tire-web-serve:
    @echo "Open http://localhost:8080/tire_pressure_calculator/web/"
    python3 -m http.server 8080

# Wheel building
build-wheel:
    #!/usr/bin/env bash
    set -euo pipefail
    uv build --wheel
    mkdir -p pypi
    rm -f pypi/motorsports_data_notebook-*.whl
    cp dist/motorsports_data_notebook-*.whl pypi/

# JupyterLite site building
# libxrk and libibt wasm wheels are resolved from PyPI by micropip at
# %pip-install time (both publish pyemscripten wheels); the only locally
# hosted wheel is the project's own (build-wheel), since
# motorsports-data-notebook is not published to PyPI.
_clean-lite-artifacts:
    rm -rf .lite_contents dist

_prepare-lite-contents:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .lite_contents/en .lite_contents/ja .lite_contents/data
    cp workspace_template/data/*.xrz .lite_contents/data/
    VERSION=$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
    for nb in workspace_template/en/*.ipynb; do
        name=$(basename "$nb" .ipynb)
        cp "$nb" ".lite_contents/en/${name}_v${VERSION}.ipynb"
    done
    for nb in workspace_template/ja/*.ipynb; do
        name=$(basename "$nb" .ipynb)
        cp "$nb" ".lite_contents/ja/${name}_v${VERSION}.ipynb"
    done

_execute-lite-notebooks:
    cd .lite_contents && uv run python ../scripts/execute_notebooks.py

_build-lite-site:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf dist .jupyterlite.doit.db
    uv run jupyter lite build --contents .lite_contents --output-dir dist
    rm -rf .lite_contents

serve-lite:
    uv run jupyter lite serve --output-dir dist

_prepare-and-execute-lite: _prepare-lite-contents _execute-lite-notebooks

# Sequential on purpose: just's [parallel] has an ETXTBSY race spawning
# shebang recipes, and build-wheel takes seconds — notebook execution is
# the long pole either way.
build-lite-full: _clean-lite-artifacts build-wheel _prepare-and-execute-lite _build-lite-site

build-and-serve-lite: build-lite-full serve-lite

# Cleanup
clean-build:
    rm -rf pypi/ dist/
