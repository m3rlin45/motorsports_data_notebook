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

# Parse run-notes into structured JSON via `claude -p`
tire-notes *ARGS:
    uv run python -m motorsports_data_notebook.tire_etl.cli enrich-notes {{ARGS}}

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

# Emscripten SDK setup (one-time, needed for building libxrk Pyodide wheel)
setup-emsdk:
    #!/usr/bin/env bash
    set -euo pipefail
    # pyodide-build needs pip on PATH for xbuildenv install
    uv pip install pip
    rm -rf .pyodide-xbuildenv*
    uv run pyodide xbuildenv install
    EMSDK_VERSION=$(uv run pyodide config get emscripten_version)
    [ -d "$HOME/emsdk" ] || git clone https://github.com/emscripten-core/emsdk.git "$HOME/emsdk"
    pushd "$HOME/emsdk"
    git config core.autocrlf false
    git checkout -- .
    git pull
    ./emsdk install "$EMSDK_VERSION"
    ./emsdk activate "$EMSDK_VERSION"
    popd

# Wheel building
build-wheel:
    #!/usr/bin/env bash
    set -euo pipefail
    uv build --wheel
    mkdir -p pypi
    rm -f pypi/motorsports_data_notebook-*.whl
    cp dist/motorsports_data_notebook-*.whl pypi/

# Download a library source for Pyodide cross-compilation
_download-pyodide-lib name repo:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(uv run python -c "import importlib.metadata; print(importlib.metadata.version('{{name}}'))")
    rm -rf build/{{name}}
    git clone --depth 1 --branch "v$VERSION" https://github.com/{{repo}}.git build/{{name}}

# Build a library for Pyodide (wasm32)
_build-pyodide-lib name:
    #!/usr/bin/env bash
    set -euo pipefail
    cd build/{{name}}
    rm -rf .pyodide-xbuildenv*
    . "$HOME/emsdk/emsdk_env.sh"
    . "$HOME/.cargo/env"
    # Install wasm-opt wrapper to strip flags unsupported by emsdk's older binaryen
    WASM_OPT="$EMSDK/upstream/bin/wasm-opt"
    [ -f "${WASM_OPT}.real" ] || mv "$WASM_OPT" "${WASM_OPT}.real"
    sed 's/\r$//' ../../scripts/wasm-opt-wrapper.sh > "$WASM_OPT"
    chmod +x "$WASM_OPT"
    export RUSTUP_TOOLCHAIN=nightly
    export CARGO_BUILD_TARGET=wasm32-unknown-emscripten
    export RUSTFLAGS="-Zemscripten-wasm-eh"
    uv pip install --project ../.. pip
    uv run --project ../.. pyodide build --exports whole_archive

# Copy a built Pyodide wheel to pypi/ (stripping dependencies)
_copy-pyodide-lib-wheel name:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f pypi/{{name}}-*.whl
    uv run python scripts/build_lite_wheel.py build/{{name}}/dist/*.whl

# Build individual libraries for Pyodide
build-libxrk-pyodide: (_download-pyodide-lib "libxrk" "m3rlin45/libxrk") (_build-pyodide-lib "libxrk") (_copy-pyodide-lib-wheel "libxrk")
build-libibt-pyodide: (_download-pyodide-lib "libibt" "m3rlin45/libibt") (_build-pyodide-lib "libibt") (_copy-pyodide-lib-wheel "libibt")

# Download a pre-built Pyodide wheel from a GitHub release
_download-pyodide-lib-release name repo:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(uv run python -c "import importlib.metadata; print(importlib.metadata.version('{{name}}'))")
    echo "Downloading {{name}} v$VERSION from {{repo}}..."
    rm -rf build/{{name}}/dist
    mkdir -p build/{{name}}/dist
    gh release download "v$VERSION" -R {{repo}} -p "*pyodide*wasm32.whl" -D build/{{name}}/dist --clobber
    rm -f pypi/{{name}}-*.whl
    # If multiple Pyodide wheels exist (e.g. cp312 + cp313), pick the latest
    WHEEL=$(ls -v build/{{name}}/dist/*.whl | tail -1)
    uv run python scripts/build_lite_wheel.py "$WHEEL"

# Download pre-built Pyodide wheels for all libraries
[parallel]
build-pyodide-libs: (_download-pyodide-lib-release "libxrk" "m3rlin45/libxrk") (_download-pyodide-lib-release "libibt" "m3rlin45/libibt")

_download-pyodide-deps:
    uv run python scripts/download_pyodide_deps.py

# JupyterLite site building
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

[parallel]
_build-lite-parallel: build-wheel build-pyodide-libs _download-pyodide-deps _prepare-and-execute-lite

build-lite-full: _clean-lite-artifacts _build-lite-parallel _build-lite-site

build-and-serve-lite: build-lite-full serve-lite

# Cleanup
clean-build:
    rm -rf build/ pypi/ dist/
