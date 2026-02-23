# Linting and formatting
format:
    uv run black src/

lint:
    uv run black --check src/

typecheck:
    uv run mypy src/

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

# Emscripten SDK setup (one-time, needed for building libxrk Pyodide wheel)
setup-emsdk:
    #!/usr/bin/env bash
    set -euo pipefail
    EMSDK_VERSION=$(uv run pyodide config get emscripten_version)
    [ -d "$HOME/emsdk" ] || git clone https://github.com/emscripten-core/emsdk.git "$HOME/emsdk"
    pushd "$HOME/emsdk"
    git config core.autocrlf false
    git checkout -- .
    git pull
    ./emsdk install "$EMSDK_VERSION"
    ./emsdk activate "$EMSDK_VERSION"
    popd
    rm -rf .pyodide-xbuildenv*
    uv run pyodide xbuildenv install

# Wheel building
build-wheel:
    #!/usr/bin/env bash
    set -euo pipefail
    uv build --wheel
    mkdir -p pypi
    cp dist/motorsports_data_notebook-*.whl pypi/

_download-libxrk:
    #!/usr/bin/env bash
    set -euo pipefail
    LIBXRK_VERSION=$(uv run python -c "import importlib.metadata; print(importlib.metadata.version('libxrk'))")
    rm -rf build/libxrk
    git clone --depth 1 --branch "v$LIBXRK_VERSION" https://github.com/m3rlin45/libxrk.git build/libxrk

_build-libxrk-pyodide:
    #!/usr/bin/env bash
    set -euo pipefail
    cd build/libxrk
    . "$HOME/emsdk/emsdk_env.sh"
    . "$HOME/.cargo/env"
    # Install wasm-opt wrapper to strip flags unsupported by emsdk's older binaryen
    WASM_OPT="$EMSDK/upstream/bin/wasm-opt"
    [ -f "${WASM_OPT}.real" ] || mv "$WASM_OPT" "${WASM_OPT}.real"
    sed 's/\r$//' scripts/wasm-opt-wrapper.sh > "$WASM_OPT"
    chmod +x "$WASM_OPT"
    export RUSTUP_TOOLCHAIN=nightly
    export CARGO_BUILD_TARGET=wasm32-unknown-emscripten
    export RUSTFLAGS="-Zemscripten-wasm-eh"
    uv run --project ../.. pyodide build --exports whole_archive

_copy-libxrk-wheel:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f pypi/libxrk-*.whl
    uv run python scripts/build_lite_wheel.py build/libxrk/dist/*.whl

build-libxrk-pyodide: _download-libxrk _build-libxrk-pyodide _copy-libxrk-wheel

_download-pyodide-deps:
    uv run python scripts/download_pyodide_deps.py

# JupyterLite site building
_prepare-lite-contents:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf .lite_contents dist
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

build-lite-full:
    #!/usr/bin/env bash
    set -euo pipefail
    # Run independent build steps in parallel
    just build-wheel &
    just build-libxrk-pyodide &
    just _download-pyodide-deps &
    just _prepare-lite-contents && just _execute-lite-notebooks &
    wait
    # Build the site from all artifacts
    just _build-lite-site

build-and-serve-lite: build-lite-full serve-lite

# Cleanup
clean-build:
    rm -rf build/ pypi/ dist/
