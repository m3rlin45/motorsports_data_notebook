#!/usr/bin/env bash
# Build a self-contained wheel that bundles motorsports_data_notebook.
#
# Usage: ./build_wheel.sh
#
# The wheel includes both inferno_driving_coach and motorsports_data_notebook
# so there's no need to publish motorsports-data-notebook separately.

set -euo pipefail
cd "$(dirname "$0")"

echo "Copying motorsports_data_notebook into src/..."
cp -r ../src/motorsports_data_notebook src/

cleanup() {
    echo "Cleaning up vendored copy..."
    rm -rf src/motorsports_data_notebook
}
trap cleanup EXIT

echo "Building wheel..."
uv build --out-dir dist

echo "Done."
ls -1 dist/*.whl 2>/dev/null | tail -1
