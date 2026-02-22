#!/bin/bash
# Build Inferno Analyzer Windows exe using Wine
# Requires: wine (with 32-bit support), xvfb, wget
#
# Install dependencies (Ubuntu/Debian):
#   sudo dpkg --add-architecture i386
#   sudo apt update
#   sudo apt install wine32:i386 wine64 xvfb wget
#
# The resulting exe is FULLY SELF-CONTAINED and does NOT require Python to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Wine configuration
export WINEPREFIX="$SCRIPT_DIR/.wine_build"
export WINEARCH=win64
export WINEDEBUG=-all

PYTHON_VERSION="3.12.4"
PYTHON_INSTALLER_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe"

PYTHON_DIR="$WINEPREFIX/drive_c/Python312"
WINE_PYTHON="$PYTHON_DIR/python.exe"

echo "========================================="
echo " Inferno Analyzer Build (Wine)"
echo "========================================="

# Check dependencies
for cmd in wine xvfb-run wget; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        echo "Install with: sudo apt install wine64 wine32:i386 xvfb wget"
        exit 1
    fi
done

# Use wine64 if available, otherwise wine
WINE_CMD="wine"
if command -v wine64 &> /dev/null; then
    WINE_CMD="wine64"
fi

# Wine Python needs valid console handles (stdin/stdout/stderr).
# In non-interactive shells (CI, piped output, etc.), Wine can't obtain
# Windows console handles, causing "init_sys_streams: Invalid handle".
# Use `script` to allocate a pseudo-TTY so Wine gets valid handles.
wine_python() {
    local cmd="$WINE_CMD \"$WINE_PYTHON\""
    local arg
    for arg in "$@"; do
        cmd+=" $(printf '%q' "$arg")"
    done
    script -qc "$cmd" /dev/null
}

# Initialize Wine prefix if needed
if [ ! -d "$WINEPREFIX" ]; then
    echo "Initializing Wine prefix..."
    wineboot --init
    wineserver -w || sleep 5
fi

# Install Python if not present
if [ ! -f "$WINE_PYTHON" ]; then
    echo "Downloading Python ${PYTHON_VERSION} installer..."
    wget -q --show-progress "$PYTHON_INSTALLER_URL" -O "/tmp/python-installer.exe"

    echo "Installing Python (silent mode, includes tkinter)..."
    xvfb-run -a $WINE_CMD "/tmp/python-installer.exe" /quiet \
        TargetDir="C:\\Python312" \
        InstallAllUsers=0 \
        PrependPath=0 \
        Include_pip=1 \
        Include_tcltk=1 \
        Include_test=0 \
        Include_doc=0

    # Wait for installer to finish
    wineserver -w || sleep 10

    rm -f "/tmp/python-installer.exe"

    if [ ! -f "$WINE_PYTHON" ]; then
        echo "Error: Python installation failed"
        exit 1
    fi

    echo "Python installed successfully."
fi

# Check Python works
echo ""
echo "Wine Python version:"
wine_python --version

# Verify tkinter works
echo "Checking tkinter..."
wine_python -c "import tkinter; print('tkinter OK')"

# Install/update pip packages
echo ""
echo "Installing Python packages..."
wine_python -m pip install --upgrade pip
# Pin versions to match main project (pyproject.toml).
# Exception: numpy<2 because numpy 2.x uses crealf() which Wine's ucrtbase.dll
# doesn't implement.
wine_python -m pip install \
    customtkinter \
    tkinterdnd2 \
    "matplotlib>=3.10.7,<4" \
    "pandas>=2,<3" \
    "numpy>=1.26,<2" \
    "pyarrow>=18,<23" \
    "libxrk>=0.10.1" \
    "pyyaml>=6,<7" \
    pyinstaller

# Remove unused pyarrow components to reduce bundle size (~33 MB)
# The apps only use core Arrow tables + compute (arrow.dll, arrow_compute.dll, arrow_python.dll)
echo "Stripping unused pyarrow components..."
PYARROW_DIR="$PYTHON_DIR/Lib/site-packages/pyarrow"
rm -f "$PYARROW_DIR"/arrow_flight.dll "$PYARROW_DIR"/arrow_flight.lib
rm -f "$PYARROW_DIR"/parquet.dll "$PYARROW_DIR"/parquet.lib
rm -f "$PYARROW_DIR"/arrow_substrait.dll "$PYARROW_DIR"/arrow_substrait.lib
rm -f "$PYARROW_DIR"/arrow_dataset.dll "$PYARROW_DIR"/arrow_dataset.lib
rm -f "$PYARROW_DIR"/arrow_acero.dll "$PYARROW_DIR"/arrow_acero.lib
rm -f "$PYARROW_DIR"/_flight.*.pyd "$PYARROW_DIR"/_parquet.*.pyd
rm -f "$PYARROW_DIR"/_dataset.*.pyd "$PYARROW_DIR"/_dataset_parquet.*.pyd
rm -f "$PYARROW_DIR"/_acero.*.pyd "$PYARROW_DIR"/_substrait.*.pyd
rm -f "$PYARROW_DIR"/_orc.*.pyd "$PYARROW_DIR"/_csv.*.pyd "$PYARROW_DIR"/_fs.*.pyd
rm -f "$PYARROW_DIR"/_parquet_encryption.*.pyd
rm -rf "$PYARROW_DIR"/include "$PYARROW_DIR"/src "$PYARROW_DIR"/tests

echo "Packages installed."

# Create build directory in Wine
BUILD_DIR="$WINEPREFIX/drive_c/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/src"
mkdir -p "$BUILD_DIR/motorsports_data_notebook_src"

# Copy source files
echo ""
echo "Copying source files..."
cp -r "$SCRIPT_DIR/src/inferno_analyzer" "$BUILD_DIR/src/"
cp -r "$REPO_ROOT/src/motorsports_data_notebook" "$BUILD_DIR/motorsports_data_notebook_src/"

# Create PyInstaller spec file
cat > "$BUILD_DIR/build.spec" << 'SPECEOF'
# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

a = Analysis(
    ['src/inferno_analyzer/main.py'],
    pathex=['src', 'motorsports_data_notebook_src'],
    binaries=[],
    datas=[
        ('motorsports_data_notebook_src', 'motorsports_data_notebook_src'),
        ('motorsports_data_notebook_src/motorsports_data_notebook/data', 'motorsports_data_notebook/data'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'customtkinter',
        'tkinterdnd2',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython', 'jupyter', 'notebook', 'qtpy', 'PyQt5', 'PyQt6', 'plotly',
        # Build tools (not needed at runtime)
        'Cython', 'setuptools', 'pip', 'pkg_resources',
        '_pyinstaller_hooks_contrib',
        # Unused pyarrow submodules
        'pyarrow.flight', 'pyarrow.parquet', 'pyarrow._dataset',
        'pyarrow._acero', 'pyarrow._substrait', 'pyarrow._orc',
        'pyarrow._csv', 'pyarrow._fs', 'pyarrow._parquet_encryption',
        'pyarrow.dataset', 'pyarrow.acero', 'pyarrow.substrait',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='InfernoAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
SPECEOF

# Run PyInstaller via xvfb
echo ""
echo "Running PyInstaller (this may take several minutes)..."
cd "$BUILD_DIR"

xvfb-run -a script -qc "$WINE_CMD \"$WINE_PYTHON\" -m PyInstaller --clean --noconfirm build.spec" /dev/null

# Check result
if [ -f "$BUILD_DIR/dist/InfernoAnalyzer.exe" ]; then
    mkdir -p "$SCRIPT_DIR/dist"
    cp "$BUILD_DIR/dist/InfernoAnalyzer.exe" "$SCRIPT_DIR/dist/"

    EXE_SIZE=$(du -h "$SCRIPT_DIR/dist/InfernoAnalyzer.exe" | cut -f1)

    echo ""
    echo "========================================="
    echo " BUILD SUCCESSFUL!"
    echo " Output: desktop_app/dist/InfernoAnalyzer.exe"
    echo " Size: $EXE_SIZE"
    echo "========================================="
    echo ""
    echo "This exe is FULLY SELF-CONTAINED."
    echo "It can run on any Windows 10/11 machine WITHOUT Python installed."
else
    echo ""
    echo "========================================="
    echo " BUILD FAILED"
    echo "========================================="
    exit 1
fi
