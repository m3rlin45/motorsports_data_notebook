#!/bin/bash
# Build Suspension Analyzer Windows exe using Wine
# Requires: wine, xvfb, wget
#
# Install dependencies (Ubuntu/Debian):
#   sudo dpkg --add-architecture i386
#   sudo apt update
#   sudo apt install wine64 wine32 xvfb wget cabextract
#
# The resulting exe is FULLY SELF-CONTAINED and does NOT require Python to run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Wine configuration
export WINEDEBUG=-all
export WINEPREFIX="$SCRIPT_DIR/.wine_build"
export WINEARCH=win64

PYTHON_VERSION="3.12.4"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe"
WINE_PYTHON="$WINEPREFIX/drive_c/Python312/python.exe"

echo "========================================="
echo " Suspension Analyzer Build (Wine)"
echo "========================================="

# Check dependencies
for cmd in wine xvfb-run wget; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        echo "Install with: sudo apt install wine64 wine32 xvfb wget"
        exit 1
    fi
done

# Initialize Wine prefix if needed
if [ ! -d "$WINEPREFIX" ]; then
    echo "Initializing Wine prefix..."
    wineboot --init 2>/dev/null || true
    # Wait for wineserver to finish
    wineserver -w 2>/dev/null || sleep 5
fi

# Install Python if not present
if [ ! -f "$WINE_PYTHON" ]; then
    echo "Downloading Python ${PYTHON_VERSION}..."
    wget -q --show-progress "$PYTHON_URL" -O "/tmp/python-installer.exe"

    echo "Installing Python (this may take a few minutes)..."
    # Use xvfb-run to handle GUI installer
    # Install to C:\Python312 for simpler path
    xvfb-run -a wine /tmp/python-installer.exe /quiet \
        InstallAllUsers=0 \
        TargetDir='C:\Python312' \
        PrependPath=1 \
        Include_pip=1 \
        Include_test=0 \
        2>/dev/null || true

    wineserver -w 2>/dev/null || sleep 10
    rm -f /tmp/python-installer.exe

    if [ ! -f "$WINE_PYTHON" ]; then
        echo "Error: Python installation failed."
        echo "Try installing Python manually in Wine."
        exit 1
    fi
    echo "Python installed successfully."
fi

# Check Python works
echo "Wine Python version:"
wine "$WINE_PYTHON" --version 2>/dev/null || {
    echo "Error: Wine Python not working"
    exit 1
}

# Install/update pip packages
echo ""
echo "Installing Python packages..."
wine "$WINE_PYTHON" -m pip install --upgrade pip -q 2>/dev/null
wine "$WINE_PYTHON" -m pip install -q \
    customtkinter \
    tkinterdnd2 \
    matplotlib \
    pandas \
    numpy \
    pyarrow \
    libxrk \
    pyinstaller \
    2>/dev/null

echo "Packages installed."

# Create build directory in Wine
BUILD_DIR="$WINEPREFIX/drive_c/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/src"
mkdir -p "$BUILD_DIR/motorsports_data_notebook_src"

# Copy source files
echo ""
echo "Copying source files..."
cp -r "$SCRIPT_DIR/src/suspension_analyzer" "$BUILD_DIR/src/"
cp -r "$REPO_ROOT/src/motorsports_data_notebook" "$BUILD_DIR/motorsports_data_notebook_src/"

# Create PyInstaller spec file for Wine
cat > "$BUILD_DIR/build.spec" << 'SPECEOF'
# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

a = Analysis(
    ['src/suspension_analyzer/main.py'],
    pathex=['src', 'motorsports_data_notebook_src'],
    binaries=[],
    datas=[('motorsports_data_notebook_src', 'motorsports_data_notebook_src')],
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'customtkinter',
        'tkinterdnd2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'notebook', 'qtpy', 'PyQt5', 'PyQt6'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SuspensionAnalyzer',
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

# Try running PyInstaller with xvfb to handle subprocess issues
xvfb-run -a wine "$WINE_PYTHON" -m PyInstaller \
    --clean \
    --noconfirm \
    build.spec \
    2>&1 | tee pyinstaller.log

# Check result
if [ -f "$BUILD_DIR/dist/SuspensionAnalyzer.exe" ]; then
    mkdir -p "$SCRIPT_DIR/dist"
    cp "$BUILD_DIR/dist/SuspensionAnalyzer.exe" "$SCRIPT_DIR/dist/"

    EXE_SIZE=$(du -h "$SCRIPT_DIR/dist/SuspensionAnalyzer.exe" | cut -f1)

    echo ""
    echo "========================================="
    echo " BUILD SUCCESSFUL!"
    echo " Output: desktop_app/dist/SuspensionAnalyzer.exe"
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
    echo ""
    echo "Check pyinstaller.log for details:"
    echo "  cat $BUILD_DIR/pyinstaller.log"
    exit 1
fi
