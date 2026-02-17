# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Driver Consistency Analyzer."""

import sys
from pathlib import Path

sys.setrecursionlimit(5000)

# Get the absolute path to the desktop_app_driver directory
spec_dir = Path(SPECPATH).resolve()
src_dir = spec_dir / "src"
repo_root = spec_dir.parent

# Add src to path for analysis
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(repo_root / "src"))

a = Analysis(
    [str(src_dir / "driver_consistency_analyzer" / "main.py")],
    pathex=[str(src_dir), str(repo_root / "src")],
    binaries=[],
    datas=[
        # Include motorsports_data_notebook package
        (str(repo_root / "src" / "motorsports_data_notebook"), "motorsports_data_notebook"),
    ],
    hiddenimports=[
        # Matplotlib backend
        "matplotlib.backends.backend_tkagg",
        # CustomTkinter
        "customtkinter",
        # tkinterdnd2
        "tkinterdnd2",
        # PIL/Pillow tkinter integration
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused heavy packages
        "IPython",
        "jupyter",
        "jupyterlab",
        "notebook",
        "ipykernel",
        "ipywidgets",
        "qtpy",
        "PyQt5",
        "PyQt6",
        "pytest",
        "black",
        "mypy",
        "plotly",
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
    name="DriverConsistencyAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app, no console
    disable_windowed_traceback=False,
)
