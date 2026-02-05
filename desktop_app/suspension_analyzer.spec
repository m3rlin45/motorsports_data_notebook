# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Suspension Analyzer."""

import sys
from pathlib import Path

# Get the absolute path to the desktop_app directory
spec_dir = Path(SPECPATH).resolve()
src_dir = spec_dir / "src"

# Add src to path for analysis
sys.path.insert(0, str(src_dir))

block_cipher = None

a = Analysis(
    [str(src_dir / "suspension_analyzer" / "main.py")],
    pathex=[str(src_dir), str(spec_dir.parent / "src")],
    binaries=[],
    datas=[
        # Include resources if any
        # (str(src_dir / "suspension_analyzer" / "resources"), "suspension_analyzer/resources"),
    ],
    hiddenimports=[
        # CustomTkinter and tkinter dependencies
        "customtkinter",
        "tkinter",
        "tkinter.filedialog",
        "tkinterdnd2",
        "tkwebview2",
        "tkwebview2.tkwebview2",
        # Plotly and visualization
        "plotly",
        "plotly.graph_objects",
        "plotly.subplots",
        "plotly.io",
        # Data processing
        "pandas",
        "numpy",
        "pyarrow",
        "pyarrow.compute",
        # libxrk
        "libxrk",
        "libxrk.base",
        # motorsports_data_notebook
        "motorsports_data_notebook",
        "motorsports_data_notebook.suspension",
        "motorsports_data_notebook.visualization",
        "motorsports_data_notebook.widgets",
        # Standard library
        "threading",
        "pathlib",
        "re",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused heavy packages
        "matplotlib",
        "seaborn",
        "jupyter",
        "jupyterlab",
        "notebook",
        "ipykernel",
        "ipywidgets",
        "pytest",
        "black",
        "mypy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SuspensionAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(src_dir / "suspension_analyzer" / "resources" / "icon.ico")
    if (src_dir / "suspension_analyzer" / "resources" / "icon.ico").exists()
    else None,
)
