# Suspension Analyzer

Standalone Windows desktop application for suspension velocity histogram analysis.

## Features

- **Drag-and-drop** XRK/XRZ file loading
- **Multi-lap analysis** - combine data from multiple laps
- **Session comparison** - compare two sessions side-by-side with grouped bar charts
- **Embedded matplotlib charts** with HiDPI support
- **Configurable motion ratios** per corner
- **Channel name mapping** for different AIM telemetry setups
- **Detailed statistics** with bump/rebound breakdown per wheel

## Windows Build Instructions

### Prerequisites

1. **Python 3.12** - Install from [python.org](https://www.python.org/downloads/) or via winget:
   ```cmd
   winget install Python.Python.3.12
   ```

2. **Poetry** (optional, for development) - Install via pip:
   ```cmd
   pip install poetry
   ```

### Building the Executable

#### Option 1: Quick Build (Recommended)

1. Copy the `desktop_app` folder to your Windows machine (e.g., `C:\SuspensionAnalyzer`)

2. Also copy the main library source:
   ```cmd
   mkdir C:\SuspensionAnalyzer\motorsports_data_notebook_src
   xcopy /E /I src\motorsports_data_notebook C:\SuspensionAnalyzer\motorsports_data_notebook_src\motorsports_data_notebook
   ```

3. Create and activate a virtual environment:
   ```cmd
   cd C:\SuspensionAnalyzer
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```cmd
   pip install customtkinter tkinterdnd2 matplotlib pandas numpy pyarrow libxrk pyinstaller
   ```

5. Run the build:
   ```cmd
   build_exe.bat
   ```

6. The executable will be at `dist\SuspensionAnalyzer.exe`

#### Option 2: Using Poetry (Development)

```cmd
cd desktop_app
poetry install
poetry run python -m suspension_analyzer
```

### Build Script (build_exe.bat)

Create this batch file for easy building:

```batch
@echo off
echo Building Suspension Analyzer...

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install customtkinter tkinterdnd2 matplotlib pandas numpy pyarrow libxrk pyinstaller

REM Build with PyInstaller
pyinstaller --onefile --windowed --name SuspensionAnalyzer ^
    --add-data "motorsports_data_notebook_src;motorsports_data_notebook_src" ^
    --collect-all pandas ^
    --collect-all customtkinter ^
    --hidden-import tkinterdnd2 ^
    --hidden-import matplotlib ^
    src/suspension_analyzer/main.py

echo Build complete! Output: dist\SuspensionAnalyzer.exe
```

## Usage

1. **Load Session A**: Drag an XRK/XRZ file onto the left panel, or click to browse
2. **Select laps**: Check the laps you want to include in the analysis
3. **View histogram**: Analysis runs automatically when laps are selected
4. **Maximize chart**: Click "Maximize" button to view full-screen histogram

### Comparing Sessions

1. Load a second XRK/XRZ file into **Session B** (middle panel)
2. Select laps from both sessions
3. The histogram automatically shows grouped bars comparing both sessions

### Viewing Statistics

1. Click **"Show Statistics"** button in the config panel
2. A popup window shows detailed statistics:
   - Summary statistics (skew, std dev, mean per corner)
   - Velocity range distribution (bump/rebound breakdown)
   - Balance analysis (front/rear, left/right)

### Configuration

- **Motion Ratios**: Set per-corner motion ratios (default: Toyota 86 ZN6)
- **Channel Names**: Update if your AIM setup uses different shock pot channel names

## Architecture

```
src/suspension_analyzer/
├── main.py              # Entry point
├── app.py               # Main CustomTkinter window
├── loader.py            # Session loading without IPython dependency
├── widgets/
│   ├── session_panel.py # File drop + lap selector
│   ├── config_panel.py  # Motion ratios, channel names, status
│   ├── chart_view.py    # Matplotlib embedded display
│   └── stats_panel.py   # Statistics tables
├── analysis/
│   └── multi_lap.py     # Multi-lap velocity aggregation
└── visualization/
    └── comparison.py    # Comparison chart generation
```

## Dependencies

- **customtkinter** - Modern tkinter theme
- **tkinterdnd2** - Drag-and-drop support
- **matplotlib** - Embedded charts with HiDPI support
- **pandas** / **numpy** / **pyarrow** - Data processing
- **libxrk** - AIM telemetry file parsing
- **motorsports-data-notebook** - Analysis library (included as source)

## Troubleshooting

### "Module not found" errors during build

Add `--collect-all <module>` to the PyInstaller command for the missing module.

### Chart appears blurry

The app includes HiDPI detection for Windows. If still blurry, try setting Windows display scaling to 100%.

### Statistics window opens behind main window

Click the window in the taskbar or use Alt+Tab to bring it to front.
