# Suspension Analyzer

Standalone Windows desktop application for suspension velocity histogram analysis.

**The built executable is fully self-contained and runs on any Windows 10/11 machine without Python installed.**

## Features

- **Drag-and-drop** XRK/XRZ file loading
- **Multi-lap analysis** - combine data from multiple laps
- **Session comparison** - compare two sessions side-by-side with grouped bar charts
- **Embedded matplotlib charts** with HiDPI support
- **Configurable motion ratios** per corner
- **Channel name mapping** for different AIM telemetry setups
- **Detailed statistics** with bump/rebound breakdown per wheel

## Quick Start (Pre-built)

If you have the pre-built `SuspensionAnalyzer.exe`, just double-click to run. No installation needed.

## Building the Executable

### Option 1: Build with Wine (Linux/WSL)

Build a Windows exe from Linux without needing Windows Python installed.

**Prerequisites (one-time setup):**

```bash
# Ubuntu/Debian
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32 xvfb wget cabextract

# Fedora
sudo dnf install wine xorg-x11-server-Xvfb wget cabextract

# Arch
sudo pacman -S wine xorg-server-xvfb wget cabextract
```

**Build:**

```bash
cd desktop_app
./build_wine.sh
```

The first run will download and install Python in Wine (~5-10 minutes). Subsequent builds are faster.

Output: `desktop_app/dist/SuspensionAnalyzer.exe`

### Option 2: Build on Windows

If you have Windows with Python installed:

1. Install Python 3.12+:
   ```cmd
   winget install Python.Python.3.12
   ```

2. Clone/copy the repository to Windows

3. Open Command Prompt in the `desktop_app` folder and run:
   ```cmd
   pip install customtkinter tkinterdnd2 matplotlib pandas numpy pyarrow libxrk pyinstaller

   pyinstaller --onefile --windowed --name SuspensionAnalyzer ^
       --add-data "../src/motorsports_data_notebook;motorsports_data_notebook" ^
       --paths src ^
       --collect-all pandas ^
       --collect-all customtkinter ^
       --collect-all tkinterdnd2 ^
       --hidden-import matplotlib.backends.backend_tkagg ^
       src/suspension_analyzer/main.py
   ```

4. The executable will be at `dist\SuspensionAnalyzer.exe`

### Option 3: Development Mode

Run directly without building an exe (requires Python):

```bash
cd desktop_app
pip install -e .
python -m suspension_analyzer
```

## Usage

### Basic Analysis

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
   - Velocity range distribution (bump/rebound breakdown per wheel)
   - Balance analysis (front/rear, left/right)
   - Interpretation guide

### Configuration

- **Motion Ratios**: Set per-corner motion ratios (default: Toyota 86 ZN6)
- **Channel Names**: Update if your AIM setup uses different shock pot channel names

## Understanding the Statistics

### Velocity Ranges

- **Slow**: Low-speed damper movement (body roll, weight transfer)
- **Fast**: Medium-speed movement (bumps, curbs)
- **High-Speed**: High-speed impacts (big bumps, kerbs)

*Friction range is excluded from statistics to avoid noise bias.*

### Skew Interpretation

- **Positive skew**: More time in rebound (extension) - damper extending
- **Negative skew**: More time in bump (compression) - damper compressing
- **Near zero**: Balanced damper response

### Balance Analysis

Compares average skew between:
- **Front vs Rear**: Indicates chassis pitch tendency
- **Left vs Right**: Indicates chassis roll tendency or track characteristics

## Architecture

```
src/suspension_analyzer/
├── main.py              # Entry point
├── app.py               # Main CustomTkinter window
├── loader.py            # Session loading (no IPython dependency)
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

## Troubleshooting

### Wine build fails with subprocess error

PyInstaller's subprocess handling can be finicky under Wine. Try:
- Ensure xvfb is installed (`sudo apt install xvfb`)
- Delete the Wine prefix and retry: `rm -rf desktop_app/.wine_build`

### Build fails with "Module not found"

Add `--collect-all <module>` to the PyInstaller command.

### Chart appears blurry on high-DPI display

The app includes automatic HiDPI detection. If still blurry, check Windows display scaling settings.

### Statistics window opens behind main window

Click the window in the taskbar or use Alt+Tab.

### "Python not found" when building on Windows

Ensure Python is installed and in your PATH:
```cmd
python --version
```

If not found, reinstall Python and check "Add to PATH" during installation.
