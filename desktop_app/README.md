# Inferno Analyzer

Standalone desktop application for motorsports telemetry analysis combining suspension velocity histograms and driver consistency analysis in a single tabbed interface.

**The built executables are fully self-contained and run without Python installed.**

- **Windows**: `InfernoAnalyzer.exe` - runs on Windows 10/11
- **Linux**: `InfernoAnalyzer` - runs on most modern Linux distributions

## Features

### Suspension Tab
- **2x2 velocity histogram** for all four shock corners (FL, FR, RL, RR)
- **Configurable motion ratios** per corner
- **Detailed statistics** with bump/rebound breakdown per wheel

### Driver Consistency Tab
- **Throttle acceptance** analysis across detected corners
- **Braking point consistency** visualization
- **Corner detection** from GPS data via curvature analysis
- **Summary, detail, and track map** view modes

### Shared
- **Drag-and-drop** XRK/XRZ file loading
- **Multi-lap analysis** - combine data from multiple laps
- **Session comparison** - compare two sessions side-by-side
- **Vehicle profiles** - auto-save/load channel mappings per logger serial number
- **Channel name mapping** for different AIM telemetry setups
- **Embedded matplotlib charts** with HiDPI support

## Quick Start (Pre-built)

If you have the pre-built `InfernoAnalyzer.exe`, just double-click to run. No installation needed.

## Building the Executable

### Option 1: Build with Wine (Linux/WSL)

Build a Windows exe from Linux without needing Windows installed.

**Prerequisites (one-time setup):**

```bash
# Ubuntu/Debian - enable 32-bit architecture first
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32:i386 xvfb wget

# Fedora
sudo dnf install wine xorg-x11-server-Xvfb wget

# Arch
sudo pacman -S wine xorg-server-xvfb wget
```

**Build:**

```bash
cd desktop_app
./build_wine.sh
```

The first run will download and install Python in Wine (~5-10 minutes). Subsequent builds are faster.

Output: `desktop_app/dist/InfernoAnalyzer.exe`

### Option 2: Build Native Linux Binary

Build a Linux executable that runs on most modern Linux distributions.

**Prerequisites:**

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

**Build:**

```bash
cd desktop_app
uv sync --extra build
uv run pyinstaller --clean --noconfirm inferno_analyzer.spec
```

Output: `desktop_app/dist/InfernoAnalyzer`

**Run:**

```bash
./dist/InfernoAnalyzer
```

Note: The Linux binary requires a display server (X11 or Wayland). On WSL2, GUI apps display automatically via WSLg.

### Option 3: Build on Windows

If you have Windows with Python installed:

1. Install Python 3.12+:
   ```cmd
   winget install Python.Python.3.12
   ```

2. Clone/copy the repository to Windows

3. Open Command Prompt in the `desktop_app` folder and run:
   ```cmd
   pip install customtkinter tkinterdnd2 matplotlib pandas numpy pyarrow libxrk pyyaml pyinstaller

   pyinstaller --onefile --windowed --name InfernoAnalyzer ^
       --add-data "../src/motorsports_data_notebook;motorsports_data_notebook" ^
       --paths src ^
       --collect-all pandas ^
       --collect-all customtkinter ^
       --collect-all tkinterdnd2 ^
       --hidden-import matplotlib.backends.backend_tkagg ^
       --hidden-import yaml ^
       src/inferno_analyzer/main.py
   ```

4. The executable will be at `dist\InfernoAnalyzer.exe`

### Option 4: Development Mode

Run directly without building an exe (requires Python):

```bash
cd desktop_app
pip install -e .
python -m inferno_analyzer
```

## Usage

### Basic Analysis

1. **Load Session A**: Drag an XRK/XRZ file onto the left panel, or click to browse
2. **Select laps**: Check the laps you want to include in the analysis
3. **Switch tabs**: Use the tab bar to switch between Suspension and Driver Consistency analysis
4. **Maximize chart**: Click "Maximize" button to view full-screen chart

### Comparing Sessions

1. Load a second XRK/XRZ file into **Session B** (middle panel)
2. Select laps from both sessions
3. Both tabs automatically show comparison views

### Vehicle Profiles

When loading a session, the app detects the logger serial number and auto-populates channel names and motion ratios from a saved profile. Click **Save Profile** to save the current configuration for that logger.

### Configuration

The config panel (right column) swaps content based on the active tab:

- **Suspension tab**: Motion ratios per corner, shock pot channel names
- **Driver tab**: Corner detection threshold, throttle threshold, sustain time, channel names

## Architecture

```
src/inferno_analyzer/
├── main.py              # Entry point (LIBXRK_BACKEND=rust)
├── app.py               # InfernoAnalyzerApp (shared sessions + CTkTabview)
├── tabs/
│   ├── base_tab.py      # BaseAnalysisTab (debounce, threading, stats window)
│   ├── suspension_tab.py
│   └── driver_tab.py
├── suspension/
│   ├── analysis/multi_lap.py
│   ├── visualization/comparison.py
│   └── widgets/{chart_view,config_panel,stats_panel}.py
└── driver/
    ├── analysis/driver_consistency.py
    ├── visualization/{corner_detail,corner_summary,track_map}.py
    └── widgets/{chart_view,config_panel,corner_selector,stats_panel}.py
```

## Troubleshooting

### Wine build fails - "wine32 is missing"

The Python Windows installer requires 32-bit Wine support:
```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine32:i386
```

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

### Linux build fails with "tkinter installation is broken"

Install the tkinter package for your distribution:
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

Then rebuild.
