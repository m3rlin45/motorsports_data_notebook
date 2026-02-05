@echo off
REM Build script for Suspension Analyzer

echo ========================================
echo  Suspension Analyzer Build Script
echo ========================================
echo.

REM Check if we're in the desktop_app directory
if not exist "pyproject.toml" (
    echo Error: Run this script from the desktop_app directory
    exit /b 1
)

REM Check for Poetry
where poetry >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Poetry not found. Please install Poetry first.
    exit /b 1
)

echo [1/4] Installing dependencies...
call poetry install --extras build
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo.
echo [2/4] Cleaning previous build...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

echo.
echo [3/4] Running PyInstaller...
call poetry run pyinstaller suspension_analyzer.spec --noconfirm
if %ERRORLEVEL% neq 0 (
    echo Error: PyInstaller build failed
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo Output: dist\SuspensionAnalyzer.exe
echo.

REM Check file size
for %%A in ("dist\SuspensionAnalyzer.exe") do (
    set size=%%~zA
    set /a sizeMB=%%~zA / 1048576
)
echo Executable size: %sizeMB% MB

echo.
echo ========================================
echo  Build completed successfully!
echo ========================================
