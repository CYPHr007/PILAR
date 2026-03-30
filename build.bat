@echo off
:: PILAR Desktop — Windows Build Script
:: =====================================
:: Usage:  build.bat
:: Output: dist\PILAR_Setup_1.0.0.exe
::
:: Requirements:
::   - Python 3.10+ in PATH
::   - pip install pyinstaller pystray pillow
::   - Inno Setup 6.x or 7.x installed
::     (https://jrsoftware.org/isinfo.php)
::
setlocal EnableDelayedExpansion

set APP_NAME=PILAR
set APP_VERSION=1.2.2
set SCRIPT_DIR=%~dp0
set DIST_DIR=%SCRIPT_DIR%dist
set INNO_V7_X86=C:\Program Files (x86)\Inno Setup 7\ISCC.exe
set INNO_V7_X64=C:\Program Files\Inno Setup 7\ISCC.exe
set INNO_V6_X86=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
set INNO_V6_X64=C:\Program Files\Inno Setup 6\ISCC.exe

echo ======================================
echo   PILAR Desktop ^— Windows Build
echo   Version: %APP_VERSION%
echo ======================================
echo.

:: ── 1. Check Python ───────────────────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found in PATH.
    echo Install Python 3.10+ from https://python.org
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   %%v

:: ── 2. Install/upgrade PyInstaller and dependencies ───────────────────────────
echo.
echo [2/4] Checking PyInstaller and tray dependencies...
python -m pip install --quiet --upgrade pyinstaller pystray pillow >nul 2>&1
if errorlevel 1 (
    echo WARNING: pip upgrade had issues. Continuing anyway...
)
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pyinstaller not available after install attempt.
    exit /b 1
)
for /f "tokens=*" %%v in ('pyinstaller --version') do echo   PyInstaller %%v

:: ── 3. Build with PyInstaller ─────────────────────────────────────────────────
echo.
echo [3/4] Building with PyInstaller (this may take several minutes)...
cd /d "%SCRIPT_DIR%"
pyinstaller --clean --noconfirm pilar.spec
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)
echo   Build complete: %DIST_DIR%\pilar\

:: ── 4. Run Inno Setup compiler ────────────────────────────────────────────────
echo.
echo [4/4] Creating Windows installer with Inno Setup...

:: Find Inno Setup (tries v7 first, then v6)
if exist "%INNO_V7_X86%" (
    set ISCC="%INNO_V7_X86%"
) else if exist "%INNO_V7_X64%" (
    set ISCC="%INNO_V7_X64%"
) else if exist "%INNO_V6_X86%" (
    set ISCC="%INNO_V6_X86%"
) else if exist "%INNO_V6_X64%" (
    set ISCC="%INNO_V6_X64%"
) else (
    echo WARNING: Inno Setup not found at:
    echo   %INNO_V7_X86%
    echo   %INNO_V7_X64%
    echo   %INNO_V6_X86%
    echo   %INNO_V6_X64%
    echo.
    echo Download from: https://jrsoftware.org/isinfo.php
    echo After install, run: iscc pilar_installer.iss
    echo.
    echo PyInstaller output is available at: %DIST_DIR%\pilar\
    echo Skipping installer packaging.
    goto :done
)

%ISCC% pilar_installer.iss
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed.
    exit /b 1
)

:done
echo.
echo ======================================
echo   BUILD COMPLETE
if exist "%DIST_DIR%\PILAR_Setup_%APP_VERSION%.exe" (
    echo   Installer: %DIST_DIR%\PILAR_Setup_%APP_VERSION%.exe
)
echo   App dir:   %DIST_DIR%\pilar\
echo ======================================
echo.
pause
