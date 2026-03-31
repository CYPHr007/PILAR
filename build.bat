@echo off
:: PILAR Desktop — Windows Build Script
:: =====================================
:: Usage:  build.bat
:: Output: dist\PILAR_Setup_X.Y.Z.exe  (Inno Setup installer)
::
:: Requirements:
::   - Python 3.10+ in PATH
::   - pip install pyinstaller pystray pillow
::   - Inno Setup 6.x or 7.x  https://jrsoftware.org/isinfo.php
::
setlocal EnableDelayedExpansion

set APP_NAME=PILAR
set APP_VERSION=1.2.11
set SCRIPT_DIR=%~dp0
set DIST_DIR=%SCRIPT_DIR%dist
set INNO_V7_X86=C:\Program Files (x86)\Inno Setup 7\ISCC.exe
set INNO_V7_X64=C:\Program Files\Inno Setup 7\ISCC.exe
set INNO_V6_X86=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
set INNO_V6_X64=C:\Program Files\Inno Setup 6\ISCC.exe
set INNO_LOCAL=%LOCALAPPDATA%\Programs\Inno Setup 7\ISCC.exe

echo ======================================
echo   PILAR Desktop ^— Windows Build
echo   Version: %APP_VERSION%
echo ======================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found in PATH.
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   %%v

:: ── 2. Install/upgrade PyInstaller ───────────────────────────────────────────
echo.
echo [2/4] Checking PyInstaller...
python -m pip install --quiet --upgrade pyinstaller pystray pillow >nul 2>&1
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pyinstaller not available.
    exit /b 1
)
for /f "tokens=*" %%v in ('pyinstaller --version') do echo   PyInstaller %%v

:: ── 3. PyInstaller build ─────────────────────────────────────────────────────
echo.
echo [3/4] Building with PyInstaller...
cd /d "%SCRIPT_DIR%"
pyinstaller --clean --noconfirm pilar.spec
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)
echo   PyInstaller OK: %DIST_DIR%\pilar\

:: ── 4. Inno Setup installer ──────────────────────────────────────────────────
echo.
echo [4/4] Building installer with Inno Setup...

if exist "%INNO_V7_X86%" ( set ISCC="%INNO_V7_X86%"
) else if exist "%INNO_V7_X64%" ( set ISCC="%INNO_V7_X64%"
) else if exist "%INNO_V6_X86%" ( set ISCC="%INNO_V6_X86%"
) else if exist "%INNO_V6_X64%" ( set ISCC="%INNO_V6_X64%"
) else if exist "%INNO_LOCAL%" ( set ISCC="%INNO_LOCAL%"
) else (
    echo WARNING: Inno Setup not found — download from https://jrsoftware.org/isinfo.php
    echo          Then run:  iscc pilar_installer.iss
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
echo   BUILD COMPLETE — v%APP_VERSION%
if exist "%DIST_DIR%\PILAR_Setup_%APP_VERSION%.exe" (
    echo   Installer: %DIST_DIR%\PILAR_Setup_%APP_VERSION%.exe
)
echo   App dir:   %DIST_DIR%\pilar\
echo ======================================
echo.
pause
