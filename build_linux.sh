#!/usr/bin/env bash
# PILAR Desktop — Linux AppImage Build Script
# ============================================
# Usage:  ./build_linux.sh
# Output: PILAR-1.0.0-x86_64.AppImage
#
# Requirements:
#   - Python 3.10+ with venv
#   - pyinstaller
#   - appimagetool (https://github.com/AppImage/AppImageKit/releases)
#   - fuse or fuse3 (for AppImage mounting)
#
# Install appimagetool:
#   wget -O appimagetool https://github.com/AppImage/AppImageKit/releases/latest/download/appimagetool-x86_64.AppImage
#   chmod +x appimagetool && sudo mv appimagetool /usr/local/bin/
#
set -euo pipefail

APP_NAME="PILAR"
APP_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
APPDIR="$DIST_DIR/${APP_NAME}.AppDir"

echo "======================================"
echo "  PILAR Desktop — Linux Build"
echo "  Version: $APP_VERSION"
echo "======================================"
echo ""

# ── 1. Check dependencies ─────────────────────────────────────────────────────
echo "[1/5] Checking dependencies..."

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PYTHON_VERSION"

if ! command -v pyinstaller &>/dev/null; then
    echo "ERROR: pyinstaller not found. Run: pip install pyinstaller"
    exit 1
fi
echo "  PyInstaller $(pyinstaller --version)"

if ! command -v appimagetool &>/dev/null; then
    echo "WARNING: appimagetool not found. Will skip AppImage packaging."
    echo "  Install from: https://github.com/AppImage/AppImageKit/releases"
    SKIP_APPIMAGE=1
else
    SKIP_APPIMAGE=0
    echo "  appimagetool found"
fi

# ── 2. Build with PyInstaller ─────────────────────────────────────────────────
echo ""
echo "[2/5] Building with PyInstaller..."

cd "$SCRIPT_DIR"

pyinstaller \
    --clean \
    --noconfirm \
    pilar.spec

echo "  PyInstaller build complete → $DIST_DIR/pilar/"

# ── 3. Create AppDir structure ────────────────────────────────────────────────
echo ""
echo "[3/5] Creating AppDir structure..."

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/lib"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy PyInstaller output into AppDir
cp -r "$DIST_DIR/pilar/." "$APPDIR/usr/bin/"

# ── 4. Create .desktop file ───────────────────────────────────────────────────
echo ""
echo "[4/5] Creating .desktop file and icon..."

cat > "$APPDIR/usr/share/applications/pilar.desktop" <<DESKTOP
[Desktop Entry]
Name=PILAR
Comment=Predictive Maintenance for Industrial Pumps
Exec=PILAR %U
Icon=pilar
Type=Application
Categories=Science;Engineering;Utility;
Keywords=maintenance;predictive;pump;industrial;monitoring;
StartupNotify=true
StartupWMClass=PILAR
Terminal=false
DESKTOP

# Create AppRun symlink
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="$HERE/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HERE/usr/lib:${LD_LIBRARY_PATH:-}"
# Set writable user data directory
export PILAR_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/pilar"
mkdir -p "$PILAR_DATA_DIR"
cd "$PILAR_DATA_DIR"
exec "$HERE/usr/bin/PILAR" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# Copy .desktop to root of AppDir (required by AppImage spec)
cp "$APPDIR/usr/share/applications/pilar.desktop" "$APPDIR/pilar.desktop"

# Create a simple SVG icon (teal circle with P)
cat > "$APPDIR/usr/share/icons/hicolor/256x256/apps/pilar.svg" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <circle cx="128" cy="128" r="120" fill="#07090f"/>
  <circle cx="128" cy="128" r="96" fill="#0d9488"/>
  <text x="128" y="168" font-family="Arial,sans-serif" font-size="110" font-weight="bold"
        text-anchor="middle" fill="white">P</text>
</svg>
SVG

# Copy icon to AppDir root (required)
cp "$APPDIR/usr/share/icons/hicolor/256x256/apps/pilar.svg" "$APPDIR/pilar.svg"

# ── 5. Package as AppImage ────────────────────────────────────────────────────
echo ""
echo "[5/5] Packaging as AppImage..."

if [ "${SKIP_APPIMAGE:-0}" = "1" ]; then
    echo "  Skipping AppImage packaging (appimagetool not found)."
    echo "  You can still run from: $APPDIR/AppRun"
else
    OUTPUT="$DIST_DIR/${APP_NAME}-${APP_VERSION}-x86_64.AppImage"
    ARCH=x86_64 appimagetool "$APPDIR" "$OUTPUT"
    chmod +x "$OUTPUT"
    echo ""
    echo "======================================"
    echo "  BUILD COMPLETE"
    echo "  Output: $OUTPUT"
    echo "  Size:   $(du -sh "$OUTPUT" | cut -f1)"
    echo "======================================"
fi

echo ""
echo "Done!"
