# -*- mode: python ; coding: utf-8 -*-
#
# PILAR Desktop — PyInstaller spec
# =================================
# Build:  pyinstaller pilar.spec
# Output: dist\pilar\  (directory build, faster startup than onefile)
#
# Bundles:
#   - launcher.py        (system tray entry point)
#   - etape7.py          (Flask web app)
#   - config.py          (configuration constants)
#   - All .pkl model files
#   - static/ directory
#   - assets/ directory (if present)

import os
from pathlib import Path

block_cipher = None

BASE = Path(SPECPATH)

# ── Collect data files ────────────────────────────────────────────────────────
datas = []

# ML model files
for pkl in BASE.glob("*.pkl"):
    datas.append((str(pkl), "."))

# JSON metadata
for json_f in BASE.glob("*.json"):
    datas.append((str(json_f), "."))

# Static assets (templates, CSS, JS, icons)
static_dir = BASE / "static"
if static_dir.exists():
    datas.append((str(static_dir), "static"))

# Config
datas.append((str(BASE / "config.py"),   "."))
datas.append((str(BASE / "etape7.py"),   "."))

# ── Hidden imports ────────────────────────────────────────────────────────────
# Flask and its ecosystem are not always auto-detected
hidden_imports = [
    # Flask
    "flask",
    "flask.templating",
    "flask_sqlalchemy",
    "werkzeug",
    "werkzeug.security",
    "werkzeug.serving",
    "jinja2",
    "jinja2.ext",
    "click",
    "itsdangerous",
    # DB
    "sqlalchemy",
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.dialects.postgresql",
    # ML
    "sklearn",
    "sklearn.ensemble",
    "sklearn.preprocessing",
    "sklearn.pipeline",
    "xgboost",
    "numpy",
    "pandas",
    "scipy",
    # Misc
    "pickle",
    "apscheduler",
    "apscheduler.schedulers.background",
    "anthropic",
    "pystray",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    # Email
    "smtplib",
    "email.mime.text",
    "email.mime.multipart",
    # Sync client (stdlib, no extras needed)
    "urllib.request",
    "urllib.error",
    "threading",
]

# ── Main launcher analysis ─────────────────────────────────────────────────────
launcher_a = Analysis(
    [str(BASE / "launcher.py")],
    pathex=[str(BASE)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "cv2", "PyQt5", "wx"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(launcher_a.pure, launcher_a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    launcher_a.scripts,
    [],
    exclude_binaries=True,
    name="PILAR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # No console window on Windows (desktop app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows icon (place pilar.ico in same dir to enable)
    icon=str(BASE / "pilar.ico") if (BASE / "pilar.ico").exists() else None,
)

coll = COLLECT(
    exe,
    launcher_a.binaries,
    launcher_a.zipfiles,
    launcher_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pilar",
)
