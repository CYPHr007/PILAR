# -*- mode: python ; coding: utf-8 -*-
#
# PILAR Desktop — PyInstaller spec
# =================================
# Build:  pyinstaller pilar.spec
# Output: dist\pilar\  (directory build, faster startup than onefile)
#
# Bundles:
#   - launcher.py        (system tray entry point)
#   - app.py          (Flask web app)
#   - config.py          (configuration constants)
#   - All .pkl model files
#   - static/ directory
#   - assets/ directory (if present)

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

BASE = Path(SPECPATH)

# ── Collect data files ────────────────────────────────────────────────────────
datas = []

# Include all sklearn data files (e.g. joblib compressed models, etc.)
datas += collect_data_files('sklearn')

# xgboost data files: VERSION (required at import time), xgboost.dll, py.typed
datas += collect_data_files('xgboost')

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

# Config + calibrator
datas.append((str(BASE / "config.py"),        "."))
datas.append((str(BASE / "app.py"),           "."))
datas.append((str(BASE / "pilar_calibrator.py"), "."))

# Agents package
agents_dir = BASE / "agents"
if agents_dir.exists():
    datas.append((str(agents_dir), "agents"))

# Modelfiles (Ollama few-shot definitions)
modelfiles_dir = BASE / "modelfiles"
if modelfiles_dir.exists():
    datas.append((str(modelfiles_dir), "modelfiles"))

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
    # ML — collect ALL sklearn submodules so RandomForest/IsoForest pickle loads work
] + collect_submodules('sklearn') + [
    "xgboost",
    "numpy",
    "pandas",
    "scipy",
    # Misc
    "packaging",
    "packaging.version",
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
    # SocketIO
    "flask_socketio",
    "simple_websocket",
    "engineio",
    "socketio",
    # SHAP
    "shap",
    # Imbalanced-learn
    "imblearn",
    "imblearn.over_sampling",
    # FPdf2
    "fpdf",
    "fpdf2",
    # pywebview
    "webview",
    "webview.platforms.winforms",
    "clr",
    "proxy_tools",
    "bottle",
    "pythonnet",
    "clr_loader",
    # Sync client (stdlib, no extras needed)
    "urllib.request",
    "urllib.error",
    "threading",
    # Agents + Ollama
    "pilar_calibrator",
    "agents",
    "agents.orchestrator",
    "agents.config",
    "agents.sla_tracker",
    "agents.diagnostic_agent",
    "agents.maintenance_agent",
    "agents.alert_agent",
    "requests",
    "autogen_agentchat",
    "autogen_agentchat.agents",
    "autogen_agentchat.teams",
    "autogen_agentchat.conditions",
    "autogen_ext",
    "autogen_ext.models.openai",
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
    excludes=["matplotlib", "cv2", "PyQt5", "wx",
              "torch", "torchvision", "torchaudio",
              "numba", "llvmlite",
              "IPython", "ipykernel", "jupyter"],
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
    # Windows icon
    icon=str(BASE / "pilar.ico") if (BASE / "pilar.ico").exists()
         else (str(BASE / "pilar.png") if (BASE / "pilar.png").exists() else None),
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
