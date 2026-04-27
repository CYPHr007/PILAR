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

# ML model files — explicit whitelist avoids shipping test/dev pickles
ML_PICKLES = [
    "failure_model.pkl",
    "isolation_forest.pkl",
    "rul_model.pkl",
    "rul_scaler.pkl",
    "scaler.pkl",
    "zone_models.pkl",
]
for name in ML_PICKLES:
    p = BASE / name
    if p.exists():
        datas.append((str(p), "."))

# JSON metadata — explicit whitelist (avoid package.json, tsconfig.json, etc.)
JSON_FILES = ["model_meta.json"]
for name in JSON_FILES:
    p = BASE / name
    if p.exists():
        datas.append((str(p), "."))

# Static assets (templates, CSS, JS, icons)
static_dir = BASE / "static"
if static_dir.exists():
    datas.append((str(static_dir), "static"))

# Config
# NOTE: pilar_calibrator and pilar_bootstrap are real Python modules
# (see hiddenimports below). Do NOT bundle them as data files — that
# causes two copies and breaks isinstance() checks across modules.
datas.append((str(BASE / "config.py"), "."))
datas.append((str(BASE / "app.py"),    "."))

# Agents package
agents_dir = BASE / "agents"
if agents_dir.exists():
    datas.append((str(agents_dir), "agents"))

# Modelfiles (kept for legacy reference — no longer used at runtime)
modelfiles_dir = BASE / "modelfiles"
if modelfiles_dir.exists():
    datas.append((str(modelfiles_dir), "modelfiles"))

# llama-cpp-python: bundle the native DLL alongside the package
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    datas += collect_dynamic_libs('llama_cpp')
except Exception:
    pass

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
    "pilar_bootstrap",
    "agents",
    "agents.orchestrator",
    "agents.config",
    "agents.sla_tracker",
    "agents.diagnostic_agent",
    "agents.maintenance_agent",
    "agents.alert_agent",
    "agents.llm_engine",
    "requests",
    # llama-cpp-python (Qwen3 4B local inference)
    "llama_cpp",
    "llama_cpp.llama",
    "llama_cpp.llama_chat_format",
    "llama_cpp.llama_tokenizer",
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
              "IPython", "ipykernel", "jupyter",
              "autogen_agentchat", "autogen_ext"],
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
    # UPX corrupts several MSVC and xgboost DLLs — exclude them.
    upx_exclude=[
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "concrt140.dll",
        "python3*.dll",
        "xgboost.dll",
        "api-ms-win-*.dll",
        "ucrtbase.dll",
    ],
    name="pilar",
)
