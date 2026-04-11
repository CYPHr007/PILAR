# -*- coding: utf-8 -*-
"""
PILAR Bootstrap — Auto-setup for Ollama + AutoGen agents
=========================================================
Runs once on first launch. Auto-detects Ollama, pulls the base model,
and creates pilar-diag / pilar-maintenance / pilar-alert from the
bundled Modelfiles. Shows a native Windows dialog if Ollama is missing.

No user action needed beyond installing Ollama itself.
"""

import os
import sys
import json
import time
import ctypes
import threading
import webbrowser
import subprocess
import urllib.request
from pathlib import Path

OLLAMA_URL          = "http://localhost:11434"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/windows"
BASE_MODEL          = "llama3.2"
REQUIRED_MODELS     = ["pilar-diag", "pilar-maintenance", "pilar-alert"]
MARKER_FILENAME     = "pilar_agents_ready.marker"


# ── Paths ────────────────────────────────────────────────────────────────────
def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.resolve()


def _modelfiles_dir() -> Path:
    base = _base_dir()
    # PyInstaller COLLECT puts bundled data under _internal/
    for c in (base / "modelfiles", base / "_internal" / "modelfiles"):
        if c.exists():
            return c
    return base / "modelfiles"


def _marker_path() -> Path:
    """Writable marker file (local appdata, not install dir)."""
    root = Path(os.environ.get("LOCALAPPDATA", _base_dir())) / "PILAR"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root / MARKER_FILENAME


# ── Ollama detection ─────────────────────────────────────────────────────────
def _ollama_running() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_binary() -> Path | None:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Ollama" / "ollama.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _start_ollama_server() -> bool:
    binary = _ollama_binary()
    if not binary:
        return False
    try:
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [str(binary), "serve"],
            creationflags=CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as e:
        print(f"[bootstrap] could not spawn ollama serve: {e}")
        return False
    for _ in range(30):
        if _ollama_running():
            return True
        time.sleep(1)
    return False


def _available_models() -> set:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return {m["name"].split(":")[0] for m in data.get("models", [])}
    except Exception:
        return set()


def _pull_base_model(notify=None) -> bool:
    if notify:
        notify("PILAR", f"Telechargement de {BASE_MODEL} (~2 GB, une seule fois)...")
    try:
        body = json.dumps({"name": BASE_MODEL, "stream": False}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/pull",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as resp:
            resp.read()
    except Exception as e:
        print(f"[bootstrap] pull {BASE_MODEL} failed: {e}")
        return False
    return BASE_MODEL in _available_models()


def _create_model(name: str, modelfile: Path) -> bool:
    try:
        content = modelfile.read_text(encoding="utf-8")
        body = json.dumps({"name": name, "modelfile": content, "stream": False}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/create",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
    except Exception as e:
        print(f"[bootstrap] create {name} failed: {e}")
        return False
    return name in _available_models()


# ── UI ───────────────────────────────────────────────────────────────────────
def _dialog_install_ollama() -> bool:
    """Blocking yes/no dialog. Returns True if user wants to install Ollama."""
    try:
        MB_YESNO, MB_ICONINFO, IDYES = 0x04, 0x40, 6
        msg = (
            "PILAR peut utiliser des agents IA locaux pour generer "
            "des diagnostics et plans de maintenance.\n\n"
            "Pour activer ces agents, installez Ollama (gratuit, hors ligne) :\n"
            "   https://ollama.com/download/windows\n\n"
            "Ouvrir la page de telechargement maintenant ?\n\n"
            "(PILAR fonctionne parfaitement sans Ollama — il utilisera "
            "des regles expertes pour l'analyse.)"
        )
        result = ctypes.windll.user32.MessageBoxW(
            0, msg, "PILAR — Activer les agents IA",
            MB_YESNO | MB_ICONINFO,
        )
        return result == IDYES
    except Exception:
        return False


# ── Main bootstrap ───────────────────────────────────────────────────────────
def bootstrap(notify=None) -> None:
    """
    Run once on first launch. Safe to call repeatedly: skips if marker exists.
    `notify(title, msg)` — optional callback for tray notifications.
    """
    marker = _marker_path()
    if marker.exists():
        return

    print("[bootstrap] PILAR AI agents setup starting")

    # 1) Ensure Ollama is up
    if not _ollama_running():
        if _ollama_binary():
            if notify:
                notify("PILAR", "Demarrage d'Ollama...")
            _start_ollama_server()

    if not _ollama_running():
        # Ollama not installed — show dialog, open download page, exit
        if _dialog_install_ollama():
            try:
                webbrowser.open(OLLAMA_DOWNLOAD_URL)
            except Exception:
                pass
        # Do NOT write marker — retry next launch
        return

    # 2) Pull base llama3.2 if missing
    available = _available_models()
    if BASE_MODEL not in available:
        if not _pull_base_model(notify=notify):
            print(f"[bootstrap] could not pull {BASE_MODEL} — aborting")
            return

    # 3) Create pilar-* models from bundled modelfiles
    mfiles_dir = _modelfiles_dir()
    if not mfiles_dir.exists():
        print(f"[bootstrap] modelfiles dir missing: {mfiles_dir}")
        return

    available = _available_models()
    ok_count = 0
    for name in REQUIRED_MODELS:
        if name in available:
            ok_count += 1
            continue
        mfile = mfiles_dir / (name.replace("-", "_") + ".modelfile")
        if not mfile.exists():
            print(f"[bootstrap] missing: {mfile}")
            continue
        if notify:
            notify("PILAR", f"Creation de l'agent {name}...")
        if _create_model(name, mfile):
            ok_count += 1

    # 4) Mark done if all required models exist
    if ok_count == len(REQUIRED_MODELS):
        try:
            marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            pass
        if notify:
            notify("PILAR", "Agents IA actives — diagnostics intelligents disponibles.")
        print("[bootstrap] done — all agents ready")
    else:
        print(f"[bootstrap] incomplete: {ok_count}/{len(REQUIRED_MODELS)} models ready")


def bootstrap_async(notify=None) -> None:
    """Non-blocking launch."""
    threading.Thread(
        target=bootstrap,
        kwargs={"notify": notify},
        daemon=True,
        name="pilar-bootstrap",
    ).start()


if __name__ == "__main__":
    bootstrap(notify=lambda t, m: print(f"[{t}] {m}"))
