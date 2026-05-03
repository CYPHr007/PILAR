# -*- coding: utf-8 -*-
"""
PILAR Bootstrap — Auto-setup for Qwen3 4B AI agents
====================================================
Runs once on first launch. Downloads the Qwen3-4B-Q4_K_M.gguf model file
(~2.6 GB) from HuggingFace to %LOCALAPPDATA%/PILAR/models/.

No separate app to install (no Ollama). The model runs directly via
llama-cpp-python, which is bundled with PILAR.

PILAR works immediately with rule-based agents while the download is in
progress. Once the download completes, Qwen3 activates automatically.
"""

import os
import sys
import time
import threading
import urllib.request
from pathlib import Path

from agents.llm_engine import MODEL_FILENAME, MODEL_DIR, MODEL_URL

MARKER_FILENAME  = "pilar_agents_ready.marker"
DECLINED_FILENAME = "pilar_agents_declined.marker"
MODEL_SIZE_BYTES  = 2_600_000_000   # ~2.6 GB — used for progress estimation


# ── Paths ─────────────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PILAR"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root


def _marker_path() -> Path:
    return _state_dir() / MARKER_FILENAME


def _declined_path() -> Path:
    return _state_dir() / DECLINED_FILENAME


def reset_bootstrap() -> None:
    """Delete markers so bootstrap runs again on next launch."""
    for p in (_marker_path(), _declined_path()):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    # Also remove the model file so it re-downloads
    model = MODEL_DIR / MODEL_FILENAME
    try:
        if model.exists():
            model.unlink()
    except Exception:
        pass


# ── Download ──────────────────────────────────────────────────────────────────

def _download_model(notify=None) -> bool:
    """
    Download Qwen3-4B-Q4_K_M.gguf from HuggingFace.
    Shows progress via notify callback. Returns True on success.
    Supports resume: skips bytes already downloaded.
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bootstrap] cannot create model dir: {e}")
        return False

    dest        = MODEL_DIR / MODEL_FILENAME
    tmp         = MODEL_DIR / (MODEL_FILENAME + ".part")  # noqa: W605
    resume_byte = tmp.stat().st_size if tmp.exists() else 0

    if notify:
        notify("PILAR", "Telechargement du modele IA Qwen3 4B (~2.6 GB)...")

    print(f"[bootstrap] downloading {MODEL_FILENAME} → {dest}")

    try:
        req = urllib.request.Request(MODEL_URL)
        if resume_byte:
            req.add_header("Range", f"bytes={resume_byte}-")
            print(f"[bootstrap] resuming from byte {resume_byte:,}")

        with urllib.request.urlopen(req, timeout=3600) as resp, \
             open(tmp, "ab" if resume_byte else "wb") as f:

            total    = int(resp.headers.get("Content-Length", MODEL_SIZE_BYTES))
            received = resume_byte
            last_pct = -1

            while True:
                chunk = resp.read(1 << 17)   # 128 KB chunks
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                pct = int(received * 100 / (total + resume_byte))
                if pct != last_pct and pct % 10 == 0:
                    last_pct = pct
                    msg = f"Modele IA : {pct}% ({received // (1024*1024)} MB)"
                    print(f"[bootstrap] {msg}")
                    if notify:
                        notify("PILAR", msg)

        # Rename .part → final
        tmp.rename(dest)
        print(f"[bootstrap] download complete: {dest}")
        return True

    except Exception as e:
        print(f"[bootstrap] download error: {e}")
        if notify:
            notify("PILAR", f"Echec du telechargement du modele IA: {e}")
        return False


# ── UI dialog ─────────────────────────────────────────────────────────────────

def _dialog_confirm_download() -> bool:
    """
    Confirm via tray notification (non-blocking).
    Returns False — the actual download is triggered by the user
    through the tray menu item 'Télécharger les modèles IA…'.
    The notify callback was already called before this to inform the user.
    """
    return False


def prompt_download_from_tray(notify=None) -> None:
    """
    Called when user clicks 'Télécharger les modèles IA…' in the tray menu.
    Shows a confirmation dialog and starts the download if confirmed.
    """
    try:
        import ctypes
        MB_YESNO, MB_ICONINFO, IDYES = 0x04, 0x40, 6
        msg = (
            "PILAR peut activer des agents IA locaux (Qwen3 4B) pour\n"
            "generer des diagnostics et plans de maintenance intelligents.\n\n"
            "Cela necessite le telechargement unique du modele (~2.6 Go).\n"
            "Le modele est stocke sur votre machine et ne necessite\n"
            "aucune connexion apres le telechargement.\n\n"
            "Telecharger le modele maintenant ?\n\n"
            "(PILAR fonctionne parfaitement sans ce modele.)"
        )
        result = ctypes.windll.user32.MessageBoxW(
            0, msg, "PILAR — Agents IA Qwen3",
            MB_YESNO | MB_ICONINFO,
        )
        if result == IDYES:
            declined = _declined_path()
            try:
                declined.unlink(missing_ok=True)
            except Exception:
                pass
            success = _download_model(notify=notify)
            if success:
                try:
                    _marker_path().write_text("ok\n", encoding="utf-8")
                except Exception:
                    pass
                if notify:
                    notify("PILAR", "Agents IA Qwen3 actives.")
                _preload_model(notify)
    except Exception as e:
        print(f"[bootstrap] prompt_download_from_tray error: {e}")


# ── Main bootstrap ────────────────────────────────────────────────────────────

def bootstrap(notify=None) -> None:
    """
    Run once on first launch. Safe to call repeatedly — skips if marker exists.
    notify(title, msg) — optional tray notification callback.
    """
    marker   = _marker_path()
    declined = _declined_path()
    dest     = MODEL_DIR / MODEL_FILENAME

    # Already set up
    if marker.exists() and dest.exists():
        return

    # Model already there but marker missing (e.g. after reset) — just mark done
    if dest.exists():
        try:
            marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            pass
        if notify:
            notify("PILAR", "Agents IA Qwen3 deja disponibles.")
        _preload_model(notify)
        return

    print("[bootstrap] PILAR AI agent setup starting")

    # User previously declined — don't prompt again
    if declined.exists():
        print("[bootstrap] user previously declined model download — skipping")
        return

    # Notify via tray (non-blocking) — user can download from tray menu
    if notify:
        notify(
            "PILAR — Agents IA disponibles",
            "Les agents IA Qwen3 4B sont disponibles. "
            "Telechargez-les depuis le menu tray pour activer les diagnostics intelligents.",
        )
    print("[bootstrap] tray notification sent — user can download from tray menu")
    return

    # Download
    success = _download_model(notify=notify)
    if not success:
        return

    # Mark done
    try:
        marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass

    if notify:
        notify("PILAR", "Agents IA Qwen3 actives — diagnostics intelligents disponibles.")
    print("[bootstrap] done — Qwen3 4B ready")

    # Warm the model in background so first real request is instant
    _preload_model(notify)


def _preload_model(notify=None) -> None:
    """Load model into memory in background after download completes."""
    def _do():
        try:
            from agents.llm_engine import preload
            ok = preload()
            if ok and notify:
                notify("PILAR", "Modele IA charge en memoire — pret.")
        except Exception as e:
            print(f"[bootstrap] preload error: {e}")

    threading.Thread(target=_do, daemon=True, name="pilar-preload").start()


def bootstrap_async(notify=None) -> None:
    """Non-blocking launch — runs bootstrap in a daemon thread."""
    threading.Thread(
        target=bootstrap,
        kwargs={"notify": notify},
        daemon=True,
        name="pilar-bootstrap",
    ).start()


if __name__ == "__main__":
    bootstrap(notify=lambda t, m: print(f"[{t}] {m}"))
