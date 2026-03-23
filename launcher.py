"""
PILAR Desktop Launcher
======================
Lance Flask en thread background, puis ouvre Edge en mode app (fenetre
sans barre URL ni onglets — comme Discord/Notion).

- Profil Edge dedie : sessions persistantes (pas besoin de re-login)
- Tray icon : Ouvrir PILAR / Quitter PILAR
- Fallback Chrome si Edge absent, puis navigateur par defaut
"""

import os
import sys
import time
import threading
import subprocess
import multiprocessing
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Config ─────────────────────────────────────────────────────────────────────
_FROZEN   = getattr(sys, "frozen", False)
APP_HOST  = "127.0.0.1"
APP_PORT  = 5000
APP_URL   = f"http://{APP_HOST}:{APP_PORT}"

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).parent.resolve()

# Profil Edge isole pour PILAR (sessions persistantes entre redemarrages)
EDGE_PROFILE = BASE_DIR / "edge_data"

# ── State ──────────────────────────────────────────────────────────────────────
_flask_started = False
_tray_icon     = None


# ── Icone tray ─────────────────────────────────────────────────────────────────
def _make_icon(size: int = 64) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, size - 2, size - 2], fill=(14, 17, 24, 255))
    pad = size // 6
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(13, 148, 136, 255))
    draw.ellipse(
        [pad + 2, pad + 2, size - pad - 2, size - pad - 2],
        outline=(20, 184, 166, 180), width=2,
    )
    try:
        from PIL import ImageFont
        font_size = size // 3
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "P", font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((size - tw) // 2, (size - th) // 2 - 1),
            "P", fill=(255, 255, 255, 255), font=font,
        )
    except Exception:
        pass
    return img


# ── Flask ──────────────────────────────────────────────────────────────────────
def _run_flask():
    global _flask_started
    if _flask_started:
        return
    _flask_started = True
    print(f"[PILAR] Starting Flask on {APP_URL}")
    os.chdir(str(BASE_DIR))
    os.environ["PILAR_LAUNCHER"] = "1"
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import etape7
    etape7.app.run(
        host=APP_HOST, port=APP_PORT,
        debug=False, use_reloader=False, threaded=True,
    )


def _wait_for_flask(timeout: float = 60.0) -> bool:
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(APP_URL + "/api/health", timeout=2)
            return True
        except urllib.error.HTTPError:
            return True          # Le serveur repond (meme avec erreur HTTP) = pret
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Fenetre Edge app mode ──────────────────────────────────────────────────────
def _find_edge() -> str | None:
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    pf64 = os.environ.get("PROGRAMFILES",       r"C:\Program Files")
    candidates = [
        os.path.join(pf86, r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(pf64, r"Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _find_chrome() -> str | None:
    pf   = os.environ.get("PROGRAMFILES",       r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    loc  = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(pf,   r"Google\Chrome\Application\chrome.exe"),
        os.path.join(pf86, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(loc,  r"Google\Chrome\Application\chrome.exe"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _open_window():
    """Ouvre PILAR dans une fenetre app (sans URL, sans onglets)."""
    edge = _find_edge()
    if edge:
        subprocess.Popen([
            edge,
            f"--app={APP_URL}",
            f"--user-data-dir={EDGE_PROFILE}",
            "--window-size=1280,800",
            "--no-first-run",
            "--disable-sync",
            "--disable-extensions",
            "--disable-default-browser-check",
            "--no-default-browser-check",
        ])
        print("[PILAR] Opened in Edge app mode")
        return

    chrome = _find_chrome()
    if chrome:
        subprocess.Popen([
            chrome,
            f"--app={APP_URL}",
            f"--user-data-dir={EDGE_PROFILE}",
            "--window-size=1280,800",
            "--no-first-run",
            "--disable-extensions",
        ])
        print("[PILAR] Opened in Chrome app mode")
        return

    # Dernier recours
    import webbrowser
    webbrowser.open(APP_URL)
    print(f"[PILAR] Opened in default browser: {APP_URL}")


# ── Tray ───────────────────────────────────────────────────────────────────────
def action_open(icon, item):
    _open_window()


def action_quit(icon, item):
    print("[PILAR] Quit requested")
    icon.stop()


def _build_menu():
    return pystray.Menu(
        pystray.MenuItem("Ouvrir PILAR", action_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quitter PILAR", action_quit),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global _tray_icon

    print(f"[PILAR] Launcher starting (frozen={_FROZEN}, base={BASE_DIR})")

    # 1. Demarrer Flask en thread daemon
    flask_thread = threading.Thread(target=_run_flask, daemon=True, name="flask")
    flask_thread.start()

    # 2. Attendre que Flask soit pret
    print("[PILAR] Waiting for Flask...")
    if not _wait_for_flask(60):
        print("[PILAR] ERROR: Flask did not start in 60 s — check logs")
        sys.exit(1)
    print("[PILAR] Flask ready")

    # 3. Ouvrir la fenetre
    _open_window()

    # 4. Tray icon — boucle principale (bloquant jusqu'a Quitter)
    icon_image = _make_icon(64)
    _tray_icon = pystray.Icon(
        name="pilar",
        icon=icon_image,
        title="PILAR",
        menu=_build_menu(),
    )
    print("[PILAR] Tray icon running")
    _tray_icon.run()

    print("[PILAR] Done")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
