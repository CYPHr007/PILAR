"""
PILAR Desktop Launcher
======================
Lance Flask en thread background, puis ouvre une fenetre native pywebview
(aucune barre URL, aucun onglet — vraie app desktop).

- Tray icon : Ouvrir PILAR / Quitter PILAR
- pywebview utilise WebView2 (Edge runtime) sous le capot
"""

import os
import sys
import time
import threading
import multiprocessing
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Config ─────────────────────────────────────────────────────────────────────
_FROZEN   = getattr(sys, "frozen", False)
APP_HOST  = "127.0.0.1"
APP_PORT  = 5000
APP_URL   = f"http://{APP_HOST}:{APP_PORT}/monitor"

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).parent.resolve()

# ── State ──────────────────────────────────────────────────────────────────────
_flask_started = False
_window        = None
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
            return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Tray actions ───────────────────────────────────────────────────────────────
def action_open(icon, item):
    global _window
    if _window is not None:
        try:
            _window.show()
            _window.restore()
        except Exception:
            pass


def action_quit(icon, item):
    global _window
    print("[PILAR] Quit requested")
    if _window is not None:
        try:
            _window.destroy()
        except Exception:
            pass
    icon.stop()


def _build_menu():
    return pystray.Menu(
        pystray.MenuItem("Ouvrir PILAR", action_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quitter PILAR", action_quit),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global _window, _tray_icon

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

    # 3. Tray icon en thread detache (pywebview doit etre dans le main thread)
    icon_image = _make_icon(64)
    _tray_icon = pystray.Icon(
        name="pilar",
        icon=icon_image,
        title="PILAR",
        menu=_build_menu(),
    )
    _tray_icon.run_detached()
    print("[PILAR] Tray icon running")

    # 4. Fenetre native pywebview — main thread
    import webview
    _window = webview.create_window(
        title="PILAR",
        url=APP_URL,
        width=1280,
        height=800,
        min_size=(900, 600),
        resizable=True,
        text_select=False,
    )
    webview.start()

    # 5. Fenetre fermee — arreter le tray
    print("[PILAR] Window closed, stopping tray")
    if _tray_icon:
        _tray_icon.stop()

    print("[PILAR] Done")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
