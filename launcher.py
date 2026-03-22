"""
PILAR Desktop Launcher
======================
System tray launcher for PILAR Desktop.
- Starts the Flask app (etape7.py) as a subprocess
- Opens the browser automatically
- Provides tray menu: Open / Restart / Quit
- Shows a teal circle icon in the system tray

Dependencies: pystray, Pillow
Install: pip install pystray pillow
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Configuration ──────────────────────────────────────────────────────────────
APP_HOST  = "127.0.0.1"
APP_PORT  = 5000
APP_URL   = f"http://{APP_HOST}:{APP_PORT}"
APP_TITLE = "PILAR"

# Path to etape7.py (same directory as this launcher)
BASE_DIR  = Path(__file__).parent.resolve()
APP_SCRIPT= BASE_DIR / "etape7.py"
PYTHON    = sys.executable

# ── State ─────────────────────────────────────────────────────────────────────
_flask_proc: subprocess.Popen | None = None
_tray_icon: pystray.Icon | None = None


# ── Icon ──────────────────────────────────────────────────────────────────────
def _make_icon(size: int = 64) -> Image.Image:
    """
    Draw a teal circle with 'P' letter — simple, clean tray icon.
    Colors match the PILAR teal dark theme.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer circle (dark background)
    draw.ellipse([2, 2, size - 2, size - 2], fill=(14, 17, 24, 255))
    # Inner teal circle
    pad = size // 6
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(13, 148, 136, 255))
    # Highlight ring
    draw.ellipse([pad + 2, pad + 2, size - pad - 2, size - pad - 2],
                 outline=(20, 184, 166, 180), width=2)
    # 'P' letter in white, centered
    font_size = size // 3
    try:
        from PIL import ImageFont
        # Try to load a system font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "P", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2
        y = (size - th) // 2 - 1
        draw.text((x, y), "P", fill=(255, 255, 255, 255), font=font)
    except Exception:
        pass
    return img


# ── Flask Process Management ───────────────────────────────────────────────────
def _start_flask() -> subprocess.Popen:
    """Launch etape7.py as a subprocess."""
    env = os.environ.copy()
    env["PILAR_LAUNCHER"] = "1"   # signals the app it's running in desktop mode
    proc = subprocess.Popen(
        [PYTHON, str(APP_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[PILAR Launcher] Flask started (PID {proc.pid})")
    return proc


def _stop_flask(proc: subprocess.Popen):
    """Gracefully terminate the Flask process."""
    if proc is None or proc.poll() is not None:
        return
    print(f"[PILAR Launcher] Stopping Flask (PID {proc.pid})...")
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception as e:
        print(f"[PILAR Launcher] Stop error: {e}")
        try:
            proc.kill()
        except Exception:
            pass
    print("[PILAR Launcher] Flask stopped")


def _log_reader(proc: subprocess.Popen):
    """Thread: reads Flask stdout and prints it."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if line:
                print(f"[Flask] {line.rstrip()}")
    except Exception:
        pass


def _wait_for_flask(timeout: float = 15.0) -> bool:
    """Poll until Flask is accepting connections (or timeout)."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(APP_URL + "/health", timeout=1)
            return True
        except Exception:
            pass
        # Also check if process has crashed
        if _flask_proc and _flask_proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def _open_browser():
    """Open the PILAR app in the default browser."""
    webbrowser.open(APP_URL)


# ── Tray Actions ──────────────────────────────────────────────────────────────
def action_open(icon: pystray.Icon, item):
    """Open the browser to PILAR."""
    _open_browser()


def action_restart(icon: pystray.Icon, item):
    """Kill Flask and restart it, then reopen browser."""
    global _flask_proc
    icon.title = "PILAR — Restarting…"
    _stop_flask(_flask_proc)
    time.sleep(1)
    _flask_proc = _start_flask()
    threading.Thread(target=_log_reader, args=(_flask_proc,), daemon=True).start()
    # Wait then open browser
    def _after_restart():
        if _wait_for_flask(20):
            _open_browser()
            icon.title = "PILAR"
        else:
            icon.title = "PILAR — Start failed"
    threading.Thread(target=_after_restart, daemon=True).start()


def action_quit(icon: pystray.Icon, item):
    """Clean shutdown: stop Flask, then exit tray."""
    global _flask_proc
    icon.title = "PILAR — Shutting down…"
    _stop_flask(_flask_proc)
    _flask_proc = None
    icon.stop()


# ── Menu ──────────────────────────────────────────────────────────────────────
def _build_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem("Open PILAR",  action_open,    default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart",     action_restart),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit PILAR",  action_quit),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _flask_proc, _tray_icon

    print("[PILAR Launcher] Starting…")

    # 1. Start Flask
    _flask_proc = _start_flask()
    threading.Thread(target=_log_reader, args=(_flask_proc,), daemon=True).start()

    # 2. Wait for Flask in background, then open browser
    def _startup_sequence():
        print("[PILAR Launcher] Waiting for Flask to be ready…")
        ready = _wait_for_flask(30)
        if ready:
            print("[PILAR Launcher] Flask ready — opening browser")
            _open_browser()
            if _tray_icon:
                _tray_icon.title = "PILAR"
        else:
            print("[PILAR Launcher] Flask did not start in time")
            if _tray_icon:
                _tray_icon.title = "PILAR — Failed to start"

    threading.Thread(target=_startup_sequence, daemon=True).start()

    # 3. Create & run system tray icon
    icon_image = _make_icon(64)
    _tray_icon = pystray.Icon(
        name  = "pilar",
        icon  = icon_image,
        title = "PILAR — Starting…",
        menu  = _build_menu(),
    )

    print("[PILAR Launcher] Tray icon running")
    _tray_icon.run()   # blocks until icon.stop() is called

    # 4. Cleanup after tray exits
    print("[PILAR Launcher] Tray closed — cleaning up")
    _stop_flask(_flask_proc)
    print("[PILAR Launcher] Done")


if __name__ == "__main__":
    main()
