"""
PILAR Desktop Launcher
======================
System tray launcher for PILAR Desktop.
- Starts the Flask app (etape7.py) as a subprocess (dev) or thread (bundled exe)
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
import multiprocessing
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Frozen detection ────────────────────────────────────────────────────────────
# When bundled by PyInstaller, sys.frozen is True and sys.executable = PILAR.exe.
# We must NOT use subprocess in that case — it would spawn another PILAR.exe
# recursively, causing infinite windows.
_FROZEN = getattr(sys, "frozen", False)

# ── Configuration ──────────────────────────────────────────────────────────────
APP_HOST  = "127.0.0.1"
APP_PORT  = 5000
APP_URL   = f"http://{APP_HOST}:{APP_PORT}"
APP_TITLE = "PILAR"

# Path resolution differs between frozen and dev modes
if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).parent.resolve()

APP_SCRIPT = BASE_DIR / "etape7.py"
PYTHON     = sys.executable  # only safe to use in dev mode

# ── State ─────────────────────────────────────────────────────────────────────
_flask_proc: subprocess.Popen | None = None
_flask_thread: threading.Thread | None = None
_tray_icon: pystray.Icon | None = None
_flask_started = False   # guard: only start Flask once (frozen mode)


# ── Icon ──────────────────────────────────────────────────────────────────────
def _make_icon(size: int = 64) -> Image.Image:
    """Draw a teal circle with 'P' letter — simple, clean tray icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.ellipse([2, 2, size - 2, size - 2], fill=(14, 17, 24, 255))
    pad = size // 6
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(13, 148, 136, 255))
    draw.ellipse([pad + 2, pad + 2, size - pad - 2, size - pad - 2],
                 outline=(20, 184, 166, 180), width=2)
    font_size = size // 3
    try:
        from PIL import ImageFont
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


# ── Flask Startup ──────────────────────────────────────────────────────────────
def _start_flask_thread():
    """Run Flask inside this same process (used when frozen as .exe)."""
    global _flask_started
    if _flask_started:
        return
    _flask_started = True

    print("[PILAR Launcher] Starting Flask in-process (frozen mode)…")
    # We need to set the working directory so etape7 can find its files
    os.chdir(str(BASE_DIR))
    os.environ["PILAR_LAUNCHER"] = "1"

    # Add BASE_DIR to sys.path so `import etape7` works
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    import etape7
    # Run with use_reloader=False and threaded=True — critical inside a thread
    etape7.app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


def _start_flask_subprocess() -> subprocess.Popen:
    """Launch etape7.py as a subprocess (dev mode only)."""
    env = os.environ.copy()
    env["PILAR_LAUNCHER"] = "1"
    proc = subprocess.Popen(
        [PYTHON, str(APP_SCRIPT)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"[PILAR Launcher] Flask started as subprocess (PID {proc.pid})")
    return proc


def _start_flask():
    """Start Flask — thread if frozen, subprocess if dev."""
    global _flask_proc, _flask_thread
    if _FROZEN:
        t = threading.Thread(target=_start_flask_thread, daemon=True, name="flask-server")
        t.start()
        _flask_thread = t
        _flask_proc = None
    else:
        _flask_proc = _start_flask_subprocess()
        threading.Thread(target=_log_reader, args=(_flask_proc,), daemon=True).start()


def _stop_flask():
    """Stop Flask — only meaningful in dev (subprocess) mode."""
    global _flask_proc
    if _FROZEN:
        # Flask thread is daemon — it dies when the process exits
        print("[PILAR Launcher] Frozen mode: Flask thread will stop with process")
        return
    if _flask_proc is None or _flask_proc.poll() is not None:
        return
    print(f"[PILAR Launcher] Stopping Flask (PID {_flask_proc.pid})…")
    try:
        if sys.platform == "win32":
            _flask_proc.terminate()
        else:
            _flask_proc.send_signal(signal.SIGTERM)
        _flask_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _flask_proc.kill()
        _flask_proc.wait()
    except Exception as e:
        print(f"[PILAR Launcher] Stop error: {e}")
        try:
            _flask_proc.kill()
        except Exception:
            pass
    print("[PILAR Launcher] Flask stopped")


def _log_reader(proc: subprocess.Popen):
    """Thread: reads Flask stdout and prints it (dev mode only)."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if line:
                print(f"[Flask] {line.rstrip()}")
    except Exception:
        pass


def _wait_for_flask(timeout: float = 30.0) -> bool:
    """Poll until Flask is accepting connections (or timeout)."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(APP_URL + "/api/health", timeout=1)
            return True
        except Exception:
            pass
        # In subprocess mode, check if it crashed
        if not _FROZEN and _flask_proc and _flask_proc.poll() is not None:
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
    """Restart Flask (dev mode) or just reopen browser (frozen mode)."""
    icon.title = "PILAR — Restarting…"
    if _FROZEN:
        # Can't restart a thread; just reopen the browser
        _open_browser()
        icon.title = "PILAR"
        return
    # Dev mode: kill and restart subprocess
    _stop_flask()
    time.sleep(1)
    _start_flask()
    def _after_restart():
        if _wait_for_flask(20):
            _open_browser()
            icon.title = "PILAR"
        else:
            icon.title = "PILAR — Start failed"
    threading.Thread(target=_after_restart, daemon=True).start()


def action_quit(icon: pystray.Icon, item):
    """Clean shutdown: stop Flask, then exit tray."""
    icon.title = "PILAR — Shutting down…"
    _stop_flask()
    icon.stop()


# ── Menu ──────────────────────────────────────────────────────────────────────
def _build_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem("Open PILAR",  action_open,  default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart",     action_restart),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit PILAR",  action_quit),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _tray_icon

    print(f"[PILAR Launcher] Starting… (frozen={_FROZEN})")
    print(f"[PILAR Launcher] BASE_DIR = {BASE_DIR}")

    # 1. Start Flask (thread or subprocess)
    _start_flask()

    # 2. Wait for Flask in background, then open browser once
    def _startup_sequence():
        print("[PILAR Launcher] Waiting for Flask to be ready…")
        ready = _wait_for_flask(60)
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
    _tray_icon.run()  # blocks until icon.stop() is called

    # 4. Cleanup
    print("[PILAR Launcher] Tray closed — cleaning up")
    _stop_flask()
    print("[PILAR Launcher] Done")


if __name__ == "__main__":
    # CRITICAL for Windows frozen executables: prevents recursive subprocess spawning
    multiprocessing.freeze_support()
    main()
