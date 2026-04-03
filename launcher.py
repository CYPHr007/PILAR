"""
PILAR Desktop Launcher
======================
Lance Flask en thread background, puis ouvre une fenetre native pywebview
(aucune barre URL, aucun onglet — vraie app desktop).

- Tray icon : Ouvrir PILAR / Quitter PILAR
- pywebview utilise WebView2 (Edge runtime) sous le capot
- Auto-updater : verifie GitHub Releases au demarrage, propose mise a jour si nouvelle version
"""

import os
import sys
import time
import threading
import multiprocessing
import tempfile
import subprocess
from pathlib import Path
from packaging.version import Version

import pystray
from PIL import Image, ImageDraw

# ── Config ─────────────────────────────────────────────────────────────────────
_FROZEN      = getattr(sys, "frozen", False)
APP_HOST     = "127.0.0.1"
APP_PORT     = 5000
APP_URL      = f"http://{APP_HOST}:{APP_PORT}/monitor"
APP_VERSION  = "1.2.18"   # bump this with every release
GITHUB_API   = "https://api.github.com/repos/CYPHr007/PILAR/releases/latest"

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).parent.resolve()

# ── State ──────────────────────────────────────────────────────────────────────
_flask_started = False
_window        = None
_tray_icon     = None

# ── Single-instance lock ───────────────────────────────────────────────────────
_FOCUS_PORT = 19847   # local-only socket used to signal the running instance

def _is_already_running() -> bool:
    """
    Try to connect to the focus-listener of an existing instance.
    If successful → send 'focus' and return True (caller should exit).
    If no listener → return False (we are the first instance).
    """
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", _FOCUS_PORT), timeout=1)
        s.sendall(b"focus")
        s.close()
        return True
    except OSError:
        return False

def _start_focus_listener():
    """
    Listen on _FOCUS_PORT. When another instance sends 'focus',
    show and restore the main window.
    Runs in a daemon thread — never blocks shutdown.
    """
    import socket
    def _serve():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.bind(("127.0.0.1", _FOCUS_PORT))
            except OSError:
                return   # port already taken — another instance is primary
            srv.listen(5)
            srv.settimeout(1)
            while True:
                try:
                    conn, _ = srv.accept()
                    with conn:
                        conn.recv(16)
                    # Bring window to front
                    if _window is not None:
                        try:
                            _window.show()
                            _window.restore()
                        except Exception:
                            pass
                except socket.timeout:
                    continue
                except Exception:
                    break
    threading.Thread(target=_serve, daemon=True, name="focus-listener").start()


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
    os.environ.setdefault("PILAR_VERSION", APP_VERSION)
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


# ── Auto-updater ───────────────────────────────────────────────────────────────
def _parse_version(tag: str) -> Version:
    """Parse a git tag like 'v1.2.3' or '1.2.3' into a Version object."""
    return Version(tag.lstrip("v"))


def _fetch_latest_release():
    """
    Interroge l'API GitHub Releases.
    Retourne (latest_version_str, download_url, is_zip) ou (None, None, False).
    Priorite : Setup.exe > .zip > zipball_url
    """
    import urllib.request
    import json
    try:
        req = urllib.request.Request(
            GITHUB_API,
            headers={"User-Agent": f"PILAR-Desktop/{APP_VERSION}"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        tag = data.get("tag_name", "")
        assets = data.get("assets", [])
        exe_url = None
        zip_url = None
        for asset in assets:
            name = asset.get("name", "")
            url  = asset.get("browser_download_url", "")
            if name.endswith(".exe") and "Setup" in name and not exe_url:
                exe_url = url
            elif name.endswith(".zip") and not zip_url:
                zip_url = url
        if exe_url:
            return tag, exe_url, False
        if zip_url:
            return tag, zip_url, True
        # Dernier recours : archive source GitHub
        return tag, data.get("zipball_url", ""), True
    except Exception as e:
        print(f"[PILAR] Update check failed: {e}")
        return None, None, False


def _download_and_install(url: str, version: str, is_zip: bool = False):
    """
    Telecharge le nouvel installeur (.exe Inno Setup) et le lance silencieusement.
    Fallback zip: extrait dans LOCALAPPDATA et lance PILAR.exe.
    """
    import urllib.request, zipfile, shutil
    global _window, _tray_icon

    print(f"[PILAR] Downloading update v{version} from {url}")

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_count * block_size * 100 // total_size)
            print(f"\r[PILAR] Download: {pct}%", end="", flush=True)

    try:
        suffix = ".zip" if is_zip else ".exe"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"PILAR_{version}_")
        tmp_path = tmp.name
        tmp.close()

        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print(f"\n[PILAR] Download complete: {tmp_path}")

        if is_zip:
            update_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "PILAR_update"
            if update_dir.exists():
                shutil.rmtree(update_dir)
            update_dir.mkdir(parents=True)
            with zipfile.ZipFile(tmp_path, "r") as z:
                z.extractall(update_dir)
            os.unlink(tmp_path)
            pilar_exe = next(update_dir.rglob("PILAR.exe"), None)
            if pilar_exe is None:
                raise FileNotFoundError("PILAR.exe introuvable dans l'archive.")
            subprocess.Popen([str(pilar_exe)], cwd=str(pilar_exe.parent), close_fds=True)
            print(f"[PILAR] Launched updated version from {pilar_exe}")
        else:
            # Inno Setup silent install — kills running PILAR via InitializeSetup,
            # installs new files, then relaunches PILAR automatically.
            # Do NOT call _window.destroy() from this background thread — pywebview
            # requires window operations on the main thread and will crash otherwise.
            # The installer's InitializeSetup() does taskkill /F on PILAR.exe anyway.
            subprocess.Popen([tmp_path, "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
                             close_fds=True)
            print("[PILAR] Silent installer launched — exiting current process")

        # Give the installer process a moment to start before we vanish
        time.sleep(1)
        os._exit(0)

    except Exception as e:
        print(f"[PILAR] Download/install failed: {e}")
        _show_error_dialog(f"Echec de la mise a jour :\n{e}")


def _show_update_dialog(latest: str, url: str, is_zip: bool = False):
    """Affiche une boite de dialogue native Windows pour proposer la mise a jour."""
    import ctypes
    MB_YESNO    = 0x04
    MB_ICONINFO = 0x40
    IDYES       = 6

    msg = (
        f"Une nouvelle version de PILAR est disponible.\n\n"
        f"Version actuelle : {APP_VERSION}\n"
        f"Nouvelle version : {latest}\n\n"
        f"Voulez-vous mettre a jour maintenant ?\n"
        f"(Telechargement automatique — l'app redemarrera une fois installe.)"
    )
    result = ctypes.windll.user32.MessageBoxW(
        0, msg, "PILAR — Mise a jour disponible", MB_YESNO | MB_ICONINFO
    )
    if result == IDYES:
        _download_and_install(url, latest, is_zip)


def _show_error_dialog(msg: str):
    import ctypes
    MB_OK        = 0x00
    MB_ICONERROR = 0x10
    ctypes.windll.user32.MessageBoxW(0, msg, "PILAR — Erreur", MB_OK | MB_ICONERROR)


def _check_update():
    """Runs in background thread — non-bloquant."""
    print(f"[PILAR] Version {APP_VERSION} — checking for updates...")
    tag, url, is_zip = _fetch_latest_release()
    if not tag or not url:
        print("[PILAR] No update info available (offline or no release).")
        return
    try:
        latest  = _parse_version(tag)
        current = _parse_version(APP_VERSION)
    except Exception as e:
        print(f"[PILAR] Version parse error: {e}")
        return

    if latest > current:
        print(f"[PILAR] Update available: {tag} (zip={is_zip})")
        _show_update_dialog(tag, url, is_zip)
    else:
        print(f"[PILAR] Up to date ({APP_VERSION}).")


# ── Tray actions ───────────────────────────────────────────────────────────────
# ── pywebview API ──────────────────────────────────────────────────────────────
class PilarAPI:
    """Python methods exposed to JS via window.pywebview.api.*"""
    def pick_file(self):
        """Open native file dialog and return chosen path (or None)."""
        import webview as _wv
        if _window is None:
            return None
        try:
            result = _window.create_file_dialog(
                _wv.OPEN_DIALOG,
                allow_multiple=False,
                file_types=('CSV files (*.csv)', 'All files (*.*)')
            )
            if result and len(result) > 0:
                return result[0]
        except Exception as e:
            print(f"[PILAR] pick_file error: {e}")
        return None


def action_open(icon, item):
    global _window
    if _window is not None:
        try:
            _window.show()
            _window.restore()
        except Exception:
            pass


def action_check_update(icon, item):
    """Manual 'Check for updates' — runs in background thread, shows dialog if found."""
    import ctypes
    # Show "checking…" balloon tip while we query GitHub
    try:
        icon.notify("Vérification des mises à jour…", "PILAR")
    except Exception:
        pass
    threading.Thread(target=_check_update_manual, daemon=True, name="updater-manual").start()


def _check_update_manual():
    """Like _check_update() but also shows 'up to date' confirmation."""
    import ctypes
    print(f"[PILAR] Manual update check — version {APP_VERSION}")
    tag, url, is_zip = _fetch_latest_release()
    if not tag or not url:
        ctypes.windll.user32.MessageBoxW(
            0,
            "Impossible de joindre le serveur de mises à jour.\nVérifiez votre connexion internet.",
            "PILAR — Vérification des mises à jour",
            0x40,  # MB_ICONINFO
        )
        return
    try:
        latest  = _parse_version(tag)
        current = _parse_version(APP_VERSION)
    except Exception as e:
        print(f"[PILAR] Version parse error: {e}")
        return

    if latest > current:
        print(f"[PILAR] Update available: {tag} (zip={is_zip})")
        _show_update_dialog(tag, url, is_zip)
    else:
        ctypes.windll.user32.MessageBoxW(
            0,
            f"PILAR est à jour.\n\nVersion installée : {APP_VERSION}",
            "PILAR — Vérification des mises à jour",
            0x40,  # MB_ICONINFO
        )
        print(f"[PILAR] Up to date ({APP_VERSION}).")


def action_quit(icon, item):
    global _window
    print("[PILAR] Quit requested — stopping all background monitors")
    try:
        import etape7 as _et
        for m in list(_et._bg_monitors.values()):
            m['stop'].set()
        _et._bg_monitors.clear()
    except Exception:
        pass
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
        pystray.MenuItem("Rechercher les mises à jour…", action_check_update),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quitter PILAR (arrêter la surveillance)", action_quit),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global _window, _tray_icon

    # Single-instance check — if another PILAR is running, focus it and exit
    if _is_already_running():
        print("[PILAR] Already running — focused existing window, exiting.")
        sys.exit(0)
    _start_focus_listener()

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

    # 3. Check for updates in background (non-blocking)
    threading.Thread(target=_check_update, daemon=True, name="updater").start()

    # 4. Tray icon en thread detache (pywebview doit etre dans le main thread)
    icon_image = _make_icon(64)
    _tray_icon = pystray.Icon(
        name="pilar",
        icon=icon_image,
        title="PILAR",
        menu=_build_menu(),
    )
    _tray_icon.run_detached()
    print("[PILAR] Tray icon running")

    # 5. Wire notification callback into etape7 so background monitor
    #    can send tray notifications
    try:
        import etape7 as _et
        def _on_alert(title, msg):
            if _tray_icon:
                try:
                    _tray_icon.notify(msg, title)
                except Exception:
                    pass
        _et._notify_callback = _on_alert
        print("[PILAR] Notification callback wired")
    except Exception as e:
        print(f"[PILAR] Could not wire notify callback: {e}")

    # 6. Fenetre native pywebview — main thread
    import webview
    _api = PilarAPI()
    _window = webview.create_window(
        title="PILAR",
        url=APP_URL,
        width=1280,
        height=800,
        min_size=(900, 600),
        resizable=True,
        text_select=False,
        js_api=_api,
    )

    # Intercept close — hide to tray instead of quitting
    def _on_closing():
        _window.hide()
        if _tray_icon:
            try:
                # Check if any background monitor is active
                import etape7 as _et2
                active = len([m for m in _et2._bg_monitors.values() if not m['stop'].is_set()])
                if active:
                    _tray_icon.notify(
                        f"PILAR surveille {active} fichier(s) en arrière-plan",
                        "PILAR"
                    )
                else:
                    _tray_icon.notify(
                        "PILAR tourne en arrière-plan — double-clic pour rouvrir",
                        "PILAR"
                    )
            except Exception:
                pass
        return False   # prevent default close

    _window.events.closing += _on_closing

    _browser_data = str(BASE_DIR / "pilar_browser_data")
    webview.start(private_mode=False, storage_path=_browser_data)

    # webview.start() returns only after action_quit calls _window.destroy()
    print("[PILAR] Window destroyed, stopping tray")
    if _tray_icon:
        _tray_icon.stop()

    print("[PILAR] Done")


def main_cli():
    """
    CLI / server mode  —  PILAR runs as a local web server in the terminal.

    Usage:
        PILAR.exe --cli [--port 5000] [--no-browser]
        pilar            (via pilar.bat wrapper on PATH)

    The Flask server starts on 127.0.0.1:<port>, the URL is printed to
    stdout, and the default browser is opened automatically unless
    --no-browser is given.  Press Ctrl-C to stop.
    """
    global APP_PORT, APP_URL
    import argparse, webbrowser, signal

    # PILAR.exe is a GUI (windowless) binary — attach/allocate a console so
    # terminal output is visible when the user runs it from cmd/PowerShell.
    if _FROZEN and sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.AttachConsole(-1)  # attach to parent console
        # If there is no parent console (e.g. double-clicked), create one
        if ctypes.windll.kernel32.GetConsoleWindow() == 0:
            ctypes.windll.kernel32.AllocConsole()
        # Re-open standard streams so print() works
        import io
        try:
            sys.stdout = io.TextIOWrapper(open("CONOUT$", "wb", buffering=0), encoding="utf-8", line_buffering=True)
            sys.stderr = sys.stdout
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="PILAR", description="PILAR Predictive Maintenance — server mode")
    parser.add_argument("--port",       type=int, default=APP_PORT, help="Port to listen on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",        help="Do not open the browser automatically")
    args, _ = parser.parse_known_args()

    port = args.port
    url  = f"http://127.0.0.1:{port}"

    # Patch the global so Flask binds on the right port
    APP_PORT = port
    APP_URL  = f"{url}/monitor"

    print("=" * 52)
    print("  PILAR — Predictive Maintenance  (server mode)")
    print(f"  Version : {APP_VERSION}")
    print(f"  URL     : {url}")
    print("  Press Ctrl-C to stop.")
    print("=" * 52)

    flask_thread = threading.Thread(target=_run_flask, daemon=True, name="flask")
    flask_thread.start()

    if not _wait_for_flask(60):
        print("[PILAR] ERROR: Flask failed to start.")
        sys.exit(1)

    print(f"[PILAR] Server ready → {url}")

    if not args.no_browser:
        webbrowser.open(APP_URL)

    # Keep the main thread alive; daemon Flask thread lives as long as we do
    try:
        signal.pause()          # Unix
    except AttributeError:
        # Windows has no signal.pause() — just sleep forever
        import time as _t
        while True:
            _t.sleep(3600)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--cli" in sys.argv or "--server" in sys.argv:
        main_cli()
    else:
        main()
