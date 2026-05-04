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
try:
    from config import APP_VERSION   # single source of truth
except ImportError:
    APP_VERSION = "4.0"              # fallback when frozen
GITHUB_API   = "https://api.github.com/repos/CYPHR007/PILAR/releases/latest"
PILAR_API    = os.environ.get("PILAR_UPDATE_URL", GITHUB_API)

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).parent.resolve()

# ── State ──────────────────────────────────────────────────────────────────────
_flask_started  = False
_window         = None
_tray_icon      = None
_UPDATE_PENDING = {}   # filled by _check_update when a newer version is found

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
    """Two-bar PILAR logo mark on dark background."""
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark rounded-square background
    r = size // 8
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=(14, 17, 24, 255))
    # Two vertical bars: white + gray, centered
    bar_w   = max(5, size // 10)
    bar_h   = size // 2
    gap     = max(3, size // 16)
    total_w = bar_w * 2 + gap
    x0 = (size - total_w) // 2
    y0 = (size - bar_h) // 2
    # Bar 1 — white
    draw.rounded_rectangle(
        [x0, y0, x0 + bar_w - 1, y0 + bar_h - 1],
        radius=max(1, bar_w // 4),
        fill=(226, 232, 240, 255),
    )
    # Bar 2 — gray
    draw.rounded_rectangle(
        [x0 + bar_w + gap, y0, x0 + bar_w + gap + bar_w - 1, y0 + bar_h - 1],
        radius=max(1, bar_w // 4),
        fill=(100, 116, 139, 255),
    )
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
    os.environ["PILAR_DESKTOP"] = "1"
    os.environ.setdefault("PILAR_VERSION", APP_VERSION)
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    import app
    app.app.run(
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
    Check PILAR website first, then fall back to GitHub API.
    Returns (version_str, download_url, is_zip) or (None, None, False).
    """
    import urllib.request
    import ssl
    import json

    def _open(url, timeout=10):
        req = urllib.request.Request(url, headers={"User-Agent": f"PILAR-Desktop/{APP_VERSION}"})
        try:
            ctx = ssl.create_default_context()
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except Exception:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)

    # --- Primary: PILAR website ---
    try:
        with _open(PILAR_API, timeout=8) as resp:
            data = json.loads(resp.read())
        tag = data.get("version", "")
        url = data.get("download_url", "")
        if tag and url:
            print(f"[PILAR] Update info from website: {tag}")
            return tag, url, url.endswith(".zip")
    except Exception as e:
        print(f"[PILAR] Website check failed: {e} — trying GitHub API")

    # --- Fallback: GitHub API ---
    try:
        with _open(GITHUB_API, timeout=8) as resp:
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
        return tag, data.get("zipball_url", ""), True
    except Exception as e:
        print(f"[PILAR] GitHub API check failed: {e}")
        return None, None, False


def _download_and_install(url: str, version: str, is_zip: bool = False):
    """
    Download the new installer (.exe) and launch it silently.
    Uses chunked download with SSL context to handle certificate errors on Windows.
    """
    import urllib.request, ssl, zipfile, shutil
    global _window, _tray_icon

    print(f"[PILAR] Downloading update v{version} from {url}")

    def _chunked_download(src_url, dest_path):
        """Download with SSL context, redirect following, progress output."""
        req = urllib.request.Request(src_url, headers={"User-Agent": f"PILAR-Desktop/{APP_VERSION}"})
        resp = None
        for verify in (True, False):
            try:
                ctx = ssl.create_default_context()
                if not verify:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=120, context=ctx)
                break
            except ssl.SSLError:
                if not verify:
                    raise
                continue
        if resp is None:
            raise RuntimeError("Could not open URL")
        total = int(resp.getheader("Content-Length") or 0)
        downloaded = 0
        with open(dest_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    print(f"[PILAR] Download: {pct}%  ({downloaded//(1024*1024)}MB/{total//(1024*1024)}MB)", end="\r", flush=True)
        print()
        resp.close()

    try:
        suffix = ".zip" if is_zip else ".exe"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"PILAR_{version}_")
        tmp_path = tmp.name
        tmp.close()

        _chunked_download(url, tmp_path)
        print(f"[PILAR] Download complete: {tmp_path}")

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
            # Inno Setup silent install — InitializeSetup() does taskkill /F on PILAR.exe
            subprocess.Popen([tmp_path, "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
                             close_fds=True)
            print("[PILAR] Silent installer launched — exiting current process")

        time.sleep(1)
        os._exit(0)

    except Exception as e:
        print(f"[PILAR] Download/install failed: {e}")
        _show_error_dialog(f"Echec de la mise a jour : {e}")

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
        # Non-blocking tray notification — no modal popup on launch
        if _tray_icon:
            try:
                _tray_icon.notify(
                    f"PILAR {tag} disponible — cliquez sur 'Rechercher les mises à jour…' pour installer.",
                    "PILAR — Mise à jour disponible",
                )
            except Exception:
                pass
        # Store for manual check trigger
        _UPDATE_PENDING['tag'] = tag
        _UPDATE_PENDING['url'] = url
        _UPDATE_PENDING['is_zip'] = is_zip
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
    # If we already detected an update at startup, use cached result
    if _UPDATE_PENDING.get('tag'):
        tag, url, is_zip = _UPDATE_PENDING['tag'], _UPDATE_PENDING['url'], _UPDATE_PENDING.get('is_zip', False)
    else:
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
        import pilar_monitor
        for m in list(pilar_monitor.active_monitors.values()):
            m['stop'].set()
        pilar_monitor.active_monitors.clear()
    except Exception:
        pass
    if _window is not None:
        try:
            _window.destroy()
        except Exception:
            pass
    icon.stop()


def action_reset_agents(icon, item):
    """Delete bootstrap markers so AI agent setup runs again on next launch."""
    import ctypes
    try:
        import pilar_bootstrap
        pilar_bootstrap.reset_bootstrap()
        ctypes.windll.user32.MessageBoxW(
            0,
            "Configuration des agents IA réinitialisée.\n\n"
            "Redémarrez PILAR pour relancer la configuration automatique.",
            "PILAR — Agents IA",
            0x40,  # MB_ICONINFO
        )
    except Exception as e:
        ctypes.windll.user32.MessageBoxW(
            0, f"Erreur : {e}", "PILAR — Agents IA", 0x10)


def action_demo(icon, item):
    """Open PILAR in demo mode (auto-login as demo@pilar.app)."""
    global _window
    demo_url = f"http://{APP_HOST}:{APP_PORT}/demo-login"
    if _window is not None:
        try:
            _window.load_url(demo_url)
            _window.show()
            _window.restore()
        except Exception:
            pass


def action_train(icon, item):
    """Open the ML training panel."""
    global _window
    train_url = f"http://{APP_HOST}:{APP_PORT}/train"
    if _window is not None:
        try:
            _window.load_url(train_url)
            _window.show()
            _window.restore()
        except Exception:
            pass


def action_download_ai(icon, item):
    """Trigger AI model download from tray menu (non-blocking)."""
    def _on_notify(title, msg):
        if _tray_icon:
            try:
                _tray_icon.notify(msg, title)
            except Exception:
                pass
    threading.Thread(
        target=lambda: _safe_bootstrap_download(_on_notify),
        daemon=True,
        name="ai-download",
    ).start()


def _safe_bootstrap_download(notify):
    try:
        import pilar_bootstrap
        pilar_bootstrap.prompt_download_from_tray(notify=notify)
    except Exception as e:
        print(f"[PILAR] AI download error: {e}")


def _build_menu():
    return pystray.Menu(
        pystray.MenuItem("Ouvrir PILAR", action_open, default=True),
        pystray.MenuItem("Mode démo (maintenance teams)", action_demo),
        pystray.MenuItem("Entraîner le modèle ML…", action_train),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Télécharger les modèles IA…", action_download_ai),
        pystray.MenuItem("Rechercher les mises à jour…", action_check_update),
        pystray.MenuItem("Réinitialiser les agents IA…", action_reset_agents),
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
    #    Must be created BEFORE bootstrap so tray notifications work.
    try:
        icon_image = Image.open(BASE_DIR / 'pilar.png').resize((64, 64), Image.LANCZOS).convert('RGBA')
    except Exception:
        icon_image = _make_icon(64)
    _tray_icon = pystray.Icon(
        name="pilar",
        icon=icon_image,
        title="PILAR",
        menu=_build_menu(),
    )
    _tray_icon.run_detached()
    print("[PILAR] Tray icon running")

    # 4b. Bootstrap PILAR AI agents (Ollama + pilar-diag/-maintenance/-alert)
    #     First run only: auto-pulls llama3.2 and creates the 3 custom models
    #     from bundled Modelfiles. Shows install dialog if Ollama is missing.
    def _bootstrap_notify(title, msg):
        if _tray_icon:
            try:
                _tray_icon.notify(msg, title)
            except Exception:
                pass
    try:
        import pilar_bootstrap
        pilar_bootstrap.bootstrap_async(notify=_bootstrap_notify)
    except Exception as e:
        print(f"[PILAR] Bootstrap init failed: {e}")

    # 5. Wire notification callback into app so background monitor
    #    can send tray notifications
    try:
        import app as _et
        def _on_alert(title, msg):
            if _tray_icon:
                try:
                    _tray_icon.notify(msg, title)
                except Exception:
                    pass
        import pilar_monitor
        pilar_monitor.notify_callback = _on_alert
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
                import pilar_monitor as _pm2
                active = len([m for m in _pm2.active_monitors.values() if not m['stop'].is_set()])
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
