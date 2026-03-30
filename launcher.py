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
APP_VERSION  = "1.2.2"   # bump this with every release
GITHUB_API   = "https://api.github.com/repos/CYPHr007/PILAR/releases/latest"

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
    - Si .exe (installer Inno Setup) : telecharge et lance directement.
    - Si .zip : telecharge, extrait dans %LOCALAPPDATA%\\PILAR_update\\,
      lance PILAR.exe depuis ce dossier, ferme l'app courante.
    """
    import urllib.request
    import zipfile
    import shutil
    global _window, _tray_icon

    print(f"[PILAR] Downloading update from {url} (zip={is_zip})")

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_count * block_size * 100 // total_size)
            print(f"\r[PILAR] Download: {pct}%", end="", flush=True)

    try:
        suffix = ".zip" if is_zip else ".exe"
        prefix = f"PILAR_{version}_"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix)
        tmp_path = tmp.name
        tmp.close()

        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print()
        print(f"[PILAR] Download complete: {tmp_path}")

        if is_zip:
            # Extrait dans un dossier dedie
            update_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "PILAR_update"
            if update_dir.exists():
                shutil.rmtree(update_dir)
            update_dir.mkdir(parents=True)
            with zipfile.ZipFile(tmp_path, "r") as z:
                z.extractall(update_dir)
            os.unlink(tmp_path)
            # Cherche PILAR.exe dans l'arborescence extraite
            pilar_exe = None
            for f in update_dir.rglob("PILAR.exe"):
                pilar_exe = f
                break
            if pilar_exe is None:
                raise FileNotFoundError("PILAR.exe introuvable dans l'archive.")
            subprocess.Popen([str(pilar_exe)], cwd=str(pilar_exe.parent), close_fds=True)
            print(f"[PILAR] Launched new version from {pilar_exe}")
        else:
            subprocess.Popen([tmp_path], close_fds=True)
            print("[PILAR] Installer launched")

        # Ferme l'app courante
        if _window is not None:
            try:
                _window.destroy()
            except Exception:
                pass
        if _tray_icon is not None:
            try:
                _tray_icon.stop()
            except Exception:
                pass

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
        f"(L'application se fermera et l'installeur se lancera automatiquement.)"
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
        pystray.MenuItem("Rechercher les mises à jour…", action_check_update),
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

    # 5. Fenetre native pywebview — main thread
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

    # 6. Fenetre fermee — arreter le tray
    print("[PILAR] Window closed, stopping tray")
    if _tray_icon:
        _tray_icon.stop()

    print("[PILAR] Done")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
