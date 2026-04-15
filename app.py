# PILAR — Predictive Maintenance for Industrial Pumps
# Author  : CYPHR007
# License : MIT — see LICENSE
# Source  : https://github.com/CYPHR007/PILAR
# ──────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template_string, render_template, session, redirect, url_for, g
import pickle, threading, smtplib, secrets as _secrets, subprocess, atexit
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from extensions import db
from datetime import datetime, timedelta, timezone
import pandas as pd, numpy as np, warnings, time, collections
from scheduler_control import env_flag_enabled, release_file_lock, try_acquire_file_lock
from pilar_logging import get_logger
from pilar_validators import validate_sensor_input, generate_csrf_token, validate_csrf_token
import pilar_ml
import pilar_email
import pilar_monitor
warnings.filterwarnings("ignore")
logger = get_logger("pilar")

# Rate limiting : {ip: [(timestamp, failed_bool), ...]}
_login_attempts = collections.defaultdict(list)
# API rate limiting : {api_key: {'count': N, 'day': 'YYYY-MM-DD'}}
_api_calls = {}

def _check_rate_limit(ip):
    """Retourne True si l'IP est bloquée."""
    now = time.time()
    attempts = _login_attempts[ip]
    # Nettoyer les vieilles entrées
    _login_attempts[ip] = [t for t in attempts if now - t < RATE_WINDOW]
    return len(_login_attempts[ip]) >= RATE_MAX

def _record_failed_login(ip):
    _login_attempts[ip].append(time.time())

app = Flask(__name__)
import os
import sys as _sys
_FROZEN  = getattr(_sys, 'frozen', False)
# PyInstaller 6.x dir-build: data files land in _internal/ (sys._MEIPASS), not next to the exe
if _FROZEN:
    _APP_DIR  = getattr(_sys, '_MEIPASS', os.path.dirname(_sys.executable))
    # User data (DB, keys) must live NEXT TO the exe, not inside _internal/
    _DATA_DIR = os.path.dirname(_sys.executable)
else:
    _APP_DIR  = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = _APP_DIR
def _pkl(name): return os.path.join(_APP_DIR, name)
from config import (
    RATE_WINDOW, RATE_MAX, SESSION_DAYS, MAX_UPLOAD_MB,
    FAILURE_ZONES, COLONNES, FEATURE_MEDIANS, CORE_FEATURES, OPTIONAL_FIELDS,
    SENSOR_BOUNDS, FLUID_RUL_FACTORS, MATERIAL_RUL_FACTORS,
    FLUID_ZONE_SENSITIVITY, NON_CENTRIFUGE_TYPES, DOMAIN_KB,
    RETRAIN_TRIGGER, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CHAT_DAILY_LIMIT,
    RUL_SCALE_FACTOR, APP_VERSION,
    DEFAULT_THRESHOLD, ZONE_ALERT_THRESHOLD,
)
_db_path = os.path.join(_DATA_DIR, "pilar.db")
db_url = (os.environ.get("DATABASE_URL")
          or os.environ.get("DATABASE_PUBLIC_URL")
          or "sqlite:///" + _db_path.replace("\\", "/"))
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
logger.info(f"DB: {db_url[:80]}...")
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    _key_path = os.path.join(_DATA_DIR, "pilar_secret.key")
    try:
        with open(_key_path) as _f:
            _secret_key = _f.read().strip() or None
    except (FileNotFoundError, IOError):
        pass
    if not _secret_key:
        _secret_key = _secrets.token_hex(32)
        try:
            with open(_key_path, "w") as _f:
                _f.write(_secret_key)
            logger.info("SECRET_KEY generated and saved to pilar_secret.key")
        except Exception:
            logger.warning("SECRET_KEY random — sessions will reset on restart")
app.config["SECRET_KEY"] = _secret_key
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=SESSION_DAYS)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT", "RENDER", "DYNO"))
db.init_app(app)
from models import (
    Team, TeamMember, User, BannedEmail, Settings, Analysis, SavedFile,
    TeamMessage, DiscoveredParam, Machine, MachineNote, AlertLog,
    MachineRequest, MachineBaseline, MachineModel, MaintenanceEvent,
    SyncQueue, LocalChatMessage,
)

# ── CSRF Protection ──────────────────────────────────────────────────────────
from flask_wtf.csrf import CSRFProtect, CSRFError

# Only enforce CSRF on HTML form submissions (POST from browser forms).
# JSON API routes (/api/*, /predire) use session+API-key auth and are exempt.
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
csrf = CSRFProtect(app)

_CSRF_PROTECTED_ROUTES = {
    '/register', '/login', '/change_password', '/forgot-password',
    '/reset-password', '/set_email', '/resend-verification',
    '/admin/set_plan', '/admin/set_quota',
    '/admin/toggle_admin', '/admin/toggle_ban', '/admin/delete_user',
    '/admin/block_email', '/admin/retrain',
    '/team/create', '/team/invite', '/team/kick', '/team/transfer', '/team/leave',
}

@app.before_request
def _enforce_csrf_on_forms():
    """Manually enforce CSRF on HTML form POST routes only."""
    from flask import request as _req
    if _req.method not in ('POST', 'PUT', 'DELETE'):
        return
    path = _req.path
    for protected in _CSRF_PROTECTED_ROUTES:
        if path == protected or path.startswith(protected + '/'):
            csrf.protect()
            return

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return jsonify({'error': 'Session expired or invalid request. Please refresh the page.'}), 400

# ── API Rate Limiting ────────────────────────────────────────────────────────
_api_rate = collections.defaultdict(list)   # key: (uid, endpoint) → [timestamps]
API_RATE_WINDOW = 60            # seconds
API_RATE_MAX    = 100           # max requests per window per user

def _check_api_rate(uid, endpoint):
    """Returns True if rate limit exceeded."""
    key = (uid, endpoint)
    now = time.time()
    _api_rate[key] = [t for t in _api_rate[key] if t > now - API_RATE_WINDOW]
    if len(_api_rate[key]) >= API_RATE_MAX:
        return True
    _api_rate[key].append(now)
    return False

@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.plot.ly https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' http://localhost:* ws://localhost:*; "
        "font-src 'self' data:; "
        "frame-ancestors 'none'"
    )
    if any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT", "RENDER", "DYNO")):
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Disable caching for all API responses so fleet/machine data is always fresh
    from flask import request as _req
    if _req.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
    return response


with app.app_context():
    try:
        db.create_all()
        logger.info("Tables created/verified")
    except Exception as e:
        logger.error(f"db.create_all() error: {e}")
    _is_sqlite = db_url.startswith('sqlite')
    if _is_sqlite:
        _migrations = [
            "ALTER TABLE analysis ADD COLUMN user_id INTEGER",
            "ALTER TABLE settings ADD COLUMN user_id INTEGER",
            "ALTER TABLE user ADD COLUMN team_id INTEGER",
            "ALTER TABLE analysis ADD COLUMN extra_params TEXT",
            "ALTER TABLE analysis ADD COLUMN confidence INTEGER",
            "ALTER TABLE analysis ADD COLUMN machine_id VARCHAR(100)",
            "ALTER TABLE user ADD COLUMN plan_expires_at DATETIME",
            "ALTER TABLE user ADD COLUMN plan_note TEXT",
            "ALTER TABLE saved_file ADD COLUMN team_id INTEGER",
            "ALTER TABLE user ADD COLUMN is_banned INTEGER DEFAULT 0",
            "ALTER TABLE analysis ADD COLUMN feedback VARCHAR(10)",
            "ALTER TABLE user ADD COLUMN onboarded INTEGER DEFAULT 0",
            "ALTER TABLE machine ADD COLUMN pump_type VARCHAR(50) DEFAULT 'centrifuge'",
            "ALTER TABLE machine ADD COLUMN fluid_type VARCHAR(50) DEFAULT 'eau'",
            "ALTER TABLE machine ADD COLUMN roue_material VARCHAR(50) DEFAULT 'inox_316'",
            "ALTER TABLE user ADD COLUMN machine_quota INTEGER DEFAULT 3",
            "ALTER TABLE machine ADD COLUMN location VARCHAR(200)",
            "ALTER TABLE machine ADD COLUMN install_date DATE",
            "ALTER TABLE machine ADD COLUMN serial_number VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN nominal_flow REAL",
            "ALTER TABLE machine ADD COLUMN nominal_pressure REAL",
            "ALTER TABLE machine ADD COLUMN power_kw REAL",
            "ALTER TABLE machine ADD COLUMN nominal_current REAL",
            "ALTER TABLE machine ADD COLUMN nominal_vibration REAL",
            "ALTER TABLE saved_file ADD COLUMN machine_id INTEGER",
            "ALTER TABLE user ADD COLUMN reset_token VARCHAR(64)",
            "ALTER TABLE user ADD COLUMN reset_token_expires DATETIME",
            "ALTER TABLE machine ADD COLUMN asset_type VARCHAR(50)",
            "ALTER TABLE machine ADD COLUMN brand VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN model_name VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN age_years REAL DEFAULT 0",
            "ALTER TABLE machine ADD COLUMN environment VARCHAR(50)",
            "ALTER TABLE machine ADD COLUMN criticality VARCHAR(20) DEFAULT 'medium'",
            "ALTER TABLE machine ADD COLUMN last_maintenance DATETIME",
        ]
    else:
        _migrations = [
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS user_id INTEGER",
            "ALTER TABLE settings ADD COLUMN IF NOT EXISTS user_id INTEGER",
            "ALTER TABLE settings DROP CONSTRAINT IF EXISTS settings_key_key",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS team_id INTEGER',
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS extra_params TEXT",
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS confidence INTEGER",
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS machine_id VARCHAR(100)",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMP',
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS plan_note VARCHAR(300)',
            "ALTER TABLE saved_file ADD COLUMN IF NOT EXISTS team_id INTEGER",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_banned BOOLEAN DEFAULT FALSE',
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS feedback VARCHAR(10)",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS onboarded BOOLEAN DEFAULT FALSE',
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS pump_type VARCHAR(50) DEFAULT 'centrifuge'",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS fluid_type VARCHAR(50) DEFAULT 'eau'",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS roue_material VARCHAR(50) DEFAULT 'inox_316'",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS machine_quota INTEGER DEFAULT 3',
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS location VARCHAR(200)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS install_date DATE",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS serial_number VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS nominal_flow REAL",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS nominal_pressure REAL",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS power_kw REAL",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS nominal_current REAL",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS nominal_vibration REAL",
            "ALTER TABLE saved_file ADD COLUMN IF NOT EXISTS machine_id INTEGER",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS reset_token VARCHAR(64)',
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS reset_token_expires TIMESTAMP',
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS asset_type VARCHAR(50)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS brand VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS model_name VARCHAR(100)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS age_years REAL DEFAULT 0",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS environment VARCHAR(50)",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS criticality VARCHAR(20) DEFAULT 'medium'",
            "ALTER TABLE machine ADD COLUMN IF NOT EXISTS last_maintenance TIMESTAMP",
        ]
    for sql in _migrations:
        try:
            db.session.execute(db.text(sql))
            db.session.commit()
            logger.debug(f"Migration OK: {sql[:50]}")
        except Exception as e:
            db.session.rollback()
            logger.debug(f"Migration skip ({sql[:40]}): {e}")
    # SuperUser persistant (set SUPER_USER_EMAIL in environment)
    _su_email = os.environ.get("SUPER_USER_EMAIL", "")
    if _su_email:
        try:
            _su = User.query.filter_by(email=_su_email).first()
            if _su:
                _changed = False
                if not _su.is_admin or _su.plan != 'pro':
                    _su.is_admin = True; _su.plan = 'pro'; _su.plan_expires_at = None; _changed = True
                if not _su.onboarded:
                    _su.onboarded = True; _changed = True
                if _changed:
                    db.session.commit()
                    logger.info(f"SuperUser: {_su_email} → admin+pro lifetime")
        except Exception as _sue:
            db.session.rollback()
            logger.info(f"SuperUser setup: {_sue}")
    # Upgrade all existing free users to pro + mark onboarded
    try:
        _free_users = User.query.filter_by(plan='free').all()
        for _u in _free_users:
            _u.plan = 'pro'
            _u.plan_expires_at = None
            _u.onboarded = True
        if _free_users:
            db.session.commit()
            logger.info(f"Auto-upgraded {len(_free_users)} free user(s) to pro")
    except Exception as _upe:
        db.session.rollback()
        logger.info(f"Auto-upgrade error: {_upe}")
    # Mark all existing users as onboarded (onboarding added post-launch)
    try:
        _not_onboarded = User.query.filter_by(onboarded=False).all()
        for _u in _not_onboarded:
            _u.onboarded = True
        if _not_onboarded:
            db.session.commit()
            logger.info(f"Marked {len(_not_onboarded)} existing user(s) as onboarded")
    except Exception as _oue:
        db.session.rollback()
        logger.info(f"Onboard migration error: {_oue}")
def _seed_demo_account():
    """Create/refresh the demo@pilar.app account with rich sample data."""
    import random, json as _json
    from datetime import timedelta
    DEMO_EMAIL = 'demo@pilar.app'
    DEMO_PASS  = 'demo1234'

    user = User.query.filter_by(email=DEMO_EMAIL).first()
    if not user:
        user = User(
            email=DEMO_EMAIL,
            password_hash=generate_password_hash(DEMO_PASS),
            email_verified=True,
            onboarded=True,
            plan='pro',
            plan_expires_at=None,
        )
        db.session.add(user)
        db.session.flush()
        logger.info("Demo account created")
    else:
        # Already exists — only seed data if machines missing
        if Machine.query.filter_by(user_id=user.id).count() > 0:
            return
        logger.info("Demo account found — seeding machines & history")

    uid = user.id
    now = datetime.now(timezone.utc)

    # ── Machines ──────────────────────────────────────────────────────────────
    machines_def = [
        dict(name='Pompe Principale',    pump_type='centrifuge', fluid_type='eau',   roue_material='inox_316',
             nominal_flow=45.0, nominal_pressure=6.5, nominal_current=18.0, nominal_vibration=2.5,
             power_kw=15.0, threshold=DEFAULT_THRESHOLD,
             description='Pompe centrifuge principale — circuit eau froide'),
        dict(name='Compresseur A',        pump_type='centrifuge', fluid_type='huile', roue_material='acier_carbone',
             nominal_flow=30.0, nominal_pressure=8.0, nominal_current=22.0, nominal_vibration=3.0,
             power_kw=22.0, threshold=50.0,
             description='Compresseur hydraulique circuit huile'),
        dict(name='Pompe de Transfert',   pump_type='centrifuge', fluid_type='acide', roue_material='inox_316l',
             nominal_flow=20.0, nominal_pressure=4.0, nominal_current=12.0, nominal_vibration=1.8,
             power_kw=7.5, threshold=40.0,
             description='Pompe transfert acide chlorhydrique dilué'),
    ]
    machine_objs = []
    for md in machines_def:
        m = Machine(
            user_id=uid, is_active=True,
            name=md['name'], description=md.get('description',''),
            pump_type=md['pump_type'], fluid_type=md['fluid_type'], roue_material=md['roue_material'],
            nominal_flow=md['nominal_flow'], nominal_pressure=md['nominal_pressure'],
            nominal_current=md['nominal_current'], nominal_vibration=md['nominal_vibration'],
            power_kw=md['power_kw'], threshold=md['threshold'],
        )
        db.session.add(m)
        machine_objs.append(m)
    db.session.flush()

    # ── Analysis history (30 days, ~2 per day per machine) ────────────────────
    rng = random.Random(42)  # deterministic seed for reproducible demo data
    for day_offset in range(30, 0, -1):
        base_ts = now - timedelta(days=day_offset)
        for mi, m in enumerate(machine_objs):
            for reading in range(2):
                ts = base_ts + timedelta(hours=rng.randint(6, 20), minutes=rng.randint(0, 59))
                # Degradation trend: risk increases in last 5 days
                base_risk = 12 + mi * 5
                if day_offset <= 5:
                    base_risk += (5 - day_offset) * 8  # spike at end
                elif day_offset <= 10:
                    base_risk += (10 - day_offset) * 2
                risk = round(min(95.0, max(5.0, base_risk + rng.uniform(-10, 15))), 1)
                pred = 1 if risk >= float(m.threshold) else 0
                zones_str = ''
                if pred == 1:
                    zone_opts = ['Cavitation', 'Roulements', 'Étanchéité', 'Moteur']
                    zones_str = rng.choice(zone_opts)
                # vary sensor values realistically
                vib  = round(rng.uniform(1.5, 5.5 if risk > 50 else 3.0), 2)
                tpal = round(rng.uniform(45, 95 if risk > 50 else 65), 1)
                hf   = round(1000 + day_offset * 48 + reading * 12 + rng.uniform(0, 10), 0)
                a = Analysis(
                    machine_type='pump',
                    temp_air=tpal,
                    temp_process=round(rng.uniform(55, 85 if risk > 50 else 70), 1),
                    vitesse=round(m.nominal_flow + rng.uniform(-5, 5), 1),
                    couple=round(m.nominal_pressure + rng.uniform(-1, 1), 2),
                    usure=hf,
                    risk=risk, prediction=pred, zones=zones_str,
                    confidence=rng.randint(70, 100),
                    mail_sent=(pred == 1 and rng.random() > 0.5),
                    user_id=uid, machine_id=m.name,
                    extra_params=_json.dumps({'vibration': vib}),
                )
                a.timestamp = ts
                db.session.add(a)

    # ── Machine notes ──────────────────────────────────────────────────────────
    notes_data = [
        (machine_objs[0].id, 'Remplacement des joints torique effectué. Vibrations revenues à la normale.'),
        (machine_objs[0].id, 'Inspection mensuelle OK. Graissage roulements réalisé.'),
        (machine_objs[1].id, 'Niveau huile vérifié et complété. Filtre changé.'),
        (machine_objs[2].id, 'Attention corrosion détectée sur le flasque — prévoir remplacement sous 3 mois.'),
    ]
    for mid, content in notes_data:
        n = MachineNote(machine_id=mid, user_id=uid, user_email=DEMO_EMAIL, content=content)
        db.session.add(n)

    db.session.commit()
    logger.info(f"Demo data seeded: {len(machine_objs)} machines, 180 analyses")

try:
    with app.app_context():
        _seed_demo_account()
except Exception as _de:
    db.session.rollback()
    logger.info(f"Demo seed error: {_de}")

# ── ML MODEL LOADING (delegated to pilar_ml) ─────────────────────────────────
pilar_ml.init_models(_APP_DIR)
# Module-level aliases for backward compatibility within app.py
model = pilar_ml.model
scaler = pilar_ml.scaler
modeles_zones = pilar_ml.modeles_zones

# All ML constants (FAILURE_ZONES, COLONNES, FEATURE_MEDIANS, SENSOR_BOUNDS,
# FLUID_RUL_FACTORS, MATERIAL_RUL_FACTORS, FLUID_ZONE_SENSITIVITY,
# NON_CENTRIFUGE_TYPES, DOMAIN_KB, RETRAIN_TRIGGER, CLAUDE_MODEL, etc.)
# are imported from config.py at the top of this file.

# ── ML HELPERS (delegated to pilar_ml) ────────────────────────────────────────
_build_model_input = pilar_ml.build_model_input
_compute_shap = pilar_ml.compute_shap
_compute_rul = pilar_ml.compute_rul
_compute_anomaly_score = pilar_ml.compute_anomaly_score

# ── BACKGROUND MONITOR (delegated to pilar_monitor) ──────────────────────────
_bg_monitors = pilar_monitor.active_monitors

# ── EMAIL (delegated to pilar_email) ──────────────────────────────────────────
GMAIL = pilar_email.GMAIL
GMAIL_PWD = pilar_email.GMAIL_PWD
_send_email = pilar_email.send_email

# ── In-app update banner ───────────────────────────────────────────────────────
_UPDATE_INFO = {'available': False, 'version': None, 'download_url': None, 'current': os.environ.get('PILAR_VERSION', APP_VERSION)}

def _check_update_background():
    """Called by launcher or on first boot — checks Railway for newer version."""
    try:
        import urllib.request as _ur, json as _j
        APP_VER = os.environ.get('PILAR_VERSION', APP_VERSION)
        update_url = os.environ.get("PILAR_UPDATE_URL", "")
        if not update_url:
            return
        with _ur.urlopen(update_url, timeout=6) as r:
            d = _j.loads(r.read().decode())
        latest = (d.get('version') or '').lstrip('v')
        current = APP_VER.lstrip('v')
        if latest and latest != current:
            _UPDATE_INFO['available'] = True
            _UPDATE_INFO['version'] = latest
            _UPDATE_INFO['download_url'] = d.get('download_url', '')
            logger.info(f"Update available: {current} → {latest}")
    except Exception as e:
        logger.info(f"Update check: {e}")

# Fire once at startup in background
threading.Thread(target=_check_update_background, daemon=True).start()
FAVICON = "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"

def current_uid():
    try:
        return session.get('user_id')
    except RuntimeError:
        return None

def get_setting(key, default="", uid=None):
    try:
        uid = uid or current_uid()
        s = Settings.query.filter_by(key=key, user_id=uid).first()
        return s.value if s else default
    except: return default

def set_setting(key, value, uid=None):
    try:
        uid = uid or current_uid()
        s = Settings.query.filter_by(key=key, user_id=uid).first()
        if s: s.value = value
        else: db.session.add(Settings(key=key, value=value, user_id=uid))
        db.session.commit()
    except Exception as e: logger.error(f"Settings error: {e}")

# ── AUTH HELPERS ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_uid():
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

def auth_optional(f):
    """Allow access without login — guest mode."""
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated

def api_or_login_required(f):
    """Accepte session Flask OU header X-Api-Key pour les endpoints API."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-Api-Key')
        if api_key:
            user = User.query.filter_by(api_key=api_key).first()
            if user:
                session['user_id'] = user.id
                return f(*args, **kwargs)
            return jsonify({'error': 'Invalid API key'}), 401
        if not current_uid():
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = current_uid()
        if not uid:
            return redirect('/login')
        user = db.session.get(User, uid)
        if not user or not user.is_admin:
            return "Accès refusé", 403
        return f(*args, **kwargs)
    return decorated

send_verify_email = pilar_email.send_verify_email
send_reset_email = pilar_email.send_reset_email


# ── AUTH PAGES ────────────────────────────────────────────────────────────────




# ── CSS & HEAD ────────────────────────────────────────────────────────────────

_SIDEBAR = """
<!-- Desktop Sidebar (shown on wide screens) -->
<aside class="sidebar" id="desktopSidebar">
  <div class="sidebar-logo">
    <div class="logo-mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
    </div>
    <div>
      <div class="logo-text">PILAR</div>
      <div class="logo-sub" data-i18n="app_subtitle">Maintenance Prédictive</div>
    </div>
  </div>
  <nav class="sidebar-nav">
    <div class="sidebar-section" data-i18n="nav_section_fleet">Machines</div>
    <a href="/machines" class="ni {fl}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span class="ni-label" data-i18n="nav_machines">Machines</span>
    </a>
    <div class="sidebar-section" data-i18n="nav_section_user">Utilisateur</div>
    <a href="/account" class="ni {a}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg>
      <span class="ni-label" data-i18n="nav_account">Compte</span>
    </a>
    <a href="/settings" class="ni {s}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
      <span class="ni-label" data-i18n="nav_settings">Réglages</span>
    </a>
  </nav>
  <div class="sidebar-footer">
    <div class="sync-status-bar" id="sidebarSyncBar" onclick="updateSyncStatus()" title="Click to refresh sync status" style="cursor:pointer">
      <span class="sync-dot offline" id="sidebarSyncDot"></span>
      <span id="sidebarSyncLabel" data-i18n="sync_offline">Hors ligne</span>
    </div>
    <button class="ni lang-toggle" id="_langBtn" onclick="_toggleLang()" title="Switch language" style="padding:6px 10px;font-size:10px;"><span id="_langLbl">EN</span></button>
  </div>
</aside>
<!-- Fleet drawer -->
<div id="fleetDrawer" style="display:none;position:fixed;top:0;right:0;bottom:0;width:420px;max-width:100vw;background:var(--surface);border-left:1px solid var(--border);z-index:200;overflow-y:auto;padding:20px;box-shadow:-4px 0 24px rgba(0,0,0,0.4)">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
    <div>
      <div style="font-size:11px;letter-spacing:3px;color:var(--teal);text-transform:uppercase;font-weight:700">Fleet</div>
      <div style="font-size:18px;font-weight:700;color:var(--text);margin-top:2px">Your Machines</div>
    </div>
    <button onclick="closeFleet()" style="background:none;border:none;color:var(--text3);cursor:pointer;padding:8px;font-size:18px;line-height:1">✕</button>
  </div>
  <div id="fleetContent"><div style="text-align:center;padding:40px;color:var(--text3)">Loading...</div></div>
  <a href="/dashboard" style="display:block;margin-top:16px;padding:12px;border:1px solid var(--border);border-radius:8px;color:var(--text3);text-decoration:none;text-align:center;font-size:11px;letter-spacing:1px">Full fleet page →</a>
</div>
<div id="fleetOverlay" onclick="closeFleet()" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:199"></div>
<script>
(function(){{
  var lbl=document.getElementById('_langLbl');
  if(lbl)lbl.textContent=(localStorage.getItem('pilar_lang')||'en').toUpperCase();
}})();
function updateSyncStatus(){{
  fetch('/api/sync/status').then(function(r){{return r.json();}}).then(function(d){{
    var dot=document.getElementById('sidebarSyncDot');
    var lbl=document.getElementById('sidebarSyncLabel');
    if(dot&&lbl){{
      if(d.online){{
        dot.className='sync-dot online';
        lbl.textContent=d.sync_url?(d.last_sync?'Synced · '+new Date(d.last_sync).toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}}):'Synced'):'Online';
      }}else{{
        dot.className='sync-dot offline';
        lbl.textContent=d.queued>0?'Offline · '+d.queued+' queued':'Offline';
      }}
    }}
  }}).catch(function(){{}});
}}
updateSyncStatus();
setInterval(updateSyncStatus,30000);
function _toggleLang(){{
  var next=LANG==='en'?'fr':'en';
  setLang(next);
  var lbl=document.getElementById('_langLbl');
  if(lbl)lbl.textContent=next.toUpperCase();
}}
function openFleet(){{
  document.getElementById('fleetDrawer').style.display='block';
  document.getElementById('fleetOverlay').style.display='block';
  fetch('/api/fleet_summary').then(r=>r.json()).then(d=>{{
    var html='';
    if(!d.machines||d.machines.length===0){{
      html='<div style="text-align:center;padding:40px;color:var(--text3)"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:12px;display:block;margin-left:auto;margin-right:auto"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2"/></svg><div>No machines yet</div><a href="/dashboard" style="display:inline-block;margin-top:12px;padding:8px 16px;background:var(--teal);color:#fff;border-radius:6px;text-decoration:none;font-size:11px;font-weight:700">Add Machine</a></div>';
    }}else{{
      d.machines.forEach(function(m){{
        var risk=m.last_risk!=null?m.last_risk:'—';
        var cls=m.last_risk==null?'':'risk>=50'?'alert':m.last_risk>=22?'amber':'ok';
        var rc=m.last_risk==null?'var(--text3)':m.last_risk>=50?'var(--red)':m.last_risk>=22?'var(--amber)':'var(--green)';
        html+='<a href="/machine/'+m.id+'" style="display:block;padding:14px;background:var(--bg);border:1px solid var(--border);border-radius:10px;text-decoration:none;margin-bottom:10px;transition:border-color .15s">';
        html+='<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">';
        html+='<div style="font-size:13px;font-weight:600;color:var(--text)">'+m.name+'</div>';
        html+='<div style="font-size:20px;font-weight:800;color:'+rc+'">'+risk+(risk!=='—'?'%':'')+'</div></div>';
        html+='<div style="font-size:11px;color:var(--text3)">'+( m.location||'' )+(m.location&&m.last_analysis?' · ':'')+( m.last_analysis?'Last: '+m.last_analysis:'' )+'</div>';
        html+='</a>';
      }});
    }}
    document.getElementById('fleetContent').innerHTML=html;
  }}).catch(function(){{
    document.getElementById('fleetContent').innerHTML='<div style="color:var(--red);text-align:center;padding:20px">Could not load fleet data</div>';
  }});
}}
function closeFleet(){{
  document.getElementById('fleetDrawer').style.display='none';
  document.getElementById('fleetOverlay').style.display='none';
}}
</script>"""

_BOTTOM_NAV = """<!-- Bottom nav (mobile fallback) -->
<nav class="bottom-nav">
<a href="/machines" class="ni {fl}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg><span data-i18n="nav_machines">Machines</span></a>
<a href="/account" class="ni {a}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg><span data-i18n="nav_account">Account</span></a>
<a href="/settings" class="ni {s}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg><span data-i18n="nav_settings">Settings</span></a>
</nav>"""

def nav(active):
    keys = {"m":"","tut":"","h":"","fl":"","a":"","s":"","tw":""}
    keys[active] = "on"
    sidebar = _SIDEBAR.format(**keys)
    # Inject admin link for admin users only
    uid = current_uid()
    if uid:
        _u = db.session.get(User, uid)
        if _u and _u.is_admin:
            _admin_link = (
                '<div class="sidebar-section">Admin</div>'
                '<a href="/admin" class="ni" style="color:var(--amber,#d97706)">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor">'
                '<path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>'
                '</svg>'
                '<span class="ni-label">Admin Panel</span>'
                '</a>'
            )
            sidebar = sidebar.replace('</nav>', _admin_link + '</nav>', 1)
    return sidebar + _BOTTOM_NAV.format(**keys)


# ── MONITOR ───────────────────────────────────────────────────────────────────


# ── ACCOUNT ───────────────────────────────────────────────────────────────────
FAV_B64 = "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"


# ── TWIN ──────────────────────────────────────────────────────────────────────

# ── HISTORY ───────────────────────────────────────────────────────────────────

# ── SETTINGS ──────────────────────────────────────────────────────────────────


# ── ASSISTANT ─────────────────────────────────────────────────────────────────

# ── TUTORIAL ───────────────────────────────────────────────────────────────────

# ── CSV ADAPTER ───────────────────────────────────────────────────────────────

# ── DEMO PAGE ─────────────────────────────────────────────────────────────────

# ── MACHINE SPACE ─────────────────────────────────────────────────────────────

# ── DASHBOARD MULTI-MACHINES ──────────────────────────────────────────────────

# ── API DOCS ──────────────────────────────────────────────────────────────────

# ── BACKEND (delegated to pilar_ml / pilar_email) ────────────────────────────
predict_risk = pilar_ml.predict_risk
envoyer_alerte = pilar_email.send_alert_email
envoyer_escalade = pilar_email.send_escalation_email

def _escalation_worker():
    """Background thread: every 5 min, escalate unacked alerts older than 30 min."""
    import time as _time
    while True:
        _time.sleep(300)
        try:
            with app.app_context():
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
                pending = AlertLog.query.filter(
                    AlertLog.acked_at == None,
                    AlertLog.escalated_at == None,
                    AlertLog.escalation_email != None,
                    AlertLog.escalation_email != '',
                    AlertLog.sent_at <= cutoff
                ).all()
                for al in pending:
                    al.escalated_at = datetime.now(timezone.utc)
                    db.session.commit()
                    threading.Thread(target=envoyer_escalade, args=(
                        al.escalation_email, al.probabilite, [], al.machine_id_str), daemon=True).start()
                    logger.info(f"escalation] AlertLog {al.id} escalated to {al.escalation_email}")
        except Exception as _esc_e:
            logger.info(f"escalation] worker error: {_esc_e}")

threading.Thread(target=_escalation_worker, daemon=True).start()

# ── WEEKLY PDF REPORTS ────────────────────────────────────────────────────────
def _generate_fleet_pdf(uid):
    """Generate a one-page fleet summary PDF for the given user and return bytes."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    machines = Machine.query.filter_by(user_id=uid, is_active=True).all()
    analyses = Analysis.query.filter_by(user_id=uid).filter(
        Analysis.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).all()
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, 'PILAR — Weekly Fleet Report', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Period: {(datetime.now(timezone.utc)-timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(4)
    # Summary line
    total_analyses = len(analyses)
    total_alerts = sum(1 for a in analyses if a.mail_sent)
    avg_risk = round(sum(a.risk for a in analyses) / total_analyses, 1) if total_analyses else 0
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, f"Analyses this week: {total_analyses}   |   Alerts triggered: {total_alerts}   |   Avg risk: {avg_risk}%", ln=True)
    pdf.ln(4)
    # Per-machine table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(7, 17, 31)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(70, 8, 'Machine', border=1, fill=True)
    pdf.cell(30, 8, 'Analyses', border=1, fill=True, align='C')
    pdf.cell(30, 8, 'Avg Risk', border=1, fill=True, align='C')
    pdf.cell(30, 8, 'Alerts', border=1, fill=True, align='C')
    pdf.cell(30, 8, 'Status', border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(30, 30, 30)
    machine_names = {m.name for m in machines}
    all_ids = {a.machine_id for a in analyses if a.machine_id} | machine_names
    for mid in (all_ids or ['(unassigned)']):
        m_analyses = [a for a in analyses if (a.machine_id or '(unassigned)') == mid]
        m_alerts = sum(1 for a in m_analyses if a.mail_sent)
        m_avg = round(sum(a.risk for a in m_analyses) / len(m_analyses), 1) if m_analyses else 0
        m_status = 'OK' if m_avg < 45 else 'WARNING' if m_avg < 70 else 'CRITICAL'
        pdf.cell(70, 7, str(mid)[:30], border=1)
        pdf.cell(30, 7, str(len(m_analyses)), border=1, align='C')
        pdf.cell(30, 7, f"{m_avg}%", border=1, align='C')
        pdf.cell(30, 7, str(m_alerts), border=1, align='C')
        pdf.cell(30, 7, m_status, border=1, align='C')
        pdf.ln()
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f"Generated by Pilar at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def _send_weekly_reports():
    """Send PDF reports to all active users who have responsible_email set."""
    with app.app_context():
        users = User.query.all()
        for u in users:
            email = get_setting('responsible_email', uid=u.id)
            if not email:
                continue
            try:
                pdf_bytes = _generate_fleet_pdf(u.id)
                if not pdf_bytes:
                    continue
                from email.mime.base import MIMEBase
                from email import encoders as _enc
                msg = MIMEMultipart()
                msg['Subject'] = f"Pilar — Weekly Fleet Report {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                msg['From'] = f"Pilar <{GMAIL}>"
                msg['To'] = email
                body = MIMEText("Please find this week's fleet report attached.", 'plain')
                msg.attach(body)
                part = MIMEBase('application', 'pdf')
                part.set_payload(pdf_bytes)
                _enc.encode_base64(part)
                _fname = 'pilar_report_' + datetime.now(timezone.utc).strftime('%Y%m%d') + '.pdf'
                part.add_header('Content-Disposition', f'attachment; filename="{_fname}"')
                msg.attach(part)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(GMAIL, GMAIL_PWD)
                    smtp.sendmail(GMAIL, email, msg.as_string())
                logger.info(f"pdf] Weekly report sent to {email}")
            except Exception as _pdf_e:
                logger.info(f"pdf] Error for user {u.id}: {_pdf_e}")

# ── ADAPTIVE PER-MACHINE ML ──────────────────────────────────────────────────
# Analysis table stores 5 sensor fields as legacy column names; the other 3
# sensors (vibration, pression_entree, courant_moteur) live in extra_params JSON.
_MACHINE_FEATURE_MAP = {
    'temp_palier':     ('col',   'temp_air'),
    'temp_moteur':     ('col',   'temp_process'),
    'debit':           ('col',   'vitesse'),
    'pression_sortie': ('col',   'couple'),
    'vibration':       ('extra', 'vibration'),
    'pression_entree': ('extra', 'pression_entree'),
    'courant_moteur':  ('extra', 'courant_moteur'),
}
_ENV_RISK_FACTOR  = {'chemical': 1.3, 'mining': 1.5, 'food': 1.1,
                     'water': 1.05, 'automotive': 1.05, 'general': 1.0}
_CRIT_RISK_FACTOR = {'critical': 1.5, 'high': 1.3, 'medium': 1.1, 'low': 1.0}
_MIN_BASELINE_SAMPLES    = 30
_MIN_SPECIALIZE_SAMPLES  = 50
_SPECIALIZE_F1_GATE      = 0.70
_BASELINE_STALE_DAYS     = 7
_PER_MACHINE_INTERVAL_S  = 6 * 3600  # 6h

def _resolve_machine(machine_ref):
    """Accept int pk, numeric string, Machine object, or name. Return Machine or None."""
    if machine_ref is None:
        return None
    if isinstance(machine_ref, Machine):
        return machine_ref
    try:
        mid = int(machine_ref)
        m = Machine.query.get(mid)
        if m: return m
    except (TypeError, ValueError):
        pass
    try:
        return Machine.query.filter_by(name=str(machine_ref)).first()
    except Exception:
        return None

def _machine_analysis_query(machine, only_normal=False):
    q = Analysis.query.filter(Analysis.machine_id == machine.name)
    if only_normal:
        q = q.filter((Analysis.prediction == 0) | (Analysis.prediction.is_(None)))
    return q

def _extract_feature_value(a, kind, key):
    if kind == 'col':
        v = getattr(a, key, None)
        return float(v) if v is not None else None
    if kind == 'extra':
        try:
            import json as _j
            d = _j.loads(a.extra_params) if a.extra_params else {}
            v = d.get(key)
            return float(v) if v is not None else None
        except Exception:
            return None
    return None

def compute_machine_baseline(machine_ref):
    """Compute per-feature normal-operation baselines. Returns dict or None if <30 samples."""
    try:
        m = _resolve_machine(machine_ref)
        if not m: return None
        rows = _machine_analysis_query(m, only_normal=True)\
                 .order_by(Analysis.timestamp.desc()).limit(500).all()
        if len(rows) < _MIN_BASELINE_SAMPLES:
            logger.debug(f"baseline[{m.id}]: only {len(rows)} samples — need {_MIN_BASELINE_SAMPLES}")
            return None
        import statistics as _st
        result = {}
        now = datetime.now(timezone.utc)
        MachineBaseline.query.filter_by(machine_id=m.id).delete()
        for feat, (kind, key) in _MACHINE_FEATURE_MAP.items():
            vals = [_extract_feature_value(a, kind, key) for a in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) < _MIN_BASELINE_SAMPLES:
                continue
            mean = _st.fmean(vals)
            std  = _st.pstdev(vals) if len(vals) > 1 else 0.0
            entry = {
                'mean': round(mean, 4),
                'std':  round(std, 4),
                'min_normal': round(mean - 2 * std, 4),
                'max_normal': round(mean + 2 * std, 4),
                'sample_count': len(vals),
            }
            result[feat] = entry
            db.session.add(MachineBaseline(
                machine_id=m.id, feature=feat,
                mean=entry['mean'], std=entry['std'],
                min_normal=entry['min_normal'], max_normal=entry['max_normal'],
                sample_count=entry['sample_count'], computed_at=now))
        db.session.commit()
        logger.info(f"baseline[{m.id}/{m.name}]: computed {len(result)} features from {len(rows)} rows")
        return result
    except Exception as _e:
        db.session.rollback()
        logger.error(f"compute_machine_baseline error: {_e}")
        return None

def compute_anomaly_scores(machine_ref, current_values):
    """Compare current sensor values to machine baseline. Returns per-feature dict."""
    try:
        m = _resolve_machine(machine_ref)
        if not m: return {}
        baselines = {b.feature: b for b in MachineBaseline.query.filter_by(machine_id=m.id).all()}
        if not baselines:
            return {}
        out = {}
        for feat, val in (current_values or {}).items():
            b = baselines.get(feat)
            if b is None or val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            std = b.std if (b.std and b.std > 1e-6) else 1e-6
            z = (v - b.mean) / std
            dev_pct = ((v - b.mean) / b.mean * 100.0) if b.mean else 0.0
            abs_z = abs(z)
            if   abs_z > 4.0: status = 'critical'
            elif abs_z > 2.5: status = 'high'
            elif abs_z > 1.5: status = 'elevated'
            else:             status = 'normal'
            out[feat] = {
                'current':       round(v, 3),
                'baseline_mean': round(b.mean, 3) if b.mean is not None else None,
                'z_score':       round(z, 2),
                'deviation_pct': round(dev_pct, 1),
                'is_anomaly':    abs_z > 2.5,
                'direction':     'above' if v > b.mean else ('below' if v < b.mean else 'equal'),
                'status':        status,
            }
        return out
    except Exception as _e:
        logger.error(f"compute_anomaly_scores error: {_e}")
        return {}

_specialize_lock = threading.Lock()

def _params_from_analysis(a):
    """Rebuild a feature-name dict from an Analysis row (cols + extra_params)."""
    params = {
        'temp_palier':          a.temp_air,
        'temp_moteur':          a.temp_process,
        'debit':                a.vitesse,
        'pression_sortie':      a.couple,
        'heure_fonctionnement': a.usure,
    }
    try:
        import json as _j
        extra = _j.loads(a.extra_params) if a.extra_params else {}
    except Exception:
        extra = {}
    for k in ('vibration', 'pression_entree', 'courant_moteur'):
        params[k] = extra.get(k, None)
    # Fill gaps with FEATURE_MEDIANS so build_model_input succeeds
    try:
        from config import FEATURE_MEDIANS as _FM
        for k, v in _FM.items():
            if params.get(k) is None:
                params[k] = v
    except Exception:
        pass
    return params

def get_machine_model(machine_ref):
    """Return (model, scaler, is_specialized). Falls back to global model on any error."""
    try:
        import pilar_ml as _pm
        m = _resolve_machine(machine_ref)
        if m:
            mm = MachineModel.query.filter_by(machine_id=m.id).first()
            if mm and mm.model_blob:
                try:
                    import pickle as _pk
                    mdl = _pk.loads(mm.model_blob)
                    scl = _pk.loads(mm.scaler_blob) if mm.scaler_blob else _pm.scaler
                    return mdl, scl, True
                except Exception as _e:
                    logger.warning(f"get_machine_model[{m.id}]: unpickle failed — {_e}")
        return _pm.model, _pm.scaler, False
    except Exception as _e:
        logger.error(f"get_machine_model fatal: {_e}")
        try:
            import pilar_ml as _pm
            return _pm.model, _pm.scaler, False
        except Exception:
            return None, None, False

def specialize_machine_model(machine_ref):
    """Train a machine-specific classifier if data is sufficient and F1 > gate.
    Returns dict {status, f1_score, samples_used, reason?}."""
    if not _specialize_lock.acquire(blocking=False):
        return {'status': 'busy', 'reason': 'another specialization is running'}
    try:
        m = _resolve_machine(machine_ref)
        if not m:
            return {'status': 'error', 'reason': 'machine not found'}
        rows = _machine_analysis_query(m).order_by(Analysis.timestamp.asc()).all()
        if len(rows) < _MIN_SPECIALIZE_SAMPLES:
            return {'status': 'insufficient_data', 'samples_used': len(rows),
                    'reason': f'need ≥{_MIN_SPECIALIZE_SAMPLES} analyses, have {len(rows)}'}
        # Build feature matrix + labels using the global feature pipeline
        import pilar_ml as _pm
        import numpy as _np, pickle as _pk
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import StandardScaler
        X_raw, y = [], []
        for a in rows:
            if a.prediction is None:
                continue
            try:
                df = _pm.build_model_input(_params_from_analysis(a))
                X_raw.append(df.iloc[0].values)
                y.append(int(a.prediction))
            except Exception:
                continue
        if len(X_raw) < _MIN_SPECIALIZE_SAMPLES:
            return {'status': 'insufficient_data', 'samples_used': len(X_raw),
                    'reason': 'too many rows dropped during feature build'}
        X = _np.array(X_raw, dtype=float)
        y = _np.array(y, dtype=int)
        if len(set(y)) < 2:
            return {'status': 'single_class', 'samples_used': len(y),
                    'reason': 'all labels identical — cannot train discriminator'}
        # Fresh StandardScaler fit on this machine's data
        scl = StandardScaler().fit(X)
        Xs = scl.transform(X)
        # Optional SMOTE if imbalanced
        pos = int(y.sum()); neg = len(y) - pos
        minority = min(pos, neg)
        if minority >= 6 and (minority / len(y)) < 0.35:
            try:
                from imblearn.over_sampling import SMOTE
                k = min(5, minority - 1)
                Xs, y = SMOTE(k_neighbors=max(1, k), random_state=42).fit_resample(Xs, y)
            except Exception as _se:
                logger.debug(f"specialize[{m.id}]: SMOTE skipped — {_se}")
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                         random_state=42)
        clf.fit(X_tr, y_tr)
        f1 = float(f1_score(y_te, clf.predict(X_te), zero_division=0))
        samples_used = int(len(rows))
        if f1 < _SPECIALIZE_F1_GATE:
            logger.info(f"specialize[{m.id}/{m.name}]: REJECTED f1={f1:.3f} < {_SPECIALIZE_F1_GATE}")
            return {'status': 'rejected_low_f1', 'f1_score': round(f1, 4),
                    'samples_used': samples_used,
                    'reason': f'F1 {f1:.2f} below gate {_SPECIALIZE_F1_GATE}'}
        # Persist
        existing = MachineModel.query.filter_by(machine_id=m.id).first()
        version = (existing.version + 1) if existing else 1
        blob_m = _pk.dumps(clf); blob_s = _pk.dumps(scl)
        if existing:
            existing.model_blob = blob_m; existing.scaler_blob = blob_s
            existing.f1_score = f1; existing.training_samples = samples_used
            existing.version = version
            existing.last_trained = datetime.now(timezone.utc)
        else:
            db.session.add(MachineModel(
                machine_id=m.id, model_blob=blob_m, scaler_blob=blob_s,
                f1_score=f1, training_samples=samples_used, version=version,
                last_trained=datetime.now(timezone.utc)))
        db.session.commit()
        try:
            s = Settings.query.filter_by(key=f'specialize_last_{m.id}', user_id=None).first()
            ts = datetime.now(timezone.utc).isoformat() + f' f1={f1:.3f} n={samples_used}'
            if s: s.value = ts
            else: db.session.add(Settings(key=f'specialize_last_{m.id}', value=ts, user_id=None))
            db.session.commit()
        except Exception:
            db.session.rollback()
        logger.info(f"specialize[{m.id}/{m.name}]: SAVED v{version} f1={f1:.3f} n={samples_used}")
        return {'status': 'ok', 'f1_score': round(f1, 4), 'samples_used': samples_used,
                'version': version}
    except Exception as _e:
        db.session.rollback()
        import traceback
        logger.error(f"specialize_machine_model error: {_e}\n{traceback.format_exc()}")
        return {'status': 'error', 'reason': str(_e)}
    finally:
        _specialize_lock.release()

# ── NL SENTENCE UNDERSTANDING ────────────────────────────────────────────────
# The operator can paste free-text notes ("grinding noise from the bearing,
# vibration spiked since Monday"). We extract risk signals via local Ollama
# (pilar-diag model), falling back to keyword heuristics when Ollama is down.
_SENTENCE_TIMEOUT_S = 25.0
_OLLAMA_URL_SENT    = "http://localhost:11434"

_SENT_KEYWORDS_RAISE = [
    ('fuite',          1.15, 'leak detected'),
    ('leak',           1.15, 'leak detected'),
    ('grind',          1.25, 'grinding noise'),
    ('grinçant',       1.25, 'grinding noise'),
    ('bruit',          1.10, 'unusual noise'),
    ('noise',          1.10, 'unusual noise'),
    ('vibration',      1.15, 'vibration reported'),
    ('fumée',          1.35, 'smoke observed'),
    ('smoke',          1.35, 'smoke observed'),
    ('brûl',           1.30, 'burning smell'),
    ('burn',           1.30, 'burning smell'),
    ('odeur',          1.15, 'abnormal smell'),
    ('smell',          1.15, 'abnormal smell'),
    ('surchauffe',     1.25, 'overheating'),
    ('overheat',       1.25, 'overheating'),
    ('chaud',          1.10, 'runs hot'),
    ('hot',            1.10, 'runs hot'),
    ('fuit',           1.15, 'leaking'),
    ('arrêt',          1.20, 'stoppage reported'),
    ('stop',           1.15, 'stop/outage reported'),
    ('broken',         1.35, 'broken component'),
    ('cassé',          1.35, 'broken component'),
    ('panne',          1.30, 'breakdown reported'),
    ('alarm',          1.20, 'alarm triggered'),
    ('alarme',         1.20, 'alarm triggered'),
]
_SENT_KEYWORDS_LOWER = [
    ('réparé',         0.85, 'recently repaired'),
    ('repaired',       0.85, 'recently repaired'),
    ('remplacé',       0.80, 'part replaced'),
    ('replaced',       0.80, 'part replaced'),
    ('neuf',           0.85, 'new installation'),
    ('new ',           0.90, 'new installation'),
    ('maintenance',    0.92, 'recent maintenance'),
    ('nettoyé',        0.92, 'recently cleaned'),
    ('cleaned',        0.92, 'recently cleaned'),
    ('normal',         0.95, 'operator reports normal'),
    ('ok ',            0.95, 'operator reports ok'),
]

def _sentence_keywords(text):
    """Heuristic fallback: scan for known phrases and combine multipliers."""
    t = (text or '').lower()
    if not t.strip():
        return {'risk_multiplier': 1.0, 'severity_hints': [], 'extracted_symptoms': [], 'source': 'none'}
    mult = 1.0
    symptoms, hints = [], []
    for kw, m, label in _SENT_KEYWORDS_RAISE:
        if kw in t:
            mult *= m
            if label not in symptoms: symptoms.append(label)
            hints.append(f'+{int((m-1)*100)}% ({label})')
    for kw, m, label in _SENT_KEYWORDS_LOWER:
        if kw in t:
            mult *= m
            if label not in symptoms: symptoms.append(label)
            hints.append(f'-{int((1-m)*100)}% ({label})')
    mult = max(0.7, min(mult, 1.6))
    return {'risk_multiplier': round(mult, 3),
            'severity_hints': hints, 'extracted_symptoms': symptoms,
            'source': 'keywords'}

def _sentence_ollama(text):
    """Ask local Ollama (pilar-diag) to emit a JSON verdict. Returns None on any failure."""
    try:
        import requests as _rq
        import json as _j
        # Ping first so we don't burn the full timeout when Ollama is down
        try:
            _rq.get(f"{_OLLAMA_URL_SENT}/api/tags", timeout=2.0)
        except Exception:
            return None
        prompt = (
            "You are a predictive-maintenance assistant. Read the operator note below "
            "and respond with ONLY a compact JSON object (no prose, no markdown) with keys:\n"
            '  risk_multiplier : float in [0.7, 1.6] (1.0 = neutral; >1 raises risk)\n'
            '  severity_hints  : array of short strings\n'
            '  extracted_symptoms : array of short strings\n\n'
            f"Operator note:\n\"\"\"\n{text.strip()}\n\"\"\"\n\nJSON:"
        )
        r = _rq.post(
            f"{_OLLAMA_URL_SENT}/api/generate",
            json={"model": "pilar-diag", "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0.1}},
            timeout=_SENTENCE_TIMEOUT_S,
        )
        r.raise_for_status()
        body = r.json().get("response", "").strip()
        data = _j.loads(body)
        mult = float(data.get("risk_multiplier", 1.0))
        mult = max(0.7, min(mult, 1.6))
        return {
            'risk_multiplier':    round(mult, 3),
            'severity_hints':     list(data.get("severity_hints", []) or [])[:8],
            'extracted_symptoms': list(data.get("extracted_symptoms", []) or [])[:8],
            'source': 'ollama',
        }
    except Exception as _e:
        logger.debug(f"sentence_ollama failed: {_e}")
        return None

def extract_sentence_signals(text):
    """Public helper: parse a free-text operator note into risk signals.
    Always returns a dict with risk_multiplier + severity_hints + extracted_symptoms."""
    if not text or not str(text).strip():
        return {'risk_multiplier': 1.0, 'severity_hints': [],
                'extracted_symptoms': [], 'source': 'none'}
    llm = _sentence_ollama(text)
    if llm is not None:
        return llm
    return _sentence_keywords(text)

# ── CONTEXTUAL PREDICTION ────────────────────────────────────────────────────
def predict_with_context(machine_ref, current_values, sentence=None, threshold=None):
    """Run a machine-aware prediction enriched with:
      - specialized classifier probability (if trained), else global risk
      - per-machine anomaly scores vs baseline
      - age / environment / criticality risk multipliers
      - free-text operator-note risk multiplier
    Returns an enriched result dict compatible with /predire.
    `current_values` uses French feature names (debit, temp_palier, ...)."""
    import pilar_ml as _pm
    result = {
        'prediction': 0, 'probabilite': 0.0, 'zones': [],
        'confidence': 100, 'imputed': [], 'anomaly_score': None,
        'shap_explanations': [], 'rul_hours': None,
        'model_type': 'global', 'model_version': None, 'model_f1': None,
        'anomaly_scores': {}, 'context_factors': {}, 'sentence_signals': None,
        'adjusted_probabilite': 0.0, 'base_probabilite': 0.0,
    }
    m = _resolve_machine(machine_ref)
    # 1. Base global prediction (zones, RUL, SHAP, etc.) — always run for consistency
    try:
        _thr = float(threshold) if threshold else DEFAULT_THRESHOLD
        _mctx = None
        if m:
            _mctx = {
                'nominal_flow':      m.nominal_flow,
                'nominal_pressure':  m.nominal_pressure,
                'nominal_current':   m.nominal_current,
                'nominal_vibration': m.nominal_vibration,
                'power_kw':          m.power_kw,
            }
        proba, pred, zones, conf, imputed, anom, shap_ex, rul = _pm.predict_risk(
            current_values, threshold=_thr, return_extra=True, machine_context=_mctx)
        result.update({'prediction': pred, 'probabilite': proba, 'zones': zones,
                       'confidence': conf, 'imputed': imputed, 'anomaly_score': anom,
                       'shap_explanations': shap_ex, 'rul_hours': rul,
                       'base_probabilite': proba})
    except Exception as _e:
        logger.error(f"predict_with_context: global predict failed — {_e}")
        return result

    # 2. If a specialized classifier exists, blend its probability into the base
    if m:
        try:
            mdl, scl, is_spec = get_machine_model(m)
            if is_spec and mdl is not None and scl is not None:
                df = _pm.build_model_input(_params_from_analysis_like(current_values))
                import numpy as _np
                Xs = scl.transform(df.values)
                try:
                    spec_p = float(mdl.predict_proba(Xs)[0][1]) * 100.0
                except Exception:
                    spec_p = float(mdl.predict(Xs)[0]) * 100.0
                mm = MachineModel.query.filter_by(machine_id=m.id).first()
                w_spec = 0.7 if (mm and (mm.f1_score or 0) >= 0.85) else 0.55
                blended = (spec_p * w_spec) + (result['probabilite'] * (1.0 - w_spec))
                result['probabilite']  = round(blended, 1)
                result['model_type']   = 'specialized'
                result['model_version']= mm.version if mm else None
                result['model_f1']     = round(mm.f1_score, 3) if (mm and mm.f1_score) else None
                # Recompute discrete prediction using the default threshold
                result['prediction']   = 1 if blended >= _thr else 0
        except Exception as _e:
            logger.debug(f"predict_with_context: specialized blend skipped — {_e}")

    # 3. Per-machine anomaly scores (z-scores vs this machine's baseline)
    if m:
        try:
            result['anomaly_scores'] = compute_anomaly_scores(m, current_values) or {}
        except Exception as _e:
            logger.debug(f"predict_with_context: anomaly_scores skipped — {_e}")

    # 4. Context multipliers
    age_f = env_f = crit_f = 1.0
    if m:
        try:
            age_f  = min(max(float(m.age_years or 0.0) / 10.0, 0.0), 2.0) or 1.0
            age_f  = max(1.0, age_f)  # age only raises risk, never lowers
        except Exception:
            age_f = 1.0
        env_f  = _ENV_RISK_FACTOR.get((m.environment or 'general').lower(), 1.0)
        crit_f = _CRIT_RISK_FACTOR.get((m.criticality or 'medium').lower(), 1.0)

    # 5. Free-text sentence signals
    sent = extract_sentence_signals(sentence) if sentence else {
        'risk_multiplier': 1.0, 'severity_hints': [],
        'extracted_symptoms': [], 'source': 'none'}
    sent_mult = float(sent.get('risk_multiplier', 1.0) or 1.0)

    total_mult = age_f * env_f * crit_f * sent_mult
    adjusted = min(100.0, result['probabilite'] * total_mult)
    result['adjusted_probabilite'] = round(adjusted, 1)
    result['context_factors'] = {
        'age_factor':      round(age_f, 3),
        'environment_factor': round(env_f, 3),
        'criticality_factor': round(crit_f, 3),
        'sentence_factor': round(sent_mult, 3),
        'total_multiplier': round(total_mult, 3),
    }
    result['sentence_signals'] = sent
    # When machine context is present, the adjusted value becomes the displayed risk
    if m:
        result['probabilite'] = result['adjusted_probabilite']
        _thr = float(threshold) if threshold else DEFAULT_THRESHOLD
        result['prediction'] = 1 if result['probabilite'] >= _thr else 0
    return result

def _params_from_analysis_like(current_values):
    """current_values uses French sensor names; pad with FEATURE_MEDIANS for build_model_input."""
    params = {}
    try:
        from config import FEATURE_MEDIANS as _FM
        for k, v in _FM.items():
            params[k] = v
    except Exception:
        pass
    for k, v in (current_values or {}).items():
        if v is None: continue
        try: params[k] = float(v)
        except (TypeError, ValueError): pass
    return params

# ── AUTO-RETRAIN ──────────────────────────────────────────────────────────────
_retrain_lock = threading.Lock()
_new_analyses_since_retrain = 0
_scheduler = None
_scheduler_lock_handle = None
_scheduler_lock_kind = None
_scheduler_status = {
    'enabled': env_flag_enabled('PILAR_ENABLE_SCHEDULER', True),
    'started': False,
    'reason': 'not_started',
    'lock_kind': None,
}
_SCHEDULER_LOCK_KEY = int(os.environ.get('PILAR_SCHEDULER_LOCK_KEY', '50495441'))
# RETRAIN_TRIGGER imported from config.py


def get_scheduler_status():
    return dict(_scheduler_status)


def _close_scheduler_lock():
    global _scheduler_lock_handle, _scheduler_lock_kind
    if _scheduler_lock_handle is None:
        return
    if _scheduler_lock_kind == 'postgres':
        try:
            _scheduler_lock_handle.close()
        except Exception:
            pass
    else:
        release_file_lock(_scheduler_lock_handle)
    _scheduler_lock_handle = None
    _scheduler_lock_kind = None


def _claim_scheduler_leadership():
    global _scheduler_lock_handle, _scheduler_lock_kind
    if not env_flag_enabled('PILAR_ENABLE_SCHEDULER', True):
        _scheduler_status.update({
            'enabled': False,
            'started': False,
            'reason': 'disabled_by_env',
            'lock_kind': None,
        })
        logger.info('[Scheduler] Scheduler disabled by PILAR_ENABLE_SCHEDULER=0')
        return False

    _scheduler_status['enabled'] = True

    if _scheduler_lock_handle is not None:
        return True

    if db_url.startswith('postgresql://'):
        conn = None
        try:
            from sqlalchemy import text as _sql_text

            with app.app_context():
                conn = db.engine.connect()
                locked = bool(conn.execute(
                    _sql_text('SELECT pg_try_advisory_lock(:lock_key)'),
                    {'lock_key': _SCHEDULER_LOCK_KEY},
                ).scalar())
            if not locked:
                conn.close()
                _scheduler_status.update({
                    'started': False,
                    'reason': 'leader_exists',
                    'lock_kind': 'postgres',
                })
                logger.info('[Scheduler] Scheduler skipped: another process owns the PostgreSQL advisory lock')
                return False
            _scheduler_lock_handle = conn
            _scheduler_lock_kind = 'postgres'
            return True
        except Exception as _lock_err:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            _scheduler_status.update({
                'started': False,
                'reason': f'lock_error:{type(_lock_err).__name__}',
                'lock_kind': 'postgres',
            })
            logger.info(f'[Scheduler] Scheduler lock error (postgres): {_lock_err}')
            return False

    lock_path = os.path.join(_DATA_DIR, 'pilar_scheduler.lock')
    handle = try_acquire_file_lock(lock_path)
    if handle is None:
        _scheduler_status.update({
            'started': False,
            'reason': 'leader_exists',
            'lock_kind': 'file',
        })
        logger.info(f'[Scheduler] Scheduler skipped: another local process owns {lock_path}')
        return False

    _scheduler_lock_handle = handle
    _scheduler_lock_kind = 'file'
    return True


def _shutdown_scheduler():
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler = None
    _close_scheduler_lock()

def _reload_models():
    """Reload pkl files into global model variables after a retrain."""
    global model, scaler, modeles_zones
    if pilar_ml.reload_models(_APP_DIR):
        model = pilar_ml.model
        scaler = pilar_ml.scaler
        modeles_zones = pilar_ml.modeles_zones
        logger.info("Models reloaded into memory")

def _auto_retrain():
    """Export Analysis DB rows to CSV and retrain the model pipeline."""
    global _new_analyses_since_retrain
    if not _retrain_lock.acquire(blocking=False):
        logger.warning("Retrain already in progress — skipped")
        return
    try:
        with app.app_context():
            import csv as _csv, tempfile as _tmp, subprocess as _sp, json as _js
            analyses = Analysis.query.order_by(Analysis.timestamp.asc()).all()
            if len(analyses) < 50:
                logger.info(f"retrain: Not enough data ({len(analyses)} rows) — skipped")
                return
            logger.info(f"retrain: Building training CSV from {len(analyses)} analyses...")
            rows = []
            for a in analyses:
                ep = {}
                try: ep = _js.loads(a.extra_params) if a.extra_params else {}
                except Exception: pass
                # Map Analysis columns back to feature names
                row = {
                    'vibration':            ep.get('vibration', FEATURE_MEDIANS['vibration']),
                    'temp_palier':          a.temp_air          if a.temp_air          is not None else FEATURE_MEDIANS['temp_palier'],
                    'debit':                a.vitesse            if a.vitesse            is not None else FEATURE_MEDIANS['debit'],
                    'pression_entree':      ep.get('pression_entree', FEATURE_MEDIANS['pression_entree']),
                    'pression_sortie':      a.couple             if a.couple             is not None else FEATURE_MEDIANS['pression_sortie'],
                    'courant_moteur':       ep.get('courant_moteur', FEATURE_MEDIANS['courant_moteur']),
                    'temp_moteur':          a.temp_process       if a.temp_process       is not None else FEATURE_MEDIANS['temp_moteur'],
                    'heure_fonctionnement': a.usure              if a.usure              is not None else FEATURE_MEDIANS['heure_fonctionnement'],
                    # target: use feedback if available, else model prediction
                    'etat_pompe_code': (1 if a.feedback in ('tp', 'fn') else 0 if a.feedback == 'fp' else a.prediction),
                }
                rows.append(row)
            # Write temp CSV
            tmp = _tmp.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8')
            writer = _csv.DictWriter(tmp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            tmp.close()
            csv_path = tmp.name
            logger.info(f"retrain: Temp CSV: {csv_path}")
            # Run retrain_real.py as subprocess
            # In a frozen PyInstaller app, 'python' is not in PATH — use sys.executable
            import sys as _sys_rt
            _retrain_script = os.path.join(_APP_DIR, 'retrain_real.py')
            if not os.path.exists(_retrain_script):
                logger.info(f"retrain: retrain_real.py not found at {_retrain_script} — skipped")
                return
            _python_exe = _sys_rt.executable if not _FROZEN else None
            if _python_exe is None or _FROZEN:
                # In frozen mode, find python.exe next to the exe or in PATH
                import shutil
                _python_exe = shutil.which('python') or shutil.which('python3') or 'python'
            result = _sp.run(
                [_python_exe, _retrain_script, csv_path],
                capture_output=True, text=True, timeout=300,
                cwd=_APP_DIR,
            )
            logger.info('retrain stdout: %s', result.stdout[-2000:] if result.stdout else '')
            if result.returncode == 0:
                logger.info('Retrain success — reloading models')
                _reload_models()
                _new_analyses_since_retrain = 0
                # Record retrain timestamp in settings
                try:
                    s = Settings.query.filter_by(key='last_auto_retrain', user_id=None).first()
                    ts = datetime.now(timezone.utc).isoformat()
                    if s: s.value = ts
                    else: db.session.add(Settings(key='last_auto_retrain', value=ts, user_id=None))
                    db.session.commit()
                except Exception: pass
                # Refresh per-machine specializations now that the global feature pipeline
                # has shifted — otherwise specialized blends drift out of sync with the base model.
                try:
                    _adaptive_sweep(force_specialize=True)
                except Exception as _as_e:
                    logger.warning(f"retrain: adaptive_sweep post-hook failed — {_as_e}")
            else:
                logger.info(f"retrain: FAILED (rc={result.returncode}): {result.stderr[-1000:]}")
            try:
                import os as _os; _os.unlink(csv_path)
            except Exception: pass
    except Exception as _re:
        logger.info(f"retrain: Error: {_re}")
    finally:
        _retrain_lock.release()


def _adaptive_sweep(force_specialize=False):
    """Every 6h: per-machine baseline refresh + specialization when enough new data.
    When force_specialize=True (called after a global retrain), re-specialize every
    machine that already has a MachineModel, regardless of data-gain threshold, so
    specialized blends stay coherent with the refreshed global feature pipeline."""
    try:
        with app.app_context():
            machines = Machine.query.filter_by(is_active=True).all()
            logger.info(f"adaptive_sweep: checking {len(machines)} machines (force_specialize={force_specialize})")
            for m in machines:
                try:
                    total = Analysis.query.filter(Analysis.machine_id == m.name).count()
                    mm = MachineModel.query.filter_by(machine_id=m.id).first()
                    # Specialization: 50+ total analyses and either untrained or 50+ new since last train
                    should_specialize = False
                    if total >= _MIN_SPECIALIZE_SAMPLES:
                        if not mm:
                            should_specialize = True
                        elif force_specialize:
                            should_specialize = True
                        else:
                            gained = total - (mm.training_samples or 0)
                            if gained >= _MIN_SPECIALIZE_SAMPLES:
                                should_specialize = True
                    if should_specialize:
                        logger.info(f"adaptive_sweep[{m.id}/{m.name}]: triggering specialization (n={total})")
                        specialize_machine_model(m)
                    # Baseline: recompute if missing or older than _BASELINE_STALE_DAYS
                    br = MachineBaseline.query.filter_by(machine_id=m.id).first()
                    stale = False
                    if not br:
                        stale = True
                    else:
                        computed = br.computed_at
                        if computed and computed.tzinfo is None:
                            computed = computed.replace(tzinfo=timezone.utc)
                        if not computed:
                            stale = True
                        else:
                            age_days = (datetime.now(timezone.utc) - computed).days
                            if age_days >= _BASELINE_STALE_DAYS:
                                stale = True
                    if stale and total >= _MIN_BASELINE_SAMPLES:
                        logger.info(f"adaptive_sweep[{m.id}/{m.name}]: refreshing baseline")
                        compute_machine_baseline(m)
                except Exception as _me:
                    logger.error(f"adaptive_sweep[{m.id}]: {_me}")
    except Exception as _e:
        logger.error(f"adaptive_sweep fatal: {_e}")


def _start_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler_status.update({
            'enabled': True,
            'started': True,
            'reason': 'already_running',
            'lock_kind': _scheduler_lock_kind,
        })
        return _scheduler

    if not _claim_scheduler_leadership():
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        _scheduler = BackgroundScheduler(timezone='UTC')
        _scheduler.add_job(_send_weekly_reports, CronTrigger(day_of_week='mon', hour=8, minute=0),
                           id='weekly_pdf_reports', replace_existing=True)
        _scheduler.add_job(_auto_retrain, CronTrigger(day_of_week='sun', hour=3, minute=0),
                           id='weekly_auto_retrain', replace_existing=True)
        _scheduler.add_job(_adaptive_sweep, CronTrigger(hour='*/6'),
                           id='adaptive_per_machine_sweep', replace_existing=True)
        _scheduler.start()
        _scheduler_status.update({
            'enabled': True,
            'started': True,
            'reason': 'leader',
            'lock_kind': _scheduler_lock_kind,
        })
        logger.info(f"APScheduler started - weekly reports Mon 08:00 UTC | auto-retrain Sun 03:00 UTC ({_scheduler_lock_kind} leader)")
        return _scheduler
    except Exception as _se:
        _scheduler_status.update({
            'enabled': True,
            'started': False,
            'reason': f'start_error:{type(_se).__name__}',
            'lock_kind': _scheduler_lock_kind,
        })
        logger.info(f"APScheduler not available: {_se}")
        _close_scheduler_lock()
        _scheduler = None
        return None


atexit.register(_shutdown_scheduler)
_start_scheduler()

# ── ROUTES AUTH ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        if current_uid(): return redirect('/monitor')
        return render_template('register.html', error=None, pending=False)
    ip = (request.headers.get('X-Forwarded-For','').split(',')[0].strip() if os.environ.get('RAILWAY_ENVIRONMENT') else '') or request.remote_addr or ''
    if _check_rate_limit(ip):
        logger.info(f"auth: Rate limit register IP={ip}")
        return render_template('register.html', error='Trop de tentatives. Réessayez dans 15 minutes.', pending=False)
    try:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password', '')
        password2 = request.form.get('password2', '')
        if not email or not password:
            return render_template('register.html', error='Email et mot de passe requis', pending=False)
        if len(password) < 8:
            return render_template('register.html', error='Mot de passe trop court (8 caractères minimum)', pending=False)
        if password != password2:
            return render_template('register.html', error='Les mots de passe ne correspondent pas', pending=False)
        if BannedEmail.query.filter_by(email=email).first():
            _record_failed_login(ip)
            return render_template('register.html', error='Cette adresse email est bloquée. Contactez le support.', pending=False)
        if User.query.filter_by(email=email).first():
            _record_failed_login(ip)
            return render_template('register.html', error='Un compte existe déjà avec cet email', pending=False)
        api_key = 'pk_' + _secrets.token_hex(24)
        is_admin = (email == os.environ.get('ADMIN_EMAIL', '').lower()) or (User.query.count() == 0)
        token = _secrets.token_hex(32)
        # Desktop app — auto-verify all accounts (no email relay needed)
        needs_verify = False
        user = User(email=email, password_hash=generate_password_hash(password, method='pbkdf2:sha256:600000'),
                    email_verified=not needs_verify, verify_token=token if needs_verify else None,
                    api_key=api_key, is_admin=is_admin, onboarded=True)
        db.session.add(user)
        db.session.commit()
        logger.info(f"auth: New user: {email} (admin={is_admin}, verified={not needs_verify}) IP={ip}")
        if needs_verify:
            base_url = request.host_url.rstrip('/')
            threading.Thread(target=send_verify_email, args=(email, token, base_url), daemon=True).start()
            session['_pending_verify'] = email
            return render_template('register.html', error=None, pending=True, resent=False, pending_email=email)
        session['user_id'] = user.id
        session.permanent = True
        return redirect('/monitor')
    except Exception as e:
        db.session.rollback()
        logger.info(f"auth: Register error: {type(e).__name__}: {e}")
        return render_template('register.html', error='Erreur serveur. Veuillez réessayer.', pending=False)

@app.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'GET':
        # Page de renvoi autonome (si l'utilisateur revient plus tard)
        email = session.get('_pending_verify', '')
        return render_template('register.html', error=None, pending=True, resent=False, pending_email=email)
    email = (request.form.get('email') or session.get('_pending_verify', '')).strip().lower()
    if email:
        user = User.query.filter_by(email=email, email_verified=False).first()
        if user:
            if not GMAIL or not GMAIL_PWD:
                logger.info(f"auth: Resend impossible: GMAIL non configuré pour {email}")
            else:
                token = _secrets.token_hex(32)
                user.verify_token = token
                try:
                    db.session.commit()
                    base_url = request.host_url.rstrip('/')
                    threading.Thread(target=send_verify_email, args=(email, token, base_url), daemon=True).start()
                    logger.info(f"auth: Resend verification email: {email}")
                except Exception as e:
                    db.session.rollback()
                    logger.info(f"auth: Resend error: {e}")
    # On affiche toujours le succès (anti-énumération)
    return render_template('register.html', error=None, pending=True, resent=True, pending_email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if current_uid(): return redirect('/monitor')
        return render_template('login.html', error=None)
    ip = (request.headers.get('X-Forwarded-For','').split(',')[0].strip() if os.environ.get('RAILWAY_ENVIRONMENT') else '') or request.remote_addr or ''
    if _check_rate_limit(ip):
        logger.info(f"auth: Rate limit login IP={ip}")
        return render_template('login.html', error='Trop de tentatives. Réessayez dans 15 minutes.')
    try:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password', '')
        if not email or not password:
            return render_template('login.html', error='Email et mot de passe requis')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            _record_failed_login(ip)
            logger.info(f"auth: Failed login: {email} IP={ip}")
            return render_template('login.html', error='Email ou mot de passe incorrect')
        if user.is_banned:
            logger.info(f"auth: Banned login attempt: {email} IP={ip}")
            return render_template('login.html', error='Ce compte a été suspendu. Contactez le support.')
        if not user.email_verified:
            # If GMAIL is not configured, no verification email was ever sent — auto-verify
            if not GMAIL:
                user.email_verified = True
                db.session.commit()
            else:
                return render_template('login.html', error='Confirmez votre email avant de vous connecter. Vérifiez vos spams.')
        session['user_id'] = user.id
        session.permanent = (request.form.get('remember') == '1')
        logger.info(f"auth: Login OK: {email} IP={ip} remember={session.permanent}")
        return redirect('/monitor')
    except Exception as e:
        db.session.rollback()
        logger.info(f"auth: Login error: {type(e).__name__}: {e}")
        return render_template('login.html', error='Erreur serveur. Veuillez réessayer.')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    uid = current_uid()
    user = User.query.get(uid)
    if not user:
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json() or {}
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    if not check_password_hash(user.password_hash, old_pw):
        return jsonify({'error': 'Current password is incorrect'}), 400
    if len(new_pw) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    user.password_hash = generate_password_hash(new_pw)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/verify-email/<token>')
def verify_email(token):
    user = User.query.filter_by(verify_token=token).first()
    if not user:
        return "<h2 style='font-family:sans-serif;color:#dc2626;padding:40px'>Lien invalide ou expiré.</h2>"
    user.email_verified = True
    user.verify_token = None
    db.session.commit()
    session['user_id'] = user.id
    session.permanent = True
    return redirect('/monitor')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot.html', error=None, error_key=None, msg=None, msg_key=None)
    email = (request.form.get('email') or '').strip().lower()
    if not email:
        return render_template('forgot.html', error='Email requis.', error_key='forgot_err_email', msg=None, msg_key=None)
    user = User.query.filter_by(email=email).first()
    # Always show success message to avoid user enumeration
    if user and not user.is_banned:
        token = _secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        db.session.commit()
        threading.Thread(target=send_reset_email, args=(email, token), daemon=True).start()
    return render_template('forgot.html', error=None, error_key=None,
        msg='Si un compte existe pour cet email, un lien de réinitialisation a été envoyé.',
        msg_key='forgot_success')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    now = datetime.now(timezone.utc)
    if not user or user.reset_token_expires is None or user.reset_token_expires.replace(tzinfo=timezone.utc) < now:
        return render_template('reset.html', token=token,
            error='Lien invalide ou expiré. Recommencez la procédure.', error_key='reset_err_expired')
    if request.method == 'GET':
        return render_template('reset.html', token=token, error=None, error_key=None)
    pw = request.form.get('password', '')
    pw2 = request.form.get('password2', '')
    if len(pw) < 8:
        return render_template('reset.html', token=token,
            error='Le mot de passe doit faire au moins 8 caractères.', error_key='reset_err_short')
    if pw != pw2:
        return render_template('reset.html', token=token,
            error='Les mots de passe ne correspondent pas.', error_key='reset_err_match')
    user.password_hash = generate_password_hash(pw)
    user.reset_token = None
    user.reset_token_expires = None
    db.session.commit()
    return redirect('/login?reset=1')

@app.route('/profile/api-key', methods=['GET', 'POST'])
@login_required
def api_key_page():
    user = db.session.get(User, current_uid())
    if request.method == 'POST':
        user.api_key = 'pk_' + _secrets.token_hex(24)
        db.session.commit()
    return jsonify({'api_key': user.api_key})

@app.route('/admin')
@admin_required
def admin():
    # Auto-expire plans
    now = datetime.now(timezone.utc)
    expired = User.query.filter(
        User.plan != 'free',
        User.plan_expires_at != None,
        User.plan_expires_at < now
    ).all()
    for u in expired:
        u.plan = 'free'
    if expired:
        db.session.commit()

    users = User.query.order_by(User.created_at.desc()).all()
    for u in users:
        u.analysis_count = Analysis.query.filter_by(user_id=u.id).count()

    total_users = len(users)
    total_analyses = Analysis.query.count()
    paid_users = sum(1 for u in users if u.plan in ('starter', 'pro'))
    mrr = sum(99 if u.plan == 'starter' else 299 if u.plan == 'pro' else 0 for u in users)
    expiring_soon = sum(1 for u in users if u.plan_expires_at and 0 <= (u.plan_expires_at - now).days < 7)

    banned_emails = BannedEmail.query.order_by(BannedEmail.banned_at.desc()).all()
    return render_template('admin.html', users=users, total_users=total_users,
                                  total_analyses=total_analyses, paid_users=paid_users,
                                  mrr=mrr, expiring_soon=expiring_soon, now=now,
                                  banned_emails=banned_emails)

@app.route('/admin/set_plan/<int:uid>', methods=['POST'])
@admin_required
def admin_set_plan(uid):
    user = db.session.get(User, uid)
    if not user:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    data = request.json or {}
    plan = data.get('plan', 'free')
    if plan not in ('free', 'starter', 'pro'):
        return jsonify({'error': 'Plan invalide'}), 400
    expires_str = data.get('expires_at', '')
    note = data.get('note', '')
    old_plan = user.plan
    user.plan = plan
    user.plan_note = note[:300] if note else None
    if expires_str:
        try:
            user.plan_expires_at = datetime.strptime(expires_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Date invalide'}), 400
    else:
        user.plan_expires_at = None
    db.session.commit()
    admin_user = db.session.get(User, current_uid())
    logger.info(f"admin] PLAN_CHANGE by {admin_user.email if admin_user else '?'}: user={user.email} {old_plan}->{plan} expires={expires_str or 'none'} note={note[:50] if note else ''}")
    return jsonify({'ok': True})

@app.route('/admin/set_quota/<int:uid>', methods=['POST'])
@admin_required
def admin_set_quota(uid):
    user = db.session.get(User, uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    data = request.json or {}
    try:
        quota = int(data.get('quota', 3))
        if quota < 0: raise ValueError
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid quota'}), 400
    user.machine_quota = quota
    db.session.commit()
    logger.info(f"admin] QUOTA_SET: user={user.email} quota={quota}")
    return jsonify({'ok': True})

@app.route('/admin/toggle_admin/<int:uid>', methods=['POST'])
@admin_required
def admin_toggle_admin(uid):
    me = db.session.get(User, current_uid())
    target = db.session.get(User, uid)
    if not target:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    if target.id == me.id:
        return jsonify({'error': 'Impossible de modifier ses propres droits'}), 400
    target.is_admin = not target.is_admin
    db.session.commit()
    action = 'GRANT_ADMIN' if target.is_admin else 'REVOKE_ADMIN'
    logger.info(f"admin] {action} by {me.email}: target={target.email}")
    return jsonify({'ok': True, 'is_admin': target.is_admin})

@app.route('/admin/toggle_ban/<int:uid>', methods=['POST'])
@admin_required
def admin_toggle_ban(uid):
    me = db.session.get(User, current_uid())
    target = db.session.get(User, uid)
    if not target:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    if target.id == me.id:
        return jsonify({'error': 'Impossible de se bannir soi-même'}), 400
    target.is_banned = not target.is_banned
    db.session.commit()
    action = 'BAN' if target.is_banned else 'UNBAN'
    logger.info(f"admin] {action} by {me.email}: target={target.email}")
    return jsonify({'ok': True, 'is_banned': target.is_banned})

@app.route('/admin/delete_user/<int:uid>', methods=['POST'])
@admin_required
def admin_delete_user(uid):
    me = db.session.get(User, current_uid())
    target = db.session.get(User, uid)
    if not target:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    if target.id == me.id:
        return jsonify({'error': 'Impossible de supprimer son propre compte'}), 400
    email = target.email
    # Supprimer les données liées
    Analysis.query.filter_by(user_id=uid).delete()
    Settings.query.filter_by(user_id=uid).delete()
    SavedFile.query.filter_by(user_id=uid).delete()
    TeamMember.query.filter_by(user_id=uid).delete()
    # Retirer de la team
    if target.team_id:
        target.team_id = None
    db.session.delete(target)
    db.session.commit()
    logger.info(f"admin] DELETE_USER by {me.email}: deleted={email}")
    return jsonify({'ok': True})

@app.route('/admin/retrain', methods=['POST'])
@admin_required
def admin_retrain():
    """Manually trigger an auto-retrain from the admin panel."""
    threading.Thread(target=_auto_retrain, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Retrain started in background — check server logs'})

@app.route('/admin/retrain/status')
@admin_required
def admin_retrain_status():
    locked = not _retrain_lock.acquire(blocking=False)
    if not locked:
        _retrain_lock.release()
    s = Settings.query.filter_by(key='last_auto_retrain', user_id=None).first()
    return jsonify({
        'in_progress': locked,
        'last_retrain': s.value if s else None,
        'analyses_since_retrain': _new_analyses_since_retrain,
        'trigger_at': RETRAIN_TRIGGER,
    })

@app.route('/admin/block_email', methods=['POST'])
@admin_required
def admin_block_email():
    me = db.session.get(User, current_uid())
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    reason = (data.get('reason') or '').strip()[:300]
    if not email:
        return jsonify({'error': 'Email requis'}), 400
    if BannedEmail.query.filter_by(email=email).first():
        return jsonify({'error': 'Email déjà bloqué'}), 409
    db.session.add(BannedEmail(email=email, reason=reason or None))
    db.session.commit()
    logger.info(f"admin] BLOCK_EMAIL by {me.email}: email={email}")
    return jsonify({'ok': True})

@app.route('/admin/unblock_email/<int:bid>', methods=['POST'])
@admin_required
def admin_unblock_email(bid):
    me = db.session.get(User, current_uid())
    entry = db.session.get(BannedEmail, bid)
    if not entry:
        return jsonify({'error': 'Introuvable'}), 404
    email = entry.email
    db.session.delete(entry)
    db.session.commit()
    logger.info(f"admin] UNBLOCK_EMAIL by {me.email}: email={email}")
    return jsonify({'ok': True})

# ── Admin System Info (read-only, no subprocess) ────────────────────────────
def _admin_sysinfo():
    """Gather read-only system info without subprocess calls."""
    import sys as _sys, platform as _plat, shutil as _shu
    lines = []
    lines.append(f"Python: {_sys.version}")
    lines.append(f"Platform: {_plat.platform()}")
    lines.append(f"Machine: {_plat.machine()}")
    lines.append(f"App dir: {os.path.dirname(os.path.abspath(__file__))}")
    # Disk
    try:
        usage = _shu.disk_usage(os.path.dirname(os.path.abspath(__file__)))
        lines.append(f"Disk: {usage.free // (1024**3)} GB free / {usage.total // (1024**3)} GB total")
    except Exception:
        pass
    # DB size
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pilar.db')
    if os.path.isfile(db_path):
        lines.append(f"DB size: {os.path.getsize(db_path) / (1024*1024):.1f} MB")
    # Model files
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ('failure_model.pkl', 'zone_models.pkl', 'isolation_forest.pkl', 'rul_model.pkl'):
        fpath = os.path.join(_app_dir, fname)
        if os.path.isfile(fpath):
            lines.append(f"{fname}: {os.path.getsize(fpath) / (1024*1024):.1f} MB")
    # Installed packages (top-level only)
    try:
        import importlib.metadata as _meta
        pkgs = sorted(_meta.distributions(), key=lambda d: d.metadata['Name'].lower())
        lines.append(f"\nInstalled packages ({len(pkgs)}):")
        for d in pkgs:
            lines.append(f"  {d.metadata['Name']} {d.version}")
    except Exception:
        pass
    return '\n'.join(lines)

@app.route('/admin/terminal', methods=['POST'])
@admin_required
def admin_terminal():
    """Read-only system info endpoint. No shell execution."""
    try:
        output = _admin_sysinfo()
        return jsonify({'output': output[:8000], 'code': 0})
    except Exception as e:
        return jsonify({'output': str(e), 'code': -1})

@app.route('/admin/impersonate/<int:uid>')
@admin_required
def impersonate(uid):
    admin_user = db.session.get(User, current_uid())
    target = db.session.get(User, uid)
    logger.info(f"admin] IMPERSONATE by {admin_user.email if admin_user else '?'}: target={target.email if target else uid}")
    session['user_id'] = uid
    return redirect('/monitor')

# ── ROUTES PAGES ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect('/monitor')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/demo-login')
def demo_login():
    """One-click demo access — logs in as demo@pilar.app, creates account if needed."""
    DEMO_EMAIL = 'demo@pilar.app'
    try:
        _seed_demo_account()
    except Exception:
        db.session.rollback()
    user = User.query.filter_by(email=DEMO_EMAIL).first()
    if not user:
        return redirect('/login')
    session.clear()
    session['user_id'] = user.id
    session.permanent = True
    logger.info(f"Demo login: uid={user.id}")
    return redirect('/monitor')

# ── ALERT ACK ─────────────────────────────────────────────────────────────────
@app.route('/alert/ack/<token>')
def alert_ack(token):
    al = AlertLog.query.filter_by(ack_token=token).first()
    if not al:
        return '<html><body style="font-family:-apple-system,\'SF Pro Display\',\'Helvetica Neue\',Arial,sans-serif;background:#07090f;color:#ffffff;display:flex;align-items:center;justify-content:center;min-height:100vh;"><div style="text-align:center"><div style="font-size:15px;letter-spacing:0.04em;color:#0d9488;font-weight:700;margin-bottom:16px;">PILAR</div><p style="color:rgba(235,235,245,0.6)">Alert not found or already processed.</p></div></body></html>', 404
    if not al.acked_at:
        al.acked_at = datetime.now(timezone.utc)
        db.session.commit()
    return '<html><body style="font-family:-apple-system,\'SF Pro Display\',\'Helvetica Neue\',Arial,sans-serif;background:#07090f;color:#ffffff;display:flex;align-items:center;justify-content:center;min-height:100vh;"><div style="text-align:center"><div style="font-size:15px;letter-spacing:0.04em;color:#0d9488;font-weight:700;margin-bottom:16px;">PILAR</div><h2 style="margin:0 0 12px;font-size:20px;font-weight:700;">Alert Acknowledged</h2><p style="color:rgba(235,235,245,0.6);font-size:14px;">This alert has been recorded. No escalation will be sent.</p><a href="/monitor" style="display:inline-block;margin-top:20px;padding:12px 24px;background:#0d9488;color:#fff;text-decoration:none;border-radius:12px;font-size:15px;font-weight:600;">Go to Dashboard</a></div></body></html>'

# ── MACHINES CRUD API ─────────────────────────────────────────────────────────
@app.route('/api/machines', methods=['GET'])
@login_required
def api_machines_list():
    uid = current_uid()
    machines = Machine.query.filter_by(user_id=uid).order_by(Machine.created_at.desc()).all()
    if not machines:
        return jsonify([])
    machine_names = [m.name for m in machines]
    machine_ids   = [m.id for m in machines]
    # Bulk-fetch latest analysis per machine (stored by machine name string)
    from sqlalchemy import func as _func
    subq = (db.session.query(Analysis.machine_id, _func.max(Analysis.timestamp).label('max_ts'))
            .filter(Analysis.machine_id.in_(machine_names))
            .group_by(Analysis.machine_id).subquery())
    latest_rows = (db.session.query(Analysis)
                   .join(subq, (Analysis.machine_id == subq.c.machine_id) &
                                (Analysis.timestamp == subq.c.max_ts))
                   .all())
    last_by_name = {a.machine_id: a for a in latest_rows}
    # Bulk-fetch latest saved file per machine (simple ordered query, most recent first)
    sf_by_mid = {}
    sf_rows = (SavedFile.query
               .filter(SavedFile.machine_id.in_(machine_ids), SavedFile.user_id == uid)
               .order_by(SavedFile.created_at.desc())
               .all())
    for _sf in sf_rows:
        if _sf.machine_id not in sf_by_mid:
            sf_by_mid[_sf.machine_id] = _sf
    result = []
    for m in machines:
        last = last_by_name.get(m.name)
        sf   = sf_by_mid.get(m.id)
        result.append({
            'id': m.id, 'name': m.name, 'description': m.description,
            'machine_type': m.machine_type, 'threshold': m.threshold,
            'pump_type': m.pump_type or 'centrifuge',
            'fluid_type': m.fluid_type or 'eau',
            'roue_material': m.roue_material or 'inox_316',
            'location': m.location or '',
            'install_date': m.install_date.isoformat() if m.install_date else None,
            'serial_number': m.serial_number or '',
            'nominal_flow': m.nominal_flow,
            'nominal_pressure': m.nominal_pressure,
            'power_kw': m.power_kw,
            'nominal_current': m.nominal_current,
            'nominal_vibration': m.nominal_vibration,
            'alert_email': m.alert_email, 'escalation_email': m.escalation_email,
            'is_active': m.is_active,
            'asset_type': m.asset_type or 'pump',
            'brand': m.brand or '',
            'model_name': m.model_name or '',
            'age_years': m.age_years or 0,
            'environment': m.environment or 'general',
            'criticality': m.criticality or 'medium',
            'last_maintenance': m.last_maintenance.isoformat() + 'Z' if m.last_maintenance else None,
            'last_risk': last.risk if last else None,
            'last_prediction': last.prediction if last else None,
            'last_seen': last.timestamp.isoformat() + 'Z' if last else None,
            'saved_file': {'id': sf.id, 'filename': sf.filename, 'row_count': sf.row_count,
                           'created_at': sf.created_at.isoformat()+'Z'} if sf else None,
            'adaptive_status': _machine_adaptive_status(m),
        })
    return jsonify(result)

def _machine_adaptive_status(m):
    """Summary dict for UI: model type + sample counts + baseline freshness."""
    try:
        total = Analysis.query.filter(Analysis.machine_id == m.name).count()
        mm = MachineModel.query.filter_by(machine_id=m.id).first()
        baseline_row = MachineBaseline.query.filter_by(machine_id=m.id).first()
        if mm and mm.f1_score:
            status = f'specialized (v{mm.version}, {mm.training_samples} samples, F1: {int(round(mm.f1_score*100))}%)'
            model_type = 'specialized'
        elif total >= _MIN_SPECIALIZE_SAMPLES:
            status = f'global_model (eligible for specialization — {total} samples)'
            model_type = 'global'
        elif total > 0:
            status = f'global_model ({total}/{_MIN_SPECIALIZE_SAMPLES} samples)'
            model_type = 'global'
        else:
            status = 'no_data'
            model_type = 'none'
        return {
            'model_type': model_type, 'status': status,
            'total_analyses': total,
            'model_f1': round(mm.f1_score, 3) if (mm and mm.f1_score) else None,
            'model_version': mm.version if mm else None,
            'last_trained': mm.last_trained.isoformat() + 'Z' if (mm and mm.last_trained) else None,
            'baseline_ready': baseline_row is not None,
            'baseline_computed_at': baseline_row.computed_at.isoformat() + 'Z' if baseline_row else None,
        }
    except Exception:
        return {'model_type': 'none', 'status': 'no_data', 'total_analyses': 0}

@app.route('/api/machines', methods=['POST'])
@login_required
def api_machines_create():
    uid = current_uid()
    user = db.session.get(User, uid)
    d = request.json or {}
    name = (d.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    if Machine.query.filter_by(user_id=uid, name=name).first():
        return jsonify({'error': 'A machine with this name already exists'}), 409
    current_count = Machine.query.filter_by(user_id=uid, is_active=True).count()
    _ = current_count  # no quota limit — all installed users have full access
    def _float_or_none(v):
        try: return float(v) if v not in (None, '') else None
        except (TypeError, ValueError): return None
    from datetime import date as _date
    _install = None
    if d.get('install_date'):
        try: _install = datetime.strptime(d['install_date'], '%Y-%m-%d').date()
        except (ValueError, TypeError): pass
    m = Machine(user_id=uid, name=name,
        description=(d.get('description') or '')[:500],
        machine_type=d.get('machine_type', 'M'),
        threshold=float(d.get('threshold') or DEFAULT_THRESHOLD),
        pump_type=(d.get('pump_type') or 'centrifuge')[:50],
        fluid_type=(d.get('fluid_type') or 'eau')[:50],
        roue_material=(d.get('roue_material') or 'inox_316')[:50],
        location=(d.get('location') or '')[:200],
        install_date=_install,
        serial_number=(d.get('serial_number') or '')[:100],
        nominal_flow=_float_or_none(d.get('nominal_flow')),
        nominal_pressure=_float_or_none(d.get('nominal_pressure')),
        power_kw=_float_or_none(d.get('power_kw')),
        nominal_current=_float_or_none(d.get('nominal_current')),
        nominal_vibration=_float_or_none(d.get('nominal_vibration')),
        alert_email=(d.get('alert_email') or '').strip(),
        escalation_email=(d.get('escalation_email') or '').strip(),
        asset_type=(d.get('asset_type') or 'pump')[:50],
        brand=(d.get('brand') or '')[:100],
        model_name=(d.get('model_name') or '')[:100],
        age_years=_float_or_none(d.get('age_years')) or 0.0,
        environment=(d.get('environment') or 'general')[:50],
        criticality=(d.get('criticality') or 'medium')[:20],
        last_maintenance=(datetime.strptime(d['last_maintenance'], '%Y-%m-%d') if d.get('last_maintenance') else None),
        is_active=True)
    db.session.add(m)
    db.session.commit()
    return jsonify({'ok': True, 'id': m.id, 'name': m.name}), 201

@app.route('/api/machines/<int:mid>', methods=['PUT'])
@login_required
def api_machines_update(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    d = request.json or {}
    if 'name' in d: m.name = (d['name'] or '').strip()[:200]
    if 'description' in d: m.description = (d['description'] or '')[:500]
    if 'machine_type' in d: m.machine_type = d['machine_type']
    if 'threshold' in d:
        try: m.threshold = float(d['threshold'])
        except (TypeError, ValueError): pass
    if 'pump_type' in d: m.pump_type = (d['pump_type'] or 'centrifuge')[:50]
    if 'fluid_type' in d: m.fluid_type = (d['fluid_type'] or 'eau')[:50]
    if 'roue_material' in d: m.roue_material = (d['roue_material'] or 'inox_316')[:50]
    if 'location' in d: m.location = (d['location'] or '')[:200]
    if 'serial_number' in d: m.serial_number = (d['serial_number'] or '')[:100]
    if 'install_date' in d:
        try: m.install_date = datetime.strptime(d['install_date'], '%Y-%m-%d').date() if d['install_date'] else None
        except (ValueError, TypeError): pass
    def _fu(v):
        try: return float(v) if v not in (None, '') else None
        except (TypeError, ValueError): return None
    if 'nominal_flow' in d: m.nominal_flow = _fu(d['nominal_flow'])
    if 'nominal_pressure' in d: m.nominal_pressure = _fu(d['nominal_pressure'])
    if 'power_kw' in d: m.power_kw = _fu(d['power_kw'])
    if 'nominal_current' in d: m.nominal_current = _fu(d['nominal_current'])
    if 'nominal_vibration' in d: m.nominal_vibration = _fu(d['nominal_vibration'])
    if 'alert_email' in d: m.alert_email = (d['alert_email'] or '').strip()
    if 'escalation_email' in d: m.escalation_email = (d['escalation_email'] or '').strip()
    if 'is_active' in d: m.is_active = bool(d['is_active'])
    if 'asset_type' in d: m.asset_type = (d['asset_type'] or 'pump')[:50]
    if 'brand' in d: m.brand = (d['brand'] or '')[:100]
    if 'model_name' in d: m.model_name = (d['model_name'] or '')[:100]
    if 'age_years' in d: m.age_years = _fu(d['age_years']) or 0.0
    if 'environment' in d: m.environment = (d['environment'] or 'general')[:50]
    if 'criticality' in d: m.criticality = (d['criticality'] or 'medium')[:20]
    if 'last_maintenance' in d:
        try:
            m.last_maintenance = datetime.strptime(d['last_maintenance'], '%Y-%m-%d') if d['last_maintenance'] else None
        except (ValueError, TypeError):
            pass
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/machines/<int:mid>', methods=['DELETE'])
@login_required
def api_machines_delete(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    db.session.delete(m)
    db.session.commit()
    return jsonify({'ok': True})

# ── MACHINE NOTES API ─────────────────────────────────────────────────────────
@app.route('/api/machines/<int:mid>/notes', methods=['GET'])
@login_required
def api_machine_notes_list(mid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    notes = MachineNote.query.filter_by(machine_id=mid).order_by(MachineNote.created_at.desc()).all()
    return jsonify([{
        'id': n.id, 'content': n.content,
        'user_email': n.user_email, 'user_id': n.user_id,
        'created_at': n.created_at.isoformat() + 'Z',
    } for n in notes])

@app.route('/api/machines/<int:mid>/notes', methods=['POST'])
@login_required
def api_machine_notes_create(mid):
    uid = current_uid()
    user = db.session.get(User, uid)
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    content = ((request.json or {}).get('content') or '').strip()
    if not content:
        return jsonify({'error': 'content required'}), 400
    n = MachineNote(machine_id=mid, user_id=uid,
                    user_email=user.email if user else '',
                    content=content[:2000])
    db.session.add(n)
    db.session.commit()
    return jsonify({'ok': True, 'id': n.id, 'created_at': n.created_at.isoformat() + 'Z'}), 201

@app.route('/api/machines/<int:mid>/notes/<int:nid>', methods=['DELETE'])
@login_required
def api_machine_notes_delete(mid, nid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    n = MachineNote.query.filter_by(id=nid, machine_id=mid).first_or_404()
    # Only note author or admin can delete
    user = db.session.get(User, uid)
    if n.user_id != uid and not (user and user.is_admin):
        return jsonify({'error': 'Forbidden'}), 403
    db.session.delete(n)
    db.session.commit()
    return jsonify({'ok': True})

# ── ADAPTIVE MODEL API (baseline + specialize + maintenance + profile) ───────
@app.route('/api/machines/<int:mid>/profile', methods=['GET'])
@login_required
def api_machine_profile(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    baseline = [{
        'feature':    b.feature,
        'mean':       b.mean, 'std': b.std,
        'min_normal': b.min_normal, 'max_normal': b.max_normal,
        'sample_count': b.sample_count,
        'computed_at': b.computed_at.isoformat() + 'Z' if b.computed_at else None,
    } for b in MachineBaseline.query.filter_by(machine_id=mid).order_by(MachineBaseline.feature).all()]
    events = [{
        'id': e.id, 'event_type': e.event_type, 'description': e.description,
        'parts_replaced': e.parts_replaced, 'cost': e.cost,
        'timestamp': e.timestamp.isoformat() + 'Z' if e.timestamp else None,
    } for e in MaintenanceEvent.query.filter_by(machine_id=mid).order_by(MaintenanceEvent.timestamp.desc()).limit(20).all()]
    mm = MachineModel.query.filter_by(machine_id=mid).first()
    return jsonify({
        'id': m.id, 'name': m.name, 'description': m.description,
        'asset_type': m.asset_type or 'pump', 'brand': m.brand or '',
        'model_name': m.model_name or '', 'age_years': m.age_years or 0,
        'environment': m.environment or 'general',
        'criticality': m.criticality or 'medium',
        'location': m.location or '',
        'serial_number': m.serial_number or '',
        'install_date': m.install_date.isoformat() if m.install_date else None,
        'last_maintenance': m.last_maintenance.isoformat() + 'Z' if m.last_maintenance else None,
        'nominal': {
            'flow':      m.nominal_flow,      'pressure':  m.nominal_pressure,
            'current':   m.nominal_current,   'vibration': m.nominal_vibration,
            'power_kw':  m.power_kw,
        },
        'baseline': baseline,
        'maintenance_events': events,
        'adaptive_status': _machine_adaptive_status(m),
        'model': {
            'f1_score':         mm.f1_score if mm else None,
            'training_samples': mm.training_samples if mm else None,
            'version':          mm.version if mm else None,
            'last_trained':     mm.last_trained.isoformat() + 'Z' if (mm and mm.last_trained) else None,
        } if mm else None,
    })

@app.route('/api/machines/<int:mid>/specialize', methods=['POST'])
@login_required
def api_machine_specialize(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    res = specialize_machine_model(m)
    return jsonify(res)

@app.route('/api/machines/<int:mid>/baseline', methods=['POST'])
@login_required
def api_machine_baseline(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    b = compute_machine_baseline(m)
    if b is None:
        return jsonify({'status': 'insufficient_data',
                        'reason': f'need ≥{_MIN_BASELINE_SAMPLES} normal analyses'}), 200
    return jsonify({'status': 'ok', 'features': b})

@app.route('/api/machines/<int:mid>/maintenance', methods=['GET'])
@login_required
def api_machine_maintenance_list(mid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    events = MaintenanceEvent.query.filter_by(machine_id=mid)\
               .order_by(MaintenanceEvent.timestamp.desc()).all()
    return jsonify([{
        'id': e.id, 'event_type': e.event_type, 'description': e.description,
        'parts_replaced': e.parts_replaced, 'cost': e.cost,
        'timestamp': e.timestamp.isoformat() + 'Z' if e.timestamp else None,
    } for e in events])

@app.route('/api/machines/<int:mid>/maintenance', methods=['POST'])
@login_required
def api_machine_maintenance_create(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    d = request.json or {}
    e = MaintenanceEvent(
        machine_id=mid,
        event_type=(d.get('event_type') or 'inspection')[:50],
        description=(d.get('description') or '')[:2000],
        parts_replaced=(d.get('parts_replaced') or '')[:500],
        cost=(float(d['cost']) if d.get('cost') not in (None, '') else None),
    )
    db.session.add(e)
    # Update machine's last_maintenance if event is preventive/repair
    if e.event_type in ('preventive', 'repair'):
        m.last_maintenance = e.timestamp or datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({'ok': True, 'id': e.id,
                    'timestamp': e.timestamp.isoformat() + 'Z' if e.timestamp else None}), 201

@app.route('/api/machines/<int:mid>/maintenance/<int:eid>', methods=['DELETE'])
@login_required
def api_machine_maintenance_delete(mid, eid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    e = MaintenanceEvent.query.filter_by(id=eid, machine_id=mid).first_or_404()
    db.session.delete(e)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/machines/<int:mid>/analyze', methods=['POST'])
@login_required
def api_machine_analyze(mid):
    """Single-shot adaptive prediction for a registered machine.
    Body: sensor values (French feature names) + optional `sentence` free-text note."""
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 503
    d = request.json or {}
    sentence = (d.get('sentence') or '').strip() or None
    # Normalise sensor values (French names only)
    sensors = {}
    for k in CORE_FEATURES:
        if k in d and d[k] not in (None, ''):
            try: sensors[k] = float(d[k])
            except (TypeError, ValueError): sensors[k] = None
    if not any(v is not None for v in sensors.values()) and not sentence:
        return jsonify({'error': 'Provide at least one sensor value or a note'}), 400
    # Bounds check
    for fld, (lo, hi) in SENSOR_BOUNDS.items():
        v = sensors.get(fld)
        if v is not None and not (lo <= v <= hi):
            return jsonify({'error': f'Out of range: {fld}={v} (expected [{lo},{hi}])'}), 400
    thr = float(m.threshold) if m.threshold else DEFAULT_THRESHOLD
    res = predict_with_context(m, sensors, sentence=sentence, threshold=thr)
    # Persist the analysis so the machine keeps learning
    try:
        extra = {k: sensors.get(k) for k in ('vibration', 'pression_entree', 'courant_moteur') if sensors.get(k) is not None}
        if sentence:
            extra['_note'] = sentence[:500]
        import json as _json
        _a = Analysis(
            machine_type='pump',
            temp_air=sensors.get('temp_palier'), temp_process=sensors.get('temp_moteur'),
            vitesse=sensors.get('debit'), couple=sensors.get('pression_sortie'),
            usure=sensors.get('heure_fonctionnement'),
            risk=res['probabilite'], prediction=res['prediction'],
            zones=', '.join([z['nom'] for z in (res.get('zones') or [])]),
            mail_sent=False, user_id=uid, machine_id=m.name,
            extra_params=_json.dumps(extra) if extra else None,
            confidence=res.get('confidence', 100))
        db.session.add(_a)
        db.session.commit()
        res['analysis_id'] = _a.id
    except Exception as _e:
        db.session.rollback()
        logger.error(f"analyze persist failed: {_e}")
    # Auto-specialize + recompute baseline when enough fresh data
    try:
        total = Analysis.query.filter(Analysis.machine_id == m.name).count()
        if total > 0 and total % _MIN_SPECIALIZE_SAMPLES == 0:
            threading.Thread(target=_bg_adaptive_retrain, args=(m.id,), daemon=True).start()
    except Exception:
        pass
    return jsonify(res)

def _bg_adaptive_retrain(mid):
    """Background worker: refresh baseline + re-specialize when data accumulates."""
    try:
        with app.app_context():
            compute_machine_baseline(mid)
            specialize_machine_model(mid)
    except Exception as _e:
        logger.error(f"bg_adaptive_retrain[{mid}]: {_e}")

# ── MACHINE MONITOR API ───────────────────────────────────────────────────────
@app.route('/api/machine_monitor/<int:mid>')
@login_required
def api_machine_monitor(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    last_a = Analysis.query.filter_by(machine_id=mid).order_by(Analysis.timestamp.desc()).first()
    if not last_a:
        return jsonify({'has_data': False})
    import json as _json
    zones = []
    try:
        zdata = _json.loads(last_a.zones) if last_a.zones else []
        if isinstance(zdata, list):
            zones = [{'nom': z.get('nom', '?'), 'proba': z.get('proba', 0)} for z in zdata]
    except Exception:
        pass
    return jsonify({
        'has_data': True,
        'risk': round(last_a.risk or 0, 1),
        'prediction': last_a.prediction or 0,
        'timestamp': last_a.timestamp.isoformat() if last_a.timestamp else None,
        'zones': zones,
    })

@app.route('/api/machine_history/<int:mid>')
@login_required
def api_machine_history(mid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    limit = min(int(request.args.get('limit', 50)), 200)
    analyses = Analysis.query.filter_by(machine_id=mid).order_by(Analysis.timestamp.desc()).limit(limit).all()
    import json as _json
    result = []
    for a in analyses:
        zones_str = '—'
        try:
            zdata = _json.loads(a.zones) if a.zones else []
            if isinstance(zdata, list) and zdata:
                zones_str = ' · '.join(z.get('nom', '?') for z in zdata[:3])
        except Exception:
            pass
        result.append({
            'timestamp': a.timestamp.isoformat() if a.timestamp else None,
            'risk': round(a.risk or 0, 1),
            'prediction': a.prediction or 0,
            'zones': zones_str,
            'confidence': None,
        })
    return jsonify({'analyses': result})

# ── MACHINE SPACE PAGE ────────────────────────────────────────────────────────
@app.route('/machine/<int:mid>')
@login_required
def machine_space(mid):
    uid = current_uid()
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    import json as _json
    sf = (SavedFile.query.filter_by(machine_id=m.id, user_id=uid)
          .order_by(SavedFile.created_at.desc()).first())
    machine_json = _json.dumps({
        'id': m.id, 'name': m.name, 'description': m.description or '',
        'pump_type': m.pump_type or 'centrifuge', 'fluid_type': m.fluid_type or 'eau',
        'roue_material': m.roue_material or 'inox_316', 'machine_type': m.machine_type or 'M',
        'threshold': m.threshold or DEFAULT_THRESHOLD, 'location': m.location or '',
        'install_date': m.install_date.isoformat() if m.install_date else '',
        'serial_number': m.serial_number or '',
        'nominal_flow': m.nominal_flow, 'nominal_pressure': m.nominal_pressure,
        'power_kw': m.power_kw, 'nominal_current': m.nominal_current,
        'nominal_vibration': m.nominal_vibration,
        'alert_email': m.alert_email or '', 'escalation_email': m.escalation_email or '',
        'saved_file': ({'id': sf.id, 'filename': sf.filename, 'row_count': sf.row_count,
                        'created_at': sf.created_at.isoformat() + 'Z'} if sf else None),
    })
    return render_template('machine_space.html', machine=m, machine_json=machine_json, sidebar_html=nav("fl"))

@app.route('/dashboard')
@app.route('/machines')
@login_required
def fleet_dashboard():
    r = _paid_required()
    if r: return r
    return render_template('dashboard_main.html')

@app.route('/api/fleet_summary')
@login_required
def api_fleet_summary():
    uid = current_uid()
    machines = Machine.query.filter_by(user_id=uid, is_active=True).order_by(Machine.name).all()
    result = []
    for m in machines:
        last_a = Analysis.query.filter_by(machine_id=m.id).order_by(Analysis.timestamp.desc()).first()
        result.append({
            'id': m.id,
            'name': m.name,
            'location': m.location,
            'last_risk': int(last_a.risk) if last_a and last_a.risk is not None else None,
            'last_analysis': last_a.timestamp.strftime('%d %b %H:%M') if last_a else None,
        })
    return jsonify({'machines': result})

@app.route('/api/machines/<int:mid>/analyze-csv', methods=['POST'])
@login_required
def api_machine_analyze_csv(mid):
    uid = current_uid()
    if _check_api_rate(uid, 'analyze-csv'):
        return jsonify({'error': 'Rate limit exceeded — max 100 requests/min'}), 429
    m = Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    try:
        import io as _io, json as _json
        content = f.read().decode('utf-8', errors='replace')
        # detect delimiter
        for delim in [',', ';', '\t', '|']:
            if delim in content.split('\n')[0]:
                break
        df = pd.read_csv(_io.StringIO(content), sep=delim)
    except Exception as e:
        return jsonify({'error': f'CSV parse error: {e}'}), 400
    # auto-map columns — pump fields (case-insensitive keyword matching)
    col_map = {}
    kw = {
        'vibration':          ['vibration','vib','vibration_mm','vib_mms'],
        'temp_palier':        ['temp_palier','bearing_temp','palier_temp','t_palier','bearing_temperature'],
        'debit':              ['debit','flow','flow_rate','flowrate'],
        'pression_entree':    ['pression_entree','inlet_pressure','pressure_in','suction_pressure'],
        'pression_sortie':    ['pression_sortie','outlet_pressure','discharge_pressure','pressure_out'],
        'courant_moteur':     ['courant_moteur','motor_current','current_motor','courant_a'],
        'temp_moteur':        ['temp_moteur','motor_temp','motor_temperature'],
        'heure_fonctionnement':['heure_fonctionnement','run_hours','operating_hours','runtime','hours'],
    }
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for field, keywords in kw.items():
        for kword in keywords:
            if kword in lower_cols:
                col_map[field] = lower_cols[kword]
                break
    if not col_map:
        return jsonify({'error': 'No recognizable columns found. Expected pump fields: vibration, temp_palier, debit, pression_entree/sortie, courant_moteur, temp_moteur, heure_fonctionnement.'}), 400
    threshold = float(m.threshold) if m.threshold else DEFAULT_THRESHOLD
    results = []
    for _, row in df.iterrows():
        try:
            params = {}
            for field, col in col_map.items():
                v = row[col]
                params[field] = float(v) if pd.notna(v) else None
            if not params:
                continue
            probabilite, prediction, zones_risque, confidence, _ = predict_risk(dict(params), threshold=threshold)
            zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
            a = Analysis(
                machine_type='pump',
                temp_air=params.get('temp_palier'), temp_process=params.get('temp_moteur'),
                vitesse=params.get('debit'), couple=params.get('pression_sortie'),
                usure=params.get('heure_fonctionnement'),
                risk=probabilite, prediction=prediction, zones=zones_str,
                confidence=confidence, user_id=uid, machine_id=m.name)
            db.session.add(a)
            results.append({'risk': probabilite, 'prediction': prediction})
        except Exception:
            continue
    if not results:
        return jsonify({'error': 'No valid rows processed'}), 400
    db.session.commit()
    failures = sum(1 for r in results if r['prediction'] == 1)
    avg_risk = round(sum(r['risk'] for r in results) / len(results), 1)
    max_risk = round(max(r['risk'] for r in results), 1)
    # Sauvegarder le fichier attaché à la machine (remplace l'ancien)
    old = SavedFile.query.filter_by(machine_id=mid, user_id=uid).first()
    if old:
        db.session.delete(old)
    sf = SavedFile(
        user_id=uid, machine_id=mid,
        filename=f.filename or f'upload_{m.name}.csv',
        content=content, row_count=len(results),
    )
    db.session.add(sf)
    db.session.commit()
    return jsonify({
        'ok': True, 'total': len(results), 'failures': failures,
        'avg_risk': avg_risk, 'max_risk': max_risk,
        'machine_name': m.name,
        'file_id': sf.id, 'filename': sf.filename,
        'saved_file': {'id': sf.id, 'filename': sf.filename,
                       'row_count': sf.row_count,
                       'created_at': sf.created_at.isoformat() + 'Z'},
    })

@app.route('/api/machines/<int:mid>/files', methods=['GET'])
@login_required
def api_machine_files(mid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    files = SavedFile.query.filter_by(machine_id=mid, user_id=uid).order_by(SavedFile.created_at.desc()).all()
    return jsonify([{
        'id': f.id, 'filename': f.filename,
        'row_count': f.row_count,
        'created_at': f.created_at.isoformat() + 'Z',
    } for f in files])

@app.route('/api/machines/<int:mid>/files/<int:fid>', methods=['DELETE'])
@login_required
def api_machine_file_delete(mid, fid):
    uid = current_uid()
    Machine.query.filter_by(id=mid, user_id=uid).first_or_404()
    sf = SavedFile.query.filter_by(id=fid, machine_id=mid, user_id=uid).first_or_404()
    db.session.delete(sf)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/machines/analyze-all', methods=['POST'])
@login_required
def api_machines_analyze_all():
    """Lance l'analyse CSV sur toutes les machines qui ont un fichier sauvegardé."""
    uid = current_uid()
    if model is None:
        return jsonify({'error': 'Modele ML non chargé — contactez l\'administrateur'}), 503
    data = request.json or {}
    machine_ids_filter = data.get('machine_ids')
    query = Machine.query.filter_by(user_id=uid, is_active=True)
    if machine_ids_filter:
        query = query.filter(Machine.id.in_(machine_ids_filter))
    machines = query.all()
    results = []
    for m in machines:
        sf = SavedFile.query.filter_by(machine_id=m.id, user_id=uid).order_by(SavedFile.created_at.desc()).first()
        if not sf:
            results.append({'machine': m.name, 'skipped': True, 'reason': 'No file'})
            continue
        try:
            import io as _io
            content = sf.content
            for delim in [',', ';', '\t', '|']:
                if delim in content.split('\n')[0]:
                    break
            df = pd.read_csv(_io.StringIO(content), sep=delim)
            col_map = {}
            kw = {
                'vibration': ['vibration','vib','vibration_mm','vib_mms'],
                'temp_palier': ['temp_palier','bearing_temp','palier_temp','t_palier','bearing_temperature'],
                'debit': ['debit','flow','flow_rate','flowrate'],
                'pression_entree': ['pression_entree','inlet_pressure','pressure_in','suction_pressure'],
                'pression_sortie': ['pression_sortie','outlet_pressure','discharge_pressure','pressure_out'],
                'courant_moteur': ['courant_moteur','motor_current','current_motor','courant_a'],
                'temp_moteur': ['temp_moteur','motor_temp','motor_temperature'],
                'heure_fonctionnement': ['heure_fonctionnement','run_hours','operating_hours','runtime','hours'],
            }
            lower_cols = {c.lower().strip(): c for c in df.columns}
            for field, keywords in kw.items():
                for kword in keywords:
                    if kword in lower_cols:
                        col_map[field] = lower_cols[kword]
                        break
            if not col_map:
                results.append({'machine': m.name, 'skipped': True, 'reason': 'Unrecognized columns'})
                continue
            threshold = float(m.threshold) if m.threshold else DEFAULT_THRESHOLD
            rows_ok = 0
            failures_m = 0
            risks = []
            for _, row in df.iterrows():
                try:
                    params = {}
                    for field, col in col_map.items():
                        v = row[col]
                        params[field] = float(v) if pd.notna(v) else None
                    probabilite, prediction, zones_risque, confidence, _ = predict_risk(dict(params), threshold=threshold)
                    zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
                    a = Analysis(
                        machine_type='pump',
                        temp_air=params.get('temp_palier'), temp_process=params.get('temp_moteur'),
                        vitesse=params.get('debit'), couple=params.get('pression_sortie'),
                        usure=params.get('heure_fonctionnement'),
                        risk=probabilite, prediction=prediction, zones=zones_str,
                        confidence=confidence, user_id=uid, machine_id=m.name)
                    db.session.add(a)
                    rows_ok += 1
                    failures_m += prediction
                    risks.append(probabilite)
                except Exception:
                    continue
            if rows_ok:
                db.session.commit()
            results.append({
                'machine': m.name,
                'skipped': False,
                'total': rows_ok,
                'failures': failures_m,
                'avg_risk': round(sum(risks)/len(risks), 1) if risks else 0,
                'max_risk': round(max(risks), 1) if risks else 0,
                'file': sf.filename,
            })
        except Exception as e:
            results.append({'machine': m.name, 'skipped': True, 'reason': str(e)})
    return jsonify({'ok': True, 'results': results})

@app.route('/api/machine-request', methods=['POST'])
@login_required
def api_machine_request():
    uid = current_uid()
    d = request.json or {}
    name = (d.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    mr = MachineRequest(user_id=uid, name=name,
        manufacturer=(d.get('manufacturer') or '').strip(),
        rpm_range=(d.get('rpm_range') or '').strip(),
        torque_range=(d.get('torque_range') or '').strip(),
        description=(d.get('description') or '').strip())
    db.session.add(mr)
    db.session.commit()
    # Email admin
    if GMAIL and GMAIL_PWD:
        def _notify():
            user = db.session.get(User, uid)
            user_email = user.email if user else 'unknown'
            html = (f'<b>New machine request from {user_email}</b><br><br>'
                    f'<b>Machine:</b> {name}<br>'
                    f'<b>Manufacturer:</b> {mr.manufacturer or "—"}<br>'
                    f'<b>RPM range:</b> {mr.rpm_range or "—"}<br>'
                    f'<b>Torque range:</b> {mr.torque_range or "—"}<br>'
                    f'<b>Description:</b> {mr.description or "—"}<br>')
            msg = MIMEMultipart('alternative')
            _admin_email = os.environ.get("ADMIN_EMAIL", "")
            msg['Subject'] = f'Pilar — Machine Request: {name}'
            msg['From'] = f'Pilar <{GMAIL}>'
            msg['To'] = _admin_email
            msg.attach(MIMEText(html, 'html'))
            try:
                if not _admin_email:
                    return
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(GMAIL, GMAIL_PWD)
                    smtp.sendmail(GMAIL, _admin_email, msg.as_string())
            except Exception as _e:
                logger.error(f"machine-request: email error: {_e}")
        with app.app_context():
            threading.Thread(target=_notify, daemon=True).start()
    return jsonify({'ok': True, 'id': mr.id})

@app.route('/onboarding')
@login_required
def onboarding():
    return redirect('/monitor')

@app.route('/onboarding/analyse', methods=['POST'])
@login_required
def onboarding_analyse():
    try:
        data = request.json or {}
        # Accept at least one pump field
        if not any(k in data for k in CORE_FEATURES):
            return jsonify({'error': 'Missing parameters'}), 400
        prob, pred, zones, conf, _ = predict_risk(data)
        uid = current_uid()
        a = Analysis(user_id=uid, machine_type='pump',
                     temp_air=data.get('temp_palier'), temp_process=data.get('temp_moteur'),
                     vitesse=data.get('debit'), couple=data.get('pression_sortie'),
                     usure=data.get('heure_fonctionnement'),
                     risk=prob, prediction=pred, confidence=conf)
        db.session.add(a)
        db.session.commit()
        verdict = ('High risk — intervention recommended before next operation cycle.' if prob >= 50
                   else 'Moderate risk — schedule inspection within 48 hours.' if prob >= 20
                   else 'All systems nominal. Continue monitoring normally.')
        return jsonify({'risk': prob, 'pred': pred, 'zones': zones, 'verdict': verdict})
    except Exception as e:
        logger.info(f"onboarding] ERROR: {type(e).__name__}: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/onboarding/complete', methods=['POST'])
@login_required
def onboarding_complete():
    uid = current_uid()
    user = db.session.get(User, uid)
    if user:
        user.onboarded = True
        db.session.commit()
    return jsonify({'ok': True})

@app.route('/onboarding/skip')
@login_required
def onboarding_skip():
    uid = current_uid()
    user = db.session.get(User, uid)
    if user:
        user.onboarded = True
        db.session.commit()
    return redirect('/monitor')

@app.route('/monitor')
def monitor():
    r = _paid_required()
    if r: return r
    import json as _json
    _last = session.get('last_result')
    _last_json = _json.dumps(_last) if _last else 'null'
    return render_template('dashboard.html', last_result=_last_json)

@app.route('/account')
def account():
    uid = current_uid()
    user = db.session.get(User, uid) if uid else None
    team = None
    members = []
    my_role = None
    if user and user.team_id:
        team = Team.query.get(user.team_id)
        if team:
            my_mbr = TeamMember.query.filter_by(team_id=team.id, user_id=uid).first()
            if my_mbr and not my_mbr.is_kicked:
                my_role = my_mbr.role
                for m in TeamMember.query.filter_by(team_id=team.id, is_kicked=False).all():
                    mu = db.session.get(User, m.user_id)
                    if mu:
                        members.append({'id': mu.id, 'email': mu.email, 'role': m.role})
            else:
                user.team_id = None
                db.session.commit()
                team = None
    return render_template('account.html', user=user, team=team, members=members, my_role=my_role)

def _paid_required():
    """All installed users have an active subscription — always allowed."""
    uid = current_uid()
    if not uid:
        return redirect('/login')
    user = db.session.get(User, uid)
    if not user:
        session.clear()
        return redirect('/login')
    return None

@app.route('/upgrade')
def upgrade():
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Get Access — Pilar</title>
<style>:root{--bg:#07090f;--surface:#0e1118;--border:#1e2433;--border2:#252d3d;--teal:#0d9488;--teal2:#14b8a6;--teal-dim:rgba(13,148,136,0.08);--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;--green:#059669;--r:12px}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,'SF Pro Display','SF Pro Text','Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;text-align:center}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:40px 32px;max-width:440px;width:100%}
h2{font-size:22px;font-weight:700;letter-spacing:-0.03em;margin-bottom:10px}
.sub{font-size:13px;color:var(--text2);line-height:1.8;margin-bottom:28px}
.features{list-style:none;margin-bottom:28px;text-align:left}
.features li{font-size:13px;color:var(--text2);padding:7px 0;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.features li:last-child{border:none}
.features li::before{content:'\2713';color:var(--green);font-weight:700;flex-shrink:0}
.badge{display:inline-block;padding:4px 12px;background:var(--teal-dim);border:1px solid rgba(13,148,136,.25);border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0;color:var(--teal2);margin-bottom:20px}
.btn{display:block;width:100%;padding:15px;background:var(--teal);color:#fff;border:none;border-radius:var(--r);font-size:15px;font-weight:600;letter-spacing:0;text-transform:none;cursor:pointer;text-decoration:none;margin-bottom:10px;transition:opacity .15s}
.btn:hover{opacity:.9}
.btn-ghost{display:block;width:100%;padding:12px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:var(--r);font-size:13px;cursor:pointer;text-decoration:none}
.divider{border:none;border-top:1px solid var(--border);margin:24px 0}
</style></head><body>
<div class="card">
<div style="width:52px;height:52px;border-radius:14px;background:var(--teal-dim);border:1px solid rgba(13,148,136,.2);display:flex;align-items:center;justify-content:center;margin:0 auto 18px">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
</div>
<div class="badge">Custom Plan</div>
<h2>Get Full Access to Pilar</h2>
<p class="sub">Your free account includes manual analysis. Unlock the full platform with a custom contract built around your operations.</p>
<ul class="features">
<li>Live Monitor (real-time sensor stream)</li>
<li>Full analysis history &amp; Digital Twin</li>
<li>AI maintenance assistant (Claude)</li>
<li>REST API access</li>
<li>Email alerts with detailed reports</li>
<li>Team collaboration</li>
<li>Custom sensor variables</li>
<li>Dedicated onboarding &amp; support</li>
</ul>
<a href="https://github.com/CYPHR007/PILAR/issues" class="btn">&#128231; Contact us to get access</a>
<hr class="divider">
<a href="/monitor" class="btn-ghost">Back to Monitor</a>
</div>
</body></html>"""
    return html

@app.route('/tutorial')
@login_required
def tutorial(): return render_template('tutorial.html')

@app.route('/adapter')
def adapter():
    r = _paid_required()
    if r: return r
    return render_template('adapter.html')

@app.route('/api/save_csv', methods=['POST'])
@auth_optional
def save_csv_file():
    uid = current_uid()
    if not uid:
        return jsonify({'error': 'login_required'}), 401
    try:
        data = request.json or {}
        filename = (data.get('filename') or 'imported.csv').strip()[:200]
        content = data.get('content', '')
        if not content:
            return jsonify({'error': 'empty'}), 400
        row_count = max(0, content.count('\n') - 1)
        user = db.session.get(User, uid)
        team_id = user.team_id if user else None
        sf = SavedFile(user_id=uid, team_id=team_id, filename=filename, content=content, row_count=row_count)
        db.session.add(sf)
        db.session.commit()
        return jsonify({'id': sf.id, 'filename': sf.filename, 'rows': row_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/saved_files')
@auth_optional
def list_saved_files():
    uid = current_uid()
    if not uid:
        return jsonify([])
    user = db.session.get(User, uid)
    if user and user.team_id:
        files = SavedFile.query.filter(
            (SavedFile.team_id == user.team_id) | (SavedFile.user_id == uid)
        ).order_by(SavedFile.created_at.desc()).limit(50).all()
    else:
        files = SavedFile.query.filter_by(user_id=uid).order_by(SavedFile.created_at.desc()).limit(20).all()
    return jsonify([{
        'id': f.id, 'filename': f.filename, 'rows': f.row_count,
        'created_at': f.created_at.isoformat(),
        'owner': f.user_id == uid
    } for f in files])

@app.route('/api/saved_files/<int:fid>')
@auth_optional
def get_saved_file(fid):
    uid = current_uid()
    user = db.session.get(User, uid) if uid else None
    if user and user.team_id:
        f = SavedFile.query.filter(
            SavedFile.id == fid,
            (SavedFile.team_id == user.team_id) | (SavedFile.user_id == uid)
        ).first_or_404()
    else:
        f = SavedFile.query.filter_by(id=fid, user_id=uid).first_or_404()
    return jsonify({'id': f.id, 'filename': f.filename, 'content': f.content, 'rows': f.row_count})

@app.route('/api/saved_files/<int:fid>/delete', methods=['POST'])
@auth_optional
def delete_saved_file(fid):
    uid = current_uid()
    user = db.session.get(User, uid) if uid else None
    f = SavedFile.query.filter_by(id=fid).first_or_404()
    # Seul le propriétaire ou le leader de la team peut supprimer
    is_owner = f.user_id == uid
    is_leader = False
    if user and user.team_id and f.team_id == user.team_id:
        mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=uid, is_kicked=False).first()
        is_leader = mbr and mbr.role == 'leader'
    if not is_owner and not is_leader:
        return jsonify({'error': 'Forbidden'}), 403
    db.session.delete(f)
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/admin/files')
@admin_required
def admin_list_files():
    rows = db.session.query(SavedFile, User.email)\
        .outerjoin(User, SavedFile.user_id == User.id)\
        .order_by(SavedFile.created_at.desc()).all()
    return jsonify([{
        'id': f.id, 'filename': f.filename, 'rows': f.row_count,
        'created_at': f.created_at.isoformat(),
        'user_email': email or 'unknown'
    } for f, email in rows])

@app.route('/admin/files/<int:fid>/download')
@admin_required
def admin_download_file(fid):
    from flask import Response
    f = SavedFile.query.get_or_404(fid)
    return Response(
        f.content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{f.filename}"'}
    )

@app.route('/twin')
def twin():
    r = _paid_required()
    if r: return r
    import json as _json
    _last = session.get('last_result')
    _last_json = _json.dumps(_last) if _last else 'null'
    return render_template('twin.html', last_result=_last_json)

@app.route('/history')
def history():
    r = _paid_required()
    if r: return r
    uid = current_uid()
    PAGE_SIZE = 50
    page = max(1, request.args.get('page', 1, type=int))
    # Load only 4 lightweight columns for stats (no text blobs)
    stat_rows = (Analysis.query.filter_by(user_id=uid)
                 .with_entities(Analysis.prediction, Analysis.risk,
                                Analysis.mail_sent, Analysis.feedback)
                 .all())
    total     = len(stat_rows)
    anomalies = sum(1 for r in stat_rows if r.prediction)
    mails     = sum(1 for r in stat_rows if r.mail_sent)
    avg_risk  = round(sum(r.risk or 0 for r in stat_rows) / total, 1) if total else 0
    labeled   = [r for r in stat_rows if r.feedback in ('tp', 'fp', 'fn')]
    reliability = round(sum(1 for r in labeled if r.feedback in ('tp', 'fn')) / len(labeled) * 100) if labeled else None
    # Paginated full rows for display
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    analyses = (Analysis.query.filter_by(user_id=uid)
                .order_by(Analysis.timestamp.desc())
                .limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE).all())
    return render_template('history.html', analyses=analyses, total=total,
                                   anomalies=anomalies, avg_risk=avg_risk, mails=mails,
                                   reliability=reliability, page=page, total_pages=total_pages)

@app.route('/analysis/<int:aid>/feedback', methods=['POST'])
@login_required
def analysis_feedback(aid):
    uid = current_uid()
    a = Analysis.query.filter_by(id=aid, user_id=uid).first_or_404()
    fb = (request.json or {}).get('feedback')
    if fb not in ('tp', 'fp', 'fn', None):
        return jsonify({'error': 'Invalid feedback value'}), 400
    a.feedback = fb
    # fn = missed failure — treat as failure for retraining
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/settings')
def settings():
    r = _paid_required()
    if r: return r
    return render_template('settings.html')

@app.route('/set_email', methods=['POST'])
@login_required
def set_email():
    import re as _re
    email = (request.json or {}).get('email', '').strip()[:200]
    if email and not _re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return jsonify({'status': 'error', 'message': 'Email invalide'}), 400
    set_setting('responsible_email', email)
    return jsonify({'status': 'ok'})

@app.route('/predire', methods=['POST'])
def predire():
    try:
        _uid = session.get('user_id', request.remote_addr)
        if _check_api_rate(_uid, 'predire'):
            return jsonify({'error': 'Rate limit exceeded — max 100 requests/min'}), 429
        data = request.json
        if not data:
            return jsonify({'error': 'Données manquantes (JSON invalide)'}), 400
        if model is None:
            return jsonify({'error': 'Modele ML non chargé — contactez l\'administrateur'}), 503
        # Champs core optionnels — None → imputé par predict_risk
        for field in CORE_FEATURES:
            if field in data and data[field] is not None:
                try:
                    data[field] = float(data[field])
                except (TypeError, ValueError):
                    data[field] = None
            else:
                data[field] = None
        if all(data.get(f) is None for f in CORE_FEATURES):
            return jsonify({'error': 'Au moins un paramètre requis'}), 400
        # Bornes physiques des capteurs pompe
        for fld, (lo, hi) in SENSOR_BOUNDS.items():
            v = data.get(fld)
            if v is not None and not (lo <= v <= hi):
                return jsonify({'error': f'Valeur hors limites : {fld} ({v}) — attendu [{lo},{hi}]'}), 400
        # Champs optionnels connus
        import json as _json
        extra_params = {}
        for field in OPTIONAL_FIELDS:
            if field in data and data[field] is not None:
                try:
                    extra_params[field] = round(float(data[field]), 3)
                except (TypeError, ValueError):
                    pass
        # Resolve machine and custom threshold
        uid = current_uid()
        machine_id_str = (data.get('machine_id') or '').strip() or None
        _machine = Machine.query.filter_by(user_id=uid, name=machine_id_str, is_active=True).first() if (uid and machine_id_str) else None
        _threshold = float(_machine.threshold) if (_machine and _machine.threshold) else DEFAULT_THRESHOLD
        _machine_context = None
        if _machine:
            _machine_context = {
                'nominal_flow':      _machine.nominal_flow,
                'nominal_pressure':  _machine.nominal_pressure,
                'nominal_current':   _machine.nominal_current,
                'nominal_vibration': _machine.nominal_vibration,
                'power_kw':          _machine.power_kw,
            }
        probabilite, prediction, zones_risque, confidence, imputed, anomaly_score, shap_explanations, rul_hours = predict_risk(data, threshold=_threshold, return_extra=True, machine_context=_machine_context)

        # Free-text operator note — parsed for risk signals (Ollama → keywords fallback)
        _sentence = (data.get('sentence') or data.get('note') or '').strip() or None
        _sentence_signals = None
        _context_factors  = None
        _anomaly_scores   = {}
        _model_type       = 'global'
        _model_f1         = None
        _model_version    = None
        _base_probabilite = probabilite

        if _machine:
            # Blend specialized classifier probability when available
            try:
                _mdl, _scl, _is_spec = get_machine_model(_machine)
                if _is_spec and _mdl is not None and _scl is not None:
                    import pilar_ml as _pm
                    _df = _pm.build_model_input(_params_from_analysis_like(data))
                    _Xs = _scl.transform(_df.values)
                    try:    _spec_p = float(_mdl.predict_proba(_Xs)[0][1]) * 100.0
                    except Exception: _spec_p = float(_mdl.predict(_Xs)[0]) * 100.0
                    _mm = MachineModel.query.filter_by(machine_id=_machine.id).first()
                    _w = 0.7 if (_mm and (_mm.f1_score or 0) >= 0.85) else 0.55
                    probabilite = round((_spec_p * _w) + (probabilite * (1.0 - _w)), 1)
                    _model_type    = 'specialized'
                    _model_f1      = round(_mm.f1_score, 3) if (_mm and _mm.f1_score) else None
                    _model_version = _mm.version if _mm else None
            except Exception as _be:
                logger.debug(f"predire: specialized blend skipped — {_be}")
            _anomaly_scores = compute_anomaly_scores(_machine, data) or {}
            _age_f  = max(1.0, min(float(_machine.age_years or 0.0) / 10.0, 2.0)) or 1.0
            _env_f  = _ENV_RISK_FACTOR.get((_machine.environment or 'general').lower(), 1.0)
            _crit_f = _CRIT_RISK_FACTOR.get((_machine.criticality or 'medium').lower(), 1.0)
        else:
            _age_f = _env_f = _crit_f = 1.0
        _sentence_signals = extract_sentence_signals(_sentence) if _sentence else None
        _sent_f = float(_sentence_signals.get('risk_multiplier', 1.0)) if _sentence_signals else 1.0
        _total_mult = _age_f * _env_f * _crit_f * _sent_f
        if _machine or _sentence_signals:
            probabilite = round(min(100.0, probabilite * _total_mult), 1)
            prediction  = 1 if probabilite >= _threshold else 0
            _context_factors = {
                'age_factor':         round(_age_f, 3),
                'environment_factor': round(_env_f, 3),
                'criticality_factor': round(_crit_f, 3),
                'sentence_factor':    round(_sent_f, 3),
                'total_multiplier':   round(_total_mult, 3),
            }

        # ── Domain knowledge corrections ──────────────────────────────────────
        # Resolve pump/fluid/material from machine record OR from request payload
        _pump_type    = ((_machine.pump_type    if _machine else None) or data.get('pump_type')    or 'centrifuge').strip()
        _fluid_type   = ((_machine.fluid_type   if _machine else None) or data.get('fluid_type')   or 'eau').strip()
        _roue_mat     = ((_machine.roue_material if _machine else None) or data.get('roue_material') or 'inox_316').strip()
        _domain_warnings = []

        # Non-centrifuge: model trained only on centrifuge data → warn
        if _pump_type in NON_CENTRIFUGE_TYPES:
            _domain_warnings.append(f'Model trained on centrifugal pumps — results for {_pump_type} are indicative only.')

        # Apply zone threshold adjustments based on fluid aggressiveness
        _zone_adj = FLUID_ZONE_SENSITIVITY.get(_fluid_type, {})
        if _zone_adj and zones_risque:
            _zone_code_map = {v: k for k, v in FAILURE_ZONES.items()}
            for z in zones_risque:
                _code = _zone_code_map.get(z['nom'])
                if _code and _code in _zone_adj:
                    z['proba'] = round(min(100.0, max(0.0, z['proba'] + _zone_adj[_code])), 1)
            zones_risque = [z for z in zones_risque if z['proba'] >= 10]
            zones_risque.sort(key=lambda x: x['proba'], reverse=True)

        # Apply RUL fluid × material scaling factor
        if rul_hours is not None:
            _f_factor = FLUID_RUL_FACTORS.get(_fluid_type, 1.0)
            _m_factor = MATERIAL_RUL_FACTORS.get(_roue_mat, 1.0)
            rul_hours = round(rul_hours * _f_factor * _m_factor, 1)

        mail_envoye = False
        email = (_machine.alert_email if _machine and _machine.alert_email else None) or get_setting('responsible_email')
        alert_threshold = _threshold  # alert fires at same threshold as prediction
        if probabilite >= alert_threshold and email:
            import secrets as _sec
            ack_tok = _sec.token_hex(24)
            esc_email = _machine.escalation_email if _machine else None
            threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data, ack_tok), daemon=True).start()
            mail_envoye = True
            _al = AlertLog(user_id=uid, machine_id_str=machine_id_str,
                email_to=email, probabilite=probabilite,
                ack_token=ack_tok, escalation_email=esc_email)
            db.session.add(_al)
        zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
        if _sentence:
            extra_params['_note'] = _sentence[:500]
        extra_json = _json.dumps(extra_params) if extra_params else None
        _a = Analysis(machine_type='pump',
            temp_air=data.get('temp_palier'), temp_process=data.get('temp_moteur'),
            vitesse=data.get('debit'), couple=data.get('pression_sortie'), usure=data.get('heure_fonctionnement'),
            risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye,
            extra_params=extra_json, confidence=confidence, user_id=uid,
            machine_id=machine_id_str)
        db.session.add(_a)
        db.session.commit()
        # Trigger auto-retrain when enough new data has accumulated
        global _new_analyses_since_retrain
        _new_analyses_since_retrain += 1
        if _new_analyses_since_retrain >= RETRAIN_TRIGGER:
            threading.Thread(target=_auto_retrain, daemon=True).start()
        # ── PILAR Agents (background thread — non-blocking) ───────────────
        # Map zone nom back to code for agents
        _zone_code_map = {v: k for k, v in FAILURE_ZONES.items()}
        _top_zone = _zone_code_map.get(zones_risque[0]['nom'], '') if zones_risque else ''
        _machine_display = (_machine.nom if _machine else machine_id_str) or 'Machine'
        _ml_result_for_agents = {
            'risk_score': probabilite,
            'zone': _top_zone,
            'rul': rul_hours,
        }
        _prev_risk = 0
        try:
            _prev = Analysis.query.filter_by(user_id=uid, machine_id=machine_id_str)\
                        .order_by(Analysis.timestamp.desc()).offset(1).first()
            if _prev:
                _prev_risk = _prev.risk or 0
        except Exception:
            pass
        def _run_agents_bg(mname, sdata, mlres, prev):
            try:
                from agents.orchestrator import run_agents
                run_agents(mname, sdata, mlres, previous_risk=prev)
            except Exception as _ae:
                logger.error(f"Agent error: {_ae}")
        threading.Thread(
            target=_run_agents_bg,
            args=(_machine_display, data, _ml_result_for_agents, _prev_risk),
            daemon=True
        ).start()
        # ─────────────────────────────────────────────────────────────────

        _result = {'prediction': prediction, 'probabilite': probabilite,
                   'zones': zones_risque, 'mail_envoye': mail_envoye,
                   'confidence': confidence, 'imputed': imputed,
                   'anomaly_score': anomaly_score,
                   'shap_explanations': shap_explanations,
                   'rul_hours': rul_hours,
                   'domain_warnings': _domain_warnings,
                   'base_probabilite':  _base_probabilite,
                   'model_type':        _model_type,
                   'model_f1':          _model_f1,
                   'model_version':     _model_version,
                   'anomaly_scores':    _anomaly_scores,
                   'context_factors':   _context_factors,
                   'sentence_signals':  _sentence_signals,
                   'last_params': {k: data.get(k) for k in CORE_FEATURES}}
        try:
            session['last_result'] = _result
            session.modified = True
        except Exception:
            pass
        return jsonify(_result)
    except Exception as e:
        db.session.rollback()
        import traceback
        logger.error(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Erreur interne — réessayez ou contactez le support'}), 500

@app.route('/api/sla')
@api_or_login_required
def api_sla_dashboard():
    """SLA dashboard — returns current status for all agents and components."""
    try:
        from agents.orchestrator import get_sla_dashboard, get_sla_history
        dashboard = get_sla_dashboard()
        history   = get_sla_history(hours=24)
        # Compute breach count
        breaches = sum(1 for r in dashboard if r.get('status') == 'breach')
        warnings = sum(1 for r in dashboard if r.get('status') == 'warn')
        return jsonify({
            'ok': True,
            'components': dashboard,
            'history_24h': history[-100:],   # last 100 events
            'summary': {
                'total':    len(dashboard),
                'ok':       sum(1 for r in dashboard if r.get('status') == 'ok'),
                'warn':     warnings,
                'breach':   breaches,
                'health':   'ok' if breaches == 0 and warnings == 0
                            else ('warn' if breaches == 0 else 'breach'),
            }
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/sla/history')
@api_or_login_required
def api_sla_history():
    """SLA event history for a specific component."""
    component = request.args.get('component')
    hours     = int(request.args.get('hours', 24))
    try:
        from agents.orchestrator import get_sla_history
        history = get_sla_history(component=component, hours=hours)
        return jsonify({'ok': True, 'events': history})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/agents/conversations')
@api_or_login_required
def api_agent_conversations():
    """Latest agent conversations for the dashboard."""
    try:
        from agents.sla_tracker import get_conversations
        limit = int(request.args.get('limit', 20))
        return jsonify({'ok': True, 'conversations': get_conversations(limit)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500




@app.route('/agents')
@login_required
def agents_dashboard():
    """Agent & SLA monitoring dashboard."""
    return render_template('agents_dashboard.html')


@app.route('/api/twin')
@api_or_login_required
def api_twin():
    try:
        uid = current_uid()
        analyses = Analysis.query.filter_by(user_id=uid).order_by(Analysis.timestamp.asc()).all()
        if not analyses: return jsonify({'has_data': False})
        last = analyses[-1]
        history_times = [a.timestamp.strftime('%H:%M') for a in analyses]
        history_risks = [a.risk for a in analyses]
        history_wear  = [a.usure for a in analyses]          # heure_fonctionnement
        history_temp  = [a.temp_air for a in analyses]       # temp_palier (bearing temp)
        future_times, future_risks, future_wear, future_temp = [], [], [], []
        now = datetime.now(timezone.utc)
        # Simulate degradation: run hours increase, bearing temp slowly rises
        chf = last.usure or FEATURE_MEDIANS['heure_fonctionnement']
        ctp = last.temp_air or FEATURE_MEDIANS['temp_palier']
        failure_hours = None
        for h in range(1, 25):
            chf = chf + 1.0
            ctp = min(ctp + 0.1, 150.0)
            _p = {
                'vibration': FEATURE_MEDIANS['vibration'],
                'temp_palier': ctp,
                'debit': last.vitesse or FEATURE_MEDIANS['debit'],
                'pression_entree': FEATURE_MEDIANS['pression_entree'],
                'pression_sortie': last.couple or FEATURE_MEDIANS['pression_sortie'],
                'courant_moteur': FEATURE_MEDIANS['courant_moteur'],
                'temp_moteur': last.temp_process or FEATURE_MEDIANS['temp_moteur'],
                'heure_fonctionnement': chf,
            }
            risk, pred, _, _, _ = predict_risk(_p)
            future_times.append((now + timedelta(hours=h)).strftime('%H:%M'))
            future_risks.append(risk); future_wear.append(round(chf,1)); future_temp.append(round(ctp,1))
            if failure_hours is None and risk >= 50: failure_hours = h
        total = len(analyses)
        avg_risk = round(sum(a.risk for a in analyses) / total, 1)
        anomaly_rate = round(sum(1 for a in analyses if a.prediction) / total * 100, 1)
        trend = 'Stable'
        if len(history_risks) >= 3:
            diff = history_risks[-1] - history_risks[-3]
            trend = 'Increasing' if diff > 2 else 'Decreasing' if diff < -2 else 'Stable'
        # ── RUL: Multi-signal Health Index + polynomial regression (degree 2) ──
        rul_hours = None
        rul_confidence = None
        rul_degradation_rates = {}
        try:
            if len(analyses) >= 3:
                import json as _json2
                t_ref = analyses[0].timestamp
                t_arr = np.array([(a.timestamp - t_ref).total_seconds() / 3600 for a in analyses], dtype=float)
                r_arr = np.array([a.risk for a in analyses], dtype=float)
                # bearing temp degradation component (0=healthy 44°C → 1=critical 70°C)
                t_bear = np.array([a.temp_air if a.temp_air is not None else FEATURE_MEDIANS['temp_palier'] for a in analyses], dtype=float)
                bear_deg = np.clip((t_bear - 44.5) / 25.5, 0.0, 1.0)
                # vibration from extra_params if stored
                vib_vals = []
                for a in analyses:
                    try:
                        ep = _json2.loads(a.extra_params) if a.extra_params else {}
                        vib_vals.append(float(ep.get('vibration', FEATURE_MEDIANS['vibration'])))
                    except Exception:
                        vib_vals.append(FEATURE_MEDIANS['vibration'])
                vib_arr = np.array(vib_vals, dtype=float)
                vib_deg = np.clip((vib_arr - 0.6) / 1.4, 0.0, 1.0)  # 0.6=ok → 2.0=critical
                # Composite Health Index: weighted combination of risk + bearing temp + vibration
                hi_arr = np.clip(0.5 * (r_arr / 100.0) + 0.3 * bear_deg + 0.2 * vib_deg, 0.0, 1.0) * 100.0
                # Polynomial fit (degree 2 if ≥5 points, linear if fewer)
                deg = min(2, len(t_arr) - 1)
                coeffs = np.polyfit(t_arr, hi_arr, deg)
                hi_pred = np.polyval(coeffs, t_arr)
                ss_res = ((hi_arr - hi_pred) ** 2).sum()
                ss_tot = ((hi_arr - hi_arr.mean()) ** 2).sum()
                r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                # Find time when HI hits 80 (failure threshold)
                threshold = 80.0
                c_shifted = coeffs.copy(); c_shifted[-1] -= threshold
                if deg == 2:
                    roots = np.roots(c_shifted)
                    future_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > t_arr[-1]]
                else:  # linear: slope*t + intercept = 80
                    future_roots = [(-c_shifted[1] / c_shifted[0])] if c_shifted[0] != 0 else []
                    future_roots = [x for x in future_roots if x > t_arr[-1]]
                if future_roots and hi_arr[-1] < threshold:
                    t_fail = min(future_roots)
                    rul_hours = max(0, round(t_fail - t_arr[-1]))
                    rul_confidence = 'high' if r2 > 0.7 else 'medium' if r2 > 0.3 else 'low'
                # Per-feature degradation rates (unit/hour)
                span = max(t_arr[-1] - t_arr[0], 1e-9)
                rul_degradation_rates = {
                    'risk_per_h': round(float(r_arr[-1] - r_arr[0]) / span, 4),
                    'bearing_temp_per_h': round(float(t_bear[-1] - t_bear[0]) / span, 4),
                    'vibration_per_h': round(float(vib_arr[-1] - vib_arr[0]) / span, 4),
                    'hi_current': round(float(hi_arr[-1]), 1),
                    'hi_slope_per_h': round(float(np.polyval(np.polyder(coeffs), t_arr[-1])), 4),
                }
        except Exception as _re:
            logger.info(f"RUL] {_re}")
        return jsonify({'has_data':True,'current_risk':last.risk,'avg_risk_24h':avg_risk,
            'anomaly_rate':anomaly_rate,'total_analyses':total,'failure_hours':failure_hours,'trend':trend,
            'history_times':history_times,'history_risks':history_risks,'history_wear':history_wear,'history_temp':history_temp,
            'future_times':future_times,'future_risks':future_risks,'future_wear':future_wear,'future_temp':future_temp,
            'rul_hours':rul_hours,'rul_confidence':rul_confidence,'rul_degradation':rul_degradation_rates,
            'last_params':{'vibration':FEATURE_MEDIANS['vibration'],'debit':last.vitesse,'pression_sortie':last.couple,'heure_fonctionnement':last.usure,'temp_palier':last.temp_air}})
    except Exception as e:
        import traceback
        logger.info(f"api_twin] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Erreur serveur'}), 500

@app.route('/api/update_info')
def api_update_info():
    return jsonify(_UPDATE_INFO)

@app.route('/api/update/check', methods=['GET'])
def api_update_check():
    """Manual update check — re-queries GitHub and returns result."""
    _check_update_background()
    APP_VER = os.environ.get('PILAR_VERSION', APP_VERSION)
    return jsonify({
        'current': APP_VER,
        'available': _UPDATE_INFO['available'],
        'latest': _UPDATE_INFO.get('version'),
        'download_url': _UPDATE_INFO.get('download_url'),
    })

@app.route('/api/update/install', methods=['POST'])
def api_update_install():
    """
    Download the latest release zip, extract to %LOCALAPPDATA%\\PILAR_update,
    launch the new PILAR.exe, then exit this process.
    Only works in the frozen desktop app (_FROZEN=True).
    """
    import urllib.request as _ur, zipfile as _zf, shutil as _sh, tempfile as _tmp
    url = _UPDATE_INFO.get('download_url', '')
    version = _UPDATE_INFO.get('version', 'unknown')
    if not _UPDATE_INFO.get('available') or not url:
        return jsonify({'error': 'No update available'}), 400

    def _do_install():
        try:
            import urllib.request, zipfile, shutil, os as _os, tempfile, sys as _sys
            # Download to temp file
            suffix = '.zip' if url.endswith('.zip') else '.exe'
            tmp_path = tempfile.mktemp(suffix=suffix, prefix=f'PILAR_{version}_')
            logger.info(f"update: Downloading {url} → {tmp_path}")
            urllib.request.urlretrieve(url, tmp_path)
            logger.info("update: Download complete")

            if suffix == '.zip':
                update_dir = _os.path.join(
                    _os.environ.get('LOCALAPPDATA', tempfile.gettempdir()), 'PILAR_update')
                if _os.path.exists(update_dir):
                    shutil.rmtree(update_dir)
                _os.makedirs(update_dir, exist_ok=True)
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    z.extractall(update_dir)
                _os.unlink(tmp_path)
                # Find PILAR.exe in extracted tree
                pilar_exe = None
                for root, dirs, files in _os.walk(update_dir):
                    for fname in files:
                        if fname.lower() == 'pilar.exe':
                            pilar_exe = _os.path.join(root, fname)
                            break
                    if pilar_exe:
                        break
                if not pilar_exe:
                    raise FileNotFoundError('PILAR.exe not found in archive')
                import subprocess
                subprocess.Popen([pilar_exe], cwd=_os.path.dirname(pilar_exe), close_fds=True)
                logger.info(f"update: Launched {pilar_exe} — exiting")
            else:
                import subprocess
                subprocess.Popen([tmp_path], close_fds=True)
                logger.info("update: Installer launched — exiting")

            import time
            time.sleep(1.5)
            _os._exit(0)
        except Exception as _e:
            logger.info(f"update: Install failed: {_e}")

    threading.Thread(target=_do_install, daemon=True, name='in-app-updater').start()
    return jsonify({'ok': True, 'message': 'Téléchargement en cours…'})

# ── SERVER-SIDE FILE PERSISTENCE ─────────────────────────────────────────────
# Saves uploaded CSVs to disk so they survive page navigation and app restarts.
# No localStorage size limit — works for any CSV size.
_SESSION_FILE_DIR = os.path.join(_DATA_DIR, 'session_files')
try:
    os.makedirs(_SESSION_FILE_DIR, exist_ok=True)
except Exception:
    pass

def _session_slot_paths(slot):
    safe = ''.join(c for c in slot if c.isalnum() or c == '_')[:20] or 'default'
    uid = str(current_uid() or 'guest')
    d = os.path.join(_SESSION_FILE_DIR, uid)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{safe}.csv'), os.path.join(d, f'{safe}.fname')

@app.route('/api/session_file/save', methods=['POST'])
def session_file_save():
    import json as _j
    data = request.json or {}
    slot  = data.get('slot', 'default')
    fname = data.get('fname', '')
    csv   = data.get('csv', '')
    try:
        csv_path, fname_path = _session_slot_paths(slot)
        with open(csv_path,   'w', encoding='utf-8') as f: f.write(csv)
        with open(fname_path, 'w', encoding='utf-8') as f: f.write(fname)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

@app.route('/api/session_file/load')
def session_file_load():
    slot = request.args.get('slot', 'default')
    try:
        csv_path, fname_path = _session_slot_paths(slot)
        with open(csv_path,   encoding='utf-8') as f: csv   = f.read()
        with open(fname_path, encoding='utf-8') as f: fname = f.read()
        return jsonify(ok=True, csv=csv, fname=fname)
    except FileNotFoundError:
        return jsonify(ok=False)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

@app.route('/api/session_file/clear', methods=['POST'])
def session_file_clear():
    slot = (request.json or {}).get('slot', 'default')
    try:
        csv_path, fname_path = _session_slot_paths(slot)
        for p in (csv_path, fname_path):
            try: os.remove(p)
            except FileNotFoundError: pass
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

# ── END SERVER-SIDE FILE PERSISTENCE ─────────────────────────────────────────

@app.route('/api/health')
def api_health():
    import os, sys, json as _json
    meta = {}
    scheduler = get_scheduler_status()
    try:
        with open('model_meta.json') as f:
            meta = _json.load(f)
    except Exception:
        pass
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'zones_loaded': len(modeles_zones),
        'python': sys.version.split()[0],
        'commit': os.environ.get('RAILWAY_GIT_COMMIT_SHA', 'local')[:7],
        'model_name': meta.get('model_name', 'unknown'),
        'recall': meta.get('recall'),
        'precision': meta.get('precision'),
        'f1': meta.get('f1'),
        'n_train': meta.get('n_train'),
        'trained_at': meta.get('trained_at'),
        'source': meta.get('source'),
        'scheduler_enabled': scheduler.get('enabled'),
        'scheduler_started': scheduler.get('started'),
        'scheduler_reason': scheduler.get('reason'),
        'scheduler_lock_kind': scheduler.get('lock_kind'),
    })

@app.route('/api/bgmonitor/start', methods=['POST'])
@login_required
def api_bgmonitor_start():
    data = request.json or {}
    path = (data.get('path') or '').strip()
    if not path or not os.path.isfile(path):
        return jsonify(error='File not found'), 400
    interval = max(1, int(data.get('interval', 5)))
    machine_id = data.get('machine_id') or None
    pilar_monitor.start_monitor(path, interval, predict_risk, machine_id)
    return jsonify(ok=True, path=path, fname=os.path.basename(path))

@app.route('/api/bgmonitor/stop', methods=['POST'])
@login_required
def api_bgmonitor_stop():
    data = request.json or {}
    path = (data.get('path') or '').strip()
    if data.get('all'):
        for m in list(_bg_monitors.values()):
            m['stop'].set()
        _bg_monitors.clear()
        return jsonify(ok=True, stopped='all')
    if path and pilar_monitor.stop_monitor(path):
        return jsonify(ok=True, stopped=path)
    return jsonify(ok=False, error='Monitor not found'), 404

@app.route('/api/bgmonitor/status')
@login_required
def api_bgmonitor_status():
    monitors = [
        {'path': m['path'], 'fname': m['fname'], 'rows': m.get('rows', 0), 'alerts': m.get('alerts', 0)}
        for m in _bg_monitors.values() if not m['stop'].is_set()
    ]
    return jsonify(monitors=monitors, count=len(monitors))

@app.route('/api/discover', methods=['POST'])
@login_required
def api_discover():
    r = _paid_required()
    if r: return jsonify({'error': 'Paid plan required'}), 403
    try:
        import json as _json, math
        data = request.json
        if not data:
            return jsonify({'ok': False})
        col_name = str(data.get('name', ''))[:100]
        values   = data.get('values', [])
        risks    = data.get('risks', [])
        if not col_name or not values:
            return jsonify({'ok': False})
        norm = ''.join(c if c.isalnum() else '_' for c in col_name.lower().strip()).strip('_')[:80]
        cl = col_name.lower()
        unit_guess = ''
        if '%' in cl or 'percent' in cl or 'humid' in cl: unit_guess = '%'
        elif 'bar' in cl or 'pres' in cl: unit_guess = 'bar'
        elif 'mm' in cl and ('vib' in cl or '/s' in cl): unit_guess = 'mm/s'
        elif 'volt' in cl or cl.endswith('_v'): unit_guess = 'V'
        elif 'amp' in cl or 'curr' in cl or cl.endswith('_a'): unit_guess = 'A'
        elif 'rpm' in cl or 'speed' in cl: unit_guess = 'rpm'
        elif 'temp' in cl or 'heat' in cl: unit_guess = 'K'
        uid = current_uid()
        dp = DiscoveredParam.query.filter_by(name=norm, user_id=uid).first()
        try:
            new_vals  = [float(v) for v in values if v is not None]
            new_risks = [float(r) for r in risks  if r is not None]
        except Exception:
            return jsonify({'ok': False})
        if not dp:
            dp = DiscoveredParam(name=norm, label=col_name, unit_guess=unit_guess, user_id=uid)
            db.session.add(dp)
        existing_vals  = _json.loads(dp.samples_json or '[]')
        existing_risks = _json.loads(dp.risks_json   or '[]')
        all_vals  = (existing_vals  + new_vals) [-500:]
        all_risks = (existing_risks + new_risks)[-500:]
        dp.samples_json = _json.dumps(all_vals)
        dp.risks_json   = _json.dumps(all_risks)
        dp.n_samples    = len(all_vals)
        dp.updated_at   = datetime.now(timezone.utc)
        if len(all_vals) >= 10:
            n = min(len(all_vals), len(all_risks))
            xv = all_vals[:n]; yr = all_risks[:n]
            mx = sum(xv)/n; my = sum(yr)/n
            num = sum((xv[i]-mx)*(yr[i]-my) for i in range(n))
            sx  = math.sqrt(sum((x-mx)**2 for x in xv) or 1)
            sy  = math.sqrt(sum((y-my)**2 for y in yr) or 1)
            dp.impact = round(num/(sx*sy), 3) if sx*sy > 0 else 0.0
        db.session.commit()
        return jsonify({'ok': True, 'impact': dp.impact, 'n_samples': dp.n_samples, 'label': dp.label})
    except Exception as e:
        db.session.rollback()
        return jsonify({'ok': False, 'error': str(e)})

# ── API v1 ────────────────────────────────────────────────────────────────────
def _api_rate_check(api_key, plan='free'):
    """Retourne (allowed, count, limit)."""
    today = datetime.now(timezone.utc).date().isoformat()
    rec = _api_calls.get(api_key, {'count': 0, 'day': ''})
    if rec['day'] != today:
        rec = {'count': 0, 'day': today}
    rec['count'] += 1
    _api_calls[api_key] = rec
    limit = 50000  # all users have full API access
    return rec['count'] <= limit, rec['count'], limit

def _resolve_api_user():
    """Retourne (user, error_response). Accepte X-Api-Key header ou session."""
    api_key = request.headers.get('X-Api-Key') or request.args.get('api_key')
    if api_key:
        user = User.query.filter_by(api_key=api_key).first()
        if not user:
            return None, (jsonify({'error': 'Invalid API key', 'code': 'AUTH_FAILED'}), 401)
        allowed, count, limit = _api_rate_check(api_key, user.plan)
        if not allowed:
            return None, (jsonify({'error': 'Rate limit exceeded', 'limit': limit,
                                   'reset': 'midnight UTC', 'code': 'RATE_LIMIT'}), 429)
        return user, None
    uid = current_uid()
    if uid:
        return db.session.get(User, uid), None
    return None, (jsonify({'error': 'Authentication required — pass X-Api-Key header', 'code': 'AUTH_REQUIRED'}), 401)

def _parse_sensor_input(data):
    """Parse et convertit les champs capteurs pompe. Retourne (params, extra, machine_id, err)."""
    import json as _json
    if not data:
        return None, None, None, (jsonify({'error': 'Empty JSON body', 'code': 'BAD_REQUEST'}), 400)
    machine_id = str(data.get('machine_id', '') or '')[:100] or None
    for field in CORE_FEATURES:
        if field in data and data[field] is not None:
            try:
                data[field] = float(data[field])
            except (TypeError, ValueError):
                data[field] = None
        else:
            data[field] = None
    if all(data.get(f) is None for f in CORE_FEATURES):
        return None, None, None, (jsonify({'error': 'At least one sensor field required',
                                           'code': 'NO_FIELDS'}), 400)
    for fld, (lo, hi) in SENSOR_BOUNDS.items():
        v = data.get(fld)
        if v is not None and not (lo <= v <= hi):
            return None, None, None, (jsonify({'error': f'Value out of range: {fld}={v} (expected [{lo},{hi}])', 'code': 'OUT_OF_RANGE'}), 400)
    extra = {}
    for field in OPTIONAL_FIELDS:
        if field in data and data[field] is not None:
            try:
                extra[field] = round(float(data[field]), 3)
            except (TypeError, ValueError):
                pass
    return data, extra, machine_id, None

@app.route('/api/v1/analyze', methods=['POST'])
def api_v1_analyze():
    user, err = _resolve_api_user()
    if err: return err
    if model is None:
        return jsonify({'error': 'Model not loaded', 'code': 'MODEL_UNAVAILABLE'}), 503
    import json as _json
    data, extra, machine_id, err = _parse_sensor_input(request.json)
    if err: return err
    try:
        _uid = user.id if user else None
        _machine = Machine.query.filter_by(user_id=_uid, name=machine_id, is_active=True).first() if (_uid and machine_id) else None
        _threshold = float(_machine.threshold) if (_machine and _machine.threshold) else DEFAULT_THRESHOLD
        probabilite, prediction, zones_risque, confidence, imputed = predict_risk(data, threshold=_threshold)
        mail_envoye = False
        email = (_machine.alert_email if _machine and _machine.alert_email else None) or get_setting('responsible_email', uid=_uid)
        if probabilite >= _threshold and email:
            import secrets as _sec
            ack_tok = _sec.token_hex(24)
            esc_email = _machine.escalation_email if _machine else None
            threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data, ack_tok), daemon=True).start()
            mail_envoye = True
            _al = AlertLog(user_id=_uid, machine_id_str=machine_id,
                email_to=email, probabilite=probabilite,
                ack_token=ack_tok, escalation_email=esc_email)
            db.session.add(_al)
        zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
        a = Analysis(machine_type='pump',
            temp_air=data.get('temp_palier'), temp_process=data.get('temp_moteur'),
            vitesse=data.get('debit'), couple=data.get('pression_sortie'), usure=data.get('heure_fonctionnement'),
            risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye,
            extra_params=_json.dumps(extra) if extra else None, confidence=confidence,
            machine_id=machine_id, user_id=_uid)
        db.session.add(a)
        db.session.commit()
        return jsonify({
            'ok': True,
            'analysis_id': a.id,
            'timestamp': a.timestamp.isoformat() + 'Z',
            'machine_id': machine_id,
            'prediction': prediction,
            'risk': probabilite,
            'alert': probabilite >= _threshold,
            'confidence': confidence,
            'imputed': imputed,
            'zones': [{'code': z['nom'].split()[0], 'name': z['nom'], 'probability': z['proba']}
                      for z in zones_risque],
            'mail_sent': mail_envoye,
        })
    except Exception as e:
        db.session.rollback()
        import traceback
        logger.info(f"api_v1_analyze] {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'code': 'INTERNAL_ERROR'}), 500

@app.route('/api/v1/analyze/batch', methods=['POST'])
def api_v1_batch():
    user, err = _resolve_api_user()
    if err: return err
    if model is None:
        return jsonify({'error': 'Model not loaded', 'code': 'MODEL_UNAVAILABLE'}), 503
    import json as _json
    body = request.json
    if not body or 'readings' not in body:
        return jsonify({'error': 'Expected {"readings": [...]}', 'code': 'BAD_REQUEST'}), 400
    readings = body['readings']
    if not isinstance(readings, list) or len(readings) == 0:
        return jsonify({'error': 'readings must be a non-empty array', 'code': 'BAD_REQUEST'}), 400
    if len(readings) > 100:
        return jsonify({'error': 'Max 100 readings per batch', 'code': 'LIMIT_EXCEEDED'}), 400
    results = []
    for i, raw in enumerate(readings):
        try:
            data, extra, machine_id, err2 = _parse_sensor_input(dict(raw) if isinstance(raw, dict) else {})
            if err2:
                results.append({'index': i, 'ok': False, 'error': 'Invalid input'})
                continue
            probabilite, prediction, zones_risque, confidence, imputed = predict_risk(data)
            a = Analysis(machine_type='pump',
                temp_air=data.get('temp_palier'), temp_process=data.get('temp_moteur'),
                vitesse=data.get('debit'), couple=data.get('pression_sortie'), usure=data.get('heure_fonctionnement'),
                risk=probabilite, prediction=prediction,
                zones=', '.join([z['nom'] for z in zones_risque]) if zones_risque else '',
                extra_params=_json.dumps(extra) if extra else None, confidence=confidence,
                machine_id=machine_id, user_id=user.id if user else None)
            db.session.add(a)
            results.append({'index': i, 'ok': True, 'analysis_id': None,
                            'prediction': prediction, 'risk': probabilite,
                            'alert': probabilite >= 50, 'confidence': confidence,
                            'machine_id': machine_id, 'imputed': imputed,
                            'zones': [{'code': z['nom'].split()[0], 'name': z['nom'],
                                       'probability': z['proba']} for z in zones_risque]})
        except Exception as e:
            results.append({'index': i, 'ok': False, 'error': str(e)})
    try:
        db.session.flush()
        for j, r in enumerate(results):
            if r.get('ok') and db.session.new:
                pass
        db.session.commit()
    except Exception as e:
        db.session.rollback()
    return jsonify({'ok': True, 'count': len(readings), 'results': results})

@app.route('/api/v1/status', methods=['GET'])
def api_v1_status():
    user, err = _resolve_api_user()
    if err: return err
    import json as _json
    meta = {}
    try:
        with open('model_meta.json') as f:
            meta = _json.load(f)
    except Exception:
        pass
    return jsonify({
        'ok': True,
        'model_loaded': model is not None,
        'model_name': meta.get('model_name', 'unknown'),
        'recall': meta.get('recall'),
        'precision': meta.get('precision'),
        'f1': meta.get('f1'),
        'n_train': meta.get('n_train'),
        'trained_at': meta.get('trained_at'),
        'zones': meta.get('zones', []),
        'plan': user.plan if user else 'guest',
    })

@app.route('/api/v1/history', methods=['GET'])
def api_v1_history():
    user, err = _resolve_api_user()
    if err: return err
    uid = user.id if user else None
    limit  = min(int(request.args.get('limit', 50)), 500)
    machine = request.args.get('machine_id')
    q = Analysis.query.filter_by(user_id=uid)
    if machine:
        q = q.filter_by(machine_id=machine)
    rows = q.order_by(Analysis.timestamp.desc()).limit(limit).all()
    return jsonify({'ok': True, 'count': len(rows), 'analyses': [
        {'id': a.id, 'timestamp': a.timestamp.isoformat() + 'Z',
         'machine_id': a.machine_id, 'machine_type': a.machine_type,
         'risk': a.risk, 'prediction': a.prediction, 'confidence': a.confidence,
         'zones': a.zones, 'temp_palier': a.temp_air, 'temp_moteur': a.temp_process,
         'debit': a.vitesse, 'pression_sortie': a.couple, 'heure_fonctionnement': a.usure}
        for a in rows
    ]})

@app.route('/api/docs')
def api_docs():
    uid = current_uid()
    api_key = ''
    if uid:
        user = db.session.get(User, uid)
        if user and user.api_key:
            api_key = user.api_key
    return render_template('api_docs.html', api_key=api_key, ak=api_key)

@app.route('/api/whatif', methods=['POST'])
@login_required
def api_whatif():
    r = _paid_required()
    if r: return jsonify({'error': 'Paid plan required'}), 403
    try:
        params = request.json
        if not params: return jsonify({'error': 'Données manquantes'}), 400
        risk, pred, zones, _, _ = predict_risk(params)
        if pred == 0: status, message = 'Normal Operation', 'No failure predicted under these conditions.'
        elif risk < 50: status, message = 'Low Risk', 'Minor anomaly. Continue monitoring.'
        else: status, message = 'High Failure Risk', 'Check vibration, bearing temperature, and run hours.'
        return jsonify({'risk':risk,'status':status,'message':message,'zones':zones})
    except Exception as e:
        logger.info(f"api_whatif] ERROR: {type(e).__name__}: {e}")
        return jsonify({'error': 'Erreur serveur'}), 500

# ── TEAM ROUTES ───────────────────────────────────────────────────────────────
@app.route('/team/create', methods=['POST'])
@login_required
def team_create():
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    if user.team_id:
        return jsonify({'error': 'Already in a team'}), 400
    name = (request.json or {}).get('name', 'My Team').strip() or 'My Team'
    team = Team(name=name)
    db.session.add(team)
    db.session.commit()
    db.session.add(TeamMember(team_id=team.id, user_id=uid, role='leader'))
    user.team_id = team.id
    db.session.commit()
    return jsonify({'ok': True, 'team_id': team.id})

@app.route('/team/invite', methods=['POST'])
@login_required
def team_invite():
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    my_mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=uid).first()
    if not my_mbr or my_mbr.role != 'leader':
        return jsonify({'error': 'Leader access required'}), 403
    email = (request.json or {}).get('email', '').strip().lower()
    target = User.query.filter_by(email=email).first()
    if not target:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    existing = TeamMember.query.filter_by(team_id=user.team_id, user_id=target.id).first()
    if existing:
        if existing.is_kicked:
            existing.is_kicked = False
            existing.role = 'member'
            target.team_id = user.team_id
            db.session.commit()
            return jsonify({'ok': True})
        return jsonify({'error': 'Déjà membre de l\'équipe'}), 409
    db.session.add(TeamMember(team_id=user.team_id, user_id=target.id, role='member'))
    target.team_id = user.team_id
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/team/kick/<int:target_uid>', methods=['POST'])
@login_required
def team_kick(target_uid):
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    my_mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=uid).first()
    if not my_mbr or my_mbr.role != 'leader':
        return jsonify({'error': 'Leader access required'}), 403
    t_mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=target_uid).first()
    if not t_mbr:
        return jsonify({'error': 'Member not found'}), 404
    if t_mbr.role == 'leader':
        return jsonify({'error': 'Cannot kick a leader'}), 400
    t_mbr.is_kicked = True
    target = db.session.get(User, target_uid)
    if target:
        target.team_id = None
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/team/transfer/<int:target_uid>', methods=['POST'])
@login_required
def team_transfer(target_uid):
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    my_mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=uid).first()
    if not my_mbr or my_mbr.role != 'leader':
        return jsonify({'error': 'Leader access required'}), 403
    t_mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=target_uid).first()
    if not t_mbr or t_mbr.is_kicked:
        return jsonify({'error': 'Member not found'}), 404
    if t_mbr.role == 'leader':
        return jsonify({'error': 'Already a leader'}), 400
    leaders_count = TeamMember.query.filter_by(team_id=user.team_id, role='leader', is_kicked=False).count()
    if leaders_count >= 2:
        my_mbr.role = 'member'
    t_mbr.role = 'leader'
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/team/leave', methods=['POST'])
@login_required
def team_leave():
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    mbr = TeamMember.query.filter_by(team_id=user.team_id, user_id=uid).first()
    if mbr:
        db.session.delete(mbr)
    user.team_id = None
    db.session.commit()
    return jsonify({'ok': True})

# ── TEAM CHAT ─────────────────────────────────────────────────────────────────
@app.route('/team/messages')
@login_required
def team_messages_get():
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user or not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    since_id = request.args.get('since', 0, type=int)
    msgs = (TeamMessage.query
            .filter(TeamMessage.team_id == user.team_id, TeamMessage.id > since_id)
            .order_by(TeamMessage.created_at.asc())
            .limit(50).all())
    return jsonify([{
        'id': m.id,
        'user_id': m.user_id,
        'email': m.user_email,
        'content': m.content,
        'ts': m.created_at.strftime('%H:%M'),
        'mine': m.user_id == uid
    } for m in msgs])

@app.route('/team/messages', methods=['POST'])
@login_required
def team_messages_post():
    uid = current_uid()
    user = db.session.get(User, uid)
    if not user or not user.team_id:
        return jsonify({'error': 'Not in a team'}), 400
    content = ((request.json or {}).get('content') or '').strip()[:1000]
    if not content:
        return jsonify({'error': 'empty'}), 400
    msg = TeamMessage(team_id=user.team_id, user_id=uid, user_email=user.email, content=content)
    db.session.add(msg)
    db.session.commit()
    return jsonify({'id': msg.id, 'ok': True})

# ── PWA ───────────────────────────────────────────────────────────────────────
@app.route('/manifest.json')
def manifest():
    from flask import Response
    import json
    data = {
        "name": "Pilar",
        "short_name": "Pilar",
        "description": "Predictive Maintenance System",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#07090f",
        "theme_color": "#07090f",
        "orientation": "portrait-primary",
        "icons": [
            {"src": f"data:image/png;base64,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", "sizes": "192x192", "type": "image/png"},
            {"src": f"data:image/png;base64,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", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}
        ]
    }
    return Response(json.dumps(data), mimetype='application/json')

@app.route('/sw.js')
def service_worker():
    from flask import Response
    sw = """
const CACHE = 'pilar-v3';
const URLS = ['/', '/account', '/twin', '/history', '/settings', '/manifest.json'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(URLS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k))))
    .then(() => clients.claim())
));
self.addEventListener('fetch', e => {
  if(e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
"""
    resp = Response(sw, mimetype='application/javascript')
    resp.headers['Cache-Control'] = 'no-store'
    return resp

# ── GLOBAL ERROR HANDLER ──────────────────────────────────────────────────────
@app.errorhandler(500)
def internal_error(e):
    import traceback
    tb = traceback.format_exc()
    logger.info(f"500 ERROR:\n{tb}")
    try: db.session.rollback()
    except: pass
    wants_json = request.headers.get('Accept','').find('application/json') >= 0 \
                 or request.headers.get('Content-Type','').find('application/json') >= 0
    if wants_json:
        return jsonify({'error': 'Internal server error'}), 500
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:'Inter','Segoe UI',system-ui,Arial,sans-serif;background:#07090f;color:#ffffff;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.c{{max-width:420px;text-align:center;padding:40px;}}.logo{{font-size:15px;letter-spacing:0.04em;color:#0d9488;font-weight:700;}}.msg{{color:rgba(235,235,245,0.6);font-size:13px;margin:16px 0 24px;line-height:1.7;}}
a{{padding:12px 24px;background:#0d9488;color:#fff;border-radius:12px;text-decoration:none;font-size:15px;font-weight:600;}}</style></head>
<body><div class="c"><div class="logo">PILAR</div><h2 style="margin:20px 0 8px;font-size:18px;font-weight:700;">Erreur serveur</h2>
<p class="msg">Une erreur inattendue s'est produite.<br>Elle a été enregistrée dans les logs.</p>
<a href="/">Retour</a></div></body></html>""", 500

@app.errorhandler(Exception)
def unhandled(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException): return e  # laisser Flask gérer 404, 405, etc.
    import traceback
    logger.info(f"Unhandled exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    try: db.session.rollback()
    except: pass
    return internal_error(e)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SYNC CLIENT & LOCAL CHAT
# ══════════════════════════════════════════════════════════════════════════════

# ── Sync Config (set via environment or config.py) ────────────────────────────
PILAR_SYNC_URL    = os.environ.get("PILAR_SYNC_URL", "")          # e.g. https://your-sync-server.example.com
PILAR_CLIENT_ID   = os.environ.get("PILAR_CLIENT_ID", "")
PILAR_CLIENT_SECRET = os.environ.get("PILAR_CLIENT_SECRET", "")

with app.app_context():
    try:
        db.create_all()
        # SQLite migrations for new tables
        _sync_migrations = [
            "ALTER TABLE sync_queue ADD COLUMN data_type VARCHAR(20) DEFAULT 'analysis'",
            "ALTER TABLE local_chat_message ADD COLUMN is_local INTEGER DEFAULT 0",
            "ALTER TABLE local_chat_message ADD COLUMN synced_at DATETIME",
        ]
        for sql in _sync_migrations:
            try:
                db.session.execute(db.text(sql))
                db.session.commit()
            except Exception:
                db.session.rollback()
        logger.info("Sync tables ready")
    except Exception as _se:
        logger.info(f"Sync table init: {_se}")


# ── Internet connectivity check (independent of sync) ────────────────────────
import socket as _socket
_inet_cache = {'online': False, 'ts': 0}
_inet_lock  = threading.Lock()

def _check_internet_now():
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(('8.8.8.8', 53))
        s.close()
        return True
    except Exception:
        return False

def _get_internet_status():
    import time as _time
    with _inet_lock:
        if _time.time() - _inet_cache['ts'] < 30:
            return _inet_cache['online']
    result = _check_internet_now()
    with _inet_lock:
        _inet_cache['online'] = result
        _inet_cache['ts'] = _time.time()
    return result


# ── SyncClient Class ──────────────────────────────────────────────────────────

class SyncClient:
    """
    Background sync client.
    - Every 60 seconds: checks connectivity, pushes queued analyses/notes, pulls chat.
    - Works transparently — no user interaction needed.
    """

    INTERVAL = 60   # seconds between sync cycles

    def __init__(self):
        self._online  = False
        self._last_sync: datetime | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_pull_ts: datetime | None = None

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def last_sync(self) -> datetime | None:
        return self._last_sync

    def start(self):
        if not PILAR_SYNC_URL or not PILAR_CLIENT_ID or not PILAR_CLIENT_SECRET:
            logger.info("[SyncClient] No sync credentials — sync disabled")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"[SyncClient] Started (target={PILAR_SYNC_URL})")

    def stop(self):
        self._stop_event.set()

    def _headers(self) -> dict:
        return {
            "X-Client-ID":     PILAR_CLIENT_ID,
            "X-Client-Secret": PILAR_CLIENT_SECRET,
            "Content-Type":    "application/json",
        }

    def _check_connectivity(self) -> bool:
        """Ping the sync server status endpoint."""
        import urllib.request, urllib.error
        try:
            req = urllib.request.Request(
                f"{PILAR_SYNC_URL}/api/sync/status",
                headers={
                    "X-Client-ID":     PILAR_CLIENT_ID,
                    "X-Client-Secret": PILAR_CLIENT_SECRET,
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _push_queue(self):
        """Push all unsynced items in SyncQueue to the server."""
        import json as _j, urllib.request
        with app.app_context():
            pending = SyncQueue.query.filter_by(synced_at=None).limit(200).all()
            if not pending:
                return

            analyses, notes, ids = [], [], []
            for item in pending:
                try:
                    d = _j.loads(item.data_json)
                    if item.data_type == "note":
                        notes.append(d)
                    else:
                        analyses.append(d)
                    ids.append(item.id)
                except Exception:
                    pass

            payload = _j.dumps({"analyses": analyses, "notes": notes}).encode()
            req = urllib.request.Request(
                f"{PILAR_SYNC_URL}/api/sync",
                data=payload,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = _j.loads(resp.read())
                if result.get("ok"):
                    now = datetime.now(timezone.utc)
                    SyncQueue.query.filter(SyncQueue.id.in_(ids)).update(
                        {SyncQueue.synced_at: now}, synchronize_session=False
                    )
                    db.session.commit()
                    logger.info(f"[SyncClient] Pushed {len(analyses)} analyses, {len(notes)} notes")
            except Exception as e:
                logger.info(f"[SyncClient] Push failed: {e}")

    def _pull_chat(self):
        """Pull new chat messages from the server."""
        import json as _j, urllib.request
        with app.app_context():
            since = self._last_pull_ts
            url   = f"{PILAR_SYNC_URL}/api/sync/pull"
            if since:
                url += f"?since={since.isoformat()}"
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _j.loads(resp.read())
                messages = data.get("messages", [])
                for m in messages:
                    # Skip if already stored
                    existing = LocalChatMessage.query.filter_by(
                        remote_id=m["id"]
                    ).first()
                    if existing:
                        continue
                    ts = None
                    try:
                        ts = datetime.fromisoformat(m["created_at"])
                    except Exception:
                        ts = datetime.now(timezone.utc)
                    msg = LocalChatMessage(
                        remote_id   = m["id"],
                        client_id   = m.get("client_id", ""),
                        sender_name = m.get("sender_name", ""),
                        room        = m.get("room", "general"),
                        content     = m.get("content", ""),
                        image_data  = m.get("image_data", "") or None,
                        image_mime  = m.get("image_mime", "") or None,
                        created_at  = ts,
                        is_local    = False,
                        synced_at   = datetime.now(timezone.utc),
                    )
                    db.session.add(msg)
                if messages:
                    db.session.commit()
                    logger.info(f"[SyncClient] Pulled {len(messages)} chat messages")
                self._last_pull_ts = datetime.now(timezone.utc)
            except Exception as e:
                logger.info(f"[SyncClient] Pull failed: {e}")

    def _push_local_chat(self):
        """Push offline-composed chat messages to the server."""
        import json as _j, urllib.request
        with app.app_context():
            pending = LocalChatMessage.query.filter_by(is_local=True, synced_at=None).limit(50).all()
            for msg in pending:
                payload = _j.dumps({
                    "room":        msg.room,
                    "content":     msg.content or "",
                    "sender_name": msg.sender_name or "",
                    "image_data":  msg.image_data or "",
                    "image_mime":  msg.image_mime or "",
                }).encode()
                req = urllib.request.Request(
                    f"{PILAR_SYNC_URL}/api/chat/messages",
                    data=payload,
                    headers=self._headers(),
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        result = _j.loads(resp.read())
                    msg.remote_id = result.get("id")
                    msg.synced_at = datetime.now(timezone.utc)
                    msg.is_local  = False
                    db.session.commit()
                except Exception as e:
                    logger.info(f"[SyncClient] Chat push failed: {e}")

    def _loop(self):
        while not self._stop_event.wait(self.INTERVAL):
            try:
                was_online = self._online
                self._online = self._check_connectivity()
                if not was_online and self._online:
                    logger.info("[SyncClient] Back online")
                elif was_online and not self._online:
                    logger.info("[SyncClient] Gone offline")

                if self._online:
                    self._push_queue()
                    self._pull_chat()
                    self._push_local_chat()
                    self._last_sync = datetime.now(timezone.utc)
            except Exception as e:
                logger.info(f"[SyncClient] Loop error: {e}")

    def enqueue_analysis(self, analysis_obj):
        """Call this after saving an Analysis to DB to queue it for sync."""
        import json as _j
        try:
            with app.app_context():
                extra = {}
                try:
                    extra = _j.loads(analysis_obj.extra_params or "{}")
                except Exception:
                    pass
                data = {
                    "remote_id":   analysis_obj.id,
                    "machine_id":  analysis_obj.machine_id or "",
                    "machine_type":analysis_obj.machine_type or "",
                    "risk":        analysis_obj.risk,
                    "prediction":  analysis_obj.prediction,
                    "zones":       analysis_obj.zones or "",
                    "extra_params":extra,
                    "feedback":    analysis_obj.feedback or "",
                    "confidence":  analysis_obj.confidence or 100,
                    "timestamp":   analysis_obj.timestamp.isoformat(),
                }
                item = SyncQueue(data_type="analysis", data_json=_j.dumps(data))
                db.session.add(item)
                db.session.commit()
        except Exception as e:
            logger.info(f"[SyncClient] Enqueue analysis error: {e}")


# Instantiate and start sync client
_sync_client = SyncClient()

with app.app_context():
    _sync_client.start()


def _queue_analysis_for_sync(analysis):
    """Helper called after each new analysis is saved."""
    if PILAR_SYNC_URL:
        try:
            _sync_client.enqueue_analysis(analysis)
        except Exception as _qe:
            logger.info(f"[SyncClient] Queue error: {_qe}")


# ── API: Chat Routes ──────────────────────────────────────────────────────────

@app.route('/api/chat/messages', methods=['GET'])
@auth_optional
def chat_get_messages():
    """Return chat messages (local DB + synced from server)."""
    room  = request.args.get('room', 'general')
    limit = min(int(request.args.get('limit', 100)), 200)
    since_str = request.args.get('since', '')
    since = None
    if since_str:
        try:
            since = datetime.fromisoformat(since_str)
        except Exception:
            pass
    q = LocalChatMessage.query.filter_by(room=room)
    if since:
        q = q.filter(LocalChatMessage.created_at > since)
    msgs = q.order_by(LocalChatMessage.created_at.asc()).limit(limit).all()
    return jsonify([{
        'id':          m.id,
        'remote_id':   m.remote_id,
        'client_id':   m.client_id,
        'sender_name': m.sender_name,
        'room':        m.room,
        'content':     m.content or '',
        'image_data':  m.image_data or '',
        'image_mime':  m.image_mime or '',
        'created_at':  m.created_at.isoformat(),
        'is_local':    m.is_local,
    } for m in msgs])


@app.route('/api/chat/messages', methods=['POST'])
@auth_optional
def chat_post_message():
    """
    Send a chat message.
    - If online: also pushes to sync server immediately.
    - If offline: stores locally with is_local=True; sync client will push later.
    Body: { room, content, sender_name, image_data (base64), image_mime }
    """
    import json as _j
    data        = request.get_json(force=True) or {}
    content     = str(data.get('content',     '')).strip()
    image_data  = data.get('image_data',  '')
    image_mime  = str(data.get('image_mime', 'image/jpeg'))[:50]
    room        = str(data.get('room',     'general'))[:100]

    # Determine sender name from logged-in user or payload
    uid         = current_uid()
    user        = db.session.get(User, uid) if uid else None
    sender_name = str(data.get('sender_name', (user.email if user else 'Anonymous')))[:200]

    if not content and not image_data:
        return jsonify({'error': 'Empty message'}), 400
    if image_data and len(image_data) > 5 * 1024 * 1024:
        return jsonify({'error': 'Image too large (max 5 MB)'}), 413

    # Store locally
    msg = LocalChatMessage(
        client_id   = PILAR_CLIENT_ID or 'local',
        sender_name = sender_name,
        room        = room,
        content     = content,
        image_data  = image_data or None,
        image_mime  = image_mime if image_data else None,
        is_local    = True,
    )
    db.session.add(msg)
    db.session.commit()

    # If online, push immediately to server
    if _sync_client.is_online and PILAR_SYNC_URL:
        import urllib.request
        payload = _j.dumps({
            'room':        room,
            'content':     content,
            'sender_name': sender_name,
            'image_data':  image_data or '',
            'image_mime':  image_mime,
        }).encode()
        req = urllib.request.Request(
            f"{PILAR_SYNC_URL}/api/chat/messages",
            data=payload,
            headers={
                'X-Client-ID':     PILAR_CLIENT_ID,
                'X-Client-Secret': PILAR_CLIENT_SECRET,
                'Content-Type':    'application/json',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                result = _j.loads(resp.read())
            msg.remote_id = result.get('id')
            msg.synced_at = datetime.now(timezone.utc)
            msg.is_local  = False
            db.session.commit()
        except Exception as _pe:
            logger.error(f"[Chat] Immediate push failed: {_pe}")

    return jsonify({
        'id':          msg.id,
        'remote_id':   msg.remote_id,
        'sender_name': sender_name,
        'room':        room,
        'content':     content,
        'created_at':  msg.created_at.isoformat(),
        'is_local':    msg.is_local,
    }), 201


@app.route('/api/sync/status')
def sync_status_route():
    """Returns sync status for the desktop UI status bar."""
    queued = SyncQueue.query.filter_by(synced_at=None).count()
    pending_chat = LocalChatMessage.query.filter_by(is_local=True, synced_at=None).count()
    online = _sync_client.is_online if PILAR_SYNC_URL else _get_internet_status()
    return jsonify({
        'online':       online,
        'queued':       queued,
        'pending_chat': pending_chat,
        'last_sync':    _sync_client.last_sync.isoformat() if _sync_client.last_sync else None,
        'sync_url':     PILAR_SYNC_URL or None,
        'client_id':    PILAR_CLIENT_ID[:8] + '…' if PILAR_CLIENT_ID else None,
    })


# ── END STEP 3 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Pilar v3 — http://localhost:{port} (debug={debug})")
    try:
        from agents.sla_tracker import init_sla_tables
        init_sla_tables()
        logger.info("SLA tables ready")
    except Exception as _ae:
        logger.info(f"SLA init skipped: {_ae}")
    app.run(debug=debug, host='0.0.0.0', port=port)
