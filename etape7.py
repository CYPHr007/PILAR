from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, g
import pickle, threading, smtplib, secrets as _secrets, subprocess
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd, numpy as np, warnings, time, collections
warnings.filterwarnings("ignore")

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
    _APP_DIR = getattr(_sys, '_MEIPASS', os.path.dirname(_sys.executable))
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
def _pkl(name): return os.path.join(_APP_DIR, name)
from config import (
    RATE_WINDOW, RATE_MAX, SESSION_DAYS,
    FAILURE_ZONES, COLONNES, FEATURE_MEDIANS, CORE_FEATURES, OPTIONAL_FIELDS,
    SENSOR_BOUNDS, FLUID_RUL_FACTORS, MATERIAL_RUL_FACTORS,
    FLUID_ZONE_SENSITIVITY, NON_CENTRIFUGE_TYPES, DOMAIN_KB,
    RETRAIN_TRIGGER, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CHAT_DAILY_LIMIT,
    RUL_SCALE_FACTOR,
)
db_url = (os.environ.get("DATABASE_URL")
          or os.environ.get("DATABASE_PUBLIC_URL")
          or "sqlite:///pilar.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
print(f"[Pilar] DB: {db_url[:40]}...")
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    _key_path = os.path.join(os.getcwd(), "pilar_secret.key")
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
            print(f"[Pilar] SECRET_KEY generated and saved to pilar_secret.key")
        except Exception:
            print("[Pilar] WARNING: SECRET_KEY random — sessions will reset on restart")
app.config["SECRET_KEY"] = _secret_key
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=SESSION_DAYS)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT", "RENDER", "DYNO"))
db = SQLAlchemy(app)

@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    if any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT", "RENDER", "DYNO")):
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# ── MODÈLES ───────────────────────────────────────────────────────────────────
class Team(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(200), default='My Team')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TeamMember(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    team_id   = db.Column(db.Integer, nullable=False)
    user_id   = db.Column(db.Integer, nullable=False)
    role      = db.Column(db.String(20), default='member')  # 'leader' or 'member'
    is_kicked = db.Column(db.Boolean, default=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(200), unique=True, nullable=False)
    password_hash  = db.Column(db.String(256), nullable=False)
    email_verified = db.Column(db.Boolean, default=True)
    verify_token   = db.Column(db.String(64))
    api_key        = db.Column(db.String(64), unique=True)
    plan           = db.Column(db.String(20), default='free')
    plan_expires_at= db.Column(db.DateTime, nullable=True)
    plan_note      = db.Column(db.String(300), nullable=True)
    is_admin       = db.Column(db.Boolean, default=False)
    is_banned      = db.Column(db.Boolean, default=False)
    team_id        = db.Column(db.Integer, nullable=True)
    onboarded      = db.Column(db.Boolean, default=False)
    machine_quota  = db.Column(db.Integer, default=3)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

class BannedEmail(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(200), unique=True, nullable=False)
    reason     = db.Column(db.String(300), nullable=True)
    banned_at  = db.Column(db.DateTime, default=datetime.utcnow)

class Settings(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    key     = db.Column(db.String(120))
    value   = db.Column(db.String(500))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class Analysis(db.Model):
    # NOTE: column names are legacy CNC-era identifiers. Pump domain mapping:
    #   temp_air     = bearing temperature  (temp_palier, °C)
    #   temp_process = motor temperature    (temp_moteur, °C)
    #   vitesse      = flow rate            (debit, m³/h)
    #   couple       = outlet pressure      (pression_sortie, bar)
    #   usure        = run hours            (heure_fonctionnement, h)
    # TODO: rename via DB migration once traffic is stable.
    id           = db.Column(db.Integer, primary_key=True)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)
    machine_type = db.Column(db.String(10))
    temp_air     = db.Column(db.Float)   # bearing temp (temp_palier)
    temp_process = db.Column(db.Float)   # motor temp   (temp_moteur)
    vitesse      = db.Column(db.Float)   # flow rate    (debit)
    couple       = db.Column(db.Float)   # outlet press (pression_sortie)
    usure        = db.Column(db.Float)   # run hours    (heure_fonctionnement)
    risk         = db.Column(db.Float)
    prediction   = db.Column(db.Integer)
    zones        = db.Column(db.String(500))
    mail_sent    = db.Column(db.Boolean, default=False)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    extra_params = db.Column(db.Text)
    confidence   = db.Column(db.Integer, default=100)
    machine_id   = db.Column(db.String(100))
    feedback     = db.Column(db.String(10))  # 'tp'=confirmed failure, 'fp'=false positive

class SavedFile(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, nullable=True)
    team_id    = db.Column(db.Integer, nullable=True)
    machine_id = db.Column(db.Integer, nullable=True)   # FK vers Machine.id
    filename   = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    row_count  = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TeamMessage(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    team_id    = db.Column(db.Integer, nullable=False)
    user_id    = db.Column(db.Integer, nullable=False)
    user_email = db.Column(db.String(200))
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DiscoveredParam(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100))
    label        = db.Column(db.String(200))
    unit_guess   = db.Column(db.String(20))
    impact       = db.Column(db.Float, default=0.0)
    n_samples    = db.Column(db.Integer, default=0)
    samples_json = db.Column(db.Text)
    risks_json   = db.Column(db.Text)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow)
    user_id      = db.Column(db.Integer, nullable=True)

class Machine(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name             = db.Column(db.String(200), nullable=False)
    description      = db.Column(db.String(500))
    machine_type     = db.Column(db.String(10), default='M')       # L / M / H quality class
    pump_type        = db.Column(db.String(50), default='centrifuge')
    fluid_type       = db.Column(db.String(50), default='eau')
    roue_material    = db.Column(db.String(50), default='inox_316')
    threshold        = db.Column(db.Float, default=45.0)
    location         = db.Column(db.String(200))
    install_date     = db.Column(db.Date, nullable=True)
    serial_number    = db.Column(db.String(100))
    nominal_flow     = db.Column(db.Float)   # m³/h — normal operating flow
    nominal_pressure = db.Column(db.Float)   # bar  — normal outlet pressure
    power_kw         = db.Column(db.Float)   # kW   — motor rated power
    nominal_current  = db.Column(db.Float)   # A    — motor nominal current
    nominal_vibration= db.Column(db.Float)   # mm/s — normal vibration level
    alert_email      = db.Column(db.String(200))
    escalation_email = db.Column(db.String(200))
    is_active        = db.Column(db.Boolean, default=True)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)

class MachineNote(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    user_id    = db.Column(db.Integer, nullable=False)
    user_email = db.Column(db.String(200))
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AlertLog(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, nullable=True)
    machine_id_str   = db.Column(db.String(100))   # matches Analysis.machine_id
    analysis_id      = db.Column(db.Integer, nullable=True)
    email_to         = db.Column(db.String(200))
    probabilite      = db.Column(db.Float)
    sent_at          = db.Column(db.DateTime, default=datetime.utcnow)
    acked_at         = db.Column(db.DateTime, nullable=True)
    ack_token        = db.Column(db.String(64), unique=True)
    escalated_at     = db.Column(db.DateTime, nullable=True)
    escalation_email = db.Column(db.String(200))

class MachineRequest(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, nullable=True)
    name         = db.Column(db.String(200))
    manufacturer = db.Column(db.String(200))
    rpm_range    = db.Column(db.String(100))
    torque_range = db.Column(db.String(100))
    description  = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status       = db.Column(db.String(20), default='pending')  # pending / integrated

with app.app_context():
    try:
        db.create_all()
        print("[Pilar] Tables créées/vérifiées")
    except Exception as e:
        print(f"[Pilar] db.create_all() error: {e}")
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
        ]
    for sql in _migrations:
        try:
            db.session.execute(db.text(sql))
            db.session.commit()
            print(f"[Pilar] Migration OK: {sql[:50]}")
        except Exception as e:
            db.session.rollback()
            print(f"[Pilar] Migration skip ({sql[:40]}): {e}")
    # SuperUser persistant
    try:
        _su = User.query.filter_by(email='aliguenbou07r@gmail.com').first()
        if _su and (not _su.is_admin or _su.plan != 'pro'):
            _su.is_admin = True
            _su.plan = 'pro'
            _su.plan_expires_at = None
            db.session.commit()
            print("[Pilar] SuperUser: aliguenbou07r@gmail.com → admin+pro lifetime")
    except Exception as _sue:
        db.session.rollback()
        print(f"[Pilar] SuperUser setup: {_sue}")
    # Upgrade all existing free users to pro — everyone who installed PILAR has a subscription
    try:
        _free_users = User.query.filter_by(plan='free').all()
        for _u in _free_users:
            _u.plan = 'pro'
            _u.plan_expires_at = None
        if _free_users:
            db.session.commit()
            print(f"[Pilar] Auto-upgraded {len(_free_users)} free user(s) to pro")
    except Exception as _upe:
        db.session.rollback()
        print(f"[Pilar] Auto-upgrade error: {_upe}")

try:
    with open(_pkl("modele_pannes.pkl"),"rb") as f: model = pickle.load(f)
    with open(_pkl("scaler.pkl"),"rb") as f: scaler = pickle.load(f)
    with open(_pkl("modeles_zones.pkl"),"rb") as f: modeles_zones = pickle.load(f)
    print(f"[Pilar] Modeles ML charges depuis {_APP_DIR}")
except FileNotFoundError as _e:
    print(f"[Pilar] FATAL: fichier modele manquant — {_e} (recherche dans: {_APP_DIR})")
    model = scaler = None
    modeles_zones = {}
except Exception as _e:
    print(f"[Pilar] FATAL: erreur chargement modele — {type(_e).__name__}: {_e}")
    model = scaler = None
    modeles_zones = {}

# ── RUL MODEL (NASA C-MAPSS) ──────────────────────────────────────────────────
_rul_model  = None
_rul_scaler = None
_RUL_SCALE = RUL_SCALE_FACTOR  # CMAPSS cycles → pump hours (see config.py)
try:
    with open(_pkl("rul_model.pkl"),  "rb") as f: _rul_model  = pickle.load(f)
    with open(_pkl("rul_scaler.pkl"), "rb") as f: _rul_scaler = pickle.load(f)
    print("[Pilar] RUL model charge (NASA C-MAPSS GBT)")
except FileNotFoundError: pass
except Exception as _e: print(f"[Pilar] RUL model load error: {_e}")

# ── ISOLATION FOREST ──────────────────────────────────────────────────────────
_iso_forest = None
_normal_samples = []   # accumulates normal scaled vectors for lazy IsoForest training
try:
    with open(_pkl("isolation_forest.pkl"),"rb") as f: _iso_forest = pickle.load(f)
    print("[Pilar] Isolation Forest chargé")
except FileNotFoundError: pass
except Exception as _e: print(f"[Pilar] Isolation Forest load error: {_e}")

# All ML constants (FAILURE_ZONES, COLONNES, FEATURE_MEDIANS, SENSOR_BOUNDS,
# FLUID_RUL_FACTORS, MATERIAL_RUL_FACTORS, FLUID_ZONE_SENSITIVITY,
# NON_CENTRIFUGE_TYPES, DOMAIN_KB, RETRAIN_TRIGGER, CLAUDE_MODEL, etc.)
# are imported from config.py at the top of this file.

# ── SHAP EXPLAINER ─────────────────────────────────────────────────────────────
_shap_explainer = None

def _init_shap_explainer():
    global _shap_explainer
    if model is None or scaler is None:
        return
    try:
        import shap
        bg = np.array([[FEATURE_MEDIANS[c] for c in COLONNES]])
        bg_scaled = scaler.transform(bg)
        _shap_explainer = shap.KernelExplainer(model.predict_proba, bg_scaled)
        print("[Pilar] SHAP explainer initialisé")
    except Exception as _e:
        print(f"[Pilar] SHAP init error: {_e}")

def _compute_shap(x_scaled):
    """Return top-3 SHAP feature impacts list, or [] on failure."""
    global _shap_explainer
    try:
        import shap
        # Try TreeExplainer first (instant for XGBoost/RF)
        try:
            exp = shap.TreeExplainer(model)
            sv = exp.shap_values(x_scaled)
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
        except Exception:
            if _shap_explainer is None:
                _init_shap_explainer()
            if _shap_explainer is None:
                return []
            sv = _shap_explainer.shap_values(x_scaled, nsamples=50, silent=True)
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
        labels = {
            'vibration': 'Vibration', 'temp_palier': 'Bearing temp',
            'debit': 'Flow rate', 'pression_entree': 'Inlet pressure',
            'pression_sortie': 'Outlet pressure', 'courant_moteur': 'Motor current',
            'temp_moteur': 'Motor temp', 'heure_fonctionnement': 'Run hours'
        }
        total_abs = sum(abs(v) for v in vals) or 1.0
        top3 = sorted(zip(COLONNES, vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        return [{'feature': labels.get(f, f),
                 'impact': ('+' if v > 0 else '') + str(round(abs(v) / total_abs * 100)) + '%',
                 'direction': 'up' if v > 0 else 'down'} for f, v in top3]
    except Exception as _e:
        print(f"[Pilar] SHAP compute error: {_e}")
        return []

def _compute_rul(x_scaled):
    """Return estimated remaining useful life in hours, or None."""
    if _rul_model is None or _rul_scaler is None:
        return None
    try:
        x_rul = _rul_scaler.transform(x_scaled)
        cycles = float(np.clip(_rul_model.predict(x_rul)[0], 0, None))
        return round(cycles * _RUL_SCALE, 1)
    except Exception as _e:
        print(f"[Pilar] RUL compute error: {_e}")
        return None

def _compute_anomaly_score(x_scaled):
    """Return 0–100 anomaly score (0=normal, 100=very anomalous), or None if model not ready."""
    if _iso_forest is None:
        return None
    try:
        raw = float(_iso_forest.score_samples(x_scaled)[0])
        # score_samples: ~0.15 = very normal, ~-0.5 = very anomalous
        return round(min(100.0, max(0.0, (0.15 - raw) / 0.65 * 100)), 1)
    except Exception as _e:
        print(f"[Pilar] IsoForest score error: {_e}")
        return None

GMAIL     = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_PWD = os.environ.get("GMAIL_APP_PASSWORD", "")

# ── In-app update banner ───────────────────────────────────────────────────────
_UPDATE_INFO = {'available': False, 'version': None, 'download_url': None}

def _check_update_background():
    """Called by launcher or on first boot — checks Railway for newer version."""
    try:
        import urllib.request as _ur, json as _j
        APP_VER = os.environ.get('PILAR_VERSION', '1.0.0')
        with _ur.urlopen('https://pilar-website.up.railway.app/api/latest', timeout=6) as r:
            d = _j.loads(r.read().decode())
        latest = (d.get('version') or '').lstrip('v')
        current = APP_VER.lstrip('v')
        if latest and latest != current:
            _UPDATE_INFO['available'] = True
            _UPDATE_INFO['version'] = latest
            _UPDATE_INFO['download_url'] = d.get('download_url', '')
            print(f"[Pilar] Update available: {current} → {latest}")
    except Exception as e:
        print(f"[Pilar] Update check: {e}")

# Fire once at startup in background
threading.Thread(target=_check_update_background, daemon=True).start()
FAVICON = "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"

def current_uid():
    return session.get('user_id')

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
    except Exception as e: print(f"Settings error: {e}")

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

def send_verify_email(email, token, base_url=None):
    if not GMAIL or not GMAIL_PWD:
        print(f"[Pilar/auth] Email non configuré (GMAIL/GMAIL_APP_PASSWORD manquants) — token pour {email}: {token}")
        return
    base = (base_url or os.environ.get("APP_URL", "")).rstrip('/')
    if not base:
        base = "https://trypilar.com"
    link = f"{base}/verify-email/{token}"
    html = f"""<div style="font-family:sans-serif;background:#07090f;color:#e2e8f0;padding:40px;border-radius:8px">
<h2 style="color:#14b8a6;letter-spacing:3px">PILAR</h2>
<p>Confirmez votre adresse email pour activer votre compte.</p>
<a href="{link}" style="display:inline-block;margin-top:16px;padding:12px 24px;background:#0d9488;color:#fff;border-radius:6px;text-decoration:none;font-weight:700">Vérifier mon email</a>
<p style="margin-top:24px;color:#64748b;font-size:12px">Lien valide 24h. Si vous n'avez pas créé de compte, ignorez cet email.</p>
<p style="color:#334155;font-size:11px">Ou copiez ce lien : {link}</p>
</div>"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Pilar — Vérifiez votre email"
    msg['From'] = f"Pilar <{GMAIL}>"
    msg['To'] = email
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email, msg.as_string())
        print(f"[Pilar/auth] Verification email sent to {email}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"[Pilar/auth] SMTP auth failed (vérifiez GMAIL_APP_PASSWORD): {e}")
    except smtplib.SMTPException as e:
        print(f"[Pilar/auth] SMTP error: {e}")
    except Exception as e:
        print(f"[Pilar/auth] Email error ({type(e).__name__}): {e}")


# ── AUTH PAGES ────────────────────────────────────────────────────────────────
_AUTH_HEAD = ("""<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#08090c">
<link rel="icon" type="image/png" href="data:image/png;base64,""" + FAVICON + """">
<title>PILAR</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>if('serviceWorker' in navigator){navigator.serviceWorker.getRegistrations().then(function(r){r.forEach(function(x){x.unregister();});});}</script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#08090c;--surface:#0f1117;--surface2:#141820;
  --border:#1c2130;--border2:#252d3d;
  --teal:#0d9488;--tl:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --r:10px;--r-sm:7px;
}
html,body{height:100%;}
body{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);overflow:hidden;}
.auth-layout{display:flex;height:100vh;}
.canvas-side{flex:1;position:relative;overflow:hidden;background:var(--bg);border-right:1px solid var(--border);}
#pillar-canvas{position:absolute;inset:0;display:block;width:100%;height:100%;}
.pilar-wm{position:absolute;bottom:40px;left:50%;transform:translateX(-50%);
  font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
  letter-spacing:9px;color:rgba(13,148,136,0.55);white-space:nowrap;pointer-events:none;}
.form-side{width:440px;min-width:360px;background:var(--surface);
  display:flex;flex-direction:column;justify-content:center;
  padding:52px 44px;overflow-y:auto;}
.form-logo{font-family:'DM Serif Display',serif;font-size:30px;color:var(--text);margin-bottom:6px;}
.form-tagline{font-size:13px;color:var(--text3);margin-bottom:40px;}
.flbl{font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--text3);
  text-transform:uppercase;margin-bottom:6px;display:block;margin-top:18px;}
.fi{width:100%;padding:11px 14px;background:rgba(255,255,255,0.035);
  border:1px solid var(--border);border-radius:var(--r);color:var(--text);
  font-size:14px;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .15s,background .15s;}
.fi:focus{border-color:var(--teal);background:rgba(255,255,255,0.055);}
.fi::placeholder{color:var(--text3);}
.btn-submit{width:100%;padding:13px;background:var(--teal);color:#fff;border:none;
  border-radius:var(--r);font-size:14px;font-weight:600;font-family:'DM Sans',sans-serif;
  cursor:pointer;margin-top:26px;transition:opacity .15s,box-shadow .15s;}
.btn-submit:hover{opacity:.88;box-shadow:0 0 0 3px rgba(13,148,136,0.22);}
.auth-err{padding:10px 14px;background:var(--red-dim);border:1px solid rgba(220,38,38,0.3);
  border-radius:var(--r-sm);font-size:12px;color:#f87171;margin-top:14px;}
.auth-link{text-align:center;margin-top:22px;font-size:12px;color:var(--text3);}
.auth-link a{color:var(--tl);text-decoration:none;}
.auth-link a:hover{text-decoration:underline;}
.verify-box{text-align:center;padding:12px 0;}
.verify-icon{width:46px;height:46px;border-radius:12px;background:var(--teal-dim);
  border:1px solid rgba(13,148,136,0.18);display:flex;align-items:center;
  justify-content:center;margin:0 auto 18px;}
.verify-title{font-family:'DM Serif Display',serif;font-size:20px;color:var(--text);margin-bottom:8px;}
.verify-sub{font-size:13px;color:var(--text2);line-height:1.65;}
.verify-note{font-size:11px;color:var(--text3);margin-top:16px;}
.btn-resend{width:100%;padding:11px;margin-top:20px;background:transparent;
  border:1px solid var(--border2);border-radius:var(--r);color:var(--text3);
  font-size:13px;font-weight:600;font-family:'DM Sans',sans-serif;cursor:pointer;
  transition:border-color .15s,color .15s;}
.btn-resend:hover{border-color:var(--teal);color:var(--tl);}
.lang-sw{position:fixed;top:14px;right:14px;display:flex;gap:2px;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:3px;z-index:99;}
.lang-sw button{padding:4px 10px;border:none;border-radius:5px;font-size:10px;
  font-weight:600;cursor:pointer;background:transparent;color:var(--text3);transition:all .15s;}
.lang-sw button.active{background:var(--teal);color:#fff;}
@media(max-width:800px){
  .auth-layout{flex-direction:column;}
  .canvas-side{flex:none;height:220px;}
  .form-side{width:100%;min-width:0;padding:32px 24px;flex:1;}
}
</style></head><body>
<div class="lang-sw" id="_authLang">
  <button id="_authEN" onclick="_authSetLang('en')">EN</button>
  <button id="_authFR" onclick="_authSetLang('fr')">FR</button>
</div>
<script>
var _aLang=localStorage.getItem('pilar_lang')||'fr';
var _TA={
en:{login_title:'Sign in',reg_title:'Create account',tagline:'Predictive maintenance platform',
  lbl_email:'Email',lbl_pw:'Password',lbl_pw2:'Confirm password',
  btn_login:'Sign in',btn_reg:'Create account',
  link_reg:'No account? <a href="/register">Create one</a>',
  link_login:'Already have an account? <a href="/login">Sign in</a>',
  verify_title:'Check your inbox',
  verify_sub:'A confirmation link was sent. Click it to activate your account.',
  verify_note:'Valid 24h \u00b7 Check spam',back_login:'Back to sign in',
  resend:'Resend email',resent:'Email sent again!'},
fr:{login_title:'Connexion',reg_title:'Cr\u00e9er un compte',tagline:'Plateforme de maintenance pr\u00e9dictive',
  lbl_email:'Email',lbl_pw:'Mot de passe',lbl_pw2:'Confirmer le mot de passe',
  btn_login:'Se connecter',btn_reg:'Cr\u00e9er mon compte',
  link_reg:'Pas encore de compte\u00a0? <a href="/register">Cr\u00e9er un compte</a>',
  link_login:'D\u00e9j\u00e0 un compte\u00a0? <a href="/login">Se connecter</a>',
  verify_title:'V\u00e9rifiez votre email',
  verify_sub:'Un lien de confirmation a \u00e9t\u00e9 envoy\u00e9. Cliquez dessus pour activer votre compte.',
  verify_note:'Valide 24h \u00b7 V\u00e9rifiez vos spams',back_login:'Retour \u00e0 la connexion',
  resend:"Renvoyer l'email",resent:'Email renvoy\u00e9\u00a0!'}
};
function _tA(k){return(_TA[_aLang]||_TA.fr)[k]||k;}
function _authSetLang(l){
  _aLang=l;localStorage.setItem('pilar_lang',l);
  document.getElementById('_authEN').className=l==='en'?'active':'';
  document.getElementById('_authFR').className=l==='fr'?'active':'';
  document.querySelectorAll('[data-t]').forEach(function(el){
    var k=el.getAttribute('data-t'),v=_tA(k);
    if(el.tagName==='INPUT'){el.placeholder=v;}else{el.innerHTML=v;}
  });
}
(function(){_authSetLang(_aLang);})();
document.addEventListener('DOMContentLoaded',function(){_authSetLang(_aLang);});
</script>
<div class="auth-layout">
  <div class="canvas-side" id="canvas-wrap">
    <canvas id="pillar-canvas"></canvas>
    <div class="pilar-wm">P I L A R</div>
  </div>
  <div class="form-side">""")

_AUTH_FOOT = """  </div>
</div>
<script>
(function(){
  'use strict';
  var PRM=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var wrap=document.getElementById('canvas-wrap');
  var cvs=document.getElementById('pillar-canvas');
  if(!wrap||!cvs) return;
  var ctx=cvs.getContext('2d');
  var S={PAUSE:'pause',DEGRADING:'degrading',THRESHOLD:'threshold',REGEN:'regen'};
  var W,H,PX,PY,PW_COL,PH_COL,CAP_EXT=15,BASE_EXT=20;
  function layout(){W=wrap.offsetWidth||window.innerWidth*0.6;H=wrap.offsetHeight||window.innerHeight;cvs.width=W;cvs.height=H;PW_COL=80;PH_COL=Math.min(H*0.70,460);PX=(W-PW_COL)/2;PY=H*0.88-PH_COL;}
  var grainPts=[],wearLines=[];
  function computeTexture(){var cH=PH_COL*0.10,sY=PY+cH,sH=PH_COL-cH*2;grainPts=[];for(var g=0;g<280;g++)grainPts.push([PX-4+Math.random()*(PW_COL+8),PY+Math.random()*PH_COL]);wearLines=[];for(var i=0;i<6;i++)wearLines.push({x1:PX+Math.random()*10,y:sY+Math.random()*sH,x2:PX+PW_COL-Math.random()*10});}
  function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
  function drawPillar(scanY,glowA){var capH=PH_COL*0.10,baseH=PH_COL*0.10,sY=PY+capH,sH=PH_COL-capH-baseH;ctx.save();var gr=ctx.createLinearGradient(PX-BASE_EXT,0,PX+PW_COL+BASE_EXT,0);gr.addColorStop(0,'#2a3049');gr.addColorStop(0.28,'#222840');gr.addColorStop(0.74,'#1e2235');gr.addColorStop(1,'#141724');rr(PX-BASE_EXT,PY+PH_COL-baseH,PW_COL+BASE_EXT*2,baseH,2);ctx.fillStyle=gr;ctx.fill();ctx.fillStyle='rgba(255,255,255,0.05)';ctx.fillRect(PX-BASE_EXT,PY+PH_COL-baseH,PW_COL+BASE_EXT*2,1);rr(PX-CAP_EXT,PY,PW_COL+CAP_EXT*2,capH,2);ctx.fillStyle=gr;ctx.fill();ctx.fillStyle='rgba(255,255,255,0.06)';ctx.fillRect(PX-CAP_EXT,PY,PW_COL+CAP_EXT*2,1);var sg=ctx.createLinearGradient(PX,0,PX+PW_COL,0);sg.addColorStop(0,'#2a3049');sg.addColorStop(0.28,'#232942');sg.addColorStop(0.72,'#1e2235');sg.addColorStop(1,'#141724');ctx.fillStyle=sg;ctx.fillRect(PX,sY,PW_COL,sH);ctx.fillStyle='rgba(255,255,255,0.055)';ctx.fillRect(PX,sY,1,sH);ctx.fillStyle='rgba(0,0,0,0.22)';ctx.fillRect(PX+PW_COL-1,sY,1,sH);ctx.fillStyle='rgba(0,0,0,0.28)';ctx.fillRect(PX-CAP_EXT,sY,PX-(PX-CAP_EXT),sH);ctx.fillRect(PX+PW_COL,sY,(PX-CAP_EXT+PW_COL+CAP_EXT*2)-(PX+PW_COL),sH);for(var i=1;i<=4;i++){var fx=PX+i*(PW_COL/5);ctx.beginPath();ctx.moveTo(fx,sY+4);ctx.lineTo(fx,sY+sH-4);ctx.strokeStyle='rgba(0,0,0,0.20)';ctx.lineWidth=1.2;ctx.stroke();ctx.beginPath();ctx.moveTo(fx+0.9,sY+4);ctx.lineTo(fx+0.9,sY+sH-4);ctx.strokeStyle='rgba(255,255,255,0.038)';ctx.lineWidth=0.5;ctx.stroke();}ctx.lineWidth=0.35;wearLines.forEach(function(m){ctx.beginPath();ctx.moveTo(m.x1,m.y);ctx.lineTo(m.x2,m.y);ctx.strokeStyle='rgba(0,0,0,0.09)';ctx.stroke();});ctx.fillStyle='rgba(255,255,255,0.016)';grainPts.forEach(function(g){ctx.fillRect(g[0],g[1],1,1);});if(scanY!==undefined&&scanY>=PY&&scanY<=PY+PH_COL){var slg=ctx.createLinearGradient(PX-BASE_EXT,0,PX+PW_COL+BASE_EXT,0);slg.addColorStop(0,'rgba(13,148,136,0.18)');slg.addColorStop(0.5,'rgba(13,148,136,0.52)');slg.addColorStop(1,'rgba(13,148,136,0.18)');ctx.fillStyle=slg;ctx.fillRect(PX-BASE_EXT-2,scanY-2,PW_COL+BASE_EXT*2+4,5);var sg2=ctx.createLinearGradient(0,scanY-18,0,scanY+18);sg2.addColorStop(0,'rgba(13,148,136,0)');sg2.addColorStop(0.5,'rgba(13,148,136,0.07)');sg2.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=sg2;ctx.fillRect(PX-BASE_EXT-8,scanY-18,PW_COL+BASE_EXT*2+16,36);}if(glowA>0){var gg=ctx.createRadialGradient(PX+PW_COL/2,PY+PH_COL/2,0,PX+PW_COL/2,PY+PH_COL/2,W*0.65);gg.addColorStop(0,'rgba(13,148,136,'+(glowA*0.09)+')');gg.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=gg;ctx.fillRect(0,0,W,H);}ctx.restore();}
  function generateCrackSeeds(){return {devFreq:2+Math.floor(Math.random()*4),devAmp:Math.PI/28+Math.random()*Math.PI/10,bifProb:0.18+Math.random()*0.32,bifInterval:11+Math.random()*13,maxDepth:Math.random()<0.25?2:3,speedBase:5000+Math.random()*4000,speedDecay:0.48+Math.random()*0.28,branchWidthR:0.42+Math.random()*0.22,branchLenR:0.26+Math.random()*0.26,accelPt:0.18+Math.random()*0.17,decelPt:0.62+Math.random()*0.18,closeSpd0:1.8+Math.random()*1.0,angMomentum:0.12+Math.random()*0.38,microDevProb:0.12+Math.random()*0.22};}
  var holes=[];
  function generateWeaknessMap(){var nC=2+Math.floor(Math.random()*2),centers=[];for(var c=0;c<nC;c++)centers.push({x:PX+PW_COL*(0.12+Math.random()*0.76),y:PY+PH_COL*(0.08+Math.random()*0.84)});var total=8+Math.floor(Math.random()*5),pts=[];for(var i=0;i<total;i++){var ctr=centers[Math.floor(Math.random()*nC)];var wx=ctr.x+(Math.random()-0.5)*PW_COL*0.55,wy=ctr.y+(Math.random()-0.5)*PH_COL*0.28;wx=Math.max(PX+3,Math.min(PX+PW_COL-3,wx));wy=Math.max(PY+3,Math.min(PY+PH_COL-3,wy));var dEdge=Math.min(wx-PX,(PX+PW_COL)-wx),wt=dEdge<10?1.4:1.0;for(var h=0;h<holes.length;h++){var hdx=wx-holes[h].x,hdy=wy-holes[h].y;if(Math.sqrt(hdx*hdx+hdy*hdy)<22){wt*=1.6;break;}}pts.push({x:wx,y:wy,w:wt});}return pts;}
  function pickCrackOrigin(wmap,seeds){var typeR=Math.random(),sx,sy,angle;if(typeR<0.15){var left=Math.random()>0.5;sx=left?PX+1+Math.random()*3:PX+PW_COL-1-Math.random()*3;sy=PY+PH_COL*(0.12+Math.random()*0.76);angle=left?(Math.random()-0.5)*0.12:Math.PI+(Math.random()-0.5)*0.12;}else if(typeR<0.52){var left=Math.random()>0.5;sx=left?PX+1+Math.random()*3:PX+PW_COL-1-Math.random()*3;sy=PY+PH_COL*(0.10+Math.random()*0.80);if(wmap.length>0){var tw=wmap[Math.floor(Math.random()*wmap.length)];angle=Math.atan2(tw.y-sy,tw.x-sx)+(Math.random()-0.5)*0.45;}else{angle=(left?0:Math.PI)+(Math.random()-0.5)*0.6;}}else{if(wmap.length>0){var totalW=0;for(var i=0;i<wmap.length;i++)totalW+=wmap[i].w;var r=Math.random()*totalW,acc=0,chosen=wmap[0];for(var i=0;i<wmap.length;i++){acc+=wmap[i].w;if(r<=acc){chosen=wmap[i];break;}}sx=chosen.x+(Math.random()-0.5)*10;sy=chosen.y+(Math.random()-0.5)*10;}else{sx=PX+PW_COL*(0.12+Math.random()*0.76);sy=PY+PH_COL*(0.12+Math.random()*0.62);}angle=Math.random()*Math.PI*2;if(Math.random()<0.35)angle=Math.PI*0.3+(Math.random()-0.5)*1.2;}return {x:Math.max(PX,Math.min(PX+PW_COL,sx)),y:Math.max(PY,Math.min(PY+PH_COL,sy)),angle:angle};}
  function CrackBranch(sx,sy,angle,maxLen,baseW,depth,seeds){this.depth=depth||0;this.baseW=Math.max(baseW,0.12);this.maxLen=maxLen;this.drawnLen=0;this.done=false;this.closing=false;this.children=[];this.startAtPx=0;this.started=false;this.startEl=0;var sd=seeds||{devFreq:3,devAmp:Math.PI/15,bifProb:0.35,bifInterval:15,maxDepth:3,speedBase:7000+Math.random()*2000,speedDecay:0.60,branchWidthR:0.54,branchLenR:0.33+Math.random()*0.26,accelPt:0.30,decelPt:0.70,closeSpd0:2.4,angMomentum:0.25,microDevProb:0.25};this._accelPt=sd.accelPt;this._decelPt=sd.decelPt;this.spd=(maxLen/(sd.speedBase+Math.random()*1200))*Math.pow(sd.speedDecay,this.depth);this.closeSpd=this.depth===0?sd.closeSpd0:1.6;function gauss(amp){var u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)*amp;}this.pts=[{x:sx,y:sy,irr:(Math.random()-0.5)*0.5,isBranch:false}];var cx=sx,cy=sy,a=angle,devStep=sd.devFreq+Math.floor(Math.random()*3),nextBranchAt=sd.bifInterval*(0.8+Math.random()*0.4),angMom=0;for(var i=1;i<=(maxLen|0)+2;i++){if(i%devStep===0){var dev=gauss(sd.devAmp)+angMom*sd.angMomentum;a+=dev;angMom=dev*0.28;devStep=sd.devFreq+Math.floor(Math.random()*3);if(Math.random()<sd.microDevProb)a+=(Math.random()-0.5)*sd.devAmp*0.45;}cx+=Math.cos(a);cy+=Math.sin(a);cy=Math.max(PY-8,Math.min(PY+PH_COL+8,cy));cx=Math.max(4,Math.min(W-4,cx));var isBranch=false;if(this.depth<sd.maxDepth&&i>=(nextBranchAt|0)&&Math.random()<sd.bifProb){var dir=(Math.random()>0.5?1:-1);var ba=a+dir*(0.18+Math.random()*0.58);var bLen=maxLen*(sd.branchLenR*(0.75+Math.random()*0.5));var child=new CrackBranch(cx,cy,ba,bLen,baseW*sd.branchWidthR,depth+1,seeds);child.startAtPx=i;this.children.push(child);isBranch=true;nextBranchAt=i+sd.bifInterval*(0.8+Math.random()*0.6);}this.pts.push({x:cx,y:cy,irr:(Math.random()-0.5)*0.5,isBranch:isBranch});}}
  CrackBranch.prototype.tick=function(dt,el){if(this.closing){this.drawnLen=Math.max(0,this.drawnLen-this.closeSpd*dt);for(var c=0;c<this.children.length;c++)this.children[c].tick(dt,0);return;}if(this.done)return;var dtMs=dt*16.67,p=this.drawnLen/this.maxLen,ap=this._accelPt||0.30,dp=this._decelPt||0.70;var sf=p<ap?0.22+(p/ap)*0.78:p<dp?1.0:1.0-((p-dp)/(1.0-dp))*0.65;var vi=this.drawnLen|0;if(vi<this.pts.length){var cp=this.pts[vi];for(var h=0;h<holes.length;h++){var hdx=cp.x-holes[h].x,hdy=cp.y-holes[h].y;if(hdx*hdx+hdy*hdy<100){sf*=1.6;break;}}}var prev=this.drawnLen;this.drawnLen=Math.min(this.maxLen,this.drawnLen+dtMs*this.spd*sf);if(this.drawnLen>=this.maxLen)this.done=true;var growth=this.drawnLen-prev;if(growth>0&&Math.random()<0.25*growth){var ni=Math.min((this.drawnLen|0),this.pts.length-1);microDust.push(new MicroDust(this.pts[ni].x,this.pts[ni].y));}for(var c=0;c<this.children.length;c++){var ch=this.children[c];if(this.drawnLen>=ch.startAtPx){if(!ch.started){ch.started=true;ch.startEl=el;}ch.tick(dt,el-ch.startEl);}}};
  CrackBranch.prototype.draw=function(ctx){var vis=Math.min((this.drawnLen|0)+1,this.pts.length);if(vis<2)return;var lx=[],ly=[],rx=[],ry=[];for(var i=0;i<vis;i++){var pt=this.pts[i],t=i/Math.max(this.maxLen,1),hw=this.baseW*(1.0-t*0.92)+0.08,tx,ty;if(i<vis-1){tx=this.pts[i+1].x-pt.x;ty=this.pts[i+1].y-pt.y;}else{tx=pt.x-this.pts[Math.max(0,i-1)].x;ty=pt.y-this.pts[Math.max(0,i-1)].y;}var tl=Math.sqrt(tx*tx+ty*ty)||1,nx=-ty/tl,ny=tx/tl;lx.push(pt.x+nx*(hw*0.62)+pt.irr);ly.push(pt.y+ny*(hw*0.62));rx.push(pt.x-nx*(hw*0.48)-pt.irr*0.5);ry.push(pt.y-ny*(hw*0.48));}ctx.save();ctx.lineCap='round';ctx.lineJoin='round';if(this.depth===0){ctx.beginPath();ctx.moveTo(this.pts[0].x,this.pts[0].y);for(var i=1;i<vis;i++)ctx.lineTo(this.pts[i].x,this.pts[i].y);ctx.strokeStyle='rgba(46,51,72,0.22)';ctx.lineWidth=this.baseW*3.5+2.5;ctx.stroke();}ctx.beginPath();ctx.moveTo(lx[0],ly[0]);for(var i=1;i<vis;i++)ctx.lineTo(lx[i],ly[i]);for(var i=vis-1;i>=0;i--)ctx.lineTo(rx[i],ry[i]);ctx.closePath();ctx.fillStyle='#050508';ctx.fill();ctx.beginPath();ctx.moveTo(lx[0],ly[0]);for(var i=1;i<vis;i++)ctx.lineTo(lx[i],ly[i]);ctx.strokeStyle='rgba(0,0,0,0.82)';ctx.lineWidth=0.7;ctx.stroke();ctx.beginPath();ctx.moveTo(rx[0],ry[0]);for(var i=1;i<vis;i++)ctx.lineTo(rx[i],ry[i]);ctx.strokeStyle='rgba(255,255,255,0.055)';ctx.lineWidth=0.45;ctx.stroke();ctx.fillStyle='#1a1e2e';for(var i=0;i<vis;i+=2){ctx.fillRect(lx[i],ly[i],1,1);if(i+1<vis)ctx.fillRect((lx[i]+lx[i+1])*0.5+0.5,(ly[i]+ly[i+1])*0.5,1,1);}ctx.restore();for(var c=0;c<this.children.length;c++){if(this.drawnLen>=this.children[c].startAtPx)this.children[c].draw(ctx);}};
  function randomSize(){var r=Math.random();return r<0.70?1.5+Math.random()*2.5:r<0.90?5+Math.random()*5:11+Math.random()*7;}
  function makeFragPts(sz){var sides=4+Math.floor(Math.random()*4),pts=[],baseA=Math.random()*Math.PI*2;for(var i=0;i<sides;i++){var a=baseA+(i/sides)*Math.PI*2+(Math.random()-0.5)*(Math.PI*2/sides)*0.38,r=sz*(0.42+Math.random()*0.58);pts.push([Math.cos(a)*r,Math.sin(a)*r]);}return pts;}
  function Crumb(x,y,sz){this.x=x;this.y=y;this.ox=x;this.oy=y;this.sz=sz;var micro=sz<4,small=sz>=4&&sz<10;this.gravity=micro?0.022:small?0.065:0.125;this.airX=micro?0.984:small?0.995:0.998;this.rotSpd=micro?(0.05+Math.random()*0.10)*((Math.random()>0.5?1:-1)):small?(0.012+Math.random()*0.015)*((Math.random()>0.5?1:-1)):(0.005+Math.random()*0.015)*((Math.random()>0.5?1:-1));this.fadeOut=micro;this.pile=!micro;var side=x<PX+PW_COL/2?-1:1;this.vx=side*(0.2+Math.random()*0.6);this.vy=0.12+Math.random()*0.35;this.rot=Math.random()*Math.PI*2;this.alpha=1;this.settled=false;this.returning=false;this.retSpeed=0;this.arrived=false;this.pts=makeFragPts(sz);var v=(Math.random()-0.5)*16;this.cr=(26+v)|0;this.cg=(30+v)|0;this.cb=(46+v)|0;this.fr=(58+(Math.random()-0.5)*10)|0;this.fg=(62+(Math.random()-0.5)*10)|0;this.fb=(85+(Math.random()-0.5)*10)|0;}
  Crumb.prototype.update=function(dt){if(this.returning||this.settled)return;this.vy+=this.gravity*dt;this.vx*=Math.pow(this.airX,dt);this.x+=this.vx*dt;this.y+=this.vy*dt;this.rot+=this.rotSpd*dt;this.rotSpd*=Math.pow(0.9985,dt);var ground=H*0.88,fadeStart=ground-22;if(this.fadeOut&&this.y>fadeStart){this.alpha=Math.max(0,1-(this.y-fadeStart)/22);if(this.alpha<=0.01){this.settled=true;return;}}if(!this.fadeOut&&this.y>=ground-22){this.vy*=0.55;this.vx*=0.75;if(this.y>=ground){this.y=ground;this.vy=0;this.vx=0;this.rotSpd=0;this.settled=true;}}};
  Crumb.prototype.updateReturn=function(dt){if(!this.returning||this.arrived)return;var dx=this.ox-this.x,dy=this.oy-this.y,dist=Math.sqrt(dx*dx+dy*dy);if(dist<2){this.x=this.ox;this.y=this.oy;this.arrived=true;return;}this.retSpeed=Math.min(this.retSpeed+0.7*dt,15);var s=Math.min(this.retSpeed*dt,dist);this.x+=dx/dist*s;this.y+=dy/dist*s;this.alpha=Math.min(1,this.alpha+0.06*dt);this.rot-=this.rotSpd*dt*0.6;};
  Crumb.prototype.draw=function(ctx,alpha){if(this.alpha<0.01||this.pts.length<3)return;var a=alpha!==undefined?alpha:this.alpha;ctx.save();ctx.globalAlpha=a;ctx.translate(this.x,this.y);ctx.rotate(this.rot);var hw=0;for(var i=0;i<this.pts.length;i++)hw=Math.max(hw,Math.abs(this.pts[i][0]));hw=Math.max(hw,1);var gr=ctx.createLinearGradient(-hw,0,hw,0);gr.addColorStop(0,'rgb('+this.fr+','+this.fg+','+this.fb+')');gr.addColorStop(0.5,'rgb('+this.cr+','+this.cg+','+this.cb+')');gr.addColorStop(1,'rgb('+(this.cr-10)+','+(this.cg-8)+','+(this.cb-12)+')');ctx.beginPath();ctx.moveTo(this.pts[0][0],this.pts[0][1]);for(var i=1;i<this.pts.length;i++)ctx.lineTo(this.pts[i][0],this.pts[i][1]);ctx.closePath();ctx.fillStyle=gr;ctx.fill();if(this.sz>=4){ctx.fillStyle='rgba(255,255,255,0.028)';for(var k=0;k<3;k++){var gx=(Math.random()-0.5)*hw*1.4,gy=(Math.random()-0.5)*hw*1.4;ctx.fillRect(gx,gy,1,1);}}ctx.restore();};
  function MicroDust(x,y){this.x=x;this.y=y;var a=Math.random()*Math.PI*2,sp=0.3+Math.random()*0.6;this.vx=Math.cos(a)*sp;this.vy=Math.sin(a)*sp;this.life=0.4+Math.random()*0.25;this.ml=this.life;}
  MicroDust.prototype.upd=function(dt){this.vx+=(Math.random()-0.5)*0.08;this.vy+=(Math.random()-0.5)*0.08;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.028*dt;};
  MicroDust.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml)*0.35;ctx.fillStyle='rgba(100,116,139,'+a+')';ctx.fillRect(this.x,this.y,1,1);};
  function DustPuff(x,y){this.x=x;this.y=y;var a=Math.random()*Math.PI*2,sp=0.5+Math.random()*1.2;this.vx=Math.cos(a)*sp;this.vy=Math.sin(a)*sp-0.4;this.life=0.55+Math.random()*0.35;this.ml=this.life;this.r=0.8+Math.random()*1.2;}
  DustPuff.prototype.upd=function(dt){this.vy+=0.015*dt;this.vx+=(Math.random()-0.5)*0.06;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.022*dt;};
  DustPuff.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml)*0.30;ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle='rgba(100,116,139,'+a+')';ctx.fill();};
  function Spark(x,y){this.x=x;this.y=y;this.vx=(Math.random()-0.5)*2.3;this.vy=-(Math.random()*2.6+0.3);this.life=0.7+Math.random()*0.4;this.ml=this.life;this.r=0.7+Math.random()*2.1;this.col=Math.random()>0.5?'#0d9488':'#14b8a6';}
  Spark.prototype.upd=function(dt){this.vy+=0.028*dt;this.vx+=(Math.random()-0.5)*0.06;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.020*dt;};
  Spark.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml);ctx.save();ctx.globalAlpha=a;ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle=this.col;ctx.fill();ctx.restore();};
  function addHole(x,y,r){holes.push({x:x,y:y,r:Math.max(1,r),a:1});}
  function drawHoles(belowY){holes.forEach(function(h){if(belowY!==undefined&&h.y>belowY)return;ctx.save();ctx.globalAlpha=h.a*0.80;ctx.beginPath();ctx.arc(h.x,h.y,h.r,0,Math.PI*2);ctx.fillStyle='rgba(8,9,12,0.94)';ctx.fill();ctx.restore();});}
  function spawnDustCloud(x,y,n){for(var i=0;i<n;i++)dustPuffs.push(new DustPuff(x,y));}
  function spawnCrumbAt(x,y){var sz=randomSize();crumbs.push(new Crumb(x,y,sz));if(sz>=4)addHole(x,y,sz*0.55);spawnDustCloud(x,y,3+Math.floor(sz*0.5));}
  function collectSpawnPts(){var pts=[];function fromBranch(ck){var vis=Math.min((ck.drawnLen|0)+1,ck.pts.length);for(var i=0;i<vis;i++){var p=ck.pts[i];if(Math.abs(p.x-PX)<9||Math.abs(p.x-(PX+PW_COL))<9)pts.push({x:p.x,y:p.y,w:4});if(p.isBranch)pts.push({x:p.x,y:p.y,w:3});}for(var c=0;c<ck.children.length;c++){if(ck.drawnLen>=ck.children[c].startAtPx)fromBranch(ck.children[c]);}}cracks.forEach(fromBranch);for(var i=0;i<cracks.length-1;i++){var viA=Math.min((cracks[i].drawnLen|0)+1,cracks[i].pts.length);for(var j=i+1;j<cracks.length;j++){var viB=Math.min((cracks[j].drawnLen|0)+1,cracks[j].pts.length);for(var pi=0;pi<viA;pi+=4){for(var pj=0;pj<viB;pj+=4){var dx=cracks[i].pts[pi].x-cracks[j].pts[pj].x,dy=cracks[i].pts[pi].y-cracks[j].pts[pj].y;if(dx*dx+dy*dy<64)pts.push({x:(cracks[i].pts[pi].x+cracks[j].pts[pj].x)*0.5,y:(cracks[i].pts[pi].y+cracks[j].pts[pj].y)*0.5,w:5});}}}}return pts;}
  var state=S.PAUSE,stateStart=0,crumbs=[],pile=[],cracks=[],microDust=[],dustPuffs=[],sparks=[];
  var crumbCount=0,CRUMB_THRESH=15,crumbTimer=0,weaknessMap=[],cycleSeeds=null;
  var crackSchedule=[4000,10000,16000],crackCreated=[false,false,false];
  var finalCrack=null,scanY=undefined,glowA=0,flashA=0,regenStart=0;
  function setClosingRecursive(ck){ck.closing=true;for(var c=0;c<ck.children.length;c++)setClosingRecursive(ck.children[c]);}
  function enter(s,now){state=s;stateStart=now;if(s===S.DEGRADING){crumbs=[];pile=[];cracks=[];microDust=[];dustPuffs=[];sparks=[];holes=[];crumbCount=0;finalCrack=null;crackCreated=[false,false,false];cycleSeeds=generateCrackSeeds();weaknessMap=generateWeaknessMap();crumbTimer=(PRM?2500:900)+Math.random()*300;scanY=undefined;glowA=0;flashA=0;}if(s===S.REGEN){regenStart=now;pile.forEach(function(c){c.returning=true;c.retSpeed=0;crumbs.push(c);});pile=[];crumbs.forEach(function(c){if(!c.settled){c.returning=true;c.retSpeed=0;}});cracks.forEach(setClosingRecursive);if(finalCrack)setClosingRecursive(finalCrack);for(var i=0;i<20;i++)sparks.push(new Spark(PX+Math.random()*PW_COL,PY+PH_COL*0.2+Math.random()*PH_COL*0.6));scanY=PY+PH_COL;glowA=0;flashA=1.0;}}
  var lastTs=0,ready=false;
  function loop(ts){if(!ready){layout();computeTexture();ready=true;stateStart=ts;lastTs=ts;}var dt=Math.min((ts-lastTs)/16.67,3.5);lastTs=ts;var el=ts-stateStart;ctx.clearRect(0,0,W,H);var gy=H*0.88;ctx.strokeStyle='rgba(28,33,48,0.60)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();var dg=ctx.createLinearGradient(0,gy,0,gy+16);dg.addColorStop(0,'rgba(28,33,48,0.18)');dg.addColorStop(1,'rgba(28,33,48,0)');ctx.fillStyle=dg;ctx.fillRect(0,gy,W,16);pile.forEach(function(c){c.draw(ctx,1);});
    switch(state){
      case S.PAUSE:drawPillar(undefined,0);if(el>2400)enter(S.DEGRADING,ts);break;
      case S.DEGRADING:crumbTimer-=dt*16.67;if(crumbTimer<=0){crumbTimer=(PRM?2600:880)+Math.random()*380;var spPts=collectSpawnPts();if(spPts.length>0){var sp=spPts[Math.floor(Math.random()*spPts.length)];spawnCrumbAt(sp.x+(Math.random()-0.5)*6,sp.y+(Math.random()-0.5)*6);}else{var side=Math.random()>0.5;spawnCrumbAt(side?PX+Math.random()*4:PX+PW_COL-Math.random()*4,PY+PH_COL*0.1+Math.random()*PH_COL*0.8);}crumbCount++;}for(var ci=0;ci<crackSchedule.length;ci++){if(!crackCreated[ci]&&el>crackSchedule[ci]){crackCreated[ci]=true;var orig=pickCrackOrigin(weaknessMap,cycleSeeds);var ck=new CrackBranch(orig.x,orig.y,orig.angle,42+Math.random()*32,1.5,0,cycleSeeds);ck.startTime=ts;cracks.push(ck);}}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});for(var i=crumbs.length-1;i>=0;i--){crumbs[i].update(dt);if(crumbs[i].settled&&crumbs[i].pile){pile.push(crumbs[i]);crumbs.splice(i,1);}}for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}for(var i=dustPuffs.length-1;i>=0;i--){dustPuffs[i].upd(dt);if(dustPuffs[i].life<=0)dustPuffs.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});dustPuffs.forEach(function(p){p.draw(ctx);});drawPillar(undefined,0);drawHoles(undefined);cracks.forEach(function(c){c.draw(ctx);});crumbs.forEach(function(c){c.draw(ctx);});if(crumbCount>=CRUMB_THRESH)enter(S.THRESHOLD,ts);break;
      case S.THRESHOLD:if(!finalCrack&&el>180){var fy=PY+PH_COL*(0.43+(Math.random()-0.5)*0.12);finalCrack=new CrackBranch(PX-BASE_EXT-8,fy,0,PW_COL+BASE_EXT*2+18,1.8,0);finalCrack.spd=(PW_COL+BASE_EXT*2+18)/1500;finalCrack.startTime=ts;}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});if(finalCrack)finalCrack.tick(dt,ts-finalCrack.startTime);for(var i=crumbs.length-1;i>=0;i--){crumbs[i].update(dt);if(crumbs[i].settled&&crumbs[i].pile){pile.push(crumbs[i]);crumbs.splice(i,1);}}for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}for(var i=dustPuffs.length-1;i>=0;i--){dustPuffs[i].upd(dt);if(dustPuffs[i].life<=0)dustPuffs.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});dustPuffs.forEach(function(p){p.draw(ctx);});drawPillar(undefined,0);drawHoles(undefined);cracks.forEach(function(c){c.draw(ctx);});if(finalCrack)finalCrack.draw(ctx);crumbs.forEach(function(c){c.draw(ctx);});if(finalCrack&&finalCrack.done&&el>1800)enter(S.REGEN,ts);break;
      case S.REGEN:var rel=ts-regenStart;flashA=Math.max(0,1-rel/300);if(rel>300){var rp=Math.min((rel-300)/2200,1);scanY=PY+PH_COL-rp*PH_COL;glowA=Math.sin(rp*Math.PI);holes.forEach(function(h){if(h.y>scanY)h.a=Math.max(0,h.a-0.05*dt);});}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});if(finalCrack)finalCrack.tick(dt,ts-finalCrack.startTime);crumbs.forEach(function(c){if(c.returning)c.updateReturn(dt);else c.update(dt);});for(var i=sparks.length-1;i>=0;i--){sparks[i].upd(dt);if(sparks[i].life<=0)sparks.splice(i,1);}crumbs.forEach(function(c){if(c.arrived&&Math.random()<0.025*dt){sparks.push(new Spark(c.ox,c.oy));c.arrived=false;}});for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});drawPillar(scanY,glowA);drawHoles(scanY);cracks.forEach(function(c){c.draw(ctx);});if(finalCrack)finalCrack.draw(ctx);crumbs.forEach(function(c){c.draw(ctx);});sparks.forEach(function(p){p.draw(ctx);});if(flashA>0&&!PRM){var fg=ctx.createRadialGradient(PX+PW_COL/2,PY+PH_COL/2,0,PX+PW_COL/2,PY+PH_COL/2,W*0.7);fg.addColorStop(0,'rgba(13,148,136,'+(flashA*0.55)+')');fg.addColorStop(0.45,'rgba(13,148,136,'+(flashA*0.12)+')');fg.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=fg;ctx.fillRect(0,0,W,H);}if(rel>2500)enter(S.PAUSE,ts);break;
    }
    requestAnimationFrame(loop);
  }
  var rsT=null;
  window.addEventListener('resize',function(){clearTimeout(rsT);rsT=setTimeout(function(){layout();computeTexture();if(state===S.PAUSE||state===S.DEGRADING)enter(S.PAUSE,performance.now());},180);},{passive:true});
  setTimeout(function(){requestAnimationFrame(loop);},400);
})();
</script>
</body></html>"""


LOGIN_HTML = _AUTH_HEAD + """
    <div class="form-logo" data-t="login_title">Connexion</div>
    <div class="form-tagline" data-t="tagline">Plateforme de maintenance prédictive</div>
    {% if error %}<div class="auth-err">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
      <label class="flbl" for="em" data-t="lbl_email">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-t="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="••••••••" autocomplete="current-password" required>
      <button type="submit" class="btn-submit" data-t="btn_login">Se connecter</button>
    </form>
    <div class="auth-link" data-t="link_reg">Pas encore de compte ? <a href="/register">Créer un compte</a></div>
""" + _AUTH_FOOT


REGISTER_HTML = _AUTH_HEAD + """
    {% if pending %}
    <div class="verify-box">
      <div class="verify-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 01-2.06 0L2 7"/></svg></div>
      <div class="verify-title" data-t="verify_title">Vérifiez votre email</div>
      <div class="verify-sub" data-t="verify_sub">Un lien de confirmation a été envoyé. Cliquez dessus pour activer votre compte.</div>
      {% if resent|default(False) %}<div style="margin-top:12px;font-size:12px;color:#059669" data-t="resent">Email renvoyé !</div>{% endif %}
      <div class="verify-note" data-t="verify_note">Valide 24h · Vérifiez vos spams</div>
      <form method="POST" action="/resend-verification">
        <input type="hidden" name="email" value="{{ pending_email|default('') }}">
        <button type="submit" class="btn-resend" data-t="resend">Renvoyer l'email</button>
      </form>
    </div>
    <div class="auth-link"><a href="/login" data-t="back_login">Retour à la connexion</a></div>
    {% else %}
    <div class="form-logo" data-t="reg_title">Créer un compte</div>
    <div class="form-tagline" data-t="tagline">Plateforme de maintenance prédictive</div>
    {% if error %}<div class="auth-err">{{ error }}</div>{% endif %}
    <form method="POST" action="/register">
      <label class="flbl" for="em" data-t="lbl_email">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-t="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="8 caractères minimum" autocomplete="new-password" required minlength="8">
      <label class="flbl" for="pw2" data-t="lbl_pw2">Confirmer le mot de passe</label>
      <input class="fi" type="password" id="pw2" name="password2" placeholder="••••••••" autocomplete="new-password" required>
      <button type="submit" class="btn-submit" data-t="btn_reg">Créer mon compte</button>
    </form>
    <div class="auth-link" data-t="link_login">Déjà un compte ? <a href="/login">Se connecter</a></div>
    {% endif %}
""" + _AUTH_FOOT


ADMIN_HTML = """<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilar Admin</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--surface:#0e1118;--surface2:#141820;--surface3:#1e2433;
  --border:#1e2433;--border2:#252d3d;
  --teal:#0d9488;--teal2:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;--green-dim:rgba(5,150,105,0.08);
  --amber:#d97706;--purple:#bf5af2;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --r:12px;--r-sm:8px;--r-xs:6px;
}
body{font-family:-apple-system,'SF Pro Text','Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:0 0 60px}
nav{background:rgba(14,17,24,0.96);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.logo{font-size:13px;font-weight:700;letter-spacing:0.1em;color:var(--teal2);text-transform:uppercase}
.nav-right{display:flex;gap:8px;align-items:center}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:20px}
.kc{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px}
.kv{font-size:26px;font-weight:800;letter-spacing:-0.02em;font-variant-numeric:tabular-nums}
.kl{font-size:10px;font-weight:600;color:var(--text3);letter-spacing:0.06em;text-transform:uppercase;margin-top:4px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;margin-bottom:14px}
.ctitle{font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:var(--text3);margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}
.toolbar{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
.toolbar input,.toolbar select{background:rgba(120,120,128,0.15);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:var(--r-sm);font-size:12px;outline:none;transition:border-color .15s}
.toolbar input{flex:1;min-width:180px}
.toolbar input:focus,.toolbar select:focus{border-color:var(--teal)}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 12px;color:var(--text3);font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:middle;color:var(--text2)}
tr:last-child td{border:none}
tr:hover td{background:rgba(255,255,255,.015)}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;white-space:nowrap}
.b-free{background:rgba(235,235,245,0.07);color:var(--text3);border:1px solid var(--border)}
.b-starter{background:var(--teal-dim);color:var(--teal2);border:1px solid rgba(13,148,136,0.25)}
.b-pro{background:rgba(191,90,242,0.1);color:var(--purple);border:1px solid rgba(191,90,242,0.25)}
.b-admin{background:rgba(217,119,6,0.1);color:var(--amber);border:1px solid rgba(217,119,6,0.25)}
.b-ok{background:var(--green-dim);color:var(--green);border:1px solid rgba(5,150,105,0.25)}
.b-warn{background:rgba(217,119,6,0.1);color:var(--amber);border:1px solid rgba(217,119,6,0.25)}
.b-exp{background:var(--red-dim);color:var(--red);border:1px solid rgba(220,38,38,0.25)}
.b-ban{background:var(--red-dim);color:var(--red);border:1px solid rgba(220,38,38,0.25)}
.btn{padding:6px 11px;border:none;border-radius:var(--r-sm);font-size:11px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:4px;transition:opacity .15s,border-color .15s;white-space:nowrap}
.btn-teal{background:var(--teal);color:#fff}.btn-teal:hover{opacity:.85}
.btn-ghost{background:var(--surface2);border:1px solid var(--border2);color:var(--text2)}.btn-ghost:hover{border-color:var(--teal);color:var(--teal2)}
.btn-red{background:var(--red-dim);border:1px solid rgba(220,38,38,0.3);color:var(--red)}.btn-red:hover{background:rgba(220,38,38,0.16)}
.btn-amber{background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.25);color:var(--amber)}.btn-amber:hover{opacity:.8}
.actions{display:flex;gap:4px;flex-wrap:wrap}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:100;align-items:center;justify-content:center;padding:16px;backdrop-filter:blur(14px)}
.overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border2);border-radius:var(--r);padding:28px;width:100%;max-width:480px;max-height:90vh;overflow-y:auto;box-shadow:0 24px 64px rgba(0,0,0,.65)}
.modal-h{font-size:15px;font-weight:700;color:var(--text);margin-bottom:4px}
.modal-sub{font-size:12px;color:var(--text3);margin-bottom:20px;word-break:break-all}
label{font-size:10px;font-weight:700;color:var(--text3);display:block;margin-bottom:5px;letter-spacing:0.04em;text-transform:uppercase}
.fi{width:100%;background:rgba(120,120,128,0.15);border:1px solid var(--border);color:var(--text);padding:10px 12px;border-radius:var(--r-sm);font-size:13px;margin-bottom:14px;outline:none;transition:border-color .15s}
.fi:focus{border-color:var(--teal)}
.plan-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px}
.plan-opt{padding:12px 6px;border:2px solid var(--border);border-radius:var(--r-sm);text-align:center;cursor:pointer;font-size:12px;font-weight:700;transition:all .15s;background:var(--surface2);color:var(--text3)}.plan-opt:hover{border-color:var(--teal2)}
.sel-free{border-color:var(--text3)!important;background:rgba(100,116,139,0.08)!important;color:var(--text2)!important}
.sel-starter{border-color:var(--teal)!important;background:var(--teal-dim)!important;color:var(--teal2)!important}
.sel-pro{border-color:var(--purple)!important;background:rgba(191,90,242,0.1)!important;color:var(--purple)!important}
.date-row{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
.date-row input{flex:1;min-width:120px;margin:0}
.msg{font-size:11px;padding:8px 12px;border-radius:var(--r-sm);margin-bottom:12px;display:none}
.msg.ok{background:var(--green-dim);border:1px solid rgba(48,209,88,0.25);color:var(--green);display:block}
.msg.err{background:var(--red-dim);border:1px solid rgba(255,69,58,0.25);color:var(--red);display:block}
.mfooter{display:flex;gap:10px;margin-top:4px}
.mfooter .btn{flex:1;justify-content:center;padding:12px}
.block-row{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.block-row input{background:rgba(120,120,128,0.15);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:var(--r-sm);font-size:12px;outline:none;flex:1;min-width:160px;transition:border-color .15s}
.block-row input:focus{border-color:var(--teal)}
.empty{color:var(--text3);font-size:12px;text-align:center;padding:14px;background:rgba(255,255,255,.02);border-radius:var(--r-sm)}
.term-wrap{background:var(--bg);border:1px solid var(--border);border-radius:var(--r-sm);overflow:hidden}
.term-hdr{display:flex;align-items:center;justify-content:space-between;padding:9px 14px;background:var(--surface2);border-bottom:1px solid var(--border)}
.term-dots{display:flex;gap:5px}.term-dots span{width:9px;height:9px;border-radius:50%}
.term-out{font-family:'SF Mono','Menlo','Consolas',monospace;font-size:11px;line-height:1.7;padding:12px 14px;min-height:160px;max-height:340px;overflow-y:auto;color:var(--text2);white-space:pre-wrap;word-break:break-all}
.term-out::-webkit-scrollbar{width:4px}.term-out::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:2px}
.l-ok{color:var(--green)}.l-err{color:var(--red)}.l-cmd{color:var(--teal2);font-weight:600}
.term-row{display:flex;align-items:center;gap:8px;padding:8px 14px;border-top:1px solid var(--border)}
.term-prompt{color:var(--teal2);font-family:'SF Mono','Menlo','Consolas',monospace;font-size:11px;white-space:nowrap}
.term-inp{flex:1;background:transparent;border:none;color:var(--text);font-family:'SF Mono','Menlo','Consolas',monospace;font-size:11px;outline:none}
.term-btn{padding:5px 12px;background:var(--teal);border:none;border-radius:var(--r-xs);color:#fff;font-size:11px;font-weight:700;cursor:pointer}.term-btn:hover{opacity:.85}.term-btn:disabled{opacity:.4;cursor:not-allowed}
.shortcuts{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
@media(max-width:640px){.kv{font-size:22px}.toolbar{flex-direction:column}table{display:block;overflow-x:auto}.modal{padding:18px}.plan-grid{grid-template-columns:1fr 1fr 1fr}}
</style>
</head>
<body>
<nav>
  <span class="logo">Pilar Admin</span>
  <div class="nav-right">
    <a href="/" class="btn btn-ghost">Application</a>
    <a href="/logout" class="btn btn-ghost">Deconnexion</a>
  </div>
</nav>
<div class="wrap">

<div class="kgrid">
  <div class="kc"><div class="kv" style="color:var(--teal2)">{{ total_users }}</div><div class="kl">Utilisateurs</div></div>
  <div class="kc"><div class="kv" style="color:var(--purple)">{{ paid_users }}</div><div class="kl">Payants</div></div>
  <div class="kc"><div class="kv" style="color:var(--green)">${{ mrr }}</div><div class="kl">MRR</div></div>
  <div class="kc"><div class="kv">{{ total_analyses }}</div><div class="kl">Analyses</div></div>
  <div class="kc"><div class="kv" style="color:var(--amber)">{{ expiring_soon }}</div><div class="kl">Expirent &lt;7j</div></div>
</div>

<div class="card">
  <div class="ctitle"><span>Gestion utilisateurs</span><span style="font-weight:400;font-size:11px;text-transform:none;letter-spacing:0;color:var(--text3)">{{ total_users }} compte{% if total_users != 1 %}s{% endif %}</span></div>
  <div class="toolbar">
    <input type="text" id="searchInput" placeholder="Rechercher par email..." oninput="filterTable()">
    <select id="planFilter" onchange="filterTable()">
      <option value="">Tous les plans</option>
      <option value="free">Free</option>
      <option value="starter">Starter</option>
      <option value="pro">Pro</option>
    </select>
  </div>
  <div class="tbl-wrap">
  <table id="usersTable">
    <thead><tr><th>Email</th><th>Plan</th><th>Expiration</th><th>Note</th><th>Quota</th><th>Analyses</th><th>Inscrit</th><th>Actions</th></tr></thead>
    <tbody>
    {% for u in users %}
    <tr data-email="{{ u.email|lower }}" data-plan="{{ u.plan }}">
      <td>
        <div style="font-weight:600;color:{% if u.is_banned %}var(--red){% else %}var(--text){% endif %};font-size:13px">{{ u.email }}</div>
        <div style="display:flex;gap:4px;margin-top:3px;flex-wrap:wrap">
          {% if u.is_admin %}<span class="badge b-admin">admin</span>{% endif %}
          {% if u.is_banned %}<span class="badge b-ban">banni</span>{% endif %}
        </div>
      </td>
      <td><span class="badge b-{{ u.plan }}">{{ u.plan }}</span></td>
      <td>{% if u.plan_expires_at %}{% if u.plan_expires_at < now %}<span class="badge b-exp">Expire {{ u.plan_expires_at.strftime('%d/%m/%y') }}</span>{% elif (u.plan_expires_at - now).days < 7 %}<span class="badge b-warn">{{ u.plan_expires_at.strftime('%d/%m/%y') }}</span>{% else %}<span style="font-size:11px">{{ u.plan_expires_at.strftime('%d/%m/%Y') }}</span>{% endif %}{% else %}<span style="color:var(--text3);font-size:11px">—</span>{% endif %}</td>
      <td style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:var(--text3)">{{ u.plan_note or '—' }}</td>
      <td style="text-align:center;font-size:12px">{{ u.machine_quota }}</td>
      <td style="text-align:center;font-size:12px">{{ u.analysis_count }}</td>
      <td style="font-size:11px;color:var(--text3);white-space:nowrap">{{ u.created_at.strftime('%d/%m/%Y') }}</td>
      <td>
        <div class="actions">
          <button class="btn btn-teal manage-btn"
            data-uid="{{ u.id }}"
            data-email="{{ u.email|e }}"
            data-plan="{{ u.plan }}"
            data-expires="{{ u.plan_expires_at.strftime('%Y-%m-%d') if u.plan_expires_at else '' }}"
            data-note="{{ u.plan_note|e if u.plan_note else '' }}"
            data-quota="{{ u.machine_quota }}">Gerer</button>
          <a href="/admin/impersonate/{{ u.id }}" class="btn btn-ghost">Voir</a>
          <button onclick="toggleAdmin({{ u.id }}, '{{ u.email|e }}')" class="btn {% if u.is_admin %}btn-red{% else %}btn-amber{% endif %}">{% if u.is_admin %}Admin -{% else %}Admin +{% endif %}</button>
          <button onclick="toggleBan({{ u.id }}, '{{ u.email|e }}', {{ 'true' if u.is_banned else 'false' }})" class="btn {% if u.is_banned %}btn-teal{% else %}btn-red{% endif %}">{% if u.is_banned %}Debannir{% else %}Bannir{% endif %}</button>
          <button onclick="deleteUser({{ u.id }}, '{{ u.email|e }}')" class="btn btn-red">Suppr.</button>
        </div>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>

<div class="card">
  <div class="ctitle">Emails bloques</div>
  <div class="block-row">
    <input id="blockEmailInput" placeholder="email@domaine.com" type="email">
    <input id="blockReasonInput" placeholder="Raison (optionnel)">
    <button onclick="blockEmail()" class="btn btn-red">Bloquer</button>
  </div>
  {% if banned_emails %}
  <div class="tbl-wrap"><table>
    <thead><tr><th>Email</th><th>Raison</th><th>Date</th><th></th></tr></thead>
    <tbody>{% for b in banned_emails %}
    <tr>
      <td style="color:var(--red);font-weight:600">{{ b.email }}</td>
      <td style="color:var(--text3);font-size:11px">{{ b.reason or '—' }}</td>
      <td style="color:var(--text3);font-size:11px;white-space:nowrap">{{ b.banned_at.strftime('%d/%m/%Y') }}</td>
      <td><button onclick="unblockEmail({{ b.id }}, '{{ b.email|e }}')" class="btn btn-ghost">Debloquer</button></td>
    </tr>{% endfor %}</tbody>
  </table></div>
  {% else %}<div class="empty">Aucun email bloque.</div>{% endif %}
</div>

<div class="card">
  <div class="ctitle">Fichiers utilisateurs</div>
  <div id="filesMsg" style="font-size:12px;color:var(--text3)">Chargement...</div>
  <div class="tbl-wrap"><table id="filesTable" style="display:none">
    <thead><tr><th>Fichier</th><th>Utilisateur</th><th>Lignes</th><th>Date</th><th></th></tr></thead>
    <tbody id="filesTbody"></tbody>
  </table></div>
</div>

<div class="card">
  <div class="ctitle">Terminal serveur</div>
  <div class="shortcuts">
    <button class="btn btn-ghost" onclick="termQuick('python --version')">python --version</button>
    <button class="btn btn-ghost" onclick="termQuick('pip list')">pip list</button>
    <button class="btn btn-ghost" onclick="termQuick('python -c &quot;from etape7 import app,db,User,Analysis;app.app_context().push();print(User.query.count(),&#39;users&#39;,Analysis.query.count(),&#39;analyses&#39;)&quot;')">stats DB</button>
    <button class="btn btn-ghost" onclick="termQuick('python -c &quot;import sys,platform;print(sys.version);print(platform.platform())&quot;')">systeme</button>
    <button class="btn btn-ghost" onclick="termQuick('python retrain_real.py')">retrain ML</button>
  </div>
  <div class="term-wrap">
    <div class="term-hdr">
      <div class="term-dots"><span style="background:var(--red)"></span><span style="background:var(--amber)"></span><span style="background:var(--green)"></span></div>
      <span style="font-size:10px;font-weight:600;color:var(--text3);letter-spacing:0.04em;text-transform:uppercase">PILAR SHELL</span>
      <button onclick="termClear()" style="background:transparent;border:1px solid var(--border);color:var(--text3);padding:3px 9px;border-radius:var(--r-xs);font-size:10px;cursor:pointer">Effacer</button>
    </div>
    <div class="term-out" id="termOut"><span style="color:var(--text3)">Pret.</span>
</div>
    <div class="term-row">
      <span class="term-prompt">pilar$&nbsp;</span>
      <input class="term-inp" id="termIn" type="text" placeholder="commande..." autocomplete="off" spellcheck="false" onkeydown="termKey(event)">
      <button class="term-btn" id="termBtn" onclick="termRun()">Run</button>
    </div>
  </div>
</div>

</div>

<div class="overlay" id="overlay" onclick="closeIfOut(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-h" id="modalTitle">Gerer l\u2019abonnement</div>
    <div class="modal-sub" id="modalSub"></div>
    <div class="msg" id="modalMsg"></div>
    <label>Plan</label>
    <div class="plan-grid">
      <div class="plan-opt" id="opt-free"    onclick="selectPlan('free')">Free<br><span style="font-size:9px;font-weight:400;opacity:.6">Limite</span></div>
      <div class="plan-opt" id="opt-starter" onclick="selectPlan('starter')">Starter<br><span style="font-size:9px;font-weight:400">$99/mo</span></div>
      <div class="plan-opt" id="opt-pro"     onclick="selectPlan('pro')">Pro<br><span style="font-size:9px;font-weight:400">$299/mo</span></div>
    </div>
    <label>Expiration</label>
    <div class="date-row">
      <input type="date" class="fi" id="expiresInput" style="margin:0">
      <button onclick="addMonths(1)" class="btn btn-ghost">+1 mois</button>
      <button onclick="addMonths(3)" class="btn btn-ghost">+3 mois</button>
      <button onclick="addMonths(12)" class="btn btn-ghost">+1 an</button>
      <button onclick="document.getElementById('expiresInput').value=''" class="btn btn-ghost">Vider</button>
    </div>
    <label>Quota machines</label>
    <input type="number" class="fi" id="quotaInput" min="0" max="999" style="width:100px">
    <label>Note paiement</label>
    <input type="text" class="fi" id="noteInput" placeholder="ex: Virement BNP ref 2026-031">
    <div class="mfooter">
      <button onclick="savePlan()" class="btn btn-teal" id="saveBtn">Enregistrer</button>
      <button onclick="closeModal()" class="btn btn-ghost">Annuler</button>
    </div>
  </div>
</div>

<script>
var _uid = null, _plan = 'free';

document.addEventListener('click', function(e) {
  var btn = e.target.closest('.manage-btn');
  if (btn) openModal(btn.dataset.uid, btn.dataset.email, btn.dataset.plan, btn.dataset.expires, btn.dataset.note, btn.dataset.quota);
});

function openModal(uid, email, plan, expires, note, quota) {
  _uid  = uid;
  _plan = plan;
  document.getElementById('modalTitle').textContent = 'Gerer l\u2019abonnement';
  document.getElementById('modalSub').textContent   = email;
  document.getElementById('expiresInput').value     = expires || '';
  document.getElementById('noteInput').value        = note    || '';
  document.getElementById('quotaInput').value       = quota   || '3';
  var msg = document.getElementById('modalMsg');
  msg.className   = 'msg';
  msg.textContent = '';
  selectPlan(plan);
  document.getElementById('overlay').classList.add('open');
}

function closeModal() { document.getElementById('overlay').classList.remove('open'); }
function closeIfOut(e) { if (e.target === document.getElementById('overlay')) closeModal(); }

function selectPlan(p) {
  _plan = p;
  ['free','starter','pro'].forEach(function(x) {
    var el = document.getElementById('opt-' + x);
    el.className = 'plan-opt' + (x === p ? ' sel-' + p : '');
  });
}

function addMonths(n) {
  var inp  = document.getElementById('expiresInput');
  var base = inp.value ? new Date(inp.value + 'T12:00:00') : new Date();
  base.setMonth(base.getMonth() + n);
  inp.value = base.toISOString().slice(0, 10);
}

async function savePlan() {
  var btn     = document.getElementById('saveBtn');
  var msg     = document.getElementById('modalMsg');
  var expires = document.getElementById('expiresInput').value;
  var note    = document.getElementById('noteInput').value;
  var quota   = parseInt(document.getElementById('quotaInput').value, 10);
  if (isNaN(quota) || quota < 0) quota = 3;
  btn.disabled    = true;
  msg.className   = 'msg';
  msg.textContent = '';
  try {
    var r = await fetch('/admin/set_plan/' + _uid, {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({plan: _plan, expires_at: expires, note: note})
    });
    var d = await r.json();
    if (!d.ok) {
      msg.className   = 'msg err';
      msg.textContent = d.error || 'Erreur serveur';
      btn.disabled    = false;
      return;
    }
    await fetch('/admin/set_quota/' + _uid, {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({quota: quota})
    });
    msg.className   = 'msg ok';
    msg.textContent = 'Enregistre avec succes.';
    setTimeout(function() { location.reload(); }, 900);
  } catch(err) {
    msg.className   = 'msg err';
    msg.textContent = 'Erreur reseau. Verifiez votre connexion et rechargez la page.';
    btn.disabled    = false;
  }
}

function filterTable() {
  var q  = document.getElementById('searchInput').value.toLowerCase();
  var pf = document.getElementById('planFilter').value;
  document.querySelectorAll('#usersTable tbody tr').forEach(function(row) {
    var email = (row.dataset.email || '').toLowerCase();
    var plan  = row.dataset.plan  || '';
    row.style.display = (!q || email.includes(q)) && (!pf || plan === pf) ? '' : 'none';
  });
}

async function toggleAdmin(uid, email) {
  if (!confirm('Modifier les droits admin de ' + email + ' ?')) return;
  try {
    var r = await fetch('/admin/toggle_admin/' + uid, {method:'POST',headers:{'Content-Type':'application/json'}});
    var d = await r.json();
    if (d.ok) location.reload(); else alert(d.error || 'Erreur');
  } catch(e) { alert('Erreur reseau'); }
}

async function toggleBan(uid, email, isBanned) {
  if (!confirm((isBanned ? 'Debannir' : 'Bannir') + ' le compte de ' + email + ' ?')) return;
  try {
    var r = await fetch('/admin/toggle_ban/' + uid, {method:'POST',headers:{'Content-Type':'application/json'}});
    var d = await r.json();
    if (d.ok) location.reload(); else alert(d.error || 'Erreur');
  } catch(e) { alert('Erreur reseau'); }
}

async function deleteUser(uid, email) {
  if (!confirm('Supprimer le compte de ' + email + ' ? Cette action est irreversible.')) return;
  try {
    var r = await fetch('/admin/delete_user/' + uid, {method:'POST',headers:{'Content-Type':'application/json'}});
    var d = await r.json();
    if (d.ok) location.reload(); else alert(d.error || 'Erreur');
  } catch(e) { alert('Erreur reseau'); }
}

async function blockEmail() {
  var email  = document.getElementById('blockEmailInput').value.trim();
  var reason = document.getElementById('blockReasonInput').value.trim();
  if (!email) { alert('Email requis'); return; }
  if (!confirm('Bloquer ' + email + ' ?')) return;
  try {
    var r = await fetch('/admin/block_email', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({email: email, reason: reason})
    });
    var d = await r.json();
    if (d.ok) location.reload(); else alert(d.error || 'Erreur');
  } catch(e) { alert('Erreur reseau'); }
}

async function unblockEmail(bid, email) {
  if (!confirm('Debloquer ' + email + ' ?')) return;
  try {
    var r = await fetch('/admin/unblock_email/' + bid, {method:'POST',headers:{'Content-Type':'application/json'}});
    var d = await r.json();
    if (d.ok) location.reload(); else alert(d.error || 'Erreur');
  } catch(e) { alert('Erreur reseau'); }
}

var _hist = [], _hi = -1;
function termKey(e) {
  if (e.key === 'Enter') { termRun(); return; }
  if (e.key === 'ArrowUp') { e.preventDefault(); if (!_hist.length) return; _hi = Math.min(_hi+1,_hist.length-1); document.getElementById('termIn').value = _hist[_hi]; }
  if (e.key === 'ArrowDown') { e.preventDefault(); if (_hi<=0){_hi=-1;document.getElementById('termIn').value='';return;} _hi--; document.getElementById('termIn').value=_hist[_hi]; }
}
async function termRun() {
  var inp = document.getElementById('termIn');
  var cmd = inp.value.trim();
  if (!cmd) return;
  _hist.unshift(cmd); _hi = -1; inp.value = '';
  var out = document.getElementById('termOut');
  var btn = document.getElementById('termBtn');
  btn.disabled = true;
  out.innerHTML += '<span class="l-cmd">$ ' + cmd.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
  out.scrollTop  = out.scrollHeight;
  try {
    var r = await fetch('/admin/terminal', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({cmd:cmd})
    });
    var d   = await r.json();
    var txt = (d.output || '(aucune sortie)').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    out.innerHTML += '<span class="' + (d.code===0?'l-ok':'l-err') + '">' + txt + '</span>\\n';
  } catch(e) { out.innerHTML += '<span class="l-err">Erreur reseau</span>\\n'; }
  btn.disabled  = false;
  out.scrollTop = out.scrollHeight;
  inp.focus();
}
function termClear() { document.getElementById('termOut').innerHTML = '<span style="color:var(--text3)">Terminal efface.</span>\\n'; }
function termQuick(cmd) { document.getElementById('termIn').value = cmd; termRun(); }

async function loadFiles() {
  try {
    var r = await fetch('/admin/files');
    var files = await r.json();
    var msg   = document.getElementById('filesMsg');
    var table = document.getElementById('filesTable');
    var tbody = document.getElementById('filesTbody');
    if (!files.length) { msg.textContent = 'Aucun fichier.'; return; }
    msg.style.display   = 'none';
    table.style.display = '';
    tbody.innerHTML = files.map(function(f) {
      return '<tr>'
        + '<td style="font-weight:600;color:var(--text)">'     + f.filename   + '</td>'
        + '<td style="color:var(--text3);font-size:11px">'     + f.user_email + '</td>'
        + '<td style="color:var(--text2)">'                    + f.rows       + '</td>'
        + '<td style="color:var(--text3);font-size:11px">'     + new Date(f.created_at).toLocaleDateString('fr') + '</td>'
        + '<td><a href="/admin/files/' + f.id + '/download" class="btn btn-ghost">Telecharger</a></td>'
        + '</tr>';
    }).join('');
  } catch(e) { document.getElementById('filesMsg').textContent = 'Erreur de chargement.'; }
}
loadFiles();
</script>
</body></html>"""

# ── CSS & HEAD ────────────────────────────────────────────────────────────────
_HEAD = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#07090f">
<link rel="icon" type="image/png" href="data:image/png;base64,{FAV}">
<title>Pilar</title>
<link rel="manifest" href="/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Pilar">
<script>
if('serviceWorker' in navigator){
  navigator.serviceWorker.getRegistrations().then(function(regs){
    return Promise.all(regs.map(function(r){return r.unregister();}));
  }).then(function(){
    navigator.serviceWorker.register('/sw.js');
  });
}
</script>

<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;}
:root{
--bg:#07090f;--surface:#0e1118;--surface2:#141820;--surface3:#1e2433;
--border:#1e2433;--border2:#252d3d;
--teal:#0d9488;--teal-light:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
--red:#dc2626;--red-dim:rgba(220,38,38,0.08);
--green:#059669;--green-dim:rgba(5,150,105,0.08);
--amber:#d97706;--purple:#bf5af2;
--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
--nav-h:64px;
--sidebar-w:200px;
--r:12px;--r-sm:8px;--r-xs:6px;
--shadow:0 4px 24px rgba(0,0,0,0.5);--shadow-sm:0 2px 8px rgba(0,0,0,0.4);
}
html,body{height:100%;overflow:hidden;}
body{font-family:'Inter','Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;}

/* ── Desktop Sidebar Layout ───────────────────────────────────────────────── */
.desktop-layout{display:flex;height:100vh;overflow:hidden;}
.sidebar{width:var(--sidebar-w,200px);background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;overflow:hidden;}
.sidebar-logo{padding:16px 16px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;}
.sidebar-logo .logo-mark{width:28px;height:28px;border-radius:7px;background:var(--teal);display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.sidebar-logo .logo-text{font-size:13px;font-weight:700;letter-spacing:3px;color:var(--teal-light);text-transform:uppercase;}
.sidebar-logo .logo-sub{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:1px;}
.sidebar-nav{flex:1;padding:8px 0;overflow-y:auto;}
.sidebar-nav::-webkit-scrollbar{width:0;}
.sidebar-section{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text3);padding:12px 16px 4px;}
.ni{display:flex;align-items:center;gap:10px;padding:8px 14px;color:var(--text3);font-size:12px;font-weight:500;text-decoration:none;border:none;background:none;cursor:pointer;width:100%;text-align:left;transition:all .15s;border-radius:0;position:relative;}
.ni:hover{background:var(--surface2);color:var(--text2);}
.ni.on{background:var(--teal-dim);color:var(--teal-light);font-weight:600;}
.ni.on::before{content:'';position:absolute;left:0;top:4px;bottom:4px;width:3px;background:var(--teal);border-radius:0 3px 3px 0;}
.ni svg{width:16px;height:16px;stroke-width:1.8;flex-shrink:0;}
.ni .ni-label{flex:1;}
.ni .ni-badge{background:var(--red);color:#fff;font-size:9px;font-weight:700;border-radius:10px;padding:1px 5px;min-width:16px;text-align:center;}
.sidebar-footer{border-top:1px solid var(--border);padding:10px 12px;display:flex;flex-direction:column;gap:4px;}
.sync-status-bar{display:flex;align-items:center;gap:6px;padding:6px 10px;background:var(--surface2);border-radius:7px;font-size:10px;color:var(--text3);}
.sync-dot{width:6px;height:6px;border-radius:50%;background:var(--text3);flex-shrink:0;}
.sync-dot.online{background:var(--green);animation:blink 2s infinite;}
.sync-dot.offline{background:var(--amber);}
.main-content{flex:1;display:flex;flex-direction:column;overflow:hidden;}
header{height:48px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;padding:0 20px;background:var(--surface);flex-shrink:0;}
.logo{font-size:13px;font-weight:700;letter-spacing:4px;color:var(--teal-light);text-transform:uppercase;}
.hd{width:1px;height:18px;background:var(--border2);}
.hsub{font-size:10px;letter-spacing:1.2px;color:var(--text3);text-transform:uppercase;}
.hright{margin-left:auto;display:flex;gap:6px;align-items:center;}
/* Bottom nav — fixed at bottom on mobile, hidden on desktop */
.bottom-nav{position:fixed;bottom:0;left:0;right:0;height:var(--nav-h);border-top:1px solid var(--border);background:var(--surface);display:flex;align-items:stretch;z-index:50;}
.page{padding-bottom:calc(var(--nav-h) + 8px);}
@media(min-width:768px){.bottom-nav{display:none;}.page{padding-bottom:0;}}
.lang-toggle{max-width:42px;color:var(--text3);}
.lang-toggle:hover span{color:var(--teal-light);}
.page{flex:1;overflow-y:auto;overflow-x:hidden;}
.page::-webkit-scrollbar{width:4px;}
.page::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.pad{padding:20px;padding-bottom:24px;max-width:1400px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px;margin-bottom:12px;box-shadow:var(--shadow-sm);}
.ctitle{font-size:11px;font-weight:600;letter-spacing:0.02em;color:var(--text2);text-transform:uppercase;margin-bottom:12px;}
.rh{display:flex;align-items:center;justify-content:space-between;padding:18px;border-radius:var(--r);border:1px solid var(--border);background:var(--surface);margin-bottom:12px;box-shadow:var(--shadow-sm);}
.rh.ok{border-color:var(--green);background:var(--green-dim);}
.rh.alert{border-color:var(--red);background:var(--red-dim);}
.rh.amber{border-color:var(--amber);background:rgba(255,159,10,0.1);}
.sb{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:0;}
.sb.ok{background:var(--green-dim);color:var(--green);}
.sb.alert{background:var(--red-dim);color:var(--red);}
.sb.amber{background:rgba(255,159,10,0.15);color:var(--amber);}
.dot{width:6px;height:6px;border-radius:50%;}
.dot.ok{background:var(--green);}.dot.alert{background:var(--red);animation:blink 1.2s infinite;}.dot.amber{background:var(--amber);}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
.rnum{font-size:48px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;}
.rnum.ok{color:var(--green);}.rnum.alert{color:var(--red);}.rnum.amber{color:var(--amber);}
.runit{font-size:20px;color:var(--text3);}
.rlbl{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;text-align:right;margin-top:3px;}
.tgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px;}
.tbtn{padding:10px 4px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);color:var(--text3);font-size:11px;cursor:pointer;text-align:center;transition:all 0.15s;}
.tbtn.on{border-color:var(--teal);background:var(--teal-dim);color:var(--teal-light);font-weight:600;}
.sensor{margin-bottom:16px;}
.srow{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.sname{font-size:12px;color:var(--text2);}
.vwrap{display:flex;align-items:center;gap:5px;}
.vi{width:72px;padding:4px 8px;background:rgba(120,120,128,0.18);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:15px;font-weight:600;text-align:right;outline:none;-webkit-appearance:none;}
.vi:focus{border-color:var(--teal);}
.vu{font-size:10px;color:var(--text3);}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;background:var(--surface3);border-radius:4px;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:var(--teal);cursor:pointer;border:2px solid rgba(255,255,255,0.15);box-shadow:0 2px 6px rgba(0,0,0,0.4);}
.rl{display:flex;justify-content:space-between;font-size:9px;color:var(--text3);margin-top:2px;}
.btn{width:100%;padding:14px;background:var(--teal);color:#fff;border:none;border-radius:12px;font-size:15px;font-weight:600;letter-spacing:0;text-transform:none;cursor:pointer;transition:opacity 0.15s;margin-top:8px;}
.btn:disabled{background:var(--surface3);color:var(--text3);cursor:not-allowed;}
.flbl{font-size:11px;font-weight:600;letter-spacing:0.02em;color:var(--text2);text-transform:uppercase;margin-bottom:7px;display:block;}
.fi{width:100%;padding:10px 12px;background:rgba(120,120,128,0.18);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:13px;outline:none;transition:border-color 0.15s;}
.fi:focus{border-color:var(--teal);}
.fi::placeholder{color:var(--text3);}
.zrow{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface2);border-radius:var(--r-sm);margin-bottom:6px;}
.zname{font-size:11px;color:var(--text2);flex:1;}
.zbw{width:80px;height:2px;background:var(--surface3);border-radius:1px;}
.zbf{height:100%;border-radius:1px;background:var(--red);}
.zp{font-size:12px;font-weight:700;color:var(--amber);min-width:34px;text-align:right;}
.kgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.kc{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px;box-shadow:var(--shadow-sm);}
.kv{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums;}
.kv.ok{color:var(--green);}.kv.alert{color:var(--red);}.kv.amber{color:var(--amber);}.kv.purple{color:var(--purple);}
.kl{font-size:11px;font-weight:600;letter-spacing:0.02em;color:var(--text3);text-transform:uppercase;margin-top:3px;}
.tw{overflow-x:auto;border-radius:var(--r);background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow-sm);}
table{width:100%;border-collapse:collapse;font-size:11px;min-width:480px;}
th{text-align:left;padding:10px 12px;color:var(--text2);font-size:11px;font-weight:600;letter-spacing:0.02em;border-bottom:1px solid var(--border);text-transform:uppercase;white-space:nowrap;}
td{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--text2);white-space:nowrap;}
tr:last-child td{border-bottom:none;}
.badge{padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0;}
.badge.ok{background:var(--green-dim);color:var(--green);}
.badge.alert{background:var(--red-dim);color:var(--red);}
.mb{padding:2px 8px;border-radius:20px;font-size:11px;background:var(--teal-dim);color:var(--teal-light);}
.cw{display:flex;flex-direction:column;height:100%;overflow:hidden;}
.cm{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}
.cm::-webkit-scrollbar{width:0;}
.msg{display:flex;flex-direction:column;gap:3px;max-width:85%;}
.msg.user{align-self:flex-end;align-items:flex-end;}
.msg.bot{align-self:flex-start;align-items:flex-start;}
.ms{font-size:8px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;}
.mb2{padding:10px 14px;border-radius:var(--r-sm);font-size:13px;line-height:1.65;}
.msg.user .mb2{background:var(--teal);color:#fff;border-radius:18px 18px 4px 18px;}
.msg.bot .mb2{background:var(--surface2);border:1px solid var(--border);color:var(--text2);border-radius:18px 18px 18px 4px;}
.typing{color:var(--text3);font-style:italic;}
.cia{padding:10px 14px;border-top:1px solid var(--border);display:flex;gap:8px;background:rgba(28,28,30,0.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);flex-shrink:0;}
.cta{flex:1;padding:10px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;color:var(--text);font-size:13px;outline:none;resize:none;font-family:inherit;max-height:100px;line-height:1.5;transition:border-color 0.15s;}
.cta:focus{border-color:var(--teal);}
.cta::placeholder{color:var(--text3);}
.bsend{padding:10px 16px;background:var(--teal);color:#fff;border:none;border-radius:12px;font-size:13px;font-weight:600;cursor:pointer;align-self:flex-end;transition:opacity 0.15s;flex-shrink:0;}
.bsend:disabled{background:var(--surface3);color:var(--text3);cursor:not-allowed;}
.ab{padding:10px 14px;background:var(--teal-dim);border:1px solid var(--teal);border-radius:var(--r-sm);font-size:11px;color:var(--teal-light);margin-bottom:10px;display:none;}
.nb{padding:5px 11px;background:transparent;border:1px solid var(--border2);border-radius:20px;color:var(--text3);font-size:11px;cursor:pointer;transition:all 0.15s;white-space:nowrap;}
.nb.on{border-color:var(--green);color:var(--green);}
.idle{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;gap:8px;color:var(--text3);text-align:center;}
.idle .l1{font-size:13px;}.idle .l2{font-size:11px;}
.lcard{flex:1;padding:14px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;text-align:center;transition:all 0.15s;}
.lcard.lactive{border-color:var(--teal);background:var(--teal-dim);}
.lcard .flag{font-size:26px;display:block;margin-bottom:6px;}
.lcard .lname{font-size:11px;font-weight:600;color:var(--text2);}
.lcard.lactive .lname{color:var(--teal-light);}
@keyframes slideInRight{from{transform:translateX(40px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes slideInLeft{from{transform:translateX(-40px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes slideInUp{from{transform:translateY(18px);opacity:0}to{transform:translateY(0);opacity:1}}
.slide-right{animation:slideInRight 0.22s cubic-bezier(0.25,0.46,0.45,0.94)}
.slide-left{animation:slideInLeft 0.22s cubic-bezier(0.25,0.46,0.45,0.94)}
.slide-up{animation:slideInUp 0.2s cubic-bezier(0.25,0.46,0.45,0.94)}
.lz-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:12px;box-shadow:var(--shadow-sm);}
.lz-empty{display:flex;flex-direction:column;align-items:center;padding:36px 24px;gap:14px;}
.lz-cta{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:16px;background:var(--teal);color:#fff;border:none;border-radius:12px;font-size:15px;font-weight:600;letter-spacing:0;cursor:pointer;transition:opacity 0.15s;}
.lz-cta:hover{opacity:0.85;}
.lz-hint{font-size:10px;color:var(--text3);letter-spacing:0.5px;text-align:center;line-height:1.7;}
.lz-conn{padding:16px;}
.lz-ch{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.lz-dot{width:8px;height:8px;border-radius:50%;background:var(--green);flex-shrink:0;animation:blink 1.5s infinite;}
.lz-fname{font-size:12px;color:var(--teal-light);font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.lz-disc{padding:5px 12px;background:transparent;border:1px solid var(--border2);border-radius:20px;color:var(--text3);font-size:11px;cursor:pointer;transition:border-color 0.15s;white-space:nowrap;}
.lz-disc:hover{border-color:var(--red);color:var(--red);}
.lz-sg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;}
.lz-si{background:var(--surface2);border-radius:var(--r-sm);padding:12px 8px;text-align:center;}
.lz-sv{display:block;font-size:26px;font-weight:800;font-variant-numeric:tabular-nums;color:var(--text);}
.lz-sl{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px;display:block;}
.lz-si.alert .lz-sv{color:var(--red);}
.lz-si.ok .lz-sv{color:var(--green);}
.lz-foot{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;}
.lz-ck{font-size:10px;color:var(--text3);flex:1;}
.lz-sel{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-xs);color:var(--text2);font-size:10px;padding:4px 8px;outline:none;}
.man-hdr{display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none;margin-bottom:0;}
.man-chv{width:14px;height:14px;stroke:var(--text3);fill:none;flex-shrink:0;transition:transform 0.2s;}
.ai-chip{display:flex;align-items:center;gap:6px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-sm);padding:7px 12px;flex:1;min-width:0;}
.ai-dot{width:7px;height:7px;border-radius:50%;background:var(--teal);flex-shrink:0;}
.ai-label{font-size:10px;font-weight:700;letter-spacing:1px;color:var(--text2);white-space:nowrap;}
.ai-val{font-size:9px;color:var(--text3);margin-left:auto;white-space:nowrap;}
</style>
<script>
var T={
fr:{nav_monitor:'Live Monitor',nav_twin:'Twin',nav_history:'Historique',nav_account:'Compte',nav_settings:'Réglages',
page_monitor:'Monitor',page_twin:'Jumeau Numérique',page_history:'Historique',page_account:'Compte',page_settings:'Réglages',
idle_l1:'Aucune analyse',idle_l2:'Configurez ci-dessous et lancez',
machine_class:'Classe machine',sensor_params:'Paramètres capteurs',
air_temp:'Température air',proc_temp:'Température process',rot_speed:'Vitesse rotation',torque:'Couple',tool_wear:'Usure outil',
p_vibration:'Vibration',p_temp_palier:'Temp. palier',p_debit:'Débit',p_pression_e:'Pression entrée',p_pression_s:'Pression sortie',p_courant:'Courant moteur',p_temp_moteur:'Temp. moteur',p_heure:'Heures fonct.',
run_btn:"Lancer l'analyse",zone_title:'Analyse zones de panne',
status_ok:'Fonctionnement normal',status_alert:'Anomalie détectée',failure_prob:'Prob. panne',
u_temp:'°C',u_speed:'m³/h',u_torque:'bar',u_wear:'h',
r_ta_min:'21.9°C',r_ta_max:'31.9°C',r_tp_min:'31.9°C',r_tp_max:'41.9°C',
r_v_min:'1000',r_v_max:'3000',r_c_min:'3',r_c_max:'80N·m',r_u_min:'0',r_u_max:'4.17h',
twin_loading:'Chargement simulation...',twin_no_data:'Aucune donnée',twin_no_data2:"Lancez d'abord une analyse dans Monitor",twin_go:'Aller à Monitor',
twin_healthy:'Système sain',twin_failure:'Panne dans ~',twin_trend:'Tendance\u00a0:',twin_cur_risk:'Risque actuel',twin_avg:'Risque moyen',twin_anom:"Taux d'anomalie",
twin_c_risk:'Risque — Historique + Simulation 24h',twin_c_wear:'Projection heures fonct.',twin_c_temp:'Température palier',twin_c_sim:'Simulateur de scénario',
twin_speed:'Débit (m³/h)',twin_torque:'Pression sortie (bar)',twin_wear:'Heures fonct. (h)',twin_airtemp:'Vibration (mm/s)',twin_sim:'Simuler',twin_sim_r:'Risque simulé',
hist_total:'Total',hist_anom:'Anomalies',hist_avg:'Risque moy.',hist_alerts:'Alertes envoyées',hist_reliability:'Fiabilité',
hist_time:'Heure',hist_class:'Classe',hist_risk:'Risque',hist_status:'Statut',hist_zones:'Zones',hist_alert:'Alerte',hist_feedback:'Retour',
hist_anomaly:'Anomalie',hist_ok:'OK',hist_sent:'Envoyé',hist_reliability_hint:'Notez les alertes avec +/- pour mesurer la précision du modèle sur vos données.',
set_email:"Email d'alerte",set_email_lbl:'Adresse destinataire',set_email_ph:'maintenance@entreprise.com',set_email_btn:'Enregistrer',set_saved:'Enregistré',
set_notif:'Notifications navigateur',set_notif_desc:'Recevez des alertes quand le risque dépasse 50%.',set_notif_btn:'Activer les notifications',set_notif_on:'Notifications activées',set_notif_blocked:'Bloqué — Activez dans les réglages',
set_sys:'Infos système',set_version:'Version',set_aimodel:'Modèle IA',set_db:'Base de données',set_lang:'Langue',
acc_guest_title:'Mode invité',acc_guest_desc:"Connectez-vous pour sauvegarder vos données,<br>rejoindre une équipe et accéder à la collaboration.",
acc_signin:'Se connecter',acc_register:'Créer un compte',acc_card_title:'Compte',acc_signout:'Déconnexion',
acc_team_title:'Équipe',acc_no_team_desc:"Créez une équipe pour partager les analyses, les scores d'anomalie et les prévisions RUL en temps réel. Jusqu'à 2 responsables peuvent gérer l'équipe.",
acc_create_ph:"Nom de l'équipe (optionnel)",acc_create_btn:'Créer une équipe',
acc_role_leader:'Responsable',acc_role_member:'Membre',acc_you:'(vous)',
acc_promote:'Promouvoir',acc_kick:'Retirer',
acc_add_title:'Ajouter un membre',acc_add_ph:'email@entreprise.com',acc_add_btn:'Ajouter',acc_added:'Membre ajouté !',
acc_members:' membre(s)',acc_leave:"Quitter l'équipe",
nav_import:'Manuel',nav_assistant:'Assistant',
live_connect:'Connecter fichier',live_on:'Live actif',live_disconnect:'Déconnecter',
live_rows:'Lignes lues',live_fail:'Pannes',live_ok:'Normal',live_last:'Dernier',
live_no_api:'Sélectionnez votre fichier CSV. Cliquez Actualiser pour relire les nouvelles lignes.',
live_refresh:'Actualiser',
tut_format:'Format CSV',tut_cols:'Colonne',tut_unit:'Unité',tut_range:'Plage',tut_desc:'Description',
tut_sample:'Télécharger exemple CSV',tut_import:'Importer un fichier',tut_speed:'Vitesse',tut_start:'Démarrer',
tut_pause:'Pause',tut_resume:'Reprendre',tut_stop:'Arrêter',tut_done:'Terminé',
tut_live:'Moniteur fichier live',tut_live_desc:"Connectez un fichier CSV mis à jour par votre SCADA. Pilar détecte les nouvelles lignes automatiquement.",
tut_connect:'Connecter fichier',tut_no_file:'Aucun fichier connecté',tut_disconnected:'Déconnecté',
ast_placeholder:'Posez votre question sur la machine...',ast_send:'Envoyer',
ast_hello:"Bonjour. Je suis votre assistant maintenance prédictive. Partagez vos relevés capteurs ou posez-moi vos questions.",
csv_detect:'Colonnes détectées',csv_bad:'Colonnes non reconnues',csv_rows:'lignes',
live_hint:'CSV · noms de colonnes libres · conversion auto',manual_title:'Analyse manuelle',
select_machine:'Choisissez votre machine',adv_params:'Paramètres avancés',not_listed:'Ma machine n\u2019est pas listée',
idle_l2b:'Sélectionnez une machine ci-dessous et lancez l\u2019analyse',
custom_machine_title:'Machine personnalisée',custom_machine_desc:'Décrivez votre machine ci-dessous. Nous l\u2019intégrerons à Pilar sous 48h.',
cust_name_lbl:'Nom / type de machine',cust_mfr_lbl:'Fabricant',cust_rpm_lbl:'Plage débit typique (m³/h)',cust_torque_lbl:'Plage pression (bar)',cust_desc_lbl:'Description',
cust_submit:'Envoyer la demande',cust_sent:'Demande reçue. Votre machine sera intégrée sous 48h.',
partial_analysis:'Analyse partielle',imputed:'estimés',discovered_param:'Paramètre découvert',
page_adapter:'Adaptateur CSV',adp_upload:'Importer CSV',adp_hint:'Tout délimiteur · noms libres · conversion unités incluse',
adp_change:'Changer',adp_preview:'Aperçu (5 lignes)',adp_download:'Convertir & Télécharger',
adp_no_map:'Aucun champ mappé',adp_source:'Colonne source',adp_samples:'Exemples',adp_field:'Champ Pilar',adp_unit:'Unité',
adp_ignore:'(Ignorer)',adp_desc:'Convertissez n\u2019importe quel CSV au format Pilar',adp_save:'Sauvegarder dans Pilar',adp_my_files:'Mes fichiers',adp_refresh:'Actualiser',adp_saved_ok:'Fichier sauvegardé',adp_no_files:'Aucun fichier sauvegardé',adp_load:'Charger',adp_delete:'Supprimer',
page_tutorial:'Import & Analyse',tut_csv_desc:"Votre CSV doit contenir ces colonnes (l'ordre n'a pas d'importance) :",tut_click_csv:'Cliquez pour sélectionner un fichier CSV',tut_no_file_sel:'Aucun fichier sélectionné',tut_progress:'Progression',tut_check_every:'Vérifier toutes les',tut_row:'Ligne',
set_domain:'Domaine',set_nav_title:'Navigation',set_twin_nav:'Jumeau Numérique',
ast_you:'Vous',ast_pilar:'Pilar IA',ast_error:'Erreur\u00a0: ',ast_net_error:'Erreur réseau. Réessayez.',
ai_warming:'se réchauffe',ai_ready:'prêt',ai_in_twin:'Jumeau Numérique',fl_title:'Vue Flotte',fl_sub:'Surveillance de vos pompes centrifuges',fl_add:'+ Ajouter une machine',fl_analyze_all:'Analyser tout',fl_analyzing:'Analyse en cours…',fl_total:'Machines actives',fl_alerts:'Alertes actives',fl_avg:'Risque moyen flotte',fl_critical:'Critiques',fl_never:'Jamais',fl_badge_ok:'Normal',fl_badge_warn:'Attention',fl_badge_crit:'Critique',fl_badge_inactive:'Inactif',fl_badge_new:'Sans données',fl_btn_analyze:'Analyser',fl_btn_csv:'Importer CSV',fl_btn_edit:'Modifier',fl_btn_rerun:'Relancer',fl_modal_add:'Ajouter une machine',fl_modal_edit:'Modifier la machine',fl_save:'Enregistrer',fl_cancel:'Annuler',fl_save_changes:'Sauvegarder les modifications',fl_name_lbl:'Nom de la machine',fl_name_ph:'ex. POMPE-01, Compresseur-A3',fl_name_hint:'Ce nom est utilisé comme machine_id dans l’API.',fl_desc_lbl:'Description',fl_desc_ph:'Optionnel — emplacement, modèle, notes',fl_class_sel:'Classe machine',fl_thresh_lbl:'Seuil d’alerte %',fl_thresh_hint:'L’alerte se déclenche au-dessus de ce seuil.',fl_pump_sel:'Type de pompe',fl_fluid_sel:'Type de fluide',fl_mat_sel:'Matériau de la roue',fl_email_lbl:'Email d’alerte principal',fl_email_ph:'tech@entreprise.com',fl_esc_lbl:'Email d’escalade',fl_esc_ph:'responsable@entreprise.com',fl_esc_hint:'Si l’alerte n’est pas acquittée sous 30 min, ce contact est notifié.',fl_upload_title:'Importer des données machine',fl_upload_desc:'CSV exporté depuis votre SCADA, API ou système capteurs. Pilar détecte les colonnes automatiquement et analyse chaque ligne. Résultats sauvegardés dans l’historique.',fl_upload_click:'Cliquez pour sélectionner un fichier CSV',fl_upload_hint:'Tout délimiteur · noms libres · conversion d’unités auto',fl_upload_run:'Lancer l’analyse',fl_upload_other:'Importer un autre fichier',fl_upload_rows:'Lignes',fl_upload_failures:'Anomalies',fl_upload_peak:'Risque max',fl_upload_avg:'Risque moy.',fl_upload_history:'Sauvegardé dans l’historique',fl_an_run:'Lancer l’analyse',fl_an_again:'Relancer',fl_an_close:'Fermer',fl_an_zones:'Zones de défaillance',fl_an_no_zone:'Aucune zone spécifique',fl_an_failure:'⚠ PANNE',fl_an_normal:'✓ NORMAL',fl_an_anom:'Score anomalie',fl_an_rul:'Délai estimé',fl_del_confirm_msg:'Supprimer cette machine ? Action irréversible.',fl_del_file_msg:'Supprimer ce fichier de la machine ?',fl_ap_title:'Analyse de la flotte',fl_ap_confirm:'Lancer l’analyse sur {n} machine(s) avec fichier CSV ?',fl_ap_done:'Analyse terminée',fl_ap_rows:'lignes',fl_ap_anom:'anomalies',fl_ap_avg:'risque moy.',fl_ap_skip:'ignorée',fl_ap_close:'Fermer',fl_empty_title:'Aucune machine configurée',fl_empty_desc:'Ajoutez votre première machine pour démarrer la surveillance de la flotte.',fl_add_first:'+ Ajouter ma première machine',fl_class_lbl:'Classe',fl_pump_lbl:'Pompe',fl_fluid_lbl:'Fluide',fl_mat_lbl:'Matériau',fl_thresh_card:'Seuil',fl_last_lbl:'Dernière mesure',fl_no_file_msg:'Aucune machine avec fichier CSV. Utilisez le bouton Importer CSV sur chaque carte.',fl_full_page:'Page flotte complète →',fl_loading:'Chargement de la flotte…',fl_load_fail:'Erreur de chargement. ',fl_error:'Erreur',fl_net_error:'Erreur réseau. Réessayez.',fl_retry:'Réessayer',fl_risk_lbl:'RISQUE'},
en:{nav_monitor:'Live Monitor',nav_twin:'Twin',nav_history:'History',nav_account:'Account',nav_settings:'Settings',
page_monitor:'Monitor',page_twin:'Digital Twin',page_history:'History',page_account:'Account',page_settings:'Settings',
idle_l1:'No analysis yet',idle_l2:'Configure below and run',
machine_class:'Machine class',sensor_params:'Sensor parameters',
air_temp:'Air temperature',proc_temp:'Process temperature',rot_speed:'Rotational speed',torque:'Torque',tool_wear:'Tool wear',
p_vibration:'Vibration',p_temp_palier:'Bearing temp.',p_debit:'Flow rate',p_pression_e:'Inlet pressure',p_pression_s:'Outlet pressure',p_courant:'Motor current',p_temp_moteur:'Motor temp.',p_heure:'Run hours',
run_btn:'Run Analysis',zone_title:'Failure zone analysis',
status_ok:'Normal Operation',status_alert:'Anomaly Detected',failure_prob:'Failure prob.',
u_temp:'°C',u_speed:'m\u00b3/h',u_torque:'bar',u_wear:'h',
r_ta_min:'295K',r_ta_max:'305K',r_tp_min:'305K',r_tp_max:'315K',
r_v_min:'1000',r_v_max:'3000',r_c_min:'3',r_c_max:'80Nm',r_u_min:'0',r_u_max:'250',
twin_loading:'Loading simulation...',twin_no_data:'No data yet',twin_no_data2:'Run an analysis on Monitor first',twin_go:'Go to Monitor',
twin_healthy:'System Healthy',twin_failure:'Failure in ~',twin_trend:'Trend:',twin_cur_risk:'Current risk',twin_avg:'Avg risk',twin_anom:'Anomaly rate',
twin_c_risk:'Risk \u2014 History + 24h Simulation',twin_c_wear:'Run hours projection',twin_c_temp:'Bearing temperature',twin_c_sim:'Scenario Simulator',
twin_speed:'Flow rate (m³/h)',twin_torque:'Outlet pressure (bar)',twin_wear:'Run hours (h)',twin_airtemp:'Vibration (mm/s)',twin_sim:'Simulate',twin_sim_r:'Simulated risk',
hist_total:'Total',hist_anom:'Anomalies',hist_avg:'Avg risk',hist_alerts:'Alerts sent',hist_reliability:'Reliability',
hist_time:'Time',hist_class:'Class',hist_risk:'Risk',hist_status:'Status',hist_zones:'Zones',hist_alert:'Alert',hist_feedback:'Feedback',
hist_anomaly:'Anomaly',hist_ok:'OK',hist_sent:'Sent',hist_reliability_hint:'Rate alerts with +/- to track model accuracy on your data.',
set_email:'Alert email',set_email_lbl:'Recipient address',set_email_ph:'maintenance@company.com',set_email_btn:'Save Email',set_saved:'Saved',
set_notif:'Browser notifications',set_notif_desc:'Receive alerts when failure risk exceeds 50%.',set_notif_btn:'Enable Notifications',set_notif_on:'Notifications Enabled',set_notif_blocked:'Blocked \u2014 Enable in Browser Settings',
set_sys:'System info',set_version:'Version',set_aimodel:'AI Model',set_db:'Database',set_lang:'Language',
acc_guest_title:'Guest Mode',acc_guest_desc:'Sign in to save your data,<br>join a team and access collaboration.',
acc_signin:'Sign In',acc_register:'Create Account',acc_card_title:'Account',acc_signout:'Sign Out',
acc_team_title:'Team',acc_no_team_desc:'Create a team to share live AI insights — anomaly scores, SHAP explanations, and RUL forecasts — with your colleagues. Up to 2 leaders can manage the team.',
acc_create_ph:'Team name (optional)',acc_create_btn:'Create Team',
acc_role_leader:'Leader',acc_role_member:'Member',acc_you:'(you)',
acc_promote:'Promote',acc_kick:'Remove',
acc_add_title:'Add Member',acc_add_ph:'email@company.com',acc_add_btn:'Add',acc_added:'Member added!',
acc_members:' member(s)',acc_leave:'Leave Team',
nav_import:'Manual',nav_assistant:'Assistant',
live_connect:'Connect File',live_on:'Live ON',live_disconnect:'Disconnect',
live_rows:'Rows read',live_fail:'Failures',live_ok:'Normal',live_last:'Last',
live_no_api:'Select your CSV file. Click Refresh to reload new rows.',
live_refresh:'Refresh',
tut_format:'CSV Format',tut_cols:'Column',tut_unit:'Unit',tut_range:'Range',tut_desc:'Description',
tut_sample:'Download sample CSV',tut_import:'Import File',tut_speed:'Speed',tut_start:'Start',
tut_pause:'Pause',tut_resume:'Resume',tut_stop:'Stop',tut_done:'Done',
tut_live:'Live File Monitor',tut_live_desc:'Connect a CSV file updated by your SCADA or system. Pilar detects new rows automatically.',
tut_connect:'Connect File',tut_no_file:'No file connected',tut_disconnected:'Disconnected',
ast_placeholder:'Ask about your machine...',ast_send:'Send',
ast_hello:'Hello. I am your predictive maintenance assistant. Share your sensor readings or ask me anything about your machine health.',
csv_detect:'Columns detected',csv_bad:'Columns not recognized',csv_rows:'rows',
live_hint:'CSV · any column names · auto unit conversion',manual_title:'Manual Analysis',
select_machine:'Select your machine',adv_params:'Advanced parameters',not_listed:'My machine is not listed',
idle_l2b:'Select a machine below and run analysis',
custom_machine_title:'Custom Machine',custom_machine_desc:'Describe your machine below. We will integrate it into Pilar within 48 hours.',
cust_name_lbl:'Machine name / type',cust_mfr_lbl:'Manufacturer',cust_rpm_lbl:'Typical flow range (m³/h)',cust_torque_lbl:'Pressure range (bar)',cust_desc_lbl:'Description',
cust_submit:'Submit Request',cust_sent:'Request received. Your machine will be integrated within 48 hours.',
partial_analysis:'Partial analysis',imputed:'estimated',discovered_param:'Discovered parameter',
page_adapter:'CSV Adapter',adp_upload:'Upload CSV',adp_hint:'Any delimiter · any column names · unit conversion included',
adp_change:'Change',adp_preview:'Preview (5 rows)',adp_download:'Convert & Download',
adp_no_map:'No fields mapped',adp_source:'Source column',adp_samples:'Samples',adp_field:'Pilar field',adp_unit:'Unit',
adp_ignore:'(Ignore)',adp_desc:'Convert any CSV to Pilar format',adp_save:'Save to Pilar',adp_my_files:'My files',adp_refresh:'Refresh',adp_saved_ok:'File saved',adp_no_files:'No saved files',adp_load:'Load',adp_delete:'Delete',
page_tutorial:'Import & Run',tut_csv_desc:'Your CSV must contain these columns (order does not matter):',tut_click_csv:'Click to select a CSV file',tut_no_file_sel:'No file selected',tut_progress:'Progress',tut_check_every:'Check every',tut_row:'Row',
set_domain:'Domain',set_nav_title:'Navigation',set_twin_nav:'Digital Twin',
ast_you:'You',ast_pilar:'Pilar AI',ast_error:'Error: ',ast_net_error:'Network error. Please retry.',
ai_warming:'warming up',ai_ready:'ready',ai_in_twin:'Digital Twin',fl_title:'Fleet Overview',fl_sub:'Monitor your centrifugal pumps',fl_add:'+ Add Machine',fl_analyze_all:'Analyze All',fl_analyzing:'Analyzing…',fl_total:'Active Machines',fl_alerts:'Active Alerts',fl_avg:'Avg Fleet Risk',fl_critical:'Critical',fl_never:'Never',fl_badge_ok:'Normal',fl_badge_warn:'Warning',fl_badge_crit:'Critical',fl_badge_inactive:'Inactive',fl_badge_new:'No data',fl_btn_analyze:'Analyze',fl_btn_csv:'Upload CSV',fl_btn_edit:'Edit',fl_btn_rerun:'Re-run',fl_modal_add:'Add Machine',fl_modal_edit:'Edit Machine',fl_save:'Save',fl_cancel:'Cancel',fl_save_changes:'Save Changes',fl_name_lbl:'Machine Name',fl_name_ph:'e.g. PUMP-01, Compressor-A3',fl_name_hint:'Used as machine_id when sending data via API.',fl_desc_lbl:'Description',fl_desc_ph:'Optional — location, model, notes',fl_class_sel:'Machine Class',fl_thresh_lbl:'Alert threshold %',fl_thresh_hint:'Alert fires above this threshold.',fl_pump_sel:'Pump Type',fl_fluid_sel:'Fluid Type',fl_mat_sel:'Impeller Material',fl_email_lbl:'Primary Alert Email',fl_email_ph:'tech@yourcompany.com',fl_esc_lbl:'Escalation Email',fl_esc_ph:'manager@yourcompany.com',fl_esc_hint:'If primary alert is unacknowledged after 30 min, this contact is notified.',fl_upload_title:'Upload Machine Data',fl_upload_desc:'Upload a CSV exported from your SCADA, PLC, or sensor system. Pilar auto-detects columns and runs a full analysis on every row. Results are saved to history.',fl_upload_click:'Click to select a CSV file',fl_upload_hint:'Any delimiter · any column names · auto unit conversion',fl_upload_run:'Run Analysis',fl_upload_other:'Upload another file',fl_upload_rows:'Rows',fl_upload_failures:'Failures',fl_upload_peak:'Peak risk',fl_upload_avg:'Avg risk',fl_upload_history:'Saved to history',fl_an_run:'Run Analysis',fl_an_again:'Run Again',fl_an_close:'Close',fl_an_zones:'Failure Zones',fl_an_no_zone:'No specific zone',fl_an_failure:'⚠ FAILURE',fl_an_normal:'✓ NORMAL',fl_an_anom:'Anomaly score',fl_an_rul:'Est. time to failure',fl_del_confirm_msg:'Delete this machine? This cannot be undone.',fl_del_file_msg:'Delete this file from the machine?',fl_ap_title:'Fleet Analysis',fl_ap_confirm:'Run analysis on {n} machine(s) with CSV file?',fl_ap_done:'Analysis complete',fl_ap_rows:'rows',fl_ap_anom:'anomalies',fl_ap_avg:'avg risk',fl_ap_skip:'skipped',fl_ap_close:'Close',fl_empty_title:'No machines yet',fl_empty_desc:'Add your first machine to start monitoring your fleet.',fl_add_first:'+ Add First Machine',fl_class_lbl:'Class',fl_pump_lbl:'Pump',fl_fluid_lbl:'Fluid',fl_mat_lbl:'Material',fl_thresh_card:'Threshold',fl_last_lbl:'Last seen',fl_no_file_msg:'No machines with CSV attached. Use the Upload CSV button on each card.',fl_full_page:'Full fleet page →',fl_loading:'Loading fleet…',fl_load_fail:'Failed to load fleet. ',fl_error:'Error',fl_net_error:'Network error. Please retry.',fl_retry:'Retry',fl_risk_lbl:'RISK'}
};
var LANG=localStorage.getItem('pilar_lang')||'en';
function t(k){return(T[LANG]&&T[LANG][k])||(T.en[k])||k;}
function setLang(l){LANG=l;localStorage.setItem('pilar_lang',l);applyLang();}
function applyLang(){
  try{
  document.querySelectorAll('[data-i18n]').forEach(function(el){
    var k=el.getAttribute('data-i18n');
    if(el.tagName==='INPUT'){el.placeholder=t(k);}else{el.textContent=t(k);}
  });
  document.querySelectorAll('[data-i18n-html]').forEach(function(el){el.innerHTML=t(el.getAttribute('data-i18n-html'));});
  document.querySelectorAll('.lcard').forEach(function(c){c.classList.toggle('lactive',c.dataset.lang===LANG);});
  document.querySelectorAll('td .badge.alert').forEach(function(el){el.textContent=t('hist_anomaly');});
  document.querySelectorAll('td .badge.ok').forEach(function(el){el.textContent=t('hist_ok');});
  document.querySelectorAll('td .mb').forEach(function(el){el.textContent=t('hist_sent');});
  try{updateSensorUnits();}catch(e){}
  document.querySelectorAll('.acc-role').forEach(function(el){el.textContent=t(el.dataset.role==='leader'?'acc_role_leader':'acc_role_member');});
  document.querySelectorAll('[data-i18n-count]').forEach(function(el){var n=el.dataset.icount||'0';el.textContent=n+t(el.getAttribute('data-i18n-count'));});
  }catch(e){console.warn('[PILAR] applyLang error:',e);}
}
function toDisplay(raw,type){
  if(LANG!=='fr')return raw;
  if(type==='temp')return+(raw-273.15).toFixed(2);
  if(type==='wear')return+(raw/60).toFixed(3);
  return raw;
}
function toRaw(disp,type){
  if(LANG!=='fr')return disp;
  if(type==='temp')return+(disp+273.15).toFixed(2);
  if(type==='wear')return+(disp*60).toFixed(1);
  return disp;
}
function updateSensorUnits(){
  // Pump sensors use fixed SI units — no locale conversion needed
}
document.addEventListener('DOMContentLoaded',applyLang);
var _tz=Intl.DateTimeFormat().resolvedOptions().timeZone;
function localTime(utcStr,opts){
  return new Date(utcStr).toLocaleString(undefined,Object.assign({timeZone:_tz},opts||{}));
}
function localTimeNow(){
  return new Date().toLocaleString(undefined,{timeZone:_tz});
}

// ── PAGE TRANSITIONS ─────────────────────────────────────────────────────────
var _PAGE_ORDER={'/monitor':0,'/tutorial':1,'/history':2,'/dashboard':3,'/account':4,'/settings':5};
var _curPage=window.location.pathname;
var _prevIdx=parseInt(sessionStorage.getItem('pilar_page_idx')||'0');
var _curIdx=_PAGE_ORDER[_curPage]!==undefined?_PAGE_ORDER[_curPage]:_prevIdx;
sessionStorage.setItem('pilar_page_idx',_curIdx);
window.addEventListener('DOMContentLoaded',function(){
  var page=document.querySelector('.page');
  if(page&&_curIdx!==_prevIdx){
    page.classList.add(_curIdx>_prevIdx?'slide-right':'slide-left');
  }
});
document.querySelectorAll('.ni').forEach(function(link){
  link.addEventListener('click',function(e){
    var href=link.getAttribute('href');
    if(!href||href==='/javascript:void(0)')return;
    var toIdx=_PAGE_ORDER[href];
    if(toIdx===undefined)return;
    sessionStorage.setItem('pilar_page_idx',_curIdx);
  });
});

// ── SLIDER NUMBER ANIMATION ───────────────────────────────────────────────────
function animateNum(el,from,to,decimals,duration){
  var start=null;
  from=parseFloat(from);to=parseFloat(to);
  function step(ts){
    if(!start)start=ts;
    var p=Math.min((ts-start)/duration,1);
    var ease=1-Math.pow(1-p,3);
    var val=from+(to-from)*ease;
    el.value=val.toFixed(decimals);
    if(p<1)requestAnimationFrame(step);
    else el.value=to.toFixed(decimals);
  }
  requestAnimationFrame(step);
}
// ── CSV INTELLIGENT PARSING ───────────────────────────────────────────────
var _CSV_FIELDS=['vibration','temp_palier','debit','pression_entree','pression_sortie','courant_moteur','temp_moteur','heure_fonctionnement'];
var _CSV_OPT_FIELDS=['temperature_ambiante','niveau_huile','tension_reseau'];
var _CSV_PATS={
  vibration:['vibration','vib','vibration_mm','vib_mms','vibration_mmps','accel','acceleration','vibr'],
  temp_palier:['temp_palier','bearing_temp','palier_temp','t_palier','tpalier','bearing_temperature','temp_roulement'],
  debit:['debit','flow','flow_rate','flowrate','debit_m3h','flow_m3h','caudal','durchfluss'],
  pression_entree:['pression_entree','inlet_pressure','pressure_in','p_in','pe','suction_pressure','pression_aspiration'],
  pression_sortie:['pression_sortie','outlet_pressure','discharge_pressure','pressure_out','p_out','ps','pression_refoulement'],
  courant_moteur:['courant_moteur','motor_current','current_motor','im','courant_a','current_a','ampere_moteur','motor_amp'],
  temp_moteur:['temp_moteur','motor_temp','motor_temperature','tm','temperature_moteur','t_moteur'],
  heure_fonctionnement:['heure_fonctionnement','run_hours','operating_hours','runtime','heures','hours','hf','total_hours']
};
var _CSV_OPT_PATS={
  temperature_ambiante:['temperature_ambiante','ambient_temp','temp_ambiante','t_amb','ambient','ta'],
  niveau_huile:['niveau_huile','oil_level','oil','huile','niveau'],
  tension_reseau:['tension_reseau','voltage','tension','v_reseau','power_supply','supply_voltage']
};
function _csvN(s){return s.toLowerCase().trim().replace(/[^a-z0-9]/g,'_').replace(/_+/g,'_').replace(/^_|_$/g,'');}
function _csvDelim(line){var sc=(line.match(/;/g)||[]).length,cc=(line.match(/,/g)||[]).length;return sc>cc?';':',';}
function detectCsvMapping(rawHeaders){
  var norm=rawHeaders.map(_csvN);
  var origLow=rawHeaders.map(function(h){return h.toLowerCase().trim();});
  var map={};
  var usedIdx=new Set();
  function matchField(field,pats){
    var best=-1,score=0;
    for(var j=0;j<norm.length;j++){
      if(usedIdx.has(j))continue;
      var h=norm[j],s=0;
      for(var p=0;p<pats.length;p++){
        if(h===pats[p]){s=100-p;break;}
        if(h.indexOf(pats[p])!==-1&&s<70-p)s=70-p;
      }
      if(s>score){score=s;best=j;}
    }
    if(best!==-1){
      var orig=origLow[best],unit=null;
      // Pump fields — no unit conversion needed (all SI)
      map[field]={idx:best,unit:unit,col:rawHeaders[best]};
      usedIdx.add(best);
    }
  }
  _CSV_FIELDS.forEach(function(f){matchField(f,_CSV_PATS[f]);});
  _CSV_OPT_FIELDS.forEach(function(f){matchField(f,_CSV_OPT_PATS[f]);});
  var unknown=[];
  for(var i=0;i<rawHeaders.length;i++){
    if(!usedIdx.has(i))unknown.push({idx:i,col:rawHeaders[i]});
  }
  map._unknown=unknown;
  return map;
}
function buildPilarRow(vals,map){
  var row={};
  for(var i=0;i<_CSV_FIELDS.length;i++){
    var f=_CSV_FIELDS[i],m=map[f];
    if(!m||m.idx>=vals.length){row[f]=null;continue;}
    var raw=(vals[m.idx]||'').toString().trim().replace(',','.');
    var v=parseFloat(raw);
    if(isNaN(v)){row[f]=null;continue;}
    row[f]=Math.round(v*100)/100;
  }
  var hasAny=_CSV_FIELDS.some(function(f){return row[f]!==null;});
  if(!hasAny)return null;
  var extra={};
  _CSV_OPT_FIELDS.forEach(function(f){
    var m=map[f];if(!m||m.idx>=vals.length)return;
    var raw=(vals[m.idx]||'').toString().trim().replace(',','.');
    var v=parseFloat(raw);if(!isNaN(v))extra[f]=Math.round(v*100)/100;
  });
  row._extra=extra;
  var unknown={};
  if(map._unknown){
    map._unknown.forEach(function(uc){
      if(uc.idx<vals.length){
        var raw=(vals[uc.idx]||'').toString().trim().replace(',','.');
        var v=parseFloat(raw);if(!isNaN(v))unknown[uc.col]=v;
      }
    });
  }
  row._unknown=unknown;
  return row;
}
</script></head>"""

_SIDEBAR = """
<!-- Desktop Sidebar (shown on wide screens) -->
<aside class="sidebar" id="desktopSidebar">
  <div class="sidebar-logo">
    <div class="logo-mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
    </div>
    <div>
      <div class="logo-text">PILAR</div>
      <div class="logo-sub">Predictive Maintenance</div>
    </div>
  </div>
  <nav class="sidebar-nav">
    <div class="sidebar-section">Analysis</div>
    <a href="/monitor" class="ni {m}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      <span class="ni-label" data-i18n="nav_monitor">Monitor</span>
    </a>
    <a href="/tutorial" class="ni {tut}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
      <span class="ni-label" data-i18n="nav_import">Import</span>
    </a>
    <a href="/history" class="ni {h}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
      <span class="ni-label" data-i18n="nav_history">History</span>
    </a>
    <a href="/twin" class="ni {tw}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
      <span class="ni-label" data-i18n="nav_twin">Digital Twin</span>
    </a>
    <div class="sidebar-section">Fleet</div>
    <a href="/dashboard" class="ni {fl}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span class="ni-label">Fleet Overview</span>
    </a>
    <div class="sidebar-section">User</div>
    <a href="/account" class="ni {a}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg>
      <span class="ni-label">Lobby</span>
    </a>
    <a href="/settings" class="ni {s}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
      <span class="ni-label" data-i18n="nav_settings">Settings</span>
    </a>
  </nav>
  <div class="sidebar-footer">
    <div class="sync-status-bar" id="sidebarSyncBar" onclick="updateSyncStatus()" title="Click to refresh sync status" style="cursor:pointer">
      <span class="sync-dot offline" id="sidebarSyncDot"></span>
      <span id="sidebarSyncLabel">Offline</span>
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
<a href="/monitor" class="ni {m}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg><span data-i18n="nav_monitor">Monitor</span></a>
<a href="/tutorial" class="ni {tut}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg><span data-i18n="nav_import">Import</span></a>
<a href="/history" class="ni {h}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg><span data-i18n="nav_history">History</span></a>
<a href="/twin" class="ni {tw}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg><span data-i18n="nav_twin">Twin</span></a>
<a href="/settings" class="ni {s}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg><span data-i18n="nav_settings">Settings</span></a>
</nav>"""

def nav(active):
    keys = {"m":"","tut":"","h":"","fl":"","a":"","s":"","tw":""}
    keys[active] = "on"
    return _SIDEBAR.format(**keys) + _BOTTOM_NAV.format(**keys)


# ── MONITOR ───────────────────────────────────────────────────────────────────
HTML = _HEAD.replace("{FAV}","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") + """
<body>
<div id="pilar-update-banner" style="display:none;position:fixed;top:0;left:0;right:0;z-index:9999;background:#0d9488;color:#fff;padding:10px 20px;font-size:13px;font-family:'DM Sans',sans-serif;display:flex;align-items:center;justify-content:space-between;gap:12px;">
  <span id="pilar-update-msg"></span>
  <div style="display:flex;gap:8px;flex-shrink:0">
    <a id="pilar-update-link" href="#" target="_blank" style="background:rgba(0,0,0,0.25);color:#fff;padding:5px 14px;border-radius:6px;font-weight:600;font-size:12px;text-decoration:none;">Download</a>
    <button onclick="document.getElementById('pilar-update-banner').style.display='none'" style="background:transparent;border:none;color:rgba(255,255,255,0.7);cursor:pointer;font-size:18px;line-height:1;padding:0 4px">&times;</button>
  </div>
</div>
<script>
fetch('/api/update_info').then(function(r){return r.json();}).then(function(d){
  if(d.available){
    var b=document.getElementById('pilar-update-banner');
    document.getElementById('pilar-update-msg').textContent='Update available — PILAR '+d.version+' is ready to install.';
    document.getElementById('pilar-update-link').href=d.download_url||'#';
    b.style.display='flex';
  }
}).catch(function(){});
</script>
<div class="desktop-layout">
""" + nav("m") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_monitor">Monitor</span>
<div class="hright"><button class="nb" id="nb" onclick="toggleN()">Notifs</button></div></header>
<div class="page pad">
  <div class="ab" id="abn" data-i18n="status_alert">Anomaly Detected</div>

  <!-- RESULT — hero -->
  <div id="res"><div class="idle"><span class="l1" data-i18n="idle_l1">No analysis yet</span><span class="l2" data-i18n="idle_l2">Connect a CSV file below to start live monitoring</span></div></div>

  <!-- LIVE FILE ZONE -->
  <div class="lz-wrap">
    <!-- Empty state -->
    <div id="lzEmpty" class="lz-empty">
      <svg style="width:32px;height:32px;stroke:var(--text3);fill:none;flex-shrink:0" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="lz-hint" data-i18n="live_hint">Connect a CSV file — Pilar analyses each new row automatically as data arrives</div>
      <button class="lz-cta" onclick="openLiveFile()">
        <svg style="width:16px;height:16px;stroke:currentColor;fill:none;flex-shrink:0" viewBox="0 0 24 24"><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z" stroke-width="2"/><polyline points="13 2 13 9 20 9" stroke-width="2"/></svg>
        <span data-i18n="live_connect">Connect CSV file</span>
      </button>
      <input type="file" id="lfInput" accept=".csv" style="display:none" onchange="onLiveFileFallback(this)">
    </div>
    <!-- Connected state -->
    <div id="lzConn" class="lz-conn" style="display:none">
      <div class="lz-ch">
        <span class="lz-dot"></span>
        <span class="lz-fname" id="liveFileName">—</span>
        <button class="lz-disc" onclick="stopLiveMonitor()" data-i18n="live_disconnect">Disconnect</button>
      </div>
      <div class="lz-sg">
        <div class="lz-si"><span class="lz-sv" id="liveRowCount">0</span><span class="lz-sl" data-i18n="live_rows">Rows</span></div>
        <div class="lz-si alert"><span class="lz-sv" id="liveFailCount">0</span><span class="lz-sl" data-i18n="live_fail">Failures</span></div>
        <div class="lz-si ok"><span class="lz-sv" id="liveOkCount">0</span><span class="lz-sl" data-i18n="live_ok">Normal</span></div>
      </div>
      <div class="lz-foot">
        <span class="lz-ck" id="liveChk"></span>
        <select class="lz-sel" id="liveIntv">
          <option value="2000">2s</option>
          <option value="5000" selected>5s</option>
          <option value="10000">10s</option>
          <option value="30000">30s</option>
        </select>
      </div>
    </div>
  </div>

  <!-- AI STACK — always visible -->
  <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
    <div class="ai-chip" id="ai-isoforest">
      <span class="ai-dot" id="ai-isoforest-dot"></span>
      <span class="ai-label">Isolation Forest</span>
      <span class="ai-val" id="ai-isoforest-val" data-i18n="ai_warming">warming up</span>
    </div>
    <div class="ai-chip">
      <span class="ai-dot" style="background:var(--teal2)"></span>
      <span class="ai-label">SHAP</span>
      <span class="ai-val" data-i18n="ai_ready">ready</span>
    </div>
    <div class="ai-chip">
      <span class="ai-dot" style="background:var(--teal2)"></span>
      <span class="ai-label">RUL</span>
      <span class="ai-val" data-i18n="ai_in_twin">Digital Twin</span>
    </div>
  </div>

  <!-- AI RESULTS — shown after each prediction -->
  <div id="ai-panel" style="display:none;margin-bottom:12px">
    <div class="card" style="padding:14px 16px">
      <div style="font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:12px">AI Insights</div>
      <div id="ai-anomaly-row" style="display:none;align-items:center;justify-content:space-between;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--border)">
        <div>
          <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase">Anomaly score</div>
          <div style="font-size:9px;color:var(--text3);margin-top:2px">Isolation Forest · unsupervised</div>
        </div>
        <span id="ai-anomaly-val" style="font-size:20px;font-weight:800">—</span>
      </div>
      <div id="ai-shap-section" style="display:none">
        <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">Top drivers · SHAP</div>
        <div id="ai-shap-body"></div>
      </div>
    </div>
  </div>

  <!-- MANUAL INPUT — sensor sliders (collapsed by default) -->
  <div class="card">
    <div class="man-hdr" onclick="toggleManual()">
      <span class="ctitle" style="margin-bottom:0" data-i18n="manual_title">Manual Analysis</span>
      <svg class="man-chv" id="manChv" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
    <div id="manBody" style="display:none;margin-top:14px">
      <div class="ctitle" data-i18n="sensor_params">Sensor readings</div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_vibration">Vibration</span><div class="vwrap"><input class="vi" type="number" id="nvib" value="2.5" min="0" max="30" step="0.1" oninput="si('svib','nvib',null)"><span class="vu">mm/s</span></div></div><input type="range" id="svib" min="0" max="30" step="0.1" value="2.5" oninput="ss('svib','nvib',1,null)"><div class="rl"><span>0</span><span>30 mm/s</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_temp_palier">Bearing temp.</span><div class="vwrap"><input class="vi" type="number" id="ntp" value="65" min="20" max="150" step="1" oninput="si('stp','ntp',null)"><span class="vu">°C</span></div></div><input type="range" id="stp" min="20" max="150" step="1" value="65" oninput="ss('stp','ntp',0,null)"><div class="rl"><span>20</span><span>150°C</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_debit">Flow rate</span><div class="vwrap"><input class="vi" type="number" id="ndbt" value="45" min="0" max="300" step="1" oninput="si('sdbt','ndbt',null)"><span class="vu">m³/h</span></div></div><input type="range" id="sdbt" min="0" max="300" step="1" value="45" oninput="ss('sdbt','ndbt',0,null)"><div class="rl"><span>0</span><span>300 m³/h</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_pression_e">Inlet pressure</span><div class="vwrap"><input class="vi" type="number" id="npe" value="1.5" min="0" max="15" step="0.1" oninput="si('spe','npe',null)"><span class="vu">bar</span></div></div><input type="range" id="spe" min="0" max="15" step="0.1" value="1.5" oninput="ss('spe','npe',1,null)"><div class="rl"><span>0</span><span>15 bar</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_pression_s">Outlet pressure</span><div class="vwrap"><input class="vi" type="number" id="nps" value="107.7" min="0" max="300" step="0.1" oninput="si('sps','nps',null)"><span class="vu">bar</span></div></div><input type="range" id="sps" min="0" max="300" step="0.5" value="107.7" oninput="ss('sps','nps',1,null)"><div class="rl"><span>0</span><span>300 bar</span></div></div>
      <div class="adv-row" onclick="toggleAdv()">
        <span data-i18n="adv_params">Advanced parameters</span>
        <svg id="advChv" viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:13px;height:13px;transition:transform .2s"><path d="M19 9l-7 7-7-7" stroke-width="2" stroke-linecap="round"/></svg>
      </div>
      <div id="advParams" style="display:none;padding-top:4px">
        <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_courant">Motor current</span><div class="vwrap"><input class="vi" type="number" id="nim" value="4.6" min="0" max="50" step="0.1" oninput="si('sim','nim',null)"><span class="vu">A</span></div></div><input type="range" id="sim" min="0" max="50" step="0.1" value="4.6" oninput="ss('sim','nim',1,null)"><div class="rl"><span>0</span><span>50 A</span></div></div>
        <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_temp_moteur">Motor temp.</span><div class="vwrap"><input class="vi" type="number" id="ntm" value="75" min="20" max="200" step="1" oninput="si('stm','ntm',null)"><span class="vu">°C</span></div></div><input type="range" id="stm" min="20" max="200" step="1" value="75" oninput="ss('stm','ntm',0,null)"><div class="rl"><span>20</span><span>200°C</span></div></div>
        <div class="sensor"><div class="srow"><span class="sname" data-i18n="p_heure">Run hours</span><div class="vwrap"><input class="vi" type="number" id="nhf" value="5000" min="0" max="100000" step="100" oninput="si('shf','nhf',null)"><span class="vu">h</span></div></div><input type="range" id="shf" min="0" max="100000" step="100" value="5000" oninput="ss('shf','nhf',0,null)"><div class="rl"><span>0</span><span>100k h</span></div></div>
      </div>
      <button class="btn" id="btn" onclick="analyse()" data-i18n="run_btn">Run Analysis</button>
    </div>
  </div>
</div>
</div>
</div>
<style>
.adv-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;margin:8px 0;border-top:1px solid var(--border);cursor:pointer;font-size:10px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase}
.adv-row:hover{color:var(--text2)}
</style>
<script>
let lastR=null,lastD=null;

function toggleAdv(){
  var p=document.getElementById('advParams'),c=document.getElementById('advChv');
  var open=p.style.display==='block';
  p.style.display=open?'none':'block';
  c.style.transform=open?'':'rotate(180deg)';
}

function toggleManual(){
  var b=document.getElementById('manBody'),c=document.getElementById('manChv'),o=b.style.display==='block';
  b.style.display=o?'none':'block';
  c.style.transform=o?'':'rotate(180deg)';
}
function updN(){const b=document.getElementById('nb');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent='Notifs ON';b.className='nb on';}else{b.textContent='Enable Notifs';b.className='nb';}}
async function toggleN(){if(Notification.permission==='granted')return;await Notification.requestPermission();updN();}
function sendN(risk,zones){if(Notification.permission!=='granted')return;new Notification('Pilar — Risk: '+risk+'%',{body:zones.length?'Zones: '+zones.map(z=>z.nom).join(', '):'No specific zone',requireInteraction:true,tag:'pilar'});}
updN();
function ss(s,n,d,type){
  var el=document.getElementById(n);
  var from=parseFloat(el.value)||0;
  var v=parseFloat(document.getElementById(s).value);
  if(type){el.dataset.raw=toRaw(v,type);}
  animateNum(el,from,v,d===null?0:d,120);
}
function si(s,n,type){
  var v=parseFloat(document.getElementById(n).value);
  if(!isNaN(v)){document.getElementById(s).value=v;if(type){document.getElementById(n).dataset.raw=toRaw(v,type);}}
}
function gv(id){return parseFloat(document.getElementById(id).value);}
async function analyse(){
  const btn=document.getElementById('btn');btn.disabled=true;btn.textContent=t('run_btn')+'\u2026';
  lastD={vibration:gv('nvib'),temp_palier:gv('ntp'),
    debit:gv('ndbt'),pression_entree:gv('npe'),pression_sortie:gv('nps'),
    courant_moteur:gv('nim'),temp_moteur:gv('ntm'),heure_fonctionnement:gv('nhf')};
  try{
    const res=await fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastD)});
    let r;
    try{r=await res.json();}catch(je){
      document.getElementById('res').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Server error ('+res.status+')</span></div>';
      btn.disabled=false;btn.textContent=t('run_btn');return;
    }
    if(r.error){
      document.getElementById('res').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">'+r.error+'</span></div>';
      btn.disabled=false;btn.textContent=t('run_btn');return;
    }
    lastR=r;
    sessionStorage.setItem('lr',JSON.stringify(lastR));
    sessionStorage.setItem('ld',JSON.stringify(lastD));
    render(lastR);
    if(lastR.probabilite>=50){sendN(lastR.probabilite,lastR.zones);const a=document.getElementById('abn');a.style.display='block';setTimeout(()=>a.style.display='none',4000);}
  }catch(err){
    document.getElementById('res').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Network error: '+err.message+'</span></div>';
  }
  btn.disabled=false;btn.textContent=t('run_btn');
}
function render(r){
  const al=r.prediction===1,cls=al?'alert':'ok',st=al?t('status_alert'):t('status_ok');
  let confH='';
  if(r.confidence!==undefined&&r.confidence<100){
    var imp=r.imputed&&r.imputed.length?r.imputed.join(', '):'';
    confH='<div style="margin-top:8px;padding:6px 10px;background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.25);border-radius:6px;font-size:10px;color:#ca8a04;display:flex;align-items:center;gap:6px"><svg style="width:12px;height:12px;fill:none;stroke:currentColor;flex-shrink:0" viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'+t('partial_analysis')+' — '+r.confidence+'%'+(imp?' ('+t('imputed')+': '+imp+')':'')+'</div>';
  }
  let zH='';
  if(al&&r.zones&&r.zones.length>0){zH='<div class="card"><div class="ctitle">'+t('zone_title')+'</div>'+r.zones.map(z=>'<div class="zrow"><span class="zname">'+z.nom+'</span><div class="zbw"><div class="zbf" style="width:'+z.proba+'%"></div></div><span class="zp">'+z.proba+'%</span></div>').join('')+'</div>';}
  let rulH='';
  if(r.rul_hours!=null){
    var rulCol=r.rul_hours<500?'var(--red)':r.rul_hours<1500?'var(--amber)':'var(--green)';
    rulH='<div class="card" style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px"><div><div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase">Est. Remaining Life</div><div style="font-size:9px;color:var(--text3);margin-top:2px">RUL · NASA C-MAPSS model</div></div><div style="text-align:right"><div style="font-size:26px;font-weight:800;color:'+rulCol+'">'+r.rul_hours+'<span style="font-size:12px;color:var(--text3)"> h</span></div></div></div>';
  }
  let warnH='';
  if(r.domain_warnings&&r.domain_warnings.length){
    warnH='<div style="padding:10px 14px;background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.25);border-radius:8px;font-size:10px;color:#fbbf24;margin-bottom:12px">'+r.domain_warnings.map(w=>'&#9888; '+w).join('<br>')+'</div>';
  }
  document.getElementById('res').innerHTML=warnH+'<div class="rh '+cls+'"><div><div class="sb '+cls+'"><span class="dot '+cls+'"></span>'+st+'</div><div style="font-size:10px;color:var(--text3);margin-top:4px">'+localTimeNow()+'</div>'+confH+'</div><div><div class="rnum '+cls+'">'+r.probabilite+'<span class="runit">%</span></div><div class="rlbl">'+t('failure_prob')+'</div></div></div>'+zH+rulH;

  // ── AI PANEL ──
  var aiPanel=document.getElementById('ai-panel');
  var hasAI=(r.anomaly_score!=null)||(r.shap_explanations&&r.shap_explanations.length);
  aiPanel.style.display=hasAI?'block':'none';

  // Anomaly score
  var anomRow=document.getElementById('ai-anomaly-row');
  var anomVal=document.getElementById('ai-anomaly-val');
  var isoChipDot=document.getElementById('ai-isoforest-dot');
  var isoChipVal=document.getElementById('ai-isoforest-val');
  if(r.anomaly_score!=null){
    var aCl=r.anomaly_score>=70?'var(--red)':r.anomaly_score>=35?'var(--amber)':'var(--green)';
    anomRow.style.display='flex';
    anomVal.textContent=r.anomaly_score+'/100';
    anomVal.style.color=aCl;
    if(isoChipDot)isoChipDot.style.background=aCl;
    if(isoChipVal)isoChipVal.textContent=r.anomaly_score+'/100';
  }else{
    anomRow.style.display='none';
    if(isoChipDot)isoChipDot.style.background='var(--text3)';
    if(isoChipVal)isoChipVal.textContent=t('ai_warming')||'warming up';
  }

  // SHAP top-3
  var shapSection=document.getElementById('ai-shap-section');
  var shapBody=document.getElementById('ai-shap-body');
  if(r.shap_explanations&&r.shap_explanations.length){
    shapSection.style.display='block';
    shapBody.innerHTML=r.shap_explanations.map(function(s){
      return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        +'<span style="font-size:11px;color:var(--text2);flex:1">'+s.feature+'</span>'
        +'<span style="font-size:13px;font-weight:700;color:'+(s.direction==='up'?'var(--red)':'var(--green)')+'">'+s.impact+' '+(s.direction==='up'?'\u2191':'\u2193')+'</span>'
        +'</div>';
    }).join('');
  }else{
    shapSection.style.display='none';
  }
}

// ── LIVE FILE MONITOR ─────────────────────────────────────────────────────
var _lfHandle=null,_lfTimer=null,_lfKnown=0,_lfFail=0,_lfOk=0,_lfMap=null,_lfDelim=',',_lfFallback=false,_lfUnknown={};

async function openLiveFile(){
  if(window.showOpenFilePicker){
    try{
      var picks=await window.showOpenFilePicker({types:[{description:'CSV',accept:{'text/csv':['.csv']}}]});
      _lfHandle=picks[0];
      _lfKnown=0;_lfFail=0;_lfOk=0;_lfMap=null;_lfFallback=false;clearTimeout(_lfTimer);
      var fname=(await _lfHandle.getFile()).name;
      _showLzConn(fname);
      _lfLoop();
    }catch(e){
      if(e.name==='AbortError')return;
      console.warn('showOpenFilePicker failed, using fallback:',e);
      document.getElementById('lfInput').click();
    }
  }else{
    document.getElementById('lfInput').click();
  }
}

function onLiveFileFallback(inp){
  var f=inp.files[0];if(!f)return;
  _lfHandle=null;_lfKnown=0;_lfFail=0;_lfOk=0;_lfMap=null;_lfFallback=true;clearTimeout(_lfTimer);
  _showLzConn(f.name);
  _readSnapshot(f);
  document.getElementById('liveChk').innerHTML='<label for="lfInput" style="padding:3px 10px;background:var(--teal);border-radius:3px;color:#fff;font-size:10px;cursor:pointer">'+t('live_refresh')+'</label>';
}

function _showLzConn(fname){
  document.getElementById('liveFileName').textContent=fname;
  document.getElementById('lzEmpty').style.display='none';
  document.getElementById('lzConn').style.display='block';
  document.getElementById('liveRowCount').textContent='0';
  document.getElementById('liveFailCount').textContent='0';
  document.getElementById('liveOkCount').textContent='0';
  document.getElementById('liveChk').textContent='';
}

function resetLiveTimer(){clearTimeout(_lfTimer);if(_lfHandle)_lfLoop();}

async function _readSnapshot(f){
  try{var text=await f.text();await _lfProcess(text);}catch(e){console.error(e);}
}

async function _lfLoop(){
  if(!_lfHandle)return;
  try{
    var file=await _lfHandle.getFile();
    var text=await file.text();
    await _lfProcess(text);
  }catch(e){console.error('live monitor error',e);}
  _schedule();
}

async function _lfProcess(text){
  var lines=text.split('\\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
  if(lines.length<2)return;
  if(!_lfMap){
    _lfDelim=_csvDelim(lines[0]);
    var rawHdr=lines[0].split(_lfDelim).map(function(s){return s.trim();});
    _lfMap=detectCsvMapping(rawHdr);
    var found=_CSV_FIELDS.filter(function(f){return !!_lfMap[f];}).length;
    var optFound=_CSV_OPT_FIELDS.filter(function(f){return !!_lfMap[f];}).length;
    var ukCount=(_lfMap._unknown||[]).length;
    if(found<1){
      document.getElementById('liveChk').textContent=t('csv_bad')+': '+_CSV_FIELDS.filter(function(f){return !_lfMap[f];}).join(', ');
      stopLiveMonitor();return;
    }
    var hint=t('csv_detect')+' ('+found+'/8)';
    if(optFound>0)hint+=' +'+optFound+' opt';
    if(ukCount>0)hint+=' +'+ukCount+' ?';
    document.getElementById('liveChk').textContent=hint;
  }
  var total=lines.length-1;
  document.getElementById('liveRowCount').textContent=total;
  if(total>_lfKnown){
    var newLines=lines.slice(_lfKnown+1);
    _lfKnown=total;
    var BATCH=10;
    for(var bi=0;bi<newLines.length;bi+=BATCH){
      var chunk=newLines.slice(bi,bi+BATCH);
      var rows=chunk.map(function(line){
        var vals=line.split(_lfDelim);
        return buildPilarRow(vals,_lfMap);
      }).filter(Boolean);
      await Promise.all(rows.map(async function(row){
        try{
          var payload=Object.assign({},row);
          if(row._extra)Object.assign(payload,row._extra);
          delete payload._extra;delete payload._unknown;
          var res=await fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
          var r=await res.json();
          if(!r.error){
            if(r.prediction===1){_lfFail++;if(r.probabilite>=50){sendN(r.probabilite,r.zones);var a=document.getElementById('abn');a.style.display='block';setTimeout(function(){a.style.display='none';},4000);}}
            else _lfOk++;
            document.getElementById('liveFailCount').textContent=_lfFail;
            document.getElementById('liveOkCount').textContent=_lfOk;
            lastR=r;render(r);
            localStorage.setItem('pilar_last_result',JSON.stringify(r));
            if(row._unknown){
              Object.keys(row._unknown).forEach(function(colName){
                var val=row._unknown[colName];
                if(!_lfUnknown[colName])_lfUnknown[colName]={vals:[],risks:[]};
                _lfUnknown[colName].vals.push(val);
                _lfUnknown[colName].risks.push(r.probabilite);
                if(_lfUnknown[colName].vals.length>=20){
                  var disc=_lfUnknown[colName];
                  _lfUnknown[colName]={vals:[],risks:[]};
                  fetch('/api/discover',{method:'POST',headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({name:colName,values:disc.vals,risks:disc.risks})}).catch(function(){});
                }
              });
            }
          }
        }catch(e){}
      }));
      document.getElementById('liveChk').textContent='Processing '+(Math.min(bi+BATCH,newLines.length))+'/'+newLines.length+'...';
      await new Promise(function(resolve){setTimeout(resolve,0);});
    }
  }
  if(!_lfFallback)document.getElementById('liveChk').textContent=t('live_last')+': '+new Date().toLocaleTimeString();
}

function _schedule(){
  var intv=parseInt(document.getElementById('liveIntv').value||5000);
  _lfTimer=setTimeout(_lfLoop,intv);
}

function stopLiveMonitor(){
  clearTimeout(_lfTimer);_lfHandle=null;_lfMap=null;
  document.getElementById('lzEmpty').style.display='flex';
  document.getElementById('lzConn').style.display='none';
  document.getElementById('lfInput').value='';
}
</script></body></html>"""


# ── ACCOUNT ───────────────────────────────────────────────────────────────────
FAV_B64 = "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"

ACCOUNT_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<div class="desktop-layout">
""" + nav("a") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub">Lobby</span></header>
<div class="page pad">

{% if not user %}
<div class="card" style="text-align:center;padding:28px">
  <div style="width:48px;height:48px;border-radius:12px;background:var(--teal-dim);border:1px solid rgba(13,148,136,0.2);display:flex;align-items:center;justify-content:center;margin:0 auto 16px"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--text3)" stroke-width="1.5"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>
  <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px" data-i18n="acc_guest_title">Guest Mode</div>
  <div style="font-size:11px;color:var(--text3);line-height:1.7;margin-bottom:20px" data-i18n-html="acc_guest_desc">Sign in to save your data.</div>
  <a href="/login" style="display:block;padding:14px;background:var(--teal);color:#fff;border-radius:12px;text-decoration:none;font-size:15px;font-weight:600;letter-spacing:0;text-transform:none;margin-bottom:10px" data-i18n="acc_signin">Sign In</a>
  <a href="/register" style="display:block;padding:14px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:12px;text-decoration:none;font-size:15px;font-weight:500;letter-spacing:0;text-transform:none" data-i18n="acc_register">Create Account</a>
</div>

{% else %}
<div class="card">
  <div class="ctitle" data-i18n="acc_card_title">Account</div>
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:13px;font-weight:600;color:var(--text)">{{ user.email }}</div>
      <div style="display:flex;gap:6px;margin-top:5px;align-items:center">
        <span style="padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:var(--teal-dim);color:var(--teal-light)">{{ user.plan }}</span>
        {% if my_role %}<span class="acc-role" data-role="{{ my_role }}" style="padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(191,90,242,0.12);color:var(--purple)">{{ my_role }}</span>{% endif %}
      </div>
    </div>
    <a href="/logout" style="padding:8px 16px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:var(--r-sm);text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap" data-i18n="acc_signout">Sign Out</a>
  </div>
</div>

{% if not team %}
<div class="card">
  <div class="ctitle" data-i18n="acc_team_title">Team</div>
  <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:14px" data-i18n="acc_no_team_desc">Create a team to share live AI insights — anomaly scores, SHAP explanations, and RUL forecasts — with your colleagues.</p>
  <input class="fi" id="tname" placeholder="Team name (optional)" data-i18n="acc_create_ph" style="margin-bottom:10px">
  <button class="btn" onclick="createTeam()" data-i18n="acc_create_btn">Create Team</button>
</div>
{% else %}
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div>
      <div class="ctitle" style="margin-bottom:2px" data-i18n="acc_team_title">Team</div>
      <div style="font-size:13px;font-weight:600;color:var(--text)">{{ team.name }}</div>
    </div>
    <span style="font-size:9px;color:var(--text3)" data-i18n-count="acc_members" data-icount="{{ members|length }}">{{ members|length }} member(s)</span>
  </div>

  {% for m in members %}
  <div style="display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--border)">
    <div style="width:32px;height:32px;border-radius:50%;background:var(--surface2);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:var(--teal-light);flex-shrink:0">{{ m.email[0].upper() }}</div>
    <div style="flex:1;min-width:0">
      <div style="font-size:12px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ m.email }}</div>
      <div style="font-size:9px;letter-spacing:1px;text-transform:uppercase;margin-top:2px;color:{% if m.role=='leader' %}var(--teal-light){% else %}var(--text3){% endif %}">
        <span class="acc-role" data-role="{{ m.role }}">{{ m.role }}</span>{% if m.id == user.id %} <span data-i18n="acc_you">(you)</span>{% endif %}
      </div>
    </div>
    {% if my_role == 'leader' and m.id != user.id %}
    <div style="display:flex;gap:5px;flex-shrink:0">
      {% if m.role == 'member' %}
      <button class="nb" onclick="transfer({{ m.id }},this)" style="font-size:9px" data-i18n="acc_promote">Promote</button>
      {% endif %}
      <button class="nb" onclick="kick({{ m.id }},this)" style="font-size:9px;color:var(--red);border-color:rgba(220,38,38,0.3)" data-i18n="acc_kick">Remove</button>
    </div>
    {% endif %}
  </div>
  {% endfor %}

  {% if my_role == 'leader' %}
  <div style="margin-top:16px">
    <div class="ctitle" data-i18n="acc_add_title">Add Member</div>
    <div style="display:flex;gap:8px">
      <input class="fi" id="inv_email" placeholder="email@company.com" data-i18n="acc_add_ph" type="email" style="flex:1;min-width:0">
      <button onclick="invite()" style="padding:10px 16px;background:var(--teal);color:#fff;border:none;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer;flex-shrink:0" data-i18n="acc_add_btn">Add</button>
    </div>
    <div id="inv_msg" style="font-size:11px;margin-top:6px;display:none"></div>
  </div>
  {% endif %}

  <button onclick="leaveTeam()" style="width:100%;margin-top:16px;padding:11px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:6px;font-size:11px;cursor:pointer" data-i18n="acc_leave">Leave Team</button>
</div>
{% endif %}

{% if user %}
<div class="card">
  <div class="ctitle">API</div>
  <div style="font-size:11px;color:var(--text3);margin-bottom:10px">Use your API key to send sensor data from PLCs or scripts.</div>
  {% if user.api_key %}
  <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <code style="font-size:11px;color:var(--teal-light);flex:1;word-break:break-all">{{ user.api_key }}</code>
    <button onclick="navigator.clipboard.writeText('{{ user.api_key }}')" style="background:none;border:1px solid var(--border2);border-radius:4px;padding:3px 10px;color:var(--text3);font-size:10px;cursor:pointer;white-space:nowrap;flex-shrink:0">COPY</button>
  </div>
  {% endif %}
  <div style="display:flex;gap:8px">
    <button onclick="rotateKey(this)" style="flex:1;padding:10px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:6px;font-size:11px;cursor:pointer">{% if user.api_key %}Regenerate key{% else %}Generate API key{% endif %}</button>
    <a href="/api/docs" style="flex:1;padding:10px;background:var(--teal);color:#fff;border-radius:6px;font-size:11px;font-weight:700;text-decoration:none;display:flex;align-items:center;justify-content:center;letter-spacing:1px;text-transform:uppercase">API Docs</a>
  </div>
</div>
{% endif %}

{% endif %}

</div>
</div>
</div>
<script>
async function createTeam(){
  const btn=event.target;btn.disabled=true;
  const name=document.getElementById('tname').value.trim()||'My Team';
  try{
    const r=await fetch('/team/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    if(!r.ok&&r.status===302){window.location='/login';return;}
    const d=await r.json();
    if(d.ok)location.reload();else{alert(d.error||'Erreur');btn.disabled=false;}
  }catch(e){alert('Erreur réseau');btn.disabled=false;}
}
async function invite(){
  const email=document.getElementById('inv_email').value.trim();
  if(!email)return;
  const msg=document.getElementById('inv_msg');
  const r=await fetch('/team/invite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email})});
  const d=await r.json();
  if(d.ok){msg.style.color='var(--green)';msg.textContent=t('acc_added');msg.style.display='block';setTimeout(()=>location.reload(),1200);}
  else{msg.style.color='var(--red)';msg.textContent=d.error||'Erreur';msg.style.display='block';}
}
async function kick(uid,btn){
  if(!confirm(LANG==='fr'?'Retirer ce membre ?':'Remove this member?'))return;
  btn.disabled=true;
  const r=await fetch('/team/kick/'+uid,{method:'POST'});
  const d=await r.json();
  if(d.ok)location.reload();else{alert(d.error||'Erreur');btn.disabled=false;}
}
async function transfer(uid,btn){
  if(!confirm(LANG==='fr'?'Promouvoir ce membre responsable ?':'Promote this member to leader?'))return;
  btn.disabled=true;
  const r=await fetch('/team/transfer/'+uid,{method:'POST'});
  const d=await r.json();
  if(d.ok)location.reload();else{alert(d.error||'Erreur');btn.disabled=false;}
}
async function leaveTeam(){
  if(!confirm(LANG==='fr'?'Quitter cette équipe ?':'Leave this team?'))return;
  const r=await fetch('/team/leave',{method:'POST'});
  const d=await r.json();
  if(d.ok)location.reload();else alert(d.error||'Erreur');
}
async function rotateKey(btn){
  btn.disabled=true;
  const r=await fetch('/profile/api-key',{method:'POST'});
  const d=await r.json();
  if(d.api_key)location.reload();else{alert('Erreur');btn.disabled=false;}
}
</script>

{% if team %}
<!-- ── TEAM CHAT ─────────────────────────────────────────────────────────── -->
<style>
#chat-bubble{position:fixed;bottom:68px;right:16px;width:40px;height:40px;border-radius:50%;background:var(--teal);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(0,0,0,.4);z-index:200;transition:transform .15s}
#chat-bubble:hover{transform:scale(1.08)}
#chat-badge{position:absolute;top:-2px;right:-2px;width:16px;height:16px;border-radius:50%;background:var(--red);font-size:9px;font-weight:700;color:#fff;display:none;align-items:center;justify-content:center}
#chat-panel{position:fixed;bottom:116px;right:16px;width:300px;max-width:calc(100vw - 32px);height:380px;background:var(--surface);border:1px solid var(--border2);border-radius:var(--r);display:none;flex-direction:column;z-index:201;box-shadow:var(--shadow)}
#chat-panel.open{display:flex}
#chat-head{padding:12px 14px;border-bottom:1px solid var(--border);font-size:13px;font-weight:700;color:var(--teal-light);display:flex;justify-content:space-between;align-items:center}
#chat-msgs{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:6px}
.cmsg{max-width:80%;padding:8px 12px;border-radius:var(--r-sm);font-size:12px;line-height:1.5;word-break:break-word}
.cmsg.mine{align-self:flex-end;background:var(--teal-dim);border:1px solid rgba(13,148,136,0.3);color:var(--text)}
.cmsg.theirs{align-self:flex-start;background:var(--surface2);border:1px solid var(--border);color:var(--text)}
.cmsg .cmeta{font-size:9px;color:var(--text3);margin-bottom:2px}
#chat-form{display:flex;padding:10px;gap:8px;border-top:1px solid var(--border)}
#chat-input{flex:1;background:rgba(120,120,128,0.18);border:1px solid var(--border);border-radius:10px;padding:8px 10px;color:var(--text);font-size:13px;outline:none}
#chat-send{background:var(--teal);border:none;border-radius:8px;padding:8px 12px;cursor:pointer;color:#fff;font-size:12px;font-weight:700}
</style>

<div id="chat-bubble" onclick="toggleChat()" title="Team Chat">
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
  <span id="chat-badge"></span>
</div>

<div id="chat-panel">
  <div id="chat-head">
    <span>{{ team.name }} — Team Chat</span>
    <button onclick="toggleChat()" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:16px;line-height:1">×</button>
  </div>
  <div id="chat-msgs"></div>
  <form id="chat-form" onsubmit="sendMsg(event)">
    <input id="chat-input" placeholder="Message..." maxlength="1000" autocomplete="off">
    <button id="chat-send" type="submit">Send</button>
  </form>
</div>

<script>
let _chatLastId = 0;
let _chatOpen = false;
let _chatUnread = 0;

function toggleChat(){
  _chatOpen = !_chatOpen;
  document.getElementById('chat-panel').classList.toggle('open', _chatOpen);
  if(_chatOpen){ _chatUnread=0; updateBadge(); fetchMsgs(); }
}

function updateBadge(){
  const b = document.getElementById('chat-badge');
  if(_chatUnread > 0){ b.textContent=_chatUnread>9?'9+':_chatUnread; b.style.display='flex'; }
  else b.style.display='none';
}

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function fetchMsgs(){
  try{
    const r = await fetch('/team/messages?since='+_chatLastId);
    if(!r.ok) return;
    const msgs = await r.json();
    if(!msgs.length) return;
    const box = document.getElementById('chat-msgs');
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
    msgs.forEach(m => {
      _chatLastId = Math.max(_chatLastId, m.id);
      const div = document.createElement('div');
      div.className = 'cmsg ' + (m.mine ? 'mine' : 'theirs');
      div.innerHTML = (!m.mine ? `<div class="cmeta">${escHtml(m.email.split('@')[0])}</div>` : '') +
                      `<div>${escHtml(m.content)}</div><div class="cmeta" style="text-align:right;margin-top:2px">${m.ts}</div>`;
      box.appendChild(div);
      if(!_chatOpen && !m.mine){ _chatUnread++; }
    });
    if(atBottom || _chatOpen) box.scrollTop = box.scrollHeight;
    if(!_chatOpen && msgs.some(m=>!m.mine)) updateBadge();
  } catch(e){}
}

async function sendMsg(e){
  e.preventDefault();
  const inp = document.getElementById('chat-input');
  const content = inp.value.trim();
  if(!content) return;
  inp.value = '';
  try{
    await fetch('/team/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    await fetchMsgs();
  } catch(e){}
}

// Ctrl+Enter ou Enter pour envoyer
document.getElementById('chat-input').addEventListener('keydown', e => {
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(e); }
});

// Polling toutes les 4 secondes
setInterval(fetchMsgs, 4000);
fetchMsgs();
</script>
{% endif %}

</body></html>"""

# ── TWIN ──────────────────────────────────────────────────────────────────────
TWIN_HTML = _HEAD.replace("{FAV}","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") + """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<body>
<div class="desktop-layout">
""" + nav("tw") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub">Digital Twin — Pump Simulation</span></header>
<div class="page pad" id="tc"><div class="idle"><span class="l1" data-i18n="twin_loading">Loading simulation...</span></div></div>
</div>
</div>
<script>
const PL={paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'rgba(235,235,245,0.3)',size:10},margin:{t:8,b:36,l:40,r:8},xaxis:{gridcolor:'rgba(84,84,88,0.36)',linecolor:'rgba(84,84,88,0.36)',tickfont:{size:9}},yaxis:{gridcolor:'rgba(84,84,88,0.36)',linecolor:'rgba(84,84,88,0.36)',tickfont:{size:9}},legend:{bgcolor:'transparent',font:{size:9}},hovermode:'x unified'};
const PC={responsive:true,displayModeBar:false};
async function load(){
  let d;
  try{
    const res=await fetch('/api/twin');
    try{d=await res.json();}catch(je){
      document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:var(--red)">Erreur serveur (code '+res.status+')</span><span class="l2">Vérifiez les logs Railway</span></div>';return;
    }
    if(d.error){document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:var(--red)">'+d.error+'</span></div>';return;}
  }catch(err){
    document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:var(--red)">Erreur réseau: '+err.message+'</span></div>';return;
  }
  if(!d.has_data){document.getElementById('tc').innerHTML='<div class="idle"><span class="l1">'+t('twin_no_data')+'</span><span class="l2">'+t('twin_no_data2')+'</span><a href="/" style="margin-top:16px;padding:12px 20px;background:var(--teal);color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">'+t('twin_go')+'</a></div>';return;}
  const bCls=d.failure_hours===null?'ok':d.failure_hours<6?'alert':'amber';
  const bT=d.failure_hours===null?t('twin_healthy'):t('twin_failure')+d.failure_hours+'h';
  const wta_disp=d.last_params.vibration||2.5;
  const wu_disp=d.last_params.heure_fonctionnement||5000;
  const rulH=d.rul_hours!=null?('<div style="margin-top:6px;padding:6px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-xs);font-size:11px;color:var(--text2)">&#x23F1; Est. remaining life: <strong style="color:'+(d.rul_hours<24?'var(--red)':d.rul_hours<72?'var(--amber)':'var(--green)')+'">~'+d.rul_hours+'h</strong>'+(d.rul_confidence?' <span style="font-size:9px;color:var(--text3)">('+d.rul_confidence+')</span>':'')+'</div>'):'';
  document.getElementById('tc').innerHTML=`
    <div class="rh ${bCls}"><div><div class="sb ${bCls}"><span class="dot ${bCls}"></span>${bT}</div><div style="font-size:10px;color:var(--text3);margin-top:4px">${t('twin_trend')} ${d.trend}</div>${rulH}</div><div><div class="rnum ${bCls}">${d.current_risk}<span class="runit">%</span></div><div class="rlbl">${t('twin_cur_risk')}</div></div></div>
    <div class="kgrid"><div class="kc"><div class="kv amber">${d.avg_risk_24h}%</div><div class="kl">${t('twin_avg')}</div></div><div class="kc"><div class="kv ${d.anomaly_rate>=30?'alert':'ok'}">${d.anomaly_rate}%</div><div class="kl">${t('twin_anom')}</div></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_risk')}</div><div id="cr" style="height:220px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_wear')}</div><div id="cw" style="height:180px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_temp')}</div><div id="ct" style="height:180px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_sim')}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
        <div><label style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:4px">Flow rate (m³/h)</label><input class="fi" type="number" id="wv" value="${d.last_params.debit||45}" step="1"></div>
        <div><label style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:4px">Outlet pressure (bar)</label><input class="fi" type="number" id="wc" value="${d.last_params.pression_sortie||107.7}" step="0.1"></div>
        <div><label style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:4px">Run hours (h)</label><input class="fi" type="number" id="wu" value="${wu_disp}" step="1"></div>
        <div><label style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:4px">Vibration (mm/s)</label><input class="fi" type="number" id="wta" value="${wta_disp}" step="0.1"></div>
      </div>
      <button class="btn" onclick="sim()">${t('twin_sim')}</button>
      <div id="wr" style="margin-top:12px"></div>
    </div>`;
  Plotly.newPlot('cr',[{x:d.history_times,y:d.history_risks,name:'History',type:'scatter',mode:'lines+markers',line:{color:'#0d9488',width:2},marker:{size:5}},{x:d.future_times,y:d.future_risks,name:'Simulated',type:'scatter',mode:'lines',line:{color:'#7c3aed',width:2,dash:'dot'},fill:'tozeroy',fillcolor:'rgba(124,58,237,0.04)'},{x:[...d.history_times,...d.future_times],y:Array(d.history_times.length+d.future_times.length).fill(50),name:'Threshold',type:'scatter',mode:'lines',line:{color:'#dc2626',width:1,dash:'dash'}}],{...PL,yaxis:{...PL.yaxis,range:[0,105]}},PC);
  Plotly.newPlot('cw',[{x:d.history_times,y:d.history_wear,name:'Actual',type:'scatter',mode:'lines+markers',line:{color:'#d97706',width:2},marker:{size:4}},{x:d.future_times,y:d.future_wear,name:'Projected',type:'scatter',mode:'lines',line:{color:'#d97706',width:2,dash:'dot'}}],PL,PC);
  Plotly.newPlot('ct',[{x:d.history_times,y:d.history_temp,name:'Actual',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2}},{x:d.future_times,y:d.future_temp,name:'Projected',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2,dash:'dot'}}],PL,PC);
}
async function sim(){
  const p={vibration:parseFloat(document.getElementById('wta').value),debit:parseFloat(document.getElementById('wv').value),pression_sortie:parseFloat(document.getElementById('wc').value),heure_fonctionnement:parseFloat(document.getElementById('wu').value)};
  const res=await fetch('/api/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
  const d=await res.json();const c={ok:'#059669',amber:'#d97706',alert:'#dc2626'};const cls=d.risk>=50?'alert':d.risk>=22?'amber':'ok';
  document.getElementById('wr').innerHTML='<div style="padding:14px;background:var(--bg);border:1px solid '+c[cls]+';border-radius:6px"><div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase">'+t('twin_sim_r')+'</div><div style="font-size:32px;font-weight:800;color:'+c[cls]+';margin:4px 0">'+d.risk+'%</div><div style="font-size:12px;font-weight:600;color:'+c[cls]+'">'+d.status+'</div><div style="font-size:11px;color:var(--text3);margin-top:3px">'+d.message+'</div>'+(d.zones.length?'<div style="font-size:10px;color:var(--amber);margin-top:6px">Zones: '+d.zones.map(z=>z.nom+' '+z.proba+'%').join(' · ')+'</div>':'')+'</div>';
}
load();
</script></body></html>"""

# ── HISTORY ───────────────────────────────────────────────────────────────────
HISTORY_HTML = _HEAD.replace("{FAV}","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") + """
<body>
<div class="desktop-layout">
""" + nav("h") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_history">History</span></header>
<div class="page pad">
  <div class="kgrid" style="grid-template-columns:repeat(auto-fit,minmax(80px,1fr))">
    <div class="kc"><div class="kv">{{ total }}</div><div class="kl" data-i18n="hist_total">Total</div></div>
    <div class="kc"><div class="kv alert">{{ anomalies }}</div><div class="kl" data-i18n="hist_anom">Anomalies</div></div>
    <div class="kc"><div class="kv amber">{{ avg_risk }}%</div><div class="kl" data-i18n="hist_avg">Avg risk</div></div>
    <div class="kc"><div class="kv ok">{{ mails }}</div><div class="kl" data-i18n="hist_alerts">Alerts sent</div></div>
    <div class="kc"><div class="kv {{ 'ok' if reliability and reliability >= 70 else 'amber' if reliability else '' }}">{% if reliability is not none %}{{ reliability }}%{% else %}—{% endif %}</div><div class="kl" data-i18n="hist_reliability">Reliability</div></div>
  </div>
  <div style="font-size:11px;color:var(--text3);margin:6px 0 16px;padding:10px 14px;background:var(--bg2);border-radius:8px;border-left:3px solid var(--teal);display:flex;align-items:center;gap:10px">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:14px;height:14px;flex-shrink:0;color:var(--teal)"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <span>Your feedback trains the AI — mark each result as <strong style="color:var(--green)">Confirmed</strong> or <strong style="color:var(--red)">False alarm</strong> to improve accuracy over time.</span>
  </div>
  <div class="tw">
    <table>
      <thead><tr><th data-i18n="hist_time">Time</th><th data-i18n="hist_class">Class</th><th data-i18n="hist_risk">Risk</th><th data-i18n="hist_status">Status</th><th data-i18n="hist_zones">Zones</th><th data-i18n="hist_alert">Alert</th><th data-i18n="hist_feedback">Feedback</th></tr></thead>
      <tbody>
      {% for a in analyses %}
      <tr><td data-utc="{{ a.timestamp.isoformat() }}Z">{{ a.timestamp.strftime('%d/%m %H:%M') }}</td><td>{{ a.machine_type }}</td><td>{{ a.risk }}%</td>
          <td><span class="badge {{ 'alert' if a.prediction else 'ok' }}">{{ 'Anomaly' if a.prediction else 'OK' }}</span></td>
          <td>{{ a.zones or '—' }}</td>
          <td>{% if a.mail_sent %}<span class="mb">Sent</span>{% else %}—{% endif %}</td>
          <td>
            <span class="fbtn" data-id="{{ a.id }}" data-fb="{{ a.feedback or '' }}" data-pred="{{ a.prediction }}">
              {% if a.prediction %}
              <button onclick="rate({{ a.id }},'tp',this)" class="fb-btn fb-confirm {{ 'fb-active-tp' if a.feedback=='tp' else '' }}">{% if a.feedback=='tp' %}✓ Confirmed{% else %}Confirm{% endif %}</button>
              <button onclick="rate({{ a.id }},'fp',this)" class="fb-btn fb-false  {{ 'fb-active-fp' if a.feedback=='fp' else '' }}">{% if a.feedback=='fp' %}✗ False alarm{% else %}False alarm{% endif %}</button>
              {% else %}
              <button onclick="rate({{ a.id }},'fn',this)" class="fb-btn fb-false {{ 'fb-active-fn' if a.feedback=='fn' else '' }}">{% if a.feedback=='fn' %}⚠ Missed failure{% else %}Missed?{% endif %}</button>
              {% endif %}
            </span>
          </td></tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% if total_pages > 1 %}
  <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-top:20px;font-size:13px;color:var(--text2)">
    {% if page > 1 %}
    <a href="/history?page={{ page - 1 }}" style="padding:6px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--teal);text-decoration:none;font-weight:600">&lsaquo; Prev</a>
    {% endif %}
    <span>Page {{ page }} / {{ total_pages }}</span>
    {% if page < total_pages %}
    <a href="/history?page={{ page + 1 }}" style="padding:6px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--teal);text-decoration:none;font-weight:600">Next &rsaquo;</a>
    {% endif %}
  </div>
  {% endif %}
</div>
</div>
</div>
<style>
.fb-btn{background:none;border:1px solid var(--border);border-radius:5px;padding:3px 9px;cursor:pointer;font-size:10px;font-weight:600;letter-spacing:.5px;color:var(--text3);transition:all .15s;white-space:nowrap}
.fb-btn+.fb-btn{margin-left:4px}
.fb-confirm:hover{border-color:var(--green);color:var(--green)}
.fb-false:hover{border-color:var(--red);color:#f87171}
.fb-active-tp{border-color:var(--green);background:rgba(16,185,129,0.12);color:var(--green)}
.fb-active-fp{border-color:var(--red);background:rgba(239,68,68,0.12);color:#f87171}
.fb-active-fn{border-color:#f97316;background:rgba(249,115,22,0.12);color:#f97316}
</style>
<script>
document.querySelectorAll('td[data-utc]').forEach(function(td){
  td.textContent=localTime(td.dataset.utc,{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
});
async function rate(id, fb, btn){
  const span=btn.closest('.fbtn');
  const prev=span.dataset.fb;
  const newFb=prev===fb?'':fb;
  try{
    await fetch('/analysis/'+id+'/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({feedback:newFb||null})});
    span.dataset.fb=newFb;
    span.querySelectorAll('.fb-btn').forEach(b=>{
      b.classList.remove('fb-active-tp','fb-active-fp','fb-active-fn');
    });
    if(newFb==='tp'){
      var b=span.querySelector('.fb-confirm');
      if(b){b.classList.add('fb-active-tp');b.textContent='✓ Confirmed';}
    } else if(newFb==='fp'){
      var b=span.querySelector('.fb-false');
      if(b){b.classList.add('fb-active-fp');b.textContent='✗ False alarm';}
    } else if(newFb==='fn'){
      var b=span.querySelector('.fb-false');
      if(b){b.classList.add('fb-active-fn');b.textContent='⚠ Missed failure';}
    } else {
      // toggled off — reset labels
      var pred=span.dataset.pred==='1';
      var bc=span.querySelector('.fb-confirm');
      var bf=span.querySelector('.fb-false');
      if(bc) bc.textContent='Confirm';
      if(bf) bf.textContent=pred?'False alarm':'Missed?';
    }
  }catch(e){}
}
</script>
</body></html>"""

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SETTINGS_HTML = _HEAD.replace("{FAV}","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") + """
<body>
<div class="desktop-layout">
""" + nav("s") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_settings">Settings</span></header>
<div class="page pad">
  <div class="card">
    <div class="ctitle" data-i18n="set_lang">Language</div>
    <div style="display:flex;gap:10px">
      <div class="lcard lactive" data-lang="en" onclick="setLang('en')">
        <span class="flag" style="font-size:13px;font-weight:700;letter-spacing:1px">EN</span><span class="lname">English</span>
      </div>
      <div class="lcard" data-lang="fr" onclick="setLang('fr')">
        <span class="flag" style="font-size:13px;font-weight:700;letter-spacing:1px">FR</span><span class="lname">Français</span>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="ctitle" data-i18n="set_email">Alert email</div>
    <label class="flbl" data-i18n="set_email_lbl">Recipient address</label>
    <input class="fi" type="email" id="em" placeholder="maintenance@company.com" data-i18n="set_email_ph">
    <div style="font-size:10px;color:var(--green);margin-top:6px;display:none" id="sv" data-i18n="set_saved">Saved</div>
    <button class="btn" style="margin-top:12px" onclick="saveEmail()" data-i18n="set_email_btn">Save Email</button>
  </div>
  <div class="card">
    <div class="ctitle" data-i18n="set_notif">Browser notifications</div>
    <p style="font-size:12px;color:var(--text2);margin-bottom:12px;line-height:1.6" data-i18n="set_notif_desc">Receive alerts when failure risk exceeds 50%.</p>
    <button class="btn" id="nb" onclick="toggleN()" style="background:var(--purple)" data-i18n="set_notif_btn">Enable Notifications</button>
  </div>
  <div class="card">
    <div class="ctitle" data-i18n="set_sys">System info</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_version">Version</span><span>Pilar v4.0</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_aimodel">AI Model</span><span>RandomForest + GBT + SHAP + RUL</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_domain">Domain</span><span>trypilar.com</span></div>
    </div>
  </div>
  <div class="card">
    <div class="ctitle">Change Password</div>
    <label class="flbl">Current password</label>
    <input class="fi" type="password" id="pwOld" placeholder="••••••••" style="margin-bottom:10px">
    <label class="flbl">New password</label>
    <input class="fi" type="password" id="pwNew" placeholder="••••••••" style="margin-bottom:10px">
    <label class="flbl">Confirm new password</label>
    <input class="fi" type="password" id="pwConf" placeholder="••••••••" style="margin-bottom:12px">
    <div id="pwMsg" style="font-size:11px;margin-bottom:10px;display:none"></div>
    <button class="btn" onclick="changePassword()">Update Password</button>
  </div>
  <div class="card">
    <div class="ctitle">Session</div>
    <button class="btn" onclick="window.location='/logout'" style="background:var(--surface2);border:1px solid var(--border2);color:var(--text2)">Sign Out</button>
  </div>
</div>
</div>
</div>
<script>
async function saveEmail(){const e=document.getElementById('em').value;if(!e)return;await fetch('/set_email',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:e})});const s=document.getElementById('sv');s.style.display='block';setTimeout(()=>s.style.display='none',3000);}
function updN(){const b=document.getElementById('nb');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent=t('set_notif_on');b.style.background='var(--green)';}else if(p==='denied'){b.textContent=t('set_notif_blocked');b.style.background='var(--red)';}else{b.textContent=t('set_notif_btn');b.style.background='var(--purple)';}}
async function toggleN(){if(Notification.permission==='granted')return;await Notification.requestPermission();updN();}
async function changePassword(){
  const old=document.getElementById('pwOld').value;
  const nw=document.getElementById('pwNew').value;
  const cf=document.getElementById('pwConf').value;
  const msg=document.getElementById('pwMsg');
  msg.style.display='block';
  if(!old||!nw||!cf){msg.style.color='var(--red)';msg.textContent='All fields required.';return;}
  if(nw!==cf){msg.style.color='var(--red)';msg.textContent='Passwords do not match.';return;}
  if(nw.length<8){msg.style.color='var(--red)';msg.textContent='Minimum 8 characters.';return;}
  const r=await fetch('/change_password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old_password:old,new_password:nw})});
  const d=await r.json();
  if(d.ok){msg.style.color='var(--green)';msg.textContent='Password updated.';document.getElementById('pwOld').value='';document.getElementById('pwNew').value='';document.getElementById('pwConf').value='';}
  else{msg.style.color='var(--red)';msg.textContent=d.error||'Error.';}
}
updN();
</script></body></html>"""


# ── ASSISTANT ─────────────────────────────────────────────────────────────────
ASSISTANT_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<div class="desktop-layout">
""" + nav("ai") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="nav_assistant">Assistant</span>
<div class="hright"><a href="/account" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px" data-i18n="nav_account">Account</a></div>
</header>
<div class="cw">
  <div class="cm" id="cm">
    <div class="msg bot"><span class="ms" id="ast-pilar-lbl">Pilar AI</span><div class="mb2" id="ast-hello">Hello. I am your predictive maintenance assistant. Share your sensor readings or ask me anything about your machine health.</div></div>
  </div>
  <div class="cia">
    <textarea class="cta" id="ci" placeholder="Ask about your machine..." rows="1" onkeydown="kd(event)" data-i18n="ast_placeholder"></textarea>
    <button class="bsend" id="sb" onclick="send()" data-i18n="ast_send">Send</button>
  </div>
</div>
</div>
</div>
<script>
var hist=[],lastCtx=null;
try{var s=localStorage.getItem('pilar_last_result');if(s)lastCtx=JSON.parse(s);}catch(e){}
document.getElementById('ast-hello').textContent=t('ast_hello');
var _astPilarLbl=document.getElementById('ast-pilar-lbl');
if(_astPilarLbl)_astPilarLbl.textContent=t('ast_pilar');
applyLang();
function kd(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function addMsg(role,txt){
  var d=document.createElement('div');d.className='msg '+role;
  var s=document.createElement('span');s.className='ms';s.textContent=role==='user'?t('ast_you'):t('ast_pilar');
  var b=document.createElement('div');b.className='mb2';b.textContent=txt;
  d.appendChild(s);d.appendChild(b);
  document.getElementById('cm').appendChild(d);
  document.getElementById('cm').scrollTop=999999;
  return b;
}
async function send(){
  var ci=document.getElementById('ci'),sb=document.getElementById('sb');
  var msg=ci.value.trim();if(!msg)return;
  ci.value='';sb.disabled=true;
  addMsg('user',msg);
  hist.push({role:'user',content:msg});
  var tb=addMsg('bot','...');tb.classList.add('typing');
  try{
    var res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,context:lastCtx,history:hist})});
    var r=await res.json();
    tb.classList.remove('typing');
    if(r.error){tb.textContent=t('ast_error')+r.error;}
    else{tb.textContent=r.reply;hist.push({role:'assistant',content:r.reply});}
  }catch(e){tb.classList.remove('typing');tb.textContent=t('ast_net_error');}
  sb.disabled=false;ci.focus();
}
</script></body></html>"""

# ── TUTORIAL ───────────────────────────────────────────────────────────────────
TUTORIAL_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<div class="desktop-layout">
""" + nav("tut") + """
<div class="main-content">
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_tutorial">Import & Run</span>
<div class="hright"><a href="/twin" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px" data-i18n="nav_twin">Twin</a></div>
</header>
<div class="page pad">

  <!-- CSV FORMAT GUIDE -->
  <div class="card">
    <div class="ctitle" data-i18n="tut_format">CSV Format</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:12px" data-i18n="tut_csv_desc">Your CSV must contain these columns (order does not matter):</p>
    <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:6px;padding:12px;overflow-x:auto;margin-bottom:12px">
      <code style="font-size:11px;color:var(--teal-light);white-space:nowrap">vibration, temp_palier, debit, pression_entree, pression_sortie, courant_moteur, temp_moteur, heure_fonctionnement</code>
    </div>
    <table style="font-size:11px;margin-top:0">
      <thead><tr><th data-i18n="tut_cols">Column</th><th data-i18n="tut_unit">Unit</th><th data-i18n="tut_range">Range</th><th data-i18n="tut_desc">Description</th></tr></thead>
      <tbody>
        <tr><td style="color:var(--teal-light)">vibration</td><td>mm/s</td><td>0–30</td><td>Vibration RMS on bearing</td></tr>
        <tr><td style="color:var(--teal-light)">temp_palier</td><td>°C</td><td>20–150</td><td>Bearing temperature</td></tr>
        <tr><td style="color:var(--teal-light)">debit</td><td>m³/h</td><td>0–300</td><td>Flow rate</td></tr>
        <tr><td style="color:var(--teal-light)">pression_entree</td><td>bar</td><td>0–15</td><td>Inlet (suction) pressure</td></tr>
        <tr><td style="color:var(--teal-light)">pression_sortie</td><td>bar</td><td>0–40</td><td>Outlet (discharge) pressure</td></tr>
        <tr><td style="color:var(--teal-light)">courant_moteur</td><td>A</td><td>0–150</td><td>Motor current draw</td></tr>
        <tr><td style="color:var(--teal-light)">temp_moteur</td><td>°C</td><td>20–200</td><td>Motor temperature</td></tr>
        <tr><td style="color:var(--teal-light)">heure_fonctionnement</td><td>h</td><td>0–100000</td><td>Cumulative run hours</td></tr>
      </tbody>
    </table>
    <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap">
      <button class="btn" style="flex:1;background:transparent;border:1px solid var(--border2);color:var(--text2)" onclick="dlSample()" data-i18n="tut_sample">Download sample CSV</button>
      <a href="/adapter" class="btn" style="flex:1;text-align:center;text-decoration:none;background:var(--teal);color:#fff">CSV Adapter</a>
    </div>
  </div>

  <!-- IMPORT -->
  <div class="card">
    <div class="ctitle" data-i18n="tut_import">Import File</div>
    <label for="csvf" style="display:block;padding:24px;border:1px dashed var(--border2);border-radius:6px;text-align:center;cursor:pointer;background:var(--surface2);transition:border-color 0.15s" id="dropz">
      <svg style="width:32px;height:32px;stroke:var(--text3);margin-bottom:8px" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
      <div style="font-size:12px;color:var(--text3)" data-i18n="tut_click_csv">Click to select a CSV file</div>
      <div style="font-size:10px;color:var(--text3);margin-top:4px" id="fname" data-i18n="tut_no_file_sel">No file selected</div>
    </label>
    <input type="file" id="csvf" accept=".csv" style="display:none" onchange="onFile(this)">
    <div style="display:flex;align-items:center;gap:10px;margin-top:14px">
      <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase" data-i18n="tut_speed">Speed</div>
      <select id="spd" class="fi" style="flex:1;padding:8px 10px">
        <option value="500">Fast — 0.5s per row</option>
        <option value="1000" selected>Normal — 1s per row</option>
        <option value="2000">Slow — 2s per row</option>
        <option value="5000">Real-time — 5s per row</option>
      </select>
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn" id="btnStart" style="flex:1" onclick="startRun()" disabled data-i18n="tut_start">Start</button>
      <button class="btn" id="btnPause" style="flex:1;background:var(--amber);display:none" onclick="togglePause()" data-i18n="tut_pause">Pause</button>
      <button class="btn" id="btnStop" style="flex:1;background:var(--red);display:none" onclick="stopRun()" data-i18n="tut_stop">Stop</button>
    </div>
  </div>

  <!-- PROGRESS -->
  <div class="card" id="progCard" style="display:none">
    <div class="ctitle" data-i18n="tut_progress">Progress</div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text3);margin-bottom:8px">
      <span id="progLbl">Row 0 / 0</span>
      <span id="progPct">0%</span>
    </div>
    <div style="height:4px;background:var(--border2);border-radius:2px;overflow:hidden">
      <div id="progBar" style="height:100%;width:0%;background:var(--teal);border-radius:2px;transition:width 0.3s"></div>
    </div>
    <div style="display:flex;gap:10px;margin-top:12px">
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--red)" id="cntFail">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px" data-i18n="live_fail">Failures</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--green)" id="cntOk">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px" data-i18n="live_ok">Normal</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--amber)" id="cntAvg">—</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px" data-i18n="twin_avg">Avg Risk</div>
      </div>
    </div>
  </div>

  <!-- LIVE RESULT -->
  <div id="liveRes"></div>

  <!-- LIVE FILE MONITOR -->
  <div class="card" style="margin-top:8px">
    <div class="ctitle" data-i18n="tut_live">Live File Monitor</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:14px" data-i18n="tut_live_desc">Connect a CSV file that your SCADA or system updates continuously. Pilar detects new rows and analyses them automatically.</p>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <div>
        <div style="font-size:12px;color:var(--text2)" id="liveName" data-i18n="tut_no_file">No file connected</div>
        <div style="font-size:10px;margin-top:3px" id="liveStatus" style="color:var(--text3)" data-i18n="tut_disconnected">Disconnected</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <div style="font-size:10px;color:var(--text3)" id="liveTotalRows"></div>
        <button class="btn" id="btnLiveConnect" onclick="connectLive()" style="width:auto;padding:10px 16px;margin-top:0" data-i18n="tut_connect">Connect File</button>
        <button class="btn" id="btnLiveStop" onclick="stopLive()" style="width:auto;padding:10px 16px;margin-top:0;background:var(--red);display:none" data-i18n="live_disconnect">Disconnect</button>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;white-space:nowrap" data-i18n="tut_check_every">Check every</div>
      <select id="liveInterval" class="fi" style="flex:1;padding:8px 10px">
        <option value="2000">2 seconds</option>
        <option value="5000" selected>5 seconds</option>
        <option value="10000">10 seconds</option>
        <option value="30000">30 seconds</option>
        <option value="60000">1 minute</option>
      </select>
    </div>
    <div style="display:flex;gap:10px;margin-bottom:12px">
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--red)" id="liveFail">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px" data-i18n="live_fail">Failures</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--green)" id="liveOk">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px" data-i18n="live_ok">Normal</div>
      </div>
    </div>
    <div style="font-size:10px;color:var(--text3);margin-bottom:8px" id="liveLastCheck"></div>
    <div id="liveLog" style="max-height:200px;overflow-y:auto"></div>
    <p style="font-size:10px;color:var(--text3);margin-top:10px;line-height:1.5">Requires Chrome or Edge. The file stays on your machine — only new rows are sent to Pilar.</p>
  </div>

</div>
</div>
</div>
<script>
var rows=[],idx=0,timer=null,paused=false,nFail=0,nOk=0,sumRisk=0;

function dlSample(){
  var csv='vibration,temp_palier,debit,pression_entree,pression_sortie,courant_moteur,temp_moteur,heure_fonctionnement\\n'+
    '2.3,63,46,1.5,4.4,17.8,74,4800\\n'+
    '1.8,58,52,1.6,4.7,16.2,70,2300\\n'+
    '5.4,82,31,0.9,3.2,21.5,88,9500\\n'+
    '9.1,95,22,0.4,2.8,19.0,79,14200\\n'+
    '2.5,66,44,1.4,4.3,18.0,75,5000\\n';
  var a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='pilar_pump_sample.csv';a.click();
}

function onFile(inp){
  var f=inp.files[0];if(!f)return;
  document.getElementById('fname').textContent=f.name+' (loading...)';
  var rd=new FileReader();
  rd.onload=function(e){
    var text=e.target.result;
    var lines=text.split('\\n').map(s=>s.trim()).filter(s=>s.length>0);
    if(lines.length<2){alert(t('csv_bad'));return;}
    var delim=_csvDelim(lines[0]);
    var rawHdr=lines[0].split(delim).map(s=>s.trim());
    var map=detectCsvMapping(rawHdr);
    var found=Object.keys(map).length;
    if(found<1){
      var missing=_CSV_FIELDS.filter(function(field){return !map[field];}).join(', ');
      alert(t('csv_bad')+': '+missing);return;
    }
    rows=[];
    for(var r=1;r<lines.length;r++){
      var vals=lines[r].split(delim);
      var obj=buildPilarRow(vals,map);
      if(obj)rows.push(obj);
    }
    var info=found+'/8 '+t('csv_detect');
    document.getElementById('fname').textContent=f.name+' — '+rows.length+' '+t('csv_rows')+' ('+info+')';
    document.getElementById('btnStart').disabled=rows.length===0;
  };
  rd.readAsText(f);
}

function startRun(){
  if(rows.length===0)return;
  idx=0;nFail=0;nOk=0;sumRisk=0;paused=false;
  document.getElementById('progCard').style.display='block';
  document.getElementById('btnStart').style.display='none';
  document.getElementById('btnPause').style.display='block';
  document.getElementById('btnStop').style.display='block';
  document.getElementById('liveRes').innerHTML='';
  runNext();
}

function runNext(){
  if(idx>=rows.length){finish();return;}
  if(paused)return;
  var row=rows[idx];
  var total=rows.length;
  var pct=Math.round(idx/total*100);
  document.getElementById('progLbl').textContent='Row '+(idx+1)+' / '+total;
  document.getElementById('progPct').textContent=pct+'%';
  document.getElementById('progBar').style.width=pct+'%';
  fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)})
    .then(function(r){return r.json();})
    .then(function(r){
      if(r.error){showRow(idx,row,null,r.error);}
      else{
        sumRisk+=r.probabilite;
        if(r.prediction===1)nFail++;else nOk++;
        document.getElementById('cntFail').textContent=nFail;
        document.getElementById('cntOk').textContent=nOk;
        document.getElementById('cntAvg').textContent=Math.round(sumRisk/(idx+1))+'%';
        showRow(idx,row,r,null);
      }
      idx++;
      timer=setTimeout(runNext,parseInt(document.getElementById('spd').value));
    })
    .catch(function(e){
      showRow(idx,row,null,'Network error');
      idx++;
      timer=setTimeout(runNext,parseInt(document.getElementById('spd').value));
    });
}

function showRow(i,row,r,err){
  var div=document.createElement('div');
  var cls='ok',pct=0,label='Normal';
  if(err){cls='amber';label='Error';}
  else if(r.prediction===1){cls='alert';pct=r.probabilite;label='Failure Risk';}
  else{pct=r.probabilite;label='Normal';}
  div.style.cssText='background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;gap:10px';
  var left='<div style="font-size:10px;color:var(--text3)">#'+(i+1)+'</div>';
  left+='<div style="font-size:12px;color:var(--text2);margin-top:2px">'+
    'vib '+(row.vibration||'—')+' mm/s'+
    ' | '++(row.temp_palier||'—')+'°C bearing'+
    ' | '+(row.debit||'—')+' m³/h flow</div>';
  var col=err?'var(--amber)':r.prediction===1?'var(--red)':'var(--green)';
  var right='<div style="text-align:right"><div style="font-size:20px;font-weight:800;color:'+col+'">'+(err?'—':pct+'%')+'</div>';
  right+='<div style="font-size:9px;color:'+col+';letter-spacing:1px;text-transform:uppercase">'+label+'</div></div>';
  div.innerHTML='<div>'+left+'</div>'+right;
  var lr=document.getElementById('liveRes');
  lr.insertBefore(div,lr.firstChild);
}

function togglePause(){
  paused=!paused;
  document.getElementById('btnPause').textContent=paused?'Resume':'Pause';
  if(!paused)runNext();
}

function stopRun(){
  clearTimeout(timer);paused=false;
  document.getElementById('btnStart').style.display='block';
  document.getElementById('btnStart').disabled=false;
  document.getElementById('btnPause').style.display='none';
  document.getElementById('btnStop').style.display='none';
  document.getElementById('btnPause').textContent='Pause';
  finish();
}

function finish(){
  clearTimeout(timer);
  document.getElementById('progLbl').textContent='Done — '+rows.length+' rows';
  document.getElementById('progPct').textContent='100%';
  document.getElementById('progBar').style.width='100%';
  document.getElementById('btnStart').style.display='block';
  document.getElementById('btnStart').textContent='Run Again';
  document.getElementById('btnStart').disabled=false;
  document.getElementById('btnPause').style.display='none';
  document.getElementById('btnStop').style.display='none';
}

// ── LIVE FILE MONITOR ──────────────────────────────────────────────────────
var liveHandle=null, liveTimer=null, liveKnownRows=0, liveFail=0, liveOk=0, liveMap=null, liveDelim=',';

async function connectLive(){
  if(!window.showOpenFilePicker){
    alert('Live file monitoring requires Chrome or Edge browser.');return;
  }
  try{
    var handles=await window.showOpenFilePicker({types:[{description:'CSV',accept:{'text/csv':['.csv']}}]});
    liveHandle=handles[0];
    var fname=(await liveHandle.getFile()).name;
    document.getElementById('liveName').textContent=fname;
    document.getElementById('liveStatus').textContent='Connected';
    document.getElementById('liveStatus').style.color='var(--green)';
    document.getElementById('btnLiveStop').style.display='inline-block';
    document.getElementById('btnLiveConnect').style.display='none';
    document.getElementById('liveLog').innerHTML='';
    liveKnownRows=0;liveFail=0;liveOk=0;
    document.getElementById('liveFail').textContent='0';
    document.getElementById('liveOk').textContent='0';
    liveCheck();
  }catch(e){if(e.name!=='AbortError')console.error(e);}
}

async function liveCheck(){
  if(!liveHandle)return;
  try{
    var file=await liveHandle.getFile();
    var text=await file.text();
    var lines=text.split('\\n').map(s=>s.trim()).filter(s=>s.length>0);
    if(lines.length<2){liveTimer=setTimeout(liveCheck,parseInt(document.getElementById('liveInterval').value));return;}
    if(!liveMap){
      liveDelim=_csvDelim(lines[0]);
      var rawHdr=lines[0].split(liveDelim).map(s=>s.trim());
      liveMap=detectCsvMapping(rawHdr);
      var coreFound=_CSV_FIELDS.filter(function(f){return !!liveMap[f];}).length;
      if(coreFound<1){
        var missing=_CSV_FIELDS.filter(function(f){return !liveMap[f];}).join(', ');
        document.getElementById('liveStatus').textContent=t('csv_bad')+': '+missing;
        document.getElementById('liveStatus').style.color='var(--red)';return;
      }
      document.getElementById('liveStatus').textContent=t('csv_detect')+' ('+coreFound+'/8)';
      document.getElementById('liveStatus').style.color='var(--green)';
    }
    var total=lines.length-1;
    if(total>liveKnownRows){
      var newLines=lines.slice(liveKnownRows+1);
      for(var r=0;r<newLines.length;r++){
        var vals=newLines[r].split(liveDelim);
        var obj=buildPilarRow(vals,liveMap);
        if(obj){
          (function(row,rowNum){
            fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)})
              .then(function(res){return res.json();})
              .then(function(r){
                if(!r.error){
                  if(r.prediction===1)liveFail++;else liveOk++;
                  document.getElementById('liveFail').textContent=liveFail;
                  document.getElementById('liveOk').textContent=liveOk;
                  var col=r.prediction===1?'var(--red)':'var(--green)';
                  var label=r.prediction===1?'FAILURE '+r.probabilite+'%':'Normal '+r.probabilite+'%';
                  var ts=new Date().toLocaleTimeString();
                  var li=document.createElement('div');
                  li.style.cssText='display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);font-size:11px';
                  li.innerHTML='<span style="color:var(--text3)">'+ts+' · Row '+rowNum+'</span><span style="color:'+col+';font-weight:700">'+label+'</span>';
                  var log=document.getElementById('liveLog');
                  log.insertBefore(li,log.firstChild);
                  if(log.children.length>50)log.removeChild(log.lastChild);
                }
              }).catch(function(){});
          })(obj, liveKnownRows+r+1);
        }
      }
      liveKnownRows=total;
      document.getElementById('liveTotalRows').textContent=total+' rows';
    }
    document.getElementById('liveLastCheck').textContent='Last check: '+new Date().toLocaleTimeString();
  }catch(e){console.error('liveCheck error',e);}
  liveTimer=setTimeout(liveCheck,parseInt(document.getElementById('liveInterval').value));
}

function stopLive(){
  clearTimeout(liveTimer);liveHandle=null;liveMap=null;
  document.getElementById('liveStatus').textContent='Disconnected';
  document.getElementById('liveStatus').style.color='var(--text3)';
  document.getElementById('btnLiveStop').style.display='none';
  document.getElementById('btnLiveConnect').style.display='inline-block';
  document.getElementById('liveName').textContent='No file connected';
}
</script></body></html>"""

# ── CSV ADAPTER ───────────────────────────────────────────────────────────────
ADAPTER_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<div class="desktop-layout">
""" + nav("") + """
<div class="main-content">
<header>
  <span class="logo">PILAR</span><div class="hd"></div>
  <span class="hsub" data-i18n="page_adapter">CSV Adapter</span>
  <div class="hright"><a href="/tutorial" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px">Import</a></div>
</header>
<div class="page pad">

  <div id="adpEmpty">
    <div class="card" style="text-align:center;padding:40px 24px">
      <div style="font-size:9px;letter-spacing:3px;color:var(--teal);text-transform:uppercase;margin-bottom:10px">CSV Adapter</div>
      <div style="font-size:13px;color:var(--text2);margin-bottom:24px;line-height:1.6" data-i18n="adp_desc">Convert any CSV to Pilar format</div>
      <label for="adpInput" class="lz-cta" style="display:inline-flex;width:auto;padding:14px 32px">
        <svg style="width:18px;height:18px;stroke:currentColor;fill:none;flex-shrink:0" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span data-i18n="adp_upload">Upload CSV</span>
      </label>
      <input type="file" id="adpInput" accept=".csv" style="display:none" onchange="adpLoad(this)">
      <div style="margin-top:16px;font-size:10px;color:var(--text3);line-height:1.7" data-i18n="adp_hint">Any delimiter · any column names · unit conversion included</div>
    </div>
  </div>

  <div id="adpMain" style="display:none">
    <div class="card" style="margin-bottom:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <span id="adpFileName" style="font-size:11px;color:var(--text2);font-weight:600">—</span>
        <button onclick="adpReset()" style="background:none;border:1px solid var(--border);border-radius:5px;padding:5px 12px;color:var(--text3);font-size:10px;cursor:pointer;letter-spacing:1px" data-i18n="adp_change">Change</button>
      </div>
      <div id="adpTable"></div>
    </div>
    <div class="card" style="margin-bottom:12px">
      <div class="ctitle" data-i18n="adp_preview">Preview (5 rows)</div>
      <div id="adpPreview" style="overflow-x:auto;font-size:10px"></div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button class="btn" onclick="adpDownload()" data-i18n="adp_download" style="flex:1">Convert & Download</button>
      <button class="btn" onclick="adpSaveToPilar()" id="adpSaveBtn" style="flex:1;background:var(--surface2);border:1px solid var(--teal);color:var(--teal)" data-i18n="adp_save">Save to Pilar</button>
    </div>
    <div id="adpSaveMsg" style="display:none;margin-top:8px;font-size:11px;color:var(--green);text-align:center"></div>
  </div>

  <!-- SAVED FILES -->
  <div id="adpSavedSection" style="display:none">
    <div class="card" style="margin-top:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <span class="ctitle" style="margin:0" data-i18n="adp_my_files">My files</span>
        <button onclick="adpLoadSaved()" style="background:none;border:none;color:var(--teal);font-size:10px;cursor:pointer;letter-spacing:1px" data-i18n="adp_refresh">Refresh</button>
      </div>
      <div id="adpFilesList"></div>
    </div>
  </div>
</div>
</div>
</div>
<script>
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
var _adpRaw=null,_adpDelim=',',_adpHdr=[],_adpMap=[];
var _ADP_KEYS=['vibration','temp_palier','debit','pression_entree','pression_sortie','courant_moteur','temp_moteur','heure_fonctionnement','temperature_ambiante','niveau_huile','tension_reseau'];
var _ADP_LBL_FR={vibration:'Vibration (mm/s)',temp_palier:'Temp. palier (°C)',debit:'Débit (m³/h)',pression_entree:'Pression entrée (bar)',pression_sortie:'Pression sortie (bar)',courant_moteur:'Courant moteur (A)',temp_moteur:'Temp. moteur (°C)',heure_fonctionnement:'Heures fonct. (h)',temperature_ambiante:'Temp. ambiante (°C)',niveau_huile:'Niveau huile (%)',tension_reseau:'Tension réseau (V)'};
var _ADP_LBL_EN={vibration:'Vibration (mm/s)',temp_palier:'Bearing temp (°C)',debit:'Flow rate (m³/h)',pression_entree:'Inlet pressure (bar)',pression_sortie:'Outlet pressure (bar)',courant_moteur:'Motor current (A)',temp_moteur:'Motor temp (°C)',heure_fonctionnement:'Run hours (h)',temperature_ambiante:'Ambient temp (°C)',niveau_huile:'Oil level (%)',tension_reseau:'Supply voltage (V)'};
var _ADP_OUT={vibration:'vibration_mms',temp_palier:'bearing_temp_c',debit:'flow_rate_m3h',pression_entree:'inlet_pressure_bar',pression_sortie:'outlet_pressure_bar',courant_moteur:'motor_current_a',temp_moteur:'motor_temp_c',heure_fonctionnement:'run_hours_h',temperature_ambiante:'ambient_temp_c',niveau_huile:'oil_level_pct',tension_reseau:'supply_voltage_v'};

function adpLoad(inp){
  var f=inp.files[0];if(!f)return;
  var rd=new FileReader();
  rd.onload=function(e){
    _adpRaw=e.target.result;
    var lines=_adpRaw.split('\\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
    if(lines.length<2)return;
    _adpDelim=_csvDelim(lines[0]);
    _adpHdr=lines[0].split(_adpDelim).map(function(s){return s.trim();});
    var autoMap=detectCsvMapping(_adpHdr);
    _adpMap=_adpHdr.map(function(col,idx){
      var pf=null,unit=null;
      _ADP_KEYS.forEach(function(k){if(autoMap[k]&&autoMap[k].idx===idx){pf=k;unit=autoMap[k].unit||null;}});
      var samples=[];
      for(var r=1;r<Math.min(4,lines.length);r++){var vs=lines[r].split(_adpDelim);if(idx<vs.length)samples.push(vs[idx].trim());}
      return {col:col,idx:idx,pf:pf,unit:unit,samples:samples};
    });
    var nr=lines.length-1;
    document.getElementById('adpFileName').textContent=f.name+' — '+_adpHdr.length+' col · '+nr+' rows';
    document.getElementById('adpEmpty').style.display='none';
    document.getElementById('adpMain').style.display='block';
    adpRenderTable();
    adpRenderPreview(lines.slice(1,6));
  };
  rd.readAsText(f);
}

function adpRenderTable(){
  var lang=localStorage.getItem('pilar_lang')||'en';
  var lbl=lang==='fr'?_ADP_LBL_FR:_ADP_LBL_EN;
  var baseOpts='<option value="">'+t('adp_ignore')+'</option>';
  _ADP_KEYS.forEach(function(k){baseOpts+='<option value="'+k+'">'+lbl[k]+'</option>';});
  var tds=function(s,extra){return '<td style="padding:7px 8px;border-bottom:1px solid var(--border)'+(extra||'')+'">'+(s||'')+'</td>';};
  var html='<table style="width:100%;border-collapse:collapse"><thead><tr>';
  ['adp_source','adp_samples','adp_field','adp_unit'].forEach(function(k){html+='<th style="text-align:left;padding:7px 8px;color:var(--text3);font-size:9px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border)">'+t(k)+'</th>';});
  html+='</tr></thead><tbody>';
  _adpMap.forEach(function(m,i){
    var dot=m.pf?'<span style="width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;margin-right:6px;flex-shrink:0"></span>':'<span style="width:7px;height:7px;border-radius:50%;background:var(--border);display:inline-block;margin-right:6px;flex-shrink:0"></span>';
    var sel=baseOpts.replace('value="'+(m.pf||'')+'"','value="'+(m.pf||'')+'" selected');
    var selEl='<select onchange="adpSetField('+i+',this.value)" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 6px;font-size:10px;width:100%">'+sel+'</select>';
    var unitEl='<span style="font-size:10px;color:var(--text3)">—</span>';
    // Pump fields use fixed SI units — no unit selector needed
    html+='<tr>'+tds('<span style="display:flex;align-items:center">'+dot+'<strong style="font-size:11px;color:var(--text)">'+esc(m.col)+'</strong></span>')+tds('<span style="color:var(--text3);font-size:10px">'+esc(m.samples.slice(0,3).join(', '))+'</span>')+tds(selEl)+tds(unitEl)+'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('adpTable').innerHTML=html;
}

function adpSetField(i,val){
  _adpMap[i].pf=val||null;_adpMap[i].unit=null;
  adpRenderTable();
  var lines=_adpRaw.split('\\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
  adpRenderPreview(lines.slice(1,6));
}
function adpSetUnit(i,val){_adpMap[i].unit=val;}

function _adpConv(raw,pf,unit){
  // Pump fields — all values are already in SI units, no conversion needed
  var v=parseFloat(raw.replace(',','.'));if(isNaN(v))return '';
  return Math.round(v*1000)/1000;
}

function _adpRowOut(rawVals){
  var out={};
  _adpMap.forEach(function(m){
    if(!m.pf)return;
    var raw=m.idx<rawVals.length?rawVals[m.idx].trim():'';
    out[m.pf]=_adpConv(raw,m.pf,m.unit);
  });
  return out;
}

function adpRenderPreview(dataLines){
  if(!dataLines||!dataLines.length){document.getElementById('adpPreview').innerHTML='';return;}
  var active=_ADP_KEYS.filter(function(k){return _adpMap.some(function(m){return m.pf===k;});});
  if(!active.length){document.getElementById('adpPreview').innerHTML='<span style="color:var(--text3)">'+t('adp_no_map')+'</span>';return;}
  var th=function(s){return '<th style="padding:6px 10px;color:var(--teal);font-size:9px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);white-space:nowrap">'+s+'</th>';};
  var td=function(s){return '<td style="padding:6px 10px;border-bottom:1px solid var(--border);color:var(--text2);white-space:nowrap">'+s+'</td>';};
  var html='<table style="border-collapse:collapse;min-width:100%"><thead><tr>';
  active.forEach(function(k){html+=th(_ADP_OUT[k]);});
  html+='</tr></thead><tbody>';
  dataLines.forEach(function(line){
    var rv=line.split(_adpDelim);var row=_adpRowOut(rv);
    html+='<tr>';active.forEach(function(k){html+=td(row[k]!==undefined?row[k]:'');});html+='</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('adpPreview').innerHTML=html;
}

function adpDownload(){
  var csv=adpGetCSV();
  if(!csv)return;
  var blob=new Blob([csv],{type:'text/csv'});
  var a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='pilar_adapted.csv';
  document.body.appendChild(a);a.click();document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function adpReset(){
  _adpRaw=null;_adpHdr=[];_adpMap=[];
  document.getElementById('adpInput').value='';
  document.getElementById('adpEmpty').style.display='block';
  document.getElementById('adpMain').style.display='none';
}

function adpGetCSV(){
  var active=_ADP_KEYS.filter(function(k){return _adpMap.some(function(m){return m.pf===k;});});
  if(!active.length)return null;
  var lines=_adpRaw.split('\\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
  var csv=active.map(function(k){return _ADP_OUT[k];}).join(',')+'\\n';
  for(var i=1;i<lines.length;i++){
    var rv=lines[i].split(_adpDelim);var row=_adpRowOut(rv);
    csv+=active.map(function(k){return row[k]!==undefined?String(row[k]):'';}).join(',')+'\\n';
  }
  return csv;
}

function adpSaveToPilar(){
  var csv=adpGetCSV();
  if(!csv){alert(t('adp_no_map'));return;}
  var fname=document.getElementById('adpFileName').textContent.split(' — ')[0]||'imported.csv';
  var base=fname.replace(/\\.[^.]+$/,'')+'_pilar.csv';
  var btn=document.getElementById('adpSaveBtn');
  btn.disabled=true;
  fetch('/api/save_csv',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:base,content:csv})})
    .then(function(r){
      if(r.status===401){window.location.href='/login';return null;}
      return r.json();
    })
    .then(function(d){
      if(!d)return;
      btn.disabled=false;
      if(d.id){
        var msg=document.getElementById('adpSaveMsg');
        msg.textContent=t('adp_saved_ok')+' — '+d.filename+' ('+d.rows+' rows)';
        msg.style.display='block';
        setTimeout(function(){msg.style.display='none';},3000);
        adpLoadSaved();
      }
    }).catch(function(){btn.disabled=false;});
}

function adpLoadSaved(){
  fetch('/api/saved_files').then(function(r){return r.json();}).then(function(files){
    var sec=document.getElementById('adpSavedSection');
    var lst=document.getElementById('adpFilesList');
    if(!files||!files.length){
      lst.innerHTML='<div style="font-size:11px;color:var(--text3);text-align:center;padding:12px">'+t('adp_no_files')+'</div>';
      sec.style.display='block';return;
    }
    var html='';
    files.forEach(function(f){
      var d=new Date(f.created_at).toLocaleDateString();
      html+='<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">';
      html+='<div><div style="font-size:11px;color:var(--text);font-weight:600">'+esc(f.filename)+'</div><div style="font-size:10px;color:var(--text3)">'+f.rows+' rows · '+d+'</div></div>';
      html+='<div style="display:flex;gap:6px">';
      html+='<button onclick="adpUseFile('+f.id+')" style="background:var(--teal);border:none;border-radius:4px;padding:5px 10px;color:#fff;font-size:10px;cursor:pointer">'+t('adp_load')+'</button>';
      html+='<button onclick="adpDelFile('+f.id+',this)" style="background:none;border:1px solid var(--border);border-radius:4px;padding:5px 10px;color:var(--text3);font-size:10px;cursor:pointer">'+t('adp_delete')+'</button>';
      html+='</div></div>';
    });
    lst.innerHTML=html;
    sec.style.display='block';
  });
}

function adpUseFile(id){
  fetch('/api/saved_files/'+id).then(function(r){return r.json();}).then(function(f){
    _adpRaw=f.content;
    var lines=_adpRaw.split('\\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
    if(lines.length<2)return;
    _adpDelim=_csvDelim(lines[0]);
    _adpHdr=lines[0].split(_adpDelim).map(function(s){return s.trim();});
    var autoMap=detectCsvMapping(_adpHdr);
    _adpMap=_adpHdr.map(function(col,idx){
      var pf=null,unit=null;
      _ADP_KEYS.forEach(function(k){if(autoMap[k]&&autoMap[k].idx===idx){pf=k;unit=autoMap[k].unit||null;}});
      var samples=[];
      for(var r=1;r<Math.min(4,lines.length);r++){var vs=lines[r].split(_adpDelim);if(idx<vs.length)samples.push(vs[idx].trim());}
      return {col:col,idx:idx,pf:pf,unit:unit,samples:samples};
    });
    document.getElementById('adpFileName').textContent=f.filename+' — '+_adpHdr.length+' col · '+f.rows+' rows';
    document.getElementById('adpEmpty').style.display='none';
    document.getElementById('adpMain').style.display='block';
    adpRenderTable();
    adpRenderPreview(lines.slice(1,6));
  });
}

function adpDelFile(id,btn){
  btn.disabled=true;
  fetch('/api/saved_files/'+id+'/delete',{method:'POST'}).then(function(){adpLoadSaved();});
}

document.addEventListener('DOMContentLoaded',adpLoadSaved);
</script></body></html>"""

# ── DEMO PAGE ─────────────────────────────────────────────────────────────────
DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilar — Interactive Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--surface:#0e1118;--surface2:#141820;--surface3:#1e2433;
  --border:#1e2433;--border2:#252d3d;
  --teal:#0d9488;--teal2:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;--green-dim:rgba(5,150,105,0.08);
  --amber:#d97706;--purple:#bf5af2;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --r:12px;--r-sm:8px;--r-xs:6px;
  --shadow:0 4px 24px rgba(0,0,0,0.5);--shadow-sm:0 2px 8px rgba(0,0,0,0.4);
}
html{scroll-behavior:smooth}
body{font-family:-apple-system,'SF Pro Display','SF Pro Text','Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

/* NAV */
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);background:rgba(28,28,30,0.92);border-bottom:1px solid var(--border)}
.nav-logo{font-size:16px;font-weight:700;letter-spacing:0.04em;color:var(--text);text-decoration:none}
.nav-right{display:flex;align-items:center;gap:8px}
.nav-tag{font-size:10px;font-weight:600;letter-spacing:0.04em;color:var(--teal2);text-transform:uppercase;border:1px solid rgba(13,148,136,0.3);padding:4px 10px;border-radius:20px}
.nav-right a{font-size:13px;font-weight:500;text-decoration:none;padding:8px 16px;border-radius:var(--r)}
.nav-login{color:var(--text2);border:1px solid var(--border2)}
.nav-login:hover{border-color:var(--teal);color:var(--teal2)}
.nav-register{background:var(--teal);color:#fff}
.nav-register:hover{opacity:0.85}

/* LAYOUT */
.wrap{max-width:1100px;margin:0 auto;padding:90px 32px 80px}
.page-eyebrow{font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--teal2);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.page-eyebrow::before{content:'';display:inline-block;width:20px;height:1px;background:var(--teal2)}
h1{font-size:clamp(40px,5vw,70px);font-weight:700;line-height:1.05;letter-spacing:-0.025em;color:var(--text);margin-bottom:16px}
h1 span{color:var(--teal2)}
.page-sub{font-size:16px;color:var(--text2);line-height:1.8;max-width:600px;margin-bottom:56px}

/* AUDIO SECTION — hidden, no audio element */
.audio-section{display:none}
.lang-toggle{display:flex;gap:2px;background:var(--surface2);border:1px solid var(--border);padding:3px;border-radius:var(--r-sm)}
.lang-btn{padding:5px 14px;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;background:transparent;color:var(--text3);transition:all .15s}
.lang-btn.active{background:var(--teal);color:#fff}

/* DEMO GRID */
.demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}

/* SCENARIO PICKER */
.scenarios{background:var(--surface);border:1px solid var(--border);padding:24px;border-radius:var(--r);box-shadow:var(--shadow-sm)}
.block-label{font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--teal2);text-transform:uppercase;margin-bottom:20px}
.scenario-list{display:grid;gap:8px}
.scenario-btn{width:100%;text-align:left;padding:14px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);cursor:pointer;transition:all .18s}
.scenario-btn:hover{border-color:var(--teal2);background:var(--teal-dim)}
.scenario-btn.active{border-color:var(--teal2);background:var(--teal-dim)}
.sc-name{font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px}
.sc-desc{font-size:12px;color:var(--text3)}
.scenario-btn.active .sc-name{color:var(--teal2)}

/* RESULT PANEL */
.result-panel{background:var(--surface);border:1px solid var(--border);padding:24px;border-radius:var(--r);box-shadow:var(--shadow-sm)}
.sensor-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}
.sensor-card{background:var(--surface2);border:1px solid var(--border);padding:14px 16px;border-radius:var(--r-sm)}
.sensor-label{font-size:10px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:6px}
.sensor-val{font-size:20px;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums}
.sensor-val.warn{color:var(--amber)}
.sensor-val.danger{color:var(--red)}
.sensor-val.ok{color:var(--green)}
.sensor-unit{font-size:11px;font-weight:400;color:var(--text3);margin-left:3px}

/* RISK RESULT */
.risk-block{border:1px solid var(--border);padding:20px 24px;margin-bottom:14px;border-radius:var(--r)}
.risk-block.danger{border-color:rgba(255,69,58,0.4);background:var(--red-dim)}
.risk-block.warn{border-color:rgba(255,159,10,0.4);background:rgba(255,159,10,0.08)}
.risk-block.ok{border-color:rgba(48,209,88,0.3);background:var(--green-dim)}
.risk-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.risk-label{font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase}
.risk-pct{font-size:36px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.risk-block.danger .risk-pct{color:var(--red)}
.risk-block.warn .risk-pct{color:var(--amber)}
.risk-block.ok .risk-pct{color:var(--green)}
.risk-bar-bg{height:6px;background:var(--surface3);border-radius:6px;overflow:hidden;margin-bottom:8px}
.risk-bar-fill{height:100%;border-radius:6px;transition:width .6s ease}
.risk-verdict{font-size:13px;color:var(--text2);line-height:1.6}

/* ZONES */
.zones{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:14px}
.zone{padding:8px;text-align:center;border:1px solid var(--border);border-radius:var(--r-xs);background:var(--surface2)}
.zone-name{font-size:10px;font-weight:700;letter-spacing:0.04em}
.zone-status{font-size:9px;margin-top:4px;font-weight:600}
.zone.high{border-color:rgba(255,69,58,0.4);background:var(--red-dim)}
.zone.high .zone-name{color:var(--red)}
.zone.high .zone-status{color:var(--red)}
.zone.med{border-color:rgba(255,159,10,0.3);background:rgba(255,159,10,0.08)}
.zone.med .zone-name{color:var(--amber)}
.zone.med .zone-status{color:var(--amber)}
.zone.low .zone-name{color:var(--text3)}
.zone.low .zone-status{color:var(--text3)}

/* CTA */
.demo-cta{margin-top:40px;border:1px solid var(--border);padding:32px;background:var(--surface);border-radius:var(--r);box-shadow:var(--shadow-sm);display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap}
.demo-cta-text h3{font-size:28px;font-weight:700;letter-spacing:-0.02em;color:var(--text);margin-bottom:6px}
.demo-cta-text p{font-size:14px;color:var(--text2);line-height:1.7}
.btn-register{display:inline-block;padding:14px 28px;background:var(--teal);color:#fff;text-decoration:none;font-size:14px;font-weight:600;border-radius:12px;transition:opacity .18s;white-space:nowrap}
.btn-register:hover{opacity:0.85}

@media(max-width:800px){
  .demo-grid{grid-template-columns:1fr}
  .zones{grid-template-columns:repeat(5,1fr)}
  .sensor-grid{grid-template-columns:1fr 1fr}
  .demo-cta{flex-direction:column}
}
@media(max-width:480px){
  .zones{grid-template-columns:repeat(3,1fr)}
  .wrap{padding:80px 20px 60px}
  nav{padding:0 20px}
}
</style>
</head>
<body>
<nav>
  <a href="/" class="nav-logo">PILAR</a>
  <div class="nav-right">
    <div class="nav-tag">Interactive Demo</div>
    <a href="/login" class="nav-login">Sign In</a>
    <a href="/register" class="nav-register">Get Started Free</a>
  </div>
</nav>

<div class="wrap">
  <div class="page-eyebrow">Live Demo</div>
  <h1>See Pilar<br><span>in action</span></h1>
  <p class="page-sub">Explore live failure scenarios below to see the model in action.</p>

  <!-- WALKTHROUGH STEPS -->
  <div style="margin-bottom:48px">
    <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--teal2);text-transform:uppercase;margin-bottom:24px;display:flex;align-items:center;gap:10px"><span style="display:inline-block;width:20px;height:1px;background:var(--teal2)"></span>How it works</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow-sm)">
      <div style="padding:20px 22px;border-right:1px solid var(--border);background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">01 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">Enter sensor readings</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">Vibration, bearing temp, flow rate, pressures, motor current, run hours — manually or via API/CSV.</div>
      </div>
      <div style="padding:20px 22px;border-right:1px solid var(--border);background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">02 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">Or upload a CSV file</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">Columns: vibration, temp_palier, debit, pression_entree, pression_sortie, courant_moteur, temp_moteur, heure_fonctionnement.</div>
      </div>
      <div style="padding:20px 22px;border-right:1px solid var(--border);background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">03 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">Auto-detect &amp; map columns</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">Pilar auto-detects your column names and maps them — confirm or correct before analysis.</div>
      </div>
      <div style="padding:20px 22px;border-right:1px solid var(--border);background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">04 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">Three AI layers run in parallel</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">RandomForest failure probability + Isolation Forest anomaly score + SHAP explanations + RUL forecast — all in one request.</div>
      </div>
      <div style="padding:20px 22px;border-right:1px solid var(--border);background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">05 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">Full AI result instantly</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">Risk %, anomaly score, top contributing sensors, failure zones — and estimated remaining life.</div>
      </div>
      <div style="padding:20px 22px;background:var(--surface)">
        <div style="font-size:10px;font-weight:600;color:var(--teal2);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px">06 / 06</div>
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px">History, Twin &amp; alerts</div>
        <div style="font-size:12px;color:var(--text2);line-height:1.7">Full analysis history in History. Digital Twin shows RUL trend. Email alerts auto-escalate.</div>
      </div>
    </div>
  </div>

  <!-- DEMO SCENARIOS + RESULT -->
  <div class="demo-grid">
    <!-- SCENARIO PICKER -->
    <div class="scenarios">
      <div class="block-label">Choose a scenario</div>
      <div class="scenario-list">
        <button class="scenario-btn active" onclick="loadScenario(0)">
          <div class="sc-name">Normal operation</div>
          <div class="sc-desc">All parameters within safe range</div>
        </button>
        <button class="scenario-btn" onclick="loadScenario(1)">
          <div class="sc-name">Bearing wear developing</div>
          <div class="sc-desc">Progressive wear, risk building up</div>
        </button>
        <button class="scenario-btn" onclick="loadScenario(2)">
          <div class="sc-name">Cavitation + seal failure</div>
          <div class="sc-desc">Low flow, pressure drop, impeller at risk</div>
        </button>
        <button class="scenario-btn" onclick="loadScenario(3)">
          <div class="sc-name">Critical — imminent failure</div>
          <div class="sc-desc">Multiple zones in alert, action required</div>
        </button>
      </div>
    </div>

    <!-- RESULT PANEL -->
    <div class="result-panel">
      <div class="block-label">Live sensor reading</div>
      <div class="sensor-grid">
        <div class="sensor-card">
          <div class="sensor-label">Vibration</div>
          <div class="sensor-val ok" id="v-vib">2.1<span class="sensor-unit">mm/s</span></div>
        </div>
        <div class="sensor-card">
          <div class="sensor-label">Bearing temp</div>
          <div class="sensor-val ok" id="v-tbear">52.4<span class="sensor-unit">°C</span></div>
        </div>
        <div class="sensor-card">
          <div class="sensor-label">Flow rate</div>
          <div class="sensor-val ok" id="v-debit">42.0<span class="sensor-unit">m³/h</span></div>
        </div>
        <div class="sensor-card">
          <div class="sensor-label">Outlet pressure</div>
          <div class="sensor-val ok" id="v-pout">3.8<span class="sensor-unit">bar</span></div>
        </div>
        <div class="sensor-card">
          <div class="sensor-label">Run hours</div>
          <div class="sensor-val ok" id="v-hrun">1240<span class="sensor-unit">h</span></div>
        </div>
        <div class="sensor-card">
          <div class="sensor-label">Status</div>
          <div class="sensor-val ok" id="v-status" style="font-size:16px">Normal</div>
        </div>
      </div>

      <div class="block-label">Failure zones</div>
      <div class="zones" id="zones-row">
        <div class="zone low"><div class="zone-name">CAV</div><div class="zone-status">LOW</div></div>
        <div class="zone low"><div class="zone-name">ROL</div><div class="zone-status">LOW</div></div>
        <div class="zone low"><div class="zone-name">ETN</div><div class="zone-status">LOW</div></div>
        <div class="zone low"><div class="zone-name">IMP</div><div class="zone-status">LOW</div></div>
        <div class="zone low"><div class="zone-name">MOT</div><div class="zone-status">LOW</div></div>
      </div>

      <div class="risk-block ok" id="risk-block">
        <div class="risk-row">
          <div class="risk-label">Failure risk</div>
          <div class="risk-pct" id="risk-pct">3%</div>
        </div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" id="risk-bar" style="width:3%;background:var(--green)"></div></div>
        <div class="risk-verdict" id="risk-verdict">All systems nominal. No intervention required.</div>
      </div>

      <div id="ai-insights" style="margin-top:16px;display:flex;flex-direction:column;gap:10px">
        <div style="padding:10px 14px;background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:5px;display:flex;align-items:center;justify-content:space-between">
          <div>
            <div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;margin-bottom:2px">Anomaly score</div>
            <div style="font-size:9px;color:var(--text3)">Isolation Forest · unsupervised</div>
          </div>
          <span id="demo-anomaly" style="font-size:18px;font-weight:800;color:#059669">12/100</span>
        </div>
        <div style="padding:10px 14px;background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:5px">
          <div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;margin-bottom:10px">AI Explanation · SHAP top-3</div>
          <div id="demo-shap"></div>
        </div>
        <div id="demo-rul-wrap" style="display:none;padding:10px 14px;background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:5px;font-size:11px;color:var(--text2)">
          &#x23F1; Est. remaining life: <strong id="demo-rul-val" style="color:var(--green)"></strong>
          <span id="demo-rul-conf" style="font-size:9px;color:var(--text3);margin-left:4px"></span>
        </div>
      </div>
    </div>
  </div>

  <!-- CTA -->
  <div class="demo-cta">
    <div class="demo-cta-text">
      <h3>Ready to connect your machines?</h3>
      <p>Free account — upload your first CSV in under 2 minutes. No credit card required.</p>
    </div>
    <a href="/register" class="btn-register">Create free account</a>
  </div>
</div>

<script>
var SCENARIOS=[
  {
    name:'Normal operation',
    vib:1.8,tbear:44.8,debit:42.0,pout:108,hrun:1240,status:'Normal',
    risk:3,bar_color:'var(--green)',block_class:'ok',
    verdict:'All systems nominal. No intervention required.',
    zones:['low','low','low','low','low'],
    zone_labels:['LOW','LOW','LOW','LOW','LOW'],
    anomaly:10,anomaly_color:'#059669',
    shap:[{f:'Vibration',v:'18%',d:'down'},{f:'Run hours',v:'12%',d:'down'},{f:'Flow rate',v:'8%',d:'up'}],
    rul:null
  },
  {
    name:'Bearing wear developing',
    vib:5.8,tbear:74.2,debit:38.5,pout:102,hrun:3820,status:'Warning',
    risk:38,bar_color:'var(--amber)',block_class:'warn',
    verdict:'Bearing temperature elevated. Schedule inspection within 48h.',
    zones:['low','med','low','low','low'],
    zone_labels:['LOW','MED','LOW','LOW','LOW'],
    anomaly:44,anomaly_color:'#d97706',
    shap:[{f:'Bearing temp',v:'41%',d:'up'},{f:'Vibration',v:'28%',d:'up'},{f:'Run hours',v:'15%',d:'up'}],
    rul:312,rul_conf:'medium'
  },
  {
    name:'Cavitation + seal failure',
    vib:9.3,tbear:81.6,debit:22.1,pout:88,hrun:5600,status:'Critical',
    risk:67,bar_color:'var(--amber)',block_class:'warn',
    verdict:'Low flow and pressure drop detected. Inspect inlet and impeller for cavitation.',
    zones:['high','med','low','med','low'],
    zone_labels:['HIGH','MED','LOW','MED','LOW'],
    anomaly:71,anomaly_color:'#dc2626',
    shap:[{f:'Flow rate',v:'38%',d:'down'},{f:'Outlet pressure',v:'29%',d:'down'},{f:'Vibration',v:'21%',d:'up'}],
    rul:74,rul_conf:'high'
  },
  {
    name:'Critical — imminent failure',
    vib:15.7,tbear:96.4,debit:14.8,pout:71,hrun:7200,status:'STOP',
    risk:91,bar_color:'var(--red)',block_class:'danger',
    verdict:'CRITICAL — Multiple failure zones active. Stop pump and inspect immediately.',
    zones:['high','high','low','high','med'],
    zone_labels:['HIGH','HIGH','LOW','HIGH','MED'],
    anomaly:88,anomaly_color:'#dc2626',
    shap:[{f:'Vibration',v:'35%',d:'up'},{f:'Bearing temp',v:'31%',d:'up'},{f:'Flow rate',v:'22%',d:'down'}],
    rul:8,rul_conf:'high'
  }
];

function loadScenario(i){
  document.querySelectorAll('.scenario-btn').forEach(function(b,j){b.classList.toggle('active',j===i)});
  var s=SCENARIOS[i];

  function setSensor(id,v,cls){var el=document.getElementById(id);if(!el)return;el.childNodes[0].textContent=v;el.className='sensor-val'+(cls?' '+cls:'');}
  setSensor('v-vib',s.vib,s.vib>10?' danger':s.vib>6?' warn':' ok');
  setSensor('v-tbear',s.tbear,s.tbear>85?' danger':s.tbear>70?' warn':' ok');
  setSensor('v-debit',s.debit,s.debit<20?' danger':s.debit<30?' warn':' ok');
  setSensor('v-pout',s.pout,s.pout<80?' danger':s.pout<95?' warn':' ok');
  setSensor('v-hrun',s.hrun,s.hrun>6000?' warn':' ok');
  var stEl=document.getElementById('v-status');if(stEl){stEl.textContent=s.status;stEl.className='sensor-val'+(s.status==='STOP'?' danger':s.status==='Critical'?' warn':' ok');}

  var zones=['CAV','ROL','ETN','IMP','MOT'];
  var zRow=document.getElementById('zones-row');
  zRow.innerHTML='';
  for(var z=0;z<5;z++){
    var d=document.createElement('div');
    d.className='zone '+s.zones[z];
    d.innerHTML='<div class="zone-name">'+zones[z]+'</div><div class="zone-status">'+s.zone_labels[z]+'</div>';
    zRow.appendChild(d);
  }

  var block=document.getElementById('risk-block');
  block.className='risk-block '+s.block_class;
  document.getElementById('risk-pct').textContent=s.risk+'%';
  var bar=document.getElementById('risk-bar');
  bar.style.width=s.risk+'%';
  bar.style.background=s.bar_color;
  document.getElementById('risk-verdict').textContent=s.verdict;

  // Anomaly score
  var aEl=document.getElementById('demo-anomaly');
  if(aEl){aEl.textContent=s.anomaly+'/100';aEl.style.color=s.anomaly_color;}

  // SHAP top-3
  var shapEl=document.getElementById('demo-shap');
  if(shapEl&&s.shap){
    shapEl.innerHTML=s.shap.map(function(x){
      return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px"><span style="font-size:11px;color:var(--text2);flex:1">'+x.f+'</span><span style="font-size:12px;font-weight:700;color:'+(x.d==='up'?'var(--red)':'var(--green)')+'">'+x.v+' '+(x.d==='up'?'\u2191':'\u2193')+'</span></div>';
    }).join('');
  }

  // RUL
  var rulWrap=document.getElementById('demo-rul-wrap');
  if(rulWrap){
    if(s.rul!=null){
      rulWrap.style.display='block';
      var rulColor=s.rul<24?'var(--red)':s.rul<72?'var(--amber)':'var(--green)';
      document.getElementById('demo-rul-val').textContent='~'+s.rul+'h';
      document.getElementById('demo-rul-val').style.color=rulColor;
      document.getElementById('demo-rul-conf').textContent=s.rul_conf?'('+s.rul_conf+')':'';
    } else {
      rulWrap.style.display='none';
    }
  }
}

</script>
</body>
</html>"""

# ── MACHINE SPACE ─────────────────────────────────────────────────────────────
MACHINE_SPACE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ machine.name }} — Pilar</title>
<style>
:root{
  --bg:#07090f;--surface:#0e1118;--surface2:#141820;--surface3:#1e2433;
  --border:#1e2433;--border2:#252d3d;
  --teal:#0d9488;--teal2:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;--green-dim:rgba(5,150,105,0.08);
  --amber:#d97706;--amber-dim:rgba(255,159,10,0.12);
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --r:12px;--r-sm:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;}
body{font-family:'Inter','Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:var(--text);}
.desktop-layout{display:flex;height:100vh;overflow:hidden;}
.sidebar{width:200px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;overflow:hidden;}
.main-content{flex:1;display:flex;flex-direction:column;overflow:hidden;}
header{height:48px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;padding:0 20px;background:var(--surface);flex-shrink:0;}
.logo{font-size:13px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;}
.hd{width:1px;height:18px;background:var(--border2);}
.hsub{font-size:10px;letter-spacing:1.2px;color:var(--text3);text-transform:uppercase;}
.sidebar-logo{padding:16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;}
.logo-mark{width:28px;height:28px;border-radius:7px;background:#0d9488;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.logo-text{font-size:13px;font-weight:700;letter-spacing:3px;color:#14b8a6;text-transform:uppercase;}
.logo-sub{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:1px;}
.sidebar-nav{flex:1;padding:8px 0;overflow-y:auto;}
.sidebar-section{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text3);padding:12px 16px 4px;}
.ni{display:flex;align-items:center;gap:10px;padding:9px 16px;color:var(--text3);text-decoration:none;font-size:13px;transition:color .15s,background .15s;cursor:pointer;border:none;background:none;width:100%;text-align:left;}
.ni:hover{color:var(--text2);background:var(--surface2);}
.ni.on{color:#14b8a6;background:rgba(13,148,136,0.08);}
.ni svg{width:16px;height:16px;flex-shrink:0;}
.ni-label{font-size:13px;}
.sidebar-footer{border-top:1px solid var(--border);padding:10px 12px;}
.sync-status-bar{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--text3);padding:4px 0;}
.sync-dot{width:6px;height:6px;border-radius:50%;background:var(--text3);}
.sync-dot.online{background:#059669;}
.sync-dot.offline{background:var(--amber);}
.bottom-nav{display:none;}
.page{flex:1;overflow-y:auto;overflow-x:hidden;}
.page::-webkit-scrollbar{width:4px;}
.page::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.pad{padding:20px;max-width:1100px;}
.machine-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:28px;}
.machine-title{font-size:26px;font-weight:700;letter-spacing:-0.03em;}
.machine-sub{font-size:13px;color:var(--text3);margin-top:5px;}
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.status-ok{background:var(--green-dim);color:var(--green);border:1px solid rgba(48,209,88,.3);}
.status-warn{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(255,159,10,.3);}
.status-crit{background:var(--red-dim);color:var(--red);border:1px solid rgba(255,69,58,.3);}
.status-none{background:rgba(84,84,88,.12);color:var(--text3);border:1px solid var(--border);}
/* Tabs */
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--border);margin-bottom:28px;}
.tab{padding:10px 18px;font-size:13px;font-weight:500;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:color .15s;}
.tab.active{color:var(--teal2);border-bottom-color:var(--teal);}
.tab:hover:not(.active){color:var(--text2);}
.tab-pane{display:none;}
.tab-pane.active{display:block;}
/* Forms */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.form-group{display:flex;flex-direction:column;gap:6px;}
.form-group.full{grid-column:1/-1;}
.form-label{font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.04em;}
.form-input,.form-select{background:rgba(120,120,128,0.18);border:1px solid var(--border2);border-radius:var(--r-sm);padding:10px 12px;font-size:13px;color:var(--text);outline:none;transition:border-color .15s;width:100%;}
.form-input:focus,.form-select:focus{border-color:var(--teal);}
.form-select option{background:#0e1118;}
.form-hint{font-size:11px;color:var(--text3);}
.section-title{font-size:12px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin:24px 0 14px;}
.section-title:first-child{margin-top:0;}
/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:24px;margin-bottom:16px;}
/* Monitor */
.monitor-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
.stat-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;}
.stat-label{font-size:10px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;}
.stat-val{font-size:28px;font-weight:700;line-height:1;}
.stat-sub{font-size:11px;color:var(--text3);margin-top:4px;}
.zones-list{display:flex;flex-direction:column;gap:8px;}
.zone-row{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-sm);}
.zone-name{font-size:13px;font-weight:600;}
.zone-bar-wrap{flex:1;margin:0 16px;height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.zone-bar-fill{height:100%;border-radius:2px;background:var(--amber);}
.zone-pct{font-size:12px;font-weight:700;color:var(--amber);min-width:36px;text-align:right;}
/* CSV upload */
.upload-area{border:2px dashed var(--border2);border-radius:var(--r);padding:48px;text-align:center;cursor:pointer;transition:border-color .2s;}
.upload-area:hover,.upload-area.drag{border-color:var(--teal);}
.upload-area input{display:none;}
.upload-icon{font-size:32px;margin-bottom:12px;}
.upload-label{font-size:14px;font-weight:600;color:var(--text2);margin-bottom:6px;}
.upload-hint{font-size:12px;color:var(--text3);}
/* Notes */
.note-input{width:100%;background:rgba(120,120,128,0.18);border:1px solid var(--border2);border-radius:var(--r-sm);padding:12px;font-size:13px;color:var(--text);outline:none;resize:vertical;min-height:80px;font-family:inherit;}
.note-input:focus{border-color:var(--teal);}
.notes-list{display:flex;flex-direction:column;gap:10px;margin-top:16px;}
.note-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px 16px;}
.note-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.note-author{font-size:11px;font-weight:600;color:var(--teal2);}
.note-time{font-size:11px;color:var(--text3);}
.note-content{font-size:13px;color:var(--text2);line-height:1.6;}
.note-del{background:none;border:none;color:var(--text3);cursor:pointer;font-size:14px;padding:2px 6px;}
.note-del:hover{color:var(--red);}
/* History table */
.history-table{width:100%;border-collapse:collapse;font-size:13px;}
.history-table th{text-align:left;padding:10px 14px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text3);border-bottom:1px solid var(--border);}
.history-table td{padding:10px 14px;border-bottom:1px solid var(--border);color:var(--text2);}
.history-table tr:last-child td{border-bottom:none;}
/* Buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;font-size:13px;font-weight:600;border:none;border-radius:var(--r-sm);cursor:pointer;transition:opacity .15s;text-decoration:none;}
.btn-primary{background:var(--teal);color:#fff;}
.btn-primary:hover{opacity:.85;}
.btn-ghost{background:transparent;color:var(--text2);border:1px solid var(--border2);}
.btn-ghost:hover{border-color:var(--teal);color:var(--teal2);}
.btn-sm{padding:6px 12px;font-size:12px;}
.toast{position:fixed;bottom:24px;right:24px;background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r-sm);padding:12px 18px;font-size:13px;color:var(--text2);box-shadow:0 4px 24px rgba(0,0,0,.5);z-index:999;opacity:0;transform:translateY(8px);transition:all .25s;}
.toast.show{opacity:1;transform:translateY(0);}
.result-box{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:20px;margin-top:16px;display:none;}
@media(max-width:700px){.form-grid{grid-template-columns:1fr;}.monitor-grid{grid-template-columns:1fr 1fr;}.pad{padding:16px;}}
</style>
</head>
<body>
<div class="desktop-layout">
{{ sidebar_html | safe }}
<div class="main-content">
<header>
  <span class="logo">PILAR</span>
  <div class="hd"></div>
  <span class="hsub" style="display:flex;align-items:center;gap:6px;text-transform:none;font-size:12px;letter-spacing:0;">
    <a href="/dashboard" style="color:var(--text3);text-decoration:none">Fleet</a>
    <span style="color:var(--border2)">›</span>
    <span style="color:var(--text)">{{ machine.name }}</span>
  </span>
</header>
<div class="page pad">
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;">
  <div>
    <div class="machine-title">{{ machine.name }}</div>
    <div class="machine-sub" id="machine-sub">{{ machine.description or (machine.pump_type or 'Pump') + ' · ' + (machine.location or 'No location set') }}</div>
  </div>
  <div id="status-pill" class="status-pill status-none">No data yet</div>
</div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('info')">Info</div>
    <div class="tab" onclick="switchTab('data')">Data</div>
    <div class="tab" onclick="switchTab('monitor')">Monitor</div>
    <div class="tab" onclick="switchTab('notes')">Notes</div>
    <div class="tab" onclick="switchTab('history')">History</div>
  </div>

  <!-- INFO TAB -->
  <div id="tab-info" class="tab-pane active">
    <p style="font-size:13px;color:var(--text3);margin-bottom:20px;">Machine profile — the model uses these values to adapt its thresholds and imputation to this specific machine.</p>
    <div class="section-title">Identity</div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Machine name</label>
        <input class="form-input" id="f-name" value="{{ machine.name }}">
      </div>
      <div class="form-group">
        <label class="form-label">Serial number</label>
        <input class="form-input" id="f-serial" placeholder="e.g. SN-20240315" value="{{ machine.serial_number or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Location</label>
        <input class="form-input" id="f-location" placeholder="e.g. Building A, Line 3" value="{{ machine.location or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Install date</label>
        <input class="form-input" id="f-install" type="date" value="{{ machine.install_date.isoformat() if machine.install_date else '' }}">
      </div>
      <div class="form-group full">
        <label class="form-label">Description / notes on machine type</label>
        <input class="form-input" id="f-desc" placeholder="Optional description" value="{{ machine.description or '' }}">
      </div>
    </div>

    <div class="section-title">Pump specs</div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Pump type</label>
        <select class="form-select" id="f-pump-type">
          <option value="centrifuge" {% if machine.pump_type=='centrifuge' %}selected{% endif %}>Centrifugal</option>
          <option value="axiale" {% if machine.pump_type=='axiale' %}selected{% endif %}>Axial</option>
          <option value="volumetrique" {% if machine.pump_type=='volumetrique' %}selected{% endif %}>Positive displacement</option>
          <option value="engrenages" {% if machine.pump_type=='engrenages' %}selected{% endif %}>Gear pump</option>
          <option value="peristaltique" {% if machine.pump_type=='peristaltique' %}selected{% endif %}>Peristaltic</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Fluid type</label>
        <select class="form-select" id="f-fluid-type">
          <option value="eau" {% if machine.fluid_type=='eau' %}selected{% endif %}>Water</option>
          <option value="huile" {% if machine.fluid_type=='huile' %}selected{% endif %}>Oil</option>
          <option value="acide" {% if machine.fluid_type=='acide' %}selected{% endif %}>Acid</option>
          <option value="boue" {% if machine.fluid_type=='boue' %}selected{% endif %}>Slurry</option>
          <option value="solvant" {% if machine.fluid_type=='solvant' %}selected{% endif %}>Solvent</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Impeller material</label>
        <select class="form-select" id="f-roue-mat">
          <option value="inox_316" {% if machine.roue_material=='inox_316' %}selected{% endif %}>Stainless 316</option>
          <option value="inox_304" {% if machine.roue_material=='inox_304' %}selected{% endif %}>Stainless 304</option>
          <option value="fonte" {% if machine.roue_material=='fonte' %}selected{% endif %}>Cast iron</option>
          <option value="bronze" {% if machine.roue_material=='bronze' %}selected{% endif %}>Bronze</option>
          <option value="plastique" {% if machine.roue_material=='plastique' %}selected{% endif %}>Plastic</option>
          <option value="duplex" {% if machine.roue_material=='duplex' %}selected{% endif %}>Duplex</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Risk alert threshold (%)</label>
        <input class="form-input" id="f-threshold" type="number" min="10" max="90" value="{{ machine.threshold or 45 }}">
        <span class="form-hint">Anomaly alert fires above this value</span>
      </div>
    </div>

    <div class="section-title">Nominal operating values <span style="font-size:10px;font-weight:400;text-transform:none;"> — used by the model as per-machine baselines</span></div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Nominal flow (m³/h)</label>
        <input class="form-input" id="f-nominal-flow" type="number" step="0.1" placeholder="e.g. 12.5" value="{{ machine.nominal_flow or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Nominal outlet pressure (bar)</label>
        <input class="form-input" id="f-nominal-pressure" type="number" step="0.1" placeholder="e.g. 6.0" value="{{ machine.nominal_pressure or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Rated motor power (kW)</label>
        <input class="form-input" id="f-power-kw" type="number" step="0.1" placeholder="e.g. 7.5" value="{{ machine.power_kw or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Nominal motor current (A)</label>
        <input class="form-input" id="f-nominal-current" type="number" step="0.1" placeholder="e.g. 15.0" value="{{ machine.nominal_current or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Normal vibration level (mm/s)</label>
        <input class="form-input" id="f-nominal-vibration" type="number" step="0.01" placeholder="e.g. 0.6" value="{{ machine.nominal_vibration or '' }}">
      </div>
    </div>

    <div class="section-title">Alerts</div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Alert email</label>
        <input class="form-input" id="f-alert-email" type="email" placeholder="maintenance@company.com" value="{{ machine.alert_email or '' }}">
      </div>
      <div class="form-group">
        <label class="form-label">Escalation email</label>
        <input class="form-input" id="f-esc-email" type="email" placeholder="manager@company.com" value="{{ machine.escalation_email or '' }}">
        <span class="form-hint">Notified if alert is not acknowledged within 30 min</span>
      </div>
    </div>

    <div style="margin-top:24px;">
      <button class="btn btn-primary" onclick="saveInfo()">Save machine profile</button>
    </div>
  </div>

  <!-- DATA TAB -->
  <div id="tab-data" class="tab-pane">
    <div class="card">
      <div style="font-size:14px;font-weight:600;margin-bottom:8px;">Upload sensor data (CSV)</div>
      <p style="font-size:13px;color:var(--text3);margin-bottom:20px;">Upload a CSV file with sensor readings. Columns are auto-detected. Each row is one reading — all rows will be analysed and saved.</p>
      <div class="upload-area" id="upload-area" onclick="document.getElementById('csv-file').click()" ondragover="event.preventDefault();this.classList.add('drag')" ondragleave="this.classList.remove('drag')" ondrop="handleDrop(event)">
        <input type="file" id="csv-file" accept=".csv" onchange="handleFileSelect(event)">
        <div class="upload-icon">📂</div>
        <div class="upload-label" id="upload-label">Drop CSV here or click to browse</div>
        <div class="upload-hint">vibration · temp_palier · debit · pression_entree/sortie · courant_moteur · temp_moteur · heure_fonctionnement</div>
      </div>
      <div id="upload-result" class="result-box"></div>
    </div>
  </div>

  <!-- MONITOR TAB -->
  <div id="tab-monitor" class="tab-pane">
    <p style="font-size:13px;color:var(--text3);margin-bottom:20px;">Latest analysis for this machine. Updates automatically when new data is uploaded.</p>
    <div class="monitor-grid">
      <div class="stat-box">
        <div class="stat-label">Risk score</div>
        <div class="stat-val" id="mon-risk" style="color:var(--text3);">—</div>
        <div class="stat-sub" id="mon-status">No data yet</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">RUL estimate</div>
        <div class="stat-val" id="mon-rul" style="color:var(--text3);">—</div>
        <div class="stat-sub">hours remaining</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Last seen</div>
        <div class="stat-val" style="font-size:15px;color:var(--text2);" id="mon-last">—</div>
        <div class="stat-sub" id="mon-confidence"></div>
      </div>
    </div>
    <div id="mon-zones-section" style="display:none;">
      <div class="section-title">Active failure zones</div>
      <div class="zones-list" id="mon-zones"></div>
    </div>
    <div id="mon-shap-section" style="display:none;margin-top:20px;">
      <div class="section-title">Top contributing factors (SHAP)</div>
      <div class="zones-list" id="mon-shap"></div>
    </div>
  </div>

  <!-- NOTES TAB -->
  <div id="tab-notes" class="tab-pane">
    <div class="card">
      <div style="font-size:14px;font-weight:600;margin-bottom:12px;">Add a note</div>
      <textarea class="note-input" id="note-input" placeholder="Observation, maintenance action, anomaly detail…"></textarea>
      <div style="margin-top:10px;">
        <button class="btn btn-primary btn-sm" onclick="addNote()">Post note</button>
      </div>
    </div>
    <div class="notes-list" id="notes-list">
      <div style="font-size:13px;color:var(--text3);text-align:center;padding:32px;">Loading notes…</div>
    </div>
  </div>

  <!-- HISTORY TAB -->
  <div id="tab-history" class="tab-pane">
    <p style="font-size:13px;color:var(--text3);margin-bottom:16px;">All analyses recorded for this machine.</p>
    <div style="overflow-x:auto;">
      <table class="history-table">
        <thead>
          <tr>
            <th>Date</th><th>Risk</th><th>Status</th><th>Zones</th><th>Confidence</th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="5" style="text-align:center;color:var(--text3);padding:32px;">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
const MID = {{ machine.id }};
const MACHINE = {{ machine_json | safe }};
let activeTab = 'info';

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const tabs = ['info','data','monitor','notes','history'];
    t.classList.toggle('active', tabs[i] === name);
  });
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  activeTab = name;
  if (name === 'monitor') loadMonitor();
  if (name === 'notes') loadNotes();
  if (name === 'history') loadHistory();
}

function toast(msg, ok) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderColor = ok ? 'rgba(48,209,88,.4)' : 'rgba(255,69,58,.4)';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

function saveInfo() {
  const payload = {
    name: document.getElementById('f-name').value.trim(),
    serial_number: document.getElementById('f-serial').value.trim(),
    location: document.getElementById('f-location').value.trim(),
    install_date: document.getElementById('f-install').value || null,
    description: document.getElementById('f-desc').value.trim(),
    pump_type: document.getElementById('f-pump-type').value,
    fluid_type: document.getElementById('f-fluid-type').value,
    roue_material: document.getElementById('f-roue-mat').value,
    threshold: parseFloat(document.getElementById('f-threshold').value) || 45,
    nominal_flow: parseFloat(document.getElementById('f-nominal-flow').value) || null,
    nominal_pressure: parseFloat(document.getElementById('f-nominal-pressure').value) || null,
    power_kw: parseFloat(document.getElementById('f-power-kw').value) || null,
    nominal_current: parseFloat(document.getElementById('f-nominal-current').value) || null,
    nominal_vibration: parseFloat(document.getElementById('f-nominal-vibration').value) || null,
    alert_email: document.getElementById('f-alert-email').value.trim(),
    escalation_email: document.getElementById('f-esc-email').value.trim(),
  };
  fetch('/api/machines/' + MID, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
    .then(r => r.json()).then(d => {
      if (d.ok) { toast('Profile saved', true); document.querySelector('.machine-title').textContent = payload.name; }
      else toast(d.error || 'Error saving', false);
    }).catch(() => toast('Network error', false));
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) uploadCSV(file);
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-area').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) uploadCSV(file);
}
function uploadCSV(file) {
  document.getElementById('upload-label').textContent = 'Uploading ' + file.name + '…';
  const fd = new FormData();
  fd.append('file', file);
  fetch('/api/machines/' + MID + '/analyze-csv', {method:'POST', body:fd})
    .then(r => r.json()).then(d => {
      const box = document.getElementById('upload-result');
      box.style.display = 'block';
      if (d.ok) {
        box.style.borderColor = 'rgba(48,209,88,.3)';
        box.innerHTML = '<span style="color:var(--green);font-weight:700;">✓ Done</span> &nbsp;'
          + d.total + ' rows analysed &nbsp;·&nbsp; '
          + '<span style="color:var(--amber);">' + d.failures + ' anomalies</span> &nbsp;·&nbsp; '
          + 'Avg risk: <b>' + d.avg_risk + '%</b> &nbsp;·&nbsp; Max risk: <b>' + d.max_risk + '%</b>';
        document.getElementById('upload-label').textContent = file.name + ' — done';
        toast('Analysis complete', true);
      } else {
        box.style.borderColor = 'rgba(255,69,58,.3)';
        box.innerHTML = '<span style="color:var(--red);">Error: ' + (d.error || 'Unknown') + '</span>';
        document.getElementById('upload-label').textContent = 'Drop CSV here or click to browse';
        toast(d.error || 'Upload failed', false);
      }
    }).catch(() => {
      toast('Network error', false);
      document.getElementById('upload-label').textContent = 'Drop CSV here or click to browse';
    });
}

function loadMonitor() {
  fetch('/api/machine_monitor/' + MID)
    .then(r => r.json()).then(d => {
      if (!d.has_data) return;
      const risk = d.risk;
      const el = document.getElementById('mon-risk');
      el.textContent = risk.toFixed(1) + '%';
      el.style.color = risk >= 75 ? 'var(--red)' : risk >= 45 ? 'var(--amber)' : 'var(--green)';
      document.getElementById('mon-status').textContent = d.prediction ? 'Anomaly detected' : 'Normal';
      const pill = document.getElementById('status-pill');
      pill.className = 'status-pill ' + (d.prediction ? (risk >= 75 ? 'status-crit' : 'status-warn') : 'status-ok');
      pill.textContent = d.prediction ? (risk >= 75 ? 'Critical' : 'Warning') : 'Healthy';
      if (d.timestamp) document.getElementById('mon-last').textContent = new Date(d.timestamp).toLocaleString();
      if (d.zones && d.zones.length) {
        document.getElementById('mon-zones-section').style.display = 'block';
        document.getElementById('mon-zones').innerHTML = d.zones.map(z =>
          '<div class="zone-row"><span class="zone-name">' + z.nom + '</span><div class="zone-bar-wrap"><div class="zone-bar-fill" style="width:' + z.proba + '%"></div></div><span class="zone-pct">' + z.proba + '%</span></div>'
        ).join('');
      }
    }).catch(() => {});
}

function loadNotes() {
  fetch('/api/machines/' + MID + '/notes')
    .then(r => r.json()).then(notes => {
      const el = document.getElementById('notes-list');
      if (!notes.length) { el.innerHTML = '<div style="font-size:13px;color:var(--text3);text-align:center;padding:32px;">No notes yet. Be the first to add one.</div>'; return; }
      el.innerHTML = notes.map(n => `
        <div class="note-card" id="note-${n.id}">
          <div class="note-meta">
            <span class="note-author">${n.user_email || 'Team'}</span>
            <div style="display:flex;align-items:center;gap:8px;">
              <span class="note-time">${new Date(n.created_at).toLocaleString()}</span>
              <button class="note-del" onclick="deleteNote(${n.id})" title="Delete">✕</button>
            </div>
          </div>
          <div class="note-content">${n.content.replace(/</g,'&lt;')}</div>
        </div>`).join('');
    });
}

function addNote() {
  const input = document.getElementById('note-input');
  const content = input.value.trim();
  if (!content) return;
  fetch('/api/machines/' + MID + '/notes', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({content})})
    .then(r => r.json()).then(d => {
      if (d.ok) { input.value = ''; loadNotes(); toast('Note added', true); }
      else toast(d.error || 'Error', false);
    });
}

function deleteNote(nid) {
  fetch('/api/machines/' + MID + '/notes/' + nid, {method:'DELETE'})
    .then(r => r.json()).then(d => {
      if (d.ok) { document.getElementById('note-' + nid).remove(); toast('Note deleted', true); }
      else toast(d.error || 'Error', false);
    });
}

function loadHistory() {
  fetch('/api/machine_history/' + MID + '?limit=50')
    .then(r => r.json()).then(d => {
      const tbody = document.getElementById('history-body');
      const analyses = d.analyses || [];
      if (!analyses.length) { tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:32px;">No analyses yet for this machine.</td></tr>'; return; }
      tbody.innerHTML = analyses.map(a => {
        const riskCol = a.risk >= 75 ? 'var(--red)' : a.risk >= 45 ? 'var(--amber)' : 'var(--green)';
        const status = a.prediction ? '<span style="color:var(--red);font-weight:600;">Anomaly</span>' : '<span style="color:var(--green);">Normal</span>';
        return `<tr>
          <td>${new Date(a.timestamp).toLocaleString()}</td>
          <td><span style="font-weight:700;color:${riskCol}">${a.risk}%</span></td>
          <td>${status}</td>
          <td style="color:var(--amber);font-size:12px;">${a.zones || '—'}</td>
          <td>${a.confidence !== undefined ? a.confidence + '%' : '—'}</td>
        </tr>`;
      }).join('');
    }).catch(() => {
      document.getElementById('history-body').innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);">Could not load history.</td></tr>';
    });
}

// Auto-load monitor on page open
loadMonitor();
setInterval(loadMonitor, 30000);
</script>
</body></html>"""

# ── DASHBOARD MULTI-MACHINES ──────────────────────────────────────────────────
DASHBOARD_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<div class="desktop-layout">
""" + nav("fl") + """
<div class="main-content">
<header>
  <span class="logo">PILAR</span>
  <div class="hd"></div>
  <span class="hsub" data-i18n="fl_title">Fleet Overview</span>
  <div class="hright" style="display:flex;gap:8px;align-items:center">
    <button id="analyze-all-btn" onclick="analyzeAll()" class="btn" style="font-size:11px;padding:6px 14px;background:var(--teal);color:#fff;display:none" data-i18n="fl_analyze_all">Analyser tout</button>
    <button onclick="openAdd()" class="btn btn-primary" style="font-size:12px;padding:7px 16px;font-weight:600">+ <span data-i18n="fl_add">Ajouter une machine</span></button>
  </div>
</header>
<div class="page pad">
<style>
.fleet-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;align-items:start;}
.machine-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:0;position:relative;transition:border-color .2s,box-shadow .2s;overflow:hidden;}
.machine-card:hover{border-color:var(--border2);}
.card-top{padding:18px 18px 14px;}
.card-name-row{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:12px;}
.machine-name{font-size:15px;font-weight:700;letter-spacing:-.01em;color:var(--text);}
.machine-desc{font-size:11px;color:var(--text3);margin-top:2px;}
.badge{display:inline-block;padding:3px 9px;font-size:10px;font-weight:700;border-radius:20px;letter-spacing:.2px;}
.badge-ok{background:var(--green-dim);color:var(--green);border:1px solid rgba(48,209,88,.3);}
.badge-warn{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(255,159,10,.3);}
.badge-crit{background:var(--red-dim);color:var(--red);border:1px solid rgba(255,69,58,.3);}
.badge-inactive{background:rgba(84,84,88,.12);color:var(--text3);border:1px solid var(--border);}
.badge-new{background:rgba(120,120,128,.12);color:var(--text3);border:1px solid var(--border);}
.risk-section{padding:14px 18px;background:rgba(0,0,0,.15);border-top:1px solid var(--border);}
.risk-label-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.risk-pct{font-size:32px;font-weight:800;line-height:1;}
.risk-sub{font-size:10px;color:var(--text3);letter-spacing:.5px;text-transform:uppercase;}
.risk-bar{height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;margin-top:8px;}
.risk-fill{height:100%;border-radius:2px;transition:width .5s cubic-bezier(.4,0,.2,1);}
.result-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border-top:1px solid var(--border);}
.rstat{padding:10px 14px;background:var(--surface);text-align:center;}
.rstat-val{font-size:16px;font-weight:700;color:var(--text);}
.rstat-lbl{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;}
.drop-zone{margin:0;padding:28px 18px;border-top:1px solid var(--border);text-align:center;cursor:pointer;transition:background .15s;position:relative;}
.drop-zone:hover,.drop-zone.drag-over{background:rgba(13,148,136,.06);border-color:rgba(13,148,136,.4);}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.dz-icon{width:24px;height:24px;color:var(--text3);margin:0 auto 8px;}
.dz-text{font-size:12px;font-weight:600;color:var(--text2);}
.dz-sub{font-size:10px;color:var(--text3);margin-top:3px;}
.file-bar{display:flex;align-items:center;gap:8px;padding:10px 16px;background:rgba(13,148,136,.05);border-top:1px solid rgba(13,148,136,.15);font-size:11px;}
.file-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--teal2);font-weight:600;}
.card-actions{display:flex;gap:1px;background:var(--border);border-top:1px solid var(--border);}
.card-action-btn{flex:1;padding:10px 8px;background:var(--surface);border:none;color:var(--text3);cursor:pointer;font-size:11px;font-weight:600;transition:background .15s,color .15s;text-align:center;}
.card-action-btn:hover{background:rgba(255,255,255,.05);color:var(--text);}
.card-action-btn.primary{color:var(--teal2);}
.card-action-btn.danger:hover{background:var(--red-dim);color:var(--red);}
.empty-state{grid-column:1/-1;text-align:center;padding:80px 32px 60px;}
.empty-icon{width:48px;height:48px;margin:0 auto 20px;color:var(--border2);}
.empty-title{font-size:17px;font-weight:700;color:var(--text);margin-bottom:8px;}
.empty-desc{font-size:13px;color:var(--text3);line-height:1.7;max-width:420px;margin:0 auto 28px;}
.summary-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.sstat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;min-width:100px;}
.sstat-val{font-size:20px;font-weight:700;line-height:1;}
.sstat-lbl{font-size:9px;font-weight:700;color:var(--text3);letter-spacing:.5px;text-transform:uppercase;margin-top:3px;}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);z-index:100;display:none;align-items:center;justify-content:center;}
.modal-overlay.open{display:flex;}
.modal{background:var(--surface);border:1px solid var(--border2);border-radius:var(--r);padding:28px;width:480px;max-width:92vw;max-height:92vh;overflow-y:auto;box-shadow:var(--shadow);}
.modal-title{font-size:17px;font-weight:700;margin-bottom:20px;}
.form-group{margin-bottom:14px;}
.form-label{font-size:10px;font-weight:700;color:var(--text3);display:block;margin-bottom:5px;letter-spacing:.5px;text-transform:uppercase;}
.form-input{width:100%;background:rgba(120,120,128,0.18);border:1px solid var(--border2);border-radius:8px;padding:9px 12px;font-size:13px;color:var(--text);outline:none;transition:border-color .15s;box-sizing:border-box;}
.form-input:focus{border-color:var(--teal);}
.form-hint{font-size:11px;color:var(--text3);margin-top:4px;}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.modal-footer{display:flex;gap:10px;justify-content:flex-end;margin-top:20px;padding-top:18px;border-top:1px solid var(--border);}
.adv-toggle{font-size:11px;color:var(--text3);cursor:pointer;margin:4px 0 12px;display:inline-flex;align-items:center;gap:4px;user-select:none;}
.adv-toggle:hover{color:var(--text2);}
#adv-fields{display:none;}
.upload-zone{border:2px dashed var(--border2);border-radius:8px;padding:22px;text-align:center;cursor:pointer;transition:border-color .15s,background .15s;position:relative;margin-bottom:4px;}
.upload-zone:hover,.upload-zone.drag-over{border-color:var(--teal);background:rgba(13,148,136,.04);}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.upload-zone-text{font-size:12px;font-weight:600;color:var(--text2);}
.upload-zone-sub{font-size:10px;color:var(--text3);margin-top:3px;}
.ap-row{display:grid;grid-template-columns:1fr auto auto auto;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--border);font-size:12px;}
.ap-row:last-child{border-bottom:none;}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border2);border-top-color:var(--teal);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px;}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:600px){.fleet-grid{grid-template-columns:1fr}.summary-row{gap:8px}}
</style>

<!-- Summary row -->
<div class="summary-row" id="summary-row" style="display:none">
  <div class="sstat"><div class="sstat-val" id="s-total">0</div><div class="sstat-lbl" data-i18n="fl_total">Machines</div></div>
  <div class="sstat"><div class="sstat-val" id="s-alerts" style="color:var(--red)">0</div><div class="sstat-lbl" data-i18n="fl_alerts">Alertes</div></div>
  <div class="sstat"><div class="sstat-val" id="s-avg">\u2014</div><div class="sstat-lbl" data-i18n="fl_avg">Risque moy.</div></div>
  <div class="sstat"><div class="sstat-val" id="s-crit" style="color:var(--red)">0</div><div class="sstat-lbl" data-i18n="fl_critical">Critiques</div></div>
</div>

<div class="fleet-grid" id="fleet-grid">
  <div class="empty-state">
    <svg class="empty-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="24" cy="24" r="20"/><path d="M24 16v8l5 5"/></svg>
    <p style="color:var(--text3);font-size:14px" data-i18n="fl_loading">Chargement\u2026</p>
  </div>
</div>
</div>

<!-- Add/Edit Machine Modal -->
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-title" id="modal-title" data-i18n="fl_modal_add">Ajouter une machine</div>
    <form id="machine-form" onsubmit="submitMachine(event)">
      <input type="hidden" id="edit-id" value="">
      <div class="form-group">
        <label class="form-label" data-i18n="fl_name_lbl">Nom de la machine</label>
        <input class="form-input" id="f-name" required autocomplete="off">
        <div class="form-hint" id="f-name-hint" data-i18n="fl_name_hint">Utilis\u00e9 comme machine_id dans l\u2019API.</div>
      </div>
      <div class="form-group">
        <label class="form-label" data-i18n="fl_desc_lbl">Description</label>
        <input class="form-input" id="f-desc" autocomplete="off">
      </div>

      <!-- File upload — only shown in add mode -->
      <div id="f-file-group" class="form-group">
        <label class="form-label">Fichier CSV <span style="font-weight:400;color:var(--text3)">(optionnel)</span></label>
        <div class="upload-zone" id="modal-drop" onclick="document.getElementById('modal-file').click()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:22px;height:22px;color:var(--text3);margin:0 auto 8px;display:block"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          <div class="upload-zone-text" id="modal-file-label" data-i18n="fl_upload_click">Cliquez pour s\u00e9lectionner un CSV</div>
          <div class="upload-zone-sub" data-i18n="fl_upload_hint">Tout d\u00e9limiteur \u00b7 noms libres \u00b7 conversion auto</div>
          <input type="file" id="modal-file" accept=".csv" style="display:none" onchange="onModalFileChange(this)">
        </div>
        <div id="modal-file-name" style="display:none;font-size:11px;color:var(--teal2);font-weight:600;margin-top:6px;padding:6px 10px;background:rgba(13,148,136,.06);border-radius:6px;border:1px solid rgba(13,148,136,.2)"></div>
      </div>

      <div class="adv-toggle" onclick="toggleAdv()" id="adv-toggle-btn">
        <svg id="adv-arrow" viewBox="0 0 16 16" fill="currentColor" style="width:12px;height:12px;transform:rotate(0deg);transition:transform .2s"><path d="M8 10.5L2.5 5h11z"/></svg>
        <span data-i18n="adv_params">Param\u00e8tres avanc\u00e9s</span>
      </div>
      <div id="adv-fields">
        <div class="form-row">
          <div class="form-group">
            <label class="form-label" data-i18n="fl_class_sel">Classe machine</label>
            <select class="form-input" id="f-type">
              <option value="L">L \u2014 Basse qualit\u00e9</option>
              <option value="M" selected>M \u2014 Standard</option>
              <option value="H">H \u2014 Haute qualit\u00e9</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" data-i18n="fl_thresh_lbl">Seuil d\u2019alerte\u00a0%</label>
            <input class="form-input" id="f-threshold" type="number" min="10" max="95" step="1" value="45">
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label" data-i18n="fl_pump_sel">Type de pompe</label>
            <select class="form-input" id="f-pump-type">
              <option value="centrifuge" selected>Centrifuge</option>
              <option value="pompe_a_vis">Pompe \u00e0 vis</option>
              <option value="pompe_a_engrenage">Pompe \u00e0 engrenages</option>
              <option value="pompe_a_palettes">Pompe \u00e0 palettes</option>
              <option value="pompe_a_piston">Pompe \u00e0 piston</option>
              <option value="peristaltique">P\u00e9ristaltique</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" data-i18n="fl_fluid_sel">Type de fluide</label>
            <select class="form-input" id="f-fluid-type">
              <option value="eau" selected>Eau</option>
              <option value="eau_chargee">Eau charg\u00e9e / boues</option>
              <option value="huile">Huile</option>
              <option value="acide">Acide</option>
              <option value="base">Base / alcalin</option>
              <option value="autre">Autre</option>
            </select>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label" data-i18n="fl_mat_sel">Mat\u00e9riau roue</label>
            <select class="form-input" id="f-roue-material">
              <option value="inox_316" selected>Inox 316</option>
              <option value="fonte">Fonte grise</option>
              <option value="titane">Titane</option>
              <option value="bronze">Bronze</option>
              <option value="plastique">Plastique / PVDF</option>
              <option value="autre">Autre</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label" data-i18n="fl_email_lbl">Email d\u2019alerte</label>
            <input class="form-input" id="f-email" type="email">
          </div>
        </div>
      </div>

      <div id="modal-result" style="display:none;margin-top:14px"></div>
      <div class="modal-footer">
        <button type="button" class="btn btn-ghost" onclick="closeModal()" data-i18n="fl_cancel">Annuler</button>
        <button type="submit" class="btn btn-primary" id="modal-submit" data-i18n="fl_save">Enregistrer</button>
      </div>
    </form>
  </div>
</div>

<!-- Quick Analyse Modal -->
<div class="modal-overlay" id="analyse-overlay" onclick="if(event.target===this)closeAnalyse()">
  <div class="modal" style="width:520px">
    <div class="modal-title" id="an-title">Analyse</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:4px">
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_vibration">Vibration <span style="font-weight:400;letter-spacing:0;text-transform:none">mm/s</span></label>
        <input class="form-input" id="an-vib" type="number" step="0.01" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_temp_palier">Temp. palier <span style="font-weight:400;letter-spacing:0;text-transform:none">\u00b0C</span></label>
        <input class="form-input" id="an-tp" type="number" step="0.1" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_debit">D\u00e9bit <span style="font-weight:400;letter-spacing:0;text-transform:none">m\u00b3/h</span></label>
        <input class="form-input" id="an-dbt" type="number" step="0.001" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_pression_e">Pression entr\u00e9e <span style="font-weight:400;letter-spacing:0;text-transform:none">bar</span></label>
        <input class="form-input" id="an-pe" type="number" step="0.01" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_pression_s">Pression sortie <span style="font-weight:400;letter-spacing:0;text-transform:none">bar</span></label>
        <input class="form-input" id="an-ps" type="number" step="0.01" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_courant">Courant moteur <span style="font-weight:400;letter-spacing:0;text-transform:none">A</span></label>
        <input class="form-input" id="an-im" type="number" step="0.1" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_temp_moteur">Temp. moteur <span style="font-weight:400;letter-spacing:0;text-transform:none">\u00b0C</span></label>
        <input class="form-input" id="an-tm" type="number" step="0.1" min="0">
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label" data-i18n="p_heure">Heures fonct. <span style="font-weight:400;letter-spacing:0;text-transform:none">h</span></label>
        <input class="form-input" id="an-hf" type="number" step="1" min="0">
      </div>
    </div>
    <div id="an-result" style="margin-top:14px;display:none"></div>
    <div class="modal-footer">
      <button type="button" class="btn btn-ghost" onclick="closeAnalyse()" data-i18n="fl_an_close">Fermer</button>
      <button type="button" class="btn btn-primary" id="an-submit" onclick="runAnalyse()" data-i18n="fl_an_run">Lancer l\u2019analyse</button>
    </div>
  </div>
</div>

<!-- Analyze All Results Modal -->
<div class="modal-overlay" id="ap-overlay" onclick="if(event.target===this)closeAP()">
  <div class="modal" style="width:560px">
    <div class="modal-title" id="ap-title" data-i18n="fl_ap_title">Analyse de la flotte</div>
    <div id="ap-content" style="min-height:60px"></div>
    <div class="modal-footer">
      <button type="button" class="btn btn-primary" onclick="closeAP()" data-i18n="fl_ap_close">Fermer</button>
    </div>
  </div>
</div>

<script>
var _machines = [];
var _MEDIANS = {vibration:0.61,temp_palier:44.8,debit:0.395,pression_entree:1.987,pression_sortie:107.73,courant_moteur:4.58,temp_moteur:58.0,heure_fonctionnement:1103.0};

// ── HELPERS ─────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function riskColor(r) {
  if (r===null||r===undefined) return 'var(--text3)';
  return r>=70?'var(--red)':r>=45?'var(--amber)':'var(--green)';
}
function riskBadge(r) {
  if (r===null||r===undefined) return '<span class="badge badge-new">'+t('fl_badge_new')+'</span>';
  if (r>=70) return '<span class="badge badge-crit">'+t('fl_badge_crit')+'</span>';
  if (r>=45) return '<span class="badge badge-warn">'+t('fl_badge_warn')+'</span>';
  return '<span class="badge badge-ok">'+t('fl_badge_ok')+'</span>';
}

// ── LOAD & RENDER ───────────────────────────────────────────────────────────
async function loadFleet() {
  try {
    var r = await fetch('/api/machines');
    if (!r.ok) throw new Error('HTTP '+r.status);
    _machines = await r.json();
    if (!Array.isArray(_machines)) throw new Error('bad');
    renderFleet();
  } catch(e) {
    document.getElementById('fleet-grid').innerHTML =
      '<div class="empty-state"><p style="color:var(--red)">'+t('fl_load_fail')+e.message+'</p>'
      +'<button class="btn btn-ghost" onclick="loadFleet()">'+t('fl_retry')+'</button></div>';
  }
}

function renderFleet() {
  var grid = document.getElementById('fleet-grid');
  var sumRow = document.getElementById('summary-row');
  var aBtn = document.getElementById('analyze-all-btn');

  if (!_machines.length) {
    sumRow.style.display = 'none';
    aBtn.style.display = 'none';
    grid.innerHTML = '<div class="empty-state">'
      +'<svg class="empty-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="8" y="12" width="32" height="28" rx="3"/><path d="M16 12V9a2 2 0 012-2h12a2 2 0 012 2v3"/><path d="M24 22v8M20 26h8"/></svg>'
      +'<div class="empty-title">'+t('fl_empty_title')+'</div>'
      +'<p class="empty-desc">'+t('fl_empty_desc')+'</p>'
      +'<button class="btn btn-primary" onclick="openAdd()" style="font-size:13px;padding:10px 22px">+ '+t('fl_add_first')+'</button>'
      +'</div>';
    return;
  }

  var active = _machines.filter(function(m){return m.is_active;});
  var withData = active.filter(function(m){return m.last_risk!==null;});
  var alerts = active.filter(function(m){return m.last_prediction===1;}).length;
  var avgRisk = withData.length ? Math.round(withData.reduce(function(s,m){return s+m.last_risk;},0)/withData.length*10)/10 : null;
  var crit = active.filter(function(m){return m.last_risk>=70;}).length;
  var hasFiles = _machines.some(function(m){return m.saved_file;});

  document.getElementById('s-total').textContent = active.length;
  document.getElementById('s-alerts').textContent = alerts;
  document.getElementById('s-avg').textContent = avgRisk!==null ? avgRisk+'%' : '\u2014';
  document.getElementById('s-crit').textContent = crit;
  sumRow.style.display = 'flex';
  aBtn.style.display = hasFiles ? '' : 'none';

  var isFr = (LANG==='fr');
  var pumpFr = {centrifuge:'Centrifuge',pompe_a_vis:'Vis',pompe_a_engrenage:'Engrenages',pompe_a_palettes:'Palettes',pompe_a_piston:'Piston',peristaltique:'P\u00e9ristaltique'};
  var pumpEn = {centrifuge:'Centrifugal',pompe_a_vis:'Screw',pompe_a_engrenage:'Gear',pompe_a_palettes:'Vane',pompe_a_piston:'Piston',peristaltique:'Peristaltic'};
  var flFr = {eau:'Eau',eau_chargee:'Eau charg\u00e9e',huile:'Huile',acide:'Acide',base:'Base',autre:'Autre'};
  var flEn = {eau:'Water',eau_chargee:'Slurry',huile:'Oil',acide:'Acid',base:'Base',autre:'Other'};
  var pL = isFr ? pumpFr : pumpEn;
  var fL = isFr ? flFr : flEn;

  grid.innerHTML = _machines.map(function(m) {
    var risk = m.last_risk;
    var rCol = riskColor(risk);
    var badge = m.is_active ? riskBadge(risk) : '<span class="badge badge-inactive">'+t('fl_badge_inactive')+'</span>';
    var pumpTxt = pL[m.pump_type]||m.pump_type||'Centrifugal';
    var fluidTxt = fL[m.fluid_type]||m.fluid_type||'Eau';

    // Risk section — only when we have data
    var riskSection = '';
    if (m.is_active && risk!==null) {
      var rFill = Math.min(risk,100);
      riskSection = '<div class="risk-section">'
        +'<div class="risk-label-row">'
        +'<div><div class="risk-pct" style="color:'+rCol+'">'+risk+'<span style="font-size:14px;font-weight:400;color:var(--text3)">%</span></div>'
        +'<div class="risk-sub">'+t('fl_risk_lbl')+'</div></div>'
        +'<div style="text-align:right">'+badge+'<div style="font-size:10px;color:var(--text3);margin-top:4px">'+pumpTxt+' \u00b7 '+fluidTxt+'</div></div>'
        +'</div>'
        +'<div class="risk-bar"><div class="risk-fill" style="width:'+rFill+'%;background:'+rCol+'"></div></div>'
        +'</div>';
    }

    // Result stats if file was analyzed
    var statsSection = '';
    if (m.saved_file && risk!==null) {
      statsSection = '<div class="result-stats">'
        +'<div class="rstat"><div class="rstat-val">'+m.saved_file.row_count+'</div><div class="rstat-lbl">'+t('fl_ap_rows')+'</div></div>'
        +'<div class="rstat"><div class="rstat-val" style="color:'+rCol+'">'+risk+'%</div><div class="rstat-lbl">'+t('fl_risk_lbl')+'</div></div>'
        +'<div class="rstat"><div class="rstat-val">'+m.threshold+'%</div><div class="rstat-lbl">'+t('fl_thresh_card')+'</div></div>'
        +'</div>';
    }

    // Drop zone when no file
    var dropZone = '';
    if (!m.saved_file) {
      dropZone = '<div class="drop-zone" id="dz-'+m.id+'" ondragover="dzOver(event,'+m.id+')" ondragleave="dzLeave('+m.id+')" ondrop="dzDrop(event,'+m.id+')">'
        +'<svg class="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        +'<div class="dz-text">'+t('fl_upload_click')+'</div>'
        +'<div class="dz-sub">'+t('fl_upload_hint')+'</div>'
        +'<input type="file" accept=".csv" onchange="dzFile(this,'+m.id+')">'
        +'</div>';
    } else {
      dropZone = '<div class="file-bar">'
        +'<svg viewBox="0 0 20 20" fill="currentColor" style="width:13px;height:13px;flex-shrink:0;color:var(--teal2)"><path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"/></svg>'
        +'<span class="file-name">'+esc(m.saved_file.filename)+'</span>'
        +'<button onclick="event.stopPropagation();rerunFile('+m.id+')" style="background:var(--teal);border:none;border-radius:5px;color:#fff;font-size:10px;font-weight:700;padding:3px 8px;cursor:pointer" id="rerun-'+m.id+'">'+t('fl_btn_rerun')+'</button>'
        +'<button onclick="event.stopPropagation();deleteFile('+m.id+','+m.saved_file.id+')" style="background:none;border:none;color:var(--text3);cursor:pointer;font-size:15px;line-height:1;padding:2px 4px">\u2715</button>'
        +'</div>';
    }

    // Card header info
    var infoLine = '';
    if (!riskSection) {
      infoLine = '<div style="font-size:11px;color:var(--text3);margin-bottom:4px">'+pumpTxt+' \u00b7 '+fluidTxt+' \u00b7 '+badge+'</div>';
    }

    return '<div class="machine-card" id="card-'+m.id+'">'
      +'<div class="card-top">'
      +'<div class="card-name-row">'
      +'<div><div class="machine-name">'+esc(m.name)+'</div>'
      +(m.description?'<div class="machine-desc">'+esc(m.description)+'</div>':'')
      +'</div>'
      +(riskSection?'':'<div>'+badge+'</div>')
      +'</div>'
      +infoLine
      +'</div>'
      +riskSection
      +statsSection
      +dropZone
      +'<div class="card-actions">'
      +'<button class="card-action-btn primary" onclick="openAnalyse('+m.id+')">\u26a1 '+t('fl_btn_analyze')+'</button>'
      +'<button class="card-action-btn" onclick="openEdit('+m.id+')">'+t('fl_btn_edit')+'</button>'
      +'<button class="card-action-btn danger" onclick="deleteMachine('+m.id+')">\u2715</button>'
      +'</div>'
      +'</div>';
  }).join('');
}

// ── DRAG & DROP on cards ─────────────────────────────────────────────────────
function dzOver(e, id) { e.preventDefault(); var dz=document.getElementById('dz-'+id); if(dz)dz.classList.add('drag-over'); }
function dzLeave(id) { var dz=document.getElementById('dz-'+id); if(dz)dz.classList.remove('drag-over'); }
function dzDrop(e, id) {
  e.preventDefault();
  dzLeave(id);
  var files = e.dataTransfer.files;
  if (files && files[0]) uploadFileToMachine(id, files[0]);
}
function dzFile(input, id) {
  if (input.files && input.files[0]) uploadFileToMachine(id, input.files[0]);
}

async function uploadFileToMachine(mid, file) {
  var dz = document.getElementById('dz-'+mid);
  if (dz) dz.innerHTML = '<div style="text-align:center;padding:8px"><span class="spinner"></span><span style="font-size:12px;color:var(--text3)">'+t('fl_analyzing')+'</span></div>';
  var fd = new FormData();
  fd.append('file', file);
  try {
    var r = await fetch('/api/machines/'+mid+'/analyze-csv', {method:'POST', body:fd});
    var d = await r.json();
    if (!r.ok||d.error) {
      if (dz) dz.innerHTML = '<div style="padding:12px;color:var(--red);font-size:12px;text-align:center">\u26a0 '+(d.error||t('fl_error'))+'</div>';
      return;
    }
    loadFleet();
  } catch(ex) {
    if (dz) dz.innerHTML = '<div style="padding:12px;color:var(--red);font-size:12px;text-align:center">'+t('fl_net_error')+'</div>';
  }
}

// ── ADD / EDIT MODAL ────────────────────────────────────────────────────────
var _modalFile = null;

function onModalFileChange(input) {
  _modalFile = input.files[0] || null;
  var lbl = document.getElementById('modal-file-label');
  var fname = document.getElementById('modal-file-name');
  if (_modalFile) {
    lbl.textContent = _modalFile.name;
    fname.textContent = _modalFile.name;
    fname.style.display = 'block';
  } else {
    lbl.setAttribute('data-i18n','fl_upload_click');
    lbl.textContent = t('fl_upload_click');
    fname.style.display = 'none';
  }
}

function toggleAdv() {
  var fields = document.getElementById('adv-fields');
  var arrow = document.getElementById('adv-arrow');
  var open = fields.style.display === 'block';
  fields.style.display = open ? 'none' : 'block';
  arrow.style.transform = open ? 'rotate(0deg)' : 'rotate(180deg)';
}

function openAdd() {
  document.getElementById('modal-title').textContent = t('fl_modal_add');
  document.getElementById('edit-id').value = '';
  document.getElementById('machine-form').reset();
  document.getElementById('f-threshold').value = 45;
  document.getElementById('modal-submit').textContent = t('fl_save');
  document.getElementById('f-file-group').style.display = '';
  document.getElementById('modal-file-name').style.display = 'none';
  document.getElementById('modal-result').style.display = 'none';
  document.getElementById('modal-result').innerHTML = '';
  _modalFile = null;
  document.getElementById('modal-overlay').classList.add('open');
  applyLang();
  setTimeout(function(){document.getElementById('f-name').focus();},100);
}

function openEdit(id) {
  var m = _machines.find(function(x){return x.id===id;});
  if (!m) return;
  document.getElementById('modal-title').textContent = t('fl_modal_edit');
  document.getElementById('edit-id').value = id;
  document.getElementById('f-name').value = m.name;
  document.getElementById('f-desc').value = m.description||'';
  document.getElementById('f-type').value = m.machine_type||'M';
  document.getElementById('f-threshold').value = m.threshold||45;
  document.getElementById('f-pump-type').value = m.pump_type||'centrifuge';
  document.getElementById('f-fluid-type').value = m.fluid_type||'eau';
  document.getElementById('f-roue-material').value = m.roue_material||'inox_316';
  document.getElementById('f-email').value = m.alert_email||'';
  document.getElementById('modal-submit').textContent = t('fl_save_changes');
  document.getElementById('f-file-group').style.display = 'none';
  document.getElementById('modal-result').style.display = 'none';
  _modalFile = null;
  document.getElementById('modal-overlay').classList.add('open');
  applyLang();
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

async function submitMachine(e) {
  e.preventDefault();
  var id = document.getElementById('edit-id').value;
  var submitBtn = document.getElementById('modal-submit');
  submitBtn.disabled = true;
  submitBtn.textContent = t('fl_analyzing');
  var body = {
    name: document.getElementById('f-name').value.trim(),
    description: document.getElementById('f-desc').value.trim(),
    machine_type: document.getElementById('f-type').value,
    threshold: parseFloat(document.getElementById('f-threshold').value)||45,
    pump_type: document.getElementById('f-pump-type').value,
    fluid_type: document.getElementById('f-fluid-type').value,
    roue_material: (document.getElementById('f-roue-material')||{}).value||'inox_316',
    alert_email: (document.getElementById('f-email')||{}).value||'',
    escalation_email: '',
  };
  var url = id ? ('/api/machines/'+id) : '/api/machines';
  var method = id ? 'PUT' : 'POST';
  var r = await fetch(url, {method:method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  var d = await r.json();
  if (!r.ok) { alert(d.error||t('fl_error')); submitBtn.disabled=false; submitBtn.textContent=t('fl_save'); return; }
  var newId = d.id || id;
  // If file selected, upload and analyze immediately
  if (_modalFile && newId) {
    var resDiv = document.getElementById('modal-result');
    resDiv.style.display = 'block';
    resDiv.innerHTML = '<div style="text-align:center;padding:12px"><span class="spinner"></span><span style="font-size:12px;color:var(--text3)">'+t('fl_analyzing')+'</span></div>';
    var fd = new FormData();
    fd.append('file', _modalFile);
    try {
      var r2 = await fetch('/api/machines/'+newId+'/analyze-csv', {method:'POST', body:fd});
      var d2 = await r2.json();
      if (!r2.ok||d2.error) {
        resDiv.innerHTML = '<div style="color:var(--red);font-size:12px;padding:8px 0">\u26a0 '+(d2.error||t('fl_error'))+'</div>';
        submitBtn.disabled=false; submitBtn.textContent=t('fl_save');
        loadFleet(); return;
      }
      var sCol = d2.failures>0?'var(--amber)':'var(--green)';
      resDiv.innerHTML = '<div style="background:rgba(13,148,136,.06);border:1px solid rgba(13,148,136,.2);border-radius:8px;padding:14px">'
        +'<div style="font-size:9px;color:var(--teal2);font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px">'+t('fl_ap_done')+' \u2014 '+esc(d2.machine_name)+'</div>'
        +'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center">'
        +'<div><div style="font-size:22px;font-weight:800">'+d2.total+'</div><div style="font-size:9px;color:var(--text3);text-transform:uppercase;margin-top:2px">'+t('fl_upload_rows')+'</div></div>'
        +'<div><div style="font-size:22px;font-weight:800;color:'+sCol+'">'+d2.failures+'</div><div style="font-size:9px;color:var(--text3);text-transform:uppercase;margin-top:2px">'+t('fl_upload_failures')+'</div></div>'
        +'<div><div style="font-size:22px;font-weight:800;color:'+sCol+'">'+d2.max_risk+'%</div><div style="font-size:9px;color:var(--text3);text-transform:uppercase;margin-top:2px">'+t('fl_upload_peak')+'</div></div>'
        +'</div></div>';
      submitBtn.textContent = t('fl_an_close');
      submitBtn.disabled = false;
      submitBtn.type = 'button';
      submitBtn.onclick = function(){closeModal(); submitBtn.type='submit'; submitBtn.onclick=null;};
      loadFleet(); return;
    } catch(ex) {
      resDiv.innerHTML = '<div style="color:var(--red);font-size:12px">'+t('fl_net_error')+'</div>';
      submitBtn.disabled=false; submitBtn.textContent=t('fl_save');
      loadFleet(); return;
    }
  }
  closeModal();
  loadFleet();
}

// ── DELETE ───────────────────────────────────────────────────────────────────
async function deleteMachine(id) {
  var m = _machines.find(function(x){return x.id===id;});
  if (!confirm(t('fl_del_confirm_msg')+(m?' ('+m.name+')':''))) return;
  await fetch('/api/machines/'+id, {method:'DELETE'});
  loadFleet();
}

async function rerunFile(mid) {
  var btn = document.getElementById('rerun-'+mid);
  if (btn) { btn.textContent='...'; btn.disabled=true; }
  try {
    var r = await fetch('/api/machines/analyze-all', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({machine_ids:[mid]})});
    await r.json();
    loadFleet();
  } catch(ex) { if(btn){btn.textContent=t('fl_btn_rerun');btn.disabled=false;} }
}

async function deleteFile(mid, fid) {
  if (!confirm(t('fl_del_file_msg'))) return;
  try { await fetch('/api/machines/'+mid+'/files/'+fid, {method:'DELETE'}); loadFleet(); }
  catch(ex) { alert(t('fl_net_error')); }
}

// ── ANALYZE ALL ──────────────────────────────────────────────────────────────
function showAP(html) { document.getElementById('ap-content').innerHTML=html; document.getElementById('ap-overlay').classList.add('open'); }
function closeAP() { document.getElementById('ap-overlay').classList.remove('open'); }

async function analyzeAll() {
  var btn = document.getElementById('analyze-all-btn');
  if (btn) { btn.textContent=t('fl_analyzing'); btn.disabled=true; }
  await loadFleet();
  var withFiles = _machines.filter(function(m){return m.saved_file;});
  if (!withFiles.length) { showAP('<p style="color:var(--text3);font-size:13px">'+t('fl_no_file_msg')+'</p>'); if(btn){btn.textContent=t('fl_analyze_all');btn.disabled=false;} return; }
  showAP('<div style="text-align:center;padding:20px"><span class="spinner"></span><span style="font-size:13px;color:var(--text3)">'+t('fl_analyzing')+'</span></div>');
  try {
    var r = await fetch('/api/machines/analyze-all', {method:'POST', headers:{'Content-Type':'application/json'}});
    var d = await r.json();
    if (!d.ok) { showAP('<p style="color:var(--red);font-size:13px">'+(d.error||t('fl_error'))+'</p>'); if(btn){btn.textContent=t('fl_analyze_all');btn.disabled=false;} return; }
    var ok = d.results.filter(function(x){return !x.skipped;});
    var skip = d.results.filter(function(x){return x.skipped;});
    var html = '<div style="font-size:11px;font-weight:700;color:var(--teal2);letter-spacing:1px;text-transform:uppercase;margin-bottom:14px">'+t('fl_ap_done')+' \u2014 '+ok.length+' machines</div>';
    if (ok.length) {
      html += '<div>';
      ok.forEach(function(x) {
        var rc = x.avg_risk>=70?'var(--red)':x.avg_risk>=45?'var(--amber)':'var(--green)';
        html += '<div class="ap-row"><div style="font-weight:600">'+esc(x.machine)+'</div>'
          +'<div style="color:var(--text3)">'+x.total+' '+t('fl_ap_rows')+'</div>'
          +'<div style="color:var(--amber)">'+x.failures+' '+t('fl_ap_anom')+'</div>'
          +'<div style="color:'+rc+'">'+x.avg_risk+'% '+t('fl_ap_avg')+'</div></div>';
      });
      html += '</div>';
    }
    if (skip.length) {
      html += '<div style="font-size:11px;color:var(--text3);padding-top:8px;border-top:1px solid var(--border)">';
      skip.forEach(function(x){html+='<div style="padding:3px 0">'+esc(x.machine)+' \u2014 '+(x.reason||t('fl_ap_skip'))+'</div>';});
      html += '</div>';
    }
    showAP(html); loadFleet();
  } catch(ex) { showAP('<p style="color:var(--red);font-size:13px">'+t('fl_net_error')+'</p>'); }
  if (btn) { btn.textContent=t('fl_analyze_all'); btn.disabled=false; }
}

// ── QUICK ANALYSE MODAL ──────────────────────────────────────────────────────
var _analyseMid = null;
function openAnalyse(id) {
  _analyseMid = id;
  var m = _machines.find(function(x){return x.id===id;});
  document.getElementById('an-title').textContent = t('fl_an_title') + (m?' \u2014 '+esc(m.name):'');
  document.getElementById('an-vib').value=_MEDIANS.vibration;
  document.getElementById('an-tp').value=_MEDIANS.temp_palier;
  document.getElementById('an-dbt').value=_MEDIANS.debit;
  document.getElementById('an-pe').value=_MEDIANS.pression_entree;
  document.getElementById('an-ps').value=_MEDIANS.pression_sortie;
  document.getElementById('an-im').value=_MEDIANS.courant_moteur;
  document.getElementById('an-tm').value=_MEDIANS.temp_moteur;
  document.getElementById('an-hf').value=_MEDIANS.heure_fonctionnement;
  document.getElementById('an-result').style.display='none';
  document.getElementById('an-result').innerHTML='';
  var btn=document.getElementById('an-submit'); btn.disabled=false; btn.textContent=t('fl_an_run');
  document.getElementById('analyse-overlay').classList.add('open');
  applyLang();
}
function closeAnalyse() { document.getElementById('analyse-overlay').classList.remove('open'); _analyseMid=null; }
function _anGv(id){return parseFloat(document.getElementById(id).value)||0;}

async function runAnalyse() {
  if (!_analyseMid) return;
  var m = _machines.find(function(x){return x.id===_analyseMid;});
  var btn = document.getElementById('an-submit');
  btn.disabled=true; btn.textContent=t('fl_analyzing');
  var payload = {
    vibration:_anGv('an-vib'),temp_palier:_anGv('an-tp'),debit:_anGv('an-dbt'),
    pression_entree:_anGv('an-pe'),pression_sortie:_anGv('an-ps'),courant_moteur:_anGv('an-im'),
    temp_moteur:_anGv('an-tm'),heure_fonctionnement:_anGv('an-hf'),
    machine_id:m?m.name:String(_analyseMid),threshold:m?m.threshold:45,
    pump_type:m?(m.pump_type||'centrifuge'):'centrifuge',
    fluid_type:m?(m.fluid_type||'eau'):'eau',
    roue_material:m?(m.roue_material||'inox_316'):'inox_316',
  };
  try {
    var r=await fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    var d=await r.json();
    var res=document.getElementById('an-result');
    res.style.display='block';
    if (d.error){res.innerHTML='<div style="color:var(--red);font-size:13px">'+d.error+'</div>';btn.disabled=false;btn.textContent=t('fl_an_run');return;}
    var risk=d.probabilite, rCol=riskColor(risk);
    var status=d.prediction===1
      ?'<span style="color:var(--red);font-weight:700">'+t('fl_an_failure')+'</span>'
      :'<span style="color:var(--green);font-weight:700">'+t('fl_an_normal')+'</span>';
    var zones=(d.zones&&d.zones.length)
      ?d.zones.map(function(z){return '<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:11px"><span style="color:var(--text2)">'+z.nom+'</span><span style="font-weight:700;color:var(--red)">'+z.proba+'%</span></div>';}).join('')
      :'<div style="font-size:11px;color:var(--text3)">'+t('fl_an_no_zone')+'</div>';
    var anom=d.anomaly_score!=null?'<div style="margin-top:8px;font-size:11px;color:var(--text3)">'+t('fl_an_anom')+': <strong style="color:var(--text)">'+d.anomaly_score+'/100</strong></div>':'';
    var rul=d.rul_hours!=null?'<div style="margin-top:4px;font-size:11px;color:var(--text3)">'+t('fl_an_rul')+': <strong style="color:var(--teal2)">'+d.rul_hours+' h</strong></div>':'';
    var warn=(d.domain_warnings&&d.domain_warnings.length)?'<div style="margin-top:8px;padding:8px;background:var(--amber-dim);border:1px solid rgba(255,159,10,0.25);border-radius:6px;font-size:11px;color:var(--amber)">'+d.domain_warnings.map(function(w){return '\u26a0 '+w;}).join('<br>')+'</div>':'';
    res.innerHTML='<div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:16px">'
      +'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">'
      +'<div style="font-size:40px;font-weight:800;color:'+rCol+';line-height:1">'+risk+'<span style="font-size:14px;color:var(--text3)">%</span></div>'
      +'<div>'+status+'</div></div>'
      +'<div style="height:3px;background:var(--border);border-radius:2px;margin-bottom:12px;overflow:hidden"><div style="height:100%;width:'+Math.min(risk,100)+'%;background:'+rCol+'"></div></div>'
      +'<div style="font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:6px">'+t('fl_an_zones')+'</div>'
      +zones+anom+rul+warn+'</div>';
    btn.textContent=t('fl_an_again'); btn.disabled=false;
    loadFleet();
  } catch(ex) {
    document.getElementById('an-result').innerHTML='<div style="color:var(--red);font-size:13px">'+t('fl_net_error')+'</div>';
    btn.disabled=false; btn.textContent=t('fl_an_run');
  }
}

loadFleet();
</script>
</div>
</div>
</div>

</body></html>"""

# ── API DOCS ──────────────────────────────────────────────────────────────────
API_DOCS_HTML = _AUTH_HEAD + """
<div style="width:100%;max-width:860px;margin:0 auto;padding:0 0 60px">
  <div style="padding:32px 0 24px;border-bottom:1px solid var(--border);margin-bottom:32px;display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-size:11px;font-weight:700;letter-spacing:0.04em;color:var(--teal-light);text-transform:uppercase">PILAR</div>
      <div style="font-size:22px;font-weight:800;color:var(--text);margin-top:6px">API Reference</div>
      <div style="font-size:12px;color:var(--text3);margin-top:4px">REST API · JSON · Authentication via API key</div>
    </div>
    <div style="text-align:right">
      <a href="/" style="font-size:12px;font-weight:500;color:var(--teal-light);text-decoration:none">App</a>
      <span style="color:var(--border2);margin:0 8px">|</span>
      <a href="/account" style="font-size:12px;font-weight:500;color:var(--teal-light);text-decoration:none">Account</a>
    </div>
  </div>

  <!-- AUTH -->
  <div style="margin-bottom:32px">
    <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text2);text-transform:uppercase;margin-bottom:12px">Authentication</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;box-shadow:var(--shadow-sm)">
      <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:14px">Every request must include your API key in the <code style="color:var(--teal-light);background:var(--surface2);padding:2px 6px;border-radius:6px">X-Api-Key</code> header.</p>
      {% if api_key %}
      <div style="background:var(--surface2);border:1px solid rgba(13,148,136,0.3);border-radius:var(--r-sm);padding:12px 16px;display:flex;align-items:center;justify-content:space-between">
        <code style="font-size:12px;color:var(--teal-light)">{{ api_key }}</code>
        <button onclick="navigator.clipboard.writeText('{{ api_key }}')" style="background:var(--teal-dim);border:1px solid var(--border2);border-radius:8px;padding:5px 12px;color:var(--teal-light);font-size:11px;font-weight:600;cursor:pointer">COPY</button>
      </div>
      {% else %}
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 16px">
        <code style="font-size:12px;color:var(--text3)">pk_your_api_key_here</code>
        <span style="font-size:11px;color:var(--text3);margin-left:12px">→ <a href="/account" style="color:var(--teal-light)">Get your key in Account</a></span>
      </div>
      {% endif %}
      <div style="margin-top:14px;font-size:11px;color:var(--text3)">Base URL: <code style="color:var(--text2)">https://trypilar.com</code></div>
    </div>
  </div>

  <!-- ENDPOINTS -->
  {% set ak = api_key if api_key else 'pk_your_api_key_here' %}

  <!-- POST /api/v1/analyze -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:var(--teal-dim);border:1px solid rgba(13,148,136,0.4);border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700;color:var(--teal-light)">POST</span>
      <code style="font-size:14px;color:var(--text);font-weight:700">/api/v1/analyze</code>
      <span style="font-size:11px;color:var(--text3)">— Single sensor reading</span>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow-sm)">
      <div style="display:grid;grid-template-columns:1fr 1fr">
        <div style="padding:16px;border-right:1px solid var(--border)">
          <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:10px">Request body</div>
          <pre style="font-size:11px;color:var(--text2);line-height:1.7;margin:0;overflow-x:auto">{
  "machine_id": "PUMP-01",
  "vibration": 2.5,
  "temp_palier": 65.0,
  "debit": 45,
  "pression_entree": 2.0,
  "pression_sortie": 115,
  "courant_moteur": 4.8,
  "temp_moteur": 70,
  "heure_fonctionnement": 5000
}</pre>
        </div>
        <div style="padding:16px">
          <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:10px">Response</div>
          <pre style="font-size:11px;color:var(--text2);line-height:1.7;margin:0;overflow-x:auto">{
  "ok": true,
  "analysis_id": 142,
  "timestamp": "2026-03-14T10:30:00Z",
  "machine_id": "PUMP-01",
  "prediction": 0,
  "risk": 12.4,
  "alert": false,
  "confidence": 100,
  "imputed": [],
  "zones": []
}</pre>
        </div>
      </div>
      <div style="padding:14px 16px;border-top:1px solid var(--border);background:var(--surface2)">
        <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:8px">cURL</div>
        <pre style="font-size:11px;color:var(--teal-light);margin:0;overflow-x:auto;white-space:pre-wrap">curl -X POST https://trypilar.com/api/v1/analyze \
  -H "X-Api-Key: {{ ak }}" \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"PUMP-01","vibration":2.5,"temp_palier":65,"debit":45,"pression_entree":2.0,"pression_sortie":115,"courant_moteur":4.8}'</pre>
      </div>
    </div>
  </div>

  <!-- POST /api/v1/analyze/batch -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:var(--teal-dim);border:1px solid rgba(13,148,136,0.4);border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700;color:var(--teal-light)">POST</span>
      <code style="font-size:14px;color:var(--text);font-weight:700">/api/v1/analyze/batch</code>
      <span style="font-size:11px;color:var(--text3)">— Up to 100 readings</span>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow-sm)">
      <div style="display:grid;grid-template-columns:1fr 1fr">
        <div style="padding:16px;border-right:1px solid var(--border)">
          <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:10px">Request body</div>
          <pre style="font-size:11px;color:var(--text2);line-height:1.7;margin:0;overflow-x:auto">{
  "readings": [
    {"machine_id":"P1","vibration":2.5,
     "temp_palier":65,"debit":45},
    {"machine_id":"P2","vibration":8.1,
     "temp_palier":92,"debit":28}
  ]
}</pre>
        </div>
        <div style="padding:16px">
          <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text3);text-transform:uppercase;margin-bottom:10px">Response</div>
          <pre style="font-size:11px;color:var(--text2);line-height:1.7;margin:0;overflow-x:auto">{
  "ok": true,
  "count": 2,
  "results": [
    {"index":0,"ok":true,
     "risk":8.1,"alert":false},
    {"index":1,"ok":true,
     "risk":67.3,"alert":true}
  ]
}</pre>
        </div>
      </div>
    </div>
  </div>

  <!-- GET /api/v1/history -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:rgba(191,90,242,0.15);border:1px solid rgba(191,90,242,0.4);border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700;color:var(--purple)">GET</span>
      <code style="font-size:14px;color:var(--text);font-weight:700">/api/v1/history</code>
      <span style="font-size:11px;color:var(--text3)">— Recent analyses</span>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:16px;box-shadow:var(--shadow-sm)">
      <div style="font-size:11px;color:var(--text2);line-height:2">
        <code style="color:var(--teal-light)">?limit=50</code> — Number of results (max 500)<br>
        <code style="color:var(--teal-light)">?machine_id=PUMP-01</code> — Filter by machine
      </div>
      <div style="margin-top:12px;padding:10px 14px;background:var(--surface2);border-radius:var(--r-sm)">
        <pre style="font-size:11px;color:var(--teal-light);margin:0">curl "https://trypilar.com/api/v1/history?limit=10&machine_id=PUMP-01" \
  -H "X-Api-Key: {{ ak }}"</pre>
      </div>
    </div>
  </div>

  <!-- GET /api/v1/status -->
  <div style="margin-bottom:32px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:rgba(191,90,242,0.15);border:1px solid rgba(191,90,242,0.4);border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700;color:var(--purple)">GET</span>
      <code style="font-size:14px;color:var(--text);font-weight:700">/api/v1/status</code>
      <span style="font-size:11px;color:var(--text3)">— Model info &amp; health</span>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:16px;box-shadow:var(--shadow-sm)">
      <pre style="font-size:11px;color:var(--text2);line-height:1.7;margin:0">{"ok":true,"model_name":"GradientBoosting","recall":98.1,"precision":89.8,"f1":93.8,"n_train":20902,"trained_at":"2026-03-14T06:29:50","plan":"free"}</pre>
    </div>
  </div>

  <!-- PARAMETERS TABLE -->
  <div style="margin-bottom:32px">
    <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text2);text-transform:uppercase;margin-bottom:14px">Parameters reference</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow-sm)">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="border-bottom:1px solid var(--border)">
          <th style="padding:10px 14px;text-align:left;color:var(--text2);font-size:11px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase">Field</th>
          <th style="padding:10px 14px;text-align:left;color:var(--text2);font-size:11px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase">Unit</th>
          <th style="padding:10px 14px;text-align:left;color:var(--text2);font-size:11px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase">Example</th>
          <th style="padding:10px 14px;text-align:left;color:var(--text2);font-size:11px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase">Required</th>
        </tr></thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>vibration</code></td><td style="padding:9px 14px;color:var(--text2)">mm/s</td><td style="padding:9px 14px;color:var(--text3)">2.5</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>temp_palier</code></td><td style="padding:9px 14px;color:var(--text2)">°C</td><td style="padding:9px 14px;color:var(--text3)">65.0</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>debit</code></td><td style="padding:9px 14px;color:var(--text2)">m³/h</td><td style="padding:9px 14px;color:var(--text3)">45.0</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>pression_entree</code></td><td style="padding:9px 14px;color:var(--text2)">bar</td><td style="padding:9px 14px;color:var(--text3)">2.0</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>pression_sortie</code></td><td style="padding:9px 14px;color:var(--text2)">bar</td><td style="padding:9px 14px;color:var(--text3)">115.0</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>courant_moteur</code></td><td style="padding:9px 14px;color:var(--text2)">A</td><td style="padding:9px 14px;color:var(--text3)">4.8</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>temp_moteur</code></td><td style="padding:9px 14px;color:var(--text2)">°C</td><td style="padding:9px 14px;color:var(--text3)">75.0</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>heure_fonctionnement</code></td><td style="padding:9px 14px;color:var(--text2)">h</td><td style="padding:9px 14px;color:var(--text3)">5000</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>machine_id</code></td><td style="padding:9px 14px;color:var(--text2)">string</td><td style="padding:9px 14px;color:var(--text3)">"PUMP-01"</td><td style="padding:9px 14px;color:var(--text3)">optional</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>temperature_ambiante</code></td><td style="padding:9px 14px;color:var(--text2)">°C</td><td style="padding:9px 14px;color:var(--text3)">25</td><td style="padding:9px 14px;color:var(--text2)">optional+</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 14px;color:var(--teal-light)"><code>niveau_huile</code></td><td style="padding:9px 14px;color:var(--text2)">%</td><td style="padding:9px 14px;color:var(--text3)">85</td><td style="padding:9px 14px;color:var(--text2)">optional+</td></tr>
          <tr><td style="padding:9px 14px;color:var(--teal-light)"><code>tension_reseau</code></td><td style="padding:9px 14px;color:var(--text2)">V</td><td style="padding:9px 14px;color:var(--text3)">400</td><td style="padding:9px 14px;color:var(--text2)">optional+</td></tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top:10px;font-size:11px;color:var(--text3)">At least one sensor field required. Missing core fields are imputed with dataset medians.</div>
  </div>

  <!-- PYTHON EXAMPLE -->
  <div style="margin-bottom:32px">
    <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text2);text-transform:uppercase;margin-bottom:14px">Python example</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;position:relative;box-shadow:var(--shadow-sm)">
      <button onclick="navigator.clipboard.writeText(document.getElementById('pyex').innerText)" style="position:absolute;top:12px;right:12px;background:var(--teal-dim);border:1px solid var(--border2);border-radius:8px;padding:5px 12px;color:var(--teal-light);font-size:11px;font-weight:600;cursor:pointer">COPY</button>
      <pre id="pyex" style="font-size:11px;color:var(--text2);line-height:1.8;margin:0;overflow-x:auto">import requests

API_KEY = "{{ ak }}"
BASE    = "https://trypilar.com"

# Single reading — centrifugal pump
resp = requests.post(f"{BASE}/api/v1/analyze",
    headers={"X-Api-Key": API_KEY},
    json={"machine_id": "PUMP-01", "vibration": 2.5, "temp_palier": 65,
          "debit": 45, "pression_entree": 2.0, "pression_sortie": 115,
          "courant_moteur": 4.8, "temp_moteur": 70, "heure_fonctionnement": 5000}
)
print(resp.json())

# Batch (PLC loop)
readings = [{"machine_id": f"P{i}", "vibration": 2.0+i*0.5, "temp_palier": 65+i*3}
            for i in range(5)]
resp = requests.post(f"{BASE}/api/v1/analyze/batch",
    headers={"X-Api-Key": API_KEY},
    json={"readings": readings}
)
for r in resp.json()["results"]:
    print(f"M{r['index']+1} — risk {r['risk']}% alert={r['alert']}")</pre>
    </div>
  </div>

  <!-- RATE LIMITS -->
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;box-shadow:var(--shadow-sm)">
    <div style="font-size:11px;font-weight:600;letter-spacing:0.04em;color:var(--text2);text-transform:uppercase;margin-bottom:12px">Rate limits</div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <tr style="border-bottom:1px solid var(--border)">
        <td style="padding:8px 0;color:var(--text2)">Free plan</td>
        <td style="padding:8px 0;color:var(--text);text-align:right;font-weight:700">1 000 requests / day</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:var(--text2)">Paid plan</td>
        <td style="padding:8px 0;color:var(--teal-light);text-align:right;font-weight:700">50 000 requests / day</td>
      </tr>
    </table>
    <div style="margin-top:10px;font-size:11px;color:var(--text3)">Resets at midnight UTC. HTTP 429 returned when exceeded.</div>
  </div>

</div>
</body></html>"""

# ── BACKEND ───────────────────────────────────────────────────────────────────
def predict_risk(params, threshold=45, return_extra=False, machine_context=None):
    global _iso_forest, _normal_samples
    # Per-machine imputation: override global medians with machine nominal values
    _medians = dict(FEATURE_MEDIANS)
    if machine_context:
        if machine_context.get('nominal_flow'):      _medians['debit']           = machine_context['nominal_flow']
        if machine_context.get('nominal_pressure'):  _medians['pression_sortie'] = machine_context['nominal_pressure']
        if machine_context.get('nominal_current'):   _medians['courant_moteur']  = machine_context['nominal_current']
        if machine_context.get('nominal_vibration'): _medians['vibration']       = machine_context['nominal_vibration']
    # Analyse partielle : imputer les features manquantes
    missing_keys = [k for k in CORE_FEATURES if params.get(k) is None]
    for k in missing_keys:
        params[k] = _medians[k]
    confidence = round((len(CORE_FEATURES) - len(missing_keys)) / len(CORE_FEATURES) * 100)
    row = [params[c] for c in COLONNES]
    donnees = pd.DataFrame([row], columns=COLONNES)
    donnees_scaled = scaler.transform(donnees)
    probabilite = round(float(model.predict_proba(donnees_scaled)[0][1]) * 100, 1)
    prediction = 1 if probabilite >= threshold else 0
    zones_risque = []
    if prediction == 1:
        for col, nom in FAILURE_ZONES.items():
            if col in modeles_zones:
                pz = round(float(modeles_zones[col].predict_proba(donnees_scaled)[0][1]) * 100, 1)
                if pz >= 30:
                    zones_risque.append({'nom': nom, 'proba': pz})
        zones_risque.sort(key=lambda x: x['proba'], reverse=True)
        # MOT: threshold rule — no trained model (constant data in training CSV)
        if 'MOT' not in modeles_zones:
            _curr = params.get('courant_moteur', _medians['courant_moteur'])
            _tmot = params.get('temp_moteur', _medians['temp_moteur'])
            # Adapt thresholds to machine nominal current (warn at +20%, critical at +50%)
            _nom_curr = (machine_context.get('nominal_current') if machine_context else None) or 18.0
            _curr_warn = _nom_curr * 1.2
            _curr_crit = _nom_curr * 1.5
            _curr_range = max(_curr_crit - _curr_warn, 1.0)
            _curr_score = min(100.0, max(0.0, (_curr - _curr_warn) / _curr_range * 100)) if _curr > _curr_warn else 0.0
            _temp_score = min(100.0, max(0.0, (_tmot - 75.0) / 35.0 * 100)) if _tmot > 85.0 else 0.0
            _pz_mot = round(max(_curr_score, _temp_score), 1)
            if _pz_mot >= 30:
                zones_risque.append({'nom': FAILURE_ZONES['MOT'], 'proba': _pz_mot})
                zones_risque.sort(key=lambda x: x['proba'], reverse=True)
    if not return_extra:
        return probabilite, prediction, zones_risque, confidence, missing_keys
    # ── Extra: Isolation Forest anomaly scoring + lazy training ───────────────
    anomaly_score = None
    try:
        if prediction == 0:
            _normal_samples.append(donnees_scaled[0].tolist())
            if len(_normal_samples) >= 30 and _iso_forest is None:
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=0.1, random_state=42)
                clf.fit(np.array(_normal_samples))
                _iso_forest = clf
                try:
                    with open(_pkl("isolation_forest.pkl"),"wb") as _f: pickle.dump(clf, _f)
                except Exception: pass
                print(f"[Pilar] Isolation Forest entraîné ({len(_normal_samples)} normaux)")
        anomaly_score = _compute_anomaly_score(donnees_scaled)
    except Exception as _ie:
        print(f"[Pilar] IsoForest pipeline error: {_ie}")
    # ── Extra: SHAP explanations ───────────────────────────────────────────────
    shap_explanations = _compute_shap(donnees_scaled)
    # ── Extra: RUL model (NASA C-MAPSS GBT) ───────────────────────────────────
    rul_hours = _compute_rul(donnees_scaled)
    return probabilite, prediction, zones_risque, confidence, missing_keys, anomaly_score, shap_explanations, rul_hours

def envoyer_alerte(email_to, probabilite, zones_risque, data, ack_token=None):
    severity = "CRITICAL" if probabilite >= 75 else "HIGH"
    sc = "#dc2626"
    zones_rows = "".join(f'<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#94a3b8;font-size:12px;">{z["nom"]}</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#dc2626;font-weight:700;">{z["proba"]}%</td></tr>' for z in zones_risque) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;">No specific zone identified</td></tr>'
    _base_url = os.environ.get('APP_URL', 'https://pilarapp.up.railway.app')
    ack_row = (f'<tr><td style="padding:0 28px 24px;"><a href="{_base_url}/alert/ack/{ack_token}" '
               'style="display:inline-block;padding:10px 22px;background:#0d9488;color:#fff;'
               'font-size:11px;font-weight:700;letter-spacing:2px;text-decoration:none;border-radius:4px;">'
               'ACKNOWLEDGE ALERT</a></td></tr>') if ack_token else ''
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;padding:40px 0;"><tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #1e2433;"><table width="100%" cellpadding="0" cellspacing="0"><tr><td><div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;">PILAR</div></td><td align="right"><span style="padding:4px 10px;background:rgba(220,38,38,0.12);border:1px solid #dc2626;border-radius:3px;color:#dc2626;font-size:10px;font-weight:700;letter-spacing:2px;">FAILURE ALERT</span></td></tr></table></td></tr>
<tr><td style="padding:28px;"><div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability</div><div style="font-size:52px;font-weight:800;color:{sc};line-height:1;">{probabilite}<span style="font-size:22px;color:#64748b;">%</span></div><div style="margin-top:8px;"><span style="padding:3px 10px;background:rgba(220,38,38,0.1);border:1px solid {sc};border-radius:3px;font-size:10px;font-weight:700;color:{sc};">SEVERITY: {severity}</span></div></td></tr>
<tr><td style="padding:0 28px 24px;"><table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;"><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Vibration</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("vibration")} mm/s</td></tr><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Bearing temp</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_palier")} °C</td></tr><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Flow rate</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("debit")} m³/h</td></tr><tr><td style="padding:8px 12px;color:#64748b;font-size:11px;">Motor temp</td><td style="padding:8px 12px;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_moteur")} °C</td></tr></table></td></tr>
<tr><td style="padding:0 28px 24px;"><div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:10px;">Failure Zones</div><table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">{zones_rows}</table></td></tr>
{ack_row}
<tr><td style="padding:16px 28px;border-top:1px solid #1e2433;background:#0a0d16;"><div style="font-size:10px;color:#64748b;">Pilar · {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div></td></tr>
</table></td></tr></table></body></html>"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Pilar Alert — Risk {probabilite}% | {severity}"
    msg['From'] = f"Pilar <{GMAIL}>"
    msg['To'] = email_to
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email_to, msg.as_string())
        print(f"Alert sent to {email_to}")
    except Exception as e:
        print(f"Email error: {e}")

def envoyer_escalade(email_to, probabilite, zones_risque, machine_id_str):
    """Send escalation email when primary alert is unacknowledged after 30 min."""
    zones_rows = "".join(f'<tr><td style="padding:8px 12px;color:#94a3b8;font-size:12px;">{z["nom"]}</td><td style="text-align:right;padding:8px 12px;color:#dc2626;font-weight:700;">{z["proba"]}%</td></tr>' for z in zones_risque) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;">No specific zone</td></tr>'
    machine_label = f' ({machine_id_str})' if machine_id_str else ''
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;"><tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #7c2d12;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #7c2d12;background:#1c0a04;">
<div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#f97316;text-transform:uppercase;">PILAR — ESCALATION ALERT</div>
<div style="font-size:12px;color:#94a3b8;margin-top:6px;">Primary contact did not acknowledge this alert within 30 minutes.</div></td></tr>
<tr><td style="padding:28px;">
<div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability{machine_label}</div>
<div style="font-size:52px;font-weight:800;color:#f97316;line-height:1;">{probabilite}<span style="font-size:22px;color:#64748b;">%</span></div></td></tr>
<tr><td style="padding:0 28px 28px;"><table width="100%">{zones_rows}</table></td></tr>
<tr><td style="padding:16px 28px;border-top:1px solid #1e2433;background:#0a0d16;">
<div style="font-size:10px;color:#64748b;">Pilar Escalation · {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div></td></tr>
</table></td></tr></table></body></html>"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Pilar ESCALATION — Risk {probabilite}% unacknowledged"
    msg['From'] = f"Pilar <{GMAIL}>"
    msg['To'] = email_to
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email_to, msg.as_string())
        print(f"Escalation sent to {email_to}")
    except Exception as e:
        print(f"Escalation email error: {e}")

def _escalation_worker():
    """Background thread: every 5 min, escalate unacked alerts older than 30 min."""
    import time as _time
    while True:
        _time.sleep(300)
        try:
            with app.app_context():
                cutoff = datetime.utcnow() - timedelta(minutes=30)
                pending = AlertLog.query.filter(
                    AlertLog.acked_at == None,
                    AlertLog.escalated_at == None,
                    AlertLog.escalation_email != None,
                    AlertLog.escalation_email != '',
                    AlertLog.sent_at <= cutoff
                ).all()
                for al in pending:
                    al.escalated_at = datetime.utcnow()
                    db.session.commit()
                    threading.Thread(target=envoyer_escalade, args=(
                        al.escalation_email, al.probabilite, [], al.machine_id_str), daemon=True).start()
                    print(f"[Pilar/escalation] AlertLog {al.id} escalated to {al.escalation_email}")
        except Exception as _esc_e:
            print(f"[Pilar/escalation] worker error: {_esc_e}")

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
        Analysis.timestamp >= datetime.utcnow() - timedelta(days=7)).all()
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, 'PILAR — Weekly Fleet Report', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Period: {(datetime.utcnow()-timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.utcnow().strftime('%Y-%m-%d')}", ln=True)
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
    pdf.cell(0, 5, f"Generated by Pilar at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def _send_weekly_reports():
    """Send PDF reports to all active users who have responsible_email set."""
    with app.app_context():
        users = User.query.filter(User.plan.in_(['pro', 'team'])).all()
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
                msg['Subject'] = f"Pilar — Weekly Fleet Report {datetime.utcnow().strftime('%Y-%m-%d')}"
                msg['From'] = f"Pilar <{GMAIL}>"
                msg['To'] = email
                body = MIMEText("Please find this week's fleet report attached.", 'plain')
                msg.attach(body)
                part = MIMEBase('application', 'pdf')
                part.set_payload(pdf_bytes)
                _enc.encode_base64(part)
                _fname = 'pilar_report_' + datetime.utcnow().strftime('%Y%m%d') + '.pdf'
                part.add_header('Content-Disposition', f'attachment; filename="{_fname}"')
                msg.attach(part)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(GMAIL, GMAIL_PWD)
                    smtp.sendmail(GMAIL, email, msg.as_string())
                print(f"[Pilar/pdf] Weekly report sent to {email}")
            except Exception as _pdf_e:
                print(f"[Pilar/pdf] Error for user {u.id}: {_pdf_e}")

# ── AUTO-RETRAIN ──────────────────────────────────────────────────────────────
_retrain_lock = threading.Lock()
_new_analyses_since_retrain = 0
# RETRAIN_TRIGGER imported from config.py

def _reload_models():
    """Reload pkl files into global model variables after a retrain."""
    global model, scaler, modeles_zones, _shap_explainer, _iso_forest, _rul_model, _rul_scaler, _normal_samples
    try:
        with open(_pkl('modele_pannes.pkl'), 'rb') as f: model = pickle.load(f)
        with open(_pkl('scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
        with open(_pkl('modeles_zones.pkl'), 'rb') as f: modeles_zones = pickle.load(f)
        _shap_explainer = None
        _normal_samples = []
        try:
            with open(_pkl('isolation_forest.pkl'), 'rb') as f: _iso_forest = pickle.load(f)
        except FileNotFoundError: _iso_forest = None
        try:
            with open(_pkl('rul_model.pkl'),  'rb') as f: _rul_model  = pickle.load(f)
            with open(_pkl('rul_scaler.pkl'), 'rb') as f: _rul_scaler = pickle.load(f)
        except FileNotFoundError: pass
        print('[Pilar/retrain] Models reloaded into memory')
    except Exception as _e:
        print(f'[Pilar/retrain] Reload error: {_e}')

def _auto_retrain():
    """Export Analysis DB rows to CSV and retrain the model pipeline."""
    global _new_analyses_since_retrain
    if not _retrain_lock.acquire(blocking=False):
        print('[Pilar/retrain] Retrain already in progress — skipped')
        return
    try:
        with app.app_context():
            import csv as _csv, tempfile as _tmp, subprocess as _sp, json as _js
            analyses = Analysis.query.order_by(Analysis.timestamp.asc()).all()
            if len(analyses) < 50:
                print(f'[Pilar/retrain] Not enough data ({len(analyses)} rows) — skipped')
                return
            print(f'[Pilar/retrain] Building training CSV from {len(analyses)} analyses...')
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
            print(f'[Pilar/retrain] Temp CSV: {csv_path}')
            # Run retrain_real.py as subprocess
            result = _sp.run(
                ['python', 'retrain_real.py', csv_path],
                capture_output=True, text=True, timeout=300
            )
            print('[Pilar/retrain] stdout:', result.stdout[-2000:] if result.stdout else '')
            if result.returncode == 0:
                print('[Pilar/retrain] Success — reloading models')
                _reload_models()
                _new_analyses_since_retrain = 0
                # Record retrain timestamp in settings
                try:
                    s = Settings.query.filter_by(key='last_auto_retrain', user_id=None).first()
                    ts = datetime.utcnow().isoformat()
                    if s: s.value = ts
                    else: db.session.add(Setting(key='last_auto_retrain', value=ts, user_id=None))
                    db.session.commit()
                except Exception: pass
            else:
                print(f'[Pilar/retrain] FAILED (rc={result.returncode}): {result.stderr[-1000:]}')
            try:
                import os as _os; _os.unlink(csv_path)
            except Exception: pass
    except Exception as _re:
        print(f'[Pilar/retrain] Error: {_re}')
    finally:
        _retrain_lock.release()

def _start_scheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        _sched = BackgroundScheduler()
        _sched.add_job(_send_weekly_reports, CronTrigger(day_of_week='mon', hour=8, minute=0),
                       id='weekly_pdf_reports', replace_existing=True)
        _sched.add_job(_auto_retrain, CronTrigger(day_of_week='sun', hour=3, minute=0),
                       id='weekly_auto_retrain', replace_existing=True)
        _sched.start()
        print("[Pilar] APScheduler started — weekly reports Mon 08:00 UTC | auto-retrain Sun 03:00 UTC")
    except Exception as _se:
        print(f"[Pilar] APScheduler not available: {_se}")

_start_scheduler()

# ── ROUTES AUTH ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        if current_uid(): return redirect('/monitor')
        return render_template_string(REGISTER_HTML, error=None, pending=False)
    ip = (request.headers.get('X-Forwarded-For','').split(',')[0].strip() if os.environ.get('RAILWAY_ENVIRONMENT') else '') or request.remote_addr or ''
    if _check_rate_limit(ip):
        print(f"[Pilar/auth] Rate limit register IP={ip}")
        return render_template_string(REGISTER_HTML, error='Trop de tentatives. Réessayez dans 15 minutes.', pending=False)
    try:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password', '')
        password2 = request.form.get('password2', '')
        if not email or not password:
            return render_template_string(REGISTER_HTML, error='Email et mot de passe requis', pending=False)
        if len(password) < 8:
            return render_template_string(REGISTER_HTML, error='Mot de passe trop court (8 caractères minimum)', pending=False)
        if password != password2:
            return render_template_string(REGISTER_HTML, error='Les mots de passe ne correspondent pas', pending=False)
        if BannedEmail.query.filter_by(email=email).first():
            _record_failed_login(ip)
            return render_template_string(REGISTER_HTML, error='Cette adresse email est bloquée. Contactez le support.', pending=False)
        if User.query.filter_by(email=email).first():
            _record_failed_login(ip)
            return render_template_string(REGISTER_HTML, error='Un compte existe déjà avec cet email', pending=False)
        api_key = 'pk_' + _secrets.token_hex(24)
        is_admin = (email == os.environ.get('ADMIN_EMAIL', '').lower()) or (User.query.count() == 0)
        token = _secrets.token_hex(32)
        # Les admins sont auto-vérifiés ; les autres doivent confirmer leur email
        needs_verify = not is_admin and bool(GMAIL)
        user = User(email=email, password_hash=generate_password_hash(password, method='pbkdf2:sha256:600000'),
                    email_verified=not needs_verify, verify_token=token if needs_verify else None,
                    api_key=api_key, is_admin=is_admin)
        db.session.add(user)
        db.session.commit()
        print(f"[Pilar/auth] New user: {email} (admin={is_admin}, verified={not needs_verify}) IP={ip}")
        if needs_verify:
            base_url = request.host_url.rstrip('/')
            threading.Thread(target=send_verify_email, args=(email, token, base_url), daemon=True).start()
            session['_pending_verify'] = email
            return render_template_string(REGISTER_HTML, error=None, pending=True, resent=False, pending_email=email)
        session['user_id'] = user.id
        session.permanent = True
        return redirect('/onboarding')
    except Exception as e:
        db.session.rollback()
        print(f"[Pilar/auth] Register error: {type(e).__name__}: {e}")
        return render_template_string(REGISTER_HTML, error='Erreur serveur. Veuillez réessayer.', pending=False)

@app.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'GET':
        # Page de renvoi autonome (si l'utilisateur revient plus tard)
        email = session.get('_pending_verify', '')
        return render_template_string(REGISTER_HTML, error=None, pending=True, resent=False, pending_email=email)
    email = (request.form.get('email') or session.get('_pending_verify', '')).strip().lower()
    if email:
        user = User.query.filter_by(email=email, email_verified=False).first()
        if user:
            if not GMAIL or not GMAIL_PWD:
                print(f"[Pilar/auth] Resend impossible: GMAIL non configuré pour {email}")
            else:
                token = _secrets.token_hex(32)
                user.verify_token = token
                try:
                    db.session.commit()
                    base_url = request.host_url.rstrip('/')
                    threading.Thread(target=send_verify_email, args=(email, token, base_url), daemon=True).start()
                    print(f"[Pilar/auth] Resend verification email: {email}")
                except Exception as e:
                    db.session.rollback()
                    print(f"[Pilar/auth] Resend error: {e}")
    # On affiche toujours le succès (anti-énumération)
    return render_template_string(REGISTER_HTML, error=None, pending=True, resent=True, pending_email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if current_uid(): return redirect('/monitor')
        return render_template_string(LOGIN_HTML, error=None)
    ip = (request.headers.get('X-Forwarded-For','').split(',')[0].strip() if os.environ.get('RAILWAY_ENVIRONMENT') else '') or request.remote_addr or ''
    if _check_rate_limit(ip):
        print(f"[Pilar/auth] Rate limit login IP={ip}")
        return render_template_string(LOGIN_HTML, error='Trop de tentatives. Réessayez dans 15 minutes.')
    try:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password', '')
        if not email or not password:
            return render_template_string(LOGIN_HTML, error='Email et mot de passe requis')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            _record_failed_login(ip)
            print(f"[Pilar/auth] Failed login: {email} IP={ip}")
            return render_template_string(LOGIN_HTML, error='Email ou mot de passe incorrect')
        if user.is_banned:
            print(f"[Pilar/auth] Banned login attempt: {email} IP={ip}")
            return render_template_string(LOGIN_HTML, error='Ce compte a été suspendu. Contactez le support.')
        if not user.email_verified:
            # If GMAIL is not configured, no verification email was ever sent — auto-verify
            if not GMAIL:
                user.email_verified = True
                db.session.commit()
            else:
                return render_template_string(LOGIN_HTML, error='Confirmez votre email avant de vous connecter. Vérifiez vos spams.')
        session['user_id'] = user.id
        session.permanent = True
        print(f"[Pilar/auth] Login OK: {email} IP={ip}")
        return redirect('/monitor' if user.onboarded else '/onboarding')
    except Exception as e:
        db.session.rollback()
        print(f"[Pilar/auth] Login error: {type(e).__name__}: {e}")
        return render_template_string(LOGIN_HTML, error='Erreur serveur. Veuillez réessayer.')

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
    now = datetime.utcnow()
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
    return render_template_string(ADMIN_HTML, users=users, total_users=total_users,
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
    print(f"[Pilar/admin] PLAN_CHANGE by {admin_user.email if admin_user else '?'}: user={user.email} {old_plan}->{plan} expires={expires_str or 'none'} note={note[:50] if note else ''}")
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
    print(f"[Pilar/admin] QUOTA_SET: user={user.email} quota={quota}")
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
    print(f"[Pilar/admin] {action} by {me.email}: target={target.email}")
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
    print(f"[Pilar/admin] {action} by {me.email}: target={target.email}")
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
    print(f"[Pilar/admin] DELETE_USER by {me.email}: deleted={email}")
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
    print(f"[Pilar/admin] BLOCK_EMAIL by {me.email}: email={email}")
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
    print(f"[Pilar/admin] UNBLOCK_EMAIL by {me.email}: email={email}")
    return jsonify({'ok': True})

# Commandes autorisées dans le terminal admin (préfixes)
_TERM_ALLOWED = [
    'python --version', 'python -V', 'python3 --version',
    'pip list', 'pip show', 'pip freeze',
    'ls', 'dir', 'pwd',
    'python -c "from etape7', 'python -c "import pickle',
    'python -c "import sys', 'python -c "import platform',
    'env | grep', 'set | findstr',
    'python retrain_real.py', 'python retrain_kaggle.py', 'python retrain_nln_emp.py',
]
_TERM_BLOCKED = ['rm ', 'del ', 'rmdir', 'curl ', 'wget ', 'nc ', 'ncat ',
                 'bash ', 'sh ', 'exec(', 'eval(', '> /', 'sudo ', 'chmod ',
                 'dd ', 'mkfs', 'kill ', 'pkill', ';rm', '&&rm', '|rm',
                 'pip install', 'pip uninstall', '__import__']

@app.route('/admin/terminal', methods=['POST'])
@admin_required
def admin_terminal():
    cmd = (request.json or {}).get('cmd', '').strip()
    if not cmd:
        return jsonify({'output': '', 'code': 0})
    cmd_lower = cmd.lower()
    # Bloquer les commandes dangereuses
    for blocked in _TERM_BLOCKED:
        if blocked.lower() in cmd_lower:
            print(f"[Pilar/terminal] BLOCKED cmd by {current_uid()}: {cmd[:80]}")
            return jsonify({'output': f'❌ Commande bloquée pour sécurité : contient "{blocked}"', 'code': 1})
    # Autoriser seulement les préfixes whitelistés
    allowed = any(cmd.startswith(a) for a in _TERM_ALLOWED)
    if not allowed:
        print(f"[Pilar/terminal] NOT_WHITELISTED cmd by {current_uid()}: {cmd[:80]}")
        return jsonify({'output': '❌ Commande non autorisée. Seules les commandes de diagnostic sont permises.', 'code': 1})
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return jsonify({'output': output.rstrip('\n')[:8000], 'code': result.returncode})
    except subprocess.TimeoutExpired:
        return jsonify({'output': 'Timeout (15s)', 'code': -1})
    except Exception as e:
        return jsonify({'output': str(e), 'code': -1})

@app.route('/admin/impersonate/<int:uid>')
@admin_required
def impersonate(uid):
    admin_user = db.session.get(User, current_uid())
    target = db.session.get(User, uid)
    print(f"[Pilar/admin] IMPERSONATE by {admin_user.email if admin_user else '?'}: target={target.email if target else uid}")
    session['user_id'] = uid
    return redirect('/monitor')

# ── ROUTES PAGES ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect('/monitor')

@app.route('/demo')
def demo():
    return DEMO_HTML

# ── ALERT ACK ─────────────────────────────────────────────────────────────────
@app.route('/alert/ack/<token>')
def alert_ack(token):
    al = AlertLog.query.filter_by(ack_token=token).first()
    if not al:
        return '<html><body style="font-family:-apple-system,\'SF Pro Display\',\'Helvetica Neue\',Arial,sans-serif;background:#07090f;color:#ffffff;display:flex;align-items:center;justify-content:center;min-height:100vh;"><div style="text-align:center"><div style="font-size:15px;letter-spacing:0.04em;color:#0d9488;font-weight:700;margin-bottom:16px;">PILAR</div><p style="color:rgba(235,235,245,0.6)">Alert not found or already processed.</p></div></body></html>', 404
    if not al.acked_at:
        al.acked_at = datetime.utcnow()
        db.session.commit()
    return '<html><body style="font-family:-apple-system,\'SF Pro Display\',\'Helvetica Neue\',Arial,sans-serif;background:#07090f;color:#ffffff;display:flex;align-items:center;justify-content:center;min-height:100vh;"><div style="text-align:center"><div style="font-size:15px;letter-spacing:0.04em;color:#0d9488;font-weight:700;margin-bottom:16px;">PILAR</div><h2 style="margin:0 0 12px;font-size:20px;font-weight:700;">Alert Acknowledged</h2><p style="color:rgba(235,235,245,0.6);font-size:14px;">This alert has been recorded. No escalation will be sent.</p><a href="/monitor" style="display:inline-block;margin-top:20px;padding:12px 24px;background:#0d9488;color:#fff;text-decoration:none;border-radius:12px;font-size:15px;font-weight:600;">Go to Dashboard</a></div></body></html>'

# ── MACHINES CRUD API ─────────────────────────────────────────────────────────
@app.route('/api/machines', methods=['GET'])
@login_required
def api_machines_list():
    uid = current_uid()
    machines = Machine.query.filter_by(user_id=uid).order_by(Machine.created_at.desc()).all()
    result = []
    for m in machines:
        last = Analysis.query.filter_by(machine_id=m.id).order_by(Analysis.timestamp.desc()).first()
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
            'last_risk': last.risk if last else None,
            'last_prediction': last.prediction if last else None,
            'last_seen': last.timestamp.isoformat() + 'Z' if last else None,
            'saved_file': (lambda sf: {'id': sf.id, 'filename': sf.filename, 'row_count': sf.row_count, 'created_at': sf.created_at.isoformat()+'Z'} if sf else None)(
                SavedFile.query.filter_by(machine_id=m.id, user_id=uid).order_by(SavedFile.created_at.desc()).first()
            ),
        })
    return jsonify(result)

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
        threshold=float(d.get('threshold') or 45),
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
    machine_json = _json.dumps({
        'id': m.id, 'name': m.name, 'description': m.description or '',
        'pump_type': m.pump_type or 'centrifuge', 'fluid_type': m.fluid_type or 'eau',
        'roue_material': m.roue_material or 'inox_316', 'machine_type': m.machine_type or 'M',
        'threshold': m.threshold or 45, 'location': m.location or '',
        'install_date': m.install_date.isoformat() if m.install_date else '',
        'serial_number': m.serial_number or '',
        'nominal_flow': m.nominal_flow, 'nominal_pressure': m.nominal_pressure,
        'power_kw': m.power_kw, 'nominal_current': m.nominal_current,
        'nominal_vibration': m.nominal_vibration,
        'alert_email': m.alert_email or '', 'escalation_email': m.escalation_email or '',
    })
    return render_template_string(MACHINE_SPACE_HTML, machine=m, machine_json=machine_json, sidebar_html=nav("fl"))

@app.route('/dashboard')
@login_required
def fleet_dashboard():
    r = _paid_required()
    if r: return r
    return DASHBOARD_HTML

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
    threshold = float(m.threshold) if m.threshold else 45.0
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
            threshold = float(m.threshold) if m.threshold else 45.0
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
            msg['Subject'] = f'Pilar — Machine Request: {name}'
            msg['From'] = f'Pilar <{GMAIL}>'
            msg['To'] = 'aliguenbou07r@gmail.com'
            msg.attach(MIMEText(html, 'html'))
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(GMAIL, GMAIL_PWD)
                    smtp.sendmail(GMAIL, 'aliguenbou07r@gmail.com', msg.as_string())
            except Exception as _e:
                print(f'[Pilar/machine-request] email error: {_e}')
        with app.app_context():
            threading.Thread(target=_notify, daemon=True).start()
    return jsonify({'ok': True, 'id': mr.id})

@app.route('/onboarding')
@login_required
def onboarding():
    return ONBOARDING_HTML

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
        print(f"[Pilar/onboarding] ERROR: {type(e).__name__}: {e}")
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
    return render_template_string(HTML)

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
    return render_template_string(ACCOUNT_HTML, user=user, team=team, members=members, my_role=my_role)

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
<a href="mailto:aliguenbou07r@gmail.com?subject=Pilar%20%E2%80%94%20Custom%20Plan%20Request&body=Hi%2C%20I%27d%20like%20to%20discuss%20a%20custom%20plan%20for%20my%20team." class="btn">&#128231; Contact us to get access</a>
<hr class="divider">
<a href="/monitor" class="btn-ghost">Back to Monitor</a>
</div>
</body></html>"""
    return html

@app.route('/tutorial')
@login_required
def tutorial(): return render_template_string(TUTORIAL_HTML)

@app.route('/adapter')
def adapter():
    r = _paid_required()
    if r: return r
    return render_template_string(ADAPTER_HTML)

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
    return render_template_string(TWIN_HTML)

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
    return render_template_string(HISTORY_HTML, analyses=analyses, total=total,
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
    return render_template_string(SETTINGS_HTML)

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
        _threshold = float(_machine.threshold) if (_machine and _machine.threshold) else 45.0
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
        return jsonify({'prediction': prediction, 'probabilite': probabilite,
                        'zones': zones_risque, 'mail_envoye': mail_envoye,
                        'confidence': confidence, 'imputed': imputed,
                        'anomaly_score': anomaly_score,
                        'shap_explanations': shap_explanations,
                        'rul_hours': rul_hours,
                        'domain_warnings': _domain_warnings})
    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"[Pilar/predire] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Erreur interne — réessayez ou contactez le support'}), 500

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
        now = datetime.utcnow()
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
            print(f"[Pilar/RUL] {_re}")
        return jsonify({'has_data':True,'current_risk':last.risk,'avg_risk_24h':avg_risk,
            'anomaly_rate':anomaly_rate,'total_analyses':total,'failure_hours':failure_hours,'trend':trend,
            'history_times':history_times,'history_risks':history_risks,'history_wear':history_wear,'history_temp':history_temp,
            'future_times':future_times,'future_risks':future_risks,'future_wear':future_wear,'future_temp':future_temp,
            'rul_hours':rul_hours,'rul_confidence':rul_confidence,'rul_degradation':rul_degradation_rates,
            'last_params':{'vibration':FEATURE_MEDIANS['vibration'],'debit':last.vitesse,'pression_sortie':last.couple,'heure_fonctionnement':last.usure,'temp_palier':last.temp_air}})
    except Exception as e:
        import traceback
        print(f"[Pilar/api_twin] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Erreur serveur'}), 500

@app.route('/api/update_info')
def api_update_info():
    return jsonify(_UPDATE_INFO)

@app.route('/api/health')
def api_health():
    import os, sys, json as _json
    meta = {}
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
    })

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
        dp.updated_at   = datetime.utcnow()
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
    today = datetime.utcnow().date().isoformat()
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
        _threshold = float(_machine.threshold) if (_machine and _machine.threshold) else 45.0
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
        print(f"[Pilar/api_v1_analyze] {type(e).__name__}: {e}\n{traceback.format_exc()}")
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
    return render_template_string(API_DOCS_HTML, api_key=api_key, ak=api_key)

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
        print(f"[Pilar/api_whatif] ERROR: {type(e).__name__}: {e}")
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
    print(f"[Pilar] 500 ERROR:\n{tb}")
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
    print(f"[Pilar] Unhandled exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    try: db.session.rollback()
    except: pass
    return internal_error(e)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SYNC CLIENT & LOCAL CHAT
# ══════════════════════════════════════════════════════════════════════════════

# ── Sync Config (set via environment or config.py) ────────────────────────────
PILAR_SYNC_URL    = os.environ.get("PILAR_SYNC_URL", "")          # e.g. https://pilar-sync.railway.app
PILAR_CLIENT_ID   = os.environ.get("PILAR_CLIENT_ID", "")
PILAR_CLIENT_SECRET = os.environ.get("PILAR_CLIENT_SECRET", "")

# ── New DB Models ─────────────────────────────────────────────────────────────

class SyncQueue(db.Model):
    """Queues local analyses/notes for uploading to the sync server."""
    __tablename__ = "sync_queue"
    id         = db.Column(db.Integer, primary_key=True)
    data_type  = db.Column(db.String(20), default="analysis")  # "analysis" | "note"
    data_json  = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    synced_at  = db.Column(db.DateTime, nullable=True)

    @property
    def is_synced(self):
        return self.synced_at is not None


class LocalChatMessage(db.Model):
    """Local storage for chat messages (online: pulled from sync server; offline: created locally)."""
    __tablename__ = "local_chat_message"
    id          = db.Column(db.Integer, primary_key=True)
    remote_id   = db.Column(db.Integer, nullable=True)   # id on sync server (null if not yet pushed)
    client_id   = db.Column(db.String(64))
    sender_name = db.Column(db.String(200))
    room        = db.Column(db.String(100), default="general")
    content     = db.Column(db.Text)
    image_data  = db.Column(db.Text)   # base64
    image_mime  = db.Column(db.String(50))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    is_local    = db.Column(db.Boolean, default=False)   # True = created offline, not yet synced
    synced_at   = db.Column(db.DateTime, nullable=True)

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
        print("[Pilar] Sync tables ready")
    except Exception as _se:
        print(f"[Pilar] Sync table init: {_se}")


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
            print("[SyncClient] No sync credentials — sync disabled")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[SyncClient] Started (target={PILAR_SYNC_URL})")

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
                    now = datetime.utcnow()
                    SyncQueue.query.filter(SyncQueue.id.in_(ids)).update(
                        {SyncQueue.synced_at: now}, synchronize_session=False
                    )
                    db.session.commit()
                    print(f"[SyncClient] Pushed {len(analyses)} analyses, {len(notes)} notes")
            except Exception as e:
                print(f"[SyncClient] Push failed: {e}")

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
                        ts = datetime.utcnow()
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
                        synced_at   = datetime.utcnow(),
                    )
                    db.session.add(msg)
                if messages:
                    db.session.commit()
                    print(f"[SyncClient] Pulled {len(messages)} chat messages")
                self._last_pull_ts = datetime.utcnow()
            except Exception as e:
                print(f"[SyncClient] Pull failed: {e}")

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
                    msg.synced_at = datetime.utcnow()
                    msg.is_local  = False
                    db.session.commit()
                except Exception as e:
                    print(f"[SyncClient] Chat push failed: {e}")

    def _loop(self):
        while not self._stop_event.wait(self.INTERVAL):
            try:
                was_online = self._online
                self._online = self._check_connectivity()
                if not was_online and self._online:
                    print("[SyncClient] Back online")
                elif was_online and not self._online:
                    print("[SyncClient] Gone offline")

                if self._online:
                    self._push_queue()
                    self._pull_chat()
                    self._push_local_chat()
                    self._last_sync = datetime.utcnow()
            except Exception as e:
                print(f"[SyncClient] Loop error: {e}")

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
            print(f"[SyncClient] Enqueue analysis error: {e}")


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
            print(f"[SyncClient] Queue error: {_qe}")


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
            msg.synced_at = datetime.utcnow()
            msg.is_local  = False
            db.session.commit()
        except Exception as _pe:
            print(f"[Chat] Immediate push failed: {_pe}")

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
    print(f"Pilar v3 — http://localhost:{port} (debug={debug})")
    app.run(debug=debug, host='0.0.0.0', port=port)