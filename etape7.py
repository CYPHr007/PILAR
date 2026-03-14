from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, g
import pickle, threading, smtplib, secrets as _secrets, subprocess
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd, warnings, time, collections
warnings.filterwarnings("ignore")

# Rate limiting : {ip: [(timestamp, failed_bool), ...]}
_login_attempts = collections.defaultdict(list)
# API rate limiting : {api_key: {'count': N, 'day': 'YYYY-MM-DD'}}
_api_calls = {}
_RATE_WINDOW = 900   # 15 minutes
_RATE_MAX    = 10    # max 10 tentatives échouées par fenêtre

def _check_rate_limit(ip):
    """Retourne True si l'IP est bloquée."""
    now = time.time()
    attempts = _login_attempts[ip]
    # Nettoyer les vieilles entrées
    _login_attempts[ip] = [t for t in attempts if now - t < _RATE_WINDOW]
    return len(_login_attempts[ip]) >= _RATE_MAX

def _record_failed_login(ip):
    _login_attempts[ip].append(time.time())

app = Flask(__name__)
import os
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
    import secrets as _s
    _secret_key = _s.token_hex(32)
    print("[Pilar] WARNING: SECRET_KEY not set — generating random key (sessions will reset on restart)")
app.config["SECRET_KEY"] = _secret_key
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("RAILWAY_ENVIRONMENT") is not None
db = SQLAlchemy(app)

@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    if os.environ.get("RAILWAY_ENVIRONMENT"):
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
    team_id        = db.Column(db.Integer, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

class Settings(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    key     = db.Column(db.String(120))
    value   = db.Column(db.String(500))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class Analysis(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)
    machine_type = db.Column(db.String(10))
    temp_air     = db.Column(db.Float)
    temp_process = db.Column(db.Float)
    vitesse      = db.Column(db.Float)
    couple       = db.Column(db.Float)
    usure        = db.Column(db.Float)
    risk         = db.Column(db.Float)
    prediction   = db.Column(db.Integer)
    zones        = db.Column(db.String(500))
    mail_sent    = db.Column(db.Boolean, default=False)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    extra_params = db.Column(db.Text)
    confidence   = db.Column(db.Integer, default=100)
    machine_id   = db.Column(db.String(100))

class SavedFile(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, nullable=True)
    filename   = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    row_count  = db.Column(db.Integer, default=0)
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

try:
    with open("modele_pannes.pkl","rb") as f: model = pickle.load(f)
    with open("scaler.pkl","rb") as f: scaler = pickle.load(f)
    with open("modeles_zones.pkl","rb") as f: modeles_zones = pickle.load(f)
    print("[Pilar] Modeles ML charges")
except FileNotFoundError as _e:
    print(f"[Pilar] FATAL: fichier modele manquant — {_e}")
    model = scaler = None
    modeles_zones = {}
except Exception as _e:
    print(f"[Pilar] FATAL: erreur chargement modele — {_e}")
    model = scaler = None
    modeles_zones = {}

FAILURE_ZONES = {"TWF":"Tool Wear Failure","HDF":"Heat Dissipation Failure","PWF":"Power Failure","OSF":"Overstrain Failure","RNF":"Random Failure"}
COLONNES = ["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","ecart_temp","puissance"]
# Médianes données réelles — utilisées pour imputer les features manquantes (analyse partielle)
FEATURE_MEDIANS = {'type':1.0,'temp_air':377.0,'temp_process':314.6,'vitesse':1298.0,'couple':257.8,'usure':8248.0}
CORE_FEATURES   = list(FEATURE_MEDIANS.keys())
OPTIONAL_FIELDS = ['humidite','vibration','pression','courant','tension']
GMAIL     = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_PWD = os.environ.get("GMAIL_APP_PASSWORD", "")
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

def send_verify_email(email, token):
    if not GMAIL or not GMAIL_PWD:
        print(f"[Pilar/auth] Email non configuré (GMAIL/GMAIL_APP_PASSWORD manquants) — token pour {email}: {token}")
        return
    base = os.environ.get("APP_URL", "http://localhost:5000").rstrip('/')
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
_AUTH_HEAD = """<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0e1118"><title>Pilar</title>
<script>
if('serviceWorker' in navigator){
  navigator.serviceWorker.getRegistrations().then(function(regs){
    regs.forEach(function(r){r.unregister();});
  });
}
</script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#07090f;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}
.ac{width:100%;max-width:380px;}
.logo{font-size:13px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;text-align:center;margin-bottom:32px;}
.card{background:#0e1118;border:1px solid #1e2433;border-radius:10px;padding:28px;}
.ctitle{font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:20px;}
.flbl{font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:7px;display:block;margin-top:14px;}
.fi{width:100%;padding:11px 14px;background:#141820;border:1px solid #252d3d;border-radius:6px;color:#e2e8f0;font-size:13px;outline:none;transition:border-color 0.15s;}
.fi:focus{border-color:#0d9488;}
.fi::placeholder{color:#475569;}
.btn{width:100%;padding:13px;background:#0d9488;color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;margin-top:20px;transition:background 0.15s;}
.btn:hover{background:#14b8a6;}
.err{padding:10px 14px;background:rgba(220,38,38,0.1);border:1px solid #dc2626;border-radius:6px;font-size:12px;color:#dc2626;margin-top:14px;display:none;}
.ok{padding:10px 14px;background:rgba(5,150,105,0.1);border:1px solid #059669;border-radius:6px;font-size:12px;color:#34d399;margin-top:14px;display:none;}
.link{text-align:center;margin-top:18px;font-size:11px;color:#64748b;}
.link a{color:#14b8a6;text-decoration:none;}
.badge{padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;}
.badge.ok{background:rgba(5,150,105,0.12);color:#059669;}
.badge.free{background:rgba(13,148,136,0.12);color:#14b8a6;}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px;}
th{text-align:left;padding:8px 10px;color:#64748b;font-size:9px;letter-spacing:1px;border-bottom:1px solid #1e2433;text-transform:uppercase;}
td{padding:8px 10px;border-bottom:1px solid #1e2433;color:#94a3b8;}
tr:last-child td{border-bottom:none;}
.kgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0;}
.kc{background:#141820;border:1px solid #1e2433;border-radius:8px;padding:14px;}
.kv{font-size:22px;font-weight:800;color:#14b8a6;}
.kl{font-size:9px;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-top:3px;}
.lang-sw{position:fixed;top:14px;right:14px;display:flex;gap:2px;background:#141820;border:1px solid #1e2433;border-radius:6px;padding:3px;z-index:99;}
.lang-sw button{padding:4px 10px;border:none;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;background:transparent;color:#64748b;transition:all .15s;}
.lang-sw button.active{background:#0d9488;color:#fff;}
</style></head><body>
<div class="lang-sw" id="_authLang">
  <button id="_authEN" onclick="_authSetLang('en')">EN</button>
  <button id="_authFR" onclick="_authSetLang('fr')">FR</button>
</div>
<script>
var _aLang=localStorage.getItem('pilar_lang')||'en';
var _TA={
en:{login_title:'SIGN IN',reg_title:'CREATE ACCOUNT',lbl_email:'Email',lbl_email_pro:'Professional Email',lbl_pw:'Password',lbl_pw_min:'Min. 8 characters',lbl_pw_conf:'Confirm Password',btn_login:'Sign In',btn_guest:'Continue without account',btn_reg:'Create Account',link_noreg:'No account yet? <a href="/register">Create one</a>',link_haveac:'Already have an account? <a href="/login">Sign In</a>',verify_title:'Check your email',verify_desc:'A confirmation link was sent to your address.<br>Click the link to activate your account.',verify_note:'Valid 24h · Check your spam',back_login:'Back to Sign In',resend_btn:'Resend email',resent_ok:'Email sent again!'},
fr:{login_title:'CONNEXION',reg_title:'CRÉER UN COMPTE',lbl_email:'Email',lbl_email_pro:'Email professionnel',lbl_pw:'Mot de passe',lbl_pw_min:'8 caractères minimum',lbl_pw_conf:'Confirmer le mot de passe',btn_login:'Se connecter',btn_guest:'Continuer sans compte',btn_reg:'Créer mon compte',link_noreg:'Pas encore de compte ? <a href="/register">Créer un compte</a>',link_haveac:'Déjà un compte ? <a href="/login">Se connecter</a>',verify_title:'Vérifiez votre email',verify_desc:'Un lien de confirmation a été envoyé à votre adresse.<br>Cliquez sur le lien pour activer votre compte.',verify_note:'Lien valide 24h · Vérifiez vos spams',back_login:'Retour à la connexion',resend_btn:'Renvoyer l\'email',resent_ok:'Email renvoyé !'}
};
function _tA(k){return(_TA[_aLang]||_TA.en)[k]||k;}
function _authSetLang(l){
  _aLang=l;localStorage.setItem('pilar_lang',l);
  document.getElementById('_authEN').className=l==='en'?'active':'';
  document.getElementById('_authFR').className=l==='fr'?'active':'';
  document.querySelectorAll('[data-iauth]').forEach(function(el){
    var k=el.getAttribute('data-iauth');var v=_tA(k);
    if(el.tagName==='INPUT'){el.placeholder=v;}else{el.innerHTML=v;}
  });
}
(function(){_authSetLang(_aLang);})();
document.addEventListener('DOMContentLoaded',function(){_authSetLang(_aLang);});
</script>"""

LOGIN_HTML = _AUTH_HEAD + """
<div class="ac">
  <div class="logo">PILAR</div>
  <div class="card">
    <div class="ctitle" data-iauth="login_title">CONNEXION</div>
    {% if error %}<div class="err" style="display:block">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
      <label class="flbl" for="em" data-iauth="lbl_email">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-iauth="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="••••••••" autocomplete="current-password" required>
      <button type="submit" class="btn" data-iauth="btn_login">Se connecter</button>
    </form>
    <a href="/" class="btn" data-iauth="btn_guest" style="display:block;text-align:center;text-decoration:none;background:transparent;border:1px solid #252d3d;color:#64748b;margin-top:8px;padding:13px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Continuer sans compte</a>
  </div>
  <div class="link" data-iauth="link_noreg">Pas encore de compte ? <a href="/register">Créer un compte</a></div>
</div>
</body></html>"""

REGISTER_HTML = _AUTH_HEAD + """
<div class="ac">
  <div class="logo">PILAR</div>
  {% if pending %}
  <div class="card" style="text-align:center;padding:32px 24px">
    <div style="width:48px;height:48px;border-radius:12px;background:rgba(13,148,136,.1);border:1px solid rgba(13,148,136,.2);display:flex;align-items:center;justify-content:center;margin:0 auto 16px">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 01-2.06 0L2 7"/></svg>
    </div>
    <div style="font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:8px" data-iauth="verify_title">Vérifiez votre email</div>
    <div style="font-size:12px;color:#64748b;line-height:1.7" data-iauth="verify_desc">Un lien de confirmation a été envoyé à votre adresse.<br>Cliquez sur le lien pour activer votre compte.</div>
    <div style="margin-top:20px;font-size:11px;color:#334155" data-iauth="verify_note">Lien valide 24h · Vérifiez vos spams</div>
    {% if resent|default(False) %}
    <div style="margin-top:16px;padding:10px 14px;background:rgba(5,150,105,0.1);border:1px solid #059669;border-radius:6px;font-size:12px;color:#34d399" data-iauth="resent_ok">Email renvoyé !</div>
    {% endif %}
    <form method="POST" action="/resend-verification" style="margin-top:20px">
      <input type="hidden" name="email" value="{{ pending_email|default('') }}">
      <button type="submit" style="width:100%;padding:11px;background:transparent;border:1px solid #1e2433;border-radius:6px;color:#64748b;font-size:11px;font-weight:700;letter-spacing:1px;cursor:pointer;transition:border-color 0.15s" onmouseover="this.style.borderColor='#0d9488';this.style.color='#14b8a6'" onmouseout="this.style.borderColor='#1e2433';this.style.color='#64748b'" data-iauth="resend_btn">Renvoyer l'email</button>
    </form>
  </div>
  <div class="link"><a href="/login" data-iauth="back_login">Retour à la connexion</a></div>
  {% else %}
  <div class="card">
    <div class="ctitle" data-iauth="reg_title">CRÉER UN COMPTE</div>
    {% if error %}<div class="err" style="display:block">{{ error }}</div>{% endif %}
    <form method="POST" action="/register">
      <label class="flbl" for="em" data-iauth="lbl_email_pro">Email professionnel</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-iauth="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" data-iauth="lbl_pw_min" placeholder="8 caractères minimum" autocomplete="new-password" required minlength="8">
      <label class="flbl" for="pw2" data-iauth="lbl_pw_conf">Confirmer le mot de passe</label>
      <input class="fi" type="password" id="pw2" name="password2" placeholder="••••••••" autocomplete="new-password" required>
      <button type="submit" class="btn" data-iauth="btn_reg">Créer mon compte</button>
    </form>
  </div>
  <div class="link" data-iauth="link_haveac">Déjà un compte ? <a href="/login">Se connecter</a></div>
  {% endif %}
</div>
</body></html>"""

ADMIN_HTML = """<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilar Admin</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#050d1a;color:#e2e8f0;min-height:100vh;padding:0 0 60px}
nav{background:#080f1e;border-bottom:1px solid #1a2a45;padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.logo{font-size:16px;font-weight:900;letter-spacing:3px;color:#14b8a6}
.nav-right{display:flex;gap:10px;align-items:center}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.kc{background:#0c1526;border:1px solid #1a2a45;border-radius:10px;padding:18px 20px}
.kv{font-size:28px;font-weight:800;letter-spacing:-1px}
.kl{font-size:11px;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-top:4px}
.card{background:#0c1526;border:1px solid #1a2a45;border-radius:12px;padding:20px;margin-bottom:20px}
.ctitle{font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:16px}
.search-bar{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.search-bar input{flex:1;min-width:200px;background:#080f1e;border:1px solid #1a2a45;color:#e2e8f0;padding:9px 14px;border-radius:6px;font-size:12px}
.search-bar select{background:#080f1e;border:1px solid #1a2a45;color:#e2e8f0;padding:9px 12px;border-radius:6px;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 10px;color:#64748b;font-size:10px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1a2a45}
td{padding:10px 10px;border-bottom:1px solid #0f1a2e;vertical-align:middle}
tr:last-child td{border:none}
tr:hover td{background:rgba(255,255,255,.015)}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.b-free{background:rgba(100,116,139,.15);color:#94a3b8}
.b-starter{background:rgba(13,148,136,.15);color:#14b8a6}
.b-pro{background:rgba(124,58,237,.15);color:#a78bfa}
.b-admin{background:rgba(245,158,11,.15);color:#fbbf24}
.b-ok{background:rgba(5,150,105,.15);color:#34d399}
.b-warn{background:rgba(220,38,38,.12);color:#f87171}
.b-exp{background:rgba(239,68,68,.12);color:#f87171}
.btn{padding:6px 14px;border:none;border-radius:5px;font-size:11px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block;letter-spacing:.5px}
.btn-teal{background:#0d9488;color:#fff}
.btn-teal:hover{background:#14b8a6}
.btn-ghost{background:transparent;border:1px solid #1e3050;color:#64748b}
.btn-ghost:hover{border-color:#0d9488;color:#14b8a6}
.btn-red{background:transparent;border:1px solid rgba(220,38,38,.3);color:#f87171}
.btn-red:hover{background:rgba(220,38,38,.1)}

/* Modal */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center;padding:16px}
.overlay.open{display:flex}
.modal{background:#0c1526;border:1px solid #1a2a45;border-radius:14px;padding:28px;width:100%;max-width:460px;max-height:90vh;overflow-y:auto}
.modal h3{font-size:14px;font-weight:700;margin-bottom:20px;color:#e2e8f0}
.fi{width:100%;background:#080f1e;border:1px solid #1a2a45;color:#e2e8f0;padding:10px 14px;border-radius:6px;font-size:12px;margin-bottom:12px}
.fi:focus{outline:none;border-color:#0d9488}
label{font-size:11px;color:#64748b;display:block;margin-bottom:4px;letter-spacing:.5px}
.plan-btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.plan-opt{padding:10px;border:2px solid #1a2a45;border-radius:8px;text-align:center;cursor:pointer;font-size:12px;font-weight:700;transition:all .15s}
.plan-opt.selected-free{border-color:#64748b;background:rgba(100,116,139,.1);color:#94a3b8}
.plan-opt.selected-starter{border-color:#0d9488;background:rgba(13,148,136,.1);color:#14b8a6}
.plan-opt.selected-pro{border-color:#7c3aed;background:rgba(124,58,237,.1);color:#a78bfa}
.plan-opt:hover{border-color:#0d9488}
.expires-wrap{display:flex;gap:8px;margin-bottom:12px}
.expires-wrap button{padding:7px 12px;background:#080f1e;border:1px solid #1a2a45;color:#94a3b8;border-radius:6px;font-size:11px;cursor:pointer;white-space:nowrap}
.expires-wrap button:hover{border-color:#0d9488;color:#14b8a6}
.msg{font-size:11px;padding:8px 12px;border-radius:6px;margin-bottom:12px;display:none}
.msg.ok{background:rgba(5,150,105,.1);border:1px solid rgba(5,150,105,.2);color:#34d399;display:block}
.msg.err{background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.2);color:#f87171;display:block}
@media(max-width:600px){
  .kv{font-size:22px}
  .search-bar{flex-direction:column}
  table{display:block;overflow-x:auto}
}
/* Terminal */
.term-wrap{background:#020910;border:1px solid #1a2a45;border-radius:10px;padding:0;overflow:hidden;margin-bottom:20px}
.term-header{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;background:#050d1a;border-bottom:1px solid #1a2a45}
.term-dots{display:flex;gap:6px}
.term-dots span{width:10px;height:10px;border-radius:50%}
.term-output{font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.7;padding:14px 16px;min-height:200px;max-height:420px;overflow-y:auto;color:#94a3b8;white-space:pre-wrap;word-break:break-all}
.term-output::-webkit-scrollbar{width:4px}
.term-output::-webkit-scrollbar-thumb{background:#1a2a45;border-radius:2px}
.term-line-ok{color:#34d399}
.term-line-err{color:#f87171}
.term-line-cmd{color:#14b8a6;font-weight:600}
.term-input-row{display:flex;align-items:center;gap:8px;padding:10px 16px;border-top:1px solid #1a2a45;background:#020910}
.term-prompt{color:#14b8a6;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;white-space:nowrap}
.term-input{flex:1;background:transparent;border:none;color:#e2e8f0;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;outline:none}
.term-run-btn{padding:5px 14px;background:#0d9488;border:none;border-radius:5px;color:#fff;font-size:11px;font-weight:700;cursor:pointer;white-space:nowrap}
.term-run-btn:hover{background:#14b8a6}
.term-run-btn:disabled{opacity:.4;cursor:not-allowed}
</style>
</head>
<body>
<nav>
  <div class="logo">PILAR ADMIN</div>
  <div class="nav-right">
    <a href="/monitor" class="btn btn-ghost">App</a>
    <a href="/logout" class="btn btn-ghost">Déconnexion</a>
  </div>
</nav>
<div class="wrap">

<!-- KPIs -->
<div class="kgrid">
  <div class="kc"><div class="kv" style="color:#14b8a6">{{ total_users }}</div><div class="kl">Utilisateurs</div></div>
  <div class="kc"><div class="kv" style="color:#a78bfa">{{ paid_users }}</div><div class="kl">Payants</div></div>
  <div class="kc"><div class="kv" style="color:#34d399">${{ mrr }}</div><div class="kl">MRR estimé</div></div>
  <div class="kc"><div class="kv">{{ total_analyses }}</div><div class="kl">Analyses</div></div>
  <div class="kc"><div class="kv" style="color:#f59e0b">{{ expiring_soon }}</div><div class="kl">Expirent &lt;7j</div></div>
</div>

<!-- FILTRES + TABLEAU -->
<div class="card">
  <div class="ctitle">Gestion abonnements</div>
  <div class="search-bar">
    <input type="text" id="searchInput" placeholder="Rechercher par email..." oninput="filterTable()">
    <select id="planFilter" onchange="filterTable()">
      <option value="">Tous les plans</option>
      <option value="free">Free</option>
      <option value="starter">Starter</option>
      <option value="pro">Pro</option>
    </select>
  </div>
  <table id="usersTable">
    <thead>
      <tr>
        <th>Email</th>
        <th>Plan</th>
        <th>Expiration</th>
        <th>Note paiement</th>
        <th>Analyses</th>
        <th>Inscrit</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody>
    {% for u in users %}
    <tr data-email="{{ u.email|lower }}" data-plan="{{ u.plan }}">
      <td>
        <div style="font-weight:600;color:#e2e8f0">{{ u.email }}</div>
        {% if u.is_admin %}<span class="badge b-admin">admin</span>{% endif %}
      </td>
      <td>
        <span class="badge b-{{ u.plan }}">{{ u.plan }}</span>
      </td>
      <td>
        {% if u.plan_expires_at %}
          {% if u.plan_expires_at < now %}
            <span class="badge b-exp">Expiré {{ u.plan_expires_at.strftime('%d/%m/%y') }}</span>
          {% elif (u.plan_expires_at - now).days < 7 %}
            <span class="badge b-warn">{{ u.plan_expires_at.strftime('%d/%m/%y') }}</span>
          {% else %}
            <span style="font-size:11px;color:#94a3b8">{{ u.plan_expires_at.strftime('%d/%m/%Y') }}</span>
          {% endif %}
        {% else %}
          <span style="color:#334155;font-size:11px">—</span>
        {% endif %}
      </td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#64748b;font-size:11px">
        {{ u.plan_note or '—' }}
      </td>
      <td style="color:#94a3b8">{{ u.analysis_count }}</td>
      <td style="color:#64748b;font-size:11px;white-space:nowrap">{{ u.created_at.strftime('%d/%m/%Y') }}</td>
      <td>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button class="btn btn-teal manage-btn"
            data-uid="{{ u.id }}"
            data-email="{{ u.email|e }}"
            data-plan="{{ u.plan }}"
            data-expires="{{ u.plan_expires_at.strftime('%Y-%m-%d') if u.plan_expires_at else '' }}"
            data-note="{{ u.plan_note|e if u.plan_note else '' }}">Gérer</button>
          <a href="/admin/impersonate/{{ u.id }}" class="btn btn-ghost">Voir</a>
        </div>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<!-- TERMINAL ADMIN -->
<div class="card">
  <div class="ctitle">Terminal</div>
  <div class="term-wrap">
    <div class="term-header">
      <div class="term-dots">
        <span style="background:#f87171"></span>
        <span style="background:#fbbf24"></span>
        <span style="background:#34d399"></span>
      </div>
      <span style="font-size:10px;color:#334155;letter-spacing:1px">PILAR SHELL — ADMIN ONLY</span>
      <button onclick="termClear()" style="background:transparent;border:1px solid #1e3050;color:#64748b;padding:3px 10px;border-radius:4px;font-size:10px;cursor:pointer">Effacer</button>
    </div>
    <div class="term-output" id="termOut"><span style="color:#334155">Prêt. Tapez une commande et appuyez sur Entrée.</span>
</div>
    <div class="term-input-row">
      <span class="term-prompt">pilar$&nbsp;</span>
      <input class="term-input" id="termIn" type="text" placeholder="ls, python --version, pip list..." autocomplete="off" spellcheck="false" onkeydown="termKey(event)">
      <button class="term-run-btn" id="termBtn" onclick="termRun()">Exécuter</button>
    </div>
  </div>
</div>

</div><!-- /wrap -->

<!-- MODAL GESTION ABONNEMENT -->
<div class="overlay" id="overlay" onclick="closeIfOut(event)">
  <div class="modal">
    <h3 id="modalTitle">Gérer l'abonnement</h3>
    <div class="msg" id="modalMsg"></div>

    <label>Plan</label>
    <div class="plan-btns">
      <div class="plan-opt" id="opt-free" onclick="selectPlan('free')">Free</div>
      <div class="plan-opt" id="opt-starter" onclick="selectPlan('starter')">Starter<br><span style="font-size:9px;font-weight:400">$99/mo</span></div>
      <div class="plan-opt" id="opt-pro" onclick="selectPlan('pro')">Pro<br><span style="font-size:9px;font-weight:400">$299/mo</span></div>
    </div>

    <label>Date d'expiration</label>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
      <input type="date" class="fi" id="expiresInput" style="flex:1;margin:0">
      <button onclick="addMonths(1)" style="padding:7px 10px;background:#080f1e;border:1px solid #1a2a45;color:#94a3b8;border-radius:6px;font-size:11px;cursor:pointer">+1 mois</button>
      <button onclick="addMonths(3)" style="padding:7px 10px;background:#080f1e;border:1px solid #1a2a45;color:#94a3b8;border-radius:6px;font-size:11px;cursor:pointer">+3 mois</button>
      <button onclick="addMonths(12)" style="padding:7px 10px;background:#080f1e;border:1px solid #1a2a45;color:#94a3b8;border-radius:6px;font-size:11px;cursor:pointer">+1 an</button>
      <button onclick="document.getElementById('expiresInput').value=''" style="padding:7px 10px;background:#080f1e;border:1px solid #1a2a45;color:#94a3b8;border-radius:6px;font-size:11px;cursor:pointer">Effacer</button>
    </div>

    <label>Note paiement (référence virement, etc.)</label>
    <input type="text" class="fi" id="noteInput" placeholder="ex: Virement BNP ref 2026-031 — Société ABC">

    <div style="display:flex;gap:10px;margin-top:8px">
      <button onclick="savePlan()" class="btn btn-teal" id="saveBtn" style="flex:1;padding:12px">Enregistrer</button>
      <button onclick="closeModal()" class="btn btn-ghost" style="padding:12px 20px">Annuler</button>
    </div>
  </div>
</div>

<script>
let _currentUid = null;
let _currentPlan = 'free';

document.addEventListener('click', function(e) {
  const btn = e.target.closest('.manage-btn');
  if (btn) openModal(btn.dataset.uid, btn.dataset.email, btn.dataset.plan, btn.dataset.expires, btn.dataset.note);
});

function openModal(uid, email, plan, expires, note) {
  _currentUid = uid;
  _currentPlan = plan;
  document.getElementById('modalTitle').textContent = 'Abonnement — ' + email;
  document.getElementById('expiresInput').value = expires;
  document.getElementById('noteInput').value = note || '';
  document.getElementById('modalMsg').className = 'msg';
  document.getElementById('modalMsg').textContent = '';
  selectPlan(plan);
  document.getElementById('overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('overlay').classList.remove('open');
}

function closeIfOut(e) {
  if (e.target === document.getElementById('overlay')) closeModal();
}

function selectPlan(plan) {
  _currentPlan = plan;
  ['free','starter','pro'].forEach(p => {
    const el = document.getElementById('opt-' + p);
    el.className = 'plan-opt';
    if (p === plan) el.classList.add('selected-' + plan);
  });
}

function addMonths(n) {
  const inp = document.getElementById('expiresInput');
  const base = inp.value ? new Date(inp.value) : new Date();
  base.setMonth(base.getMonth() + n);
  inp.value = base.toISOString().slice(0, 10);
}

async function savePlan() {
  const btn = document.getElementById('saveBtn');
  btn.disabled = true;
  const msg = document.getElementById('modalMsg');
  const expires = document.getElementById('expiresInput').value;
  const note = document.getElementById('noteInput').value;
  try {
    const r = await fetch('/admin/set_plan/' + _currentUid, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({plan: _currentPlan, expires_at: expires, note: note})
    });
    const d = await r.json();
    if (d.ok) {
      msg.className = 'msg ok';
      msg.textContent = 'Abonnement mis à jour.';
      setTimeout(() => location.reload(), 1000);
    } else {
      msg.className = 'msg err';
      msg.textContent = d.error || 'Erreur';
      btn.disabled = false;
    }
  } catch(e) {
    msg.className = 'msg err';
    msg.textContent = 'Erreur réseau';
    btn.disabled = false;
  }
}

function filterTable() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  const pf = document.getElementById('planFilter').value;
  document.querySelectorAll('#usersTable tbody tr').forEach(row => {
    const email = row.dataset.email || '';
    const plan = row.dataset.plan || '';
    const matchEmail = !q || email.includes(q);
    const matchPlan = !pf || plan === pf;
    row.style.display = matchEmail && matchPlan ? '' : 'none';
  });
}

/* ── Terminal ── */
let _termHistory = [];
let _termHistIdx = -1;

function termKey(e) {
  if (e.key === 'Enter') { termRun(); return; }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (_termHistory.length === 0) return;
    _termHistIdx = Math.min(_termHistIdx + 1, _termHistory.length - 1);
    document.getElementById('termIn').value = _termHistory[_termHistIdx];
  }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (_termHistIdx <= 0) { _termHistIdx = -1; document.getElementById('termIn').value = ''; return; }
    _termHistIdx--;
    document.getElementById('termIn').value = _termHistory[_termHistIdx];
  }
}

async function termRun() {
  const inp = document.getElementById('termIn');
  const cmd = inp.value.trim();
  if (!cmd) return;
  _termHistory.unshift(cmd);
  _termHistIdx = -1;
  inp.value = '';
  const out = document.getElementById('termOut');
  const btn = document.getElementById('termBtn');
  btn.disabled = true;
  out.innerHTML += '<span class="term-line-cmd">$ ' + cmd.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\n';
  out.scrollTop = out.scrollHeight;
  try {
    const r = await fetch('/admin/terminal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cmd: cmd})
    });
    const d = await r.json();
    const txt = (d.output || '').replace(/</g,'&lt;').replace(/>/g,'&gt;') || '(aucune sortie)';
    const cls = d.code === 0 ? 'term-line-ok' : 'term-line-err';
    out.innerHTML += '<span class="' + cls + '">' + txt + '</span>\n';
  } catch(e) {
    out.innerHTML += '<span class="term-line-err">Erreur réseau</span>\n';
  }
  btn.disabled = false;
  out.scrollTop = out.scrollHeight;
  inp.focus();
}

function termClear() {
  document.getElementById('termOut').innerHTML = '<span style="color:#334155">Terminal effacé.</span>\n';
}
</script>
</body></html>"""

# ── CSS & HEAD ────────────────────────────────────────────────────────────────
_HEAD = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0e1118">
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
:root{--bg:#07090f;--surface:#0e1118;--surface2:#141820;--border:#1e2433;--border2:#252d3d;--teal:#0d9488;--teal-light:#14b8a6;--teal-dim:rgba(13,148,136,0.08);--red:#dc2626;--red-dim:rgba(220,38,38,0.08);--green:#059669;--green-dim:rgba(5,150,105,0.08);--amber:#d97706;--purple:#7c3aed;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;--nav-h:60px;}
html,body{height:100%;overflow:hidden;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;}
header{height:52px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;padding:0 20px;background:var(--surface);flex-shrink:0;}
.logo{font-size:13px;font-weight:700;letter-spacing:4px;color:var(--teal-light);text-transform:uppercase;}
.hd{width:1px;height:18px;background:var(--border2);}
.hsub{font-size:10px;letter-spacing:1.2px;color:var(--text3);text-transform:uppercase;}
.hright{margin-left:auto;display:flex;gap:6px;align-items:center;}
.bottom-nav{height:var(--nav-h);border-top:1px solid var(--border);background:var(--surface);display:flex;align-items:stretch;flex-shrink:0;}
.ni{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;text-decoration:none;color:var(--text3);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;border:none;background:none;cursor:pointer;padding:8px 0;}
.ni.on{color:var(--teal-light);}
.ni svg{width:20px;height:20px;stroke-width:1.8;}
.lang-toggle{max-width:42px;color:var(--text3);}
.lang-toggle:hover span{color:var(--teal-light);}
.page{flex:1;overflow-y:auto;overflow-x:hidden;}
.page::-webkit-scrollbar{width:0;}
.pad{padding:16px;padding-bottom:80px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:12px;}
.ctitle{font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:12px;}
.rh{display:flex;align-items:center;justify-content:space-between;padding:18px;border-radius:8px;border:1px solid var(--border);background:var(--surface);margin-bottom:12px;}
.rh.ok{border-color:var(--green);background:var(--green-dim);}
.rh.alert{border-color:var(--red);background:var(--red-dim);}
.rh.amber{border-color:var(--amber);background:rgba(217,119,6,0.06);}
.sb{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;}
.sb.ok{background:rgba(5,150,105,0.15);color:var(--green);}
.sb.alert{background:rgba(220,38,38,0.15);color:var(--red);}
.sb.amber{background:rgba(217,119,6,0.12);color:var(--amber);}
.dot{width:6px;height:6px;border-radius:50%;}
.dot.ok{background:var(--green);}.dot.alert{background:var(--red);animation:blink 1.2s infinite;}.dot.amber{background:var(--amber);}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
.rnum{font-size:48px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;}
.rnum.ok{color:var(--green);}.rnum.alert{color:var(--red);}.rnum.amber{color:var(--amber);}
.runit{font-size:20px;color:var(--text3);}
.rlbl{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;text-align:right;margin-top:3px;}
.tgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px;}
.tbtn{padding:10px 4px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text3);font-size:11px;cursor:pointer;text-align:center;transition:all 0.15s;}
.tbtn.on{border-color:var(--teal);background:var(--teal-dim);color:var(--teal-light);font-weight:600;}
.sensor{margin-bottom:16px;}
.srow{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.sname{font-size:12px;color:var(--text2);}
.vwrap{display:flex;align-items:center;gap:5px;}
.vi{width:72px;padding:4px 8px;background:var(--surface2);border:1px solid var(--border2);border-radius:4px;color:var(--text);font-size:15px;font-weight:600;text-align:right;outline:none;-webkit-appearance:none;}
.vi:focus{border-color:var(--teal);}
.vu{font-size:10px;color:var(--text3);}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:var(--border2);border-radius:2px;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--teal);cursor:pointer;border:2px solid var(--bg);}
.rl{display:flex;justify-content:space-between;font-size:9px;color:var(--text3);margin-top:2px;}
.btn{width:100%;padding:14px;background:var(--teal);color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:background 0.15s;margin-top:8px;}
.btn:disabled{background:var(--border2);color:var(--text3);cursor:not-allowed;}
.flbl{font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:7px;display:block;}
.fi{width:100%;padding:10px 12px;background:var(--surface2);border:1px solid var(--border2);border-radius:6px;color:var(--text);font-size:13px;outline:none;transition:border-color 0.15s;}
.fi:focus{border-color:var(--teal);}
.fi::placeholder{color:var(--text3);}
.zrow{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface2);border-radius:6px;margin-bottom:6px;}
.zname{font-size:11px;color:var(--text2);flex:1;}
.zbw{width:80px;height:2px;background:var(--border2);border-radius:1px;}
.zbf{height:100%;border-radius:1px;background:var(--red);}
.zp{font-size:12px;font-weight:700;color:var(--amber);min-width:34px;text-align:right;}
.kgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.kc{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;}
.kv{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums;}
.kv.ok{color:var(--green);}.kv.alert{color:var(--red);}.kv.amber{color:var(--amber);}.kv.purple{color:var(--purple);}
.kl{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;margin-top:3px;}
.tw{overflow-x:auto;border-radius:6px;background:var(--surface);border:1px solid var(--border);}
table{width:100%;border-collapse:collapse;font-size:11px;min-width:480px;}
th{text-align:left;padding:10px 12px;color:var(--text3);font-size:9px;font-weight:500;letter-spacing:1px;border-bottom:1px solid var(--border);text-transform:uppercase;white-space:nowrap;}
td{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--text2);white-space:nowrap;}
tr:last-child td{border-bottom:none;}
.badge{padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;}
.badge.ok{background:rgba(5,150,105,0.12);color:var(--green);}
.badge.alert{background:rgba(220,38,38,0.12);color:var(--red);}
.mb{padding:2px 8px;border-radius:3px;font-size:10px;background:rgba(13,148,136,0.12);color:var(--teal-light);}
.cw{display:flex;flex-direction:column;height:100%;overflow:hidden;}
.cm{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}
.cm::-webkit-scrollbar{width:0;}
.msg{display:flex;flex-direction:column;gap:3px;max-width:85%;}
.msg.user{align-self:flex-end;align-items:flex-end;}
.msg.bot{align-self:flex-start;align-items:flex-start;}
.ms{font-size:8px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;}
.mb2{padding:10px 14px;border-radius:8px;font-size:13px;line-height:1.65;}
.msg.user .mb2{background:var(--teal);color:#fff;border-radius:8px 8px 2px 8px;}
.msg.bot .mb2{background:var(--surface2);border:1px solid var(--border);color:var(--text2);border-radius:8px 8px 8px 2px;}
.typing{color:var(--text3);font-style:italic;}
.cia{padding:10px 14px;border-top:1px solid var(--border);display:flex;gap:8px;background:var(--surface);flex-shrink:0;}
.cta{flex:1;padding:10px 12px;background:var(--surface2);border:1px solid var(--border2);border-radius:8px;color:var(--text);font-size:13px;outline:none;resize:none;font-family:inherit;max-height:100px;line-height:1.5;transition:border-color 0.15s;}
.cta:focus{border-color:var(--teal);}
.cta::placeholder{color:var(--text3);}
.bsend{padding:10px 16px;background:var(--teal);color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;align-self:flex-end;transition:background 0.15s;flex-shrink:0;}
.bsend:disabled{background:var(--border2);color:var(--text3);cursor:not-allowed;}
.ab{padding:10px 14px;background:var(--teal-dim);border:1px solid var(--teal);border-radius:6px;font-size:11px;color:var(--teal-light);margin-bottom:10px;display:none;}
.nb{padding:5px 11px;background:transparent;border:1px solid var(--border2);border-radius:4px;color:var(--text3);font-size:10px;cursor:pointer;transition:all 0.15s;white-space:nowrap;}
.nb.on{border-color:var(--green);color:var(--green);}
.idle{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;gap:8px;color:var(--text3);text-align:center;}
.idle .l1{font-size:13px;}.idle .l2{font-size:11px;}
.lcard{flex:1;padding:14px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;cursor:pointer;text-align:center;transition:all 0.15s;}
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
.lz-wrap{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:12px;}
.lz-empty{display:flex;flex-direction:column;align-items:center;padding:36px 24px;gap:14px;}
.lz-cta{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:16px;background:var(--teal);color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:background 0.15s;}
.lz-cta:hover{background:#0f766e;}
.lz-hint{font-size:10px;color:var(--text3);letter-spacing:0.5px;text-align:center;line-height:1.7;}
.lz-conn{padding:16px;}
.lz-ch{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.lz-dot{width:8px;height:8px;border-radius:50%;background:var(--green);flex-shrink:0;animation:blink 1.5s infinite;}
.lz-fname{font-size:12px;color:var(--teal-light);font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.lz-disc{padding:5px 12px;background:transparent;border:1px solid var(--border2);border-radius:4px;color:var(--text3);font-size:10px;cursor:pointer;transition:border-color 0.15s;white-space:nowrap;}
.lz-disc:hover{border-color:var(--red);color:var(--red);}
.lz-sg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;}
.lz-si{background:var(--surface2);border-radius:8px;padding:12px 8px;text-align:center;}
.lz-sv{display:block;font-size:26px;font-weight:800;font-variant-numeric:tabular-nums;color:var(--text);}
.lz-sl{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px;display:block;}
.lz-si.alert .lz-sv{color:var(--red);}
.lz-si.ok .lz-sv{color:var(--green);}
.lz-foot{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;}
.lz-ck{font-size:10px;color:var(--text3);flex:1;}
.lz-sel{background:var(--surface2);border:1px solid var(--border2);border-radius:4px;color:var(--text2);font-size:10px;padding:4px 8px;outline:none;}
.man-hdr{display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none;margin-bottom:0;}
.man-chv{width:14px;height:14px;stroke:var(--text3);fill:none;flex-shrink:0;transition:transform 0.2s;}
</style>
<script>
const T={
fr:{nav_monitor:'Monitor',nav_twin:'Twin',nav_history:'Historique',nav_account:'Compte',nav_settings:'Réglages',
page_monitor:'Monitor',page_twin:'Jumeau Numérique',page_history:'Historique',page_account:'Compte',page_settings:'Réglages',
idle_l1:'Aucune analyse',idle_l2:'Configurez ci-dessous et lancez',
machine_class:'Classe machine',sensor_params:'Paramètres capteurs',
air_temp:'Température air',proc_temp:'Température process',rot_speed:'Vitesse rotation',torque:'Couple',tool_wear:'Usure outil',
run_btn:"Lancer l'analyse",zone_title:'Analyse zones de panne',
status_ok:'Fonctionnement normal',status_alert:'Anomalie détectée',failure_prob:'Prob. panne',
u_temp:'°C',u_speed:'tr/min',u_torque:'N·m',u_wear:'h',
r_ta_min:'21.9°C',r_ta_max:'31.9°C',r_tp_min:'31.9°C',r_tp_max:'41.9°C',
r_v_min:'1000',r_v_max:'3000',r_c_min:'3',r_c_max:'80N·m',r_u_min:'0',r_u_max:'4.17h',
twin_loading:'Chargement simulation...',twin_no_data:'Aucune donnée',twin_no_data2:"Lancez d'abord une analyse dans Monitor",twin_go:'Aller à Monitor',
twin_healthy:'Système sain',twin_failure:'Panne dans ~',twin_trend:'Tendance\u00a0:',twin_cur_risk:'Risque actuel',twin_avg:'Risque moyen',twin_anom:"Taux d'anomalie",
twin_c_risk:'Risque — Historique + Simulation 24h',twin_c_wear:'Projection usure outil',twin_c_temp:'Température process',twin_c_sim:'Simulateur de scénario',
twin_speed:'Vitesse (tr/min)',twin_torque:'Couple (N·m)',twin_wear:'Usure outil (h)',twin_airtemp:'Temp. air (°C)',twin_sim:'Simuler',twin_sim_r:'Risque simulé',
hist_total:'Total',hist_anom:'Anomalies',hist_avg:'Risque moy.',hist_alerts:'Alertes envoyées',
hist_time:'Heure',hist_class:'Classe',hist_risk:'Risque',hist_status:'Statut',hist_zones:'Zones',hist_alert:'Alerte',
hist_anomaly:'Anomalie',hist_ok:'OK',hist_sent:'Envoyé',
set_email:"Email d'alerte",set_email_lbl:'Adresse destinataire',set_email_ph:'maintenance@entreprise.com',set_email_btn:'Enregistrer',set_saved:'Enregistré',
set_notif:'Notifications navigateur',set_notif_desc:'Recevez des alertes quand le risque dépasse 50%.',set_notif_btn:'Activer les notifications',set_notif_on:'Notifications activées',set_notif_blocked:'Bloqué — Activez dans les réglages',
set_sys:'Infos système',set_version:'Version',set_aimodel:'Modèle IA',set_db:'Base de données',set_lang:'Langue',
acc_guest_title:'Mode invité',acc_guest_desc:"Connectez-vous pour sauvegarder vos données,<br>rejoindre une équipe et accéder à la collaboration.",
acc_signin:'Se connecter',acc_register:'Créer un compte',acc_card_title:'Compte',acc_signout:'Déconnexion',
acc_team_title:'Équipe',acc_no_team_desc:"Créez une équipe pour collaborer. Jusqu'à 2 responsables peuvent gérer l'équipe.",
acc_create_ph:"Nom de l'équipe (optionnel)",acc_create_btn:'Créer une équipe',
acc_role_leader:'Responsable',acc_role_member:'Membre',acc_you:'(vous)',
acc_promote:'Promouvoir',acc_kick:'Retirer',
acc_add_title:'Ajouter un membre',acc_add_ph:'email@entreprise.com',acc_add_btn:'Ajouter',acc_added:'Membre ajouté !',
acc_members:' membre(s)',acc_leave:"Quitter l'équipe",
nav_import:'Import',nav_assistant:'Assistant',
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
live_hint:'CSV · noms de colonnes libres · conversion auto',manual_title:'Saisie manuelle',
partial_analysis:'Analyse partielle',imputed:'estimés',discovered_param:'Paramètre découvert',
page_adapter:'Adaptateur CSV',adp_upload:'Importer CSV',adp_hint:'Tout délimiteur · noms libres · conversion unités incluse',
adp_change:'Changer',adp_preview:'Aperçu (5 lignes)',adp_download:'Convertir & Télécharger',
adp_no_map:'Aucun champ mappé',adp_source:'Colonne source',adp_samples:'Exemples',adp_field:'Champ Pilar',adp_unit:'Unité',
adp_ignore:'(Ignorer)',adp_desc:'Convertissez n\u2019importe quel CSV au format Pilar',adp_save:'Sauvegarder dans Pilar',adp_my_files:'Mes fichiers',adp_refresh:'Actualiser',adp_saved_ok:'Fichier sauvegardé',adp_no_files:'Aucun fichier sauvegardé',adp_load:'Charger',adp_delete:'Supprimer',
page_tutorial:'Import & Analyse',tut_csv_desc:"Votre CSV doit contenir ces colonnes (l'ordre n'a pas d'importance) :",tut_click_csv:'Cliquez pour sélectionner un fichier CSV',tut_no_file_sel:'Aucun fichier sélectionné',tut_progress:'Progression',tut_check_every:'Vérifier toutes les',tut_row:'Ligne',
set_domain:'Domaine',set_nav_title:'Navigation',set_twin_nav:'Jumeau Numérique',
ast_you:'Vous',ast_pilar:'Pilar IA',ast_error:'Erreur\u00a0: ',ast_net_error:'Erreur réseau. Réessayez.'},
en:{nav_monitor:'Monitor',nav_twin:'Twin',nav_history:'History',nav_account:'Account',nav_settings:'Settings',
page_monitor:'Monitor',page_twin:'Digital Twin',page_history:'History',page_account:'Account',page_settings:'Settings',
idle_l1:'No analysis yet',idle_l2:'Configure below and run',
machine_class:'Machine class',sensor_params:'Sensor parameters',
air_temp:'Air temperature',proc_temp:'Process temperature',rot_speed:'Rotational speed',torque:'Torque',tool_wear:'Tool wear',
run_btn:'Run Analysis',zone_title:'Failure zone analysis',
status_ok:'Normal Operation',status_alert:'Anomaly Detected',failure_prob:'Failure prob.',
u_temp:'K',u_speed:'rpm',u_torque:'Nm',u_wear:'min',
r_ta_min:'295K',r_ta_max:'305K',r_tp_min:'305K',r_tp_max:'315K',
r_v_min:'1000',r_v_max:'3000',r_c_min:'3',r_c_max:'80Nm',r_u_min:'0',r_u_max:'250',
twin_loading:'Loading simulation...',twin_no_data:'No data yet',twin_no_data2:'Run an analysis on Monitor first',twin_go:'Go to Monitor',
twin_healthy:'System Healthy',twin_failure:'Failure in ~',twin_trend:'Trend:',twin_cur_risk:'Current risk',twin_avg:'Avg risk',twin_anom:'Anomaly rate',
twin_c_risk:'Risk \u2014 History + 24h Simulation',twin_c_wear:'Tool wear projection',twin_c_temp:'Process temperature',twin_c_sim:'Scenario Simulator',
twin_speed:'Speed (rpm)',twin_torque:'Torque (Nm)',twin_wear:'Tool wear (min)',twin_airtemp:'Air temp (K)',twin_sim:'Simulate',twin_sim_r:'Simulated risk',
hist_total:'Total',hist_anom:'Anomalies',hist_avg:'Avg risk',hist_alerts:'Alerts sent',
hist_time:'Time',hist_class:'Class',hist_risk:'Risk',hist_status:'Status',hist_zones:'Zones',hist_alert:'Alert',
hist_anomaly:'Anomaly',hist_ok:'OK',hist_sent:'Sent',
set_email:'Alert email',set_email_lbl:'Recipient address',set_email_ph:'maintenance@company.com',set_email_btn:'Save Email',set_saved:'Saved',
set_notif:'Browser notifications',set_notif_desc:'Receive alerts when failure risk exceeds 50%.',set_notif_btn:'Enable Notifications',set_notif_on:'Notifications Enabled',set_notif_blocked:'Blocked \u2014 Enable in Browser Settings',
set_sys:'System info',set_version:'Version',set_aimodel:'AI Model',set_db:'Database',set_lang:'Language',
acc_guest_title:'Guest Mode',acc_guest_desc:'Sign in to save your data,<br>join a team and access collaboration.',
acc_signin:'Sign In',acc_register:'Create Account',acc_card_title:'Account',acc_signout:'Sign Out',
acc_team_title:'Team',acc_no_team_desc:'Create a team to collaborate with colleagues. Up to 2 leaders can manage the team.',
acc_create_ph:'Team name (optional)',acc_create_btn:'Create Team',
acc_role_leader:'Leader',acc_role_member:'Member',acc_you:'(you)',
acc_promote:'Promote',acc_kick:'Remove',
acc_add_title:'Add Member',acc_add_ph:'email@company.com',acc_add_btn:'Add',acc_added:'Member added!',
acc_members:' member(s)',acc_leave:'Leave Team',
nav_import:'Import',nav_assistant:'Assistant',
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
live_hint:'CSV · any column names · auto unit conversion',manual_title:'Manual Input',
partial_analysis:'Partial analysis',imputed:'estimated',discovered_param:'Discovered parameter',
page_adapter:'CSV Adapter',adp_upload:'Upload CSV',adp_hint:'Any delimiter · any column names · unit conversion included',
adp_change:'Change',adp_preview:'Preview (5 rows)',adp_download:'Convert & Download',
adp_no_map:'No fields mapped',adp_source:'Source column',adp_samples:'Samples',adp_field:'Pilar field',adp_unit:'Unit',
adp_ignore:'(Ignore)',adp_desc:'Convert any CSV to Pilar format',adp_save:'Save to Pilar',adp_my_files:'My files',adp_refresh:'Refresh',adp_saved_ok:'File saved',adp_no_files:'No saved files',adp_load:'Load',adp_delete:'Delete',
page_tutorial:'Import & Run',tut_csv_desc:'Your CSV must contain these columns (order does not matter):',tut_click_csv:'Click to select a CSV file',tut_no_file_sel:'No file selected',tut_progress:'Progress',tut_check_every:'Check every',tut_row:'Row',
set_domain:'Domain',set_nav_title:'Navigation',set_twin_nav:'Digital Twin',
ast_you:'You',ast_pilar:'Pilar AI',ast_error:'Error: ',ast_net_error:'Network error. Please retry.'}
};
let LANG=localStorage.getItem('pilar_lang')||'en';
function t(k){return(T[LANG]&&T[LANG][k])||(T.en[k])||k;}
function setLang(l){LANG=l;localStorage.setItem('pilar_lang',l);applyLang();}
function applyLang(){
  document.querySelectorAll('[data-i18n]').forEach(function(el){
    var k=el.getAttribute('data-i18n');
    if(el.tagName==='INPUT'){el.placeholder=t(k);}else{el.textContent=t(k);}
  });
  document.querySelectorAll('[data-i18n-html]').forEach(function(el){el.innerHTML=t(el.getAttribute('data-i18n-html'));});
  document.querySelectorAll('.lcard').forEach(function(c){c.classList.toggle('lactive',c.dataset.lang===LANG);});
  document.querySelectorAll('td .badge.alert').forEach(function(el){el.textContent=t('hist_anomaly');});
  document.querySelectorAll('td .badge.ok').forEach(function(el){el.textContent=t('hist_ok');});
  document.querySelectorAll('td .mb').forEach(function(el){el.textContent=t('hist_sent');});
  if(document.getElementById('nta'))updateSensorUnits();
  document.querySelectorAll('.acc-role').forEach(function(el){el.textContent=t(el.dataset.role==='leader'?'acc_role_leader':'acc_role_member');});
  document.querySelectorAll('[data-i18n-count]').forEach(function(el){var n=el.dataset.icount||'0';el.textContent=n+t(el.getAttribute('data-i18n-count'));});
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
  var fr=LANG==='fr';
  var nta=document.getElementById('nta'),ntp=document.getElementById('ntp'),nu=document.getElementById('nu');
  if(!nta)return;
  var ta_raw=parseFloat(nta.dataset.raw)||300;
  var tp_raw=parseFloat(ntp.dataset.raw)||310;
  var u_raw=parseFloat(nu.dataset.raw)||100;
  var ta_min=fr?21.85:295,ta_max=fr?31.85:305;
  document.getElementById('sta').min=ta_min;document.getElementById('sta').max=ta_max;
  nta.min=ta_min;nta.max=ta_max;
  var ta_d=toDisplay(ta_raw,'temp');nta.value=ta_d;document.getElementById('sta').value=ta_d;
  var tp_min=fr?31.85:305,tp_max=fr?41.85:315;
  document.getElementById('stp').min=tp_min;document.getElementById('stp').max=tp_max;
  ntp.min=tp_min;ntp.max=tp_max;
  var tp_d=toDisplay(tp_raw,'temp');ntp.value=tp_d;document.getElementById('stp').value=tp_d;
  var u_max=fr?+(250/60).toFixed(3):250,u_step=fr?0.001:1;
  document.getElementById('su').min=0;document.getElementById('su').max=u_max;document.getElementById('su').step=u_step;
  nu.min=0;nu.max=u_max;
  var u_d=toDisplay(u_raw,'wear');nu.value=u_d;document.getElementById('su').value=u_d;
  var ids={'vu-ta':t('u_temp'),'vu-tp':t('u_temp'),'vu-v':t('u_speed'),'vu-c':t('u_torque'),'vu-u':t('u_wear'),
    'rl-ta-min':t('r_ta_min'),'rl-ta-max':t('r_ta_max'),'rl-tp-min':t('r_tp_min'),'rl-tp-max':t('r_tp_max'),
    'rl-v-min':t('r_v_min'),'rl-v-max':t('r_v_max'),'rl-c-min':t('r_c_min'),'rl-c-max':t('r_c_max'),
    'rl-u-min':t('r_u_min'),'rl-u-max':t('r_u_max')};
  Object.keys(ids).forEach(function(id){var e=document.getElementById(id);if(e)e.textContent=ids[id];});
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
var _PAGE_ORDER={'/':0,'/tutorial':1,'/history':2,'/account':3,'/settings':4};
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
var _CSV_FIELDS=['type','temp_air','temp_process','vitesse','couple','usure'];
var _CSV_OPT_FIELDS=['humidite','vibration','pression','courant','tension'];
var _CSV_PATS={
  type:['type','machine_type','product_type','machine','classe','class','category','categorie'],
  temp_air:['air_temp','temp_air','temperature_air','air_temperature','t_air','tair','ambient_temp','temp_ambiante','temp_ambiant'],
  temp_process:['process_temp','temp_process','temperature_process','process_temperature','t_process','tprocess','proc_temp'],
  vitesse:['speed','vitesse','rpm','rotational_speed','rotation_speed','speed_rpm','rot_speed'],
  couple:['torque','couple','torque_nm','couple_nm','moment'],
  usure:['wear','usure','tool_wear','wear_min','tool_wear_min','outil','usure_outil','wear_time']
};
var _CSV_OPT_PATS={
  humidite:['humidity','humidite','humid','rh','relative_humidity','moisture','humidite_relative','humidity_pct'],
  vibration:['vibration','vib','vibration_mm','vib_mms','vibration_mmps','accel','acceleration'],
  pression:['pressure','pression','pres','bar','press','pression_bar','pressure_bar'],
  courant:['current','courant','ampere','amp','current_a','courant_a','intensite'],
  tension:['voltage','tension','volt','v_in','tension_v','voltage_v','alimentation']
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
      if(field==='temp_air'||field==='temp_process'){
        if(orig.indexOf('celsius')!==-1||orig.indexOf('_c')!==-1)unit='C';
        else if(orig.indexOf('fahrenheit')!==-1||orig.indexOf('_f')!==-1)unit='F';
        else if(orig.indexOf('kelvin')!==-1||orig.indexOf('_k')!==-1)unit='K';
      }
      if(field==='usure'){
        if(orig.indexOf('hour')!==-1||orig.indexOf('heure')!==-1||orig.endsWith('_h'))unit='h';
        else if(orig.indexOf('min')!==-1)unit='min';
      }
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
    if(f==='type'){
      var sv=raw.toLowerCase();
      if(sv==='l'||sv==='low')row.type=0;
      else if(sv==='m'||sv==='medium'||sv==='med')row.type=1;
      else if(sv==='h'||sv==='high')row.type=2;
      else{var n=parseFloat(raw);if(isNaN(n)){row.type=null;continue;}row.type=Math.min(2,Math.max(0,Math.round(n)));}
    }else{
      var v=parseFloat(raw);if(isNaN(v)){row[f]=null;continue;}
      if(f==='temp_air'||f==='temp_process'){
        var u=m.unit||(v<200?'C':'K');
        if(u==='C')v+=273.15;else if(u==='F')v=(v-32)*5/9+273.15;
      }
      if(f==='usure'){var u=m.unit||(v<20?'h':'min');if(u==='h')v*=60;}
      row[f]=Math.round(v*100)/100;
    }
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

_NAV = """<nav class="bottom-nav">
<a href="/monitor" class="ni {m}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg><span data-i18n="nav_monitor">Monitor</span></a>
<a href="/tutorial" class="ni {tut}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg><span data-i18n="nav_import">Import</span></a>
<a href="/history" class="ni {h}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg><span data-i18n="nav_history">History</span></a>
<a href="/account" class="ni {a}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg><span data-i18n="nav_account">Account</span></a>
<a href="/settings" class="ni {s}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg><span data-i18n="nav_settings">Settings</span></a>
<button class="ni lang-toggle" id="_langBtn" onclick="_toggleLang()" title="Switch language" style="background:none;border:none;cursor:pointer"><span id="_langLbl" style="font-size:10px;font-weight:700;letter-spacing:1px">EN</span></button>
</nav>
<script>
(function(){{
  var lbl=document.getElementById('_langLbl');
  if(lbl)lbl.textContent=(localStorage.getItem('pilar_lang')||'en').toUpperCase();
}})();
function _toggleLang(){{
  var next=LANG==='en'?'fr':'en';
  setLang(next);
  var lbl=document.getElementById('_langLbl');
  if(lbl)lbl.textContent=next.toUpperCase();
}}
</script>"""

def nav(active):
    keys = {"m":"","tut":"","h":"","a":"","s":""}
    keys[active] = "on"
    return _NAV.format(**keys)


# ── MONITOR ───────────────────────────────────────────────────────────────────
HTML = _HEAD.replace("{FAV}","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") + """
<body>
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_monitor">Monitor</span>
<div class="hright"><button class="nb" id="nb" onclick="toggleN()">Notifs</button></div></header>
<div class="page pad">
  <div class="ab" id="abn" data-i18n="status_alert">Anomaly Detected</div>

  <!-- RESULT — hero -->
  <div id="res"><div class="idle"><span class="l1" data-i18n="idle_l1">No analysis yet</span><span class="l2" data-i18n="idle_l2">Connect a file or use manual input</span></div></div>

  <!-- LIVE FILE ZONE — primary -->
  <div class="lz-wrap">
    <input type="file" id="lfInput" accept=".csv" style="display:none" onchange="onLiveFile(this)">
    <div id="lzEmpty" class="lz-empty">
      <label for="lfInput" class="lz-cta">
        <svg style="width:18px;height:18px;stroke:currentColor;fill:none;flex-shrink:0" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span data-i18n="live_connect">Connect File</span>
      </label>
      <span class="lz-hint" data-i18n="live_hint">CSV · any column names · auto unit conversion</span>
    </div>
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
        <select id="liveIntv" class="lz-sel" onchange="resetLiveTimer()">
          <option value="2000">2s</option>
          <option value="5000" selected>5s</option>
          <option value="10000">10s</option>
          <option value="30000">30s</option>
        </select>
      </div>
    </div>
  </div>

  <!-- MANUAL INPUT — secondary, collapsible -->
  <div class="card">
    <div class="man-hdr" onclick="toggleManual()">
      <span class="ctitle" style="margin-bottom:0" data-i18n="manual_title">Manual Input</span>
      <svg class="man-chv" id="manChv" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
    <div id="manBody" style="display:none;margin-top:14px">
      <div class="ctitle" data-i18n="machine_class">Machine class</div>
      <div class="tgrid">
        <div class="tbtn on" data-val="0" onclick="selT(this)">L — Low</div>
        <div class="tbtn" data-val="1" onclick="selT(this)">M — Med</div>
        <div class="tbtn" data-val="2" onclick="selT(this)">H — High</div>
      </div>
      <div class="ctitle" data-i18n="sensor_params">Sensor parameters</div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="air_temp">Air temperature</span><div class="vwrap"><input class="vi" type="number" id="nta" value="300" min="295" max="305" step="0.1" data-raw="300" oninput="si('sta','nta','temp')"><span class="vu" id="vu-ta">K</span></div></div><input type="range" id="sta" min="295" max="305" step="0.1" value="300" oninput="ss('sta','nta',1,'temp')"><div class="rl"><span id="rl-ta-min">295K</span><span id="rl-ta-max">305K</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="proc_temp">Process temperature</span><div class="vwrap"><input class="vi" type="number" id="ntp" value="310" min="305" max="315" step="0.1" data-raw="310" oninput="si('stp','ntp','temp')"><span class="vu" id="vu-tp">K</span></div></div><input type="range" id="stp" min="305" max="315" step="0.1" value="310" oninput="ss('stp','ntp',1,'temp')"><div class="rl"><span id="rl-tp-min">305K</span><span id="rl-tp-max">315K</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="rot_speed">Rotational speed</span><div class="vwrap"><input class="vi" type="number" id="nv" value="1500" min="1000" max="3000" step="10" oninput="si('sv','nv',null)"><span class="vu" id="vu-v">rpm</span></div></div><input type="range" id="sv" min="1000" max="3000" step="10" value="1500" oninput="ss('sv','nv',0,null)"><div class="rl"><span id="rl-v-min">1000</span><span id="rl-v-max">3000</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="torque">Torque</span><div class="vwrap"><input class="vi" type="number" id="nc" value="40" min="3" max="80" step="0.1" oninput="si('sc','nc',null)"><span class="vu" id="vu-c">Nm</span></div></div><input type="range" id="sc" min="3" max="80" step="0.1" value="40" oninput="ss('sc','nc',1,null)"><div class="rl"><span id="rl-c-min">3</span><span id="rl-c-max">80Nm</span></div></div>
      <div class="sensor"><div class="srow"><span class="sname" data-i18n="tool_wear">Tool wear</span><div class="vwrap"><input class="vi" type="number" id="nu" value="100" min="0" max="250" step="1" data-raw="100" oninput="si('su','nu','wear')"><span class="vu" id="vu-u">min</span></div></div><input type="range" id="su" min="0" max="250" step="1" value="100" oninput="ss('su','nu',0,'wear')"><div class="rl"><span id="rl-u-min">0</span><span id="rl-u-max">250</span></div></div>
      <button class="btn" id="btn" onclick="analyse()" data-i18n="run_btn">Run Analysis</button>
    </div>
  </div>
</div>""" + nav("m") + """
<script>
let mT=0,lastR=null,lastD=null;
function toggleManual(){var b=document.getElementById('manBody'),c=document.getElementById('manChv'),o=b.style.display==='block';b.style.display=o?'none':'block';c.style.transform=o?'':'rotate(180deg)';}
function updN(){const b=document.getElementById('nb');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent='Notifs ON';b.className='nb on';}else{b.textContent='Enable Notifs';b.className='nb';}}
async function toggleN(){if(Notification.permission==='granted')return;await Notification.requestPermission();updN();}
function sendN(risk,zones){if(Notification.permission!=='granted')return;new Notification('Pilar — Risk: '+risk+'%',{body:zones.length?'Zones: '+zones.map(z=>z.nom).join(', '):'No specific zone',requireInteraction:true,tag:'pilar'});}
updN();
function selT(el){document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('on'));el.classList.add('on');mT=parseInt(el.dataset.val);}
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
  lastD={type:mT,
    temp_air:toRaw(gv('nta'),'temp'),
    temp_process:toRaw(gv('ntp'),'temp'),
    vitesse:gv('nv'),couple:gv('nc'),
    usure:toRaw(gv('nu'),'wear')};
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
  document.getElementById('res').innerHTML='<div class="rh '+cls+'"><div><div class="sb '+cls+'"><span class="dot '+cls+'"></span>'+st+'</div><div style="font-size:10px;color:var(--text3);margin-top:4px">'+localTimeNow()+'</div>'+confH+'</div><div><div class="rnum '+cls+'">'+r.probabilite+'<span class="runit">%</span></div><div class="rlbl">'+t('failure_prob')+'</div></div></div>'+zH;
}

// ── LIVE FILE MONITOR ─────────────────────────────────────────────────────
var _lfHandle=null,_lfTimer=null,_lfKnown=0,_lfFail=0,_lfOk=0,_lfMap=null,_lfDelim=',',_lfFallback=false,_lfUnknown={};

function onLiveFile(inp){
  var f=inp.files[0];if(!f)return;
  _lfKnown=0;_lfFail=0;_lfOk=0;_lfMap=null;_lfFallback=false;clearTimeout(_lfTimer);
  document.getElementById('liveFileName').textContent=f.name;
  document.getElementById('lzEmpty').style.display='none';
  document.getElementById('lzConn').style.display='block';
  document.getElementById('liveRowCount').textContent='0';
  document.getElementById('liveFailCount').textContent='0';
  document.getElementById('liveOkCount').textContent='0';
  document.getElementById('liveChk').textContent='';
  if(window.showOpenFilePicker){
    _connectWithAPI(f.name);
  }else{
    _lfFallback=true;
    _readSnapshot(f);
    _showRefreshBtn();
  }
}

async function _connectWithAPI(targetName){
  try{
    var picks=await window.showOpenFilePicker({types:[{description:'CSV',accept:{'text/csv':['.csv']}}]});
    _lfHandle=picks[0];
    _lfLoop();
  }catch(e){
    _lfFallback=true;
    _showRefreshBtn();
  }
}

function _showRefreshBtn(){
  document.getElementById('liveChk').innerHTML='<label for="lfInput" style="padding:3px 10px;background:var(--teal);border-radius:3px;color:#fff;font-size:10px;cursor:pointer">'+t('live_refresh')+'</label>';
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
    var hint=t('csv_detect')+' ('+found+'/6)';
    if(optFound>0)hint+=' +'+optFound+' opt';
    if(ukCount>0)hint+=' +'+ukCount+' ?';
    document.getElementById('liveChk').textContent=hint;
  }
  var total=lines.length-1;
  document.getElementById('liveRowCount').textContent=total;
  if(total>_lfKnown){
    var newLines=lines.slice(_lfKnown+1);
    var lastRow=null;
    newLines.forEach(function(line){
      var vals=line.split(_lfDelim);
      var obj=buildPilarRow(vals,_lfMap);
      if(obj)lastRow=obj;
    });
    _lfKnown=total;
    if(lastRow){
      try{
        var payload=Object.assign({},lastRow);
        if(lastRow._extra)Object.assign(payload,lastRow._extra);
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
          if(lastRow._unknown){
            Object.keys(lastRow._unknown).forEach(function(colName){
              var val=lastRow._unknown[colName];
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
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_account">Account</span></header>
<div class="page pad">

{% if not user %}
<div class="card" style="text-align:center;padding:28px">
  <div style="width:48px;height:48px;border-radius:12px;background:rgba(13,148,136,.08);border:1px solid rgba(13,148,136,.15);display:flex;align-items:center;justify-content:center;margin:0 auto 16px"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.5"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>
  <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px" data-i18n="acc_guest_title">Guest Mode</div>
  <div style="font-size:11px;color:var(--text3);line-height:1.7;margin-bottom:20px" data-i18n-html="acc_guest_desc">Sign in to save your data.</div>
  <a href="/login" style="display:block;padding:13px;background:var(--teal);color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px" data-i18n="acc_signin">Sign In</a>
  <a href="/register" style="display:block;padding:13px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:6px;text-decoration:none;font-size:12px;letter-spacing:1px;text-transform:uppercase" data-i18n="acc_register">Create Account</a>
</div>

{% else %}
<div class="card">
  <div class="ctitle" data-i18n="acc_card_title">Account</div>
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:13px;font-weight:600;color:var(--text)">{{ user.email }}</div>
      <div style="display:flex;gap:6px;margin-top:5px;align-items:center">
        <span style="padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;background:rgba(13,148,136,0.12);color:var(--teal-light);text-transform:uppercase">{{ user.plan }}</span>
        {% if my_role %}<span class="acc-role" data-role="{{ my_role }}" style="padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;background:rgba(124,58,237,0.12);color:#a78bfa;text-transform:uppercase">{{ my_role }}</span>{% endif %}
      </div>
    </div>
    <a href="/logout" style="padding:8px 14px;background:transparent;border:1px solid var(--border2);color:var(--text3);border-radius:6px;text-decoration:none;font-size:11px;font-weight:600;white-space:nowrap" data-i18n="acc_signout">Sign Out</a>
  </div>
</div>

{% if not team %}
<div class="card">
  <div class="ctitle" data-i18n="acc_team_title">Team</div>
  <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:14px" data-i18n="acc_no_team_desc">Create a team to collaborate with colleagues.</p>
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

</div>""" + nav("a") + """
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
</script></body></html>"""

# ── TWIN ──────────────────────────────────────────────────────────────────────
TWIN_HTML = _HEAD.replace("{FAV}","iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAHxUlEQVR4nO2Za4wT1xXHz713PLbHj8Xe2V17H973C3YXNgQKIWlE2rJSFDWCqlUkmoeafilVUrWVKqVqxIe0EoqqNIqqqEraokStRBIR2kDDpoFNgeUN5rE89gVre9/s+v0az8y9tx9MUSlSpNiTOkj+f7Vn7vndc+45555BnHO4n4VLbUCxKgOUWmWAUqsMcI/+z3m57IF7hBAy/J2fo7IHSq0yQKklGPs6zvnnp1GEkLGnHN3v7bRhHuAcAPhEcGY5EjMJwl2O4AAACIGmUdld0dHcAMZlW8MAGGOE4OP+kXf3Dq5wOChj//UjRwghhOPJ1DNbt3S2+ChlhHyVADjnGKNYMjU1s1AluzWdEwKEYH577xGlDCGQ5crQ3GIylbbbJM65IU4wBoAxTgh+ffcHr779gd0mbVi9srvFG01kzKIJAOU0ze2wXRoL+K/f/Me/ThPB9KsdTxvlBCOzEKVMIKKSo51NtVs2PTA2NSuaBIyxklNXtjaoOj1+YVQQRKpTAxc1Jgvl42F+KfzL377dWO+VLKblaCKr6ACIYGCAHBaz7HZGE6m5W8u/+fkPaypdRoWQYQCM80+Pn9u0tschSb//677RqZlwJC5gwWQSVF1b4XD2dzQ+/9QTiWTy1KWxb25aiwxKRAaEEGMcY3Ru5JpHdjkkCQDSioZz6c0dVVUuGyAUiWcvBpZimSzn3Olw1MgVl66N96/qzD9Y5OoGtBIYI0ppOqOs6W5XUrFkJmvOxQb6W5tX9sxMB65dverr6NncW19r44qqZTLJnraGcCyhalrx1kPxIcQ5p5SevXw9mkh2tzUtBG/MxZQ6Gz0ZSA4fGtz+1Danzb5797u9D20e6PMGYrytXvZ53KGYbhVNbY31vOhAMiCEOKDL4wGft2opmrRU+Tw0EM3k5sdHHnl4/Ww0O72U2rhpw8TElekG1zce7HHWNAKAC5ZvLUcBIc54kQehWACE0Gxg4uLhD70DW1qbXIoS1SrI6aHD69eur27u9Pv9lEPf2nUNtZ7Jc8N9dRWpRBxjnEsl4xNj8xL3NLYXaUCxAIzzWl/zIwPb2nu7K+qa3BilFkOuxq7J8bGcIGlgooxN3QxFgqPe9t6UjjDYgbLlTGaBVG2saShydTDgEHMQRfOaNT0OlzubWJ4Khk5cC6zuaZ9RzMcOHujvaunvbPF/NjiVFNubG89Nzk+FQqlkLJ7VzTY7MZmKByj+EANCML+4FJhbXNvVGImnMxoMDf69wec7fXXWmp0XBRwmlQ/3tcyFgg9963HZit2yyz863e6rddhtxZez4s8AcM69NVUjYzc0ED0eBwDsCiQjsavrulqRKnFGRXvFGf+FUEZ8rrEOAOKpzMKtcH93O+McF13LDOuF+rpbj5zxP/7oRs65xSxGsO2t/aeavJUY4+Dclda2Douo5C0ePn95XW8nQrfvCUXKgEKGEOKce6qq6jzV/xw+CwDprMKoblshh7M8qmLJWcm5nlYUBPDJsTM+b031V60XAoD87o6M31CyuY8ODS9GEppGMcEYI13NWSxSTaVzy6Nfc9qkno4WQ4InL8NCCCPEGOvtaL0RCK72mHW5wiSYKKMCIgiBpmlEFLyVFc2+BsYYxoZNQ4ydSiAA+Ntnp0/4A06HnVLKGUMIYYwxwrFkcpGf37HdgNx/15JGhRCljBD8uz+//4f3Pna7nbqq5dsczhniCACZTEI4lnjh6Sd//P2tlFJCiCHrFuiBO0OHO209QsA42/jAKqfDbhZN/9Mq58dBSi7X19UCAAjje99QmAzwQPH5pJiLwRf2QL703gpHUhlFVVVMUEdTIwBMzy9qOrWazQ67dX4pDAAe2S1ZrBx4PuPruq6qms0mEYwppcG5BQCUzmTqPdWuCmdh1kNBdYADgKrp33th57FzF/cePLLj5dcopdcnA99+/qWlSIxg/Mob7+x68y8EY4SAEAKABEL27B/a+fofCcaargPA/kMnfvCLXcHZxe0/+/WBoeMAwO4aJX1pAPloqfdUW0XxwVUrX/rRM/sGjwZn5vq626xWsaWh1mqx1MqyV3ZLVivjHOWvbIxdGr9x+NTVaDwhEEIIaWnwWkXzE49t6u1q/9P7H8Pt2d6XD5BXPu5jqfT+oZNtrT5PtZxIZTAW8rurc67z2/+jjCGEzl8Ze/KxjT1tvvcODOW3QKdUUbVjZy8vLsVefO67BR+kwguKxWKeuxXWNG3PGzslq1XX6Z3yhDkngBnnhBCCMQAcOeUfnQytcNo/OnwS/tMEWSzChWsTN0PTmzes4QCFJYICARBC8US63itvG/h6XXUlAJgEklFyBBMAUDlTqI4RWliOnPCPRONxIpi2bx3Y+ZNnA7MLR09fRAA6pbpOX3z2O3KFY8fLr2FUYD78wgD5ZSaDs5WyPBmcUzVN1ykAjAZmK2XXeDCUTKdzOo0l0u/sPfjqW3sYhw8/Pc4BOWySZDX3dbXuOzQciceWokmzVQrNL775yk+XwrF9nxzFGBfAUCC3klNF0ZTLqRazmI/djJKzmEVV1QjB+W4nnVFE0SRZzJlsDmNkMYuqphOCKaUAwDkIAlE1zWo2A0Aqk7VL1gIsKf0HjrwBBZfC0gMUqfv+I18ZoNQqA5RaZYBSqwxQapUBSq0yQKl13wP8GxwKx1pBe9uwAAAAAElFTkSuQmCC") + """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<body>
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_twin">Digital Twin</span></header>
<div class="page pad" id="tc"><div class="idle"><span class="l1" data-i18n="twin_loading">Loading simulation...</span></div></div>""" + nav("t") + """
<script>
const PL={paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#64748b',size:10},margin:{t:8,b:36,l:40,r:8},xaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},yaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},legend:{bgcolor:'transparent',font:{size:9}},hovermode:'x unified'};
const PC={responsive:true,displayModeBar:false};
async function load(){
  let d;
  try{
    const res=await fetch('/api/twin');
    try{d=await res.json();}catch(je){
      document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Erreur serveur (code '+res.status+')</span><span class="l2">Vérifiez les logs Railway</span></div>';return;
    }
    if(d.error){document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">'+d.error+'</span></div>';return;}
  }catch(err){
    document.getElementById('tc').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Erreur réseau: '+err.message+'</span></div>';return;
  }
  if(!d.has_data){document.getElementById('tc').innerHTML='<div class="idle"><span class="l1">'+t('twin_no_data')+'</span><span class="l2">'+t('twin_no_data2')+'</span><a href="/" style="margin-top:16px;padding:12px 20px;background:var(--teal);color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">'+t('twin_go')+'</a></div>';return;}
  const bCls=d.failure_hours===null?'ok':d.failure_hours<6?'alert':'amber';
  const bT=d.failure_hours===null?t('twin_healthy'):t('twin_failure')+d.failure_hours+'h';
  const wta_disp=toDisplay(d.last_params.temp_air,'temp');
  const wu_disp=toDisplay(d.last_params.usure,'wear');
  document.getElementById('tc').innerHTML=`
    <div class="rh ${bCls}"><div><div class="sb ${bCls}"><span class="dot ${bCls}"></span>${bT}</div><div style="font-size:10px;color:var(--text3);margin-top:4px">${t('twin_trend')} ${d.trend}</div></div><div><div class="rnum ${bCls}">${d.current_risk}<span class="runit">%</span></div><div class="rlbl">${t('twin_cur_risk')}</div></div></div>
    <div class="kgrid"><div class="kc"><div class="kv amber">${d.avg_risk_24h}%</div><div class="kl">${t('twin_avg')}</div></div><div class="kc"><div class="kv ${d.anomaly_rate>=30?'alert':'ok'}">${d.anomaly_rate}%</div><div class="kl">${t('twin_anom')}</div></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_risk')}</div><div id="cr" style="height:220px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_wear')}</div><div id="cw" style="height:180px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_temp')}</div><div id="ct" style="height:180px"></div></div>
    <div class="card"><div class="ctitle">${t('twin_c_sim')}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
        <div><label class="flbl">${t('twin_speed')}</label><input class="fi" type="number" id="wv" value="${d.last_params.vitesse}" step="10"></div>
        <div><label class="flbl">${t('twin_torque')}</label><input class="fi" type="number" id="wc" value="${d.last_params.couple}" step="0.1"></div>
        <div><label class="flbl">${t('twin_wear')}</label><input class="fi" type="number" id="wu" value="${wu_disp}" step="0.001" data-raw="${d.last_params.usure}"></div>
        <div><label class="flbl">${t('twin_airtemp')}</label><input class="fi" type="number" id="wta" value="${wta_disp}" step="0.1" data-raw="${d.last_params.temp_air}"></div>
      </div>
      <button class="btn" onclick="sim()">${t('twin_sim')}</button>
      <div id="wr" style="margin-top:12px"></div>
    </div>`;
  Plotly.newPlot('cr',[{x:d.history_times,y:d.history_risks,name:'History',type:'scatter',mode:'lines+markers',line:{color:'#14b8a6',width:2},marker:{size:5}},{x:d.future_times,y:d.future_risks,name:'Simulated',type:'scatter',mode:'lines',line:{color:'#7c3aed',width:2,dash:'dot'},fill:'tozeroy',fillcolor:'rgba(124,58,237,0.04)'},{x:[...d.history_times,...d.future_times],y:Array(d.history_times.length+d.future_times.length).fill(50),name:'Threshold',type:'scatter',mode:'lines',line:{color:'#dc2626',width:1,dash:'dash'}}],{...PL,yaxis:{...PL.yaxis,range:[0,105]}},PC);
  Plotly.newPlot('cw',[{x:d.history_times,y:d.history_wear,name:'Actual',type:'scatter',mode:'lines+markers',line:{color:'#d97706',width:2},marker:{size:4}},{x:d.future_times,y:d.future_wear,name:'Projected',type:'scatter',mode:'lines',line:{color:'#d97706',width:2,dash:'dot'}}],PL,PC);
  Plotly.newPlot('ct',[{x:d.history_times,y:d.history_temp,name:'Actual',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2}},{x:d.future_times,y:d.future_temp,name:'Projected',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2,dash:'dot'}}],PL,PC);
}
async function sim(){
  const wta_raw=parseFloat(document.getElementById('wta').dataset.raw)||toRaw(parseFloat(document.getElementById('wta').value),'temp');
  const wu_raw=parseFloat(document.getElementById('wu').dataset.raw)||toRaw(parseFloat(document.getElementById('wu').value),'wear');
  const p={type:1,temp_air:wta_raw,temp_process:wta_raw+10,vitesse:parseFloat(document.getElementById('wv').value),couple:parseFloat(document.getElementById('wc').value),usure:wu_raw};
  const res=await fetch('/api/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
  const d=await res.json();const c={ok:'#059669',amber:'#d97706',alert:'#dc2626'};const cls=d.risk>=50?'alert':d.risk>=22?'amber':'ok';
  document.getElementById('wr').innerHTML='<div style="padding:14px;background:var(--bg);border:1px solid '+c[cls]+';border-radius:6px"><div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase">'+t('twin_sim_r')+'</div><div style="font-size:32px;font-weight:800;color:'+c[cls]+';margin:4px 0">'+d.risk+'%</div><div style="font-size:12px;font-weight:600;color:'+c[cls]+'">'+d.status+'</div><div style="font-size:11px;color:var(--text3);margin-top:3px">'+d.message+'</div>'+(d.zones.length?'<div style="font-size:10px;color:var(--amber);margin-top:6px">Zones: '+d.zones.map(z=>z.nom+' '+z.proba+'%').join(' · ')+'</div>':'')+'</div>';
}
load();
</script></body></html>"""

# ── HISTORY ───────────────────────────────────────────────────────────────────
HISTORY_HTML = _HEAD.replace("{FAV}","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") + """
<body>
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_history">History</span></header>
<div class="page pad">
  <div class="kgrid">
    <div class="kc"><div class="kv">{{ total }}</div><div class="kl" data-i18n="hist_total">Total</div></div>
    <div class="kc"><div class="kv alert">{{ anomalies }}</div><div class="kl" data-i18n="hist_anom">Anomalies</div></div>
    <div class="kc"><div class="kv amber">{{ avg_risk }}%</div><div class="kl" data-i18n="hist_avg">Avg risk</div></div>
    <div class="kc"><div class="kv ok">{{ mails }}</div><div class="kl" data-i18n="hist_alerts">Alerts sent</div></div>
  </div>
  <div class="tw">
    <table>
      <thead><tr><th data-i18n="hist_time">Time</th><th data-i18n="hist_class">Class</th><th data-i18n="hist_risk">Risk</th><th data-i18n="hist_status">Status</th><th data-i18n="hist_zones">Zones</th><th data-i18n="hist_alert">Alert</th></tr></thead>
      <tbody>
      {% for a in analyses %}
      <tr><td data-utc="{{ a.timestamp.isoformat() }}Z">{{ a.timestamp.strftime('%d/%m %H:%M') }}</td><td>{{ a.machine_type }}</td><td>{{ a.risk }}%</td>
          <td><span class="badge {{ 'alert' if a.prediction else 'ok' }}">{{ 'Anomaly' if a.prediction else 'OK' }}</span></td>
          <td>{{ a.zones or '—' }}</td>
          <td>{% if a.mail_sent %}<span class="mb">Sent</span>{% else %}—{% endif %}</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>""" + nav("h") + """
<script>
document.querySelectorAll('td[data-utc]').forEach(function(td){
  td.textContent=localTime(td.dataset.utc,{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
});
</script>
</body></html>"""

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SETTINGS_HTML = _HEAD.replace("{FAV}","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") + """
<body>
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
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_version">Version</span><span>Pilar v3.0</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_aimodel">AI Model</span><span>Claude Haiku</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)" data-i18n="set_domain">Domain</span><span>trypilar.com</span></div>
    </div>
  </div>
  <div class="card">
    <div class="ctitle" data-i18n="set_nav_title">Navigation</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <a href="/twin" style="display:flex;justify-content:space-between;font-size:12px;color:var(--text2);text-decoration:none;padding:8px 0"><span data-i18n="set_twin_nav">Digital Twin</span><span style="color:var(--teal-light)">›</span></a>
    </div>
  </div>
</div>""" + nav("s") + """
<script>
async function saveEmail(){const e=document.getElementById('em').value;if(!e)return;await fetch('/set_email',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:e})});const s=document.getElementById('sv');s.style.display='block';setTimeout(()=>s.style.display='none',3000);}
function updN(){const b=document.getElementById('nb');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent=t('set_notif_on');b.style.background='var(--green)';}else if(p==='denied'){b.textContent=t('set_notif_blocked');b.style.background='var(--red)';}else{b.textContent=t('set_notif_btn');b.style.background='var(--purple)';}}
async function toggleN(){if(Notification.permission==='granted')return;await Notification.requestPermission();updN();}
updN();
</script></body></html>"""


# ── ASSISTANT ─────────────────────────────────────────────────────────────────
ASSISTANT_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
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
""" + nav("ai") + """
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
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_tutorial">Import & Run</span>
<div class="hright"><a href="/twin" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px" data-i18n="nav_twin">Twin</a></div>
</header>
<div class="page pad">

  <!-- CSV FORMAT GUIDE -->
  <div class="card">
    <div class="ctitle" data-i18n="tut_format">CSV Format</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:12px" data-i18n="tut_csv_desc">Your CSV must contain these columns (order does not matter):</p>
    <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:6px;padding:12px;overflow-x:auto;margin-bottom:12px">
      <code style="font-size:11px;color:var(--teal-light);white-space:nowrap">type, temp_air, temp_process, vitesse, couple, usure</code>
    </div>
    <table style="font-size:11px;margin-top:0">
      <thead><tr><th data-i18n="tut_cols">Column</th><th data-i18n="tut_unit">Unit</th><th data-i18n="tut_range">Range</th><th data-i18n="tut_desc">Description</th></tr></thead>
      <tbody>
        <tr><td style="color:var(--teal-light)">type</td><td>—</td><td>0, 1 or 2</td><td>Machine class (L=0, M=1, H=2)</td></tr>
        <tr><td style="color:var(--teal-light)">temp_air</td><td>K</td><td>295–310</td><td>Air temperature (Kelvin)</td></tr>
        <tr><td style="color:var(--teal-light)">temp_process</td><td>K</td><td>305–320</td><td>Process temperature (Kelvin)</td></tr>
        <tr><td style="color:var(--teal-light)">vitesse</td><td>rpm</td><td>1000–3000</td><td>Rotational speed</td></tr>
        <tr><td style="color:var(--teal-light)">couple</td><td>Nm</td><td>3–80</td><td>Torque</td></tr>
        <tr><td style="color:var(--teal-light)">usure</td><td>min</td><td>0–250</td><td>Tool wear time</td></tr>
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
""" + nav("tut") + """
<script>
var rows=[],idx=0,timer=null,paused=false,nFail=0,nOk=0,sumRisk=0;

function dlSample(){
  var csv='type,temp_air,temp_process,vitesse,couple,usure\\n'+
    '0,300.1,310.5,1500,40.2,45\\n'+
    '1,301.3,311.8,1800,55.0,120\\n'+
    '2,299.7,309.2,2200,65.3,200\\n'+
    '0,302.0,312.1,1200,30.5,210\\n'+
    '1,300.5,310.9,1600,48.0,80\\n';
  var a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='pilar_sample.csv';a.click();
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
    if(found<6){
      var missing=_CSV_FIELDS.filter(function(field){return !map[field];}).join(', ');
      alert(t('csv_bad')+': '+missing);return;
    }
    rows=[];
    for(var r=1;r<lines.length;r++){
      var vals=lines[r].split(delim);
      var obj=buildPilarRow(vals,map);
      if(obj)rows.push(obj);
    }
    var info=found+'/6 '+t('csv_detect');
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
    'Type '+(row.type===0?'L':row.type===1?'M':'H')+
    ' | '+row.vitesse+'rpm | '+row.usure+'min wear</div>';
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
      var found=Object.keys(liveMap).length;
      if(found<6){
        var missing=_CSV_FIELDS.filter(function(f){return !liveMap[f];}).join(', ');
        document.getElementById('liveStatus').textContent=t('csv_bad')+': '+missing;
        document.getElementById('liveStatus').style.color='var(--red)';return;
      }
      document.getElementById('liveStatus').textContent=t('csv_detect')+' ('+found+'/6)';
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
</div>""" + nav("") + """
<script>
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
var _adpRaw=null,_adpDelim=',',_adpHdr=[],_adpMap=[];
var _ADP_KEYS=['type','temp_air','temp_process','vitesse','couple','usure','humidite','vibration','pression','courant','tension'];
var _ADP_LBL_FR={type:'Type machine',temp_air:'Temp. air',temp_process:'Temp. process',vitesse:'Vitesse (rpm)',couple:'Couple (Nm)',usure:'Usure outil',humidite:'Humidité (%)',vibration:'Vibration (mm/s)',pression:'Pression (bar)',courant:'Courant (A)',tension:'Tension (V)'};
var _ADP_LBL_EN={type:'Machine type',temp_air:'Air temp',temp_process:'Process temp',vitesse:'Speed (rpm)',couple:'Torque (Nm)',usure:'Tool wear',humidite:'Humidity (%)',vibration:'Vibration (mm/s)',pression:'Pressure (bar)',courant:'Current (A)',tension:'Voltage (V)'};
var _ADP_OUT={type:'type',temp_air:'air_temperature_k',temp_process:'process_temperature_k',vitesse:'rotational_speed_rpm',couple:'torque_nm',usure:'tool_wear_min',humidite:'humidity_pct',vibration:'vibration_mms',pression:'pressure_bar',courant:'current_a',tension:'voltage_v'};

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
    if(m.pf==='temp_air'||m.pf==='temp_process'){
      unitEl='<select id="u'+i+'" onchange="adpSetUnit('+i+',this.value)" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 6px;font-size:10px">';
      ['K','°C','°F'].forEach(function(u){unitEl+='<option'+(u===(m.unit||'K')?' selected':'')+'>'+u+'</option>';});
      unitEl+='</select>';
    } else if(m.pf==='usure'){
      unitEl='<select id="u'+i+'" onchange="adpSetUnit('+i+',this.value)" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:5px 6px;font-size:10px">';
      ['min','h'].forEach(function(u){unitEl+='<option'+(u===(m.unit||'min')?' selected':'')+'>'+u+'</option>';});
      unitEl+='</select>';
    }
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
  if(pf==='type'){var s=raw.toLowerCase();return s==='l'||s==='low'?'0':s==='m'||s==='medium'||s==='med'?'1':s==='h'||s==='high'?'2':raw;}
  var v=parseFloat(raw.replace(',','.'));if(isNaN(v))return '';
  if(pf==='temp_air'||pf==='temp_process'){var u=unit||(v<200?'°C':'K');if(u==='°C')v+=273.15;else if(u==='°F')v=(v-32)*5/9+273.15;}
  if(pf==='usure'){var u=unit||(v<20?'h':'min');if(u==='h')v*=60;}
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

# ── LANDING PAGE ──────────────────────────────────────────────────────────────
LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilar — Predictive Maintenance for Industry</title>
<meta name="description" content="Predict machine failures before they happen. Pilar uses AI to analyze industrial sensors and alert you hours before breakdowns occur.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050d1a;--bg2:#080f1e;--surface:#0c1526;--surface2:#111c30;
  --border:#1a2a45;--border2:#1e3050;
  --teal:#0d9488;--teal2:#14b8a6;--teal3:#5eead4;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --red:#ef4444;--amber:#f59e0b;--green:#22c55e;
  --grad:linear-gradient(135deg,#0d9488 0%,#0891b2 50%,#7c3aed 100%);
}
html{scroll-behavior:smooth}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}

/* NAV */
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 24px;height:64px;display:flex;align-items:center;justify-content:space-between;background:rgba(5,13,26,0.85);backdrop-filter:blur(12px);border-bottom:1px solid rgba(255,255,255,0.04)}
.nav-logo{font-size:20px;font-weight:900;letter-spacing:3px;color:#fff;text-decoration:none;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-links{display:flex;align-items:center;gap:8px}
.nav-links a{text-decoration:none}
.btn-ghost{padding:8px 18px;background:transparent;border:1px solid var(--border2);color:var(--text2);border-radius:6px;font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;cursor:pointer;transition:all .2s}
.btn-ghost:hover{border-color:var(--teal);color:var(--teal2)}
.btn-teal{padding:9px 20px;background:var(--teal);color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;transition:all .2s;text-decoration:none;display:inline-block}
.btn-teal:hover{background:var(--teal2);transform:translateY(-1px)}

/* HERO */
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:100px 24px 60px;position:relative;overflow:hidden}
.hero-glow{position:absolute;width:800px;height:800px;border-radius:50%;background:radial-gradient(circle,rgba(13,148,136,.15) 0%,transparent 70%);top:50%;left:50%;transform:translate(-50%,-60%);pointer-events:none}
.hero-glow2{position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,.08) 0%,transparent 70%);top:30%;right:10%;pointer-events:none}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(13,148,136,.1);border:1px solid rgba(13,148,136,.2);border-radius:100px;font-size:11px;font-weight:600;color:var(--teal2);letter-spacing:1px;text-transform:uppercase;margin-bottom:28px}
.badge-dot{width:6px;height:6px;border-radius:50%;background:var(--teal2);box-shadow:0 0 8px var(--teal2);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
h1{font-size:clamp(36px,7vw,80px);font-weight:900;line-height:1.05;letter-spacing:-2px;margin-bottom:24px;max-width:900px}
h1 span{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:clamp(15px,2.5vw,20px);color:var(--text2);max-width:580px;line-height:1.7;margin-bottom:40px;font-weight:400}
.hero-cta{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin-bottom:60px}
.btn-hero{padding:16px 36px;background:var(--teal);color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;transition:all .3s;text-decoration:none;display:inline-flex;align-items:center;gap:8px;box-shadow:0 0 40px rgba(13,148,136,.3)}
.btn-hero:hover{background:var(--teal2);transform:translateY(-2px);box-shadow:0 0 60px rgba(13,148,136,.5)}
.btn-hero-ghost{padding:16px 32px;background:transparent;border:1px solid var(--border2);color:var(--text2);border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;text-decoration:none;display:inline-block}
.btn-hero-ghost:hover{border-color:var(--teal);color:var(--teal2)}

/* MOCKUP PREVIEW */
.hero-preview{width:100%;max-width:900px;border-radius:16px;overflow:hidden;border:1px solid var(--border);box-shadow:0 40px 100px rgba(0,0,0,.6),0 0 80px rgba(13,148,136,.08);background:var(--surface);padding:0}
.preview-bar{height:36px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 14px;gap:6px}
.preview-dot{width:10px;height:10px;border-radius:50%}
.preview-screen{padding:20px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.preview-card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:14px}
.preview-label{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
.preview-val{font-size:22px;font-weight:800;margin-bottom:2px}
.preview-bar2{height:4px;border-radius:2px;background:var(--border);margin-top:8px;overflow:hidden}
.preview-fill{height:100%;border-radius:2px;animation:grow 2s ease-out forwards}
@keyframes grow{from{width:0}to{width:var(--w)}}
.preview-alert{grid-column:1/-1;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:12px;display:flex;align-items:center;gap:10px}
.preview-alert-dot{width:8px;height:8px;border-radius:50%;background:#ef4444;flex-shrink:0;animation:pulse 1.5s infinite}
.preview-normal{grid-column:1/-1;background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.15);border-radius:8px;padding:12px;display:flex;align-items:center;gap:10px}

/* STATS BAR */
.stats-bar{background:var(--surface);border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:32px 24px;display:flex;flex-wrap:wrap;justify-content:center;gap:0}
.stat-item{flex:1;min-width:160px;max-width:220px;text-align:center;padding:0 24px;position:relative}
.stat-item+.stat-item::before{content:'';position:absolute;left:0;top:10%;height:80%;width:1px;background:var(--border)}
.stat-num{font-size:36px;font-weight:900;letter-spacing:-1px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.stat-lbl{font-size:11px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:4px}

/* SECTIONS */
section{padding:80px 24px;max-width:1100px;margin:0 auto}
.section-label{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--teal2);margin-bottom:12px}
.section-title{font-size:clamp(28px,4vw,44px);font-weight:800;letter-spacing:-1px;line-height:1.15;margin-bottom:16px}
.section-title span{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.section-sub{font-size:16px;color:var(--text2);max-width:560px;line-height:1.7}

/* HOW IT WORKS */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px;margin-top:48px}
.step{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:28px;position:relative;overflow:hidden;transition:border-color .2s}
.step:hover{border-color:rgba(13,148,136,.3)}
.step-num{font-size:64px;font-weight:900;position:absolute;right:16px;top:8px;color:rgba(255,255,255,.03);letter-spacing:-2px;line-height:1}
.step-icon{width:44px;height:44px;border-radius:10px;background:rgba(13,148,136,.1);border:1px solid rgba(13,148,136,.2);display:flex;align-items:center;justify-content:center;margin-bottom:16px;font-size:20px}
.step-title{font-size:15px;font-weight:700;margin-bottom:8px}
.step-desc{font-size:13px;color:var(--text2);line-height:1.65}

/* PERFORMANCE */
.perf-grid{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:center;margin-top:48px}
.perf-metrics{display:grid;gap:16px}
.metric-row{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;display:flex;align-items:center;justify-content:space-between}
.metric-name{font-size:13px;color:var(--text2)}
.metric-bar-wrap{flex:1;height:6px;background:var(--border);border-radius:3px;margin:0 16px;overflow:hidden}
.metric-bar{height:100%;border-radius:3px;background:var(--grad)}
.metric-val{font-size:16px;font-weight:800;color:var(--teal2);min-width:48px;text-align:right}
.perf-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:32px;text-align:center}
.perf-big{font-size:72px;font-weight:900;letter-spacing:-3px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}
.perf-unit{font-size:18px;font-weight:600;color:var(--teal2)}
.perf-desc{font-size:13px;color:var(--text2);margin-top:12px;line-height:1.6}

/* FEATURES */
.features-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:48px}
.feature-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;transition:all .2s}
.feature-card:hover{border-color:rgba(13,148,136,.3);transform:translateY(-2px)}
.feature-icon{font-size:28px;margin-bottom:14px}
.feature-title{font-size:15px;font-weight:700;margin-bottom:8px}
.feature-desc{font-size:13px;color:var(--text2);line-height:1.65}

/* ROI */
.roi-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:48px}
.roi-card{border-radius:12px;padding:28px}
.roi-before{background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.15)}
.roi-after{background:rgba(13,148,136,.05);border:1px solid rgba(13,148,136,.15)}
.roi-label{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px}
.roi-before .roi-label{color:#f87171}
.roi-after .roi-label{color:var(--teal2)}
.roi-item{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.roi-item:last-child{border:none}
.roi-item-label{font-size:12px;color:var(--text2)}
.roi-item-val{font-size:13px;font-weight:700}
.roi-before .roi-item-val{color:#f87171}
.roi-after .roi-item-val{color:var(--teal2)}

/* INDUSTRIES */
.industries{display:flex;flex-wrap:wrap;gap:12px;margin-top:32px}
.industry-chip{padding:10px 20px;background:var(--surface);border:1px solid var(--border);border-radius:100px;font-size:12px;font-weight:600;color:var(--text2);transition:all .2s;cursor:default}
.industry-chip:hover{border-color:var(--teal);color:var(--teal2);background:rgba(13,148,136,.05)}

/* PRICING */
.pricing-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-top:48px;max-width:900px;margin-left:auto;margin-right:auto}
.plan-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:32px;position:relative;transition:all .2s}
.plan-card:hover{transform:translateY(-4px)}
.plan-card.featured{border-color:var(--teal);background:linear-gradient(135deg,rgba(13,148,136,.08),rgba(8,145,178,.04))}
.plan-badge{position:absolute;top:-12px;left:50%;transform:translateX(-50%);padding:4px 16px;background:var(--teal);color:#fff;border-radius:100px;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;white-space:nowrap}
.plan-name{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text2);margin-bottom:12px}
.plan-price{font-size:48px;font-weight:900;letter-spacing:-2px;line-height:1;margin-bottom:4px}
.plan-price span{font-size:16px;font-weight:500;color:var(--text3)}
.plan-desc{font-size:12px;color:var(--text3);margin-bottom:24px;line-height:1.6}
.plan-features{list-style:none;margin-bottom:28px}
.plan-features li{font-size:13px;color:var(--text2);padding:7px 0;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.plan-features li:last-child{border:none}
.plan-features li::before{content:'\2713';color:var(--teal2);font-weight:700;flex-shrink:0}
.plan-features li.off{color:var(--text3)}
.plan-features li.off::before{content:'—';color:var(--text3)}
.plan-btn{display:block;width:100%;padding:13px;border-radius:8px;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;text-align:center;text-decoration:none;transition:all .2s;cursor:pointer;border:none}
.plan-btn-primary{background:var(--teal);color:#fff}
.plan-btn-primary:hover{background:var(--teal2)}
.plan-btn-ghost{background:transparent;border:1px solid var(--border2);color:var(--text2)}
.plan-btn-ghost:hover{border-color:var(--teal);color:var(--teal2)}

/* FINAL CTA */
.cta-section{background:linear-gradient(135deg,rgba(13,148,136,.1),rgba(124,58,237,.06));border:1px solid rgba(13,148,136,.15);border-radius:20px;padding:64px 40px;text-align:center;margin:0 24px 80px}
.cta-section h2{font-size:clamp(26px,4vw,42px);font-weight:900;letter-spacing:-1px;margin-bottom:16px}
.cta-section p{font-size:16px;color:var(--text2);margin-bottom:36px;max-width:460px;margin-left:auto;margin-right:auto;line-height:1.7}

/* FOOTER */
footer{border-top:1px solid var(--border);padding:32px 24px;text-align:center}
.footer-logo{font-size:16px;font-weight:900;letter-spacing:3px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
.footer-text{font-size:11px;color:var(--text3);line-height:1.7}
.footer-links{display:flex;flex-wrap:wrap;justify-content:center;gap:16px;margin-top:12px}
.footer-links a{font-size:11px;color:var(--text3);text-decoration:none;transition:color .2s}
.footer-links a:hover{color:var(--teal2)}

/* DIVIDER */
.full-divider{width:100%;height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent)}

/* RESPONSIVE */
@media(max-width:768px){
  .perf-grid,.roi-grid{grid-template-columns:1fr}
  .preview-screen{grid-template-columns:1fr 1fr}
  .stat-item+.stat-item::before{display:none}
  .cta-section{padding:40px 24px}
  .nav-links .btn-hero-ghost{display:none}
}
@media(max-width:480px){
  .preview-screen{grid-template-columns:1fr}
  .pricing-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <a href="/" class="nav-logo">PILAR</a>
  <div class="nav-links">
    <div style="display:flex;gap:2px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:6px;padding:3px;margin-right:4px">
      <button id="_lEN" onclick="_lp('en')" style="padding:4px 10px;border:none;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;background:transparent;color:#64748b;transition:all .15s">EN</button>
      <button id="_lFR" onclick="_lp('fr')" style="padding:4px 10px;border:none;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;background:transparent;color:#64748b;transition:all .15s">FR</button>
    </div>
    <a href="/login" class="btn-ghost" data-ilp="nav_signin">Sign In</a>
    <a href="/register" class="btn-teal" data-ilp="nav_start">Get Started Free</a>
  </div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-glow2"></div>
  <div class="badge"><span class="badge-dot"></span><span data-ilp="hero_badge">AI-Powered Predictive Maintenance</span></div>
  <h1 data-ilp="hero_h">Stop fixing machines.<br><span>Predict failures first.</span></h1>
  <p class="hero-sub" data-ilp="hero_sub">Pilar analyzes your industrial sensors in real time and alerts you hours before a breakdown — before it costs you anything.</p>
  <div class="hero-cta">
    <a href="/register" class="btn-hero">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      <span data-ilp="hero_cta1">Start for free</span>
    </a>
    <a href="#how-it-works" class="btn-hero-ghost" data-ilp="hero_cta2">See how it works</a>
  </div>

  <!-- APP PREVIEW -->
  <div class="hero-preview">
    <div class="preview-bar">
      <div class="preview-dot" style="background:#ef4444"></div>
      <div class="preview-dot" style="background:#f59e0b"></div>
      <div class="preview-dot" style="background:#22c55e"></div>
      <span style="font-size:10px;color:#475569;margin-left:8px">trypilar.com — Monitor</span>
    </div>
    <div class="preview-screen">
      <div class="preview-card">
        <div class="preview-label">Failure Risk</div>
        <div class="preview-val" style="color:#ef4444">72%</div>
        <div class="preview-bar2"><div class="preview-fill" style="--w:72%;background:linear-gradient(90deg,#f59e0b,#ef4444)"></div></div>
      </div>
      <div class="preview-card">
        <div class="preview-label">Rotational Speed</div>
        <div class="preview-val" style="color:#14b8a6">1 486 rpm</div>
        <div class="preview-bar2"><div class="preview-fill" style="--w:55%;background:var(--teal)"></div></div>
      </div>
      <div class="preview-card">
        <div class="preview-label">Tool Wear</div>
        <div class="preview-val" style="color:#f59e0b">198 min</div>
        <div class="preview-bar2"><div class="preview-fill" style="--w:78%;background:linear-gradient(90deg,#0d9488,#f59e0b)"></div></div>
      </div>
      <div class="preview-alert">
        <div class="preview-alert-dot"></div>
        <div>
          <div style="font-size:12px;font-weight:700;color:#f87171">Tool Wear Failure detected — TWF zone</div>
          <div style="font-size:10px;color:#94a3b8;margin-top:2px">Confidence 94% — Alert email sent to maintenance team</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- STATS BAR -->
<div class="stats-bar">
  <div class="stat-item"><div class="stat-num">98.1%</div><div class="stat-lbl" data-ilp="stat1">Recall — failures caught</div></div>
  <div class="stat-item"><div class="stat-num">5</div><div class="stat-lbl" data-ilp="stat2">Failure zones detected</div></div>
  <div class="stat-item"><div class="stat-num">&lt;5s</div><div class="stat-lbl" data-ilp="stat3">Analysis per reading</div></div>
  <div class="stat-item"><div class="stat-num">20K+</div><div class="stat-lbl" data-ilp="stat4">Industrial records trained</div></div>
</div>

<!-- HOW IT WORKS -->
<section id="how-it-works">
  <div class="section-label" data-ilp="how_lbl">Process</div>
  <div class="section-title" data-ilp="how_title">Three steps to <span>zero unplanned downtime</span></div>
  <div class="section-sub" data-ilp="how_sub">Connect your machines, let Pilar learn, and receive precise alerts before failures happen.</div>
  <div class="steps">
    <div class="step">
      <div class="step-num">01</div>
      <div class="step-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div class="step-title" data-ilp="s1t">Connect your sensors</div>
      <div class="step-desc" data-ilp="s1d">Send readings via REST API from your PLCs, SCADA systems, or CSV files. Pilar accepts temperature, speed, torque, vibration, and more.</div>
    </div>
    <div class="step">
      <div class="step-num">02</div>
      <div class="step-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
      </div>
      <div class="step-title" data-ilp="s2t">AI detects anomalies</div>
      <div class="step-desc" data-ilp="s2d">Our ML model — trained on 20,000+ industrial records — analyzes each reading across 5 failure zones in real time with 98.1% recall.</div>
    </div>
    <div class="step">
      <div class="step-num">03</div>
      <div class="step-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 9.81a19.79 19.79 0 01-3.07-8.64 2 2 0 012-2.18h3a2 2 0 012 1.72 12.05 12.05 0 00.66 2.65 2 2 0 01-.45 2.11L8.09 6.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.05 12.05 0 002.65.66A2 2 0 0122 16.92z"/></svg>
      </div>
      <div class="step-title" data-ilp="s3t">Your team gets alerted</div>
      <div class="step-desc" data-ilp="s3d">When risk exceeds threshold, Pilar sends an instant email alert with the failure zone, risk level, and recommended action — before breakdown occurs.</div>
    </div>
  </div>
</section>

<div class="full-divider"></div>

<!-- PERFORMANCE -->
<section>
  <div class="section-label" data-ilp="perf_lbl">Model Performance</div>
  <div class="section-title" data-ilp="perf_title">Built on <span>real industrial data</span></div>
  <div class="section-sub" data-ilp="perf_sub">Trained on 20,902 machine readings across multiple industrial environments — not just synthetic data.</div>
  <div class="perf-grid">
    <div class="perf-metrics">
      <div class="metric-row">
        <div class="metric-name">Recall (failures caught)</div>
        <div class="metric-bar-wrap"><div class="metric-bar" style="width:98.1%"></div></div>
        <div class="metric-val">98.1%</div>
      </div>
      <div class="metric-row">
        <div class="metric-name">Precision</div>
        <div class="metric-bar-wrap"><div class="metric-bar" style="width:89.8%"></div></div>
        <div class="metric-val">89.8%</div>
      </div>
      <div class="metric-row">
        <div class="metric-name">F1 Score</div>
        <div class="metric-bar-wrap"><div class="metric-bar" style="width:93.8%"></div></div>
        <div class="metric-val">93.8%</div>
      </div>
      <div class="metric-row">
        <div class="metric-name">Zones detected</div>
        <div class="metric-bar-wrap"><div class="metric-bar" style="width:100%"></div></div>
        <div class="metric-val">TWF HDF PWF OSF RNF</div>
      </div>
    </div>
    <div class="perf-card">
      <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text3);margin-bottom:20px">Failures missed per 1000 alerts</div>
      <div class="perf-big">19</div>
      <div class="perf-unit">out of 1000</div>
      <div class="perf-desc">Only 1.9% of real failures go undetected. In industrial maintenance, that number matters.</div>
      <div style="margin-top:24px;padding:14px;background:rgba(13,148,136,.06);border:1px solid rgba(13,148,136,.12);border-radius:8px">
        <div style="font-size:11px;color:var(--teal2);font-weight:600">Auto-improving model</div>
        <div style="font-size:11px;color:var(--text3);margin-top:4px">Pilar learns from your confirmed labels and retrains automatically on your own data.</div>
      </div>
    </div>
  </div>
</section>

<div class="full-divider"></div>

<!-- ROI -->
<section>
  <div class="section-label" data-ilp="roi_lbl">Business Impact</div>
  <div class="section-title" data-ilp="roi_title">The real cost of <span>reactive maintenance</span></div>
  <div class="section-sub" data-ilp="roi_sub">Every unplanned breakdown costs production time, emergency repairs, and team morale. Pilar changes the equation.</div>
  <div class="roi-grid">
    <div class="roi-card roi-before">
      <div class="roi-label" data-ilp="roi_before">Without Pilar</div>
      <div class="roi-item"><span class="roi-item-label">Downtime per incident</span><span class="roi-item-val">4 – 48 hours</span></div>
      <div class="roi-item"><span class="roi-item-label">Repair cost (average)</span><span class="roi-item-val">$3,000 – $50,000</span></div>
      <div class="roi-item"><span class="roi-item-label">Detection method</span><span class="roi-item-val">Machine breaks down</span></div>
      <div class="roi-item"><span class="roi-item-label">Maintenance planning</span><span class="roi-item-val">Reactive / calendar-based</span></div>
      <div class="roi-item"><span class="roi-item-label">Alert time</span><span class="roi-item-val">0 minutes</span></div>
    </div>
    <div class="roi-card roi-after">
      <div class="roi-label" data-ilp="roi_after">With Pilar</div>
      <div class="roi-item"><span class="roi-item-label">Downtime per incident</span><span class="roi-item-val">Planned → near zero</span></div>
      <div class="roi-item"><span class="roi-item-label">Repair cost (average)</span><span class="roi-item-val">Parts only — no emergency</span></div>
      <div class="roi-item"><span class="roi-item-label">Detection method</span><span class="roi-item-val">AI alert before failure</span></div>
      <div class="roi-item"><span class="roi-item-label">Maintenance planning</span><span class="roi-item-val">Predictive — data-driven</span></div>
      <div class="roi-item"><span class="roi-item-label">Alert time</span><span class="roi-item-val">Hours in advance</span></div>
    </div>
  </div>
</section>

<div class="full-divider"></div>

<!-- FEATURES -->
<section>
  <div class="section-label" data-ilp="feat_lbl">Features</div>
  <div class="section-title" data-ilp="feat_title">Everything your maintenance<br><span>team needs</span></div>
  <div class="features-grid">
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><path d="M4.93 4.93a10 10 0 000 14.14M19.07 4.93a10 10 0 010 14.14M9 12a3 3 0 106 0 3 3 0 00-6 0"/></svg></div>
      <div class="feature-title">Live Sensor Monitoring</div>
      <div class="feature-desc">Real-time analysis of temperature, speed, torque, tool wear and more. Upload CSV files or connect directly via API.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg></div>
      <div class="feature-title">5-Zone Failure Detection</div>
      <div class="feature-desc">Identifies Tool Wear (TWF), Heat Dissipation (HDF), Power Failure (PWF), Overstrain (OSF), and Random (RNF) failure modes.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 01-2.06 0L2 7"/></svg></div>
      <div class="feature-title">Instant Email Alerts</div>
      <div class="feature-desc">Automated alert emails to your maintenance team when failure probability exceeds threshold — with failure zone and risk level.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
      <div class="feature-title">Digital Twin Simulation</div>
      <div class="feature-desc">Simulate your machine behavior over 24 hours. Adjust parameters and see failure risk projections before changing settings.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4M8 14h.01M12 14h.01M16 14h.01"/></svg></div>
      <div class="feature-title">AI Maintenance Assistant</div>
      <div class="feature-desc">Ask questions about your machines in plain language. Get root cause hypotheses, maintenance recommendations, and technical guidance.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="1.5"><path d="M18 8h1a4 4 0 010 8h-1M2 8h16v9a4 4 0 01-4 4H6a4 4 0 01-4-4V8zM6 1v3M10 1v3M14 1v3"/></svg></div>
      <div class="feature-title">REST API Integration</div>
      <div class="feature-desc">Connect any PLC, SCADA, or IoT device via our documented REST API. Python examples and Curl snippets included.</div>
    </div>
  </div>
</section>

<div class="full-divider"></div>

<!-- INDUSTRIES -->
<section>
  <div class="section-label" data-ilp="ind_lbl">Industries</div>
  <div class="section-title" data-ilp="ind_title">Built for <span>industrial environments</span></div>
  <div class="section-sub" data-ilp="ind_sub">Any process that uses rotating machinery, temperature-controlled equipment, or precision tools can benefit from predictive monitoring.</div>
  <div class="industries">
    <div class="industry-chip">Chemical Processing</div>
    <div class="industry-chip">Automotive Manufacturing</div>
    <div class="industry-chip">Food & Beverage</div>
    <div class="industry-chip">Pharmaceutical</div>
    <div class="industry-chip">Plastics & Rubber</div>
    <div class="industry-chip">Metal Fabrication</div>
    <div class="industry-chip">Packaging Lines</div>
    <div class="industry-chip">Water Treatment</div>
    <div class="industry-chip">Cement & Mining</div>
    <div class="industry-chip">Paper & Printing</div>
  </div>
</section>

<div class="full-divider"></div>

<!-- PRICING -->
<section id="pricing">
  <div class="section-label" style="text-align:center" data-ilp="pricing_lbl">Pricing</div>
  <div class="section-title" style="text-align:center" data-ilp="pricing_title">Start free, <span>scale when ready</span></div>
  <div class="pricing-grid">
    <div class="plan-card">
      <div class="plan-name">Free</div>
      <div class="plan-price">$0 <span>/ forever</span></div>
      <div class="plan-desc">Get started immediately. No credit card required.</div>
      <ul class="plan-features">
        <li>Live sensor analysis</li>
        <li>Manual CSV analysis</li>
        <li>Real-time failure alerts</li>
        <li class="off">Analysis history</li>
        <li class="off">Digital Twin simulation</li>
        <li class="off">AI maintenance assistant</li>
        <li class="off">API access</li>
        <li class="off">Team collaboration</li>
      </ul>
      <a href="/register" class="plan-btn plan-btn-ghost">Get started free</a>
    </div>
    <div class="plan-card featured">
      <div class="plan-badge">Most Popular</div>
      <div class="plan-name">Starter</div>
      <div class="plan-price">$99 <span>/ month</span></div>
      <div class="plan-desc">For small teams monitoring up to 10 machines.</div>
      <ul class="plan-features">
        <li>Everything in Free</li>
        <li>Full analysis history</li>
        <li>Digital Twin simulation</li>
        <li>AI maintenance assistant</li>
        <li>REST API — 1,000 calls/day</li>
        <li>Email alerts with PDF report</li>
        <li class="off">Team collaboration</li>
        <li class="off">Custom sensor variables</li>
      </ul>
      <a href="/register" class="plan-btn plan-btn-primary">Start Starter</a>
    </div>
    <div class="plan-card">
      <div class="plan-name">Pro</div>
      <div class="plan-price">$299 <span>/ month</span></div>
      <div class="plan-desc">For industrial teams managing large fleets of machines.</div>
      <ul class="plan-features">
        <li>Everything in Starter</li>
        <li>Team collaboration (unlimited)</li>
        <li>Custom sensor variables</li>
        <li>API — 50,000 calls/day</li>
        <li>Adaptive model on your data</li>
        <li>Priority support</li>
        <li>CSV/Excel bulk import</li>
        <li>Dedicated onboarding</li>
      </ul>
      <a href="/register" class="plan-btn plan-btn-ghost">Start Pro</a>
    </div>
  </div>
</section>

<!-- FINAL CTA -->
<div style="max-width:1100px;margin:0 auto;padding:0 24px">
<div class="cta-section">
  <h2 data-ilp="cta_h">Your machines are sending signals.<br><span style="background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Are you listening?</span></h2>
  <p data-ilp="cta_p">Join the teams that stopped reacting to breakdowns and started preventing them. Free to start, no setup fees.</p>
  <a href="/register" class="btn-hero" style="display:inline-flex">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
    <span data-ilp="cta_btn">Start monitoring for free</span>
  </a>
</div>
</div>

<script>
var _TLP={
en:{nav_signin:'Sign In',nav_start:'Get Started Free',hero_badge:'AI-Powered Predictive Maintenance',hero_h:'Stop fixing machines.<br><span>Predict failures first.</span>',hero_sub:'Pilar analyzes your industrial sensors in real time and alerts you hours before a breakdown \u2014 before it costs you anything.',hero_cta1:'Start for free',hero_cta2:'See how it works',stat1:'Recall \u2014 failures caught',stat2:'Failure zones detected',stat3:'Analysis per reading',stat4:'Industrial records trained',how_lbl:'Process',how_title:'Three steps to <span>zero unplanned downtime</span>',how_sub:'Connect your machines, let Pilar learn, and receive precise alerts before failures happen.',s1t:'Connect your sensors',s1d:'Send readings via REST API from your PLCs, SCADA systems, or CSV files. Pilar accepts temperature, speed, torque, vibration, and more.',s2t:'AI detects anomalies',s2d:'Our ML model \u2014 trained on 20,000+ industrial records \u2014 analyzes each reading across 5 failure zones in real time with 98.1% recall.',s3t:'Your team gets alerted',s3d:'When risk exceeds threshold, Pilar sends an instant email alert with the failure zone, risk level, and recommended action \u2014 before breakdown occurs.',perf_lbl:'Model Performance',perf_title:'Built on <span>real industrial data</span>',perf_sub:'Trained on 20,902 machine readings across multiple industrial environments \u2014 not just synthetic data.',roi_lbl:'Business Impact',roi_title:'The real cost of <span>reactive maintenance</span>',roi_sub:'Every unplanned breakdown costs production time, emergency repairs, and team morale. Pilar changes the equation.',roi_before:'Without Pilar',roi_after:'With Pilar',feat_lbl:'Features',feat_title:'Everything your maintenance<br><span>team needs</span>',ind_lbl:'Industries',ind_title:'Built for <span>industrial environments</span>',ind_sub:'Any process that uses rotating machinery, temperature-controlled equipment, or precision tools can benefit from predictive monitoring.',pricing_lbl:'Pricing',pricing_title:'Start free, <span>scale when ready</span>',cta_h:'Your machines are sending signals.<br><span style="background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Are you listening?</span>',cta_p:'Join the teams that stopped reacting to breakdowns and started preventing them. Free to start, no setup fees.',cta_btn:'Start monitoring for free',footer_sign:'Sign In',footer_reg:'Create Account',footer_copy:'\u00a9 2026 Pilar. All rights reserved.'},
fr:{nav_signin:'Connexion',nav_start:'Commencer gratuitement',hero_badge:'Maintenance Pr\u00e9dictive par IA',hero_h:'Arr\u00eatez de r\u00e9parer.<br><span>Pr\u00e9disez les pannes.</span>',hero_sub:'Pilar analyse vos capteurs industriels en temps r\u00e9el et vous alerte des heures avant une panne \u2014 avant que \u00e7a ne co\u00fbte quoi que ce soit.',hero_cta1:'Commencer gratuitement',hero_cta2:'Voir comment \u00e7a marche',stat1:'Rappel \u2014 pannes d\u00e9tect\u00e9es',stat2:'Zones de panne identifi\u00e9es',stat3:'Analyse par mesure',stat4:'Enregistrements industriels entra\u00een\u00e9s',how_lbl:'Processus',how_title:'Trois \u00e9tapes vers <span>z\u00e9ro arr\u00eat impr\u00e9vu</span>',how_sub:'Connectez vos machines, laissez Pilar apprendre, recevez des alertes pr\u00e9cises avant les pannes.',s1t:'Connectez vos capteurs',s1d:'Envoyez des mesures via API REST depuis vos automates, SCADA ou fichiers CSV. Pilar accepte temp\u00e9rature, vitesse, couple, usure et bien plus.',s2t:"L'IA d\u00e9tecte les anomalies",s2d:"Notre mod\u00e8le ML \u2014 entra\u00een\u00e9 sur 20 000+ enregistrements industriels \u2014 analyse chaque mesure sur 5 zones de panne en temps r\u00e9el avec 98,1% de rappel.",s3t:"Votre \u00e9quipe est alert\u00e9e",s3d:"Quand le risque d\u00e9passe le seuil, Pilar envoie un email d'alerte instantan\u00e9 avec la zone de panne, le niveau de risque et l'action recommand\u00e9e \u2014 avant la panne.",perf_lbl:'Performance du mod\u00e8le',perf_title:'Bas\u00e9 sur des <span>donn\u00e9es industrielles r\u00e9elles</span>',perf_sub:'Entra\u00een\u00e9 sur 20 902 mesures machines dans plusieurs environnements industriels \u2014 pas seulement des donn\u00e9es synth\u00e9tiques.',roi_lbl:'Impact business',roi_title:'Le vrai co\u00fbt de la <span>maintenance r\u00e9active</span>',roi_sub:'Chaque arr\u00eat impr\u00e9vu co\u00fbte du temps de production, des r\u00e9parations en urgence et du moral. Pilar change la donne.',roi_before:'Sans Pilar',roi_after:'Avec Pilar',feat_lbl:'Fonctionnalit\u00e9s',feat_title:'Tout ce dont votre \u00e9quipe de<br><span>maintenance a besoin</span>',ind_lbl:'Secteurs',ind_title:'Con\u00e7u pour les <span>environnements industriels</span>',ind_sub:"Tout proc\u00e9d\u00e9 utilisant des machines tournantes, des \u00e9quipements \u00e0 temp\u00e9rature contr\u00f4l\u00e9e ou des outils de pr\u00e9cision peut b\u00e9n\u00e9ficier de la surveillance pr\u00e9dictive.",pricing_lbl:'Tarifs',pricing_title:'Commencez gratuitement, <span>montez en puissance</span>',cta_h:'Vos machines envoient des signaux.<br><span style="background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Les \u00e9coutez-vous\u00a0?</span>',cta_p:'Rejoignez les \u00e9quipes qui ont arr\u00eat\u00e9 de r\u00e9agir aux pannes et ont commenc\u00e9 \u00e0 les pr\u00e9venir. Gratuit pour commencer, sans frais de mise en place.',cta_btn:'Commencer la surveillance gratuitement',footer_sign:'Connexion',footer_reg:'Cr\u00e9er un compte',footer_copy:'\u00a9 2026 Pilar. Tous droits r\u00e9serv\u00e9s.'}
};
function _lp(l){
  localStorage.setItem('pilar_lang',l);
  var teal='#0d9488';var dim='#64748b';
  document.getElementById('_lEN').style.background=l==='en'?teal:'transparent';
  document.getElementById('_lEN').style.color=l==='en'?'#fff':dim;
  document.getElementById('_lFR').style.background=l==='fr'?teal:'transparent';
  document.getElementById('_lFR').style.color=l==='fr'?'#fff':dim;
  var tr=_TLP[l]||_TLP.en;
  document.querySelectorAll('[data-ilp]').forEach(function(el){
    var k=el.getAttribute('data-ilp');
    if(tr[k]!==undefined){el.innerHTML=tr[k];}
  });
}
(function(){_lp(localStorage.getItem('pilar_lang')||'en');})();
</script>
<!-- FOOTER -->
<footer>
  <div class="footer-logo">PILAR</div>
  <div class="footer-text">Predictive Maintenance Intelligence &mdash; Built for industrial teams worldwide</div>
  <div class="footer-links">
    <a href="/login" data-ilp="footer_sign">Sign In</a>
    <a href="/register" data-ilp="footer_reg">Create Account</a>
    <a href="/api/docs">API Docs</a>
    <a href="mailto:contact@trypilar.com">Contact</a>
  </div>
  <div style="font-size:10px;color:#334155;margin-top:16px" data-ilp="footer_copy">&copy; 2026 Pilar. All rights reserved.</div>
</footer>

</body>
</html>"""

# ── API DOCS ──────────────────────────────────────────────────────────────────
API_DOCS_HTML = _AUTH_HEAD + """
<div style="width:100%;max-width:860px;margin:0 auto;padding:0 0 60px">
  <div style="padding:32px 0 24px;border-bottom:1px solid #1e2433;margin-bottom:32px;display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase">PILAR</div>
      <div style="font-size:22px;font-weight:800;color:#e2e8f0;margin-top:6px">API Reference</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px">REST API · JSON · Authentication via API key</div>
    </div>
    <div style="text-align:right">
      <a href="/" style="font-size:10px;color:#0d9488;text-decoration:none;letter-spacing:1px">App</a>
      <span style="color:#1e2433;margin:0 8px">|</span>
      <a href="/account" style="font-size:10px;color:#0d9488;text-decoration:none;letter-spacing:1px">Account</a>
    </div>
  </div>

  <!-- AUTH -->
  <div style="margin-bottom:32px">
    <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:12px">Authentication</div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:20px">
      <p style="font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:14px">Every request must include your API key in the <code style="color:#14b8a6;background:#0a0d16;padding:2px 6px;border-radius:3px">X-Api-Key</code> header.</p>
      {% if api_key %}
      <div style="background:#0a0d16;border:1px solid #0d9488;border-radius:6px;padding:12px 16px;display:flex;align-items:center;justify-content:space-between">
        <code style="font-size:12px;color:#14b8a6">{{ api_key }}</code>
        <button onclick="navigator.clipboard.writeText('{{ api_key }}')" style="background:none;border:1px solid #1e2433;border-radius:4px;padding:4px 10px;color:#64748b;font-size:10px;cursor:pointer;letter-spacing:1px">COPY</button>
      </div>
      {% else %}
      <div style="background:#0a0d16;border:1px solid #1e2433;border-radius:6px;padding:12px 16px">
        <code style="font-size:12px;color:#475569">pk_your_api_key_here</code>
        <span style="font-size:10px;color:#64748b;margin-left:12px">→ <a href="/account" style="color:#0d9488">Get your key in Account</a></span>
      </div>
      {% endif %}
      <div style="margin-top:14px;font-size:11px;color:#64748b">Base URL: <code style="color:#94a3b8">https://trypilar.com</code></div>
    </div>
  </div>

  <!-- ENDPOINTS -->
  {% set ak = api_key if api_key else 'pk_your_api_key_here' %}

  <!-- POST /api/v1/analyze -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:rgba(13,148,136,0.12);border:1px solid #0d9488;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;color:#14b8a6;letter-spacing:1px">POST</span>
      <code style="font-size:14px;color:#e2e8f0;font-weight:700">/api/v1/analyze</code>
      <span style="font-size:11px;color:#64748b">— Single sensor reading</span>
    </div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;overflow:hidden">
      <div style="display:grid;grid-template-columns:1fr 1fr">
        <div style="padding:16px;border-right:1px solid #1e2433">
          <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin-bottom:10px">Request body</div>
          <pre style="font-size:11px;color:#94a3b8;line-height:1.7;margin:0;overflow-x:auto">{
  "type": 1,
  "temp_air": 377.0,
  "temp_process": 314.6,
  "vitesse": 1298,
  "couple": 257.8,
  "usure": 8248,
  "machine_id": "PUMP-01",
  "humidite": 65
}</pre>
        </div>
        <div style="padding:16px">
          <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin-bottom:10px">Response</div>
          <pre style="font-size:11px;color:#94a3b8;line-height:1.7;margin:0;overflow-x:auto">{
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
      <div style="padding:14px 16px;border-top:1px solid #1e2433;background:#0a0d16">
        <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin-bottom:8px">cURL</div>
        <pre style="font-size:11px;color:#14b8a6;margin:0;overflow-x:auto;white-space:pre-wrap">curl -X POST https://trypilar.com/api/v1/analyze \
  -H "X-Api-Key: {{ ak }}" \
  -H "Content-Type: application/json" \
  -d '{"type":1,"temp_air":377,"temp_process":314,"vitesse":1298,"couple":258,"usure":8248}'</pre>
      </div>
    </div>
  </div>

  <!-- POST /api/v1/analyze/batch -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:rgba(13,148,136,0.12);border:1px solid #0d9488;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;color:#14b8a6;letter-spacing:1px">POST</span>
      <code style="font-size:14px;color:#e2e8f0;font-weight:700">/api/v1/analyze/batch</code>
      <span style="font-size:11px;color:#64748b">— Up to 100 readings</span>
    </div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;overflow:hidden">
      <div style="display:grid;grid-template-columns:1fr 1fr">
        <div style="padding:16px;border-right:1px solid #1e2433">
          <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin-bottom:10px">Request body</div>
          <pre style="font-size:11px;color:#94a3b8;line-height:1.7;margin:0;overflow-x:auto">{
  "readings": [
    {"machine_id":"M1","temp_air":377,
     "vitesse":1300,"couple":260},
    {"machine_id":"M2","temp_air":390,
     "vitesse":850,"couple":420}
  ]
}</pre>
        </div>
        <div style="padding:16px">
          <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin-bottom:10px">Response</div>
          <pre style="font-size:11px;color:#94a3b8;line-height:1.7;margin:0;overflow-x:auto">{
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
      <span style="background:rgba(99,102,241,0.12);border:1px solid #6366f1;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;color:#818cf8;letter-spacing:1px">GET</span>
      <code style="font-size:14px;color:#e2e8f0;font-weight:700">/api/v1/history</code>
      <span style="font-size:11px;color:#64748b">— Recent analyses</span>
    </div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#94a3b8;line-height:2">
        <code style="color:#14b8a6">?limit=50</code> — Number of results (max 500)<br>
        <code style="color:#14b8a6">?machine_id=PUMP-01</code> — Filter by machine
      </div>
      <div style="margin-top:12px;padding:10px 14px;background:#0a0d16;border-radius:6px">
        <pre style="font-size:11px;color:#14b8a6;margin:0">curl "https://trypilar.com/api/v1/history?limit=10&machine_id=PUMP-01" \
  -H "X-Api-Key: {{ ak }}"</pre>
      </div>
    </div>
  </div>

  <!-- GET /api/v1/status -->
  <div style="margin-bottom:32px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
      <span style="background:rgba(99,102,241,0.12);border:1px solid #6366f1;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;color:#818cf8;letter-spacing:1px">GET</span>
      <code style="font-size:14px;color:#e2e8f0;font-weight:700">/api/v1/status</code>
      <span style="font-size:11px;color:#64748b">— Model info &amp; health</span>
    </div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:16px">
      <pre style="font-size:11px;color:#94a3b8;line-height:1.7;margin:0">{"ok":true,"model_name":"GradientBoosting","recall":98.1,"precision":89.8,"f1":93.8,"n_train":20902,"trained_at":"2026-03-14T06:29:50","plan":"free"}</pre>
    </div>
  </div>

  <!-- PARAMETERS TABLE -->
  <div style="margin-bottom:32px">
    <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:14px">Parameters reference</div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;overflow:hidden">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="border-bottom:1px solid #1e2433">
          <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:9px;letter-spacing:1px;text-transform:uppercase">Field</th>
          <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:9px;letter-spacing:1px;text-transform:uppercase">Unit</th>
          <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:9px;letter-spacing:1px;text-transform:uppercase">Example</th>
          <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:9px;letter-spacing:1px;text-transform:uppercase">Required</th>
        </tr></thead>
        <tbody>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>type</code></td><td style="padding:9px 14px;color:#94a3b8">0/1/2 or L/M/H</td><td style="padding:9px 14px;color:#64748b">1</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>temp_air</code></td><td style="padding:9px 14px;color:#94a3b8">K</td><td style="padding:9px 14px;color:#64748b">377.0</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>temp_air_c</code></td><td style="padding:9px 14px;color:#94a3b8">°C (auto-converted)</td><td style="padding:9px 14px;color:#64748b">103.85</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>temp_process</code></td><td style="padding:9px 14px;color:#94a3b8">K</td><td style="padding:9px 14px;color:#64748b">314.6</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>vitesse</code></td><td style="padding:9px 14px;color:#94a3b8">rpm</td><td style="padding:9px 14px;color:#64748b">1298</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>couple</code></td><td style="padding:9px 14px;color:#94a3b8">Nm</td><td style="padding:9px 14px;color:#64748b">257.8</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>usure</code></td><td style="padding:9px 14px;color:#94a3b8">min</td><td style="padding:9px 14px;color:#64748b">8248</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>machine_id</code></td><td style="padding:9px 14px;color:#94a3b8">string</td><td style="padding:9px 14px;color:#64748b">"PUMP-01"</td><td style="padding:9px 14px;color:#64748b">optional</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>humidite</code></td><td style="padding:9px 14px;color:#94a3b8">%</td><td style="padding:9px 14px;color:#64748b">65</td><td style="padding:9px 14px;color:#94a3b8">optional+</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>vibration</code></td><td style="padding:9px 14px;color:#94a3b8">mm/s</td><td style="padding:9px 14px;color:#64748b">2.1</td><td style="padding:9px 14px;color:#94a3b8">optional+</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>pression</code></td><td style="padding:9px 14px;color:#94a3b8">bar</td><td style="padding:9px 14px;color:#64748b">4.8</td><td style="padding:9px 14px;color:#94a3b8">optional+</td></tr>
          <tr style="border-bottom:1px solid #1e2433"><td style="padding:9px 14px;color:#14b8a6"><code>courant</code></td><td style="padding:9px 14px;color:#94a3b8">A</td><td style="padding:9px 14px;color:#64748b">12.4</td><td style="padding:9px 14px;color:#94a3b8">optional+</td></tr>
          <tr><td style="padding:9px 14px;color:#14b8a6"><code>tension</code></td><td style="padding:9px 14px;color:#94a3b8">V</td><td style="padding:9px 14px;color:#64748b">380</td><td style="padding:9px 14px;color:#94a3b8">optional+</td></tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top:10px;font-size:11px;color:#64748b">At least one sensor field required. Missing core fields are imputed with dataset medians.</div>
  </div>

  <!-- PYTHON EXAMPLE -->
  <div style="margin-bottom:32px">
    <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:14px">Python example</div>
    <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:20px;position:relative">
      <button onclick="navigator.clipboard.writeText(document.getElementById('pyex').innerText)" style="position:absolute;top:12px;right:12px;background:none;border:1px solid #1e2433;border-radius:4px;padding:4px 10px;color:#64748b;font-size:10px;cursor:pointer;letter-spacing:1px">COPY</button>
      <pre id="pyex" style="font-size:11px;color:#94a3b8;line-height:1.8;margin:0;overflow-x:auto">import requests

API_KEY = "{{ ak }}"
BASE    = "https://trypilar.com"

# Single reading
resp = requests.post(f"{BASE}/api/v1/analyze",
    headers={"X-Api-Key": API_KEY},
    json={"machine_id": "PUMP-01", "temp_air": 377, "vitesse": 1298,
          "couple": 258, "usure": 8248}
)
print(resp.json())

# Batch (PLC loop)
readings = [{"machine_id": f"M{i}", "vitesse": 1200+i*50, "couple": 250+i*10}
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
  <div style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;padding:20px">
    <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:12px">Rate limits</div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <tr style="border-bottom:1px solid #1e2433">
        <td style="padding:8px 0;color:#94a3b8">Free plan</td>
        <td style="padding:8px 0;color:#e2e8f0;text-align:right;font-weight:700">1 000 requests / day</td>
      </tr>
      <tr>
        <td style="padding:8px 0;color:#94a3b8">Paid plan</td>
        <td style="padding:8px 0;color:#14b8a6;text-align:right;font-weight:700">50 000 requests / day</td>
      </tr>
    </table>
    <div style="margin-top:10px;font-size:11px;color:#64748b">Resets at midnight UTC. HTTP 429 returned when exceeded.</div>
  </div>

</div>
</body></html>"""

# ── BACKEND ───────────────────────────────────────────────────────────────────
def predict_risk(params):
    # Analyse partielle : imputer les features manquantes avec les médianes AI4I
    missing_keys = [k for k in CORE_FEATURES if params.get(k) is None]
    for k in missing_keys:
        params[k] = FEATURE_MEDIANS[k]
    confidence = round((len(CORE_FEATURES) - len(missing_keys)) / len(CORE_FEATURES) * 100)
    ecart_temp = params['temp_process'] - params['temp_air']
    puissance = params['vitesse'] * params['couple']
    donnees = pd.DataFrame([[params['type'], params['temp_air'], params['temp_process'],
        params['vitesse'], params['couple'], params['usure'], ecart_temp, puissance]], columns=COLONNES)
    donnees_scaled = scaler.transform(donnees)
    probabilite = round(float(model.predict_proba(donnees_scaled)[0][1]) * 100, 1)
    prediction = 1 if probabilite >= 22 else 0
    zones_risque = []
    if prediction == 1:
        for col, nom in FAILURE_ZONES.items():
            if col in modeles_zones:
                pz = round(float(modeles_zones[col].predict_proba(donnees_scaled)[0][1]) * 100, 1)
                if pz >= 30:
                    zones_risque.append({'nom': nom, 'proba': pz})
        zones_risque.sort(key=lambda x: x['proba'], reverse=True)
    return probabilite, prediction, zones_risque, confidence, missing_keys

def envoyer_alerte(email_to, probabilite, zones_risque, data):
    machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
    mtype = machine_types.get(data.get('type', 0), 'Unknown')
    severity = "CRITICAL" if probabilite >= 75 else "HIGH"
    sc = "#dc2626"
    zones_rows = "".join(f'<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#94a3b8;font-size:12px;">{z["nom"]}</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#dc2626;font-weight:700;">{z["proba"]}%</td></tr>' for z in zones_risque) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;">No specific zone identified</td></tr>'
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;padding:40px 0;"><tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #1e2433;"><table width="100%" cellpadding="0" cellspacing="0"><tr><td><div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;">PILAR</div></td><td align="right"><span style="padding:4px 10px;background:rgba(220,38,38,0.12);border:1px solid #dc2626;border-radius:3px;color:#dc2626;font-size:10px;font-weight:700;letter-spacing:2px;">FAILURE ALERT</span></td></tr></table></td></tr>
<tr><td style="padding:28px;"><div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability</div><div style="font-size:52px;font-weight:800;color:{sc};line-height:1;">{probabilite}<span style="font-size:22px;color:#64748b;">%</span></div><div style="margin-top:8px;"><span style="padding:3px 10px;background:rgba(220,38,38,0.1);border:1px solid {sc};border-radius:3px;font-size:10px;font-weight:700;color:{sc};">SEVERITY: {severity}</span></div></td></tr>
<tr><td style="padding:0 28px 24px;"><table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;"><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Class</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{mtype}</td></tr><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Air temp</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_air")} K</td></tr><tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Speed</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("vitesse")} rpm</td></tr><tr><td style="padding:8px 12px;color:#64748b;font-size:11px;">Tool wear</td><td style="padding:8px 12px;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("usure")} min</td></tr></table></td></tr>
<tr><td style="padding:0 28px 24px;"><div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:10px;">Failure Zones</div><table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">{zones_rows}</table></td></tr>
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
            threading.Thread(target=send_verify_email, args=(email, token), daemon=True).start()
            session['_pending_verify'] = email
            return render_template_string(REGISTER_HTML, error=None, pending=True, resent=False, pending_email=email)
        session['user_id'] = user.id
        session.permanent = True
        return redirect('/monitor')
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
                    threading.Thread(target=send_verify_email, args=(email, token), daemon=True).start()
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
        if not user.email_verified:
            return render_template_string(LOGIN_HTML, error='Confirmez votre email avant de vous connecter. Vérifiez vos spams.')
        session['user_id'] = user.id
        session.permanent = True
        print(f"[Pilar/auth] Login OK: {email} IP={ip}")
        return redirect('/monitor')
    except Exception as e:
        db.session.rollback()
        print(f"[Pilar/auth] Login error: {type(e).__name__}: {e}")
        return render_template_string(LOGIN_HTML, error='Erreur serveur. Veuillez réessayer.')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

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

    return render_template_string(ADMIN_HTML, users=users, total_users=total_users,
                                  total_analyses=total_analyses, paid_users=paid_users,
                                  mrr=mrr, expiring_soon=expiring_soon, now=now)

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

@app.route('/admin/terminal', methods=['POST'])
@admin_required
def admin_terminal():
    cmd = (request.json or {}).get('cmd', '').strip()
    if not cmd:
        return jsonify({'output': '', 'code': 0})
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return jsonify({'output': output.rstrip('\n'), 'code': result.returncode})
    except subprocess.TimeoutExpired:
        return jsonify({'output': 'Timeout (30s)', 'code': -1})
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
    if current_uid():
        return redirect('/monitor')
    return LANDING_HTML

@app.route('/monitor')
def monitor(): return render_template_string(HTML)

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
    """Returns redirect response if user doesn't have paid plan, else None."""
    uid = current_uid()
    if not uid:
        return redirect('/login')
    user = db.session.get(User, uid)
    if not user:
        session.clear()
        return redirect('/login')
    if user.is_admin or user.plan in ('starter', 'pro'):
        return None
    return redirect('/upgrade')

@app.route('/upgrade')
def upgrade():
    uid = current_uid()
    user = db.session.get(User, uid) if uid else None
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Upgrade — Pilar</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui,sans-serif;background:#050d1a;color:#e2e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;text-align:center}
.card{background:#0c1526;border:1px solid #1a2a45;border-radius:16px;padding:40px 32px;max-width:420px;width:100%}
.icon{font-size:40px;margin-bottom:16px}
h2{font-size:22px;font-weight:800;letter-spacing:-0.5px;margin-bottom:8px}
p{font-size:13px;color:#64748b;line-height:1.7;margin-bottom:28px}
.features{list-style:none;margin-bottom:28px;text-align:left}
.features li{font-size:12px;color:#94a3b8;padding:6px 0;border-bottom:1px solid #1a2a45;display:flex;align-items:center;gap:8px}
.features li:last-child{border:none}
.features li::before{content:'\2713';color:#14b8a6;font-weight:700}
.btn{display:block;width:100%;padding:14px;background:#0d9488;color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;cursor:pointer;text-decoration:none;margin-bottom:10px}
.btn:hover{background:#14b8a6}
.btn-ghost{display:block;width:100%;padding:12px;background:transparent;border:1px solid #1e3050;color:#64748b;border-radius:8px;font-size:12px;cursor:pointer;text-decoration:none}
</style></head><body>
<div class="card">
<div style="width:48px;height:48px;border-radius:12px;background:rgba(13,148,136,.1);border:1px solid rgba(13,148,136,.2);display:flex;align-items:center;justify-content:center;margin:0 auto 16px"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg></div>
<h2>Upgrade to Starter</h2>
<p>This feature is available on paid plans. Upgrade to unlock the full Pilar experience.</p>
<ul class="features">
<li>Full analysis history</li>
<li>Digital Twin simulation</li>
<li>AI maintenance assistant</li>
<li>REST API access</li>
<li>Email alerts with reports</li>
</ul>
<a href="mailto:contact@trypilar.com?subject=Upgrade%20Pilar%20Starter" class="btn">Contact us to upgrade — $99/mo</a>
<a href="/monitor" class="btn-ghost">Back to Monitor</a>
</div>
</body></html>"""
    return html

@app.route('/assistant')
@login_required
def assistant():
    r = _paid_required()
    if r: return r
    return render_template_string(ASSISTANT_HTML)

@app.route('/tutorial')
@login_required
def tutorial(): return render_template_string(TUTORIAL_HTML)

@app.route('/adapter')
@auth_optional
def adapter(): return render_template_string(ADAPTER_HTML)

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
        sf = SavedFile(user_id=uid, filename=filename, content=content, row_count=row_count)
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
    files = SavedFile.query.filter_by(user_id=uid).order_by(SavedFile.created_at.desc()).limit(20).all()
    return jsonify([{'id': f.id, 'filename': f.filename, 'rows': f.row_count,
                     'created_at': f.created_at.isoformat()} for f in files])

@app.route('/api/saved_files/<int:fid>')
@auth_optional
def get_saved_file(fid):
    uid = current_uid()
    f = SavedFile.query.filter_by(id=fid, user_id=uid).first_or_404()
    return jsonify({'id': f.id, 'filename': f.filename, 'content': f.content, 'rows': f.row_count})

@app.route('/api/saved_files/<int:fid>/delete', methods=['POST'])
@auth_optional
def delete_saved_file(fid):
    uid = current_uid()
    f = SavedFile.query.filter_by(id=fid, user_id=uid).first_or_404()
    db.session.delete(f)
    db.session.commit()
    return jsonify({'ok': True})

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
    analyses = Analysis.query.filter_by(user_id=uid).order_by(Analysis.timestamp.desc()).all()
    total = len(analyses)
    anomalies = sum(1 for a in analyses if a.prediction)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1) if total > 0 else 0
    mails = sum(1 for a in analyses if a.mail_sent)
    return render_template_string(HISTORY_HTML, analyses=analyses, total=total,
                                   anomalies=anomalies, avg_risk=avg_risk, mails=mails)

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
        # Bornes physiques des capteurs
        _bounds = {'temp_air':(200,600),'temp_process':(200,700),'vitesse':(0,50000),'couple':(0,2000),'usure':(0,500000),'type':(0,3)}
        for fld, (lo, hi) in _bounds.items():
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
        probabilite, prediction, zones_risque, confidence, imputed = predict_risk(data)
        mail_envoye = False
        email = get_setting('responsible_email')
        if probabilite >= 50 and email:
            threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data), daemon=True).start()
            mail_envoye = True
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
        extra_json = _json.dumps(extra_params) if extra_params else None
        db.session.add(Analysis(machine_type=machine_types.get(int(data.get('type') or 1), 'Unknown'),
            temp_air=data.get('temp_air'), temp_process=data.get('temp_process'),
            vitesse=data.get('vitesse'), couple=data.get('couple'), usure=data.get('usure'),
            risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye,
            extra_params=extra_json, confidence=confidence, user_id=current_uid()))
        db.session.commit()
        return jsonify({'prediction': prediction, 'probabilite': probabilite,
                        'zones': zones_risque, 'mail_envoye': mail_envoye,
                        'confidence': confidence, 'imputed': imputed})
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
        history_wear  = [a.usure for a in analyses]
        history_temp  = [a.temp_process for a in analyses]
        future_times, future_risks, future_wear, future_temp = [], [], [], []
        now = datetime.utcnow()
        cu, ctp = last.usure, last.temp_process
        failure_hours = None
        for h in range(1, 25):
            cu = min(cu + 1.5, 250); ctp = min(ctp + 0.05, 315)
            risk, pred, _, _, _ = predict_risk({'type':1,'temp_air':last.temp_air,'temp_process':ctp,'vitesse':last.vitesse,'couple':last.couple,'usure':cu})
            future_times.append((now + timedelta(hours=h)).strftime('%H:%M'))
            future_risks.append(risk); future_wear.append(round(cu,1)); future_temp.append(round(ctp,2))
            if failure_hours is None and risk >= 50: failure_hours = h
        total = len(analyses)
        avg_risk = round(sum(a.risk for a in analyses) / total, 1)
        anomaly_rate = round(sum(1 for a in analyses if a.prediction) / total * 100, 1)
        trend = 'Stable'
        if len(history_risks) >= 3:
            diff = history_risks[-1] - history_risks[-3]
            trend = 'Increasing' if diff > 2 else 'Decreasing' if diff < -2 else 'Stable'
        return jsonify({'has_data':True,'current_risk':last.risk,'avg_risk_24h':avg_risk,
            'anomaly_rate':anomaly_rate,'total_analyses':total,'failure_hours':failure_hours,'trend':trend,
            'history_times':history_times,'history_risks':history_risks,'history_wear':history_wear,'history_temp':history_temp,
            'future_times':future_times,'future_risks':future_risks,'future_wear':future_wear,'future_temp':future_temp,
            'last_params':{'temp_air':last.temp_air,'vitesse':last.vitesse,'couple':last.couple,'usure':last.usure}})
    except Exception as e:
        import traceback
        print(f"[Pilar/api_twin] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Erreur serveur'}), 500

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
@auth_optional
def api_discover():
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
    limit = 50000 if plan != 'free' else 1000
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
    """Parse et convertit les champs capteurs. Retourne (params, extra, machine_id, err)."""
    import json as _json
    if not data:
        return None, None, None, (jsonify({'error': 'Empty JSON body', 'code': 'BAD_REQUEST'}), 400)
    # Support Celsius alternative keys
    if 'temp_air_c' in data:
        data['temp_air'] = float(data['temp_air_c']) + 273.15
    if 'temp_process_c' in data:
        data['temp_process'] = float(data['temp_process_c']) + 273.15
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
    _bounds = {'temp_air':(200,600),'temp_process':(200,700),'vitesse':(0,50000),'couple':(0,2000),'usure':(0,500000),'type':(0,3)}
    for fld, (lo, hi) in _bounds.items():
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
        probabilite, prediction, zones_risque, confidence, imputed = predict_risk(data)
        mail_envoye = False
        email = get_setting('responsible_email', uid=user.id if user else None)
        if probabilite >= 50 and email:
            threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data), daemon=True).start()
            mail_envoye = True
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
        a = Analysis(machine_type=machine_types.get(int(data.get('type') or 1), 'Unknown'),
            temp_air=data.get('temp_air'), temp_process=data.get('temp_process'),
            vitesse=data.get('vitesse'), couple=data.get('couple'), usure=data.get('usure'),
            risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye,
            extra_params=_json.dumps(extra) if extra else None, confidence=confidence,
            machine_id=machine_id, user_id=user.id if user else None)
        db.session.add(a)
        db.session.commit()
        return jsonify({
            'ok': True,
            'analysis_id': a.id,
            'timestamp': a.timestamp.isoformat() + 'Z',
            'machine_id': machine_id,
            'prediction': prediction,
            'risk': probabilite,
            'alert': probabilite >= 50,
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
            a = Analysis(machine_type={0:'Low',1:'Medium',2:'High'}.get(int(data.get('type') or 1),'Unknown'),
                temp_air=data.get('temp_air'), temp_process=data.get('temp_process'),
                vitesse=data.get('vitesse'), couple=data.get('couple'), usure=data.get('usure'),
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
         'zones': a.zones, 'temp_air': a.temp_air, 'temp_process': a.temp_process,
         'vitesse': a.vitesse, 'couple': a.couple, 'usure': a.usure}
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
@api_or_login_required
def api_whatif():
    try:
        params = request.json
        if not params: return jsonify({'error': 'Données manquantes'}), 400
        params['temp_process'] = params['temp_air'] + 10
        risk, pred, zones, _, _ = predict_risk(params)
        if pred == 0: status, message = 'Normal Operation', 'No failure predicted under these conditions.'
        elif risk < 50: status, message = 'Low Risk', 'Minor anomaly. Continue monitoring.'
        else: status, message = 'High Failure Risk', 'Reduce tool wear or torque immediately.'
        return jsonify({'risk':risk,'status':status,'message':message,'zones':zones})
    except Exception as e:
        print(f"[Pilar/api_whatif] ERROR: {type(e).__name__}: {e}")
        return jsonify({'error': 'Erreur serveur'}), 500

@app.route('/chat', methods=['POST'])
@api_or_login_required
def chat():
    import anthropic as _anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[Pilar/chat] ANTHROPIC_API_KEY manquante — configurez-la sur Railway")
        return jsonify({'reply': None, 'error': 'API key not configured'}), 503

    # Rate limit : 30 messages / 10 minutes par utilisateur
    _chat_uid = str(current_uid() or request.remote_addr or 'anon')
    _now_ts = time.time()
    _chat_window = _login_attempts  # réutilise la structure {key: [timestamps]}
    _chat_key = f'chat_{_chat_uid}'
    _login_attempts[_chat_key] = [t for t in _login_attempts[_chat_key] if _now_ts - t < 600]
    if len(_login_attempts[_chat_key]) >= 30:
        return jsonify({'reply': None, 'error': 'Trop de messages — réessayez dans quelques minutes'}), 429
    _login_attempts[_chat_key].append(_now_ts)

    data = request.json
    message = (data.get('message') or '').strip()[:2000]
    if not message:
        return jsonify({'reply': '', 'error': 'Empty message'}), 400

    context = data.get('context')
    chat_history = (data.get('history') or [])[-20:]  # max 20 messages d'historique

    # ── Bloc contexte machine ────────────────────────────────────────────────
    if context:
        r = context.get('result', {})
        d = context.get('data', {})
        machine_types = {0: 'Low (L)', 1: 'Medium (M)', 2: 'High (H)'}
        mtype = machine_types.get(d.get('type', 0), 'Unknown')
        zones_str = ', '.join([f"{z['nom']} ({z['proba']}%)" for z in r.get('zones', [])]) or 'aucune zone identifiée'
        status_str = 'ANOMALIE DÉTECTÉE' if r.get('prediction') else 'Fonctionnement normal'
        ctx_block = f"""
=== DERNIÈRE ANALYSE MACHINE ===
Statut      : {status_str}
Risque      : {r.get('probabilite')}%
Classe      : {mtype}
Temp. air   : {d.get('temp_air')} K  |  Temp. process : {d.get('temp_process')} K
Vitesse     : {d.get('vitesse')} rpm |  Couple         : {d.get('couple')} Nm
Usure outil : {d.get('usure')} min
Zones risque: {zones_str}
================================
"""
    else:
        ctx_block = "\n[Aucune analyse machine disponible. Si l'utilisateur pose une question sur sa machine, invite-le à lancer une analyse depuis l'onglet Monitor.]\n"

    # ── System prompt expert maintenance ─────────────────────────────────────
    system_prompt = f"""Tu es Pilar, un assistant IA expert en maintenance prédictive industrielle, intégré dans une plateforme SaaS B2B pour PME industrielles.
{ctx_block}
Directives :
- Tu es un expert technique : capteurs, thermique, vibrations, usure, défaillances mécaniques et électriques
- Si des données d'analyse sont disponibles, analyse-les précisément et donne des recommandations concrètes et actionnables
- Si l'utilisateur décrit un symptôme machine, propose un diagnostic différentiel et des actions correctives priorisées
- Réponds en français si l'utilisateur écrit en français, en anglais sinon — détecte automatiquement la langue
- Réponses concises, structurées et techniques — ton ingénieur de maintenance expérimenté
- Ne dis jamais que tu ne peux pas répondre — donne toujours une réponse utile et directe
- Pour les questions hors maintenance, réponds quand même de façon complète"""

    # ── Historique : chat_history inclut le message courant en dernier ────────
    messages = [{"role": h['role'], "content": h['content']} for h in chat_history[:-1]]
    messages.append({"role": "user", "content": message})

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        reply = response.content[0].text
        print(f"[Pilar/chat] OK — {len(reply)} chars")
        return jsonify({'reply': reply})
    except _anthropic.AuthenticationError as e:
        print(f"[Pilar/chat] Auth error: {e}")
        return jsonify({'reply': None, 'error': 'Invalid API key'}), 401
    except _anthropic.RateLimitError as e:
        print(f"[Pilar/chat] Rate limit: {e}")
        return jsonify({'reply': None, 'error': 'Rate limit reached'}), 429
    except Exception as e:
        print(f"[Pilar/chat] Error {type(e).__name__}: {e}")
        return jsonify({'reply': None, 'error': 'Erreur serveur'}), 500


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
        "theme_color": "#0e1118",
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
<style>body{{font-family:sans-serif;background:#07090f;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.c{{max-width:420px;text-align:center;padding:40px;}}.logo{{font-size:13px;letter-spacing:4px;color:#14b8a6;font-weight:700;}}.msg{{color:#94a3b8;font-size:13px;margin:16px 0 24px;line-height:1.7;}}
a{{padding:12px 24px;background:#0d9488;color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:2px;}}</style></head>
<body><div class="c"><div class="logo">PILAR</div><h2 style="margin:20px 0 8px;font-size:18px;">Erreur serveur</h2>
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Pilar v3 — http://localhost:{port} (debug={debug})")
    app.run(debug=debug, host='0.0.0.0', port=port)