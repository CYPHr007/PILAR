from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, g
import pickle, threading, smtplib, secrets as _secrets
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
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=90)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
# HTTPS only en production (Railway), désactivé en local pour dev
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("RAILWAY_ENVIRONMENT") is not None
db = SQLAlchemy(app)

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
        ]
    else:
        _migrations = [
            "ALTER TABLE analysis ADD COLUMN IF NOT EXISTS user_id INTEGER",
            "ALTER TABLE settings ADD COLUMN IF NOT EXISTS user_id INTEGER",
            "ALTER TABLE settings DROP CONSTRAINT IF EXISTS settings_key_key",
            'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS team_id INTEGER',
        ]
    for sql in _migrations:
        try:
            db.session.execute(db.text(sql))
            db.session.commit()
            print(f"[Pilar] Migration OK: {sql[:50]}")
        except Exception as e:
            db.session.rollback()
            print(f"[Pilar] Migration skip ({sql[:40]}): {e}")

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
        api_key = request.headers.get('X-Api-Key') or request.args.get('api_key')
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
    base = os.environ.get("APP_URL", "http://localhost:5000")
    link = f"{base}/verify-email/{token}"
    html = f"""<div style="font-family:sans-serif;background:#07090f;color:#e2e8f0;padding:40px;border-radius:8px">
<h2 style="color:#14b8a6;letter-spacing:3px">PILAR</h2>
<p>Confirmez votre adresse email pour activer votre compte.</p>
<a href="{link}" style="display:inline-block;margin-top:16px;padding:12px 24px;background:#0d9488;color:#fff;border-radius:6px;text-decoration:none;font-weight:700">Vérifier mon email</a>
<p style="margin-top:24px;color:#64748b;font-size:12px">Lien valide 24h. Si vous n'avez pas créé de compte, ignorez cet email.</p>
</div>"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Pilar — Vérifiez votre email"
    msg['From'] = f"Pilar <{GMAIL}>"
    msg['To'] = email
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email, msg.as_string())
        print(f"[Pilar/auth] Verification email sent to {email}")
    except Exception as e:
        print(f"[Pilar/auth] Email error: {e}")


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
</style></head><body>"""

LOGIN_HTML = _AUTH_HEAD + """
<div class="ac">
  <div class="logo">PILAR</div>
  <div class="card">
    <div class="ctitle">Connexion</div>
    {% if error %}<div class="err" style="display:block">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
      <label class="flbl" for="em">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="••••••••" autocomplete="current-password" required>
      <button type="submit" class="btn">Se connecter</button>
    </form>
    <a href="/" class="btn" style="display:block;text-align:center;text-decoration:none;background:transparent;border:1px solid #252d3d;color:#64748b;margin-top:8px;padding:13px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase">Continuer sans compte</a>
  </div>
  <div class="link">Pas encore de compte ? <a href="/register">Créer un compte</a></div>
</div>
</body></html>"""

REGISTER_HTML = _AUTH_HEAD + """
<div class="ac">
  <div class="logo">PILAR</div>
  <div class="card">
    <div class="ctitle">Créer un compte</div>
    {% if error %}<div class="err" style="display:block">{{ error }}</div>{% endif %}
    <form method="POST" action="/register">
      <label class="flbl" for="em">Email professionnel</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="8 caractères minimum" autocomplete="new-password" required minlength="8">
      <label class="flbl" for="pw2">Confirmer le mot de passe</label>
      <input class="fi" type="password" id="pw2" name="password2" placeholder="••••••••" autocomplete="new-password" required>
      <button type="submit" class="btn">Créer mon compte</button>
    </form>
  </div>
  <div class="link">Déjà un compte ? <a href="/login">Se connecter</a></div>
</div>
</body></html>"""

ADMIN_HTML = _AUTH_HEAD + """
<div style="width:100%;max-width:800px;">
  <div class="logo">PILAR — Admin</div>
  <div class="card" style="margin-bottom:16px">
    <div class="ctitle">Vue d'ensemble</div>
    <div class="kgrid">
      <div class="kc"><div class="kv">{{ total_users }}</div><div class="kl">Utilisateurs</div></div>
      <div class="kc"><div class="kv">{{ total_analyses }}</div><div class="kl">Analyses</div></div>
      <div class="kc"><div class="kv" style="color:#d97706">{{ unverified }}</div><div class="kl">Non vérifiés</div></div>
    </div>
  </div>
  <div class="card">
    <div class="ctitle">Utilisateurs</div>
    <table>
      <thead><tr><th>Email</th><th>Plan</th><th>Vérifié</th><th>Inscrit</th><th>Analyses</th><th>Actions</th></tr></thead>
      <tbody>
      {% for u in users %}
      <tr>
        <td>{{ u.email }}{% if u.is_admin %} <span class="badge ok">admin</span>{% endif %}</td>
        <td><span class="badge free">{{ u.plan }}</span></td>
        <td>{% if u.email_verified %}<span class="badge ok">✓</span>{% else %}<span style="color:#dc2626">✗</span>{% endif %}</td>
        <td>{{ u.created_at.strftime('%d/%m/%Y') }}</td>
        <td>{{ u.analysis_count }}</td>
        <td><a href="/admin/impersonate/{{ u.id }}" style="color:#14b8a6;font-size:11px">Voir</a></td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  <div class="link" style="margin-top:16px"><a href="/">← Retour à l'app</a> · <a href="/logout">Déconnexion</a></div>
</div>
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
ast_hello:"Bonjour. Je suis votre assistant maintenance prédictive. Partagez vos relevés capteurs ou posez-moi vos questions."},
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
ast_hello:'Hello. I am your predictive maintenance assistant. Share your sensor readings or ask me anything about your machine health.'}
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
</script></head>"""

_NAV = """<nav class="bottom-nav">
<a href="/" class="ni {m}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg><span data-i18n="nav_monitor">Monitor</span></a>
<a href="/tutorial" class="ni {tut}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg><span data-i18n="nav_import">Import</span></a>
<a href="/history" class="ni {h}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg><span data-i18n="nav_history">History</span></a>
<a href="/account" class="ni {a}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg><span data-i18n="nav_account">Account</span></a>
<a href="/settings" class="ni {s}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg><span data-i18n="nav_settings">Settings</span></a>
</nav>"""

def nav(active):
    keys = {"m":"","tut":"","h":"","a":"","s":""}
    keys[active] = "on"
    return _NAV.format(**keys)


# ── MONITOR ───────────────────────────────────────────────────────────────────
HTML = _HEAD.replace("{FAV}","iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAHxUlEQVR4nO2Za4wT1xXHz713PLbHj8Xe2V17H973C3YXNgQKIWlE2rJSFDWCqlUkmoeafilVUrWVKqVqxIe0EoqqNIqqqEraokStRBIR2kDDpoFNgeUN5rE89gVre9/s+v0az8y9tx9MUSlSpNiTOkj+f7Vn7vndc+45555BnHO4n4VLbUCxKgOUWmWAUqsMcI/+z3m57IF7hBAy/J2fo7IHSq0yQKklGPs6zvnnp1GEkLGnHN3v7bRhHuAcAPhEcGY5EjMJwl2O4AAACIGmUdld0dHcAMZlW8MAGGOE4OP+kXf3Dq5wOChj//UjRwghhOPJ1DNbt3S2+ChlhHyVADjnGKNYMjU1s1AluzWdEwKEYH577xGlDCGQ5crQ3GIylbbbJM65IU4wBoAxTgh+ffcHr779gd0mbVi9srvFG01kzKIJAOU0ze2wXRoL+K/f/Me/ThPB9KsdTxvlBCOzEKVMIKKSo51NtVs2PTA2NSuaBIyxklNXtjaoOj1+YVQQRKpTAxc1Jgvl42F+KfzL377dWO+VLKblaCKr6ACIYGCAHBaz7HZGE6m5W8u/+fkPaypdRoWQYQCM80+Pn9u0tschSb//677RqZlwJC5gwWQSVF1b4XD2dzQ+/9QTiWTy1KWxb25aiwxKRAaEEGMcY3Ru5JpHdjkkCQDSioZz6c0dVVUuGyAUiWcvBpZimSzn3Olw1MgVl66N96/qzD9Y5OoGtBIYI0ppOqOs6W5XUrFkJmvOxQb6W5tX9sxMB65dverr6NncW19r44qqZTLJnraGcCyhalrx1kPxIcQ5p5SevXw9mkh2tzUtBG/MxZQ6Gz0ZSA4fGtz+1Danzb5797u9D20e6PMGYrytXvZ53KGYbhVNbY31vOhAMiCEOKDL4wGft2opmrRU+Tw0EM3k5sdHHnl4/Ww0O72U2rhpw8TElekG1zce7HHWNAKAC5ZvLUcBIc54kQehWACE0Gxg4uLhD70DW1qbXIoS1SrI6aHD69eur27u9Pv9lEPf2nUNtZ7Jc8N9dRWpRBxjnEsl4xNj8xL3NLYXaUCxAIzzWl/zIwPb2nu7K+qa3BilFkOuxq7J8bGcIGlgooxN3QxFgqPe9t6UjjDYgbLlTGaBVG2saShydTDgEHMQRfOaNT0OlzubWJ4Khk5cC6zuaZ9RzMcOHujvaunvbPF/NjiVFNubG89Nzk+FQqlkLJ7VzTY7MZmKByj+EANCML+4FJhbXNvVGImnMxoMDf69wec7fXXWmp0XBRwmlQ/3tcyFgg9963HZit2yyz863e6rddhtxZez4s8AcM69NVUjYzc0ED0eBwDsCiQjsavrulqRKnFGRXvFGf+FUEZ8rrEOAOKpzMKtcH93O+McF13LDOuF+rpbj5zxP/7oRs65xSxGsO2t/aeavJUY4+Dclda2Douo5C0ePn95XW8nQrfvCUXKgEKGEOKce6qq6jzV/xw+CwDprMKoblshh7M8qmLJWcm5nlYUBPDJsTM+b031V60XAoD87o6M31CyuY8ODS9GEppGMcEYI13NWSxSTaVzy6Nfc9qkno4WQ4InL8NCCCPEGOvtaL0RCK72mHW5wiSYKKMCIgiBpmlEFLyVFc2+BsYYxoZNQ4ydSiAA+Ntnp0/4A06HnVLKGUMIYYwxwrFkcpGf37HdgNx/15JGhRCljBD8uz+//4f3Pna7nbqq5dsczhniCACZTEI4lnjh6Sd//P2tlFJCiCHrFuiBO0OHO209QsA42/jAKqfDbhZN/9Mq58dBSi7X19UCAAjje99QmAzwQPH5pJiLwRf2QL703gpHUhlFVVVMUEdTIwBMzy9qOrWazQ67dX4pDAAe2S1ZrBx4PuPruq6qms0mEYwppcG5BQCUzmTqPdWuCmdh1kNBdYADgKrp33th57FzF/cePLLj5dcopdcnA99+/qWlSIxg/Mob7+x68y8EY4SAEAKABEL27B/a+fofCcaargPA/kMnfvCLXcHZxe0/+/WBoeMAwO4aJX1pAPloqfdUW0XxwVUrX/rRM/sGjwZn5vq626xWsaWh1mqx1MqyV3ZLVivjHOWvbIxdGr9x+NTVaDwhEEIIaWnwWkXzE49t6u1q/9P7H8Pt2d6XD5BXPu5jqfT+oZNtrT5PtZxIZTAW8rurc67z2/+jjCGEzl8Ze/KxjT1tvvcODOW3QKdUUbVjZy8vLsVefO67BR+kwguKxWKeuxXWNG3PGzslq1XX6Z3yhDkngBnnhBCCMQAcOeUfnQytcNo/OnwS/tMEWSzChWsTN0PTmzes4QCFJYICARBC8US63itvG/h6XXUlAJgEklFyBBMAUDlTqI4RWliOnPCPRONxIpi2bx3Y+ZNnA7MLR09fRAA6pbpOX3z2O3KFY8fLr2FUYD78wgD5ZSaDs5WyPBmcUzVN1ykAjAZmK2XXeDCUTKdzOo0l0u/sPfjqW3sYhw8/Pc4BOWySZDX3dbXuOzQciceWokmzVQrNL775yk+XwrF9nxzFGBfAUCC3klNF0ZTLqRazmI/djJKzmEVV1QjB+W4nnVFE0SRZzJlsDmNkMYuqphOCKaUAwDkIAlE1zWo2A0Aqk7VL1gIsKf0HjrwBBZfC0gMUqfv+I18ZoNQqA5RaZYBSqwxQapUBSq0yQKl13wP8GxwKx1pBe9uwAAAAAElFTkSuQmCC") + """
<body>
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub" data-i18n="page_monitor">Monitor</span>
<div class="hright">
  <label for="lfInput" class="nb" id="btnFile" style="cursor:pointer">
    <svg style="width:13px;height:13px;vertical-align:middle;stroke:currentColor;fill:none;margin-right:3px" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-width="2"/></svg>
    <span id="btnFileLbl" data-i18n="live_connect">Connect File</span>
  </label>
  <input type="file" id="lfInput" accept=".csv" style="display:none" onchange="onLiveFile(this)">
  <button class="nb" id="nb" onclick="toggleN()">Notifs</button>
</div></header>
<div class="page pad">
  <div class="ab" id="abn">Alert dispatched</div>

  <!-- LIVE FILE STATUS BAR -->
  <div id="liveBar" style="display:none;background:var(--surface);border:1px solid var(--teal);border-radius:8px;padding:12px 14px;margin-bottom:12px">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:8px">
        <span style="width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;animation:blink 1.5s infinite"></span>
        <span style="font-size:11px;color:var(--teal-light);font-weight:600" id="liveFileName">—</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <select id="liveIntv" style="background:var(--surface2);border:1px solid var(--border2);border-radius:4px;color:var(--text2);font-size:10px;padding:4px 6px;outline:none" onchange="resetLiveTimer()">
          <option value="2000">2s</option>
          <option value="5000" selected>5s</option>
          <option value="10000">10s</option>
          <option value="30000">30s</option>
        </select>
        <span style="font-size:10px;color:var(--text3)" id="liveChk"></span>
        <button onclick="stopLiveMonitor()" style="padding:4px 10px;background:transparent;border:1px solid var(--border2);border-radius:4px;color:var(--text3);font-size:10px;cursor:pointer">Disconnect</button>
      </div>
    </div>
    <div style="margin-top:8px;display:flex;gap:12px;font-size:10px;color:var(--text3)">
      <span><span data-i18n="live_rows">Rows read</span>: <strong style="color:var(--text2)" id="liveRowCount">0</strong></span>
      <span><span data-i18n="live_fail">Failures</span>: <strong style="color:var(--red)" id="liveFailCount">0</strong></span>
      <span><span data-i18n="live_ok">Normal</span>: <strong style="color:var(--green)" id="liveOkCount">0</strong></span>
    </div>
  </div>

  <div id="res"><div class="idle"><span class="l1" data-i18n="idle_l1">No analysis yet</span><span class="l2" data-i18n="idle_l2">Configure below and run</span></div></div>
  <div class="card">
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
</div>""" + nav("m") + """
<script>
let mT=0,lastR=null,lastD=null;
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
      document.getElementById('res').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Erreur serveur (réponse non-JSON, code '+res.status+')</span></div>';
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
    document.getElementById('res').innerHTML='<div class="idle"><span class="l1" style="color:#dc2626">Erreur réseau: '+err.message+'</span></div>';
  }
  btn.disabled=false;btn.textContent=t('run_btn');
}
function render(r){
  const al=r.prediction===1,cls=al?'alert':'ok',st=al?t('status_alert'):t('status_ok');
  let zH='';
  if(al&&r.zones.length>0){zH='<div class="card"><div class="ctitle">'+t('zone_title')+'</div>'+r.zones.map(z=>'<div class="zrow"><span class="zname">'+z.nom+'</span><div class="zbw"><div class="zbf" style="width:'+z.proba+'%"></div></div><span class="zp">'+z.proba+'%</span></div>').join('')+'</div>';}
  document.getElementById('res').innerHTML='<div class="rh '+cls+'"><div><div class="sb '+cls+'"><span class="dot '+cls+'"></span>'+st+'</div><div style="font-size:10px;color:var(--text3);margin-top:4px">'+localTimeNow()+'</div></div><div><div class="rnum '+cls+'">'+r.probabilite+'<span class="runit">%</span></div><div class="rlbl">'+t('failure_prob')+'</div></div></div>'+zH;
}

// ── LIVE FILE MONITOR ─────────────────────────────────────────────────────
var _lfHandle=null,_lfTimer=null,_lfKnown=0,_lfFail=0,_lfOk=0,_lfHdr=null;

function onLiveFile(inp){
  var f=inp.files[0];if(!f)return;
  _lfKnown=0;_lfFail=0;_lfOk=0;_lfHdr=null;clearTimeout(_lfTimer);
  document.getElementById('liveFileName').textContent=f.name;
  document.getElementById('liveBar').style.display='block';
  document.getElementById('btnFileLbl').textContent=t('live_on');
  document.getElementById('btnFile').classList.add('on');
  document.getElementById('liveRowCount').textContent='0';
  document.getElementById('liveFailCount').textContent='0';
  document.getElementById('liveOkCount').textContent='0';
  // Try to get a live handle for auto-polling (Chrome/Edge)
  if(window.showOpenFilePicker){
    _connectWithAPI(f.name);
  }else{
    // Fallback: read snapshot + show refresh button
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
    // User dismissed or blocked — fall back to snapshot mode
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
  var lines=text.split('\n').map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
  if(lines.length<2)return;
  if(!_lfHdr){
    _lfHdr=lines[0].split(',').map(function(s){return s.trim().toLowerCase();});
    var needed=['type','temp_air','temp_process','vitesse','couple','usure'];
    for(var i=0;i<needed.length;i++){
      if(_lfHdr.indexOf(needed[i])<0){
        document.getElementById('liveChk').textContent='Error: missing column '+needed[i];
        stopLiveMonitor();return;
      }
    }
  }
  var total=lines.length-1;
  document.getElementById('liveRowCount').textContent=total;
  if(total>_lfKnown){
    var newLines=lines.slice(_lfKnown+1);
    var lastRow=null;
    newLines.forEach(function(line){
      var vals=line.split(',');
      var obj={};
      _lfHdr.forEach(function(col,ci){obj[col]=parseFloat(vals[ci]);});
      if(!isNaN(obj.type)&&!isNaN(obj.temp_air))lastRow=obj;
    });
    _lfKnown=total;
    if(lastRow){
      try{
        var res=await fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastRow)});
        var r=await res.json();
        if(!r.error){
          if(r.prediction===1){_lfFail++;if(r.probabilite>=50){sendN(r.probabilite,r.zones);var a=document.getElementById('abn');a.style.display='block';setTimeout(function(){a.style.display='none';},4000);}}
          else _lfOk++;
          document.getElementById('liveFailCount').textContent=_lfFail;
          document.getElementById('liveOkCount').textContent=_lfOk;
          lastR=r;render(r);
          localStorage.setItem('pilar_last_result',JSON.stringify(r));
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
  clearTimeout(_lfTimer);_lfHandle=null;_lfHdr=null;
  document.getElementById('liveBar').style.display='none';
  document.getElementById('btnFileLbl').textContent=t('live_connect');
  document.getElementById('btnFile').classList.remove('on');
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
  <div style="font-size:32px;margin-bottom:12px">👤</div>
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
</script></body></html>"""

# ── TWIN ──────────────────────────────────────────────────────────────────────
TWIN_HTML = _HEAD.replace("{FAV}","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") + """
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
        <span class="flag">🇬🇧</span><span class="lname">English</span>
      </div>
      <div class="lcard" data-lang="fr" onclick="setLang('fr')">
        <span class="flag">🇫🇷</span><span class="lname">Français</span>
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
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)">Domain</span><span>trypilar.com</span></div>
    </div>
  </div>
  <div class="card">
    <div class="ctitle">Navigation</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <a href="/twin" style="display:flex;justify-content:space-between;font-size:12px;color:var(--text2);text-decoration:none;padding:8px 0"><span>Digital Twin Simulation</span><span style="color:var(--teal-light)">›</span></a>
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
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub">Assistant</span>
<div class="hright"><a href="/account" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px">Account</a></div>
</header>
<div class="cw">
  <div class="cm" id="cm">
    <div class="msg bot"><span class="ms">Pilar AI</span><div class="mb2" id="ast-hello">Hello. I am your predictive maintenance assistant. Share your sensor readings or ask me anything about your machine health.</div></div>
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
applyLang();
function kd(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function addMsg(role,txt){
  var d=document.createElement('div');d.className='msg '+role;
  var s=document.createElement('span');s.className='ms';s.textContent=role==='user'?'You':'Pilar AI';
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
    if(r.error){tb.textContent='Error: '+r.error;}
    else{tb.textContent=r.reply;hist.push({role:'assistant',content:r.reply});}
  }catch(e){tb.classList.remove('typing');tb.textContent='Network error. Please retry.';}
  sb.disabled=false;ci.focus();
}
</script></body></html>"""

# ── TUTORIAL ───────────────────────────────────────────────────────────────────
TUTORIAL_HTML = _HEAD.replace("{FAV}", FAV_B64) + """
<body>
<header><span class="logo">PILAR</span><div class="hd"></div><span class="hsub">Import & Run</span>
<div class="hright"><a href="/twin" style="font-size:10px;color:var(--text3);text-decoration:none;letter-spacing:1px">Twin</a></div>
</header>
<div class="page pad">

  <!-- CSV FORMAT GUIDE -->
  <div class="card">
    <div class="ctitle">CSV Format</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin-bottom:12px">Your CSV must contain these columns (order does not matter):</p>
    <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:6px;padding:12px;overflow-x:auto;margin-bottom:12px">
      <code style="font-size:11px;color:var(--teal-light);white-space:nowrap">type, temp_air, temp_process, vitesse, couple, usure</code>
    </div>
    <table style="font-size:11px;margin-top:0">
      <thead><tr><th>Column</th><th>Unit</th><th>Range</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td style="color:var(--teal-light)">type</td><td>—</td><td>0, 1 or 2</td><td>Machine class (L=0, M=1, H=2)</td></tr>
        <tr><td style="color:var(--teal-light)">temp_air</td><td>K</td><td>295–310</td><td>Air temperature (Kelvin)</td></tr>
        <tr><td style="color:var(--teal-light)">temp_process</td><td>K</td><td>305–320</td><td>Process temperature (Kelvin)</td></tr>
        <tr><td style="color:var(--teal-light)">vitesse</td><td>rpm</td><td>1000–3000</td><td>Rotational speed</td></tr>
        <tr><td style="color:var(--teal-light)">couple</td><td>Nm</td><td>3–80</td><td>Torque</td></tr>
        <tr><td style="color:var(--teal-light)">usure</td><td>min</td><td>0–250</td><td>Tool wear time</td></tr>
      </tbody>
    </table>
    <button class="btn" style="margin-top:14px;background:transparent;border:1px solid var(--border2);color:var(--text2)" onclick="dlSample()">Download sample CSV</button>
  </div>

  <!-- IMPORT -->
  <div class="card">
    <div class="ctitle">Import File</div>
    <label for="csvf" style="display:block;padding:24px;border:1px dashed var(--border2);border-radius:6px;text-align:center;cursor:pointer;background:var(--surface2);transition:border-color 0.15s" id="dropz">
      <svg style="width:32px;height:32px;stroke:var(--text3);margin-bottom:8px" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
      <div style="font-size:12px;color:var(--text3)">Click to select a CSV file</div>
      <div style="font-size:10px;color:var(--text3);margin-top:4px" id="fname">No file selected</div>
    </label>
    <input type="file" id="csvf" accept=".csv" style="display:none" onchange="onFile(this)">
    <div style="display:flex;align-items:center;gap:10px;margin-top:14px">
      <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase">Speed</div>
      <select id="spd" class="fi" style="flex:1;padding:8px 10px">
        <option value="500">Fast — 0.5s per row</option>
        <option value="1000" selected>Normal — 1s per row</option>
        <option value="2000">Slow — 2s per row</option>
        <option value="5000">Real-time — 5s per row</option>
      </select>
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn" id="btnStart" style="flex:1" onclick="startRun()" disabled>Start</button>
      <button class="btn" id="btnPause" style="flex:1;background:var(--amber);display:none" onclick="togglePause()">Pause</button>
      <button class="btn" id="btnStop" style="flex:1;background:var(--red);display:none" onclick="stopRun()">Stop</button>
    </div>
  </div>

  <!-- PROGRESS -->
  <div class="card" id="progCard" style="display:none">
    <div class="ctitle">Progress</div>
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
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px">Failures</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--green)" id="cntOk">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px">Normal</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--amber)" id="cntAvg">—</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px">Avg Risk</div>
      </div>
    </div>
  </div>

  <!-- LIVE RESULT -->
  <div id="liveRes"></div>

  <!-- LIVE FILE MONITOR -->
  <div class="card" style="margin-top:8px">
    <div class="ctitle">Live File Monitor</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:14px">Connect a CSV file that your SCADA or system updates continuously. Pilar detects new rows and analyses them automatically.</p>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <div>
        <div style="font-size:12px;color:var(--text2)" id="liveName">No file connected</div>
        <div style="font-size:10px;margin-top:3px" id="liveStatus" style="color:var(--text3)">Disconnected</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <div style="font-size:10px;color:var(--text3)" id="liveTotalRows"></div>
        <button class="btn" id="btnLiveConnect" onclick="connectLive()" style="width:auto;padding:10px 16px;margin-top:0">Connect File</button>
        <button class="btn" id="btnLiveStop" onclick="stopLive()" style="width:auto;padding:10px 16px;margin-top:0;background:var(--red);display:none">Disconnect</button>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;white-space:nowrap">Check every</div>
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
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px">Failures</div>
      </div>
      <div style="flex:1;background:var(--surface2);border-radius:6px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:800;color:var(--green)" id="liveOk">0</div>
        <div style="font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-top:2px">Normal</div>
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
    var lines=e.target.result.split('\\n').map(s=>s.trim()).filter(s=>s.length>0);
    if(lines.length<2){alert('File must have at least 2 rows (header + data).');return;}
    var hdr=lines[0].split(',').map(s=>s.trim().toLowerCase());
    var needed=['type','temp_air','temp_process','vitesse','couple','usure'];
    for(var i=0;i<needed.length;i++){
      if(hdr.indexOf(needed[i])<0){alert('Missing column: '+needed[i]);return;}
    }
    rows=[];
    for(var r=1;r<lines.length;r++){
      var vals=lines[r].split(',');
      var obj={};
      hdr.forEach(function(col,ci){obj[col]=parseFloat(vals[ci]);});
      if(!isNaN(obj.type)&&!isNaN(obj.temp_air))rows.push(obj);
    }
    document.getElementById('fname').textContent=f.name+' — '+rows.length+' rows loaded';
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
var liveHandle=null, liveTimer=null, liveKnownRows=0, liveFail=0, liveOk=0;

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
    var lines=text.split('\n').map(s=>s.trim()).filter(s=>s.length>0);
    if(lines.length<2){liveTimer=setTimeout(liveCheck,parseInt(document.getElementById('liveInterval').value));return;}
    var hdr=lines[0].split(',').map(s=>s.trim().toLowerCase());
    var needed=['type','temp_air','temp_process','vitesse','couple','usure'];
    for(var i=0;i<needed.length;i++){if(hdr.indexOf(needed[i])<0){
      document.getElementById('liveStatus').textContent='Error: missing column '+needed[i];
      document.getElementById('liveStatus').style.color='var(--red)';return;
    }}
    var total=lines.length-1;
    if(total>liveKnownRows){
      var newLines=lines.slice(liveKnownRows+1);
      for(var r=0;r<newLines.length;r++){
        var vals=newLines[r].split(',');
        var obj={};
        hdr.forEach(function(col,ci){obj[col]=parseFloat(vals[ci]);});
        if(!isNaN(obj.type)&&!isNaN(obj.temp_air)){
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
  clearTimeout(liveTimer);liveHandle=null;
  document.getElementById('liveStatus').textContent='Disconnected';
  document.getElementById('liveStatus').style.color='var(--text3)';
  document.getElementById('btnLiveStop').style.display='none';
  document.getElementById('btnLiveConnect').style.display='inline-block';
  document.getElementById('liveName').textContent='No file connected';
}
</script></body></html>"""

# ── BACKEND ───────────────────────────────────────────────────────────────────
def predict_risk(params):
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
    return probabilite, prediction, zones_risque

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
        if current_uid(): return redirect('/')
        return render_template_string(REGISTER_HTML, error=None)
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    if _check_rate_limit(ip):
        print(f"[Pilar/auth] Rate limit register IP={ip}")
        return render_template_string(REGISTER_HTML, error='Trop de tentatives. Réessayez dans 15 minutes.')
    try:
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password', '')
        password2 = request.form.get('password2', '')
        if not email or not password:
            return render_template_string(REGISTER_HTML, error='Email et mot de passe requis')
        if len(password) < 8:
            return render_template_string(REGISTER_HTML, error='Mot de passe trop court (8 caractères minimum)')
        if password != password2:
            return render_template_string(REGISTER_HTML, error='Les mots de passe ne correspondent pas')
        if User.query.filter_by(email=email).first():
            _record_failed_login(ip)
            return render_template_string(REGISTER_HTML, error='Un compte existe déjà avec cet email')
        api_key = 'pk_' + _secrets.token_hex(24)
        is_admin = (email == os.environ.get('ADMIN_EMAIL', '').lower()) or (User.query.count() == 0)
        user = User(email=email, password_hash=generate_password_hash(password, method='pbkdf2:sha256:600000'),
                    email_verified=True, api_key=api_key, is_admin=is_admin)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        session.permanent = True
        print(f"[Pilar/auth] New user: {email} (admin={is_admin}) IP={ip}")
        return redirect('/')
    except Exception as e:
        db.session.rollback()
        print(f"[Pilar/auth] Register error: {type(e).__name__}: {e}")
        return render_template_string(REGISTER_HTML, error='Erreur serveur. Veuillez réessayer.')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if current_uid(): return redirect('/')
        return render_template_string(LOGIN_HTML, error=None)
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
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
        session['user_id'] = user.id
        session.permanent = True
        print(f"[Pilar/auth] Login OK: {email} IP={ip}")
        return redirect('/')
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
    return redirect('/')

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
    users = User.query.order_by(User.created_at.desc()).all()
    for u in users:
        u.analysis_count = Analysis.query.filter_by(user_id=u.id).count()
    total_users = len(users)
    total_analyses = Analysis.query.count()
    unverified = sum(1 for u in users if not u.email_verified)
    return render_template_string(ADMIN_HTML, users=users, total_users=total_users,
                                  total_analyses=total_analyses, unverified=unverified)

@app.route('/admin/impersonate/<int:uid>')
@admin_required
def impersonate(uid):
    session['user_id'] = uid
    return redirect('/')

# ── ROUTES PAGES ──────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template_string(HTML)

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

@app.route('/assistant')
@login_required
def assistant(): return render_template_string(ASSISTANT_HTML)

@app.route('/tutorial')
@login_required
def tutorial(): return render_template_string(TUTORIAL_HTML)

@app.route('/twin')
def twin(): return render_template_string(TWIN_HTML)

@app.route('/history')
def history():
    uid = current_uid()
    analyses = Analysis.query.filter_by(user_id=uid).order_by(Analysis.timestamp.desc()).all()
    total = len(analyses)
    anomalies = sum(1 for a in analyses if a.prediction)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1) if total > 0 else 0
    mails = sum(1 for a in analyses if a.mail_sent)
    return render_template_string(HISTORY_HTML, analyses=analyses, total=total,
                                   anomalies=anomalies, avg_risk=avg_risk, mails=mails)

@app.route('/settings')
def settings(): return render_template_string(SETTINGS_HTML)

@app.route('/set_email', methods=['POST'])
@login_required
def set_email():
    set_setting('responsible_email', request.json.get('email', ''))
    return jsonify({'status': 'ok'})

@app.route('/predire', methods=['POST'])
def predire():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Données manquantes (JSON invalide)'}), 400
        if model is None:
            return jsonify({'error': 'Modele ML non chargé — contactez l\'administrateur'}), 503
        required = ['type', 'temp_air', 'temp_process', 'vitesse', 'couple', 'usure']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
            try:
                data[field] = float(data[field])
            except (TypeError, ValueError):
                return jsonify({'error': f'Valeur invalide pour {field}: doit etre numerique'}), 400
        probabilite, prediction, zones_risque = predict_risk(data)
        mail_envoye = False
        email = get_setting('responsible_email')
        if probabilite >= 50 and email:
            threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data), daemon=True).start()
            mail_envoye = True
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
        db.session.add(Analysis(machine_type=machine_types.get(data['type'], 'Unknown'),
            temp_air=data['temp_air'], temp_process=data['temp_process'],
            vitesse=data['vitesse'], couple=data['couple'], usure=data['usure'],
            risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye,
            user_id=current_uid()))
        db.session.commit()
        return jsonify({'prediction': prediction, 'probabilite': probabilite,
                        'zones': zones_risque, 'mail_envoye': mail_envoye})
    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"[Pilar/predire] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Erreur modèle: {type(e).__name__}: {str(e)}'}), 500

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
            risk, pred, _ = predict_risk({'type':1,'temp_air':last.temp_air,'temp_process':ctp,'vitesse':last.vitesse,'couple':last.couple,'usure':cu})
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
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500

@app.route('/api/health')
def api_health():
    import os, sys
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'zones_loaded': len(modeles_zones),
        'python': sys.version.split()[0],
        'commit': os.environ.get('RAILWAY_GIT_COMMIT_SHA', 'local')[:7],
    })

@app.route('/api/whatif', methods=['POST'])
@api_or_login_required
def api_whatif():
    try:
        params = request.json
        if not params: return jsonify({'error': 'Données manquantes'}), 400
        params['temp_process'] = params['temp_air'] + 10
        risk, pred, zones = predict_risk(params)
        if pred == 0: status, message = 'Normal Operation', 'No failure predicted under these conditions.'
        elif risk < 50: status, message = 'Low Risk', 'Minor anomaly. Continue monitoring.'
        else: status, message = 'High Failure Risk', 'Reduce tool wear or torque immediately.'
        return jsonify({'risk':risk,'status':status,'message':message,'zones':zones})
    except Exception as e:
        print(f"[Pilar/api_whatif] ERROR: {type(e).__name__}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@api_or_login_required
def chat():
    import anthropic as _anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[Pilar/chat] ANTHROPIC_API_KEY manquante — configurez-la sur Railway")
        return jsonify({'reply': None, 'error': 'API key not configured'}), 503

    data = request.json
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'reply': '', 'error': 'Empty message'}), 400

    context = data.get('context')
    chat_history = data.get('history', [])

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
        return jsonify({'reply': None, 'error': str(e)}), 500


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
        return jsonify({'error': str(e)}), 500
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