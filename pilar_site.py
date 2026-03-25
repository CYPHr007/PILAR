"""
PILAR Website — Access Management
Public landing, request access (→ Google Sheet), admin panel, member download area.
Deploy on Render / Railway (free tier).

ENV variables:
  SECRET_KEY        — Flask secret (set a long random string in prod)
  ADMIN_PASSWORD    — Admin panel password (default: pilar-admin-2026)
  GOOGLE_SHEET_URL  — Google Apps Script URL to mirror requests to a Sheet (optional)
  DOWNLOAD_URL      — Direct URL to the PILAR Pilot .exe (GitHub Release, Drive, etc.)
  APP_VERSION       — Display version string, e.g. "v1.0-pilot"
  BASE_URL          — Your domain, e.g. https://pilar.io (for signup links)
"""
import os, secrets, hashlib, json
from datetime import datetime, timezone
from functools import wraps

from flask import (Flask, request, render_template_string, redirect,
                   url_for, session, jsonify, abort)
from flask_sqlalchemy import SQLAlchemy

try:
    import requests as _http
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pilar-dev-secret-change-in-prod')

db_path = os.path.join(os.path.dirname(__file__), 'pilar_site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Config ───────────────────────────────────────────────────────────────────
ADMIN_PASSWORD   = os.environ.get('ADMIN_PASSWORD',   'pilar-admin-2026')
GOOGLE_SHEET_URL = os.environ.get('GOOGLE_SHEET_URL', '')
DOWNLOAD_URL     = os.environ.get('DOWNLOAD_URL',     '#')
APP_VERSION      = os.environ.get('APP_VERSION',      'v1.0-pilot')
BASE_URL         = os.environ.get('BASE_URL',         'http://localhost:8080')

# ── Models ───────────────────────────────────────────────────────────────────
class AccessRequest(db.Model):
    __tablename__ = 'access_requests'
    id           = db.Column(db.Integer, primary_key=True)
    email        = db.Column(db.String(200), nullable=False)
    name         = db.Column(db.String(200), default='')
    company      = db.Column(db.String(200), default='')
    submitted_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status       = db.Column(db.String(20), default='pending')  # pending | approved | rejected
    signup_token = db.Column(db.String(64), unique=True)
    approved_at  = db.Column(db.DateTime)

class User(db.Model):
    __tablename__ = 'users'
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(200), unique=True, nullable=False)
    name       = db.Column(db.String(200), default='')
    company    = db.Column(db.String(200), default='')
    pw_hash    = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    request_id = db.Column(db.Integer, db.ForeignKey('access_requests.id'), nullable=True)

class InviteToken(db.Model):
    __tablename__ = 'invite_tokens'
    id         = db.Column(db.Integer, primary_key=True)
    token      = db.Column(db.String(64), unique=True, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    max_uses   = db.Column(db.Integer, default=5)
    use_count  = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    label      = db.Column(db.String(200), default='')

# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def wrap(*a, **kw):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.url))
        return f(*a, **kw)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*a, **kw):
        if not session.get('admin'):
            return redirect(url_for('admin_login'))
        return f(*a, **kw)
    return wrap

def _sheet(data):
    """Mirror data to Google Sheet via Apps Script (best-effort)."""
    if GOOGLE_SHEET_URL and _HAS_REQUESTS:
        try: _http.post(GOOGLE_SHEET_URL, json=data, timeout=4)
        except Exception: pass

def _signup_link(token):
    return f"{BASE_URL}/signup/{token}"

# ── CSS / Design tokens ───────────────────────────────────────────────────────
_FONTS = "@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');"

_BASE_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#08090c;--surface:#0f1117;--surface2:#131720;
  --border:#1c2130;--border2:#252d3d;
  --accent:#0d9488;--accent-dim:rgba(13,148,136,.12);
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --red:#ef4444;--red-dim:rgba(239,68,68,.15);
  --green:#22c55e;--green-dim:rgba(34,197,94,.12);
  --amber:#f59e0b;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:var(--accent);text-decoration:none}
input,textarea,select{font-family:inherit}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
"""

# ── Shared HTML pieces ─────────────────────────────────────────────────────────
def _page(title, body_html, extra_css='', show_nav=False, nav_user=None):
    nav = ''
    if show_nav:
        user_part = f'<span class="nav-user">{nav_user}</span>' if nav_user else ''
        nav = f'''<nav>
  <a href="/" class="nav-logo">P I L A R</a>
  <div class="nav-right">
    {user_part}
    <a href="/logout" class="nav-link">Sign out</a>
  </div>
</nav>'''
    return render_template_string(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — PILAR</title>
<style>
{_FONTS}
{_BASE_CSS}
nav{{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 2rem;height:56px;
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(8,9,12,.9);backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}}
.nav-logo{{font-family:'JetBrains Mono',monospace;font-size:.8rem;letter-spacing:.32em;
  color:var(--text);text-transform:uppercase}}
.nav-right{{display:flex;align-items:center;gap:1.5rem}}
.nav-link{{font-size:.82rem;color:var(--text3)}}
.nav-link:hover{{color:var(--text2)}}
.nav-user{{font-size:.78rem;color:var(--text3);font-family:'JetBrains Mono',monospace}}
{extra_css}
</style>
</head>
<body>
{nav}
{body_html}
</body>
</html>""")


# ─────────────────────────────────────────────────────────────────────────────
#  LANDING PAGE  /
# ─────────────────────────────────────────────────────────────────────────────
LANDING_CSS = """
.grid-bg{position:fixed;inset:0;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),
    linear-gradient(90deg,var(--border) 1px,transparent 1px);
  background-size:60px 60px;opacity:.03;pointer-events:none;z-index:0}
.hero-glow{position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);
  width:700px;height:500px;
  background:radial-gradient(ellipse,rgba(13,148,136,.07) 0%,transparent 70%);
  pointer-events:none}
section{position:relative;z-index:1}
.container{max-width:1100px;margin:0 auto;padding:0 2rem}
/* NAV */
#lp-nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 2rem;height:60px;
  display:flex;align-items:center;justify-content:space-between;
  transition:background .3s,border-bottom .3s}
#lp-nav.scrolled{background:rgba(8,9,12,.85);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border)}
.nav-logo{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:500;
  letter-spacing:.35em;color:var(--text);text-transform:uppercase}
.lp-nav-right{display:flex;align-items:center;gap:2rem}
.lp-nav-link{font-size:.875rem;color:var(--text3);transition:color .2s}
.lp-nav-link:hover{color:var(--text)}
.lp-nav-cta{font-size:.8rem;font-weight:500;color:var(--text2);padding:.4rem .9rem;
  border:1px solid var(--border2);border-radius:4px;transition:border-color .2s,color .2s}
.lp-nav-cta:hover{border-color:var(--accent);color:var(--text)}
/* HERO */
#hero{min-height:100vh;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:120px 2rem 80px;position:relative;overflow:hidden}
.hero-inner{max-width:760px;position:relative;z-index:1}
.hero-label{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.7rem;
  letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:2rem;
  padding:.3rem .7rem;border:1px solid rgba(13,148,136,.25);border-radius:3px}
h1{font-family:'DM Serif Display',serif;font-size:clamp(2.8rem,6vw,5.2rem);
  line-height:1.08;font-weight:400;color:var(--text);margin-bottom:1.5rem;letter-spacing:-.02em}
.hero-sub{font-size:clamp(1rem,2vw,1.2rem);color:var(--text2);font-weight:300;
  max-width:560px;margin:0 auto 2.5rem;line-height:1.7}
.hero-actions{display:flex;align-items:center;justify-content:center;gap:1.5rem;flex-wrap:wrap}
.btn-primary{display:inline-flex;align-items:center;font-size:.875rem;font-weight:500;
  color:#fff;background:var(--accent);padding:.7rem 1.6rem;border-radius:4px;
  border:1px solid transparent;transition:opacity .2s,box-shadow .2s,transform .15s}
.btn-primary:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 0 20px rgba(13,148,136,.3)}
.btn-ghost{font-size:.875rem;color:var(--text3);transition:color .2s}
.btn-ghost:hover{color:var(--text)}
/* PROOF */
#proof{border-top:1px solid var(--border);border-bottom:1px solid var(--border);
  background:var(--surface);padding:2rem 0}
.proof-stats{display:flex;align-items:center;justify-content:center;flex-wrap:wrap}
.proof-stat{padding:0 2.5rem;font-family:'JetBrains Mono',monospace;font-size:.9rem;
  color:var(--text2);font-weight:500}
.proof-stat+.proof-stat{border-left:1px solid var(--border2)}
.proof-caption{text-align:center;font-size:.8rem;color:var(--text3);margin-top:1.2rem;font-weight:300}
/* SECTIONS */
.section-label{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--text3);margin-bottom:.75rem}
.section-title{font-family:'DM Serif Display',serif;font-size:clamp(1.8rem,3.5vw,2.8rem);
  font-weight:400;line-height:1.15;letter-spacing:-.01em}
#product{padding:100px 0}
.product-grid{display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:start}
.product-body{font-size:1.05rem;color:var(--text2);line-height:1.8;font-weight:300}
.sensor-list{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1.5rem}
.sensor-list-title{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text3);margin-bottom:1rem}
.sensor-item{display:flex;align-items:center;gap:.75rem;padding:.55rem 0;
  border-bottom:1px solid var(--border);font-size:.9rem;color:var(--text2)}
.sensor-item:last-child{border-bottom:none}
.sensor-dot{width:5px;height:5px;border-radius:50%;background:var(--border2);flex-shrink:0}
.sensor-item.active .sensor-dot{background:var(--accent)}
#how{padding:100px 0;border-top:1px solid var(--border)}
.steps-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:3rem;margin-top:3rem}
.step{padding-top:1.5rem;border-top:1px solid var(--border)}
.step-num{font-family:'DM Serif Display',serif;font-size:3rem;color:var(--border2);
  line-height:1;margin-bottom:1.2rem}
.step-title{font-size:1.1rem;font-weight:500;color:var(--text);margin-bottom:.75rem}
.step-body{font-size:.9rem;color:var(--text3);line-height:1.7;font-weight:300}
#outcomes{padding:100px 0}
.outcomes-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;
  border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:3rem}
.outcome{padding:2rem 1.75rem;background:var(--surface);transition:background .2s}
.outcome:hover{background:var(--surface2)}
.outcome-num{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--text3);margin-bottom:.75rem}
.outcome-text{font-family:'DM Serif Display',serif;font-size:1.3rem;line-height:1.35;color:var(--text)}
#industries{padding:100px 0;border-top:1px solid var(--border)}
.industry-tags{display:flex;gap:.75rem;flex-wrap:wrap;margin:2rem 0 1.5rem}
.tag{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--text2);
  padding:.4rem .9rem;border:1px solid var(--border2);border-radius:3px;letter-spacing:.05em}
/* CTA / FORM */
#cta{padding:100px 0;text-align:center;border-top:1px solid var(--border)}
.cta-title{font-family:'DM Serif Display',serif;font-size:clamp(2rem,4vw,3.2rem);
  font-weight:400;margin-bottom:1rem;letter-spacing:-.015em}
.cta-sub{font-size:1rem;color:var(--text3);margin-bottom:2.5rem;font-weight:300}
.cta-form{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;margin-bottom:1rem}
.cta-input{width:220px;padding:.7rem 1rem;background:var(--surface);border:1px solid var(--border2);
  border-radius:4px;color:var(--text);font-size:.875rem;outline:none;transition:border-color .2s}
.cta-input::placeholder{color:var(--text3)}
.cta-input:focus{border-color:var(--accent)}
.cta-submit{padding:.7rem 1.4rem;background:var(--accent);border:none;border-radius:4px;
  color:#fff;font-size:.875rem;font-weight:500;cursor:pointer;
  transition:opacity .2s,box-shadow .2s,transform .15s}
.cta-submit:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 0 20px rgba(13,148,136,.3)}
.cta-submit:disabled{opacity:.5;cursor:default;transform:none;box-shadow:none}
.cta-note{font-size:.78rem;color:var(--text3)}
.form-msg{font-size:.85rem;margin-top:.75rem;min-height:1.2rem}
.form-msg.ok{color:var(--accent)}
.form-msg.err{color:var(--red)}
footer{border-top:1px solid var(--border);padding:2rem 0}
.footer-inner{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem}
.footer-logo{font-family:'JetBrains Mono',monospace;font-size:.78rem;letter-spacing:.3em;
  color:var(--text3);text-transform:uppercase}
.footer-right{display:flex;align-items:center;gap:2rem;flex-wrap:wrap}
.footer-link{font-size:.78rem;color:var(--text3)}
.footer-link:hover{color:var(--text2)}
.footer-copy{font-size:.78rem;color:var(--text3)}
.reveal{opacity:0;transform:translateY(16px);transition:opacity .4s ease,transform .4s ease}
.reveal.visible{opacity:1;transform:translateY(0)}
@media(max-width:768px){
  .product-grid,.steps-grid,.outcomes-grid{grid-template-columns:1fr}
  .steps-grid{gap:2rem}.outcomes-grid{gap:1px}
  .proof-stat{padding:0 1.2rem}
  .container{padding:0 1.25rem}
  #hero{padding:100px 1.25rem 60px}
}
@media(max-width:480px){
  .lp-nav-right .lp-nav-link{display:none}
  .proof-stats{flex-direction:column;gap:1rem;text-align:center}
  .proof-stat+.proof-stat{border-left:none;border-top:1px solid var(--border2)}
  .proof-stat{padding:.6rem 0}
}
"""

LANDING_HTML = """
<div class="grid-bg"></div>
<nav id="lp-nav">
  <a href="#" class="nav-logo">P I L A R</a>
  <div class="lp-nav-right">
    <a href="#product" class="lp-nav-link">Product</a>
    <a href="#cta" class="lp-nav-link">Contact</a>
    <a href="#cta" class="lp-nav-cta">Request Access</a>
  </div>
</nav>

<section id="hero">
  <div class="hero-glow"></div>
  <div class="hero-inner">
    <div class="hero-label">Early Access — Predictive Maintenance</div>
    <h1>Know before<br>it breaks.</h1>
    <p class="hero-sub">Pilar monitors your existing pump sensors and surfaces failure signals days before they become downtime.</p>
    <div class="hero-actions">
      <a href="#cta" class="btn-primary">Request Early Access</a>
      <a href="#how" class="btn-ghost">See how it works →</a>
    </div>
  </div>
</section>

<section id="proof">
  <div class="container">
    <div class="proof-stats">
      <div class="proof-stat">&lt; 24h setup</div>
      <div class="proof-stat">No new hardware</div>
      <div class="proof-stat">Early access open</div>
    </div>
    <p class="proof-caption">Designed for maintenance engineers in chemical, water treatment, and food processing facilities.</p>
  </div>
</section>

<section id="product">
  <div class="container">
    <div class="section-header reveal">
      <div class="section-label">What Pilar does</div>
      <h2 class="section-title">Condition monitoring,<br>without the project.</h2>
    </div>
    <div class="product-grid">
      <div class="reveal">
        <p class="product-body">Most industrial pumps run without any condition monitoring. When they fail, it's sudden, expensive, and avoidable.<br><br>Pilar connects to your existing sensor data and builds a behavioral baseline for each machine. Deviations trigger early warnings — not alarms.</p>
      </div>
      <div class="sensor-list reveal">
        <div class="sensor-list-title">Supported sensors</div>
        <div class="sensor-item active"><div class="sensor-dot"></div>Vibration</div>
        <div class="sensor-item active"><div class="sensor-dot"></div>Bearing temperature</div>
        <div class="sensor-item active"><div class="sensor-dot"></div>Motor temperature</div>
        <div class="sensor-item active"><div class="sensor-dot"></div>Flow rate</div>
        <div class="sensor-item"><div class="sensor-dot"></div>Inlet pressure</div>
        <div class="sensor-item"><div class="sensor-dot"></div>Outlet pressure</div>
        <div class="sensor-item"><div class="sensor-dot"></div>Motor current</div>
        <div class="sensor-item"><div class="sensor-dot"></div>Run hours</div>
      </div>
    </div>
  </div>
</section>

<section id="how">
  <div class="container">
    <div class="section-header reveal">
      <div class="section-label">How it works</div>
      <h2 class="section-title">Three steps to visibility.</h2>
    </div>
    <div class="steps-grid">
      <div class="step reveal">
        <div class="step-num">01</div>
        <div class="step-title">Connect</div>
        <p class="step-body">Point Pilar at your existing sensor feed or upload historical CSV data. No new hardware required.</p>
      </div>
      <div class="step reveal">
        <div class="step-num">02</div>
        <div class="step-title">Learn</div>
        <p class="step-body">Pilar builds a behavioral model of your specific pump under normal operating conditions.</p>
      </div>
      <div class="step reveal">
        <div class="step-num">03</div>
        <div class="step-title">Alert</div>
        <p class="step-body">When sensor patterns diverge from baseline, you receive an early warning with the likely cause and estimated time to failure.</p>
      </div>
    </div>
  </div>
</section>

<section id="outcomes">
  <div class="container">
    <div class="section-header reveal">
      <div class="section-label">Key outcomes</div>
      <h2 class="section-title">What operators report.</h2>
    </div>
    <div class="outcomes-grid">
      <div class="outcome reveal"><div class="outcome-num">01</div><div class="outcome-text">Catch failures 24–72h early</div></div>
      <div class="outcome reveal"><div class="outcome-num">02</div><div class="outcome-text">No hardware investment</div></div>
      <div class="outcome reveal"><div class="outcome-num">03</div><div class="outcome-text">Deployed in under a day</div></div>
    </div>
  </div>
</section>

<section id="industries">
  <div class="container">
    <div class="section-header reveal">
      <div class="section-label">Built for</div>
      <h2 class="section-title">Process industries.</h2>
    </div>
    <div class="industry-tags reveal">
      <div class="tag">Chemical</div>
      <div class="tag">Water Treatment</div>
      <div class="tag">Food &amp; Beverage</div>
    </div>
    <p class="industry-note reveal">If you run centrifugal pumps, Pilar works with your setup.</p>
  </div>
</section>

<section id="cta">
  <div class="container">
    <div class="reveal">
      <div class="section-label" style="display:flex;justify-content:center">Early access</div>
      <h2 class="cta-title">Ready to stop guessing?</h2>
      <p class="cta-sub">Join the early access program. We onboard 3 new facilities per month.</p>
      <form class="cta-form" id="req-form">
        <input class="cta-input" type="text"  name="name"    placeholder="Your name"      required>
        <input class="cta-input" type="text"  name="company" placeholder="Company"        required>
        <input class="cta-input" type="email" name="email"   placeholder="Work email"     required>
        <button class="cta-submit" type="submit">Request Access</button>
      </form>
      <div class="form-msg" id="form-msg">No commitment. Response within 48 hours.</div>
    </div>
  </div>
</section>

<footer>
  <div class="container">
    <div class="footer-inner">
      <div class="footer-logo">P I L A R</div>
      <div class="footer-right">
        <a href="#product" class="footer-link">Product</a>
        <span class="footer-copy">© 2026 PILAR</span>
      </div>
    </div>
  </div>
</footer>

<script>
var lpNav = document.getElementById('lp-nav');
window.addEventListener('scroll', function(){ lpNav.classList.toggle('scrolled', scrollY > 20); }, {passive:true});
var io = new IntersectionObserver(function(entries){
  entries.forEach(function(e){ if(e.isIntersecting) e.target.classList.add('visible'); });
}, {threshold:.12});
document.querySelectorAll('.reveal').forEach(function(el){ io.observe(el); });

document.getElementById('req-form').addEventListener('submit', async function(e){
  e.preventDefault();
  var btn = this.querySelector('button');
  var msg = document.getElementById('form-msg');
  btn.disabled = true; btn.textContent = 'Sending...';
  var fd = new FormData(this);
  var body = {name: fd.get('name'), company: fd.get('company'), email: fd.get('email')};
  try {
    var r = await fetch('/api/request-access', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    var j = await r.json();
    if(r.ok){
      msg.className='form-msg ok'; msg.textContent = j.message || 'Request received. We will be in touch within 48 hours.';
      this.innerHTML = ''; // hide form
    } else {
      msg.className='form-msg err'; msg.textContent = j.error || 'Something went wrong. Try again.';
      btn.disabled=false; btn.textContent='Request Access';
    }
  } catch(err){
    msg.className='form-msg err'; msg.textContent = 'Network error. Please try again.';
    btn.disabled=false; btn.textContent='Request Access';
  }
});
</script>
"""


@app.route('/')
def index():
    return _page('Predictive Maintenance for Industry', LANDING_HTML, LANDING_CSS)


# ─────────────────────────────────────────────────────────────────────────────
#  API — REQUEST ACCESS
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/request-access', methods=['POST'])
def api_request_access():
    data = request.get_json(silent=True) or {}
    email   = (data.get('email')   or '').strip().lower()
    name    = (data.get('name')    or '').strip()
    company = (data.get('company') or '').strip()
    if not email or '@' not in email:
        return jsonify(error='A valid email is required.'), 400
    existing = AccessRequest.query.filter_by(email=email).first()
    if existing:
        return jsonify(message='We already have your request. We will be in touch soon.'), 200
    req = AccessRequest(email=email, name=name, company=company)
    db.session.add(req)
    db.session.commit()
    _sheet({'email': email, 'name': name, 'company': company,
            'submitted_at': req.submitted_at.isoformat(), 'type': 'access_request'})
    return jsonify(message='Request received. We will be in touch within 48 hours.'), 200


# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN
# ─────────────────────────────────────────────────────────────────────────────
ADMIN_CSS = """
.page{padding:80px 0 60px;max-width:1100px;margin:0 auto;padding-left:2rem;padding-right:2rem}
h2{font-family:'DM Serif Display',serif;font-size:2rem;font-weight:400;margin-bottom:2rem}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:2rem}
.tab{font-size:.85rem;color:var(--text3);padding:.6rem 1.2rem;cursor:pointer;
  border-bottom:2px solid transparent;transition:color .2s}
.tab.active{color:var(--text);border-bottom-color:var(--accent)}
.tab-panel{display:none}.tab-panel.active{display:block}
table{width:100%;border-collapse:collapse;font-size:.875rem}
th{text-align:left;padding:.6rem .75rem;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--text3);border-bottom:1px solid var(--border);font-weight:500;font-family:'JetBrains Mono',monospace}
td{padding:.7rem .75rem;border-bottom:1px solid var(--border);color:var(--text2);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--surface)}
.badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.65rem;
  letter-spacing:.08em;padding:.2rem .5rem;border-radius:3px;font-weight:500}
.badge-pending{color:var(--amber);background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.2)}
.badge-approved{color:var(--green);background:var(--green-dim);border:1px solid rgba(34,197,94,.2)}
.badge-rejected{color:var(--red);background:var(--red-dim);border:1px solid rgba(239,68,68,.2)}
.btn-sm{display:inline-flex;align-items:center;font-size:.75rem;font-weight:500;padding:.3rem .7rem;
  border-radius:3px;border:none;cursor:pointer;transition:opacity .2s;font-family:inherit}
.btn-approve{background:var(--green-dim);color:var(--green);border:1px solid rgba(34,197,94,.25)}
.btn-approve:hover{opacity:.8}
.btn-reject{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.btn-reject:hover{opacity:.8}
.btn-copy{background:var(--accent-dim);color:var(--accent);border:1px solid rgba(13,148,136,.25)}
.btn-copy:hover{opacity:.8}
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1.25rem}
.stat-val{font-family:'DM Serif Display',serif;font-size:2.2rem;font-weight:400;
  color:var(--text);line-height:1}
.stat-label{font-size:.78rem;color:var(--text3);margin-top:.4rem;font-family:'JetBrains Mono',monospace;
  letter-spacing:.08em;text-transform:uppercase}
.link-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;
  align-items:center;justify-content:center}
.link-modal.open{display:flex}
.modal-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:2rem;max-width:520px;width:90%}
.modal-title{font-family:'DM Serif Display',serif;font-size:1.4rem;margin-bottom:1rem}
.modal-url{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--text2);
  background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:.75rem;
  word-break:break-all;margin-bottom:1rem}
.modal-actions{display:flex;gap:.75rem;justify-content:flex-end}
.section-settings{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1.5rem;margin-bottom:2rem}
.setting-row{display:flex;align-items:center;gap:1rem;padding:.6rem 0;border-bottom:1px solid var(--border)}
.setting-row:last-child{border-bottom:none}
.setting-label{font-size:.85rem;color:var(--text2);width:180px;flex-shrink:0}
.setting-val{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--text3);flex:1;word-break:break-all}
.setting-input{flex:1;background:var(--bg);border:1px solid var(--border2);border-radius:4px;
  color:var(--text);font-family:'JetBrains Mono',monospace;font-size:.8rem;padding:.4rem .75rem;outline:none}
.setting-input:focus{border-color:var(--accent)}
"""

def _admin_body():
    reqs = AccessRequest.query.order_by(AccessRequest.submitted_at.desc()).all()
    users = User.query.order_by(User.created_at.desc()).all()
    n_pending  = sum(1 for r in reqs if r.status == 'pending')
    n_approved = sum(1 for r in reqs if r.status == 'approved')
    n_users    = len(users)

    def req_row(r):
        badge = f'<span class="badge badge-{r.status}">{r.status}</span>'
        date  = r.submitted_at.strftime('%b %d, %Y') if r.submitted_at else '—'
        link_btn = ''
        if r.status == 'approved' and r.signup_token:
            link_btn = f'<button class="btn-sm btn-copy" onclick="copyLink(\'{r.signup_token}\')">Copy link</button> '
        approve_btn = '' if r.status == 'approved' else f'<button class="btn-sm btn-approve" onclick="approveReq({r.id})">Approve</button> '
        reject_btn  = '' if r.status == 'rejected' else f'<button class="btn-sm btn-reject" onclick="rejectReq({r.id})">Reject</button>'
        return f'''<tr id="req-{r.id}">
          <td>{r.email}</td><td>{r.name or '—'}</td><td>{r.company or '—'}</td>
          <td>{date}</td><td>{badge}</td>
          <td style="white-space:nowrap">{link_btn}{approve_btn}{reject_btn}</td>
        </tr>'''

    def user_row(u):
        date = u.created_at.strftime('%b %d, %Y') if u.created_at else '—'
        return f'<tr><td>{u.email}</td><td>{u.name or "—"}</td><td>{u.company or "—"}</td><td>{date}</td></tr>'

    return f"""
<div class="page">
  <h2>Admin — PILAR</h2>
  <div class="stats-row">
    <div class="stat-card"><div class="stat-val">{len(reqs)}</div><div class="stat-label">Requests</div></div>
    <div class="stat-card"><div class="stat-val">{n_pending}</div><div class="stat-label">Pending</div></div>
    <div class="stat-card"><div class="stat-val">{n_approved}</div><div class="stat-label">Approved</div></div>
    <div class="stat-card"><div class="stat-val">{n_users}</div><div class="stat-label">Users</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="showTab('requests',this)">Access Requests</div>
    <div class="tab" onclick="showTab('users',this)">Users</div>
    <div class="tab" onclick="showTab('settings',this)">Settings</div>
  </div>

  <div id="tab-requests" class="tab-panel active">
    <table>
      <thead><tr><th>Email</th><th>Name</th><th>Company</th><th>Date</th><th>Status</th><th>Actions</th></tr></thead>
      <tbody>{''.join(req_row(r) for r in reqs) or '<tr><td colspan="6" style="color:var(--text3);padding:2rem .75rem">No requests yet.</td></tr>'}</tbody>
    </table>
  </div>

  <div id="tab-users" class="tab-panel">
    <table>
      <thead><tr><th>Email</th><th>Name</th><th>Company</th><th>Joined</th></tr></thead>
      <tbody>{''.join(user_row(u) for u in users) or '<tr><td colspan="4" style="color:var(--text3);padding:2rem .75rem">No users yet.</td></tr>'}</tbody>
    </table>
  </div>

  <div id="tab-settings" class="tab-panel">
    <div class="section-settings">
      <div class="setting-row">
        <div class="setting-label">Download URL</div>
        <input class="setting-input" id="s-download" value="{DOWNLOAD_URL}" placeholder="https://github.com/.../releases/...">
        <button class="btn-sm btn-copy" style="flex-shrink:0" onclick="saveSetting('download_url', document.getElementById('s-download').value)">Save</button>
      </div>
      <div class="setting-row">
        <div class="setting-label">App version</div>
        <input class="setting-input" id="s-version" value="{APP_VERSION}" placeholder="v1.0-pilot">
        <button class="btn-sm btn-copy" style="flex-shrink:0" onclick="saveSetting('app_version', document.getElementById('s-version').value)">Save</button>
      </div>
      <div class="setting-row">
        <div class="setting-label">Base URL</div>
        <div class="setting-val">{BASE_URL}</div>
      </div>
      <div class="setting-row">
        <div class="setting-label">Google Sheet URL</div>
        <div class="setting-val">{'Set via ENV' if GOOGLE_SHEET_URL else 'Not configured'}</div>
      </div>
    </div>
    <p style="font-size:.8rem;color:var(--text3)">To change Base URL or Google Sheet URL, set the ENV variables on your server.</p>
  </div>
</div>

<div class="link-modal" id="link-modal">
  <div class="modal-box">
    <div class="modal-title">Signup link</div>
    <p style="font-size:.85rem;color:var(--text3);margin-bottom:1rem">Share this link with the approved user. It expires once they create their account.</p>
    <div class="modal-url" id="modal-url"></div>
    <div class="modal-actions">
      <button class="btn-sm btn-copy" onclick="doCopy()">Copy to clipboard</button>
      <button class="btn-sm" style="background:var(--surface2);color:var(--text3);border:1px solid var(--border2)"
        onclick="document.getElementById('link-modal').classList.remove('open')">Close</button>
    </div>
  </div>
</div>

<script>
function showTab(id, el){{
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  el.classList.add('active');
}}
async function approveReq(id){{
  var r = await fetch('/admin/approve/'+id, {{method:'POST'}});
  var j = await r.json();
  if(r.ok){{ showLink(j.signup_link); refreshRow(id, j); }}
  else alert(j.error||'Error');
}}
async function rejectReq(id){{
  if(!confirm('Reject this request?')) return;
  var r = await fetch('/admin/reject/'+id, {{method:'POST'}});
  var j = await r.json();
  if(r.ok) refreshRow(id, j);
  else alert(j.error||'Error');
}}
function refreshRow(id, j){{
  var row = document.getElementById('req-'+id);
  if(row) row.querySelector('td:nth-child(5)').innerHTML =
    '<span class="badge badge-'+j.status+'">'+j.status+'</span>';
  if(j.signup_link){{
    var td = row.querySelector('td:last-child');
    td.innerHTML = '<button class="btn-sm btn-copy" onclick="copyLink(\''+j.signup_token+'\')">Copy link</button> '
      + '<button class="btn-sm btn-reject" onclick="rejectReq('+id+')">Reject</button>';
  }}
}}
function copyLink(token){{
  var url = '{BASE_URL}/signup/'+token;
  showLink(url);
}}
function showLink(url){{
  document.getElementById('modal-url').textContent = url;
  document.getElementById('link-modal').classList.add('open');
}}
function doCopy(){{
  var t = document.getElementById('modal-url').textContent;
  navigator.clipboard.writeText(t).then(()=>{{
    var btn = document.querySelector('.link-modal .btn-copy');
    btn.textContent='Copied!'; setTimeout(()=>btn.textContent='Copy to clipboard',2000);
  }});
}}
async function saveSetting(key, val){{
  var r = await fetch('/admin/settings', {{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{key, value: val}})}});
  var j = await r.json();
  if(!r.ok) alert(j.error||'Error saving'); else alert('Saved. Restart server to apply.');
}}
document.getElementById('link-modal').addEventListener('click', function(e){{
  if(e.target===this) this.classList.remove('open');
}});
</script>
"""

ADMIN_LOGIN_HTML = """
<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem">
  <div style="width:100%;max-width:360px">
    <div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;letter-spacing:.3em;
      text-transform:uppercase;color:var(--text3);text-align:center;margin-bottom:2rem">P I L A R</div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:2rem">
      <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;margin-bottom:1.5rem">Admin access</div>
      {% if error %}<div style="color:var(--red);font-size:.85rem;margin-bottom:1rem">{{ error }}</div>{% endif %}
      <form method="POST" action="/admin/login">
        <div style="margin-bottom:1rem">
          <label style="font-size:.78rem;color:var(--text3);letter-spacing:.08em;text-transform:uppercase;
            font-family:'JetBrains Mono',monospace;display:block;margin-bottom:.4rem">Password</label>
          <input type="password" name="password" autofocus
            style="width:100%;background:var(--bg);border:1px solid var(--border2);border-radius:4px;
            color:var(--text);font-size:.875rem;padding:.65rem .9rem;outline:none;
            font-family:inherit;transition:border-color .2s"
            onfocus="this.style.borderColor='var(--accent)'"
            onblur="this.style.borderColor='var(--border2)'" required>
        </div>
        <button type="submit"
          style="width:100%;padding:.7rem;background:var(--accent);border:none;border-radius:4px;
          color:#fff;font-size:.875rem;font-weight:500;cursor:pointer;font-family:inherit">
          Sign in
        </button>
      </form>
    </div>
  </div>
</div>
"""


@app.route('/admin')
@admin_required
def admin_dashboard():
    return _page('Admin', _admin_body(), ADMIN_CSS, show_nav=True, nav_user='admin')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        pw = request.form.get('password', '')
        if pw == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        error = 'Incorrect password.'
    return _page('Admin Login', ADMIN_LOGIN_HTML, show_nav=False)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/approve/<int:req_id>', methods=['POST'])
@admin_required
def admin_approve(req_id):
    req = AccessRequest.query.get_or_404(req_id)
    if not req.signup_token:
        req.signup_token = secrets.token_urlsafe(32)
    req.status = 'approved'
    req.approved_at = datetime.now(timezone.utc)
    db.session.commit()
    link = _signup_link(req.signup_token)
    return jsonify(status='approved', signup_link=link, signup_token=req.signup_token)

@app.route('/admin/reject/<int:req_id>', methods=['POST'])
@admin_required
def admin_reject(req_id):
    req = AccessRequest.query.get_or_404(req_id)
    req.status = 'rejected'
    db.session.commit()
    return jsonify(status='rejected')

@app.route('/admin/settings', methods=['POST'])
@admin_required
def admin_settings():
    # Runtime settings — writes to a local JSON file
    data = request.get_json(silent=True) or {}
    key  = data.get('key', '')
    val  = data.get('value', '')
    cfg_path = os.path.join(os.path.dirname(__file__), 'pilar_site_settings.json')
    cfg = {}
    if os.path.exists(cfg_path):
        try: cfg = json.load(open(cfg_path))
        except Exception: pass
    cfg[key] = val
    json.dump(cfg, open(cfg_path, 'w'))
    return jsonify(ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNUP  /signup/<token>
# ─────────────────────────────────────────────────────────────────────────────
SIGNUP_CSS = """
.auth-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}
.auth-box{width:100%;max-width:440px}
.auth-logo{font-family:'JetBrains Mono',monospace;font-size:.85rem;letter-spacing:.3em;
  text-transform:uppercase;color:var(--text3);text-align:center;margin-bottom:2rem}
.auth-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:2rem}
.auth-title{font-family:'DM Serif Display',serif;font-size:1.6rem;margin-bottom:.5rem}
.auth-sub{font-size:.875rem;color:var(--text3);margin-bottom:1.75rem;font-weight:300}
.field{margin-bottom:1rem}
.field label{display:block;font-size:.72rem;color:var(--text3);letter-spacing:.1em;
  text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:.4rem}
.field input{width:100%;background:var(--bg);border:1px solid var(--border2);border-radius:4px;
  color:var(--text);font-size:.875rem;padding:.65rem .9rem;outline:none;font-family:inherit;
  transition:border-color .2s}
.field input:focus{border-color:var(--accent)}
.auth-btn{width:100%;padding:.75rem;background:var(--accent);border:none;border-radius:4px;
  color:#fff;font-size:.9rem;font-weight:500;cursor:pointer;font-family:inherit;margin-top:.25rem;
  transition:opacity .2s}
.auth-btn:hover{opacity:.9}
.auth-error{color:var(--red);font-size:.85rem;margin-bottom:1rem}
.auth-link{font-size:.82rem;color:var(--text3);text-align:center;margin-top:1.25rem}
.auth-link a{color:var(--accent)}
"""

@app.route('/signup/<token>', methods=['GET', 'POST'])
def signup(token):
    req = AccessRequest.query.filter_by(signup_token=token, status='approved').first()
    if not req:
        body = """<div class="auth-wrap"><div class="auth-box">
          <div class="auth-logo">P I L A R</div>
          <div class="auth-card">
            <div class="auth-title">Invalid link</div>
            <p class="auth-sub">This signup link is invalid or has already been used.<br>
            <a href="/#cta">Request access again →</a></p>
          </div></div></div>"""
        return _page('Invalid Link', body, SIGNUP_CSS), 404

    existing = User.query.filter_by(request_id=req.id).first()
    if existing:
        body = """<div class="auth-wrap"><div class="auth-box">
          <div class="auth-logo">P I L A R</div>
          <div class="auth-card">
            <div class="auth-title">Already registered</div>
            <p class="auth-sub">An account for this invitation already exists.<br>
            <a href="/login">Sign in →</a></p>
          </div></div></div>"""
        return _page('Already Registered', body, SIGNUP_CSS)

    error = ''
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        company = request.form.get('company', '').strip()
        pw      = request.form.get('password', '')
        pw2     = request.form.get('password2', '')
        if not name: error = 'Name is required.'
        elif len(pw) < 8: error = 'Password must be at least 8 characters.'
        elif pw != pw2: error = 'Passwords do not match.'
        else:
            user = User(email=req.email, name=name, company=company,
                        pw_hash=_hash(pw), request_id=req.id)
            db.session.add(user)
            db.session.commit()
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_name'] = user.name
            return redirect(url_for('download_page'))

    err_html = f'<div class="auth-error">{error}</div>' if error else ''
    body = f"""<div class="auth-wrap"><div class="auth-box">
      <div class="auth-logo">P I L A R</div>
      <div class="auth-card">
        <div class="auth-title">Create your account</div>
        <p class="auth-sub">You've been granted access to PILAR Pilot.</p>
        {err_html}
        <form method="POST">
          <div class="field">
            <label>Email</label>
            <input type="text" value="{req.email}" disabled style="opacity:.5;cursor:not-allowed">
          </div>
          <div class="field">
            <label>Full name</label>
            <input type="text" name="name" required autofocus placeholder="Jean Dupont">
          </div>
          <div class="field">
            <label>Company</label>
            <input type="text" name="company" placeholder="{req.company or 'Your company'}">
          </div>
          <div class="field">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Min. 8 characters">
          </div>
          <div class="field">
            <label>Confirm password</label>
            <input type="password" name="password2" required placeholder="Repeat password">
          </div>
          <button type="submit" class="auth-btn">Create account &amp; get access</button>
        </form>
      </div>
    </div></div>"""
    return _page('Create Account', body, SIGNUP_CSS)


# ─────────────────────────────────────────────────────────────────────────────
#  TEAM INVITE  /signup/invite/<token>
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/signup/invite/<inv_token>', methods=['GET', 'POST'])
def signup_invite(inv_token):
    inv = InviteToken.query.filter_by(token=inv_token).first()
    if not inv or inv.use_count >= inv.max_uses:
        body = """<div class="auth-wrap"><div class="auth-box">
          <div class="auth-logo">P I L A R</div>
          <div class="auth-card">
            <div class="auth-title">Invalid invite</div>
            <p class="auth-sub">This invite link is no longer valid.</p>
          </div></div></div>"""
        return _page('Invalid Invite', body, SIGNUP_CSS), 404

    error = ''
    if request.method == 'POST':
        email   = request.form.get('email', '').strip().lower()
        name    = request.form.get('name', '').strip()
        company = request.form.get('company', '').strip()
        pw      = request.form.get('password', '')
        pw2     = request.form.get('password2', '')
        if User.query.filter_by(email=email).first():
            error = 'An account with this email already exists.'
        elif not name: error = 'Name is required.'
        elif not email or '@' not in email: error = 'Valid email required.'
        elif len(pw) < 8: error = 'Password must be at least 8 characters.'
        elif pw != pw2: error = 'Passwords do not match.'
        else:
            # Create an auto-approved access request
            req = AccessRequest(email=email, name=name, company=company,
                                status='approved', approved_at=datetime.now(timezone.utc))
            db.session.add(req)
            db.session.flush()
            user = User(email=email, name=name, company=company,
                        pw_hash=_hash(pw), request_id=req.id)
            db.session.add(user)
            inv.use_count += 1
            db.session.commit()
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_name'] = user.name
            return redirect(url_for('download_page'))

    inviter = User.query.get(inv.created_by)
    inviter_name = inviter.name or inviter.email if inviter else 'a PILAR user'
    label = f' ({inv.label})' if inv.label else ''
    err_html = f'<div class="auth-error">{error}</div>' if error else ''
    spots = inv.max_uses - inv.use_count
    body = f"""<div class="auth-wrap"><div class="auth-box">
      <div class="auth-logo">P I L A R</div>
      <div class="auth-card">
        <div class="auth-title">You've been invited</div>
        <p class="auth-sub">{inviter_name} gave you access to PILAR Pilot{label}.<br>
          <span style="font-size:.78rem;color:var(--text3)">{spots} spot{'s' if spots!=1 else ''} remaining on this invite.</span>
        </p>
        {err_html}
        <form method="POST">
          <div class="field"><label>Email</label>
            <input type="email" name="email" required placeholder="you@company.com"></div>
          <div class="field"><label>Full name</label>
            <input type="text" name="name" required autofocus></div>
          <div class="field"><label>Company</label>
            <input type="text" name="company"></div>
          <div class="field"><label>Password</label>
            <input type="password" name="password" required placeholder="Min. 8 characters"></div>
          <div class="field"><label>Confirm password</label>
            <input type="password" name="password2" required></div>
          <button type="submit" class="auth-btn">Create account &amp; get access</button>
        </form>
      </div></div></div>"""
    return _page('Join PILAR', body, SIGNUP_CSS)


# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN  /login
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('download_page'))
    error = ''
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pw    = request.form.get('password', '')
        user  = User.query.filter_by(email=email, pw_hash=_hash(pw)).first()
        if user:
            session['user_id']    = user.id
            session['user_email'] = user.email
            session['user_name']  = user.name
            next_url = request.args.get('next') or url_for('download_page')
            return redirect(next_url)
        error = 'Incorrect email or password.'
    err_html = f'<div class="auth-error">{error}</div>' if error else ''
    body = f"""<div class="auth-wrap"><div class="auth-box">
      <div class="auth-logo">P I L A R</div>
      <div class="auth-card">
        <div class="auth-title">Sign in</div>
        <p class="auth-sub" style="margin-bottom:1.5rem">Access your PILAR Pilot account.</p>
        {err_html}
        <form method="POST">
          <div class="field"><label>Email</label>
            <input type="email" name="email" required autofocus placeholder="you@company.com"></div>
          <div class="field"><label>Password</label>
            <input type="password" name="password" required></div>
          <button type="submit" class="auth-btn">Sign in</button>
        </form>
        <div class="auth-link">No account? <a href="/#cta">Request access →</a></div>
      </div></div></div>"""
    return _page('Sign In', body, SIGNUP_CSS)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ─────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD (member area)  /download
# ─────────────────────────────────────────────────────────────────────────────
DOWNLOAD_CSS = """
.dl-wrap{padding-top:56px;min-height:100vh}
.dl-hero{padding:60px 0 50px;border-bottom:1px solid var(--border)}
.dl-container{max-width:880px;margin:0 auto;padding:0 2rem}
.dl-welcome{font-size:.78rem;color:var(--text3);font-family:'JetBrains Mono',monospace;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:.75rem}
.dl-title{font-family:'DM Serif Display',serif;font-size:2.4rem;font-weight:400;line-height:1.15;
  margin-bottom:.5rem}
.dl-sub{font-size:.95rem;color:var(--text3);font-weight:300}
.dl-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;padding:2.5rem 0}
.dl-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.75rem}
.dl-card-label{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text3);margin-bottom:1rem}
.dl-card-title{font-size:1.1rem;font-weight:500;color:var(--text);margin-bottom:.4rem}
.dl-card-sub{font-size:.85rem;color:var(--text3);margin-bottom:1.5rem;line-height:1.6}
.dl-btn{display:inline-flex;align-items:center;gap:.5rem;font-size:.875rem;font-weight:500;
  color:#fff;background:var(--accent);padding:.65rem 1.4rem;border-radius:4px;
  border:none;cursor:pointer;font-family:inherit;
  transition:opacity .2s,box-shadow .2s,transform .15s;text-decoration:none}
.dl-btn:hover{opacity:.9;transform:translateY(-1px);box-shadow:0 0 16px rgba(13,148,136,.3)}
.dl-btn-sec{background:var(--surface2);color:var(--text2);border:1px solid var(--border2)}
.dl-btn-sec:hover{box-shadow:none;border-color:var(--border)}
.version-badge{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--accent);
  background:rgba(13,148,136,.1);border:1px solid rgba(13,148,136,.2);border-radius:3px;
  padding:.15rem .5rem;margin-left:.5rem;vertical-align:middle}
.invite-section{padding:2.5rem 0;border-top:1px solid var(--border)}
.invite-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.75rem}
.invite-form{display:flex;gap:.75rem;flex-wrap:wrap;margin-top:1.25rem}
.invite-input{flex:1;min-width:200px;background:var(--bg);border:1px solid var(--border2);
  border-radius:4px;color:var(--text);font-size:.875rem;padding:.65rem .9rem;outline:none;
  font-family:inherit;transition:border-color .2s}
.invite-input:focus{border-color:var(--accent)}
.invite-input::placeholder{color:var(--text3)}
.invite-result{margin-top:1rem;padding:.75rem 1rem;background:var(--accent-dim);
  border:1px solid rgba(13,148,136,.2);border-radius:4px;display:none}
.invite-url{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--text2);
  word-break:break-all;margin-bottom:.5rem}
.install-steps{margin-top:1.25rem;display:flex;flex-direction:column;gap:.6rem}
.install-step{display:flex;gap:.75rem;align-items:flex-start;font-size:.875rem;color:var(--text2)}
.step-n{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--text3);
  background:var(--surface2);border:1px solid var(--border);border-radius:3px;
  padding:.15rem .45rem;flex-shrink:0;margin-top:.1rem}
@media(max-width:640px){.dl-grid{grid-template-columns:1fr}}
"""

@app.route('/download')
@login_required
def download_page():
    user = User.query.get(session['user_id'])
    user_invites = InviteToken.query.filter_by(created_by=user.id).order_by(InviteToken.created_at.desc()).all()

    invite_rows = ''
    for inv in user_invites:
        invite_rows += f'''<div style="display:flex;align-items:center;gap:.75rem;padding:.5rem 0;
          border-bottom:1px solid var(--border);font-size:.82rem">
          <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--text3);flex:1;
            word-break:break-all">{BASE_URL}/signup/invite/{inv.token}</span>
          <span style="color:var(--text3)">{inv.use_count}/{inv.max_uses} used</span>
          <button class="dl-btn dl-btn-sec" style="padding:.25rem .6rem;font-size:.72rem"
            onclick="navigator.clipboard.writeText('{BASE_URL}/signup/invite/{inv.token}');
            this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)">Copy</button>
        </div>'''

    first_name = (user.name or user.email).split()[0]
    body = f"""<div class="dl-wrap">
  <div class="dl-hero">
    <div class="dl-container">
      <div class="dl-welcome">Welcome back, {first_name}</div>
      <div class="dl-title">PILAR Pilot <span class="version-badge">{APP_VERSION}</span></div>
      <div class="dl-sub">Your early access to predictive maintenance for centrifugal pumps.</div>
    </div>
  </div>

  <div class="dl-container">
    <div class="dl-grid">
      <div class="dl-card">
        <div class="dl-card-label">Download</div>
        <div class="dl-card-title">PILAR Pilot — Windows</div>
        <div class="dl-card-sub">Desktop application for Windows 10/11.<br>
          Requires no internet connection after setup.</div>
        <a href="{DOWNLOAD_URL}" class="dl-btn" {'target="_blank"' if DOWNLOAD_URL != '#' else ''}>
          ↓ Download PILAR Pilot
        </a>
      </div>

      <div class="dl-card">
        <div class="dl-card-label">Installation</div>
        <div class="dl-card-title">Getting started</div>
        <div class="install-steps">
          <div class="install-step"><span class="step-n">1</span>Run the installer (PILAR_Setup.exe)</div>
          <div class="install-step"><span class="step-n">2</span>Launch PILAR from your desktop</div>
          <div class="install-step"><span class="step-n">3</span>Create your admin account on first launch</div>
          <div class="install-step"><span class="step-n">4</span>Upload a CSV or connect your sensor feed</div>
        </div>
      </div>
    </div>

    <div class="invite-section">
      <div class="invite-box">
        <div class="dl-card-label">Team access</div>
        <div class="dl-card-title">Invite your team</div>
        <div class="dl-card-sub">Generate an invite link to share with colleagues at your facility.<br>
          Each link can be used up to 5 times by default.</div>
        <form class="invite-form" id="invite-form">
          <input class="invite-input" name="label" placeholder="Label, e.g. &quot;Maintenance team — Lyon plant&quot;">
          <input class="invite-input" name="max_uses" type="number" min="1" max="20" value="5"
            style="max-width:90px" placeholder="5">
          <button type="submit" class="dl-btn">Generate invite link</button>
        </form>
        <div class="invite-result" id="invite-result">
          <div class="invite-url" id="invite-url"></div>
          <button class="dl-btn dl-btn-sec" style="font-size:.78rem;padding:.3rem .75rem"
            onclick="navigator.clipboard.writeText(document.getElementById('invite-url').textContent);
            this.textContent='Copied!';setTimeout(()=>this.textContent='Copy to clipboard',1500)">
            Copy to clipboard
          </button>
        </div>
        {'<div style="margin-top:1.5rem">'+invite_rows+'</div>' if invite_rows else ''}
      </div>
    </div>
  </div>
</div>

<script>
document.getElementById('invite-form').addEventListener('submit', async function(e){{
  e.preventDefault();
  var fd = new FormData(this);
  var r = await fetch('/api/invite', {{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{label: fd.get('label'), max_uses: parseInt(fd.get('max_uses'))||5}})}});
  var j = await r.json();
  if(r.ok){{
    var box = document.getElementById('invite-result');
    document.getElementById('invite-url').textContent = j.link;
    box.style.display = 'block';
  }} else alert(j.error||'Error generating link');
}});
</script>
"""
    return _page('Your Download', body, DOWNLOAD_CSS, show_nav=True,
                 nav_user=user.email)


@app.route('/api/invite', methods=['POST'])
@login_required
def api_invite():
    data = request.get_json(silent=True) or {}
    label     = (data.get('label') or '').strip()[:200]
    max_uses  = min(int(data.get('max_uses', 5)), 20)
    token     = secrets.token_urlsafe(24)
    inv = InviteToken(token=token, created_by=session['user_id'],
                      max_uses=max_uses, label=label)
    db.session.add(inv)
    db.session.commit()
    return jsonify(link=f"{BASE_URL}/signup/invite/{token}")


# ─────────────────────────────────────────────────────────────────────────────
#  BOOT
# ─────────────────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    # Load runtime settings if saved
    cfg_path = os.path.join(os.path.dirname(__file__), 'pilar_site_settings.json')
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
            if 'download_url': DOWNLOAD_URL = cfg.get('download_url', DOWNLOAD_URL)
            if 'app_version':  APP_VERSION  = cfg.get('app_version',  APP_VERSION)
        except Exception:
            pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"PILAR Site — http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
