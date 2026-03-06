from flask import Flask, request, jsonify, render_template_string
import pickle
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pilar.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True)
    value = db.Column(db.String(500))

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    machine_type = db.Column(db.String(10))
    temp_air = db.Column(db.Float)
    temp_process = db.Column(db.Float)
    vitesse = db.Column(db.Float)
    couple = db.Column(db.Float)
    usure = db.Column(db.Float)
    risk = db.Column(db.Float)
    prediction = db.Column(db.Integer)
    zones = db.Column(db.String(500))
    mail_sent = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

with open("modele_pannes.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("modeles_zones.pkl", "rb") as f:
    modeles_zones = pickle.load(f)

FAILURE_ZONES = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failure'
}
COLONNES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'ecart_temp']
GMAIL = "guenbourali77@gmail.com"
GMAIL_PWD = "lpxm bplq znnx sbcx"
FAVICON = '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'

def get_setting(key, default=''):
    try:
        s = Settings.query.filter_by(key=key).first()
        return s.value if s else default
    except:
        return default

def set_setting(key, value):
    try:
        s = Settings.query.filter_by(key=key).first()
        if s:
            s.value = value
        else:
            s = Settings(key=key, value=value)
            db.session.add(s)
        db.session.commit()
    except Exception as e:
        print(f"Settings error: {e}")

# ─── SHARED CSS + BASE LAYOUT ────────────────────────────────────────────────
BASE_HEAD = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0e1118">
<link rel="icon" type="image/png" href="data:image/png;base64,{FAVICON}">
<title>Pilar</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;}}
:root{{
  --bg:#07090f;--surface:#0e1118;--surface2:#141820;
  --border:#1e2433;--border2:#252d3d;
  --teal:#0d9488;--teal-light:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;--green-dim:rgba(5,150,105,0.08);
  --amber:#d97706;--purple:#7c3aed;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --nav-h:60px;
}}
html,body{{height:100%;overflow:hidden;}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;}}
/* ── HEADER ── */
header{{height:52px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;padding:0 20px;background:var(--surface);flex-shrink:0;}}
.logo{{font-size:13px;font-weight:700;letter-spacing:4px;color:var(--teal-light);text-transform:uppercase;}}
.header-divider{{width:1px;height:18px;background:var(--border2);}}
.header-sub{{font-size:10px;letter-spacing:1.2px;color:var(--text3);text-transform:uppercase;}}
.header-right{{margin-left:auto;display:flex;gap:6px;align-items:center;}}
/* ── BOTTOM NAV ── */
.bottom-nav{{height:var(--nav-h);border-top:1px solid var(--border);background:var(--surface);display:flex;align-items:stretch;flex-shrink:0;}}
.nav-item{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;text-decoration:none;color:var(--text3);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;transition:color 0.15s;border:none;background:none;cursor:pointer;padding:8px 0;}}
.nav-item.active{{color:var(--teal-light);}}
.nav-item svg{{width:20px;height:20px;stroke-width:1.8;}}
/* ── PAGE CONTENT ── */
.page{{flex:1;overflow-y:auto;overflow-x:hidden;}}
.page::-webkit-scrollbar{{width:0;}}
.pad{{padding:16px;}}
/* ── CARDS ── */
.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:12px;}}
.card-title{{font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:12px;}}
/* ── RISK DISPLAY ── */
.risk-hero{{display:flex;align-items:center;justify-content:space-between;padding:18px;border-radius:8px;border:1px solid var(--border);background:var(--surface);margin-bottom:12px;transition:border-color 0.3s,background 0.3s;}}
.risk-hero.ok{{border-color:var(--green);background:var(--green-dim);}}
.risk-hero.alert{{border-color:var(--red);background:var(--red-dim);}}
.risk-status{{display:flex;flex-direction:column;gap:6px;}}
.status-badge{{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;}}
.status-badge.ok{{background:rgba(5,150,105,0.15);color:var(--green);}}
.status-badge.alert{{background:rgba(220,38,38,0.15);color:var(--red);}}
.status-dot{{width:6px;height:6px;border-radius:50%;}}
.status-dot.ok{{background:var(--green);}}
.status-dot.alert{{background:var(--red);animation:blink 1.2s infinite;}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.2;}}}}
.risk-num{{font-size:48px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;}}
.risk-num.ok{{color:var(--green);}} .risk-num.alert{{color:var(--red);}}
.risk-unit{{font-size:20px;color:var(--text3);}}
.risk-lbl{{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;text-align:right;margin-top:3px;}}
/* ── TYPE SELECTOR ── */
.type-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px;}}
.type-btn{{padding:10px 4px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text3);font-size:11px;cursor:pointer;text-align:center;transition:all 0.15s;}}
.type-btn.active{{border-color:var(--teal);background:var(--teal-dim);color:var(--teal-light);font-weight:600;}}
/* ── SENSORS ── */
.sensor{{margin-bottom:16px;}}
.sensor-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.sensor-name{{font-size:12px;color:var(--text2);}}
.val-wrap{{display:flex;align-items:center;gap:5px;}}
.val-input{{width:72px;padding:4px 8px;background:var(--surface2);border:1px solid var(--border2);border-radius:4px;color:var(--text);font-size:15px;font-weight:600;text-align:right;outline:none;-webkit-appearance:none;}}
.val-input:focus{{border-color:var(--teal);}}
.val-unit{{font-size:10px;color:var(--text3);}}
input[type=range]{{-webkit-appearance:none;width:100%;height:3px;background:var(--border2);border-radius:2px;outline:none;}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--teal);cursor:pointer;border:2px solid var(--bg);}}
.range-labels{{display:flex;justify-content:space-between;font-size:9px;color:var(--text3);margin-top:2px;}}
/* ── BUTTONS ── */
.btn-primary{{width:100%;padding:14px;background:var(--teal);color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:background 0.15s;margin-top:8px;}}
.btn-primary:hover{{background:var(--teal-light);}}
.btn-primary:disabled{{background:var(--border2);color:var(--text3);cursor:not-allowed;}}
/* ── FIELD ── */
.field-label{{font-size:9px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:7px;display:block;}}
.field-input{{width:100%;padding:10px 12px;background:var(--surface2);border:1px solid var(--border2);border-radius:6px;color:var(--text);font-size:13px;outline:none;transition:border-color 0.15s;}}
.field-input:focus{{border-color:var(--teal);}}
.field-input::placeholder{{color:var(--text3);}}
/* ── ZONE ROW ── */
.zone-row{{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface2);border-radius:6px;margin-bottom:6px;}}
.zone-name{{font-size:11px;color:var(--text2);flex:1;}}
.zone-bar-wrap{{width:80px;height:2px;background:var(--border2);border-radius:1px;}}
.zone-bar-fill{{height:100%;border-radius:1px;background:var(--red);}}
.zone-proba{{font-size:12px;font-weight:700;color:var(--amber);min-width:34px;text-align:right;}}
/* ── KPI GRID ── */
.kpi-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}}
.kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;}}
.kpi-val{{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums;}}
.kpi-val.ok{{color:var(--green);}} .kpi-val.alert{{color:var(--red);}} .kpi-val.amber{{color:var(--amber);}} .kpi-val.purple{{color:var(--purple);}}
.kpi-lbl{{font-size:9px;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;margin-top:3px;}}
/* ── HISTORY TABLE ── */
.tbl-wrap{{overflow-x:auto;border-radius:6px;background:var(--surface);border:1px solid var(--border);}}
table{{width:100%;border-collapse:collapse;font-size:11px;min-width:500px;}}
th{{text-align:left;padding:10px 12px;color:var(--text3);font-size:9px;font-weight:500;letter-spacing:1px;border-bottom:1px solid var(--border);text-transform:uppercase;white-space:nowrap;}}
td{{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--text2);white-space:nowrap;}}
tr:last-child td{{border-bottom:none;}}
.badge{{padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;}}
.badge.ok{{background:rgba(5,150,105,0.12);color:var(--green);}}
.badge.alert{{background:rgba(220,38,38,0.12);color:var(--red);}}
.mail-badge{{padding:2px 8px;border-radius:3px;font-size:10px;background:rgba(13,148,136,0.12);color:var(--teal-light);}}
/* ── CHAT ── */
.chat-wrap{{display:flex;flex-direction:column;height:100%;overflow:hidden;}}
.chat-msgs{{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}}
.chat-msgs::-webkit-scrollbar{{width:0;}}
.msg{{display:flex;flex-direction:column;gap:3px;max-width:85%;}}
.msg.user{{align-self:flex-end;align-items:flex-end;}}
.msg.bot{{align-self:flex-start;align-items:flex-start;}}
.msg-sender{{font-size:8px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;}}
.msg-bubble{{padding:10px 14px;border-radius:8px;font-size:13px;line-height:1.65;}}
.msg.user .msg-bubble{{background:var(--teal);color:#fff;border-radius:8px 8px 2px 8px;}}
.msg.bot .msg-bubble{{background:var(--surface2);border:1px solid var(--border);color:var(--text2);border-radius:8px 8px 8px 2px;}}
.typing-bubble{{color:var(--text3);font-style:italic;}}
.chat-input-area{{padding:10px 14px;border-top:1px solid var(--border);display:flex;gap:8px;background:var(--surface);flex-shrink:0;}}
.chat-textarea{{flex:1;padding:10px 12px;background:var(--surface2);border:1px solid var(--border2);border-radius:8px;color:var(--text);font-size:13px;outline:none;resize:none;font-family:inherit;max-height:100px;line-height:1.5;transition:border-color 0.15s;}}
.chat-textarea:focus{{border-color:var(--teal);}}
.chat-textarea::placeholder{{color:var(--text3);}}
.btn-send{{padding:10px 16px;background:var(--teal);color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;align-self:flex-end;transition:background 0.15s;flex-shrink:0;}}
.btn-send:disabled{{background:var(--border2);color:var(--text3);cursor:not-allowed;}}
/* ── ALERT BANNER ── */
.alert-banner{{padding:10px 14px;background:var(--teal-dim);border:1px solid var(--teal);border-radius:6px;font-size:11px;color:var(--teal-light);margin-bottom:10px;display:none;}}
/* ── NOTIF BTN ── */
.notif-btn{{padding:5px 11px;background:transparent;border:1px solid var(--border2);border-radius:4px;color:var(--text3);font-size:10px;cursor:pointer;transition:all 0.15s;white-space:nowrap;}}
.notif-btn.enabled{{border-color:var(--green);color:var(--green);}}
/* ── IDLE ── */
.idle{{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;gap:8px;color:var(--text3);text-align:center;}}
.idle .l1{{font-size:13px;}} .idle .l2{{font-size:11px;}}
/* ── DESKTOP OVERRIDES ── */
@media(min-width:768px){{
  .bottom-nav{{display:none;}}
  body{{overflow:hidden;}}
  .desktop-layout{{display:grid;grid-template-columns:300px 1fr 300px;flex:1;overflow:hidden;}}
  .desktop-col{{border-right:1px solid var(--border);overflow-y:auto;}}
  .desktop-col:last-child{{border-right:none;}}
  .desktop-col::-webkit-scrollbar{{width:3px;}}
  .desktop-col::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px;}}
  .page.desktop-hidden{{display:none;}}
}}
</style></head>"""

with open('/home/claude/etape7_new.py', 'a') as f:
    f.write(BASE_HEAD + '\n')
print("BASE_HEAD written")

# ─── NAV HTML ─────────────────────────────────────────────────────────────────
def nav_html(active):
    items = [
        ('/', 'monitor', '<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>', 'Monitor'),
        ('/twin', 'twin', '<path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/>', 'Twin'),
        ('/history', 'history', '<path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>', 'History'),
        ('/assistant', 'assistant', '<path d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>', 'Assistant'),
        ('/settings', 'settings', '<path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>', 'Settings'),
    ]
    html = '<nav class="bottom-nav">'
    for href, key, icon_path, label in items:
        cls = 'nav-item active' if active == key else 'nav-item'
        html += f'<a href="{href}" class="{cls}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><{icon_path}></svg>{label}</a>'
    html += '</nav>'
    return html


# ─── MONITOR PAGE ─────────────────────────────────────────────────────────────
HTML = BASE_HEAD + """
<body>
<header>
  <span class="logo">PILAR</span>
  <div class="header-divider"></div>
  <span class="header-sub">Monitor</span>
  <div class="header-right">
    <button class="notif-btn" id="notif-btn" onclick="toggleNotif()">Notifications</button>
  </div>
</header>
<div class="page pad">
  <div class="alert-banner" id="mail-notif">Alert dispatched</div>
  <div id="results-area">
    <div class="idle"><span class="l1">No analysis yet</span><span class="l2">Configure and run below</span></div>
  </div>
  <div class="card">
    <div class="card-title">Machine class</div>
    <div class="type-grid">
      <div class="type-btn active" data-val="0" onclick="selectType(this)">L — Low</div>
      <div class="type-btn" data-val="1" onclick="selectType(this)">M — Med</div>
      <div class="type-btn" data-val="2" onclick="selectType(this)">H — High</div>
    </div>
    <div class="card-title">Sensor parameters</div>
    <div class="sensor">
      <div class="sensor-row"><span class="sensor-name">Air temperature</span><div class="val-wrap"><input class="val-input" type="number" id="n_ta" value="300" min="295" max="305" step="0.1" oninput="si('temp_air','n_ta')"><span class="val-unit">K</span></div></div>
      <input type="range" id="temp_air" min="295" max="305" step="0.1" value="300" oninput="ss('temp_air','n_ta',1)">
      <div class="range-labels"><span>295K</span><span>305K</span></div>
    </div>
    <div class="sensor">
      <div class="sensor-row"><span class="sensor-name">Process temperature</span><div class="val-wrap"><input class="val-input" type="number" id="n_tp" value="310" min="305" max="315" step="0.1" oninput="si('temp_process','n_tp')"><span class="val-unit">K</span></div></div>
      <input type="range" id="temp_process" min="305" max="315" step="0.1" value="310" oninput="ss('temp_process','n_tp',1)">
      <div class="range-labels"><span>305K</span><span>315K</span></div>
    </div>
    <div class="sensor">
      <div class="sensor-row"><span class="sensor-name">Rotational speed</span><div class="val-wrap"><input class="val-input" type="number" id="n_v" value="1500" min="1000" max="3000" step="10" oninput="si('vitesse','n_v')"><span class="val-unit">rpm</span></div></div>
      <input type="range" id="vitesse" min="1000" max="3000" step="10" value="1500" oninput="ss('vitesse','n_v',0)">
      <div class="range-labels"><span>1000</span><span>3000rpm</span></div>
    </div>
    <div class="sensor">
      <div class="sensor-row"><span class="sensor-name">Torque</span><div class="val-wrap"><input class="val-input" type="number" id="n_c" value="40" min="3" max="80" step="0.1" oninput="si('couple','n_c')"><span class="val-unit">Nm</span></div></div>
      <input type="range" id="couple" min="3" max="80" step="0.1" value="40" oninput="ss('couple','n_c',1)">
      <div class="range-labels"><span>3Nm</span><span>80Nm</span></div>
    </div>
    <div class="sensor">
      <div class="sensor-row"><span class="sensor-name">Tool wear</span><div class="val-wrap"><input class="val-input" type="number" id="n_u" value="100" min="0" max="250" step="1" oninput="si('usure','n_u')"><span class="val-unit">min</span></div></div>
      <input type="range" id="usure" min="0" max="250" step="1" value="100" oninput="ss('usure','n_u',0)">
      <div class="range-labels"><span>0</span><span>250min</span></div>
    </div>
    <button class="btn-primary" id="btn" onclick="analyse()">Run Analysis</button>
  </div>
</div>
""" + nav_html('monitor') + """
<script>
let mType=0,lastResult=null,lastData=null;
function updateNotif(){const b=document.getElementById('notif-btn');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent='Notifs ON';b.className='notif-btn enabled';}else if(p==='denied'){b.textContent='Blocked';b.className='notif-btn';}else{b.textContent='Enable Notifs';b.className='notif-btn';}}
async function toggleNotif(){if(Notification.permission==='granted')return;if(Notification.permission==='denied'){alert('Enable in browser settings');return;}await Notification.requestPermission();updateNotif();}
function sendNotif(risk,zones){if(Notification.permission!=='granted')return;new Notification(`Pilar — Risk: ${risk}%`,{body:zones.length?'Zones: '+zones.map(z=>z.nom).join(', '):'No specific zone',requireInteraction:true,tag:'pilar-alert'});}
updateNotif();
function selectType(el){document.querySelectorAll('.type-btn').forEach(b=>b.classList.remove('active'));el.classList.add('active');mType=parseInt(el.dataset.val);}
function ss(sid,nid,d){document.getElementById(nid).value=parseFloat(document.getElementById(sid).value).toFixed(d);}
function si(sid,nid){const v=parseFloat(document.getElementById(nid).value);if(!isNaN(v))document.getElementById(sid).value=v;}
function gv(id){return parseFloat(document.getElementById(id).value);}
async function analyse(){
  const btn=document.getElementById('btn');btn.disabled=true;btn.textContent='Analyzing...';
  lastData={type:mType,temp_air:gv('n_ta'),temp_process:gv('n_tp'),vitesse:gv('n_v'),couple:gv('n_c'),usure:gv('n_u')};
  const res=await fetch('/predire',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastData)});
  lastResult=await res.json();
  sessionStorage.setItem('lastResult',JSON.stringify(lastResult));
  sessionStorage.setItem('lastData',JSON.stringify(lastData));
  renderResults(lastResult);
  if(lastResult.probabilite>=50){sendNotif(lastResult.probabilite,lastResult.zones);const n=document.getElementById('mail-notif');n.style.display='block';setTimeout(()=>n.style.display='none',4000);}
  btn.disabled=false;btn.textContent='Run Analysis';
}
function renderResults(r){
  const a=r.prediction===1,cls=a?'alert':'ok',st=a?'Anomaly Detected':'Normal Operation';
  let zH='';
  if(a&&r.zones.length>0){zH='<div class="card"><div class="card-title">Failure zone analysis</div>'+r.zones.map(z=>`<div class="zone-row"><span class="zone-name">${z.nom}</span><div class="zone-bar-wrap"><div class="zone-bar-fill" style="width:${z.proba}%"></div></div><span class="zone-proba">${z.proba}%</span></div>`).join('')+'</div>';}
  document.getElementById('results-area').innerHTML=`
    <div class="risk-hero ${cls}">
      <div class="risk-status">
        <div class="status-badge ${cls}"><span class="status-dot ${cls}"></span>${st}</div>
        <div style="font-size:10px;color:var(--text3);margin-top:4px">${new Date().toLocaleString('en-GB')}</div>
      </div>
      <div>
        <div class="risk-num ${cls}">${r.probabilite}<span class="risk-unit">%</span></div>
        <div class="risk-lbl">Failure prob.</div>
      </div>
    </div>${zH}`;
}
</script>
</body></html>"""


# ─── ASSISTANT PAGE ───────────────────────────────────────────────────────────
ASSISTANT_HTML = BASE_HEAD + """
<body style="overflow:hidden;">
<header>
  <span class="logo">PILAR</span>
  <div class="header-divider"></div>
  <span class="header-sub">AI Assistant</span>
  <div class="header-right" style="font-size:9px;color:var(--text3);">Claude Haiku</div>
</header>
<div class="page chat-wrap">
  <div class="chat-msgs" id="chat-msgs">
    <div class="msg bot"><span class="msg-sender">Pilar</span>
      <div class="msg-bubble">Hello! I'm Pilar, your industrial AI assistant. I can help you with:<br><br>
      • Machine failure analysis and prevention<br>
      • Sensor data interpretation<br>
      • Maintenance recommendations<br>
      • Any technical or general questions<br><br>
      What can I help you with today?</div>
    </div>
  </div>
  <div class="chat-input-area">
    <textarea class="chat-textarea" id="chat-input" placeholder="Ask anything..." rows="1"
      oninput="autoResize(this)"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}"></textarea>
    <button class="btn-send" id="btn-send" onclick="send()">Send</button>
  </div>
</div>
""" + nav_html('assistant') + """
<script>
let history=[];
let msgId=0;

function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,100)+'px';}

function addMsg(role,text,typing=false){
  const id='m'+(++msgId),d=document.createElement('div');
  d.className='msg '+role;d.id=id;
  const escaped=text.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
  d.innerHTML=`<span class="msg-sender">${role==='user'?'You':'Pilar'}</span><div class="msg-bubble${typing?' typing-bubble':''}">${escaped}</div>`;
  const c=document.getElementById('chat-msgs');c.appendChild(d);c.scrollTop=c.scrollHeight;
  return id;
}
function removeMsg(id){const el=document.getElementById(id);if(el)el.remove();}

async function send(){
  const input=document.getElementById('chat-input');
  const msg=input.value.trim();if(!msg)return;
  history.push({role:'user',content:msg});
  addMsg('user',msg);
  input.value='';input.style.height='auto';
  const tid=addMsg('bot','Thinking...', true);
  document.getElementById('btn-send').disabled=true;

  // Get context from sessionStorage if available
  let context=null;
  try{
    const lr=sessionStorage.getItem('lastResult');
    const ld=sessionStorage.getItem('lastData');
    if(lr&&ld)context={result:JSON.parse(lr),data:JSON.parse(ld)};
  }catch(e){}

  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:msg,context,history:history.slice(-20)})});
  const data=await res.json();
  history.push({role:'assistant',content:data.reply});
  removeMsg(tid);
  addMsg('bot',data.reply);
  document.getElementById('btn-send').disabled=false;
}
</script>
</body></html>"""


# ─── TWIN PAGE ────────────────────────────────────────────────────────────────
TWIN_HTML = BASE_HEAD + """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<body>
<header>
  <span class="logo">PILAR</span>
  <div class="header-divider"></div>
  <span class="header-sub">Digital Twin</span>
</header>
<div class="page pad" id="twin-content">
  <div class="idle"><span class="l1">Loading simulation...</span></div>
</div>
""" + nav_html('twin') + """
<script>
const PL={paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#64748b',size:10},margin:{t:8,b:36,l:40,r:8},xaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},yaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},legend:{bgcolor:'transparent',font:{size:9}},hovermode:'x unified'};
const PC={responsive:true,displayModeBar:false};

async function loadTwin(){
  const res=await fetch('/api/twin');const d=await res.json();
  if(!d.has_data){document.getElementById('twin-content').innerHTML=`<div class="idle"><span class="l1">No data yet</span><span class="l2">Run an analysis on Monitor first</span><a href="/" style="margin-top:16px;padding:12px 20px;background:var(--teal);color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">Go to Monitor</a></div>`;return;}
  const bCls=d.failure_hours===null?'ok':d.failure_hours<6?'alert':'amber';
  const bTitle=d.failure_hours===null?'System Healthy':'Failure predicted in '+d.failure_hours+'h';
  document.getElementById('twin-content').innerHTML=`
    <div class="risk-hero ${bCls}" style="margin-bottom:12px">
      <div><div class="status-badge ${bCls}"><span class="status-dot ${bCls}"></span>${bTitle}</div>
        <div style="font-size:10px;color:var(--text3);margin-top:4px">Trend: ${d.trend}</div></div>
      <div><div class="risk-num ${bCls}">${d.current_risk}<span class="risk-unit">%</span></div><div class="risk-lbl">Current risk</div></div>
    </div>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-val amber">${d.avg_risk_24h}%</div><div class="kpi-lbl">Avg risk</div></div>
      <div class="kpi-card"><div class="kpi-val ${d.anomaly_rate>=30?'alert':'ok'}">${d.anomaly_rate}%</div><div class="kpi-lbl">Anomaly rate</div></div>
    </div>
    <div class="card"><div class="card-title">Risk — History + 24h Simulation</div><div id="ch-risk" style="height:220px"></div></div>
    <div class="card"><div class="card-title">Tool wear projection</div><div id="ch-wear" style="height:180px"></div></div>
    <div class="card"><div class="card-title">Process temperature</div><div id="ch-temp" style="height:180px"></div></div>
    <div class="card">
      <div class="card-title">Scenario Simulator</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
        <div><label style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:5px">Speed (rpm)</label><input class="field-input" type="number" id="wi-v" value="${d.last_params.vitesse}" step="10"></div>
        <div><label style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:5px">Torque (Nm)</label><input class="field-input" type="number" id="wi-c" value="${d.last_params.couple}" step="0.1"></div>
        <div><label style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:5px">Tool wear (min)</label><input class="field-input" type="number" id="wi-u" value="${d.last_params.usure}" step="1"></div>
        <div><label style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:5px">Air temp (K)</label><input class="field-input" type="number" id="wi-ta" value="${d.last_params.temp_air}" step="0.1"></div>
      </div>
      <button class="btn-primary" onclick="runWhatIf()">Simulate</button>
      <div id="wi-result" style="margin-top:12px"></div>
    </div>`;
  Plotly.newPlot('ch-risk',[
    {x:d.history_times,y:d.history_risks,name:'History',type:'scatter',mode:'lines+markers',line:{color:'#14b8a6',width:2},marker:{size:5}},
    {x:d.future_times,y:d.future_risks,name:'Simulated',type:'scatter',mode:'lines',line:{color:'#7c3aed',width:2,dash:'dot'},fill:'tozeroy',fillcolor:'rgba(124,58,237,0.04)'},
    {x:[...d.history_times,...d.future_times],y:Array(d.history_times.length+d.future_times.length).fill(50),name:'Threshold',type:'scatter',mode:'lines',line:{color:'#dc2626',width:1,dash:'dash'}}
  ],{...PL,yaxis:{...PL.yaxis,range:[0,105]}},PC);
  Plotly.newPlot('ch-wear',[{x:d.history_times,y:d.history_wear,name:'Actual',type:'scatter',mode:'lines+markers',line:{color:'#d97706',width:2},marker:{size:4}},{x:d.future_times,y:d.future_wear,name:'Projected',type:'scatter',mode:'lines',line:{color:'#d97706',width:2,dash:'dot'}}],PL,PC);
  Plotly.newPlot('ch-temp',[{x:d.history_times,y:d.history_temp,name:'Actual',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2}},{x:d.future_times,y:d.future_temp,name:'Projected',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2,dash:'dot'}}],PL,PC);
}
async function runWhatIf(){
  const p={type:1,temp_air:parseFloat(document.getElementById('wi-ta').value),temp_process:parseFloat(document.getElementById('wi-ta').value)+10,vitesse:parseFloat(document.getElementById('wi-v').value),couple:parseFloat(document.getElementById('wi-c').value),usure:parseFloat(document.getElementById('wi-u').value)};
  const res=await fetch('/api/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
  const d=await res.json();
  const c={ok:'#059669',amber:'#d97706',alert:'#dc2626'};const cls=d.risk>=50?'alert':d.risk>=22?'amber':'ok';
  document.getElementById('wi-result').innerHTML=`<div style="padding:14px;background:var(--bg);border:1px solid ${c[cls]};border-radius:6px"><div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase">Simulated risk</div><div style="font-size:32px;font-weight:800;color:${c[cls]};margin:4px 0">${d.risk}%</div><div style="font-size:12px;font-weight:600;color:${c[cls]}">${d.status}</div><div style="font-size:11px;color:var(--text3);margin-top:3px">${d.message}</div>${d.zones.length?`<div style="font-size:10px;color:var(--amber);margin-top:6px">Zones: ${d.zones.map(z=>z.nom+' '+z.proba+'%').join(' · ')}</div>`:''}</div>`;
}
loadTwin();
</script>
</body></html>"""


# ─── HISTORY PAGE ─────────────────────────────────────────────────────────────
HISTORY_HTML = BASE_HEAD + """
<body>
<header>
  <span class="logo">PILAR</span>
  <div class="header-divider"></div>
  <span class="header-sub">History</span>
</header>
<div class="page pad">
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-val">{{ total }}</div><div class="kpi-lbl">Total analyses</div></div>
    <div class="kpi-card"><div class="kpi-val alert">{{ anomalies }}</div><div class="kpi-lbl">Anomalies</div></div>
    <div class="kpi-card"><div class="kpi-val amber">{{ avg_risk }}%</div><div class="kpi-lbl">Avg risk</div></div>
    <div class="kpi-card"><div class="kpi-val ok">{{ mails }}</div><div class="kpi-lbl">Alerts sent</div></div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Time</th><th>Class</th><th>Risk</th><th>Status</th><th>Zones</th><th>Alert</th></tr></thead>
      <tbody>
        {% for a in analyses %}
        <tr>
          <td>{{ a.timestamp.strftime('%d/%m %H:%M') }}</td>
          <td>{{ a.machine_type }}</td>
          <td>{{ a.risk }}%</td>
          <td><span class="badge {{ 'alert' if a.prediction else 'ok' }}">{{ 'Anomaly' if a.prediction else 'Normal' }}</span></td>
          <td>{{ a.zones or '—' }}</td>
          <td>{% if a.mail_sent %}<span class="mail-badge">Sent</span>{% else %}—{% endif %}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
""" + nav_html('history') + """
</body></html>"""

# ─── SETTINGS PAGE ────────────────────────────────────────────────────────────
SETTINGS_HTML = BASE_HEAD + """
<body>
<header>
  <span class="logo">PILAR</span>
  <div class="header-divider"></div>
  <span class="header-sub">Settings</span>
</header>
<div class="page pad">
  <div class="card">
    <div class="card-title">Alert email</div>
    <label class="field-label">Recipient email address</label>
    <input class="field-input" type="email" id="email" placeholder="maintenance@company.com">
    <div style="font-size:10px;color:var(--green);margin-top:6px;display:none" id="saved-msg">Saved successfully</div>
    <button class="btn-primary" style="margin-top:12px" onclick="saveEmail()">Save Email</button>
  </div>
  <div class="card">
    <div class="card-title">Browser notifications</div>
    <p style="font-size:12px;color:var(--text2);margin-bottom:12px;line-height:1.6">Enable push notifications to receive alerts when failure risk exceeds 50%, even when the app is in the background.</p>
    <button class="btn-primary" id="notif-btn" onclick="toggleNotif()" style="background:var(--purple)">Enable Notifications</button>
  </div>
  <div class="card">
    <div class="card-title">System info</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)">Version</span><span>Pilar v2.0</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)">AI Model</span><span>Claude Haiku</span></div>
      <div style="display:flex;justify-content:space-between;font-size:12px"><span style="color:var(--text3)">Database</span><span>SQLite</span></div>
    </div>
  </div>
</div>
""" + nav_html('settings') + """
<script>
async function saveEmail(){
  const email=document.getElementById('email').value;
  if(!email)return;
  await fetch('/set_email',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email})});
  const s=document.getElementById('saved-msg');s.style.display='block';setTimeout(()=>s.style.display='none',3000);
}
function updateNotif(){const b=document.getElementById('notif-btn');if(!b)return;const p=Notification.permission;if(p==='granted'){b.textContent='Notifications Enabled';b.style.background='var(--green)';}else if(p==='denied'){b.textContent='Blocked — Enable in Settings';b.style.background='var(--red)';}}
async function toggleNotif(){if(Notification.permission==='granted')return;await Notification.requestPermission();updateNotif();}
updateNotif();
</script>
</body></html>"""


# ─── BACKEND FUNCTIONS ────────────────────────────────────────────────────────
def predict_risk(params):
    ecart_temp = params['temp_process'] - params['temp_air']
    donnees = pd.DataFrame([[
        params['type'], params['temp_air'], params['temp_process'],
        params['vitesse'], params['couple'], params['usure'], ecart_temp
    ]], columns=COLONNES)
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
    zones_rows = "".join(f"""<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#94a3b8;font-size:12px;">{z['nom']}</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;"><span style="color:#dc2626;font-weight:700;">{z['proba']}%</span></td></tr>""" for z in zones_risque) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;font-size:12px;">No specific zone identified</td></tr>'
    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;padding:40px 0;"><tr><td align="center">
<table width="540" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #1e2433;">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td><div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;">PILAR</div></td>
    <td align="right"><span style="padding:4px 10px;background:rgba(220,38,38,0.12);border:1px solid #dc2626;border-radius:3px;color:#dc2626;font-size:10px;font-weight:700;letter-spacing:2px;">FAILURE ALERT</span></td>
  </tr></table>
</td></tr>
<tr><td style="padding:28px;">
  <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability</div>
  <div style="font-size:52px;font-weight:800;color:{sc};line-height:1;">{probabilite}<span style="font-size:24px;color:#64748b;">%</span></div>
  <div style="margin-top:8px;"><span style="padding:3px 10px;background:rgba(220,38,38,0.1);border:1px solid {sc};border-radius:3px;font-size:10px;font-weight:700;color:{sc};">SEVERITY: {severity}</span></div>
</td></tr>
<tr><td style="padding:0 28px 24px;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">
    <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Machine class</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{mtype}</td></tr>
    <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Air temperature</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_air")} K</td></tr>
    <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Speed</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("vitesse")} rpm</td></tr>
    <tr><td style="padding:8px 12px;color:#64748b;font-size:11px;">Tool wear</td><td style="padding:8px 12px;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("usure")} min</td></tr>
  </table>
</td></tr>
<tr><td style="padding:0 28px 24px;">
  <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:10px;">Failure Zones</div>
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">{zones_rows}</table>
</td></tr>
<tr><td style="padding:16px 28px;border-top:1px solid #1e2433;background:#0a0d16;">
  <div style="font-size:10px;color:#64748b;">Pilar Predictive Maintenance · {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
</td></tr>
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

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/assistant')
def assistant():
    return render_template_string(ASSISTANT_HTML)

@app.route('/twin')
def twin():
    return render_template_string(TWIN_HTML)

@app.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()).all()
    total = len(analyses)
    anomalies = sum(1 for a in analyses if a.prediction)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1) if total > 0 else 0
    mails = sum(1 for a in analyses if a.mail_sent)
    return render_template_string(HISTORY_HTML, analyses=analyses, total=total,
                                   anomalies=anomalies, avg_risk=avg_risk, mails=mails)

@app.route('/settings')
def settings():
    return render_template_string(SETTINGS_HTML)

@app.route('/set_email', methods=['POST'])
def set_email():
    data = request.json
    set_setting('responsible_email', data.get('email', ''))
    return jsonify({'status': 'ok'})

@app.route('/predire', methods=['POST'])
def predire():
    data = request.json
    probabilite, prediction, zones_risque = predict_risk(data)
    mail_envoye = False
    email = get_setting('responsible_email')
    if probabilite >= 50 and email:
        threading.Thread(target=envoyer_alerte, args=(email, probabilite, zones_risque, data), daemon=True).start()
        mail_envoye = True
    machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
    zones_str = ', '.join([z['nom'] for z in zones_risque]) if zones_risque else ''
    db.session.add(Analysis(
        machine_type=machine_types.get(data['type'], 'Unknown'),
        temp_air=data['temp_air'], temp_process=data['temp_process'],
        vitesse=data['vitesse'], couple=data['couple'], usure=data['usure'],
        risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye))
    db.session.commit()
    return jsonify({'prediction': prediction, 'probabilite': probabilite,
                    'zones': zones_risque, 'mail_envoye': mail_envoye})

@app.route('/api/twin')
def api_twin():
    analyses = Analysis.query.order_by(Analysis.timestamp.asc()).all()
    if not analyses:
        return jsonify({'has_data': False})
    last = analyses[-1]
    history_times = [a.timestamp.strftime('%H:%M') for a in analyses]
    history_risks = [a.risk for a in analyses]
    history_wear  = [a.usure for a in analyses]
    history_temp  = [a.temp_process for a in analyses]
    future_times, future_risks, future_wear, future_temp = [], [], [], []
    now = datetime.utcnow()
    cur_u, cur_tp = last.usure, last.temp_process
    failure_hours = None
    for h in range(1, 25):
        cur_u = min(cur_u + 1.5, 250)
        cur_tp = min(cur_tp + 0.05, 315)
        risk, pred, _ = predict_risk({'type': 1, 'temp_air': last.temp_air, 'temp_process': cur_tp,
                                       'vitesse': last.vitesse, 'couple': last.couple, 'usure': cur_u})
        future_times.append((now + timedelta(hours=h)).strftime('%H:%M'))
        future_risks.append(risk)
        future_wear.append(round(cur_u, 1))
        future_temp.append(round(cur_tp, 2))
        if failure_hours is None and risk >= 50:
            failure_hours = h
    total = len(analyses)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1)
    anomaly_rate = round(sum(1 for a in analyses if a.prediction) / total * 100, 1)
    trend = 'Stable'
    if len(history_risks) >= 3:
        diff = history_risks[-1] - history_risks[-3]
        trend = 'Increasing' if diff > 2 else 'Decreasing' if diff < -2 else 'Stable'
    return jsonify({'has_data': True, 'current_risk': last.risk, 'avg_risk_24h': avg_risk,
        'anomaly_rate': anomaly_rate, 'total_analyses': total, 'failure_hours': failure_hours,
        'trend': trend, 'history_times': history_times, 'history_risks': history_risks,
        'history_wear': history_wear, 'history_temp': history_temp,
        'future_times': future_times, 'future_risks': future_risks,
        'future_wear': future_wear, 'future_temp': future_temp,
        'last_params': {'temp_air': last.temp_air, 'vitesse': last.vitesse,
                        'couple': last.couple, 'usure': last.usure}})

@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    params = request.json
    params['temp_process'] = params['temp_air'] + 10
    risk, pred, zones = predict_risk(params)
    if pred == 0:
        status, message = 'Normal Operation', 'No failure predicted under these conditions.'
    elif risk < 50:
        status, message = 'Low Risk', 'Minor anomaly. Continue monitoring.'
    else:
        status, message = 'High Failure Risk', 'Reduce tool wear or torque immediately.'
    return jsonify({'risk': risk, 'status': status, 'message': message, 'zones': zones})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    context = data.get('context')
    chat_history = data.get('history', [])

    # Build rich system prompt
    if context:
        r = context.get('result', {})
        d = context.get('data', {})
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        mtype = machine_types.get(d.get('type', 0), 'Unknown')
        zones_str = ', '.join([f"{z['nom']} ({z['proba']}%)" for z in r.get('zones', [])]) or 'none'
        ctx_block = f"""
Current machine state from last analysis:
- Status: {'ANOMALY DETECTED' if r.get('prediction') else 'Normal'} | Risk: {r.get('probabilite')}% | Class: {mtype}
- Air temp: {d.get('temp_air')} K | Process temp: {d.get('temp_process')} K
- Speed: {d.get('vitesse')} rpm | Torque: {d.get('couple')} Nm | Tool wear: {d.get('usure')} min
- Failure zones: {zones_str}
"""
    else:
        ctx_block = ""

    system_prompt = f"""You are Pilar, an advanced AI assistant embedded in an industrial predictive maintenance platform.
{ctx_block}
Your capabilities:
- Deep expertise in industrial machinery, predictive maintenance, failure analysis
- Knowledge of sensors (temperature, vibration, torque, RPM, tool wear)
- Understanding of ML models for failure prediction
- General knowledge on any topic the user asks about
- Ability to explain complex technical concepts simply

Behavior guidelines:
- Answer the user's ACTUAL question directly — never deflect or give generic responses
- If they ask about machine data, analyze it specifically
- If they ask general questions (coding, science, math, etc.), answer those fully
- Be conversational and helpful, not robotic
- Use the machine context only when relevant to the question
- Respond in the same language as the user (French or English)
- Give detailed, useful answers — not just 1-2 sentences unless the question is simple"""

    messages = [{"role": h['role'], "content": h['content']} for h in chat_history[:-1]]
    messages.append({"role": "user", "content": message})

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        reply = response.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        reply = "Je suis disponible pour répondre à vos questions. / I'm available to answer your questions."
    return jsonify({'reply': reply})

if __name__ == '__main__':
    print("Pilar v2 — http://localhost:5000")
    app.run(debug=True, host='0.0.0.0')