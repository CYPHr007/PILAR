from flask import Flask, request, jsonify, render_template_string
import pickle
import threading
import smtplib
import os
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

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pilar — Predictive Maintenance</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #07090f; --surface: #0e1118; --surface2: #141820;
      --border: #1e2433; --border2: #252d3d;
      --teal: #0d9488; --teal-light: #14b8a6; --teal-dim: rgba(13,148,136,0.08);
      --red: #dc2626; --red-dim: rgba(220,38,38,0.08);
      --green: #059669; --green-dim: rgba(5,150,105,0.08);
      --amber: #d97706;
      --text: #e2e8f0; --text2: #94a3b8; --text3: #64748b;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    header { padding: 0 32px; height: 52px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; flex-shrink: 0; background: var(--surface); }
    .logo { font-size: 13px; font-weight: 700; letter-spacing: 4px; color: var(--teal-light); text-transform: uppercase; }
    .header-divider { width: 1px; height: 20px; background: var(--border2); }
    .header-sub { color: var(--text3); font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }
    .nav-links { margin-left: auto; display: flex; gap: 6px; }
    .nav-link { padding: 6px 14px; background: transparent; border: 1px solid var(--border2); border-radius: 4px; color: var(--text3); font-size: 11px; letter-spacing: 0.5px; text-decoration: none; transition: all 0.15s; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    .notif-btn { padding: 6px 14px; background: transparent; border: 1px solid var(--border2); border-radius: 4px; color: var(--text3); font-size: 11px; cursor: pointer; transition: all 0.15s; }
    .notif-btn.enabled { border-color: var(--green); color: var(--green); }
    .notif-btn.denied { border-color: var(--red); color: var(--red); }
    main { display: grid; grid-template-columns: 300px 1fr 280px; flex: 1; overflow: hidden; }
    .panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
    .panel:last-child { border-right: none; }
    .panel-body { padding: 20px 18px; overflow-y: auto; flex: 1; }
    .panel-body::-webkit-scrollbar { width: 3px; }
    .panel-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
    .field-label { font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; margin-bottom: 8px; display: block; }
    .type-selector { display: grid; grid-template-columns: repeat(3,1fr); gap: 5px; margin-bottom: 20px; }
    .type-btn { padding: 8px 4px; background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; color: var(--text3); font-size: 11px; cursor: pointer; text-align: center; transition: all 0.15s; }
    .type-btn:hover { border-color: var(--border2); color: var(--text2); }
    .type-btn.active { border-color: var(--teal); background: var(--teal-dim); color: var(--teal-light); font-weight: 600; }
    .sensor { margin-bottom: 18px; }
    .sensor-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; }
    .sensor-name { font-size: 11px; color: var(--text2); }
    .value-input-wrap { display: flex; align-items: center; gap: 4px; }
    .value-input { width: 68px; padding: 3px 7px; background: var(--surface2); border: 1px solid var(--border2); border-radius: 3px; color: var(--text); font-size: 14px; font-weight: 600; text-align: right; outline: none; transition: border-color 0.15s; }
    .value-input:focus { border-color: var(--teal); }
    .sensor-unit { font-size: 10px; color: var(--text3); }
    input[type=range] { -webkit-appearance: none; width: 100%; height: 2px; background: var(--border2); border-radius: 1px; outline: none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 11px; height: 11px; border-radius: 50%; background: var(--teal); cursor: pointer; }
    .range-labels { display: flex; justify-content: space-between; font-size: 9px; color: var(--text3); margin-top: 3px; }
    .email-section { margin-bottom: 18px; }
    .email-input { width: 100%; padding: 8px 10px; background: var(--surface2); border: 1px solid var(--border2); border-radius: 4px; color: var(--text); font-size: 11px; outline: none; transition: border-color 0.15s; }
    .email-input:focus { border-color: var(--teal); }
    .email-input::placeholder { color: var(--text3); }
    .email-saved { font-size: 10px; color: var(--green); margin-top: 4px; display: none; }
    .btn-analyze { width: 100%; padding: 11px; background: var(--teal); color: #fff; border: none; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; cursor: pointer; transition: background 0.15s; margin-top: 6px; }
    .btn-analyze:hover { background: var(--teal-light); }
    .btn-analyze:disabled { background: var(--border2); color: var(--text3); cursor: not-allowed; }
    .status-card { padding: 18px 20px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); display: flex; align-items: center; gap: 16px; margin-bottom: 20px; transition: border-color 0.3s; }
    .status-card.ok { border-color: var(--green); background: var(--green-dim); }
    .status-card.alert { border-color: var(--red); background: var(--red-dim); }
    .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 3px; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }
    .status-badge.ok { background: rgba(5,150,105,0.15); color: var(--green); }
    .status-badge.alert { background: rgba(220,38,38,0.15); color: var(--red); }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; }
    .status-dot.ok { background: var(--green); }
    .status-dot.alert { background: var(--red); animation: blink 1.2s infinite; }
    @keyframes blink { 0%,100%{opacity:1;}50%{opacity:0.2;} }
    .risk-block { margin-left: auto; text-align: right; }
    .risk-number { font-size: 38px; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; }
    .risk-number.ok { color: var(--green); }
    .risk-number.alert { color: var(--red); }
    .risk-label { font-size: 9px; color: var(--text3); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }
    .section-title { font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; margin-bottom: 10px; }
    .zone-row { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: var(--surface); border-radius: 4px; border: 1px solid var(--border); margin-bottom: 6px; }
    .zone-name { font-size: 11px; color: var(--text2); min-width: 160px; }
    .zone-bar-wrap { flex: 1; height: 2px; background: var(--border2); border-radius: 1px; }
    .zone-bar-fill { height: 100%; border-radius: 1px; background: var(--red); transition: width 0.4s; }
    .zone-proba { font-size: 12px; font-weight: 700; color: var(--amber); min-width: 36px; text-align: right; }
    .idle-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 6px; color: var(--text3); }
    .idle-state .line1 { font-size: 12px; letter-spacing: 1px; }
    .idle-state .line2 { font-size: 10px; }
    .history-table { margin-top: 20px; }
    table { width: 100%; border-collapse: collapse; font-size: 10px; }
    th { text-align: left; padding: 6px 8px; color: var(--text3); font-weight: 500; letter-spacing: 1px; border-bottom: 1px solid var(--border); text-transform: uppercase; }
    td { padding: 8px 8px; border-bottom: 1px solid var(--border); color: var(--text2); }
    td.ok { color: var(--green); }
    td.alert { color: var(--red); }
    tr:last-child td { border-bottom: none; }
    .alert-notif { margin-top: 8px; padding: 8px 12px; background: var(--teal-dim); border: 1px solid var(--teal); border-radius: 4px; font-size: 10px; color: var(--teal-light); letter-spacing: 0.5px; display: none; }
    .chat-panel { display: flex; flex-direction: column; overflow: hidden; }
    .chat-header { padding: 0 16px; height: 52px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; flex-shrink: 0; background: var(--surface); }
    .chat-title { font-size: 11px; font-weight: 600; color: var(--text2); letter-spacing: 1px; }
    .chat-subtitle { font-size: 9px; color: var(--text3); margin-top: 2px; letter-spacing: 0.5px; }
    .chat-messages { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
    .chat-messages::-webkit-scrollbar { width: 3px; }
    .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    .msg { display: flex; flex-direction: column; gap: 2px; max-width: 88%; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.bot { align-self: flex-start; align-items: flex-start; }
    .msg-sender { font-size: 8px; letter-spacing: 1.5px; color: var(--text3); text-transform: uppercase; }
    .msg-bubble { padding: 8px 12px; border-radius: 6px; font-size: 11px; line-height: 1.6; }
    .msg.user .msg-bubble { background: var(--teal); color: #fff; border-radius: 6px 6px 2px 6px; }
    .msg.bot .msg-bubble { background: var(--surface2); border: 1px solid var(--border); color: var(--text2); border-radius: 6px 6px 6px 2px; }
    .msg-bubble.typing { color: var(--text3); font-style: italic; }
    .chat-input-area { padding: 10px 12px; border-top: 1px solid var(--border); display: flex; gap: 6px; flex-shrink: 0; }
    .chat-input { flex: 1; padding: 8px 10px; background: var(--surface2); border: 1px solid var(--border2); border-radius: 4px; color: var(--text); font-size: 11px; outline: none; resize: none; font-family: inherit; transition: border-color 0.15s; max-height: 80px; }
    .chat-input:focus { border-color: var(--teal); }
    .chat-input::placeholder { color: var(--text3); }
    .btn-send { padding: 8px 12px; background: var(--teal); color: #fff; border: none; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; align-self: flex-end; transition: background 0.15s; flex-shrink: 0; }
    .btn-send:hover { background: var(--teal-light); }
    .btn-send:disabled { background: var(--border2); color: var(--text3); cursor: not-allowed; }
  </style>
</head>
<body>

<header>
  <span class="logo">Pilar</span>
  <div class="header-divider"></div>
  <span class="header-sub">Predictive Maintenance System</span>
  <div class="nav-links">
    <button class="notif-btn" id="notif-btn" onclick="toggleNotifications()">Notifications Off</button>
    <a class="nav-link" href="/twin">Digital Twin</a>
    <a class="nav-link" href="/history">History</a>
  </div>
</header>

<main>
  <div class="panel">
    <div class="panel-body">
      <span class="field-label">Alert recipient (email)</span>
      <div class="email-section">
        <input class="email-input" type="email" id="email_responsable"
          placeholder="maintenance@company.com" onchange="saveEmail(this.value)">
        <div class="email-saved" id="email-saved">Email address saved</div>
      </div>
      <span class="field-label">Machine class</span>
      <div class="type-selector">
        <div class="type-btn active" data-val="0" onclick="selectType(this)">L &mdash; Low</div>
        <div class="type-btn" data-val="1" onclick="selectType(this)">M &mdash; Med</div>
        <div class="type-btn" data-val="2" onclick="selectType(this)">H &mdash; High</div>
      </div>
      <span class="field-label">Sensor parameters</span>
      <div class="sensor">
        <div class="sensor-header">
          <span class="sensor-name">Air temperature</span>
          <div class="value-input-wrap">
            <input class="value-input" type="number" id="num_temp_air" value="300" min="295" max="305" step="0.1" oninput="syncFromInput('temp_air','num_temp_air')">
            <span class="sensor-unit">K</span>
          </div>
        </div>
        <input type="range" id="temp_air" min="295" max="305" step="0.1" value="300" oninput="syncFromSlider('temp_air','num_temp_air',1)">
        <div class="range-labels"><span>295 K</span><span>305 K</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header">
          <span class="sensor-name">Process temperature</span>
          <div class="value-input-wrap">
            <input class="value-input" type="number" id="num_temp_process" value="310" min="305" max="315" step="0.1" oninput="syncFromInput('temp_process','num_temp_process')">
            <span class="sensor-unit">K</span>
          </div>
        </div>
        <input type="range" id="temp_process" min="305" max="315" step="0.1" value="310" oninput="syncFromSlider('temp_process','num_temp_process',1)">
        <div class="range-labels"><span>305 K</span><span>315 K</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header">
          <span class="sensor-name">Rotational speed</span>
          <div class="value-input-wrap">
            <input class="value-input" type="number" id="num_vitesse" value="1500" min="1000" max="3000" step="10" oninput="syncFromInput('vitesse','num_vitesse')">
            <span class="sensor-unit">rpm</span>
          </div>
        </div>
        <input type="range" id="vitesse" min="1000" max="3000" step="10" value="1500" oninput="syncFromSlider('vitesse','num_vitesse',0)">
        <div class="range-labels"><span>1000</span><span>3000 rpm</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header">
          <span class="sensor-name">Torque</span>
          <div class="value-input-wrap">
            <input class="value-input" type="number" id="num_couple" value="40" min="3" max="80" step="0.1" oninput="syncFromInput('couple','num_couple')">
            <span class="sensor-unit">Nm</span>
          </div>
        </div>
        <input type="range" id="couple" min="3" max="80" step="0.1" value="40" oninput="syncFromSlider('couple','num_couple',1)">
        <div class="range-labels"><span>3 Nm</span><span>80 Nm</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header">
          <span class="sensor-name">Tool wear</span>
          <div class="value-input-wrap">
            <input class="value-input" type="number" id="num_usure" value="100" min="0" max="250" step="1" oninput="syncFromInput('usure','num_usure')">
            <span class="sensor-unit">min</span>
          </div>
        </div>
        <input type="range" id="usure" min="0" max="250" step="1" value="100" oninput="syncFromSlider('usure','num_usure',0)">
        <div class="range-labels"><span>0</span><span>250 min</span></div>
      </div>
      <button class="btn-analyze" id="btn" onclick="analyser()">Run Analysis</button>
      <div class="alert-notif" id="mail-notif">Alert notification dispatched</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-body" id="panel-results">
      <div class="idle-state">
        <span class="line1">No analysis data</span>
        <span class="line2">Configure parameters and run analysis</span>
      </div>
    </div>
  </div>

  <div class="chat-panel">
    <div class="chat-header">
      <div class="chat-title">Pilar Assistant</div>
      <div class="chat-subtitle">Powered by Claude Haiku &middot; Industrial AI</div>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="msg bot">
        <span class="msg-sender">Pilar</span>
        <div class="msg-bubble">Welcome. I am your industrial maintenance assistant. Run an analysis or ask me anything about machine health and failure prevention.</div>
      </div>
    </div>
    <div class="chat-input-area">
      <textarea class="chat-input" id="chat-input" placeholder="Ask a question..." rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"></textarea>
      <button class="btn-send" id="btn-send" onclick="sendChat()">Send</button>
    </div>
  </div>
</main>

<script>
  let machineType = 0, runHistory = [], lastResult = null, lastData = null, chatHistory = [];

  // ── NOTIFICATIONS ──────────────────────────────
  function updateNotifBtn() {
    const btn = document.getElementById('notif-btn');
    const p = Notification.permission;
    if (p === 'granted') {
      btn.textContent = 'Notifications On';
      btn.className = 'notif-btn enabled';
    } else if (p === 'denied') {
      btn.textContent = 'Notifications Blocked';
      btn.className = 'notif-btn denied';
    } else {
      btn.textContent = 'Enable Notifications';
      btn.className = 'notif-btn';
    }
  }

  async function toggleNotifications() {
    if (Notification.permission === 'granted') return;
    if (Notification.permission === 'denied') {
      alert('Notifications are blocked. Please enable them in your browser settings for this site.');
      return;
    }
    const result = await Notification.requestPermission();
    updateNotifBtn();
    if (result === 'granted') {
      new Notification('Pilar — Notifications Enabled', {
        body: 'You will receive alerts when failure risk exceeds 50%.',
        tag: 'pilar-init'
      });
    }
  }

  function sendBrowserNotif(risk, zones) {
    if (Notification.permission !== 'granted') return;
    const severity = risk >= 75 ? 'CRITICAL' : 'HIGH';
    const zonesText = zones.length > 0
      ? 'Zones: ' + zones.map(z => z.nom).join(', ')
      : 'No specific zone identified';
    new Notification(`Pilar Alert — ${severity} | Risk: ${risk}%`, {
      body: zonesText,
      requireInteraction: true,
      tag: 'pilar-alert'
    });
  }

  updateNotifBtn();

  // ── CONTROLS ───────────────────────────────────
  function selectType(el) {
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    machineType = parseInt(el.dataset.val);
  }
  function syncFromSlider(sliderId, numId, dec) {
    document.getElementById(numId).value = parseFloat(document.getElementById(sliderId).value).toFixed(dec);
  }
  function syncFromInput(sliderId, numId) {
    const v = parseFloat(document.getElementById(numId).value);
    if (!isNaN(v)) document.getElementById(sliderId).value = v;
  }
  function getVal(id) { return parseFloat(document.getElementById(id).value); }

  async function saveEmail(email) {
    if (!email) return;
    await fetch('/set_email', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({email}) });
    const s = document.getElementById('email-saved');
    s.style.display = 'block';
    setTimeout(() => s.style.display = 'none', 3000);
  }

  // ── ANALYSIS ───────────────────────────────────
  async function analyser() {
    const btn = document.getElementById('btn');
    btn.disabled = true; btn.textContent = 'Analyzing...';
    lastData = {
      type: machineType,
      temp_air: getVal('num_temp_air'), temp_process: getVal('num_temp_process'),
      vitesse: getVal('num_vitesse'), couple: getVal('num_couple'), usure: getVal('num_usure')
    };
    const res = await fetch('/predire', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(lastData) });
    lastResult = await res.json();
    const now = new Date().toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
    runHistory.unshift({ time: now, risk: lastResult.probabilite, status: lastResult.prediction });
    if (runHistory.length > 6) runHistory.pop();
    renderResults(lastResult);

    if (lastResult.probabilite >= 50) {
      sendBrowserNotif(lastResult.probabilite, lastResult.zones);
      const n = document.getElementById('mail-notif');
      n.style.display = 'block';
      setTimeout(() => n.style.display = 'none', 5000);
    }

    btn.disabled = false; btn.textContent = 'Run Analysis';
  }

  function renderResults(result) {
    const isAlert = result.prediction === 1;
    const cls = isAlert ? 'alert' : 'ok';
    const statusText = isAlert ? 'Anomaly Detected' : 'Normal Operation';
    let zonesHTML = '';
    if (isAlert && result.zones.length > 0) {
      zonesHTML = `<div style="margin-bottom:20px">
        <div class="section-title" style="margin-bottom:10px">Failure zone analysis</div>
        ${result.zones.map(z => `
          <div class="zone-row">
            <span class="zone-name">${z.nom}</span>
            <div class="zone-bar-wrap"><div class="zone-bar-fill" style="width:${z.proba}%"></div></div>
            <span class="zone-proba">${z.proba}%</span>
          </div>`).join('')}
      </div>`;
    }
    let histHTML = '';
    if (runHistory.length > 1) {
      histHTML = `<div class="history-table">
        <div class="section-title" style="margin-bottom:8px">Recent analyses</div>
        <table>
          <thead><tr><th>Time</th><th>Risk</th><th>Status</th></tr></thead>
          <tbody>${runHistory.slice(1).map(h => `
            <tr>
              <td>${h.time}</td><td>${h.risk}%</td>
              <td class="${h.status ? 'alert' : 'ok'}">${h.status ? 'Anomaly' : 'Normal'}</td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
    }
    document.getElementById('panel-results').innerHTML = `
      <div class="status-card ${cls}">
        <div>
          <div class="status-badge ${cls}">
            <span class="status-dot ${cls}"></span>${statusText}
          </div>
          <div style="margin-top:6px;font-size:10px;color:var(--text3)">${new Date().toLocaleString('en-GB')}</div>
        </div>
        <div class="risk-block">
          <div class="risk-number ${cls}">${result.probabilite}<span style="font-size:18px;color:var(--text3)">%</span></div>
          <div class="risk-label">Failure probability</div>
        </div>
      </div>
      ${zonesHTML}${histHTML}`;
  }

  // ── CHAT ───────────────────────────────────────
  let msgId = 0;
  function addMessage(role, text, typing = false) {
    const id = 'msg-' + (++msgId);
    const div = document.createElement('div');
    div.className = 'msg ' + role; div.id = id;
    div.innerHTML = `<span class="msg-sender">${role === 'user' ? 'You' : 'Pilar'}</span>
      <div class="msg-bubble${typing ? ' typing' : ''}">${text}</div>`;
    const c = document.getElementById('chat-messages');
    c.appendChild(div); c.scrollTop = c.scrollHeight;
    return id;
  }
  function removeMessage(id) { const el = document.getElementById(id); if (el) el.remove(); }

  async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim(); if (!msg) return;
    chatHistory.push({role:'user', content: msg});
    addMessage('user', msg); input.value = '';
    const typingId = addMessage('bot', 'Processing...', true);
    document.getElementById('btn-send').disabled = true;
    const context = lastResult ? {
      risk: lastResult.probabilite,
      status: lastResult.prediction === 1 ? 'anomaly' : 'normal',
      zones: lastResult.zones, data: lastData
    } : null;
    const res = await fetch('/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ message: msg, context, history: chatHistory.slice(-10) })
    });
    const data = await res.json();
    chatHistory.push({role:'assistant', content: data.reply});
    removeMessage(typingId); addMessage('bot', data.reply);
    document.getElementById('btn-send').disabled = false;
  }
</script>
</body>
</html>
"""

TWIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pilar — Digital Twin</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #07090f; --surface: #0e1118; --surface2: #141820;
      --border: #1e2433; --border2: #252d3d;
      --teal: #0d9488; --teal-light: #14b8a6; --teal-dim: rgba(13,148,136,0.08);
      --red: #dc2626; --red-dim: rgba(220,38,38,0.08);
      --green: #059669; --green-dim: rgba(5,150,105,0.08);
      --amber: #d97706; --purple: #7c3aed;
      --text: #e2e8f0; --text2: #94a3b8; --text3: #64748b;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    header { padding: 0 32px; height: 52px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; background: var(--surface); }
    .logo { font-size: 13px; font-weight: 700; letter-spacing: 4px; color: var(--teal-light); text-transform: uppercase; }
    .header-divider { width: 1px; height: 20px; background: var(--border2); }
    .header-sub { color: var(--text3); font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }
    .nav-links { margin-left: auto; display: flex; gap: 6px; }
    .nav-link { padding: 6px 14px; background: transparent; border: 1px solid var(--border2); border-radius: 4px; color: var(--text3); font-size: 11px; text-decoration: none; transition: all 0.15s; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    .content { padding: 24px 28px; }
    .top-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; }
    .kpi-value { font-size: 26px; font-weight: 800; font-variant-numeric: tabular-nums; }
    .kpi-label { font-size: 9px; color: var(--text3); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px; }
    .kpi-value.ok { color: var(--green); } .kpi-value.alert { color: var(--red); }
    .kpi-value.amber { color: var(--amber); } .kpi-value.purple { color: var(--purple); }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 18px; }
    .chart-card.full { grid-column: 1 / -1; }
    .chart-title { font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; margin-bottom: 14px; }
    .banner { padding: 14px 18px; border-radius: 6px; border: 1px solid; display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
    .banner.safe { background: var(--green-dim); border-color: var(--green); }
    .banner.warning { background: rgba(217,119,6,0.06); border-color: var(--amber); }
    .banner.danger { background: var(--red-dim); border-color: var(--red); }
    .banner-title { font-size: 13px; font-weight: 700; }
    .banner-sub { font-size: 11px; color: var(--text3); margin-top: 2px; }
    .banner-title.safe { color: var(--green); } .banner-title.warning { color: var(--amber); } .banner-title.danger { color: var(--red); }
    .whatif-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 20px; margin-bottom: 20px; }
    .whatif-title { font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; margin-bottom: 18px; }
    .whatif-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 16px; }
    .whatif-field { display: flex; flex-direction: column; gap: 5px; }
    .whatif-label { font-size: 9px; color: var(--text3); letter-spacing: 1px; text-transform: uppercase; }
    .whatif-input { padding: 7px 9px; background: var(--bg); border: 1px solid var(--border2); border-radius: 4px; color: var(--text); font-size: 12px; font-weight: 600; outline: none; width: 100%; transition: border-color 0.15s; }
    .whatif-input:focus { border-color: var(--teal); }
    .btn-simulate { padding: 9px 20px; background: var(--purple); color: #fff; border: none; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; cursor: pointer; transition: opacity 0.15s; }
    .btn-simulate:hover { opacity: 0.85; }
    .no-data { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px; gap: 10px; color: var(--text3); }
    .no-data .l1 { font-size: 13px; letter-spacing: 1px; }
    .no-data .l2 { font-size: 11px; }
  </style>
</head>
<body>
<header>
  <span class="logo">Pilar</span>
  <div class="header-divider"></div>
  <span class="header-sub">Digital Twin &mdash; Predictive Simulation</span>
  <div class="nav-links">
    <a class="nav-link" href="/">Monitor</a>
    <a class="nav-link" href="/history">History</a>
  </div>
</header>
<div class="content" id="main-content">
  <div class="no-data"><span class="l1">Loading simulation data...</span></div>
</div>
<script>
  async function loadTwin() {
    const res = await fetch('/api/twin');
    const d = await res.json();
    if (!d.has_data) {
      document.getElementById('main-content').innerHTML = `
        <div class="no-data">
          <span class="l1">No analysis data available</span>
          <span class="l2">Run at least one analysis on the Monitor page to initialize the twin</span>
          <a href="/" style="margin-top:14px;padding:9px 18px;background:var(--teal);color:#fff;border-radius:4px;text-decoration:none;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Go to Monitor</a>
        </div>`;
      return;
    }
    const bannerCls = d.failure_hours === null ? 'safe' : d.failure_hours < 6 ? 'danger' : 'warning';
    const bannerTitle = d.failure_hours === null
      ? 'System Healthy &mdash; No failure predicted within 24 hours'
      : d.failure_hours < 6
        ? `Critical &mdash; Failure predicted within ${d.failure_hours} hour(s)`
        : `Warning &mdash; Failure predicted in approximately ${d.failure_hours} hours`;
    const bannerSub = d.failure_hours === null
      ? `Current risk: ${d.current_risk}% &nbsp;&middot;&nbsp; Trend: ${d.trend}`
      : `Current risk: ${d.current_risk}% &nbsp;&middot;&nbsp; Immediate maintenance action recommended`;

    document.getElementById('main-content').innerHTML = `
      <div class="banner ${bannerCls}">
        <div>
          <div class="banner-title ${bannerCls}">${bannerTitle}</div>
          <div class="banner-sub">${bannerSub}</div>
        </div>
      </div>
      <div class="top-grid">
        <div class="kpi-card"><div class="kpi-value ${d.current_risk>=50?'alert':d.current_risk>=22?'amber':'ok'}">${d.current_risk}%</div><div class="kpi-label">Current risk</div></div>
        <div class="kpi-card"><div class="kpi-value amber">${d.avg_risk_24h}%</div><div class="kpi-label">Average risk</div></div>
        <div class="kpi-card"><div class="kpi-value ${d.anomaly_rate>=30?'alert':'ok'}">${d.anomaly_rate}%</div><div class="kpi-label">Anomaly rate</div></div>
        <div class="kpi-card"><div class="kpi-value purple">${d.total_analyses}</div><div class="kpi-label">Total analyses</div></div>
      </div>
      <div class="chart-card full" style="margin-bottom:16px">
        <div class="chart-title">Failure risk &mdash; Historical data and 24-hour simulation</div>
        <div id="chart-risk" style="height:270px"></div>
      </div>
      <div class="chart-grid">
        <div class="chart-card"><div class="chart-title">Tool wear progression</div><div id="chart-wear" style="height:210px"></div></div>
        <div class="chart-card"><div class="chart-title">Process temperature trend</div><div id="chart-temp" style="height:210px"></div></div>
      </div>
      <div class="whatif-card">
        <div class="whatif-title">Scenario Simulator &mdash; Test operating conditions</div>
        <div class="whatif-grid">
          <div class="whatif-field"><label class="whatif-label">Machine class</label>
            <select class="whatif-input" id="wi-type"><option value="0">L &mdash; Low</option><option value="1">M &mdash; Medium</option><option value="2">H &mdash; High</option></select></div>
          <div class="whatif-field"><label class="whatif-label">Air temp (K)</label><input class="whatif-input" type="number" id="wi-temp-air" value="${d.last_params.temp_air}" step="0.1"></div>
          <div class="whatif-field"><label class="whatif-label">Speed (rpm)</label><input class="whatif-input" type="number" id="wi-vitesse" value="${d.last_params.vitesse}" step="10"></div>
          <div class="whatif-field"><label class="whatif-label">Torque (Nm)</label><input class="whatif-input" type="number" id="wi-couple" value="${d.last_params.couple}" step="0.1"></div>
          <div class="whatif-field"><label class="whatif-label">Tool wear (min)</label><input class="whatif-input" type="number" id="wi-usure" value="${d.last_params.usure}" step="1"></div>
        </div>
        <button class="btn-simulate" onclick="runWhatIf()">Run Simulation</button>
        <div id="whatif-result" style="margin-top:14px"></div>
      </div>`;
    plotRisk(d); plotWear(d); plotTemp(d);
  }

  const layout = {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    font:{color:'#64748b',size:10}, margin:{t:8,b:36,l:42,r:16},
    xaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},
    yaxis:{gridcolor:'#1e2433',linecolor:'#1e2433',tickfont:{size:9}},
    legend:{bgcolor:'transparent',font:{size:9}}, hovermode:'x unified'
  };

  function plotRisk(d) {
    Plotly.newPlot('chart-risk', [
      {x:d.history_times,y:d.history_risks,name:'Historical',type:'scatter',mode:'lines+markers',line:{color:'#14b8a6',width:2},marker:{color:d.history_risks.map(r=>r>=50?'#dc2626':r>=22?'#d97706':'#059669'),size:5}},
      {x:d.future_times,y:d.future_risks,name:'Simulated (24h)',type:'scatter',mode:'lines',line:{color:'#7c3aed',width:2,dash:'dot'},fill:'tozeroy',fillcolor:'rgba(124,58,237,0.04)'},
      {x:[...d.history_times,...d.future_times],y:Array(d.history_times.length+d.future_times.length).fill(50),name:'Alert threshold',type:'scatter',mode:'lines',line:{color:'#dc2626',width:1,dash:'dash'}}
    ], {...layout,yaxis:{...layout.yaxis,range:[0,105]}});
  }
  function plotWear(d) {
    Plotly.newPlot('chart-wear',[
      {x:d.history_times,y:d.history_wear,name:'Actual',type:'scatter',mode:'lines+markers',line:{color:'#d97706',width:2},marker:{size:4}},
      {x:d.future_times,y:d.future_wear,name:'Projected',type:'scatter',mode:'lines',line:{color:'#d97706',width:2,dash:'dot'},fill:'tozeroy',fillcolor:'rgba(217,119,6,0.04)'}
    ], layout);
  }
  function plotTemp(d) {
    Plotly.newPlot('chart-temp',[
      {x:d.history_times,y:d.history_temp,name:'Actual',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2}},
      {x:d.future_times,y:d.future_temp,name:'Projected',type:'scatter',mode:'lines',line:{color:'#dc2626',width:2,dash:'dot'}}
    ], layout);
  }

  async function runWhatIf() {
    const params = {
      type: parseInt(document.getElementById('wi-type').value),
      temp_air: parseFloat(document.getElementById('wi-temp-air').value),
      temp_process: parseFloat(document.getElementById('wi-temp-air').value)+10,
      vitesse: parseFloat(document.getElementById('wi-vitesse').value),
      couple: parseFloat(document.getElementById('wi-couple').value),
      usure: parseFloat(document.getElementById('wi-usure').value)
    };
    const res = await fetch('/api/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    const d = await res.json();
    const colors = {ok:'#059669',amber:'#d97706',alert:'#dc2626'};
    const cls = d.risk>=50?'alert':d.risk>=22?'amber':'ok';
    document.getElementById('whatif-result').innerHTML = `
      <div style="padding:14px 18px;background:var(--bg);border:1px solid ${colors[cls]};border-radius:4px;display:flex;align-items:center;gap:18px;">
        <div>
          <div style="font-size:9px;letter-spacing:1.5px;color:var(--text3);text-transform:uppercase;margin-bottom:3px;">Simulated risk</div>
          <div style="font-size:32px;font-weight:800;color:${colors[cls]}">${d.risk}%</div>
        </div>
        <div style="flex:1">
          <div style="font-size:12px;font-weight:600;color:${colors[cls]};margin-bottom:3px;">${d.status}</div>
          <div style="font-size:11px;color:var(--text3)">${d.message}</div>
          ${d.zones.length>0?`<div style="margin-top:7px;font-size:10px;color:var(--amber)">Zones at risk: ${d.zones.map(z=>z.nom+' — '+z.proba+'%').join(' &nbsp;|&nbsp; ')}</div>`:''}
        </div>
      </div>`;
  }
  loadTwin();
</script>
</body>
</html>
"""

HISTORY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pilar — History</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #07090f; --surface: #0e1118; --surface2: #141820;
      --border: #1e2433; --border2: #252d3d;
      --teal: #0d9488; --teal-light: #14b8a6;
      --red: #dc2626; --green: #059669; --amber: #d97706;
      --text: #e2e8f0; --text2: #94a3b8; --text3: #64748b;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    header { padding: 0 32px; height: 52px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; background: var(--surface); }
    .logo { font-size: 13px; font-weight: 700; letter-spacing: 4px; color: var(--teal-light); text-transform: uppercase; }
    .header-divider { width: 1px; height: 20px; background: var(--border2); }
    .header-sub { color: var(--text3); font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }
    .nav-links { margin-left: auto; display: flex; gap: 6px; }
    .nav-link { padding: 6px 14px; background: transparent; border: 1px solid var(--border2); border-radius: 4px; color: var(--text3); font-size: 11px; text-decoration: none; transition: all 0.15s; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    .content { padding: 24px 28px; }
    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; }
    .stat-value { font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }
    .stat-value.alert { color: var(--red); } .stat-value.ok { color: var(--green); } .stat-value.amber { color: var(--amber); }
    .stat-label { font-size: 9px; color: var(--text3); margin-top: 4px; letter-spacing: 1.5px; text-transform: uppercase; }
    .section-title { font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; margin-bottom: 12px; }
    table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 6px; overflow: hidden; font-size: 11px; }
    th { text-align: left; padding: 10px 14px; color: var(--text3); font-weight: 500; letter-spacing: 1px; border-bottom: 1px solid var(--border); font-size: 9px; text-transform: uppercase; }
    td { padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text2); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--surface2); }
    .badge { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }
    .badge.ok { background: rgba(5,150,105,0.12); color: var(--green); }
    .badge.alert { background: rgba(220,38,38,0.12); color: var(--red); }
    .mail-badge { padding: 2px 8px; border-radius: 3px; font-size: 10px; background: rgba(13,148,136,0.12); color: var(--teal-light); }
  </style>
</head>
<body>
<header>
  <span class="logo">Pilar</span>
  <div class="header-divider"></div>
  <span class="header-sub">Analysis History</span>
  <div class="nav-links">
    <a class="nav-link" href="/twin">Digital Twin</a>
    <a class="nav-link" href="/">Monitor</a>
  </div>
</header>
<div class="content">
  <div class="stats">
    <div class="stat-card"><div class="stat-value">{{ total }}</div><div class="stat-label">Total analyses</div></div>
    <div class="stat-card"><div class="stat-value alert">{{ anomalies }}</div><div class="stat-label">Anomalies detected</div></div>
    <div class="stat-card"><div class="stat-value amber">{{ avg_risk }}%</div><div class="stat-label">Average risk score</div></div>
    <div class="stat-card"><div class="stat-value ok">{{ mails }}</div><div class="stat-label">Alerts dispatched</div></div>
  </div>
  <div class="section-title">Analysis log</div>
  <table>
    <thead>
      <tr><th>Timestamp</th><th>Class</th><th>Air Temp</th><th>Speed</th><th>Torque</th><th>Tool Wear</th><th>Risk</th><th>Status</th><th>Failure Zones</th><th>Alert</th></tr>
    </thead>
    <tbody>
      {% for a in analyses %}
      <tr>
        <td>{{ a.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</td>
        <td>{{ a.machine_type }}</td><td>{{ a.temp_air }} K</td>
        <td>{{ a.vitesse }} rpm</td><td>{{ a.couple }} Nm</td><td>{{ a.usure }} min</td>
        <td>{{ a.risk }}%</td>
        <td><span class="badge {{ 'alert' if a.prediction else 'ok' }}">{{ 'Anomaly' if a.prediction else 'Normal' }}</span></td>
        <td>{{ a.zones or '&mdash;' }}</td>
        <td>{% if a.mail_sent %}<span class="mail-badge">Sent</span>{% else %}&mdash;{% endif %}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>
"""

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
    severity_color = "#dc2626"
    zones_rows = "".join(f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#94a3b8;font-size:12px;">{z['nom']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;">
            <span style="color:#dc2626;font-weight:700;font-size:12px;">{z['proba']}%</span>
          </td>
        </tr>""" for z in zones_risque) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;font-size:12px;">No specific zone identified</td></tr>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#07090f;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;padding:40px 0;">
    <tr><td align="center">
      <table width="580" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;overflow:hidden;">
        <tr><td style="padding:24px 32px;border-bottom:1px solid #1e2433;background:#0a0d16;">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td><div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;">PILAR</div>
              <div style="font-size:10px;color:#64748b;margin-top:3px;letter-spacing:1px;">Predictive Maintenance System</div></td>
            <td align="right"><span style="padding:4px 12px;background:rgba(220,38,38,0.12);border:1px solid #dc2626;border-radius:3px;color:#dc2626;font-size:10px;font-weight:700;letter-spacing:2px;">FAILURE ALERT</span></td>
          </tr></table>
        </td></tr>
        <tr><td style="padding:32px 32px 24px;">
          <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:8px;">Failure Probability</div>
          <div style="font-size:56px;font-weight:800;color:{severity_color};line-height:1;">{probabilite}<span style="font-size:28px;color:#64748b;">%</span></div>
          <div style="margin-top:8px;"><span style="padding:3px 10px;background:rgba(220,38,38,0.1);border:1px solid {severity_color};border-radius:3px;font-size:10px;font-weight:700;letter-spacing:2px;color:{severity_color};">SEVERITY: {severity}</span></div>
        </td></tr>
        <tr><td style="padding:0 32px 24px;">
          <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:12px;">Machine Parameters</div>
          <table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">
            <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Machine class</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{mtype}</td></tr>
            <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Air temperature</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get('temp_air')} K</td></tr>
            <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Rotational speed</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get('vitesse')} rpm</td></tr>
            <tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Torque</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get('couple')} Nm</td></tr>
            <tr><td style="padding:8px 12px;color:#64748b;font-size:11px;">Tool wear</td><td style="padding:8px 12px;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get('usure')} min</td></tr>
          </table>
        </td></tr>
        <tr><td style="padding:0 32px 24px;">
          <div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:12px;">Identified Failure Zones</div>
          <table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">{zones_rows}</table>
        </td></tr>
        <tr><td style="padding:20px 32px;border-top:1px solid #1e2433;background:#0a0d16;">
          <div style="font-size:10px;color:#64748b;">This alert was automatically generated by the Pilar Predictive Maintenance System.<br>
          Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Pilar Alert — Failure Risk {probabilite}% | {severity}"
    msg['From'] = f"Pilar Maintenance System <{GMAIL}>"
    msg['To'] = email_to
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email_to, msg.as_string())
        print(f"Alert dispatched to {email_to}")
    except Exception as e:
        print(f"Email error: {e}")

@app.route('/')
def index():
    return render_template_string(HTML)

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
    wear_rate, temp_drift = 1.5, 0.05
    future_times, future_risks, future_wear, future_temp = [], [], [], []
    now = datetime.utcnow()
    cur_usure, cur_temp_p = last.usure, last.temp_process
    failure_hours = None
    for h in range(1, 25):
        cur_usure = min(cur_usure + wear_rate, 250)
        cur_temp_p = min(cur_temp_p + temp_drift, 315)
        params = {'type': 1, 'temp_air': last.temp_air, 'temp_process': cur_temp_p,
                  'vitesse': last.vitesse, 'couple': last.couple, 'usure': cur_usure}
        risk, pred, _ = predict_risk(params)
        future_times.append((now + timedelta(hours=h)).strftime('%H:%M'))
        future_risks.append(risk)
        future_wear.append(round(cur_usure, 1))
        future_temp.append(round(cur_temp_p, 2))
        if failure_hours is None and risk >= 50:
            failure_hours = h
    total = len(analyses)
    anomaly_rate = round(sum(1 for a in analyses if a.prediction) / total * 100, 1)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1)
    if len(history_risks) >= 3:
        diff = history_risks[-1] - history_risks[-3]
        trend = 'Increasing' if diff > 2 else 'Stable' if abs(diff) <= 2 else 'Decreasing'
    else:
        trend = 'Stable'
    return jsonify({
        'has_data': True, 'current_risk': last.risk, 'avg_risk_24h': avg_risk,
        'anomaly_rate': anomaly_rate, 'total_analyses': total,
        'failure_hours': failure_hours, 'trend': trend,
        'history_times': history_times, 'history_risks': history_risks,
        'history_wear': history_wear, 'history_temp': history_temp,
        'future_times': future_times, 'future_risks': future_risks,
        'future_wear': future_wear, 'future_temp': future_temp,
        'last_params': {'temp_air': last.temp_air, 'vitesse': last.vitesse,
                        'couple': last.couple, 'usure': last.usure}
    })

@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    params = request.json
    params['temp_process'] = params['temp_air'] + 10
    risk, pred, zones_risque = predict_risk(params)
    if pred == 0:
        status, message = 'Normal Operation', 'No failure predicted under these operating conditions.'
    elif risk < 50:
        status, message = 'Low Risk', 'Minor anomaly detected. Continue monitoring.'
    else:
        status, message = 'High Failure Risk', 'These conditions are likely to cause a failure. Reduce tool wear or torque immediately.'
    return jsonify({'risk': risk, 'status': status, 'message': message, 'zones': zones_risque})

@app.route('/set_email', methods=['POST'])
def set_email():
    data = request.json
    email = data.get('email', '')
    set_setting('responsible_email', email)
    print(f"Responsible email updated: {email}")
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
    analysis = Analysis(
        machine_type=machine_types.get(data['type'], 'Unknown'),
        temp_air=data['temp_air'], temp_process=data['temp_process'],
        vitesse=data['vitesse'], couple=data['couple'], usure=data['usure'],
        risk=probabilite, prediction=prediction, zones=zones_str, mail_sent=mail_envoye
    )
    db.session.add(analysis)
    db.session.commit()
    return jsonify({'prediction': prediction, 'probabilite': probabilite,
                    'zones': zones_risque, 'mail_envoye': mail_envoye})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    context = data.get('context')
    chat_history = data.get('history', [])
    if context:
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        mtype = machine_types.get(context['data'].get('type', 0), 'Unknown')
        zones_str = ', '.join([f"{z['nom']} ({z['proba']}%)" for z in context['zones']]) or 'none detected'
        d = context['data']
        system_prompt = f"""You are Pilar, an industrial AI assistant specialized in predictive maintenance.
Current machine state (reference only if relevant):
- Status: {context['status']} | Risk: {context['risk']}% | Class: {mtype}
- Air temp: {d.get('temp_air')} K | Speed: {d.get('vitesse')} rpm | Torque: {d.get('couple')} Nm | Tool wear: {d.get('usure')} min
- Failure zones: {zones_str}
Respond professionally and concisely. Only reference machine data when directly relevant. Maximum 3 sentences for simple queries."""
    else:
        system_prompt = "You are Pilar, an industrial AI assistant specialized in predictive maintenance. Respond professionally and concisely."
    messages = [{"role": h['role'], "content": h['content']} for h in chat_history[:-1]]
    messages.append({"role": "user", "content": message})
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=400,
            system=system_prompt, messages=messages
        )
        reply = response.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        reply = f"Current failure risk is {context['risk']}%." if context else "I am available to assist with maintenance and reliability questions."
    return jsonify({'reply': reply})

if __name__ == '__main__':
    print("Pilar system started — http://localhost:5000")
    app.run(debug=True)