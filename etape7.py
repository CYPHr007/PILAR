import logging
import os
import pickle
import smtplib
import threading
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pilar.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

COLONNES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'ecart_temp']

GMAIL = os.getenv("PILAR_GMAIL", "")
GMAIL_PWD = os.getenv("PILAR_GMAIL_PWD", "")
responsable_email: dict[str, str] = {"email": ""}
_email_lock = threading.Lock()

# ──────────────────────────────────────────────
# MAIN HTML
# ──────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pilar — Machine Monitor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0b0f1a; --surface: #111827; --border: #1f2937;
      --teal: #0d9488; --teal-light: #14b8a6;
      --red: #ef4444; --green: #10b981; --amber: #f59e0b;
      --text: #f1f5f9; --muted: #64748b; --label: #94a3b8;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    header { padding: 16px 32px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
    .logo { font-size: 18px; font-weight: 700; letter-spacing: 3px; color: var(--teal-light); }
    .sep { color: var(--border); }
    .page-title { color: var(--muted); font-size: 13px; letter-spacing: 1px; }
    .nav-links { margin-left: auto; display: flex; gap: 8px; }
    .nav-link { padding: 7px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--label); font-size: 12px; text-decoration: none; transition: all 0.15s; display: inline-flex; align-items: center; gap: 5px; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    main { display: grid; grid-template-columns: 320px 1fr 300px; flex: 1; overflow: hidden; }
    .panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
    .panel:last-child { border-right: none; }
    .panel-body { padding: 24px 20px; overflow-y: auto; flex: 1; }
    .panel-body::-webkit-scrollbar { width: 4px; }
    .panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    .section-label { font-size: 10px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 14px; }
    .type-selector { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; margin-bottom: 24px; }
    .type-btn { padding: 9px 6px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--muted); font-size: 12px; cursor: pointer; text-align: center; transition: all 0.15s; }
    .type-btn:hover { border-color: var(--teal); color: var(--text); }
    .type-btn.active { border-color: var(--teal); background: rgba(13,148,136,0.12); color: var(--teal-light); font-weight: 600; }
    .sensor { margin-bottom: 20px; }
    .sensor-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .sensor-name { font-size: 12px; color: var(--label); }
    .value-input-wrap { display: flex; align-items: center; gap: 4px; }
    .value-input { width: 72px; padding: 4px 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 5px; color: var(--text); font-size: 15px; font-weight: 600; text-align: right; outline: none; transition: border-color 0.15s; }
    .value-input:focus { border-color: var(--teal); }
    .sensor-unit { font-size: 11px; color: var(--muted); }
    input[type=range] { -webkit-appearance: none; width: 100%; height: 3px; background: var(--border); border-radius: 2px; outline: none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 13px; height: 13px; border-radius: 50%; background: var(--teal); cursor: pointer; }
    .range-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 4px; }
    .email-section { margin-bottom: 20px; }
    .email-input { width: 100%; padding: 9px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 7px; color: var(--text); font-size: 12px; outline: none; transition: border-color 0.15s; }
    .email-input:focus { border-color: var(--teal); }
    .email-input::placeholder { color: var(--muted); }
    .email-saved { font-size: 10px; color: var(--green); margin-top: 5px; display: none; }
    .btn-analyze { width: 100%; padding: 13px; background: var(--teal); color: #fff; border: none; border-radius: 7px; font-size: 13px; font-weight: 600; letter-spacing: 1px; cursor: pointer; transition: background 0.15s; margin-top: 8px; }
    .btn-analyze:hover { background: var(--teal-light); }
    .btn-analyze:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }
    .status-card { padding: 22px 24px; border-radius: 10px; border: 1px solid var(--border); background: var(--surface); display: flex; align-items: center; gap: 20px; margin-bottom: 24px; transition: border-color 0.3s; }
    .status-card.ok { border-color: var(--green); }
    .status-card.alert { border-color: var(--red); }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
    .status-dot.ok { background: var(--green); box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .status-dot.alert { background: var(--red); box-shadow: 0 0 8px rgba(239,68,68,0.5); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;}50%{opacity:0.3;} }
    .status-label { font-size: 10px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 3px; }
    .status-text { font-size: 20px; font-weight: 700; }
    .status-text.ok { color: var(--green); }
    .status-text.alert { color: var(--red); }
    .risk-block { margin-left: auto; text-align: right; }
    .risk-number { font-size: 42px; font-weight: 800; line-height: 1; }
    .risk-number.ok { color: var(--green); }
    .risk-number.alert { color: var(--red); }
    .risk-suffix { font-size: 18px; color: var(--muted); }
    .risk-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; margin-top: 3px; }
    .zone-row { display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: var(--surface); border-radius: 7px; border: 1px solid var(--border); margin-bottom: 8px; }
    .zone-name { font-size: 12px; color: var(--label); min-width: 160px; }
    .zone-bar-wrap { flex: 1; height: 3px; background: var(--border); border-radius: 2px; }
    .zone-bar-fill { height: 100%; border-radius: 2px; background: var(--red); transition: width 0.4s; }
    .zone-proba { font-size: 13px; font-weight: 600; color: var(--amber); min-width: 38px; text-align: right; }
    .idle-msg { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 8px; color: var(--muted); }
    .idle-msg .big { font-size: 13px; letter-spacing: 1px; }
    .idle-msg .small { font-size: 11px; }
    .history-wrap { margin-top: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th { text-align: left; padding: 7px 10px; color: var(--muted); font-weight: 500; letter-spacing: 1px; border-bottom: 1px solid var(--border); }
    td { padding: 9px 10px; border-bottom: 1px solid var(--border); color: var(--label); }
    td.ok { color: var(--green); }
    td.alert { color: var(--red); }
    tr:last-child td { border-bottom: none; }
    .mail-notif { margin-top: 10px; padding: 8px 12px; background: rgba(13,148,136,0.1); border: 1px solid var(--teal); border-radius: 6px; font-size: 11px; color: var(--teal-light); display: none; }
    .chat-panel { display: flex; flex-direction: column; overflow: hidden; }
    .chat-header { padding: 16px 20px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .chat-title { font-size: 12px; font-weight: 600; color: var(--label); letter-spacing: 1px; }
    .chat-subtitle { font-size: 10px; color: var(--muted); margin-top: 2px; }
    .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .chat-messages::-webkit-scrollbar { width: 3px; }
    .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    .msg { display: flex; flex-direction: column; gap: 3px; max-width: 90%; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.bot { align-self: flex-start; align-items: flex-start; }
    .msg-sender { font-size: 9px; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; }
    .msg-bubble { padding: 9px 13px; border-radius: 10px; font-size: 12px; line-height: 1.5; color: var(--text); }
    .msg.user .msg-bubble { background: var(--teal); border-radius: 10px 10px 2px 10px; }
    .msg.bot .msg-bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 10px 10px 10px 2px; color: var(--label); }
    .msg-bubble.typing { color: var(--muted); font-style: italic; }
    .chat-input-area { padding: 12px 16px; border-top: 1px solid var(--border); display: flex; gap: 8px; flex-shrink: 0; }
    .chat-input { flex: 1; padding: 9px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 7px; color: var(--text); font-size: 12px; outline: none; resize: none; font-family: inherit; transition: border-color 0.15s; max-height: 80px; }
    .chat-input:focus { border-color: var(--teal); }
    .chat-input::placeholder { color: var(--muted); }
    .btn-send { padding: 9px 14px; background: var(--teal); color: #fff; border: none; border-radius: 7px; font-size: 12px; font-weight: 600; cursor: pointer; align-self: flex-end; transition: background 0.15s; flex-shrink: 0; }
    .btn-send:hover { background: var(--teal-light); }
    .btn-send:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }
  </style>
</head>
<body>
<header>
  <span class="logo">PILAR</span>
  <span class="sep">/</span>
  <span class="page-title">Machine Monitor — Anomaly Detection</span>
  <div class="nav-links">
    <a class="nav-link" href="/twin"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" y1="22" x2="12" y2="12"/></svg> Digital Twin</a>
    <a class="nav-link" href="/history"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg> History</a>
  </div>
</header>
<main>
  <div class="panel">
    <div class="panel-body">
      <div class="section-label">Responsible email</div>
      <div class="email-section">
        <input class="email-input" type="email" id="email_responsable"
          placeholder="responsible@company.com" onchange="saveEmail(this.value)">
        <div class="email-saved" id="email-saved">✅ Email saved</div>
      </div>
      <div class="section-label">Machine type</div>
      <div class="type-selector">
        <div class="type-btn active" data-val="0" onclick="selectType(this)">L — Low</div>
        <div class="type-btn" data-val="1" onclick="selectType(this)">M — Medium</div>
        <div class="type-btn" data-val="2" onclick="selectType(this)">H — High</div>
      </div>
      <div class="section-label">Sensor readings</div>
      <div class="sensor">
        <div class="sensor-header"><span class="sensor-name">Air temperature</span>
          <div class="value-input-wrap"><input class="value-input" type="number" id="num_temp_air" value="300" min="295" max="305" step="0.1" oninput="syncFromInput('temp_air','num_temp_air')"><span class="sensor-unit">K</span></div>
        </div>
        <input type="range" id="temp_air" min="295" max="305" step="0.1" value="300" oninput="syncFromSlider('temp_air','num_temp_air',1)">
        <div class="range-labels"><span>295</span><span>305 K</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header"><span class="sensor-name">Process temperature</span>
          <div class="value-input-wrap"><input class="value-input" type="number" id="num_temp_process" value="310" min="305" max="315" step="0.1" oninput="syncFromInput('temp_process','num_temp_process')"><span class="sensor-unit">K</span></div>
        </div>
        <input type="range" id="temp_process" min="305" max="315" step="0.1" value="310" oninput="syncFromSlider('temp_process','num_temp_process',1)">
        <div class="range-labels"><span>305</span><span>315 K</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header"><span class="sensor-name">Rotational speed</span>
          <div class="value-input-wrap"><input class="value-input" type="number" id="num_vitesse" value="1500" min="1000" max="3000" step="10" oninput="syncFromInput('vitesse','num_vitesse')"><span class="sensor-unit">rpm</span></div>
        </div>
        <input type="range" id="vitesse" min="1000" max="3000" step="10" value="1500" oninput="syncFromSlider('vitesse','num_vitesse',0)">
        <div class="range-labels"><span>1000</span><span>3000 rpm</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header"><span class="sensor-name">Torque</span>
          <div class="value-input-wrap"><input class="value-input" type="number" id="num_couple" value="40" min="3" max="80" step="0.1" oninput="syncFromInput('couple','num_couple')"><span class="sensor-unit">Nm</span></div>
        </div>
        <input type="range" id="couple" min="3" max="80" step="0.1" value="40" oninput="syncFromSlider('couple','num_couple',1)">
        <div class="range-labels"><span>3</span><span>80 Nm</span></div>
      </div>
      <div class="sensor">
        <div class="sensor-header"><span class="sensor-name">Tool wear</span>
          <div class="value-input-wrap"><input class="value-input" type="number" id="num_usure" value="100" min="0" max="250" step="1" oninput="syncFromInput('usure','num_usure')"><span class="sensor-unit">min</span></div>
        </div>
        <input type="range" id="usure" min="0" max="250" step="1" value="100" oninput="syncFromSlider('usure','num_usure',0)">
        <div class="range-labels"><span>0</span><span>250 min</span></div>
      </div>
      <button class="btn-analyze" id="btn" onclick="analyser()">Run Analysis</button>
      <div class="mail-notif" id="mail-notif">Alert dispatched to responsible</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-body" id="panel-results">
      <div class="idle-msg">
        <svg width="200" height="150" viewBox="0 0 200 150" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:8px">
          <line x1="100" y1="52" x2="100" y2="29" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>
          <line x1="122" y1="65" x2="148" y2="40" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>
          <line x1="122" y1="85" x2="148" y2="110" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>
          <line x1="78" y1="85" x2="52" y2="110" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>
          <line x1="78" y1="65" x2="52" y2="40" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>
          <rect x="74" y="52" width="52" height="46" rx="4" stroke="#2d3748" stroke-width="1.5" fill="#111827"/>
          <rect x="81" y="60" width="38" height="4" rx="1" fill="#1e293b"/>
          <rect x="81" y="68" width="24" height="4" rx="1" fill="#1e293b"/>
          <rect x="81" y="76" width="30" height="4" rx="1" fill="#1e293b"/>
          <rect x="81" y="84" width="16" height="4" rx="1" fill="#1e293b"/>
          <rect x="78" y="11" width="44" height="18" rx="3" stroke="#2d3748" stroke-width="1.5" fill="#0b0f1a"/>
          <text x="100" y="23" text-anchor="middle" fill="#374151" font-size="7" font-family="monospace">AIR TEMP</text>
          <rect x="148" y="29" width="44" height="18" rx="3" stroke="#2d3748" stroke-width="1.5" fill="#0b0f1a"/>
          <text x="170" y="41" text-anchor="middle" fill="#374151" font-size="7" font-family="monospace">SPEED</text>
          <rect x="148" y="103" width="44" height="18" rx="3" stroke="#2d3748" stroke-width="1.5" fill="#0b0f1a"/>
          <text x="170" y="115" text-anchor="middle" fill="#374151" font-size="7" font-family="monospace">TORQUE</text>
          <rect x="8" y="103" width="44" height="18" rx="3" stroke="#2d3748" stroke-width="1.5" fill="#0b0f1a"/>
          <text x="30" y="115" text-anchor="middle" fill="#374151" font-size="7" font-family="monospace">WEAR</text>
          <rect x="8" y="29" width="44" height="18" rx="3" stroke="#2d3748" stroke-width="1.5" fill="#0b0f1a"/>
          <text x="30" y="41" text-anchor="middle" fill="#374151" font-size="7" font-family="monospace">PROC TEMP</text>
        </svg>
        <span class="big">Awaiting input</span>
        <span class="small">Configure sensor parameters and run analysis</span>
      </div>
    </div>
  </div>

  <div class="chat-panel">
    <div class="chat-header">
      <div class="chat-title">Diagnostic Console</div>
      <div class="chat-subtitle">Maintenance intelligence</div>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="msg bot">
        <span class="msg-sender">Pilar</span>
        <div class="msg-bubble">System online. Run an analysis or ask about machine parameters, failure modes, and maintenance procedures.</div>
      </div>
    </div>
    <div class="chat-input-area">
      <textarea class="chat-input" id="chat-input" placeholder="Ask me anything..." rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"></textarea>
      <button class="btn-send" id="btn-send" onclick="sendChat()">Send</button>
    </div>
  </div>
</main>

<script>
  let machineType = 0, history = [], lastResult = null, lastData = null, chatHistory = [];

  function selectType(el) {
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    machineType = parseInt(el.dataset.val);
  }
  function syncFromSlider(sliderId, numId, decimals) {
    document.getElementById(numId).value = parseFloat(document.getElementById(sliderId).value).toFixed(decimals);
  }
  function syncFromInput(sliderId, numId) {
    const val = parseFloat(document.getElementById(numId).value);
    if (!isNaN(val)) document.getElementById(sliderId).value = val;
  }
  function getVal(id) { return parseFloat(document.getElementById(id).value); }

  async function saveEmail(email) {
    if (!email) return;
    await fetch('/set_email', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({email}) });
    const s = document.getElementById('email-saved');
    s.style.display = 'block';
    setTimeout(() => s.style.display = 'none', 3000);
  }

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
    history.unshift({ time: now, risk: lastResult.probabilite, status: lastResult.prediction });
    if (history.length > 5) history.pop();
    renderResults(lastResult);
    if (lastResult.mail_envoye) {
      const n = document.getElementById('mail-notif');
      n.style.display = 'block';
      setTimeout(() => n.style.display = 'none', 5000);
    }
    btn.disabled = false; btn.textContent = 'Run Analysis';
  }

  function renderResults(result) {
    const isAlert = result.prediction === 1, cls = isAlert ? 'alert' : 'ok';
    let zonesHTML = '';
    if (isAlert && result.zones.length > 0) {
      zonesHTML = `<div style="margin-bottom:24px"><div class="section-label" style="margin-bottom:12px">Failure zones</div>
        ${result.zones.map(z => `<div class="zone-row">
          <span class="zone-name">${z.nom}</span>
          <div class="zone-bar-wrap"><div class="zone-bar-fill" style="width:${z.proba}%"></div></div>
          <span class="zone-proba">${z.proba}%</span></div>`).join('')}</div>`;
    }
    let histHTML = '';
    if (history.length > 1) {
      histHTML = `<div class="history-wrap"><div class="section-label" style="margin-bottom:10px">Recent runs</div>
        <table><thead><tr><th>Time</th><th>Risk</th><th>Status</th></tr></thead><tbody>
        ${history.slice(1).map(h => `<tr><td>${h.time}</td><td>${h.risk}%</td>
          <td class="${h.status?'alert':'ok'}">${h.status?'Anomaly':'Normal'}</td></tr>`).join('')}
        </tbody></table></div>`;
    }
    document.getElementById('panel-results').innerHTML = `
      <div class="status-card ${cls}">
        <div class="status-dot ${cls}"></div>
        <div><div class="status-label">Machine status</div>
          <div class="status-text ${cls}">${isAlert ? 'Anomaly detected' : 'Normal operation'}</div></div>
        <div class="risk-block">
          <div class="risk-number ${cls}">${result.probabilite}<span class="risk-suffix">%</span></div>
          <div class="risk-label">Failure risk</div>
        </div>
      </div>${zonesHTML}${histHTML}`;
  }

  let msgId = 0;
  function addMessage(role, text, typing = false) {
    const id = 'msg-' + (++msgId), div = document.createElement('div');
    div.className = 'msg ' + role; div.id = id;
    div.innerHTML = `<span class="msg-sender">${role==='user'?'You':'Pilar'}</span>
      <div class="msg-bubble${typing?' typing':''}">${text}</div>`;
    const c = document.getElementById('chat-messages');
    c.appendChild(div); c.scrollTop = c.scrollHeight;
    return id;
  }
  function removeMessage(id) { const el = document.getElementById(id); if(el) el.remove(); }

  async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim(); if (!msg) return;
    chatHistory.push({role:'user', content: msg});
    addMessage('user', msg); input.value = '';
    const typingId = addMessage('bot', 'Thinking...', true);
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

# ──────────────────────────────────────────────
# DIGITAL TWIN HTML
# ──────────────────────────────────────────────
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
      --bg: #0b0f1a; --surface: #111827; --border: #1f2937;
      --teal: #0d9488; --teal-light: #14b8a6;
      --red: #ef4444; --green: #10b981; --amber: #f59e0b; --purple: #8b5cf6;
      --text: #f1f5f9; --muted: #64748b; --label: #94a3b8;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    header { padding: 16px 32px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
    .logo { font-size: 18px; font-weight: 700; letter-spacing: 3px; color: var(--teal-light); }
    .sep { color: var(--border); }
    .page-title { color: var(--muted); font-size: 13px; letter-spacing: 1px; }
    .nav-links { margin-left: auto; display: flex; gap: 8px; }
    .nav-link { padding: 7px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--label); font-size: 12px; text-decoration: none; transition: all 0.15s; display: inline-flex; align-items: center; gap: 5px; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    .content { padding: 28px 32px; }
    .top-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
    .kpi-value { font-size: 28px; font-weight: 800; }
    .kpi-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }
    .kpi-value.ok { color: var(--green); }
    .kpi-value.alert { color: var(--red); }
    .kpi-value.amber { color: var(--amber); }
    .kpi-value.purple { color: var(--purple); }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
    .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
    .chart-card.full { grid-column: 1 / -1; }
    .chart-title { font-size: 11px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 16px; }
    .whatif-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 24px; margin-bottom: 24px; }
    .whatif-title { font-size: 11px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 20px; }
    .whatif-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 20px; }
    .whatif-field { display: flex; flex-direction: column; gap: 6px; }
    .whatif-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; }
    .whatif-input { padding: 8px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 13px; font-weight: 600; outline: none; width: 100%; transition: border-color 0.15s; }
    .whatif-input:focus { border-color: var(--teal); }
    .btn-simulate { padding: 11px 24px; background: var(--purple); color: #fff; border: none; border-radius: 7px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
    .btn-simulate:hover { opacity: 0.85; }
    .prediction-banner { padding: 16px 20px; border-radius: 10px; border: 1px solid; display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
    .prediction-banner.safe { background: rgba(16,185,129,0.05); border-color: var(--green); }
    .prediction-banner.warning { background: rgba(245,158,11,0.05); border-color: var(--amber); }
    .prediction-banner.danger { background: rgba(239,68,68,0.05); border-color: var(--red); }
    .banner-icon { display: flex; align-items: center; flex-shrink: 0; }
    .banner-title { font-size: 15px; font-weight: 700; }
    .banner-sub { font-size: 12px; color: var(--muted); margin-top: 3px; }
    .banner-title.safe { color: var(--green); }
    .banner-title.warning { color: var(--amber); }
    .banner-title.danger { color: var(--red); }
    .no-data { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px; gap: 12px; color: var(--muted); }
    .no-data .big { font-size: 14px; letter-spacing: 1px; }
    .no-data .small { font-size: 12px; }
  </style>
</head>
<body>
<header>
  <span class="logo">PILAR</span>
  <span class="sep">/</span>
  <span class="page-title">Digital Twin — Predictive Simulation</span>
  <div class="nav-links">
    <a class="nav-link" href="/"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg> Monitor</a>
    <a class="nav-link" href="/history"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg> History</a>
  </div>
</header>

<div class="content" id="main-content">
  <div class="no-data">
    <span class="big">Loading twin...</span>
    <span class="small">Fetching machine data</span>
  </div>
</div>

<script>
  async function loadTwin() {
    const res = await fetch('/api/twin');
    const d = await res.json();

    if (!d.has_data) {
      document.getElementById('main-content').innerHTML = `
        <div class="no-data">
          <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="8" y="20" width="28" height="28" stroke="#2d3748" stroke-width="1.5"/>
            <rect x="20" y="8" width="28" height="28" stroke="#1e293b" stroke-width="1.5"/>
            <line x1="8" y1="20" x2="20" y2="8" stroke="#2d3748" stroke-width="1.5"/>
            <line x1="36" y1="20" x2="48" y2="8" stroke="#1e293b" stroke-width="1.5"/>
            <line x1="8" y1="48" x2="20" y2="36" stroke="#2d3748" stroke-width="1.5"/>
            <line x1="36" y1="48" x2="48" y2="36" stroke="#2d3748" stroke-width="1.5"/>
          </svg>
          <span class="big">No data yet</span>
          <span class="small">Run at least one analysis on the Monitor page first</span>
          <a href="/" style="margin-top:12px;padding:10px 20px;background:var(--teal);color:#fff;border-radius:7px;text-decoration:none;font-size:13px;font-weight:600;">Go to Monitor</a>
        </div>`;
      return;
    }

    const bannerCls = d.failure_hours === null ? 'safe' : d.failure_hours < 6 ? 'danger' : 'warning';
    const BANNER_ICONS = {
      safe:    `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
      danger:  `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
      warning: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
    };
    const bannerIcon = BANNER_ICONS[bannerCls];
    const bannerTitle = d.failure_hours === null
      ? 'Machine healthy — No failure predicted in next 24h'
      : d.failure_hours < 6
        ? `Critical — Failure predicted in ~${d.failure_hours}h`
        : `Warning — Failure predicted in ~${d.failure_hours}h`;
    const bannerSub = d.failure_hours === null
      ? `Current risk: ${d.current_risk}% — Trend: ${d.trend}`
      : `Current risk: ${d.current_risk}% — Immediate action recommended`;

    document.getElementById('main-content').innerHTML = `
      <div class="prediction-banner ${bannerCls}">
        <span class="banner-icon">${bannerIcon}</span>
        <div>
          <div class="banner-title ${bannerCls}">${bannerTitle}</div>
          <div class="banner-sub">${bannerSub}</div>
        </div>
      </div>

      <div class="top-grid">
        <div class="kpi-card">
          <div class="kpi-value ${d.current_risk >= 50 ? 'alert' : d.current_risk >= 22 ? 'amber' : 'ok'}">${d.current_risk}%</div>
          <div class="kpi-label">Current risk</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value amber">${d.avg_risk_24h}%</div>
          <div class="kpi-label">Avg risk (history)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value ${d.anomaly_rate >= 30 ? 'alert' : 'ok'}">${d.anomaly_rate}%</div>
          <div class="kpi-label">Anomaly rate</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value purple">${d.total_analyses}</div>
          <div class="kpi-label">Total analyses</div>
        </div>
      </div>

      <div class="chart-card full" style="margin-bottom:20px">
        <div class="chart-title">Risk evolution — History + 24h simulation</div>
        <div id="chart-risk" style="height:280px"></div>
      </div>

      <div class="chart-grid">
        <div class="chart-card">
          <div class="chart-title">Tool wear projection</div>
          <div id="chart-wear" style="height:220px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Temperature trend</div>
          <div id="chart-temp" style="height:220px"></div>
        </div>
      </div>

      <div class="whatif-card">
        <div class="whatif-title">What-If Simulator — Test future scenarios</div>
        <div class="whatif-grid">
          <div class="whatif-field">
            <label class="whatif-label">Machine type</label>
            <select class="whatif-input" id="wi-type">
              <option value="0">L — Low</option>
              <option value="1">M — Medium</option>
              <option value="2">H — High</option>
            </select>
          </div>
          <div class="whatif-field">
            <label class="whatif-label">Air temp (K)</label>
            <input class="whatif-input" type="number" id="wi-temp-air" value="${d.last_params.temp_air}" step="0.1">
          </div>
          <div class="whatif-field">
            <label class="whatif-label">Speed (rpm)</label>
            <input class="whatif-input" type="number" id="wi-vitesse" value="${d.last_params.vitesse}" step="10">
          </div>
          <div class="whatif-field">
            <label class="whatif-label">Torque (Nm)</label>
            <input class="whatif-input" type="number" id="wi-couple" value="${d.last_params.couple}" step="0.1">
          </div>
          <div class="whatif-field">
            <label class="whatif-label">Tool wear (min)</label>
            <input class="whatif-input" type="number" id="wi-usure" value="${d.last_params.usure}" step="1">
          </div>
        </div>
        <button class="btn-simulate" onclick="runWhatIf()">🔮 Simulate 24h forecast</button>
        <div id="whatif-result" style="margin-top:16px"></div>
      </div>
    `;

    plotRisk(d);
    plotWear(d);
    plotTemp(d);
  }

  const layout_base = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#94a3b8', size: 11 },
    margin: { t: 10, b: 40, l: 45, r: 20 },
    xaxis: { gridcolor: '#1f2937', linecolor: '#1f2937', tickfont: { size: 10 } },
    yaxis: { gridcolor: '#1f2937', linecolor: '#1f2937', tickfont: { size: 10 } },
    legend: { bgcolor: 'transparent', font: { size: 10 } },
    hovermode: 'x unified'
  };

  function plotRisk(d) {
    const traces = [
      {
        x: d.history_times, y: d.history_risks, name: 'Historical risk',
        type: 'scatter', mode: 'lines+markers',
        line: { color: '#14b8a6', width: 2 },
        marker: { color: d.history_risks.map(r => r >= 50 ? '#ef4444' : r >= 22 ? '#f59e0b' : '#10b981'), size: 6 }
      },
      {
        x: d.future_times, y: d.future_risks, name: 'Simulated (24h)',
        type: 'scatter', mode: 'lines',
        line: { color: '#8b5cf6', width: 2, dash: 'dot' },
        fill: 'tozeroy', fillcolor: 'rgba(139,92,246,0.05)'
      },
      {
        x: [...d.history_times, ...d.future_times],
        y: Array(d.history_times.length + d.future_times.length).fill(50),
        name: 'Alert threshold', type: 'scatter', mode: 'lines',
        line: { color: '#ef4444', width: 1, dash: 'dash' }
      }
    ];
    Plotly.newPlot('chart-risk', traces, {...layout_base, yaxis: {...layout_base.yaxis, range: [0, 105], title: {text:'Risk %', font:{size:10}}}});
  }

  function plotWear(d) {
    const traces = [
      { x: d.history_times, y: d.history_wear, name: 'Actual wear', type: 'scatter', mode: 'lines+markers', line: {color:'#f59e0b', width:2}, marker: {size:5} },
      { x: d.future_times, y: d.future_wear, name: 'Projected wear', type: 'scatter', mode: 'lines', line: {color:'#f59e0b', width:2, dash:'dot'}, fill:'tozeroy', fillcolor:'rgba(245,158,11,0.05)' }
    ];
    Plotly.newPlot('chart-wear', traces, {...layout_base, yaxis: {...layout_base.yaxis, title:{text:'min', font:{size:10}}}});
  }

  function plotTemp(d) {
    const traces = [
      { x: d.history_times, y: d.history_temp, name: 'Process temp', type: 'scatter', mode: 'lines', line: {color:'#ef4444', width:2} },
      { x: d.future_times, y: d.future_temp, name: 'Projected temp', type: 'scatter', mode: 'lines', line: {color:'#ef4444', width:2, dash:'dot'} }
    ];
    Plotly.newPlot('chart-temp', traces, {...layout_base, yaxis: {...layout_base.yaxis, title:{text:'K', font:{size:10}}}});
  }

  async function runWhatIf() {
    const params = {
      type: parseInt(document.getElementById('wi-type').value),
      temp_air: parseFloat(document.getElementById('wi-temp-air').value),
      temp_process: parseFloat(document.getElementById('wi-temp-air').value) + 10,
      vitesse: parseFloat(document.getElementById('wi-vitesse').value),
      couple: parseFloat(document.getElementById('wi-couple').value),
      usure: parseFloat(document.getElementById('wi-usure').value)
    };

    const res = await fetch('/api/whatif', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(params)
    });
    const d = await res.json();

    const cls = d.risk >= 50 ? 'alert' : d.risk >= 22 ? 'amber' : 'ok';
    const colors = { alert: '#ef4444', amber: '#f59e0b', ok: '#10b981' };
    const color = colors[cls];

    document.getElementById('whatif-result').innerHTML = `
      <div style="padding:16px 20px;background:var(--bg);border:1px solid ${color};border-radius:8px;display:flex;align-items:center;gap:20px;">
        <div>
          <div style="font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:4px;">Simulated risk</div>
          <div style="font-size:36px;font-weight:800;color:${color}">${d.risk}%</div>
        </div>
        <div style="flex:1">
          <div style="font-size:13px;font-weight:600;color:${color};margin-bottom:4px;">${d.status}</div>
          <div style="font-size:11px;color:var(--muted)">${d.message}</div>
          ${d.zones.length > 0 ? `<div style="margin-top:8px;font-size:11px;color:var(--amber)">⚠️ Zones at risk: ${d.zones.map(z => z.nom + ' ' + z.proba + '%').join(' · ')}</div>` : ''}
        </div>
      </div>`;
  }

  loadTwin();
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────
# HISTORY HTML
# ──────────────────────────────────────────────
HISTORY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pilar — History</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0b0f1a; --surface: #111827; --border: #1f2937;
      --teal: #0d9488; --teal-light: #14b8a6;
      --red: #ef4444; --green: #10b981; --amber: #f59e0b;
      --text: #f1f5f9; --muted: #64748b; --label: #94a3b8;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    header { padding: 16px 32px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
    .logo { font-size: 18px; font-weight: 700; letter-spacing: 3px; color: var(--teal-light); }
    .sep { color: var(--border); }
    .page-title { color: var(--muted); font-size: 13px; letter-spacing: 1px; }
    .nav-links { margin-left: auto; display: flex; gap: 8px; }
    .nav-link { padding: 7px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--label); font-size: 12px; text-decoration: none; display: inline-flex; align-items: center; gap: 5px; }
    .nav-link:hover { border-color: var(--teal); color: var(--teal-light); }
    .content { padding: 30px 32px; }
    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
    .stat-value { font-size: 32px; font-weight: 700; }
    .stat-value.alert { color: var(--red); }
    .stat-value.ok { color: var(--green); }
    .stat-value.amber { color: var(--amber); }
    .stat-label { font-size: 11px; color: var(--muted); margin-top: 4px; letter-spacing: 1px; }
    .section-label { font-size: 10px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 14px; }
    table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 10px; overflow: hidden; font-size: 12px; }
    th { text-align: left; padding: 12px 16px; color: var(--muted); font-weight: 500; letter-spacing: 1px; border-bottom: 1px solid var(--border); font-size: 11px; }
    td { padding: 11px 16px; border-bottom: 1px solid var(--border); color: var(--label); }
    tr:last-child td { border-bottom: none; }
    .badge { padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }
    .badge.ok { background: rgba(16,185,129,0.15); color: var(--green); }
    .badge.alert { background: rgba(239,68,68,0.15); color: var(--red); }
    .mail-badge { padding: 3px 9px; border-radius: 20px; font-size: 11px; background: rgba(13,148,136,0.15); color: var(--teal-light); }
  </style>
</head>
<body>
<header>
  <span class="logo">PILAR</span>
  <span class="sep">/</span>
  <span class="page-title">Analysis History</span>
  <div class="nav-links">
    <a class="nav-link" href="/twin"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" y1="22" x2="12" y2="12"/></svg> Digital Twin</a>
    <a class="nav-link" href="/"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg> Monitor</a>
  </div>
</header>
<div class="content">
  <div class="stats">
    <div class="stat-card"><div class="stat-value">{{ total }}</div><div class="stat-label">Total analyses</div></div>
    <div class="stat-card"><div class="stat-value alert">{{ anomalies }}</div><div class="stat-label">Anomalies detected</div></div>
    <div class="stat-card"><div class="stat-value amber">{{ avg_risk }}%</div><div class="stat-label">Average risk score</div></div>
    <div class="stat-card"><div class="stat-value ok">{{ mails }}</div><div class="stat-label">Alerts sent</div></div>
  </div>
  <div class="section-label">All analyses</div>
  <table>
    <thead>
      <tr><th>Date & Time</th><th>Type</th><th>Air Temp</th><th>Speed</th><th>Torque</th><th>Tool Wear</th><th>Risk</th><th>Status</th><th>Failure Zones</th><th>Alert</th></tr>
    </thead>
    <tbody>
      {% for a in analyses %}
      <tr>
        <td>{{ a.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</td>
        <td>{{ a.machine_type }}</td>
        <td>{{ a.temp_air }} K</td>
        <td>{{ a.vitesse }} rpm</td>
        <td>{{ a.couple }} Nm</td>
        <td>{{ a.usure }} min</td>
        <td>{{ a.risk }}%</td>
        <td><span class="badge {{ 'alert' if a.prediction else 'ok' }}">{{ 'Anomaly' if a.prediction else 'Normal' }}</span></td>
        <td>{{ a.zones or '—' }}</td>
        <td>{% if a.mail_sent %}<span class="mail-badge">Sent</span>{% else %}—{% endif %}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>
"""

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
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
    zones_html = "".join(f"<li style='padding:5px 0;'>{z['nom']} — <strong>{z['proba']}%</strong></li>" for z in zones_risque) or "<li>Zone not identified</li>"
    html = f"""
    <div style="font-family:Arial;max-width:600px;margin:0 auto;background:#0b0f1a;color:#f1f5f9;padding:30px;border-radius:12px;">
        <h1 style="color:#ef4444;margin-bottom:5px;letter-spacing:2px;">FAILURE ALERT</h1>
        <p style="color:#64748b;margin-bottom:25px;">Pilar — Industrial Predictive Maintenance</p>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px;margin-bottom:20px;">
            <h2 style="color:#f59e0b;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">Risk Score</h2>
            <p style="font-size:48px;font-weight:800;color:#ef4444;margin:0;">{probabilite}%</p>
        </div>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px;margin-bottom:20px;">
            <h2 style="color:#f59e0b;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:15px;">Machine Parameters</h2>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="color:#64748b;padding:6px 0;">Type</td><td style="color:#f1f5f9;font-weight:600;">{mtype}</td></tr>
                <tr><td style="color:#64748b;padding:6px 0;">Air temp</td><td style="color:#f1f5f9;font-weight:600;">{data.get('temp_air')} K</td></tr>
                <tr><td style="color:#64748b;padding:6px 0;">Speed</td><td style="color:#f1f5f9;font-weight:600;">{data.get('vitesse')} rpm</td></tr>
                <tr><td style="color:#64748b;padding:6px 0;">Torque</td><td style="color:#f1f5f9;font-weight:600;">{data.get('couple')} Nm</td></tr>
                <tr><td style="color:#64748b;padding:6px 0;">Tool wear</td><td style="color:#f1f5f9;font-weight:600;">{data.get('usure')} min</td></tr>
            </table>
        </div>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px;margin-bottom:20px;">
            <h2 style="color:#f59e0b;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">Failure Zones</h2>
            <ul style="color:#f1f5f9;padding-left:20px;">{zones_html}</ul>
        </div>
        <p style="color:#64748b;font-size:11px;text-align:center;">Pilar — Industrial Predictive Maintenance System</p>
    </div>"""
    msg = MIMEMultipart('alternative')
    msg["Subject"] = f"[PILAR ALERT] Failure risk: {probabilite}%"
    msg['From'] = GMAIL
    msg['To'] = email_to
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL, GMAIL_PWD)
            smtp.sendmail(GMAIL, email_to, msg.as_string())
        logging.info("Alert email sent to %s", email_to)
    except Exception as e:
        logging.error("Failed to send alert email to %s: %s", email_to, e)

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
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
    return render_template_string(HISTORY_HTML, analyses=analyses, total=total, anomalies=anomalies, avg_risk=avg_risk, mails=mails)

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

    # Simulate next 24h based on last params + wear trend
    wear_rate = 1.5  # min per hour
    temp_drift = 0.05  # K per hour
    future_times, future_risks, future_wear, future_temp = [], [], [], []

    now = datetime.utcnow()
    cur_usure = last.usure
    cur_temp_p = last.temp_process
    failure_hours = None

    for h in range(1, 25):
        cur_usure = min(cur_usure + wear_rate, 250)
        cur_temp_p = min(cur_temp_p + temp_drift, 315)
        params = {
            'type': 1, 'temp_air': last.temp_air,
            'temp_process': cur_temp_p, 'vitesse': last.vitesse,
            'couple': last.couple, 'usure': cur_usure
        }
        risk, pred, _ = predict_risk(params)
        t = (now + timedelta(hours=h)).strftime('%H:%M')
        future_times.append(t)
        future_risks.append(risk)
        future_wear.append(round(cur_usure, 1))
        future_temp.append(round(cur_temp_p, 2))
        if failure_hours is None and risk >= 50:
            failure_hours = h

    total = len(analyses)
    anomalies = sum(1 for a in analyses if a.prediction)
    avg_risk = round(sum(a.risk for a in analyses) / total, 1)
    anomaly_rate = round(anomalies / total * 100, 1)

    # Trend
    if len(history_risks) >= 3:
        trend = 'increasing ↑' if history_risks[-1] > history_risks[-3] else 'stable →' if abs(history_risks[-1] - history_risks[-3]) < 2 else 'decreasing ↓'
    else:
        trend = 'stable →'

    return jsonify({
        'has_data': True,
        'current_risk': last.risk,
        'avg_risk_24h': avg_risk,
        'anomaly_rate': anomaly_rate,
        'total_analyses': total,
        'failure_hours': failure_hours,
        'trend': trend,
        'history_times': history_times,
        'history_risks': history_risks,
        'history_wear': history_wear,
        'history_temp': history_temp,
        'future_times': future_times,
        'future_risks': future_risks,
        'future_wear': future_wear,
        'future_temp': future_temp,
        'last_params': {
            'temp_air': last.temp_air, 'vitesse': last.vitesse,
            'couple': last.couple, 'usure': last.usure
        }
    })

@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    params = request.json
    params['temp_process'] = params['temp_air'] + 10
    risk, pred, zones_risque = predict_risk(params)
    if pred == 0:
        status = 'Normal operation'
        message = 'No failure predicted with these parameters.'
    elif risk < 50:
        status = 'Low anomaly risk'
        message = 'Minor risk detected. Monitor closely.'
    else:
        status = 'High failure risk — Action required'
        message = 'These parameters will likely lead to failure. Reduce tool wear or torque.'
    return jsonify({'risk': risk, 'status': status, 'message': message, 'zones': zones_risque})

@app.route("/set_email", methods=["POST"])
def set_email():
    data = request.get_json(silent=True) or {}
    with _email_lock:
        responsable_email["email"] = data.get("email", "")
    logging.info("Responsible email set: %s", responsable_email["email"])
    return jsonify({"status": "ok"})

@app.route("/predire", methods=["POST"])
def predire():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required = {"type", "temp_air", "temp_process", "vitesse", "couple", "usure"}
    missing = required - data.keys()
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(sorted(missing))}"}), 400

    probabilite, prediction, zones_risque = predict_risk(data)

    mail_envoye = False
    with _email_lock:
        recipient = responsable_email["email"]
    if probabilite >= 50 and recipient:
        threading.Thread(
            target=envoyer_alerte,
            args=(recipient, probabilite, zones_risque, data),
            daemon=True,
        ).start()
        mail_envoye = True

    machine_types = {0: "Low", 1: "Medium", 2: "High"}
    zones_str = ", ".join(z["nom"] for z in zones_risque)
    analysis = Analysis(
        machine_type=machine_types.get(data["type"], "Unknown"),
        temp_air=data["temp_air"],
        temp_process=data["temp_process"],
        vitesse=data["vitesse"],
        couple=data["couple"],
        usure=data["usure"],
        risk=probabilite,
        prediction=prediction,
        zones=zones_str,
        mail_sent=mail_envoye,
    )
    db.session.add(analysis)
    db.session.commit()

    return jsonify({"prediction": prediction, "probabilite": probabilite, "zones": zones_risque, "mail_envoye": mail_envoye})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    context = data.get('context')
    chat_history = data.get('history', [])

    if context:
        machine_types = {0: 'Low', 1: 'Medium', 2: 'High'}
        mtype = machine_types.get(context['data'].get('type', 0), 'Unknown')
        zones_str = ', '.join([f"{z['nom']} ({z['proba']}%)" for z in context['zones']]) or 'none'
        d = context['data']
        system_prompt = f"""You are Pilar, an intelligent industrial AI assistant specialized in machine maintenance and predictive analytics.

Current machine state (use only if relevant to the question):
- Status: {context['status']} | Failure risk: {context['risk']}%
- Type: {mtype} | Air temp: {d.get('temp_air')}K | Process temp: {d.get('temp_process')}K
- Speed: {d.get('vitesse')} rpm | Torque: {d.get('couple')} Nm | Tool wear: {d.get('usure')} min
- Failure zones: {zones_str}

Instructions:
- Answer the user's question directly and naturally
- Only mention machine data if the user asks about it or if it's directly relevant
- Be conversational, helpful and concise
- Never automatically summarize the machine state unless asked
- Max 3 sentences for simple questions, more detail only when needed"""
    else:
        system_prompt = "You are Pilar, an intelligent industrial AI assistant. Answer any questions helpfully. If asked about specific machine data, suggest running an analysis first."

    messages = [{"role": h['role'], "content": h['content']} for h in chat_history[:-1]]
    messages.append({"role": "user", "content": message})

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system_prompt,
            messages=messages,
        )
        reply = response.content[0].text
    except Exception as e:
        logging.error("Claude API error: %s", e)
        reply = (
            "I'm here to help with any questions about machine maintenance and predictive analytics."
            if not context
            else f"Machine is at {context['risk']}% failure risk."
        )

    return jsonify({'reply': reply})

if __name__ == "__main__":
    logging.info("Pilar server starting on http://localhost:5000")
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")