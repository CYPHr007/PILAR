from flask import Flask, jsonify, render_template_string
import pickle
import pandas as pd
import time
import threading
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ===== Load models =====
with open("modele_pannes.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("modeles_zones.pkl", "rb") as f:
    modeles_zones = pickle.load(f)

zones = {
    "TWF": "🔧 Usure outil",
    "HDF": "🌡️ Surchauffe",
    "PWF": "⚡ Surcharge électrique",
    "OSF": "⚙️ Contrainte mécanique",
    "RNF": "❓ Panne aléatoire",
}

COLONNES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "ecart_temp",
]

historique = []
hist_lock = threading.Lock()

# === Your HTML (unchanged except fetch cache-bust already OK) ===
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Pilar — Live Monitor</title>
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
        .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-left: auto; box-shadow: 0 0 8px rgba(16,185,129,0.5); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
        .content { padding: 30px 32px; }
        .section-label { font-size: 10px; letter-spacing: 2px; color: var(--muted); text-transform: uppercase; margin-bottom: 14px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; background: var(--surface); border-radius: 10px; overflow: hidden; }
        th { text-align: left; padding: 12px 16px; color: var(--muted); font-weight: 500; letter-spacing: 1px; border-bottom: 1px solid var(--border); font-size: 11px; }
        td { padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--label); }
        tr:last-child td { border-bottom: none; }
        .badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
        .badge.ok { background: rgba(16,185,129,0.15); color: var(--green); }
        .badge.alert { background: rgba(239,68,68,0.15); color: var(--red); }
        .zone-tag { display: inline-block; padding: 2px 8px; background: rgba(245,158,11,0.15); color: var(--amber); border-radius: 4px; font-size: 11px; margin: 2px; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
        .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
        .stat-value { font-size: 28px; font-weight: 700; }
        .stat-label { font-size: 11px; color: var(--muted); margin-top: 4px; letter-spacing: 1px; }
        .stat-value.alert { color: var(--red); }
        .stat-value.ok { color: var(--green); }
    </style>
    <script>
        async function refresh() {
            try {
                const res = await fetch('/data?t=' + Date.now(), { cache: "no-store" });
                if (!res.ok) return;
                const data = await res.json();

                const total = data.length;
                const alertes = data.filter(d => d.alert).length;
                const dernierRisque = total > 0 ? data[0].probabilite : 0;
                const statut = total > 0 && data[0].alert ? 'alert' : 'ok';

                document.getElementById('stat-total').textContent = total;
                document.getElementById('stat-alertes').textContent = alertes;
                document.getElementById('stat-alertes').className = 'stat-value ' + (alertes > 0 ? 'alert' : 'ok');
                document.getElementById('stat-risque').textContent = dernierRisque + '%';
                document.getElementById('stat-risque').className = 'stat-value ' + statut;
                document.getElementById('stat-statut').textContent = statut === 'alert' ? 'Anomaly' : 'Normal';
                document.getElementById('stat-statut').className = 'stat-value ' + statut;

                let html = '';
                data.forEach(row => {
                    const zonesHtml = row.zones && row.zones.length > 0
                        ? row.zones.map(z => `<span class="zone-tag">${z.nom} ${z.proba}%</span>`).join('')
                        : '<span style="color:var(--muted);font-size:11px">—</span>';

                    html += `<tr>
                        <td>${row.temps}</td>
                        <td>${row.temp_air}K</td>
                        <td>${row.vitesse} rpm</td>
                        <td>${row.probabilite}%</td>
                        <td><span class="badge ${row.alert ? 'alert' : 'ok'}">${row.statut}</span></td>
                        <td>${zonesHtml}</td>
                    </tr>`;
                });
                document.getElementById('tbody').innerHTML =
                    html || '<tr><td colspan="6" style="text-align:center;color:var(--muted)">Waiting for data...</td></tr>';

            } catch (e) {}
        }
        setInterval(refresh, 1000);
        refresh();
    </script>
</head>
<body>
<header>
    <span class="logo">PILAR</span>
    <span class="sep">/</span>
    <span class="page-title">Live Monitoring</span>
    <div class="live-dot"></div>
</header>
<div class="content">
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="stat-total">—</div>
            <div class="stat-label">Total readings</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-alertes">—</div>
            <div class="stat-label">Anomalies detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-risque">—</div>
            <div class="stat-label">Last risk score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-statut">—</div>
            <div class="stat-label">Current status</div>
        </div>
    </div>
    <div class="section-label">Live feed</div>
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Air Temp</th>
                <th>Speed</th>
                <th>Risk</th>
                <th>Status</th>
                <th>Failure zones</th>
            </tr>
        </thead>
        <tbody id="tbody">
            <tr><td colspan="6" style="text-align:center;color:var(--muted)">Waiting for data...</td></tr>
        </tbody>
    </table>
</div>
</body>
</html>
"""

def lire_capteurs():
    """
    Live loop: every cycle, read LAST ROW of CSV and append one entry to historique.
    Even if the CSV overwrites the last row, the dashboard still updates because timestamp changes.
    """
    while True:
        try:
            df = pd.read_csv("capteurs_live.csv")
            if df.empty:
                time.sleep(1)
                continue

            row = df.iloc[-1]

            ecart_temp = float(row["temp_process"]) - float(row["temp_air"])

            donnees = pd.DataFrame([[
                row["type"],
                float(row["temp_air"]),
                float(row["temp_process"]),
                float(row["vitesse"]),
                float(row["couple"]),
                float(row["usure"]),
                ecart_temp
            ]], columns=COLONNES)

            donnees_scaled = scaler.transform(donnees)

            probabilite = round(float(model.predict_proba(donnees_scaled)[0][1]) * 100, 1)
            prediction = 1 if probabilite >= 46 else 0

            zones_risque = []
            if prediction == 1:
                for code_zone, nom_zone in zones.items():
                    if code_zone in modeles_zones:
                        pz = round(float(modeles_zones[code_zone].predict_proba(donnees_scaled)[0][1]) * 100, 1)
                        if pz >= 30:
                            zones_risque.append({"nom": nom_zone, "proba": pz})
                zones_risque.sort(key=lambda x: x["proba"], reverse=True)

            entry = {
                "temps": time.strftime("%H:%M:%S"),
                "temp_air": float(row["temp_air"]),
                "vitesse": float(row["vitesse"]),
                "probabilite": probabilite,
                "statut": "Anomaly" if prediction == 1 else "Normal",
                "alert": bool(prediction == 1),
                "zones": zones_risque,
            }

            with hist_lock:
                historique.append(entry)
                if len(historique) > 20:
                    historique.pop(0)

            print(f"[{entry['temps']}] Risque: {probabilite}% — {'ALERTE' if prediction == 1 else 'Normal'}")

        except Exception as e:
            print(f"Erreur: {e}")

        time.sleep(1)  # one cycle per second


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/data")
def data():
    with hist_lock:
        payload = list(reversed(historique))

    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


if __name__ == "__main__":
    threading.Thread(target=lire_capteurs, daemon=True).start()
    print("✅ Live monitor démarré sur http://localhost:5001")
    app.run(port=5001, debug=False, threaded=True)