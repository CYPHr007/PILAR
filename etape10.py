from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# =========================
# 🔹 Chargement des modèles
# =========================
with open("modele_pannes.pkl", "rb") as f:
    model_principal = pickle.load(f)

with open("modeles_zones.pkl", "rb") as f:
    modeles_zones = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# 🔹 Mapping zones
# =========================
ZONES = {
    'HDF': '🌡️ Surchauffe',
    'TWF': '🔧 Usure outil',
    'PWF': '⚡ Surcharge électrique',
    'OSF': '⚙️ Contrainte mécanique',
    'RNF': '❓ Panne aléatoire'
}

# =========================
# 🔹 HTML (UI complète)
# =========================
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Analyse de panne industrielle</title>
<style>
body {
    background:#0f0f1a;
    color:white;
    font-family:Arial;
    padding:30px;
}
.container {
    max-width:700px;
    margin:auto;
}
.card {
    background:#16213e;
    border-radius:14px;
    padding:25px;
    margin-bottom:20px;
}
h1 {
    text-align:center;
    color:#00d4ff;
}
.grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:15px;
}
label {
    color:#aaa;
    font-size:14px;
}
input, select {
    width:100%;
    padding:10px;
    background:#0f0f1a;
    border:1px solid #2a2a4a;
    border-radius:8px;
    color:white;
}
button {
    margin-top:15px;
    width:100%;
    padding:15px;
    font-size:18px;
    font-weight:bold;
    background:#00d4ff;
    color:#0f0f1a;
    border:none;
    border-radius:12px;
    cursor:pointer;
}
button:hover {
    background:#00b8d9;
}
#result, #zones {
    display:none;
}
.ok { background:#0a2e1a; border:2px solid #00c853; }
.bad { background:#2e0a0a; border:2px solid #ff1744; }
.big {
    font-size:42px;
    font-weight:bold;
    margin:10px 0;
}
.bar {
    background:#0f0f1a;
    border-radius:10px;
    height:14px;
    overflow:hidden;
}
.fill {
    height:100%;
}
.zone {
    margin-top:10px;
}
</style>
</head>

<body>
<div class="container">
<h1>🔧 Analyse de panne industrielle</h1>

<div class="card">
<div class="grid">
<div>
<label>Type machine</label>
<select id="type">
<option value="0">Low</option>
<option value="1" selected>Medium</option>
<option value="2">High</option>
</select>
</div>
<div>
<label>Température air (K)</label>
<input id="temp_air" type="number" value="300">
</div>
<div>
<label>Température process (K)</label>
<input id="temp_process" type="number" value="310">
</div>
<div>
<label>Vitesse (RPM)</label>
<input id="vitesse" type="number" value="1500">
</div>
<div>
<label>Couple (Nm)</label>
<input id="couple" type="number" value="40">
</div>
<div>
<label>Usure outil (min)</label>
<input id="usure" type="number" value="100">
</div>
</div>

<button onclick="analyser()">🔍 Analyser la machine</button>
</div>

<div class="card" id="result">
<div id="titre"></div>
<div class="big" id="proba"></div>
<div class="bar">
<div class="fill" id="jauge"></div>
</div>
</div>

<div class="card" id="zones">
<h3>📍 Zones à risque</h3>
<div id="zones_list"></div>
</div>

</div>

<script>
async function analyser() {
    const payload = {
        type: Number(type.value),
        temp_air: Number(temp_air.value),
        temp_process: Number(temp_process.value),
        vitesse: Number(vitesse.value),
        couple: Number(couple.value),
        usure: Number(usure.value)
    };

    const res = await fetch("/predire", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    result.style.display = "block";
    proba.innerText = data.probabilite + "%";
    jauge.style.width = data.probabilite + "%";

    if (data.prediction === 1) {
        result.className = "card bad";
        titre.innerText = "🔴 Panne probable";
        jauge.style.background = "#ff1744";
    } else {
        result.className = "card ok";
        titre.innerText = "🟢 Machine saine";
        jauge.style.background = "#00c853";
    }

    zones_list.innerHTML = "";
    if (data.zones.length > 0) {
        zones.style.display = "block";
        data.zones.forEach(z => {
            zones_list.innerHTML += `
            <div class="zone">
                ${z.nom} — <b>${z.pct}%</b>
                <div class="bar">
                    <div class="fill" style="width:${z.pct}%;background:#ffa500"></div>
                </div>
            </div>`;
        });
    } else {
        zones.style.display = "none";
    }
}
</script>
</body>
</html>
"""

# =========================
# 🔹 Routes
# =========================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predire", methods=["POST"])
def predire():
    try:
        d = request.get_json()

        ecart_temp = d["temp_process"] - d["temp_air"]

        X = np.array([[ 
            d["type"],
            d["temp_air"],
            d["temp_process"],
            d["vitesse"],
            d["couple"],
            d["usure"],
            ecart_temp
        ]])

        Xs = scaler.transform(X)

        pred = int(model_principal.predict(Xs)[0])
        proba = round(model_principal.predict_proba(Xs)[0][1] * 100, 1)

        zones_res = []
        for k, nom in ZONES.items():
            if k in modeles_zones:
                p = round(modeles_zones[k].predict_proba(Xs)[0][1] * 100, 1)
                if p >= 30:
                    zones_res.append({"nom": nom, "pct": p})

        zones_res.sort(key=lambda x: x["pct"], reverse=True)

        return jsonify({
            "prediction": pred,
            "probabilite": proba,
            "zones": zones_res
        })

    except Exception as e:
        print("❌ ERREUR:", e)
        return jsonify({"error": str(e)}), 500

# =========================
if __name__ == "__main__":
    print("✅ Serveur actif sur http://localhost:5000")
    app.run()