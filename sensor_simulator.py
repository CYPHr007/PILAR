"""
╔══════════════════════════════════════════════════════════════╗
║         PILAR — Simulateur de Capteurs Industriels           ║
║         Inspiré de Kapachim Maroc — Jorf Lasfar              ║
║                                                              ║
║  Simule 3 machines en temps réel et envoie les données       ║
║  à Pilar automatiquement toutes les X secondes               ║
╚══════════════════════════════════════════════════════════════╝

Usage :
  python sensor_simulator.py                        → local (localhost:5000)
  python sensor_simulator.py --url https://your-app.example.com
  python sensor_simulator.py --url https://your-app.example.com --interval 10
"""

import requests
import time
import random
import math
import argparse
import sys
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Simulateur capteurs Pilar')
parser.add_argument('--url', default='http://localhost:5000', help='URL de Pilar')
parser.add_argument('--interval', type=float, default=5.0, help='Secondes entre chaque envoi')
parser.add_argument('--machine', default='all', help='all | reacteur | pompe | compresseur')
args = parser.parse_args()

PILAR_URL = args.url.rstrip('/')
INTERVAL  = args.interval

# ── COULEURS CONSOLE ──────────────────────────────────────────
class C:
    TEAL    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    GREY    = '\033[90m'
    BOLD    = '\033[1m'
    RESET   = '\033[0m'
    CLEAR   = '\033[2J\033[H'

def color_risk(r):
    if r < 25:  return f"{C.GREEN}{r:.1f}%{C.RESET}"
    if r < 50:  return f"{C.YELLOW}{r:.1f}%{C.RESET}"
    if r < 75:  return f"{C.RED}{r:.1f}%{C.RESET}"
    return f"{C.RED}{C.BOLD}{r:.1f}%{C.RESET}"

# ── DÉFINITION DES MACHINES ───────────────────────────────────
# Basé sur l'industrie chimique de Kapachim : 
# - Réacteur de mélange tensioactifs
# - Pompe de transfert acide/base
# - Compresseur pour conditionnement

MACHINES = {
    "RÉACTEUR-01": {
        "type":        1,         # Medium
        "description": "Réacteur mélange tensioactifs",
        # Valeurs nominales (normales)
        "temp_air_base":     298.0,   # 25°C
        "temp_process_base": 308.0,   # 35°C
        "vitesse_base":      1450.0,  # RPM
        "couple_base":       42.0,    # Nm
        "usure_initiale":    0.0,     # min
        # Variabilité normale
        "temp_air_noise":    0.3,
        "temp_process_noise":0.5,
        "vitesse_noise":     20.0,
        "couple_noise":      2.5,
        # Limites d'alarme
        "temp_process_max":  315.0,
        "vitesse_min":       1200.0,
        "couple_max":        65.0,
        "usure_max":         220.0,
        # Vitesse d'usure (min/cycle)
        "usure_rate":        0.8,
    },
    "POMPE-02": {
        "type":        0,         # Low
        "description": "Pompe transfert produit fini",
        "temp_air_base":     297.0,
        "temp_process_base": 305.0,
        "vitesse_base":      1750.0,
        "couple_base":       28.0,
        "usure_initiale":    45.0,
        "temp_air_noise":    0.2,
        "temp_process_noise":0.4,
        "vitesse_noise":     30.0,
        "couple_noise":      1.5,
        "temp_process_max":  312.0,
        "vitesse_min":       1500.0,
        "couple_max":        50.0,
        "usure_max":         200.0,
        "usure_rate":        0.6,
    },
    "COMPRESSEUR-03": {
        "type":        2,         # High
        "description": "Compresseur conditionnement",
        "temp_air_base":     299.0,
        "temp_process_base": 310.0,
        "vitesse_base":      2200.0,
        "couple_base":       55.0,
        "usure_initiale":    120.0,
        "temp_air_noise":    0.4,
        "temp_process_noise":0.8,
        "vitesse_noise":     40.0,
        "couple_noise":      3.0,
        "temp_process_max":  318.0,
        "vitesse_min":       1800.0,
        "couple_max":        80.0,
        "usure_max":         240.0,
        "usure_rate":        1.2,
    }
}

# ── ÉTAT DES MACHINES ─────────────────────────────────────────
machine_state = {}
for name, cfg in MACHINES.items():
    machine_state[name] = {
        "usure":        cfg["usure_initiale"],
        "temp_drift":   0.0,    # dérive progressive de température
        "vitesse_drift":0.0,    # dérive progressive de vitesse
        "anomalie_active": False,
        "anomalie_countdown": 0,
        "cycle": 0,
        "last_risk": 0.0,
        "last_prediction": 0,
    }

# ── GÉNÉRATEUR DE DONNÉES RÉALISTES ───────────────────────────
def generer_donnees(name, cfg, state):
    """
    Génère des données capteurs réalistes avec :
    - Bruit gaussien normal
    - Dérive progressive simulant l'usure
    - Anomalies occasionnelles (surcharge, surchauffe)
    """
    state["cycle"] += 1
    cycle = state["cycle"]

    # Usure progressive
    state["usure"] = min(state["usure"] + cfg["usure_rate"] + random.gauss(0, 0.1),
                         cfg["usure_max"])

    # Dérive thermique lente (toutes les 20 cycles, légère montée)
    if cycle % 20 == 0:
        state["temp_drift"] += random.gauss(0.1, 0.05)
        state["temp_drift"] = max(0, min(state["temp_drift"], 8.0))

    # Dérive de vitesse (légère baisse avec l'usure)
    state["vitesse_drift"] = -(state["usure"] / cfg["usure_max"]) * 50

    # Anomalie aléatoire (5% de chance par cycle, si pas déjà une)
    if not state["anomalie_active"] and random.random() < 0.05:
        state["anomalie_active"] = True
        state["anomalie_countdown"] = random.randint(3, 8)

    if state["anomalie_active"]:
        state["anomalie_countdown"] -= 1
        anomalie_factor = 1.0
        if state["anomalie_countdown"] <= 0:
            state["anomalie_active"] = False
        # Amplitude de l'anomalie
        anomalie_factor = 1.5 + random.gauss(0, 0.3)
    else:
        anomalie_factor = 1.0

    # ── Calcul des valeurs ──
    temp_air = cfg["temp_air_base"] + random.gauss(0, cfg["temp_air_noise"])

    temp_process = (cfg["temp_process_base"]
                    + state["temp_drift"]
                    + random.gauss(0, cfg["temp_process_noise"] * anomalie_factor))

    vitesse = (cfg["vitesse_base"]
               + state["vitesse_drift"]
               + random.gauss(0, cfg["vitesse_noise"] * anomalie_factor))

    couple = (cfg["couple_base"]
              + (state["usure"] / cfg["usure_max"]) * 10
              + random.gauss(0, cfg["couple_noise"] * anomalie_factor))

    # Clamp aux limites physiques
    temp_air     = max(293.0, min(305.0, temp_air))
    temp_process = max(295.0, min(320.0, temp_process))
    vitesse      = max(800.0,  min(3000.0, vitesse))
    couple       = max(5.0,   min(100.0, couple))
    usure        = round(state["usure"], 1)

    return {
        "type":         cfg["type"],
        "temp_air":     round(temp_air, 2),
        "temp_process": round(temp_process, 2),
        "vitesse":      round(vitesse, 1),
        "couple":       round(couple, 2),
        "usure":        usure,
        "_anomalie":    state["anomalie_active"],
    }

# ── ENVOI À PILAR ─────────────────────────────────────────────
def envoyer_a_pilar(name, donnees):
    payload = {k: v for k, v in donnees.items() if not k.startswith('_')}
    try:
        r = requests.post(f"{PILAR_URL}/predire", json=payload, timeout=8)
        if r.status_code == 200:
            res = r.json()
            return res.get('probabilite', 0), res.get('prediction', 0), res.get('zones', [])
        else:
            return None, None, None
    except requests.exceptions.ConnectionError:
        return None, None, None
    except Exception as e:
        return None, None, None

# ── AFFICHAGE CONSOLE ──────────────────────────────────────────
def afficher_dashboard(stats):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{C.TEAL}{C.BOLD}{'═'*65}{C.RESET}")
    print(f"{C.TEAL}{C.BOLD}  🏭 PILAR — Simulateur Kapachim Jorf Lasfar  {C.GREY}[{now}]{C.RESET}")
    print(f"{C.TEAL}{C.BOLD}{'═'*65}{C.RESET}")
    print(f"  {C.GREY}Cible : {PILAR_URL}  |  Intervalle : {INTERVAL}s{C.RESET}\n")

    for name, info in stats.items():
        d     = info['donnees']
        risk  = info['risk']
        pred  = info['pred']
        zones = info['zones']
        ok    = info['ok']
        anom  = d.get('_anomalie', False)

        status = f"{C.RED}⚠ ANOMALIE{C.RESET}" if anom else f"{C.GREEN}● Normal{C.RESET}"
        conn   = f"{C.GREEN}✓ Envoyé{C.RESET}" if ok else f"{C.RED}✗ Hors ligne{C.RESET}"

        print(f"  {C.BOLD}{name}{C.RESET}  {status}  {conn}")
        print(f"  {C.GREY}{'─'*60}{C.RESET}")

        # Paramètres
        print(f"  {C.GREY}T.Air:{C.RESET} {d['temp_air']}K  "
              f"{C.GREY}T.Process:{C.RESET} {d['temp_process']}K  "
              f"{C.GREY}Vitesse:{C.RESET} {d['vitesse']}rpm")
        print(f"  {C.GREY}Couple:{C.RESET} {d['couple']}Nm  "
              f"{C.GREY}Usure:{C.RESET} {d['usure']}min")

        # Barre de risque
        if risk is not None:
            bar_len = 30
            filled  = int(risk / 100 * bar_len)
            bar_col = C.GREEN if risk < 25 else C.YELLOW if risk < 50 else C.RED
            bar     = f"{bar_col}{'█' * filled}{'░' * (bar_len - filled)}{C.RESET}"
            print(f"  Risque : [{bar}] {color_risk(risk)}")
            if zones:
                zone_names = [z['nom'] for z in zones]
                print(f"  {C.RED}⚡ Zones : {', '.join(zone_names)}{C.RESET}")
        else:
            print(f"  {C.GREY}Risque : (pas de réponse de Pilar){C.RESET}")

        print()

    print(f"  {C.GREY}Ctrl+C pour arrêter{C.RESET}")

# ── BOUCLE PRINCIPALE ─────────────────────────────────────────
def main():
    print(f"\n{C.TEAL}{C.BOLD}  PILAR Simulateur démarré{C.RESET}")
    print(f"  Connexion à {PILAR_URL}...\n")

    # Test de connexion initial
    try:
        r = requests.get(PILAR_URL, timeout=5)
        print(f"  {C.GREEN}✓ Pilar accessible (HTTP {r.status_code}){C.RESET}\n")
    except:
        print(f"  {C.YELLOW}⚠ Pilar non accessible — les données seront envoyées quand même{C.RESET}\n")

    time.sleep(1)
    cycle_global = 0

    try:
        while True:
            cycle_global += 1
            stats = {}

            for name, cfg in MACHINES.items():
                if args.machine != 'all' and args.machine.lower() not in name.lower():
                    continue

                state   = machine_state[name]
                donnees = generer_donnees(name, cfg, state)
                risk, pred, zones = envoyer_a_pilar(name, donnees)

                if risk is not None:
                    state["last_risk"]       = risk
                    state["last_prediction"] = pred

                stats[name] = {
                    "donnees": donnees,
                    "risk":    risk if risk is not None else state["last_risk"],
                    "pred":    pred,
                    "zones":   zones or [],
                    "ok":      risk is not None,
                }

            # Affichage
            print(C.CLEAR, end='')
            afficher_dashboard(stats)

            # Résumé console simple (sans CLEAR)
            pannes = sum(1 for s in stats.values() if s['risk'] and s['risk'] >= 50)
            if pannes:
                print(f"  {C.RED}{C.BOLD}🚨 {pannes} machine(s) en zone de risque élevé !{C.RESET}\n")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  {C.TEAL}Simulateur arrêté. Bonne maintenance ! 🔧{C.RESET}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()