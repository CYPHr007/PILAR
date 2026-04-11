# -*- coding: utf-8 -*-
"""
PILAR DiagnosticAgent
---------------------
Analyses sensor data + ML output and produces a human-readable diagnosis
in French using the local Ollama LLM.

Falls back to a rule-based diagnosis if Ollama is not running.
"""

import time
import requests
from agents.config import llm_config, ZONE_LABELS, SLA
LLM_CONFIG = llm_config("diagnostic")
from agents import sla_tracker

OLLAMA_URL = LLM_CONFIG["config_list"][0]["base_url"].replace("/v1", "")


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _rule_based_diagnosis(data: dict) -> str:
    """Fallback when Ollama is not running."""
    zone = data.get("zone", "")
    risk = data.get("risk_score", 0)
    vib  = data.get("vibration", 0)
    temp = data.get("temp_palier", 0)
    tp   = data.get("temp_moteur", 0)
    deb  = data.get("debit", 0)

    lines = []

    if risk >= 75:
        lines.append("CRITIQUE : risque de panne imminent detecte.")
    elif risk >= 50:
        lines.append("ATTENTION : degradation significative en cours.")
    else:
        lines.append("Etat surveillee : anomalie faible detectee.")

    if zone == "CAV":
        lines.append("Cavitation probable : debit trop faible ou pression d'aspiration insuffisante.")
        lines.append("Action : verifier la vanne d'aspiration et le niveau du reservoir.")
    elif zone == "ROL":
        lines.append("Defaut de roulement detecte via vibrations elevees.")
        lines.append("Action : planifier le remplacement des roulements sous 48-72h.")
    elif zone == "ETN":
        lines.append("Probleme d'etancheite possible (joint mecanique ou garniture).")
        lines.append("Action : inspecter les joints et verifier les fuites visuellement.")
    elif zone == "IMP":
        lines.append("Usure ou dommage sur l'impulseur detecte.")
        lines.append("Action : inspection interne recommandee, mesurer le jeu axial.")
    elif zone == "MOT":
        lines.append("Anomalie moteur : surchauffe ou surcharge electrique.")
        lines.append("Action : verifier le courant moteur, la ventilation et la temperature ambiante.")

    if vib > 20:
        lines.append(f"Vibration elevee ({vib:.1f} mm/s) -- depasse le seuil nominal.")
    if temp > 80:
        lines.append(f"Temperature palier elevee ({temp:.0f}C) -- risque de fusion graisse.")
    if tp > 100:
        lines.append(f"Temperature moteur critique ({tp:.0f}C) -- arret possible.")

    return " ".join(lines)


def _llm_diagnosis(data: dict) -> str:
    """Call Ollama for an intelligent diagnosis."""
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", "inconnu"))
    model   = LLM_CONFIG["config_list"][0]["model"]

    prompt = f"""Tu es un expert en maintenance industrielle de machines tournantes (pompes, compresseurs, ventilateurs).
Voici les donnees capteurs en temps reel :

  Vibration         : {data.get('vibration', 0):.2f} mm/s
  Temperature palier: {data.get('temp_palier', 0):.1f} C
  Debit             : {data.get('debit', 0):.2f} m3/h
  Pression entree   : {data.get('pression_entree', 0):.2f} bar
  Pression sortie   : {data.get('pression_sortie', 0):.2f} bar
  Courant moteur    : {data.get('courant_moteur', 0):.1f} A
  Temperature moteur: {data.get('temp_moteur', 0):.1f} C
  Heures fonct.     : {data.get('heure_fonctionnement', 0):.0f} h

Analyse ML PILAR :
  Score de risque   : {data.get('risk_score', 0):.1f}%
  Zone defaillance  : {zone_fr}
  RUL estime        : {data.get('rul', 'N/A')} heures restantes

Reponds UNIQUEMENT avec :
1. Diagnostic (1 phrase)
2. Cause probable (1 phrase)
3. Action recommandee (1 phrase)
4. Urgence : Immediate / 48h / Planifiee

Sois concis et precis. Pas de markdown, pas de listes, juste 4 lignes."""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=SLA["agent_response_seconds"]
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[LLM indisponible: {e}] " + _rule_based_diagnosis(data)


def diagnose(data: dict) -> dict:
    """
    Main entry point. Takes sensor + ML data, returns diagnosis dict.

    data = {
        'vibration', 'temp_palier', 'debit', 'pression_entree',
        'pression_sortie', 'courant_moteur', 'temp_moteur',
        'heure_fonctionnement', 'risk_score', 'zone', 'rul'
    }
    """
    t0 = time.perf_counter()

    if _ollama_available():
        text = _llm_diagnosis(data)
        source = "ollama"
    else:
        text = _rule_based_diagnosis(data)
        source = "rule_based"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Record SLA
    sla_tracker.record(
        component="DiagnosticAgent",
        metric="response_ms",
        value=elapsed_ms,
        threshold=SLA["agent_response_seconds"] * 1000,
        detail=f"source={source}"
    )

    return {
        "diagnosis": text,
        "source": source,
        "elapsed_ms": round(elapsed_ms),
        "zone": data.get("zone"),
        "risk_score": data.get("risk_score"),
    }
