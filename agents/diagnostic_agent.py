# -*- coding: utf-8 -*-
"""
PILAR DiagnosticAgent
---------------------
Analyses sensor data + ML output and produces a human-readable diagnosis
in French. Uses Qwen3 4B locally via llama-cpp-python.
Falls back to rule-based diagnosis if the model is not yet downloaded.
"""

import time
from agents.config import ZONE_LABELS, SLA
from agents import sla_tracker
from agents import llm_engine


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_diagnosis(data: dict) -> str:
    zone = data.get("zone", "")
    risk = data.get("risk_score", 0)
    vib  = data.get("vibration", 0)
    temp = data.get("temp_palier", 0)
    tp   = data.get("temp_moteur", 0)

    lines = []

    if risk >= 75:
        lines.append("CRITIQUE : risque de panne imminent detecte.")
    elif risk >= 50:
        lines.append("ATTENTION : degradation significative en cours.")
    else:
        lines.append("Etat surveille : anomalie faible detectee.")

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
        lines.append(f"Vibration elevee ({vib:.1f} mm/s) — depasse le seuil nominal.")
    if temp > 80:
        lines.append(f"Temperature palier elevee ({temp:.0f}°C) — risque de fusion graisse.")
    if tp > 100:
        lines.append(f"Temperature moteur critique ({tp:.0f}°C) — arret possible.")

    return " ".join(lines)


# ── LLM diagnosis ──────────────────────────────────────────────────────────────

_SYSTEM = (
    "Tu es un expert en maintenance industrielle de machines industrielles "
    "(pompes, compresseurs, ventilateurs, turbines, convoyeurs, systèmes robotiques). "
    "Tu analyses des donnees capteurs en temps reel et produis un diagnostic concis. "
    "Reponds UNIQUEMENT avec 4 lignes, sans markdown :\n"
    "DIAGNOSTIC: (1 phrase)\n"
    "CAUSE: (1 phrase)\n"
    "ACTION: (1 phrase)\n"
    "URGENCE: Immediate / 48h / Planifiee"
)


def _llm_diagnosis(data: dict) -> str | None:
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", "inconnu"))

    user = (
        f"Donnees capteurs :\n"
        f"  Vibration              : {data.get('vibration', 0):.2f} mm/s\n"
        f"  Temperature roulement  : {data.get('temp_palier', 0):.1f} °C\n"
        f"  Debit                  : {data.get('debit', 0):.2f} m3/h\n"
        f"  Pression entree        : {data.get('pression_entree', 0):.2f} bar\n"
        f"  Pression sortie        : {data.get('pression_sortie', 0):.2f} bar\n"
        f"  Courant moteur         : {data.get('courant_moteur', 0):.1f} A\n"
        f"  Temperature moteur     : {data.get('temp_moteur', 0):.1f} °C\n"
        f"  Heures fonctionnement  : {data.get('heure_fonctionnement', 0):.0f} h\n\n"
        f"Analyse ML PILAR :\n"
        f"  Score de risque        : {data.get('risk_score', 0):.1f}%\n"
        f"  Zone de defaillance    : {zone_fr}\n"
        f"  RUL estime             : {data.get('rul', 'N/A')} heures restantes"
    )

    return llm_engine.query(_SYSTEM, user, max_tokens=300, temperature=0.2)


# ── Public API ─────────────────────────────────────────────────────────────────

def diagnose(data: dict) -> dict:
    """
    Main entry point. Takes merged sensor + ML data dict, returns diagnosis dict.
    Keys: diagnosis, source, elapsed_ms, zone, risk_score
    """
    t0 = time.perf_counter()

    text = _llm_diagnosis(data)
    if text:
        source = "qwen3"
    else:
        text   = _rule_based_diagnosis(data)
        source = "rule_based"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    sla_tracker.record(
        component="DiagnosticAgent",
        metric="response_ms",
        value=elapsed_ms,
        threshold=SLA["agent_response_seconds"] * 1000,
        detail=f"source={source}",
    )

    return {
        "diagnosis":  text,
        "source":     source,
        "elapsed_ms": round(elapsed_ms),
        "zone":       data.get("zone"),
        "risk_score": data.get("risk_score"),
    }
