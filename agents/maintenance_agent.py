# -*- coding: utf-8 -*-
"""
PILAR MaintenanceAgent
----------------------
Generates a 5-step maintenance plan based on the diagnosis + machine data.
Uses Qwen3 4B locally. Falls back to rule-based plans when model unavailable.
"""

import time
from agents.config import ZONE_LABELS, SLA
from agents import sla_tracker
from agents import llm_engine


# ── Rule-based fallback data ───────────────────────────────────────────────────

ZONE_PLANS = {
    "CAV": [
        "Verifier et nettoyer le filtre d'aspiration",
        "Controler le niveau et la pression du reservoir",
        "Verifier l'ouverture complete de la vanne d'aspiration",
        "Mesurer la pression d'aspiration (NPSH disponible)",
        "Inspecter la tuyauterie pour presence d'air",
    ],
    "ROL": [
        "Mesurer le niveau de vibrations avec accelerometre",
        "Inspecter visuellement les roulements (bruit, chaleur)",
        "Verifier le jeu axial et radial",
        "Regraisser ou remplacer les roulements",
        "Controler l'alignement arbre-moteur",
    ],
    "ETN": [
        "Inspecter les joints mecaniques pour fuites",
        "Verifier l'etat des garnitures et presse-etoupes",
        "Controler la pression de la chambre de joint",
        "Remplacer joints et garnitures si deteriores",
        "Verifier le liquide de purge si joint double",
    ],
    "IMP": [
        "Ouvrir la pompe et inspecter l'impulseur",
        "Mesurer l'usure et le jeu de l'impulseur",
        "Verifier la presence de corps etrangers",
        "Remplacer l'impulseur si usure > 10%",
        "Controler le diffuseur et la volute",
    ],
    "MOT": [
        "Mesurer le courant moteur sur les 3 phases",
        "Verifier la temperature de l'enroulement",
        "Controler la ventilation et nettoyer les ailettes",
        "Verifier la resistance d'isolement (megohmmetre)",
        "Controler les connexions electriques (serrage)",
    ],
}

_ZONE_PARTS = {
    "CAV": ["Filtre aspiration", "Joint torique"],
    "ROL": ["Roulement SKF/FAG", "Graisse haute temperature", "Joint d'etancheite"],
    "ETN": ["Joint mecanique", "Garniture presse-etoupe", "O-rings"],
    "IMP": ["Impulseur de rechange", "Anneau d'usure", "Vis inox"],
    "MOT": ["Condensateur", "Ventilateur moteur", "Bornier de connexion"],
}

_ZONE_DURATION = {
    "CAV": "1-2 heures",
    "ROL": "2-4 heures",
    "ETN": "3-6 heures",
    "IMP": "4-8 heures (demontage complet)",
    "MOT": "2-4 heures",
}

_URGENCY_DELAY = {
    "critical": "Intervention immediate (<4h)",
    "warning":  "Intervention sous 48h",
    "low":      "Planifier a la prochaine maintenance preventive",
}


def _rule_based_plan(data: dict) -> dict:
    zone    = data.get("zone", "")
    risk    = data.get("risk_score", 0)
    urgency = "critical" if risk >= 75 else ("warning" if risk >= 50 else "low")
    return {
        "urgency":            urgency,
        "delay":              _URGENCY_DELAY[urgency],
        "rul_hours":          data.get("rul", "N/A"),
        "steps":              ZONE_PLANS.get(zone, ["Inspection generale recommandee"]),
        "parts_needed":       _ZONE_PARTS.get(zone, ["Pieces selon inspection"]),
        "estimated_duration": _ZONE_DURATION.get(zone, "A evaluer sur site"),
        "source":             "rule_based",
    }


# ── LLM plan ──────────────────────────────────────────────────────────────────

_SYSTEM = (
    "Tu es un planificateur de maintenance industrielle expert, spécialisé dans les machines industrielles de tout type. "
    "A partir d'un diagnostic et de donnees machine, tu generes un plan d'intervention precis. "
    "Reponds UNIQUEMENT dans ce format, sans markdown :\n"
    "URGENCE: Immediate / 48h / Planifiee\n"
    "DUREE: ...\n"
    "PIECES: piece1, piece2, piece3\n"
    "1. etape\n2. etape\n3. etape\n4. etape\n5. etape"
)


def _parse_llm_plan(raw: str, data: dict) -> dict:
    """Parse structured LLM output into a plan dict."""
    lines   = raw.splitlines()
    steps   = []
    parts   = []
    delay   = _URGENCY_DELAY["warning"]
    dur     = "A evaluer"
    risk    = data.get("risk_score", 0)
    urgency = "critical" if risk >= 75 else ("warning" if risk >= 50 else "low")

    for line in lines:
        l = line.strip()
        if l.upper().startswith("URGENCE:"):
            val = l.split(":", 1)[1].strip().lower()
            if "immed" in val:
                delay = _URGENCY_DELAY["critical"]
            elif "48" in val:
                delay = _URGENCY_DELAY["warning"]
            else:
                delay = _URGENCY_DELAY["low"]
        elif l.upper().startswith("DUREE:"):
            dur = l.split(":", 1)[1].strip()
        elif l.upper().startswith("PIECES:"):
            parts = [p.strip() for p in l.split(":", 1)[1].split(",") if p.strip()]
        elif l and l[0].isdigit() and len(l) > 2 and l[1] in ".):":
            body = l[2:].strip()
            if body:
                steps.append(body)

    if not steps:
        steps = ZONE_PLANS.get(data.get("zone", ""), ["Inspection generale"])

    return {
        "urgency":            urgency,
        "delay":              delay,
        "rul_hours":          data.get("rul", "N/A"),
        "steps":              steps[:5],
        "parts_needed":       parts or _ZONE_PARTS.get(data.get("zone", ""), []),
        "estimated_duration": dur,
        "source":             "qwen3",
    }


def _llm_plan(data: dict, diagnosis: str, history_summary: str = "") -> dict | None:
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))

    user = (
        f"Diagnostic : {diagnosis}\n"
        f"Zone de defaillance : {zone_fr}\n"
        f"Score de risque : {data.get('risk_score', 0):.0f}%\n"
        f"RUL estime : {data.get('rul', 'N/A')} heures\n"
        f"Heures de fonctionnement : {data.get('heure_fonctionnement', 0):.0f}h"
        + (f"\nHistorique : {history_summary}" if history_summary else "")
    )

    raw = llm_engine.query(_SYSTEM, user, max_tokens=400, temperature=0.2)
    if raw is None:
        return None
    return _parse_llm_plan(raw, data)


# ── Public API ─────────────────────────────────────────────────────────────────

def plan(data: dict, diagnosis: str, history_summary: str = "") -> dict:
    """
    Generate a maintenance plan. Returns dict with keys:
    urgency, delay, rul_hours, steps, parts_needed, estimated_duration,
    source, elapsed_ms
    """
    t0 = time.perf_counter()

    result = _llm_plan(data, diagnosis, history_summary)
    if result is None:
        result = _rule_based_plan(data)

    elapsed_ms        = (time.perf_counter() - t0) * 1000
    result["elapsed_ms"] = round(elapsed_ms)

    sla_tracker.record(
        component="MaintenanceAgent",
        metric="response_ms",
        value=elapsed_ms,
        threshold=SLA["agent_response_seconds"] * 1000,
        detail=f"source={result.get('source')} zone={data.get('zone')}",
    )

    return result
