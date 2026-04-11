# -*- coding: utf-8 -*-
"""
PILAR MaintenanceAgent
----------------------
Generates a maintenance plan based on diagnosis + machine history.
Uses Ollama locally if available.
"""

import time
import requests
from agents.config import llm_config, ZONE_LABELS, SLA
LLM_CONFIG = llm_config("maintenance")
from agents import sla_tracker

OLLAMA_URL = LLM_CONFIG["config_list"][0]["base_url"].replace("/v1", "")


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# Rule-based maintenance plans per zone
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

URGENCY_DELAY = {
    "critical": "Intervention immediate (<4h)",
    "warning":  "Intervention sous 48h",
    "low":      "Planifier a la prochaine maintenance preventive",
}


def _rule_based_plan(data: dict, diagnosis: str) -> dict:
    """Fallback maintenance plan."""
    zone    = data.get("zone", "")
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    steps   = ZONE_PLANS.get(zone, ["Inspection generale recommandee"])
    urgency = "critical" if risk >= 75 else ("warning" if risk >= 50 else "low")

    return {
        "urgency": urgency,
        "delay": URGENCY_DELAY[urgency],
        "rul_hours": rul,
        "steps": steps,
        "parts_needed": _estimate_parts(zone),
        "estimated_duration": _estimate_duration(zone),
        "source": "rule_based",
    }


def _estimate_parts(zone: str) -> list:
    parts = {
        "CAV": ["Filtre aspiration", "Joint torique"],
        "ROL": ["Roulement SKF / FAG", "Graisse haute temperature", "Joint d'etancheite"],
        "ETN": ["Joint mecanique", "Garniture presse-etoupe", "O-rings"],
        "IMP": ["Impulseur de rechange", "Anneau d'usure", "Vis inox"],
        "MOT": ["Condensateur", "Ventilateur moteur", "Bornier de connexion"],
    }
    return parts.get(zone, ["Pieces generiques selon inspection"])


def _estimate_duration(zone: str) -> str:
    durations = {
        "CAV": "1-2 heures",
        "ROL": "2-4 heures",
        "ETN": "3-6 heures",
        "IMP": "4-8 heures (demontage complet)",
        "MOT": "2-4 heures",
    }
    return durations.get(zone, "A evaluer sur site")


def _llm_plan(data: dict, diagnosis: str, history_summary: str = "") -> dict:
    """Generate maintenance plan via Ollama."""
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    model   = LLM_CONFIG["config_list"][0]["model"]
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    hf      = data.get("heure_fonctionnement", 0)

    prompt = f"""Tu es un planificateur de maintenance industrielle.

Diagnostic recu : {diagnosis}
Zone de defaillance : {zone_fr}
Score de risque : {risk:.0f}%
RUL estime : {rul} heures restantes
Heures de fonctionnement : {hf:.0f}h
{f"Historique : {history_summary}" if history_summary else ""}

Genere un plan de maintenance en 5 etapes ordonnees.
Indique :
- Urgence (Immediate / 48h / Planifie)
- Duree estimee
- Pieces necessaires (3 max)
- Les 5 etapes d'intervention

Format :
URGENCE: ...
DUREE: ...
PIECES: ...
ETAPES:
1. ...
2. ...
3. ...
4. ...
5. ...

Sois precis et concis. Pas de markdown."""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=SLA["agent_response_seconds"]
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()

        # Parse the structured response
        lines  = raw.split("\n")
        steps  = []
        parts  = []
        delay  = "48h"
        dur    = "A evaluer"

        for line in lines:
            l = line.strip()
            if l.startswith("URGENCE:"):
                val = l.split(":", 1)[1].strip().lower()
                if "immed" in val:
                    delay = URGENCY_DELAY["critical"]
                elif "48" in val:
                    delay = URGENCY_DELAY["warning"]
                else:
                    delay = URGENCY_DELAY["low"]
            elif l.startswith("DUREE:"):
                dur = l.split(":", 1)[1].strip()
            elif l.startswith("PIECES:"):
                parts = [p.strip() for p in l.split(":", 1)[1].split(",")]
            elif l and l[0].isdigit() and "." in l[:3]:
                steps.append(l.split(".", 1)[1].strip())

        if not steps:
            steps = _rule_based_plan(data, diagnosis)["steps"]

        urgency = "critical" if risk >= 75 else ("warning" if risk >= 50 else "low")
        return {
            "urgency": urgency,
            "delay": delay,
            "rul_hours": rul,
            "steps": steps[:5],
            "parts_needed": parts or _estimate_parts(data.get("zone", "")),
            "estimated_duration": dur,
            "source": "ollama",
            "raw_llm": raw,
        }

    except Exception as e:
        result = _rule_based_plan(data, diagnosis)
        result["llm_error"] = str(e)
        return result


def plan(data: dict, diagnosis: str, history_summary: str = "") -> dict:
    """
    Generate a maintenance plan.

    Returns dict with keys:
        urgency, delay, rul_hours, steps, parts_needed,
        estimated_duration, source, elapsed_ms
    """
    t0 = time.perf_counter()

    if _ollama_available():
        result = _llm_plan(data, diagnosis, history_summary)
    else:
        result = _rule_based_plan(data, diagnosis)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    result["elapsed_ms"] = round(elapsed_ms)

    sla_tracker.record(
        component="MaintenanceAgent",
        metric="response_ms",
        value=elapsed_ms,
        threshold=SLA["agent_response_seconds"] * 1000,
        detail=f"source={result.get('source')} zone={data.get('zone')}"
    )

    return result
