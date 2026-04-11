# -*- coding: utf-8 -*-
"""
PILAR AlertAgent
----------------
Decides whether to send an alert and generates intelligent alert messages.
Uses Ollama locally if available, rule-based fallback otherwise.
"""

import time
import requests
from datetime import datetime
from agents.config import llm_config, ZONE_LABELS, SLA
LLM_CONFIG = llm_config("alert")
from agents import sla_tracker

OLLAMA_URL = LLM_CONFIG["config_list"][0]["base_url"].replace("/v1", "")


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _should_alert(risk_score: float, zone: str, previous_risk: float = 0) -> bool:
    """Decide if an alert should be sent."""
    if risk_score >= 75:
        return True
    if risk_score >= 50 and (risk_score - previous_risk) >= 15:
        return True
    return False


def _rule_based_message(machine_name: str, data: dict, diagnosis: str) -> str:
    """Fallback alert message."""
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")

    urgency = "CRITIQUE" if risk >= 75 else "ATTENTION"

    return f"""[PILAR] {urgency} -- Machine : {machine_name}
Date        : {ts}
Risque      : {risk:.0f}%
Zone        : {zone_fr}
RUL estime  : {rul} heures

Diagnostic  : {diagnosis}

Connectez-vous sur PILAR pour voir le detail complet."""


def _llm_message(machine_name: str, data: dict, diagnosis: str) -> str:
    """Generate a smart alert message via Ollama."""
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    model   = LLM_CONFIG["config_list"][0]["model"]
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")

    prompt = f"""Tu rediges un message d'alerte de maintenance industrielle professionnel.
Machine     : {machine_name}
Risque      : {risk:.0f}%
Zone        : {zone_fr}
RUL estime  : {rul} heures
Diagnostic  : {diagnosis}

Redige un message d'alerte court (5 lignes max) pour le responsable maintenance.
Commence par le niveau d'urgence (CRITIQUE / ATTENTION / INFO).
Mentionne la machine, le probleme, et l'action recommandee.
Pas de markdown. Professionnel et direct."""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=SLA["agent_response_seconds"]
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return _rule_based_message(machine_name, data, diagnosis)


def process(machine_name: str, data: dict, diagnosis: str,
            previous_risk: float = 0) -> dict:
    """
    Decide if alert needed and generate the message.

    Returns:
        {
            'should_alert': bool,
            'message': str,
            'urgency': str,   # 'critical' | 'warning' | 'none'
            'elapsed_ms': int
        }
    """
    t0 = time.perf_counter()

    risk = data.get("risk_score", 0)
    alert_needed = _should_alert(risk, data.get("zone"), previous_risk)

    if not alert_needed:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sla_tracker.record("AlertAgent", "response_ms", elapsed_ms,
                           SLA["alert_max_seconds"] * 1000, "no_alert")
        return {
            "should_alert": False,
            "message": "",
            "urgency": "none",
            "elapsed_ms": round(elapsed_ms)
        }

    if _ollama_available():
        message = _llm_message(machine_name, data, diagnosis)
        source  = "ollama"
    else:
        message = _rule_based_message(machine_name, data, diagnosis)
        source  = "rule_based"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    urgency    = "critical" if risk >= 75 else "warning"

    sla_tracker.record(
        component="AlertAgent",
        metric="response_ms",
        value=elapsed_ms,
        threshold=SLA["alert_max_seconds"] * 1000,
        detail=f"source={source} urgency={urgency}"
    )

    return {
        "should_alert": True,
        "message": message,
        "urgency": urgency,
        "source": source,
        "elapsed_ms": round(elapsed_ms)
    }
