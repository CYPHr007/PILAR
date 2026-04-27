# -*- coding: utf-8 -*-
"""
PILAR AlertAgent
----------------
Decides whether to send an alert and generates a professional alert message.
Uses Qwen3 4B locally. Falls back to rule-based message when model unavailable.
"""

import time
from datetime import datetime
from agents.config import ZONE_LABELS, SLA
from agents import sla_tracker
from agents import llm_engine


# ── Alert decision ─────────────────────────────────────────────────────────────

def _should_alert(risk_score: float, previous_risk: float = 0) -> bool:
    if risk_score >= 75:
        return True
    if risk_score >= 50 and (risk_score - previous_risk) >= 15:
        return True
    return False


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_message(machine_name: str, data: dict, diagnosis: str) -> str:
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")
    urgency = "CRITIQUE" if risk >= 75 else "ATTENTION"

    return (
        f"[PILAR] {urgency} — Machine : {machine_name}\n"
        f"Date       : {ts}\n"
        f"Risque     : {risk:.0f}%  |  Zone : {zone_fr}  |  RUL : {rul}h\n"
        f"Diagnostic : {diagnosis}\n"
        f"Connectez-vous sur PILAR pour voir le detail complet."
    )


# ── LLM alert ─────────────────────────────────────────────────────────────────

_SYSTEM = (
    "Tu rediges des alertes de maintenance industrielle professionnelles en francais. "
    "Sois direct, precis, 5 lignes maximum. "
    "Commence obligatoirement par CRITIQUE: ou ATTENTION: selon l'urgence. "
    "Pas de markdown, pas de liste a puces."
)


def _llm_message(machine_name: str, data: dict, diagnosis: str) -> str | None:
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")

    user = (
        f"Machine    : {machine_name}\n"
        f"Date       : {ts}\n"
        f"Risque     : {risk:.0f}%\n"
        f"Zone       : {zone_fr}\n"
        f"RUL estime : {rul} heures\n"
        f"Diagnostic : {diagnosis}\n\n"
        f"Redige le message d'alerte pour le responsable maintenance."
    )

    return llm_engine.query(_SYSTEM, user, max_tokens=200, temperature=0.3)


# ── Public API ─────────────────────────────────────────────────────────────────

def process(machine_name: str, data: dict, diagnosis: str,
            previous_risk: float = 0) -> dict:
    """
    Decide if an alert is needed and generate the message.

    Returns dict with keys:
        should_alert, message, urgency, source, elapsed_ms
    """
    t0   = time.perf_counter()
    risk = data.get("risk_score", 0)

    if not _should_alert(risk, previous_risk):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sla_tracker.record("AlertAgent", "response_ms", elapsed_ms,
                           SLA["alert_max_seconds"] * 1000, "no_alert")
        return {
            "should_alert": False,
            "message":      "",
            "urgency":      "none",
            "source":       "none",
            "elapsed_ms":   round(elapsed_ms),
        }

    message = _llm_message(machine_name, data, diagnosis)
    if message:
        source = "qwen3"
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
        detail=f"source={source} urgency={urgency}",
    )

    return {
        "should_alert": True,
        "message":      message,
        "urgency":      urgency,
        "source":       source,
        "elapsed_ms":   round(elapsed_ms),
    }
