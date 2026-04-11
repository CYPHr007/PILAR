# -*- coding: utf-8 -*-
"""
PILAR Agent Orchestrator -- AutoGen 0.4 + Ollama
-------------------------------------------------
Three specialized agents collaborate to produce a full analysis:

  DiagnosticAgent  (llama3.2 -- fast)
  MaintenanceAgent (mistral  -- quality)
  AlertAgent       (llama3.2 -- fast)

All agents run locally via Ollama. No cloud dependency.
Falls back to rule-based logic if Ollama is not running.
"""

import asyncio
import time
import requests
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="autogen_ext")

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False

from agents.config import MODELS, MODELS_FALLBACK, SLA, ZONE_LABELS
from agents import sla_tracker

OLLAMA_URL = "http://localhost:11434"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _available_models() -> set:
    """Return set of model names available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            return {m["name"].split(":")[0] for m in r.json().get("models", [])}
    except Exception:
        pass
    return set()


def _resolve_model(model_key: str) -> str:
    """Use custom pilar-* model if available, else fall back to base model."""
    preferred = MODELS[model_key]
    available = _available_models()
    if preferred.split(":")[0] in available:
        return preferred
    fallback = MODELS_FALLBACK.get(model_key, "llama3.2")
    print(f"[Orchestrator] {preferred} not found — using {fallback}")
    return fallback


def _make_client(model_key: str):
    return OpenAIChatCompletionClient(
        model=_resolve_model(model_key),
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        model_capabilities={
            "vision": False,
            "function_calling": False,
            "json_output": False,
        }
    )


# ── Rule-based fallbacks ──────────────────────────────────────────────────────

ZONE_PLANS = {
    "CAV": ["Verifier filtre aspiration", "Controler pression aspiration",
            "Verifier vanne aspiration", "Mesurer NPSH disponible",
            "Inspecter tuyauterie (presence air)"],
    "ROL": ["Mesurer vibrations (accelerometre)", "Inspecter roulements (bruit/chaleur)",
            "Verifier jeu axial et radial", "Regraisser ou remplacer roulements",
            "Controler alignement arbre-moteur"],
    "ETN": ["Inspecter joints mecaniques", "Verifier garnitures presse-etoupe",
            "Controler pression chambre joint", "Remplacer joints si deteriores",
            "Verifier liquide de purge (joint double)"],
    "IMP": ["Ouvrir pompe, inspecter impulseur", "Mesurer usure et jeu impulseur",
            "Verifier corps etrangers", "Remplacer impulseur si usure > 10%",
            "Controler diffuseur et volute"],
    "MOT": ["Mesurer courant moteur (3 phases)", "Verifier temperature enroulement",
            "Nettoyer ventilation / ailettes", "Mesurer resistance isolement",
            "Controler connexions electriques"],
}


def _rule_diagnosis(data: dict) -> str:
    zone    = data.get("zone", "")
    risk    = data.get("risk_score", 0)
    vib     = data.get("vibration", 0)
    temp    = data.get("temp_palier", 0)
    zone_fr = ZONE_LABELS.get(zone, zone)
    sev     = "CRITIQUE" if risk >= 75 else "ATTENTION"
    parts   = [f"{sev}: risque {risk:.0f}% -- zone {zone_fr}."]
    if vib  > 20: parts.append(f"Vibration elevee ({vib:.1f} mm/s).")
    if temp > 80: parts.append(f"Temp palier elevee ({temp:.0f}C).")
    parts.append(f"Action recommandee: {ZONE_PLANS.get(zone, ['Inspection generale'])[0]}.")
    return " ".join(parts)


def _rule_alert(machine_name: str, data: dict, diagnosis: str) -> str:
    from datetime import datetime
    risk    = data.get("risk_score", 0)
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), "")
    sev     = "CRITIQUE" if risk >= 75 else "ATTENTION"
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")
    return (f"[PILAR] {sev} -- {machine_name}\n"
            f"Date: {ts} | Risque: {risk:.0f}% | Zone: {zone_fr}\n"
            f"{diagnosis}\n"
            f"Connectez-vous sur PILAR pour le detail complet.")


# ── AutoGen async pipeline ────────────────────────────────────────────────────

async def _autogen_pipeline(machine_name: str, data: dict) -> dict:
    zone_fr = ZONE_LABELS.get(data.get("zone", ""), data.get("zone", ""))
    risk    = data.get("risk_score", 0)
    rul     = data.get("rul", "N/A")

    sensor_text = (
        f"Machine: {machine_name} | Risque: {risk:.0f}% | Zone: {zone_fr} | RUL: {rul}h\n"
        f"Vibration: {data.get('vibration',0):.1f}mm/s | "
        f"Temp palier: {data.get('temp_palier',0):.1f}C | "
        f"Temp moteur: {data.get('temp_moteur',0):.1f}C\n"
        f"Debit: {data.get('debit',0):.1f}m3/h | "
        f"Pression sortie: {data.get('pression_sortie',0):.1f}bar | "
        f"Courant: {data.get('courant_moteur',0):.1f}A | "
        f"Heures: {data.get('heure_fonctionnement',0):.0f}h"
    )

    # ── Step 1: Diagnostic (llama3.2 -- fast) ────────────────────────────
    diag_client = _make_client("diagnostic")
    diag_agent  = AssistantAgent(
        name="DiagnosticAgent",
        model_client=diag_client,
        system_message=(
            "Tu es expert en maintenance industrielle de machines tournantes. "
            "Quand tu recois des donnees capteurs, reponds EXACTEMENT en 3 lignes:\n"
            "DIAGNOSTIC: (1 phrase)\n"
            "CAUSE: (1 phrase)\n"
            "URGENCE: Immediate / 48h / Planifie\n"
            "Reponds en francais. Sois concis."
        ),
    )
    diag_team   = RoundRobinGroupChat(
        [diag_agent],
        termination_condition=MaxMessageTermination(2)
    )
    diag_result = await diag_team.run(task=f"Analyse ces donnees:\n{sensor_text}")
    diag_text   = next(
        (m.content for m in diag_result.messages if m.source != "user"),
        _rule_diagnosis(data)
    )
    await diag_client.close()

    # ── Step 2: Maintenance plan (mistral -- quality) ─────────────────────
    maint_client = _make_client("maintenance")
    maint_agent  = AssistantAgent(
        name="MaintenanceAgent",
        model_client=maint_client,
        system_message=(
            "Tu es planificateur de maintenance industrielle. "
            "Propose un plan en 5 etapes ordonnees. Format EXACT:\n"
            "DELAI: Immediate / 48h / Planifie\n"
            "PIECES: piece1, piece2, piece3\n"
            "1. etape\n2. etape\n3. etape\n4. etape\n5. etape\n"
            "Reponds en francais. Sois precis et court."
        ),
    )
    maint_team   = RoundRobinGroupChat(
        [maint_agent],
        termination_condition=MaxMessageTermination(2)
    )
    maint_result = await maint_team.run(
        task=f"Diagnostic: {diag_text}\nZone: {zone_fr} | Risque: {risk:.0f}%"
    )
    plan_text    = next(
        (m.content for m in maint_result.messages if m.source != "user"), ""
    )
    await maint_client.close()

    # ── Step 3: Alert message (llama3.2 -- fast) ─────────────────────────
    alert_client = _make_client("alert")
    alert_agent  = AssistantAgent(
        name="AlertAgent",
        model_client=alert_client,
        system_message=(
            "Tu rediges des alertes de maintenance industrielle professionnelles. "
            "4 lignes max. Commence par CRITIQUE: ou ATTENTION: "
            "Reponds en francais. Pas de markdown."
        ),
    )
    alert_team   = RoundRobinGroupChat(
        [alert_agent],
        termination_condition=MaxMessageTermination(2)
    )
    alert_result = await alert_team.run(
        task=(f"Machine: {machine_name} | Risque: {risk:.0f}%\n"
              f"Diagnostic: {diag_text}")
    )
    alert_text   = next(
        (m.content for m in alert_result.messages if m.source != "user"),
        _rule_alert(machine_name, data, diag_text)
    )
    await alert_client.close()

    # Parse steps from plan
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            steps.append(line[2:].strip())
    if not steps:
        steps = ZONE_PLANS.get(data.get("zone", ""), ["Inspection generale"])

    return {
        "diagnosis": diag_text,
        "plan_raw":  plan_text,
        "alert_msg": alert_text,
        "steps":     steps[:5],
        "source":    "autogen+ollama",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_agents(machine_name: str, sensor_data: dict,
               ml_result: dict, previous_risk: float = 0,
               history_summary: str = "") -> dict:
    """
    Full agent pipeline for one machine analysis.
    Runs AutoGen+Ollama if available, falls back to rule-based otherwise.
    Safe to call from a background thread.
    """
    t0   = time.perf_counter()
    data = {**sensor_data, **ml_result}
    risk = data.get("risk_score", 0)

    if HAS_AUTOGEN and _ollama_available():
        try:
            ag = asyncio.run(_autogen_pipeline(machine_name, data))
            diagnosis_text = ag["diagnosis"]
            alert_msg      = ag["alert_msg"]
            steps          = ag["steps"]
            source         = "autogen+ollama"
        except Exception as e:
            print(f"[Orchestrator] AutoGen error: {e} -- rule-based fallback")
            diagnosis_text = _rule_diagnosis(data)
            alert_msg      = _rule_alert(machine_name, data, diagnosis_text)
            steps          = ZONE_PLANS.get(data.get("zone", ""), ["Inspection generale"])
            source         = "rule_based"
    else:
        diagnosis_text = _rule_diagnosis(data)
        alert_msg      = _rule_alert(machine_name, data, diagnosis_text)
        steps          = ZONE_PLANS.get(data.get("zone", ""), ["Inspection generale"])
        source         = "rule_based"

    # Alert decision
    prev         = previous_risk or 0
    should_alert = risk >= 75 or (risk >= 50 and (risk - prev) >= 15)
    urgency      = "critical" if risk >= 75 else ("warning" if risk >= 50 else "none")

    total_ms = (time.perf_counter() - t0) * 1000

    sla_tracker.record(
        "Orchestrator", "pipeline_ms", total_ms,
        SLA["agent_response_seconds"] * 3 * 1000,
        f"machine={machine_name} source={source}"
    )

    # Log conversation for dashboard
    sla_tracker.log_conversation(
        machine=machine_name, risk_score=risk, zone=data.get("zone", ""),
        diag=diagnosis_text, plan="", alert=alert_msg,
        steps=steps, source=source, total_ms=round(total_ms)
    )

    return {
        "diagnosis": {
            "diagnosis":  diagnosis_text,
            "source":     source,
            "elapsed_ms": round(total_ms),
            "zone":       data.get("zone"),
            "risk_score": risk,
        },
        "alert": {
            "should_alert": should_alert,
            "message":      alert_msg if should_alert else "",
            "urgency":      urgency,
            "elapsed_ms":   round(total_ms),
        },
        "maintenance": {
            "urgency":   urgency,
            "delay":     "Immediate (<4h)" if risk >= 75 else "Sous 48h",
            "rul_hours": data.get("rul", "N/A"),
            "steps":     steps,
            "source":    source,
        },
        "sla_ok":   total_ms <= SLA["agent_response_seconds"] * 3 * 1000,
        "total_ms": round(total_ms),
    }


def get_sla_dashboard() -> list:
    return sla_tracker.get_dashboard()


def get_sla_history(component: str = None, hours: int = 24) -> list:
    return sla_tracker.get_history(component=component, hours=hours)
