# -*- coding: utf-8 -*-
"""
PILAR Agent Orchestrator — Qwen3 4B (llama-cpp-python)
-------------------------------------------------------
Three specialized agents collaborate to produce a full analysis:

  DiagnosticAgent   → diagnosis text
  MaintenanceAgent  → 5-step maintenance plan
  AlertAgent        → alert message + send decision

All agents share a single Qwen3 4B model loaded once at startup.
No Ollama, no AutoGen, no cloud dependency. Fully offline after first download.
Falls back to rule-based logic if the model is not yet downloaded.
"""

import time
from agents.config import SLA
from agents import sla_tracker
from agents import diagnostic_agent
from agents import maintenance_agent
from agents import alert_agent


# ── Public API ────────────────────────────────────────────────────────────────

def run_agents(machine_name: str, sensor_data: dict,
               ml_result: dict, previous_risk: float = 0,
               history_summary: str = "") -> dict:
    """
    Full agent pipeline for one machine analysis.
    Safe to call from a background thread.

    Returns dict with keys: diagnosis, alert, maintenance, sla_ok, total_ms
    """
    t0   = time.perf_counter()
    data = {**sensor_data, **ml_result}
    risk = data.get("risk_score", 0)

    # ── Step 1: Diagnostic ────────────────────────────────────────────────────
    diag_result = diagnostic_agent.diagnose(data)
    diagnosis   = diag_result["diagnosis"]
    source      = diag_result["source"]

    # ── Step 2: Maintenance plan ──────────────────────────────────────────────
    plan_result = maintenance_agent.plan(data, diagnosis, history_summary)

    # ── Step 3: Alert decision + message ─────────────────────────────────────
    alert_result = alert_agent.process(machine_name, data, diagnosis, previous_risk)

    total_ms = (time.perf_counter() - t0) * 1000

    sla_tracker.record(
        "Orchestrator", "pipeline_ms", total_ms,
        SLA["agent_response_seconds"] * 3 * 1000,
        f"machine={machine_name} source={source}",
    )

    sla_tracker.log_conversation(
        machine=machine_name,
        risk_score=risk,
        zone=data.get("zone", ""),
        diag=diagnosis,
        plan="",
        alert=alert_result.get("message", ""),
        steps=plan_result.get("steps", []),
        source=source,
        total_ms=round(total_ms),
    )

    urgency = "critical" if risk >= 75 else ("warning" if risk >= 50 else "none")

    return {
        "diagnosis": {
            "diagnosis":  diagnosis,
            "source":     source,
            "elapsed_ms": diag_result["elapsed_ms"],
            "zone":       data.get("zone"),
            "risk_score": risk,
        },
        "alert": {
            "should_alert": alert_result["should_alert"],
            "message":      alert_result.get("message", ""),
            "urgency":      urgency,
            "elapsed_ms":   alert_result["elapsed_ms"],
        },
        "maintenance": {
            "urgency":            urgency,
            "delay":              plan_result["delay"],
            "rul_hours":          data.get("rul", "N/A"),
            "steps":              plan_result["steps"],
            "parts_needed":       plan_result.get("parts_needed", []),
            "estimated_duration": plan_result.get("estimated_duration", ""),
            "source":             plan_result["source"],
        },
        "sla_ok":   total_ms <= SLA["agent_response_seconds"] * 3 * 1000,
        "total_ms": round(total_ms),
    }


def get_sla_dashboard() -> list:
    return sla_tracker.get_dashboard()


def get_sla_history(component: str = None, hours: int = 24) -> list:
    return sla_tracker.get_history(component=component, hours=hours)
