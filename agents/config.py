# -*- coding: utf-8 -*-
"""
PILAR Agents — Shared configuration
====================================
LLM backend: Qwen3 4B via llama-cpp-python (local, offline, no token cost).
Falls back to rule-based logic when model is not yet downloaded.
"""

# SLA thresholds
SLA = {
    "prediction_max_ms":      500,    # ML prediction in <500ms
    "alert_max_seconds":      120,    # alert notification in <120s
    "data_freshness_minutes":  60,    # sensor data must be <60 min old
    "agent_response_seconds": 120,    # LLM agent must answer in <120s (Qwen3 4B CPU)
    "model_min_recall":        0.85,  # model recall must stay above 85%
    "report_delay_hours":      24,    # weekly report within 24h
}

# Zone labels in French
ZONE_LABELS = {
    "CAV": "Cavitation",
    "ROL": "Roulement",
    "ETN": "Etancheite",
    "IMP": "Impulseur",
    "MOT": "Moteur",
}
