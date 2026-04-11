# -*- coding: utf-8 -*-
"""
PILAR Agents -- Shared configuration
Ollama local LLM + AutoGen multi-agent framework.
Zero cloud dependency -- runs fully offline on desktop.
"""

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Model assignment per agent role
# pilar-* models are fine-tuned via Ollama Modelfiles with few-shot industrial examples
# Falls back to base models if custom models not yet created
MODELS = {
    "diagnostic":   "pilar-diag",         # llama3.2 + few-shot maintenance examples
    "alert":        "pilar-alert",         # llama3.2 + few-shot alert format examples
    "maintenance":  "pilar-maintenance",   # mistral  + few-shot 5-step plan examples
    "watchdog":     "pilar-diag",          # reuse diagnostic model for SLA summaries
}

# Fallback base models if custom models unavailable
MODELS_FALLBACK = {
    "diagnostic":   "llama3.2",
    "alert":        "llama3.2",
    "maintenance":  "llama3.2",
    "watchdog":     "llama3.2",
}

def ollama_config(model_key: str) -> dict:
    """AutoGen-compatible LLM config for a given agent role."""
    return {
        "config_list": [{
            "model": MODELS[model_key],
            "base_url": OLLAMA_BASE_URL,
            "api_key": "ollama",
        }],
        "temperature": 0.2,
        "cache_seed": None,
    }

# SLA thresholds (internal)
SLA = {
    "prediction_max_ms":      500,    # ML prediction in <500ms
    "alert_max_seconds":      120,    # alert notification in <120s (CPU inference)
    "data_freshness_minutes":  60,    # sensor data must be <60 min old
    "agent_response_seconds": 200,    # LLM agent must answer in <200s (CPU-only, no GPU)
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

# Legacy (kept for compatibility)
LLM_CONFIG = ollama_config("diagnostic")


def llm_config(model_key: str) -> dict:
    return ollama_config(model_key)
