# -*- coding: utf-8 -*-
"""
PILAR SLA Tracker
-----------------
Measures and stores SLA metrics for every agent and system component.
Writes to pilar.db (existing Flask DB) so the dashboard can read it.
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from agents.config import SLA

DB_PATH = Path(__file__).parent.parent / "pilar.db"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_sla_tables():
    """Create SLA tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine     TEXT,
            risk_score  REAL,
            zone        TEXT,
            diag_text   TEXT,
            plan_text   TEXT,
            alert_text  TEXT,
            steps       TEXT,
            source      TEXT,
            total_ms    INTEGER
        );

        CREATE TABLE IF NOT EXISTS sla_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP,
            component   TEXT NOT NULL,
            metric      TEXT NOT NULL,
            value       REAL NOT NULL,
            threshold   REAL NOT NULL,
            status      TEXT NOT NULL,   -- 'ok' | 'warn' | 'breach'
            detail      TEXT
        );

        CREATE TABLE IF NOT EXISTS sla_status (
            component   TEXT PRIMARY KEY,
            last_ts     DATETIME,
            metric      TEXT,
            last_value  REAL,
            threshold   REAL,
            status      TEXT,
            detail      TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sla_events_ts
            ON sla_events(ts DESC);
        CREATE INDEX IF NOT EXISTS idx_sla_events_comp
            ON sla_events(component);
        """)


def record(component: str, metric: str, value: float,
           threshold: float, detail: str = ""):
    """Record one SLA measurement."""
    if value <= threshold:
        status = "ok"
    elif value <= threshold * 1.5:
        status = "warn"
    else:
        status = "breach"

    ts = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sla_events(ts,component,metric,value,threshold,status,detail) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, component, metric, value, threshold, status, detail)
        )
        conn.execute(
            """INSERT INTO sla_status(component,last_ts,metric,last_value,threshold,status,detail)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(component) DO UPDATE SET
                   last_ts=excluded.last_ts,
                   metric=excluded.metric,
                   last_value=excluded.last_value,
                   threshold=excluded.threshold,
                   status=excluded.status,
                   detail=excluded.detail""",
            (component, ts, metric, value, threshold, status, detail)
        )


def log_conversation(machine: str, risk_score: float, zone: str,
                     diag: str, plan: str, alert: str,
                     steps: list, source: str, total_ms: int):
    """Store a full agent conversation for the dashboard."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO agent_conversations
               (machine, risk_score, zone, diag_text, plan_text, alert_text, steps, source, total_ms)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (machine, risk_score, zone, diag, plan, alert,
             json.dumps(steps, ensure_ascii=False), source, total_ms)
        )


def get_conversations(limit: int = 20) -> list[dict]:
    """Return latest agent conversations for the dashboard."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_conversations ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["steps"] = json.loads(d["steps"] or "[]")
        except Exception:
            d["steps"] = []
        result.append(d)
    return result


@contextmanager
def measure(component: str, metric: str, threshold: float, detail: str = ""):
    """
    Context manager that times a block and records the SLA result.

    Usage:
        with measure("PredictionAgent", "response_ms", 500):
            result = model.predict(X)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record(component, metric, elapsed_ms, threshold, detail)


def get_dashboard() -> list[dict]:
    """Return current SLA status for all components."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sla_status ORDER BY component"
        ).fetchall()
    return [dict(r) for r in rows]


def get_history(component: str = None, hours: int = 24) -> list[dict]:
    """Return SLA event history."""
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with get_conn() as conn:
        if component:
            rows = conn.execute(
                "SELECT * FROM sla_events WHERE component=? AND ts>=? ORDER BY ts DESC LIMIT 500",
                (component, since)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sla_events WHERE ts>=? ORDER BY ts DESC LIMIT 1000",
                (since,)
            ).fetchall()
    return [dict(r) for r in rows]


def check_data_freshness(machine_id: str, last_analysis_dt: datetime):
    """Check if machine data is fresh enough per SLA."""
    age_minutes = (datetime.utcnow() - last_analysis_dt).total_seconds() / 60
    record(
        component=f"DataFreshness:{machine_id}",
        metric="data_age_minutes",
        value=age_minutes,
        threshold=SLA["data_freshness_minutes"],
        detail=f"Last analysis: {last_analysis_dt.strftime('%Y-%m-%d %H:%M')}"
    )
    return age_minutes <= SLA["data_freshness_minutes"]


def check_model_recall(recall: float):
    """Record model recall against SLA threshold."""
    record(
        component="MLModel",
        metric="recall",
        value=recall * 100,
        threshold=SLA["model_min_recall"] * 100,
        detail=f"Recall={recall*100:.1f}%"
    )
    return recall >= SLA["model_min_recall"]


# Auto-init on import
try:
    init_sla_tables()
except Exception:
    pass
