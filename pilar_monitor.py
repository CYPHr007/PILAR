# -*- coding: utf-8 -*-
"""
PILAR -- Background CSV file monitor.
Watches a CSV file for new rows and runs predictions on each.
"""
import os
import re
import threading
from typing import Dict, Optional, Callable, List
from pilar_logging import get_logger

logger = get_logger("pilar.monitor")

# Active monitors: path -> {stop, thread, rows, alerts, fname}
active_monitors: Dict[str, dict] = {}

# Desktop notification callback (set by launcher.py)
notify_callback: Optional[Callable] = None

# Column name patterns for fuzzy CSV header matching
_CSV_PATTERNS = {
    "vibration": [
        "vibration", "vib", "vibration_mm", "vib_mms", "vibration_mmps",
        "accel", "acceleration", "vibr",
    ],
    "temp_palier": [
        "temp_palier", "bearing_temp", "palier_temp", "t_palier", "tpalier",
        "bearing_temperature", "temp_roulement",
    ],
    "debit": [
        "debit", "flow", "flow_rate", "flowrate", "debit_m3h", "flow_m3h",
        "caudal", "durchfluss",
    ],
    "pression_entree": [
        "pression_entree", "inlet_pressure", "pressure_in", "p_in", "pe",
        "suction_pressure", "pression_aspiration",
    ],
    "pression_sortie": [
        "pression_sortie", "outlet_pressure", "discharge_pressure",
        "pressure_out", "p_out", "ps", "pression_refoulement",
    ],
    "courant_moteur": [
        "courant_moteur", "motor_current", "current_motor", "im", "courant_a",
        "current_a", "ampere_moteur", "motor_amp",
    ],
    "temp_moteur": [
        "temp_moteur", "motor_temp", "motor_temperature", "tm",
        "temperature_moteur", "t_moteur",
    ],
    "heure_fonctionnement": [
        "heure_fonctionnement", "run_hours", "operating_hours", "runtime",
        "heures", "hours", "hf", "total_hours",
    ],
}


def _normalize_header(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def detect_column_mapping(headers: List[str]) -> Dict[str, int]:
    """Map CSV headers to sensor fields using fuzzy matching."""
    norm = [_normalize_header(h) for h in headers]
    used: set = set()
    col_map: Dict[str, int] = {}

    for field, pats in _CSV_PATTERNS.items():
        best_idx, best_score = -1, 0
        for j, h in enumerate(norm):
            if j in used:
                continue
            score = 0
            for pi, p in enumerate(pats):
                if h == p:
                    score = 100 - pi
                    break
                elif p in h and score < 70 - pi:
                    score = 70 - pi
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx >= 0:
            col_map[field] = best_idx
            used.add(best_idx)

    return col_map


def _build_row(vals: List[str], col_map: Dict[str, int]) -> Optional[dict]:
    row = {}
    for field, idx in col_map.items():
        if idx < len(vals):
            try:
                row[field] = round(float(vals[idx].strip().replace(",", ".")), 2)
            except (ValueError, TypeError):
                row[field] = None
        else:
            row[field] = None
    return row if any(v is not None for v in row.values()) else None


def _notify(title: str, msg: str) -> None:
    """Send desktop notification via launcher callback, else log."""
    try:
        if notify_callback:
            notify_callback(title, msg)
        else:
            logger.warning("ALERT %s: %s", title, msg)
    except Exception as e:
        logger.error("Notification error: %s", e)


def monitor_loop(
    path: str,
    interval: int,
    stop_ev: threading.Event,
    predict_fn: Callable,
    machine_id: Optional[str] = None,
    save_fn: Optional[Callable] = None,
    entry: Optional[dict] = None,
) -> None:
    """
    Watch a CSV file for new rows and run predictions.

    predict_fn: callable(params_dict) -> (probabilite, prediction, zones, confidence, _)
    save_fn:    callable(params, prob, pred, zones, confidence) -> None
    """
    import datetime as _dt
    known_rows = 0
    col_map = None
    delim = ","
    if entry is None:
        entry = active_monitors.get(path, {})
    logger.info("Watching: %s", path)

    while not stop_ev.is_set():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = [l for l in content.split("\n") if l.strip()]
            if len(lines) < 2:
                stop_ev.wait(interval)
                continue

            if col_map is None:
                header = lines[0]
                delim = ";" if header.count(";") > header.count(",") else ","
                headers = [h.strip() for h in header.split(delim)]
                col_map = detect_column_mapping(headers)
                known_rows = len(lines) - 1
                logger.info("Mapped %d/8 columns — waiting for new rows", len(col_map))
                stop_ev.wait(interval)
                continue

            total = len(lines) - 1
            if total > known_rows:
                new_lines = lines[known_rows + 1 :]
                known_rows = total
                entry["rows"] = known_rows

                for line in new_lines:
                    vals = [v.strip() for v in line.split(delim)]
                    row = _build_row(vals, col_map)
                    if not row:
                        continue
                    if machine_id:
                        row["machine_id"] = machine_id
                    try:
                        result = predict_fn(row)
                        prob  = result[0]
                        pred  = result[1]
                        zones = result[2] if len(result) > 2 else []
                        conf  = result[3] if len(result) > 3 else 100

                        # Update live status
                        entry["last_risk"] = round(prob, 1)
                        entry["last_ts"]   = _dt.datetime.utcnow().isoformat() + "Z"

                        # Persist to DB if save_fn provided
                        if save_fn:
                            try:
                                save_fn(row, prob, pred, zones, conf)
                            except Exception as se:
                                logger.error("save_fn error: %s", se)

                        if pred == 1 and prob >= 45:
                            entry["alerts"] = entry.get("alerts", 0) + 1
                            fname = os.path.basename(path)
                            zone_str = (
                                ", ".join(z["nom"] for z in zones)
                                if zones
                                else "Anomaly detected"
                            )
                            _notify(
                                f"PILAR Alert \u2014 {fname}",
                                f"Risk: {prob:.0f}%  {zone_str}",
                            )
                    except Exception as e:
                        logger.error("Prediction error in monitor: %s", e)

        except FileNotFoundError:
            logger.warning("File not found — stopping monitor: %s", path)
            break
        except Exception as e:
            logger.error("Monitor read error: %s", e)

        stop_ev.wait(interval)

    active_monitors.pop(path, None)
    logger.info("Monitor stopped: %s", path)


def start_monitor(
    path: str,
    interval: int,
    predict_fn: Callable,
    machine_id: Optional[str] = None,
    save_fn: Optional[Callable] = None,
) -> dict:
    """Start a background monitor for a CSV file. Returns the monitor entry.

    save_fn: optional callable(params, prob, pred, zones, confidence) called for
             every new row so results can be persisted to the database.
    """
    # Stop any existing monitor for this path
    if path in active_monitors:
        active_monitors[path]["stop"].set()

    stop_ev = threading.Event()
    entry = {
        "stop": stop_ev,
        "path": path,
        "fname": os.path.basename(path),
        "rows": 0,
        "alerts": 0,
        "last_risk": None,
        "last_ts": None,
        "machine_db_id": None,
    }
    active_monitors[path] = entry

    t = threading.Thread(
        target=monitor_loop,
        args=(path, interval, stop_ev, predict_fn, machine_id, save_fn, entry),
        daemon=True,
        name=f"bgmon-{os.path.basename(path)}",
    )
    entry["thread"] = t
    t.start()
    return entry


def stop_monitor(path: str) -> bool:
    """Stop a running monitor. Returns True if it was running."""
    entry = active_monitors.get(path)
    if entry:
        entry["stop"].set()
        return True
    return False
