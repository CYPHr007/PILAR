#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PILAR Embedded — Standalone inference CLI
==========================================
Runs the PILAR ML pipeline without Flask, a database, or any server dependency.
Designed for edge deployment inside robots, autonomous machines, and industrial
controllers running Linux or Windows.

Usage
-----
  # One-shot from stdin
  echo '{"vibration": 2.5, "temp_palier": 65, "courant_moteur": 4.8}' | python pilar_embedded.py

  # One-shot from file
  python pilar_embedded.py --input reading.json

  # Polling mode — re-reads the file every N seconds
  python pilar_embedded.py --input /var/pilar/latest.json --watch --interval 5

  # Adapter mode — reads from a configured sensor source (Modbus/file/HTTP/GPIO)
  python pilar_embedded.py --adapter adapters/examples/modbus_tcp.json

  # Daemon mode — minimal HTTP server on a local port
  python pilar_embedded.py --daemon --port 7000

Output
------
Each inference emits one JSON line to stdout:

  {
    "timestamp": "2026-04-27T09:12:00Z",
    "machine_id": "ROBOT-ARM-04",
    "risk_score": 78.4,
    "status": "critical",
    "zone": "ROL",
    "zone_label": "Bearing Failure",
    "rul_hours": 38.5,
    "anomaly_score": 62.1,
    "confidence": 87,
    "top_sensors": [
      {"feature": "vibration",      "impact": "+43%", "direction": "up"},
      {"feature": "temp_palier",    "impact": "+31%", "direction": "up"},
      {"feature": "courant_moteur", "impact": "+18%", "direction": "up"}
    ],
    "missing_sensors": ["debit"]
  }

If risk_score is below the threshold, status is "normal" and zone/zone_label are null.

Environment variables
---------------------
  PILAR_MODELS_DIR   Path to directory containing .pkl model files (default: same dir as this script)
  PILAR_LOG_DIR      Path for log files (default: ~/.pilar/logs on Linux, %LOCALAPPDATA%/PILAR on Windows)
  PILAR_MACHINE_ID   Default machine ID when not supplied in sensor data
  PILAR_THRESHOLD    Override decision threshold (0–100, default: from config.py)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Resolve model directory before any PILAR import ──────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_MODELS_DIR = Path(os.environ.get("PILAR_MODELS_DIR", _SCRIPT_DIR))

# ── PILAR imports (order matters: calibrator before pilar_ml) ─────────────────
sys.path.insert(0, str(_SCRIPT_DIR))

import pilar_calibrator  # noqa: F401 — registers PlattModel so pickle can load failure_model.pkl
import pilar_ml as ml
from pilar_logging import get_logger
from config import FAILURE_ZONES, DEFAULT_THRESHOLD

logger = get_logger("pilar.embedded")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> bool:
    ml.init_models(str(_MODELS_DIR))
    ok = ml.model is not None and ml.scaler is not None
    if ok:
        logger.info("Models loaded from %s", _MODELS_DIR)
    else:
        logger.error("Failed to load models from %s — check .pkl files exist", _MODELS_DIR)
    return ok


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(sensor_data: dict, machine_id: str = None,
                  threshold: float = None, machine_cfg=None) -> dict:
    """
    Run the full PILAR pipeline on one sensor reading.
    Returns a structured result dict ready for JSON serialisation.

    sensor_data keys (all optional — missing fields are imputed from medians):
        vibration, temp_palier, debit, pression_entree, pression_sortie,
        courant_moteur, temp_moteur, heure_fonctionnement

    Additional keys accepted but not used by the ML model:
        machine_id, machine_type, timestamp
    """
    if ml.model is None or ml.scaler is None:
        raise RuntimeError("Models not loaded — call _load_models() first")

    # Resolve machine config
    if machine_cfg is None and hasattr(run_inference, "_default_machine_cfg"):
        machine_cfg = run_inference._default_machine_cfg

    # Resolve threshold (machine config > CLI arg > env > config.py default)
    if threshold is None:
        if machine_cfg is not None and machine_cfg.risk_threshold is not None:
            threshold = machine_cfg.risk_threshold
        else:
            threshold = float(os.environ.get("PILAR_THRESHOLD", DEFAULT_THRESHOLD))

    mid = (
        machine_id
        or sensor_data.get("machine_id")
        or os.environ.get("PILAR_MACHINE_ID", "MACHINE-01")
    )

    # Extract sensor fields only (drop metadata keys)
    _META_KEYS = {"machine_id", "machine_type", "timestamp"}
    params = {k: v for k, v in sensor_data.items() if k not in _META_KEYS}

    # Build machine context from OEM config
    machine_context = machine_cfg.as_machine_context() if machine_cfg else None

    # Run prediction
    (
        risk_score,
        prediction,
        zones_risque,
        confidence,
        missing_keys,
        anomaly_score,
        shap_explanations,
        rul_hours,
    ) = ml.predict_risk(params, threshold=threshold, return_extra=True,
                        machine_context=machine_context)

    # Determine status
    if prediction == 0:
        status = "normal"
    elif risk_score >= 75:
        status = "critical"
    else:
        status = "warning"

    # Filter zones by machine config (disabled zones removed from output)
    if machine_cfg is not None:
        zones_risque = [
            z for z in zones_risque
            if machine_cfg.zone_enabled(
                next((k for k, v in FAILURE_ZONES.items() if v == z["nom"]), "")
            )
        ]

    # Top zone — apply custom label from machine config
    top_zone = zones_risque[0] if zones_risque else None
    zone_code = None
    zone_label = None
    if top_zone:
        zone_code = next(
            (k for k, v in FAILURE_ZONES.items() if v == top_zone["nom"]), None
        )
        zone_label = (
            machine_cfg.zone_label(zone_code)
            if (machine_cfg and zone_code)
            else top_zone["nom"]
        )

    # Top sensors from SHAP
    top_sensors = [
        {
            "feature": s["feature_key"],
            "impact": s["impact"],
            "direction": s["direction"],
        }
        for s in shap_explanations
    ]

    result = {
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine_id":      mid,
        "risk_score":      risk_score,
        "status":          status,
        "zone":            zone_code,
        "zone_label":      zone_label,
        "rul_hours":       rul_hours,
        "anomaly_score":   anomaly_score,
        "confidence":      confidence,
        "top_sensors":     top_sensors,
        "missing_sensors": missing_keys,
        "all_zones": [
            {
                "zone": next((k for k, v in FAILURE_ZONES.items() if v == z["nom"]), z["nom"]),
                "zone_label": (
                    machine_cfg.zone_label(
                        next((k for k, v in FAILURE_ZONES.items() if v == z["nom"]), "")
                    ) if machine_cfg else z["nom"]
                ),
                "probability": z["proba"],
            }
            for z in zones_risque
        ],
    }

    # Attach machine type metadata if config loaded
    if machine_cfg is not None:
        result["machine_type"] = machine_cfg.machine_type
        if machine_cfg.is_certified:
            result["model_certified"] = True
            result["certification_recall"] = machine_cfg.certification.get("recall")

    return result


def emit(result: dict) -> None:
    """Write one JSON line to stdout and flush."""
    print(json.dumps(result, ensure_ascii=False), flush=True)


# ── Input readers ─────────────────────────────────────────────────────────────

def read_stdin() -> dict:
    raw = sys.stdin.read().strip()
    return json.loads(raw)


def read_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Modes ─────────────────────────────────────────────────────────────────────

def mode_adapter(args):
    """
    Adapter mode: read from a configured sensor source (Modbus/file/HTTP/GPIO)
    and emit inference results in a continuous loop.
    """
    from adapters.base import load_adapter

    adapter = load_adapter(args.adapter)
    interval = args.interval or adapter.interval
    machine_id = args.machine_id or adapter.machine_id

    logger.info(
        "Adapter '%s' running — machine=%s interval=%.1fs (Ctrl+C to stop)",
        adapter.__class__.__name__, machine_id, interval,
    )

    try:
        while True:
            data = adapter.read()
            if data is not None:
                result = run_inference(data, machine_id=machine_id, threshold=args.threshold)
                emit(result)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Adapter stopped")
    finally:
        adapter.close()


def mode_once(args):
    """Read one JSON object and exit."""
    if args.input:
        data = read_file(args.input)
    else:
        data = read_stdin()
    result = run_inference(data, machine_id=args.machine_id, threshold=args.threshold)
    emit(result)


def mode_watch(args):
    """Poll a file every --interval seconds and emit on change or always."""
    path = args.input
    interval = args.interval
    last_mtime = None
    logger.info("Watching %s every %ds (Ctrl+C to stop)", path, interval)
    while True:
        try:
            mtime = os.path.getmtime(path)
            if mtime != last_mtime:
                last_mtime = mtime
                data = read_file(path)
                result = run_inference(data, machine_id=args.machine_id, threshold=args.threshold)
                emit(result)
        except FileNotFoundError:
            logger.warning("Sensor file not found: %s — retrying in %ds", path, interval)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in %s: %s", path, e)
        except Exception as e:
            logger.error("Watch loop error: %s", e)
        time.sleep(interval)


def mode_daemon(args):
    """Minimal HTTP server — POST /analyze with JSON body, returns JSON result."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *a):
            logger.debug("HTTP %s", format % a)

        def do_POST(self):
            if self.path != "/analyze":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                result = run_inference(
                    data,
                    machine_id=args.machine_id,
                    threshold=args.threshold,
                )
                payload = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({
                    "status": "ok",
                    "model_loaded": ml.model is not None,
                    "zones_loaded": len(ml.modeles_zones),
                    "rul_loaded": ml._rul_model is not None,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    logger.info(
        "PILAR Embedded daemon listening on port %d  (POST /analyze, GET /health)",
        args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Daemon stopped")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PILAR Embedded — standalone machine health inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", metavar="DIR",
        help="Directory containing .pkl model files (default: PILAR_MODELS_DIR env or script dir)",
    )
    parser.add_argument(
        "--adapter", metavar="CONFIG",
        help="JSON adapter config file (modbus/file/http/gpio — see adapters/examples/)",
    )
    parser.add_argument(
        "--input", metavar="FILE",
        help="JSON file with sensor readings (default: read from stdin)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Poll --input file continuously instead of reading once",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, metavar="SECONDS",
        help="Polling interval for --watch mode (default: 5s)",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as a minimal HTTP server (POST /analyze)",
    )
    parser.add_argument(
        "--port", type=int, default=7000,
        help="HTTP port for --daemon mode (default: 7000)",
    )
    parser.add_argument(
        "--machine-config", metavar="FILE",
        help="OEM machine config JSON (oem/examples/) — sets nominals, zone labels, thresholds",
    )
    parser.add_argument(
        "--machine-id", metavar="ID",
        help="Machine identifier to include in output (overrides PILAR_MACHINE_ID env)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="0-100",
        help="Decision threshold for failure (default: from config.py)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )

    args = parser.parse_args()

    # Override models dir from CLI arg
    global _MODELS_DIR
    if args.models:
        _MODELS_DIR = Path(args.models).resolve()

    import logging
    logging.getLogger("pilar").setLevel(getattr(logging, args.log_level))

    # Load OEM machine config if provided — stored as default on run_inference
    if args.machine_config:
        from oem.machine_config import MachineConfig
        mc = MachineConfig.from_file(args.machine_config)
        run_inference._default_machine_cfg = mc
        logger.info("Machine config: %s — %s", mc.machine_type, mc.certification_summary())

    if not _load_models():
        sys.exit(1)

    if args.daemon:
        mode_daemon(args)
    elif args.adapter:
        mode_adapter(args)
    elif args.watch:
        if not args.input:
            parser.error("--watch requires --input FILE")
        mode_watch(args)
    else:
        mode_once(args)


if __name__ == "__main__":
    main()
