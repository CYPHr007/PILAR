# -*- coding: utf-8 -*-
"""
PILAR Embedded — File Adapter
==============================
Watches a local file for new sensor data. Useful for machines that log
readings to a shared file (historians, PLCs that write to a network share,
or any system that exports CSV/JSON periodically).

Supports:
  - JSON file  — one object per file, updated in-place
  - JSONL file — newline-delimited JSON, one reading per line (reads last line)
  - CSV file   — one reading per row (reads last row, auto-detects columns)

Requires: no extra dependencies beyond numpy/pandas (already in requirements_embedded.txt)

Config example — JSON
---------------------
{
  "adapter":    "file",
  "machine_id": "PUMP-01",
  "interval":   5,
  "path":       "/var/pilar/sensors/latest.json",
  "format":     "json",
  "column_map": {
    "vib":          "vibration",
    "bearing_temp": "temp_palier",
    "flow":         "debit",
    "p_in":         "pression_entree",
    "p_out":        "pression_sortie",
    "motor_amps":   "courant_moteur",
    "motor_temp":   "temp_moteur",
    "run_hours":    "heure_fonctionnement"
  }
}

Config example — CSV (last row is the most recent reading)
-----------------------------------------------------------
{
  "adapter":    "file",
  "machine_id": "CONVEYOR-02",
  "path":       "/mnt/plc_export/sensor_log.csv",
  "format":     "csv",
  "column_map": {
    "Vibration_mms": "vibration",
    "MotorCurrent":  "courant_moteur"
  }
}

Notes
-----
- column_map is optional. If your CSV/JSON already uses PILAR sensor names,
  you do not need to provide it.
- The adapter emits a new reading only when the file modification time changes,
  so it will not re-emit the same data on every poll.
- For CSV files, the last non-empty row is used as the current reading.
"""

from __future__ import annotations
import csv
import json
import os
from typing import Optional

from adapters.base import BaseAdapter
from pilar_logging import get_logger

logger = get_logger("pilar.adapter.file")

# PILAR canonical sensor names — used for fuzzy auto-detection when no column_map given
_PILAR_SENSORS = {
    "vibration", "temp_palier", "debit",
    "pression_entree", "pression_sortie",
    "courant_moteur", "temp_moteur", "heure_fonctionnement",
}


class FileAdapter(BaseAdapter):

    def __init__(self, config: dict):
        super().__init__(config)
        self._path: str = config["path"]
        self._fmt: str = config.get("format", "json").lower()
        self._col_map: dict = config.get("column_map", {})
        self._last_mtime: float = 0.0

        if self._fmt not in ("json", "jsonl", "csv"):
            raise ValueError(f"Unsupported file format: '{self._fmt}'. Use json, jsonl, or csv.")

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> Optional[dict]:
        if not os.path.exists(self._path):
            logger.warning("Sensor file not found: %s", self._path)
            return None

        mtime = os.path.getmtime(self._path)
        if mtime == self._last_mtime:
            return None  # file unchanged — no new data

        self._last_mtime = mtime

        try:
            raw = self._read_raw()
        except Exception as e:
            logger.error("File read error (%s): %s", self._path, e)
            return None

        if raw is None:
            return None

        data = self._apply_map(raw)
        data["machine_id"] = self.machine_id
        logger.debug("File read: %s", data)
        return data

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_raw(self) -> Optional[dict]:
        if self._fmt == "json":
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)

        elif self._fmt == "jsonl":
            last_line = None
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if last_line is None:
                return None
            return json.loads(last_line)

        elif self._fmt == "csv":
            last_row = None
            with open(self._path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_row = row
            if last_row is None:
                return None
            # Convert all values to float where possible
            result = {}
            for k, v in last_row.items():
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = v
            return result

        return None

    def _apply_map(self, raw: dict) -> dict:
        """
        Apply column_map to rename fields to PILAR sensor names.
        If no column_map is given, pass through fields that are already PILAR names.
        """
        if not self._col_map:
            # No map provided — keep only known PILAR sensor keys
            return {k: v for k, v in raw.items() if k in _PILAR_SENSORS}

        result = {}
        for src_key, pilar_key in self._col_map.items():
            if src_key in raw:
                try:
                    result[pilar_key] = float(raw[src_key])
                except (ValueError, TypeError):
                    logger.warning("Non-numeric value for %s: %s", src_key, raw[src_key])

        # Also carry over any fields already using PILAR names
        for k, v in raw.items():
            if k in _PILAR_SENSORS and k not in result:
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    pass

        return result

    def close(self) -> None:
        pass  # no persistent connection to close
