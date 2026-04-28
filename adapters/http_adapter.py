# -*- coding: utf-8 -*-
"""
PILAR Embedded — HTTP Adapter
==============================
Polls a REST/HTTP endpoint on a configurable interval and extracts sensor
values from the JSON response. Useful for machines or controllers that
already expose a local HTTP API (common in modern CNCs, collaborative robots,
and IoT-enabled PLCs).

Requires: no extra dependencies — uses stdlib urllib only.

Config example
--------------
{
  "adapter":    "http",
  "machine_id": "CNC-01",
  "interval":   10,

  "url":     "http://controller.local/api/sensors",
  "method":  "GET",
  "headers": {
    "Authorization": "Bearer my-token",
    "Accept": "application/json"
  },

  "field_map": {
    "sensors.vibration":     "vibration",
    "sensors.bearing_temp":  "temp_palier",
    "sensors.flow":          "debit",
    "inlet_pressure":        "pression_entree",
    "outlet_pressure":       "pression_sortie",
    "motor.current":         "courant_moteur",
    "motor.temperature":     "temp_moteur",
    "run_hours":             "heure_fonctionnement"
  },

  "timeout": 5
}

Field map syntax
----------------
Keys are dot-notation paths into the JSON response.
"sensors.vibration" → response["sensors"]["vibration"]
"motor.current"     → response["motor"]["current"]
Top-level keys need no dot: "vibration" → response["vibration"]

POST body (optional)
--------------------
To POST a request body instead of GET:
  "method": "POST",
  "body": {"device_id": "sensor-board-01"}
"""

from __future__ import annotations
import json
import urllib.error
import urllib.request
from typing import Optional, Any

from adapters.base import BaseAdapter
from pilar_logging import get_logger

logger = get_logger("pilar.adapter.http")


def _get_nested(obj: Any, path: str) -> Optional[Any]:
    """Resolve a dot-notation path into a nested dict. Returns None if missing."""
    parts = path.split(".")
    for part in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
        if obj is None:
            return None
    return obj


class HttpAdapter(BaseAdapter):

    def __init__(self, config: dict):
        super().__init__(config)
        self._url: str = config["url"]
        self._method: str = config.get("method", "GET").upper()
        self._headers: dict = config.get("headers", {})
        self._body: Optional[dict] = config.get("body")
        self._field_map: dict = config.get("field_map", {})
        self._timeout: int = int(config.get("timeout", 5))

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> Optional[dict]:
        try:
            response_json = self._fetch()
        except Exception as e:
            logger.error("HTTP fetch error (%s): %s", self._url, e)
            return None

        if response_json is None:
            return None

        data = self._extract(response_json)
        if not data:
            logger.warning("HTTP response yielded no sensor values from %s", self._url)
            return None

        data["machine_id"] = self.machine_id
        logger.debug("HTTP read: %s", data)
        return data

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch(self) -> Optional[dict]:
        body_bytes = None
        if self._body is not None:
            body_bytes = json.dumps(self._body).encode()

        req = urllib.request.Request(
            self._url,
            data=body_bytes,
            headers=self._headers,
            method=self._method,
        )
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
            return json.loads(raw)

    def _extract(self, response: dict) -> dict:
        """Map field_map paths to PILAR sensor names."""
        data: dict = {}

        if not self._field_map:
            # No map — try to use response keys directly if they match PILAR names
            _PILAR = {"vibration","temp_palier","debit","pression_entree",
                      "pression_sortie","courant_moteur","temp_moteur","heure_fonctionnement"}
            for k, v in response.items():
                if k in _PILAR:
                    try:
                        data[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            return data

        for path, pilar_key in self._field_map.items():
            val = _get_nested(response, path)
            if val is None:
                logger.debug("Field not found in response: %s", path)
                continue
            try:
                data[pilar_key] = float(val)
            except (ValueError, TypeError):
                logger.warning("Non-numeric value at %s: %s", path, val)

        return data

    def close(self) -> None:
        pass  # stateless HTTP — nothing to close
