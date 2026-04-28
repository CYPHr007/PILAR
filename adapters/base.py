# -*- coding: utf-8 -*-
"""
PILAR Embedded — BaseAdapter
============================
All sensor adapters implement this interface.

An adapter is responsible for one thing: reading one sensor sample from a
data source and returning it as a plain dict with PILAR feature names as keys.

  read()  → dict | None    (None = no new data yet)
  close() → None           (clean up connections)

Config files are JSON. Each adapter type defines its own required fields
under a shared envelope:

  {
    "adapter":    "modbus" | "file" | "http" | "gpio",
    "machine_id": "ROBOT-01",          # optional — overridden by CLI --machine-id
    "interval":   5,                   # polling interval in seconds (adapters that need it)
    ...adapter-specific fields...
  }

Usage from pilar_embedded.py:

  from adapters import load_adapter
  adapter = load_adapter("config.json")
  data = adapter.read()
  if data:
      result = run_inference(data)
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseAdapter(ABC):
    """Abstract base for all PILAR sensor adapters."""

    def __init__(self, config: dict):
        self.config = config
        self.machine_id: str = config.get("machine_id", "MACHINE-01")
        self.interval: float = float(config.get("interval", 5.0))

    @abstractmethod
    def read(self) -> Optional[dict]:
        """
        Read one sensor sample.
        Returns a dict with PILAR sensor keys, or None if no new data is available.
        Sensor keys (all optional — missing ones are imputed from medians):
            vibration, temp_palier, debit, pression_entree, pression_sortie,
            courant_moteur, temp_moteur, heure_fonctionnement
        """

    def close(self) -> None:
        """Release any held connections or file handles."""


def load_adapter(config_path: str) -> BaseAdapter:
    """
    Instantiate the correct adapter from a JSON config file.

    Example config files are in adapters/examples/.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    adapter_type = config.get("adapter", "").lower()

    if adapter_type == "modbus":
        from adapters.modbus_adapter import ModbusAdapter
        return ModbusAdapter(config)
    elif adapter_type == "file":
        from adapters.file_adapter import FileAdapter
        return FileAdapter(config)
    elif adapter_type == "http":
        from adapters.http_adapter import HttpAdapter
        return HttpAdapter(config)
    elif adapter_type == "gpio":
        from adapters.gpio_adapter import GpioAdapter
        return GpioAdapter(config)
    else:
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            "Supported: modbus, file, http, gpio"
        )
