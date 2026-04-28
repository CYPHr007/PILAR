# -*- coding: utf-8 -*-
"""
PILAR Embedded — OEM Machine Config
=====================================
Loads and validates an OEM machine configuration JSON file.

The machine config lets an OEM describe exactly what "normal" looks like
for their specific machine model — sensor nominals, physical bounds, which
failure zones are relevant, custom zone labels, and alert thresholds.

This config is passed as machine_context to predict_risk(), which uses it to:
  - Impute missing sensors from machine-specific nominals (not global medians)
  - Apply per-machine current threshold for MOT zone
  - Carry machine metadata into the output event

Config format
-------------
See oem/examples/ for full working examples.

Minimum valid config:
  {
    "machine_type": "my_machine",
    "sensors": {
      "vibration": {"nominal": 1.2},
      "courant_moteur": {"nominal": 8.5}
    }
  }

Full config with all options:
  {
    "machine_type":  "industrial_robot_arm",
    "machine_name":  "6-Axis Robot Arm",
    "manufacturer":  "ACME Robotics",
    "model_ref":     "AR-2000",

    "sensors": {
      "vibration":            {"unit": "mm/s", "nominal": 1.2,  "min": 0.0, "max": 20.0},
      "temp_palier":          {"unit": "°C",   "nominal": 45.0, "min": 10,  "max": 120},
      "debit":                {"unit": "L/min","nominal": 8.0,  "min": 0,   "max": 60},
      "pression_entree":      {"unit": "bar",  "nominal": 5.0,  "min": 0,   "max": 20},
      "pression_sortie":      {"unit": "bar",  "nominal": 18.0, "min": 0,   "max": 50},
      "courant_moteur":       {"unit": "A",    "nominal": 8.5,  "min": 0,   "max": 25},
      "temp_moteur":          {"unit": "°C",   "nominal": 65.0, "min": 10,  "max": 150},
      "heure_fonctionnement": {"unit": "h",    "nominal": 0,    "min": 0,   "max": 100000}
    },

    "failure_zones": {
      "CAV": {"enabled": false},
      "ROL": {"enabled": true,  "label": "Joint Bearing Failure"},
      "ETN": {"enabled": true,  "label": "Seal / Gearbox Leak"},
      "IMP": {"enabled": false},
      "MOT": {"enabled": true,  "label": "Servo Motor Fault"}
    },

    "thresholds": {
      "risk_alert":        50,
      "zone_alert":        40,
      "nominal_current_A": 8.5
    },

    "certification": {
      "certified_by":      "PILAR",
      "certified_date":    "2026-04-27",
      "model_version":     "1.3.3",
      "training_samples":  12000,
      "recall":            0.98,
      "precision":         0.97,
      "f1":                0.975
    }
  }
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

# All PILAR sensor feature names
_ALL_SENSORS = {
    "vibration", "temp_palier", "debit",
    "pression_entree", "pression_sortie",
    "courant_moteur", "temp_moteur", "heure_fonctionnement",
}

_ALL_ZONES = {"CAV", "ROL", "ETN", "IMP", "MOT"}

# Default zone labels (from config.py FAILURE_ZONES)
_DEFAULT_ZONE_LABELS = {
    "CAV": "Cavitation",
    "ROL": "Bearing Failure",
    "ETN": "Seal Failure",
    "IMP": "Impeller Wear",
    "MOT": "Motor Fault",
}


class MachineConfig:
    """
    Loaded OEM machine configuration. Provides typed access to all config
    sections and generates the machine_context dict consumed by predict_risk().
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._validate()

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> "MachineConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Machine config not found: {path}")
        with open(p, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "MachineConfig":
        return cls(d)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if "machine_type" not in self._cfg:
            raise ValueError("machine_config: 'machine_type' is required")
        sensors = self._cfg.get("sensors", {})
        unknown = set(sensors.keys()) - _ALL_SENSORS
        if unknown:
            raise ValueError(f"machine_config: unknown sensor(s): {unknown}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def machine_type(self) -> str:
        return self._cfg["machine_type"]

    @property
    def machine_name(self) -> str:
        return self._cfg.get("machine_name", self.machine_type)

    @property
    def manufacturer(self) -> str:
        return self._cfg.get("manufacturer", "")

    @property
    def model_ref(self) -> str:
        return self._cfg.get("model_ref", "")

    # ── Sensor nominals ───────────────────────────────────────────────────────

    def get_nominal(self, sensor: str) -> Optional[float]:
        """Return the nominal (normal operating) value for a sensor, or None."""
        s = self._cfg.get("sensors", {}).get(sensor, {})
        val = s.get("nominal")
        return float(val) if val is not None else None

    def get_sensor_bounds(self, sensor: str) -> Optional[tuple]:
        """Return (min, max) physical bounds for a sensor, or None."""
        s = self._cfg.get("sensors", {}).get(sensor, {})
        if "min" in s and "max" in s:
            return (float(s["min"]), float(s["max"]))
        return None

    # ── Zones ─────────────────────────────────────────────────────────────────

    def zone_enabled(self, zone_code: str) -> bool:
        zones = self._cfg.get("failure_zones", {})
        if zone_code not in zones:
            return True  # not configured = enabled by default
        return bool(zones[zone_code].get("enabled", True))

    def zone_label(self, zone_code: str) -> str:
        zones = self._cfg.get("failure_zones", {})
        if zone_code in zones and "label" in zones[zone_code]:
            return zones[zone_code]["label"]
        return _DEFAULT_ZONE_LABELS.get(zone_code, zone_code)

    def enabled_zones(self) -> dict:
        """Return {zone_code: label} for all enabled zones."""
        return {
            z: self.zone_label(z)
            for z in _ALL_ZONES
            if self.zone_enabled(z)
        }

    # ── Thresholds ────────────────────────────────────────────────────────────

    @property
    def risk_threshold(self) -> Optional[float]:
        return self._cfg.get("thresholds", {}).get("risk_alert")

    @property
    def zone_threshold(self) -> Optional[float]:
        return self._cfg.get("thresholds", {}).get("zone_alert")

    @property
    def nominal_current(self) -> Optional[float]:
        # Look in thresholds first, then sensors
        t = self._cfg.get("thresholds", {}).get("nominal_current_A")
        if t is not None:
            return float(t)
        return self.get_nominal("courant_moteur")

    # ── machine_context for predict_risk() ───────────────────────────────────

    def as_machine_context(self) -> dict:
        """
        Return a dict compatible with predict_risk(machine_context=...).
        Passes machine-specific nominals so missing sensors are imputed
        from this machine's baseline rather than the global medians.
        """
        ctx: dict = {"machine_type": self.machine_type}

        if self.get_nominal("debit") is not None:
            ctx["nominal_flow"] = self.get_nominal("debit")
        if self.get_nominal("pression_sortie") is not None:
            ctx["nominal_pressure"] = self.get_nominal("pression_sortie")
        if self.nominal_current is not None:
            ctx["nominal_current"] = self.nominal_current
        if self.get_nominal("vibration") is not None:
            ctx["nominal_vibration"] = self.get_nominal("vibration")

        return ctx

    # ── Certification info ────────────────────────────────────────────────────

    @property
    def is_certified(self) -> bool:
        return "certification" in self._cfg

    @property
    def certification(self) -> dict:
        return self._cfg.get("certification", {})

    def certification_summary(self) -> str:
        if not self.is_certified:
            return "Not certified — using default PILAR model"
        c = self.certification
        return (
            f"Certified by {c.get('certified_by','PILAR')} "
            f"on {c.get('certified_date','?')} — "
            f"Recall {c.get('recall',0)*100:.1f}% "
            f"F1 {c.get('f1',0)*100:.1f}% "
            f"({c.get('training_samples',0):,} samples)"
        )

    # ── Serialise ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return dict(self._cfg)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cfg, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"MachineConfig(type={self.machine_type!r}, "
            f"name={self.machine_name!r}, certified={self.is_certified})"
        )
