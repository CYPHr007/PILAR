# -*- coding: utf-8 -*-
"""
PILAR Embedded — Modbus Adapter
================================
Reads sensor registers from a Modbus device and maps them to PILAR sensor names.

Supports:
  - Modbus/TCP  (Ethernet — most modern PLCs, drives, controllers)
  - Modbus/RTU  (RS-485 serial — older field devices)

Requires: pip install pymodbus>=3.0

Config example
--------------
{
  "adapter":    "modbus",
  "machine_id": "ROBOT-ARM-04",
  "interval":   5,

  "connection": {
    "type":    "tcp",
    "host":    "192.168.1.10",
    "port":    502,
    "unit_id": 1
  },

  "registers": {
    "vibration":             {"address": 100, "type": "holding", "scale": 0.01},
    "temp_palier":           {"address": 101, "type": "holding", "scale": 0.1},
    "debit":                 {"address": 102, "type": "holding", "scale": 0.001},
    "pression_entree":       {"address": 103, "type": "holding", "scale": 0.01},
    "pression_sortie":       {"address": 104, "type": "holding", "scale": 0.01},
    "courant_moteur":        {"address": 105, "type": "holding", "scale": 0.01},
    "temp_moteur":           {"address": 106, "type": "holding", "scale": 0.1},
    "heure_fonctionnement":  {"address": 107, "type": "holding", "scale": 1}
  }
}

For RTU connections, use:
  "connection": {
    "type":     "rtu",
    "port":     "/dev/ttyUSB0",
    "baudrate": 9600,
    "stopbits": 1,
    "bytesize": 8,
    "parity":   "N",
    "unit_id":  1
  }

Register types: "holding" (FC03), "input" (FC04).
Scale: raw_register_value × scale = physical_value.
  e.g. register = 4250 with scale=0.01 → vibration = 42.50 mm/s
"""

from __future__ import annotations
import time
from typing import Optional

from adapters.base import BaseAdapter
from pilar_logging import get_logger

logger = get_logger("pilar.adapter.modbus")


class ModbusAdapter(BaseAdapter):

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = None
        self._registers: dict = config.get("registers", {})
        self._conn_cfg: dict = config.get("connection", {})
        self._unit_id: int = int(self._conn_cfg.get("unit_id", 1))
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient, ModbusSerialClient
        except ImportError:
            raise ImportError(
                "pymodbus is required for the Modbus adapter. "
                "Install it with: pip install pymodbus>=3.0"
            )

        conn_type = self._conn_cfg.get("type", "tcp").lower()

        if conn_type == "tcp":
            host = self._conn_cfg.get("host", "127.0.0.1")
            port = int(self._conn_cfg.get("port", 502))
            self._client = ModbusTcpClient(host, port=port)
            logger.info("Modbus/TCP connecting to %s:%d unit=%d", host, port, self._unit_id)
        elif conn_type == "rtu":
            serial_port = self._conn_cfg.get("port", "/dev/ttyUSB0")
            self._client = ModbusSerialClient(
                port=serial_port,
                baudrate=int(self._conn_cfg.get("baudrate", 9600)),
                stopbits=int(self._conn_cfg.get("stopbits", 1)),
                bytesize=int(self._conn_cfg.get("bytesize", 8)),
                parity=self._conn_cfg.get("parity", "N"),
            )
            logger.info("Modbus/RTU connecting to %s baud=%d unit=%d",
                        serial_port, self._conn_cfg.get("baudrate", 9600), self._unit_id)
        else:
            raise ValueError(f"Unknown Modbus connection type: '{conn_type}'. Use 'tcp' or 'rtu'.")

        if not self._client.connect():
            logger.warning("Modbus connection failed — will retry on next read()")

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> Optional[dict]:
        if self._client is None:
            return None

        if not self._client.is_socket_open():
            logger.warning("Modbus socket closed — reconnecting")
            if not self._client.connect():
                logger.error("Modbus reconnect failed")
                return None

        data: dict = {}

        for sensor_name, reg_cfg in self._registers.items():
            address = int(reg_cfg["address"])
            scale = float(reg_cfg.get("scale", 1.0))
            reg_type = reg_cfg.get("type", "holding").lower()

            try:
                if reg_type == "holding":
                    result = self._client.read_holding_registers(
                        address, count=1, slave=self._unit_id
                    )
                elif reg_type == "input":
                    result = self._client.read_input_registers(
                        address, count=1, slave=self._unit_id
                    )
                else:
                    logger.warning("Unknown register type '%s' for %s — skipping", reg_type, sensor_name)
                    continue

                if result.isError():
                    logger.warning("Modbus read error on %s (addr %d): %s", sensor_name, address, result)
                    continue

                raw = result.registers[0]
                data[sensor_name] = round(raw * scale, 4)

            except Exception as e:
                logger.error("Modbus read exception for %s: %s", sensor_name, e)

        if not data:
            logger.warning("No registers could be read from Modbus device")
            return None

        data["machine_id"] = self.machine_id
        logger.debug("Modbus read: %s", data)
        return data

    def close(self) -> None:
        if self._client:
            self._client.close()
            logger.info("Modbus connection closed")
