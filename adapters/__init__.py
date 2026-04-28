# -*- coding: utf-8 -*-
"""
PILAR Embedded — Sensor Adapter Layer

Usage:
    from adapters import load_adapter

    adapter = load_adapter("adapters/examples/modbus_tcp.json")
    data = adapter.read()   # returns dict | None
    adapter.close()

Available adapters: modbus, file, http, gpio
"""

from adapters.base import BaseAdapter, load_adapter
from adapters.file_adapter import FileAdapter
from adapters.http_adapter import HttpAdapter

__all__ = [
    "BaseAdapter",
    "load_adapter",
    "FileAdapter",
    "HttpAdapter",
]

# Heavy adapters (optional deps) are importable but not pre-loaded:
#   from adapters.modbus_adapter import ModbusAdapter  — requires pymodbus
#   from adapters.gpio_adapter   import GpioAdapter    — requires spidev (Raspberry Pi only)
