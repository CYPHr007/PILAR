# -*- coding: utf-8 -*-
"""
PILAR Embedded — GPIO Adapter (Raspberry Pi)
=============================================
Reads analog sensor values directly from GPIO pins via an ADC chip.
Designed for setups where PILAR runs on a Raspberry Pi wired directly
to analog sensors (4–20 mA current loops, 0–10 V transducers, thermistors).

Requires: pip install gpiozero spidev  (on Raspberry Pi)
ADC supported: MCP3008 (8-channel, SPI) — most common, cheap, reliable

Wiring (MCP3008 SPI0):
  MCP3008 VDD  → 3.3V (pin 1)
  MCP3008 VREF → 3.3V (pin 1)
  MCP3008 AGND → GND  (pin 6)
  MCP3008 DGND → GND  (pin 6)
  MCP3008 CLK  → SCLK (pin 23)
  MCP3008 DOUT → MISO (pin 21)
  MCP3008 DIN  → MOSI (pin 19)
  MCP3008 CS   → CE0  (pin 24)  ← SPI bus 0 device 0

Config example
--------------
{
  "adapter":    "gpio",
  "machine_id": "PUMP-FLOOR-1",
  "interval":   2,

  "adc": {
    "chip":     "MCP3008",
    "spi_bus":  0,
    "spi_device": 0,
    "vref":     3.3
  },

  "channels": {
    "vibration": {
      "channel":  0,
      "type":     "voltage",
      "v_min":    0.0,
      "v_max":    3.3,
      "val_min":  0.0,
      "val_max":  50.0
    },
    "temp_palier": {
      "channel":  1,
      "type":     "current_loop",
      "ma_min":   4,
      "ma_max":   20,
      "val_min":  0.0,
      "val_max":  150.0,
      "shunt_ohm": 165
    },
    "courant_moteur": {
      "channel":  2,
      "type":     "voltage",
      "v_min":    0.0,
      "v_max":    3.3,
      "val_min":  0.0,
      "val_max":  10.0
    }
  }
}

Channel types
-------------
voltage      — linear mapping from [v_min, v_max] → [val_min, val_max]
current_loop — 4–20 mA industrial standard. The shunt resistor (shunt_ohm)
               converts current to voltage across the ADC input.
               Typical: 165 Ω → 4 mA = 0.66 V, 20 mA = 3.3 V

Notes
-----
- Only channels listed in config["channels"] are read; the rest are skipped.
- This adapter only works on Raspberry Pi with gpiozero + spidev installed.
- On other platforms it raises ImportError with a clear message.
- Run PILAR with --log-level DEBUG to see raw ADC readings for calibration.
"""

from __future__ import annotations
from typing import Optional

from adapters.base import BaseAdapter
from pilar_logging import get_logger

logger = get_logger("pilar.adapter.gpio")


class GpioAdapter(BaseAdapter):

    def __init__(self, config: dict):
        super().__init__(config)
        self._adc_cfg: dict = config.get("adc", {})
        self._channels: dict = config.get("channels", {})
        self._adc = None
        self._mcp = None
        self._init_adc()

    # ── ADC initialisation ────────────────────────────────────────────────────

    def _init_adc(self) -> None:
        chip = self._adc_cfg.get("chip", "MCP3008").upper()
        if chip != "MCP3008":
            raise ValueError(f"Unsupported ADC chip: '{chip}'. Only MCP3008 is supported.")

        try:
            import spidev
        except ImportError:
            raise ImportError(
                "spidev is required for GPIO adapter. "
                "Install it with: pip install spidev  (Raspberry Pi only)"
            )

        spi_bus = int(self._adc_cfg.get("spi_bus", 0))
        spi_device = int(self._adc_cfg.get("spi_device", 0))
        self._vref = float(self._adc_cfg.get("vref", 3.3))

        self._spi = spidev.SpiDev()
        self._spi.open(spi_bus, spi_device)
        self._spi.max_speed_hz = 1_350_000
        logger.info("MCP3008 ADC ready on SPI%d.%d (Vref=%.1fV)", spi_bus, spi_device, self._vref)

    # ── ADC read ──────────────────────────────────────────────────────────────

    def _read_channel(self, channel: int) -> float:
        """Read raw 10-bit value from MCP3008 channel and return voltage."""
        if channel < 0 or channel > 7:
            raise ValueError(f"MCP3008 channel must be 0–7, got {channel}")
        adc_bytes = self._spi.xfer2([1, (8 + channel) << 4, 0])
        raw = ((adc_bytes[1] & 3) << 8) + adc_bytes[2]  # 0–1023
        voltage = (raw / 1023.0) * self._vref
        logger.debug("ADC ch%d raw=%d voltage=%.3fV", channel, raw, voltage)
        return voltage

    # ── Conversion ────────────────────────────────────────────────────────────

    @staticmethod
    def _voltage_to_value(voltage: float, v_min: float, v_max: float,
                           val_min: float, val_max: float) -> float:
        """Linear interpolation: voltage range → physical value range."""
        if v_max == v_min:
            return val_min
        ratio = (voltage - v_min) / (v_max - v_min)
        return round(val_min + ratio * (val_max - val_min), 4)

    @staticmethod
    def _current_loop_to_value(voltage: float, shunt_ohm: float,
                                ma_min: float, ma_max: float,
                                val_min: float, val_max: float) -> float:
        """
        Convert ADC voltage to physical value via 4–20 mA current loop.
        current_mA = (voltage / shunt_ohm) * 1000
        Then linear map [ma_min, ma_max] → [val_min, val_max].
        """
        current_ma = (voltage / shunt_ohm) * 1000.0
        if ma_max == ma_min:
            return val_min
        ratio = (current_ma - ma_min) / (ma_max - ma_min)
        ratio = max(0.0, min(1.0, ratio))  # clamp to valid range
        return round(val_min + ratio * (val_max - val_min), 4)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> Optional[dict]:
        data: dict = {}

        for sensor_name, ch_cfg in self._channels.items():
            channel = int(ch_cfg["channel"])
            ch_type = ch_cfg.get("type", "voltage").lower()

            try:
                voltage = self._read_channel(channel)

                if ch_type == "voltage":
                    val = self._voltage_to_value(
                        voltage,
                        float(ch_cfg.get("v_min", 0.0)),
                        float(ch_cfg.get("v_max", self._vref)),
                        float(ch_cfg["val_min"]),
                        float(ch_cfg["val_max"]),
                    )
                elif ch_type == "current_loop":
                    val = self._current_loop_to_value(
                        voltage,
                        float(ch_cfg.get("shunt_ohm", 165.0)),
                        float(ch_cfg.get("ma_min", 4.0)),
                        float(ch_cfg.get("ma_max", 20.0)),
                        float(ch_cfg["val_min"]),
                        float(ch_cfg["val_max"]),
                    )
                else:
                    logger.warning("Unknown channel type '%s' for %s", ch_type, sensor_name)
                    continue

                data[sensor_name] = val

            except Exception as e:
                logger.error("GPIO read error on channel %d (%s): %s", channel, sensor_name, e)

        if not data:
            return None

        data["machine_id"] = self.machine_id
        return data

    def close(self) -> None:
        if hasattr(self, "_spi") and self._spi:
            self._spi.close()
            logger.info("SPI/ADC closed")
