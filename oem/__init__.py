# -*- coding: utf-8 -*-
"""
PILAR Embedded — OEM Configuration & Certification Layer

Usage:
    from oem import MachineConfig

    mc = MachineConfig.from_file("oem/examples/robot_arm.json")
    context = mc.as_machine_context()   # pass to predict_risk()
    labels  = mc.enabled_zones()        # {zone_code: label}

Certification (CLI):
    python -m oem.certify \\
        --data  machine_data.csv \\
        --config oem/examples/robot_arm.json \\
        --out   ./certified/robot_arm/
"""

from oem.machine_config import MachineConfig

__all__ = ["MachineConfig"]
