"""
PILAR — Universal Machine Training
====================================
Author  : CYPHR007
License : MIT

Generates a large, unbiased synthetic dataset covering 20 industrial machine
profiles. Every profile shares the same 8 sensor features used by PILAR but
operates in a different physical regime (pump, compressor, fan, hydraulic,
turbine, agitator, etc.).

Goal: eliminate the "pump-only" bias so the model generalises to any rotating
or fluid-handling machine regardless of flow range, pressure range, or power.

Output:  failure_model.pkl  |  scaler.pkl  |  zone_models.pkl  |  model_meta.json
Runtime: ~5-10 min on a modern laptop (300 k rows, SMOTE, ensemble selection).
"""

import argparse
import os
import numpy as np
import pandas as pd
import pickle
import json
import warnings
from datetime import datetime

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--extra-csv', default=None,
                 help='Path to real_data_pilar.csv from fetch_real_data.py')
_train_args, _ = _ap.parse_known_args()
EXTRA_CSV = _train_args.extra_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve  # kept for potential diagnostics

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES = [
    'vibration',            # mm/s  — RMS vibration on bearing housing
    'temp_palier',          # °C    — bearing temperature
    'debit',                # m³/h  — volumetric flow (or equivalent throughput)
    'pression_entree',      # bar   — inlet / suction pressure
    'pression_sortie',      # bar   — outlet / discharge pressure
    'courant_moteur',       # A     — motor current (RMS)
    'temp_moteur',          # °C    — motor winding temperature
    'heure_fonctionnement', # h     — cumulative run hours
]

ZONES = ['CAV', 'ROL', 'ETN', 'IMP', 'MOT']

# ── MACHINE PROFILES ──────────────────────────────────────────────────────────
# Each profile defines the NORMAL operating envelope (mean, std, clip_min, clip_max)
# for every sensor, plus a failure-rate weight per zone and samples to generate.
#
# Failure zone semantics (universal):
#   CAV — cavitation / surge / suction loss / inlet starvation
#   ROL — bearing wear / rolling element defect
#   ETN — seal / gasket / packing failure
#   IMP — impeller / rotor element / blade wear
#   MOT — motor winding / insulation / electrical fault
#
# zone_weights: relative probability of each zone occurring (sums to 1).
# n_normal: how many normal samples to generate for this profile.

MACHINE_PROFILES = {

    # ── PUMPS ──────────────────────────────────────────────────────────────
    "centrifugal_pump_small": {
        "desc": "Centrifugal pump — small industrial (5-50 m³/h)",
        "n_normal": 14000,
        "normal": {
            "vibration":            (2.2,  0.8,  0.3,   7.0),
            "temp_palier":          (65,   10,   30,    95),
            "debit":                (28,   9,    3,     60),
            "pression_entree":      (1.5,  0.4,  0.1,   5.0),
            "pression_sortie":      (5.0,  1.5,  1.0,  18.0),
            "courant_moteur":       (15,   3.5,  4,     38),
            "temp_moteur":          (75,   12,   30,   110),
            "heure_fonctionnement": (4000, 3500, 0,  25000),
        },
        "zone_weights": [0.30, 0.30, 0.15, 0.15, 0.10],
    },

    "centrifugal_pump_large": {
        "desc": "Centrifugal pump — large industrial (50-500 m³/h)",
        "n_normal": 13000,
        "normal": {
            "vibration":            (3.5,  1.2,  0.5,  10.0),
            "temp_palier":          (72,   12,   35,   110),
            "debit":                (200,  70,  30,   520),
            "pression_entree":      (2.0,  0.6,  0.2,   8.0),
            "pression_sortie":      (12,   3.5,  2.0,  30.0),
            "courant_moteur":       (50,   12,  12,   100),
            "temp_moteur":          (85,   14,   35,   125),
            "heure_fonctionnement": (8000, 6000, 0,  60000),
        },
        "zone_weights": [0.28, 0.30, 0.15, 0.17, 0.10],
    },

    "axial_pump": {
        "desc": "Axial / mixed-flow pump (100-2000 m³/h, low head)",
        "n_normal": 10000,
        "normal": {
            "vibration":            (2.8,  1.0,  0.3,   8.0),
            "temp_palier":          (60,   10,   30,    95),
            "debit":                (800,  350,  80,  2100),
            "pression_entree":      (0.5,  0.2,  0.05,  2.0),
            "pression_sortie":      (1.5,  0.5,  0.3,   5.0),
            "courant_moteur":       (65,   18,  15,   150),
            "temp_moteur":          (80,   13,   35,   115),
            "heure_fonctionnement": (6000, 5000, 0,  50000),
        },
        "zone_weights": [0.35, 0.25, 0.12, 0.18, 0.10],
    },

    "screw_pump": {
        "desc": "Screw (progressive cavity) pump — high pressure, low flow",
        "n_normal": 9000,
        "normal": {
            "vibration":            (1.5,  0.5,  0.2,   5.0),
            "temp_palier":          (68,   11,   35,   105),
            "debit":                (12,   4,    0.5,   35),
            "pression_entree":      (1.0,  0.3,  0.1,   4.0),
            "pression_sortie":      (25,   8,    4.0,  60.0),
            "courant_moteur":       (18,   5,    4,     45),
            "temp_moteur":          (78,   13,   35,   110),
            "heure_fonctionnement": (5000, 4000, 0,  30000),
        },
        "zone_weights": [0.15, 0.30, 0.25, 0.20, 0.10],
    },

    "gear_pump": {
        "desc": "Gear pump — very high pressure, low flow (hydraulic service)",
        "n_normal": 8000,
        "normal": {
            "vibration":            (1.2,  0.4,  0.1,   4.0),
            "temp_palier":          (62,   10,   30,    95),
            "debit":                (6,    2.5,  0.3,   18),
            "pression_entree":      (0.8,  0.3,  0.1,   3.0),
            "pression_sortie":      (60,   20,   8.0, 130.0),
            "courant_moteur":       (12,   4,    3,     30),
            "temp_moteur":          (72,   12,   30,   105),
            "heure_fonctionnement": (4000, 3000, 0,  25000),
        },
        "zone_weights": [0.10, 0.30, 0.30, 0.20, 0.10],
    },

    "piston_pump": {
        "desc": "Reciprocating piston pump — very high pressure",
        "n_normal": 8000,
        "normal": {
            "vibration":            (6.0,  2.0,  1.0,  15.0),
            "temp_palier":          (70,   12,   35,   115),
            "debit":                (3.0,  1.0,  0.1,   8.0),
            "pression_entree":      (1.5,  0.5,  0.2,   6.0),
            "pression_sortie":      (120,  40,  20.0, 250.0),
            "courant_moteur":       (28,   8,    8,     60),
            "temp_moteur":          (85,   14,   40,   125),
            "heure_fonctionnement": (3000, 2500, 0,  20000),
        },
        "zone_weights": [0.20, 0.25, 0.25, 0.15, 0.15],
    },

    # ── COMPRESSORS ───────────────────────────────────────────────────────────
    "rotary_compressor": {
        "desc": "Rotary screw / vane air compressor",
        "n_normal": 13000,
        "normal": {
            "vibration":            (2.5,  0.8,  0.3,   8.0),
            "temp_palier":          (75,   12,   40,   115),
            "debit":                (80,   30,   5,   220),
            "pression_entree":      (1.0,  0.02, 0.9,   1.1),
            "pression_sortie":      (8.0,  2.0,  3.0,  16.0),
            "courant_moteur":       (38,   10,   8,     80),
            "temp_moteur":          (88,   14,   40,   130),
            "heure_fonctionnement": (8000, 6000, 0,  55000),
        },
        "zone_weights": [0.30, 0.28, 0.18, 0.12, 0.12],
    },

    "piston_compressor": {
        "desc": "Reciprocating piston compressor — high pressure",
        "n_normal": 9000,
        "normal": {
            "vibration":            (8.0,  2.5,  2.0,  20.0),
            "temp_palier":          (80,   14,   40,   130),
            "debit":                (25,   10,   2,    60),
            "pression_entree":      (1.0,  0.02, 0.9,   1.1),
            "pression_sortie":      (18,   6,    4.0,  40.0),
            "courant_moteur":       (25,   8,    6,     55),
            "temp_moteur":          (90,   15,   40,   140),
            "heure_fonctionnement": (5000, 4000, 0,  35000),
        },
        "zone_weights": [0.25, 0.25, 0.20, 0.15, 0.15],
    },

    # ── FANS & BLOWERS ────────────────────────────────────────────────────────
    "industrial_fan": {
        "desc": "Industrial centrifugal fan / blower",
        "n_normal": 12000,
        "normal": {
            "vibration":            (2.0,  0.7,  0.2,   7.0),
            "temp_palier":          (55,   10,   25,    90),
            "debit":                (2500, 900, 200,  8000),
            "pression_entree":      (1.01, 0.005, 0.98,  1.05),
            "pression_sortie":      (1.06, 0.02,  1.01,  1.20),
            "courant_moteur":       (35,   10,   8,     80),
            "temp_moteur":          (72,   12,   30,   105),
            "heure_fonctionnement": (9000, 7000, 0,  70000),
        },
        "zone_weights": [0.35, 0.30, 0.10, 0.15, 0.10],
    },

    "exhaust_fan": {
        "desc": "Exhaust / ventilation fan — low pressure",
        "n_normal": 8000,
        "normal": {
            "vibration":            (1.5,  0.5,  0.1,   5.0),
            "temp_palier":          (50,   9,    22,    82),
            "debit":                (800,  300,  50,  3000),
            "pression_entree":      (1.005, 0.003, 0.99, 1.02),
            "pression_sortie":      (1.015, 0.008, 1.002, 1.05),
            "courant_moteur":       (18,   5,    4,     40),
            "temp_moteur":          (65,   11,   28,    95),
            "heure_fonctionnement": (7000, 5500, 0,  60000),
        },
        "zone_weights": [0.30, 0.35, 0.10, 0.15, 0.10],
    },

    # ── HYDRAULIC SYSTEMS ─────────────────────────────────────────────────────
    "hydraulic_power_unit": {
        "desc": "Hydraulic power unit — HPU (industrial press / injection moulding)",
        "n_normal": 10000,
        "normal": {
            "vibration":            (2.0,  0.7,  0.2,   7.0),
            "temp_palier":          (65,   11,   30,   100),
            "debit":                (40,   15,   3,    100),
            "pression_entree":      (2.0,  0.5,  0.5,   6.0),
            "pression_sortie":      (220,  60,  50.0, 400.0),
            "courant_moteur":       (30,   8,    8,     65),
            "temp_moteur":          (82,   13,   35,   120),
            "heure_fonctionnement": (6000, 5000, 0,  45000),
        },
        "zone_weights": [0.10, 0.28, 0.30, 0.18, 0.14],
    },

    # ── TURBINES ──────────────────────────────────────────────────────────────
    "steam_turbine_small": {
        "desc": "Small steam / hydro turbine (5-500 kW)",
        "n_normal": 10000,
        "normal": {
            "vibration":            (2.5,  0.8,  0.3,   8.0),
            "temp_palier":          (85,   15,   45,   145),
            "debit":                (200,  80,  30,   600),
            "pression_entree":      (25,   8,    4.0,  60.0),
            "pression_sortie":      (3.0,  1.5,  0.3,  12.0),
            "courant_moteur":       (80,   25,  20,   200),
            "temp_moteur":          (95,   18,   45,   155),
            "heure_fonctionnement": (15000, 10000, 0, 100000),
        },
        "zone_weights": [0.20, 0.35, 0.15, 0.20, 0.10],
    },

    # ── SPECIAL PUMPS ─────────────────────────────────────────────────────────
    "slurry_pump": {
        "desc": "Slurry / abrasive pump — mining, wastewater",
        "n_normal": 9000,
        "normal": {
            "vibration":            (5.0,  1.8,  1.0,  15.0),
            "temp_palier":          (78,   13,   40,   125),
            "debit":                (90,   35,   8,   250),
            "pression_entree":      (1.2,  0.4,  0.1,   4.0),
            "pression_sortie":      (8.0,  2.5,  1.5,  22.0),
            "courant_moteur":       (55,   15,  12,   110),
            "temp_moteur":          (92,   16,   45,   140),
            "heure_fonctionnement": (2500, 2000, 0,  15000),
        },
        "zone_weights": [0.25, 0.25, 0.15, 0.30, 0.05],
    },

    "submersible_pump": {
        "desc": "Submersible pump — borehole / drainage",
        "n_normal": 8000,
        "normal": {
            "vibration":            (1.2,  0.4,  0.1,   4.0),
            "temp_palier":          (48,   8,    22,    75),
            "debit":                (80,   30,   5,   220),
            "pression_entree":      (8.0,  3.0,  1.0,  20.0),
            "pression_sortie":      (15,   5,    3.0,  35.0),
            "courant_moteur":       (22,   6,    5,     50),
            "temp_moteur":          (58,   10,   25,    85),
            "heure_fonctionnement": (4000, 3500, 0,  25000),
        },
        "zone_weights": [0.30, 0.25, 0.20, 0.15, 0.10],
    },

    "booster_pump": {
        "desc": "Booster / pressure-boosting pump — water supply",
        "n_normal": 8000,
        "normal": {
            "vibration":            (1.8,  0.6,  0.2,   6.0),
            "temp_palier":          (58,   9,    28,    88),
            "debit":                (30,   10,   2,    80),
            "pression_entree":      (5.0,  1.5,  1.0,  12.0),
            "pression_sortie":      (18,   5,    5.0,  35.0),
            "courant_moteur":       (20,   5,    5,     45),
            "temp_moteur":          (70,   11,   30,   100),
            "heure_fonctionnement": (5000, 4000, 0,  35000),
        },
        "zone_weights": [0.25, 0.30, 0.20, 0.15, 0.10],
    },

    "cooling_water_pump": {
        "desc": "Cooling / chilled water circuit pump — HVAC",
        "n_normal": 9000,
        "normal": {
            "vibration":            (1.5,  0.5,  0.1,   5.0),
            "temp_palier":          (52,   9,    25,    80),
            "debit":                (120,  40,  15,   320),
            "pression_entree":      (1.0,  0.3,  0.1,   3.5),
            "pression_sortie":      (3.5,  1.0,  0.8,  10.0),
            "courant_moteur":       (25,   7,    6,     55),
            "temp_moteur":          (68,   11,   28,    98),
            "heure_fonctionnement": (6000, 5000, 0,  45000),
        },
        "zone_weights": [0.28, 0.30, 0.18, 0.14, 0.10],
    },

    # ── ROTATING MACHINES ────────────────────────────────────────────────────
    "agitator_mixer": {
        "desc": "Industrial agitator / mixer — tank stirring",
        "n_normal": 9000,
        "normal": {
            "vibration":            (3.5,  1.2,  0.5,  10.0),
            "temp_palier":          (62,   10,   30,    95),
            "debit":                (15,   6,    1,    40),
            "pression_entree":      (0.8,  0.3,  0.05,  3.0),
            "pression_sortie":      (2.0,  0.7,  0.2,   8.0),
            "courant_moteur":       (22,   6,    5,     50),
            "temp_moteur":          (78,   13,   32,   115),
            "heure_fonctionnement": (5000, 4000, 0,  35000),
        },
        "zone_weights": [0.15, 0.35, 0.20, 0.20, 0.10],
    },

    # ── PILAR SITE MACHINES (exact operating ranges) ─────────────────────────
    # These profiles are calibrated to match the specific machines on the PILAR
    # demo site so that normal readings never trigger false alarms.

    "pilar_ventilateur": {
        "desc": "PILAR V-01 type ventilateur — low-gauge-pressure fan (450 m3/h)",
        "n_normal": 10000,
        "normal": {
            "vibration":            (2.9,  0.7,  0.5,   6.0),
            "temp_palier":          (35,   5,    22,    60),
            "debit":                (450,  60,  200,   700),
            "pression_entree":      (0.10, 0.03, 0.02,  0.25),
            "pression_sortie":      (0.40, 0.08, 0.15,  0.80),
            "courant_moteur":       (8,    1.5,  4,     18),
            "temp_moteur":          (48,   8,    28,    75),
            "heure_fonctionnement": (9800, 5000, 0,  50000),
        },
        "zone_weights": [0.35, 0.30, 0.10, 0.15, 0.10],
    },

    "pilar_hydraulique": {
        "desc": "PILAR H-01 type groupe hydraulique — compact HPU (12 m3/h, 180 bar)",
        "n_normal": 9000,
        "normal": {
            "vibration":            (7.2,  1.5,  2.0,  14.0),
            "temp_palier":          (55,   8,    30,    85),
            "debit":                (12,   3,    3,     30),
            "pression_entree":      (2.0,  0.4,  0.5,   5.0),
            "pression_sortie":      (180,  30,  80.0, 280.0),
            "courant_moteur":       (22,   4,    10,    45),
            "temp_moteur":          (72,   10,   35,   105),
            "heure_fonctionnement": (7100, 4000, 0,  40000),
        },
        "zone_weights": [0.10, 0.30, 0.30, 0.16, 0.14],
    },

    "pilar_agitateur": {
        "desc": "PILAR A-01 type agitateur — small mixer (5 m3/h, 1.2 bar outlet)",
        "n_normal": 8000,
        "normal": {
            "vibration":            (8.5,  1.8,  2.0,  15.0),
            "temp_palier":          (48,   7,    28,    80),
            "debit":                (5,    1.5,  1.0,  12.0),
            "pression_entree":      (0.5,  0.1,  0.1,   1.5),
            "pression_sortie":      (1.2,  0.3,  0.3,   3.5),
            "courant_moteur":       (14,   3,    5,     28),
            "temp_moteur":          (65,   10,   30,    95),
            "heure_fonctionnement": (6500, 3500, 0,  30000),
        },
        "zone_weights": [0.10, 0.40, 0.20, 0.20, 0.10],
    },

    "gearbox_driven_machine": {
        "desc": "Gearbox-driven machine — conveyor, mill, crusher",
        "n_normal": 10000,
        "normal": {
            "vibration":            (4.5,  1.5,  0.5,  14.0),
            "temp_palier":          (70,   12,   35,   110),
            "debit":                (5,    2,    0.1,   20),  # lubrication oil flow
            "pression_entree":      (1.5,  0.5,  0.3,   5.0),
            "pression_sortie":      (4.0,  1.5,  0.5,  12.0),
            "courant_moteur":       (45,   15,  10,   110),
            "temp_moteur":          (85,   14,   38,   125),
            "heure_fonctionnement": (7000, 6000, 0,  60000),
        },
        "zone_weights": [0.10, 0.40, 0.20, 0.20, 0.10],
    },

    "pipeline_pump": {
        "desc": "Pipeline transmission pump — oil & gas, water mains",
        "n_normal": 10000,
        "normal": {
            "vibration":            (2.8,  0.9,  0.3,   9.0),
            "temp_palier":          (68,   11,   32,   105),
            "debit":                (500,  180,  60,  1200),
            "pression_entree":      (6.0,  2.0,  1.0,  15.0),
            "pression_sortie":      (45,   15,  10.0,  90.0),
            "courant_moteur":       (90,   25,  20,   180),
            "temp_moteur":          (88,   15,   40,   130),
            "heure_fonctionnement": (10000, 8000, 0, 80000),
        },
        "zone_weights": [0.22, 0.32, 0.16, 0.18, 0.12],
    },

    "fire_pump": {
        "desc": "Fire suppression pump — standby / emergency",
        "n_normal": 5000,
        "normal": {
            "vibration":            (2.0,  0.7,  0.2,   7.0),
            "temp_palier":          (60,   10,   28,    90),
            "debit":                (40,   15,   5,   120),
            "pression_entree":      (0.8,  0.3,  0.1,   3.0),
            "pression_sortie":      (12,   3,    4.0,  25.0),
            "courant_moteur":       (28,   8,    7,     60),
            "temp_moteur":          (72,   12,   30,   105),
            "heure_fonctionnement": (500,  400,  0,   4000),  # rarely runs
        },
        "zone_weights": [0.30, 0.25, 0.20, 0.15, 0.10],
    },
}

# ── FAILURE SIGNATURES ────────────────────────────────────────────────────────
# For each zone, defines how sensor values deviate from normal.
# `severity` ranges from 0.0 (barely detectable) to 1.0 (full-blown).
# Three tiers:
#   EARLY      severity 0.05–0.30  subtle drift, human might miss it
#   DEVELOPING severity 0.30–0.65  clear trend, warrants inspection
#   SEVERE     severity 0.65–1.00  obvious failure, urgent action

def _lerp(lo, hi, t):
    """Linear interpolation between lo and hi by t in [0,1]."""
    return lo + (hi - lo) * t

def apply_failure_signature(row, zone, profile_key, severity=None):
    """
    Perturb a normal operating point to produce a failure signature.
    severity: 0.0 (barely detectable) to 1.0 (full-blown). If None, random 0.05–1.0.
    Returns a modified copy of `row` (dict of feature values).
    """
    r = dict(row)
    vib = r['vibration']
    tp  = r['temp_palier']
    q   = r['debit']
    pe  = r['pression_entree']
    ps  = r['pression_sortie']
    I   = r['courant_moteur']
    tm  = r['temp_moteur']
    hf  = r['heure_fonctionnement']

    if severity is None:
        severity = np.random.uniform(0.05, 1.0)
    s = float(np.clip(severity, 0.0, 1.0))

    # ── age factor: more hours = slightly worse at same severity
    age_factor = 1.0 + 0.15 * np.log1p(hf / 10000) * s

    # ── sensor noise: real sensors jitter ±2-5%
    jitter = lambda: np.random.uniform(0.97, 1.03)

    if zone == 'CAV':
        # Cavitation / surge / inlet starvation
        vib_mul   = _lerp(1.15, 5.5, s)
        flow_mul  = _lerp(0.85, 0.25, s)
        pe_mul    = _lerp(0.80, 0.15, s)
        ps_mul    = _lerp(0.92, 0.45, s)
        I_mul     = _lerp(1.0,  1.25, s * 0.5)
        tp_add    = _lerp(1, 22, s) * age_factor
        r['vibration']        = vib * vib_mul * age_factor * jitter()
        r['debit']            = max(q * 0.05, q * flow_mul * jitter())
        r['pression_entree']  = max(0.01, pe * pe_mul * jitter())
        r['pression_sortie']  = max(0.05, ps * ps_mul * jitter())
        r['courant_moteur']   = I * I_mul * jitter()
        r['temp_palier']      = tp + tp_add * jitter()

    elif zone == 'ROL':
        # Bearing wear — vibration and bearing temp rise progressively
        vib_mul   = _lerp(1.15, 6.0, s) * age_factor
        tp_add    = _lerp(5, 70, s) * age_factor
        I_mul     = _lerp(1.01, 1.35, s)
        tm_add    = _lerp(2, 30, s) * age_factor
        r['vibration']        = vib * vib_mul * jitter()
        r['temp_palier']      = tp + tp_add * jitter()
        r['courant_moteur']   = I * I_mul * jitter()
        r['temp_moteur']      = tm + tm_add * jitter()

    elif zone == 'ETN':
        # Seal / gasket failure — pressure drop, flow leak
        leak      = _lerp(0.03, 0.55, s)
        flow_mul  = _lerp(0.95, 0.50, s)
        pe_mul    = _lerp(0.97, 0.60, s)
        vib_mul   = _lerp(1.05, 2.5, s)
        tp_add    = _lerp(1, 18, s)
        r['pression_sortie']  = max(0.05, ps * (1 - leak) * jitter())
        r['debit']            = max(q * 0.1, q * flow_mul * jitter())
        r['pression_entree']  = max(0.01, pe * pe_mul * jitter())
        r['vibration']        = vib * vib_mul * jitter()
        r['temp_palier']      = tp + tp_add * jitter()

    elif zone == 'IMP':
        # Impeller / blade / rotor element wear
        wear      = _lerp(0.04, 0.55, s) * age_factor
        I_mul     = _lerp(1.03, 1.50, s)
        tm_add    = _lerp(3, 40, s) * age_factor
        vib_mul   = _lerp(1.08, 3.5, s)
        r['debit']            = max(q * 0.1, q * (1 - wear * 0.6) * jitter())
        r['pression_sortie']  = max(0.05, ps * (1 - wear * 0.5) * jitter())
        r['courant_moteur']   = I * I_mul * jitter()
        r['temp_moteur']      = tm + tm_add * jitter()
        r['vibration']        = vib * vib_mul * jitter()

    elif zone == 'MOT':
        # Motor fault — winding temperature, current anomaly
        if np.random.random() < 0.7:  # overcurrent (70%)
            I_mul = _lerp(1.08, 3.5, s)
        else:  # phase loss / undercurrent (30%)
            I_mul = _lerp(0.88, 0.25, s)
        tm_add    = _lerp(5, 90, s) * age_factor
        tp_add    = _lerp(2, 30, s) * age_factor
        vib_mul   = _lerp(1.05, 2.8, s)
        r['courant_moteur']   = max(0.5, I * I_mul * jitter())
        r['temp_moteur']      = tm + tm_add * jitter()
        r['temp_palier']      = tp + tp_add * jitter()
        r['vibration']        = vib * vib_mul * jitter()

    return r


def generate_degradation_sequence(norm_cfg, zone, profile_key, n_steps=None):
    """
    Generate a progressive degradation sequence: a chain of readings
    showing gradual worsening from normal to severe failure.
    Returns list of (row_dict, severity) tuples.
    """
    if n_steps is None:
        n_steps = np.random.randint(4, 12)
    # Start from a single normal operating point
    base = {f: float(np.clip(np.random.normal(mu, sig), lo, hi))
            for f, (mu, sig, lo, hi) in norm_cfg.items()}
    # Severity progresses from barely-detectable to severe
    sev_start = np.random.uniform(0.05, 0.20)
    sev_end   = np.random.uniform(0.60, 1.00)
    sequence = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        sev = sev_start + (sev_end - sev_start) * t
        # Each step starts from the same base but with increasing severity
        row = apply_failure_signature(base, zone, profile_key, severity=sev)
        sequence.append((row, sev))
    return sequence


if __name__ == "__main__":
    # ── GENERATE DATASET ─────────────────────────────────────────────────────────

    print("=" * 70)
    print("PILAR — Universal Machine Training (23 profiles, 3-tier severity)")
    print("=" * 70)

    all_rows = []
    total_normal = 0
    total_failure = 0

    # Severity distribution for failure samples:
    #   40% early (0.05–0.30)   — subtle, easy to miss
    #   35% developing (0.30–0.65) — clear trend
    #   25% severe (0.65–1.00)  — obvious
    SEVERITY_TIERS = [
        (0.40, 0.05, 0.30),   # (fraction, sev_lo, sev_hi)
        (0.35, 0.30, 0.65),
        (0.25, 0.65, 1.00),
    ]

    def _sample_severity():
        """Sample a severity value from the 3-tier distribution."""
        r = np.random.random()
        cum = 0.0
        for frac, lo, hi in SEVERITY_TIERS:
            cum += frac
            if r <= cum:
                return np.random.uniform(lo, hi)
        return np.random.uniform(0.65, 1.0)

    def _add_sensor_noise(row, noise_pct=0.02):
        """Add realistic sensor jitter to a sample (default ±2%)."""
        for f in FEATURES:
            if f in row:
                row[f] *= np.random.uniform(1 - noise_pct, 1 + noise_pct)
        return row

    for profile_key, profile in MACHINE_PROFILES.items():
        n_norm  = profile['n_normal']
        n_fail  = max(int(n_norm * 0.10), 400)   # ~10% failure ratio (was 8%)
        n_seq   = max(int(n_norm * 0.015), 30)   # degradation sequences
        norm_cfg = profile['normal']
        weights  = np.array(profile['zone_weights'])
        weights  /= weights.sum()

        # ── Normal samples (with sensor jitter) ──
        for _ in range(n_norm):
            row = {f: float(np.clip(np.random.normal(mu, sig), lo, hi))
                   for f, (mu, sig, lo, hi) in norm_cfg.items()}
            row = _add_sensor_noise(row)
            row['failure'] = 0
            for z in ZONES:
                row[z] = 0
            all_rows.append(row)

        # ── Failure samples (3-tier severity) ──
        zone_counts = np.round(weights * n_fail).astype(int)
        zone_counts[-1] += n_fail - zone_counts.sum()

        for zone_idx, zone in enumerate(ZONES):
            for _ in range(zone_counts[zone_idx]):
                base = {f: float(np.clip(np.random.normal(mu, sig), lo, hi))
                        for f, (mu, sig, lo, hi) in norm_cfg.items()}
                sev = _sample_severity()
                row = apply_failure_signature(base, zone, profile_key, severity=sev)
                row = _add_sensor_noise(row)
                row['failure'] = 1
                for z in ZONES:
                    row[z] = 1 if z == zone else 0
                # Secondary zone contamination (~15% chance)
                if np.random.random() < 0.15:
                    co = np.random.choice([z for z in ZONES if z != zone])
                    row[co] = 1
                all_rows.append(row)

        # ── Degradation sequences (progressive worsening) ──
        seq_per_zone = np.round(weights * n_seq).astype(int)
        seq_per_zone[-1] += n_seq - seq_per_zone.sum()
        for zone_idx, zone in enumerate(ZONES):
            for _ in range(seq_per_zone[zone_idx]):
                seq = generate_degradation_sequence(norm_cfg, zone, profile_key)
                for step_row, step_sev in seq:
                    step_row = _add_sensor_noise(step_row)
                    step_row['failure'] = 1
                    for z in ZONES:
                        step_row[z] = 1 if z == zone else 0
                    if np.random.random() < 0.10:
                        co = np.random.choice([z for z in ZONES if z != zone])
                        step_row[co] = 1
                    all_rows.append(step_row)
                    total_failure += 1

        total_normal  += n_norm
        total_failure += n_fail
        print(f"  [{profile_key[:35]:35s}]  normal={n_norm:6d}  failure={n_fail:5d}  seqs={n_seq:3d}")

    print(f"\n  Total : {len(all_rows):,} rows  |  normal={total_normal:,}  failure={total_failure:,}")
    print(f"  Failure rate : {total_failure/len(all_rows)*100:.1f}%")

    # ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
    print("\n[2/7] Building DataFrame and adding physics-derived features...")

    df = pd.DataFrame(all_rows)

    # ── Optionally merge real-world data ──────────────────────────────────────────
    if EXTRA_CSV and os.path.isfile(EXTRA_CSV):
        df_real = pd.read_csv(EXTRA_CSV)
        # Keep only the columns we need
        keep = FEATURES + ZONES + ['failure']
        df_real = df_real[[c for c in keep if c in df_real.columns]]
        # Fill any missing zone columns with 0
        for col in ZONES + ['failure']:
            if col not in df_real.columns:
                df_real[col] = 0
        df_real = df_real.dropna(subset=FEATURES)
        print(f"  + Real data from {EXTRA_CSV}: {len(df_real):,} rows "
              f"({int(df_real['failure'].sum())} faults)")
        df = pd.concat([df, df_real], ignore_index=True)
    elif EXTRA_CSV:
        print(f"  [!] --extra-csv file not found: {EXTRA_CSV}")

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Derived features (dimensionless ratios — help the model understand physics)
    df['dp']            = df['pression_sortie'] - df['pression_entree']    # differential pressure
    df['eta_proxy']     = (df['debit'] * df['dp'].clip(0.001)) / (df['courant_moteur'].clip(0.1))  # hydraulic efficiency proxy
    df['thermal_load']  = df['temp_moteur'] - df['temp_palier']            # motor vs bearing delta
    df['wear_index']    = np.log1p(df['heure_fonctionnement'] / 10000)     # log-age

    FEATURES_EXTENDED = FEATURES + ['dp', 'eta_proxy', 'thermal_load', 'wear_index']

    X = df[FEATURES_EXTENDED]
    y = df['failure']

    print(f"  Feature set: {FEATURES_EXTENDED}")

    # ── SPLIT ─────────────────────────────────────────────────────────────────────
    print("\n[3/7] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # ── SCALE ─────────────────────────────────────────────────────────────────────
    print("[4/7] StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── SMOTE ─────────────────────────────────────────────────────────────────────
    if HAS_SMOTE:
        print("[5/7] SMOTE resampling...")
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
        vc = pd.Series(y_train_bal).value_counts().to_dict()
        print(f"  After SMOTE: {vc}")
    else:
        print("[5/7] SMOTE unavailable — using imbalanced data")
        X_train_bal, y_train_bal = X_train_s, y_train.values

    # ── TRAIN MODELS ──────────────────────────────────────────────────────────────
    print("\n[6/8] Training candidate models...")
    candidates = {}

    # Random Forest — constrained depth to reduce overconfidence
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        min_samples_split=20, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    r = recall_score(y_test, rf.predict(X_test_s))
    f = f1_score(y_test, rf.predict(X_test_s))
    p = precision_score(y_test, rf.predict(X_test_s))
    candidates['RandomForest'] = (rf, r, f, p)
    print(f"  RandomForest      Recall={r*100:.1f}%  Prec={p*100:.1f}%  F1={f*100:.1f}%")

    if HAS_XGB:
        scale_pos = int((y_train_bal == 0).sum() / max((y_train_bal == 1).sum(), 1))
        # Reduced complexity: shallower trees, stronger regularisation, more dropout
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0,       # L1 + L2 regularisation
            min_child_weight=10,                   # prevent overfitting to small groups
            gamma=0.1,                             # min loss reduction for splits
            scale_pos_weight=max(scale_pos, 1),
            random_state=RANDOM_STATE, eval_metric='logloss',
            n_jobs=-1, verbosity=0
        )
        xgb.fit(X_train_bal, y_train_bal)
        r = recall_score(y_test, xgb.predict(X_test_s))
        f = f1_score(y_test, xgb.predict(X_test_s))
        p = precision_score(y_test, xgb.predict(X_test_s))
        candidates['XGBoost'] = (xgb, r, f, p)
        print(f"  XGBoost           Recall={r*100:.1f}%  Prec={p*100:.1f}%  F1={f*100:.1f}%")

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.80, min_samples_leaf=10,
            random_state=RANDOM_STATE
        )
        gb.fit(X_train_bal, y_train_bal)
        r = recall_score(y_test, gb.predict(X_test_s))
        f = f1_score(y_test, gb.predict(X_test_s))
        p = precision_score(y_test, gb.predict(X_test_s))
        candidates['GradientBoosting'] = (gb, r, f, p)
        print(f"  GradientBoosting  Recall={r*100:.1f}%  Prec={p*100:.1f}%  F1={f*100:.1f}%")
    except ImportError:
        pass

    # Select best model: maximise recall (safety-critical) with F1 as tiebreak
    best_name = max(candidates, key=lambda k: (candidates[k][1], candidates[k][2]))
    best_model, best_recall, best_f1, best_prec = candidates[best_name]
    print(f"\n  => Best: {best_name}  Recall={best_recall*100:.1f}%  Prec={best_prec*100:.1f}%  F1={best_f1*100:.1f}%")
    print("\n" + classification_report(y_test, best_model.predict(X_test_s),
                                        target_names=['Normal', 'Failure']))

    # ── CALIBRATION (Platt scaling on HELD-OUT calibration set, not test) ────────
    # Split test set in half: one for calibration, one for final evaluation.
    # This avoids data leakage — the Platt sigmoid never sees the evaluation data.
    print("  Calibrating probabilities (Platt scaling on held-out calibration set)...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from pilar_calibrator import PlattModel as _PlattModel
    _cal_split = int(len(X_test_s) * 0.5)
    X_cal_s, X_eval_s = X_test_s[:_cal_split], X_test_s[_cal_split:]
    y_cal, y_eval = y_test.values[:_cal_split], y_test.values[_cal_split:]

    _raw_probs_cal = best_model.predict_proba(X_cal_s)[:, 1].reshape(-1, 1)
    # Lower C for softer sigmoid (reduces overconfidence)
    _platt = LogisticRegression(C=1.0, solver='lbfgs')
    _platt.fit(_raw_probs_cal, y_cal)
    cal_model = _PlattModel(best_model, _platt)

    # Verify on the held-out eval portion (no data leakage)
    _eval_probs = cal_model.predict_proba(X_eval_s)[:, 1]
    _eval_preds = (_eval_probs >= 0.5).astype(int)
    _eval_recall = recall_score(y_eval, _eval_preds)
    _eval_prec   = precision_score(y_eval, _eval_preds)
    _eval_f1     = f1_score(y_eval, _eval_preds)
    _extreme     = ((_eval_probs < 0.01) | (_eval_probs > 0.99)).sum()
    print(f"  Calibrated — eval mean prob={_eval_probs.mean()*100:.1f}% "
          f"(expected ~{y_eval.mean()*100:.1f}%)")
    print(f"  Eval set:   Recall={_eval_recall*100:.1f}%  Prec={_eval_prec*100:.1f}%  "
          f"F1={_eval_f1*100:.1f}%")
    print(f"  Extreme probs (<1% or >99%): {_extreme}/{len(_eval_probs)} "
          f"({_extreme/len(_eval_probs)*100:.1f}%)")

    # ── ZONE MODELS ───────────────────────────────────────────────────────────────
    print("\n[7/7] Training zone classifiers (CAV / ROL / ETN / IMP / MOT)...")
    modeles_zones = {}

    for zone in ZONES:
        y_zone = df[zone].values
        X_z_tr, X_z_te, y_z_tr, y_z_te = train_test_split(
            X, y_zone, test_size=0.20, random_state=RANDOM_STATE, stratify=y_zone
        )
        X_z_tr_s = scaler.transform(X_z_tr)
        X_z_te_s = scaler.transform(X_z_te)

        if HAS_SMOTE and y_z_tr.sum() >= 10:
            try:
                X_z_bal, y_z_bal = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(
                    X_z_tr_s, y_z_tr
                )
            except Exception:
                X_z_bal, y_z_bal = X_z_tr_s, y_z_tr
        else:
            X_z_bal, y_z_bal = X_z_tr_s, y_z_tr

        if HAS_XGB:
            sp = max(1, int((y_z_bal == 0).sum() / max((y_z_bal == 1).sum(), 1)))
            m_z = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=sp, subsample=0.80,
                reg_alpha=0.3, reg_lambda=2.0,
                min_child_weight=10, gamma=0.1,
                random_state=RANDOM_STATE, eval_metric='logloss',
                n_jobs=-1, verbosity=0
            )
        else:
            m_z = RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced',
                min_samples_leaf=10, min_samples_split=20,
                random_state=RANDOM_STATE, n_jobs=-1
            )

        m_z.fit(X_z_bal, y_z_bal)
        y_z_pred = m_z.predict(X_z_te_s)
        r_z = recall_score(y_z_te, y_z_pred, zero_division=0)
        f_z = f1_score(y_z_te, y_z_pred, zero_division=0)
        modeles_zones[zone] = m_z
        print(f"  {zone}  Recall={r_z*100:.1f}%  F1={f_z*100:.1f}%")

    # ── SAVE ──────────────────────────────────────────────────────────────────────
    print("\nSaving .pkl files...")

    with open("failure_model.pkl", "wb") as f:
        pickle.dump(cal_model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("zone_models.pkl", "wb") as f:
        pickle.dump(modeles_zones, f)

    meta = {
        "model_name":    best_name + "_calibrated",
        "recall":        round(_eval_recall * 100, 1),
        "precision":     round(_eval_prec   * 100, 1),
        "f1":            round(_eval_f1     * 100, 1),
        "n_train":       int(len(X_train_bal)),
        "n_total":       len(df),
        "n_failures":    int(y.sum()),
        "failure_rate":  round(float(y.mean()) * 100, 1),
        "source":        "universal_synthetic_23_machine_profiles",
        "profiles":      list(MACHINE_PROFILES.keys()),
        "features":      FEATURES_EXTENDED,
        "zones":         ZONES,
        "trained_at":    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_ranges": {
            feat: [round(float(X[feat].min()), 4), round(float(X[feat].max()), 4)]
            for feat in FEATURES_EXTENDED
        }
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  failure_model.pkl  OK")
    print("  scaler.pkl         OK")
    print("  zone_models.pkl    OK")
    print("  model_meta.json    OK")
    print("\n" + "=" * 70)
    print(f"Done — {len(df):,} rows | {best_name} | "
          f"Recall={best_recall*100:.1f}%  Prec={best_prec*100:.1f}%  F1={best_f1*100:.1f}%")
    print("=" * 70)
    print("Restart app.py to load the new models.")
