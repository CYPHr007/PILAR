# PILAR — Predictive Maintenance for Industrial Machines
# Author  : CYPHR007
# License : MIT — see LICENSE
# Source  : https://github.com/CYPHR007/PILAR

"""
Pilar — Central Configuration
==============================
All tunable constants live here. Edit this file instead of hunting
through 7 000 lines of app code. Grouped by concern.
"""

# ── APP ───────────────────────────────────────────────────────────────────────
APP_VERSION  = '1.5.2'
SESSION_DAYS = 30        # session cookie lifetime (30 days — reasonable for industrial use)
MAX_UPLOAD_MB = 16       # max file upload size in megabytes

# ── SECURITY / RATE LIMITING ──────────────────────────────────────────────────
RATE_WINDOW = 900    # seconds — sliding window for failed login attempts
RATE_MAX    = 10     # max failed logins allowed per window before IP block

# ── ML — SENSOR FEATURES (default profile: hydraulic/pump — retrain for any machine) ──────────
FAILURE_ZONES = {
    'CAV': 'Cavitation',
    'ROL': 'Bearing Failure',
    'ETN': 'Seal Failure',
    'IMP': 'Impeller Wear',
    'MOT': 'Motor Fault',
}

COLONNES = [
    'vibration', 'temp_palier', 'debit',
    'pression_entree', 'pression_sortie',
    'courant_moteur', 'temp_moteur', 'heure_fonctionnement',
]

# Medians from UCI hydraulic test rig + 3-phase motor formula (400 V / 0.92 / 0.85).
# Used to impute missing sensor values during partial analyses.
FEATURE_MEDIANS = {
    'vibration':            13.628,
    'temp_palier':          51.649,
    'debit':                2.456,
    'pression_entree':      48.134,
    'pression_sortie':      632.639,
    'courant_moteur':       2.534,
    'temp_moteur':          44.291,
    'heure_fonctionnement': 110159.5,
}

CORE_FEATURES   = list(FEATURE_MEDIANS.keys())
OPTIONAL_FIELDS = ['temperature_ambiante', 'niveau_huile', 'tension_reseau']

# ── ML — PREDICTION THRESHOLDS ───────────────────────────────────────────────
DEFAULT_THRESHOLD       = 30    # failure probability % to trigger alert (lower = more sensitive = fewer missed failures)
ZONE_ALERT_THRESHOLD    = 30    # minimum zone probability % to include in results

# Physical validation bounds (min, max) — API rejects values outside these ranges.
SENSOR_BOUNDS = {
    'vibration':            (0,      25),
    'temp_palier':          (0,      60),
    'debit':                (0,       3),
    'pression_entree':      (0,      63),
    'pression_sortie':      (0,     880),
    'courant_moteur':       (0,       6),
    'temp_moteur':          (0,      84),
    'heure_fonctionnement': (0,  250000),
}

# ── DOMAIN CORRECTIONS ────────────────────────────────────────────────────────
# RUL multipliers: aggressive fluids shorten life; titanium extends it.
FLUID_RUL_FACTORS = {
    'eau':        1.0,
    'eau_chargee': 0.5,
    'huile':      1.3,
    'acide':      0.35,
    'base':       0.4,
    'autre':      0.8,
}

MATERIAL_RUL_FACTORS = {
    'inox_316':  1.0,
    'fonte':     0.6,
    'titane':    1.8,
    'bronze':    0.9,
    'plastique': 0.7,
    'autre':     0.85,
}

# Zone probability adjustments per fluid type.
# Positive delta = lower effective threshold (fault easier to trigger).
FLUID_ZONE_SENSITIVITY = {
    'acide':       {'ETN': 15, 'ROL': 5},
    'base':        {'ETN': 10},
    'eau_chargee': {'IMP': 15, 'CAV': 10},
    'huile':       {'MOT': -5},
    'eau':   {},
    'autre': {},
}

# Machine sub-types outside the hydraulic training distribution — predictions are indicative only.
NON_CENTRIFUGE_TYPES = {
    'pompe_a_vis', 'pompe_a_engrenage', 'pompe_a_palettes',
    'pompe_a_piston', 'peristaltique',
}

# ── RUL MODEL ─────────────────────────────────────────────────────────────────
# Converts NASA C-MAPSS degradation cycles to machine operating hours.
# Scale = machine MTBF (5 000 h) / C-MAPSS max cycles (361).
RUL_SCALE_FACTOR = 5000.0 / 361.0   # ~13.85 h/cycle

# ── AUTO-RETRAIN ──────────────────────────────────────────────────────────────
RETRAIN_TRIGGER = 5000  # fire auto-retrain after this many new analyses (high: avoid corrupting universal model)
MIN_TRAIN_ROWS  = 50    # minimum DB rows required to attempt a retrain

# ── ISOLATION FOREST ─────────────────────────────────────────────────────
ISO_CONTAMINATION       = 0.1    # expected anomaly fraction
ISO_MIN_NORMAL_SAMPLES  = 30     # minimum normal samples before first training
ISO_RETRAIN_INTERVAL    = 50     # retrain every N new normal samples
ISO_MAX_SAMPLES         = 500    # keep last N samples for training

# ── MOT THRESHOLD RULE (fallback when no trained MOT model) ─────────────
MOT_CURRENT_WARN_PCT  = 1.2     # warn at 120% of nominal current
MOT_CURRENT_CRIT_PCT  = 1.5     # critical at 150% of nominal current
MOT_TEMP_WARN         = 85.0    # motor temp warning threshold (C)
MOT_TEMP_RANGE        = 35.0    # temp range for 0-100% scoring above warn
MOT_DEFAULT_NOMINAL_A = 2.5     # default nominal current if machine has none

# ── BATCH API ────────────────────────────────────────────────────────────
BATCH_MAX_READINGS = 100  # max readings per batch API call

# ── ESCALATION ───────────────────────────────────────────────────────────
ESCALATION_DELAY_MIN = 30  # minutes before escalation if alert not acked

# ── AI CHAT ───────────────────────────────────────────────────────────────────
CLAUDE_MODEL      = 'claude-haiku-4-5-20251001'
CLAUDE_MAX_TOKENS = 512
CHAT_DAILY_LIMIT  = 100   # max chat messages per user per day

# Injected into every Claude system prompt — edit here to improve AI advice.
# Sources: UCI hydraulic, NLN-EMP 4TU, ESPset, Sulzer/Xylem/KSB cases,
#          Cutsforth methodology, Chen et al. PMC review.
DOMAIN_KB = (
    "=== MACHINE DOMAIN KNOWLEDGE BASE ===\n"
    "SIGNAL -> FAULT MAPPING (default profile: hydraulic/pump — applies broadly to rotating and industrial machines):\n"
    "- Vibration up + flow down          -> Cavitation (CAV): check NPSH, inlet filter, speed\n"
    "- Vibration up + bearing temp up    -> Bearing wear (ROL): lubrication, alignment, BEP margin\n"
    "- Outlet pressure down for flow     -> Seal leakage (ETN) or impeller wear (IMP)\n"
    "- Motor current up + motor temp up  -> Motor fault (MOT): rotor bar, winding, electrical bearing\n"
    "- 3-phase current imbalance         -> Stator asymmetry or phase loss\n"
    "\n"
    "REAL-WORLD THRESHOLDS:\n"
    "- Vibration > 3.8 mm/s (0.15 ips) = excessive, action required (Xylem metal processing case)\n"
    "- 30-min bearing temp cadence sufficient to catch impending failure (KSB Guard)\n"
    "- RMS current trend + temp combined = earlier MOT detection (Cutsforth / San Jose Water)\n"
    "- Vibration kurtosis > 6 = early bearing defect signature\n"
    "\n"
    "DIAGNOSIS METHODOLOGY:\n"
    "- Combine process sensors (P, Q, T) with CM signals (vibration, current) for best accuracy\n"
    "- Far-from-BEP operation accelerates bearing wear and cavitation\n"
    "- Contextual metadata (fluid type, speed, load) essential for fault signatures\n"
    "\n"
    "TRAINING DATA:\n"
    "- Main model: UCI hydraulic test rig (2205 cycles, centrifugal pump)\n"
    "- RUL: NASA C-MAPSS FD001 mapped to pump domain (~13.85 h/cycle)\n"
    "- MOT zone: proxy-trained; NLN-EMP real current data will improve it\n"
    "\n"
    "CORRECTIVE ACTIONS:\n"
    "- CAV: verify NPSH_a > NPSH_r, clear inlet filter, reduce speed\n"
    "- ROL: vibration spectrum, re-lubricate, check alignment\n"
    "- ETN: inspect mechanical seal, O-rings, stuffing box\n"
    "- IMP: check impeller erosion, measure hydraulic efficiency\n"
    "- MOT: insulation resistance test, winding temp, coupling alignment\n"
    "==================================="
)
