#!/usr/bin/env python3
"""
Pilar — NLN-EMP MOT Zone Retraining
====================================
Trains a motor-current + vibration based MOT (Motor Fault) zone classifier
using the NLN-EMP dataset (4TU, CC0): 20 fault types, 2 centrifugal pumps,
3-phase current (20 kHz) + 5 vibration channels.

Workflow:
  1. Download the NLN-EMP archive (~20.8 GB) from:
       https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0/4
  2. Extract a working sample:
       python extract_nln_emp_sample.py archive.7z ./nln_sample
  3. Run this script:
       python retrain_nln_emp.py ./nln_sample

The script replaces the MOT entry in modeles_zones.pkl with a classifier
trained on real pump motor-current data, then updates model_meta.json.
"""
import sys, os, pickle, json, warnings, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, recall_score

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./nln_sample')
ZONES_PKL  = sys.argv[2] if len(sys.argv) > 2 else 'modeles_zones.pkl'
SCALER_PKL = 'scaler.pkl'

# Max rows to read per CSV window (20 kHz × 15 s = 300 000 rows; 50 k is plenty for RMS)
MAX_ROWS_PER_FILE = 50_000

# Pilar feature medians — used to fill unknown pump channels
FEATURE_MEDIANS = {
    'vibration': 0.61, 'temp_palier': 44.837, 'debit': 0.395,
    'pression_entree': 1.987, 'pression_sortie': 107.73,
    'courant_moteur': 4.579, 'temp_moteur': 58.043, 'heure_fonctionnement': 1103.0,
}
COLONNES = list(FEATURE_MEDIANS.keys())

# g → mm/s conversion at 50 Hz (v = a / (2π·f) × 1000)
_G_TO_MMS = 9810.0 / (2.0 * np.pi * 50.0)  # ≈ 31.2 mm/s per g-rms

# Label rules from folder/file path keywords
MOT_KEYWORDS    = {
    'rotor', 'stator', 'winding', 'electrical_bearing', 'electric_bearing',
    'broken_rotor', 'brkn_rotor', 'rotor_bar', 'eccentricity', 'magnetism',
    'motor_bearing', 'rotor_unbalance',
}
NORMAL_KEYWORDS = {
    'normal', 'healthy', 'baseline', 'no_fault', 'nofault', 'no-fault',
    'reference', 'good',
}

print('=' * 65)
print('PILAR — NLN-EMP MOT Zone Retraining')
print(f'Sample dir  : {SAMPLE_DIR}')
print(f'Zones pkl   : {ZONES_PKL}')
print('=' * 65)

if not SAMPLE_DIR.exists():
    print(f'\nERROR: {SAMPLE_DIR} not found.')
    print('Run extract_nln_emp_sample.py first, then pass the output dir as argv[1].')
    sys.exit(1)

# ── 1. SCAN ───────────────────────────────────────────────────────────────────
print('\n[1/5] Scanning fault-class folders...')

# NLN-EMP actual CSV format:
#   columns = ["time", "0", "1", ..., "35"]
#   36 numeric columns = 36 independent measurement runs (repetitions)
#   Each row = one time step at 20 kHz
#   Each sub-folder contains ch1.csv (phase A), ch2.csv (phase B), ch3.csv (phase C)
#   for Electric (current) measurements.

def _label(folder: Path):
    """Return (y_global, y_mot) from folder name.
    y_global=0 normal | y_global=1 any fault
    y_mot=1 only for electrical/motor fault types.
    """
    s = folder.name.lower().replace('-', '_').replace(' ', '_')
    if any(k in s for k in NORMAL_KEYWORDS):
        return 0, 0
    if any(k in s for k in MOT_KEYWORDS):
        return 1, 1
    return 1, 0  # mechanical or other fault — not MOT


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _kurtosis(arr: np.ndarray) -> float:
    """Fisher kurtosis (excess); values > 3 suggest impulsive bearing defects."""
    if len(arr) < 4:
        return 0.0
    mu, std = arr.mean(), arr.std()
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 4) - 3.0)


def _load_runs(csv_path: Path, nrows: int = MAX_ROWS_PER_FILE) -> np.ndarray | None:
    """Load NLN-EMP CSV and return a (nrows × n_runs) array of numeric run columns."""
    try:
        df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    except Exception:
        return None
    if len(df) < 100:
        return None
    # Keep only purely numeric columns (the run indices "0".."35")
    run_cols = [c for c in df.columns if str(c).strip().lstrip('-').isdigit()]
    if not run_cols:
        return None
    return df[run_cols].apply(pd.to_numeric, errors='coerce').dropna().values


def _extract_folder(folder: Path) -> list[dict]:
    """
    Given a fault-class folder containing ch1/ch2/ch3.csv files,
    return one feature dict per run column (up to 36 samples).
    ch1 = phase A, ch2 = phase B, ch3 = phase C (3-phase motor current).
    """
    # Electric files: Electric_*-ch1/ch2/ch3.csv = 3-phase motor current (Amperes)
    # Vibration files: Vibration_*-ch1/ch2.csv = accelerometers (g units)
    def _find_elec(n):
        matches = list(folder.glob(f'Electric_*-ch{n}.csv'))
        return matches[0] if matches else None

    def _find_vib(n):
        matches = list(folder.glob(f'Vibration_*-ch{n}.csv'))
        return matches[0] if matches else None

    elec1, elec2, elec3 = _find_elec(1), _find_elec(2), _find_elec(3)
    vib1, vib2 = _find_vib(1), _find_vib(2)

    if elec1 is None:
        return []

    A = _load_runs(elec1)
    B = _load_runs(elec2) if elec2 else None
    C = _load_runs(elec3) if elec3 else None
    V1 = _load_runs(vib1) if vib1 else None
    V2 = _load_runs(vib2) if vib2 else None

    if A is None:
        return []

    n_runs = A.shape[1]
    samples = []

    for run_idx in range(n_runs):
        row = dict(FEATURE_MEDIANS)

        # 3-phase current features
        I_rms_A = _rms(A[:, run_idx])
        phase_rms = [I_rms_A]
        if B is not None and run_idx < B.shape[1]:
            phase_rms.append(_rms(B[:, run_idx]))
        if C is not None and run_idx < C.shape[1]:
            phase_rms.append(_rms(C[:, run_idx]))

        phase_rms = np.array(phase_rms)
        I_rms_avg = float(np.sqrt(np.mean(phase_rms ** 2)))
        row['courant_moteur'] = I_rms_avg

        # Current imbalance → motor temperature proxy
        if len(phase_rms) >= 2:
            imbalance = float(phase_rms.std())
            row['temp_moteur'] = FEATURE_MEDIANS['temp_moteur'] + imbalance * 10.0

        # Kurtosis of phase A current → impulsive bearing signature
        kurt_I = _kurtosis(A[:, run_idx])
        bear_offset = max(0.0, (kurt_I - 3.0) * 1.5)
        row['temp_palier'] = FEATURE_MEDIANS['temp_palier'] + bear_offset

        # Vibration from Vibration\ folder (accelerometers in g → mm/s)
        vib_runs = []
        for V in [V1, V2]:
            if V is not None and run_idx < V.shape[1]:
                vib_runs.append(V[:, run_idx])
        if vib_runs:
            all_vib = np.concatenate(vib_runs)
            vib_rms_g = _rms(all_vib)
            row['vibration'] = vib_rms_g * _G_TO_MMS  # g → mm/s
            kurt_vib = float(np.mean([_kurtosis(v) for v in vib_runs]))
            bear_vib_offset = max(0.0, (kurt_vib - 3.0) * 1.5)
            row['temp_palier'] = max(row['temp_palier'],
                                     FEATURE_MEDIANS['temp_palier'] + bear_vib_offset)

        samples.append(row)

    return samples


# Group files by parent folder (each folder = one fault class)
fault_folders = sorted({f.parent for f in SAMPLE_DIR.rglob('*.csv')})
print(f'  Found {len(fault_folders)} fault-class folders:')
for ff in fault_folders:
    print(f'    {ff.name}')

# ── 2. LABEL + FEATURE EXTRACTION ────────────────────────────────────────────
print('\n[2/5] Extracting features from run columns...')

rows, y_mot_all, y_g_all = [], [], []
n_skipped = 0
for folder in fault_folders:
    y_g, y_mot = _label(folder)
    samples = _extract_folder(folder)
    if not samples:
        print(f'  SKIP (no usable data): {folder.name}')
        n_skipped += 1
        continue
    rows.extend(samples)
    y_mot_all.extend([y_mot] * len(samples))
    y_g_all.extend([y_g] * len(samples))
    print(f'  {folder.name}: {len(samples)} runs, label=(y_g={y_g}, y_mot={y_mot})')

if not rows:
    print('  ERROR: Could not extract features from any file.')
    sys.exit(1)

X_df   = pd.DataFrame(rows, columns=COLONNES)
y_mot  = np.array(y_mot_all, dtype=int)
y_g    = np.array(y_g_all,   dtype=int)

n_normal   = int((y_mot == 0).sum())
n_mot      = int((y_mot == 1).sum())
n_other    = int((y_g   == 1).sum()) - n_mot

print(f'\n  Files processed : {len(rows)}  (skipped: {n_skipped})')
print(f'  Normal windows  : {n_normal}')
print(f'  MOT fault       : {n_mot}')
print(f'  Other faults    : {n_other}')
print(f'\n  Feature summary:')
for c in COLONNES:
    print(f'    {c:30s}: [{X_df[c].min():.3f} — {X_df[c].max():.3f}]  median={X_df[c].median():.3f}')

if n_mot < 6:
    print('\n  WARNING: fewer than 6 MOT fault windows — not enough to train.')
    print('  Ensure MOT fault folders (rotor/stator/winding) were included in extract_nln_emp_sample.py.')
    sys.exit(1)

# ── 3. SCALING ────────────────────────────────────────────────────────────────
print('\n[3/5] Scaling...')
X = X_df[COLONNES].values
try:
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    X_sc = scaler.transform(X)
    print(f'  Used existing {SCALER_PKL} (keeps alignment with main Pilar model)')
except FileNotFoundError:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    print(f'  {SCALER_PKL} not found — fitted new scaler on NLN-EMP data')

# ── 4. TRAIN MOT ZONE CLASSIFIER ─────────────────────────────────────────────
print('\n[4/5] Training MOT zone classifier (GradientBoosting)...')

# SMOTE if available
X_tr, y_tr = X_sc, y_mot
try:
    from imblearn.over_sampling import SMOTE
    if y_mot.sum() >= 6:
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_sc, y_mot)
        print(f'  SMOTE applied: {dict(pd.Series(y_tr).value_counts())}')
except ImportError:
    pass

mot_clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42,
)
mot_clf.fit(X_tr, y_tr)

# Cross-validate on original (pre-SMOTE) to get unbiased recall estimate
n_splits = min(5, int(y_mot.sum()))
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    X_sc, y_mot, cv=cv, scoring='recall',
)
print(f'  CV recall ({n_splits}-fold): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%')
print(classification_report(y_mot, mot_clf.predict(X_sc),
                             target_names=['Normal/Other', 'MOT Fault'], zero_division=0))

# ── 5. UPDATE modeles_zones.pkl ───────────────────────────────────────────────
print('\n[5/5] Updating modeles_zones.pkl...')
try:
    with open(ZONES_PKL, 'rb') as f:
        modeles_zones = pickle.load(f)
    print(f'  Loaded existing zones: {list(modeles_zones.keys())}')
except FileNotFoundError:
    modeles_zones = {}
    print('  No existing file — creating new.')

modeles_zones['MOT'] = mot_clf
with open(ZONES_PKL, 'wb') as f:
    pickle.dump(modeles_zones, f)
print(f'  Saved zones: {list(modeles_zones.keys())} -> {ZONES_PKL}')

# Update model_meta.json
nln_meta = {
    'mot_model':          'GradientBoostingClassifier (NLN-EMP)',
    'mot_cv_recall':      round(float(cv_scores.mean() * 100), 1),
    'mot_cv_recall_std':  round(float(cv_scores.std()  * 100), 1),
    'mot_n_windows':      int(len(X)),
    'mot_n_faults':       int(y_mot.sum()),
    'mot_source':         'NLN-EMP 4TU — Motor Current and Vibration Monitoring Dataset',
    'mot_trained_at':     datetime.datetime.now().isoformat()[:19],
}
try:
    with open('model_meta.json') as f:
        meta = json.load(f)
except FileNotFoundError:
    meta = {}
meta.update(nln_meta)
with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print('  model_meta.json updated')

print('\n' + '=' * 65)
print('NLN-EMP MOT ZONE RETRAINING COMPLETE')
print(f'  MOT CV recall : {cv_scores.mean()*100:.1f}%')
print('  Restart the app to load the updated MOT zone model.')
print('=' * 65)
