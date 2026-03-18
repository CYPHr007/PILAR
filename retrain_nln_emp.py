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
print('\n[1/5] Scanning CSV files...')
csv_files = sorted(SAMPLE_DIR.rglob('*.csv'))
print(f'  Found {len(csv_files)} CSV files')
if not csv_files:
    print('  ERROR: No CSV files found. Check extraction.')
    sys.exit(1)

# ── 2. LABEL + FEATURE EXTRACTION ────────────────────────────────────────────
print('\n[2/5] Extracting features from time-series windows...')

def _label(path: Path):
    """Return (y_global, y_mot) from file path.
    y_global=0 normal | y_global=1 any fault
    y_mot=1 only for electrical/motor fault types.
    """
    s = str(path).lower().replace('-', '_').replace(' ', '_')
    if any(k in s for k in NORMAL_KEYWORDS):
        return 0, 0
    if any(k in s for k in MOT_KEYWORDS):
        return 1, 1
    return 1, 0  # mechanical or other fault — not MOT


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _kurtosis(arr: np.ndarray) -> float:
    """Fisher kurtosis (excess); values > 3 suggest impulsive bearing defects."""
    n = len(arr)
    if n < 4:
        return 0.0
    mu = arr.mean()
    std = arr.std()
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 4) - 3.0)


def _extract(csv_path: Path) -> dict | None:
    """Read one NLN-EMP window CSV and return a Pilar-compatible feature row."""
    try:
        df = pd.read_csv(csv_path, nrows=MAX_ROWS_PER_FILE, low_memory=False)
    except Exception as e:
        return None

    if len(df) < 100:
        return None

    cols_lower = {c.lower().strip(): c for c in df.columns}

    def _find(*keys):
        for k in keys:
            if k in cols_lower:
                return cols_lower[k]
        return None

    # Detect 3-phase current columns
    cu = _find('current_u_a', 'current_u', 'ia', 'phase_a', 'i_u')
    cv = _find('current_v_a', 'current_v', 'ib', 'phase_b', 'i_v')
    cw = _find('current_w_a', 'current_w', 'ic', 'phase_c', 'i_w')
    current_cols = [c for c in [cu, cv, cw] if c is not None]
    if not current_cols:
        # fallback: any column with 'current' in name
        current_cols = [cols_lower[k] for k in cols_lower if 'current' in k]

    # Detect vibration columns
    vib_cols = [cols_lower[k] for k in cols_lower
                if 'vibration' in k or 'vib_ch' in k or 'accel' in k
                or (k.startswith('vib') and k != 'vibration')]

    row = dict(FEATURE_MEDIANS)  # start from pump medians

    # ── Motor current features ────────────────────────────────────────────────
    if current_cols:
        try:
            I = df[current_cols].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(I) > 50:
                phase_rms = np.array([_rms(I[:, i]) for i in range(I.shape[1])])
                I_rms_avg = float(np.sqrt(np.mean(phase_rms ** 2)))
                row['courant_moteur'] = I_rms_avg

                # Current imbalance (std of per-phase RMS) → motor temp proxy
                # Higher imbalance correlates with winding / rotor bar faults
                if len(phase_rms) >= 3:
                    imbalance = float(phase_rms.std())
                    # +10°C per ampere of imbalance (empirical from NLN-EMP cases)
                    row['temp_moteur'] = FEATURE_MEDIANS['temp_moteur'] + imbalance * 10.0
        except Exception:
            pass

    # ── Vibration features ────────────────────────────────────────────────────
    if vib_cols:
        try:
            V = df[vib_cols].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(V) > 50:
                # RMS across all channels (in g) → convert to mm/s
                vib_rms_g = float(np.sqrt(np.mean(V ** 2)))
                row['vibration'] = vib_rms_g * _G_TO_MMS

                # Mean kurtosis across channels → bearing health indicator
                kurt = float(np.mean([_kurtosis(V[:, i]) for i in range(V.shape[1])]))
                # bearing temp offset: kurtosis > 3 → excess heat from impacts
                bear_offset = max(0.0, (kurt - 3.0) * 1.5)
                row['temp_palier'] = FEATURE_MEDIANS['temp_palier'] + bear_offset
        except Exception:
            pass

    return row


rows, y_mot_all, y_g_all = [], [], []
n_skipped = 0
for i, fp in enumerate(csv_files):
    if (i + 1) % 50 == 0:
        print(f'  ... {i+1}/{len(csv_files)} files processed')
    y_g, y_mot = _label(fp)
    feats = _extract(fp)
    if feats is None:
        n_skipped += 1
        continue
    rows.append(feats)
    y_mot_all.append(y_mot)
    y_g_all.append(y_g)

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
print(f'  Saved zones: {list(modeles_zones.keys())} → {ZONES_PKL}')

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
