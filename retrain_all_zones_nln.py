#!/usr/bin/env python3
"""
retrain_all_zones_nln.py — Retrain ROL, IMP, MOT zones on full NLN-EMP data
=============================================================================
Reads nln_emp_features.csv (output of build_nln_features.py) and retrains
the ROL, IMP, and MOT zone classifiers with real physical fault labels.

CAV and ETN zones remain unchanged (no NLN-EMP fault type maps to them).
MECH class (loose foot, soft foot) is excluded from zone training — these
are structural/mounting faults, not a Pilar zone.

Usage:
    python retrain_all_zones_nln.py [features_csv] [zones_pkl]
"""
import sys, pickle, json, warnings, datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

FEATURES_CSV = sys.argv[1] if len(sys.argv) > 1 else 'nln_emp_features.csv'
ZONES_PKL    = sys.argv[2] if len(sys.argv) > 2 else 'modeles_zones.pkl'
SCALER_PKL   = 'scaler.pkl'

COLONNES = [
    'vibration','temp_palier','debit','pression_entree','pression_sortie',
    'courant_moteur','temp_moteur','heure_fonctionnement'
]

ZONES_TO_TRAIN = ['MOT', 'ROL', 'IMP']

# Each zone's positive label in the 'zone' column of the CSV
ZONE_POSITIVE = {
    'MOT': ['MOT'],
    'ROL': ['ROL'],
    'IMP': ['IMP'],
}
# Samples included as negative examples for each zone
# MECH excluded from all — not a real Pilar zone
ZONE_NEGATIVE = {
    'MOT': ['NORMAL', 'ROL', 'IMP'],   # exclude MECH
    'ROL': ['NORMAL', 'MOT', 'IMP'],
    'IMP': ['NORMAL', 'MOT', 'ROL'],
}

print('=' * 65)
print('PILAR — NLN-EMP Full Zone Retrain (ROL, IMP, MOT)')
print(f'Features : {FEATURES_CSV}')
print(f'Zones pkl: {ZONES_PKL}')
print('=' * 65)

# ── Load features ─────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
# Drop helper columns not in the 8-feature inference pipeline
df = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')
print(f'\nLoaded {len(df)} samples from {FEATURES_CSV}')
print(f'Zone distribution:\n{df["zone"].value_counts().to_string()}')

# Show temp_moteur separation per zone (key MOT indicator)
print('\ntemp_moteur by zone (neg_seq proxy):')
print(df.groupby('zone')['temp_moteur'].agg(['mean','std','min','max']).round(2).to_string())

# ── Load scaler ───────────────────────────────────────────────────────────────
import pickle as pkl
try:
    with open(SCALER_PKL, 'rb') as f:
        scaler = pkl.load(f)
    print(f'\nUsing existing {SCALER_PKL}')
except FileNotFoundError:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[COLONNES].values)
    print(f'\nFitted new scaler on NLN-EMP data')

# ── Load existing zone models ──────────────────────────────────────────────────
try:
    with open(ZONES_PKL, 'rb') as f:
        modeles_zones = pkl.load(f)
    print(f'Loaded existing zones: {list(modeles_zones.keys())}')
except FileNotFoundError:
    modeles_zones = {}

# ── Train each zone ────────────────────────────────────────────────────────────
results = {}

for zone in ZONES_TO_TRAIN:
    print(f'\n{"-"*55}')
    print(f'Training {zone} classifier...')

    pos_mask = df['zone'].isin(ZONE_POSITIVE[zone])
    neg_mask = df['zone'].isin(ZONE_NEGATIVE[zone])
    sub = df[pos_mask | neg_mask].copy()
    sub['y'] = pos_mask[pos_mask | neg_mask].astype(int)

    n_pos = int(sub['y'].sum())
    n_neg = int((sub['y'] == 0).sum())
    print(f'  Samples: {n_pos} positive, {n_neg} negative, total={len(sub)}')

    if n_pos < 5:
        print(f'  SKIP: fewer than 5 positive samples for {zone}')
        continue

    X = scaler.transform(sub[COLONNES].values)
    y = sub['y'].values

    # SMOTE if available
    X_tr, y_tr = X, y
    try:
        from imblearn.over_sampling import SMOTE
        if n_pos >= 6:
            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X, y)
            print(f'  SMOTE: {dict(pd.Series(y_tr).value_counts())}')
    except ImportError:
        pass

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_tr, y_tr)

    n_splits = min(5, n_pos)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    cv_scores = cross_val_score(cv_clf, X, y, cv=cv, scoring='recall')
    print(f'  CV recall ({n_splits}-fold): {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%')
    print(classification_report(y, clf.predict(X),
                                 target_names=['Normal/Other', f'{zone} Fault'],
                                 zero_division=0))

    modeles_zones[zone] = clf
    results[zone] = {
        'cv_recall': round(float(cv_scores.mean() * 100), 1),
        'cv_recall_std': round(float(cv_scores.std() * 100), 1),
        'n_pos': n_pos,
        'n_neg': n_neg,
    }

# ── Save ───────────────────────────────────────────────────────────────────────
with open(ZONES_PKL, 'wb') as f:
    pkl.dump(modeles_zones, f)
print(f'\nSaved zones: {list(modeles_zones.keys())} -> {ZONES_PKL}')

# Update model_meta.json
try:
    with open('model_meta.json') as f:
        meta = json.load(f)
except FileNotFoundError:
    meta = {}

for zone, r in results.items():
    meta[f'{zone.lower()}_model']          = f'GradientBoostingClassifier (NLN-EMP)'
    meta[f'{zone.lower()}_cv_recall']      = r['cv_recall']
    meta[f'{zone.lower()}_cv_recall_std']  = r['cv_recall_std']
    meta[f'{zone.lower()}_n_pos']          = r['n_pos']
    meta[f'{zone.lower()}_source']         = 'NLN-EMP 4TU full dataset (26 faults x 3 loads)'
    meta[f'{zone.lower()}_trained_at']     = datetime.datetime.now().isoformat()[:19]

with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print('model_meta.json updated')

print('\n' + '=' * 65)
print('NLN-EMP FULL ZONE RETRAIN COMPLETE')
for zone, r in results.items():
    print(f'  {zone}: CV recall {r["cv_recall"]:.1f}% +/- {r["cv_recall_std"]:.1f}%  (n={r["n_pos"]} fault samples)')
print('  CAV, ETN: unchanged (no NLN-EMP mapping)')
print('  Restart the app to load updated zone models.')
print('=' * 65)
