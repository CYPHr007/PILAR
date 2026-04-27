"""
PILAR — Combined Multi-Source Training
========================================
Merges three datasets to train a machine-agnostic failure model:

  1. pilar_kaggle_dataset.csv  — 220K real pump sensor rows (fully PILAR-mapped)
  2. factory_sensor_simulator_2040.csv — 500K multi-machine rows, 33 machine types
       Mapped: Vibration_mms→vibration, Temperature_C→temp_palier,
               Operational_Hours→heure_fonctionnement
       Imputed: debit, pression_entree, pression_sortie, courant_moteur,
                temp_moteur → Kaggle medians
  3. synthetic_stages.csv — 69K synthetic rows (all 8 features)

Output: failure_model.pkl | scaler.pkl | isolation_forest.pkl | model_meta.json

Usage:
    py -3.14 train_combined.py
    py -3.14 train_combined.py --no-factory   # Kaggle + synthetic only
    py -3.14 train_combined.py --no-synthetic  # Kaggle + factory only
    py -3.14 train_combined.py --keep-zones    # preserve existing zone_models.pkl
"""

import os
import sys
import pickle
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

FEATURES = [
    'vibration', 'temp_palier', 'debit', 'pression_entree',
    'pression_sortie', 'courant_moteur', 'temp_moteur', 'heure_fonctionnement',
]

KAGGLE_MEDIANS = {
    'vibration':              13.628,
    'temp_palier':            51.649,
    'debit':                   2.456,
    'pression_entree':        48.134,
    'pression_sortie':       632.639,
    'courant_moteur':          2.534,
    'temp_moteur':            44.291,
    'heure_fonctionnement': 110159.5,
}


# ── 1. Load datasets ──────────────────────────────────────────────────────────

def load_kaggle():
    path = os.path.join(DATA_DIR, 'pilar_kaggle_dataset.csv')
    if not os.path.exists(path):
        print("  [SKIP] pilar_kaggle_dataset.csv not found")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={'is_failure': 'failure'})
    df = df[FEATURES + ['failure']].copy()
    df['source'] = 'kaggle'
    n_fail = int(df['failure'].sum())
    print(f"  Kaggle:   {len(df):>7,} rows  |  failures: {n_fail:,}  ({n_fail/len(df)*100:.1f}%)")
    return df


def load_factory(kaggle_medians: dict):
    path = os.path.join(DATA_DIR, 'factory_sensor_simulator_2040.csv')
    if not os.path.exists(path):
        print("  [SKIP] factory_sensor_simulator_2040.csv not found")
        return None
    raw = pd.read_csv(path)

    df = pd.DataFrame()

    # Direct mappings (same physical units / comparable scale)
    df['vibration']            = raw['Vibration_mms'].clip(lower=0)
    df['temp_palier']          = raw['Temperature_C']
    df['heure_fonctionnement'] = raw['Operational_Hours']
    df['failure']              = raw['Failure_Within_7_Days'].astype(int)
    df['source']               = 'factory'

    # Hydraulic pressure available for ~6% of rows (hydraulic machines)
    df['pression_sortie'] = raw['Hydraulic_Pressure_bar'].fillna(
        kaggle_medians['pression_sortie']
    )

    # Remaining sensors: impute with Kaggle medians (same strategy as inference)
    for col in ['debit', 'pression_entree', 'courant_moteur', 'temp_moteur']:
        df[col] = kaggle_medians[col]

    # Drop rows with NaN in mapped columns
    df = df.dropna(subset=['vibration', 'temp_palier', 'heure_fonctionnement'])

    n_fail = int(df['failure'].sum())
    print(f"  Factory:  {len(df):>7,} rows  |  failures: {n_fail:,}  ({n_fail/len(df)*100:.1f}%)")
    print(f"             33 machine types, 3 direct + 5 imputed sensor cols")
    return df


def load_synthetic():
    path = os.path.join(DATA_DIR, 'synthetic_stages.csv')
    if not os.path.exists(path):
        print("  [SKIP] synthetic_stages.csv not found")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={'is_failure': 'failure'})
    df = df[FEATURES + ['failure']].copy()
    df['source'] = 'synthetic'
    n_fail = int(df['failure'].sum())
    print(f"  Synthetic:{len(df):>7,} rows  |  failures: {n_fail:,}  ({n_fail/len(df)*100:.1f}%)")
    return df


# ── 2. Derive features ────────────────────────────────────────────────────────

def add_derived_features(df):
    df = df.copy()
    df['dp']           = df['pression_sortie'] - df['pression_entree']
    df['eta_proxy']    = (df['debit'] * df['dp'].clip(lower=0.001)) / df['courant_moteur'].clip(lower=0.1)
    df['thermal_load'] = df['temp_moteur'] - df['temp_palier']
    df['wear_index']   = np.log1p(df['heure_fonctionnement'] / 10000)
    return df


# ── 3. Train ──────────────────────────────────────────────────────────────────

MAX_TRAIN_ROWS = 200_000   # cap to keep training under ~5 min on CPU


def train(df, features_ext):
    X = df[features_ext].copy()
    y = df['failure'].copy()

    mask = X.notna().all(axis=1) & y.notna()
    if not mask.all():
        dropped = (~mask).sum()
        print(f"  Dropping {dropped:,} rows with NaN values")
        X, y = X[mask], y[mask]

    # Stratified subsample so training stays fast while preserving all sources
    if len(X) > MAX_TRAIN_ROWS:
        idx = (pd.DataFrame({'y': y, 'src': df.loc[X.index, 'source']})
               .groupby('y', group_keys=False)
               .apply(lambda g: g.sample(
                   min(len(g), int(MAX_TRAIN_ROWS * len(g) / len(X))),
                   random_state=RANDOM_STATE))
               .index)
        X, y = X.loc[idx], y.loc[idx]
        print(f"  Subsampled to {len(X):,} rows (stratified, class balance preserved)")

    print(f"\n[3/7] Train/test split (80/20)  —  {len(X):,} rows  {int(y.sum()):,} failures "
          f"({y.mean()*100:.1f}%)", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("[4/7] StandardScaler...", flush=True)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if HAS_SMOTE:
        print("[5/7] SMOTE resampling...", flush=True)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
        vc = pd.Series(y_train_bal).value_counts().to_dict()
        print(f"  After SMOTE: {vc}")
    else:
        print("[5/7] No SMOTE — using class_weight='balanced'", flush=True)
        X_train_bal, y_train_bal = X_train_s, y_train.values

    print("\n[6/7] Training candidate models...", flush=True)
    candidates = {}

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
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
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0, min_child_weight=10, gamma=0.1,
            scale_pos_weight=max(scale_pos, 1),
            random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1, verbosity=0
        )
        xgb.fit(X_train_bal, y_train_bal)
        r = recall_score(y_test, xgb.predict(X_test_s))
        f = f1_score(y_test, xgb.predict(X_test_s))
        p = precision_score(y_test, xgb.predict(X_test_s))
        candidates['XGBoost'] = (xgb, r, f, p)
        print(f"  XGBoost           Recall={r*100:.1f}%  Prec={p*100:.1f}%  F1={f*100:.1f}%")

    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.80, min_samples_leaf=20, random_state=RANDOM_STATE
    )
    gb.fit(X_train_bal, y_train_bal)
    r = recall_score(y_test, gb.predict(X_test_s))
    f = f1_score(y_test, gb.predict(X_test_s))
    p = precision_score(y_test, gb.predict(X_test_s))
    candidates['GradientBoosting'] = (gb, r, f, p)
    print(f"  GradientBoosting  Recall={r*100:.1f}%  Prec={p*100:.1f}%  F1={f*100:.1f}%")

    best_name = max(candidates, key=lambda k: (candidates[k][2], candidates[k][1]))
    best_model, best_recall, best_f1, best_prec = candidates[best_name]
    print(f"\n  => Best: {best_name}  Recall={best_recall*100:.1f}%  Prec={best_prec*100:.1f}%  F1={best_f1*100:.1f}%")
    print("\n" + classification_report(y_test, best_model.predict(X_test_s),
                                       target_names=['Normal', 'Failure']))

    # Platt calibration
    print("  Calibrating probabilities (Platt scaling)...", flush=True)
    from sklearn.linear_model import LogisticRegression
    from pilar_calibrator import PlattModel

    cal_split = int(len(X_test_s) * 0.5)
    X_cal_s, X_eval_s = X_test_s[:cal_split], X_test_s[cal_split:]
    y_cal  = y_test.values[:cal_split]
    y_eval = y_test.values[cal_split:]

    raw_probs = best_model.predict_proba(X_cal_s)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(raw_probs, y_cal)
    cal_model = PlattModel(best_model, platt)

    eval_probs = cal_model.predict_proba(X_eval_s)[:, 1]
    eval_preds = (eval_probs >= 0.5).astype(int)
    eval_recall = recall_score(y_eval, eval_preds)
    eval_prec   = precision_score(y_eval, eval_preds)
    eval_f1     = f1_score(y_eval, eval_preds)
    print(f"  Calibrated eval: Recall={eval_recall*100:.1f}%  Prec={eval_prec*100:.1f}%  F1={eval_f1*100:.1f}%")

    return cal_model, scaler, best_name, eval_recall, eval_prec, eval_f1, X, y, features_ext


def train_isolation_forest(df, scaler, features_ext):
    print("\n[7/7] Isolation Forest on NORMAL samples...", flush=True)
    normal   = df[df['failure'] == 0][features_ext].dropna()
    n_sample = min(len(normal), 50000)
    sample   = normal.sample(n=n_sample, random_state=RANDOM_STATE)
    scaled   = scaler.transform(sample)

    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(scaled)
    print(f"  Fitted on {n_sample:,} normal samples")
    return iso


# ── 4. Save ───────────────────────────────────────────────────────────────────

def save_models(cal_model, scaler, iso, best_name,
                eval_recall, eval_prec, eval_f1,
                X, y, features_ext, source_counts, keep_zones=False):
    print("\nSaving .pkl files...", flush=True)

    with open(os.path.join(SCRIPT_DIR, 'failure_model.pkl'), 'wb') as f:
        pickle.dump(cal_model, f)
    with open(os.path.join(SCRIPT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(SCRIPT_DIR, 'isolation_forest.pkl'), 'wb') as f:
        pickle.dump(iso, f)

    if not keep_zones:
        zpath = os.path.join(SCRIPT_DIR, 'zone_models.pkl')
        if os.path.exists(zpath):
            print("  zone_models.pkl: KEPT (no zone labels in combined data)")

    meta = {
        'model_name':     best_name + '_calibrated',
        'recall':         round(eval_recall * 100, 1),
        'precision':      round(eval_prec   * 100, 1),
        'f1':             round(eval_f1     * 100, 1),
        'n_total':        len(X),
        'n_failures':     int(y.sum()),
        'failure_rate':   round(float(y.mean()) * 100, 1),
        'source':         'combined_kaggle_factory_synthetic',
        'source_counts':  source_counts,
        'features':       features_ext,
        'trained_at':     datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'feature_ranges': {
            feat: [round(float(X[feat].min()), 4), round(float(X[feat].max()), 4)]
            for feat in features_ext
        },
    }
    with open(os.path.join(SCRIPT_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("  failure_model.pkl     OK")
    print("  scaler.pkl            OK")
    print("  isolation_forest.pkl  OK")
    print("  model_meta.json       OK")


def print_config_hints(X, features_ext):
    core = [f for f in features_ext if f in FEATURES]
    print("\n" + "=" * 70)
    print("SUGGESTED config.py UPDATES:")
    print("=" * 70)
    print("\nFEATURE_MEDIANS = {")
    for feat in core:
        print(f"    '{feat}': {round(float(X[feat].median()), 3)},")
    print("}")
    print("\nSENSOR_BOUNDS = {")
    for feat in core:
        lo = round(float(X[feat].quantile(0.001)), 3)
        hi = round(float(X[feat].quantile(0.999)), 3)
        margin = (hi - lo) * 0.1
        print(f"    '{feat}': ({round(lo - margin, 3)}, {round(hi + margin, 3)}),")
    print("}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PILAR combined training')
    parser.add_argument('--no-factory',   action='store_true', help='Skip factory dataset')
    parser.add_argument('--no-synthetic', action='store_true', help='Skip synthetic dataset')
    parser.add_argument('--keep-zones',   action='store_true', help='Keep existing zone_models.pkl')
    args = parser.parse_args()

    print("=" * 70)
    print("PILAR — Combined Multi-Source Training")
    print("=" * 70)

    # -- Load
    print("\n[1/7] Loading datasets...", flush=True)
    parts = []

    df_kaggle = load_kaggle()
    if df_kaggle is not None:
        parts.append(df_kaggle)

    if not args.no_factory:
        df_factory = load_factory(KAGGLE_MEDIANS)
        if df_factory is not None:
            parts.append(df_factory)
    else:
        print("  [SKIP] factory dataset (--no-factory)")

    if not args.no_synthetic:
        df_synth = load_synthetic()
        if df_synth is not None:
            parts.append(df_synth)
    else:
        print("  [SKIP] synthetic dataset (--no-synthetic)")

    if not parts:
        print("ERROR: No datasets loaded. Exiting.")
        sys.exit(1)

    df = pd.concat(parts, ignore_index=True)
    source_counts = df['source'].value_counts().to_dict()

    total_fail = int(df['failure'].sum())
    print(f"\n  Combined: {len(df):,} rows  |  failures: {total_fail:,}  ({total_fail/len(df)*100:.1f}%)")
    print(f"  Sources:  {source_counts}")

    # -- Derive features
    print("\n[2/7] Computing derived features (dp, eta_proxy, thermal_load, wear_index)...", flush=True)
    df = add_derived_features(df)
    features_ext = FEATURES + ['dp', 'eta_proxy', 'thermal_load', 'wear_index']
    print(f"  Total features: {len(features_ext)}  ({features_ext})")

    # -- Train
    cal_model, scaler, best_name, eval_recall, eval_prec, eval_f1, \
        X, y, features_ext = train(df, features_ext)

    iso = train_isolation_forest(df, scaler, features_ext)

    save_models(cal_model, scaler, iso, best_name,
                eval_recall, eval_prec, eval_f1,
                X, y, features_ext, source_counts,
                keep_zones=args.keep_zones)

    print_config_hints(X, features_ext)

    print("\n" + "=" * 70)
    print(f"DONE — Restart app.py to load new models.")
    print("=" * 70)


if __name__ == '__main__':
    main()
