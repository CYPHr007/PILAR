"""
PILAR — Kaggle Real-Data Training
===================================
Trains production models from the Kaggle pump sensor dataset (real labeled data).
Generates: failure_model.pkl | scaler.pkl | zone_models.pkl | isolation_forest.pkl | model_meta.json

The Kaggle dataset has real NORMAL/BROKEN/RECOVERING labels but NO zone labels.
Strategy:
  - failure_model.pkl  ← trained on real Kaggle data (binary: failure yes/no)
  - zone_models.pkl    ← kept from synthetic (no zone labels in Kaggle)
  - isolation_forest.pkl ← fitted on real NORMAL samples
  - scaler.pkl         ← fitted on real data distribution

Usage:
  py -3.14 train_kaggle.py
  py -3.14 train_kaggle.py --keep-zones   # preserve existing zone_models.pkl
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest,
)

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

PILAR_MAP = {
    'vibration':       'sensor_06',
    'temp_palier':     'sensor_02',
    'debit':           'sensor_00',
    'pression_entree': 'sensor_01',
    'pression_sortie': 'sensor_04',
    'courant_moteur':  'sensor_18',
    'temp_moteur':     'sensor_10',
}

FEATURES = [
    'vibration', 'temp_palier', 'debit', 'pression_entree',
    'pression_sortie', 'courant_moteur', 'temp_moteur', 'heure_fonctionnement',
]

ZONES = ['CAV', 'ROL', 'ETN', 'IMP', 'MOT']

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sensor.csv')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_prepare():
    print("=" * 70)
    print("PILAR — Kaggle Real-Data Training")
    print("=" * 70)

    print("\n[1/7] Loading Kaggle sensor data...", flush=True)
    raw = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    raw = raw.drop(columns=['Unnamed: 0'], errors='ignore')
    print(f"  Raw: {raw.shape[0]:,} rows, {raw.shape[1]} columns")

    df = pd.DataFrame()
    for pilar_name, kaggle_name in PILAR_MAP.items():
        df[pilar_name] = raw[kaggle_name]
    df['machine_status'] = raw['machine_status']
    df['timestamp'] = raw['timestamp']

    # Forward-fill + back-fill (time series)
    feat_cols = list(PILAR_MAP.keys())
    df[feat_cols] = df[feat_cols].ffill().bfill()

    # Cumulative minutes as heure_fonctionnement proxy
    df['heure_fonctionnement'] = np.arange(len(df), dtype=float)

    # Binary target
    df['failure'] = (df['machine_status'] != 'NORMAL').astype(int)

    print(f"  Labels: {df.machine_status.value_counts().to_dict()}")
    print(f"  Failure rate: {df.failure.mean()*100:.1f}%")

    return df


def build_features(df):
    print("\n[2/7] Building features (8 core + 4 derived)...", flush=True)

    df['dp'] = df['pression_sortie'] - df['pression_entree']
    df['eta_proxy'] = (df['debit'] * df['dp'].clip(lower=0.001)) / df['courant_moteur'].clip(lower=0.1)
    df['thermal_load'] = df['temp_moteur'] - df['temp_palier']
    df['wear_index'] = np.log1p(df['heure_fonctionnement'] / 10000)

    features_ext = FEATURES + ['dp', 'eta_proxy', 'thermal_load', 'wear_index']
    print(f"  Features: {features_ext}")
    return df, features_ext


def train_failure_model(df, features_ext):
    X = df[features_ext].copy()
    y = df['failure'].copy()

    # Drop rows with NaN in any feature or label (can come from extra CSV blend)
    mask = X.notna().all(axis=1) & y.notna()
    if not mask.all():
        print(f"  Dropping {(~mask).sum():,} rows with NaN values before training")
        X = X[mask]
        y = y[mask]

    print(f"\n[3/7] Train/test split (80/20)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("[4/7] StandardScaler...", flush=True)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if HAS_SMOTE:
        print("[5/7] SMOTE resampling...", flush=True)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
        vc = pd.Series(y_train_bal).value_counts().to_dict()
        print(f"  After SMOTE: {vc}")
    else:
        print("[5/7] SMOTE unavailable — using class_weight", flush=True)
        X_train_bal, y_train_bal = X_train_s, y_train.values

    print("\n[6/7] Training candidate models...", flush=True)
    candidates = {}

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
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
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
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.80, min_samples_leaf=10, random_state=RANDOM_STATE
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
    y_cal, y_eval = y_test.values[:cal_split], y_test.values[cal_split:]

    raw_probs = best_model.predict_proba(X_cal_s)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(raw_probs, y_cal)
    cal_model = PlattModel(best_model, platt)

    eval_probs = cal_model.predict_proba(X_eval_s)[:, 1]
    eval_preds = (eval_probs >= 0.5).astype(int)
    eval_recall = recall_score(y_eval, eval_preds)
    eval_prec = precision_score(y_eval, eval_preds)
    eval_f1 = f1_score(y_eval, eval_preds)
    print(f"  Calibrated eval: Recall={eval_recall*100:.1f}%  Prec={eval_prec*100:.1f}%  F1={eval_f1*100:.1f}%")

    return cal_model, scaler, best_name, eval_recall, eval_prec, eval_f1, X, y, features_ext


def train_isolation_forest(df, scaler, features_ext):
    print("\n[7/7] Training Isolation Forest on NORMAL samples...", flush=True)
    normal = df[df['failure'] == 0][features_ext]
    n_samples = min(len(normal), 50000)
    normal_sample = normal.sample(n=n_samples, random_state=RANDOM_STATE)
    normal_scaled = scaler.transform(normal_sample)

    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(normal_scaled)
    print(f"  Fitted on {n_samples:,} normal samples")
    return iso


def save_models(cal_model, scaler, iso, best_name, eval_recall, eval_prec, eval_f1,
                X, y, features_ext, keep_zones=False):
    print("\nSaving .pkl files...", flush=True)

    with open(os.path.join(SCRIPT_DIR, "failure_model.pkl"), "wb") as f:
        pickle.dump(cal_model, f)

    with open(os.path.join(SCRIPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(SCRIPT_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)

    if not keep_zones:
        # No zone labels in Kaggle → preserve existing zone_models.pkl
        zpath = os.path.join(SCRIPT_DIR, "zone_models.pkl")
        if os.path.exists(zpath):
            print("  zone_models.pkl: KEPT (no zone labels in Kaggle data)")
        else:
            print("  zone_models.pkl: NOT FOUND — run train_universal.py to generate")

    meta = {
        "model_name": best_name + "_calibrated",
        "recall": round(eval_recall * 100, 1),
        "precision": round(eval_prec * 100, 1),
        "f1": round(eval_f1 * 100, 1),
        "n_total": len(X),
        "n_failures": int(y.sum()),
        "failure_rate": round(float(y.mean()) * 100, 1),
        "source": "kaggle_pump_sensor_real_data",
        "sensor_mapping": PILAR_MAP,
        "features": features_ext,
        "zones": ZONES,
        "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_ranges": {
            feat: [round(float(X[feat].min()), 4), round(float(X[feat].max()), 4)]
            for feat in features_ext
        },
    }
    with open(os.path.join(SCRIPT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  failure_model.pkl   OK (real-data trained)")
    print("  scaler.pkl          OK")
    print("  isolation_forest.pkl OK")
    print("  model_meta.json     OK")


def update_config(X, features_ext):
    """Print suggested config.py updates based on real data ranges."""
    print("\n" + "=" * 70)
    print("SUGGESTED config.py UPDATES (copy-paste into config.py):")
    print("=" * 70)

    core = [f for f in features_ext if f in FEATURES]
    print("\nFEATURE_MEDIANS = {")
    for feat in core:
        print(f"    '{feat}': {round(float(X[feat].median()), 3)},")
    print("}")

    print("\nSENSOR_BOUNDS = {")
    for feat in core:
        lo = round(float(X[feat].min()), 3)
        hi = round(float(X[feat].max()), 3)
        margin = (hi - lo) * 0.1
        print(f"    '{feat}': ({max(0, round(lo - margin, 3))}, {round(hi + margin, 3)}),")
    print("}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep-zones', action='store_true',
                    help='Keep existing zone_models.pkl untouched')
    ap.add_argument('--extra-csv', default=None,
                    help='Path to extra CSV with columns matching PILAR features + is_failure column')
    args = ap.parse_args()

    df = load_and_prepare()

    # ── Blend in extra CSV if provided ────────────────────────────────────────
    if args.extra_csv:
        import os as _os
        extra_path = args.extra_csv
        if not _os.path.isabs(extra_path):
            extra_path = _os.path.join(_os.path.dirname(__file__), extra_path)
        print(f"\n[+] Loading extra training data: {extra_path}", flush=True)
        extra = pd.read_csv(extra_path)
        # Normalize label column
        if 'is_failure' in extra.columns:
            extra['failure'] = extra['is_failure'].astype(int)
        elif 'failure' in extra.columns:
            extra['failure'] = extra['failure'].astype(int)
        else:
            raise ValueError("Extra CSV must have an 'is_failure' or 'failure' column")
        # Only keep columns that exist in df
        shared_cols = [c for c in extra.columns if c in df.columns]
        extra_aligned = extra[shared_cols].copy()
        # Add missing columns with median fill
        for col in df.columns:
            if col not in extra_aligned.columns:
                extra_aligned[col] = df[col].median() if df[col].dtype in ['float64','int64'] else None
        df = pd.concat([df, extra_aligned], ignore_index=True)
        n_extra_fail = extra['failure'].sum()
        n_extra_norm = len(extra) - n_extra_fail
        print(f"  Extra rows added: {len(extra):,}  (normal={n_extra_norm:,}, failure={n_extra_fail:,})")
        print(f"  Combined dataset: {len(df):,} rows")

    df, features_ext = build_features(df)
    cal_model, scaler, best_name, eval_r, eval_p, eval_f1, X, y, features_ext = \
        train_failure_model(df, features_ext)
    iso = train_isolation_forest(df, scaler, features_ext)
    save_models(cal_model, scaler, iso, best_name, eval_r, eval_p, eval_f1,
                X, y, features_ext, keep_zones=args.keep_zones)
    update_config(X, features_ext)

    print("\n" + "=" * 70)
    print("Done — restart app.py to load the new models.")
    print("=" * 70)
