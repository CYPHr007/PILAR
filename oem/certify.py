#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PILAR OEM Certification Pipeline
==================================
Trains and validates a PILAR model on OEM-specific machine data.
Produces a certified model package ready to deploy with pilar_embedded.py.

Usage
-----
  python -m oem.certify \\
    --data     /path/to/machine_data.csv \\
    --config   oem/examples/robot_arm.json \\
    --out      ./certified_models/robot_arm/ \\
    --label-col  failure \\
    --label-val  1

Arguments
---------
  --data        CSV file with sensor readings + a failure label column
  --config      OEM machine config JSON (defines machine_type, sensor nominals, etc.)
  --out         Output directory for the certified model package
  --label-col   Column name in the CSV that contains the failure label (default: failure)
  --label-val   Value in label-col that means "failure" (default: 1)
                Use --label-val BROKEN if your labels are text

CSV format
----------
The CSV must contain at least some of the PILAR sensor columns:
  vibration, temp_palier, debit, pression_entree, pression_sortie,
  courant_moteur, temp_moteur, heure_fonctionnement

Plus a label column (default: "failure") with values 0/1 or custom text.

Columns may be named differently — provide a column_map in your machine config:
  "column_map": {
    "Vibration_mms":  "vibration",
    "MotorAmps":      "courant_moteur"
  }

Output
------
The --out directory contains:
  failure_model.pkl   — trained GradientBoosting + Platt calibration
  scaler.pkl          — fitted StandardScaler
  isolation_forest.pkl— fitted IsolationForest on normal samples
  machine_config.json — your config updated with certification results
  certification_report.json — detailed metrics and model parameters
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import pilar_calibrator  # noqa: F401
from pilar_calibrator import PlattModel
from oem.machine_config import MachineConfig

# PILAR sensor features
_COLONNES = [
    "vibration", "temp_palier", "debit",
    "pression_entree", "pression_sortie",
    "courant_moteur", "temp_moteur", "heure_fonctionnement",
]
_DERIVED = ["dp", "eta_proxy", "thermal_load", "wear_index"]

RANDOM_STATE = 42


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 derived features used by the 12-feature model."""
    df = df.copy()
    dp = df["pression_sortie"] - df["pression_entree"]
    df["dp"] = dp
    df["eta_proxy"] = (
        df["debit"] * dp.clip(lower=0.001)
    ) / df["courant_moteur"].clip(lower=0.1)
    df["thermal_load"] = df["temp_moteur"] - df["temp_palier"]
    df["wear_index"] = np.log1p(df["heure_fonctionnement"] / 10000)
    return df


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    csv_path: str,
    machine_cfg: MachineConfig,
    label_col: str,
    label_val: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load CSV, apply column_map from machine config, impute missing sensors,
    add derived features. Returns (X_df, y_array).
    """
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns from {csv_path}")

    # Apply column map if defined in machine config
    col_map = machine_cfg.to_dict().get("column_map", {})
    if col_map:
        df = df.rename(columns={v: k for k, v in col_map.items()})
        print(f"  Applied column_map: {col_map}")

    # Extract label
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert label to binary
    raw_labels = df[label_col]
    try:
        y = (raw_labels.astype(float) == float(label_val)).astype(int).values
    except (ValueError, TypeError):
        y = (raw_labels.astype(str) == str(label_val)).astype(int).values

    failure_rate = y.mean() * 100
    print(f"  Labels: {y.sum():,} failures / {len(y):,} total ({failure_rate:.1f}%)")

    # Keep sensor columns that exist in the CSV
    present = [c for c in _COLONNES if c in df.columns]
    missing_cols = [c for c in _COLONNES if c not in df.columns]
    if missing_cols:
        print(f"  Missing sensor columns (will use machine nominals): {missing_cols}")

    X = df[present].copy()

    # Impute missing columns from machine-specific nominals or global medians
    _global_medians = {
        "vibration": 13.6, "temp_palier": 51.6, "debit": 2.5,
        "pression_entree": 48.1, "pression_sortie": 632.6,
        "courant_moteur": 2.5, "temp_moteur": 44.3, "heure_fonctionnement": 110000,
    }
    for col in missing_cols:
        nominal = machine_cfg.get_nominal(col)
        fill_val = nominal if nominal is not None else _global_medians.get(col, 0.0)
        X[col] = fill_val

    # Reorder to canonical order
    X = X[_COLONNES]

    # Impute NaN within present columns
    for col in _COLONNES:
        if X[col].isna().any():
            fill = machine_cfg.get_nominal(col) or _global_medians.get(col, 0.0)
            X[col] = X[col].fillna(fill)

    # Add derived features
    X = _add_derived(X)
    feature_cols = _COLONNES + _DERIVED

    return X[feature_cols], y


# ── Model training ────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Train GradientBoosting + Platt calibration.
    Returns (PlattModel, StandardScaler, train_metrics, test_metrics).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Handle class imbalance with class_weight
    neg, pos = np.bincount(y_train)
    scale_pos = neg / max(pos, 1)
    print(f"  Class balance — neg:{neg:,} pos:{pos:,}  scale_pos_weight={scale_pos:.2f}")

    # Try SMOTE if available and imbalance is significant
    if scale_pos > 3:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, pos - 1))
            X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
            print(f"  SMOTE applied: {len(y_train):,} samples after resampling")
        except Exception:
            pass

    # GradientBoosting
    print("  Training GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train_s, y_train)

    # Platt calibration on a held-out calibration set
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_test_s, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
    )
    raw_probs = gb.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(raw_probs, y_cal)

    calibrated = PlattModel(gb, platt)

    # Evaluate on validation set
    probs_val = calibrated.predict_proba(X_val)[:, 1]
    preds_val = (probs_val >= 0.30).astype(int)

    test_metrics = {
        "recall":    round(float(recall_score(y_val, preds_val, zero_division=0)), 4),
        "precision": round(float(precision_score(y_val, preds_val, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_val, preds_val, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_val, probs_val)) if len(np.unique(y_val)) > 1 else 0.0, 4),
        "confusion_matrix": confusion_matrix(y_val, preds_val).tolist(),
        "n_test":    int(len(y_val)),
    }

    print(f"  Recall={test_metrics['recall']:.3f}  "
          f"Precision={test_metrics['precision']:.3f}  "
          f"F1={test_metrics['f1']:.3f}  "
          f"AUC={test_metrics['roc_auc']:.3f}")

    return calibrated, scaler, test_metrics


def train_isolation_forest(X_scaled: np.ndarray, y: np.ndarray) -> IsolationForest:
    """Fit IsolationForest on normal samples only."""
    normal_mask = y == 0
    X_normal = X_scaled[normal_mask]
    clf = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    clf.fit(X_normal)
    print(f"  IsolationForest fitted on {len(X_normal):,} normal samples")
    return clf


# ── Output ────────────────────────────────────────────────────────────────────

def save_package(
    out_dir: str,
    model: PlattModel,
    scaler: StandardScaler,
    iso: IsolationForest,
    machine_cfg: MachineConfig,
    metrics: dict,
    training_meta: dict,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save pkl files
    for name, obj in [
        ("failure_model.pkl",    model),
        ("scaler.pkl",           scaler),
        ("isolation_forest.pkl", iso),
    ]:
        with open(os.path.join(out_dir, name), "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {name}")

    # Certification metadata
    cert = {
        "certified_by":      "PILAR",
        "certified_date":    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_version":     "1.3.3",
        "training_samples":  training_meta["n_train"],
        "recall":            metrics["recall"],
        "precision":         metrics["precision"],
        "f1":                metrics["f1"],
        "roc_auc":           metrics["roc_auc"],
    }

    # Update machine config with certification
    cfg_dict = machine_cfg.to_dict()
    cfg_dict["certification"] = cert
    cfg_path = os.path.join(out_dir, "machine_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved machine_config.json")

    # Full certification report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine_type": machine_cfg.machine_type,
        "machine_name": machine_cfg.machine_name,
        "manufacturer": machine_cfg.manufacturer,
        "model_ref":    machine_cfg.model_ref,
        "training": training_meta,
        "metrics":  metrics,
        "certification": cert,
        "model_parameters": {
            "algorithm":     "GradientBoosting + PlattCalibration",
            "n_estimators":  300,
            "max_depth":     4,
            "learning_rate": 0.05,
            "features":      _COLONNES + _DERIVED,
        },
        "deployment": {
            "command": f"python pilar_embedded.py --machine-config {cfg_path}",
            "min_recall_warning": "If recall < 0.90, collect more failure samples before deploying",
        }
    }
    report_path = os.path.join(out_dir, "certification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved certification_report.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PILAR OEM Certification — train and validate a machine-specific model"
    )
    parser.add_argument("--data",       required=True, help="CSV file with sensor data + labels")
    parser.add_argument("--config",     required=True, help="OEM machine config JSON")
    parser.add_argument("--out",        required=True, help="Output directory for certified model package")
    parser.add_argument("--label-col",  default="failure", help="Label column name (default: failure)")
    parser.add_argument("--label-val",  default="1",       help="Failure label value (default: 1)")
    args = parser.parse_args()

    print(f"\nPILAR OEM Certification")
    print(f"{'='*50}")

    # Load machine config
    machine_cfg = MachineConfig.from_file(args.config)
    print(f"Machine : {machine_cfg.machine_name} ({machine_cfg.machine_type})")

    # Load data
    print(f"\nLoading data...")
    X, y = load_data(args.data, machine_cfg, args.label_col, args.label_val)
    n_train = int(len(y) * 0.8)

    # Train
    print(f"\nTraining model ({len(y):,} samples)...")
    model, scaler, metrics = train_model(X, y)

    # IsolationForest
    print(f"\nFitting anomaly detector...")
    X_scaled = scaler.transform(X)
    iso = train_isolation_forest(X_scaled, y)

    # Save
    print(f"\nSaving certified model package to {args.out}/")
    save_package(
        args.out, model, scaler, iso, machine_cfg, metrics,
        training_meta={"n_train": n_train, "n_total": int(len(y)),
                       "failure_rate": round(float(y.mean()), 4)}
    )

    # Summary
    print(f"\n{'='*50}")
    print(f"Certification complete")
    print(f"  Recall    : {metrics['recall']*100:.1f}%")
    print(f"  Precision : {metrics['precision']*100:.1f}%")
    print(f"  F1        : {metrics['f1']*100:.1f}%")
    print(f"  AUC-ROC   : {metrics['roc_auc']*100:.1f}%")

    if metrics["recall"] < 0.85:
        print(f"\n  WARNING: Recall {metrics['recall']*100:.1f}% is below 85%.")
        print(f"  Collect more failure samples before deploying to production.")
    elif metrics["recall"] >= 0.95:
        print(f"\n  Model is production-ready.")

    print(f"\nDeploy with:")
    print(f"  python pilar_embedded.py --machine-config {args.out}/machine_config.json")


if __name__ == "__main__":
    main()
