"""
PILAR — 3-class (normal / developing / severe) training
========================================================
Tests the paper's hypothesis (Abdallah et al., Sensors 2023) that a 3-class
classifier can learn a distinctive "near-failure" signature rather than
collapsing it into a binary "failure vs normal" decision.

Classes:
  0  NORMAL      — no failure signature applied
  1  DEVELOPING  — failure signature with severity 0.05–0.65
  2  SEVERE      — failure signature with severity 0.65–1.00

The commercial thesis: class 1 is the warning that actually saves customers
money. The paper saw 96% recall on the intermediate class; we check whether
PILAR's synthetic data reproduces that pattern.

Usage:
  py -3.14 train_3class.py                  # full train
  py -3.14 train_3class.py --scale 0.3      # faster dev run
  py -3.14 train_3class.py --model xgb      # use XGBoost (default: gbt)

Outputs:
  failure_model_3class.pkl  — the 3-class classifier
  scaler_3class.pkl         — its StandardScaler
  meta_3class.json          — metrics + confusion matrix
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from train_universal import (
    MACHINE_PROFILES,
    FEATURES,
    ZONES,
    apply_failure_signature,
    generate_degradation_sequence,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")

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


SEVERITY_TIERS = [
    (0.40, 0.05, 0.30),
    (0.35, 0.30, 0.65),
    (0.25, 0.65, 1.00),
]

CLASS_NAMES = ["normal", "developing", "severe"]
DEVELOPING_THRESHOLD = 0.65   # severity < this → developing, else severe


def _sample_severity() -> float:
    r = np.random.random()
    cum = 0.0
    for frac, lo, hi in SEVERITY_TIERS:
        cum += frac
        if r <= cum:
            return float(np.random.uniform(lo, hi))
    return float(np.random.uniform(0.65, 1.0))


def _add_sensor_noise(row: dict, noise_pct: float = 0.02) -> dict:
    for f in FEATURES:
        if f in row:
            row[f] *= np.random.uniform(1 - noise_pct, 1 + noise_pct)
    return row


def _severity_to_class(severity: float) -> int:
    """0 = normal (severity ~0), 1 = developing (0<sev<0.65), 2 = severe (>=0.65)."""
    if severity <= 0.0:
        return 0
    return 1 if severity < DEVELOPING_THRESHOLD else 2


def build_3class_dataset(scale: float = 1.0) -> pd.DataFrame:
    """Build the synthetic dataset with a 3-level severity_class label."""
    rng = np.random.default_rng(RANDOM_STATE)
    all_rows = []
    for profile_key, profile in MACHINE_PROFILES.items():
        n_norm = max(int(profile["n_normal"] * scale), 400)
        n_fail = max(int(n_norm * 0.10), 100)
        n_seq = max(int(n_norm * 0.015), 20)
        norm_cfg = profile["normal"]
        weights = np.array(profile["zone_weights"], dtype=float)
        weights /= weights.sum()

        # ── Normal samples (class 0) ────────────────────────────────────────
        for _ in range(n_norm):
            row = {
                f: float(np.clip(np.random.normal(mu, sig), lo, hi))
                for f, (mu, sig, lo, hi) in norm_cfg.items()
            }
            row = _add_sensor_noise(row)
            row["severity"] = 0.0
            row["severity_class"] = 0
            row["failure"] = 0
            for z in ZONES:
                row[z] = 0
            row["profile"] = profile_key
            all_rows.append(row)

        # ── Failure samples (class 1 or 2 depending on severity) ────────────
        zone_counts = np.round(weights * n_fail).astype(int)
        zone_counts[-1] += n_fail - zone_counts.sum()
        for zi, zone in enumerate(ZONES):
            for _ in range(zone_counts[zi]):
                base = {
                    f: float(np.clip(np.random.normal(mu, sig), lo, hi))
                    for f, (mu, sig, lo, hi) in norm_cfg.items()
                }
                sev = _sample_severity()
                row = apply_failure_signature(base, zone, profile_key, severity=sev)
                row = _add_sensor_noise(row)
                row["severity"] = float(sev)
                row["severity_class"] = _severity_to_class(sev)
                row["failure"] = 1
                for z in ZONES:
                    row[z] = 1 if z == zone else 0
                if np.random.random() < 0.15:
                    co = np.random.choice([z for z in ZONES if z != zone])
                    row[co] = 1
                row["profile"] = profile_key
                all_rows.append(row)

        # ── Degradation sequences (each step labelled by its own severity) ──
        seq_per_zone = np.round(weights * n_seq).astype(int)
        seq_per_zone[-1] += n_seq - seq_per_zone.sum()
        for zi, zone in enumerate(ZONES):
            for _ in range(seq_per_zone[zi]):
                seq = generate_degradation_sequence(norm_cfg, zone, profile_key)
                for step_row, step_sev in seq:
                    step_row = _add_sensor_noise(step_row)
                    step_row["severity"] = float(step_sev)
                    step_row["severity_class"] = _severity_to_class(step_sev)
                    step_row["failure"] = 1
                    for z in ZONES:
                        step_row[z] = 1 if z == zone else 0
                    if np.random.random() < 0.10:
                        co = np.random.choice([z for z in ZONES if z != zone])
                        step_row[co] = 1
                    step_row["profile"] = profile_key
                    all_rows.append(step_row)

    df = pd.DataFrame(all_rows)
    # Physics-derived features — identical to train_universal.py
    df["dp"] = df["pression_sortie"] - df["pression_entree"]
    df["eta_proxy"] = (
        df["debit"] * df["dp"].clip(0.001)
    ) / df["courant_moteur"].clip(0.1)
    df["thermal_load"] = df["temp_moteur"] - df["temp_palier"]
    df["wear_index"] = np.log1p(df["heure_fonctionnement"] / 10000)
    return df


FEATURES_EXTENDED = FEATURES + ["dp", "eta_proxy", "thermal_load", "wear_index"]


def _build_model(model_name: str, n_classes: int):
    if model_name == "xgb":
        if not HAS_XGB:
            raise SystemExit("XGBoost not installed. Use --model gbt or rf.")
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0,
            min_child_weight=10, gamma=0.1,
            objective="multi:softprob", num_class=n_classes,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            n_jobs=-1, verbosity=0,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            min_samples_split=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    # default: GradientBoosting
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.80, min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Scale n_normal per profile (0.3 = ~3x faster). Default 1.0")
    ap.add_argument("--model", choices=["gbt", "xgb", "rf"], default="gbt",
                    help="Classifier. Default 'gbt' (GradientBoosting).")
    ap.add_argument("--out-model", default="failure_model_3class.pkl")
    ap.add_argument("--out-scaler", default="scaler_3class.pkl")
    ap.add_argument("--out-meta", default="meta_3class.json")
    args = ap.parse_args()

    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print(f"PILAR — 3-class training (model={args.model}, scale={args.scale})")
    print("=" * 70)

    # ── 1) Build dataset ────────────────────────────────────────────────────
    print("\n[1/5] Building 3-class dataset...")
    t0 = time.time()
    df = build_3class_dataset(scale=args.scale)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    class_counts = df["severity_class"].value_counts().sort_index().to_dict()
    print(f"  rows={len(df):,}  time={time.time()-t0:.1f}s")
    print(f"  class counts: " + ", ".join(
        f"{CLASS_NAMES[int(c)]}={n:,}" for c, n in class_counts.items()
    ))

    X = df[FEATURES_EXTENDED]
    y = df["severity_class"].values

    # ── 2) Split ────────────────────────────────────────────────────────────
    print("\n[2/5] Stratified 80/20 split...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # ── 3) Scale ────────────────────────────────────────────────────────────
    print("[3/5] StandardScaler...")
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    # ── 4) SMOTE (multi-class) ──────────────────────────────────────────────
    if HAS_SMOTE:
        print("[4/5] Multi-class SMOTE...")
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            Xs_bal, y_bal = sm.fit_resample(Xs_tr, y_tr)
            bal_counts = pd.Series(y_bal).value_counts().sort_index().to_dict()
            print(f"  after SMOTE: " + ", ".join(
                f"{CLASS_NAMES[int(c)]}={n:,}" for c, n in bal_counts.items()
            ))
        except Exception as e:
            print(f"  SMOTE failed: {e} — falling back to class_weight='balanced'")
            Xs_bal, y_bal = Xs_tr, y_tr
    else:
        print("[4/5] imblearn unavailable — relying on class_weight='balanced'")
        Xs_bal, y_bal = Xs_tr, y_tr

    # ── 5) Train + evaluate ─────────────────────────────────────────────────
    print(f"\n[5/5] Training {args.model}...")
    t0 = time.time()
    n_classes = len(CLASS_NAMES)
    model = _build_model(args.model, n_classes)
    model.fit(Xs_bal, y_bal)
    train_time = time.time() - t0
    y_pred = model.predict(Xs_te)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    # Per-class recall (diagonal / row sum)
    per_class_recall = []
    for i in range(n_classes):
        row_sum = int(cm[i].sum())
        per_class_recall.append(float(cm[i, i] / row_sum) if row_sum else 0.0)

    # Macro / weighted metrics
    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"  trained in {train_time:.1f}s\n")
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>12s} " + " ".join(f"{n:>11s}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>12s} " + " ".join(f"{int(v):>11d}" for v in row))

    print(f"\nPer-class recall:")
    for name, r in zip(CLASS_NAMES, per_class_recall):
        print(f"  {name:>12s}  {r*100:5.1f}%")
    print(f"  macro F1 : {macro_f1*100:5.1f}%   weighted F1: {weighted_f1*100:5.1f}%")

    # Paper-style verdict
    dev_recall = per_class_recall[1]
    if dev_recall >= 0.90:
        verdict = f"STRONG: developing-class recall {dev_recall*100:.1f}% matches the paper's 96% finding."
    elif dev_recall >= 0.75:
        verdict = f"MODERATE: developing-class recall {dev_recall*100:.1f}% is useful but below paper's 96%."
    else:
        verdict = f"WEAK: developing-class recall only {dev_recall*100:.1f}% — 3-class approach not validated."
    print(f"\nVerdict: {verdict}")

    # ── Save ────────────────────────────────────────────────────────────────
    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)
    with open(args.out_scaler, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_type": args.model,
        "classes": CLASS_NAMES,
        "developing_threshold_severity": DEVELOPING_THRESHOLD,
        "n_train": int(len(Xs_bal)),
        "n_test": int(len(Xs_te)),
        "class_counts_total": {CLASS_NAMES[int(c)]: int(n) for c, n in class_counts.items()},
        "per_class_recall": {CLASS_NAMES[i]: round(r, 4) for i, r in enumerate(per_class_recall)},
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_recall": round(float(macro_recall), 4),
        "confusion_matrix": cm.tolist(),
        "features": FEATURES_EXTENDED,
        "scale_factor": args.scale,
        "verdict": verdict,
        "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved:")
    print(f"  {args.out_model}")
    print(f"  {args.out_scaler}")
    print(f"  {args.out_meta}")


if __name__ == "__main__":
    main()
