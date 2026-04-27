# -*- coding: utf-8 -*-
"""
PILAR -- ML pipeline functions.
Extracted from app.py for maintainability. All functions are stateless
except for the global model references passed in at init time.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from pilar_logging import get_logger
from config import (
    COLONNES, CORE_FEATURES, FEATURE_MEDIANS, FAILURE_ZONES, RUL_SCALE_FACTOR,
    DEFAULT_THRESHOLD, ZONE_ALERT_THRESHOLD,
    ISO_CONTAMINATION, ISO_MIN_NORMAL_SAMPLES, ISO_RETRAIN_INTERVAL, ISO_MAX_SAMPLES,
    MOT_CURRENT_WARN_PCT, MOT_CURRENT_CRIT_PCT, MOT_TEMP_WARN, MOT_TEMP_RANGE,
    MOT_DEFAULT_NOMINAL_A,
)

logger = get_logger("pilar.ml")

# ── Module-level state (set by init_models) ──────────────────────────────────
model = None
scaler = None
modeles_zones: Dict = {}
_rul_model = None
_rul_scaler = None
_iso_forest = None
_normal_samples: List = []
_shap_explainer = None

_DERIVED_FEATURES = ["dp", "eta_proxy", "thermal_load", "wear_index"]
_RUL_SCALE = RUL_SCALE_FACTOR


def init_models(app_dir: str) -> None:
    """Load all ML models from pkl files. Call once at startup."""
    global model, scaler, modeles_zones, _rul_model, _rul_scaler, _iso_forest

    def _pkl(name: str) -> str:
        return os.path.join(app_dir, name)

    # Main failure model + scaler
    try:
        import pilar_calibrator as _  # noqa: F811 — registers PlattModel for pickle
        with open(_pkl("failure_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(_pkl("scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        logger.info("ML models loaded from %s", app_dir)
    except FileNotFoundError as e:
        logger.error("Model file missing: %s (search dir: %s)", e, app_dir)
        _log_model_error(f"FATAL: model file missing — {e} (dir: {app_dir})")
    except Exception as e:
        import traceback
        msg = f"Model load error — {type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(msg)
        _log_model_error(msg)

    # Zone models
    try:
        with open(_pkl("zone_models.pkl"), "rb") as f:
            modeles_zones = pickle.load(f)
        logger.info("Zone models loaded (%d zones)", len(modeles_zones))
    except FileNotFoundError:
        logger.warning("zone_models.pkl absent — zone analysis disabled")
    except Exception as e:
        logger.error("Zone model load error: %s", e)

    # RUL model (NASA C-MAPSS)
    try:
        with open(_pkl("rul_model.pkl"), "rb") as f:
            _rul_model = pickle.load(f)
        with open(_pkl("rul_scaler.pkl"), "rb") as f:
            _rul_scaler = pickle.load(f)
        logger.info("RUL model loaded (NASA C-MAPSS GBT)")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("RUL model load error: %s", e)

    # Isolation Forest
    try:
        with open(_pkl("isolation_forest.pkl"), "rb") as f:
            _iso_forest = pickle.load(f)
        logger.info("Isolation Forest loaded")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("Isolation Forest load error: %s", e)


def _log_model_error(msg: str) -> None:
    """Write model load error to a log file accessible in frozen (no-console) mode."""
    try:
        log_path = os.path.join(
            os.path.expanduser("~"), "AppData", "Local", "PILAR", "model_load.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        from datetime import datetime
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def build_model_input(params: dict) -> pd.DataFrame:
    """
    Build a DataFrame row matching the loaded scaler's feature set.
    Auto-detects 8-column (legacy) vs 12-column (universal) models.
    """
    row = {c: float(params[c]) for c in COLONNES}
    df_row = pd.DataFrame([row], columns=COLONNES)

    n_expected = getattr(scaler, "n_features_in_", len(COLONNES))
    if n_expected > len(COLONNES):
        dp = df_row["pression_sortie"] - df_row["pression_entree"]
        df_row["dp"] = dp
        df_row["eta_proxy"] = (
            df_row["debit"] * dp.clip(lower=0.001)
        ) / df_row["courant_moteur"].clip(lower=0.1)
        df_row["thermal_load"] = df_row["temp_moteur"] - df_row["temp_palier"]
        df_row["wear_index"] = np.log1p(df_row["heure_fonctionnement"] / 10000)

    return df_row


def _trim_to_n(x_scaled: np.ndarray, n: int) -> np.ndarray:
    """Trim scaled array to first n features (backward compat)."""
    if x_scaled.shape[1] > n:
        return x_scaled[:, :n]
    return x_scaled


def compute_rul(donnees_raw: pd.DataFrame) -> Optional[float]:
    """Return estimated remaining useful life in hours, or None."""
    if _rul_model is None or _rul_scaler is None:
        return None
    try:
        n = getattr(_rul_scaler, "n_features_in_", 8)
        raw = (
            donnees_raw.iloc[:, :n].values
            if hasattr(donnees_raw, "iloc")
            else donnees_raw[:, :n]
        )
        x_rul = _rul_scaler.transform(raw)
        cycles = float(np.clip(_rul_model.predict(x_rul)[0], 0, None))
        return round(cycles * _RUL_SCALE, 1)
    except Exception as e:
        logger.error("RUL compute error: %s", e)
        return None


def compute_anomaly_score(x_scaled: np.ndarray) -> Optional[float]:
    """Return 0-100 anomaly score (0=normal, 100=very anomalous), or None."""
    if _iso_forest is None:
        return None
    try:
        n = getattr(_iso_forest, "n_features_in_", 8)
        decision = float(_iso_forest.decision_function(_trim_to_n(x_scaled, n))[0])
        return round(min(100.0, max(0.0, -decision / 0.15 * 100)), 1)
    except Exception as e:
        logger.error("IsoForest score error: %s", e)
        return None


def _extract_shap_vals(sv) -> np.ndarray:
    """
    Normalise SHAP output across library versions.
    - list of arrays [neg, pos]  → positive-class row 0 (old SHAP)
    - ndarray shape (1, F, 2)    → positive-class row 0 (new SHAP TreeExplainer)
    - ndarray shape (1, F)       → row 0 (regression / single output)
    """
    if isinstance(sv, list):
        return np.array(sv[1][0])
    arr = np.array(sv)
    if arr.ndim == 3:          # (samples, features, classes) — take positive class
        return arr[0, :, 1]
    return arr[0]              # (samples, features)


def compute_shap(x_scaled: np.ndarray) -> List[Dict[str, str]]:
    """Return top-3 SHAP feature impacts, or [] on failure."""
    global _shap_explainer
    try:
        import shap

        # Use all feature names (8 core + 4 derived if universal model)
        n_feats = x_scaled.shape[1]
        if n_feats > len(COLONNES):
            feature_names = COLONNES + _DERIVED_FEATURES[: n_feats - len(COLONNES)]
        else:
            feature_names = COLONNES[:n_feats]

        # Try TreeExplainer first (instant for XGBoost/RF/GB).
        # Unwrap PlattModel so TreeExplainer sees the native tree model directly.
        try:
            _base = model.base if hasattr(model, "base") else model
            exp = shap.TreeExplainer(_base)
            sv = exp.shap_values(x_scaled)
            vals = _extract_shap_vals(sv)
        except Exception:
            if _shap_explainer is None:
                _init_shap_explainer()
            if _shap_explainer is None:
                return []
            sv = _shap_explainer.shap_values(x_scaled, nsamples=50, silent=True)
            vals = _extract_shap_vals(sv)

        total_abs = sum(abs(v) for v in vals) or 1.0
        top3 = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[
            :3
        ]
        return [
            {
                "feature_key": f,
                "impact": ("+" if v > 0 else "")
                + str(round(abs(v) / total_abs * 100))
                + "%",
                "direction": "up" if v > 0 else "down",
            }
            for f, v in top3
        ]
    except Exception as e:
        logger.error("SHAP compute error: %s", e)
        return []


def _init_shap_explainer() -> None:
    global _shap_explainer
    if model is None or scaler is None:
        return
    try:
        import shap
        bg = build_model_input(FEATURE_MEDIANS)
        bg_scaled = scaler.transform(bg)
        _shap_explainer = shap.KernelExplainer(model.predict_proba, bg_scaled)
        logger.info("SHAP KernelExplainer initialized")
    except Exception as e:
        logger.error("SHAP init error: %s", e)


def predict_risk(
    params: dict,
    threshold: Optional[float] = None,
    return_extra: bool = False,
    machine_context: Optional[dict] = None,
) -> tuple:
    """
    Core prediction function.

    Returns:
        If return_extra=False: (probabilite, prediction, zones_risque, confidence, missing_keys)
        If return_extra=True:  + (anomaly_score, shap_explanations, rul_hours)
    """
    global _iso_forest, _normal_samples
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # Per-machine imputation
    _medians = dict(FEATURE_MEDIANS)
    if machine_context:
        if machine_context.get("nominal_flow"):
            _medians["debit"] = machine_context["nominal_flow"]
        if machine_context.get("nominal_pressure"):
            _medians["pression_sortie"] = machine_context["nominal_pressure"]
        if machine_context.get("nominal_current"):
            _medians["courant_moteur"] = machine_context["nominal_current"]
        if machine_context.get("nominal_vibration"):
            _medians["vibration"] = machine_context["nominal_vibration"]

    # Impute missing features
    missing_keys = [k for k in CORE_FEATURES if params.get(k) is None]
    for k in missing_keys:
        params[k] = _medians[k]

    confidence = round(
        (len(CORE_FEATURES) - len(missing_keys)) / len(CORE_FEATURES) * 100
    )
    donnees = build_model_input(params)
    donnees_scaled = scaler.transform(donnees)
    probabilite = round(float(model.predict_proba(donnees_scaled)[0][1]) * 100, 1)
    prediction = 1 if probabilite >= threshold else 0

    # Zone analysis
    zones_risque: List[Dict[str, Any]] = []
    if prediction == 1:
        for col, nom in FAILURE_ZONES.items():
            if col in modeles_zones:
                pz = round(
                    float(modeles_zones[col].predict_proba(donnees_scaled)[0][1]) * 100,
                    1,
                )
                if pz >= ZONE_ALERT_THRESHOLD:
                    zones_risque.append({"nom": nom, "proba": pz})
        zones_risque.sort(key=lambda x: x["proba"], reverse=True)

        # MOT fallback: threshold rule when no trained model
        if "MOT" not in modeles_zones:
            _curr = params.get("courant_moteur", _medians["courant_moteur"])
            _tmot = params.get("temp_moteur", _medians["temp_moteur"])
            _nom_curr = (
                (machine_context.get("nominal_current") if machine_context else None)
                or MOT_DEFAULT_NOMINAL_A
            )
            _curr_warn = _nom_curr * MOT_CURRENT_WARN_PCT
            _curr_crit = _nom_curr * MOT_CURRENT_CRIT_PCT
            _curr_range = max(_curr_crit - _curr_warn, 1.0)
            _curr_score = (
                min(100.0, max(0.0, (_curr - _curr_warn) / _curr_range * 100))
                if _curr > _curr_warn
                else 0.0
            )
            _temp_score = (
                min(100.0, max(0.0, (_tmot - (MOT_TEMP_WARN - 10)) / MOT_TEMP_RANGE * 100))
                if _tmot > MOT_TEMP_WARN
                else 0.0
            )
            _pz_mot = round(max(_curr_score, _temp_score), 1)
            if _pz_mot >= ZONE_ALERT_THRESHOLD:
                zones_risque.append({"nom": FAILURE_ZONES["MOT"], "proba": _pz_mot})
                zones_risque.sort(key=lambda x: x["proba"], reverse=True)

    if not return_extra:
        return probabilite, prediction, zones_risque, confidence, missing_keys

    # ── Extra: Isolation Forest ──────────────────────────────────────────────
    anomaly_score = None
    try:
        if prediction == 0:
            _normal_samples.append(donnees_scaled[0].tolist())
            _n = len(_normal_samples)
            if _n >= ISO_MIN_NORMAL_SAMPLES and (_iso_forest is None or _n % ISO_RETRAIN_INTERVAL == 0):
                from sklearn.ensemble import IsolationForest

                clf = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42)
                clf.fit(np.array(_normal_samples[-ISO_MAX_SAMPLES:]))
                _iso_forest = clf
                try:
                    pkl_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "isolation_forest.pkl",
                    )
                    with open(pkl_path, "wb") as f:
                        pickle.dump(clf, f)
                except Exception:
                    pass
                logger.info("Isolation Forest retrained (%d normal samples)", _n)
        anomaly_score = compute_anomaly_score(donnees_scaled)
    except Exception as e:
        logger.error("IsoForest pipeline error: %s", e)

    # ── Extra: SHAP ──────────────────────────────────────────────────────────
    shap_explanations = compute_shap(donnees_scaled)

    # ── Extra: RUL ───────────────────────────────────────────────────────────
    rul_hours = compute_rul(donnees)

    return (
        probabilite,
        prediction,
        zones_risque,
        confidence,
        missing_keys,
        anomaly_score,
        shap_explanations,
        rul_hours,
    )


def reload_models(app_dir: str) -> bool:
    """Hot-reload all models. Returns True on success."""
    try:
        init_models(app_dir)
        return model is not None and scaler is not None
    except Exception as e:
        logger.error("Model reload failed: %s", e)
        return False
