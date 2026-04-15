"""Re-export shim — canonical name for the ML pipeline module.

The actual implementation lives in `pilar_ml.py` (historical name used by
PyInstaller spec, tests, and many in-line imports). Prefer `ml_engine` in new
code; both names resolve to the same module.
"""
from pilar_ml import *  # noqa: F401,F403
from pilar_ml import (  # noqa: F401
    init_models, reload_models, predict_risk,
    build_model_input, compute_shap, compute_rul, compute_anomaly_score,
    model, scaler, modeles_zones,
)
