# -*- coding: utf-8 -*-
"""
PILAR -- Platt calibration wrapper for failure model.
Kept in a separate module so pickle can resolve the class on load.
"""
import numpy as np


class PlattModel:
    """XGBoost/RF + Platt sigmoid calibration. Pickle-safe."""

    def __init__(self, base, platt):
        self.base  = base   # fitted base classifier
        self.platt = platt  # fitted LogisticRegression on raw probs

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn compatibility
    @property
    def classes_(self):
        return np.array([0, 1])
