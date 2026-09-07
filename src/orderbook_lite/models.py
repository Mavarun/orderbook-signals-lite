"""Logistic and ridge baselines (sklearn) for next-mid direction."""

from __future__ import annotations

from typing import Literal

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BaselineKind = Literal["logistic", "ridge"]


def build_baseline(kind: BaselineKind = "logistic", C: float = 1.0) -> Pipeline:
    """StandardScaler + logistic or ridge classifier."""
    if kind == "logistic":
        clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    elif kind == "ridge":
        clf = RidgeClassifier(alpha=1.0 / C if C else 1.0)
    else:
        raise ValueError(f"unknown baseline kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
