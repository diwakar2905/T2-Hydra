"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate standard regression metrics."""

    return RegressionMetrics(
        rmse=float(mean_squared_error(y_true, y_pred, squared=False)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Calculate weighted multiclass classification metrics."""

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )
