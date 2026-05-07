"""Baseline sklearn and XGBoost model factory functions."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class BaselineModelBundle:
    regression: dict[str, object]
    classification: dict[str, object]


def build_baseline_models(random_state: int = 42) -> BaselineModelBundle:
    """Create baseline regression and classification estimators."""

    regression = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        ),
    }
    classification = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        ),
    }
    return BaselineModelBundle(regression=regression, classification=classification)
