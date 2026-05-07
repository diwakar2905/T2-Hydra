"""Prediction service for drought probability and severity."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from utils.feature_engineering import MODEL_FEATURES, classify_drought_score, prepare_single_observation


@dataclass(frozen=True)
class PredictionResult:
    drought_probability: float
    drought_severity: str
    severity_score: float
    confidence_score: float
    risk_factors: list[dict[str, float | str]]


class PredictionService:
    """Generate drought predictions from climate observations."""

    def predict(self, payload: dict[str, float]) -> dict[str, object]:
        """Predict drought probability, severity class, and confidence."""

        features = prepare_single_observation(payload)
        score = float(features["drought_severity_score"].iloc[0])
        probability = self._probability_from_score(score)
        severity = classify_drought_score(score)
        confidence = self._confidence_from_inputs(features)
        result = PredictionResult(
            drought_probability=round(probability, 4),
            drought_severity=severity,
            severity_score=round(score, 2),
            confidence_score=round(confidence, 4),
            risk_factors=self._risk_factors(features),
        )
        return asdict(result)

    @staticmethod
    def model_input(payload: dict[str, float]) -> pd.DataFrame:
        """Return engineered model features for explainability."""

        return prepare_single_observation(payload)[MODEL_FEATURES]

    @staticmethod
    def _probability_from_score(score: float) -> float:
        return 1 / (1 + math.exp(-0.09 * (score - 45)))

    @staticmethod
    def _confidence_from_inputs(features: pd.DataFrame) -> float:
        humidity = float(features["humidity"].iloc[0])
        rainfall = float(features["rainfall"].iloc[0])
        completeness = features.notna().mean(axis=1).iloc[0]
        signal_strength = min(1.0, abs(50 - humidity) / 50 + min(rainfall, 20) / 40)
        return max(0.62, min(0.96, 0.65 + 0.25 * signal_strength + 0.10 * completeness))

    @staticmethod
    def _risk_factors(features: pd.DataFrame) -> list[dict[str, float | str]]:
        row = features.iloc[0]
        factors = {
            "Rainfall anomaly": abs(float(row["rainfall_anomaly"])),
            "Temperature anomaly": abs(float(row["temperature_anomaly"])),
            "Humidity deficit": max(0.0, 55.0 - float(row["humidity"])),
            "Solar radiation": float(row["solar_radiation"]),
            "Wind speed": float(row["wind_speed"]),
        }
        ranked = sorted(factors.items(), key=lambda item: item[1], reverse=True)[:4]
        return [{"feature": key, "impact": round(value, 3)} for key, value in ranked]
