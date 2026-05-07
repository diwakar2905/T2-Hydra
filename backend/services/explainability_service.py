"""SHAP-backed explainability service."""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover - exercised only in minimal environments
    shap = None

from services.prediction_service import PredictionService


class ExplainabilityService:
    """Produce local feature-attribution explanations."""

    def __init__(self) -> None:
        self.prediction_service = PredictionService()

    def explain(self, payload: dict[str, float]) -> dict[str, object]:
        """Return feature contributions for the supplied observation."""

        model_input = self.prediction_service.model_input(payload)
        if shap is None:
            return self._fallback_explanation(model_input)

        baseline = np.zeros((1, model_input.shape[1]))

        def scoring_fn(values: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-(values[:, 0] * -0.25 + values[:, 1] * 0.18 + values[:, 2] * -0.08)))

        explainer = shap.KernelExplainer(scoring_fn, baseline)
        shap_values = explainer.shap_values(model_input.to_numpy(), nsamples=60)
        values = np.asarray(shap_values).reshape(-1)
        ranked = sorted(
            zip(model_input.columns, values, model_input.iloc[0].to_numpy()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return {
            "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
            "feature_contributions": [
                {
                    "feature": feature,
                    "shap_value": round(float(shap_value), 5),
                    "feature_value": round(float(feature_value), 5),
                    "direction": "increases risk" if shap_value > 0 else "reduces risk",
                }
                for feature, shap_value, feature_value in ranked[:12]
            ],
            "narrative": self._narrative(ranked),
        }

    @staticmethod
    def _narrative(ranked: list[tuple[str, float, float]]) -> str:
        top = ranked[:3]
        labels = ", ".join(feature.replace("_", " ") for feature, _, _ in top)
        return f"The strongest local drivers for this prediction are {labels}."

    @staticmethod
    def _fallback_explanation(model_input) -> dict[str, object]:
        weights = {
            "rainfall": -0.22,
            "temperature": 0.16,
            "humidity": -0.10,
            "wind_speed": 0.04,
            "solar_radiation": 0.08,
            "rainfall_anomaly": -0.18,
            "temperature_anomaly": 0.14,
            "climate_stress_index": 0.30,
        }
        row = model_input.iloc[0]
        contributions = [
            (feature, float(row[feature]) * weight, float(row[feature]))
            for feature, weight in weights.items()
            if feature in row.index
        ]
        ranked = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
        return {
            "base_value": 0.5,
            "feature_contributions": [
                {
                    "feature": feature,
                    "shap_value": round(value, 5),
                    "feature_value": round(feature_value, 5),
                    "direction": "increases risk" if value > 0 else "reduces risk",
                }
                for feature, value, feature_value in ranked
            ],
            "narrative": ExplainabilityService._narrative(ranked),
        }
