"""Forecasting service for future drought trends."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from utils.feature_engineering import add_climate_features, classify_drought_score


class ForecastingService:
    """Create scenario-based drought forecasts."""

    def forecast(self, payload: dict[str, float]) -> dict[str, object]:
        """Forecast future drought trend and climate risk."""

        horizon_days = int(payload.get("horizon_days", 30))
        horizon_days = max(7, min(horizon_days, 180))
        history = self._synthetic_history(payload)
        future = self._project_future(history, horizon_days)
        enriched = add_climate_features(pd.concat([history, future], ignore_index=True)).tail(horizon_days)

        forecast_points = [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "severity_score": round(float(row.drought_severity_score), 2),
                "drought_class": classify_drought_score(float(row.drought_severity_score)),
                "rainfall": round(float(row.rainfall), 2),
                "temperature": round(float(row.temperature), 2),
            }
            for row in enriched.itertuples()
        ]
        mean_risk = float(enriched["drought_severity_score"].mean())
        return {
            "horizon_days": horizon_days,
            "predicted_climate_risk": round(mean_risk, 2),
            "future_drought_trend": self._trend_label(enriched["drought_severity_score"]),
            "forecast": forecast_points,
        }

    @staticmethod
    def _synthetic_history(payload: dict[str, float], days: int = 90) -> pd.DataFrame:
        end = pd.Timestamp.today().normalize()
        dates = pd.date_range(end=end, periods=days, freq="D")
        month_angle = 2 * np.pi * dates.month / 12
        rng = np.random.default_rng(42)
        rainfall_base = float(payload.get("rainfall", 3.0))
        temperature_base = float(payload.get("temperature", 30.0))
        return pd.DataFrame(
            {
                "date": dates,
                "rainfall": np.maximum(0, rainfall_base + 1.8 * np.sin(month_angle) + rng.normal(0, 0.6, days)),
                "temperature": temperature_base + 2.0 * np.cos(month_angle) + rng.normal(0, 0.4, days),
                "humidity": float(payload.get("humidity", 55.0)) + rng.normal(0, 3.5, days),
                "wind_speed": float(payload.get("wind_speed", 2.5)) + rng.normal(0, 0.25, days),
                "solar_radiation": float(payload.get("solar_radiation", 18.0)) + rng.normal(0, 1.0, days),
                "pressure": float(payload.get("pressure", 100.0)) + rng.normal(0, 0.3, days),
            }
        )

    @staticmethod
    def _project_future(history: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        last_date = pd.Timestamp(history["date"].max())
        dates = [last_date + timedelta(days=index) for index in range(1, horizon_days + 1)]
        recent = history.tail(30)
        trend = np.linspace(0, 1, horizon_days)
        seasonal = np.sin(2 * np.pi * pd.DatetimeIndex(dates).dayofyear / 365)
        return pd.DataFrame(
            {
                "date": dates,
                "rainfall": np.maximum(0, recent["rainfall"].mean() - 0.8 * trend + 0.9 * seasonal),
                "temperature": recent["temperature"].mean() + 1.1 * trend + 1.2 * seasonal,
                "humidity": np.clip(recent["humidity"].mean() - 4.0 * trend, 15, 100),
                "wind_speed": recent["wind_speed"].mean() + 0.2 * trend,
                "solar_radiation": recent["solar_radiation"].mean() + 1.0 * trend,
                "pressure": recent["pressure"].mean(),
            }
        )

    @staticmethod
    def _trend_label(scores: pd.Series) -> str:
        slope = np.polyfit(np.arange(len(scores)), scores.to_numpy(), 1)[0]
        if slope > 0.15:
            return "Increasing drought risk"
        if slope < -0.15:
            return "Decreasing drought risk"
        return "Stable drought risk"
