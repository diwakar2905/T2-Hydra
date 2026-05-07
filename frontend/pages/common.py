"""Shared frontend helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


DEFAULT_PAYLOAD = {
    "rainfall": 2.8,
    "temperature": 33.5,
    "humidity": 42.0,
    "wind_speed": 3.1,
    "solar_radiation": 21.5,
    "pressure": 100.7,
    "horizon_days": 45,
}


def climate_input_form(key_prefix: str = "climate") -> dict[str, float]:
    col1, col2, col3 = st.columns(3)
    payload = {
        "rainfall": col1.number_input("Rainfall (mm/day)", 0.0, 300.0, DEFAULT_PAYLOAD["rainfall"], key=f"{key_prefix}_rainfall"),
        "temperature": col1.number_input("Temperature (C)", -20.0, 60.0, DEFAULT_PAYLOAD["temperature"], key=f"{key_prefix}_temperature"),
        "humidity": col2.number_input("Humidity (%)", 0.0, 100.0, DEFAULT_PAYLOAD["humidity"], key=f"{key_prefix}_humidity"),
        "wind_speed": col2.number_input("Wind speed (m/s)", 0.0, 40.0, DEFAULT_PAYLOAD["wind_speed"], key=f"{key_prefix}_wind"),
        "solar_radiation": col3.number_input(
            "Solar radiation (MJ/m2/day)",
            0.0,
            45.0,
            DEFAULT_PAYLOAD["solar_radiation"],
            key=f"{key_prefix}_solar",
        ),
        "pressure": col3.number_input("Pressure (kPa)", 80.0, 120.0, DEFAULT_PAYLOAD["pressure"], key=f"{key_prefix}_pressure"),
    }
    return payload


def fallback_prediction(payload: dict[str, float]) -> dict[str, object]:
    rainfall_stress = max(0, 8 - payload["rainfall"]) * 6
    heat_stress = max(0, payload["temperature"] - 28) * 3
    humidity_stress = max(0, 55 - payload["humidity"]) * 0.7
    severity_score = min(100, rainfall_stress + heat_stress + humidity_stress)
    drought_class = "Severe Drought" if severity_score >= 60 else "Mild Drought" if severity_score >= 35 else "No Drought"
    return {
        "drought_probability": round(severity_score / 100, 4),
        "drought_severity": drought_class,
        "severity_score": round(severity_score, 2),
        "confidence_score": 0.74,
        "risk_factors": [
            {"feature": "Rainfall deficit", "impact": round(rainfall_stress, 2)},
            {"feature": "Heat stress", "impact": round(heat_stress, 2)},
            {"feature": "Humidity deficit", "impact": round(humidity_stress, 2)},
        ],
    }


def fallback_forecast(payload: dict[str, float], horizon_days: int = 45) -> dict[str, object]:
    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon_days, freq="D")
    base = fallback_prediction(payload)["severity_score"]
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "severity_score": [min(100, max(0, base + index * 0.22)) for index in range(horizon_days)],
        }
    )
    frame["drought_class"] = frame["severity_score"].map(
        lambda score: "Severe Drought" if score >= 60 else "Mild Drought" if score >= 35 else "No Drought"
    )
    frame["rainfall"] = payload["rainfall"]
    frame["temperature"] = payload["temperature"]
    return {
        "horizon_days": horizon_days,
        "predicted_climate_risk": round(float(frame["severity_score"].mean()), 2),
        "future_drought_trend": "Increasing drought risk",
        "forecast": frame.to_dict("records"),
    }


def api_or_fallback(callable_fn, fallback_fn, *args):
    try:
        return callable_fn(*args)
    except Exception as exc:
        st.warning(f"Using local fallback because the backend is unavailable: {exc}")
        return fallback_fn(*args[1:]) if len(args) > 1 else fallback_fn(*args)
