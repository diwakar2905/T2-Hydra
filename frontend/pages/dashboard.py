"""Home dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages.common import DEFAULT_PAYLOAD, fallback_forecast, fallback_prediction


def render_dashboard(api_url: str, ui: dict[str, object]) -> None:
    prediction = _safe_predict(api_url, ui)
    forecast = _safe_forecast(api_url, ui)
    forecast_frame = pd.DataFrame(forecast["forecast"])

    ui["metric_row"](
        [
            ("Drought probability", f"{prediction['drought_probability'] * 100:.1f}%", None),
            ("Severity", str(prediction["drought_severity"]), None),
            ("Forecast risk", f"{forecast['predicted_climate_risk']:.1f}/100", forecast["future_drought_trend"]),
            ("Confidence", f"{prediction['confidence_score'] * 100:.1f}%", None),
        ]
    )
    left, right = st.columns([1, 2])
    left.plotly_chart(ui["risk_gauge"](prediction["severity_score"], "Current drought severity"), use_container_width=True)
    right.plotly_chart(ui["severity_line_chart"](forecast_frame), use_container_width=True)

    st.subheader("Climate Pattern Correlation")
    st.plotly_chart(ui["climate_heatmap"](forecast_frame), use_container_width=True)


def _safe_predict(api_url: str, ui: dict[str, object]) -> dict[str, object]:
    try:
        return ui["post_json"](api_url, "/predict", DEFAULT_PAYLOAD)
    except Exception:
        return fallback_prediction(DEFAULT_PAYLOAD)


def _safe_forecast(api_url: str, ui: dict[str, object]) -> dict[str, object]:
    try:
        return ui["post_json"](api_url, "/forecast", DEFAULT_PAYLOAD)
    except Exception:
        return fallback_forecast(DEFAULT_PAYLOAD)
