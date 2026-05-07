"""Prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages.common import climate_input_form, fallback_prediction


def render_prediction(api_url: str, ui: dict[str, object]) -> None:
    st.subheader("Drought Prediction")
    payload = climate_input_form("prediction")
    if st.button("Run prediction", type="primary"):
        try:
            result = ui["post_json"](api_url, "/predict", payload)
        except Exception as exc:
            st.warning(f"Backend unavailable, using local fallback: {exc}")
            result = fallback_prediction(payload)

        ui["metric_row"](
            [
                ("Probability", f"{result['drought_probability'] * 100:.1f}%", None),
                ("Severity", str(result["drought_severity"]), None),
                ("Confidence", f"{result['confidence_score'] * 100:.1f}%", None),
            ]
        )
        col1, col2 = st.columns([1, 1])
        col1.plotly_chart(ui["risk_gauge"](result["severity_score"], "Severity score"), use_container_width=True)
        col2.dataframe(pd.DataFrame(result["risk_factors"]), use_container_width=True, hide_index=True)
