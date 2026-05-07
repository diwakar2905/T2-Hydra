"""Explainability page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages.common import climate_input_form, fallback_prediction


def render_explainability(api_url: str, ui: dict[str, object]) -> None:
    st.subheader("Explainable AI")
    payload = climate_input_form("explain")

    if st.button("Explain prediction", type="primary"):
        try:
            explanation = ui["post_json"](api_url, "/explain", payload)
        except Exception as exc:
            st.warning(f"Backend unavailable, using local explanation: {exc}")
            prediction = fallback_prediction(payload)
            explanation = {
                "base_value": 0.5,
                "feature_contributions": [
                    {
                        "feature": factor["feature"],
                        "shap_value": factor["impact"] / 100,
                        "feature_value": factor["impact"],
                        "direction": "increases risk",
                    }
                    for factor in prediction["risk_factors"]
                ],
                "narrative": "The prediction is primarily driven by rainfall deficit, heat stress, and humidity deficit.",
            }

        st.info(explanation["narrative"])
        contributions = pd.DataFrame(explanation["feature_contributions"])
        st.plotly_chart(ui["feature_importance_chart"](explanation["feature_contributions"]), use_container_width=True)
        st.dataframe(contributions, use_container_width=True, hide_index=True)
