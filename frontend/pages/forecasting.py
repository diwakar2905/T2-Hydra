"""Forecasting page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pages.common import climate_input_form, fallback_forecast


def render_forecasting(api_url: str, ui: dict[str, object]) -> None:
    st.subheader("Drought Forecasting")
    payload = climate_input_form("forecast")
    payload["horizon_days"] = st.slider("Forecast horizon", 7, 180, 45)

    if st.button("Generate forecast", type="primary"):
        try:
            result = ui["post_json"](api_url, "/forecast", payload)
        except Exception as exc:
            st.warning(f"Backend unavailable, using local fallback: {exc}")
            result = fallback_forecast(payload, int(payload["horizon_days"]))

        frame = pd.DataFrame(result["forecast"])
        ui["metric_row"](
            [
                ("Forecast horizon", f"{result['horizon_days']} days", None),
                ("Climate risk", f"{result['predicted_climate_risk']:.1f}/100", result["future_drought_trend"]),
            ]
        )
        st.plotly_chart(ui["severity_line_chart"](frame), use_container_width=True)
        st.dataframe(frame.tail(15), use_container_width=True, hide_index=True)
