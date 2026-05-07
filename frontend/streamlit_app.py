"""T2-HYDRO Streamlit frontend."""

from __future__ import annotations

import requests
import streamlit as st

from components.cards import metric_row
from components.charts import climate_heatmap, feature_importance_chart, model_comparison_chart, severity_line_chart
from components.gauges import risk_gauge
from components.sidebar import configure_sidebar
from pages.dashboard import render_dashboard
from pages.explainability import render_explainability
from pages.forecasting import render_forecasting
from pages.model_analysis import render_model_analysis
from pages.prediction import render_prediction

st.set_page_config(page_title="T2-HYDRO", page_icon="TH", layout="wide")


def post_json(api_url: str, path: str, payload: dict[str, float]) -> dict[str, object]:
    response = requests.post(f"{api_url}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"]


def main() -> None:
    api_url = configure_sidebar()
    st.title("T2-HYDRO")
    st.caption("AI drought prediction, forecasting, and climate analytics")

    page = st.sidebar.radio(
        "Navigation",
        ["Home Dashboard", "Prediction", "Forecasting", "Explainability", "Model Performance"],
    )

    shared = {
        "post_json": post_json,
        "metric_row": metric_row,
        "risk_gauge": risk_gauge,
        "severity_line_chart": severity_line_chart,
        "climate_heatmap": climate_heatmap,
        "feature_importance_chart": feature_importance_chart,
        "model_comparison_chart": model_comparison_chart,
    }

    if page == "Home Dashboard":
        render_dashboard(api_url, shared)
    elif page == "Prediction":
        render_prediction(api_url, shared)
    elif page == "Forecasting":
        render_forecasting(api_url, shared)
    elif page == "Explainability":
        render_explainability(api_url, shared)
    else:
        render_model_analysis(shared)


if __name__ == "__main__":
    main()
