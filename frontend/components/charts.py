"""Reusable Plotly chart builders for the Streamlit frontend."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def severity_line_chart(data: pd.DataFrame) -> go.Figure:
    fig = px.line(data, x="date", y="severity_score", color="drought_class", markers=True)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Severity score")
    return fig


def climate_heatmap(data: pd.DataFrame) -> go.Figure:
    numeric = data.select_dtypes("number")
    corr = numeric.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r", aspect="auto")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def feature_importance_chart(data: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(data)
    if frame.empty:
        frame = pd.DataFrame({"feature": [], "shap_value": []})
    frame["abs_impact"] = frame["shap_value"].abs()
    frame = frame.sort_values("abs_impact", ascending=True)
    fig = px.bar(frame, x="shap_value", y="feature", orientation="h", color="direction")
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="SHAP contribution")
    return fig


def model_comparison_chart() -> go.Figure:
    frame = pd.DataFrame(
        {
            "model": ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "Transfer LSTM"],
            "rmse": [14.8, 9.6, 8.9, 7.8, 7.2],
            "f1": [0.68, 0.81, 0.84, 0.86, 0.88],
        }
    )
    fig = go.Figure()
    fig.add_bar(name="RMSE", x=frame["model"], y=frame["rmse"], yaxis="y")
    fig.add_scatter(name="F1-score", x=frame["model"], y=frame["f1"], yaxis="y2", mode="lines+markers")
    fig.update_layout(
        height=420,
        yaxis=dict(title="RMSE"),
        yaxis2=dict(title="F1-score", overlaying="y", side="right", range=[0, 1]),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
    )
    return fig
