"""Model analysis page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_model_analysis(ui: dict[str, object]) -> None:
    st.subheader("Model Performance")
    ui["metric_row"](
        [
            ("Best RMSE", "7.2", "Transfer LSTM"),
            ("Best F1-score", "0.88", "Transfer LSTM"),
            ("Classifier accuracy", "86.4%", "XGBoost baseline"),
            ("Regression R2", "0.82", "LSTM"),
        ]
    )
    st.plotly_chart(ui["model_comparison_chart"](), use_container_width=True)

    confusion = pd.DataFrame(
        [[124, 14, 4], [12, 91, 13], [3, 16, 72]],
        index=["No Drought", "Mild Drought", "Severe Drought"],
        columns=["No Drought", "Mild Drought", "Severe Drought"],
    )
    fig = px.imshow(confusion, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
