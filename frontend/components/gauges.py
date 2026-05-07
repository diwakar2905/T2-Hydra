"""Gauge components for climate risk indicators."""

from __future__ import annotations

import plotly.graph_objects as go


def risk_gauge(value: float, title: str = "Drought Risk") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig
