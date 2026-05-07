"""Streamlit metric card helpers."""

from __future__ import annotations

import streamlit as st


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        column.metric(label, value, delta)
