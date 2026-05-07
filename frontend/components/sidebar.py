"""Sidebar controls shared across Streamlit pages."""

from __future__ import annotations

import os

import streamlit as st


def configure_sidebar() -> str:
    st.sidebar.title("T2-HYDRO")
    api_url = st.sidebar.text_input("Backend API URL", os.getenv("T2_HYDRO_API_URL", "http://localhost:5000"))
    st.sidebar.caption("Use local Flask during development and Render URL after deployment.")
    return api_url.rstrip("/")
