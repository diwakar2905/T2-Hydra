"""API smoke tests for T2-HYDRO."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app import create_app  # noqa: E402


def test_health_endpoint() -> None:
    client = create_app().test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_endpoint() -> None:
    client = create_app().test_client()
    response = client.post(
        "/predict",
        json={
            "rainfall": 2.0,
            "temperature": 34.0,
            "humidity": 40.0,
            "wind_speed": 3.0,
            "solar_radiation": 22.0,
            "pressure": 101.0,
        },
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert "drought_probability" in payload["data"]


def test_forecast_endpoint() -> None:
    client = create_app().test_client()
    response = client.post("/forecast", json={"horizon_days": 10})
    payload = response.get_json()
    assert response.status_code == 200
    assert len(payload["data"]["forecast"]) == 10
