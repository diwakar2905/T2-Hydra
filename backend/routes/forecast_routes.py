"""Forecasting API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.forecasting_service import ForecastingService

forecast_bp = Blueprint("forecast", __name__)
forecasting_service = ForecastingService()


@forecast_bp.post("/forecast")
def forecast():
    """Return future drought trend and predicted climate risk."""

    payload = request.get_json(silent=True) or {}
    result = forecasting_service.forecast(payload)
    return jsonify({"status": "success", "data": result})
