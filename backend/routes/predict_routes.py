"""Prediction API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.prediction_service import PredictionService

predict_bp = Blueprint("predict", __name__)
prediction_service = PredictionService()


@predict_bp.post("/predict")
def predict():
    """Return drought probability, severity, and confidence."""

    payload = request.get_json(silent=True) or {}
    result = prediction_service.predict(payload)
    return jsonify({"status": "success", "data": result})
