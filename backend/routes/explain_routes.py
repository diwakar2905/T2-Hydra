"""Explainability API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.explainability_service import ExplainabilityService

explain_bp = Blueprint("explain", __name__)
explainability_service = ExplainabilityService()


@explain_bp.post("/explain")
def explain():
    """Return SHAP-style local feature explanations."""

    payload = request.get_json(silent=True) or {}
    result = explainability_service.explain(payload)
    return jsonify({"status": "success", "data": result})
