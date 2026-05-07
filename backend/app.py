"""Flask REST API entry point for T2-HYDRO."""

from __future__ import annotations

import logging
from http import HTTPStatus
from werkzeug.exceptions import HTTPException

from flask import Flask, jsonify
from flask_cors import CORS

from config.settings import settings
from routes.explain_routes import explain_bp
from routes.forecast_routes import forecast_bp
from routes.predict_routes import predict_bp


def create_app() -> Flask:
    """Create and configure the Flask application."""

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": settings.cors_origin}})

    app.register_blueprint(predict_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(explain_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": settings.app_name, "environment": settings.environment})

    @app.errorhandler(Exception)
    def handle_exception(error: Exception):
        """Return JSON for HTTP exceptions and a 500 for others.

        This prevents common 404s from being reported as internal server
        errors in development (useful for health checks and missing routes).
        """
        app.logger.exception("Unhandled API error: %s", error)

        if isinstance(error, HTTPException):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": error.name,
                        "detail": str(error),
                    }
                ),
                error.code,
            )

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Unable to process request",
                    "detail": str(error),
                }
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=settings.environment == "development")
