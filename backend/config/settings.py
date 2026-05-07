"""Application configuration for T2-HYDRO."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = "T2-HYDRO"
    environment: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_dir: str = os.getenv("MODEL_DIR", "saved_models")
    nasa_power_base_url: str = os.getenv(
        "NASA_POWER_BASE_URL",
        "https://power.larc.nasa.gov/api/temporal/daily/point",
    )
    cors_origin: str = os.getenv("CORS_ORIGIN", "*")


settings = Settings()
