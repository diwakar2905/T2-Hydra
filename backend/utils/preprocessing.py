"""Data ingestion and preprocessing utilities for climate features."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

NASA_POWER_PARAMETERS = {
    "PRECTOTCORR": "rainfall",
    "T2M": "temperature",
    "RH2M": "humidity",
    "WS2M": "wind_speed",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
    "PS": "pressure",
}


def fetch_nasa_power_data(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    base_url: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily climate observations from NASA POWER API."""

    params = {
        "parameters": ",".join(NASA_POWER_PARAMETERS.keys()),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = requests.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()["properties"]["parameter"]

    frame = pd.DataFrame(payload).rename(columns=NASA_POWER_PARAMETERS)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    return frame.reset_index()


def clean_climate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema, replace NASA missing markers, and impute gaps."""

    frame = data.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")

    numeric_columns = [column for column in frame.columns if column != "date"]
    frame[numeric_columns] = frame[numeric_columns].replace([-999, -999.0], np.nan)
    frame[numeric_columns] = frame[numeric_columns].interpolate(limit_direction="both")
    frame[numeric_columns] = frame[numeric_columns].fillna(frame[numeric_columns].median())
    return frame


def scale_features(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale selected features and return the fitted scaler."""

    frame = data.copy()
    columns = list(feature_columns)
    fitted_scaler = scaler or StandardScaler()
    frame[columns] = fitted_scaler.fit_transform(frame[columns])
    return frame, fitted_scaler
