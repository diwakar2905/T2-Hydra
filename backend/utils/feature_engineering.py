"""Feature engineering for drought analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "rainfall",
    "temperature",
    "humidity",
    "wind_speed",
    "solar_radiation",
    "pressure",
]

MODEL_FEATURES = [
    *BASE_FEATURES,
    "month_sin",
    "month_cos",
    "season_code",
    "rainfall_7d_avg",
    "rainfall_30d_avg",
    "temperature_7d_avg",
    "temperature_30d_avg",
    "rainfall_trend",
    "temperature_trend",
    "rainfall_anomaly",
    "temperature_anomaly",
    "climate_stress_index",
]


def add_climate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, rolling, anomaly, and drought stress features."""

    frame = data.copy()
    if "date" not in frame.columns:
        frame["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(frame), freq="D")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    month = frame["date"].dt.month
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["season_code"] = month.map(_season_code).astype(int)

    frame["rainfall_7d_avg"] = frame["rainfall"].rolling(7, min_periods=1).mean()
    frame["rainfall_30d_avg"] = frame["rainfall"].rolling(30, min_periods=1).mean()
    frame["temperature_7d_avg"] = frame["temperature"].rolling(7, min_periods=1).mean()
    frame["temperature_30d_avg"] = frame["temperature"].rolling(30, min_periods=1).mean()
    frame["rainfall_trend"] = frame["rainfall"].diff().rolling(7, min_periods=1).mean().fillna(0)
    frame["temperature_trend"] = frame["temperature"].diff().rolling(7, min_periods=1).mean().fillna(0)

    rainfall_baseline = frame.groupby(month)["rainfall"].transform("mean")
    temperature_baseline = frame.groupby(month)["temperature"].transform("mean")
    frame["rainfall_anomaly"] = frame["rainfall"] - rainfall_baseline
    frame["temperature_anomaly"] = frame["temperature"] - temperature_baseline
    frame["climate_stress_index"] = calculate_climate_stress_index(frame)
    frame["drought_severity_score"] = calculate_drought_severity_score(frame)
    frame["drought_class"] = frame["drought_severity_score"].map(classify_drought_score)
    return frame


def calculate_climate_stress_index(data: pd.DataFrame) -> pd.Series:
    """Compute a 0-1 climate stress proxy from water deficit and heat load."""

    rainfall_deficit = _minmax(-data["rainfall_anomaly"])
    heat_anomaly = _minmax(data["temperature_anomaly"])
    humidity_deficit = _minmax(100 - data["humidity"])
    wind_load = _minmax(data["wind_speed"])
    solar_load = _minmax(data["solar_radiation"])
    return (
        0.35 * rainfall_deficit
        + 0.25 * heat_anomaly
        + 0.15 * humidity_deficit
        + 0.15 * solar_load
        + 0.10 * wind_load
    ).clip(0, 1)


def calculate_drought_severity_score(data: pd.DataFrame) -> pd.Series:
    """Estimate drought severity on a 0-100 scale."""

    rainfall_component = _minmax(-data["rainfall_anomaly"]) * 45
    heat_component = _minmax(data["temperature_anomaly"]) * 25
    humidity_component = _minmax(100 - data["humidity"]) * 20
    trend_component = _minmax(-data["rainfall_trend"]) * 10
    return (rainfall_component + heat_component + humidity_component + trend_component).clip(0, 100)


def classify_drought_score(score: float) -> str:
    """Map a drought severity score to the supported class labels."""

    if score >= 60:
        return "Severe Drought"
    if score >= 35:
        return "Mild Drought"
    return "No Drought"


def prepare_single_observation(payload: dict[str, float]) -> pd.DataFrame:
    """Convert an API payload into a feature-complete single-row DataFrame."""

    today = pd.Timestamp.today().normalize()
    defaults = {
        "rainfall": 3.0,
        "temperature": 30.0,
        "humidity": 55.0,
        "wind_speed": 2.5,
        "solar_radiation": 18.0,
        "pressure": 100.0,
    }
    row = {key: float(payload.get(key, value)) for key, value in defaults.items()}
    frame = pd.DataFrame([{**row, "date": today}])
    return add_climate_features(frame)


def _season_code(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8, 9):
        return 2
    return 3


def _minmax(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)
