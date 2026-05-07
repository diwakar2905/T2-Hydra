"""Temporal sequence generation for LSTM models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_sequences(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window feature sequences and aligned targets."""

    features = data[feature_columns].to_numpy(dtype=np.float32)
    target = data[target_column].to_numpy(dtype=np.float32)
    x_values: list[np.ndarray] = []
    y_values: list[float] = []

    for index in range(sequence_length, len(data)):
        x_values.append(features[index - sequence_length : index])
        y_values.append(target[index])

    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)
