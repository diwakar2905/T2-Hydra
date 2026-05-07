"""PyTorch LSTM models for drought severity forecasting."""

from __future__ import annotations

import torch
from torch import nn


class DroughtLSTM(nn.Module):
    """Sequence model for drought severity regression or classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass over climate sequences."""

        outputs, _ = self.lstm(inputs)
        last_hidden_state = outputs[:, -1, :]
        return self.head(last_hidden_state)
