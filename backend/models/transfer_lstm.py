"""Transfer-learning wrapper for LSTM drought models."""

from __future__ import annotations

from torch import nn

from models.lstm_model import DroughtLSTM


class TransferDroughtLSTM(nn.Module):
    """Fine-tune an LSTM encoder with a task-specific prediction head."""

    def __init__(self, base_model: DroughtLSTM, output_size: int = 1, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = base_model.lstm
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        hidden_size = base_model.lstm.hidden_size
        self.transfer_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs)
        return self.transfer_head(outputs[:, -1, :])
