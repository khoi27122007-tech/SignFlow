from __future__ import annotations
import torch
from torch import nn


class LandmarkGRU(nn.Module):
    """Sequence classifier. Input shape: (batch, frames, 63)."""
    def __init__(self, num_classes: int, input_dim=63, hidden_dim=128,
                 num_layers=2, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1, :])
