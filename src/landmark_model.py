from __future__ import annotations
import torch
from torch import nn


class LandmarkMLP(nn.Module):
    """Classifier for 21 normalized hand landmarks (21 x 3 = 63 inputs)."""
    def __init__(self, num_classes: int, input_dim: int = 63, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_landmark_model(checkpoint, num_classes, device="cpu"):
    model = LandmarkMLP(num_classes)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model
