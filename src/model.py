import torch
from torch import nn


class LandmarkClassifier(nn.Module):
    """MLP classifier for 21 normalized 3D hand landmarks (63 features)."""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)
