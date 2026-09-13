"""Compact CNN specification for training when PyTorch is installed."""
import torch
from torch import nn

class AudioCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__(); self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); self.head = nn.Linear(64, classes)
    def forward(self, x): return self.head(self.features(x).flatten(1))
