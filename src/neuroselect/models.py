"""Compact convolutional recurrent CTC model."""

from __future__ import annotations

import torch
from torch import nn


class ConvCTC(nn.Module):
    def __init__(self, channels: int, vocabulary_size: int, hidden: int = 64):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.sequence = nn.GRU(
            hidden, hidden // 2, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(hidden, vocabulary_size)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        features = self.frontend(signals.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.sequence(features)
        return self.classifier(encoded)
