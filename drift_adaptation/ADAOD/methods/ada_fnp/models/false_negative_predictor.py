'''False-negative count predictor and its regression loss.'''

from __future__ import annotations

import torch
from torch import nn


class FalseNegativePredictor(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.regressor(pooled).squeeze(1)


def false_negative_prediction_loss(
    source_prediction: torch.Tensor,
    source_target: torch.Tensor,
    labeled_target_prediction: torch.Tensor | None = None,
    labeled_target: torch.Tensor | None = None,
) -> torch.Tensor:
    '''Compute source MSE plus optional labeled-target MSE.'''

    loss = (source_prediction - source_target).square().mean()
    if labeled_target_prediction is not None or labeled_target is not None:
        if labeled_target_prediction is None or labeled_target is None:
            raise ValueError('both labeled-target tensors are required')
        if labeled_target_prediction.numel() == 0:
            return loss
        loss = loss + (
            labeled_target_prediction - labeled_target
        ).square().mean()
    return loss
