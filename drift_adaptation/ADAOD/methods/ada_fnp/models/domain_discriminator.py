'''Progressive-DA image-level domain discriminator.'''

import torch
from torch import nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 512, hidden_channels: int = 64):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(3):
            layers.extend((
                nn.Conv2d(channels, hidden_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ))
            channels = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)

    def source_probability(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).sigmoid().mean(dim=(-2, -1))
