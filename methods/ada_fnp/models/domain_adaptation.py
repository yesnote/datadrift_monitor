'''Domain-adaptation model and loss components for ADA-FNP.'''

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional


class _GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, inputs: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.neg().mul(ctx.scale), None


def gradient_reverse(inputs: Tensor, scale: float = 1.0) -> Tensor:
    '''Return ``inputs`` while multiplying its backward gradient by ``-scale``.'''
    if scale < 0:
        raise ValueError('gradient reversal scale must be non-negative')
    return _GradientReverseFunction.apply(inputs, float(scale))


class AdaFnpDomainDiscriminator(nn.Module):
    '''Progressive-DA image-level domain discriminator used by ADA-FNP.'''

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

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)

    def source_probability(self, features: Tensor) -> Tensor:
        return self(features).sigmoid().mean(dim=(-2, -1))


def domain_discriminator_loss(
    source_logits: Tensor, target_logits: Tensor
) -> Tensor:
    '''Compute balanced source=1 and target=0 discriminator loss.'''
    source_loss = functional.binary_cross_entropy_with_logits(
        source_logits, torch.ones_like(source_logits)
    )
    target_loss = functional.binary_cross_entropy_with_logits(
        target_logits, torch.zeros_like(target_logits)
    )
    return 0.5 * (source_loss + target_loss)


def compute_multi_target_domain_loss(
    source_logits: Tensor, target_logits: Sequence[Tensor]
) -> Tensor:
    '''Average adversarial losses across the available target branches.'''
    if not target_logits:
        raise ValueError('at least one target-domain logit tensor is required')
    losses = [
        domain_discriminator_loss(source_logits, logits)
        for logits in target_logits
    ]
    return torch.stack(losses).mean()
