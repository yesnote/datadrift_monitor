'''Domain and supervised-loss helpers for ADA-FNP.'''

from typing import Dict, Mapping

import torch
from torch.nn import functional as functional


def domain_discriminator_loss(
    source_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    source = functional.binary_cross_entropy_with_logits(
        source_logits, torch.ones_like(source_logits)
    )
    target = functional.binary_cross_entropy_with_logits(
        target_logits, torch.zeros_like(target_logits)
    )
    return 0.5 * (source + target)


def prefix_losses(
    losses: Mapping[str, torch.Tensor], prefix: str, weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    if weight < 0:
        raise ValueError('loss weight must be non-negative')
    return {f'{prefix}.{key}': value * weight for key, value in losses.items()}
