'''Raw ADA-FNP acquisition score components.'''

import torch


def foreground_entropy(
    probabilities: torch.Tensor, background_index: int = -1
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError('probabilities must have shape [boxes, classes]')
    if probabilities.shape[0] == 0:
        return probabilities.new_zeros(())
    background_index %= probabilities.shape[1]
    keep = torch.ones(
        probabilities.shape[1], dtype=torch.bool, device=probabilities.device
    )
    keep[background_index] = False
    foreground = probabilities[:, keep]
    terms = torch.where(
        foreground > 0,
        -foreground * foreground.clamp_min(torch.finfo(foreground.dtype).tiny).log(),
        torch.zeros_like(foreground),
    )
    return terms.sum(dim=1).mean()


def domain_diversity(source_probability: torch.Tensor, epsilon: float = 1e-6):
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError('epsilon must be between zero and one')
    if ((source_probability < 0) | (source_probability > 1)).any():
        raise ValueError('source domain probability must be in [0, 1]')
    probability = source_probability.clamp(min=epsilon, max=1.0)
    score = (1.0 - probability) / probability
    if not torch.isfinite(score).all():
        raise ValueError('domain diversity produced a non-finite value')
    return score
