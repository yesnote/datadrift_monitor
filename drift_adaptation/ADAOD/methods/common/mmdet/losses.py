'''Small loss-dictionary operations shared by ADA detectors.'''

from typing import Dict, Mapping


def _scale_loss_value(value: object, weight: float) -> object:
    if isinstance(value, list):
        return [item * weight for item in value]
    if isinstance(value, tuple):
        return tuple(item * weight for item in value)
    return value * weight


def prefix_losses(
    losses: Mapping[str, object],
    prefix: str,
    weight: float = 1.0,
) -> Dict[str, object]:
    if weight < 0:
        raise ValueError('loss weight must be non-negative')
    return {
        '{}.{}'.format(prefix, key): _scale_loss_value(value, weight)
        for key, value in losses.items()
    }
