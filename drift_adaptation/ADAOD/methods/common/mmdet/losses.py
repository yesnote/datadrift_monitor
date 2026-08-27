'''Small loss-dictionary operations shared by ADA detectors.'''

from typing import Dict, Mapping


def prefix_losses(
    losses: Mapping[str, object],
    prefix: str,
    weight: float = 1.0,
) -> Dict[str, object]:
    if weight < 0:
        raise ValueError('loss weight must be non-negative')
    return {
        '{}.{}'.format(prefix, key): value * weight
        for key, value in losses.items()
    }
