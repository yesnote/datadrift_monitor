'''Oracle annotation access isolated behind a post-acquisition reveal boundary.'''

from __future__ import annotations

from typing import Tuple

from .image_identity import SampleIdentity
from .pool import PoolState


def selected_samples_for_reveal(
    pool_state: PoolState,
    expected_namespace: str,
) -> Tuple[SampleIdentity, ...]:
    '''Return the committed labeled set after validating its dataset namespace.'''

    if not isinstance(pool_state, PoolState):
        raise TypeError('pool_state must be a PoolState')
    if not isinstance(expected_namespace, str) or not expected_namespace:
        raise ValueError('expected_namespace must be a non-empty string')
    wrong_namespace = tuple(
        sample
        for sample in pool_state.universe
        if sample.namespace != expected_namespace
    )
    if wrong_namespace:
        raise ValueError(
            'pool contains a sample from the wrong namespace: {}'.format(
                wrong_namespace[0].qualified_id
            )
        )
    if len(pool_state.labeled) != len(set(pool_state.labeled)):
        raise ValueError('committed labeled pool contains duplicate sample IDs')
    return tuple(sorted(pool_state.labeled, key=lambda sample: sample.sample_id))
