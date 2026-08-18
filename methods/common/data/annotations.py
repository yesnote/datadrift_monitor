'''Oracle annotation access isolated behind a post-acquisition reveal boundary.'''

from __future__ import annotations

from types import MappingProxyType
from typing import Generic, Mapping, Protocol, Sequence, Tuple, TypeVar

from .image_identity import SampleIdentity
from .pool import PoolState


AnnotationT = TypeVar('AnnotationT')


class OracleAnnotationProvider(Protocol, Generic[AnnotationT]):
    '''Interface implemented by an experiment-only annotation oracle.'''

    def reveal(
        self, samples: Sequence[SampleIdentity]
    ) -> Mapping[SampleIdentity, AnnotationT]:
        '''Return annotations for exactly the samples crossing the reveal boundary.'''


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


def labeled_samples_for_adaptation(
    pool_state: PoolState,
    expected_namespace: str,
) -> Tuple[SampleIdentity, ...]:
    '''Return selected samples, refusing an empty target-labeled sampler request.'''

    selected = selected_samples_for_reveal(pool_state, expected_namespace)
    if not selected:
        raise RuntimeError(
            'target-labeled adaptation sampling is unavailable before acquisition'
        )
    return selected


def reveal_acquired_annotations(
    provider: OracleAnnotationProvider[AnnotationT],
    previous: PoolState,
    current: PoolState,
) -> Mapping[SampleIdentity, AnnotationT]:
    '''Reveal only the samples acquired by one validated pool transition.'''

    if previous.universe != current.universe:
        raise ValueError('pool transition changed the target universe')
    if previous.round_budgets != current.round_budgets:
        raise ValueError('pool transition changed the budget plan')
    if len(current.history) != len(previous.history) + 1:
        raise ValueError('annotation reveal requires exactly one acquisition transition')
    if current.history[:-1] != previous.history:
        raise ValueError('pool transition rewrote acquisition history')
    if current.labeled[: len(previous.labeled)] != previous.labeled:
        raise ValueError('pool transition rewrote the labeled pool')

    acquired = current.history[-1].selected
    if not acquired:
        return MappingProxyType({})
    revealed = dict(provider.reveal(acquired))
    acquired_set = set(acquired)
    if set(revealed) != acquired_set:
        missing = acquired_set - set(revealed)
        extra = set(revealed) - acquired_set
        raise ValueError(
            'oracle returned the wrong sample set (missing={}, extra={})'.format(
                sorted(sample.qualified_id for sample in missing),
                sorted(sample.qualified_id for sample in extra),
            )
        )
    return MappingProxyType({sample: revealed[sample] for sample in acquired})
