'''Deterministic multiplicative acquisition scoring and exact top-K selection.'''

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

from methods.common.data.image_identity import SampleIdentity


def _score_value(value: float, component_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('component {} must be numeric'.format(component_name))
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError('component {} must be finite'.format(component_name))
    if converted < 0.0:
        raise ValueError('component {} must not be negative'.format(component_name))
    return converted


@dataclass(frozen=True)
class AcquisitionScore:
    '''Normalized score components and their deterministic final product.'''

    sample: SampleIdentity
    components: Mapping[str, float]
    detection_count: int
    empty_detection_score: float = 0.0
    final_score: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, SampleIdentity):
            raise TypeError('score sample must be a SampleIdentity')
        if isinstance(self.detection_count, bool) or not isinstance(
            self.detection_count, int
        ):
            raise TypeError('detection_count must be an integer')
        if self.detection_count < 0:
            raise ValueError('detection_count must not be negative')
        empty_detection_score = _score_value(
            self.empty_detection_score, 'empty_detection_score'
        )
        object.__setattr__(self, 'empty_detection_score', empty_detection_score)
        if not self.components:
            raise ValueError('at least one normalized component is required')
        normalized_components = {}
        for component_name in sorted(self.components):
            if not isinstance(component_name, str) or not component_name:
                raise ValueError('component names must be non-empty strings')
            normalized_components[component_name] = _score_value(
                self.components[component_name], component_name
            )
        object.__setattr__(
            self, 'components', MappingProxyType(normalized_components)
        )
        final_score = (
            empty_detection_score
            if self.detection_count == 0
            else math.prod(normalized_components[name] for name in sorted(normalized_components))
        )
        object.__setattr__(self, 'final_score', final_score)

def build_product_scores(
    components: Mapping[str, Mapping[SampleIdentity, float]],
    detection_counts: Mapping[SampleIdentity, int],
    *,
    empty_detection_score: float = 0.0,
) -> Tuple[AcquisitionScore, ...]:
    '''Combine same-domain normalized components for every unlabeled sample.'''

    if not components:
        raise ValueError('at least one score component is required')
    reference_samples = set(detection_counts)
    if any(not isinstance(sample, SampleIdentity) for sample in reference_samples):
        raise TypeError('detection-count keys must be SampleIdentity instances')
    for component_name, sample_scores in components.items():
        if not isinstance(component_name, str) or not component_name:
            raise ValueError('component names must be non-empty strings')
        if set(sample_scores) != reference_samples:
            raise ValueError('components and detection counts must cover the same samples')
    return tuple(
        AcquisitionScore(
            sample=sample,
            components={
                component_name: components[component_name][sample]
                for component_name in sorted(components)
            },
            detection_count=detection_counts[sample],
            empty_detection_score=empty_detection_score,
        )
        for sample in sorted(reference_samples)
    )


def select_top_k(
    scores: Sequence[AcquisitionScore],
    k: int,
) -> Tuple[AcquisitionScore, ...]:
    '''Select exactly K records, breaking score ties by sample_id then namespace.'''

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError('k must be an integer')
    if k < 0:
        raise ValueError('k must not be negative')
    records = tuple(scores)
    if any(not isinstance(record, AcquisitionScore) for record in records):
        raise TypeError('scores must contain AcquisitionScore records')
    samples = [record.sample for record in records]
    if len(samples) != len(set(samples)):
        raise ValueError('scores contain duplicate samples')
    if k > len(records):
        raise ValueError('cannot select more records than are available')
    ranked = sorted(
        records,
        key=lambda record: (
            -record.final_score,
            record.sample.sample_id,
            record.sample.namespace,
        ),
    )
    return tuple(ranked[:k])
