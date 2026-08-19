'''Numerically explicit acquisition-score normalization.'''

from __future__ import annotations

import math
import statistics
from typing import Dict, Mapping, Sequence, Tuple

from methods.common.data.image_identity import SampleIdentity


def _finite_float(value: float, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('{} must be numeric'.format(context))
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError('{} must be finite'.format(context))
    return converted


def lower_clamped_standardize(
    values: Sequence[float],
    *,
    offset: float = 0.5,
    lower_bound: float = 0.0,
) -> Tuple[float, ...]:
    '''Apply the Eq. 10 transform ``max((x-mean)/(6*std)+offset, lower)``.

    A constant component has no ranking information and is therefore mapped to
    the neutral value ``offset`` (0.5 by default) for every sample.
    '''

    converted = tuple(
        _finite_float(value, 'values[{}]'.format(index))
        for index, value in enumerate(values)
    )
    offset = _finite_float(offset, 'offset')
    lower_bound = _finite_float(lower_bound, 'lower_bound')
    if not converted:
        return ()
    mean = statistics.fmean(converted)
    standard_deviation = statistics.pstdev(converted, mu=mean)
    if standard_deviation == 0.0:
        neutral = max(offset, lower_bound)
        return tuple(neutral for _ in converted)
    return tuple(
        max((value - mean) / (6.0 * standard_deviation) + offset, lower_bound)
        for value in converted
    )


def standardize_components(
    components: Mapping[str, Mapping[SampleIdentity, float]],
    *,
    constant_component_value: float = 0.5,
) -> Dict[str, Dict[SampleIdentity, float]]:
    '''Standardize each component over one identical unlabeled sample set.'''

    if not components:
        raise ValueError('at least one score component is required')
    standardized = {}
    reference_samples = None
    for component_name in sorted(components):
        if not isinstance(component_name, str) or not component_name:
            raise ValueError('component names must be non-empty strings')
        sample_scores = components[component_name]
        sample_set = set(sample_scores)
        if any(not isinstance(sample, SampleIdentity) for sample in sample_set):
            raise TypeError('component score keys must be SampleIdentity instances')
        if reference_samples is None:
            reference_samples = sample_set
        elif sample_set != reference_samples:
            raise ValueError('all score components must cover the same sample set')
        ordered_samples = sorted(sample_set)
        normalized_values = lower_clamped_standardize(
            [sample_scores[sample] for sample in ordered_samples],
            offset=constant_component_value,
        )
        standardized[component_name] = {
            sample: value for sample, value in zip(ordered_samples, normalized_values)
        }
    return standardized
