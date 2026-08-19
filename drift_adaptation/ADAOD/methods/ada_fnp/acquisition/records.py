'''Typed ADA-FNP raw scores and common normalization bridge.'''

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from methods.common.acquisition.normalization import standardize_components
from methods.common.acquisition.selection import (
    AcquisitionScore, build_product_scores,
)
from methods.common.data.image_identity import SampleIdentity


@dataclass(frozen=True)
class RawAdaFnpScore:
    sample: SampleIdentity
    false_negative: float
    localization: float
    entropy: float
    diversity: float
    source_domain_probability: float
    detection_count: int

    def __post_init__(self) -> None:
        values = (
            self.false_negative, self.localization, self.entropy,
            self.diversity, self.source_domain_probability,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f'non-finite score for {self.sample.qualified_id}')
        if any(float(value) < 0 for value in values):
            raise ValueError(f'negative score for {self.sample.qualified_id}')
        if self.source_domain_probability > 1:
            raise ValueError(
                f'invalid source probability for {self.sample.qualified_id}'
            )
        if self.detection_count < 0:
            raise ValueError('detection_count must not be negative')
        if self.detection_count == 0 and (
            self.localization != 0 or self.entropy != 0
        ):
            raise ValueError('empty detections require zero localization and entropy')


def normalize_scores(
    records: Sequence[RawAdaFnpScore],
    *,
    constant_component_value: float = 0.5,
    empty_detection_score: float = 0.0,
) -> Tuple[AcquisitionScore, ...]:
    samples = [record.sample for record in records]
    if len(samples) != len(set(samples)):
        raise ValueError('raw scores contain duplicate samples')
    components = {
        'false_negative': {
            record.sample: record.false_negative for record in records
        },
        'localization': {record.sample: record.localization for record in records},
        'entropy': {record.sample: record.entropy for record in records},
        'diversity': {record.sample: record.diversity for record in records},
    }
    normalized = standardize_components(
        components, constant_component_value=constant_component_value
    )
    detection_counts = {
        record.sample: record.detection_count for record in records
    }
    return build_product_scores(
        normalized,
        detection_counts,
        empty_detection_score=empty_detection_score,
    )
