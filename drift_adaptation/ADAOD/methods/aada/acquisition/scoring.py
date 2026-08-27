'''AADA entropy-weighted domain-diversity acquisition score.'''

import math
from dataclasses import dataclass

from methods.common.data.image_identity import SampleIdentity


@dataclass(frozen=True)
class RawAadaScore:
    sample: SampleIdentity
    entropy: float
    diversity: float
    source_domain_probability: float
    detection_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample, SampleIdentity):
            raise TypeError('AADA score sample must be a SampleIdentity')
        values = (
            self.entropy,
            self.diversity,
            self.source_domain_probability,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError('AADA score components must be finite')
        if any(float(value) < 0 for value in values):
            raise ValueError('AADA score components must not be negative')
        if self.source_domain_probability > 1:
            raise ValueError('source-domain probability must not exceed one')
        if isinstance(self.detection_count, bool) or not isinstance(
            self.detection_count,
            int,
        ):
            raise TypeError('detection_count must be an integer')
        if self.detection_count < 0:
            raise ValueError('detection_count must not be negative')
        if self.detection_count == 0 and self.entropy != 0:
            raise ValueError('empty detections require zero entropy')

    @property
    def final_score(self) -> float:
        if self.detection_count == 0:
            return 0.0
        return float(self.entropy) * float(self.diversity)
