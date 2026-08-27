'''Detector schedule used for ADA-FNP paper comparisons.'''

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


INITIAL_DETECTOR_SEGMENT = (0, 5000)
ADAPTATION_DETECTOR_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (5000, 10000),
    (10000, 15000),
    (15000, 20000),
    (20000, 25000),
    (25000, 40000),
)
DETECTOR_TRAINING_SEGMENTS = (
    INITIAL_DETECTOR_SEGMENT,
    *ADAPTATION_DETECTOR_SEGMENTS,
)
DETECTOR_CHECKPOINT_ITERATIONS = tuple(
    end_iteration for _, end_iteration in DETECTOR_TRAINING_SEGMENTS
)
ACQUISITION_MILESTONES = tuple(
    start_iteration
    for start_iteration, _ in ADAPTATION_DETECTOR_SEGMENTS
)
ACQUISITION_ROUND_COUNT = len(ACQUISITION_MILESTONES)
MAXIMUM_DETECTOR_ITERATION = ADAPTATION_DETECTOR_SEGMENTS[-1][1]


class DetectorTrainingMode(str, Enum):
    INITIALIZATION = 'initialization'
    UNLABELED_ADAPTATION = 'unlabeled_adaptation'
    ADAPTATION = 'adaptation'


@dataclass(frozen=True)
class DetectorTrainingPhase:
    start_iteration: int
    end_iteration: int
    mode: DetectorTrainingMode


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    if value < 0:
        raise ValueError('{} must not be negative'.format(name))
    return value


def resolve_detector_training_phase(
    start_iteration: int,
    end_iteration: int,
    labeled_sample_count: int,
) -> DetectorTrainingPhase:
    '''Resolve one segment of the ADA-FNP 40k comparison protocol.'''

    start_iteration = _non_negative_integer(
        start_iteration,
        'start_iteration',
    )
    end_iteration = _non_negative_integer(end_iteration, 'end_iteration')
    labeled_sample_count = _non_negative_integer(
        labeled_sample_count,
        'labeled_sample_count',
    )
    segment = (start_iteration, end_iteration)
    if segment == INITIAL_DETECTOR_SEGMENT:
        if labeled_sample_count != 0:
            raise ValueError(
                'initial detector training must precede target acquisition'
            )
        return DetectorTrainingPhase(
            start_iteration,
            end_iteration,
            DetectorTrainingMode.INITIALIZATION,
        )
    if segment not in ADAPTATION_DETECTOR_SEGMENTS:
        raise ValueError('unsupported detector-training segment')
    mode = (
        DetectorTrainingMode.ADAPTATION
        if labeled_sample_count
        else DetectorTrainingMode.UNLABELED_ADAPTATION
    )
    return DetectorTrainingPhase(start_iteration, end_iteration, mode)
