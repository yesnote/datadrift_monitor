'''Iteration, round, and budget rules for the ADA-FNP workflow.'''

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple


INITIAL_DETECTOR_SEGMENT = (0, 5000)
ADAPTATION_DETECTOR_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (5000, 10000),
    (10000, 15000),
    (15000, 20000),
    (20000, 25000),
    (25000, 40000),
)
ACQUISITION_MILESTONES: Tuple[int, ...] = tuple(
    start_iteration
    for start_iteration, _ in ADAPTATION_DETECTOR_SEGMENTS
)
ACQUISITION_ROUND_COUNT = len(ACQUISITION_MILESTONES)
MAXIMUM_DETECTOR_ITERATION = ADAPTATION_DETECTOR_SEGMENTS[-1][1]
FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND = 2000


class DetectorTrainingMode(str, Enum):
    INITIALIZATION = 'initialization'
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
    '''Resolve one supported detector-training segment.'''

    start_iteration = _non_negative_integer(
        start_iteration, 'start_iteration'
    )
    end_iteration = _non_negative_integer(end_iteration, 'end_iteration')
    labeled_sample_count = _non_negative_integer(
        labeled_sample_count, 'labeled_sample_count'
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
    return DetectorTrainingPhase(
        start_iteration,
        end_iteration,
        DetectorTrainingMode.ADAPTATION,
    )


def validate_false_negative_training_round(
    round_index: int,
    iteration: int = 0,
) -> None:
    '''Validate one false-negative predictor training position.'''

    round_index = _non_negative_integer(round_index, 'round_index')
    iteration = _non_negative_integer(iteration, 'iteration')
    if not 1 <= round_index <= ACQUISITION_ROUND_COUNT:
        raise ValueError(
            'round_index must be between 1 and {}'.format(
                ACQUISITION_ROUND_COUNT
            )
        )
    if iteration > FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND:
        raise ValueError('false-negative training iteration exceeds its round')


def resolve_total_budget(config: Mapping) -> int:
    '''Resolve the exact integer annotation budget for one scenario.'''

    acquisition = config['acquisition']
    if 'total_budget' in acquisition:
        return int(acquisition['total_budget'])
    target_size = int(config['dataset']['target']['expected_train_images'])
    percentage = float(acquisition['budget_percent'])
    if not 0.0 <= percentage <= 100.0:
        raise ValueError('budget_percent must be between 0 and 100')
    return int(target_size * percentage / 100.0 + 0.5)
