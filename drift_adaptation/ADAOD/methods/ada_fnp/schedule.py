'''False-negative predictor schedule for ADA-FNP.'''

from methods.common.protocols.ada_fnp_detection import (
    ACQUISITION_ROUND_COUNT,
)


FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND = 2000


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    if value < 0:
        raise ValueError('{} must not be negative'.format(name))
    return value


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
