'''Pure ADA-FNP phase validation and selected-annotation reveal helpers.'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from methods.common.data.cityscapes import (
    LabeledTargetManifest,
    materialize_target_train_labeled,
)
from methods.common.data.pool import PoolState


INITIAL_SEGMENT = (0, 5000)
ADAPTATION_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (5000, 10000),
    (10000, 15000),
    (15000, 20000),
    (20000, 25000),
    (25000, 40000),
)
FNPM_MILESTONES: Tuple[int, ...] = (5000, 10000, 15000, 20000, 25000)
FNPM_ITERATIONS_PER_ROUND = 2000


class DetectorStageMode(str, Enum):
    INITIALIZATION = 'initialization'
    ADAPTATION = 'adaptation'


@dataclass(frozen=True)
class DetectorPhase:
    '''Validated detector-training segment and its dataset access mode.'''

    start_iteration: int
    end_iteration: int
    labeled_sample_count: int
    mode: DetectorStageMode
    initialize_teacher_at_end: bool
    use_pseudo_labels: bool
    use_target_labeled_sampler: bool


@dataclass(frozen=True)
class FnpmRoundPhase:
    '''Validated position inside one fresh 2k FNPM optimization round.'''

    round_index: int
    detector_iteration: int
    iteration: int
    max_iterations: int = FNPM_ITERATIONS_PER_ROUND


def _iteration(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    if value < 0:
        raise ValueError('{} must not be negative'.format(name))
    return value


def resolve_detector_phase(
    start_iteration: int,
    end_iteration: int,
    labeled_sample_count: int,
) -> DetectorPhase:
    '''Resolve the strict 0-to-5k initialization or a later adaptation segment.'''

    start_iteration = _iteration(start_iteration, 'start_iteration')
    end_iteration = _iteration(end_iteration, 'end_iteration')
    labeled_sample_count = _iteration(labeled_sample_count, 'labeled_sample_count')
    segment = (start_iteration, end_iteration)
    if segment == INITIAL_SEGMENT:
        if labeled_sample_count != 0:
            raise ValueError('0-to-5k initialization must precede target acquisition')
        return DetectorPhase(
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            labeled_sample_count=0,
            mode=DetectorStageMode.INITIALIZATION,
            initialize_teacher_at_end=True,
            use_pseudo_labels=False,
            use_target_labeled_sampler=False,
        )
    if segment not in ADAPTATION_SEGMENTS:
        raise ValueError(
            'detector segment must be 0-to-5k or one configured later segment'
        )
    return DetectorPhase(
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        labeled_sample_count=labeled_sample_count,
        mode=DetectorStageMode.ADAPTATION,
        initialize_teacher_at_end=False,
        use_pseudo_labels=True,
        use_target_labeled_sampler=labeled_sample_count > 0,
    )


def resolve_fnpm_round_phase(
    round_index: int,
    detector_iteration: int,
    iteration: int = 0,
) -> FnpmRoundPhase:
    '''Validate the round/milestone pair and its resumable local iteration.'''

    round_index = _iteration(round_index, 'round_index')
    detector_iteration = _iteration(detector_iteration, 'detector_iteration')
    iteration = _iteration(iteration, 'iteration')
    if not 1 <= round_index <= len(FNPM_MILESTONES):
        raise ValueError('FNPM round_index must be between 1 and 5')
    expected_milestone = FNPM_MILESTONES[round_index - 1]
    if detector_iteration != expected_milestone:
        raise ValueError('FNPM round does not match its detector milestone')
    if iteration > FNPM_ITERATIONS_PER_ROUND:
        raise ValueError('FNPM local iteration must not exceed 2000')
    return FnpmRoundPhase(round_index, detector_iteration, iteration)


@dataclass(frozen=True)
class RevealRequest:
    '''Typed input to the C-to-F reveal executor.'''

    oracle_json_path: Path
    output_path: Path
    pool_state: PoolState

    def __post_init__(self) -> None:
        object.__setattr__(self, 'oracle_json_path', Path(self.oracle_json_path))
        object.__setattr__(self, 'output_path', Path(self.output_path))
        if not isinstance(self.pool_state, PoolState):
            raise TypeError('pool_state must be a PoolState')


def execute_reveal(request: RevealRequest) -> LabeledTargetManifest:
    '''Materialize the target-labeled dataset for one committed pool state.'''

    if not isinstance(request, RevealRequest):
        raise TypeError('request must be a RevealRequest')
    return materialize_target_train_labeled(
        request.oracle_json_path,
        request.pool_state,
        request.output_path,
    )


def labeled_manifest_for_sampler(
    phase: DetectorPhase,
    manifest: Optional[LabeledTargetManifest],
) -> Optional[Path]:
    '''Return a labeled manifest only when the phase may sample selected targets.'''

    if not isinstance(phase, DetectorPhase):
        raise TypeError('phase must be a DetectorPhase')
    if not phase.use_target_labeled_sampler:
        return None
    if manifest is None or not manifest.has_samples:
        raise RuntimeError('adaptation requested a nonempty target-labeled manifest')
    if manifest.image_count != phase.labeled_sample_count:
        raise ValueError('phase labeled count does not match the revealed manifest')
    return manifest.path
