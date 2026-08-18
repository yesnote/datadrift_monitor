from pathlib import Path

import pytest

from methods.ada_fnp.phases import (
    ADAPTATION_SEGMENTS,
    DetectorStageMode,
    LabeledTargetManifest,
    labeled_manifest_for_sampler,
    resolve_detector_phase,
    resolve_fnpm_round_phase,
)


def test_initial_phase_is_strict_and_has_no_target_labeled_sampler() -> None:
    phase = resolve_detector_phase(0, 5000, 0)

    assert phase.mode is DetectorStageMode.INITIALIZATION
    assert phase.initialize_teacher_at_end
    assert not phase.use_pseudo_labels
    assert not phase.use_target_labeled_sampler
    assert labeled_manifest_for_sampler(phase, None) is None

    with pytest.raises(ValueError, match='precede target acquisition'):
        resolve_detector_phase(0, 5000, 1)


@pytest.mark.parametrize(('start_iteration', 'end_iteration'), ADAPTATION_SEGMENTS)
def test_later_segments_use_adaptation_mode(
    start_iteration: int,
    end_iteration: int,
) -> None:
    phase = resolve_detector_phase(start_iteration, end_iteration, 2)

    assert phase.mode is DetectorStageMode.ADAPTATION
    assert not phase.initialize_teacher_at_end
    assert phase.use_pseudo_labels
    assert phase.use_target_labeled_sampler


def test_adaptation_does_not_request_empty_labeled_manifest() -> None:
    phase = resolve_detector_phase(5000, 10000, 0)

    assert not phase.use_target_labeled_sampler
    assert labeled_manifest_for_sampler(phase, None) is None


def test_sampler_requires_matching_nonempty_manifest() -> None:
    phase = resolve_detector_phase(5000, 10000, 2)
    empty = LabeledTargetManifest(Path('empty.json'), '0' * 64, 0, 0)
    mismatched = LabeledTargetManifest(Path('one.json'), '1' * 64, 1, 1)
    matching = LabeledTargetManifest(Path('two.json'), '2' * 64, 2, 1)

    with pytest.raises(RuntimeError, match='nonempty'):
        labeled_manifest_for_sampler(phase, empty)
    with pytest.raises(ValueError, match='does not match'):
        labeled_manifest_for_sampler(phase, mismatched)
    assert labeled_manifest_for_sampler(phase, matching) == Path('two.json')


@pytest.mark.parametrize(
    ('start_iteration', 'end_iteration'),
    ((0, 4999), (1, 5000), (5000, 9999), (25000, 39999), (40000, 45000)),
)
def test_invalid_detector_segments_are_rejected(
    start_iteration: int,
    end_iteration: int,
) -> None:
    with pytest.raises(ValueError, match='segment'):
        resolve_detector_phase(start_iteration, end_iteration, 0)


def test_phase_counts_require_nonnegative_integers() -> None:
    with pytest.raises(TypeError):
        resolve_detector_phase(0.0, 5000, 0)
    with pytest.raises(ValueError, match='negative'):
        resolve_detector_phase(5000, 10000, -1)


@pytest.mark.parametrize(
    ('round_index', 'detector_iteration'),
    ((1, 5000), (2, 10000), (3, 15000), (4, 20000), (5, 25000)),
)
def test_fnpm_round_phase_matches_detector_milestone(
    round_index: int,
    detector_iteration: int,
) -> None:
    phase = resolve_fnpm_round_phase(round_index, detector_iteration, iteration=150)

    assert phase.round_index == round_index
    assert phase.detector_iteration == detector_iteration
    assert phase.iteration == 150
    assert phase.max_iterations == 2000


def test_fnpm_round_phase_rejects_wrong_milestone_and_resume_position() -> None:
    with pytest.raises(ValueError, match='milestone'):
        resolve_fnpm_round_phase(2, 5000)
    with pytest.raises(ValueError, match='exceed'):
        resolve_fnpm_round_phase(1, 5000, iteration=2001)
