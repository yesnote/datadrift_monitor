import copy

import pytest
import torch
from torch import nn

from methods.ada_fnp.models.fnpm import FalseNegativePredictionModule, fnpm_loss
from methods.ada_fnp.training.fnpm_trainer import (
    FNPM_ITERATIONS_PER_ROUND,
    FNPM_LEARNING_RATE,
    FnpmSupervision,
    build_fnpm_resume_payload,
    build_fnpm_round_optimization,
    compute_fnpm_training_loss,
    extract_teacher_supervision,
    module_state_sha256,
    restore_fnpm_resume_payload,
    run_fnpm_steps,
)


class TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 2, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


def _extractor(teacher: nn.Module, batch):
    return teacher(batch['inputs']), batch['counts']


def _batch(iteration: int):
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    inputs = inputs / 16.0 + float(iteration) / 100.0
    counts = torch.tensor(
        [float(iteration % 3), float((iteration + 1) % 3)]
    )
    return {'inputs': inputs, 'counts': counts}


def _empty_target(iteration: int):
    del iteration
    return {
        'inputs': torch.empty(0, 2, 2, 2),
        'counts': torch.empty(0),
    }


def _assert_modules_equal(left: nn.Module, right: nn.Module) -> None:
    assert tuple(left.state_dict()) == tuple(right.state_dict())
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name]), name


def test_only_fnpm_receives_gradients_and_teacher_state_is_unchanged() -> None:
    torch.manual_seed(2)
    teacher = TinyTeacher()
    fnpm = FalseNegativePredictionModule(in_channels=2)
    optimizer, scheduler = build_fnpm_round_optimization(fnpm)
    teacher_hash = module_state_sha256(teacher)
    fnpm_hash = module_state_sha256(fnpm)
    callback_state = {}

    def checking_extractor(model, batch):
        callback_state['grad_enabled'] = torch.is_grad_enabled()
        callback_state['teacher_training'] = model.training
        return _extractor(model, batch)

    result = run_fnpm_steps(
        fnpm,
        teacher,
        optimizer,
        scheduler,
        _batch,
        checking_extractor,
        start_iteration=0,
        end_iteration=1,
    )

    assert result.iteration == 1
    assert result.teacher_state_sha256 == teacher_hash
    assert module_state_sha256(teacher) == teacher_hash
    assert module_state_sha256(fnpm) != fnpm_hash
    assert callback_state == {'grad_enabled': False, 'teacher_training': False}
    assert teacher.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in fnpm.parameters())


def test_empty_labeled_target_uses_source_mean_only() -> None:
    torch.manual_seed(3)
    fnpm = FalseNegativePredictionModule(in_channels=2)
    source = FnpmSupervision(
        torch.randn(2, 2, 2, 2), torch.tensor([0.0, 2.0])
    )
    empty_target = FnpmSupervision(
        torch.empty(0, 2, 2, 2), torch.empty(0)
    )

    actual = compute_fnpm_training_loss(fnpm, source, empty_target)
    expected = fnpm_loss(fnpm(source.features), source.false_negative_counts)

    assert torch.equal(actual, expected)


def test_labeled_target_mean_is_added_to_source_mean() -> None:
    torch.manual_seed(4)
    fnpm = FalseNegativePredictionModule(in_channels=2)
    source = FnpmSupervision(
        torch.randn(2, 2, 2, 2), torch.tensor([0.0, 1.0])
    )
    target = FnpmSupervision(
        torch.randn(1, 2, 2, 2), torch.tensor([2.0])
    )

    actual = compute_fnpm_training_loss(fnpm, source, target)
    expected = fnpm_loss(
        fnpm(source.features),
        source.false_negative_counts,
        fnpm(target.features),
        target.false_negative_counts,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    'counts',
    (
        torch.tensor([float('nan'), 0.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.5, 1.0]),
    ),
)
def test_teacher_targets_must_be_finite_nonnegative_integer_counts(counts) -> None:
    teacher = TinyTeacher()
    batch = {'inputs': torch.zeros(2, 2, 2, 2), 'counts': counts}

    with pytest.raises(ValueError, match='counts'):
        extract_teacher_supervision(teacher, batch, _extractor)


def test_teacher_feature_and_count_shapes_must_match() -> None:
    teacher = TinyTeacher()
    batch = {
        'inputs': torch.zeros(2, 2, 2, 2),
        'counts': torch.zeros(1),
    }

    with pytest.raises(ValueError, match='shapes do not match'):
        extract_teacher_supervision(teacher, batch, _extractor)

    class WrongShapeFnpm(nn.Module):
        def forward(self, features):
            return torch.zeros((len(features), 1), requires_grad=True)

    supervision = FnpmSupervision(torch.zeros(2, 2, 2, 2), torch.zeros(2))
    with pytest.raises(ValueError, match='shapes do not match'):
        compute_fnpm_training_loss(WrongShapeFnpm(), supervision)

    with pytest.raises(ValueError, match='finite'):
        compute_fnpm_training_loss(
            FalseNegativePredictionModule(in_channels=2),
            FnpmSupervision(
                torch.zeros(2, 2, 2, 2),
                torch.tensor([0.0, float('inf')]),
            ),
        )


def test_new_round_retains_weights_but_resets_optimizer_and_cosine() -> None:
    torch.manual_seed(5)
    teacher = TinyTeacher()
    fnpm = FalseNegativePredictionModule(in_channels=2)
    round_one_optimizer, round_one_scheduler = build_fnpm_round_optimization(fnpm)
    run_fnpm_steps(
        fnpm,
        teacher,
        round_one_optimizer,
        round_one_scheduler,
        _batch,
        _extractor,
        start_iteration=0,
        end_iteration=3,
    )
    retained_hash = module_state_sha256(fnpm)
    decayed_lr = round_one_optimizer.param_groups[0]['lr']

    round_two_optimizer, round_two_scheduler = build_fnpm_round_optimization(fnpm)

    assert module_state_sha256(fnpm) == retained_hash
    assert round_two_optimizer is not round_one_optimizer
    assert round_two_scheduler is not round_one_scheduler
    assert decayed_lr < FNPM_LEARNING_RATE
    assert round_two_optimizer.param_groups[0]['lr'] == FNPM_LEARNING_RATE
    assert round_two_scheduler.T_max == FNPM_ITERATIONS_PER_ROUND
    assert round_two_scheduler.last_epoch == 0


def test_uninterrupted_and_resumed_training_are_exact() -> None:
    torch.manual_seed(6)
    teacher = TinyTeacher()
    initial = FalseNegativePredictionModule(in_channels=2)
    uninterrupted = copy.deepcopy(initial)
    interrupted = copy.deepcopy(initial)

    full_optimizer, full_scheduler = build_fnpm_round_optimization(uninterrupted)
    full_result = run_fnpm_steps(
        uninterrupted,
        teacher,
        full_optimizer,
        full_scheduler,
        _batch,
        _extractor,
        start_iteration=0,
        end_iteration=6,
        labeled_target_batch_provider=_empty_target,
    )

    split_optimizer, split_scheduler = build_fnpm_round_optimization(interrupted)
    first_result = run_fnpm_steps(
        interrupted,
        teacher,
        split_optimizer,
        split_scheduler,
        _batch,
        _extractor,
        start_iteration=0,
        end_iteration=3,
        labeled_target_batch_provider=_empty_target,
    )
    payload = build_fnpm_resume_payload(
        interrupted,
        split_optimizer,
        split_scheduler,
        round_index=1,
        iteration=first_result.iteration,
    )
    resumed = FalseNegativePredictionModule(in_channels=2)
    resumed_optimizer, resumed_scheduler = build_fnpm_round_optimization(resumed)
    resume_iteration = restore_fnpm_resume_payload(
        payload,
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        expected_round_index=1,
    )
    second_result = run_fnpm_steps(
        resumed,
        teacher,
        resumed_optimizer,
        resumed_scheduler,
        _batch,
        _extractor,
        start_iteration=resume_iteration,
        end_iteration=6,
        labeled_target_batch_provider=_empty_target,
    )

    _assert_modules_equal(uninterrupted, resumed)
    assert full_optimizer.state_dict() == resumed_optimizer.state_dict()
    assert full_scheduler.state_dict() == resumed_scheduler.state_dict()
    assert full_result.losses == first_result.losses + second_result.losses
    assert payload['iteration'] == 3
    assert set(payload) == {
        'schema_version',
        'round_index',
        'iteration',
        'model',
        'optimizer',
        'scheduler',
    }
