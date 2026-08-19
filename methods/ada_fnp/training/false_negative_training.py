'''False-negative predictor supervision and round optimization.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
from torch import nn

from methods.ada_fnp.models.false_negative_predictor import (
    false_negative_prediction_loss,
)
from methods.ada_fnp.schedule import (
    ACQUISITION_ROUND_COUNT,
    FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND,
)


FALSE_NEGATIVE_CHECKPOINT_SCHEMA_VERSION = 1

TeacherBatchExtractor = Callable[
    [nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]
]
BatchProvider = Callable[[int], Any]


@dataclass(frozen=True)
class FalseNegativeSupervision:
    '''Frozen teacher features and integer false-negative count targets.'''

    features: torch.Tensor
    false_negative_counts: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.features, torch.Tensor) or not isinstance(
            self.false_negative_counts, torch.Tensor
        ):
            raise TypeError(
                'false-negative supervision requires feature and count tensors'
            )
        if self.features.ndim != 4:
            raise ValueError('features must have shape [N, C, H, W]')
        if self.false_negative_counts.ndim != 1:
            raise ValueError('false-negative counts must have shape [N]')
        if self.features.shape[0] != self.false_negative_counts.shape[0]:
            raise ValueError(
                'feature and false-negative count batch shapes do not match'
            )
        if not torch.isfinite(self.features).all():
            raise ValueError('features must be finite')
        if not torch.isfinite(self.false_negative_counts).all():
            raise ValueError('false-negative counts must be finite')
        if (self.false_negative_counts < 0).any():
            raise ValueError('false-negative counts must not be negative')
        if self.false_negative_counts.is_floating_point() and not torch.equal(
            self.false_negative_counts,
            self.false_negative_counts.round(),
        ):
            raise ValueError('false-negative counts must be integer-valued')

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FalseNegativeTrainingResult:
    '''Result of a contiguous section of one predictor-training round.'''

    iteration: int
    losses: Tuple[float, ...]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    return value


def _validate_supervision(
    features: torch.Tensor,
    counts: torch.Tensor,
    *,
    allow_empty: bool,
) -> FalseNegativeSupervision:
    supervision = FalseNegativeSupervision(features, counts)
    if not allow_empty and features.shape[0] == 0:
        raise ValueError('source supervision must not be empty')
    return FalseNegativeSupervision(
        supervision.features.detach(),
        supervision.false_negative_counts.detach(),
    )


def extract_teacher_false_negative_supervision(
    teacher: nn.Module,
    batch: Any,
    extractor: TeacherBatchExtractor,
    *,
    allow_empty: bool = False,
) -> FalseNegativeSupervision:
    '''Run detector inference and matching under teacher eval and no-grad.'''

    if not isinstance(teacher, nn.Module):
        raise TypeError('teacher must be a torch module')
    if not callable(extractor):
        raise TypeError('extractor must be callable')
    previous_training = teacher.training
    teacher.eval()
    try:
        with torch.no_grad():
            extracted = extractor(teacher, batch)
    finally:
        teacher.train(previous_training)
    if not isinstance(extracted, tuple) or len(extracted) != 2:
        raise TypeError('teacher extractor must return (features, counts)')
    return _validate_supervision(
        extracted[0], extracted[1], allow_empty=allow_empty
    )


def build_false_negative_round_optimization(
    predictor: nn.Module,
    learning_rate: float,
) -> Tuple[torch.optim.SGD, torch.optim.lr_scheduler.CosineAnnealingLR]:
    '''Create fresh SGD and cosine scheduling for one training round.'''

    if not isinstance(predictor, nn.Module):
        raise TypeError('predictor must be a torch module')
    learning_rate = float(learning_rate)
    if learning_rate <= 0:
        raise ValueError('learning rate must be positive')
    parameters = tuple(
        parameter
        for parameter in predictor.parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError('predictor has no trainable parameters')
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND,
    )
    return optimizer, scheduler


def _validate_prediction_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    domain: str,
) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError(
            '{} prediction and count target shapes do not match'.format(domain)
        )
    if not torch.isfinite(prediction).all():
        raise ValueError('{} prediction must be finite'.format(domain))
    return target


def compute_false_negative_training_loss(
    predictor: nn.Module,
    source: FalseNegativeSupervision,
    labeled_target: Optional[FalseNegativeSupervision] = None,
) -> torch.Tensor:
    '''Compute source mean MSE plus optional labeled-target mean MSE.'''

    if not isinstance(source, FalseNegativeSupervision):
        raise TypeError('source must be FalseNegativeSupervision')
    if source.batch_size == 0:
        raise ValueError('source supervision must not be empty')
    source_prediction = predictor(source.features)
    source_target = _validate_prediction_target(
        source_prediction,
        source.false_negative_counts,
        'source',
    )
    target_prediction = None
    target_count = None
    if labeled_target is not None:
        if not isinstance(labeled_target, FalseNegativeSupervision):
            raise TypeError(
                'labeled_target must be FalseNegativeSupervision or None'
            )
        if labeled_target.batch_size > 0:
            target_prediction = predictor(labeled_target.features)
            target_count = _validate_prediction_target(
                target_prediction,
                labeled_target.false_negative_counts,
                'labeled-target',
            )
    loss = false_negative_prediction_loss(
        source_prediction,
        source_target,
        target_prediction,
        target_count,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('false-negative prediction loss must be finite scalar')
    return loss


def _validate_optimizer_scope(
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {
        id(parameter)
        for parameter in predictor.parameters()
        if parameter.requires_grad
    }
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    if actual != expected:
        raise ValueError(
            'optimizer must contain exactly the trainable predictor parameters'
        )


def train_false_negative_step(
    predictor: nn.Module,
    source: FalseNegativeSupervision,
    labeled_target: Optional[FalseNegativeSupervision],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    '''Update only the predictor and advance its scheduler exactly once.'''

    _validate_optimizer_scope(predictor, optimizer)
    if scheduler.optimizer is not optimizer:
        raise ValueError('scheduler must belong to the predictor optimizer')
    predictor.train()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_false_negative_training_loss(
        predictor, source, labeled_target
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def run_false_negative_training_steps(
    predictor: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    source_batch_provider: BatchProvider,
    teacher_batch_extractor: TeacherBatchExtractor,
    *,
    start_iteration: int,
    end_iteration: int,
    labeled_target_batch_provider: Optional[BatchProvider] = None,
) -> FalseNegativeTrainingResult:
    '''Run one contiguous section of a predictor-training round.'''

    start_iteration = _integer(start_iteration, 'start_iteration')
    end_iteration = _integer(end_iteration, 'end_iteration')
    if not (
        0
        <= start_iteration
        <= end_iteration
        <= FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND
    ):
        raise ValueError(
            'iteration slice must stay within 0 through {}'.format(
                FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND
            )
        )
    if not callable(source_batch_provider) or not callable(
        teacher_batch_extractor
    ):
        raise TypeError('source provider and teacher extractor must be callable')
    if labeled_target_batch_provider is not None and not callable(
        labeled_target_batch_provider
    ):
        raise TypeError('labeled-target provider must be callable or None')

    previous_teacher_training = teacher.training
    losses = []
    teacher.eval()
    try:
        for iteration in range(start_iteration, end_iteration):
            source = extract_teacher_false_negative_supervision(
                teacher,
                source_batch_provider(iteration),
                teacher_batch_extractor,
                allow_empty=False,
            )
            labeled_target = None
            if labeled_target_batch_provider is not None:
                target_batch = labeled_target_batch_provider(iteration)
                if target_batch is not None:
                    labeled_target = extract_teacher_false_negative_supervision(
                        teacher,
                        target_batch,
                        teacher_batch_extractor,
                        allow_empty=True,
                    )
            losses.append(
                train_false_negative_step(
                    predictor,
                    source,
                    labeled_target,
                    optimizer,
                    scheduler,
                )
            )
    finally:
        teacher.train(previous_teacher_training)
    return FalseNegativeTrainingResult(end_iteration, tuple(losses))


def build_false_negative_checkpoint_payload(
    predictor: nn.Module,
    *,
    round_index: int,
) -> dict:
    '''Capture the predictor weights retained between acquisition rounds.'''

    round_index = _integer(round_index, 'round_index')
    if not 1 <= round_index <= ACQUISITION_ROUND_COUNT:
        raise ValueError(
            'round_index must be between 1 and {}'.format(
                ACQUISITION_ROUND_COUNT
            )
        )
    return {
        'schema_version': FALSE_NEGATIVE_CHECKPOINT_SCHEMA_VERSION,
        'round_index': round_index,
        'model': predictor.state_dict(),
    }


def restore_false_negative_checkpoint_payload(
    payload: Mapping[str, Any],
    predictor: nn.Module,
    *,
    expected_round_index: int,
) -> None:
    '''Strictly restore predictor weights from one completed round.'''

    expected_keys = {'schema_version', 'round_index', 'model'}
    if set(payload) != expected_keys:
        raise ValueError('predictor checkpoint has an invalid schema')
    if payload['schema_version'] != FALSE_NEGATIVE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('unsupported predictor checkpoint schema version')
    expected_round_index = _integer(
        expected_round_index, 'expected_round_index'
    )
    if not 1 <= expected_round_index <= ACQUISITION_ROUND_COUNT:
        raise ValueError(
            'expected_round_index must be between 1 and {}'.format(
                ACQUISITION_ROUND_COUNT
            )
        )
    payload_round_index = _integer(
        payload['round_index'], 'payload round_index'
    )
    if not 1 <= payload_round_index <= ACQUISITION_ROUND_COUNT:
        raise ValueError(
            'payload round_index must be between 1 and {}'.format(
                ACQUISITION_ROUND_COUNT
            )
        )
    if payload_round_index != expected_round_index:
        raise ValueError('predictor checkpoint belongs to a different round')
    predictor.load_state_dict(payload['model'], strict=True)
