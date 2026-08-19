'''Pure PyTorch FNPM round optimization and exact resume helpers.'''

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
from torch import nn

from methods.ada_fnp.models.fnpm import fnpm_loss
from methods.ada_fnp.phases import FNPM_ITERATIONS_PER_ROUND


FNPM_STATE_SCHEMA_VERSION = 1

TeacherBatchExtractor = Callable[[nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]]
BatchProvider = Callable[[int], Any]


@dataclass(frozen=True)
class FnpmSupervision:
    '''Frozen teacher features and integer false-negative count targets.'''

    features: torch.Tensor
    false_negative_counts: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.features, torch.Tensor) or not isinstance(
            self.false_negative_counts, torch.Tensor
        ):
            raise TypeError('FNPM supervision requires feature and count tensors')
        if self.features.ndim != 4:
            raise ValueError('FNPM features must have shape [N, C, H, W]')
        if self.false_negative_counts.ndim != 1:
            raise ValueError('false-negative counts must have shape [N]')
        if self.features.shape[0] != self.false_negative_counts.shape[0]:
            raise ValueError('feature and false-negative count batch shapes do not match')
        if not torch.isfinite(self.features).all():
            raise ValueError('FNPM features must be finite')
        if not torch.isfinite(self.false_negative_counts).all():
            raise ValueError('false-negative counts must be finite')
        if (self.false_negative_counts < 0).any():
            raise ValueError('false-negative counts must not be negative')
        if self.false_negative_counts.is_floating_point() and not torch.equal(
            self.false_negative_counts, self.false_negative_counts.round()
        ):
            raise ValueError('false-negative counts must be integer-valued')

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FnpmRunResult:
    '''Result of a contiguous section of one 2k FNPM round.'''

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
) -> FnpmSupervision:
    supervision = FnpmSupervision(features, counts)
    if not allow_empty and features.shape[0] == 0:
        raise ValueError('source FNPM supervision must not be empty')
    return FnpmSupervision(
        supervision.features.detach(), supervision.false_negative_counts.detach()
    )


def extract_teacher_supervision(
    teacher: nn.Module,
    batch: Any,
    extractor: TeacherBatchExtractor,
    *,
    allow_empty: bool = False,
) -> FnpmSupervision:
    '''Run injected detector inference/matching under teacher eval and no-grad.'''

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


def module_state_sha256(module: nn.Module) -> str:
    '''Hash tensor keys, metadata, and exact bytes from a module state dict.'''

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError('module state must contain tensors only')
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_fnpm_round_optimization(
    fnpm: nn.Module,
    learning_rate: float,
) -> Tuple[torch.optim.SGD, torch.optim.lr_scheduler.CosineAnnealingLR]:
    '''Create a fresh SGD and 2k cosine schedule without touching FNPM weights.'''

    if not isinstance(fnpm, nn.Module):
        raise TypeError('fnpm must be a torch module')
    learning_rate = float(learning_rate)
    if learning_rate <= 0:
        raise ValueError('FNPM learning rate must be positive')
    parameters = tuple(parameter for parameter in fnpm.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError('fnpm has no trainable parameters')
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FNPM_ITERATIONS_PER_ROUND
    )
    return optimizer, scheduler


def _validate_prediction_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    domain: str,
) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError('{} prediction and count target shapes do not match'.format(domain))
    if not torch.isfinite(prediction).all():
        raise ValueError('{} FNPM prediction must be finite'.format(domain))
    return target


def compute_fnpm_training_loss(
    fnpm: nn.Module,
    source: FnpmSupervision,
    labeled_target: Optional[FnpmSupervision] = None,
) -> torch.Tensor:
    '''Compute source-domain mean MSE plus optional labeled-target mean MSE.'''

    if not isinstance(source, FnpmSupervision):
        raise TypeError('source must be FnpmSupervision')
    if source.batch_size == 0:
        raise ValueError('source FNPM supervision must not be empty')
    source_prediction = fnpm(source.features)
    source_target = _validate_prediction_target(
        source_prediction, source.false_negative_counts, 'source'
    )
    target_prediction = None
    target_count = None
    if labeled_target is not None:
        if not isinstance(labeled_target, FnpmSupervision):
            raise TypeError('labeled_target must be FnpmSupervision or None')
        if labeled_target.batch_size > 0:
            target_prediction = fnpm(labeled_target.features)
            target_count = _validate_prediction_target(
                target_prediction,
                labeled_target.false_negative_counts,
                'labeled-target',
            )
    loss = fnpm_loss(
        source_prediction,
        source_target,
        target_prediction,
        target_count,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('FNPM loss must be a finite scalar')
    return loss


def _validate_optimizer_scope(fnpm: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {id(parameter) for parameter in fnpm.parameters() if parameter.requires_grad}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    if actual != expected:
        raise ValueError('optimizer must contain exactly the trainable FNPM parameters')


def train_fnpm_step(
    fnpm: nn.Module,
    source: FnpmSupervision,
    labeled_target: Optional[FnpmSupervision],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    '''Update only FNPM once and advance the per-round scheduler exactly once.'''

    _validate_optimizer_scope(fnpm, optimizer)
    if scheduler.optimizer is not optimizer:
        raise ValueError('scheduler must belong to the FNPM optimizer')
    fnpm.train()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_fnpm_training_loss(fnpm, source, labeled_target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def run_fnpm_steps(
    fnpm: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    source_batch_provider: BatchProvider,
    teacher_batch_extractor: TeacherBatchExtractor,
    *,
    start_iteration: int,
    end_iteration: int,
    labeled_target_batch_provider: Optional[BatchProvider] = None,
) -> FnpmRunResult:
    '''Run a deterministic contiguous slice of a 2k round for resume support.'''

    start_iteration = _integer(start_iteration, 'start_iteration')
    end_iteration = _integer(end_iteration, 'end_iteration')
    if not 0 <= start_iteration <= end_iteration <= FNPM_ITERATIONS_PER_ROUND:
        raise ValueError('FNPM iteration slice must stay within 0 through 2000')
    if not callable(source_batch_provider) or not callable(teacher_batch_extractor):
        raise TypeError('source provider and teacher extractor must be callable')
    if labeled_target_batch_provider is not None and not callable(
        labeled_target_batch_provider
    ):
        raise TypeError('labeled-target provider must be callable or None')

    previous_teacher_training = teacher.training
    teacher_hash = module_state_sha256(teacher)
    losses = []
    teacher.eval()
    try:
        for iteration in range(start_iteration, end_iteration):
            source = extract_teacher_supervision(
                teacher,
                source_batch_provider(iteration),
                teacher_batch_extractor,
                allow_empty=False,
            )
            labeled_target = None
            if labeled_target_batch_provider is not None:
                target_batch = labeled_target_batch_provider(iteration)
                if target_batch is not None:
                    labeled_target = extract_teacher_supervision(
                        teacher,
                        target_batch,
                        teacher_batch_extractor,
                        allow_empty=True,
                    )
            losses.append(
                train_fnpm_step(
                    fnpm, source, labeled_target, optimizer, scheduler
                )
            )
    finally:
        teacher.train(previous_teacher_training)
    final_teacher_hash = module_state_sha256(teacher)
    if final_teacher_hash != teacher_hash:
        raise RuntimeError('teacher state changed during FNPM feature extraction')
    return FnpmRunResult(end_iteration, tuple(losses))


def build_fnpm_resume_payload(
    fnpm: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    round_index: int,
    iteration: int,
) -> dict:
    '''Capture every mutable FNPM round state needed for exact continuation.'''

    round_index = _integer(round_index, 'round_index')
    iteration = _integer(iteration, 'iteration')
    if not 1 <= round_index <= 5:
        raise ValueError('round_index must be between 1 and 5')
    if not 0 <= iteration <= FNPM_ITERATIONS_PER_ROUND:
        raise ValueError('iteration must be between 0 and 2000')
    _validate_optimizer_scope(fnpm, optimizer)
    if scheduler.optimizer is not optimizer:
        raise ValueError('scheduler must belong to the FNPM optimizer')
    return {
        'schema_version': FNPM_STATE_SCHEMA_VERSION,
        'round_index': round_index,
        'iteration': iteration,
        'model': copy.deepcopy(fnpm.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
    }


def restore_fnpm_resume_payload(
    payload: Mapping[str, Any],
    fnpm: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    expected_round_index: int,
) -> int:
    '''Strictly restore one FNPM round and return its completed-step count.'''

    expected_keys = {
        'schema_version',
        'round_index',
        'iteration',
        'model',
        'optimizer',
        'scheduler',
    }
    if set(payload) != expected_keys:
        raise ValueError('FNPM resume payload has an invalid schema')
    if payload['schema_version'] != FNPM_STATE_SCHEMA_VERSION:
        raise ValueError('unsupported FNPM resume schema version')
    expected_round_index = _integer(expected_round_index, 'expected_round_index')
    if not 1 <= expected_round_index <= 5:
        raise ValueError('expected_round_index must be between 1 and 5')
    payload_round_index = _integer(payload['round_index'], 'payload round_index')
    if not 1 <= payload_round_index <= 5:
        raise ValueError('payload round_index must be between 1 and 5')
    if payload_round_index != expected_round_index:
        raise ValueError('FNPM resume payload belongs to a different round')
    iteration = _integer(payload['iteration'], 'iteration')
    if not 0 <= iteration <= FNPM_ITERATIONS_PER_ROUND:
        raise ValueError('resume iteration must be between 0 and 2000')
    fnpm.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    _validate_optimizer_scope(fnpm, optimizer)
    if scheduler.optimizer is not optimizer:
        raise ValueError('restored scheduler is detached from its optimizer')
    if scheduler.T_max != FNPM_ITERATIONS_PER_ROUND:
        raise ValueError('restored scheduler is not a fresh 2k cosine schedule')
    return iteration
