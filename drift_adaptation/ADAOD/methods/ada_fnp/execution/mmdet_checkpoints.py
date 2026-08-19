'''Strict MMEngine detector checkpoint validation and persistence.'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from torch import nn

from methods.common.contracts import ArtifactRef, StageSpec
from methods.common.engine.checkpoint import load_checkpoint
from methods.common.engine.context import ExecutionContext


_DETECTOR_EXECUTOR_KEY = 'ada_fnp.train_detector'
_DETECTOR_CHECKPOINT_TYPE = 'detector_checkpoint'


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    '''Return the underlying model wrapped by distributed containers.'''

    while hasattr(model, 'module'):
        model = model.module
    return model


def validate_detector_continuation_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    expected_iterations: Sequence[int],
    *,
    context: ExecutionContext,
) -> None:
    '''Validate a detector checkpoint before continuing the 40k schedule.'''

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_root = (context.run_directory / 'checkpoints').resolve()
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as error:
        raise ValueError(
            'detector continuation checkpoint must stay in the run checkpoint '
            'directory'
        ) from error

    relative_path = checkpoint_path.relative_to(
        context.run_directory
    ).as_posix()
    matching_artifacts = []
    for completed_stage in context.state_store.load().completed_stages:
        if completed_stage.get('executor_key') != _DETECTOR_EXECUTOR_KEY:
            continue
        artifact_value = completed_stage.get('result', {}).get(
            'checkpoint_artifact'
        )
        if (
            not artifact_value
            or artifact_value.get('relative_path') != relative_path
        ):
            continue
        artifact = ArtifactRef(**artifact_value)
        if artifact.artifact_type != _DETECTOR_CHECKPOINT_TYPE:
            raise ValueError(
                'continuation artifact is not a detector checkpoint'
            )
        if artifact.producer_stage_id != completed_stage.get('stage_id'):
            raise ValueError(
                'continuation artifact producer does not match its stage'
            )
        if artifact.artifact_id != artifact.sha256:
            raise ValueError(
                'continuation artifact ID must equal its SHA256'
            )
        matching_artifacts.append(artifact)
    if len(matching_artifacts) != 1:
        raise ValueError(
            'detector continuation checkpoint requires exactly one completed '
            'artifact'
        )
    context.artifact_store.verify(matching_artifacts[0])

    # MMEngine checkpoints contain HistoryBuffer objects, so the full pickle
    # load is permitted only after the run-local path and digest are verified.
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError('detector continuation checkpoint must be a mapping')
    required_sections = {'state_dict', 'optimizer', 'param_schedulers', 'meta'}
    missing_sections = sorted(required_sections.difference(checkpoint))
    if missing_sections:
        raise ValueError(
            'detector continuation checkpoint is missing: {}'.format(
                ', '.join(missing_sections)
            )
        )

    state_dict = checkpoint['state_dict']
    if not isinstance(state_dict, Mapping):
        raise TypeError('detector checkpoint state_dict must be a mapping')
    expected_state = unwrap_distributed_model(model).state_dict()
    checkpoint_state = dict(state_dict)
    if (
        checkpoint_state
        and all(key.startswith('module.') for key in checkpoint_state)
        and not any(key.startswith('module.') for key in expected_state)
    ):
        checkpoint_state = {
            key[len('module.'):]: value
            for key, value in checkpoint_state.items()
        }

    missing_keys = sorted(set(expected_state).difference(checkpoint_state))
    unexpected_keys = sorted(set(checkpoint_state).difference(expected_state))
    if missing_keys or unexpected_keys:
        raise ValueError(
            'detector checkpoint model keys differ '
            '(missing={}, unexpected={})'.format(missing_keys, unexpected_keys)
        )
    shape_mismatches = sorted(
        key
        for key, expected_value in expected_state.items()
        if not hasattr(checkpoint_state[key], 'shape')
        or tuple(checkpoint_state[key].shape) != tuple(expected_value.shape)
    )
    if shape_mismatches:
        raise ValueError(
            'detector checkpoint tensor shapes differ: {}'.format(
                ', '.join(shape_mismatches)
            )
        )

    optimizer_state = checkpoint['optimizer']
    scheduler_state = checkpoint['param_schedulers']
    if not isinstance(optimizer_state, Mapping) or not optimizer_state:
        raise ValueError(
            'detector continuation requires nonempty optimizer state'
        )
    if (
        not isinstance(scheduler_state, (Mapping, Sequence))
        or isinstance(scheduler_state, (str, bytes))
        or not scheduler_state
    ):
        raise ValueError(
            'detector continuation requires nonempty param-scheduler state'
        )

    meta = checkpoint['meta']
    if not isinstance(meta, Mapping):
        raise TypeError('detector checkpoint meta must be a mapping')
    expected_iteration_set = {int(value) for value in expected_iterations}
    global_iteration = meta.get('global_iteration')
    if global_iteration not in expected_iteration_set:
        raise ValueError(
            'detector checkpoint global iteration {} is not one of {}'.format(
                global_iteration, tuple(expected_iterations)
            )
        )
    runner_iteration = meta.get('iter')
    if runner_iteration not in (global_iteration, global_iteration + 1):
        raise ValueError(
            'detector checkpoint runner iteration {} is incompatible with '
            'global iteration {}'.format(runner_iteration, global_iteration)
        )


def bind_exact_continuation_iteration(
    runner: Any,
    checkpoint_path: Path,
    expected_iteration: int,
) -> None:
    '''Normalize MMEngine's offset while continuing a detector segment.'''

    checkpoint_path = Path(checkpoint_path).resolve()
    original_load_checkpoint = runner.load_checkpoint

    def load_checkpoint_with_exact_iteration(filename, *args, **kwargs):
        loaded = original_load_checkpoint(filename, *args, **kwargs)
        if Path(str(filename)).resolve() != checkpoint_path:
            raise ValueError(
                'Runner loaded an unexpected continuation checkpoint'
            )
        meta = loaded.get('meta')
        if not isinstance(meta, MutableMapping):
            raise TypeError('detector checkpoint meta must be mutable')
        if meta.get('global_iteration') != expected_iteration:
            raise ValueError('Runner checkpoint global iteration changed')
        if meta.get('iter') not in (expected_iteration, expected_iteration + 1):
            raise ValueError('Runner checkpoint iteration is incompatible')
        meta['iter'] = expected_iteration
        return loaded

    runner.load_checkpoint = load_checkpoint_with_exact_iteration


def save_atomic_runner_checkpoint(
    runner: Any,
    checkpoint_path: Path,
    stage: StageSpec,
    iteration: int,
) -> Path:
    '''Persist detector continuation state and atomically install the file.'''

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(
        '.{}.tmp.pth'.format(checkpoint_path.stem)
    )
    runner.save_checkpoint(
        str(temporary_path.parent),
        filename=temporary_path.name,
        save_optimizer=True,
        save_param_scheduler=True,
        by_epoch=False,
        meta={
            'adaod_stage_id': stage.stage_id,
            'global_iteration': iteration,
            'iter': iteration,
        },
    )
    if not temporary_path.is_file():
        raise RuntimeError('MMEngine did not write the requested checkpoint')
    os.replace(str(temporary_path), str(checkpoint_path))
    return checkpoint_path
