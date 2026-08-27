'''Shared MMDetection runtime for segmented ADA detector experiments.'''

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch
from torch import nn

from methods.common.contracts import StageSpec
from methods.common.engine.context import ExecutionContext
from methods.common.execution.mmdet_checkpoints import (
    bind_exact_continuation_iteration,
    save_atomic_runner_checkpoint,
    unwrap_distributed_model,
    validate_detector_continuation_checkpoint,
)
from methods.common.protocols.ada_fnp_detection import (
    DetectorTrainingMode,
    DetectorTrainingPhase,
)


class MissingMmdetDependencyError(EnvironmentError):
    '''Raised when the pinned OpenMMLab runtime is unavailable.'''


@dataclass(frozen=True)
class MmdetRuntime:
    load_config: Callable[[Path], MutableMapping[str, Any]]
    import_custom_modules: Callable[[Mapping[str, Any]], None]
    build_runner: Callable[[Mapping[str, Any]], Any]
    progress_hook: Callable[..., Any]
    build_model: Callable[[Mapping[str, Any]], nn.Module]
    build_dataloader: Callable[[Mapping[str, Any], int], Iterable]
    load_model_checkpoint: Callable[[nn.Module, Path], None]


def load_mmdet_runtime() -> MmdetRuntime:
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner, load_checkpoint as mm_load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from methods.common.mmdet.progress import (
            AdaodConsoleQuietRunner,
            TqdmProgressHook,
        )
    except ImportError as error:
        raise MissingMmdetDependencyError(
            'ADAOD execution requires the pinned MMCV/MMEngine/MMDetection '
            'environment from requirements/runtime.txt'
        ) from error
    init_default_scope('mmdet')

    def import_custom_modules(config: Mapping[str, Any]) -> None:
        custom_imports = config.get('custom_imports')
        if custom_imports:
            import_modules_from_strings(**custom_imports)

    return MmdetRuntime(
        load_config=lambda path: Config.fromfile(
            str(path),
            import_custom_modules=False,
        ),
        import_custom_modules=import_custom_modules,
        build_runner=AdaodConsoleQuietRunner.from_cfg,
        progress_hook=TqdmProgressHook,
        build_model=MODELS.build,
        build_dataloader=lambda config, seed: Runner.build_dataloader(
            config,
            seed=seed,
        ),
        load_model_checkpoint=lambda model, path: mm_load_checkpoint(
            model,
            str(path),
            map_location='cpu',
            strict=True,
        ),
    )


def require_cuda_runtime() -> MmdetRuntime:
    runtime = load_mmdet_runtime()
    if not torch.cuda.is_available():
        raise MissingMmdetDependencyError(
            'ADAOD detector stages require CUDA-enabled PyTorch'
        )
    return runtime


@contextmanager
def preserve_random_state():
    '''Keep observational evaluation from changing the training RNG.'''

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_inference_model(
    runtime: MmdetRuntime,
    context: ExecutionContext,
    checkpoint_path: Path,
    config_loader: Callable[[MmdetRuntime, ExecutionContext], MutableMapping],
):
    config = config_loader(runtime, context)
    model = runtime.build_model(config['model'])
    runtime.load_model_checkpoint(model, checkpoint_path)
    model = model.cuda()
    model.eval()
    return model, config


def train_detector_segment(
    stage: StageSpec,
    context: ExecutionContext,
    phase: DetectorTrainingPhase,
    checkpoint_path: Path,
    continuation_checkpoint: Optional[Path],
    *,
    config_builder: Callable,
    required_log_keys: Mapping[DetectorTrainingMode, tuple],
    initialize_completed_model: Optional[Callable[[nn.Module], None]] = None,
) -> Path:
    runtime = require_cuda_runtime()
    config = config_builder(
        runtime,
        context,
        phase,
        continuation_checkpoint,
        stage.stage_id,
    )
    runner = runtime.build_runner(config)
    runner.register_hook(
        runtime.progress_hook(
            context.progress,
            task_total=phase.end_iteration - phase.start_iteration,
            task_unit='iter',
            required_keys=required_log_keys[phase.mode],
        ),
        priority='LOWEST',
    )
    if continuation_checkpoint is not None:
        validate_detector_continuation_checkpoint(
            continuation_checkpoint,
            runner.model,
            (phase.start_iteration, phase.end_iteration),
            context=context,
            executor_key=stage.executor_key,
        )
        bind_exact_continuation_iteration(
            runner,
            continuation_checkpoint,
            phase.start_iteration,
        )
    runner.train()
    if int(runner.iter) != phase.end_iteration:
        raise RuntimeError(
            'MMEngine stopped at iteration {}, expected {}'.format(
                runner.iter,
                phase.end_iteration,
            )
        )
    model = unwrap_distributed_model(runner.model)
    if (
        phase.mode is DetectorTrainingMode.INITIALIZATION
        and initialize_completed_model is not None
    ):
        initialize_completed_model(model)
    return save_atomic_runner_checkpoint(
        runner,
        checkpoint_path,
        stage,
        phase.end_iteration,
    )


def evaluate_detector_checkpoint(
    stage: StageSpec,
    context: ExecutionContext,
    checkpoint_path: Path,
    *,
    config_loader: Callable[[MmdetRuntime, ExecutionContext], MutableMapping],
    configure_test_dataloader: Callable,
) -> Mapping[str, float]:
    iteration = int(stage.payload['iteration'])
    with preserve_random_state():
        runtime = require_cuda_runtime()
        config = config_loader(runtime, context)
        configure_test_dataloader(config['test_dataloader'], context)
        config['load_from'] = None
        config['resume'] = False
        config['work_dir'] = str(
            context.run_directory
            / 'mmengine/evaluations'
            / 'iter_{:05d}'.format(iteration)
        )
        runner = runtime.build_runner(config)
        runtime.load_model_checkpoint(runner.model, checkpoint_path)
        runner.register_hook(
            runtime.progress_hook(context.progress, task_unit='batch'),
            priority='LOWEST',
        )
        metrics = dict(runner.test())
    ap50_keys = [key for key in metrics if key.split('/')[-1] == 'AP50']
    if len(ap50_keys) != 1:
        raise ValueError(
            'Detectron2 VOC evaluator must return exactly one AP50 metric'
        )
    ap50 = float(metrics[ap50_keys[0]])
    if not math.isfinite(ap50) or not 0.0 <= ap50 <= 100.0:
        raise ValueError(
            'Detectron2 VOC AP50 must be a finite percentage in [0, 100]'
        )
    return {'AP50': ap50}
