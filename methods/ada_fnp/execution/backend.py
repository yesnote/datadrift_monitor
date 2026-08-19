'''MMEngine adapter for ADA-FNP training, FNPM, scoring, and evaluation.'''

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Protocol
from typing import Sequence, Tuple

import torch
from torch import nn

from methods.ada_fnp.acquisition.records import RawAdaFnpScore
from methods.ada_fnp.acquisition.scoring import (
    domain_diversity,
    foreground_entropy,
)
from methods.ada_fnp.models.fnpm import FalseNegativePredictionModule
from methods.ada_fnp.phases import DetectorPhase, DetectorStageMode
from methods.ada_fnp.training.false_negative_targets import count_false_negatives
from methods.common.contracts import ArtifactRef, StageSpec
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine.checkpoint import load_checkpoint
from methods.common.engine.context import ExecutionContext
from methods.common.mmdet.models.backbones.vgg16_caffe import CHECKPOINT_PATH

from .artifacts import completed_checkpoint
from .paths import dataset_cache_directory


class ExecutionDependencyError(EnvironmentError):
    '''Raised before a model stage when its runtime is unavailable.'''


@dataclass(frozen=True)
class FnpmSession:
    '''Runtime objects consumed by the shared, resumable FNPM trainer.'''

    fnpm: nn.Module
    teacher: nn.Module
    source_batch_provider: Callable[[int], Any]
    teacher_batch_extractor: Callable[
        [nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]
    ]
    labeled_target_batch_provider: Optional[Callable[[int], Any]] = None


class AdaFnpExecutionBackend(Protocol):
    def train_detector(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        phase: DetectorPhase,
        checkpoint_path: Path,
        resume_from: Optional[Path],
    ) -> Path:
        ...

    def create_fnpm_session(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        checkpoint_path: Optional[Path],
    ) -> FnpmSession:
        ...

    def score_pool(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        samples: Sequence[SampleIdentity],
    ) -> Sequence[RawAdaFnpScore]:
        ...

    def evaluate(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        checkpoint_path: Path,
    ) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class MmdetRuntime:
    '''Small injectable surface over MMEngine and MMDetection registries.'''

    load_config: Callable[[Path], MutableMapping[str, Any]]
    import_custom_modules: Callable[[Mapping[str, Any]], None]
    build_runner: Callable[[Mapping[str, Any]], Any]
    build_model: Callable[[Mapping[str, Any]], nn.Module]
    build_dataloader: Callable[[Mapping[str, Any], int], Iterable]
    load_model_checkpoint: Callable[[nn.Module, Path], None]


def _load_mmdet_runtime() -> MmdetRuntime:
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner, load_checkpoint as mm_load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
    except ImportError as error:
        raise ExecutionDependencyError(
            'ADA-FNP execution requires the pinned MMCV/MMEngine/MMDetection '
            'environment from requirements/runtime.txt'
        ) from error

    # Runner initializes this scope itself, but FNPM, acquisition, and resume
    # paths also build models and dataloaders directly from the registries.
    init_default_scope('mmdet')

    def import_custom_modules(config: Mapping[str, Any]) -> None:
        custom_imports = config.get('custom_imports')
        if custom_imports:
            import_modules_from_strings(**custom_imports)

    return MmdetRuntime(
        load_config=lambda path: Config.fromfile(str(path)),
        import_custom_modules=import_custom_modules,
        build_runner=Runner.from_cfg,
        build_model=MODELS.build,
        build_dataloader=lambda config, seed: Runner.build_dataloader(
            config, seed=seed
        ),
        load_model_checkpoint=lambda model, path: mm_load_checkpoint(
            model, str(path), map_location='cpu', strict=True
        ),
    )


def _materialize_config_replacement(
    value: Mapping[str, Any], name: str
) -> MutableMapping[str, Any]:
    '''Consume an MMEngine merge directive before runtime construction.'''

    resolved = copy.deepcopy(dict(value))
    delete_directive = resolved.pop('_delete_', None)
    if delete_directive is not None and delete_directive is not True:
        raise ValueError('{} _delete_ directive must be true'.format(name))
    return resolved


def _labeled_manifest(context: ExecutionContext) -> Optional[Path]:
    round_index = context.state_store.load().active_round
    if round_index == 0:
        return None
    path = (
        context.run_directory / 'datasets' /
        'target_train_labeled_round_{:02d}.json'.format(round_index)
    )
    if not path.is_file():
        raise FileNotFoundError('selected-target manifest is missing: {!s}'.format(path))
    return path


def _configure_dataset_paths(
    dataset: MutableMapping[str, Any],
    context: ExecutionContext,
    labeled_manifest: Optional[Path],
    unlabeled_manifest: Optional[Path],
) -> None:
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            _configure_dataset_paths(
                child, context, labeled_manifest, unlabeled_manifest
            )
        return
    ann_file = dataset.get('ann_file')
    if ann_file is None:
        return
    configured_ann_file = Path(str(ann_file))
    filename = configured_ann_file.name
    cache = dataset_cache_directory(context)
    if configured_ann_file.is_absolute():
        resolved_ann_file = configured_ann_file.resolve()
        allowed_roots = (context.run_directory, cache)
        if not any(
            resolved_ann_file == root or root in resolved_ann_file.parents
            for root in allowed_roots
        ):
            raise ValueError('absolute annotation file is outside run/cache roots')
    elif filename == 'target_train_labeled.json':
        if labeled_manifest is None:
            raise RuntimeError('adaptation dataset requested before annotation reveal')
        resolved_ann_file = labeled_manifest
    elif filename == 'target_train_unlabeled.json':
        if unlabeled_manifest is None:
            raise RuntimeError('target-unlabeled dataset requires an active pool')
        resolved_ann_file = unlabeled_manifest
    else:
        resolved_ann_file = cache / filename
    if not resolved_ann_file.is_file():
        raise FileNotFoundError('MMDetection annotation file is missing: {!s}'.format(
            resolved_ann_file
        ))
    dataset['ann_file'] = str(resolved_ann_file)
    dataset['data_root'] = str(context.repository_root)
    dataset['data_prefix'] = dict(img='')


def _configure_dataloader(
    dataloader: MutableMapping[str, Any],
    context: ExecutionContext,
    labeled_manifest: Optional[Path],
    unlabeled_manifest: Optional[Path] = None,
) -> None:
    _configure_dataset_paths(
        dataloader['dataset'], context, labeled_manifest, unlabeled_manifest
    )
    sampler = dataloader.get('sampler')
    if isinstance(sampler, MutableMapping) and 'seed' in sampler:
        sampler['seed'] = int(context.config['seed'])


def _config_path(context: ExecutionContext) -> Path:
    return context.repository_root / 'methods/ada_fnp/configs/cityscapes_to_foggy.py'


def _base_config(runtime: MmdetRuntime, context: ExecutionContext):
    config = runtime.load_config(_config_path(context))
    runtime.import_custom_modules(config)
    config['work_dir'] = str(context.run_directory / 'mmengine')
    config['launcher'] = context.config['runtime']['launcher']
    config['randomness'] = dict(
        seed=int(context.config['seed']),
        deterministic=bool(context.config['runtime']['deterministic']),
    )
    config['env_cfg']['cudnn_benchmark'] = bool(
        context.config['runtime']['cudnn_benchmark']
    )
    backbone = config['model']['detector']['backbone']
    backbone['pretrained_checkpoint'] = str(
        context.repository_root / CHECKPOINT_PATH
    )
    return config


def build_detector_stage_config(
    runtime: MmdetRuntime,
    context: ExecutionContext,
    phase: DetectorPhase,
    resume_from: Optional[Path],
    producer_stage_id: str,
):
    '''Resolve one segment while retaining the global iteration schedule.'''

    config = _base_config(runtime, context)
    mode = (
        'initial'
        if phase.mode is DetectorStageMode.INITIALIZATION
        else 'adaptation'
    )
    override = copy.deepcopy(config['stage_overrides'][mode])
    config['train_dataloader'] = _materialize_config_replacement(
        override['train_dataloader'], '{} train_dataloader'.format(mode)
    )
    config['model'].update(override['model'])
    config['custom_hooks'] = override['custom_hooks']
    active_pool = _active_pool(context)
    unlabeled_manifest = _write_unlabeled_pool_manifest(
        context, active_pool.unlabeled, active_pool, producer_stage_id
    )
    _configure_dataloader(
        config['train_dataloader'],
        context,
        _labeled_manifest(context),
        unlabeled_manifest,
    )
    config['train_cfg']['max_iters'] = phase.end_iteration
    config['train_cfg']['type'] = 'ADAODSegmentedIterBasedTrainLoop'
    config['train_cfg']['val_interval'] = phase.end_iteration + 1
    config['val_cfg'] = None
    config['val_dataloader'] = None
    config['val_evaluator'] = None
    config['load_from'] = str(resume_from) if resume_from is not None else None
    config['resume'] = resume_from is not None
    checkpoint_hook = config['default_hooks']['checkpoint']
    checkpoint_hook['interval'] = phase.end_iteration - phase.start_iteration
    checkpoint_hook['by_epoch'] = False
    return config


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, 'module'):
        model = model.module
    return model


def validate_detector_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    expected_iterations: Sequence[int],
    *,
    context: ExecutionContext,
) -> None:
    '''Fail before Runner resume if any reproducibility state is incomplete.'''

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_root = (context.run_directory / 'checkpoints').resolve()
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as error:
        raise ValueError(
            'detector resume checkpoint must stay in the run checkpoint directory'
        ) from error
    relative_path = checkpoint_path.relative_to(
        context.run_directory
    ).as_posix()
    artifacts = []
    for completed in context.state_store.load().completed_stages:
        if completed.get('executor_key') != 'ada_fnp.train_detector':
            continue
        value = completed.get('result', {}).get('checkpoint_artifact')
        if not value or value.get('relative_path') != relative_path:
            continue
        artifact = ArtifactRef(**value)
        if artifact.artifact_type != 'detector_checkpoint':
            raise ValueError('resume artifact is not a detector checkpoint')
        if artifact.producer_stage_id != completed.get('stage_id'):
            raise ValueError('resume artifact producer does not match its stage')
        if artifact.artifact_id != artifact.sha256:
            raise ValueError('resume artifact ID must equal its SHA256')
        artifacts.append(artifact)
    if len(artifacts) != 1:
        raise ValueError(
            'detector resume checkpoint requires exactly one completed artifact'
        )
    context.artifact_store.verify(artifacts[0])

    # This full pickle load is allowed only after the run-local path and
    # recorded SHA256 above have been verified. MMEngine checkpoints contain
    # HistoryBuffer objects that cannot be loaded through weights_only=True.
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError('detector resume checkpoint must be a mapping')
    required = {'state_dict', 'optimizer', 'param_schedulers', 'meta'}
    missing_sections = sorted(required.difference(checkpoint))
    if missing_sections:
        raise ValueError(
            'detector resume checkpoint is missing: {}'.format(
                ', '.join(missing_sections)
            )
        )
    state_dict = checkpoint['state_dict']
    if not isinstance(state_dict, Mapping):
        raise TypeError('detector checkpoint state_dict must be a mapping')
    expected_state = _unwrap_model(model).state_dict()
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
        raise ValueError('detector resume requires nonempty optimizer state')
    if (
        not isinstance(scheduler_state, (Mapping, Sequence))
        or isinstance(scheduler_state, (str, bytes))
        or not scheduler_state
    ):
        raise ValueError(
            'detector resume requires nonempty param-scheduler state'
        )
    meta = checkpoint['meta']
    if not isinstance(meta, Mapping):
        raise TypeError('detector checkpoint meta must be a mapping')
    iteration = meta.get('global_iteration')
    if iteration not in set(int(value) for value in expected_iterations):
        raise ValueError(
            'detector checkpoint global iteration {} is not one of {}'.format(
                iteration, tuple(expected_iterations)
            )
        )
    runner_iteration = meta.get('iter')
    if runner_iteration not in (iteration, iteration + 1):
        raise ValueError(
            'detector checkpoint runner iteration {} is incompatible with '
            'global iteration {}'.format(runner_iteration, iteration)
        )


def _bind_exact_runner_resume_iteration(
    runner: Any,
    checkpoint_path: Path,
    expected_iteration: int,
) -> None:
    '''Normalize MMEngine's legacy by-iteration checkpoint offset in memory.'''

    checkpoint_path = Path(checkpoint_path).resolve()
    load_checkpoint = runner.load_checkpoint

    def load_checkpoint_with_exact_iteration(filename, *args, **kwargs):
        loaded = load_checkpoint(filename, *args, **kwargs)
        if Path(str(filename)).resolve() != checkpoint_path:
            raise ValueError('Runner loaded an unexpected resume checkpoint')
        meta = loaded.get('meta')
        if not isinstance(meta, MutableMapping):
            raise TypeError('detector checkpoint meta must be mutable')
        if meta.get('global_iteration') != expected_iteration:
            raise ValueError('Runner checkpoint global iteration changed')
        if meta.get('iter') not in (
            expected_iteration, expected_iteration + 1
        ):
            raise ValueError('Runner checkpoint iteration is incompatible')
        meta['iter'] = expected_iteration
        return loaded

    runner.load_checkpoint = load_checkpoint_with_exact_iteration


def _atomic_runner_checkpoint(
    runner: Any,
    checkpoint_path: Path,
    stage: StageSpec,
    iteration: int,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(
        '.{}.tmp.pth'.format(checkpoint_path.stem)
    )
    runner.save_checkpoint(
        str(temporary.parent),
        filename=temporary.name,
        save_optimizer=True,
        save_param_scheduler=True,
        by_epoch=False,
        meta={
            'adaod_stage_id': stage.stage_id,
            'global_iteration': iteration,
            'iter': iteration,
        },
    )
    if not temporary.is_file():
        raise RuntimeError('MMEngine did not write the requested checkpoint')
    os.replace(str(temporary), str(checkpoint_path))
    return checkpoint_path


class _CyclingProvider:
    def __init__(self, dataloader: Iterable, branch: str) -> None:
        self.dataloader = dataloader
        self.branch = branch
        self.iterator = iter(dataloader)

    def __call__(self, iteration: int):
        del iteration
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return self.branch, batch


def _single_dataset_loader(
    dataset: Mapping[str, Any], batch_size: int
) -> MutableMapping[str, Any]:
    return dict(
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=None,
        collate_fn=dict(type='pseudo_collate'),
        dataset=copy.deepcopy(dataset),
    )


def _write_unlabeled_pool_manifest(
    context: ExecutionContext,
    samples: Sequence[SampleIdentity],
    pool: Optional[PoolState] = None,
    producer_stage_id: str = 'target_pool_materialization',
) -> Path:
    source_path = dataset_cache_directory(context) / 'target_train_unlabeled.json'
    if pool is None:
        pool = _active_pool(context)
    if tuple(samples) != pool.unlabeled:
        raise ValueError('requested scoring samples do not match the active pool')
    if not samples:
        raise RuntimeError('ADA-FNP requires a nonempty target-unlabeled pool')
    with source_path.open('r', encoding='utf-8') as stream:
        source = json.load(stream)
    images_by_sample = {
        SampleIdentity.parse(image['sample_id']): image
        for image in source['images']
    }
    if len(images_by_sample) != len(source['images']):
        raise ValueError('target pool cache contains duplicate sample IDs')
    if set(images_by_sample) != set(pool.universe):
        raise ValueError('target pool cache does not match the committed universe')
    unknown = tuple(sample for sample in samples if sample not in images_by_sample)
    if unknown:
        raise ValueError(
            'current pool sample is absent from the target cache: {}'.format(
                unknown[0].qualified_id
            )
        )
    output = (
        context.run_directory / 'datasets' /
        'target_train_unlabeled_pool_{:02d}.json'.format(
            context.state_store.load().active_round
        )
    )
    relative = output.relative_to(context.run_directory).as_posix()
    context.artifact_store.write_json(relative, {
        'info': dict(source.get('info', {})),
        'images': [images_by_sample[sample] for sample in samples],
        'annotations': [],
        'categories': source['categories'],
    }, 'target_unlabeled_annotations', producer_stage_id)
    return output


def _active_pool(context: ExecutionContext) -> PoolState:
    round_index = context.state_store.load().active_round
    path = (
        context.run_directory / 'artifacts/pool' /
        'round_{:02d}.json'.format(round_index)
    )
    if not path.is_file():
        raise FileNotFoundError('active target pool is missing: {!s}'.format(path))
    with path.open('r', encoding='utf-8') as stream:
        return PoolState.from_dict(json.load(stream))


def _pool_samples_by_image_id(
    manifest_path: Path,
    expected_samples: Sequence[SampleIdentity],
) -> Mapping[int, SampleIdentity]:
    with manifest_path.open('r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    mapping = {}
    for image in manifest['images']:
        image_id = int(image['id'])
        sample = SampleIdentity.parse(image['sample_id'])
        if image_id in mapping:
            raise ValueError('target pool manifest contains duplicate image IDs')
        mapping[image_id] = sample
    if (
        len(mapping) != len(expected_samples)
        or set(mapping.values()) != set(expected_samples)
    ):
        raise ValueError('target pool manifest does not cover expected samples')
    return mapping


def _teacher_supervision_extractor(teacher: nn.Module, value: Any):
    branch, batch = value
    branch_batch = {
        'inputs': batch['inputs'][branch],
        'data_samples': batch['data_samples'][branch],
    }
    processed = teacher.detector.data_preprocessor(branch_batch, training=False)
    inputs = processed['inputs']
    data_samples = processed['data_samples']
    features = teacher.extract_domain_feature(inputs)
    predictions = teacher.predict(inputs, data_samples, rescale=False)
    counts = []
    for prediction, data_sample in zip(predictions, data_samples):
        instances = getattr(prediction, 'pred_instances', prediction)
        ground_truth = data_sample.gt_instances
        counts.append(count_false_negatives(
            instances.bboxes,
            instances.scores,
            instances.labels,
            ground_truth.bboxes,
            ground_truth.labels,
            iou_threshold=0.5,
            max_detections=100,
        ))
    return features, features.new_tensor(counts)


class MmdetExecutionBackend:
    '''Run ADA-FNP stages through the pinned MMEngine/MMDetection stack.'''

    def __init__(
        self,
        runtime_loader: Callable[[], MmdetRuntime] = _load_mmdet_runtime,
        *,
        require_cuda: bool = True,
    ) -> None:
        self.runtime_loader = runtime_loader
        self.require_cuda = require_cuda

    def _runtime(self) -> MmdetRuntime:
        runtime = self.runtime_loader()
        if self.require_cuda and not torch.cuda.is_available():
            raise ExecutionDependencyError(
                'ADA-FNP model stages require a CUDA-enabled PyTorch runtime'
            )
        return runtime

    def _inference_model(
        self,
        runtime: MmdetRuntime,
        context: ExecutionContext,
        checkpoint_path: Path,
    ) -> Tuple[nn.Module, MutableMapping[str, Any]]:
        config = _base_config(runtime, context)
        model = runtime.build_model(config['model'])
        runtime.load_model_checkpoint(model, checkpoint_path)
        if self.require_cuda:
            model = model.cuda()
        model.eval()
        return model, config

    def train_detector(
        self, stage, context, phase, checkpoint_path, resume_from
    ) -> Path:
        runtime = self._runtime()
        config = build_detector_stage_config(
            runtime, context, phase, resume_from, stage.stage_id
        )
        runner = runtime.build_runner(config)
        if resume_from is not None:
            validate_detector_resume_checkpoint(
                resume_from,
                runner.model,
                (phase.start_iteration, phase.end_iteration),
                context=context,
            )
            _bind_exact_runner_resume_iteration(
                runner, resume_from, phase.start_iteration
            )
        runner.train()
        if int(runner.iter) != phase.end_iteration:
            raise RuntimeError(
                'MMEngine stopped at iteration {}, expected {}'.format(
                    runner.iter, phase.end_iteration
                )
            )
        model = _unwrap_model(runner.model)
        if phase.initialize_teacher_at_end:
            model.teacher.load_state_dict(model.student.state_dict(), strict=True)
        return _atomic_runner_checkpoint(
            runner, checkpoint_path, stage, phase.end_iteration
        )

    def create_fnpm_session(self, stage, context, checkpoint_path):
        del stage
        runtime = self._runtime()
        detector_checkpoint = completed_checkpoint(context, 'detector_checkpoint')
        if detector_checkpoint is None:
            raise FileNotFoundError('FNPM training requires a detector checkpoint')
        model, config = self._inference_model(
            runtime, context, detector_checkpoint
        )
        model = _unwrap_model(model)
        fnpm = FalseNegativePredictionModule(in_channels=512)
        if checkpoint_path is None:
            previous = completed_checkpoint(context, 'fnpm_checkpoint')
            if previous is not None:
                payload = load_checkpoint(previous)
                fnpm.load_state_dict(payload['fnpm']['model'], strict=True)
        if self.require_cuda:
            fnpm = fnpm.cuda()

        initial_datasets = config['stage_overrides']['initial'][
            'train_dataloader'
        ]['dataset']['datasets']
        source_loader_config = _single_dataset_loader(
            initial_datasets[0], int(context.config['training']['source_batch_size'])
        )
        _configure_dataloader(source_loader_config, context, None)
        source_loader = runtime.build_dataloader(
            source_loader_config, int(context.config['seed'])
        )
        labeled_provider = None
        labeled_manifest = _labeled_manifest(context)
        if labeled_manifest is not None:
            adaptation_datasets = config['stage_overrides']['adaptation'][
                'train_dataloader'
            ]['dataset']['datasets']
            labeled_loader_config = _single_dataset_loader(
                adaptation_datasets[1],
                int(context.config['training']['target_labeled_batch_size']),
            )
            _configure_dataloader(
                labeled_loader_config, context, labeled_manifest
            )
            labeled_provider = _CyclingProvider(
                runtime.build_dataloader(
                    labeled_loader_config, int(context.config['seed'])
                ),
                'target_labeled',
            )
        return FnpmSession(
            fnpm=fnpm,
            teacher=model.teacher,
            source_batch_provider=_CyclingProvider(source_loader, 'source'),
            teacher_batch_extractor=_teacher_supervision_extractor,
            labeled_target_batch_provider=labeled_provider,
        )

    def score_pool(self, stage, context, samples):
        runtime = self._runtime()
        detector_checkpoint = completed_checkpoint(context, 'detector_checkpoint')
        fnpm_checkpoint = completed_checkpoint(context, 'fnpm_checkpoint')
        if detector_checkpoint is None or fnpm_checkpoint is None:
            raise FileNotFoundError('pool scoring requires detector and FNPM checkpoints')
        model, config = self._inference_model(
            runtime, context, detector_checkpoint
        )
        model = _unwrap_model(model)
        fnpm = FalseNegativePredictionModule(in_channels=512)
        payload = load_checkpoint(fnpm_checkpoint)
        fnpm.load_state_dict(payload['fnpm']['model'], strict=True)
        if self.require_cuda:
            fnpm = fnpm.cuda()
        fnpm.eval()
        acquisition_dataset = copy.deepcopy(config['target_acquisition_dataset'])
        pool_manifest = _write_unlabeled_pool_manifest(
            context, samples, producer_stage_id=stage.stage_id
        )
        acquisition_dataset['ann_file'] = str(pool_manifest)
        samples_by_image_id = _pool_samples_by_image_id(
            pool_manifest, samples
        )
        dataloader_config = dict(
            batch_size=1,
            num_workers=4,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=acquisition_dataset,
        )
        _configure_dataloader(dataloader_config, context, None)
        dataloader = runtime.build_dataloader(
            dataloader_config, int(context.config['seed'])
        )
        records = []
        seen_samples = set()
        teacher = model.teacher
        with torch.no_grad():
            for batch in dataloader:
                processed = teacher.detector.data_preprocessor(
                    batch, training=False
                )
                inputs = processed['inputs']
                data_samples = processed['data_samples']
                features = teacher.extract_domain_feature(inputs)
                predictions = model.predict_teacher_fixed_proposals(
                    inputs,
                    data_samples,
                    passes=int(context.config['acquisition']['mc_passes']),
                )
                fn_scores = fnpm(features)
                source_probabilities = teacher.domain_discriminator.source_probability(
                    features
                )
                if len(predictions) != len(data_samples):
                    raise ValueError(
                        'acquisition predictions and data samples differ'
                    )
                for index, (prediction, data_sample) in enumerate(
                    zip(predictions, data_samples)
                ):
                    image_id = int(data_sample.metainfo['img_id'])
                    if image_id not in samples_by_image_id:
                        raise ValueError(
                            'acquisition dataloader returned an unknown image ID'
                        )
                    sample = samples_by_image_id[image_id]
                    if sample in seen_samples:
                        raise ValueError(
                            'acquisition dataloader returned a sample twice'
                        )
                    count = len(prediction.bboxes)
                    localization = (
                        prediction.box_variances.mean()
                        if count else features.new_zeros(())
                    )
                    entropy = foreground_entropy(prediction.class_probabilities)
                    source_probability = source_probabilities[index].mean()
                    diversity = domain_diversity(source_probability)
                    records.append(RawAdaFnpScore(
                        sample=sample,
                        false_negative=float(fn_scores[index].detach().cpu()),
                        localization=float(localization.detach().cpu()),
                        entropy=float(entropy.detach().cpu()),
                        diversity=float(diversity.detach().cpu()),
                        source_domain_probability=float(
                            source_probability.detach().cpu()
                        ),
                        detection_count=count,
                    ))
                    seen_samples.add(sample)
        if seen_samples != set(samples):
            raise ValueError('acquisition dataloader does not cover the pool')
        return tuple(records)

    def evaluate(self, stage, context, checkpoint_path):
        del stage
        runtime = self._runtime()
        config = _base_config(runtime, context)
        _configure_dataloader(config['test_dataloader'], context, None)
        config['load_from'] = str(checkpoint_path)
        config['resume'] = False
        config['work_dir'] = str(context.run_directory / 'mmengine/evaluation')
        metrics = dict(runtime.build_runner(config).test())
        ap50_keys = [key for key in metrics if key.split('/')[-1] == 'AP50']
        if len(ap50_keys) != 1:
            raise ValueError('VOC evaluator must return exactly one AP50 metric')
        ap50 = float(metrics[ap50_keys[0]])
        if not math.isfinite(ap50) or not 0.0 <= ap50 <= 100.0:
            raise ValueError('PT VOC AP50 must be a finite percentage in [0, 100]')
        return {'AP50': ap50}


__all__ = [
    'AdaFnpExecutionBackend',
    'ExecutionDependencyError',
    'FnpmSession',
    'MmdetExecutionBackend',
    'MmdetRuntime',
    'build_detector_stage_config',
]
