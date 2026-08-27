'''MMDetection configuration plumbing shared by ADA methods.'''

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional

from methods.common.engine.context import ExecutionContext
from methods.common.execution.run_files import (
    dataset_cache_directory,
    materialize_unlabeled_pool_manifest,
    read_active_pool,
    target_labeled_manifest_path,
)
from methods.common.mmdet.models.backbones.vgg16_caffe_checkpoint import (
    CHECKPOINT_PATH,
)
from methods.common.protocols.ada_fnp_detection import (
    DetectorTrainingMode,
    DetectorTrainingPhase,
)


def materialize_config_replacement(
    value: Mapping[str, Any],
    name: str,
) -> MutableMapping[str, Any]:
    resolved = copy.deepcopy(dict(value))
    delete_directive = resolved.pop('_delete_', None)
    if delete_directive is not None and delete_directive is not True:
        raise ValueError('{} _delete_ directive must be true'.format(name))
    return resolved


def _labeled_target_manifest(context: ExecutionContext) -> Optional[Path]:
    round_index = context.state_store.load().active_round
    if round_index == 0:
        return None
    path = target_labeled_manifest_path(context, round_index)
    if not path.is_file():
        raise FileNotFoundError(
            'selected-target manifest is missing: {!s}'.format(path)
        )
    return path


def _configure_dataset_paths(
    dataset: MutableMapping[str, Any],
    context: ExecutionContext,
    labeled_manifest: Optional[Path],
    unlabeled_manifest: Optional[Path],
) -> None:
    children = dataset.get('datasets')
    if children is not None:
        for child in children:
            _configure_dataset_paths(
                child,
                context,
                labeled_manifest,
                unlabeled_manifest,
            )
        return
    ann_file = dataset.get('ann_file')
    if ann_file is None:
        return
    configured_ann_file = Path(str(ann_file))
    filename = configured_ann_file.name
    cache_directory = dataset_cache_directory(context)
    if configured_ann_file.is_absolute():
        resolved_ann_file = configured_ann_file.resolve()
        allowed_roots = (context.run_directory, cache_directory)
        if not any(
            resolved_ann_file == root or root in resolved_ann_file.parents
            for root in allowed_roots
        ):
            raise ValueError(
                'absolute annotation file is outside run/cache roots'
            )
    elif filename == 'target_train_labeled.json':
        if labeled_manifest is None:
            raise RuntimeError(
                'adaptation dataset requested before annotation reveal'
            )
        resolved_ann_file = labeled_manifest
    elif filename == 'target_train_unlabeled.json':
        if unlabeled_manifest is None:
            raise RuntimeError(
                'target-unlabeled dataset requires an active pool'
            )
        resolved_ann_file = unlabeled_manifest
    else:
        resolved_ann_file = cache_directory / filename
    if not resolved_ann_file.is_file():
        raise FileNotFoundError(
            'MMDetection annotation file is missing: {!s}'.format(
                resolved_ann_file
            )
        )
    dataset['ann_file'] = str(resolved_ann_file)
    dataset['data_root'] = str(context.repository_root)
    dataset['data_prefix'] = dict(img='')


def configure_dataloader(
    dataloader: MutableMapping[str, Any],
    context: ExecutionContext,
    labeled_manifest: Optional[Path] = None,
    unlabeled_manifest: Optional[Path] = None,
) -> None:
    dataset = dataloader.get('dataset')
    if not isinstance(dataset, MutableMapping):
        raise TypeError('dataloader dataset must be a mutable mapping')
    _configure_dataset_paths(
        dataset,
        context,
        labeled_manifest,
        unlabeled_manifest,
    )
    sampler = dataloader.get('sampler')
    if isinstance(sampler, MutableMapping) and 'seed' in sampler:
        sampler['seed'] = int(context.config['seed'])


def configure_multi_source_loader(
    dataloader: MutableMapping[str, Any],
    source_ratio: tuple,
) -> None:
    if not source_ratio or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in source_ratio
    ):
        raise ValueError('multi-source batch sizes must be positive integers')
    sampler = dataloader.get('sampler')
    if not isinstance(sampler, MutableMapping):
        raise TypeError('multi-source dataloader requires a sampler mapping')
    batch_size = sum(source_ratio)
    dataloader['batch_size'] = batch_size
    sampler['batch_size'] = batch_size
    sampler['source_ratio'] = list(source_ratio)


def positive_batch_size(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    if value <= 0:
        raise ValueError('{} must be positive'.format(name))
    return value


def apply_detector_training_config(
    config: MutableMapping[str, Any],
    context: ExecutionContext,
) -> None:
    '''Apply the shared detector, loader, optimizer, and schedule settings.'''

    resolved = context.config
    training = resolved['training']
    detector = resolved['detector']
    inference = resolved['inference']
    faster_rcnn = config['model']['detector']
    bbox_head = faster_rcnn['roi_head']['bbox_head']
    bbox_head['num_classes'] = int(detector['num_classes'])
    bbox_head['reg_class_agnostic'] = bool(
        detector['class_agnostic_bbox_regression']
    )
    source_batch_size = int(training['source_batch_size'])
    target_labeled_batch_size = int(training['target_labeled_batch_size'])
    target_unlabeled_batch_size = int(training['target_unlabeled_batch_size'])
    stage_overrides = config['stage_overrides']
    configure_multi_source_loader(
        stage_overrides['initial']['train_dataloader'],
        (source_batch_size, target_unlabeled_batch_size),
    )
    configure_multi_source_loader(
        stage_overrides['unlabeled_adaptation']['train_dataloader'],
        (source_batch_size, target_unlabeled_batch_size),
    )
    configure_multi_source_loader(
        stage_overrides['adaptation']['train_dataloader'],
        (
            source_batch_size,
            target_labeled_batch_size,
            target_unlabeled_batch_size,
        ),
    )
    evaluation_batch_size = positive_batch_size(
        inference['evaluation_batch_size'],
        'evaluation_batch_size',
    )
    for dataloader_name in ('val_dataloader', 'test_dataloader'):
        dataloader = config.get(dataloader_name)
        if dataloader is not None:
            dataloader['batch_size'] = evaluation_batch_size
    optimizer = config['optim_wrapper']['optimizer']
    optimizer['lr'] = float(training['lr'])
    optimizer['momentum'] = float(training['momentum'])
    optimizer['weight_decay'] = float(training['weight_decay'])
    max_norm = float(training['gradient_clip_max_norm'])
    norm_type = float(training['gradient_clip_norm_type'])
    if max_norm <= 0 or norm_type <= 0:
        raise ValueError('gradient clipping values must be positive')
    config['optim_wrapper']['clip_grad'] = dict(
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=True,
    )
    scheduler_configs = config['param_scheduler']
    schedulers = {
        scheduler['type']: scheduler
        for scheduler in scheduler_configs
    }
    if (
        len(scheduler_configs) != 2
        or set(schedulers) != {'LinearLR', 'MultiStepLR'}
    ):
        raise ValueError(
            'the ADA-FNP comparison protocol requires LinearLR and '
            'MultiStepLR schedulers'
        )
    warmup_iterations = int(training['warmup_iterations'])
    maximum_iterations = int(training['max_iterations'])
    if warmup_iterations <= 0 or maximum_iterations <= warmup_iterations:
        raise ValueError('training iteration limits are invalid')
    linear_scheduler = schedulers['LinearLR']
    linear_scheduler.update(
        start_factor=float(training['warmup_start_factor']),
        begin=0,
        end=warmup_iterations,
    )
    multi_step_scheduler = schedulers['MultiStepLR']
    multi_step_scheduler.update(
        begin=0,
        end=maximum_iterations,
        milestones=list(training['lr_milestones']),
        gamma=float(training['lr_decay_factor']),
    )
    config['train_cfg']['max_iters'] = maximum_iterations


def load_method_config(
    runtime,
    context: ExecutionContext,
    config_path: Path,
    apply_method_config: Callable[[MutableMapping, ExecutionContext], None],
    *,
    load_pretrained_backbone: bool = False,
) -> MutableMapping[str, Any]:
    config = runtime.load_config(config_path)
    runtime.import_custom_modules(config)
    apply_detector_training_config(config, context)
    apply_method_config(config, context)
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
    if load_pretrained_backbone:
        backbone['pretrained_checkpoint'] = str(
            context.repository_root / CHECKPOINT_PATH
        )
    else:
        backbone['pretrained_checkpoint'] = None
        backbone['pretrained_sha256'] = None
    return config


def build_segment_config(
    runtime,
    context: ExecutionContext,
    phase: DetectorTrainingPhase,
    continuation_checkpoint: Optional[Path],
    producer_stage_id: str,
    *,
    base_config_loader: Callable,
) -> MutableMapping[str, Any]:
    config = base_config_loader(
        runtime,
        context,
        load_pretrained_backbone=(
            phase.mode is DetectorTrainingMode.INITIALIZATION
            and continuation_checkpoint is None
        ),
    )
    modes = {
        DetectorTrainingMode.INITIALIZATION: 'initial',
        DetectorTrainingMode.UNLABELED_ADAPTATION: 'unlabeled_adaptation',
        DetectorTrainingMode.ADAPTATION: 'adaptation',
    }
    mode = modes[phase.mode]
    override = copy.deepcopy(config['stage_overrides'][mode])
    config['train_dataloader'] = materialize_config_replacement(
        override['train_dataloader'],
        '{} train_dataloader'.format(mode),
    )
    config['model'].update(override.get('model', {}))
    config['custom_hooks'] = override.get('custom_hooks', [])
    active_pool = read_active_pool(context)
    unlabeled_manifest = materialize_unlabeled_pool_manifest(
        context,
        active_pool.unlabeled,
        producer_stage_id,
        pool=active_pool,
    )
    configure_dataloader(
        config['train_dataloader'],
        context,
        _labeled_target_manifest(context),
        unlabeled_manifest,
    )
    config['train_cfg']['max_iters'] = phase.end_iteration
    config['train_cfg']['type'] = 'ADAODSegmentedIterBasedTrainLoop'
    config['train_cfg']['val_interval'] = phase.end_iteration + 1
    config['val_cfg'] = None
    config['val_dataloader'] = None
    config['val_evaluator'] = None
    config['load_from'] = (
        str(continuation_checkpoint)
        if continuation_checkpoint is not None
        else None
    )
    config['resume'] = continuation_checkpoint is not None
    checkpoint_hook = config['default_hooks']['checkpoint']
    checkpoint_hook['interval'] = phase.end_iteration - phase.start_iteration
    checkpoint_hook['by_epoch'] = False
    return config


def build_single_dataset_dataloader(
    dataset: Mapping[str, Any],
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> MutableMapping[str, Any]:
    batch_size = positive_batch_size(batch_size, 'batch_size')
    return dict(
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=drop_last,
        sampler=dict(type='DefaultSampler', shuffle=shuffle),
        batch_sampler=None,
        collate_fn=dict(type='pseudo_collate'),
        dataset=copy.deepcopy(dataset),
    )
