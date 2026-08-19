'''MMDetection configuration materialization for ADA-FNP execution.'''

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Protocol

from methods.ada_fnp.schedule import (
    DetectorTrainingMode,
    DetectorTrainingPhase,
)
from methods.common.engine.context import ExecutionContext
from methods.common.mmdet.models.backbones.vgg16_caffe_checkpoint import (
    CHECKPOINT_PATH,
)

from .run_files import (
    dataset_cache_directory,
    materialize_unlabeled_pool_manifest,
    read_active_pool,
    target_labeled_manifest_path,
)


class MmdetConfigRuntime(Protocol):
    '''Runtime operations needed to load an MMEngine configuration.'''

    def load_config(self, path: Path) -> MutableMapping[str, Any]:
        ...

    def import_custom_modules(self, config: Mapping[str, Any]) -> None:
        ...


def materialize_config_replacement(
    value: Mapping[str, Any],
    name: str,
) -> MutableMapping[str, Any]:
    '''Return a standalone config after consuming MMEngine's merge directive.'''

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
    '''Resolve dataset manifests and deterministic sampler state in place.'''

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


def _config_path(context: ExecutionContext) -> Path:
    return (
        context.repository_root
        / 'methods/ada_fnp/configs/cityscapes_to_foggy.py'
    )


def _configure_multi_source_loader(
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


def apply_resolved_experiment_config(
    config: MutableMapping[str, Any],
    context: ExecutionContext,
) -> None:
    '''Project the resolved ADAOD config onto the MMDetection config.'''

    resolved = context.config
    training = resolved['training']
    detector = resolved['detector']
    domain_adaptation = resolved['domain_adaptation']
    acquisition = resolved['acquisition']
    pseudo_label = resolved['pseudo_label']

    model = config['model']
    faster_rcnn = model['detector']
    bbox_head = faster_rcnn['roi_head']['bbox_head']
    bbox_head['num_classes'] = int(detector['num_classes'])
    bbox_head['dropout'] = float(detector['dropout_probability'])
    bbox_head['reg_class_agnostic'] = bool(
        detector['class_agnostic_bbox_regression']
    )
    model['grl_scale'] = float(
        domain_adaptation['gradient_reversal_scale']
    )
    model['domain_loss_weight'] = float(domain_adaptation['loss_weight'])
    model['mc_passes'] = int(acquisition['mc_passes'])
    model['localization_variance_threshold'] = float(
        pseudo_label['localization_variance_threshold']
    )

    source_batch_size = int(training['source_batch_size'])
    target_labeled_batch_size = int(
        training['target_labeled_batch_size']
    )
    target_unlabeled_batch_size = int(
        training['target_unlabeled_batch_size']
    )
    stage_overrides = config['stage_overrides']
    _configure_multi_source_loader(
        stage_overrides['initial']['train_dataloader'],
        (source_batch_size, target_unlabeled_batch_size),
    )
    _configure_multi_source_loader(
        stage_overrides['adaptation']['train_dataloader'],
        (
            source_batch_size,
            target_labeled_batch_size,
            target_unlabeled_batch_size,
        ),
    )

    optimizer = config['optim_wrapper']['optimizer']
    optimizer['lr'] = float(training['lr'])
    optimizer['momentum'] = float(training['momentum'])
    optimizer['weight_decay'] = float(training['weight_decay'])

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
            'ADA-FNP requires one LinearLR and one MultiStepLR scheduler'
        )
    warmup_iterations = int(training['warmup_iterations'])
    maximum_iterations = int(training['max_iterations'])
    if warmup_iterations <= 0 or maximum_iterations <= warmup_iterations:
        raise ValueError('training iteration limits are invalid')
    linear_scheduler = schedulers['LinearLR']
    linear_scheduler['start_factor'] = float(
        training['warmup_start_factor']
    )
    linear_scheduler['begin'] = 0
    linear_scheduler['end'] = warmup_iterations
    multi_step_scheduler = schedulers['MultiStepLR']
    multi_step_scheduler['begin'] = 0
    multi_step_scheduler['end'] = maximum_iterations
    multi_step_scheduler['milestones'] = list(training['lr_milestones'])
    multi_step_scheduler['gamma'] = float(training['lr_decay_factor'])
    config['train_cfg']['max_iters'] = maximum_iterations

    teacher_decay = float(training['teacher_ema_decay'])
    if not 0.0 <= teacher_decay <= 1.0:
        raise ValueError('teacher EMA decay must be between zero and one')
    adaptation_hooks = stage_overrides['adaptation']['custom_hooks']
    teacher_hooks = [
        hook
        for hook in adaptation_hooks
        if hook.get('type') == 'MeanTeacherHook'
    ]
    if len(teacher_hooks) != 1:
        raise ValueError('adaptation requires exactly one MeanTeacherHook')
    teacher_hooks[0]['momentum'] = 1.0 - teacher_decay


def load_base_config(
    runtime: MmdetConfigRuntime,
    context: ExecutionContext,
    *,
    load_pretrained_backbone: bool = False,
) -> MutableMapping[str, Any]:
    '''Load the scenario config and bind all run-specific runtime settings.'''

    config = runtime.load_config(_config_path(context))
    runtime.import_custom_modules(config)
    apply_resolved_experiment_config(config, context)
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


def build_detector_stage_config(
    runtime: MmdetConfigRuntime,
    context: ExecutionContext,
    phase: DetectorTrainingPhase,
    continuation_checkpoint: Optional[Path],
    producer_stage_id: str,
) -> MutableMapping[str, Any]:
    '''Build one detector segment while preserving the global schedule.'''

    config = load_base_config(
        runtime,
        context,
        load_pretrained_backbone=(
            phase.mode is DetectorTrainingMode.INITIALIZATION
            and continuation_checkpoint is None
        ),
    )
    mode = (
        'initial'
        if phase.mode is DetectorTrainingMode.INITIALIZATION
        else 'adaptation'
    )
    override = copy.deepcopy(config['stage_overrides'][mode])
    config['train_dataloader'] = materialize_config_replacement(
        override['train_dataloader'],
        '{} train_dataloader'.format(mode),
    )
    config['model'].update(override['model'])
    config['custom_hooks'] = override['custom_hooks']

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
    '''Build a dataloader config for one detector dataset.'''

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('batch_size must be an integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
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
