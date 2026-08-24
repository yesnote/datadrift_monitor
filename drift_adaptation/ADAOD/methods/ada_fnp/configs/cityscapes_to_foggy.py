'''ADA-FNP reproduction config for Cityscapes to Foggy Cityscapes.'''

_base_ = [
    '../../../configs/_base_/datasets/cityscapes_to_foggy.py',
    '../../../configs/_base_/models/faster_rcnn_vgg16.py',
    '../../../configs/_base_/default_runtime.py',
]

from copy import deepcopy

from methods.ada_fnp.configs.default import get_config as _get_method_config


custom_imports = dict(
    imports=['methods.ada_fnp.registration'],
    allow_failed_imports=False,
)

_METHOD_CONFIG = _get_method_config()
_DROPOUT_PROBABILITY = _METHOD_CONFIG['detector']['dropout_probability']
_MC_PASSES = _METHOD_CONFIG['acquisition']['mc_passes']
_LOCALIZATION_VARIANCE_THRESHOLD = _METHOD_CONFIG['pseudo_label'][
    'localization_variance_threshold'
]
_SOURCE_BATCH_SIZE = _METHOD_CONFIG['training']['source_batch_size']
_TARGET_LABELED_BATCH_SIZE = _METHOD_CONFIG['training'][
    'target_labeled_batch_size'
]
_TARGET_UNLABELED_BATCH_SIZE = _METHOD_CONFIG['training'][
    'target_unlabeled_batch_size'
]
_source_train_dataset = deepcopy(_base_.source_train_dataset)
_target_acquisition_dataset = deepcopy(_base_.target_acquisition_dataset)
_target_labeled_dataset = deepcopy(_base_.target_labeled_dataset)
_target_unlabeled_dataset = deepcopy(_base_.target_unlabeled_dataset)
_INITIAL_BRANCHES = (
    'source',
    'target_unlabeled_weak',
    'target_unlabeled_strong',
)
_ADAPTATION_BRANCHES = (
    'source',
    'target_labeled',
    'target_unlabeled_weak',
    'target_unlabeled_strong',
)


def _dataset_for_branches(dataset, branch_field):
    resolved = deepcopy(dataset)
    multi_branch = resolved['pipeline'][-1]
    if multi_branch['type'] != 'MultiBranch':
        raise ValueError('training dataset must end with MultiBranch')
    multi_branch['branch_field'] = list(branch_field)
    return resolved


def _train_loader(datasets, batch_size, source_ratio, branch_field):
    return dict(
        _delete_=True,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(
            type='MultiSourceSampler',
            batch_size=batch_size,
            source_ratio=list(source_ratio),
            shuffle=True,
            seed=0,
        ),
        batch_sampler=None,
        collate_fn=dict(type='pseudo_collate'),
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                _dataset_for_branches(dataset, branch_field)
                for dataset in datasets
            ],
        ),
    )


# Initial UDA has no revealed target annotation file.  Adaptation adds the
# selected-only target dataset after each reveal stage has materialized it.
_initial_stage_train_dataloader = _train_loader(
    (_source_train_dataset, _target_unlabeled_dataset),
    batch_size=_SOURCE_BATCH_SIZE + _TARGET_UNLABELED_BATCH_SIZE,
    source_ratio=(_SOURCE_BATCH_SIZE, _TARGET_UNLABELED_BATCH_SIZE),
    branch_field=_INITIAL_BRANCHES,
)
_adaptation_stage_train_dataloader = _train_loader(
    (
        _source_train_dataset,
        _target_labeled_dataset,
        _target_unlabeled_dataset,
    ),
    batch_size=(
        _SOURCE_BATCH_SIZE
        + _TARGET_LABELED_BATCH_SIZE
        + _TARGET_UNLABELED_BATCH_SIZE
    ),
    source_ratio=(
        _SOURCE_BATCH_SIZE,
        _TARGET_LABELED_BATCH_SIZE,
        _TARGET_UNLABELED_BATCH_SIZE,
    ),
    branch_field=_ADAPTATION_BRANCHES,
)
train_dataloader = deepcopy(_initial_stage_train_dataloader)
# The score-pool executor consumes this deterministic, annotation-free view.
target_acquisition_dataset = deepcopy(_target_acquisition_dataset)


_detector = deepcopy(_base_.model)
_detector['roi_head']['type'] = 'AdaFnpMonteCarloDropoutRoIHead'
_detector['roi_head']['bbox_head'].update(
    dropout=_DROPOUT_PROBABILITY,
    reg_class_agnostic=_METHOD_CONFIG['detector'][
        'class_agnostic_bbox_regression'
    ],
)

model = dict(
    _delete_=True,
    type='AdaFnpDetector',
    detector=_detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=deepcopy(_detector['data_preprocessor']),
    ),
    domain_discriminator=dict(
        type='AdaFnpDomainDiscriminator',
        in_channels=512,
        hidden_channels=64,
    ),
    grl_scale=_METHOD_CONFIG['domain_adaptation'][
        'gradient_reversal_scale'
    ],
    domain_loss_weight=_METHOD_CONFIG['domain_adaptation']['loss_weight'],
    enable_unsupervised_loss=False,
    mc_passes=_MC_PASSES,
    localization_variance_threshold=_LOCALIZATION_VARIANCE_THRESHOLD,
    semi_train_cfg=dict(freeze_teacher=True),
    semi_test_cfg=dict(
        predict_on='teacher',
        forward_on='teacher',
        extract_feat_on='teacher',
    ),
)

_mean_teacher_hook = dict(
    type='MeanTeacherHook',
    momentum=1.0 - _METHOD_CONFIG['training']['teacher_ema_decay'],
    interval=1,
    skip_buffer=True,
    priority=49,
)

# MultiBranch emits the exact keys consumed by AdaFnpDetector. No executor
# rename is necessary: the detector owns weak-teacher pseudo-label generation
# and strong-student loss routing.
stage_overrides = dict(
    initial=dict(
        train_dataloader=deepcopy(_initial_stage_train_dataloader),
        model=dict(enable_unsupervised_loss=False),
        custom_hooks=[],
    ),
    adaptation=dict(
        train_dataloader=deepcopy(_adaptation_stage_train_dataloader),
        model=dict(enable_unsupervised_loss=True),
        custom_hooks=[deepcopy(_mean_teacher_hook)],
    ),
)

# The initial UDA segment has no running teacher. The stage executor copies the
# student branch at 5k, then installs the adaptation hook below.
custom_hooks = []

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(
        max_norm=_METHOD_CONFIG['training']['gradient_clip_max_norm'],
        norm_type=_METHOD_CONFIG['training']['gradient_clip_norm_type'],
        error_if_nonfinite=True,
    ),
    optimizer=dict(
        type='SGD',
        lr=_METHOD_CONFIG['training']['lr'],
        momentum=_METHOD_CONFIG['training']['momentum'],
        weight_decay=_METHOD_CONFIG['training']['weight_decay'],
    ),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=_METHOD_CONFIG['training']['warmup_start_factor'],
        begin=0,
        end=_METHOD_CONFIG['training']['warmup_iterations'],
        by_epoch=False,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=_METHOD_CONFIG['training']['max_iterations'],
        by_epoch=False,
        milestones=list(_METHOD_CONFIG['training']['lr_milestones']),
        gamma=_METHOD_CONFIG['training']['lr_decay_factor'],
    ),
]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=_METHOD_CONFIG['training']['max_iterations'],
    val_interval=_METHOD_CONFIG['training']['acquisition_milestones'][0],
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

del (
    _ADAPTATION_BRANCHES,
    _DROPOUT_PROBABILITY,
    _INITIAL_BRANCHES,
    _LOCALIZATION_VARIANCE_THRESHOLD,
    _MC_PASSES,
    _METHOD_CONFIG,
    _SOURCE_BATCH_SIZE,
    _TARGET_LABELED_BATCH_SIZE,
    _TARGET_UNLABELED_BATCH_SIZE,
    _adaptation_stage_train_dataloader,
    _detector,
    _initial_stage_train_dataloader,
    _mean_teacher_hook,
    _source_train_dataset,
    _target_acquisition_dataset,
    _target_labeled_dataset,
    _target_unlabeled_dataset,
)
