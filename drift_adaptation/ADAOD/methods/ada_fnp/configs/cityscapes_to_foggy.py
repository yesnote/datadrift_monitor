'''ADA-FNP reproduction config for Cityscapes to Foggy Cityscapes.'''

_base_ = [
    '../../../configs/_base_/datasets/cityscapes_to_foggy.py',
    '../../../configs/_base_/models/faster_rcnn_vgg16.py',
    '../../../configs/_base_/schedules/iter_40k.py',
    '../../../configs/_base_/default_runtime.py',
]

from copy import deepcopy

from configs._base_.datasets.cityscapes_to_foggy import (
    source_train_dataset as _source_train_dataset,
    target_acquisition_dataset as _target_acquisition_dataset,
    target_labeled_dataset as _target_labeled_dataset,
    target_unlabeled_dataset as _target_unlabeled_dataset,
)
from configs._base_.models.faster_rcnn_vgg16_factory import (
    build_faster_rcnn_vgg16 as _build_faster_rcnn_vgg16,
)
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
initial_stage_train_dataloader = _train_loader(
    (_source_train_dataset, _target_unlabeled_dataset),
    batch_size=_SOURCE_BATCH_SIZE + _TARGET_UNLABELED_BATCH_SIZE,
    source_ratio=(_SOURCE_BATCH_SIZE, _TARGET_UNLABELED_BATCH_SIZE),
    branch_field=_INITIAL_BRANCHES,
)
adaptation_stage_train_dataloader = _train_loader(
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
train_dataloader = deepcopy(initial_stage_train_dataloader)
# The score-pool executor consumes this deterministic, annotation-free view.
target_acquisition_dataset = deepcopy(_target_acquisition_dataset)


detector = _build_faster_rcnn_vgg16()
detector['roi_head']['type'] = 'ADAFNPRoIHead'
detector['roi_head']['bbox_head'].update(
    dropout=_DROPOUT_PROBABILITY,
    reg_class_agnostic=_METHOD_CONFIG['detector'][
        'class_agnostic_bbox_regression'
    ],
)

model = dict(
    _delete_=True,
    type='ADAFNPDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=deepcopy(detector['data_preprocessor']),
    ),
    domain_discriminator=dict(
        type='ADAFNPDomainDiscriminator',
        in_channels=512,
        hidden_channels=64,
    ),
    grl_scale=1.0,
    domain_loss_weight=_METHOD_CONFIG['training']['adversarial_weight'],
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

mean_teacher_hook = dict(
    type='MeanTeacherHook',
    momentum=_METHOD_CONFIG['training']['mmdet_ema_momentum'],
    interval=1,
    skip_buffer=True,
    priority=49,
)

# MultiBranch emits the exact keys consumed by ADAFNPDetector. No executor
# rename is necessary: the detector owns weak-teacher pseudo-label generation
# and strong-student loss routing.
stage_overrides = dict(
    initial=dict(
        train_dataloader=deepcopy(initial_stage_train_dataloader),
        model=dict(enable_unsupervised_loss=False),
        custom_hooks=[],
    ),
    adaptation=dict(
        train_dataloader=deepcopy(adaptation_stage_train_dataloader),
        model=dict(enable_unsupervised_loss=True),
        custom_hooks=[deepcopy(mean_teacher_hook)],
    ),
)

# The initial UDA segment has no running teacher. The stage executor copies the
# student branch at 5k, then installs the adaptation hook below.
custom_hooks = []

del _METHOD_CONFIG
