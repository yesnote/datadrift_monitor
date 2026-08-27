'''AADA on the ADA-FNP Cityscapes-to-Foggy comparison protocol.'''

_base_ = [
    '../../../configs/_base_/datasets/cityscapes_to_foggy.py',
    '../../../configs/_base_/models/faster_rcnn_vgg16.py',
    '../../../configs/_base_/schedules/ada_fnp_40k.py',
    '../../../configs/_base_/default_runtime.py',
]

from copy import deepcopy

from methods.aada.configs.default import get_config as _get_method_config


custom_imports = dict(
    imports=['methods.aada.registration'],
    allow_failed_imports=False,
)

_METHOD_CONFIG = _get_method_config()
_SOURCE_BATCH_SIZE = _METHOD_CONFIG['training']['source_batch_size']
_TARGET_LABELED_BATCH_SIZE = _METHOD_CONFIG['training'][
    'target_labeled_batch_size'
]
_TARGET_UNLABELED_BATCH_SIZE = _METHOD_CONFIG['training'][
    'target_unlabeled_batch_size'
]
_source_train_dataset = deepcopy(_base_.source_train_dataset)
_target_labeled_dataset = deepcopy(_base_.target_labeled_dataset)
_target_unlabeled_dataset = deepcopy(_base_.target_unlabeled_dataset)
_target_acquisition_dataset = deepcopy(_base_.target_acquisition_dataset)
_BASIC_BRANCH_PIPELINE = deepcopy(_base_.weak_pipeline)
_INITIAL_BRANCHES = ('source', 'target_unlabeled')
_ADAPTATION_BRANCHES = (
    'source',
    'target_labeled',
    'target_unlabeled',
)


def _dataset_for_branch(dataset, branch, branch_field):
    resolved = deepcopy(dataset)
    multi_branch = resolved['pipeline'][-1]
    if multi_branch.get('type') != 'MultiBranch':
        raise ValueError('AADA training dataset must end with MultiBranch')
    resolved['pipeline'][-1] = dict(
        type='MultiBranch',
        branch_field=list(branch_field),
        **{branch: deepcopy(_BASIC_BRANCH_PIPELINE)},
    )
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
                _dataset_for_branch(dataset, branch, branch_field)
                for dataset, branch in datasets
            ],
        ),
    )


_initial_stage_train_dataloader = _train_loader(
    (
        (_source_train_dataset, 'source'),
        (_target_unlabeled_dataset, 'target_unlabeled'),
    ),
    _SOURCE_BATCH_SIZE + _TARGET_UNLABELED_BATCH_SIZE,
    (_SOURCE_BATCH_SIZE, _TARGET_UNLABELED_BATCH_SIZE),
    _INITIAL_BRANCHES,
)
_adaptation_stage_train_dataloader = _train_loader(
    (
        (_source_train_dataset, 'source'),
        (_target_labeled_dataset, 'target_labeled'),
        (_target_unlabeled_dataset, 'target_unlabeled'),
    ),
    (
        _SOURCE_BATCH_SIZE
        + _TARGET_LABELED_BATCH_SIZE
        + _TARGET_UNLABELED_BATCH_SIZE
    ),
    (
        _SOURCE_BATCH_SIZE,
        _TARGET_LABELED_BATCH_SIZE,
        _TARGET_UNLABELED_BATCH_SIZE,
    ),
    _ADAPTATION_BRANCHES,
)

train_dataloader = deepcopy(_initial_stage_train_dataloader)
target_acquisition_dataset = deepcopy(_target_acquisition_dataset)

_detector = deepcopy(_base_.model)
_detector['roi_head']['type'] = 'ClassProbabilityRoIHead'
_detector['roi_head']['bbox_head']['reg_class_agnostic'] = False

model = dict(
    _delete_=True,
    type='AadaDetector',
    detector=_detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=deepcopy(_detector['data_preprocessor']),
    ),
    domain_discriminator=dict(
        type='ProgressiveDomainDiscriminator',
        in_channels=512,
        hidden_channels=64,
    ),
    grl_scale=_METHOD_CONFIG['domain_adaptation'][
        'gradient_reversal_scale'
    ],
    domain_loss_weight=_METHOD_CONFIG['domain_adaptation']['loss_weight'],
)

stage_overrides = dict(
    initial=dict(
        train_dataloader=deepcopy(_initial_stage_train_dataloader),
        model={},
        custom_hooks=[],
    ),
    unlabeled_adaptation=dict(
        train_dataloader=deepcopy(_initial_stage_train_dataloader),
        model={},
        custom_hooks=[],
    ),
    adaptation=dict(
        train_dataloader=deepcopy(_adaptation_stage_train_dataloader),
        model={},
        custom_hooks=[],
    ),
)
custom_hooks = []

del (
    _ADAPTATION_BRANCHES,
    _BASIC_BRANCH_PIPELINE,
    _INITIAL_BRANCHES,
    _METHOD_CONFIG,
    _SOURCE_BATCH_SIZE,
    _TARGET_LABELED_BATCH_SIZE,
    _TARGET_UNLABELED_BATCH_SIZE,
    _adaptation_stage_train_dataloader,
    _detector,
    _initial_stage_train_dataloader,
    _source_train_dataset,
    _target_acquisition_dataset,
    _target_labeled_dataset,
    _target_unlabeled_dataset,
)
