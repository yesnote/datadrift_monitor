'''ADA-FNP reproduction config for Cityscapes to Foggy Cityscapes.'''

from copy import deepcopy

from configs._base_.datasets.cityscapes_to_foggy import (
    source_train_dataset as _source_train_dataset,
    target_labeled_dataset as _target_labeled_dataset,
    target_unlabeled_dataset as _target_unlabeled_dataset,
)
from configs._base_.models.faster_rcnn_vgg16_factory import (
    build_faster_rcnn_vgg16 as _build_faster_rcnn_vgg16,
)


_base_ = [
    '../../../configs/_base_/datasets/cityscapes_to_foggy.py',
    '../../../configs/_base_/models/faster_rcnn_vgg16.py',
    '../../../configs/_base_/schedules/iter_40k.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['methods.ada_fnp.registration'],
    allow_failed_imports=False,
)

experiment_name = 'ada-fnp_cityscapes-to-foggy_faster-rcnn-vgg16'


def _train_loader(datasets, batch_size, source_ratio):
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
            datasets=[deepcopy(dataset) for dataset in datasets],
        ),
    )


# Initial UDA has no revealed target annotation file.  Adaptation adds the
# selected-only target dataset after each reveal stage has materialized it.
initial_stage_train_dataloader = _train_loader(
    (_source_train_dataset, _target_unlabeled_dataset),
    batch_size=8,
    source_ratio=(4, 4),
)
adaptation_stage_train_dataloader = _train_loader(
    (
        _source_train_dataset,
        _target_labeled_dataset,
        _target_unlabeled_dataset,
    ),
    batch_size=12,
    source_ratio=(4, 4, 4),
)
train_dataloader = deepcopy(initial_stage_train_dataloader)


detector = _build_faster_rcnn_vgg16()
detector['roi_head']['type'] = 'ADAFNPRoIHead'
detector['roi_head']['bbox_head'].update(
    dropout=0.1,
    reg_class_agnostic=True,
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
    domain_loss_weight=0.01,
    enable_unsupervised_loss=False,
    semi_train_cfg=dict(freeze_teacher=True),
    semi_test_cfg=dict(
        predict_on='teacher',
        forward_on='teacher',
        extract_feat_on='teacher',
    ),
)

mean_teacher_hook = dict(
    type='MeanTeacherHook',
    momentum=0.0004,
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
        use_pseudo_labels=False,
        branch_routing=dict(
            detector_inputs=[
                'source',
                'target_unlabeled_weak',
                'target_unlabeled_strong',
            ],
            pseudo_teacher=None,
            pseudo_student=None,
        ),
    ),
    adaptation=dict(
        train_dataloader=deepcopy(adaptation_stage_train_dataloader),
        model=dict(enable_unsupervised_loss=True),
        custom_hooks=[deepcopy(mean_teacher_hook)],
        use_pseudo_labels=True,
        branch_routing=dict(
            detector_inputs=[
                'source',
                'target_labeled',
                'target_unlabeled_weak',
                'target_unlabeled_strong',
            ],
            pseudo_teacher='target_unlabeled_weak',
            pseudo_student='target_unlabeled_strong',
        ),
    ),
)

# The initial UDA segment has no running teacher. The stage executor copies the
# student branch at 5k, then installs the adaptation hook below.
custom_hooks = []

ada_fnp = dict(
    acquisition_milestones=[5000, 10000, 15000, 20000, 25000],
    adversarial_loss_weight=0.01,
    ema_decay=0.9996,
    fnpm=dict(
        iterations_per_round=2000,
        lr=0.0001,
        matcher_iou_threshold=0.5,
        max_detections=100,
    ),
    acquisition=dict(
        budget_percent=1.0,
        mc_passes=10,
        dropout_probability=0.1,
        domain_probability_epsilon=0.000001,
        constant_score_normalized_value=0.5,
        empty_detection_final_score=0.0,
    ),
    pseudo_label=dict(
        localization_variance_threshold=0.1,
        hard_confidence_threshold=None,
    ),
    unlabeled_target_losses=[
        'rpn_cls',
        'roi_cls',
    ],
)
