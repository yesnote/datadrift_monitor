'''Cityscapes to Foggy Cityscapes dataset contract for MMDetection 3.3.'''

_base_ = []

from methods.common.data.cityscapes import CITYSCAPES_CLASSES

# CocoDataset is MMDetection's concrete BaseDetDataset reader for the
# generated COCO-format manifests.  The manifests are build artifacts and
# must never be written beside the source gtFine annotations.
dataset_type = 'CocoDataset'
data_root = ''
backend_args = None

classes = CITYSCAPES_CLASSES
metainfo = dict(classes=classes)
foggy_beta = 0.02
expected_source_train_images = 2975
expected_target_train_images = 2975
expected_target_val_images = 500

dataset_cache_root = 'work_dirs/.dataset_cache/cityscapes-to-foggy'
source_train_ann_file = f'{dataset_cache_root}/source_train.json'
target_pool_ann_file = f'{dataset_cache_root}/target_train_unlabeled.json'
# The reveal stage must override or create this selected-only manifest.  The
# prepared target_train_oracle.json is intentionally absent from this config.
target_labeled_ann_file = f'{dataset_cache_root}/target_train_labeled.json'
target_val_ann_file = f'{dataset_cache_root}/target_val.json'

branch_field = [
    'source',
    'target_labeled',
    'target_unlabeled_weak',
    'target_unlabeled_strong',
]
pack_meta_keys = (
    'img_id',
    'img_path',
    'ori_shape',
    'img_shape',
    'scale_factor',
    'flip',
    'flip_direction',
    'homography_matrix',
)

# Detectron2/PT uses a 600-pixel short edge and its default 1333-pixel maximum
# long edge. Geometry runs once before MultiBranch so weak and strong target
# views differ only in photometric appearance.
shared_geometric_pipeline = [
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
]
weak_pipeline = [
    dict(type='PackDetInputs', meta_keys=pack_meta_keys),
]
strong_pipeline = [
    dict(type='PTStrongAugmentation', p=1.0),
    dict(type='PackDetInputs', meta_keys=pack_meta_keys),
]
source_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    *shared_geometric_pipeline,
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        source=weak_pipeline,
    ),
]
target_labeled_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    *shared_geometric_pipeline,
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        target_labeled=weak_pipeline,
    ),
]
target_unlabeled_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    *shared_geometric_pipeline,
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        target_unlabeled_weak=weak_pipeline,
        target_unlabeled_strong=strong_pipeline,
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=pack_meta_keys[:-1]),
]

# Pool scoring must not add random data uncertainty to ADA-FNP's MC Dropout
# uncertainty. It uses the same deterministic resize as evaluation.
target_acquisition_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=pack_meta_keys),
]

source_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=source_train_ann_file,
    # Cache file_name fields already contain repository-relative image paths.
    data_prefix=dict(img=''),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    pipeline=source_train_pipeline,
    backend_args=backend_args,
)
target_labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=target_labeled_ann_file,
    data_prefix=dict(img=''),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    pipeline=target_labeled_pipeline,
    backend_args=backend_args,
)
target_unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=target_pool_ann_file,
    data_prefix=dict(img=''),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    pipeline=target_unlabeled_pipeline,
    backend_args=backend_args,
)
target_acquisition_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=target_pool_ann_file,
    data_prefix=dict(img=''),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=False, min_size=1),
    pipeline=target_acquisition_pipeline,
    backend_args=backend_args,
)

# The base loader is safe before the first acquisition: it never attempts to
# open the selected-only target manifest.  Method adaptation stages add the
# target_labeled dataset only after the reveal stage has created it.
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='MultiSourceSampler',
        batch_size=2,
        source_ratio=[1, 1],
        shuffle=True,
        seed=0,
    ),
    batch_sampler=None,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            source_train_dataset,
            target_unlabeled_dataset,
        ],
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=target_val_ann_file,
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# The cache stores Detectron2-compatible zero-based, half-open boxes. Use PT's
# VOC2012-style continuous AP without MMDetection's legacy ``+1`` arithmetic.
val_evaluator = dict(
    type='PTVOCMetric',
    iou_thrs=0.5,
    metric='mAP',
    eval_mode='area',
)
test_evaluator = val_evaluator
