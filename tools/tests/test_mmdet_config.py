'''Dependency-free checks for the shared MMDetection config hierarchy.'''

from pathlib import Path, PurePosixPath
import runpy

import pytest

from methods.common.data.cityscapes import CITYSCAPES_CLASSES


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = REPOSITORY_ROOT / 'configs' / '_base_'
METHOD_CONFIG = (
    REPOSITORY_ROOT / 'methods' / 'ada_fnp' / 'configs'
    / 'cityscapes_to_foggy.py'
)


def _load(relative_path: str) -> dict:
    return runpy.run_path(str(BASE_ROOT / relative_path))


def test_config_sources_compile_without_openmmlab_dependencies() -> None:
    paths = sorted(BASE_ROOT.rglob('*.py')) + [METHOD_CONFIG]

    for path in paths:
        source = path.read_text(encoding='utf-8')
        compile(source, str(path), 'exec')


def test_c_to_f_dataset_uses_coco_caches_and_multibranch_pipelines() -> None:
    config = _load('datasets/cityscapes_to_foggy.py')

    assert config['dataset_type'] == 'CocoDataset'
    assert config['classes'] == CITYSCAPES_CLASSES

    annotation_keys = (
        'source_train_ann_file',
        'target_pool_ann_file',
        'target_labeled_ann_file',
        'target_val_ann_file',
    )
    for key in annotation_keys:
        path = PurePosixPath(config[key])
        assert not path.is_absolute()
        assert path.parts[:2] == ('work_dirs', '.dataset_cache')
        assert path.suffix == '.json'

    assert config['source_train_ann_file'].endswith('/source_train.json')
    assert config['target_pool_ann_file'].endswith(
        '/target_train_unlabeled.json')
    assert config['target_labeled_ann_file'].endswith(
        '/target_train_labeled.json')
    assert config['target_val_ann_file'].endswith('/target_val.json')

    assert config['source_train_dataset']['data_prefix'] == dict(img='')
    target_datasets = config['train_dataloader']['dataset']['datasets'][1:]
    assert all(
        dataset['data_prefix'] == dict(img='')
        for dataset in target_datasets
    )
    assert config['val_dataloader']['dataset']['data_prefix'] == dict(img='')
    assert [
        transform['type'] for transform in config['test_pipeline']
    ] == [
        'LoadImageFromFile', 'Resize', 'LoadAnnotations', 'PackDetInputs']

    assert config['foggy_beta'] == 0.02
    assert config['expected_source_train_images'] == 2975
    assert config['expected_target_train_images'] == 2975
    assert config['expected_target_val_images'] == 500

    text = repr(config)
    assert 'beta_0.005' not in text
    assert 'beta_0.01_' not in text
    assert 'target_train_oracle.json' not in text

    unlabeled_branch = config['target_unlabeled_pipeline'][-1]
    assert unlabeled_branch['type'] == 'MultiBranch'
    assert 'target_unlabeled_weak' in unlabeled_branch
    assert 'target_unlabeled_strong' in unlabeled_branch
    assert config['shared_geometric_pipeline'] == [
        dict(type='Resize', scale=(1333, 600), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
    ]
    assert [transform['type'] for transform in config['weak_pipeline']] == [
        'PackDetInputs']
    assert [transform['type'] for transform in config['strong_pipeline']] == [
        'PTStrongAugmentation', 'PackDetInputs']
    assert config['strong_pipeline'][0] == dict(
        type='PTStrongAugmentation', p=1.0)
    assert sum(
        transform['type'] == 'PTStrongAugmentation'
        for transform in config['strong_pipeline']) == 1
    assert all(
        transform['type'] != 'PhotoMetricDistortion'
        for pipeline in (config['weak_pipeline'], config['strong_pipeline'])
        for transform in pipeline)
    assert [
        transform['type']
        for transform in config['target_unlabeled_pipeline'][:-1]
    ] == [
        'LoadImageFromFile', 'LoadEmptyAnnotations', 'Resize', 'RandomFlip']
    assert sum(
        transform['type'] == 'RandomFlip'
        for transform in config['target_unlabeled_pipeline']) == 1
    acquisition_pipeline = config['target_acquisition_pipeline']
    assert [transform['type'] for transform in acquisition_pipeline] == [
        'LoadImageFromFile', 'LoadEmptyAnnotations', 'Resize', 'PackDetInputs']
    assert acquisition_pipeline[2] == dict(
        type='Resize', scale=(1333, 600), keep_ratio=True)
    assert all(
        transform['type'] not in {'RandomFlip', 'PTStrongAugmentation'}
        for transform in acquisition_pipeline)
    assert config['target_acquisition_dataset']['ann_file'] == (
        config['target_pool_ann_file'])
    assert config['target_acquisition_dataset']['pipeline'] is (
        config['target_acquisition_pipeline'])
    assert 'homography_matrix' in config['pack_meta_keys']


def test_dataset_preserves_annotation_boundary_and_ap50_protocol() -> None:
    config = _load('datasets/cityscapes_to_foggy.py')
    datasets = config['train_dataloader']['dataset']['datasets']

    assert len(datasets) == 2
    assert datasets[0]['ann_file'] == config['source_train_ann_file']
    assert datasets[1]['ann_file'] == config['target_pool_ann_file']
    assert all(
        dataset['ann_file'] != config['target_labeled_ann_file']
        for dataset in datasets)
    assert config['target_pool_ann_file'] != config['target_val_ann_file']
    assert all(
        dataset['filter_cfg']['filter_empty_gt'] is False
        for dataset in datasets)

    evaluator = config['val_evaluator']
    assert evaluator == dict(
        type='PTVOCMetric', iou_thrs=0.5, metric='mAP', eval_mode='area')
    assert config['test_evaluator'] is evaluator


def test_vgg16_faster_rcnn_is_bn_free_and_single_stride() -> None:
    from configs._base_.models.faster_rcnn_vgg16_factory import (
        VGG16_CAFFE_CHECKPOINT,
        VGG16_CAFFE_SHA256,
    )

    config = _load('models/faster_rcnn_vgg16.py')
    model = config['model']

    assert config['custom_imports'] == dict(
        imports=['methods.common.mmdet.registration'],
        allow_failed_imports=False,
    )
    assert model['type'] == 'FasterRCNN'
    assert model['backbone']['type'] == 'ADAODVGG16'
    assert model['backbone']['frozen_stages'] == 2
    assert model['backbone']['pretrained_checkpoint'] == VGG16_CAFFE_CHECKPOINT
    assert model['backbone']['pretrained_sha256'] == VGG16_CAFFE_SHA256
    assert 'norm_cfg' not in model['backbone']
    assert 'neck' not in model
    assert model['rpn_head']['in_channels'] == 512
    assert model['rpn_head']['anchor_generator']['scales'] == [8, 16, 32]
    assert model['rpn_head']['anchor_generator']['strides'] == [16]
    assert model['rpn_head']['anchor_generator']['center_offset'] == 0.0
    assert model['train_cfg']['rpn']['assigner']['min_pos_iou'] == 0.0
    assert model['train_cfg']['rpn']['assigner']['match_low_quality'] is True
    assert model['train_cfg']['rpn']['sampler']['pos_fraction'] == 0.25
    assert model['train_cfg']['rpn_proposal']['nms_pre'] == 12000
    assert model['train_cfg']['rpn_proposal']['max_per_img'] == 2000
    assert model['test_cfg']['rpn']['nms_pre'] == 6000
    assert model['test_cfg']['rpn']['max_per_img'] == 1000
    assert model['roi_head']['bbox_roi_extractor']['featmap_strides'] == [16]
    assert model['roi_head']['bbox_roi_extractor']['roi_layer']['aligned'] is True

    bbox_head = model['roi_head']['bbox_head']
    assert bbox_head['type'] == 'ADAODVGGShared2FCBBoxHead'
    assert bbox_head['fc_out_channels'] == 1024
    assert bbox_head['num_classes'] == 8
    assert bbox_head['dropout'] == 0.0
    assert bbox_head['reg_class_agnostic'] is False
    assert model['data_preprocessor'] == dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=1,
    )


def test_vgg16_factory_returns_independent_detector_configs() -> None:
    from configs._base_.models.faster_rcnn_vgg16_factory import (
        build_faster_rcnn_vgg16,
    )

    first = build_faster_rcnn_vgg16()
    second = build_faster_rcnn_vgg16()
    first['roi_head']['bbox_head']['dropout'] = 0.75
    first['data_preprocessor']['mean'][0] = -1

    assert second['roi_head']['bbox_head']['dropout'] == 0.0
    assert second['data_preprocessor']['mean'][0] == 103.53


def test_iter_schedule_and_runtime_are_iteration_based() -> None:
    schedule = _load('schedules/iter_40k.py')
    runtime = _load('default_runtime.py')

    assert schedule['optim_wrapper']['optimizer'] == dict(
        type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    assert schedule['train_cfg'] == dict(
        type='IterBasedTrainLoop', max_iters=40000, val_interval=5000)
    warmup, scheduler = schedule['param_scheduler']
    assert warmup == dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=400,
        by_epoch=False,
    )
    assert scheduler['type'] == 'MultiStepLR'
    assert scheduler['begin'] == 0
    assert scheduler['by_epoch'] is False
    assert scheduler['end'] == 40000
    assert scheduler['milestones'] == [30000, 35000]
    assert scheduler['gamma'] == 0.1
    assert runtime['default_hooks']['checkpoint']['by_epoch'] is False
    assert runtime['log_processor']['by_epoch'] is False
    assert runtime['randomness'] == dict(seed=0, deterministic=True)


def test_mmengine_loads_composed_config_when_available() -> None:
    pytest.importorskip('mmcv')
    pytest.importorskip('mmdet.registry')
    config_module = pytest.importorskip('mmengine.config')

    config = config_module.Config.fromfile(str(METHOD_CONFIG))

    assert config.model.type == 'ADAFNPDetector'
    assert config.model.data_preprocessor.type == 'MultiBranchDataPreprocessor'
    assert config.model.detector.backbone.type == 'ADAODVGG16'
    assert config.model.detector.roi_head.type == 'ADAFNPRoIHead'
    assert config.model.detector.roi_head.bbox_head.dropout == 0.1
    assert config.train_cfg.max_iters == 40000
    assert config.val_evaluator.type == 'PTVOCMetric'
    assert config.val_evaluator.eval_mode == 'area'
