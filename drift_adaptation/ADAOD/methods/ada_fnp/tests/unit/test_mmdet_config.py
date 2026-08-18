'''Static tests for the method-owned MMDetection config overlay.'''

from copy import deepcopy
from pathlib import Path
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = (
    REPOSITORY_ROOT / 'methods' / 'ada_fnp' / 'configs'
    / 'cityscapes_to_foggy.py'
)
BASE_MODEL_PATH = (
    REPOSITORY_ROOT / 'configs' / '_base_' / 'models'
    / 'faster_rcnn_vgg16.py'
)
BASE_SCHEDULE_PATH = (
    REPOSITORY_ROOT / 'configs' / '_base_' / 'schedules' / 'iter_40k.py'
)


def _load_config() -> dict:
    return runpy.run_path(str(CONFIG_PATH))


def test_config_inherits_only_the_supported_c_to_f_stack() -> None:
    config = _load_config()

    assert config['_base_'] == [
        '../../../configs/_base_/datasets/cityscapes_to_foggy.py',
        '../../../configs/_base_/models/faster_rcnn_vgg16.py',
        '../../../configs/_base_/schedules/iter_40k.py',
        '../../../configs/_base_/default_runtime.py',
    ]
    assert config['custom_imports'] == dict(
        imports=['methods.ada_fnp.registration'],
        allow_failed_imports=False,
    )


def test_config_wraps_a_complete_faster_rcnn_without_mutating_base() -> None:
    base_before = runpy.run_path(str(BASE_MODEL_PATH))['model']
    config = _load_config()
    model = config['model']
    detector = model['detector']

    assert model['type'] == 'ADAFNPDetector'
    assert model['data_preprocessor']['type'] == 'MultiBranchDataPreprocessor'
    assert model['data_preprocessor']['data_preprocessor'] == (
        detector['data_preprocessor'])
    assert detector['type'] == 'FasterRCNN'
    assert detector['data_preprocessor']['type'] == 'DetDataPreprocessor'
    assert detector['backbone']['type'] == 'ADAODVGG16'
    assert detector['rpn_head']['type'] == 'RPNHead'
    assert detector['roi_head']['type'] == 'ADAFNPRoIHead'
    assert detector['train_cfg']['rpn']['assigner']['type'] == 'MaxIoUAssigner'
    assert detector['test_cfg']['rcnn']['max_per_img'] == 100

    bbox_head = detector['roi_head']['bbox_head']
    assert bbox_head['dropout'] == 0.1
    assert bbox_head['reg_class_agnostic'] is False
    assert model['mc_passes'] == 10
    assert model['localization_variance_threshold'] == 0.1
    assert model['domain_discriminator'] == dict(
        type='ADAFNPDomainDiscriminator',
        in_channels=512,
        hidden_channels=64,
    )
    assert model['grl_scale'] == 1.0
    assert model['domain_loss_weight'] == 0.01

    expected_detector = deepcopy(base_before)
    expected_detector['roi_head']['type'] = 'ADAFNPRoIHead'
    expected_detector['roi_head']['bbox_head'].update(
        dropout=0.1,
        reg_class_agnostic=False,
    )
    assert detector == expected_detector
    assert detector['data_preprocessor'] == base_before['data_preprocessor']
    assert detector['backbone'] == base_before['backbone']
    assert detector['backbone']['pretrained_checkpoint'] == (
        'work_dirs/pretrained/vgg16_caffe.pth'
    )
    assert detector['backbone']['pretrained_sha256'] == (
        '736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e'
    )
    assert detector['roi_head']['bbox_head']['fc_out_channels'] == 1024
    assert detector['roi_head']['bbox_roi_extractor']['roi_layer']['aligned'] is True
    assert detector['rpn_head']['anchor_generator']['scales'] == [8, 16, 32]
    assert detector['rpn_head']['anchor_generator']['center_offset'] == 0.0
    assert detector['train_cfg']['rpn']['assigner']['min_pos_iou'] == 0.0
    assert detector['train_cfg']['rpn']['assigner']['match_low_quality'] is True
    assert detector['train_cfg']['rpn_proposal']['nms_pre'] == 12000
    assert detector['test_cfg']['rpn']['nms_pre'] == 6000

    detector['backbone']['frozen_stages'] = 5
    detector['roi_head']['bbox_head']['dropout'] = 0.5
    base_after = runpy.run_path(str(BASE_MODEL_PATH))['model']
    assert base_before == base_after
    assert base_after['backbone']['frozen_stages'] == 2
    assert base_after['roi_head']['bbox_head']['dropout'] == 0.0


def test_config_references_pt_warmup_schedule() -> None:
    warmup, multistep = runpy.run_path(str(BASE_SCHEDULE_PATH))[
        'param_scheduler'
    ]

    assert warmup == dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=400,
        by_epoch=False,
    )
    assert multistep['type'] == 'MultiStepLR'
    assert multistep['milestones'] == [30000, 35000]
    assert multistep['end'] == 40000


def test_initial_stage_has_no_missing_labeled_target_manifest() -> None:
    config = _load_config()
    stage = config['stage_overrides']['initial']
    dataloader = stage['train_dataloader']
    datasets = dataloader['dataset']['datasets']

    assert dataloader['batch_size'] == 8
    assert dataloader['sampler']['batch_size'] == 8
    assert dataloader['sampler']['source_ratio'] == [4, 4]
    assert len(datasets) == 2
    assert all(
        dataset['pipeline'][-1]['branch_field'] == [
            'source',
            'target_unlabeled_weak',
            'target_unlabeled_strong',
        ]
        for dataset in datasets
    )
    assert datasets[0]['ann_file'].endswith('/source_train.json')
    assert datasets[1]['ann_file'].endswith('/target_train_unlabeled.json')
    assert 'target_train_labeled.json' not in repr(dataloader)
    assert stage['model'] == dict(enable_unsupervised_loss=False)
    assert stage['custom_hooks'] == []
    assert stage['use_pseudo_labels'] is False
    assert stage['branch_routing']['detector_inputs'] == [
        'source',
        'target_unlabeled_weak',
        'target_unlabeled_strong',
    ]
    assert stage['branch_routing']['pseudo_teacher'] is None
    assert stage['branch_routing']['pseudo_student'] is None
    assert config['train_dataloader'] == dataloader
    assert config['model']['enable_unsupervised_loss'] is False


def test_adaptation_stage_adds_revealed_target_and_pseudo_labels() -> None:
    config = _load_config()
    stage = config['stage_overrides']['adaptation']
    dataloader = stage['train_dataloader']
    datasets = dataloader['dataset']['datasets']

    assert dataloader['batch_size'] == 12
    assert dataloader['sampler']['batch_size'] == 12
    assert dataloader['sampler']['source_ratio'] == [4, 4, 4]
    assert len(datasets) == 3
    assert all(
        dataset['pipeline'][-1]['branch_field'] == [
            'source',
            'target_labeled',
            'target_unlabeled_weak',
            'target_unlabeled_strong',
        ]
        for dataset in datasets
    )
    assert datasets[1]['ann_file'].endswith('/target_train_labeled.json')
    assert datasets[2]['ann_file'].endswith('/target_train_unlabeled.json')
    assert stage['model'] == dict(enable_unsupervised_loss=True)
    teacher_hook, = stage['custom_hooks']
    assert teacher_hook == config['mean_teacher_hook']
    assert stage['use_pseudo_labels'] is True
    assert stage['branch_routing']['detector_inputs'] == [
        'source',
        'target_labeled',
        'target_unlabeled_weak',
        'target_unlabeled_strong',
    ]
    assert stage['branch_routing']['pseudo_teacher'] == (
        'target_unlabeled_weak')
    assert stage['branch_routing']['pseudo_student'] == (
        'target_unlabeled_strong')
    assert 'target_train_oracle.json' not in repr(config)


def test_score_pool_dataset_is_deterministic_and_annotation_free() -> None:
    config = _load_config()
    dataset = config['target_acquisition_dataset']
    pipeline = dataset['pipeline']

    assert dataset['ann_file'].endswith('/target_train_unlabeled.json')
    assert [transform['type'] for transform in pipeline] == [
        'LoadImageFromFile', 'LoadEmptyAnnotations', 'Resize', 'PackDetInputs']
    assert pipeline[2] == dict(
        type='Resize', scale=(1333, 600), keep_ratio=True)
    assert all(
        transform['type'] not in {'RandomFlip', 'PTStrongAugmentation'}
        for transform in pipeline)


def test_config_matches_ada_fnp_reproduction_assumptions() -> None:
    config = _load_config()
    method = config['ada_fnp']

    assert method['acquisition_milestones'] == [
        5000, 10000, 15000, 20000, 25000]
    assert method['adversarial_loss_weight'] == 0.01
    assert method['fnpm'] == dict(
        iterations_per_round=2000,
        lr=0.0001,
        matcher_iou_threshold=0.5,
        max_detections=100,
    )
    assert method['acquisition']['mc_passes'] == 10
    assert method['pseudo_label'] == dict(
        localization_variance_threshold=0.1,
        hard_confidence_threshold=None,
    )
    assert method['unlabeled_target_losses'] == ['rpn_cls', 'roi_cls']
    assert config['model']['mc_passes'] == method['acquisition']['mc_passes']
    assert config['model']['localization_variance_threshold'] == (
        method['pseudo_label']['localization_variance_threshold'])

    assert config['custom_hooks'] == []
    teacher_hook = config['mean_teacher_hook']
    assert teacher_hook['type'] == 'MeanTeacherHook'
    assert teacher_hook['momentum'] == 0.0004
    assert teacher_hook['interval'] == 1
    assert teacher_hook['skip_buffer'] is True
    assert method['ema_decay'] == 0.9996
