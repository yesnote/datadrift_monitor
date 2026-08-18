'''Static tests for the method-owned MMDetection config overlay.'''

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
    assert bbox_head['reg_class_agnostic'] is True
    assert model['domain_discriminator'] == dict(
        type='ADAFNPDomainDiscriminator',
        in_channels=512,
        hidden_channels=64,
    )
    assert model['grl_scale'] == 1.0
    assert model['domain_loss_weight'] == 0.01

    detector['backbone']['frozen_stages'] = 5
    detector['roi_head']['bbox_head']['dropout'] = 0.5
    base_after = runpy.run_path(str(BASE_MODEL_PATH))['model']
    assert base_before == base_after
    assert base_after['backbone']['frozen_stages'] == -1
    assert base_after['roi_head']['bbox_head']['dropout'] == 0.0


def test_initial_stage_has_no_missing_labeled_target_manifest() -> None:
    config = _load_config()
    stage = config['stage_overrides']['initial']
    dataloader = stage['train_dataloader']
    datasets = dataloader['dataset']['datasets']

    assert dataloader['batch_size'] == 8
    assert dataloader['sampler']['batch_size'] == 8
    assert dataloader['sampler']['source_ratio'] == [4, 4]
    assert len(datasets) == 2
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

    assert config['custom_hooks'] == []
    teacher_hook = config['mean_teacher_hook']
    assert teacher_hook['type'] == 'MeanTeacherHook'
    assert teacher_hook['momentum'] == 0.0004
    assert teacher_hook['interval'] == 1
    assert teacher_hook['skip_buffer'] is True
    assert method['ema_decay'] == 0.9996
