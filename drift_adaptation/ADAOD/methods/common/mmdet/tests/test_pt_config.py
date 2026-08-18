'''Static PT detector and schedule parity checks without MMDetection runtime.'''

from pathlib import Path
import runpy

from configs._base_.models.faster_rcnn_vgg16_factory import (
    VGG16_CAFFE_CHECKPOINT,
    VGG16_CAFFE_MD5,
    VGG16_CAFFE_SHA256,
    VGG16_CAFFE_SIZE_BYTES,
    VGG16_CAFFE_URL,
    build_faster_rcnn_vgg16,
)


def test_pt_vgg_asset_provenance_and_backbone_contract():
    assert VGG16_CAFFE_CHECKPOINT == 'work_dirs/pretrained/vgg16_caffe.pth'
    assert VGG16_CAFFE_URL == (
        'https://zenodo.org/records/4515252/files/vgg16_caffe.pth?download=1'
    )
    assert VGG16_CAFFE_SHA256 == (
        '736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e'
    )
    assert VGG16_CAFFE_MD5 == '433ad40ddbd662d6448e13a6cef812f2'
    assert VGG16_CAFFE_SIZE_BYTES == 553433685

    model = build_faster_rcnn_vgg16()
    assert model['backbone'] == {
        'type': 'ADAODVGG16',
        'frozen_stages': 2,
        'pretrained_checkpoint': VGG16_CAFFE_CHECKPOINT,
        'pretrained_sha256': VGG16_CAFFE_SHA256,
        'init_cfg': None,
    }


def test_pt_caffe_preprocessing_and_faster_rcnn_mechanics():
    model = build_faster_rcnn_vgg16()

    assert model['data_preprocessor'] == {
        'type': 'DetDataPreprocessor',
        'mean': [103.53, 116.28, 123.675],
        'std': [1.0, 1.0, 1.0],
        'bgr_to_rgb': False,
        'pad_size_divisor': 1,
    }
    assert model['rpn_head']['anchor_generator']['scales'] == [8, 16, 32]
    assert model['rpn_head']['anchor_generator']['strides'] == [16]
    assert model['rpn_head']['anchor_generator']['center_offset'] == 0.0
    rpn_assigner = model['train_cfg']['rpn']['assigner']
    assert rpn_assigner['min_pos_iou'] == 0.0
    assert rpn_assigner['match_low_quality'] is True
    assert model['train_cfg']['rpn']['sampler']['pos_fraction'] == 0.25
    assert model['train_cfg']['rpn_proposal']['nms_pre'] == 12000
    assert model['train_cfg']['rpn_proposal']['max_per_img'] == 2000
    assert model['test_cfg']['rpn']['nms_pre'] == 6000
    assert model['test_cfg']['rpn']['max_per_img'] == 1000

    roi_layer = model['roi_head']['bbox_roi_extractor']['roi_layer']
    assert roi_layer == {
        'type': 'RoIAlign',
        'output_size': 7,
        'sampling_ratio': 0,
        'aligned': True,
    }
    bbox_head = model['roi_head']['bbox_head']
    assert bbox_head['fc_out_channels'] == 1024
    assert bbox_head['reg_class_agnostic'] is False


def test_pt_linear_warmup_and_multistep_schedule():
    schedule_path = (
        Path(__file__).resolve().parents[4]
        / 'configs' / '_base_' / 'schedules' / 'iter_40k.py'
    )
    schedule = runpy.run_path(str(schedule_path))

    assert schedule['optim_wrapper']['optimizer'] == {
        'type': 'SGD',
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001,
    }
    assert schedule['param_scheduler'] == [
        {
            'type': 'LinearLR',
            'start_factor': 0.001,
            'begin': 0,
            'end': 400,
            'by_epoch': False,
        },
        {
            'type': 'MultiStepLR',
            'begin': 0,
            'end': 40000,
            'by_epoch': False,
            'milestones': [30000, 35000],
            'gamma': 0.1,
        },
    ]
