'''Independent Faster R-CNN VGG16 config construction.'''

from copy import deepcopy

from methods.common.mmdet.models.backbones.vgg16_caffe_checkpoint import (
    CHECKPOINT_PATH as VGG16_CAFFE_CHECKPOINT,
    SHA256 as VGG16_CAFFE_SHA256,
)

# PT loads only convolution tensors from this Caffe-converted VGG16 asset.
# Source: Zenodo record 4515252, file ``vgg16_caffe.pth``.
_DETECTOR_TEMPLATE = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type='VGG16Backbone',
        frozen_stages=2,
        pretrained_checkpoint=VGG16_CAFFE_CHECKPOINT,
        pretrained_sha256=VGG16_CAFFE_SHA256,
        init_cfg=None,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=512,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[16],
            # Detectron2 v0.5 defaults to anchors centered on integer pixels.
            center_offset=0.0,
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                aligned=True,
            ),
            out_channels=512,
            featmap_strides=[16],
        ),
        bbox_head=dict(
            type='VGGShared2FCBBoxHead',
            in_channels=512,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            dropout=0.0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                # Detectron2 always retains each GT's best anchor, independent
                # of the positive IoU threshold.
                min_pos_iou=0.0,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)


def build_faster_rcnn_vgg16() -> dict:
    '''Return a mutation-independent Faster R-CNN config dictionary.'''

    return deepcopy(_DETECTOR_TEMPLATE)
