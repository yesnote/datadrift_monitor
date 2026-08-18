'''Integration coverage for the MMDetection 3.3 ADA-FNP registrations.'''

import importlib.util
from unittest.mock import patch

import pytest
import torch


_MISSING_DEPENDENCIES = [
    package for package in ('mmcv', 'mmengine')
    if importlib.util.find_spec(package) is None
]
if _MISSING_DEPENDENCIES:
    pytest.skip(
        'MMDetection integration dependencies are missing: {}'.format(
            ', '.join(_MISSING_DEPENDENCIES)),
        allow_module_level=True)

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

import methods.ada_fnp.registration  # noqa: F401, E402
from methods.ada_fnp.models.detector import (  # noqa: E402
    SOURCE_BRANCH, TARGET_UNLABELED_STRONG_BRANCH,
    TARGET_UNLABELED_WEAK_BRANCH)


def _faster_rcnn_config():
    train_cfg = ConfigDict(
        rpn=ConfigDict(
            assigner=ConfigDict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=ConfigDict(
                type='RandomSampler',
                num=16,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=ConfigDict(
            nms_pre=20,
            max_per_img=5,
            nms=ConfigDict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=ConfigDict(
            assigner=ConfigDict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=ConfigDict(
                type='RandomSampler',
                num=8,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
    test_cfg = ConfigDict(
        rpn=ConfigDict(
            nms_pre=20,
            max_per_img=5,
            nms=ConfigDict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=ConfigDict(
            score_thr=0.0,
            nms=ConfigDict(type='nms', iou_threshold=0.5),
            max_per_img=5))
    return dict(
        type='FasterRCNN',
        backbone=dict(type='ADAODVGG16'),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=16,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2],
                ratios=[1.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='ADAFNPRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=2, sampling_ratio=0),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='ADAODVGGShared2FCBBoxHead',
                in_channels=512,
                roi_feat_size=2,
                fc_out_channels=16,
                num_classes=2,
                reg_class_agnostic=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        train_cfg=train_cfg,
        test_cfg=test_cfg)


def _unlabeled_sample(homography, label=99):
    data_sample = DetDataSample(
        metainfo=dict(
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0),
            homography_matrix=homography))
    data_sample.gt_instances = InstanceData(
        bboxes=torch.tensor([[30.0, 30.0, 40.0, 40.0]]),
        labels=torch.tensor([label]))
    data_sample.ignored_instances = InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([label]))
    return data_sample


def test_registry_builds_teacher_student_and_reuses_rpn_proposals():
    assert MODELS.get('ADAFNPDetector') is not None
    assert MODELS.get('ADAFNPBranch') is not None
    assert MODELS.get('ADAFNPRoIHead') is not None
    assert MODELS.get('ADAFNPDomainDiscriminator') is not None

    model = MODELS.build(dict(
        type='ADAFNPDetector',
        detector=_faster_rcnn_config(),
        enable_unsupervised_loss=False))
    assert model.student.detector is not model.teacher.detector
    assert model.student.domain_discriminator is not (
        model.teacher.domain_discriminator)
    assert not any(parameter.requires_grad
                   for parameter in model.teacher.parameters())

    batch_inputs = torch.randn(1, 3, 64, 64)
    data_sample = DetDataSample(
        metainfo=dict(
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0)))
    rpn_predict = model.teacher.detector.rpn_head.predict
    bbox_head = model.teacher.detector.roi_head.bbox_head
    dropout_states = []
    bbox_forward = bbox_head.forward

    def record_dropout_state(*args, **kwargs):
        dropout_states.append([
            module.training for module in bbox_head.shared_dropouts
        ])
        return bbox_forward(*args, **kwargs)

    with patch.object(
            model.teacher.detector.rpn_head,
            'predict',
            wraps=rpn_predict) as wrapped_predict, patch.object(
                bbox_head, 'forward', side_effect=record_dropout_state):
            results = model.predict_teacher_fixed_proposals(
                batch_inputs, [data_sample], passes=2)

    assert wrapped_predict.call_count == 1
    assert dropout_states == [[True, True], [True, True]]
    assert not any(module.training for module in bbox_head.shared_dropouts)
    assert len(results) == 1
    assert results[0].proposal_indices.dtype == torch.long
    assert results[0].class_probabilities.shape[-1] == 3
    assert results[0].box_variances.shape[-1] == 4


def test_loss_orchestration_uses_weak_teacher_and_strong_student():
    model = MODELS.build(dict(
        type='ADAFNPDetector',
        detector=_faster_rcnn_config(),
        enable_unsupervised_loss=False))
    weak_homography = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    strong_homography = torch.tensor([
        [1.0, 0.0, 3.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ])
    weak_sample = _unlabeled_sample(weak_homography)
    strong_sample = _unlabeled_sample(strong_homography)
    branch_inputs = {
        SOURCE_BRANCH: torch.randn(1, 3, 64, 64),
        TARGET_UNLABELED_WEAK_BRANCH: torch.randn(1, 3, 64, 64),
        TARGET_UNLABELED_STRONG_BRANCH: torch.randn(1, 3, 64, 64),
    }
    branch_samples = {
        SOURCE_BRANCH: [DetDataSample()],
        TARGET_UNLABELED_WEAK_BRANCH: [weak_sample],
        TARGET_UNLABELED_STRONG_BRANCH: [strong_sample],
    }
    source_losses = {
        'loss_rpn_cls': torch.tensor(1.0),
        'loss_bbox': torch.tensor(2.0),
    }

    with patch.object(
            model.student, 'loss', return_value=source_losses
    ) as student_loss, patch.object(
            model.student,
            'domain_logits',
            side_effect=[torch.zeros(1), torch.zeros(1)]
    ) as domain_logits, patch.object(
            model, 'predict_teacher_fixed_proposals'
    ) as teacher_predict:
        losses = model.loss(branch_inputs, branch_samples)

    teacher_predict.assert_not_called()
    assert student_loss.call_count == 1
    assert domain_logits.call_args_list[0].args[0] is (
        branch_inputs[SOURCE_BRANCH])
    assert domain_logits.call_args_list[1].args[0] is (
        branch_inputs[TARGET_UNLABELED_STRONG_BRANCH])
    assert 'target_unlabeled_strong.loss_rpn_cls' not in losses

    teacher_result = InstanceData(
        bboxes=torch.tensor([
            [2.0, 4.0, 6.0, 8.0],
            [20.0, 40.0, 60.0, 80.0],
        ]),
        class_probabilities=torch.tensor([
            [0.001, 0.002, 0.997],
            [0.8, 0.1, 0.1],
        ]),
        box_variances=torch.tensor([
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
        ]))
    strong_losses = {
        'loss_rpn_cls': torch.tensor(3.0),
        'loss_rpn_bbox': torch.tensor(4.0),
        'loss_cls': torch.tensor(5.0),
        'loss_bbox': torch.tensor(6.0),
    }
    model.enable_unsupervised_loss = True
    with patch.object(
            model.student,
            'loss',
            side_effect=[source_losses, strong_losses]
    ) as student_loss, patch.object(
            model.student,
            'domain_logits',
            side_effect=[torch.zeros(1), torch.zeros(1)]
    ), patch.object(
            model,
            'predict_teacher_fixed_proposals',
            return_value=[teacher_result]
    ) as teacher_predict:
        losses = model.loss(branch_inputs, branch_samples)

    teacher_predict.assert_called_once()
    assert teacher_predict.call_args.args[0] is (
        branch_inputs[TARGET_UNLABELED_WEAK_BRANCH])
    clean_weak_sample = teacher_predict.call_args.args[1][0]
    assert 'gt_instances' not in clean_weak_sample
    assert 'ignored_instances' not in clean_weak_sample
    assert teacher_predict.call_args.kwargs['passes'] == 10

    strong_loss_inputs, pseudo_samples = student_loss.call_args_list[1].args
    assert strong_loss_inputs is branch_inputs[TARGET_UNLABELED_STRONG_BRANCH]
    assert pseudo_samples[0] is not strong_sample
    assert torch.allclose(
        pseudo_samples[0].gt_instances.bboxes,
        torch.tensor([[4.0, 6.0, 6.0, 8.0]]))
    assert pseudo_samples[0].gt_instances.labels.tolist() == [1]
    assert 'ignored_instances' not in pseudo_samples[0]
    assert strong_sample.gt_instances.labels.tolist() == [99]
    assert set(key for key in losses if key.startswith(
        'target_unlabeled_strong.')) == {
            'target_unlabeled_strong.loss_rpn_cls',
            'target_unlabeled_strong.loss_cls',
        }
