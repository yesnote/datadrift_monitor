'''Faster R-CNN student/teacher ownership and ADA-FNP loss routing.'''

import copy
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from mmengine.structures import InstanceData
from mmdet.models.detectors import FasterRCNN, SemiBaseDetector
from mmdet.registry import MODELS

from methods.ada_fnp.acquisition.mc_dropout import (
    monte_carlo_dropout_enabled,
)
from methods.ada_fnp.training.pseudo_labeling import (
    select_classification_losses,
    select_pseudo_labels,
)
from methods.common.mmdet.losses import prefix_losses

from methods.common.mmdet.models.progressive_domain_adaptation import (
    compute_multi_target_domain_loss,
    gradient_reverse,
)
from .mc_dropout_roi_head import run_rpn_once_for_fixed_roi


SOURCE_BRANCH = 'source'
TARGET_LABELED_BRANCH = 'target_labeled'
TARGET_UNLABELED_WEAK_BRANCH = 'target_unlabeled_weak'
TARGET_UNLABELED_STRONG_BRANCH = 'target_unlabeled_strong'
_PSEUDO_METRIC_KEYS = (
    'target_unlabeled_strong.pseudo_candidates',
    'target_unlabeled_strong.pseudo_variance_kept',
    'target_unlabeled_strong.pseudo_confidence_kept',
    'target_unlabeled_strong.pseudo_kept',
)
_ALLOWED_BRANCHES = {
    SOURCE_BRANCH,
    TARGET_LABELED_BRANCH,
    TARGET_UNLABELED_WEAK_BRANCH,
    TARGET_UNLABELED_STRONG_BRANCH,
}


def validate_loss_branches(
    inputs: Mapping[str, Tensor], data_samples: Mapping[str, Sequence]
) -> None:
    '''Validate the explicit ADA-FNP source/target branch contract.'''
    input_keys = set(inputs)
    sample_keys = set(data_samples)
    if input_keys != sample_keys:
        raise ValueError(
            'input and data-sample branches must match: {} != {}'.format(
                sorted(input_keys), sorted(sample_keys)
            )
        )
    unknown = input_keys.difference(_ALLOWED_BRANCHES)
    if unknown:
        raise ValueError(
            'unknown ADA-FNP branches: {}'.format(', '.join(sorted(unknown)))
        )
    required = {
        SOURCE_BRANCH,
        TARGET_UNLABELED_WEAK_BRANCH,
        TARGET_UNLABELED_STRONG_BRANCH,
    }
    missing = required.difference(input_keys)
    if missing:
        raise ValueError(
            'missing ADA-FNP branches: {}'.format(', '.join(sorted(missing)))
        )
    for branch in input_keys:
        if inputs[branch].shape[0] != len(data_samples[branch]):
            raise ValueError(
                f'{branch} input and data-sample batch sizes differ'
            )
    if (
        inputs[TARGET_UNLABELED_WEAK_BRANCH].shape[0]
        != inputs[TARGET_UNLABELED_STRONG_BRANCH].shape[0]
    ):
        raise ValueError('target weak and strong batch sizes must match')
    if (
        inputs[TARGET_UNLABELED_WEAK_BRANCH].shape[-2:]
        != inputs[TARGET_UNLABELED_STRONG_BRANCH].shape[-2:]
    ):
        raise ValueError('target weak and strong spatial shapes must match')


def validate_shared_target_geometry(
    weak_samples: Sequence, strong_samples: Sequence
) -> None:
    '''Require PT weak/strong views to share one geometric transform.'''
    if len(weak_samples) != len(strong_samples):
        raise ValueError('target weak and strong sample counts must match')
    required_meta = ('img_id', 'ori_shape', 'img_shape', 'homography_matrix')
    for index, (weak_sample, strong_sample) in enumerate(
        zip(weak_samples, strong_samples)
    ):
        weak_meta = weak_sample.metainfo
        strong_meta = strong_sample.metainfo
        missing = [
            key
            for key in required_meta
            if key not in weak_meta or key not in strong_meta
        ]
        if missing:
            raise ValueError(
                'paired target sample {} is missing geometry metadata: {}'.format(
                    index, ', '.join(missing)
                )
            )
        if weak_meta['img_id'] != strong_meta['img_id']:
            raise ValueError('target weak and strong image ids must match')
        for key in ('ori_shape', 'img_shape', 'flip', 'flip_direction'):
            if weak_meta.get(key) != strong_meta.get(key):
                raise ValueError(
                    f'target weak and strong {key} metadata must match'
                )
        weak_homography = torch.as_tensor(weak_meta['homography_matrix'])
        strong_homography = torch.as_tensor(
            strong_meta['homography_matrix'],
            device=weak_homography.device,
            dtype=weak_homography.dtype,
        )
        if (
            weak_homography.shape != (3, 3)
            or strong_homography.shape != (3, 3)
        ):
            raise ValueError('target view homographies must have shape [3, 3]')
        if not torch.equal(weak_homography, strong_homography):
            raise ValueError(
                'target weak and strong homography matrices must match'
            )


def select_domain_target_branches(branches: Mapping) -> Sequence[str]:
    '''Choose one unlabeled DA view, avoiding duplicate weak/strong weighting.

    The strong view is used because it is the student optimization view. The
    weak view is reserved for teacher pseudo-label generation.
    '''
    return tuple(
        branch
        for branch in (
            TARGET_LABELED_BRANCH,
            TARGET_UNLABELED_STRONG_BRANCH,
        )
        if branch in branches
    )


def route_detection_losses(
    source_losses: Mapping[str, object],
    target_labeled_losses: Optional[Mapping[str, object]] = None,
    target_unlabeled_strong_losses: Optional[Mapping[str, object]] = None,
    enable_unsupervised_loss: bool = True,
) -> Dict[str, object]:
    '''Prefix supervised losses and retain only classification pseudo losses.'''
    routed = prefix_losses(source_losses, SOURCE_BRANCH, weight=1)
    if target_labeled_losses is not None:
        routed.update(
            prefix_losses(
                target_labeled_losses, TARGET_LABELED_BRANCH, weight=1
            )
        )
    if enable_unsupervised_loss:
        if target_unlabeled_strong_losses is None:
            raise ValueError('target-unlabeled strong losses are required')
        classification_losses = select_classification_losses(
            target_unlabeled_strong_losses
        )
        if not classification_losses:
            raise ValueError(
                'target-unlabeled strong loss output has no RPN/RoI '
                'classification terms'
            )
        routed.update(
            prefix_losses(
                classification_losses,
                TARGET_UNLABELED_STRONG_BRANCH,
                weight=1,
            )
        )
    return routed


class AdaFnpDetectorBranch(nn.Module):
    '''One EMA branch owning a Faster R-CNN and domain discriminator.'''

    def __init__(
        self,
        detector: Mapping,
        domain_discriminator: Mapping,
        grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.detector = MODELS.build(detector)
        if not isinstance(self.detector, FasterRCNN):
            raise TypeError('ADA-FNP supports FasterRCNN only')
        self.domain_discriminator = MODELS.build(domain_discriminator)
        if grl_scale < 0:
            raise ValueError('grl_scale must be non-negative')
        self.grl_scale = float(grl_scale)

    def forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples=None,
        mode: str = 'tensor',
    ):
        return self.detector(batch_inputs, batch_data_samples, mode=mode)

    def loss(self, batch_inputs: Tensor, batch_data_samples) -> dict:
        return self.detector.loss(batch_inputs, batch_data_samples)

    def predict(
        self, batch_inputs: Tensor, batch_data_samples, rescale: bool = True
    ):
        return self.detector.predict(
            batch_inputs, batch_data_samples, rescale=rescale
        )

    def extract_feat(self, batch_inputs: Tensor):
        return self.detector.extract_feat(batch_inputs)

    def extract_domain_feature(self, batch_inputs: Tensor) -> Tensor:
        features = self.extract_feat(batch_inputs)
        if not isinstance(features, tuple) or len(features) != 1:
            raise ValueError(
                'ADA-FNP VGG16 must expose exactly one conv5_3 feature'
            )
        return features[0]

    def domain_logits(
        self, batch_inputs: Tensor, reverse_gradient: bool = True
    ) -> Tensor:
        features = self.extract_domain_feature(batch_inputs)
        if reverse_gradient:
            features = gradient_reverse(features, self.grl_scale)
        return self.domain_discriminator(features)

    def predict_fixed_proposals(
        self, batch_inputs: Tensor, batch_data_samples, passes: int = 10
    ):
        '''Run RPN once, then reuse its proposals for all RoI passes.'''
        features = self.extract_feat(batch_inputs)
        roi_head = self.detector.roi_head
        if not hasattr(roi_head, 'predict_fixed_proposals'):
            raise TypeError(
                'FasterRCNN roi_head must be '
                'AdaFnpMonteCarloDropoutRoIHead'
            )
        return run_rpn_once_for_fixed_roi(
            lambda: self.detector.rpn_head.predict(
                features, batch_data_samples, rescale=False
            ),
            lambda proposals: roi_head.predict_fixed_proposals(
                features, proposals, batch_data_samples, passes=passes
            ),
        )


class AdaFnpDetector(SemiBaseDetector):
    '''ADA-FNP student/teacher detector with explicit loss branches.'''

    def __init__(
        self,
        detector: Mapping,
        domain_discriminator: Optional[Mapping] = None,
        grl_scale: float = 1.0,
        domain_loss_weight: float = 0.01,
        enable_unsupervised_loss: bool = True,
        mc_passes: int = 10,
        localization_variance_threshold: float = 0.1,
        confidence_threshold: float = 0.5,
        semi_train_cfg: Optional[Mapping] = None,
        semi_test_cfg: Optional[Mapping] = None,
        data_preprocessor: Optional[Mapping] = None,
        init_cfg=None,
    ) -> None:
        if domain_loss_weight < 0:
            raise ValueError('domain_loss_weight must be non-negative')
        if mc_passes < 2:
            raise ValueError('mc_passes must be at least two')
        if localization_variance_threshold < 0:
            raise ValueError(
                'localization_variance_threshold must be non-negative'
            )
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError('confidence_threshold must be in [0, 1]')
        if domain_discriminator is None:
            domain_discriminator = dict(
                type='ProgressiveDomainDiscriminator',
                in_channels=512,
                hidden_channels=64,
            )
        branch = dict(
            type='AdaFnpDetectorBranch',
            detector=detector,
            domain_discriminator=domain_discriminator,
            grl_scale=grl_scale,
        )
        resolved_train_cfg = dict(freeze_teacher=True)
        resolved_train_cfg.update(dict(semi_train_cfg or {}))
        resolved_test_cfg = dict(
            predict_on='teacher',
            forward_on='teacher',
            extract_feat_on='teacher',
        )
        resolved_test_cfg.update(dict(semi_test_cfg or {}))
        super().__init__(
            detector=branch,
            semi_train_cfg=resolved_train_cfg,
            semi_test_cfg=resolved_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.domain_loss_weight = float(domain_loss_weight)
        self.enable_unsupervised_loss = bool(enable_unsupervised_loss)
        self.mc_passes = int(mc_passes)
        self.localization_variance_threshold = float(
            localization_variance_threshold
        )
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def _clone_without_unlabeled_annotations(data_samples):
        '''Clone samples and remove every field that can expose GT.'''
        cloned_samples = copy.deepcopy(data_samples)
        for data_sample in cloned_samples:
            for field in (
                'gt_instances',
                'ignored_instances',
                'gt_sem_seg',
                'gt_panoptic_seg',
            ):
                if field in data_sample:
                    delattr(data_sample, field)
        return cloned_samples

    def _generate_strong_pseudo_samples(
        self, weak_inputs: Tensor, weak_data_samples, strong_data_samples
    ):
        clean_weak_samples = self._clone_without_unlabeled_annotations(
            weak_data_samples
        )
        clean_strong_samples = self._clone_without_unlabeled_annotations(
            strong_data_samples
        )
        validate_shared_target_geometry(
            clean_weak_samples, clean_strong_samples
        )
        teacher_results = self.predict_teacher_fixed_proposals(
            weak_inputs, clean_weak_samples, passes=self.mc_passes
        )
        if len(teacher_results) != len(strong_data_samples):
            raise ValueError(
                'teacher predictions and strong samples must align'
            )

        pseudo_metrics = {
            key: weak_inputs.new_zeros(()) for key in _PSEUDO_METRIC_KEYS
        }
        candidate_key, variance_key, confidence_key, kept_key = (
            _PSEUDO_METRIC_KEYS
        )
        for result, strong_sample in zip(
            teacher_results, clean_strong_samples
        ):
            pseudo = select_pseudo_labels(
                result.bboxes,
                result.labels,
                result.scores,
                result.box_variances,
                variance_threshold=self.localization_variance_threshold,
                confidence_threshold=self.confidence_threshold,
            )
            pseudo_metrics[candidate_key] = (
                pseudo_metrics[candidate_key]
                + result.scores.new_tensor(len(result.scores))
            )
            pseudo_metrics[variance_key] = (
                pseudo_metrics[variance_key]
                + pseudo['variance_keep'].sum().to(weak_inputs.dtype)
            )
            pseudo_metrics[confidence_key] = (
                pseudo_metrics[confidence_key]
                + pseudo['confidence_keep'].sum().to(weak_inputs.dtype)
            )
            pseudo_metrics[kept_key] = (
                pseudo_metrics[kept_key]
                + pseudo['keep'].sum().to(weak_inputs.dtype)
            )
            gt_instances = InstanceData()
            gt_instances.bboxes = pseudo['boxes']
            gt_instances.labels = pseudo['labels']
            gt_instances.scores = pseudo['scores']
            strong_sample.gt_instances = gt_instances
        return clean_strong_samples, pseudo_metrics

    def loss(
        self,
        multi_batch_inputs: Mapping[str, Tensor],
        multi_batch_data_samples: Mapping[str, Sequence],
    ) -> dict:
        validate_loss_branches(
            multi_batch_inputs, multi_batch_data_samples
        )
        source_losses = self.student.loss(
            multi_batch_inputs[SOURCE_BRANCH],
            multi_batch_data_samples[SOURCE_BRANCH],
        )

        target_labeled_losses = None
        if TARGET_LABELED_BRANCH in multi_batch_inputs:
            target_labeled_losses = self.student.loss(
                multi_batch_inputs[TARGET_LABELED_BRANCH],
                multi_batch_data_samples[TARGET_LABELED_BRANCH],
            )

        target_unlabeled_strong_losses = None
        pseudo_metrics = {}
        if self.enable_unsupervised_loss:
            pseudo_strong_samples, pseudo_metrics = (
                self._generate_strong_pseudo_samples(
                    multi_batch_inputs[TARGET_UNLABELED_WEAK_BRANCH],
                    multi_batch_data_samples[TARGET_UNLABELED_WEAK_BRANCH],
                    multi_batch_data_samples[
                        TARGET_UNLABELED_STRONG_BRANCH
                    ],
                )
            )
            target_unlabeled_strong_losses = self.student.loss(
                multi_batch_inputs[TARGET_UNLABELED_STRONG_BRANCH],
                pseudo_strong_samples,
            )

        losses = route_detection_losses(
            source_losses,
            target_labeled_losses,
            target_unlabeled_strong_losses,
            enable_unsupervised_loss=self.enable_unsupervised_loss,
        )
        losses.update(pseudo_metrics)

        source_logits = self.student.domain_logits(
            multi_batch_inputs[SOURCE_BRANCH]
        )
        target_logits = [
            self.student.domain_logits(multi_batch_inputs[branch])
            for branch in select_domain_target_branches(multi_batch_inputs)
        ]
        losses['domain.loss_adv'] = self.domain_loss_weight * (
            compute_multi_target_domain_loss(source_logits, target_logits)
        )
        return losses

    @torch.no_grad()
    def predict_teacher_fixed_proposals(
        self, batch_inputs: Tensor, batch_data_samples, passes: int = 10
    ):
        bbox_head = self.teacher.detector.roi_head.bbox_head
        dropout_modules = getattr(bbox_head, 'shared_dropouts', ())
        with monte_carlo_dropout_enabled(self.teacher, dropout_modules):
            return self.teacher.predict_fixed_proposals(
                batch_inputs, batch_data_samples, passes=passes
            )
