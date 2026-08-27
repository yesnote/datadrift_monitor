'''AADA Faster R-CNN with supervised target reveal and domain alignment.'''

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor
from mmdet.models.detectors import BaseDetector, FasterRCNN
from mmdet.registry import MODELS

from methods.common.mmdet.losses import prefix_losses
from methods.common.mmdet.models.progressive_domain_adaptation import (
    compute_multi_target_domain_loss,
    gradient_reverse,
)


SOURCE_BRANCH = 'source'
TARGET_LABELED_BRANCH = 'target_labeled'
TARGET_UNLABELED_BRANCH = 'target_unlabeled'
_ALLOWED_BRANCHES = {
    SOURCE_BRANCH,
    TARGET_LABELED_BRANCH,
    TARGET_UNLABELED_BRANCH,
}


def _validate_loss_branches(
    inputs: Mapping[str, Tensor],
    data_samples: Mapping[str, Sequence],
) -> None:
    if set(inputs) != set(data_samples):
        raise ValueError('AADA input and data-sample branches must match')
    unknown = set(inputs).difference(_ALLOWED_BRANCHES)
    if unknown:
        raise ValueError(
            'unknown AADA branches: {}'.format(', '.join(sorted(unknown)))
        )
    required = {SOURCE_BRANCH, TARGET_UNLABELED_BRANCH}
    missing = required.difference(inputs)
    if missing:
        raise ValueError(
            'missing AADA branches: {}'.format(', '.join(sorted(missing)))
        )
    for branch in inputs:
        if inputs[branch].shape[0] != len(data_samples[branch]):
            raise ValueError(
                '{} input and data-sample batch sizes differ'.format(branch)
            )


class AadaDetector(BaseDetector):
    '''AADA detector under the ADA-FNP C-to-F comparison protocol.'''

    def __init__(
        self,
        detector: Mapping,
        domain_discriminator: Optional[Mapping] = None,
        grl_scale: float = 1.0,
        domain_loss_weight: float = 0.01,
        data_preprocessor: Optional[Mapping] = None,
        init_cfg=None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.detector = MODELS.build(detector)
        if not isinstance(self.detector, FasterRCNN):
            raise TypeError('AADA supports FasterRCNN only')
        if domain_discriminator is None:
            domain_discriminator = dict(
                type='ProgressiveDomainDiscriminator',
                in_channels=512,
                hidden_channels=64,
            )
        self.domain_discriminator = MODELS.build(domain_discriminator)
        if grl_scale < 0:
            raise ValueError('grl_scale must be non-negative')
        if domain_loss_weight < 0:
            raise ValueError('domain_loss_weight must be non-negative')
        self.grl_scale = float(grl_scale)
        self.domain_loss_weight = float(domain_loss_weight)

    def extract_feat(self, batch_inputs: Tensor):
        return self.detector.extract_feat(batch_inputs)

    def extract_domain_feature(self, batch_inputs: Tensor) -> Tensor:
        features = self.extract_feat(batch_inputs)
        if not isinstance(features, tuple) or len(features) != 1:
            raise ValueError('AADA VGG16 must expose one conv5_3 feature')
        return features[0]

    def domain_logits(
        self,
        batch_inputs: Tensor,
        reverse_gradient: bool = True,
    ) -> Tensor:
        features = self.extract_domain_feature(batch_inputs)
        if reverse_gradient:
            features = gradient_reverse(features, self.grl_scale)
        return self.domain_discriminator(features)

    def loss(
        self,
        multi_batch_inputs: Mapping[str, Tensor],
        multi_batch_data_samples: Mapping[str, Sequence],
    ) -> dict:
        _validate_loss_branches(
            multi_batch_inputs,
            multi_batch_data_samples,
        )
        losses = prefix_losses(
            self.detector.loss(
                multi_batch_inputs[SOURCE_BRANCH],
                multi_batch_data_samples[SOURCE_BRANCH],
            ),
            SOURCE_BRANCH,
        )
        if TARGET_LABELED_BRANCH in multi_batch_inputs:
            losses.update(prefix_losses(
                self.detector.loss(
                    multi_batch_inputs[TARGET_LABELED_BRANCH],
                    multi_batch_data_samples[TARGET_LABELED_BRANCH],
                ),
                TARGET_LABELED_BRANCH,
            ))
        source_logits = self.domain_logits(
            multi_batch_inputs[SOURCE_BRANCH]
        )
        target_logits = [
            self.domain_logits(multi_batch_inputs[branch])
            for branch in (TARGET_LABELED_BRANCH, TARGET_UNLABELED_BRANCH)
            if branch in multi_batch_inputs
        ]
        losses['domain.loss_adv'] = self.domain_loss_weight * (
            compute_multi_target_domain_loss(source_logits, target_logits)
        )
        return losses

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples,
        rescale: bool = True,
    ):
        return self.detector.predict(
            batch_inputs,
            batch_data_samples,
            rescale=rescale,
        )

    def _forward(self, batch_inputs: Tensor, batch_data_samples=None):
        return self.detector(
            batch_inputs,
            batch_data_samples,
            mode='tensor',
        )

    @torch.no_grad()
    def predict_with_class_probabilities(
        self,
        batch_inputs: Tensor,
        batch_data_samples,
    ):
        features = self.extract_feat(batch_inputs)
        proposals = self.detector.rpn_head.predict(
            features,
            batch_data_samples,
            rescale=False,
        )
        roi_head = self.detector.roi_head
        if not hasattr(roi_head, 'predict_with_class_probabilities'):
            raise TypeError(
                'AADA requires ClassProbabilityRoIHead for acquisition'
            )
        return roi_head.predict_with_class_probabilities(
            features,
            proposals,
            batch_data_samples,
        )
