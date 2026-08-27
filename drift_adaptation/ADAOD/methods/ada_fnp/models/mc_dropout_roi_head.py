'''Fixed-proposal Monte Carlo Dropout inference for ADA-FNP.'''

from typing import Callable, List, Sequence, Tuple, TypeVar

import torch
from mmcv.ops import batched_nms
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox2roi, get_box_tensor

from methods.common.mmdet.models.roi_heads.class_probability_roi_head import (
    ClassProbabilityRoIHead,
)

P = TypeVar('P')
R = TypeVar('R')


def run_rpn_once_for_fixed_roi(
    rpn_predict: Callable[[], P], roi_predict: Callable[[P], R]
) -> R:
    '''Materialize proposals once and reuse them for all RoI passes.'''
    proposals = rpn_predict()
    return roi_predict(proposals)


class AdaFnpMonteCarloDropoutRoIHead(ClassProbabilityRoIHead):
    '''Standard Faster R-CNN RoI head with fixed-proposal MC inference.'''

    def _empty_fixed_result(self, reference: Tensor) -> InstanceData:
        result = InstanceData()
        result.bboxes = reference.new_zeros((0, 4))
        result.scores = reference.new_zeros((0, ))
        result.labels = reference.new_zeros((0, ), dtype=torch.long)
        result.proposal_indices = reference.new_zeros(
            (0, ), dtype=torch.long
        )
        result.class_probabilities = reference.new_zeros(
            (0, self.bbox_head.num_classes + 1)
        )
        result.box_variances = reference.new_zeros((0, 4))
        return result

    def predict_fixed_proposals(
        self,
        features: Tuple[Tensor, ...],
        rpn_results_list: Sequence[InstanceData],
        batch_data_samples: Sequence,
        passes: int = 10,
    ) -> List[InstanceData]:
        '''Run the bbox head repeatedly on one immutable proposal set.

        Class-specific bbox-head deltas are tracked for every fixed
        ``(proposal, class)`` pair. The mean foreground probability chooses
        the pseudo class, its mean delta is decoded once, and its unbiased
        delta variance is retained through one class-aware NMS call.
        '''
        if passes < 2:
            raise ValueError('fixed-proposal MC inference needs 2+ passes')
        if self.bbox_head.reg_class_agnostic:
            raise ValueError(
                'ADA-FNP fixed-proposal inference requires '
                'class-specific bbox regression'
            )
        if len(rpn_results_list) != len(batch_data_samples):
            raise ValueError('proposal and data-sample batch sizes differ')

        proposals = [result.bboxes for result in rpn_results_list]
        num_proposals = [len(proposal) for proposal in proposals]
        rois = bbox2roi(proposals)
        if rois.numel() == 0:
            return [
                self._empty_fixed_result(features[0])
                for _ in batch_data_samples
            ]

        bbox_features = self._fixed_bbox_features(features, rois)
        probability_passes = []
        bbox_delta_passes = []
        expected_bbox_channels = self.bbox_head.num_classes * 4
        for _ in range(passes):
            class_logits, bbox_deltas = self.bbox_head(bbox_features)
            if (
                bbox_deltas is None
                or bbox_deltas.shape[-1] != expected_bbox_channels
            ):
                raise ValueError(
                    'fixed-proposal inference requires one bbox delta per '
                    'proposal and foreground class'
                )
            probability_passes.append(
                self._class_probabilities(class_logits)
            )
            bbox_delta_passes.append(bbox_deltas)

        probability_samples = torch.stack(probability_passes, dim=0)
        bbox_delta_samples = torch.stack(bbox_delta_passes, dim=0)
        split_probabilities = [
            sample.split(num_proposals, dim=0)
            for sample in probability_samples
        ]
        split_deltas = [
            sample.split(num_proposals, dim=0)
            for sample in bbox_delta_samples
        ]
        split_rois = rois.split(num_proposals, dim=0)

        results = []
        for image_index, data_sample in enumerate(batch_data_samples):
            image_rois = split_rois[image_index]
            if len(image_rois) == 0:
                results.append(self._empty_fixed_result(features[0]))
                continue
            image_probabilities = torch.stack([
                split_probabilities[pass_index][image_index]
                for pass_index in range(passes)
            ])
            image_deltas = torch.stack([
                split_deltas[pass_index][image_index]
                for pass_index in range(passes)
            ]).reshape(
                passes,
                len(image_rois),
                self.bbox_head.num_classes,
                4,
            )
            mean_probabilities = image_probabilities.mean(dim=0)
            foreground_scores, labels = mean_probabilities[:, :-1].max(
                dim=1
            )
            mean_deltas = image_deltas.mean(dim=0)
            delta_variances = image_deltas.var(dim=0, unbiased=True)
            proposal_indices = torch.arange(
                len(image_rois), device=image_rois.device
            )
            selected_deltas = mean_deltas[proposal_indices, labels]
            selected_variances = delta_variances[proposal_indices, labels]
            decoded = self.bbox_head.bbox_coder.decode(
                image_rois[:, 1:],
                selected_deltas,
                max_shape=data_sample.metainfo['img_shape'],
            )
            decoded_boxes = get_box_tensor(decoded).reshape(-1, 4)

            test_cfg = self.test_cfg
            valid = foreground_scores > test_cfg.score_thr
            if not valid.any():
                results.append(self._empty_fixed_result(features[0]))
                continue
            valid_proposal_indices = proposal_indices[valid]
            detections, kept_indices = batched_nms(
                decoded_boxes[valid],
                foreground_scores[valid],
                labels[valid],
                test_cfg.nms,
            )
            if test_cfg.max_per_img > 0:
                detections = detections[:test_cfg.max_per_img]
                kept_indices = kept_indices[:test_cfg.max_per_img]
            kept_proposal_indices = valid_proposal_indices[kept_indices]
            result = InstanceData()
            result.bboxes = detections[:, :4]
            result.scores = detections[:, 4]
            result.labels = labels[kept_proposal_indices]
            result.proposal_indices = kept_proposal_indices
            result.class_probabilities = mean_probabilities[
                kept_proposal_indices
            ]
            result.box_variances = selected_variances[
                kept_proposal_indices
            ]
            results.append(result)
        return results
