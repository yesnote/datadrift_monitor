'''Faster R-CNN RoI inference that preserves foreground probabilities.'''

from typing import List, Sequence, Tuple

import torch
from mmcv.ops import batched_nms
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.structures.bbox import bbox2roi, get_box_tensor


class ClassProbabilityRoIHead(StandardRoIHead):
    '''Standard RoI head with deterministic class-probability predictions.'''

    def _fixed_bbox_features(
        self,
        features: Tuple[Tensor, ...],
        rois: Tensor,
    ) -> Tensor:
        bbox_features = self.bbox_roi_extractor(
            features[:self.bbox_roi_extractor.num_inputs],
            rois,
        )
        if self.with_shared_head:
            bbox_features = self.shared_head(bbox_features)
        return bbox_features

    def _class_probabilities(self, class_logits: Tensor) -> Tensor:
        if self.bbox_head.custom_cls_channels:
            return self.bbox_head.loss_cls.get_activation(class_logits)
        return class_logits.softmax(dim=-1)

    def _empty_probability_result(self, reference: Tensor) -> InstanceData:
        result = InstanceData()
        result.bboxes = reference.new_zeros((0, 4))
        result.scores = reference.new_zeros((0,))
        result.labels = reference.new_zeros((0,), dtype=torch.long)
        result.proposal_indices = reference.new_zeros((0,), dtype=torch.long)
        result.class_probabilities = reference.new_zeros(
            (0, self.bbox_head.num_classes + 1)
        )
        return result

    def _postprocess_probability_predictions(
        self,
        image_rois: Tensor,
        class_probabilities: Tensor,
        bbox_deltas: Tensor,
        data_sample,
    ) -> InstanceData:
        if self.bbox_head.reg_class_agnostic:
            raise ValueError(
                'class-probability inference requires class-specific bbox '
                'regression'
            )
        proposal_count = len(image_rois)
        foreground_scores, labels = class_probabilities[:, :-1].max(dim=1)
        deltas = bbox_deltas.reshape(
            proposal_count,
            self.bbox_head.num_classes,
            4,
        )
        proposal_indices = torch.arange(
            proposal_count,
            device=image_rois.device,
        )
        selected_deltas = deltas[proposal_indices, labels]
        decoded = self.bbox_head.bbox_coder.decode(
            image_rois[:, 1:],
            selected_deltas,
            max_shape=data_sample.metainfo['img_shape'],
        )
        decoded_boxes = get_box_tensor(decoded).reshape(-1, 4)
        valid = foreground_scores > self.test_cfg.score_thr
        if not valid.any():
            return self._empty_probability_result(image_rois)
        valid_proposal_indices = proposal_indices[valid]
        detections, kept_indices = batched_nms(
            decoded_boxes[valid],
            foreground_scores[valid],
            labels[valid],
            self.test_cfg.nms,
        )
        if self.test_cfg.max_per_img > 0:
            detections = detections[:self.test_cfg.max_per_img]
            kept_indices = kept_indices[:self.test_cfg.max_per_img]
        kept_proposal_indices = valid_proposal_indices[kept_indices]
        result = InstanceData()
        result.bboxes = detections[:, :4]
        result.scores = detections[:, 4]
        result.labels = labels[kept_proposal_indices]
        result.proposal_indices = kept_proposal_indices
        result.class_probabilities = class_probabilities[
            kept_proposal_indices
        ]
        return result

    def predict_with_class_probabilities(
        self,
        features: Tuple[Tensor, ...],
        rpn_results_list: Sequence[InstanceData],
        batch_data_samples: Sequence,
    ) -> List[InstanceData]:
        '''Run one deterministic RoI pass and retain full class probabilities.'''

        if len(rpn_results_list) != len(batch_data_samples):
            raise ValueError('proposal and data-sample batch sizes differ')
        proposals = [result.bboxes for result in rpn_results_list]
        proposal_counts = [len(proposal) for proposal in proposals]
        rois = bbox2roi(proposals)
        if rois.numel() == 0:
            return [
                self._empty_probability_result(features[0])
                for _ in batch_data_samples
            ]
        bbox_features = self._fixed_bbox_features(features, rois)
        class_logits, bbox_deltas = self.bbox_head(bbox_features)
        expected_bbox_channels = self.bbox_head.num_classes * 4
        if bbox_deltas is None or bbox_deltas.shape[-1] != expected_bbox_channels:
            raise ValueError(
                'class-probability inference requires one bbox delta per '
                'proposal and foreground class'
            )
        probabilities = self._class_probabilities(class_logits)
        split_probabilities = probabilities.split(proposal_counts, dim=0)
        split_deltas = bbox_deltas.split(proposal_counts, dim=0)
        split_rois = rois.split(proposal_counts, dim=0)
        results = []
        for index, data_sample in enumerate(batch_data_samples):
            if not proposal_counts[index]:
                results.append(self._empty_probability_result(features[0]))
                continue
            results.append(self._postprocess_probability_predictions(
                split_rois[index],
                split_probabilities[index],
                split_deltas[index],
                data_sample,
            ))
        return results
