'''Fixed-proposal RoI prediction for ADA-FNP Faster R-CNN.'''

from typing import Callable, List, Sequence, Tuple, TypeVar

import torch
from torch import Tensor, nn

from methods.ada_fnp.acquisition.geometry import normalize_xyxy_box_samples

P = TypeVar('P')
R = TypeVar('R')


def summarize_mc_predictions(
        probability_samples: Tensor,
        box_samples: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    '''Average MC predictions before NMS and compute unbiased box variance.

    Args:
        probability_samples: Tensor shaped ``[M, N, C + 1]``.
        box_samples: Decoded boxes shaped ``[M, N, 4]`` or
            ``[M, N, C, 4]``.
    '''
    if probability_samples.ndim != 3:
        raise ValueError('probability_samples must have shape [M, N, C + 1]')
    if box_samples.ndim not in (3, 4) or box_samples.shape[-1] != 4:
        raise ValueError(
            'box_samples must have shape [M, N, 4] or [M, N, C, 4]')
    if probability_samples.shape[:2] != box_samples.shape[:2]:
        raise ValueError('probability and box sample axes must match')
    if (box_samples.ndim == 4 and
            box_samples.shape[2] != probability_samples.shape[2] - 1):
        raise ValueError(
            'class-specific box count must match foreground probabilities')
    if probability_samples.shape[0] < 2:
        raise ValueError('MC prediction requires at least two passes')
    mean_probabilities = probability_samples.mean(dim=0)
    mean_boxes = box_samples.mean(dim=0)
    box_variances = box_samples.var(dim=0, unbiased=True)
    return mean_probabilities, mean_boxes, box_variances


def map_multiclass_nms_to_proposals(
        kept_flat_indices: Tensor,
        num_foreground_classes: int,
        proposal_indices: Tensor = None) -> Tensor:
    '''Map MMDet multiclass-NMS flattened indices back to proposal indices.'''
    if kept_flat_indices.ndim != 1:
        raise ValueError('kept_flat_indices must be one-dimensional')
    if kept_flat_indices.dtype != torch.long:
        raise TypeError('kept_flat_indices must use torch.long dtype')
    if num_foreground_classes <= 0:
        raise ValueError('num_foreground_classes must be positive')
    local_indices, _ = split_multiclass_nms_indices(
        kept_flat_indices, num_foreground_classes)
    if proposal_indices is None:
        return local_indices
    if proposal_indices.ndim != 1:
        raise ValueError('proposal_indices must be one-dimensional')
    if local_indices.numel() and local_indices.max() >= len(proposal_indices):
        raise IndexError('NMS index exceeds the proposal index mapping')
    return proposal_indices[local_indices]


def split_multiclass_nms_indices(
        kept_flat_indices: Tensor,
        num_foreground_classes: int) -> Tuple[Tensor, Tensor]:
    '''Recover proposal and class indices from MMDet's flattened NMS indices.'''
    if kept_flat_indices.ndim != 1:
        raise ValueError('kept_flat_indices must be one-dimensional')
    if kept_flat_indices.dtype != torch.long:
        raise TypeError('kept_flat_indices must use torch.long dtype')
    if num_foreground_classes <= 0:
        raise ValueError('num_foreground_classes must be positive')
    proposal_indices = torch.div(
        kept_flat_indices, num_foreground_classes, rounding_mode='floor')
    class_indices = torch.remainder(
        kept_flat_indices, num_foreground_classes)
    return proposal_indices, class_indices


def run_rpn_once_for_fixed_roi(
        rpn_predict: Callable[[], P],
        roi_predict: Callable[[P], R]) -> R:
    '''Materialize proposals once and reuse them for all RoI passes.'''
    proposals = rpn_predict()
    return roi_predict(proposals)


try:
    from mmengine.structures import InstanceData
    from mmdet.models.layers import multiclass_nms
    from mmdet.models.roi_heads import StandardRoIHead
    from mmdet.structures.bbox import bbox2roi, get_box_tensor
except ModuleNotFoundError as exc:
    _MMDET_IMPORT_ERROR = exc

    class ADAFNPRoIHead(nn.Module):
        '''Placeholder that keeps the tensor helpers importable without MMDet.'''

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                'ADAFNPRoIHead requires MMDetection 3.3 with MMCV and '
                'MMEngine. The tensor helpers in this module remain usable '
                'without those dependencies.') from _MMDET_IMPORT_ERROR
else:
    _MMDET_IMPORT_ERROR = None

    class ADAFNPRoIHead(StandardRoIHead):
        '''Standard Faster R-CNN RoI head with fixed-proposal MC inference.'''

        def _fixed_bbox_features(self, features: Tuple[Tensor, ...],
                                 rois: Tensor) -> Tensor:
            bbox_features = self.bbox_roi_extractor(
                features[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_features = self.shared_head(bbox_features)
            return bbox_features

        def _class_probabilities(self, class_logits: Tensor) -> Tensor:
            if self.bbox_head.custom_cls_channels:
                return self.bbox_head.loss_cls.get_activation(class_logits)
            return class_logits.softmax(dim=-1)

        def _empty_fixed_result(self, reference: Tensor) -> InstanceData:
            result = InstanceData()
            result.bboxes = reference.new_zeros((0, 4))
            result.scores = reference.new_zeros((0, ))
            result.labels = reference.new_zeros((0, ), dtype=torch.long)
            result.proposal_indices = reference.new_zeros(
                (0, ), dtype=torch.long)
            result.class_probabilities = reference.new_zeros(
                (0, self.bbox_head.num_classes + 1))
            result.box_variances = reference.new_zeros((0, 4))
            return result

        def predict_fixed_proposals(
                self,
                features: Tuple[Tensor, ...],
                rpn_results_list: Sequence[InstanceData],
                batch_data_samples: Sequence,
                passes: int = 10) -> List[InstanceData]:
            '''Run the bbox head repeatedly on one immutable proposal set.

            Class-specific box trajectories are tracked for every fixed
            ``(proposal, class)`` pair. Probabilities and decoded boxes are
            averaged before a single multiclass NMS call.
            '''
            if passes < 2:
                raise ValueError('fixed-proposal MC inference needs 2+ passes')
            if self.bbox_head.reg_class_agnostic:
                raise ValueError(
                    'ADA-FNP fixed-proposal inference requires '
                    'class-specific bbox regression')
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
                if (bbox_deltas is None or
                        bbox_deltas.shape[-1] != expected_bbox_channels):
                    raise ValueError(
                        'fixed-proposal inference requires one bbox delta per '
                        'proposal and foreground class')
                probability_passes.append(
                    self._class_probabilities(class_logits))
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
                decoded_passes = []
                for pass_index in range(passes):
                    decoded = self.bbox_head.bbox_coder.decode(
                        image_rois[:, 1:],
                        split_deltas[pass_index][image_index],
                        max_shape=data_sample.metainfo['img_shape'])
                    decoded_tensor = get_box_tensor(decoded)
                    decoded_passes.append(decoded_tensor.reshape(
                        len(image_rois), self.bbox_head.num_classes, 4))
                image_boxes = torch.stack(decoded_passes)
                mean_probabilities, mean_boxes, _ = (
                    summarize_mc_predictions(
                        image_probabilities, image_boxes))
                normalized_boxes = normalize_xyxy_box_samples(
                    image_boxes, data_sample.metainfo['img_shape'])
                box_variances = normalized_boxes.var(dim=0, unbiased=True)

                test_cfg = self.test_cfg
                detections, labels, kept_indices = multiclass_nms(
                    mean_boxes.flatten(1),
                    mean_probabilities,
                    test_cfg.score_thr,
                    test_cfg.nms,
                    test_cfg.max_per_img,
                    return_inds=True)
                proposal_indices, class_indices = (
                    split_multiclass_nms_indices(
                        kept_indices, self.bbox_head.num_classes))
                if not torch.equal(labels, class_indices):
                    raise RuntimeError(
                        'multiclass NMS labels do not match flattened indices')
                result = InstanceData()
                result.bboxes = detections[:, :4]
                result.scores = detections[:, 4]
                result.labels = labels
                result.proposal_indices = proposal_indices
                result.class_probabilities = mean_probabilities[
                    proposal_indices]
                result.box_variances = box_variances[
                    proposal_indices, class_indices]
                results.append(result)
            return results


__all__ = [
    'ADAFNPRoIHead',
    'map_multiclass_nms_to_proposals',
    'normalize_xyxy_box_samples',
    'run_rpn_once_for_fixed_roi',
    'split_multiclass_nms_indices',
    'summarize_mc_predictions',
]
