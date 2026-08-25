'''Uncertainty-filtered pseudo labels and loss selection.'''

from __future__ import annotations

from typing import Dict, Mapping

import torch


def select_pseudo_labels(
    mean_boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    coordinate_variances: torch.Tensor,
    variance_threshold: float = 0.1,
    confidence_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    '''Keep confident detections whose mean localization variance is small.

    Labels, scores, boxes, and variances must come from the same foreground
    argmax class for each fixed proposal after MC averaging and NMS.
    '''

    if variance_threshold < 0:
        raise ValueError('variance_threshold must be non-negative')
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError('confidence_threshold must be in [0, 1]')
    if mean_boxes.ndim != 2 or mean_boxes.shape[-1] != 4:
        raise ValueError('mean boxes must have shape [N, 4]')
    if coordinate_variances.shape != mean_boxes.shape:
        raise ValueError('coordinate variances must match mean box shape')
    if labels.ndim != 1 or len(labels) != len(mean_boxes):
        raise ValueError('label count must match boxes')
    if labels.dtype != torch.long:
        raise TypeError('labels must use torch.long dtype')
    if scores.ndim != 1 or len(scores) != len(mean_boxes):
        raise ValueError('score count must match boxes')
    if not torch.isfinite(mean_boxes).all():
        raise ValueError('mean boxes must be finite')
    if not torch.isfinite(coordinate_variances).all():
        raise ValueError('coordinate variances must be finite')
    if (coordinate_variances < 0).any():
        raise ValueError('coordinate variances must be non-negative')
    if not torch.isfinite(scores).all():
        raise ValueError('scores must be finite')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('scores must be in [0, 1]')
    variance_keep = (
        coordinate_variances.mean(dim=1) <= variance_threshold
    )
    confidence_keep = scores >= confidence_threshold
    keep = variance_keep & confidence_keep
    return {
        'boxes': mean_boxes[keep],
        'labels': labels[keep],
        'scores': scores[keep],
        'variance_keep': variance_keep,
        'confidence_keep': confidence_keep,
        'keep': keep,
    }


def select_classification_losses(
    losses: Mapping[str, object],
) -> Dict[str, object]:
    '''Select the RPN and RoI classification terms from MMDet loss output.'''

    allowed = {'loss_rpn_cls', 'loss_cls'}
    return {key: value for key, value in losses.items() if key in allowed}
