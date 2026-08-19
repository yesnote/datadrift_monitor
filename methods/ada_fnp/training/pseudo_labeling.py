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
) -> Dict[str, torch.Tensor]:
    '''Keep detections whose mean localization variance is small.

    Labels and scores must come from the same multiclass-NMS candidate as each
    box. Recomputing an argmax from proposal probabilities can associate a
    class-specific box trajectory with the wrong class.
    '''

    if variance_threshold < 0:
        raise ValueError('variance_threshold must be non-negative')
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
    keep = coordinate_variances.mean(dim=1) <= variance_threshold
    return {
        'boxes': mean_boxes[keep],
        'labels': labels[keep],
        'scores': scores[keep],
        'keep': keep,
    }


def select_classification_losses(
    losses: Mapping[str, object],
) -> Dict[str, object]:
    '''Select the RPN and RoI classification terms from MMDet loss output.'''

    allowed = {'loss_rpn_cls', 'loss_cls'}
    return {key: value for key, value in losses.items() if key in allowed}
