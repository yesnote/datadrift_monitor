'''TIDE-style class-aware false-negative target construction.'''

from __future__ import annotations

import torch


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError('boxes1 must have shape [N, 4]')
    if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
        raise ValueError('boxes2 must have shape [M, 4]')
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(torch.finfo(intersection.dtype).eps)


def count_false_negatives(
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    prediction_labels: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    ground_truth_labels: torch.Tensor,
    ground_truth_ignore: torch.Tensor | None = None,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> int:
    '''Count unmatched non-ignore GT after score-ordered one-to-one matching.'''

    if ground_truth_ignore is None:
        ground_truth_ignore = torch.zeros_like(ground_truth_labels, dtype=torch.bool)
    if len(ground_truth_ignore) != len(ground_truth_boxes):
        raise ValueError('ground_truth_ignore length mismatch')
    valid_gt = ~ground_truth_ignore.bool()
    boxes = ground_truth_boxes[valid_gt]
    labels = ground_truth_labels[valid_gt]
    if len(boxes) == 0:
        return 0
    if len(prediction_boxes) == 0:
        return len(boxes)
    order = torch.argsort(prediction_scores, descending=True, stable=True)
    order = order[:max_detections]
    ious = pairwise_iou(prediction_boxes[order], boxes)
    matched = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    for prediction_index, original_index in enumerate(order):
        candidates = (labels == prediction_labels[original_index]) & ~matched
        if not candidates.any():
            continue
        candidate_indices = candidates.nonzero(as_tuple=False).flatten()
        candidate_ious = ious[prediction_index, candidate_indices]
        best_value, best_offset = candidate_ious.max(dim=0)
        if best_value >= iou_threshold:
            matched[candidate_indices[best_offset]] = True
    return int((~matched).sum().item())
