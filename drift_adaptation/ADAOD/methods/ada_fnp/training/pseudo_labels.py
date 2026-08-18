'''Uncertainty-filtered pseudo labels and view projection.'''

from typing import Dict, Mapping

import torch


def project_boxes(boxes: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    '''Project axis-aligned xyxy boxes through a 3 by 3 homography.'''

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('boxes must have shape [N, 4]')
    if homography.shape != (3, 3):
        raise ValueError('homography must have shape [3, 3]')
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    corners = torch.stack((
        torch.stack((x1, y1), dim=1),
        torch.stack((x2, y1), dim=1),
        torch.stack((x2, y2), dim=1),
        torch.stack((x1, y2), dim=1),
    ), dim=1)
    homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), dim=-1)
    projected = homogeneous @ homography.transpose(0, 1)
    projected = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    minimum = projected.amin(dim=1)
    maximum = projected.amax(dim=1)
    return torch.cat((minimum, maximum), dim=1)


def weak_to_strong_homography(
    weak_homography: torch.Tensor,
    strong_homography: torch.Tensor,
) -> torch.Tensor:
    '''Return the transform from weak-view pixels to strong-view pixels.'''

    if weak_homography.shape != (3, 3):
        raise ValueError('weak_homography must have shape [3, 3]')
    if strong_homography.shape != (3, 3):
        raise ValueError('strong_homography must have shape [3, 3]')
    strong_homography = strong_homography.to(
        device=weak_homography.device, dtype=weak_homography.dtype)
    return strong_homography @ torch.linalg.inv(weak_homography)


def select_pseudo_labels(
    mean_boxes: torch.Tensor,
    mean_probabilities: torch.Tensor,
    coordinate_variances: torch.Tensor,
    variance_threshold: float = 0.1,
    background_index: int = -1,
) -> Dict[str, torch.Tensor]:
    '''Keep proposals whose mean normalized localization variance is small.'''

    if variance_threshold < 0:
        raise ValueError('variance_threshold must be non-negative')
    if coordinate_variances.shape != mean_boxes.shape:
        raise ValueError('coordinate variances must match mean box shape')
    if mean_probabilities.ndim != 2 or len(mean_probabilities) != len(mean_boxes):
        raise ValueError('probability count must match boxes')
    if mean_probabilities.shape[1] < 2:
        raise ValueError('probabilities must contain foreground and background')
    background_index %= mean_probabilities.shape[1]
    keep_classes = torch.ones(
        mean_probabilities.shape[1], dtype=torch.bool,
        device=mean_probabilities.device,
    )
    keep_classes[background_index] = False
    scores, labels = mean_probabilities[:, keep_classes].max(dim=1)
    keep = coordinate_variances.mean(dim=1) <= variance_threshold
    return {
        'boxes': mean_boxes[keep],
        'labels': labels[keep],
        'scores': scores[keep],
        'keep': keep,
    }


def project_pseudo_labels(
    mean_boxes: torch.Tensor,
    mean_probabilities: torch.Tensor,
    coordinate_variances: torch.Tensor,
    weak_homography: torch.Tensor,
    strong_homography: torch.Tensor,
    variance_threshold: float = 0.1,
) -> Dict[str, torch.Tensor]:
    '''Filter weak-view predictions and project them into the strong view.'''

    selected = select_pseudo_labels(
        mean_boxes,
        mean_probabilities,
        coordinate_variances,
        variance_threshold=variance_threshold,
    )
    transform = weak_to_strong_homography(
        weak_homography, strong_homography)
    return {
        **selected,
        'boxes': project_boxes(selected['boxes'], transform),
        'weak_to_strong': transform,
    }


def classification_only_losses(
    losses: Mapping[str, object],
) -> Dict[str, object]:
    '''Select the RPN and RoI classification terms from MMDet loss output.'''

    allowed = {'loss_rpn_cls', 'loss_cls'}
    return {key: value for key, value in losses.items() if key in allowed}
