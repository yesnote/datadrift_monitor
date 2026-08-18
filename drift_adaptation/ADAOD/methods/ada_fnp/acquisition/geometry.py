'''Shared box-coordinate transforms for ADA-FNP MC inference.'''

from typing import Sequence

import torch
from torch import Tensor


def normalize_xyxy_box_samples(
    box_samples: Tensor, image_shape: Sequence[int]
) -> Tensor:
    if box_samples.ndim != 3 or box_samples.shape[-1] != 4:
        raise ValueError('box_samples must have shape [passes, boxes, 4]')
    if len(image_shape) < 2:
        raise ValueError('image_shape must provide height and width')
    image_height, image_width = image_shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError('image dimensions must be positive')
    x1, y1, x2, y2 = box_samples.unbind(dim=-1)
    return torch.stack((
        (x1 + x2) * 0.5 / image_width,
        (y1 + y2) * 0.5 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    ), dim=-1)
