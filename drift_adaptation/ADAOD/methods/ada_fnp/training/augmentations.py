'''PT strong photometric augmentation for MMDetection BGR images.'''

import random
from typing import MutableMapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


class _PTGaussianBlur:
    '''Reference PT Gaussian blur with Python-RNG radius sampling.'''

    def __init__(self, sigma: Sequence[float] = (0.1, 2.0)) -> None:
        if len(sigma) != 2 or sigma[0] <= 0 or sigma[0] > sigma[1]:
            raise ValueError('sigma must be a positive increasing pair')
        self.sigma = (float(sigma[0]), float(sigma[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class PTStrongAugmentation:
    '''Apply the exact PT strong photometric pipeline to a BGR uint8 image.

    ``p`` is an outer ablation gate. Its default of one does not consume a
    random draw, so the internal torchvision transforms retain their reference
    RNG sequence. The four PT component probabilities remain fixed.
    '''

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError('p must be in the interval [0, 1]')
        self.p = float(p)
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                _PTGaussianBlur((0.1, 2.0)),
            ], p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])

    @staticmethod
    def _validate_image(image: object) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("results['img'] must be a numpy array")
        if image.dtype != np.uint8:
            raise TypeError("results['img'] must have uint8 dtype")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("results['img'] must have shape [H, W, 3]")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("results['img'] height and width must be positive")
        return image

    def __call__(self, results: MutableMapping) -> MutableMapping:
        if 'img' not in results:
            raise KeyError("results must contain an 'img' entry")
        bgr_image = self._validate_image(results['img'])
        if self.p == 0.0:
            results['img'] = np.ascontiguousarray(bgr_image)
            return results
        if self.p < 1.0 and torch.rand(()) >= self.p:
            results['img'] = np.ascontiguousarray(bgr_image)
            return results

        rgb_image = np.ascontiguousarray(bgr_image[..., ::-1])
        pil_image = Image.fromarray(rgb_image)
        augmented_rgb = np.asarray(self.augmentation(pil_image))
        results['img'] = np.ascontiguousarray(augmented_rgb[..., ::-1])
        return results


__all__ = ['PTStrongAugmentation']
