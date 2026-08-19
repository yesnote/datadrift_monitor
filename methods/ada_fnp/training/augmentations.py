'''PT strong photometric augmentation for MMDetection BGR images.'''

import random
from typing import MutableMapping

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


class _PTGaussianBlur:
    '''Reference PT Gaussian blur with Python-RNG radius sampling.'''

    def __call__(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(0.1, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class PTStrongAugmentation:
    '''Apply the exact PT strong photometric pipeline to a BGR uint8 image.

    The four PT component probabilities remain fixed.
    '''

    def __init__(self) -> None:
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                _PTGaussianBlur(),
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

        # PT reads images in Detectron2's default BGR format, then passes the
        # array directly to a PIL RGB image. Preserve that literal
        # channel behavior for reproduction instead of correcting it with a
        # BGR/RGB swap here.
        pil_image = Image.fromarray(np.ascontiguousarray(bgr_image))
        augmented_image = np.asarray(self.augmentation(pil_image))
        results['img'] = np.array(
            augmented_image, dtype=np.uint8, copy=True, order='C'
        )
        return results
