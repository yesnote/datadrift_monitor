'''Unit tests for the PT strong photometric augmentation.'''

import random

import numpy as np
import pytest
import torch
from PIL import Image

from methods.ada_fnp.training.augmentations import (
    PTStrongAugmentation, _PTGaussianBlur)


def _image() -> np.ndarray:
    values = np.arange(18 * 24 * 3, dtype=np.uint16).reshape(18, 24, 3)
    return (values % 256).astype(np.uint8)


def test_zero_probability_is_exact_identity():
    image = _image()
    metadata = object()
    results = {'img': image.copy(), 'img_id': 7, 'metadata': metadata}

    returned = PTStrongAugmentation(p=0.0)(results)

    assert returned is results
    assert np.array_equal(results['img'], image)
    assert results['img_id'] == 7
    assert results['metadata'] is metadata


def test_fixed_torch_and_python_seeds_replay_exactly():
    transform = PTStrongAugmentation(p=1.0)
    image = _image()

    random.seed(5678)
    torch.manual_seed(1234)
    first = transform({'img': image.copy()})['img']
    random.seed(5678)
    torch.manual_seed(1234)
    second = transform({'img': image.copy()})['img']

    assert np.array_equal(first, second)


def test_pt_gaussian_blur_replays_from_python_seed():
    blur = _PTGaussianBlur((0.1, 2.0))
    image = Image.fromarray(_image()[..., ::-1])

    random.seed(17)
    first = np.asarray(blur(image))
    random.seed(17)
    replay = np.asarray(blur(image))
    random.seed(18)
    different_radius = np.asarray(blur(image))

    assert np.array_equal(first, replay)
    assert not np.array_equal(first, different_radius)


def test_preserves_bgr_array_contract_and_other_keys():
    transform = PTStrongAugmentation(p=1.0)
    image = _image()[:, ::-1]
    annotations = [{'bbox': [1, 2, 3, 4]}]
    results = {'img': image, 'annotations': annotations, 'img_id': 11}

    torch.manual_seed(9)
    returned = transform(results)

    assert returned is results
    assert results['img'].shape == image.shape
    assert results['img'].dtype == np.uint8
    assert results['img'].flags.c_contiguous
    assert results['img'].flags.writeable
    assert results['annotations'] is annotations
    assert results['img_id'] == 11


def test_preserves_literal_pt_bgr_as_pil_channel_behavior():
    transform = PTStrongAugmentation(p=1.0)
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    def zero_pil_red_channel(pil_image):
        array = np.asarray(pil_image).copy()
        array[..., 0] = 0
        return Image.fromarray(array)

    transform.augmentation = zero_pil_red_channel
    augmented = transform({'img': image.copy()})['img']

    assert np.all(augmented[..., 0] == 0)
    assert np.all(augmented[..., 1] == 20)
    assert np.all(augmented[..., 2] == 30)


@pytest.mark.parametrize(
    'image,exception', [
        (torch.zeros(4, 4, 3), TypeError),
        (np.zeros((4, 4, 3), dtype=np.float32), TypeError),
        (np.zeros((4, 4), dtype=np.uint8), ValueError),
        (np.zeros((4, 4, 4), dtype=np.uint8), ValueError),
        (np.zeros((0, 4, 3), dtype=np.uint8), ValueError),
    ])
def test_invalid_images_fail_fast(image, exception):
    with pytest.raises(exception):
        PTStrongAugmentation()(dict(img=image))


def test_missing_image_and_invalid_outer_probability_fail_fast():
    with pytest.raises(KeyError, match='img'):
        PTStrongAugmentation()({})
    with pytest.raises(ValueError, match='interval'):
        PTStrongAugmentation(p=1.1)
