'''Tests for torchvision VGG16 state-dict mapping.'''

import pytest
import torch

from methods.common.mmdet.models.backbones.vgg16 import (
    _required_torchvision_shapes, map_torchvision_vgg16_state_dict)


def _meta_state_dict():
    return {
        key: torch.empty(shape, device='meta')
        for key, shape in _required_torchvision_shapes().items()
    }


def test_torchvision_mapping_covers_backbone_and_fc6_fc7_without_copying():
    source = _meta_state_dict()
    source['classifier.6.weight'] = torch.empty((1000, 4096), device='meta')
    source['classifier.6.bias'] = torch.empty((1000, ), device='meta')

    mapped = map_torchvision_vgg16_state_dict(
        source,
        backbone_prefix='backbone.',
        bbox_head_prefix='roi_head.bbox_head.')

    assert len(mapped) == 30
    assert mapped['backbone.features.conv1_1.weight'] is source[
        'features.0.weight']
    assert mapped['backbone.features.conv5_3.bias'] is source[
        'features.28.bias']
    assert mapped['roi_head.bbox_head.shared_fcs.0.weight'] is source[
        'classifier.0.weight']
    assert mapped['roi_head.bbox_head.shared_fcs.1.bias'] is source[
        'classifier.3.bias']
    assert not any('classifier.6' in key for key in mapped)


def test_torchvision_mapping_rejects_missing_or_wrong_shaped_weights():
    source = _meta_state_dict()
    source.pop('features.0.weight')
    with pytest.raises(KeyError, match='features.0.weight'):
        map_torchvision_vgg16_state_dict(source)

    source = _meta_state_dict()
    source['features.0.weight'] = torch.empty((1, ), device='meta')
    with pytest.raises(ValueError, match='unexpected shape'):
        map_torchvision_vgg16_state_dict(source)
