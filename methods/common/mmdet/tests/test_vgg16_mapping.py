'''Tests for PT Caffe and torchvision VGG16 state-dict mapping.'''

import hashlib

import pytest
import torch

from methods.common.assets import AssetVerificationError
from methods.common.mmdet.models.backbones import vgg16
from methods.common.mmdet.models.backbones.vgg16 import (
    VGG16Backbone,
    _required_caffe_conv_shapes,
    _required_torchvision_shapes,
    map_caffe_vgg16_state_dict,
    map_torchvision_vgg16_state_dict,
)


def _meta_state_dict():
    return {
        key: torch.empty(shape, device='meta')
        for key, shape in _required_torchvision_shapes().items()
    }


def _meta_caffe_state_dict():
    return {
        key: torch.empty(shape, device='meta')
        for key, shape in _required_caffe_conv_shapes().items()
    }


def test_caffe_mapping_is_conv_only_and_preserves_tensor_references():
    source = _meta_caffe_state_dict()
    source['classifier.0.weight'] = torch.empty((4096, 512 * 7 * 7), device='meta')

    mapped = map_caffe_vgg16_state_dict(source, backbone_prefix='backbone.')

    assert len(mapped) == 26
    assert mapped['backbone.features.conv1_1.weight'] is source[
        'features.0.weight']
    assert mapped['backbone.features.conv5_3.bias'] is source[
        'features.28.bias']
    assert not any('classifier' in key or 'shared_fcs' in key for key in mapped)


def test_caffe_mapping_rejects_missing_or_wrong_shaped_weights():
    source = _meta_caffe_state_dict()
    source.pop('features.0.weight')
    with pytest.raises(KeyError, match='features.0.weight'):
        map_caffe_vgg16_state_dict(source)

    source = _meta_caffe_state_dict()
    source['features.0.weight'] = torch.empty((1, ), device='meta')
    with pytest.raises(ValueError, match='unexpected shape'):
        map_caffe_vgg16_state_dict(source)


def test_backbone_verifies_and_loads_caffe_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'vgg16_caffe.pth'
    checkpoint.write_bytes(b'verified checkpoint placeholder')
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = {
        key: torch.tensor(0.25).expand(shape)
        for key, shape in _required_caffe_conv_shapes().items()
    }
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return source

    monkeypatch.setattr(vgg16.torch, 'load', fake_load)
    model = VGG16Backbone(
        pretrained_checkpoint=str(checkpoint),
        pretrained_sha256=expected_sha256,
    )

    assert calls == [(
        checkpoint.resolve(),
        {'map_location': 'cpu', 'weights_only': True},
    )]
    assert torch.all(model.features.conv1_1.weight == 0.25)
    assert torch.all(model.features.conv5_3.bias == 0.25)


def test_backbone_rejects_checkpoint_digest_before_deserialization(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / 'vgg16_caffe.pth'
    checkpoint.write_bytes(b'wrong checkpoint')
    monkeypatch.setattr(
        vgg16.torch,
        'load',
        lambda *args, **kwargs: pytest.fail('unverified checkpoint was loaded'),
    )

    with pytest.raises(AssetVerificationError, match='SHA-256 mismatch'):
        VGG16Backbone(
            pretrained_checkpoint=str(checkpoint),
            pretrained_sha256='0' * 64,
        )


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
