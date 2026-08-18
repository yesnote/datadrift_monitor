'''MMDetection-dependent checks for the VGG bbox head.'''

import inspect

import pytest
import torch

try:
    import mmdet  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip('MMDetection environment is not available', allow_module_level=True)

from methods.common.mmdet.models.roi_heads.vgg_bbox_head import (
    VGGShared2FCBBoxHead)


def test_vgg_bbox_head_defaults_and_dropout_forward():
    signature = inspect.signature(VGGShared2FCBBoxHead.__init__)
    assert signature.parameters['fc_out_channels'].default == 1024
    assert signature.parameters['dropout'].default == 0.1

    head = VGGShared2FCBBoxHead(
        in_channels=8,
        roi_feat_size=2,
        num_classes=3,
        fc_out_channels=16)
    assert len(head.shared_fcs) == 2
    assert len(head.shared_dropouts) == 2
    assert all(layer.p == 0.1 for layer in head.shared_dropouts)

    head.eval()
    cls_score, bbox_pred = head(torch.randn(2, 8, 2, 2))
    assert cls_score.shape == (2, 4)
    assert bbox_pred.shape == (2, 12)


def test_vgg_bbox_head_uses_c2_xavier_only_for_shared_fcs(monkeypatch):
    head = VGGShared2FCBBoxHead(
        in_channels=8,
        roi_feat_size=2,
        num_classes=3,
        fc_out_channels=16,
    )
    kaiming_calls = []
    constant_calls = []
    original_kaiming = torch.nn.init.kaiming_uniform_
    original_constant = torch.nn.init.constant_

    def record_kaiming(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        kaiming_calls.append((tensor, a, mode, nonlinearity))
        return original_kaiming(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

    def record_constant(tensor, value):
        constant_calls.append((tensor, value))
        return original_constant(tensor, value)

    monkeypatch.setattr(torch.nn.init, 'kaiming_uniform_', record_kaiming)
    monkeypatch.setattr(torch.nn.init, 'constant_', record_constant)

    head.init_weights()

    assert len(kaiming_calls) == len(head.shared_fcs)
    for call, fc in zip(kaiming_calls, head.shared_fcs):
        tensor, a, mode, nonlinearity = call
        assert tensor is fc.weight
        assert a == 1
        assert mode == 'fan_in'
        assert nonlinearity == 'leaky_relu'
        assert any(
            tensor is fc.bias and value == 0
            for tensor, value in constant_calls
        )
    assert not any(tensor is head.fc_cls.weight for tensor, *_ in kaiming_calls)
    assert not any(tensor is head.fc_reg.weight for tensor, *_ in kaiming_calls)
    predictor_configs = {
        config['override']['name']: config
        for config in head.init_cfg
        if isinstance(config.get('override'), dict)
    }
    assert predictor_configs['fc_cls']['std'] == 0.01
    assert predictor_configs['fc_reg']['std'] == 0.001
