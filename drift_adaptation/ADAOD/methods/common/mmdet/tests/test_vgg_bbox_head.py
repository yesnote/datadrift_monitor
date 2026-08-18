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
    assert signature.parameters['fc_out_channels'].default == 4096
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
