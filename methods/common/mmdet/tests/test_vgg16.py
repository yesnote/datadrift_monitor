'''PyTorch-only tests for the BN-free VGG16 backbone.'''

import torch
from torch import nn

from methods.common.mmdet.models.backbones.vgg16 import VGG16Backbone


def test_vgg16_has_expected_named_layers_without_batch_norm_or_pool5():
    model = VGG16Backbone()
    module_names = dict(model.features.named_modules())

    assert 'conv1_1' in module_names
    assert 'conv5_3' in module_names
    assert 'pool4' in module_names
    assert 'pool5' not in module_names
    assert sum(isinstance(module, nn.MaxPool2d)
               for module in model.modules()) == 4
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                   for module in model.modules())


def test_vgg16_output_has_stride_16_and_512_channels():
    model = VGG16Backbone().eval()
    inputs = torch.randn(2, 3, 64, 80)

    with torch.no_grad():
        outputs = model(inputs)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 512, 4, 5)


def test_vgg16_freezes_only_requested_leading_stages():
    model = VGG16Backbone(frozen_stages=2)
    model.train()

    assert not model.features.conv1_1.weight.requires_grad
    assert not model.features.conv2_2.weight.requires_grad
    assert model.features.conv3_1.weight.requires_grad
    assert not model.features.relu1_1.training
    assert model.features.relu3_1.training
