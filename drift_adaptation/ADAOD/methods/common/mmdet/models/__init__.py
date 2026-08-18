'''PyTorch model primitives shared by MMDetection integrations.'''

from .backbones.vgg16 import VGG16Backbone, map_torchvision_vgg16_state_dict
from .layers.gradient_reversal import GradientReversal, gradient_reverse

__all__ = [
    'GradientReversal',
    'VGG16Backbone',
    'gradient_reverse',
    'map_torchvision_vgg16_state_dict',
]
