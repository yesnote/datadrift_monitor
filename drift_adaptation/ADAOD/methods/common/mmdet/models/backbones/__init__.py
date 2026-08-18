'''Backbones maintained outside the vendored MMDetection package.'''

from .vgg16 import VGG16Backbone, map_torchvision_vgg16_state_dict

__all__ = ['VGG16Backbone', 'map_torchvision_vgg16_state_dict']
