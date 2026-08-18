'''MMDetection-independent entry points for ADAOD model extensions.

Import :mod:`methods.common.mmdet.registration` from an MMEngine config to
register the components with MMDetection.  Keeping registration separate lets
the PyTorch-only primitives remain testable before MMDetection is installed.
'''

from .models.backbones.vgg16 import (VGG16Backbone,
                                     map_torchvision_vgg16_state_dict)
from .models.layers.gradient_reversal import (GradientReversal,
                                               gradient_reverse)

__all__ = [
    'GradientReversal',
    'VGG16Backbone',
    'gradient_reverse',
    'map_torchvision_vgg16_state_dict',
]
