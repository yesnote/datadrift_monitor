'''Register ADAOD common components with MMDetection.

Use this module in an MMEngine config::

    custom_imports = dict(
        imports=['methods.common.mmdet.registration'],
        allow_failed_imports=False)
'''

try:
    from mmdet.registry import MODELS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'ADAOD MMDetection registrations require the repository-local '
        'MMDetection 3.3 package and its MMCV/MMEngine dependencies. '
        'See requirements/README.md for the candidate environment.'
    ) from exc

from .models.backbones.vgg16 import VGG16Backbone
from .models.layers.gradient_reversal import GradientReversal
from .models.roi_heads.vgg_bbox_head import VGGShared2FCBBoxHead
from .metrics import PTVOCMetric

MODELS.register_module(name='ADAODVGG16', module=VGG16Backbone)
MODELS.register_module(
    name='ADAODVGGShared2FCBBoxHead', module=VGGShared2FCBBoxHead)
MODELS.register_module(
    name='ADAODGradientReversal', module=GradientReversal)

__all__ = [
    'GradientReversal',
    'PTVOCMetric',
    'VGG16Backbone',
    'VGGShared2FCBBoxHead',
]
