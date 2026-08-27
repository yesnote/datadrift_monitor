'''Register ADAOD common components with MMDetection.

Use this module in an MMEngine config::

    custom_imports = dict(
        imports=['methods.common.mmdet.registration'],
        allow_failed_imports=False)
'''

try:
    from mmdet.registry import LOOPS, METRICS, MODELS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'ADAOD MMDetection registrations require the repository-local '
        'MMDetection 3.3 package and its MMCV/MMEngine dependencies. '
        'See requirements/README.md for the candidate environment.'
    ) from exc

from .models.backbones.vgg16 import VGG16Backbone
from .models.progressive_domain_adaptation import (
    ProgressiveDomainDiscriminator,
)
from .models.roi_heads.vgg_bbox_head import VGGShared2FCBBoxHead
from .models.roi_heads.class_probability_roi_head import (
    ClassProbabilityRoIHead,
)
from .metrics.detectron2_voc_metric import Detectron2PascalVocMetric
from .loops.segmented_iter_loop import ADAODSegmentedIterBasedTrainLoop

MODELS.register_module(name='VGG16Backbone', module=VGG16Backbone)
MODELS.register_module(
    name='ProgressiveDomainDiscriminator',
    module=ProgressiveDomainDiscriminator,
)
MODELS.register_module(
    name='VGGShared2FCBBoxHead', module=VGGShared2FCBBoxHead)
MODELS.register_module(
    name='ClassProbabilityRoIHead',
    module=ClassProbabilityRoIHead,
)
LOOPS.register_module(
    name='ADAODSegmentedIterBasedTrainLoop',
    module=ADAODSegmentedIterBasedTrainLoop,
)
METRICS.register_module(
    name='Detectron2PascalVocMetric',
    module=Detectron2PascalVocMetric,
)
