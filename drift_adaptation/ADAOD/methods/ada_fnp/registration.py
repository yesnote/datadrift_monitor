'''Register ADA-FNP Faster R-CNN components with MMDetection.'''

try:
    from mmdet.registry import MODELS, TRANSFORMS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'ADA-FNP registrations require MMDetection 3.3 with its MMCV and '
        'MMEngine dependencies.') from exc

# Importing the common registration keeps the vendored mmdet package pristine.
import methods.common.mmdet.registration  # noqa: F401
from methods.ada_fnp.models.domain_discriminator import DomainDiscriminator
from methods.ada_fnp.models.detector import ADAFNPBranch, ADAFNPDetector
from methods.ada_fnp.models.roi_head import ADAFNPRoIHead
from methods.ada_fnp.training.augmentations import PTStrongAugmentation

MODELS.register_module(
    name='ADAFNPDomainDiscriminator', module=DomainDiscriminator)
MODELS.register_module(name='ADAFNPBranch', module=ADAFNPBranch)
MODELS.register_module(name='ADAFNPDetector', module=ADAFNPDetector)
MODELS.register_module(name='ADAFNPRoIHead', module=ADAFNPRoIHead)
TRANSFORMS.register_module(
    name='PTStrongAugmentation', module=PTStrongAugmentation)

__all__ = [
    'ADAFNPBranch',
    'ADAFNPDetector',
    'ADAFNPRoIHead',
    'DomainDiscriminator',
    'PTStrongAugmentation',
]
