'''Register ADA-FNP Faster R-CNN components with MMDetection.'''

try:
    from mmdet.registry import MODELS, TRANSFORMS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'ADA-FNP registrations require MMDetection 3.3 with its MMCV and '
        'MMEngine dependencies.') from exc

# Importing the common registration keeps the vendored mmdet package pristine.
import methods.common.mmdet.registration  # noqa: F401
from methods.ada_fnp.models.detector import (
    AdaFnpDetector,
    AdaFnpDetectorBranch,
)
from methods.ada_fnp.models.domain_adaptation import AdaFnpDomainDiscriminator
from methods.ada_fnp.models.mc_dropout_roi_head import (
    AdaFnpMonteCarloDropoutRoIHead,
)
from methods.ada_fnp.probabilistic_teacher_augmentation import (
    ProbabilisticTeacherStrongAugmentation,
)

MODELS.register_module(
    name='AdaFnpDomainDiscriminator', module=AdaFnpDomainDiscriminator)
MODELS.register_module(
    name='AdaFnpDetectorBranch', module=AdaFnpDetectorBranch)
MODELS.register_module(name='AdaFnpDetector', module=AdaFnpDetector)
MODELS.register_module(
    name='AdaFnpMonteCarloDropoutRoIHead',
    module=AdaFnpMonteCarloDropoutRoIHead,
)
TRANSFORMS.register_module(
    name='ProbabilisticTeacherStrongAugmentation',
    module=ProbabilisticTeacherStrongAugmentation,
)
