'''Register AADA components with MMDetection.'''

try:
    from mmdet.registry import MODELS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'AADA registration requires the pinned MMDetection environment'
    ) from exc

import methods.common.mmdet.registration  # noqa: F401
from methods.aada.models.detector import AadaDetector


MODELS.register_module(name='AadaDetector', module=AadaDetector)
