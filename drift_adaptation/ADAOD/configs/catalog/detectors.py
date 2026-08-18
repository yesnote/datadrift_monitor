'''Detector metadata independent of concrete ADA methods.'''

from copy import deepcopy
from typing import Mapping, Tuple


_DETECTORS = {
    'faster-rcnn-vgg16': {
        'name': 'faster-rcnn-vgg16',
        'architecture': 'FasterRCNN',
        'backbone': 'VGG16',
        'batch_normalization': False,
        'num_classes': 8,
        'capabilities': (
            'two_stage',
            'roi_features',
            'fixed_proposal_roi_inference',
            'domain_feature_tap',
            'mc_dropout',
        ),
    },
}


def list_detectors() -> Tuple[str, ...]:
    '''Return detector keys in deterministic order.'''

    return tuple(sorted(_DETECTORS))


def get_detector(key: str) -> Mapping:
    '''Return independent metadata for one detector.'''

    try:
        return deepcopy(_DETECTORS[key])
    except KeyError as error:
        choices = ', '.join(list_detectors())
        raise KeyError(
            f'unknown detector {key!r}; available: {choices}'
        ) from error

