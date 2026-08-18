'''Dataset scenarios supported by the current repository foundation.'''

from copy import deepcopy
from typing import Mapping, Tuple


_CITYSCAPES_CLASSES: Tuple[str, ...] = (
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
)


_DATASETS = {
    'cityscapes-to-foggy': {
        'source': {
            'name': 'cityscapes',
            'image_root': 'data/leftImg8bit',
            'annotation_root': 'data/gtFine',
            'split': 'train',
            'expected_images': 2975,
            'sample_id_namespace': 'cityscapes.train',
            'annotation_access': 'labeled',
        },
        'target': {
            'name': 'foggy_cityscapes',
            'image_root': 'data/leftImg8bit_foggy',
            'annotation_root': 'data/gtFine',
            'train_split': 'train',
            'eval_split': 'val',
            'beta': 0.02,
            'filename_suffix': '_leftImg8bit_foggy_beta_0.02.png',
            'expected_train_images': 2975,
            'expected_eval_images': 500,
            'train_sample_id_namespace': (
                'foggy-cityscapes.beta-0.02.train'
            ),
            'eval_sample_id_namespace': 'foggy-cityscapes.beta-0.02.val',
            'train_annotation_access': 'oracle_only',
            'eval_annotation_access': 'evaluator_only',
        },
        'classes': _CITYSCAPES_CLASSES,
    },
}


def list_datasets() -> Tuple[str, ...]:
    '''Return dataset scenario keys in deterministic order.'''

    return tuple(sorted(_DATASETS))


def get_dataset(key: str) -> Mapping:
    '''Return an independent dataset scenario configuration.'''

    try:
        return deepcopy(_DATASETS[key])
    except KeyError as error:
        choices = ', '.join(list_datasets())
        raise KeyError(
            f'unknown dataset scenario {key!r}; available: {choices}'
        ) from error
