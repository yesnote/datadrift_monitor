'''Runtime locations and reproducibility settings shared by all methods.'''

from copy import deepcopy
from typing import Mapping, Tuple


_RUNTIMES = {
    'default': {
        'name': 'default',
        'work_root': 'work_dirs',
        'dataset_cache_root': 'work_dirs/.dataset_cache',
        'launcher': 'none',
        'deterministic': True,
        'cudnn_benchmark': False,
    },
}


def list_runtimes() -> Tuple[str, ...]:
    '''Return runtime keys in deterministic order.'''

    return tuple(sorted(_RUNTIMES))


def get_runtime(key: str) -> Mapping:
    '''Return an independent runtime configuration.'''

    try:
        return deepcopy(_RUNTIMES[key])
    except KeyError as error:
        choices = ', '.join(list_runtimes())
        raise KeyError(
            f'unknown runtime {key!r}; available: {choices}'
        ) from error

