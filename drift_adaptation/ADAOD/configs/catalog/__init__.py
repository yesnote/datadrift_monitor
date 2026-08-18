'''Small, method-neutral catalog accessors.'''

from .datasets import get_dataset, list_datasets
from .detectors import get_detector, list_detectors
from .runtime import get_runtime, list_runtimes

__all__ = (
    'get_dataset',
    'get_detector',
    'get_runtime',
    'list_datasets',
    'list_detectors',
    'list_runtimes',
)

