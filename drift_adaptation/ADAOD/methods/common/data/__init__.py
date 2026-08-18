'''Method-independent dataset identity and active-pool primitives.'''

from .annotations import OracleAnnotationProvider, reveal_acquired_annotations
from .image_identity import SampleIdentity
from .pool import AcquisitionRound, PoolState, split_budget

__all__ = [
    'AcquisitionRound',
    'OracleAnnotationProvider',
    'PoolState',
    'SampleIdentity',
    'reveal_acquired_annotations',
    'split_budget',
]
