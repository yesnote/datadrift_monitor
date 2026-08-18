'''Method-independent acquisition artifacts, normalization, and selection.'''

from .artifacts import (
    JsonArtifact,
    JsonArtifactRecord,
    canonical_json_bytes,
    read_json_artifact,
    sha256_file,
    write_json_artifact,
)
from .normalization import lower_clamped_standardize, standardize_components
from .selection import AcquisitionScore, build_product_scores, select_top_k

__all__ = [
    'AcquisitionScore',
    'JsonArtifact',
    'JsonArtifactRecord',
    'build_product_scores',
    'canonical_json_bytes',
    'lower_clamped_standardize',
    'read_json_artifact',
    'select_top_k',
    'sha256_file',
    'standardize_components',
    'write_json_artifact',
]
