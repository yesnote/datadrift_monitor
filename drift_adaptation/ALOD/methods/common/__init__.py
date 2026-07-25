"""Method-runtime helpers shared by active learning methods.

This package must not depend on runner/tooling helpers. Runner and preparation
helpers belong in tools.common and may call into this package when they need
pure data utilities.
"""

from .coco_pool import (
    build_coco_subset,
    build_coco_subset_ordered,
    category_counts,
    image_ids,
    load_coco_pool,
    read_coco_json,
    write_candidate_pool_from_selection,
    write_next_round_pool_split,
    write_coco_json,
)
from .detections import load_detection_records
from .image_identity import (
    canonical_image_ids,
    normalize_image_id,
    normalize_image_ids,
    validate_image_ids_subset,
)
from .io import read_json, to_jsonable, write_json
from .paths import is_relative_to
from .results import acquisition_result
from .selection import (
    deterministic_random_sample,
    fill_to_budget,
    image_id_sort_key,
    ranked_ids_by_score,
    top_k_by_score,
    unique_in_order,
)
from .vectors import l2_normalize

__all__ = [
    'build_coco_subset',
    'build_coco_subset_ordered',
    'category_counts',
    'canonical_image_ids',
    'acquisition_result',
    'deterministic_random_sample',
    'fill_to_budget',
    'image_ids',
    'image_id_sort_key',
    'is_relative_to',
    'load_coco_pool',
    'load_detection_records',
    'l2_normalize',
    'normalize_image_id',
    'normalize_image_ids',
    'ranked_ids_by_score',
    'read_json',
    'read_coco_json',
    'to_jsonable',
    'top_k_by_score',
    'unique_in_order',
    'validate_image_ids_subset',
    'write_candidate_pool_from_selection',
    'write_coco_json',
    'write_next_round_pool_split',
    'write_json',
]
