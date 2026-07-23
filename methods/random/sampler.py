"""Random active learning sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from methods.common.coco_pool import image_ids, read_coco_json
from methods.common.selection import deterministic_random_sample

PoolInput = Union[Path, Dict[str, Any]]


def _load_pool(pool: PoolInput) -> Dict[str, Any]:
    if isinstance(pool, Path):
        return read_coco_json(pool)
    return pool


def sample(pool: PoolInput, budget: int, seed: int = 0) -> List[Any]:
    """Return a deterministic random subset of image ids from a COCO pool."""

    return deterministic_random_sample(image_ids(_load_pool(pool)), budget, seed=seed)
