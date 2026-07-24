"""Small COCO-style annotation pool helpers.

These utilities intentionally avoid MMDetection imports so they can be used by
lightweight samplers before detector dependencies load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from methods.common.io import read_json, write_json

JsonDict = Dict[str, Any]
PoolInput = Union[Path, JsonDict]


def read_coco_json(path: Path) -> JsonDict:
    return read_json(Path(path))


def write_coco_json(data: JsonDict, path: Path) -> None:
    write_json(path, data)


def load_coco_pool(pool: PoolInput) -> JsonDict:
    if isinstance(pool, Path):
        return read_coco_json(pool)
    return pool


def image_ids(coco_data: JsonDict) -> List[Any]:
    return [image['id'] for image in coco_data.get('images', [])]


def category_counts(records: Iterable[Dict[str, Any]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for record in records:
        category_id = record.get('category_id')
        if category_id is not None:
            counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def _metadata_from(oracle_data: JsonDict) -> JsonDict:
    subset = {'images': [], 'categories': oracle_data.get('categories', [])}
    for optional_key in ('info', 'licenses'):
        if optional_key in oracle_data:
            subset[optional_key] = oracle_data[optional_key]
    return subset


def build_coco_subset(
    oracle_data: JsonDict,
    selected_image_ids: Iterable[Any],
    include_annotations: bool,
) -> JsonDict:
    selected = set(selected_image_ids)
    subset = _metadata_from(oracle_data)
    subset['images'] = [
        image for image in oracle_data.get('images', [])
        if image.get('id') in selected
    ]
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle_data.get('annotations', [])
            if ann.get('image_id') in selected
        ]
    return subset


def build_coco_subset_ordered(
    oracle_data: JsonDict,
    selected_image_ids: Iterable[Any],
    include_annotations: bool,
) -> JsonDict:
    """Build a COCO subset preserving the supplied image-id order."""

    subset = _metadata_from(oracle_data)
    image_lookup = {
        image.get('id'): image for image in oracle_data.get('images', [])
    }
    ordered_ids = [
        image_id for image_id in selected_image_ids
        if image_id in image_lookup
    ]
    selected = set(ordered_ids)
    subset['images'] = [image_lookup[image_id] for image_id in ordered_ids]
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle_data.get('annotations', [])
            if ann.get('image_id') in selected
        ]
    return subset


def write_coco_pool_split(
    oracle_data: JsonDict,
    labeled_image_ids: Iterable[Any],
    unlabeled_image_ids: Iterable[Any],
    out_labeled_json: Path,
    out_unlabeled_json: Path,
    labeled_include_annotations: bool,
) -> Tuple[List[Any], List[Any]]:
    """Write labeled/unlabeled COCO files from explicit image-id lists."""

    labeled_ids = list(labeled_image_ids)
    unlabeled_ids = list(unlabeled_image_ids)
    write_coco_json(
        build_coco_subset_ordered(
            oracle_data,
            labeled_ids,
            include_annotations=labeled_include_annotations,
        ),
        out_labeled_json,
    )
    write_coco_json(
        build_coco_subset_ordered(
            oracle_data,
            unlabeled_ids,
            include_annotations=False,
        ),
        out_unlabeled_json,
    )
    return labeled_ids, unlabeled_ids


def update_labeled_unlabeled_from_oracle(
    oracle_json: Path,
    last_labeled_json: Path,
    selected_image_ids: Sequence[Any],
    out_labeled_json: Path,
    out_unlabeled_json: Path,
) -> Tuple[List[Any], List[Any]]:
    """Create next labeled/unlabeled JSONs from oracle annotations.

    Returns the final labeled and unlabeled image id lists in oracle image order.
    Selected ids already present in the labeled set are ignored.
    """

    oracle_data = read_coco_json(oracle_json)
    last_labeled_data = read_coco_json(last_labeled_json)

    oracle_ids = image_ids(oracle_data)
    oracle_id_set = set(oracle_ids)
    last_labeled_set = set(image_ids(last_labeled_data))

    new_selected = [
        image_id for image_id in selected_image_ids
        if image_id in oracle_id_set and image_id not in last_labeled_set
    ]
    labeled_set = last_labeled_set.union(new_selected)

    labeled_ids = [image_id for image_id in oracle_ids if image_id in labeled_set]
    unlabeled_ids = [image_id for image_id in oracle_ids if image_id not in labeled_set]

    write_coco_json(
        build_coco_subset(oracle_data, labeled_ids, include_annotations=True),
        out_labeled_json,
    )
    write_coco_json(
        build_coco_subset(oracle_data, unlabeled_ids, include_annotations=False),
        out_unlabeled_json,
    )
    return labeled_ids, unlabeled_ids
