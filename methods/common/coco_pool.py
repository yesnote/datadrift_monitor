"""Small COCO-style annotation pool helpers.

These utilities intentionally avoid MMDetection imports so they can be used by
lightweight samplers before detector dependencies load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

JsonDict = Dict[str, Any]


def read_coco_json(path: Path) -> JsonDict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_coco_json(data: JsonDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False)


def image_ids(coco_data: JsonDict) -> List[Any]:
    return [image['id'] for image in coco_data.get('images', [])]


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
