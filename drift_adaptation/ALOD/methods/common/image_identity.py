"""Canonical image-id helpers for method runtime artifacts."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping


def normalize_image_id(value: Any) -> Any:
    """Return a JSON/Python scalar representation for an image id."""

    if hasattr(value, 'item'):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def normalize_image_ids(values: Iterable[Any]) -> List[Any]:
    """Return image ids as plain JSON/Python scalar values."""

    return [normalize_image_id(value) for value in values]


def canonical_image_ids(coco_data: Mapping[str, Any]) -> List[Any]:
    """Return canonical COCO image ids from an annotation payload."""

    return [
        normalize_image_id(image['id'])
        for image in coco_data.get('images', [])
    ]


def validate_image_ids_subset(
    ids: Iterable[Any],
    valid_ids: Iterable[Any],
    artifact_name: str,
) -> None:
    """Raise if any artifact image id is outside the valid canonical set."""

    valid = {normalize_image_id(image_id) for image_id in valid_ids}
    missing = [
        normalize_image_id(image_id)
        for image_id in ids
        if normalize_image_id(image_id) not in valid
    ]
    if missing:
        raise ValueError(
            '%s contains image ids not found in the canonical pool: %s'
            % (artifact_name, sorted(set(missing), key=str)[:10])
        )
