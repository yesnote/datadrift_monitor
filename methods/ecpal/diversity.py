"""ECPAL Error-Count Diversity selection."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from methods.common.selection import image_id_sort_key


def _composition_vector(record: Mapping[str, Any]) -> np.ndarray:
    components = record.get('components', {})
    values = np.asarray([
        float(components.get('pi_cls', 0.0)),
        float(components.get('pi_loc', 0.0)),
        float(components.get('pi_miss', 0.0)),
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('ECPAL composition contains non-finite values')
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(3, 1.0 / 3.0, dtype=np.float64)
    return values / total


def jensen_shannon_distance(first: Iterable[float], second: Iterable[float]) -> float:
    p = np.asarray(list(first), dtype=np.float64)
    q = np.asarray(list(second), dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError('JS distance vectors must have the same shape')
    if p.size == 0:
        return 0.0
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = p / p.sum() if p.sum() > 0.0 else np.full_like(p, 1.0 / p.size)
    q = q / q.sum() if q.sum() > 0.0 else np.full_like(q, 1.0 / q.size)
    m = 0.5 * (p + q)

    def kl(lhs: np.ndarray, rhs: np.ndarray) -> float:
        mask = lhs > 0.0
        return float(np.sum(lhs[mask] * np.log(lhs[mask] / rhs[mask])))

    divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(math.sqrt(max(divergence, 0.0)))


def _selection_sort_key(record: Mapping[str, Any], distance: float = 0.0) -> tuple:
    return (
        -float(distance),
        -float(record.get('score', 0.0)),
        image_id_sort_key(record.get('image_id')),
    )


def farthest_first_select(
    candidates: Iterable[Mapping[str, Any]],
    budget: int,
) -> Dict[str, Any]:
    """Select candidates by ECD farthest-first with ECA tie-breaks."""

    records = [dict(candidate) for candidate in candidates]
    if budget <= 0 or not records:
        return {'selected_image_ids': [], 'selected_records': []}

    by_id = {record['image_id']: record for record in records}
    vectors = {record['image_id']: _composition_vector(record) for record in records}
    selected_ids: List[Any] = []
    selected_records: List[Dict[str, Any]] = []

    first = sorted(records, key=lambda record: _selection_sort_key(record))[0]
    selected_ids.append(first['image_id'])
    first_record = dict(first)
    first_record.setdefault('metadata', {})
    first_record['metadata'] = dict(first_record['metadata'])
    first_record['metadata']['nearest_selected_distance'] = None
    first_record['metadata']['selection_reason'] = 'max_eca'
    first_record['selection_rank'] = 1
    selected_records.append(first_record)

    while len(selected_ids) < min(int(budget), len(records)):
        selected_set = set(selected_ids)
        best_record = None
        best_distance = -1.0
        for record in records:
            image_id = record['image_id']
            if image_id in selected_set:
                continue
            distance = min(
                jensen_shannon_distance(vectors[image_id], vectors[selected_id])
                for selected_id in selected_ids
            )
            if best_record is None or _selection_sort_key(record, distance) < _selection_sort_key(best_record, best_distance):
                best_record = record
                best_distance = distance
        if best_record is None:
            break
        image_id = best_record['image_id']
        selected_ids.append(image_id)
        selected_record = dict(by_id[image_id])
        selected_record.setdefault('metadata', {})
        selected_record['metadata'] = dict(selected_record['metadata'])
        selected_record['metadata']['nearest_selected_distance'] = float(best_distance)
        selected_record['metadata']['selection_reason'] = 'farthest_first'
        selected_record['selection_rank'] = len(selected_ids)
        selected_records.append(selected_record)

    return {
        'selected_image_ids': selected_ids,
        'selected_records': selected_records,
    }


def attach_selection_metadata(
    candidates: Iterable[Mapping[str, Any]],
    selected_records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    selected_by_id = {record['image_id']: dict(record) for record in selected_records}
    output = []
    for candidate in candidates:
        record = dict(candidate)
        record.setdefault('metadata', {})
        record['metadata'] = dict(record['metadata'])
        selected = selected_by_id.get(record['image_id'])
        if selected is not None:
            record['selection_rank'] = selected.get('selection_rank')
            record['metadata'].update(selected.get('metadata', {}))
        output.append(record)
    return output
