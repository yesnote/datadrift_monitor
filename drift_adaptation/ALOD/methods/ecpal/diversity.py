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


def _composition_matrix(records: List[Mapping[str, Any]]) -> np.ndarray:
    vector_by_id = {
        record['image_id']: _composition_vector(record)
        for record in records
    }
    return np.stack(
        [vector_by_id[record['image_id']] for record in records],
        axis=0,
    )


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


def _jensen_shannon_distance_to_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return JS distances from each row of ``matrix`` to ``vector``."""

    p = np.asarray(matrix, dtype=np.float64)
    q = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    if p.ndim != 2:
        raise ValueError('JS distance matrix must be 2D')
    if q.shape[1] != p.shape[1]:
        raise ValueError('JS distance vectors must have the same shape')
    if p.shape[1] == 0:
        return np.zeros((p.shape[0], ), dtype=np.float64)

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    p_sums = p.sum(axis=1, keepdims=True)
    q_sum = q.sum()
    p = np.divide(
        p,
        p_sums,
        out=np.full_like(p, 1.0 / p.shape[1]),
        where=p_sums > 0.0,
    )
    if q_sum > 0.0:
        q = q / q_sum
    else:
        q = np.full_like(q, 1.0 / q.shape[1])

    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_terms = np.where(p > 0.0, p * np.log(p / m), 0.0)
        q_terms = np.where(q > 0.0, q * np.log(q / m), 0.0)
    divergence = 0.5 * p_terms.sum(axis=1) + 0.5 * q_terms.sum(axis=1)
    return np.sqrt(np.maximum(divergence, 0.0))


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
    vectors = _composition_matrix(records)
    image_ids = [record['image_id'] for record in records]
    indices_by_id: Dict[Any, List[int]] = {}
    for index, image_id in enumerate(image_ids):
        indices_by_id.setdefault(image_id, []).append(index)
    selected_ids: List[Any] = []
    selected_records: List[Dict[str, Any]] = []

    first_index = min(
        range(len(records)),
        key=lambda index: _selection_sort_key(records[index]),
    )
    first = records[first_index]
    selected_ids.append(first['image_id'])
    first_record = dict(first)
    first_record.setdefault('metadata', {})
    first_record['metadata'] = dict(first_record['metadata'])
    first_record['metadata']['nearest_selected_distance'] = None
    first_record['metadata']['selection_reason'] = 'max_eca'
    first_record['selection_rank'] = 1
    selected_records.append(first_record)

    target = min(int(budget), len(records))
    available = np.ones((len(records), ), dtype=np.bool_)
    for index in indices_by_id.get(first['image_id'], []):
        available[index] = False
    min_distances = _jensen_shannon_distance_to_vector(vectors, vectors[first_index])

    while len(selected_ids) < target:
        best_index = None
        best_distance = -1.0
        for index, record in enumerate(records):
            if not available[index]:
                continue
            distance = float(min_distances[index])
            if (
                best_index is None
                or _selection_sort_key(record, distance)
                < _selection_sort_key(records[best_index], best_distance)
            ):
                best_index = index
                best_distance = distance
        if best_index is None:
            break
        image_id = records[best_index]['image_id']
        selected_ids.append(image_id)
        selected_record = dict(by_id[image_id])
        selected_record.setdefault('metadata', {})
        selected_record['metadata'] = dict(selected_record['metadata'])
        selected_record['metadata']['nearest_selected_distance'] = float(best_distance)
        selected_record['metadata']['selection_reason'] = 'farthest_first'
        selected_record['selection_rank'] = len(selected_ids)
        selected_records.append(selected_record)
        for index in indices_by_id.get(image_id, []):
            available[index] = False
        new_distances = _jensen_shannon_distance_to_vector(vectors, vectors[best_index])
        min_distances = np.minimum(min_distances, new_distances)

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
