"""Method-neutral greedy k-center selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from methods.common.vectors import (
    l2_normalize_rows,
    squared_euclidean_to_centers,
    squared_euclidean_to_vector,
)


def _as_feature_matrix(features: Any, name: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError('%s must be a 2D feature matrix' % name)
    if not np.all(np.isfinite(values)):
        raise ValueError('%s contains non-finite values' % name)
    return values


def greedy_k_center_select(
    candidate_image_ids: Sequence[Any],
    candidate_features: Any,
    budget: int,
    center_features: Optional[Any] = None,
    normalize: bool = False,
    batch_size: int = 512,
    center_batch_size: int = 2048,
) -> Dict[str, Any]:
    """Select candidates by farthest-first k-center greedy.

    ``center_features`` represents already labeled images and any selected
    images are added to this center set after each greedy step.
    """

    candidate_ids = list(candidate_image_ids)
    features = _as_feature_matrix(candidate_features, 'candidate_features')
    if len(candidate_ids) != features.shape[0]:
        raise ValueError(
            'candidate_image_ids has %d ids but candidate_features has %d rows'
            % (len(candidate_ids), features.shape[0]))

    if normalize:
        features = l2_normalize_rows(features)

    centers = None
    if center_features is not None:
        centers = _as_feature_matrix(center_features, 'center_features')
        if centers.shape[1] != features.shape[1]:
            raise ValueError(
                'center feature dim %d does not match candidate feature dim %d'
                % (centers.shape[1], features.shape[1]))
        if normalize:
            centers = l2_normalize_rows(centers)

    target = min(int(budget), len(candidate_ids))
    if target <= 0:
        return {
            'selected_image_ids': [],
            'selected_records': [],
            'candidate_records': [],
            'initial_min_distances': [],
            'final_min_distances': [],
        }

    if centers is None or centers.shape[0] == 0:
        min_distances = np.full((features.shape[0], ), np.inf, dtype=np.float64)
    else:
        min_distances = squared_euclidean_to_centers(
            features,
            centers,
            batch_size=batch_size,
            center_batch_size=center_batch_size,
        )
    initial_min_distances = min_distances.copy()

    selected_indices: List[int] = []
    selected_records: List[Dict[str, Any]] = []
    available = np.ones((features.shape[0], ), dtype=np.bool_)
    for rank in range(1, target + 1):
        masked = np.where(available, min_distances, -np.inf)
        selected_index = int(np.argmax(masked))
        if not np.isfinite(masked[selected_index]):
            remaining = np.flatnonzero(available)
            if remaining.size == 0:
                break
            selected_index = int(remaining[0])
        selected_indices.append(selected_index)
        available[selected_index] = False
        selected_records.append({
            'image_id': candidate_ids[selected_index],
            'selection_rank': rank,
            'selection_distance': float(min_distances[selected_index])
            if np.isfinite(min_distances[selected_index]) else None,
        })
        new_distances = squared_euclidean_to_vector(features, features[selected_index])
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_indices] = -np.inf

    selected_set = set(selected_indices)
    selected_rank = {
        index: rank for rank, index in enumerate(selected_indices, start=1)
    }
    candidate_records = []
    final_distances = min_distances.copy()
    for index, image_id in enumerate(candidate_ids):
        is_selected = index in selected_set
        candidate_records.append({
            'image_id': image_id,
            'rank': index + 1,
            'score': float(initial_min_distances[index])
            if np.isfinite(initial_min_distances[index]) else None,
            'source': 'kcenter',
            'components': {
                'initial_min_distance': float(initial_min_distances[index])
                if np.isfinite(initial_min_distances[index]) else None,
                'final_min_distance': float(final_distances[index])
                if np.isfinite(final_distances[index]) else None,
            },
            'metadata': {
                'selected_by_kcenter': is_selected,
                'kcenter_rank': selected_rank.get(index),
            },
        })

    return {
        'selected_image_ids': [candidate_ids[index] for index in selected_indices],
        'selected_records': selected_records,
        'candidate_records': candidate_records,
        'initial_min_distances': initial_min_distances.tolist(),
        'final_min_distances': final_distances.tolist(),
    }
