"""Common candidate artifact helpers.

This module owns candidate formatting and persistence only. Method-specific
candidate scoring and selection stays in the individual method modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from methods.common.image_identity import normalize_image_id
from methods.common.io import write_json
from methods.common.selection import image_id_sort_key, unique_in_order


CandidateRecord = Dict[str, Any]


def candidate_image_ids(records: Iterable[Mapping[str, Any]]) -> List[Any]:
    """Return candidate image ids in record order."""

    ids = []
    for record in records:
        if record.get('image_id') is not None:
            ids.append(normalize_image_id(record['image_id']))
    return ids


def unique_candidate_image_ids(records: Iterable[Mapping[str, Any]]) -> List[Any]:
    """Return unique candidate image ids in first-seen order."""

    return unique_in_order(candidate_image_ids(records))


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _selected_rank_map(selected_image_ids: Iterable[Any]) -> Dict[Any, int]:
    selected = unique_in_order(
        normalize_image_id(image_id)
        for image_id in selected_image_ids
    )
    return {image_id: index for index, image_id in enumerate(selected, start=1)}


def normalize_candidate_record(
    record: Mapping[str, Any],
    *,
    source: str,
    score_key: Optional[str] = None,
    component_keys: Sequence[str] = (),
    metadata_keys: Sequence[str] = (),
) -> CandidateRecord:
    """Convert one method record into the common candidate schema."""

    if record.get('image_id') is None:
        raise KeyError('Candidate record is missing image_id')

    score = record.get(score_key) if score_key else record.get('score')
    components = {
        key: record[key]
        for key in component_keys
        if key in record
    }
    metadata = {
        key: record[key]
        for key in metadata_keys
        if key in record
    }

    normalized: CandidateRecord = {
        'image_id': normalize_image_id(record['image_id']),
        'rank': record.get('rank'),
        'score': _maybe_float(score),
        'source': str(record.get('source', source)),
        'selected': bool(record.get('selected', False)),
        'selection_rank': record.get('selection_rank'),
        'category_id': (
            normalize_image_id(record['category_id'])
            if record.get('category_id') is not None else None
        ),
        'components': components,
        'metadata': metadata,
    }
    return normalized


def rank_candidate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    score_key: str = 'score',
    descending: bool = True,
) -> List[CandidateRecord]:
    """Return records sorted by score with deterministic 1-based ranks."""

    def sort_key(item):
        index, record = item
        score = record.get(score_key)
        missing_score = score is None
        value = float(score) if score is not None else 0.0
        ordered_value = -value if descending else value
        return (
            missing_score,
            ordered_value,
            image_id_sort_key(record.get('image_id')),
            index,
        )

    ranked = []
    for rank, (_, record) in enumerate(
            sorted(enumerate(records), key=sort_key),
            start=1):
        item = dict(record)
        item['rank'] = rank
        ranked.append(item)
    return ranked


def build_candidate_artifact(
    *,
    method: str,
    stage: str,
    round_index: int,
    budget: int,
    candidates: Iterable[Mapping[str, Any]],
    selected_image_ids: Iterable[Any] = (),
    candidate_pool_json: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a compact common candidate artifact payload."""

    selected_ranks = _selected_rank_map(selected_image_ids)
    normalized = []
    for index, record in enumerate(candidates, start=1):
        item = dict(record)
        if item.get('image_id') is None:
            raise KeyError('Candidate record is missing image_id')
        image_id = normalize_image_id(item['image_id'])
        item['image_id'] = image_id
        item.setdefault('rank', index)
        item.setdefault('source', stage)
        item.setdefault('components', {})
        item.setdefault('metadata', {})
        if item.get('category_id') is not None:
            item['category_id'] = normalize_image_id(item['category_id'])
        item['selected'] = image_id in selected_ranks
        item['selection_rank'] = selected_ranks.get(image_id)
        normalized.append(item)

    artifact = {
        'method': method,
        'stage': stage,
        'round_index': int(round_index),
        'budget': int(budget),
        'candidate_count': len(normalized),
        'unique_image_count': len(unique_candidate_image_ids(normalized)),
        'selected_count': len(selected_ranks),
        'candidates': normalized,
    }
    if candidate_pool_json:
        artifact['candidate_pool_json'] = str(candidate_pool_json)
    return artifact


def write_candidate_artifact(path: Any, artifact: Mapping[str, Any]) -> Path:
    """Write a candidate artifact JSON."""

    return write_json(Path(path), artifact, indent=2)
