"""ECPAL compact feature artifact parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from methods.common.image_identity import normalize_image_id, validate_image_ids_subset
from methods.common.io import read_json
from methods.common.selection import image_id_sort_key


IMAGE_RECORD_KEYS = ('images', 'records', 'features', 'results', 'detections')
DETECTION_FEATURE_KEYS = ('p_max', 'A_cls', 'n_sup', 'mu_iou')
MISS_FEATURE_KEYS = ('R_amt', 'R_prob')


def _load_image_records(path: Path) -> List[Mapping[str, Any]]:
    payload = read_json(Path(path))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in IMAGE_RECORD_KEYS:
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise ValueError('Unsupported ECPAL feature JSON schema: %s' % path)
    else:
        raise ValueError('Unsupported ECPAL feature JSON schema: %s' % path)

    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError('ECPAL image record %d in %s must be an object' % (index, path))
        normalized.append(record)
    return normalized


def _finite_float(value: Any, field: str, path: Path) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError('ECPAL feature %s must be finite in %s' % (field, path))
    return result


def _normalize_bbox(value: Any, field: str, path: Path) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError('ECPAL %s must be a bbox sequence in %s' % (field, path))
    if len(value) < 4:
        raise ValueError('ECPAL %s must contain at least four values in %s' % (field, path))
    return [_finite_float(item, field, path) for item in value[:4]]


def _normalize_detection(det: Mapping[str, Any], image_id: Any, path: Path) -> Dict[str, Any]:
    missing = [key for key in ('bbox', 'category_id') if key not in det]
    if missing:
        raise KeyError(
            'ECPAL final detection for image %s in %s is missing keys: %s'
            % (image_id, path, ', '.join(missing)))

    record = dict(det)
    record['image_id'] = image_id
    record['bbox'] = _normalize_bbox(record['bbox'], 'bbox', path)
    record['category_id'] = normalize_image_id(record['category_id'])

    if 'p_max' not in record and 'score' in record:
        record['p_max'] = record['score']
    if 'score' not in record and 'p_max' in record:
        record['score'] = record['p_max']

    missing_features = [key for key in DETECTION_FEATURE_KEYS if key not in record]
    if missing_features:
        raise KeyError(
            'ECPAL final detection for image %s in %s is missing features: %s'
            % (image_id, path, ', '.join(missing_features)))

    for key in DETECTION_FEATURE_KEYS:
        record[key] = _finite_float(record[key], key, path)
    record['score'] = _finite_float(record.get('score', record['p_max']), 'score', path)
    return record


def _normalize_miss_features(value: Any, image_id: Any, path: Path) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError('ECPAL miss_features for image %s in %s must be an object' % (image_id, path))
    missing = [key for key in MISS_FEATURE_KEYS if key not in value]
    if missing:
        raise KeyError(
            'ECPAL miss_features for image %s in %s is missing keys: %s'
            % (image_id, path, ', '.join(missing)))
    return {key: _finite_float(value[key], key, path) for key in MISS_FEATURE_KEYS}


def normalize_feature_record(record: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    if 'image_id' not in record:
        raise KeyError('ECPAL image record in %s is missing image_id' % path)
    image_id = normalize_image_id(record['image_id'])

    detections = record.get('final_detections', record.get('detections', []))
    if detections is None:
        detections = []
    if not isinstance(detections, list):
        raise ValueError('ECPAL final_detections for image %s in %s must be a list' % (image_id, path))

    return {
        'image_id': image_id,
        'final_detections': [
            _normalize_detection(det, image_id, path)
            for det in detections
        ],
        'miss_features': _normalize_miss_features(record.get('miss_features'), image_id, path),
    }


def load_feature_records(path: Path) -> List[Dict[str, Any]]:
    """Load image-level ECPAL feature records from JSON."""

    feature_path = Path(path)
    records = [
        normalize_feature_record(record, feature_path)
        for record in _load_image_records(feature_path)
    ]
    seen = set()
    duplicates = []
    for record in records:
        image_id = record['image_id']
        if image_id in seen:
            duplicates.append(image_id)
        seen.add(image_id)
    if duplicates:
        raise ValueError(
            'ECPAL feature JSON contains duplicate image ids: %s'
            % sorted(set(duplicates), key=str)[:10])
    return sorted(records, key=lambda record: image_id_sort_key(record['image_id']))


def records_by_image(records: Iterable[Mapping[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {normalize_image_id(record['image_id']): dict(record) for record in records}


def filter_feature_records(
    records: Iterable[Mapping[str, Any]],
    image_ids: Iterable[Any],
    artifact_name: str = 'ECPAL feature records',
    require_all: bool = True,
) -> List[Dict[str, Any]]:
    """Return records for ``image_ids`` and validate image-id coverage."""

    ordered_ids = [normalize_image_id(image_id) for image_id in image_ids]
    record_map = records_by_image(records)
    validate_image_ids_subset(record_map, ordered_ids, artifact_name)
    missing = [image_id for image_id in ordered_ids if image_id not in record_map]
    if missing and require_all:
        raise ValueError(
            '%s is missing %d pool image id(s), first ids: %s'
            % (artifact_name, len(missing), missing[:10]))
    return [record_map[image_id] for image_id in ordered_ids if image_id in record_map]
