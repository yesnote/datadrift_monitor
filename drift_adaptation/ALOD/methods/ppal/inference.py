"""PPAL detector artifact loading.

PPAL method code consumes artifacts produced by the local MMDetection backend:

* ``RetinaHeadUncertainty`` writes ``*.bbox.json`` records with
  ``cls_uncertainty`` for DCUS.
* ``RetinaQualityEMAHead`` stores ``bbox_head.class_quality`` in
  ``latest.pth`` for DCUS class reweighting.
* ``RetinaHeadFeat`` writes ``image_dis.npy`` for CCMS diversity selection.

The loaders here keep that backend boundary explicit so ``dcus.py`` and
``ccms.py`` can focus on acquisition logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


UNCERTAINTY_RESULT_KEYS = ('image_id', 'category_id', 'bbox', 'score', 'cls_uncertainty')
CLASS_QUALITY_KEY = 'bbox_head.class_quality'


def _as_path(path: Any) -> Path:
    return path if isinstance(path, Path) else Path(path)


def class_quality_checkpoint_path(result_json: Any) -> Path:
    """Return the checkpoint path paired with a PPAL uncertainty result JSON."""

    return _as_path(result_json).parent / 'latest.pth'


def load_uncertainty_detections(path: Any) -> List[Dict[str, Any]]:
    """Load and validate PPAL uncertainty detections."""

    result_path = _as_path(path)
    with result_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in ('detections', 'annotations', 'results'):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        if records is None:
            raise ValueError('Unsupported PPAL uncertainty JSON schema: %s' % result_path)
    else:
        raise ValueError('Unsupported PPAL uncertainty JSON schema: %s' % result_path)

    normalized = []
    for index, record in enumerate(records):
        missing = [key for key in UNCERTAINTY_RESULT_KEYS if key not in record]
        if missing:
            raise KeyError(
                'PPAL uncertainty record %d in %s is missing keys: %s'
                % (index, result_path, ', '.join(missing)))
        normalized.append(dict(record))
    return normalized


def load_class_quality(checkpoint_path: Any) -> np.ndarray:
    """Load ``bbox_head.class_quality`` from a PPAL detector checkpoint."""

    ckpt_path = _as_path(checkpoint_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PPAL class quality loading requires PyTorch because %s is a '
            'MMDetection checkpoint.' % ckpt_path) from exc

    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict')
    if not isinstance(state_dict, dict) or CLASS_QUALITY_KEY not in state_dict:
        raise KeyError('Checkpoint is missing state_dict[%r]: %s' % (CLASS_QUALITY_KEY, ckpt_path))

    value = state_dict[CLASS_QUALITY_KEY]
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    qualities = np.asarray(value, dtype=np.float64).reshape(-1)
    if qualities.size == 0:
        raise ValueError('PPAL class quality is empty: %s' % ckpt_path)
    if not np.all(np.isfinite(qualities)):
        raise ValueError('PPAL class quality contains non-finite values: %s' % ckpt_path)
    return qualities


def load_image_distance_cache(path: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Load the ``image_dis.npy`` matrix and its image id vector."""

    cache_path = _as_path(path)
    with cache_path.open('rb') as handle:
        distance_matrix = np.load(handle)
        image_ids = np.load(handle).reshape(-1)

    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError('PPAL image distance cache must be a square matrix: %s' % cache_path)
    if distance_matrix.shape[0] != image_ids.shape[0]:
        raise ValueError(
            'PPAL image distance cache has %d matrix rows but %d image ids: %s'
            % (distance_matrix.shape[0], image_ids.shape[0], cache_path))
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError('PPAL image distance cache contains non-finite values: %s' % cache_path)
    return distance_matrix, image_ids
