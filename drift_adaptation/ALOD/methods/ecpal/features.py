"""Feature vector helpers for compact ECPAL artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


COMMON_FEATURE_KEYS = ('p_max', 'A_cls', 'n_sup', 'mu_iou')
CLASSIFICATION_FEATURE_KEYS = ('p_max', 'A_cls')
LOCALIZATION_FEATURE_KEYS = ('n_sup', 'mu_iou')
MISS_FEATURE_KEYS = ('R_amt', 'R_prob')


def _feature_vector(record: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    values = [float(record.get(key, 0.0)) for key in keys]
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError('ECPAL feature vector contains non-finite values')
    return vector


def common_feature_vector(detection: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(detection, COMMON_FEATURE_KEYS)


def classification_feature_vector(detection: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(detection, CLASSIFICATION_FEATURE_KEYS)


def localization_feature_vector(detection: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(detection, LOCALIZATION_FEATURE_KEYS)


def miss_feature_vector(image_record: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(image_record.get('miss_features', {}), MISS_FEATURE_KEYS)
