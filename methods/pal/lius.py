"""PAL LIUS scoring.

This implements the paper's class-wise logistic uncertainty idea with a small
NumPy logistic model. GUIDE is intentionally separate so LIUS can be validated
first.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from methods.pal.inference import lius_feature


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class BinaryLogisticModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    constant_probability: Optional[float] = None

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        learning_rate: float = 0.1,
        max_iter: int = 300,
        l2: float = 1e-4,
        min_samples: int = 4,
    ) -> 'BinaryLogisticModel':
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            return cls.constant(0.5, n_features=2)
        if x.shape[0] < min_samples or len(set(y.astype(int).tolist())) < 2:
            probability = float((y.sum() + 1.0) / (len(y) + 2.0))
            return cls.constant(probability, n_features=x.shape[1])

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        x_norm = (x - mean) / scale

        weights = np.zeros(x_norm.shape[1], dtype=np.float64)
        positive_rate = min(max(float(y.mean()), 1e-6), 1.0 - 1e-6)
        bias = math.log(positive_rate / (1.0 - positive_rate))

        for _ in range(max_iter):
            probs = _sigmoid(np.matmul(x_norm, weights) + bias)
            error = probs - y
            grad_w = np.matmul(x_norm.T, error) / len(y) + l2 * weights
            grad_b = float(error.mean())
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        return cls(weights=weights, bias=float(bias), mean=mean, scale=scale)

    @classmethod
    def constant(cls, probability: float, n_features: int = 2) -> 'BinaryLogisticModel':
        return cls(
            weights=np.zeros(n_features, dtype=np.float64),
            bias=0.0,
            mean=np.zeros(n_features, dtype=np.float64),
            scale=np.ones(n_features, dtype=np.float64),
            constant_probability=float(min(max(probability, 1e-6), 1.0 - 1e-6)),
        )

    def predict_probability(self, feature: Sequence[float]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        x = np.asarray(feature, dtype=np.float64)
        x_norm = (x - self.mean) / self.scale
        return float(_sigmoid(np.asarray([np.dot(x_norm, self.weights) + self.bias]))[0])


def train_classwise_models(
    matched_detections: Iterable[Dict[str, Any]],
    min_samples: int = 4,
) -> Dict[Any, BinaryLogisticModel]:
    features_by_class: Dict[Any, List[Sequence[float]]] = defaultdict(list)
    targets_by_class: Dict[Any, List[int]] = defaultdict(list)
    for det in matched_detections:
        category_id = det.get('category_id')
        if category_id is None:
            continue
        features_by_class[category_id].append(lius_feature(det))
        targets_by_class[category_id].append(int(det.get('target', 0)))

    models = {}
    for category_id, features in features_by_class.items():
        models[category_id] = BinaryLogisticModel.fit(
            features,
            targets_by_class[category_id],
            min_samples=min_samples,
        )
    return models


def score_unlabeled_detections(
    detections: Iterable[Dict[str, Any]],
    models: Dict[Any, BinaryLogisticModel],
) -> List[Dict[str, Any]]:
    scored = []
    for det in detections:
        record = dict(det)
        model = models.get(record.get('category_id'))
        probability = model.predict_probability(lius_feature(record)) if model else 0.5
        record['tp_probability'] = probability
        record['lius_score'] = binary_entropy(probability)
        scored.append(record)
    return scored
