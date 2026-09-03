"""ECPAL predictor fitting built on common AL predictor wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from methods.common.predictors import BinaryLogisticModel, PoissonCountModel


@dataclass
class ECPALPredictors:
    fdp: BinaryLogisticModel
    cecp: BinaryLogisticModel
    lecp: BinaryLogisticModel
    mocp: PoissonCountModel
    diagnostics: Dict[str, Any]


def _positive_rate(targets: Iterable[int]) -> float:
    values = [int(value) for value in targets]
    return float(sum(values) / len(values)) if values else 0.0


def fit_predictors(label_data: Dict[str, Any], min_samples: int = 4) -> ECPALPredictors:
    detections = list(label_data.get('detection_examples', []))
    images = list(label_data.get('image_examples', []))
    foreground = [example for example in detections if example.get('is_foreground')]

    fdp_features = [example['common_feature'] for example in detections]
    fdp_targets = [example['y_fg'] for example in detections]
    cecp_features = [example['classification_feature'] for example in foreground]
    cecp_targets = [example['y_cls'] for example in foreground]
    lecp_features = [example['localization_feature'] for example in foreground]
    lecp_targets = [example['y_loc'] for example in foreground]
    mocp_features = [example['miss_feature'] for example in images]
    mocp_targets = [example['n_miss'] for example in images]

    fdp = BinaryLogisticModel.fit(
        fdp_features, fdp_targets, min_samples=min_samples, n_features=4)
    cecp = BinaryLogisticModel.fit(
        cecp_features, cecp_targets, min_samples=min_samples, n_features=2)
    lecp = BinaryLogisticModel.fit(
        lecp_features, lecp_targets, min_samples=min_samples, n_features=2)
    mocp = PoissonCountModel.fit(
        mocp_features, mocp_targets, min_samples=min_samples, n_features=2)

    diagnostics = {
        'sample_counts': {
            'fdp': len(fdp_targets),
            'cecp': len(cecp_targets),
            'lecp': len(lecp_targets),
            'mocp': len(mocp_targets),
        },
        'positive_rates': {
            'fdp': _positive_rate(fdp_targets),
            'cecp': _positive_rate(cecp_targets),
            'lecp': _positive_rate(lecp_targets),
        },
        'fallbacks': {
            'fdp': fdp.fallback_reason,
            'cecp': cecp.fallback_reason,
            'lecp': lecp.fallback_reason,
            'mocp': mocp.fallback_reason,
        },
    }
    return ECPALPredictors(fdp=fdp, cecp=cecp, lecp=lecp, mocp=mocp, diagnostics=diagnostics)
