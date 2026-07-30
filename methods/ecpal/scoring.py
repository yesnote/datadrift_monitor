"""ECPAL error-count scoring and candidate ranking."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from methods.common.selection import image_id_sort_key
from methods.ecpal.features import (
    classification_feature_vector,
    common_feature_vector,
    localization_feature_vector,
    miss_feature_vector,
)
from methods.ecpal.predictors import ECPALPredictors


ERROR_KEYS = ('cls', 'loc', 'miss')


def actual_mean_counts(image_examples: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    examples = list(image_examples)
    if not examples:
        return {'cls': 0.0, 'loc': 0.0, 'miss': 0.0}
    return {
        'cls': float(np.mean([float(example.get('n_cls', 0.0)) for example in examples])),
        'loc': float(np.mean([float(example.get('n_loc', 0.0)) for example in examples])),
        'miss': float(np.mean([float(example.get('n_miss', 0.0)) for example in examples])),
    }


def inverse_scale_weights(mean_counts: Mapping[str, float], weight_eps: float = 1e-6) -> Dict[str, float]:
    raw = {
        key: 1.0 / (max(float(mean_counts.get(key, 0.0)), 0.0) + float(weight_eps))
        for key in ERROR_KEYS
    }
    total = sum(raw.values())
    if total <= 0.0 or not np.isfinite(total):
        return {key: 1.0 for key in ERROR_KEYS}
    return {key: float(3.0 * raw[key] / total) for key in ERROR_KEYS}


def predict_error_profile(record: Mapping[str, Any], predictors: ECPALPredictors) -> Dict[str, float]:
    n_cls = 0.0
    n_loc = 0.0
    for detection in record.get('final_detections', []) or []:
        q_fg = predictors.fdp.predict_probability(common_feature_vector(detection))
        q_cls_given_fg = predictors.cecp.predict_probability(classification_feature_vector(detection))
        q_loc_given_fg = predictors.lecp.predict_probability(localization_feature_vector(detection))
        n_cls += q_fg * q_cls_given_fg
        n_loc += q_fg * q_loc_given_fg

    n_miss = predictors.mocp.predict_count(miss_feature_vector(record))
    return {
        'n_cls_hat': float(n_cls),
        'n_loc_hat': float(n_loc),
        'n_miss_hat': float(n_miss),
    }


def weighted_profile(
    profile: Mapping[str, float],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    return {
        'cls': float(weights.get('cls', 1.0)) * float(profile.get('n_cls_hat', 0.0)),
        'loc': float(weights.get('loc', 1.0)) * float(profile.get('n_loc_hat', 0.0)),
        'miss': float(weights.get('miss', 1.0)) * float(profile.get('n_miss_hat', 0.0)),
    }


def weighted_eca(weighted: Mapping[str, float]) -> float:
    return float(sum(max(float(weighted.get(key, 0.0)), 0.0) for key in ERROR_KEYS))


def weighted_composition(
    weighted: Mapping[str, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    denominator = weighted_eca(weighted) + 3.0 * float(eps)
    return {
        key: float((max(float(weighted.get(key, 0.0)), 0.0) + float(eps)) / denominator)
        for key in ERROR_KEYS
    }


def score_unlabeled_records(
    records: Iterable[Mapping[str, Any]],
    predictors: ECPALPredictors,
    weights: Mapping[str, float],
    eps: float = 1e-12,
) -> List[Dict[str, Any]]:
    scored = []
    for record in records:
        profile = predict_error_profile(record, predictors)
        weighted = weighted_profile(profile, weights)
        eca = weighted_eca(weighted)
        composition = weighted_composition(weighted, eps=eps)
        scored.append({
            'image_id': record['image_id'],
            'score': eca,
            'profile': profile,
            'weighted_profile': weighted,
            'composition': composition,
            'final_detection_count': len(record.get('final_detections', []) or []),
        })
    return sorted(scored, key=lambda item: (-float(item['score']), image_id_sort_key(item['image_id'])))


def candidate_limit(budget: int, candidate_expand_ratio: float, pool_size: int) -> int:
    if budget <= 0 or pool_size <= 0:
        return 0
    limit = int(math.ceil(float(candidate_expand_ratio) * int(budget)))
    return max(0, min(limit, int(pool_size)))


def build_candidate_records(
    scored_records: Iterable[Mapping[str, Any]],
    budget: int,
    candidate_expand_ratio: float = 2.0,
) -> List[Dict[str, Any]]:
    ranked = sorted(scored_records, key=lambda item: (-float(item['score']), image_id_sort_key(item['image_id'])))
    limit = candidate_limit(budget, candidate_expand_ratio, len(ranked))
    candidates = []
    for rank, item in enumerate(ranked[:limit], start=1):
        profile = item['profile']
        composition = item['composition']
        components = {
            'n_cls_hat': float(profile['n_cls_hat']),
            'n_loc_hat': float(profile['n_loc_hat']),
            'n_miss_hat': float(profile['n_miss_hat']),
            'weighted_eca': float(item['score']),
            'pi_cls': float(composition['cls']),
            'pi_loc': float(composition['loc']),
            'pi_miss': float(composition['miss']),
        }
        candidates.append({
            'image_id': item['image_id'],
            'rank': rank,
            'score': float(item['score']),
            'source': 'ecd',
            'components': components,
            'metadata': {
                'candidate_rank': rank,
                'final_detection_count': int(item.get('final_detection_count', 0)),
            },
        })
    return candidates
