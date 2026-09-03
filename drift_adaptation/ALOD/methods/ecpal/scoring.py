"""ECPAL error-count and error-uncertainty scoring."""

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
PROFILE_KEYS = {
    'eca': ('n_cls_hat', 'n_loc_hat', 'n_miss_hat'),
    'eua': ('u_cls', 'u_loc', 'u_miss'),
}
SCORE_FIELDS = {
    'eca': 'weighted_eca',
    'eua': 'weighted_eua',
}


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


def bernoulli_entropy(probability: float) -> float:
    p = float(probability)
    if not np.isfinite(p):
        raise ValueError('Bernoulli entropy probability must be finite')
    p = min(max(p, 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def poisson_entropy(mean: float) -> float:
    value = float(mean)
    if not np.isfinite(value):
        raise ValueError('Poisson entropy mean must be finite')
    if value <= 0.0:
        return 0.0
    try:
        from scipy.stats import poisson
    except ImportError as exc:
        raise ImportError(
            'scipy is required for ECPAL EUA Poisson entropy. '
            'Install it with `pip install -r requirements.txt`.'
        ) from exc
    entropy = float(poisson.entropy(value))
    if not np.isfinite(entropy):
        raise ValueError('Poisson entropy produced a non-finite value')
    return entropy


def predict_eca_profile(record: Mapping[str, Any], predictors: ECPALPredictors) -> Dict[str, float]:
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


def predict_eua_profile(record: Mapping[str, Any], predictors: ECPALPredictors) -> Dict[str, float]:
    u_cls = 0.0
    u_loc = 0.0
    for detection in record.get('final_detections', []) or []:
        q_fg = predictors.fdp.predict_probability(common_feature_vector(detection))
        q_cls_given_fg = predictors.cecp.predict_probability(classification_feature_vector(detection))
        q_loc_given_fg = predictors.lecp.predict_probability(localization_feature_vector(detection))
        u_cls += q_fg * bernoulli_entropy(q_cls_given_fg)
        u_loc += q_fg * bernoulli_entropy(q_loc_given_fg)

    n_miss = predictors.mocp.predict_count(miss_feature_vector(record))
    return {
        'u_cls': float(u_cls),
        'u_loc': float(u_loc),
        'u_miss': float(poisson_entropy(n_miss)),
    }


def mean_eua_profile(
    records: Iterable[Mapping[str, Any]],
    predictors: ECPALPredictors,
) -> Dict[str, float]:
    profiles = [predict_eua_profile(record, predictors) for record in records]
    if not profiles:
        return {'cls': 0.0, 'loc': 0.0, 'miss': 0.0}
    return {
        'cls': float(np.mean([profile['u_cls'] for profile in profiles])),
        'loc': float(np.mean([profile['u_loc'] for profile in profiles])),
        'miss': float(np.mean([profile['u_miss'] for profile in profiles])),
    }


def _profile_keys(score_mode: str) -> tuple:
    normalized = score_mode.lower()
    if normalized not in PROFILE_KEYS:
        raise ValueError('Unsupported ECPAL score mode: %s' % score_mode)
    return PROFILE_KEYS[normalized]


def weighted_profile(
    profile: Mapping[str, float],
    weights: Mapping[str, float],
    score_mode: str = 'eca',
) -> Dict[str, float]:
    cls_key, loc_key, miss_key = _profile_keys(score_mode)
    return {
        'cls': float(weights.get('cls', 1.0)) * float(profile.get(cls_key, 0.0)),
        'loc': float(weights.get('loc', 1.0)) * float(profile.get(loc_key, 0.0)),
        'miss': float(weights.get('miss', 1.0)) * float(profile.get(miss_key, 0.0)),
    }


def weighted_amount(weighted: Mapping[str, float]) -> float:
    return float(sum(max(float(weighted.get(key, 0.0)), 0.0) for key in ERROR_KEYS))


def weighted_eca(weighted: Mapping[str, float]) -> float:
    return weighted_amount(weighted)


def weighted_composition(
    weighted: Mapping[str, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    denominator = weighted_amount(weighted) + 3.0 * float(eps)
    return {
        key: float((max(float(weighted.get(key, 0.0)), 0.0) + float(eps)) / denominator)
        for key in ERROR_KEYS
    }


def score_unlabeled_records(
    records: Iterable[Mapping[str, Any]],
    predictors: ECPALPredictors,
    weights: Mapping[str, float],
    score_mode: str = 'eca',
    eps: float = 1e-12,
) -> List[Dict[str, Any]]:
    normalized_mode = score_mode.lower()
    if normalized_mode == 'eca':
        profile_fn = predict_eca_profile
    elif normalized_mode == 'eua':
        profile_fn = predict_eua_profile
    else:
        raise ValueError('Unsupported ECPAL score mode: %s' % score_mode)

    scored = []
    for record in records:
        profile = profile_fn(record, predictors)
        weighted = weighted_profile(profile, weights, score_mode=normalized_mode)
        amount = weighted_amount(weighted)
        composition = weighted_composition(weighted, eps=eps)
        scored.append({
            'image_id': record['image_id'],
            'score': amount,
            'score_mode': normalized_mode,
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
    source: str = 'eca-full',
) -> List[Dict[str, Any]]:
    ranked = sorted(scored_records, key=lambda item: (-float(item['score']), image_id_sort_key(item['image_id'])))
    limit = candidate_limit(budget, candidate_expand_ratio, len(ranked))
    candidates = []
    for rank, item in enumerate(ranked[:limit], start=1):
        profile = item['profile']
        composition = item['composition']
        score_mode = str(item.get('score_mode', 'eca')).lower()
        cls_key, loc_key, miss_key = _profile_keys(score_mode)
        components = {
            cls_key: float(profile[cls_key]),
            loc_key: float(profile[loc_key]),
            miss_key: float(profile[miss_key]),
            SCORE_FIELDS[score_mode]: float(item['score']),
            'pi_cls': float(composition['cls']),
            'pi_loc': float(composition['loc']),
            'pi_miss': float(composition['miss']),
        }
        candidates.append({
            'image_id': item['image_id'],
            'rank': rank,
            'score': float(item['score']),
            'score_mode': score_mode,
            'source': source,
            'components': components,
            'metadata': {
                'candidate_rank': rank,
                'final_detection_count': int(item.get('final_detection_count', 0)),
            },
        })
    return candidates
