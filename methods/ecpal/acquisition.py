"""ECPAL acquisition orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from methods.common.coco_pool import image_ids, read_coco_json
from methods.common.selection import fill_to_budget
from methods.ecpal.diversity import attach_selection_metadata, farthest_first_select
from methods.ecpal.inference import filter_feature_records, load_feature_records
from methods.ecpal.labels import build_training_labels
from methods.ecpal.predictors import fit_predictors
from methods.ecpal.scoring import (
    actual_mean_counts,
    build_candidate_records,
    inverse_scale_weights,
    score_unlabeled_records,
)


def select_ecpal_images(
    labeled_pool: Mapping[str, Any],
    unlabeled_pool: Mapping[str, Any],
    labeled_features: List[Mapping[str, Any]],
    unlabeled_features: List[Mapping[str, Any]],
    budget: int,
    candidate_expand_ratio: float = 2.0,
    foreground_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.1,
    eps: float = 1e-12,
    weight_eps: float = 1e-6,
    seed: int = 0,
) -> Dict[str, Any]:
    labeled_ids = image_ids(dict(labeled_pool))
    unlabeled_ids = image_ids(dict(unlabeled_pool))
    labeled_records = filter_feature_records(
        labeled_features,
        labeled_ids,
        artifact_name='ECPAL labeled features',
        require_all=True,
    )
    unlabeled_records = filter_feature_records(
        unlabeled_features,
        unlabeled_ids,
        artifact_name='ECPAL unlabeled features',
        require_all=True,
    )

    label_data = build_training_labels(
        labeled_records,
        labeled_pool,
        foreground_iou_threshold=foreground_iou_threshold,
        background_iou_threshold=background_iou_threshold,
    )
    predictors = fit_predictors(label_data)
    mean_counts = actual_mean_counts(label_data.get('image_examples', []))
    weights = inverse_scale_weights(mean_counts, weight_eps=weight_eps)
    scored = score_unlabeled_records(
        unlabeled_records,
        predictors,
        weights,
        eps=eps,
    )
    candidates = build_candidate_records(
        scored,
        budget=budget,
        candidate_expand_ratio=candidate_expand_ratio,
    )
    selection = farthest_first_select(candidates, budget=budget)
    selected = list(selection['selected_image_ids'])
    if len(selected) < min(int(budget), len(unlabeled_ids)):
        selected = fill_to_budget(selected, unlabeled_ids, budget, seed=seed)

    candidate_records = attach_selection_metadata(candidates, selection['selected_records'])
    diagnostics = {
        'mode': 'ecpal',
        'stage': 'ecd',
        'budget': int(budget),
        'candidate_expand_ratio': float(candidate_expand_ratio),
        'thresholds': {
            'foreground_iou': float(foreground_iou_threshold),
            'background_iou': float(background_iou_threshold),
        },
        'eps': float(eps),
        'weight_eps': float(weight_eps),
        'pool_counts': {
            'labeled_images': len(labeled_ids),
            'unlabeled_images': len(unlabeled_ids),
            'labeled_feature_records': len(labeled_records),
            'unlabeled_feature_records': len(unlabeled_records),
        },
        'label_summary': label_data.get('summary', {}),
        'predictors': predictors.diagnostics,
        'mean_labeled_true_counts': mean_counts,
        'scale_weights': weights,
        'scored_image_count': len(scored),
        'candidate_count': len(candidates),
        'selected_count': len(selected[:budget]),
        'selected_image_ids': selected[:budget],
    }
    return {
        'selected_image_ids': selected[:budget],
        'candidate_records': candidate_records,
        'diagnostics': diagnostics,
        'mode': 'ecpal',
        'stage': 'ecd',
    }


def sample_ecpal_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_features_json: Path,
    unlabeled_features_json: Path,
    budget: int,
    candidate_expand_ratio: float = 2.0,
    foreground_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.1,
    eps: float = 1e-12,
    weight_eps: float = 1e-6,
    seed: int = 0,
    oracle_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run ECPAL acquisition from compact feature artifacts.

    ``oracle_json`` is accepted for runner symmetry with other methods but is
    intentionally unused here; pool updates are handled by common runner code.
    """

    del oracle_json
    labeled_pool = read_coco_json(Path(labeled_pool_json))
    unlabeled_pool = read_coco_json(Path(unlabeled_pool_json))
    labeled_features = load_feature_records(Path(labeled_features_json))
    unlabeled_features = load_feature_records(Path(unlabeled_features_json))
    result = select_ecpal_images(
        labeled_pool=labeled_pool,
        unlabeled_pool=unlabeled_pool,
        labeled_features=labeled_features,
        unlabeled_features=unlabeled_features,
        budget=budget,
        candidate_expand_ratio=candidate_expand_ratio,
        foreground_iou_threshold=foreground_iou_threshold,
        background_iou_threshold=background_iou_threshold,
        eps=eps,
        weight_eps=weight_eps,
        seed=seed,
    )
    result['diagnostics']['inputs'] = {
        'labeled_pool_json': str(labeled_pool_json),
        'unlabeled_pool_json': str(unlabeled_pool_json),
        'labeled_features_json': str(labeled_features_json),
        'unlabeled_features_json': str(unlabeled_features_json),
    }
    return result
