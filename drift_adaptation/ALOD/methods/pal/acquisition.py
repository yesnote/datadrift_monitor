"""PAL acquisition orchestration for LIUS-only and full PAL."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from methods.common.coco_pool import image_ids, read_coco_json
from methods.common.matching import match_detections_to_ground_truth
from methods.common.selection import deterministic_random_sample
from methods.pal.embeddings import build_image_embeddings, read_embedding_cache
from methods.pal.guide import (
    allocate_class_budgets as allocate_guide_class_budgets,
    build_class_candidates,
    compute_class_weights as compute_guide_class_weights,
    group_detections_by_image,
    score_guide_candidates,
    select_deduplicated_candidates,
)
from methods.pal.inference import (
    detection_confidence,
    detection_pre_nms_count,
    filter_pool_detections,
    load_detection_records,
)
from methods.pal.lius import score_unlabeled_detections, train_classwise_models


__all__ = [
    'allocate_class_budgets',
    'compute_class_weights',
    'sample_lius_only_from_files',
    'sample_pal_from_files',
    'select_full_pal_images',
    'select_lius_images',
]


def _sort_image_id(image_id: Any) -> tuple:
    if isinstance(image_id, int):
        return (0, image_id)
    return (1, str(image_id))


def _category_counts_from_annotations(annotations: Iterable[Dict[str, Any]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = defaultdict(int)
    for ann in annotations:
        if ann.get('category_id') is not None:
            counts[ann['category_id']] += 1
    return counts


def _category_counts_from_detections(detections: Iterable[Dict[str, Any]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = defaultdict(int)
    for det in detections:
        if det.get('category_id') is not None:
            counts[det['category_id']] += 1
    return counts


def compute_class_weights(
    labeled_counts: Dict[Any, int],
    unlabeled_counts: Dict[Any, int],
    categories: Iterable[Any],
) -> Dict[Any, float]:
    """Compute PAL rarity weights from labelled and unlabelled class counts."""

    total_labeled = float(sum(labeled_counts.values()))
    total_unlabeled = float(sum(unlabeled_counts.values()))
    weights = {}
    for category_id in categories:
        labeled_ratio = labeled_counts.get(category_id, 0) / total_labeled if total_labeled else 0.0
        unlabeled_ratio = unlabeled_counts.get(category_id, 0) / total_unlabeled if total_unlabeled else 0.0
        weights[category_id] = max(1.0 - 0.5 * (labeled_ratio + unlabeled_ratio), 0.0)
    return weights


def _candidate_image_scores_by_class(
    scored_detections: Iterable[Dict[str, Any]],
) -> Dict[Any, Dict[Any, float]]:
    scores: Dict[Any, Dict[Any, float]] = defaultdict(dict)
    for det in scored_detections:
        category_id = det.get('category_id')
        image_id = det.get('image_id')
        if category_id is None or image_id is None:
            continue
        score = float(det.get('lius_score', 0.0))
        current = scores[category_id].get(image_id)
        if current is None or score > current:
            scores[category_id][image_id] = score
    return scores


def _candidate_records_by_class(
    scored_detections: Iterable[Dict[str, Any]],
) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
    candidates: Dict[Any, Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    for det in scored_detections:
        category_id = det.get('category_id')
        image_id = det.get('image_id')
        if category_id is None or image_id is None:
            continue
        record = {
            'category_id': category_id,
            'image_id': image_id,
            'lius_score': float(det.get('lius_score', 0.0)),
            'tp_probability': float(det.get('tp_probability', 0.5)),
            'score': detection_confidence(det),
            'pre_nms_count': detection_pre_nms_count(det),
        }
        current = candidates[category_id].get(image_id)
        if current is None or (
                record['lius_score'], record['score']) > (
                    current['lius_score'], current['score']):
            candidates[category_id][image_id] = record
    return candidates


def allocate_class_budgets(
    budget: int,
    labeled_counts: Dict[Any, int],
    unlabeled_counts: Dict[Any, int],
    class_capacities: Dict[Any, int],
) -> Dict[Any, int]:
    """Allocate PAL class budgets with largest-remainder rounding."""

    if budget <= 0:
        return {category_id: 0 for category_id in class_capacities}

    categories = sorted(class_capacities, key=lambda item: str(item))
    rarity = compute_class_weights(labeled_counts, unlabeled_counts, categories)
    total_rarity = sum(rarity.values())
    if total_rarity <= 0.0:
        rarity = {category_id: 1.0 for category_id in categories}
        total_rarity = float(len(categories))

    raw = {
        category_id: budget * rarity[category_id] / total_rarity
        for category_id in categories
    }
    budgets = {
        category_id: min(int(math.floor(raw[category_id])), class_capacities[category_id])
        for category_id in categories
    }

    remaining = budget - sum(budgets.values())
    ranked = sorted(
        categories,
        key=lambda category_id: (
            -(raw[category_id] - math.floor(raw[category_id])),
            -rarity[category_id],
            str(category_id),
        ),
    )
    while remaining > 0:
        progressed = False
        for category_id in ranked:
            if budgets[category_id] < class_capacities[category_id]:
                budgets[category_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return budgets


def _rank_image_scores(image_scores: Dict[Any, float]) -> List[Any]:
    ranked = sorted(
        image_scores.items(),
        key=lambda item: (-float(item[1]), _sort_image_id(item[0])),
    )
    return [image_id for image_id, _ in ranked]


def _unique_values(values: Iterable[Any]) -> List[Any]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _load_candidate_embeddings(
    embedding_source: str,
    embedding_path: Optional[Path],
    detections: Sequence[Dict[str, Any]],
    candidate_image_ids: Sequence[Any],
) -> Dict[Any, Any]:
    source = embedding_source.lower().replace('_', '-')
    if source in ('external', 'cache'):
        if embedding_path is None:
            raise ValueError('PAL full mode requires pal_embedding_path when embedding_source=%s'
                             % embedding_source)
        embeddings = read_embedding_cache(embedding_path)
    elif source in ('detection', 'detector', 'class-scores', 'class-score'):
        embeddings = build_image_embeddings(
            detections,
            image_ids=candidate_image_ids,
            backend='detection',
        )
    elif source in ('vit', 'vision-transformer'):
        embeddings = build_image_embeddings(
            detections,
            image_ids=candidate_image_ids,
            backend='vit',
        )
    else:
        raise ValueError('Unsupported PAL embedding source: %s' % embedding_source)

    missing = [image_id for image_id in candidate_image_ids if image_id not in embeddings]
    if missing:
        raise ValueError(
            'Missing PAL RCSP embeddings for %d candidate image(s), first ids: %s'
            % (len(missing), missing[:10]))
    return embeddings


def _append_selected(
    selected: List[Any],
    selected_details: List[Dict[str, Any]],
    selected_set: set,
    record: Dict[str, Any],
    stage: str,
) -> None:
    image_id = record['image_id']
    selected.append(image_id)
    selected_set.add(image_id)
    detail = dict(record)
    detail['selection_stage'] = stage
    selected_details.append(detail)


def select_lius_images(
    labeled_pool: Dict[str, Any],
    unlabeled_pool: Dict[str, Any],
    labeled_detections: Sequence[Dict[str, Any]],
    unlabeled_detections: Sequence[Dict[str, Any]],
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    labeled_ids = set(image_ids(labeled_pool))
    unlabeled_ids = image_ids(unlabeled_pool)

    labeled_dets = filter_pool_detections(labeled_detections, labeled_ids)
    unlabeled_dets = filter_pool_detections(unlabeled_detections, unlabeled_ids)

    matched = match_detections_to_ground_truth(
        labeled_dets,
        labeled_pool,
        iou_threshold=iou_threshold,
    )
    models = train_classwise_models(matched)
    scored = score_unlabeled_detections(unlabeled_dets, models)

    class_image_scores = _candidate_image_scores_by_class(scored)
    labeled_counts = _category_counts_from_annotations(labeled_pool.get('annotations', []))
    unlabeled_counts = _category_counts_from_detections(unlabeled_dets)
    class_capacities = {
        category_id: len(image_scores)
        for category_id, image_scores in class_image_scores.items()
    }
    class_budgets = allocate_class_budgets(
        budget,
        labeled_counts,
        unlabeled_counts,
        class_capacities,
    )

    candidate_scores: Dict[Any, float] = {}
    for category_id, class_budget in class_budgets.items():
        if class_budget <= 0:
            continue
        class_candidates = _rank_image_scores(class_image_scores.get(category_id, {}))
        for image_id in class_candidates[:2 * class_budget]:
            score = class_image_scores[category_id][image_id]
            if image_id not in candidate_scores or score > candidate_scores[image_id]:
                candidate_scores[image_id] = score

    selected = _rank_image_scores(candidate_scores)[:budget]
    if len(selected) < budget:
        global_scores: Dict[Any, float] = defaultdict(float)
        for det in scored:
            image_id = det.get('image_id')
            if image_id is not None:
                global_scores[image_id] = max(global_scores[image_id], float(det.get('lius_score', 0.0)))
        for image_id in _rank_image_scores(global_scores):
            if image_id not in selected:
                selected.append(image_id)
            if len(selected) == budget:
                break

    if len(selected) < budget:
        remaining = [image_id for image_id in unlabeled_ids if image_id not in set(selected)]
        selected.extend(deterministic_random_sample(remaining, budget - len(selected), seed=seed))

    return {
        'selected_image_ids': selected[:budget],
        'mode': 'lius',
        'class_budgets': class_budgets,
        'matched_detection_count': len(matched),
        'scored_detection_count': len(scored),
    }


def select_full_pal_images(
    labeled_pool: Dict[str, Any],
    unlabeled_pool: Dict[str, Any],
    labeled_detections: Sequence[Dict[str, Any]],
    unlabeled_detections: Sequence[Dict[str, Any]],
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
    alpha: float = 0.9,
    beta: float = 0.04,
    gamma: float = 0.02,
    embedding_source: str = 'external',
    embedding_path: Optional[Path] = None,
) -> Dict[str, Any]:
    labeled_ids = set(image_ids(labeled_pool))
    unlabeled_ids = image_ids(unlabeled_pool)

    labeled_dets = filter_pool_detections(labeled_detections, labeled_ids)
    unlabeled_dets = filter_pool_detections(unlabeled_detections, unlabeled_ids)

    matched = match_detections_to_ground_truth(
        labeled_dets,
        labeled_pool,
        iou_threshold=iou_threshold,
    )
    models = train_classwise_models(matched)
    scored = score_unlabeled_detections(unlabeled_dets, models)

    labeled_counts = _category_counts_from_annotations(labeled_pool.get('annotations', []))
    unlabeled_counts = _category_counts_from_detections(unlabeled_dets)
    class_candidates = _candidate_records_by_class(scored)
    class_capacities = {
        category_id: len(image_records)
        for category_id, image_records in class_candidates.items()
    }
    all_categories = set(labeled_counts) | set(unlabeled_counts) | set(class_capacities)
    class_weights = compute_guide_class_weights(labeled_counts, unlabeled_counts, all_categories)
    class_budgets = allocate_guide_class_budgets(class_weights, class_capacities, budget)

    candidates_by_class = build_class_candidates(
        scored,
        class_budgets,
        candidate_multiplier=2,
        score_key='lius_score',
    )
    candidate_image_ids = [
        candidate['image_id']
        for candidates in candidates_by_class.values()
        for candidate in candidates
    ]
    unique_candidate_image_ids = _unique_values(candidate_image_ids)
    embeddings = _load_candidate_embeddings(
        embedding_source,
        embedding_path,
        scored,
        unique_candidate_image_ids,
    ) if unique_candidate_image_ids else {}

    image_detections = group_detections_by_image(scored)
    scored_by_class = score_guide_candidates(
        candidates_by_class,
        image_detections,
        class_weights,
        embeddings,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lius_key='lius_score',
    )
    all_candidates = [
        candidate
        for candidates in scored_by_class.values()
        for candidate in candidates
    ]
    selection = select_deduplicated_candidates(
        scored_by_class,
        class_budgets,
        budget,
        class_weights=class_weights,
        score_key='pal_score',
    )
    selected = list(selection['selected_image_ids'])
    selected_details = list(selection['selected_candidates'])
    selected_set = set(selected)

    if len(selected) < budget:
        global_scores: Dict[Any, float] = defaultdict(float)
        for det in scored:
            image_id = det.get('image_id')
            if image_id is not None:
                global_scores[image_id] = max(global_scores[image_id], float(det.get('lius_score', 0.0)))
        for image_id in _rank_image_scores(global_scores):
            if image_id in selected_set:
                continue
            record = {
                'image_id': image_id,
                'category_id': None,
                'lius_score': float(global_scores[image_id]),
                'pal_score': float(global_scores[image_id]),
            }
            _append_selected(selected, selected_details, selected_set, record, 'global_lius_refill')
            if len(selected) == budget:
                break

    if len(selected) < budget:
        remaining = [image_id for image_id in unlabeled_ids if image_id not in selected_set]
        for image_id in deterministic_random_sample(remaining, budget - len(selected), seed=seed):
            record = {
                'image_id': image_id,
                'category_id': None,
                'lius_score': 0.0,
                'pal_score': 0.0,
            }
            _append_selected(selected, selected_details, selected_set, record, 'random_refill')

    return {
        'selected_image_ids': selected[:budget],
        'mode': 'full',
        'class_budgets': class_budgets,
        'class_weights': class_weights,
        'matched_detection_count': len(matched),
        'scored_detection_count': len(scored),
        'candidate_image_count': len(unique_candidate_image_ids),
        'embedding_source': embedding_source,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'class_selected_counts': selection['class_selected_counts'],
        'unfilled_guide_budget': selection['unfilled_budget'],
        'selected_details': selected_details[:budget],
        'candidate_scores': all_candidates,
    }


def sample_lius_only_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_detections_json: Path,
    unlabeled_detections_json: Path,
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    labeled_pool = read_coco_json(labeled_pool_json)
    unlabeled_pool = read_coco_json(unlabeled_pool_json)
    labeled_detections = load_detection_records(labeled_detections_json)
    unlabeled_detections = load_detection_records(unlabeled_detections_json)
    return select_lius_images(
        labeled_pool=labeled_pool,
        unlabeled_pool=unlabeled_pool,
        labeled_detections=labeled_detections,
        unlabeled_detections=unlabeled_detections,
        budget=budget,
        iou_threshold=iou_threshold,
        seed=seed,
    )


def sample_pal_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_detections_json: Path,
    unlabeled_detections_json: Path,
    budget: int,
    mode: str = 'lius',
    iou_threshold: float = 0.5,
    seed: int = 0,
    alpha: float = 0.9,
    beta: float = 0.04,
    gamma: float = 0.02,
    embedding_source: str = 'external',
    embedding_path: Optional[Path] = None,
) -> Dict[str, Any]:
    labeled_pool = read_coco_json(labeled_pool_json)
    unlabeled_pool = read_coco_json(unlabeled_pool_json)
    labeled_detections = load_detection_records(labeled_detections_json)
    unlabeled_detections = load_detection_records(unlabeled_detections_json)
    normalized_mode = mode.lower()
    if normalized_mode in ('lius', 'lius_only', 'lius-only'):
        return select_lius_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=budget,
            iou_threshold=iou_threshold,
            seed=seed,
        )
    if normalized_mode in ('full', 'guide'):
        return select_full_pal_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=budget,
            iou_threshold=iou_threshold,
            seed=seed,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            embedding_source=embedding_source,
            embedding_path=embedding_path,
        )
    raise ValueError('Unsupported PAL mode: %s' % mode)
