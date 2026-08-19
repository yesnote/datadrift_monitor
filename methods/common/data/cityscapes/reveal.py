'''Selected-only target annotation reveal for Cityscapes-to-Foggy.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from methods.common.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState

from .layout import TARGET_TRAIN_NAMESPACE, cityscapes_categories


@dataclass(frozen=True)
class LabeledTargetManifest:
    '''Content reference for a selected-only target-labeled COCO manifest.'''

    path: Path
    sha256: str
    image_count: int
    annotation_count: int


def _selected_samples_for_reveal(
    pool_state: PoolState,
    expected_namespace: str,
) -> Tuple[SampleIdentity, ...]:
    if not isinstance(pool_state, PoolState):
        raise TypeError('pool_state must be a PoolState')
    if not isinstance(expected_namespace, str) or not expected_namespace:
        raise ValueError('expected_namespace must be a non-empty string')
    wrong_namespace = tuple(
        sample
        for sample in pool_state.universe
        if sample.namespace != expected_namespace
    )
    if wrong_namespace:
        raise ValueError(
            'pool contains a sample from the wrong namespace: {}'.format(
                wrong_namespace[0].qualified_id
            )
        )
    if len(pool_state.labeled) != len(set(pool_state.labeled)):
        raise ValueError('committed labeled pool contains duplicate sample IDs')
    return tuple(sorted(pool_state.labeled, key=lambda sample: sample.sample_id))


def _empty_labeled_dataset() -> dict:
    return {
        'info': {
            'scenario': 'cityscapes-to-foggy',
            'split': 'target_train_labeled',
        },
        'images': [],
        'annotations': [],
        'categories': cityscapes_categories(),
    }


def _load_oracle_coco(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('target oracle is not valid UTF-8 JSON') from error
    if not isinstance(value, Mapping):
        raise ValueError('target oracle JSON root must be an object')
    for key in ('images', 'annotations', 'categories'):
        if not isinstance(value.get(key), list):
            raise ValueError('target oracle {} must be a list'.format(key))
    if value['categories'] != cityscapes_categories():
        raise ValueError('target oracle categories do not match canonical C-to-F IDs')
    return value


def _oracle_images_by_sample(
    images: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[SampleIdentity, Mapping[str, Any]], Dict[int, SampleIdentity]]:
    by_sample = {}
    by_image_id = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError('target oracle images must be objects')
        sample_value = image.get('sample_id')
        image_id = image.get('id')
        try:
            sample = SampleIdentity.parse(sample_value)
        except (TypeError, ValueError) as error:
            raise ValueError('target oracle contains an invalid sample_id') from error
        if sample.namespace != TARGET_TRAIN_NAMESPACE:
            raise ValueError(
                'target oracle sample has the wrong namespace: {}'.format(
                    sample.qualified_id
                )
            )
        if sample in by_sample:
            raise ValueError(
                'target oracle contains duplicate sample ID {}'.format(
                    sample.qualified_id
                )
            )
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError('target oracle image IDs must be integers')
        if image_id in by_image_id:
            raise ValueError('target oracle contains duplicate image IDs')
        by_sample[sample] = image
        by_image_id[image_id] = sample
    return by_sample, by_image_id


def _validated_oracle_annotations(
    annotations: Sequence[Mapping[str, Any]],
    images_by_id: Mapping[int, SampleIdentity],
) -> Tuple[Mapping[str, Any], ...]:
    annotation_ids = set()
    validated = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError('target oracle annotations must be objects')
        annotation_id = annotation.get('id')
        image_id = annotation.get('image_id')
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            raise ValueError('target oracle annotation IDs must be integers')
        if annotation_id in annotation_ids:
            raise ValueError('target oracle contains duplicate annotation IDs')
        if image_id not in images_by_id:
            raise ValueError('target oracle annotation references an unknown image')
        annotation_ids.add(annotation_id)
        validated.append(annotation)
    return tuple(validated)


def reveal_selected_annotations(
    oracle_json_path: Path,
    pool_state: PoolState,
    output_path: Path,
) -> LabeledTargetManifest:
    '''Atomically reveal annotations for exactly the committed labeled pool.'''

    selected = _selected_samples_for_reveal(pool_state, TARGET_TRAIN_NAMESPACE)
    output_path = Path(output_path)
    if not selected:
        labeled_dataset = _empty_labeled_dataset()
    else:
        oracle = _load_oracle_coco(Path(oracle_json_path))
        images_by_sample, samples_by_image_id = _oracle_images_by_sample(
            oracle['images']
        )
        if set(images_by_sample) != set(pool_state.universe):
            missing = sorted(
                sample.qualified_id
                for sample in set(pool_state.universe) - set(images_by_sample)
            )
            unknown = sorted(
                sample.qualified_id
                for sample in set(images_by_sample) - set(pool_state.universe)
            )
            raise ValueError(
                'target oracle and committed pool differ '
                '(missing={}, unknown={})'.format(missing[:3], unknown[:3])
            )
        unknown_selected = tuple(
            sample for sample in selected if sample not in images_by_sample
        )
        if unknown_selected:
            raise ValueError(
                'selected sample is unknown to the target oracle: {}'.format(
                    unknown_selected[0].qualified_id
                )
            )
        validated_annotations = _validated_oracle_annotations(
            oracle['annotations'], samples_by_image_id
        )
        selected_image_ids = {images_by_sample[sample]['id'] for sample in selected}
        selected_annotations = [
            dict(annotation)
            for annotation in validated_annotations
            if annotation['image_id'] in selected_image_ids
        ]
        selected_annotations.sort(key=lambda annotation: annotation['id'])
        labeled_dataset = {
            'info': {
                'scenario': 'cityscapes-to-foggy',
                'split': 'target_train_labeled',
            },
            'images': [dict(images_by_sample[sample]) for sample in selected],
            'annotations': selected_annotations,
            'categories': [dict(category) for category in oracle['categories']],
        }
    payload = canonical_json_bytes(labeled_dataset)
    atomic_write_bytes(output_path, payload)
    return LabeledTargetManifest(
        path=output_path,
        sha256=sha256_bytes(payload),
        image_count=len(labeled_dataset['images']),
        annotation_count=len(labeled_dataset['annotations']),
    )
