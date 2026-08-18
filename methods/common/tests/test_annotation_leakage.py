import json
from pathlib import Path

import pytest

from methods.ada_fnp.phases import (
    RevealRequest,
    execute_reveal,
    labeled_manifest_for_sampler,
    resolve_detector_phase,
)
from methods.common.data.cityscapes import (
    CATEGORY_IDS,
    CITYSCAPES_CLASSES,
    TARGET_TRAIN_NAMESPACE,
    materialize_target_train_labeled,
)
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState


def _identity(sample_id: str, namespace: str = TARGET_TRAIN_NAMESPACE):
    return SampleIdentity(namespace, sample_id)


def _categories():
    return [
        {'id': CATEGORY_IDS[name], 'name': name, 'supercategory': 'object'}
        for name in CITYSCAPES_CLASSES
    ]


def _write_oracle(path: Path, samples, *, duplicate_sample: bool = False) -> dict:
    images = []
    annotations = []
    for index, sample in enumerate(samples, start=1):
        image = {
            'id': index * 10,
            'file_name': 'data/leftImg8bit_foggy/train/{}.png'.format(
                sample.sample_id
            ),
            'width': 10,
            'height': 8,
            'sample_id': sample.qualified_id,
        }
        images.append(image)
        annotations.append(
            {
                'id': index,
                'image_id': image['id'],
                'category_id': 3,
                'bbox': [1.0, 1.0, 2.0, 2.0],
                'area': 4.0,
                'segmentation': [[1.0, 1.0, 3.0, 1.0, 3.0, 3.0]],
                'iscrowd': 0,
                'selection_probe': sample.sample_id,
            }
        )
    if duplicate_sample:
        duplicate = dict(images[0])
        duplicate['id'] = 999
        images.append(duplicate)
    value = {
        'info': {'split': 'target_train_oracle'},
        'images': images,
        'annotations': annotations,
        'categories': _categories(),
    }
    path.write_text(json.dumps(value), encoding='utf-8')
    return value


def test_reveal_contains_exactly_selected_annotations_and_is_idempotent(
    tmp_path,
) -> None:
    samples = (_identity('city/a'), _identity('city/b'), _identity('city/c'))
    oracle_path = tmp_path / 'target_train_oracle.json'
    oracle = _write_oracle(oracle_path, samples)
    pool_state = PoolState.initialize(samples, total_budget=3).acquire((samples[1],))
    output_path = tmp_path / 'target_train_labeled.json'
    request = RevealRequest(oracle_path, output_path, pool_state)

    first = execute_reveal(request)
    first_bytes = output_path.read_bytes()
    second = execute_reveal(request)
    labeled = json.loads(output_path.read_text(encoding='utf-8'))

    assert first.sha256 == second.sha256
    assert first_bytes == output_path.read_bytes()
    assert first.image_count == 1
    assert first.annotation_count == 1
    assert labeled['images'] == [oracle['images'][1]]
    assert labeled['annotations'] == [oracle['annotations'][1]]
    assert labeled['categories'] == oracle['categories']
    assert 'city/a' not in first_bytes.decode('utf-8')
    assert 'city/c' not in first_bytes.decode('utf-8')
    assert not tuple(tmp_path.glob('*.tmp'))

    phase = resolve_detector_phase(5000, 10000, labeled_sample_count=1)
    assert labeled_manifest_for_sampler(phase, first) == output_path


def test_empty_pre_acquisition_manifest_never_reads_or_exposes_oracle(tmp_path) -> None:
    samples = (_identity('city/a'), _identity('city/b'))
    pool_state = PoolState.initialize(samples, total_budget=0)
    missing_oracle = tmp_path / 'must-not-be-opened-oracle.json'
    output_path = tmp_path / 'target_train_labeled.json'

    manifest = execute_reveal(RevealRequest(missing_oracle, output_path, pool_state))
    raw = output_path.read_text(encoding='utf-8')
    value = json.loads(raw)

    assert not manifest.has_samples
    assert value['images'] == []
    assert value['annotations'] == []
    assert value['categories'] == _categories()
    assert 'oracle' not in raw.lower()
    assert labeled_manifest_for_sampler(
        resolve_detector_phase(0, 5000, 0), manifest
    ) is None
    assert labeled_manifest_for_sampler(
        resolve_detector_phase(5000, 10000, 0), manifest
    ) is None


def test_reveal_rejects_unknown_duplicate_and_wrong_namespace(tmp_path) -> None:
    known = (_identity('city/a'), _identity('city/b'))
    oracle_path = tmp_path / 'target_train_oracle.json'
    _write_oracle(oracle_path, known)

    unknown = _identity('city/unknown')
    unknown_pool = PoolState.initialize((known[0], unknown), 1).acquire((unknown,))
    with pytest.raises(ValueError, match='committed pool differ'):
        materialize_target_train_labeled(
            oracle_path, unknown_pool, tmp_path / 'unknown.json'
        )

    _write_oracle(oracle_path, known, duplicate_sample=True)
    valid_pool = PoolState.initialize(known, 1).acquire((known[0],))
    with pytest.raises(ValueError, match='duplicate sample ID'):
        materialize_target_train_labeled(
            oracle_path, valid_pool, tmp_path / 'duplicate.json'
        )

    wrong = _identity('city/a', namespace='cityscapes.train')
    wrong_pool = PoolState.initialize((wrong,), 1).acquire((wrong,))
    with pytest.raises(ValueError, match='wrong namespace'):
        materialize_target_train_labeled(
            oracle_path, wrong_pool, tmp_path / 'wrong-namespace.json'
        )


def test_oracle_annotation_cannot_reference_an_unknown_image(tmp_path) -> None:
    sample = _identity('city/a')
    oracle_path = tmp_path / 'target_train_oracle.json'
    oracle = _write_oracle(oracle_path, (sample,))
    oracle['annotations'][0]['image_id'] = 123456
    oracle_path.write_text(json.dumps(oracle), encoding='utf-8')
    pool = PoolState.initialize((sample,), 1).acquire((sample,))

    with pytest.raises(ValueError, match='unknown image'):
        materialize_target_train_labeled(
            oracle_path, pool, tmp_path / 'unknown-image.json'
        )
