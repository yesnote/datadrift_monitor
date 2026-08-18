import json
import os
from pathlib import Path
import struct

import pytest

from methods.common.data.cityscapes import (
    CITYSCAPES_CLASSES,
    CLEAR_SUFFIX,
    FOGGY_002_SUFFIX,
    POLYGON_SUFFIX,
    prepare_cityscapes_to_foggy,
    validate_cityscapes_to_foggy_layout,
)
from tools.common.paths import repository_root


def _write_scene(
    repository: Path,
    split: str,
    city: str,
    stem: str,
    *,
    clear: bool,
    foggy: bool,
    objects,
) -> None:
    png_header = (
        b'\x89PNG\r\n\x1a\n'
        + struct.pack('>I', 13)
        + b'IHDR'
        + struct.pack('>II', 10, 8)
    )
    if clear:
        path = repository / 'data' / 'leftImg8bit' / split / city
        path.mkdir(parents=True, exist_ok=True)
        (path / '{}{}'.format(stem, CLEAR_SUFFIX)).write_bytes(png_header)
    if foggy:
        path = repository / 'data' / 'leftImg8bit_foggy' / split / city
        path.mkdir(parents=True, exist_ok=True)
        (path / '{}{}'.format(stem, FOGGY_002_SUFFIX)).write_bytes(png_header)
    path = repository / 'data' / 'gtFine' / split / city
    path.mkdir(parents=True, exist_ok=True)
    (path / '{}{}'.format(stem, POLYGON_SUFFIX)).write_text(
        json.dumps(
            {
                'imgWidth': 10,
                'imgHeight': 8,
                'objects': objects,
            }
        ),
        encoding='utf-8',
    )


def test_png_dimensions_reads_only_header(tmp_path, monkeypatch):
    from methods.common.data import cityscapes

    _write_scene(
        tmp_path, 'train', 'city', 'city_000000_000001',
        clear=True, foggy=False, objects=[],
    )
    image = next((tmp_path / 'data' / 'leftImg8bit').rglob('*.png'))
    monkeypatch.setattr(
        Path,
        'read_bytes',
        lambda path: (_ for _ in ()).throw(
            AssertionError('converter must not read the whole PNG')
        ),
    )
    assert cityscapes._read_png_dimensions(image) == (10, 8)


def _tiny_layout(repository: Path) -> None:
    objects = [
        {
            'label': 'person',
            'polygon': [[-2, -1], [5, -1], [5, 4], [-2, 4]],
        },
        {
            'label': 'persongroup',
            'polygon': [[6, 1], [12, 1], [12, 5], [6, 5]],
        },
        {
            'label': 'car',
            'deleted': True,
            'polygon': [[1, 1], [3, 1], [3, 3], [1, 3]],
        },
        {
            'label': 'road',
            'polygon': [[0, 0], [10, 0], [10, 8], [0, 8]],
        },
        {
            'label': 'bicycle',
            'polygon': [[1, 1], [1, 2], [1, 3]],
        },
    ]
    _write_scene(
        repository,
        'train',
        'aachen',
        'aachen_000000_000001',
        clear=True,
        foggy=True,
        objects=objects,
    )
    _write_scene(
        repository,
        'train',
        'bochum',
        'bochum_000000_000002',
        clear=True,
        foggy=True,
        objects=[{'label': 'sky', 'polygon': [[0, 0], [2, 0], [2, 2]]}],
    )
    _write_scene(
        repository,
        'val',
        'frankfurt',
        'frankfurt_000000_000003',
        clear=False,
        foggy=True,
        objects=objects[:1],
    )


def _prepare(repository: Path):
    return prepare_cityscapes_to_foggy(
        repository / 'data' / 'leftImg8bit',
        repository / 'data' / 'leftImg8bit_foggy',
        repository / 'data' / 'gtFine',
        repository / 'work_dirs' / '.dataset_cache' / 'cityscapes-to-foggy',
        repository,
        expected_train_images=2,
        expected_val_images=1,
    )


def test_conversion_is_deterministic_coco_and_preserves_empty_images(tmp_path) -> None:
    _tiny_layout(tmp_path)

    first_manifest = _prepare(tmp_path)
    cache = tmp_path / 'work_dirs' / '.dataset_cache' / 'cityscapes-to-foggy'
    first_source_bytes = (cache / 'source_train.json').read_bytes()
    second_manifest = _prepare(tmp_path)

    assert first_manifest == second_manifest
    assert first_source_bytes == (cache / 'source_train.json').read_bytes()
    assert not tuple(cache.glob('*.tmp'))

    source = json.loads(first_source_bytes)
    assert len(source['images']) == 2
    assert source['images'][0]['sample_id'] == (
        'cityscapes.train:aachen/aachen_000000_000001'
    )
    assert source['images'][1]['sample_id'].endswith('bochum/bochum_000000_000002')
    assert [category['id'] for category in source['categories']] == list(range(1, 9))
    assert [category['name'] for category in source['categories']] == list(
        CITYSCAPES_CLASSES
    )
    assert len(source['annotations']) == 2
    assert source['annotations'][0]['category_id'] == 1
    assert source['annotations'][0]['bbox'] == [0.0, 0.0, 5.0, 4.0]
    assert source['annotations'][0]['area'] == 20.0
    assert source['annotations'][0]['iscrowd'] == 0
    assert source['annotations'][1]['bbox'] == [6.0, 1.0, 4.0, 4.0]
    assert source['annotations'][1]['iscrowd'] == 1
    assert {annotation['image_id'] for annotation in source['annotations']} == {1}


def test_target_unlabeled_has_no_oracle_information(tmp_path) -> None:
    _tiny_layout(tmp_path)
    manifest = _prepare(tmp_path)
    cache = tmp_path / 'work_dirs' / '.dataset_cache' / 'cityscapes-to-foggy'
    raw_unlabeled = (cache / 'target_train_unlabeled.json').read_text(
        encoding='utf-8'
    )
    unlabeled = json.loads(raw_unlabeled)
    oracle = json.loads((cache / 'target_train_oracle.json').read_text('utf-8'))

    assert 'oracle' not in raw_unlabeled.lower()
    assert unlabeled['annotations'] == []
    assert all(
        set(image) == {'id', 'file_name', 'width', 'height', 'sample_id'}
        for image in unlabeled['images']
    )
    assert all(
        image['sample_id'].startswith('foggy-cityscapes.beta-0.02.train:')
        for image in unlabeled['images']
    )
    assert len(oracle['annotations']) == 2
    assert manifest['outputs']['target_train_unlabeled']['annotations'] == 0
    assert len(manifest['fingerprint']) == 64
    assert len(manifest['input_fingerprint']) == 64


def test_layout_rejects_count_mismatch_and_non_bijective_scenes(tmp_path) -> None:
    _tiny_layout(tmp_path)
    with pytest.raises(ValueError, match='exactly 3'):
        validate_cityscapes_to_foggy_layout(
            tmp_path / 'data' / 'leftImg8bit',
            tmp_path / 'data' / 'leftImg8bit_foggy',
            tmp_path / 'data' / 'gtFine',
            expected_train_images=3,
            expected_val_images=1,
        )

    missing = next(
        (tmp_path / 'data' / 'leftImg8bit_foggy' / 'train').rglob(
            '*{}'.format(FOGGY_002_SUFFIX)
        )
    )
    missing.unlink()
    with pytest.raises(ValueError, match='scene-bijective'):
        validate_cityscapes_to_foggy_layout(
            tmp_path / 'data' / 'leftImg8bit',
            tmp_path / 'data' / 'leftImg8bit_foggy',
            tmp_path / 'data' / 'gtFine',
            expected_train_images=None,
            expected_val_images=1,
        )


def test_cache_output_must_stay_under_dataset_cache(tmp_path) -> None:
    _tiny_layout(tmp_path)
    with pytest.raises(ValueError, match='dataset_cache'):
        prepare_cityscapes_to_foggy(
            tmp_path / 'data' / 'leftImg8bit',
            tmp_path / 'data' / 'leftImg8bit_foggy',
            tmp_path / 'data' / 'gtFine',
            tmp_path / 'outside-cache',
            tmp_path,
            expected_train_images=2,
            expected_val_images=1,
        )


@pytest.mark.skipif(
    os.environ.get('ADAOD_VALIDATE_REAL_CITYSCAPES') != '1',
    reason='set ADAOD_VALIDATE_REAL_CITYSCAPES=1 for the real-layout audit',
)
def test_optional_real_cityscapes_layout() -> None:
    root = repository_root()
    layout = validate_cityscapes_to_foggy_layout(
        root / 'data' / 'leftImg8bit',
        root / 'data' / 'leftImg8bit_foggy',
        root / 'data' / 'gtFine',
    )
    assert len(layout.source_train) == 2975
    assert len(layout.target_train) == 2975
    assert len(layout.target_val) == 500
