'''Deterministic Cityscapes polygon conversion for the C-to-F scenario.'''

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from methods.common.artifacts import atomic_write_bytes, sha256_file

from .annotations import selected_samples_for_reveal
from .image_identity import SampleIdentity
from .pool import PoolState


CITYSCAPES_CLASSES: Tuple[str, ...] = (
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
)
CATEGORY_IDS = {
    class_name: category_id
    for category_id, class_name in enumerate(CITYSCAPES_CLASSES, start=1)
}
CLEAR_SUFFIX = '_leftImg8bit.png'
FOGGY_002_SUFFIX = '_leftImg8bit_foggy_beta_0.02.png'
POLYGON_SUFFIX = '_gtFine_polygons.json'
SOURCE_NAMESPACE = 'cityscapes.train'
TARGET_TRAIN_NAMESPACE = 'foggy-cityscapes.beta-0.02.train'
TARGET_VAL_NAMESPACE = 'foggy-cityscapes.beta-0.02.val'
EXPECTED_TRAIN_IMAGES = 2975
EXPECTED_VAL_IMAGES = 500


@dataclass(frozen=True)
class SceneAsset:
    '''One image and its matching gtFine polygon file.'''

    scene_id: str
    image_path: Path
    polygon_path: Path


@dataclass(frozen=True)
class CityscapesToFoggyLayout:
    '''Validated C-to-F inputs in deterministic scene order.'''

    source_train: Tuple[SceneAsset, ...]
    target_train: Tuple[SceneAsset, ...]
    target_val: Tuple[SceneAsset, ...]


@dataclass(frozen=True)
class LabeledTargetManifest:
    '''Content reference for a selected-only target-labeled COCO manifest.'''

    path: Path
    sha256: str
    image_count: int
    annotation_count: int

    @property
    def has_samples(self) -> bool:
        return self.image_count > 0


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError('{} directory does not exist: {}'.format(description, path))


def _inventory(root: Path, split: str, suffix: str) -> Dict[str, Path]:
    split_root = root / split
    _require_directory(split_root, '{} split'.format(split))
    inventory = {}
    for path in sorted(split_root.rglob('*{}'.format(suffix))):
        if not path.is_file():
            continue
        relative = path.relative_to(split_root).as_posix()
        scene_id = relative[: -len(suffix)]
        if not scene_id:
            raise ValueError('could not derive a scene ID from {}'.format(path))
        if scene_id in inventory:
            raise ValueError('duplicate scene ID {} in {}'.format(scene_id, split_root))
        inventory[scene_id] = path
    return inventory


def _validate_count(
    name: str,
    inventory: Mapping[str, Path],
    expected_count: Optional[int],
) -> None:
    if expected_count is None:
        return
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError('expected image counts must be integers or None')
    if expected_count < 0:
        raise ValueError('expected image counts must not be negative')
    if len(inventory) != expected_count:
        raise ValueError(
            '{} must contain exactly {} scenes, found {}'.format(
                name, expected_count, len(inventory)
            )
        )


def _require_bijection(
    left_name: str,
    left: Mapping[str, Path],
    right_name: str,
    right: Mapping[str, Path],
) -> None:
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if left_only or right_only:
        raise ValueError(
            '{} and {} are not scene-bijective '
            '(only {}: {}, only {}: {})'.format(
                left_name,
                right_name,
                left_name,
                left_only[:3],
                right_name,
                right_only[:3],
            )
        )


def _pair_assets(
    images: Mapping[str, Path],
    polygons: Mapping[str, Path],
) -> Tuple[SceneAsset, ...]:
    return tuple(
        SceneAsset(scene_id, images[scene_id], polygons[scene_id])
        for scene_id in sorted(images)
    )


def validate_cityscapes_to_foggy_layout(
    clear_image_root: Path,
    foggy_image_root: Path,
    polygon_root: Path,
    *,
    expected_train_images: Optional[int] = EXPECTED_TRAIN_IMAGES,
    expected_val_images: Optional[int] = EXPECTED_VAL_IMAGES,
) -> CityscapesToFoggyLayout:
    '''Validate exact counts and scene bijections needed by C-to-F only.'''

    clear_image_root = Path(clear_image_root)
    foggy_image_root = Path(foggy_image_root)
    polygon_root = Path(polygon_root)
    _require_directory(clear_image_root, 'clear Cityscapes image root')
    _require_directory(foggy_image_root, 'Foggy Cityscapes image root')
    _require_directory(polygon_root, 'gtFine polygon root')

    clear_train = _inventory(clear_image_root, 'train', CLEAR_SUFFIX)
    foggy_train = _inventory(foggy_image_root, 'train', FOGGY_002_SUFFIX)
    foggy_val = _inventory(foggy_image_root, 'val', FOGGY_002_SUFFIX)
    polygon_train = _inventory(polygon_root, 'train', POLYGON_SUFFIX)
    polygon_val = _inventory(polygon_root, 'val', POLYGON_SUFFIX)

    for name, inventory, expected in (
        ('source train', clear_train, expected_train_images),
        ('target train', foggy_train, expected_train_images),
        ('target train oracle', polygon_train, expected_train_images),
        ('target val', foggy_val, expected_val_images),
        ('target val evaluator', polygon_val, expected_val_images),
    ):
        _validate_count(name, inventory, expected)

    _require_bijection('source train', clear_train, 'target train', foggy_train)
    _require_bijection('source train', clear_train, 'train polygons', polygon_train)
    _require_bijection('target val', foggy_val, 'val polygons', polygon_val)
    return CityscapesToFoggyLayout(
        source_train=_pair_assets(clear_train, polygon_train),
        target_train=_pair_assets(foggy_train, polygon_train),
        target_val=_pair_assets(foggy_val, polygon_val),
    )


def _read_polygon_file(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid gtFine polygon JSON: {}'.format(path)) from error
    if not isinstance(value, Mapping):
        raise ValueError('gtFine polygon JSON root must be an object: {}'.format(path))
    return value


def _positive_dimension(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('{} must be a positive integer in {}'.format(name, path))
    return value


def _read_png_dimensions(path: Path) -> Tuple[int, int]:
    with path.open('rb') as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b'\x89PNG\r\n\x1a\n'
        or header[12:16] != b'IHDR'
    ):
        raise ValueError('image is not a supported PNG file: {}'.format(path))
    width, height = struct.unpack('>II', header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError('PNG dimensions must be positive: {}'.format(path))
    return width, height


def _category(label: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(label, str):
        return None
    if label in CATEGORY_IDS:
        return CATEGORY_IDS[label], 0
    if label.endswith('group'):
        base_label = label[: -len('group')]
        if base_label in CATEGORY_IDS:
            return CATEGORY_IDS[base_label], 1
    return None


def _clipped_polygon(
    polygon: Any,
    width: int,
    height: int,
) -> Optional[Tuple[Sequence[float], Sequence[float], float]]:
    if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
        return None
    points = []
    for point in polygon:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            return None
        x, y = point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            return None
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))))
    if len(points) < 3:
        return None
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    double_area = sum(
        x_first * y_second - x_second * y_first
        for (x_first, y_first), (x_second, y_second) in zip(
            points, points[1:] + points[:1]
        )
    )
    area = abs(double_area) / 2.0
    if area <= 0.0:
        return None
    flattened = [coordinate for point in points for coordinate in point]
    return flattened, (x_min, y_min, box_width, box_height), area


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError(
            'dataset paths must be addressed through repository-relative junctions'
        ) from error


def _categories() -> list:
    return [
        {'id': CATEGORY_IDS[name], 'name': name, 'supercategory': 'object'}
        for name in CITYSCAPES_CLASSES
    ]


def _build_coco_dataset(
    assets: Sequence[SceneAsset],
    namespace: str,
    repository_root: Path,
    *,
    include_annotations: bool,
    split_name: str,
) -> dict:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, asset in enumerate(assets, start=1):
        width, height = _read_png_dimensions(asset.image_path)
        identity = SampleIdentity(namespace, asset.scene_id)
        images.append(
            {
                'id': image_id,
                'file_name': _repository_relative(asset.image_path, repository_root),
                'width': width,
                'height': height,
                'sample_id': identity.qualified_id,
            }
        )
        if not include_annotations:
            continue
        polygon_data = _read_polygon_file(asset.polygon_path)
        polygon_width = _positive_dimension(
            polygon_data.get('imgWidth'), 'imgWidth', asset.polygon_path
        )
        polygon_height = _positive_dimension(
            polygon_data.get('imgHeight'), 'imgHeight', asset.polygon_path
        )
        if (polygon_width, polygon_height) != (width, height):
            raise ValueError(
                'image and polygon dimensions differ for {}'.format(asset.scene_id)
            )
        objects = polygon_data.get('objects')
        if not isinstance(objects, list):
            raise ValueError('objects must be a list in {}'.format(asset.polygon_path))
        for instance in objects:
            if not isinstance(instance, Mapping) or instance.get('deleted', False):
                continue
            category = _category(instance.get('label'))
            if category is None:
                continue
            geometry = _clipped_polygon(instance.get('polygon'), width, height)
            if geometry is None:
                continue
            segmentation, bbox, area = geometry
            category_id, iscrowd = category
            annotations.append(
                {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'segmentation': [list(segmentation)],
                    'bbox': list(bbox),
                    'area': area,
                    'iscrowd': iscrowd,
                }
            )
            annotation_id += 1
    return {
        'info': {'scenario': 'cityscapes-to-foggy', 'split': split_name},
        'images': images,
        'annotations': annotations,
        'categories': _categories(),
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def _empty_labeled_dataset() -> dict:
    return {
        'info': {
            'scenario': 'cityscapes-to-foggy',
            'split': 'target_train_labeled',
        },
        'images': [],
        'annotations': [],
        'categories': _categories(),
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
    if value['categories'] != _categories():
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


def materialize_target_train_labeled(
    oracle_json_path: Path,
    pool_state: PoolState,
    output_path: Path,
) -> LabeledTargetManifest:
    '''Atomically materialize annotations for exactly the committed labeled pool.'''

    selected = selected_samples_for_reveal(pool_state, TARGET_TRAIN_NAMESPACE)
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
        selected_image_ids = {
            images_by_sample[sample]['id'] for sample in selected
        }
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
    payload = _canonical_bytes(labeled_dataset)
    atomic_write_bytes(output_path, payload)
    return LabeledTargetManifest(
        path=output_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        image_count=len(labeled_dataset['images']),
        annotation_count=len(labeled_dataset['annotations']),
    )


def _input_fingerprint(
    layout: CityscapesToFoggyLayout,
    repository_root: Path,
) -> str:
    entries = []
    paths = {
        asset.image_path
        for assets in (layout.source_train, layout.target_train, layout.target_val)
        for asset in assets
    }
    polygon_paths = {
        asset.polygon_path
        for assets in (layout.source_train, layout.target_train, layout.target_val)
        for asset in assets
    }
    for path in sorted(paths, key=lambda value: value.as_posix()):
        entries.append(
            {
                'path': _repository_relative(path, repository_root),
                'size': path.stat().st_size,
            }
        )
    for path in sorted(polygon_paths, key=lambda value: value.as_posix()):
        entries.append(
            {
                'path': _repository_relative(path, repository_root),
                'size': path.stat().st_size,
                'sha256': sha256_file(path),
            }
        )
    return hashlib.sha256(_canonical_bytes({'files': entries})).hexdigest()


def _cache_directory(
    repository_root: Path,
    cache_directory: Path,
) -> Path:
    repository_root = repository_root.absolute()
    cache_root = (repository_root / 'work_dirs' / '.dataset_cache').resolve()
    candidate = Path(cache_directory)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(cache_root)
    except ValueError as error:
        raise ValueError('cache output must stay under work_dirs/.dataset_cache') from error
    return candidate


def prepare_cityscapes_to_foggy(
    clear_image_root: Path,
    foggy_image_root: Path,
    polygon_root: Path,
    cache_directory: Path,
    repository_root: Path,
    *,
    expected_train_images: Optional[int] = EXPECTED_TRAIN_IMAGES,
    expected_val_images: Optional[int] = EXPECTED_VAL_IMAGES,
) -> Mapping[str, Any]:
    '''Create the four C-to-F COCO caches and a fingerprinted manifest.'''

    repository_root = Path(repository_root).absolute()
    cache_directory = _cache_directory(repository_root, cache_directory)
    layout = validate_cityscapes_to_foggy_layout(
        clear_image_root,
        foggy_image_root,
        polygon_root,
        expected_train_images=expected_train_images,
        expected_val_images=expected_val_images,
    )
    datasets = {
        'source_train': _build_coco_dataset(
            layout.source_train,
            SOURCE_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='source_train',
        ),
        'target_train_unlabeled': _build_coco_dataset(
            layout.target_train,
            TARGET_TRAIN_NAMESPACE,
            repository_root,
            include_annotations=False,
            split_name='target_train_unlabeled',
        ),
        'target_train_oracle': _build_coco_dataset(
            layout.target_train,
            TARGET_TRAIN_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='target_train_oracle',
        ),
        'target_val': _build_coco_dataset(
            layout.target_val,
            TARGET_VAL_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='target_val',
        ),
    }

    output_records = {}
    for output_name in sorted(datasets):
        output_path = cache_directory / '{}.json'.format(output_name)
        payload = _canonical_bytes(datasets[output_name])
        atomic_write_bytes(output_path, payload)
        output_records[output_name] = {
            'path': output_path.relative_to(repository_root).as_posix(),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'images': len(datasets[output_name]['images']),
            'annotations': len(datasets[output_name]['annotations']),
        }

    manifest_body = {
        'schema_version': 1,
        'scenario': 'cityscapes-to-foggy',
        'beta': '0.02',
        'categories': _categories(),
        'input_fingerprint': _input_fingerprint(layout, repository_root),
        'outputs': output_records,
    }
    manifest = dict(manifest_body)
    manifest['fingerprint'] = hashlib.sha256(
        _canonical_bytes(manifest_body)
    ).hexdigest()
    atomic_write_bytes(cache_directory / 'manifest.json', _canonical_bytes(manifest))
    return manifest
