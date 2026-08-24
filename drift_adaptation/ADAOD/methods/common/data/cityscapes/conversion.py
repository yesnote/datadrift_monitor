'''Deterministic Cityscapes polygon conversion for the C-to-F scenario.'''

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

from methods.common.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from methods.common.data.image_identity import SampleIdentity

if TYPE_CHECKING:
    from methods.common.progress import ProgressReporter

from .layout import (
    ANNOTATION_POLICY,
    CATEGORY_IDS,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_VAL_IMAGES,
    SOURCE_NAMESPACE,
    TARGET_TRAIN_NAMESPACE,
    TARGET_VAL_NAMESPACE,
    CityscapesToFoggyLayout,
    SceneAsset,
    cityscapes_categories,
    validate_cityscapes_to_foggy_layout,
)


CACHE_SCHEMA_VERSION = 2


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


def _category(label: Any) -> Optional[int]:
    if not isinstance(label, str):
        return None
    return CATEGORY_IDS.get(label)


def _pt_bbox_from_polygon(
    polygon: Any,
    width: int,
    height: int,
) -> Optional[Tuple[Sequence[float], float]]:
    '''Return the box produced by PT's VOC converter and Detectron2 loader.

    PT clips its serialized VOC coordinates to ``[1, size - 1]`` only when
    they cross an image boundary. Detectron2 subtracts one from ``xmin`` and
    ``ymin`` while retaining the inclusive VOC maxima as exclusive upper
    bounds. The returned COCO xywh box reproduces those effective zero-based,
    half-open xyxy coordinates without materializing an intermediate XML.
    '''

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
        points.append((x, y))
    if not points:
        return None
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min <= 0.0:
        x_min = 1.0
    if x_max >= float(width):
        x_max = float(width - 1)
    if y_min <= 0.0:
        y_min = 1.0
    if y_max >= float(height):
        y_max = float(height - 1)

    # Detectron2's Pascal VOC reader shifts only the lower bounds. The VOC
    # inclusive maxima then serve as half-open upper bounds.
    x_min -= 1.0
    y_min -= 1.0
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    area = box_width * box_height
    return (x_min, y_min, box_width, box_height), area


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError(
            'dataset paths must be addressed through repository-relative junctions'
        ) from error


def _build_coco_dataset(
    assets: Sequence[SceneAsset],
    namespace: str,
    repository_root: Path,
    *,
    include_annotations: bool,
    split_name: str,
    progress: Optional['ProgressReporter'] = None,
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
            if progress is not None:
                progress.advance()
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
            geometry = _pt_bbox_from_polygon(instance.get('polygon'), width, height)
            if geometry is None:
                continue
            bbox, area = geometry
            annotations.append(
                {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category,
                    'bbox': list(bbox),
                    'area': area,
                    'iscrowd': 0,
                }
            )
            annotation_id += 1
        if progress is not None:
            progress.advance()
    return {
        'info': {'scenario': 'cityscapes-to-foggy', 'split': split_name},
        'images': images,
        'annotations': annotations,
        'categories': cityscapes_categories(),
    }


def _input_fingerprint(
    layout: CityscapesToFoggyLayout,
    repository_root: Path,
    progress: Optional['ProgressReporter'] = None,
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
    if progress is not None:
        progress.start_task(len(paths) + len(polygon_paths), 'file')
    for path in sorted(paths, key=lambda value: value.as_posix()):
        entries.append(
            {
                'path': _repository_relative(path, repository_root),
                'size': path.stat().st_size,
            }
        )
        if progress is not None:
            progress.advance()
    for path in sorted(polygon_paths, key=lambda value: value.as_posix()):
        entries.append(
            {
                'path': _repository_relative(path, repository_root),
                'size': path.stat().st_size,
                'sha256': sha256_file(path),
            }
        )
        if progress is not None:
            progress.advance()
    return sha256_bytes(canonical_json_bytes({'files': entries}))


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
    progress: Optional['ProgressReporter'] = None,
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
        progress=progress,
    )
    if progress is not None:
        progress.start_task(
            len(layout.source_train)
            + len(layout.target_train) * 2
            + len(layout.target_val),
            'image',
        )
    datasets = {
        'source_train': _build_coco_dataset(
            layout.source_train,
            SOURCE_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='source_train',
            progress=progress,
        ),
        'target_train_unlabeled': _build_coco_dataset(
            layout.target_train,
            TARGET_TRAIN_NAMESPACE,
            repository_root,
            include_annotations=False,
            split_name='target_train_unlabeled',
            progress=progress,
        ),
        'target_train_oracle': _build_coco_dataset(
            layout.target_train,
            TARGET_TRAIN_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='target_train_oracle',
            progress=progress,
        ),
        'target_val': _build_coco_dataset(
            layout.target_val,
            TARGET_VAL_NAMESPACE,
            repository_root,
            include_annotations=True,
            split_name='target_val',
            progress=progress,
        ),
    }

    output_records = {}
    if progress is not None:
        progress.start_task(len(datasets), 'file')
    for output_name in sorted(datasets):
        output_path = cache_directory / '{}.json'.format(output_name)
        payload = canonical_json_bytes(datasets[output_name])
        atomic_write_bytes(output_path, payload)
        output_records[output_name] = {
            'path': output_path.relative_to(repository_root).as_posix(),
            'sha256': sha256_bytes(payload),
            'images': len(datasets[output_name]['images']),
            'annotations': len(datasets[output_name]['annotations']),
        }
        if progress is not None:
            progress.advance()

    manifest_body = {
        'schema_version': CACHE_SCHEMA_VERSION,
        'scenario': 'cityscapes-to-foggy',
        'beta': '0.02',
        'annotation_policy': dict(ANNOTATION_POLICY),
        'categories': cityscapes_categories(),
        'input_fingerprint': _input_fingerprint(
            layout,
            repository_root,
            progress,
        ),
        'outputs': output_records,
    }
    manifest = dict(manifest_body)
    manifest['fingerprint'] = sha256_bytes(canonical_json_bytes(manifest_body))
    if progress is not None:
        progress.start_task(1, 'file')
    atomic_write_bytes(
        cache_directory / 'manifest.json', canonical_json_bytes(manifest)
    )
    if progress is not None:
        progress.advance()
    return manifest
