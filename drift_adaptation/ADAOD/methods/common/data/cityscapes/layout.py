'''Cityscapes-to-Foggy taxonomy and dataset layout validation.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from methods.common.progress import ProgressReporter


CITYSCAPES_CLASSES: Tuple[str, ...] = (
    'truck',
    'car',
    'rider',
    'person',
    'train',
    'motorcycle',
    'bicycle',
    'bus',
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
ANNOTATION_POLICY = {
    'name': 'pt-voc-8class-exact-v1',
    'label_matching': 'exact',
    'group_labels': 'excluded',
    'coordinates': 'pt-voc-inclusive-to-zero-based-half-open-coco-xywh',
}


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


def cityscapes_categories() -> list:
    '''Return the canonical eight-class C-to-F COCO taxonomy.'''

    return [
        {'id': CATEGORY_IDS[name], 'name': name, 'supercategory': 'object'}
        for name in CITYSCAPES_CLASSES
    ]


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError('{} directory does not exist: {}'.format(description, path))


def _inventory(
    root: Path,
    split: str,
    suffix: str,
    progress: Optional['ProgressReporter'] = None,
) -> Dict[str, Path]:
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
        if progress is not None:
            progress.advance()
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
    progress: Optional['ProgressReporter'] = None,
) -> CityscapesToFoggyLayout:
    '''Validate exact counts and scene bijections needed by C-to-F only.'''

    clear_image_root = Path(clear_image_root)
    foggy_image_root = Path(foggy_image_root)
    polygon_root = Path(polygon_root)
    _require_directory(clear_image_root, 'clear Cityscapes image root')
    _require_directory(foggy_image_root, 'Foggy Cityscapes image root')
    _require_directory(polygon_root, 'gtFine polygon root')

    if progress is not None:
        progress.start_task(None, 'file')
    clear_train = _inventory(
        clear_image_root, 'train', CLEAR_SUFFIX, progress)
    foggy_train = _inventory(
        foggy_image_root, 'train', FOGGY_002_SUFFIX, progress)
    foggy_val = _inventory(
        foggy_image_root, 'val', FOGGY_002_SUFFIX, progress)
    polygon_train = _inventory(
        polygon_root, 'train', POLYGON_SUFFIX, progress)
    polygon_val = _inventory(
        polygon_root, 'val', POLYGON_SUFFIX, progress)

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
