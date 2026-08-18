'''Prepare deterministic C-to-F COCO caches from repository dataset junctions.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Optional, Sequence


if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from configs.catalog import get_dataset, get_runtime
from methods.common.data.cityscapes import (
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_VAL_IMAGES,
    prepare_cityscapes_to_foggy,
)
from tools.common.paths import repository_relative_path, repository_root


def build_parser() -> argparse.ArgumentParser:
    dataset = get_dataset('cityscapes-to-foggy')
    runtime = get_runtime('default')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--clear-images',
        default=dataset['source']['image_root'],
        help='repository-relative clear Cityscapes junction',
    )
    parser.add_argument(
        '--foggy-images',
        default=dataset['target']['image_root'],
        help='repository-relative Foggy Cityscapes junction',
    )
    parser.add_argument(
        '--polygons',
        default=dataset['source']['annotation_root'],
        help='repository-relative gtFine junction',
    )
    parser.add_argument(
        '--cache-directory',
        default='{}/cityscapes-to-foggy'.format(runtime['dataset_cache_root']),
        help='repository-relative output below work_dirs/.dataset_cache',
    )
    return parser


def _dataset_junction(repository: Path, value: str) -> Path:
    relative = PurePosixPath(repository_relative_path(value))
    if len(relative.parts) < 3 or relative.parts[:2] != ('data', 'Cityscapes'):
        raise ValueError(
            'dataset inputs must be addressed below data/Cityscapes'
        )
    return repository.joinpath(*relative.parts)


def _repository_output(repository: Path, value: str) -> Path:
    relative = PurePosixPath(repository_relative_path(value))
    return repository.joinpath(*relative.parts)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    repository_root_path: Optional[Path] = None,
    expected_train_images: int = EXPECTED_TRAIN_IMAGES,
    expected_val_images: int = EXPECTED_VAL_IMAGES,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = (repository_root_path or repository_root()).absolute()
    try:
        clear_images = _dataset_junction(root, args.clear_images)
        foggy_images = _dataset_junction(root, args.foggy_images)
        polygons = _dataset_junction(root, args.polygons)
        cache_directory = _repository_output(root, args.cache_directory)
        manifest = prepare_cityscapes_to_foggy(
            clear_images,
            foggy_images,
            polygons,
            cache_directory,
            root,
            expected_train_images=expected_train_images,
            expected_val_images=expected_val_images,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
