"""Prepare tiny deterministic PASCAL VOC pools for pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle)


def subset_from_ids(
    oracle: Dict[str, Any],
    selected_image_ids: Iterable[int],
    include_annotations: bool,
) -> Dict[str, Any]:
    selected = set(int(image_id) for image_id in selected_image_ids)
    subset = {
        'categories': oracle['categories'],
        'images': [
            image for image in oracle['images']
            if int(image['id']) in selected
        ],
        'annotations': [],
    }
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle['annotations']
            if int(ann['image_id']) in selected
        ]
    return subset


def read_voc_split_ids(vocdevkit: Path, year: str, split: str) -> List[str]:
    split_file = vocdevkit / ('VOC%s' % year) / 'ImageSets' / 'Main' / ('%s.txt' % split)
    if not split_file.exists():
        raise FileNotFoundError('VOC split file does not exist: %s' % split_file)
    return [
        line.strip().split()[0]
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def write_smoke_test_split(
    vocdevkit: Path,
    output_path: Path,
    n_test: int,
    seed: int,
    year: str = '2007',
    split: str = 'test',
) -> None:
    if n_test <= 0:
        raise ValueError('n_test must be positive')
    sample_ids = read_voc_split_ids(vocdevkit, year=year, split=split)
    if n_test > len(sample_ids):
        raise ValueError('n_test must be <= VOC%s %s size' % (year, split))
    rng = np.random.RandomState(seed)
    selected = [sample_ids[int(index)] for index in rng.permutation(len(sample_ids))[:n_test]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(selected) + '\n', encoding='utf-8')


def prepare_smoke_pools(
    oracle: Dict[str, Any],
    output_dir: Path,
    n_labeled: int,
    n_unlabeled: int,
    seed: int,
) -> Dict[str, Path]:
    if n_labeled <= 0:
        raise ValueError('n_labeled must be positive')
    if n_unlabeled <= 0:
        raise ValueError('n_unlabeled must be positive')
    total = n_labeled + n_unlabeled
    if total > len(oracle['images']):
        raise ValueError('n_labeled + n_unlabeled exceeds oracle image count')

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(oracle['images']))
    labeled_indices = permutation[:n_labeled]
    unlabeled_indices = permutation[n_labeled:total]
    labeled_ids = [int(oracle['images'][int(index)]['id']) for index in labeled_indices]
    unlabeled_ids = [int(oracle['images'][int(index)]['id']) for index in unlabeled_indices]
    smoke_ids = labeled_ids + unlabeled_ids

    oracle_path = output_dir / ('smoke_oracle_%d_seed%d.json' % (total, seed))
    labeled_path = output_dir / ('smoke_voc_labeled_%d_seed%d.json' % (n_labeled, seed))
    unlabeled_path = output_dir / ('smoke_voc_unlabeled_%d_seed%d.json' % (n_unlabeled, seed))

    smoke_oracle = subset_from_ids(oracle, smoke_ids, include_annotations=True)
    write_json(smoke_oracle, oracle_path)
    write_json(subset_from_ids(smoke_oracle, labeled_ids, include_annotations=True), labeled_path)
    write_json(subset_from_ids(smoke_oracle, unlabeled_ids, include_annotations=False), unlabeled_path)

    return {
        'oracle': oracle_path,
        'labeled': labeled_path,
        'unlabeled': unlabeled_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare tiny VOC active-learning smoke files')
    parser.add_argument('--oracle-input', type=Path, default=Path('data/VOC0712/annotations/trainval_0712.json'))
    parser.add_argument('--vocdevkit', type=Path, default=Path('data/VOCdevkit'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/active_learning/voc_smoke'))
    parser.add_argument('--n-labeled', type=int, default=8)
    parser.add_argument('--n-unlabeled', type=int, default=8)
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle = load_json(args.oracle_input)
    paths = prepare_smoke_pools(
        oracle,
        output_dir=args.output_dir,
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        seed=args.seed,
    )
    test_path = args.output_dir / ('voc2007_test_%d_seed%d.txt' % (args.n_test, args.seed))
    write_smoke_test_split(args.vocdevkit, test_path, n_test=args.n_test, seed=args.seed)

    print('Smoke oracle: %s' % paths['oracle'])
    print('Smoke labeled: %s' % paths['labeled'])
    print('Smoke unlabeled: %s' % paths['unlabeled'])
    print('Smoke VOC2007 test split: %s' % test_path)
    print('Smoke counts: labeled=%d unlabeled=%d test=%d' % (
        args.n_labeled,
        args.n_unlabeled,
        args.n_test,
    ))


if __name__ == '__main__':
    main()
