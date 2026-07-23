"""Prepare PASCAL VOC files required by the PPAL/PAL active-learning runner.

The script mirrors the PPAL reference data preparation behavior without
importing MMDetection/MMCV. It converts VOC2007+VOC2012 trainval annotations to
a COCO-style oracle JSON, then creates deterministic initial labeled/unlabeled
pool JSON files.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)
VOC_CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}


def _read_split_ids(voc_year_dir: Path, split: str) -> List[str]:
    split_file = voc_year_dir / 'ImageSets' / 'Main' / ('%s.txt' % split)
    if not split_file.exists():
        raise FileNotFoundError('VOC split file does not exist: %s' % split_file)
    return [
        line.strip().split()[0]
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _parse_object(obj: ET.Element) -> Tuple[int, List[int], int]:
    name = obj.findtext('name')
    if name not in VOC_CLASS_TO_ID:
        raise ValueError('Unsupported VOC class name: %s' % name)
    difficult = int(obj.findtext('difficult', default='0'))
    box = obj.find('bndbox')
    if box is None:
        raise ValueError('VOC object is missing bndbox')

    # Match the PPAL/MMDetection converter: VOC 1-based coordinates become
    # 0-based x1/y1/x2/y2 before converting to COCO xywh.
    x1 = int(box.findtext('xmin')) - 1
    y1 = int(box.findtext('ymin')) - 1
    x2 = int(box.findtext('xmax')) - 1
    y2 = int(box.findtext('ymax')) - 1
    return VOC_CLASS_TO_ID[name], [x1, y1, x2, y2], difficult


def _segmentation_from_xyxy(bbox: Sequence[int]) -> List[List[int]]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return [[x1, y1, x1, y2, x2, y2, x2, y1]]


def _annotation_from_xyxy(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox_xyxy: Sequence[int],
    difficult: int,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return {
        'segmentation': _segmentation_from_xyxy(bbox_xyxy),
        'area': int(width * height),
        'ignore': 0,
        'iscrowd': 1 if difficult else 0,
        'image_id': int(image_id),
        'bbox': [x1, y1, width, height],
        'category_id': int(category_id),
        'id': int(annotation_id),
    }


def _parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, List[int], int]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    if size is None:
        raise ValueError('VOC XML is missing size: %s' % xml_path)
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    objects = [_parse_object(obj) for obj in root.findall('object')]
    return width, height, objects


def _categories() -> List[Dict[str, Any]]:
    return [
        {'supercategory': 'none', 'id': int(index), 'name': name}
        for index, name in enumerate(VOC_CLASSES)
    ]


def build_voc0712_oracle(vocdevkit: Path, split: str = 'trainval') -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    annotation_id = 0
    image_id = 0

    for year in ('2007', '2012'):
        year_dir = vocdevkit / ('VOC%s' % year)
        if not year_dir.exists():
            raise FileNotFoundError('Expected VOC year directory: %s' % year_dir)

        for sample_id in _read_split_ids(year_dir, split):
            xml_path = year_dir / 'Annotations' / ('%s.xml' % sample_id)
            width, height, objects = _parse_voc_xml(xml_path)
            images.append({
                'id': int(image_id),
                'file_name': 'VOC%s/JPEGImages/%s.jpg' % (year, sample_id),
                'height': int(height),
                'width': int(width),
            })
            for category_id, bbox_xyxy, difficult in objects:
                annotations.append(_annotation_from_xyxy(
                    annotation_id,
                    image_id,
                    category_id,
                    bbox_xyxy,
                    difficult,
                ))
                annotation_id += 1
            image_id += 1

    return {
        'images': images,
        'type': 'instance',
        'categories': _categories(),
        'annotations': annotations,
    }


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle)


def _subset_from_ids(
    oracle: Dict[str, Any],
    selected_image_ids: Iterable[int],
    include_annotations: bool,
) -> Dict[str, Any]:
    selected = set(int(image_id) for image_id in selected_image_ids)
    subset = {
        'categories': oracle['categories'],
        'images': [image for image in oracle['images'] if int(image['id']) in selected],
        'annotations': [],
    }
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle['annotations']
            if int(ann['image_id']) in selected
        ]
    return subset


def write_initial_splits(
    oracle: Dict[str, Any],
    output_dir: Path,
    n_labeled: int,
    n_diff: int,
    seed: int,
    dataset_prefix: str = 'voc',
) -> List[Tuple[Path, Path]]:
    if n_labeled <= 0:
        raise ValueError('n_labeled must be positive')
    all_images = oracle['images']
    if n_labeled >= len(all_images):
        raise ValueError('n_labeled must be smaller than image count')

    rng = np.random.RandomState(seed)
    written = []
    for split_index in range(1, n_diff + 1):
        permutation = rng.permutation(len(all_images))
        labeled_indices = set(int(index) for index in permutation[:n_labeled])
        labeled_ids = [int(all_images[index]['id']) for index in labeled_indices]
        unlabeled_ids = [
            int(image['id']) for index, image in enumerate(all_images)
            if index not in labeled_indices
        ]

        stem = '%s_%d' % (dataset_prefix, n_labeled)
        labeled_path = output_dir / ('%s_labeled_%d.json' % (stem, split_index))
        unlabeled_path = output_dir / ('%s_unlabeled_%d.json' % (stem, split_index))
        write_json(_subset_from_ids(oracle, labeled_ids, include_annotations=True), labeled_path)
        write_json(_subset_from_ids(oracle, unlabeled_ids, include_annotations=False), unlabeled_path)
        written.append((labeled_path, unlabeled_path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare VOC active-learning JSON files')
    parser.add_argument('--vocdevkit', type=Path, default=Path('data/VOCdevkit'))
    parser.add_argument('--oracle-output', type=Path, default=Path('data/VOC0712/annotations/trainval_0712.json'))
    parser.add_argument('--split-output-dir', type=Path, default=Path('data/active_learning/voc'))
    parser.add_argument('--n-labeled', type=int, default=827)
    parser.add_argument('--n-diff', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', default='trainval')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle = build_voc0712_oracle(args.vocdevkit, split=args.split)
    write_json(oracle, args.oracle_output)
    written = write_initial_splits(
        oracle,
        output_dir=args.split_output_dir,
        n_labeled=args.n_labeled,
        n_diff=args.n_diff,
        seed=args.seed,
    )

    print('Oracle: %s' % args.oracle_output)
    print('Images: %d' % len(oracle['images']))
    print('Annotations: %d' % len(oracle['annotations']))
    for labeled_path, unlabeled_path in written:
        print('Labeled: %s' % labeled_path)
        print('Unlabeled: %s' % unlabeled_path)
    print('Image file names are relative to VOCdevkit-style roots, e.g. VOC2007/JPEGImages/000001.jpg.')


if __name__ == '__main__':
    main()
