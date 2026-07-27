"""Dataset specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    dataset_type: str
    oracle_path: str
    init_label_json: str
    init_unlabeled_json: str
    image_root: str
    test_ann_file: str
    test_img_prefix: str
    dataset_prep: Dict[str, object] = field(default_factory=dict)
    mmdet_common_cfg_options: Dict[str, object] = field(default_factory=dict)
    mmdet_eval_cfg_options: Dict[str, object] = field(default_factory=dict)

    def to_config(self) -> Dict[str, object]:
        common_options = {
            'data.train.img_prefix': self.image_root,
            'data.val.ann_file': self.test_ann_file,
            'data.val.img_prefix': self.test_img_prefix,
            'data.test.img_prefix': self.image_root,
        }
        common_options.update(self.mmdet_common_cfg_options)

        eval_options = {
            'data.test.ann_file': self.test_ann_file,
            'data.test.img_prefix': self.test_img_prefix,
        }
        eval_options.update(self.mmdet_eval_cfg_options)

        cfg = {
            'oracle_path': self.oracle_path,
            'init_label_json': self.init_label_json,
            'init_unlabeled_json': self.init_unlabeled_json,
            'init_model': None,
            'image_root': self.image_root,
            'mmdet_common_cfg_options': common_options,
            'mmdet_eval_cfg_options': eval_options,
        }
        if self.dataset_prep:
            cfg['dataset_prep'] = dict(self.dataset_prep)
        return cfg


VOC = DatasetSpec(
    name='voc',
    aliases=('voc', 'pascal-voc', 'pascal_voc'),
    dataset_type='voc',
    oracle_path='data/VOC0712/annotations/trainval_0712.json',
    init_label_json='data/active_learning/voc/voc_827_labeled_1.json',
    init_unlabeled_json='data/active_learning/voc/voc_827_unlabeled_1.json',
    image_root='data/VOCdevkit/',
    test_ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
    test_img_prefix='data/VOCdevkit/VOC2007/',
    dataset_prep=dict(
        type='voc0712',
        vocdevkit='data/VOCdevkit',
        oracle_output='data/VOC0712/annotations/trainval_0712.json',
        split_output_dir='data/active_learning/voc',
        n_labeled=827,
        n_diff=1,
        seed=0,
        split='trainval',
        dataset_prefix='voc',
    ),
)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace('_', '-')


def list_datasets() -> List[DatasetSpec]:
    return [VOC]


def resolve_dataset(name: str) -> Optional[DatasetSpec]:
    normalized = normalize_token(name)
    for dataset in list_datasets():
        aliases = {normalize_token(value) for value in dataset.aliases}
        if normalized == dataset.name or normalized in aliases:
            return dataset
    return None
