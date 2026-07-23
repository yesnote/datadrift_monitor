"""Dataset specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


VOC_PREP_HINT = (
    'prepare VOC pools with: python -B datasets/prepare_voc_active_learning.py '
    '--vocdevkit data/VOCdevkit --n-labeled 827 --n-diff 1 --seed 0'
)
SMOKE_PREP_HINT = (
    'prepare smoke VOC pools with: python -B datasets/prepare_voc_smoke_active_learning.py '
    '--oracle-input data/VOC0712/annotations/trainval_0712.json '
    '--vocdevkit data/VOCdevkit --n-labeled 8 --n-unlabeled 8 --n-test 8 --seed 0'
)


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
    initial_pool_prep_hint: Optional[str] = None
    common_cfg_options: Dict[str, object] = field(default_factory=dict)
    eval_cfg_options: Dict[str, object] = field(default_factory=dict)

    def to_config(self) -> Dict[str, object]:
        common_options = {
            'data.train.img_prefix': self.image_root,
            'data.val.ann_file': self.test_ann_file,
            'data.val.img_prefix': self.test_img_prefix,
            'data.test.img_prefix': self.image_root,
        }
        common_options.update(self.common_cfg_options)

        eval_options = {
            'data.test.ann_file': self.test_ann_file,
            'data.test.img_prefix': self.test_img_prefix,
        }
        eval_options.update(self.eval_cfg_options)

        cfg = {
            'oracle_path': self.oracle_path,
            'init_label_json': self.init_label_json,
            'init_unlabeled_json': self.init_unlabeled_json,
            'init_model': None,
            'image_root': self.image_root,
            'common_cfg_options': common_options,
            'eval_cfg_options': eval_options,
        }
        if self.initial_pool_prep_hint:
            cfg['initial_pool_prep_hint'] = self.initial_pool_prep_hint
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
    initial_pool_prep_hint=VOC_PREP_HINT,
)

VOC_SMOKE = DatasetSpec(
    name='voc',
    aliases=VOC.aliases,
    dataset_type='voc',
    oracle_path='data/active_learning/voc_smoke/smoke_oracle_16_seed0.json',
    init_label_json='data/active_learning/voc_smoke/smoke_voc_labeled_8_seed0.json',
    init_unlabeled_json='data/active_learning/voc_smoke/smoke_voc_unlabeled_8_seed0.json',
    image_root='data/VOCdevkit/',
    test_ann_file='data/active_learning/voc_smoke/voc2007_test_8_seed0.txt',
    test_img_prefix='data/VOCdevkit/VOC2007/',
    initial_pool_prep_hint=SMOKE_PREP_HINT,
    common_cfg_options={'data.workers_per_gpu': 0},
    eval_cfg_options={'data.workers_per_gpu': 0},
)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace('_', '-')


def list_datasets(smoke: bool = False) -> List[DatasetSpec]:
    return [VOC_SMOKE if smoke else VOC]


def resolve_dataset(name: str, smoke: bool = False) -> Optional[DatasetSpec]:
    normalized = normalize_token(name)
    for dataset in list_datasets(smoke=smoke):
        aliases = {normalize_token(value) for value in dataset.aliases}
        if normalized == dataset.name or normalized in aliases:
            return dataset
    return None
