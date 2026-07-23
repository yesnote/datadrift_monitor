"""Active-learning method specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .datasets import DatasetSpec, normalize_token


PAL_FULL_ALIASES = ('pal', 'pal:full', 'pal/full', 'pal:guide', 'pal/guide')
PAL_LIUS_ALIASES = ('pal:lius', 'pal/lius')
PPAL_ALIASES = ('ppal', 'ppal:dcus-ccms', 'ppal/dcus-ccms', 'ppal:full', 'ppal/full')


def normalize_method_alias(value: str) -> str:
    return normalize_token(value).replace('\\', '/')


def _uncertainty_pool_size(budget: int, budget_expand_ratio: int, gpus: int) -> int:
    expanded = budget * budget_expand_ratio
    return expanded + gpus - (expanded % gpus)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    method: str
    aliases: Tuple[str, ...]
    description: str
    output_name: str
    smoke_output_name: str
    cfg_overrides: Dict[str, object] = field(default_factory=dict)
    round_num: int = 7
    smoke_round_num: int = 1
    budget: int = 414
    smoke_budget: int = 2

    def default_alias(self) -> str:
        if self.method == 'pal' and self.cfg_overrides.get('pal_mode') == 'lius':
            return 'pal:lius'
        return self.method

    def output_dir(self, smoke: bool) -> str:
        return 'work_dirs/%s' % (self.smoke_output_name if smoke else self.output_name)

    def to_config(self, dataset: DatasetSpec, smoke: bool, gpus: int) -> Dict[str, object]:
        budget = self.smoke_budget if smoke else self.budget
        cfg: Dict[str, object] = {
            'round_num': self.smoke_round_num if smoke else self.round_num,
            'budget': budget,
            'output_dir': self.output_dir(smoke),
        }
        if self.key == 'ppal':
            cfg.update(_ppal_config(dataset, budget=budget, smoke=smoke, gpus=gpus))
        elif self.key == 'pal_full':
            cfg.update(_pal_config(mode='full', smoke=smoke))
        elif self.key == 'pal_lius':
            cfg.update(_pal_config(mode='lius', smoke=smoke))
        cfg.update(self.cfg_overrides)
        return cfg


def _ppal_config(dataset: DatasetSpec, budget: int, smoke: bool, gpus: int) -> Dict[str, object]:
    if smoke:
        uncertainty_pool_size = 4
        score_thr = 0.0
        cfg: Dict[str, object] = {'uncertainty_pool_size': uncertainty_pool_size}
    else:
        budget_expand_ratio = 4
        uncertainty_pool_size = _uncertainty_pool_size(budget, budget_expand_ratio, gpus)
        score_thr = 0.05
        cfg = {
            'budget_expand_ratio': budget_expand_ratio,
            'uncertainty_pool_size': uncertainty_pool_size,
        }
    cfg.update({
        'uncertainty_sampler_config': dict(
            type='DCUSSampler',
            n_sample_images=uncertainty_pool_size,
            oracle_annotation_path=dataset.oracle_path,
            score_thr=score_thr,
            class_weight_ub=0.2,
            class_weight_alpha=0.3,
            dataset_type=dataset.dataset_type,
        ),
        'diversity_sampler_config': dict(
            type='DiversitySampler',
            n_sample_images=budget,
            oracle_annotation_path=dataset.oracle_path,
            dataset_type=dataset.dataset_type,
        ),
    })
    return cfg


def _pal_config(mode: str, smoke: bool) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        'pal_mode': mode,
        'pal_iou_threshold': 0.5,
        'pal_labeled_detections': 'pal_labeled_detections.bbox.json',
        'pal_unlabeled_detections': 'pal_unlabeled_detections.bbox.json',
    }
    if smoke:
        cfg['pal_cfg_options'] = {
            'model.test_cfg.score_thr': 0.0,
            'model.test_cfg.max_per_img': 50,
        }
    if mode == 'full':
        cfg.update({
            'pal_alpha': 0.9,
            'pal_beta': 0.04,
            'pal_gamma': 0.02,
            'pal_embedding_source': 'detection' if smoke else 'external',
            'pal_diagnostics_file': 'pal_diagnostics.json',
        })
        if not smoke:
            embedding_path = 'work_dirs/pal_embeddings/voc_google_vit_embeddings.npy'
            cfg['pal_embedding_path'] = embedding_path
            cfg['required_files'] = [embedding_path]
    return cfg


METHODS = (
    MethodSpec(
        key='ppal',
        method='ppal',
        aliases=PPAL_ALIASES,
        description='PPAL DCUS+CCMS acquisition.',
        output_name='retinanet_voc_ppal_7rounds_5percent_to_20percent',
        smoke_output_name='smoke_retinanet_voc_ppal_1round',
    ),
    MethodSpec(
        key='pal_full',
        method='pal',
        aliases=PAL_FULL_ALIASES,
        description='PAL full LIUS+GUIDE acquisition.',
        output_name='retinanet_voc_pal_7rounds_5percent_to_20percent',
        smoke_output_name='smoke_retinanet_voc_pal_guide_1round',
        cfg_overrides={'pal_mode': 'full'},
    ),
    MethodSpec(
        key='pal_lius',
        method='pal',
        aliases=PAL_LIUS_ALIASES,
        description='PAL LIUS-only acquisition.',
        output_name='retinanet_voc_pal_lius_7rounds_5percent_to_20percent',
        smoke_output_name='smoke_retinanet_voc_pal_lius_1round',
        cfg_overrides={'pal_mode': 'lius'},
    ),
    MethodSpec(
        key='random',
        method='random',
        aliases=('random',),
        description='Random acquisition baseline.',
        output_name='retinanet_voc_random_7rounds_5percent_to_20percent',
        smoke_output_name='smoke_retinanet_voc_random_1round',
    ),
    MethodSpec(
        key='entropy',
        method='entropy',
        aliases=('entropy',),
        description='Entropy acquisition baseline.',
        output_name='retinanet_voc_entropy_7rounds_5percent_to_20percent',
        smoke_output_name='smoke_retinanet_voc_entropy_1round',
    ),
)


def list_methods() -> List[MethodSpec]:
    return list(METHODS)


def resolve_method_spec(method: str) -> Optional[MethodSpec]:
    normalized = normalize_method_alias(method)
    for spec in METHODS:
        aliases = {normalize_method_alias(value) for value in spec.aliases}
        if normalized == normalize_method_alias(spec.key) or normalized in aliases:
            return spec
    return None


def resolve_method_alias(method: str) -> Optional[Tuple[str, Dict[str, object]]]:
    spec = resolve_method_spec(method)
    if spec is None:
        return None
    return spec.method, dict(spec.cfg_overrides)
