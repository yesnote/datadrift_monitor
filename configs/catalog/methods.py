"""Active-learning method specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .datasets import DatasetSpec, normalize_token


PAL_FULL_ALIASES = ('pal', 'pal:full', 'pal/full', 'pal:guide', 'pal/guide')
PAL_LIUS_ALIASES = ('pal:lius', 'pal/lius')
PPAL_ALIASES = ('ppal', 'ppal:dcus-ccms', 'ppal/dcus-ccms', 'ppal:full', 'ppal/full')
ECPAL_ALIASES = ('ecpal', 'ecpal:full', 'ecpal/full')
CORESET_ALIASES = ('coreset', 'core-set', 'core_set', 'kcenter', 'k-center')


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
    cfg_overrides: Dict[str, object] = field(default_factory=dict)
    round_num: int = 7
    budget: int = 414

    def default_alias(self) -> str:
        if self.method == 'pal' and self.cfg_overrides.get('pal_mode') == 'lius':
            return 'pal:lius'
        return self.method

    def output_dir(self) -> str:
        return 'work_dirs/%s' % self.output_name

    def to_config(self, dataset: DatasetSpec, gpus: int) -> Dict[str, object]:
        budget = self.budget
        cfg: Dict[str, object] = {
            'round_num': self.round_num,
            'budget': budget,
            'output_dir': self.output_dir(),
        }
        if self.key == 'ppal':
            cfg.update(_ppal_config(dataset, budget=budget, gpus=gpus))
        elif self.key == 'pal_full':
            cfg.update(_pal_config(mode='full'))
        elif self.key == 'pal_lius':
            cfg.update(_pal_config(mode='lius'))
        elif self.key == 'ecpal':
            cfg.update(_ecpal_config())
        elif self.key == 'coreset':
            cfg.update(_coreset_config())
        cfg.update(self.cfg_overrides)
        return cfg


def _ppal_config(dataset: DatasetSpec, budget: int, gpus: int) -> Dict[str, object]:
    budget_expand_ratio = 4
    uncertainty_pool_size = _uncertainty_pool_size(budget, budget_expand_ratio, gpus)
    cfg: Dict[str, object] = {
        'budget_expand_ratio': budget_expand_ratio,
        'uncertainty_pool_size': uncertainty_pool_size,
    }
    cfg.update({
        'uncertainty_sampler_config': dict(
            type='DCUSSampler',
            n_sample_images=uncertainty_pool_size,
            oracle_annotation_path=dataset.oracle_path,
            score_thr=0.05,
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


def _pal_config(mode: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        'pal_mode': mode,
        'pal_iou_threshold': 0.5,
        'pal_labeled_detections': 'pal_labeled_detections.bbox.json',
        'pal_unlabeled_detections': 'pal_unlabeled_detections.bbox.json',
    }
    if mode == 'full':
        cfg.update({
            'pal_alpha': 0.9,
            'pal_beta': 0.04,
            'pal_gamma': 0.02,
            'pal_embedding_source': 'external',
            'pal_diagnostics_file': 'pal_diagnostics.json',
        })
        embedding_path = 'work_dirs/pal_embeddings/voc_google_vit_embeddings.npy'
        cfg['pal_embedding_path'] = embedding_path
        cfg['pal_embedding_prep'] = dict(
            type='google_vit',
            output_path=embedding_path,
            model_name='google/vit-base-patch16-224-in21k',
            batch_size=16,
            device='auto',
            embedding_output='pooler',
            normalize=True,
            progress=False,
        )
    return cfg


def _ecpal_config() -> Dict[str, object]:
    return {
        'ecpal_candidate_expand_ratio': 2,
        'ecpal_foreground_iou_threshold': 0.5,
        'ecpal_background_iou_threshold': 0.1,
        'ecpal_eps': 1e-12,
        'ecpal_weight_eps': 1e-6,
        'ecpal_diagnostics_file': 'ecpal_diagnostics.json',
        'ecpal_candidates_file': 'ecpal_candidates.json',
        'ecpal_labeled_features': 'ecpal_labeled_features.json',
        'ecpal_unlabeled_features': 'ecpal_unlabeled_features.json',
    }


def _coreset_config() -> Dict[str, object]:
    return {
        'coreset_labeled_features': 'coreset_labeled_features.npz',
        'coreset_unlabeled_features': 'coreset_unlabeled_features.npz',
        'coreset_candidates_file': 'coreset_candidates.json',
        'coreset_diagnostics_file': 'coreset_diagnostics.json',
        'coreset_normalize_features': False,
        'coreset_distance_batch_size': 512,
        'coreset_center_batch_size': 2048,
    }


METHODS = (
    MethodSpec(
        key='ppal',
        method='ppal',
        aliases=PPAL_ALIASES,
        description='PPAL DCUS+CCMS acquisition.',
        output_name='retinanet_voc_ppal_7rounds_5percent_to_20percent',
    ),
    MethodSpec(
        key='pal_full',
        method='pal',
        aliases=PAL_FULL_ALIASES,
        description='PAL full LIUS+GUIDE acquisition.',
        output_name='retinanet_voc_pal_7rounds_5percent_to_20percent',
        cfg_overrides={'pal_mode': 'full'},
    ),
    MethodSpec(
        key='pal_lius',
        method='pal',
        aliases=PAL_LIUS_ALIASES,
        description='PAL LIUS-only acquisition.',
        output_name='retinanet_voc_pal_lius_7rounds_5percent_to_20percent',
        cfg_overrides={'pal_mode': 'lius'},
    ),
    MethodSpec(
        key='ecpal',
        method='ecpal',
        aliases=ECPAL_ALIASES,
        description='ECPAL error-count prediction acquisition.',
        output_name='retinanet_voc_ecpal_7rounds_5percent_to_20percent',
    ),
    MethodSpec(
        key='coreset',
        method='coreset',
        aliases=CORESET_ALIASES,
        description='Core-set greedy k-center acquisition.',
        output_name='retinanet_voc_coreset_7rounds_5percent_to_20percent',
    ),
    MethodSpec(
        key='random',
        method='random',
        aliases=('random',),
        description='Random acquisition baseline.',
        output_name='retinanet_voc_random_7rounds_5percent_to_20percent',
    ),
    MethodSpec(
        key='entropy',
        method='entropy',
        aliases=('entropy',),
        description='Entropy acquisition baseline.',
        output_name='retinanet_voc_entropy_7rounds_5percent_to_20percent',
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
