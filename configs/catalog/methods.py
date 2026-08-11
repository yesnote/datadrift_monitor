"""Active-learning method specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .datasets import DatasetSpec, normalize_token


PAL_FULL_ALIASES = ('pal', 'pal:full', 'pal/full', 'pal:guide', 'pal/guide')
PAL_LIUS_ALIASES = ('pal:lius', 'pal/lius')
PPAL_ALIASES = ('ppal', 'ppal:dcus-ccms', 'ppal/dcus-ccms', 'ppal:full', 'ppal/full')
ECPAL_ECA_ONLY_ALIASES = ('ecpal:eca-only', 'ecpal/eca-only')
ECPAL_EUA_ONLY_ALIASES = ('ecpal:eua-only', 'ecpal/eua-only')
ECPAL_ECA_FULL_ALIASES = ('ecpal:eca-full', 'ecpal/eca-full')
ECPAL_EUA_FULL_ALIASES = ('ecpal:eua-full', 'ecpal/eua-full')
CORESET_ALIASES = ('coreset', 'core-set', 'core_set', 'kcenter', 'k-center')
MIAL_ALIASES = ('mial', 'mi-aod', 'miaod')


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
    output_token: str
    cfg_overrides: Dict[str, object] = field(default_factory=dict)

    def default_alias(self) -> str:
        if self.method == 'pal' and self.cfg_overrides.get('pal_mode') == 'lius':
            return 'pal:lius'
        if self.method == 'ecpal':
            return 'ecpal:%s' % self.cfg_overrides.get('ecpal_mode')
        return self.method

    def output_dir(self, dataset: DatasetSpec, detector_name: str) -> str:
        return 'work_dirs/%s_%s_%s_%s' % (
            detector_name,
            dataset.name,
            self.output_token,
            dataset.protocol_slug,
        )

    def to_config(
        self,
        dataset: DatasetSpec,
        detector_name: str,
        gpus: int,
    ) -> Dict[str, object]:
        budget = dataset.budget
        cfg: Dict[str, object] = {
            'output_dir': self.output_dir(dataset, detector_name),
        }
        if self.key == 'ppal':
            cfg.update(_ppal_config(dataset, budget=budget, gpus=gpus))
        elif self.key == 'pal_full':
            cfg.update(_pal_config(mode='full', dataset=dataset))
        elif self.key == 'pal_lius':
            cfg.update(_pal_config(mode='lius', dataset=dataset))
        elif self.key.startswith('ecpal_'):
            cfg.update(_ecpal_config(str(self.cfg_overrides['ecpal_mode'])))
        elif self.key == 'coreset':
            cfg.update(_coreset_config())
        elif self.key == 'mial':
            cfg.update(_mial_config())
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


def _pal_config(mode: str, dataset: DatasetSpec) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        'pal_mode': mode,
        'pal_iou_threshold': 0.5,
        'pal_labeled_detections': 'pal_labeled_detections.bbox.json',
        'pal_unlabeled_detections': 'pal_unlabeled_detections.bbox.json',
    }
    if mode == 'lius':
        cfg['pal_candidate_multiplier'] = 1
    if mode == 'full':
        cfg.update({
            'pal_alpha': 0.9,
            'pal_beta': 0.04,
            'pal_gamma': 0.02,
            'pal_embedding_source': 'external',
            'pal_diagnostics_file': 'pal_diagnostics.json',
        })
        embedding_path = (
            'work_dirs/pal_embeddings/%s_google_vit_embeddings.npy'
            % dataset.name
        )
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


def _ecpal_config(mode: str) -> Dict[str, object]:
    normalized_mode = mode.lower().replace('_', '-')
    candidate_expand_ratio = 1 if normalized_mode.endswith('-only') else 2
    file_stem = 'ecpal_%s' % normalized_mode.replace('-', '_')
    return {
        'ecpal_mode': normalized_mode,
        'ecpal_candidate_expand_ratio': candidate_expand_ratio,
        'ecpal_foreground_iou_threshold': 0.5,
        'ecpal_background_iou_threshold': 0.1,
        'ecpal_eps': 1e-12,
        'ecpal_weight_eps': 1e-6,
        'ecpal_diagnostics_file': '%s_diagnostics.json' % file_stem,
        'ecpal_candidates_file': '%s_candidates.json' % file_stem,
        'ecpal_labeled_features': 'ecpal_labeled_features.json',
        'ecpal_unlabeled_features': 'ecpal_unlabeled_features.json',
    }


def _coreset_config() -> Dict[str, object]:
    return {
        'coreset_labeled_features': 'coreset_labeled_features.npz',
        'coreset_unlabeled_features': 'coreset_unlabeled_features.npz',
        'coreset_candidates_file': 'coreset_candidates.json',
        'coreset_diagnostics_file': 'coreset_diagnostics.json',
        'coreset_distance_batch_size': 512,
        'coreset_center_batch_size': 2048,
    }


def _mial_config() -> Dict[str, object]:
    return {
        'gpus': 1,
        'mial_lambda': 0.5,
        'mial_topk': 10000,
        'mial_uncertainty_file': 'mial_uncertainty.json',
        'mial_candidates_file': 'mial_candidates.json',
        'mial_diagnostics_file': 'mial_diagnostics.json',
    }


METHODS = (
    MethodSpec(
        key='ppal',
        method='ppal',
        aliases=PPAL_ALIASES,
        description='PPAL DCUS+CCMS acquisition.',
        output_token='ppal',
    ),
    MethodSpec(
        key='pal_full',
        method='pal',
        aliases=PAL_FULL_ALIASES,
        description='PAL full LIUS+GUIDE acquisition.',
        output_token='pal',
        cfg_overrides={'pal_mode': 'full'},
    ),
    MethodSpec(
        key='pal_lius',
        method='pal',
        aliases=PAL_LIUS_ALIASES,
        description='PAL LIUS-only acquisition.',
        output_token='pal_lius',
        cfg_overrides={'pal_mode': 'lius'},
    ),
    MethodSpec(
        key='ecpal_eca_only',
        method='ecpal',
        aliases=ECPAL_ECA_ONLY_ALIASES,
        description='ECPAL ECA-only uncertainty acquisition.',
        output_token='ecpal_eca_only',
        cfg_overrides={
            'ecpal_mode': 'eca-only',
        },
    ),
    MethodSpec(
        key='ecpal_eua_only',
        method='ecpal',
        aliases=ECPAL_EUA_ONLY_ALIASES,
        description='ECPAL EUA-only uncertainty acquisition.',
        output_token='ecpal_eua_only',
        cfg_overrides={
            'ecpal_mode': 'eua-only',
        },
    ),
    MethodSpec(
        key='ecpal_eca_full',
        method='ecpal',
        aliases=ECPAL_ECA_FULL_ALIASES,
        description='ECPAL ECA candidate acquisition with ECA-profile diversity.',
        output_token='ecpal_eca_full',
        cfg_overrides={
            'ecpal_mode': 'eca-full',
        },
    ),
    MethodSpec(
        key='ecpal_eua_full',
        method='ecpal',
        aliases=ECPAL_EUA_FULL_ALIASES,
        description='ECPAL EUA candidate acquisition with EUA-profile diversity.',
        output_token='ecpal_eua_full',
        cfg_overrides={
            'ecpal_mode': 'eua-full',
        },
    ),
    MethodSpec(
        key='coreset',
        method='coreset',
        aliases=CORESET_ALIASES,
        description='Core-set greedy k-center acquisition.',
        output_token='coreset',
    ),
    MethodSpec(
        key='mial',
        method='mial',
        aliases=MIAL_ALIASES,
        description='MIAL/MI-AOD instance-discrepancy acquisition.',
        output_token='mial',
    ),
    MethodSpec(
        key='random',
        method='random',
        aliases=('random',),
        description='Random acquisition baseline.',
        output_token='random',
    ),
    MethodSpec(
        key='entropy',
        method='entropy',
        aliases=('entropy',),
        description='Entropy acquisition baseline.',
        output_token='entropy',
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
