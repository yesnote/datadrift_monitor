"""PPAL acquisition orchestration using local method modules."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List

from methods.common.results import acquisition_result


@dataclass(frozen=True)
class PPALStep:
    name: str
    description: str


PPAL_STEPS = (
    PPALStep('train', 'Train RetinaNet with the labeled pool.'),
    PPALStep('eval', 'Evaluate the current checkpoint on VOC test data.'),
    PPALStep('uncertainty_inference', 'Run RetinaHeadUncertainty on unlabeled data.'),
    PPALStep('dcus_acquisition', 'Select the expanded uncertainty pool with DCUS.'),
    PPALStep('diversity_inference', 'Export image distance features for the uncertainty pool.'),
    PPALStep('diversity_acquisition', 'Select the final budget with CCMS.'),
)

SAMPLER_TYPES = {
    'DCUSSampler': 'dcus',
    'DCUS': 'dcus',
    'DiversitySampler': 'ccms',
    'CCMS': 'ccms',
}


def describe_steps() -> List[Dict[str, str]]:
    return [dict(name=step.name, description=step.description) for step in PPAL_STEPS]


def sampler_config_names() -> Iterable[str]:
    return ('uncertainty_sampler_config', 'diversity_sampler_config')


def local_ppal_available() -> bool:
    return find_spec('methods.ppal') is not None


def _resolve_sampler_config(sampler_config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    resolved = dict(sampler_config)
    oracle_path = resolved.get('oracle_annotation_path')
    if oracle_path:
        oracle = Path(str(oracle_path))
        if not oracle.is_absolute():
            resolved['oracle_annotation_path'] = str((repo_root / oracle).resolve())
    return resolved


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError('%s does not exist: %s' % (description, path))


def build_local_sampler(sampler_config: Dict[str, Any], repo_root: Path) -> Any:
    """Build a PPAL sampler from local method modules."""

    resolved = _resolve_sampler_config(sampler_config, repo_root)
    sampler_type = resolved.pop('type', None)
    if sampler_type not in SAMPLER_TYPES:
        raise KeyError('Unsupported PPAL sampler type: %s' % sampler_type)
    if SAMPLER_TYPES[sampler_type] == 'dcus':
        from methods.ppal.dcus import DCUSSampler

        return DCUSSampler(**resolved)
    if SAMPLER_TYPES[sampler_type] == 'ccms':
        from methods.ppal.ccms import DiversitySampler

        return DiversitySampler(**resolved)
    raise KeyError('Unsupported PPAL sampler type: %s' % sampler_type)


def run_uncertainty_acquisition(
    cfg: Dict[str, Any],
    repo_root: Path,
    round_index: int,
    result_json: Path,
    last_labeled_json: Path,
    out_labeled_json: Path,
    out_unlabeled_json: Path,
) -> Dict[str, Any]:
    """Run PPAL DCUS acquisition with the local method implementation."""

    _require_file(result_json, 'PPAL uncertainty inference result')
    _require_file(last_labeled_json, 'PPAL previous labeled pool')
    _require_file(result_json.parent / 'latest.pth', 'PPAL checkpoint for class quality')
    out_labeled_json.parent.mkdir(parents=True, exist_ok=True)

    sampler = build_local_sampler(cfg['uncertainty_sampler_config'], repo_root)
    if hasattr(sampler, 'set_round'):
        sampler.set_round(round_index)
    result = sampler.al_round(
        str(result_json),
        str(last_labeled_json),
        str(out_labeled_json),
        str(out_unlabeled_json),
    )
    metrics = dict(result.get('metrics', {}))
    selected = result.get('selected_image_ids', [])
    out_labeled = str(out_labeled_json)
    out_unlabeled = str(out_unlabeled_json)
    return acquisition_result(
        method='ppal',
        stage='dcus',
        runner_stage='uncertainty',
        round_index=round_index,
        budget=int(cfg['uncertainty_sampler_config'].get('n_sample_images', 0)),
        selected_image_ids=selected,
        inputs={
            'labeled_pool_json': str(last_labeled_json),
            'uncertainty_result_json': str(result_json),
            'class_quality_checkpoint': str(result_json.parent / 'latest.pth'),
        },
        outputs={
            'labeled_pool_json': out_labeled,
            'unlabeled_pool_json': out_unlabeled,
        },
        metrics=metrics,
        out_labeled_json=out_labeled,
        out_unlabeled_json=out_unlabeled,
    )


def run_diversity_acquisition(
    cfg: Dict[str, Any],
    repo_root: Path,
    round_index: int,
    result_json: Path,
    image_distance_npy: Path,
    last_labeled_json: Path,
    out_labeled_json: Path,
    out_unlabeled_json: Path,
) -> Dict[str, Any]:
    """Run PPAL diversity acquisition with the local method implementation."""

    _require_file(result_json, 'PPAL uncertainty inference result')
    _require_file(image_distance_npy, 'PPAL diversity image distance cache')
    _require_file(last_labeled_json, 'PPAL previous labeled pool')
    out_labeled_json.parent.mkdir(parents=True, exist_ok=True)

    sampler = build_local_sampler(cfg['diversity_sampler_config'], repo_root)
    if hasattr(sampler, 'set_round'):
        sampler.set_round(round_index)
    result = sampler.al_round(
        str(result_json),
        str(image_distance_npy),
        str(last_labeled_json),
        str(out_labeled_json),
        str(out_unlabeled_json),
    )
    metrics = dict(result.get('metrics', {}))
    selected = result.get('selected_image_ids', [])
    out_labeled = str(out_labeled_json)
    out_unlabeled = str(out_unlabeled_json)
    return acquisition_result(
        method='ppal',
        stage='ccms',
        runner_stage='diversity',
        round_index=round_index,
        budget=int(cfg['diversity_sampler_config'].get('n_sample_images', 0)),
        selected_image_ids=selected,
        inputs={
            'labeled_pool_json': str(last_labeled_json),
            'uncertainty_result_json': str(result_json),
            'image_distance_npy': str(image_distance_npy),
        },
        outputs={
            'labeled_pool_json': out_labeled,
            'unlabeled_pool_json': out_unlabeled,
        },
        metrics=metrics,
        out_labeled_json=out_labeled,
        out_unlabeled_json=out_unlabeled,
    )
