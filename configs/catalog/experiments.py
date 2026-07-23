"""Supported active-learning experiment catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .datasets import DatasetSpec, list_datasets, resolve_dataset
from .detectors import DetectorSpec, list_detectors, resolve_detector
from .methods import MethodSpec
from .methods import list_methods, normalize_method_alias, resolve_method_alias as _resolve_method_alias
from .methods import resolve_method_spec


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    method_key: str
    method: str
    detector: str
    dataset: str
    description: str
    aliases: Tuple[str, ...] = ()
    cfg_overrides: Dict[str, object] = field(default_factory=dict)
    smoke: bool = False


@dataclass(frozen=True)
class CatalogSelection:
    preset: ExperimentPreset
    method: str
    method_alias: str
    cfg_overrides: Dict[str, object] = field(default_factory=dict)
    method_spec: Optional[MethodSpec] = None
    detector_spec: Optional[DetectorSpec] = None
    dataset_spec: Optional[DatasetSpec] = None


_PRESET_NAMES = {
    ('ppal', False): 'ppal-retinanet-voc',
    ('ppal', True): 'ppal-retinanet-voc-smoke',
    ('pal_full', False): 'pal-retinanet-voc',
    ('pal_full', True): 'pal-retinanet-voc-smoke',
    ('pal_lius', False): 'pal-lius-retinanet-voc',
    ('pal_lius', True): 'pal-lius-retinanet-voc-smoke',
    ('random', False): 'random-retinanet-voc',
    ('random', True): 'random-retinanet-voc-smoke',
    ('entropy', False): 'entropy-retinanet-voc',
    ('entropy', True): 'entropy-retinanet-voc-smoke',
}


def _preset_aliases(method: MethodSpec, smoke: bool) -> Tuple[str, ...]:
    if smoke:
        return tuple('smoke:' + value for value in method.aliases)
    return method.aliases


def _build_preset(method: MethodSpec, detector: DetectorSpec, dataset: DatasetSpec, smoke: bool) -> ExperimentPreset:
    key = (method.key, smoke)
    method_description = method.description.rstrip('.')
    return ExperimentPreset(
        name=_PRESET_NAMES[key],
        method_key=method.key,
        method=method.method,
        detector=detector.name,
        dataset=dataset.name,
        description='%s on %s/%s.' % (method_description, detector.name, dataset.name.upper()),
        aliases=_preset_aliases(method, smoke),
        cfg_overrides=dict(method.cfg_overrides),
        smoke=smoke,
    )


def list_presets() -> List[ExperimentPreset]:
    presets = []
    for smoke in (False, True):
        for detector in list_detectors():
            for dataset in list_datasets(smoke=smoke):
                for method in list_methods():
                    presets.append(_build_preset(method, detector, dataset, smoke=smoke))
    return presets


def resolve_method_alias(method: str) -> Optional[Tuple[str, Dict[str, object]]]:
    return _resolve_method_alias(method)


def _preset_names(preset: ExperimentPreset) -> Iterable[str]:
    yield preset.name
    for alias in preset.aliases:
        yield alias


def _find_preset_by_name(name: str) -> Optional[ExperimentPreset]:
    normalized = normalize_method_alias(name)
    for preset in list_presets():
        if normalized in {normalize_method_alias(value) for value in _preset_names(preset)}:
            return preset
    return None


def _find_preset_by_combo(method: str, detector: str, dataset: str, smoke: bool) -> Optional[ExperimentPreset]:
    method_spec = resolve_method_spec(method)
    detector_spec = resolve_detector(detector)
    dataset_spec = resolve_dataset(dataset, smoke=smoke)
    if method_spec is None or detector_spec is None or dataset_spec is None:
        return None
    return _build_preset(method_spec, detector_spec, dataset_spec, smoke=smoke)


def _selection_from_preset(preset: ExperimentPreset, method_alias: Optional[str] = None) -> Optional[CatalogSelection]:
    method_spec = resolve_method_spec(method_alias or preset.method_key)
    detector_spec = resolve_detector(preset.detector)
    dataset_spec = resolve_dataset(preset.dataset, smoke=preset.smoke)
    if method_spec is None or detector_spec is None or dataset_spec is None:
        return None
    return CatalogSelection(
        preset=preset,
        method=method_spec.method,
        method_alias=normalize_method_alias(method_alias or method_spec.default_alias()),
        cfg_overrides=dict(method_spec.cfg_overrides),
        method_spec=method_spec,
        detector_spec=detector_spec,
        dataset_spec=dataset_spec,
    )


def resolve_experiment(
    method: Optional[str] = None,
    detector: Optional[str] = None,
    dataset: Optional[str] = None,
    preset: Optional[str] = None,
    smoke: bool = False,
) -> Optional[CatalogSelection]:
    """Resolve a user-facing selection to an experiment config and base method."""

    if preset:
        matched = _find_preset_by_name(preset)
        if matched is None:
            return None
        return _selection_from_preset(matched, method_alias=method)

    if not method or not detector or not dataset:
        return None

    matched = _find_preset_by_combo(method, detector, dataset, smoke)
    if matched is None:
        return None

    return _selection_from_preset(matched, method_alias=method)


def _dedupe(values: Iterable[object]) -> List[object]:
    result = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def build_experiment_config(selection: CatalogSelection) -> Dict[str, object]:
    """Build runner config from catalog specs instead of experiment globals."""

    if selection.method_spec is None or selection.detector_spec is None or selection.dataset_spec is None:
        raise ValueError('Catalog selection is missing resolved specs.')

    smoke = selection.preset.smoke
    gpus = 1 if smoke else 8
    cfg: Dict[str, object] = {
        'python_path': 'python',
        'port': 29500,
        'gpus': gpus,
    }
    cfg.update(selection.dataset_spec.to_config())
    detector_cfg = selection.detector_spec.to_config(smoke=smoke)
    detector_required = detector_cfg.pop('required_files', [])
    cfg.update(detector_cfg)
    method_cfg = selection.method_spec.to_config(selection.dataset_spec, smoke=smoke, gpus=gpus)
    method_required = method_cfg.pop('required_files', [])
    cfg.update(method_cfg)
    cfg.update(selection.cfg_overrides)
    cfg['required_files'] = _dedupe(list(detector_required) + list(method_required))
    return cfg
