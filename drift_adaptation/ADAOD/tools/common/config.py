'''Deterministic composition of method and method-neutral catalogs.'''

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping, MutableMapping, Optional

from configs.catalog import get_dataset, get_detector, get_runtime
from methods.common.contracts import MethodManifest

from .paths import repository_relative_path


def _deep_merge(
    base: MutableMapping[str, Any],
    overlay: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = deepcopy(dict(base[key]))
            base[key] = _deep_merge(nested, value)
        else:
            base[key] = deepcopy(value)
    return base


def _validate_repository_paths(config: Mapping[str, Any]) -> None:
    dataset = config['dataset']
    for domain in ('source', 'target'):
        domain_config = dataset[domain]
        repository_relative_path(domain_config['image_root'])
        repository_relative_path(domain_config['annotation_root'])
    runtime = config['runtime']
    repository_relative_path(runtime['work_root'])
    repository_relative_path(runtime['dataset_cache_root'])


def compose_config(
    manifest: MethodManifest,
    dataset_key: Optional[str] = None,
    detector_key: Optional[str] = None,
    runtime_key: str = 'default',
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    '''Compose one independent config without method-specific branches.'''

    method_config = manifest.config_factory()
    if not isinstance(method_config, Mapping):
        raise TypeError('method config factory must return a mapping')
    config = deepcopy(dict(method_config))
    if config.get('method') != manifest.key:
        raise ValueError('method config key does not match its manifest')

    selected_dataset = dataset_key or config.get('scenario')
    if not selected_dataset:
        raise ValueError('dataset scenario must be selected')
    configured_detector = config.get('detector', {})
    selected_detector = detector_key or configured_detector.get('name')
    if not selected_detector:
        raise ValueError('detector must be selected')

    dataset_config = get_dataset(str(selected_dataset))
    detector_config = get_detector(str(selected_detector))
    runtime_config = get_runtime(runtime_key)
    config['scenario'] = str(selected_dataset)
    config['dataset'] = _deep_merge(
        deepcopy(dict(config.get('dataset', {}))), dataset_config
    )
    config['detector'] = _deep_merge(
        deepcopy(dict(configured_detector)), detector_config
    )
    config['runtime'] = deepcopy(dict(runtime_config))
    if overrides:
        config = dict(_deep_merge(config, overrides))

    if tuple(config['dataset']['classes']) != tuple(dataset_config['classes']):
        raise ValueError('method and dataset class mappings do not match')
    if config['detector']['num_classes'] != len(config['dataset']['classes']):
        raise ValueError('detector class count does not match the dataset')
    _validate_repository_paths(config)
    return config


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'unsupported config value: {type(value).__name__}')


def config_fingerprint(config: Mapping[str, Any]) -> str:
    '''Return a stable SHA256 fingerprint for a resolved configuration.'''

    serialized = json.dumps(
        config,
        default=_json_default,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    return sha256(serialized.encode('utf-8')).hexdigest()

