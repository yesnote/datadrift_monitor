'''Deterministic method-manifest discovery.'''

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Mapping, Optional

from .contracts import MethodManifest


def discover_methods(
    methods_root: Optional[Path] = None,
) -> Mapping[str, MethodManifest]:
    '''Discover concrete manifests without a hard-coded method list.'''

    root = methods_root or Path(__file__).resolve().parents[1]
    discovered: Dict[str, MethodManifest] = {}
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or directory.name.startswith(('_', '.')):
            continue
        if directory.name == 'common' or not (directory / 'manifest.py').is_file():
            continue
        module = importlib.import_module(f'methods.{directory.name}.manifest')
        manifest = getattr(module, 'MANIFEST', None)
        if not isinstance(manifest, MethodManifest):
            raise TypeError(f'{module.__name__}.MANIFEST must be a MethodManifest')
        if manifest.key in discovered:
            raise ValueError(f'duplicate method key: {manifest.key}')
        discovered[manifest.key] = manifest
    return discovered
