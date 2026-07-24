"""Automatic input preparation for the active-learning runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tools.common.pal_embeddings import ensure_pal_embeddings
from tools.common.paths import assert_not_code_refs
from tools.common.pretrained import ensure_pretrained
from tools.common.voc import ensure_voc_active_learning


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    resolved = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    assert_not_code_refs(resolved, root)
    return resolved


def _ensure_dataset(cfg: Mapping[str, Any], root: Path) -> Optional[Dict[str, object]]:
    dataset_prep = cfg.get('dataset_prep')
    if not isinstance(dataset_prep, dict):
        return None
    kind = str(dataset_prep.get('type', '')).lower()
    if kind != 'voc0712':
        raise ValueError('Unsupported dataset preparation type: %s'
                         % dataset_prep.get('type'))
    return ensure_voc_active_learning(dataset_prep, root)


def _ensure_pretrained(cfg: Mapping[str, Any], root: Path) -> Optional[Dict[str, object]]:
    pretrained = cfg.get('pretrained')
    if not isinstance(pretrained, dict):
        return None
    return ensure_pretrained(pretrained, root)


def _ensure_pal_embeddings(cfg: Mapping[str, Any], root: Path) -> Optional[Dict[str, object]]:
    embedding_prep = cfg.get('pal_embedding_prep')
    if not isinstance(embedding_prep, dict):
        return None
    if str(cfg.get('pal_embedding_source', '')).lower() != 'external':
        return None
    if str(cfg.get('pal_mode', '')).lower() not in ('full', 'guide'):
        return None

    ann_paths = [
        _resolve_path(cfg['init_label_json'], root),
        _resolve_path(cfg['init_unlabeled_json'], root),
    ]
    image_root = _resolve_path(cfg.get('image_root', 'data/VOCdevkit'), root)
    for path in ann_paths + [image_root]:
        assert_not_code_refs(path, root)
    return ensure_pal_embeddings(embedding_prep, ann_paths, image_root, root)


def prepare_required_inputs(
    cfg: Mapping[str, Any],
    root: Path,
) -> List[Dict[str, object]]:
    """Prepare all catalog-declared inputs needed before AL rounds start."""

    results: List[Dict[str, object]] = []
    for step in (_ensure_dataset, _ensure_pretrained, _ensure_pal_embeddings):
        result = step(cfg, root)
        if result is not None:
            results.append(result)
    return results
