"""Shared active-learning diagnostics helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return to_jsonable(value.tolist())
    if hasattr(value, 'item'):
        return value.item()
    return value


def write_diagnostics(path: Path, diagnostics: Dict[str, Any]) -> Path:
    """Write a diagnostics dictionary as stable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(to_jsonable(diagnostics), handle, indent=2, ensure_ascii=False)
    return path


def print_acquisition_summary(
    method: str,
    round_index: int,
    selected_count: int,
    diagnostics_path: Path,
    stage: Optional[str] = None,
    labeled_json: Optional[Path] = None,
    unlabeled_json: Optional[Path] = None,
) -> None:
    """Emit concise, method-neutral acquisition output."""

    parts = ['ALOD acquisition: method=%s' % method]
    if stage:
        parts.append('stage=%s' % stage)
    parts.append('round=%d' % round_index)
    parts.append('selected=%d' % selected_count)
    print(' '.join(parts), flush=True)
    print('wrote diagnostics: %s' % diagnostics_path, flush=True)
    if labeled_json is not None:
        print('wrote labeled pool: %s' % labeled_json, flush=True)
    if unlabeled_json is not None:
        print('wrote unlabeled pool: %s' % unlabeled_json, flush=True)
