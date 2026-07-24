"""Shared active-learning diagnostics helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from methods.common.io import to_jsonable, write_json


def write_diagnostics(path: Path, diagnostics: Dict[str, Any]) -> Path:
    """Write a diagnostics dictionary as stable UTF-8 JSON."""

    return write_json(path, diagnostics, indent=2)


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
