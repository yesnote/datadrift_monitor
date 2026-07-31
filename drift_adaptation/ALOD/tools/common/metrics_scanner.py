"""Discover ALOD experiment runs under ``work_dirs`` for metric viewing."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TIMESTAMP_RE = re.compile(r'^\d{2}-\d{2}-\d{4}_\d{2};\d{2}$')


@dataclass(frozen=True)
class RunRef:
    path: str
    experiment: str
    run_id: str
    method: str
    detector: str
    dataset: str
    seeds: tuple
    rounds: Optional[int] = None
    budget: Optional[int] = None
    status: str = 'unknown'
    has_aggregate: bool = False
    legacy: bool = False

    @property
    def label(self) -> str:
        return '%s / %s / %s / %s' % (
            self.method.upper(),
            self.detector,
            self.dataset,
            self.run_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _experiment_metadata_from_name(name: str) -> Dict[str, str]:
    parts = name.split('_')
    detector = parts[0] if len(parts) > 0 else 'unknown'
    dataset = parts[1] if len(parts) > 1 else 'unknown'
    method = parts[2] if len(parts) > 2 else 'unknown'
    return {'method': method, 'detector': detector, 'dataset': dataset}


def _seed_number(path: Path) -> Optional[int]:
    if not path.name.startswith('seed_'):
        return None
    try:
        return int(path.name.split('_', 1)[1])
    except ValueError:
        return None


def _seed_dirs(run_dir: Path) -> List[Path]:
    seeds = []
    for child in run_dir.iterdir() if run_dir.exists() else []:
        if child.is_dir() and _seed_number(child) is not None:
            seeds.append(child)
    return sorted(seeds, key=lambda path: _seed_number(path) or 0)


def _status_from_seed_runs(seed_runs: Sequence[Dict[str, Any]]) -> str:
    if not seed_runs:
        return 'unknown'
    statuses = {str(run.get('status', 'unknown')) for run in seed_runs}
    if len(statuses) == 1:
        return statuses.pop()
    if 'failed' in statuses:
        return 'failed'
    if 'running' in statuses:
        return 'running'
    return 'mixed'


def _run_from_aggregate(path: Path) -> Optional[RunRef]:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    run_dir = path.parent
    experiment = run_dir.parent.name
    fallback = _experiment_metadata_from_name(experiment)
    seeds = tuple(int(seed) for seed in data.get('seeds', []))
    seed_runs = data.get('seed_runs', [])
    if not seeds and seed_runs:
        seeds = tuple(int(run['seed']) for run in seed_runs if 'seed' in run)
    if not seeds:
        seeds = tuple(
            seed for seed in (_seed_number(seed_dir) for seed_dir in _seed_dirs(run_dir))
            if seed is not None
        )

    return RunRef(
        path=str(run_dir),
        experiment=experiment,
        run_id=str(data.get('run_id') or run_dir.name),
        method=str(data.get('method') or fallback['method']).lower(),
        detector=str(data.get('detector') or fallback['detector']).lower(),
        dataset=str(data.get('dataset') or fallback['dataset']).lower(),
        seeds=tuple(sorted(seeds)),
        rounds=data.get('rounds'),
        budget=data.get('budget'),
        status=_status_from_seed_runs(seed_runs),
        has_aggregate=True,
        legacy=False,
    )


def _run_from_run_summary(path: Path) -> Optional[RunRef]:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    summary_dir = path.parent
    if summary_dir.name.startswith('seed_'):
        run_dir = summary_dir.parent
        experiment = run_dir.parent.name
        run_id = run_dir.name
        seed_dirs = _seed_dirs(run_dir)
        seeds = tuple(
            seed for seed in (_seed_number(seed_dir) for seed_dir in seed_dirs)
            if seed is not None
        )
        legacy = False
    else:
        run_dir = summary_dir
        if TIMESTAMP_RE.match(run_dir.name):
            experiment = run_dir.parent.name
            run_id = run_dir.name
        else:
            experiment = run_dir.name
            run_id = run_dir.name
        seed = data.get('seed')
        seeds = (int(seed),) if seed is not None else tuple()
        legacy = True

    fallback = _experiment_metadata_from_name(experiment)
    return RunRef(
        path=str(run_dir),
        experiment=experiment,
        run_id=run_id,
        method=str(data.get('method') or fallback['method']).lower(),
        detector=str(data.get('detector') or fallback['detector']).lower(),
        dataset=str(data.get('dataset') or fallback['dataset']).lower(),
        seeds=tuple(sorted(seeds)),
        rounds=data.get('rounds'),
        budget=data.get('budget'),
        status=str(data.get('status', 'unknown')),
        has_aggregate=(run_dir / 'aggregate_summary.json').exists(),
        legacy=legacy,
    )


def scan_runs(work_dir: Path) -> List[RunRef]:
    """Return discovered ALOD run references below ``work_dir``."""

    root = Path(work_dir)
    if not root.exists():
        return []

    discovered: Dict[Path, RunRef] = {}
    for aggregate_path in root.rglob('aggregate_summary.json'):
        run = _run_from_aggregate(aggregate_path)
        if run is not None:
            discovered[Path(run.path).resolve()] = run

    for summary_path in root.rglob('run_summary.json'):
        summary_dir = summary_path.parent
        run_dir = summary_dir.parent if summary_dir.name.startswith('seed_') else summary_dir
        resolved = run_dir.resolve()
        if resolved in discovered:
            continue
        run = _run_from_run_summary(summary_path)
        if run is not None:
            discovered[Path(run.path).resolve()] = run

    return sorted(
        discovered.values(),
        key=lambda run: (run.method, run.detector, run.dataset, run.experiment, run.run_id),
    )


def unique_values(runs: Iterable[RunRef], field: str) -> List[str]:
    values = {str(getattr(run, field)) for run in runs}
    return sorted(values)
