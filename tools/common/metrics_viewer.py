"""Metric parsing helpers for the ALOD Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tools.common.metrics_scanner import RunRef


def _read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _seed_dir(run_dir: Path, seed: int, legacy: bool = False) -> Path:
    if legacy:
        return run_dir
    return run_dir / ('seed_%d' % int(seed))


def _latest_eval_json(round_dir: Path) -> Optional[Path]:
    files = sorted(round_dir.glob('eval_*.json'))
    return files[-1] if files else None


def _load_eval_metrics(seed_dir: Path, round_index: int) -> Dict[str, float]:
    eval_json = _latest_eval_json(seed_dir / ('round_%02d' % round_index))
    if eval_json is None:
        return {}
    try:
        data = _read_json(eval_json)
    except (OSError, json.JSONDecodeError):
        return {}
    metric = data.get('metric', data)
    return {
        key: float(value)
        for key, value in metric.items()
        if isinstance(value, (int, float))
    }


def _labeled_count(seed_dir: Path, round_index: int) -> Optional[int]:
    if round_index <= 0:
        path = seed_dir / 'round_00' / 'annotations' / 'labeled.json'
    else:
        path = seed_dir / ('round_%02d' % round_index) / 'annotations' / 'new_labeled.json'
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return len(data.get('images', []))


def _round_duration(seed_dir: Path, round_index: int) -> Optional[float]:
    path = seed_dir / ('round_%02d' % round_index) / 'round_summary.json'
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    steps = data.get('steps', [])
    if not isinstance(steps, list):
        return None
    durations = [
        float(step.get('duration_sec', 0.0))
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get('duration_sec'), (int, float))
    ]
    return sum(durations) if durations else None


def _round_indices_from_run(run: RunRef) -> List[int]:
    run_dir = Path(run.path)
    if run.rounds:
        return list(range(1, int(run.rounds) + 1))
    round_dirs = set()
    for seed in run.seeds or (0,):
        seed_path = _seed_dir(run_dir, int(seed), legacy=run.legacy)
        for child in seed_path.glob('round_*'):
            if child.is_dir():
                try:
                    index = int(child.name.split('_', 1)[1])
                except ValueError:
                    continue
                if index > 0:
                    round_dirs.add(index)
    return sorted(round_dirs)


def _aggregate_rows(run: RunRef) -> List[Dict[str, Any]]:
    path = Path(run.path) / 'aggregate_summary.json'
    if not path.exists():
        return []
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return []

    rows: List[Dict[str, Any]] = []
    run_dir = Path(run.path)
    for round_summary in data.get('rounds_summary', []):
        round_index = int(round_summary.get('round_index', 0))
        if round_index <= 0:
            continue
        metrics = round_summary.get('metrics', {})
        for metric_name, metric_data in metrics.items():
            values = metric_data.get('values', {})
            for seed_text, value in values.items():
                try:
                    seed = int(seed_text)
                except ValueError:
                    continue
                labeled_images = _labeled_count(
                    _seed_dir(run_dir, seed, legacy=False),
                    round_index,
                )
                rows.append(_metric_row(
                    run,
                    seed='seed_%d' % seed,
                    series_type='seed',
                    round_index=round_index,
                    labeled_images=labeled_images,
                    metric=metric_name,
                    value=float(value),
                    std=None,
                ))

            mean_value = metric_data.get('mean')
            if mean_value is not None:
                seed_counts = [
                    _labeled_count(_seed_dir(run_dir, int(seed), legacy=False), round_index)
                    for seed in values.keys()
                    if str(seed).isdigit()
                ]
                seed_counts = [count for count in seed_counts if count is not None]
                labeled_images = mean(seed_counts) if seed_counts else None
                rows.append(_metric_row(
                    run,
                    seed='mean',
                    series_type='mean',
                    round_index=round_index,
                    labeled_images=labeled_images,
                    metric=metric_name,
                    value=float(mean_value),
                    std=(float(metric_data['std']) if metric_data.get('std') is not None else None),
                ))
    return rows


def _fallback_metric_rows(run: RunRef) -> List[Dict[str, Any]]:
    run_dir = Path(run.path)
    round_indices = _round_indices_from_run(run)
    seeds = run.seeds or (0,)
    seed_metric_values: Dict[tuple, List[float]] = {}
    rows: List[Dict[str, Any]] = []

    for round_index in round_indices:
        for seed in seeds:
            seed_path = _seed_dir(run_dir, int(seed), legacy=run.legacy)
            metrics = _load_eval_metrics(seed_path, round_index)
            labeled_images = _labeled_count(seed_path, round_index)
            for metric_name, value in metrics.items():
                rows.append(_metric_row(
                    run,
                    seed='seed_%d' % int(seed),
                    series_type='seed',
                    round_index=round_index,
                    labeled_images=labeled_images,
                    metric=metric_name,
                    value=float(value),
                    std=None,
                ))
                seed_metric_values.setdefault((round_index, metric_name), []).append(float(value))

    for (round_index, metric_name), values in sorted(seed_metric_values.items()):
        seed_counts = [
            _labeled_count(_seed_dir(run_dir, int(seed), legacy=run.legacy), round_index)
            for seed in seeds
        ]
        seed_counts = [count for count in seed_counts if count is not None]
        rows.append(_metric_row(
            run,
            seed='mean',
            series_type='mean',
            round_index=round_index,
            labeled_images=(mean(seed_counts) if seed_counts else None),
            metric=metric_name,
            value=mean(values),
            std=(pstdev(values) if len(values) > 1 else 0.0),
        ))
    return rows


def _metric_row(
    run: RunRef,
    seed: str,
    series_type: str,
    round_index: int,
    labeled_images: Optional[float],
    metric: str,
    value: float,
    std: Optional[float],
) -> Dict[str, Any]:
    curve_label = '%s %s %s' % (run.method.upper(), run.run_id, seed)
    if metric:
        curve_label = '%s %s' % (curve_label, metric)
    row = {
        'run_path': run.path,
        'experiment': run.experiment,
        'run_id': run.run_id,
        'method': run.method.upper(),
        'detector': run.detector,
        'dataset': run.dataset,
        'seed': seed,
        'series_type': series_type,
        'round': int(round_index),
        'labeled_images': labeled_images,
        'metric': metric,
        'value': float(value),
        'std': std,
        'curve_label': curve_label,
    }
    if std is not None:
        row['value_low'] = float(value) - float(std)
        row['value_high'] = float(value) + float(std)
    else:
        row['value_low'] = None
        row['value_high'] = None
    return row


def load_validation_frame(runs: Sequence[RunRef]):
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for run in runs:
        aggregate = _aggregate_rows(run)
        rows.extend(aggregate if aggregate else _fallback_metric_rows(run))
    return pd.DataFrame(rows)


def load_round_summary_frame(runs: Sequence[RunRef]):
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run.path)
        for seed in run.seeds or (0,):
            seed_path = _seed_dir(run_dir, int(seed), legacy=run.legacy)
            for round_index in _round_indices_from_run(run):
                metrics = _load_eval_metrics(seed_path, round_index)
                rows.append({
                    'method': run.method.upper(),
                    'detector': run.detector,
                    'dataset': run.dataset,
                    'run_id': run.run_id,
                    'seed': 'seed_%d' % int(seed),
                    'round': round_index,
                    'labeled_images': _labeled_count(seed_path, round_index),
                    'duration_min': (
                        _round_duration(seed_path, round_index) / 60.0
                        if _round_duration(seed_path, round_index) is not None else None
                    ),
                    **metrics,
                })
    return pd.DataFrame(rows)

