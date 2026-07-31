"""Training log parsing helpers for the ALOD metrics dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.common.metrics_scanner import RunRef


TRAIN_LOG_KEY_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*):\s*([-+0-9.eE]+)')


def _seed_dir(run_dir: Path, seed: int, legacy: bool = False) -> Path:
    return run_dir if legacy else run_dir / ('seed_%d' % int(seed))


def _round_dir(run: RunRef, seed: int, round_index: int) -> Path:
    return _seed_dir(Path(run.path), seed, legacy=run.legacy) / ('round_%02d' % int(round_index))


def _candidate_json_logs(round_dir: Path) -> List[Path]:
    return sorted(round_dir.glob('*.log.json'))


def _json_log_has_train_records(path: Path) -> bool:
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('mode') == 'train':
                    return True
    except OSError:
        return False
    return False


def _latest_train_json_log(round_dir: Path) -> Optional[Path]:
    candidates = [path for path in _candidate_json_logs(round_dir) if _json_log_has_train_records(path)]
    return candidates[-1] if candidates else None


def _numeric_items(record: Dict[str, Any]) -> Dict[str, float]:
    ignored = {'epoch', 'iter', 'memory', 'data_time', 'time'}
    return {
        key: float(value)
        for key, value in record.items()
        if key not in ignored and isinstance(value, (int, float))
    }


def _parse_json_train_log(
    path: Path,
    run: RunRef,
    seed: int,
    round_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    step = 0
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('mode') != 'train':
                continue
            step += 1
            epoch = int(record.get('epoch', 0) or 0)
            iteration = int(record.get('iter', 0) or 0)
            for key, value in _numeric_items(record).items():
                rows.append(_train_row(
                    run=run,
                    seed=seed,
                    round_index=round_index,
                    epoch=epoch,
                    iteration=iteration,
                    local_step=step,
                    key=key,
                    value=value,
                    source=str(path),
                ))
    return rows


def _parse_text_train_log(
    path: Path,
    run: RunRef,
    seed: int,
    round_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    step = 0
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            pairs = dict(TRAIN_LOG_KEY_RE.findall(line))
            if not pairs:
                continue
            numeric: Dict[str, float] = {}
            for key, value in pairs.items():
                try:
                    numeric[key] = float(value)
                except ValueError:
                    continue
            metric_keys = {
                key for key in numeric
                if key == 'loss' or key.startswith('loss_') or key in ('lr', 'grad_norm')
            }
            if not metric_keys:
                continue
            step += 1
            epoch = int(numeric.get('epoch', 0))
            iteration = int(numeric.get('iter', step))
            for key in sorted(metric_keys):
                rows.append(_train_row(
                    run=run,
                    seed=seed,
                    round_index=round_index,
                    epoch=epoch,
                    iteration=iteration,
                    local_step=step,
                    key=key,
                    value=numeric[key],
                    source=str(path),
                ))
    return rows


def _train_row(
    run: RunRef,
    seed: int,
    round_index: int,
    epoch: int,
    iteration: int,
    local_step: int,
    key: str,
    value: float,
    source: str,
) -> Dict[str, Any]:
    seed_label = 'seed_%d' % int(seed)
    curve_label = '%s %s %s R%02d %s' % (
        run.method.upper(),
        run.run_id,
        seed_label,
        int(round_index),
        key,
    )
    return {
        'run_path': run.path,
        'experiment': run.experiment,
        'run_id': run.run_id,
        'method': run.method.upper(),
        'detector': run.detector,
        'dataset': run.dataset,
        'seed': seed_label,
        'round': int(round_index),
        'epoch': int(epoch),
        'iter': int(iteration),
        'local_step': int(local_step),
        'key': key,
        'value': float(value),
        'curve_label': curve_label,
        'source': source,
    }


def available_rounds(run: RunRef) -> List[int]:
    rounds = set()
    seeds = run.seeds or (0,)
    for seed in seeds:
        seed_path = _seed_dir(Path(run.path), int(seed), legacy=run.legacy)
        for child in seed_path.glob('round_*'):
            if not child.is_dir():
                continue
            try:
                round_index = int(child.name.split('_', 1)[1])
            except ValueError:
                continue
            if round_index > 0:
                rounds.add(round_index)
    return sorted(rounds)


def available_train_keys(runs: Sequence[RunRef], max_files: int = 8) -> List[str]:
    keys = set()
    checked = 0
    for run in runs:
        for seed in run.seeds or (0,):
            for round_index in available_rounds(run):
                path = _latest_train_json_log(_round_dir(run, int(seed), round_index))
                if path is None:
                    continue
                for row in _parse_json_train_log(path, run, int(seed), round_index):
                    keys.add(row['key'])
                checked += 1
                if checked >= max_files:
                    return sorted(keys)
    return sorted(keys)


def load_train_frame(
    runs: Sequence[RunRef],
    seeds: Optional[Iterable[str]] = None,
    rounds: Optional[Iterable[int]] = None,
):
    import pandas as pd

    seed_filter = set(seeds) if seeds else None
    round_filter = {int(round_index) for round_index in rounds} if rounds else None
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for seed in run.seeds or (0,):
            seed_label = 'seed_%d' % int(seed)
            if seed_filter is not None and seed_label not in seed_filter:
                continue
            for round_index in available_rounds(run):
                if round_filter is not None and int(round_index) not in round_filter:
                    continue
                round_path = _round_dir(run, int(seed), round_index)
                json_log = _latest_train_json_log(round_path)
                if json_log is not None:
                    rows.extend(_parse_json_train_log(json_log, run, int(seed), round_index))
                    continue
                text_log = round_path / 'logs' / 'train.log'
                if text_log.exists():
                    rows.extend(_parse_text_train_log(text_log, run, int(seed), round_index))
    return pd.DataFrame(rows)

