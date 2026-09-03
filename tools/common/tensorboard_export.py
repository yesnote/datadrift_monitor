"""Export ALOD experiment metrics to TensorBoard event files."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type

from tools.common.metrics_scanner import RunRef, scan_runs


EVENT_FILE_PATTERN = 'events.out.tfevents.*'
TRAIN_LOG_KEY_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*):\s*([-+0-9.eE]+)')
IGNORED_TRAIN_KEYS = {'epoch', 'iter', 'memory', 'data_time', 'time'}


@dataclass(frozen=True)
class TensorboardExportSummary:
    run_path: str
    log_dir: str
    seed_count: int
    round_count: int
    training_points: int
    active_learning_points: int
    aggregate_points: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_tensorboard() -> Type[Any]:
    """Return ``SummaryWriter`` or raise an actionable dependency error."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            'TensorBoard is required. Install the ALOD dependencies with '
            '`pip install -r requirements.txt`.'
        ) from exc
    return SummaryWriter


def _read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _seed_source_dir(run: RunRef, seed: int) -> Path:
    run_dir = Path(run.path)
    return run_dir if run.legacy else run_dir / ('seed_%d' % seed)


def _round_dirs(seed_dir: Path) -> List[Tuple[int, Path]]:
    rounds = []
    for path in seed_dir.glob('round_*'):
        if not path.is_dir():
            continue
        try:
            round_index = int(path.name.split('_', 1)[1])
        except ValueError:
            continue
        if round_index > 0:
            rounds.append((round_index, path))
    return sorted(rounds)


def _candidate_train_json_logs(round_dir: Path) -> List[Path]:
    return sorted(round_dir.glob('*.log.json'), reverse=True)


def _json_train_records(path: Path) -> List[Dict[str, float]]:
    records = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get('mode') != 'train':
                    continue
                record = {
                    key: number
                    for key, value in payload.items()
                    if key not in IGNORED_TRAIN_KEYS
                    and (number := _numeric(value)) is not None
                }
                if record:
                    records.append(record)
    except OSError:
        return []
    return records


def _text_train_records(path: Path) -> List[Dict[str, float]]:
    records = []
    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                record = {}
                for key, value in TRAIN_LOG_KEY_RE.findall(line):
                    if key in IGNORED_TRAIN_KEYS:
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if math.isfinite(number):
                        record[key] = number
                if record and any(
                    key == 'loss' or key.startswith('loss_') or key in ('lr', 'grad_norm')
                    for key in record
                ):
                    records.append(record)
    except OSError:
        return []
    return records


def _train_records(round_dir: Path) -> List[Dict[str, float]]:
    for path in _candidate_train_json_logs(round_dir):
        records = _json_train_records(path)
        if records:
            return records
    text_log = round_dir / 'logs' / 'train.log'
    return _text_train_records(text_log) if text_log.exists() else []


def _latest_eval_metrics(round_dir: Path) -> Dict[str, float]:
    paths = sorted(round_dir.glob('eval_*.json'), reverse=True)
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        metrics = payload.get('metric', payload) if isinstance(payload, Mapping) else {}
        result = {
            str(key): number
            for key, value in metrics.items()
            if (number := _numeric(value)) is not None
        }
        if result:
            return result
    return {}


def _count_annotation_items(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    if path.suffix.lower() == '.json':
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        images = payload.get('images') if isinstance(payload, Mapping) else None
        return len(images) if isinstance(images, list) else None
    try:
        with path.open('r', encoding='utf-8') as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _labeled_count(seed_dir: Path, round_index: int) -> Optional[int]:
    if round_index <= 1:
        path = seed_dir / 'round_00' / 'annotations' / 'labeled.json'
    else:
        path = (
            seed_dir
            / ('round_%02d' % (round_index - 1))
            / 'annotations'
            / 'new_labeled.json'
        )
    return _count_annotation_items(path)


def _round_summary(round_dir: Path) -> Dict[str, Any]:
    path = round_dir / 'round_summary.json'
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _round_duration(summary: Mapping[str, Any]) -> Optional[float]:
    durations = []
    steps = summary.get('steps', [])
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        duration = _numeric(step.get('duration_sec'))
        if duration is not None:
            durations.append(duration)
    return sum(durations) if durations else None


def _aggregate_seed_metrics(run_dir: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    path = run_dir / 'aggregate_summary.json'
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

    result: Dict[Tuple[int, int], Dict[str, float]] = {}
    for round_summary in payload.get('rounds_summary', []):
        if not isinstance(round_summary, Mapping):
            continue
        round_index = int(round_summary.get('round_index', 0) or 0)
        metrics = round_summary.get('metrics', {})
        if round_index <= 0 or not isinstance(metrics, Mapping):
            continue
        for metric_name, metric_summary in metrics.items():
            if not isinstance(metric_summary, Mapping):
                continue
            values = metric_summary.get('values', {})
            if not isinstance(values, Mapping):
                continue
            for seed_text, value in values.items():
                try:
                    seed = int(seed_text)
                except (TypeError, ValueError):
                    continue
                number = _numeric(value)
                if number is not None:
                    result.setdefault((seed, round_index), {})[str(metric_name)] = number
    return result


def _remove_managed_event_files(log_dir: Path) -> None:
    if not log_dir.exists():
        return
    for path in log_dir.rglob(EVENT_FILE_PATTERN):
        if path.is_file():
            path.unlink()
    manifest = log_dir / 'tensorboard_export.json'
    if manifest.exists():
        manifest.unlink()


def _write_training_events(
    writer_type: Type[Any],
    output_dir: Path,
    records: Iterable[Dict[str, float]],
) -> int:
    records = list(records)
    if not records:
        return 0
    writer = writer_type(log_dir=str(output_dir))
    point_count = 0
    try:
        for step, record in enumerate(records, start=1):
            for key, value in sorted(record.items()):
                writer.add_scalar('train/%s' % key, value, step)
                point_count += 1
        writer.flush()
    finally:
        writer.close()
    return point_count


def _write_active_learning_events(
    writer_type: Type[Any],
    output_dir: Path,
    points: Iterable[Tuple[str, float, int]],
) -> int:
    points = list(points)
    if not points:
        return 0
    writer = writer_type(log_dir=str(output_dir))
    try:
        for tag, value, step in points:
            writer.add_scalar(tag, value, step)
        writer.flush()
    finally:
        writer.close()
    return len(points)


def _aggregate_points(run_dir: Path) -> List[Tuple[str, float, int]]:
    path = run_dir / 'aggregate_summary.json'
    if not path.exists():
        return []
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return []

    points = []
    for round_summary in payload.get('rounds_summary', []):
        if not isinstance(round_summary, Mapping):
            continue
        round_index = int(round_summary.get('round_index', 0) or 0)
        if round_index <= 0:
            continue
        metrics = round_summary.get('metrics', {})
        if isinstance(metrics, Mapping):
            for metric_name, metric_summary in metrics.items():
                if not isinstance(metric_summary, Mapping):
                    continue
                for statistic in ('mean', 'std'):
                    value = _numeric(metric_summary.get(statistic))
                    if value is not None:
                        points.append((
                            'validation_%s/%s' % (statistic, metric_name),
                            value,
                            round_index,
                        ))
        for key in ('selected_count', 'duration_sec'):
            summary = round_summary.get(key, {})
            if not isinstance(summary, Mapping):
                continue
            for statistic in ('mean', 'std'):
                value = _numeric(summary.get(statistic))
                if value is not None:
                    points.append(('%s/%s' % (key, statistic), value, round_index))
    return points


def export_run(run: RunRef) -> TensorboardExportSummary:
    """Rebuild TensorBoard events for one discovered ALOD run."""

    writer_type = require_tensorboard()
    run_dir = Path(run.path)
    log_dir = run_dir / 'tensorboard'
    _remove_managed_event_files(log_dir)
    aggregate_metrics = _aggregate_seed_metrics(run_dir)

    training_points = 0
    active_learning_points = 0
    round_count = 0
    seeds = tuple(int(seed) for seed in (run.seeds or (0,)))
    for seed in seeds:
        seed_dir = _seed_source_dir(run, seed)
        rounds = _round_dirs(seed_dir)
        round_count += len(rounds)
        active_points = []
        for round_index, round_dir in rounds:
            training_points += _write_training_events(
                writer_type,
                log_dir / ('seed_%d' % seed) / 'train' / ('round_%02d' % round_index),
                _train_records(round_dir),
            )
            metrics = _latest_eval_metrics(round_dir)
            if not metrics:
                metrics = aggregate_metrics.get((seed, round_index), {})
            for metric_name, value in sorted(metrics.items()):
                active_points.append(('validation/%s' % metric_name, value, round_index))

            labeled_count = _labeled_count(seed_dir, round_index)
            if labeled_count is not None:
                active_points.append(('pool/labeled_images', float(labeled_count), round_index))

            summary = _round_summary(round_dir)
            selected_count = _numeric(summary.get('selected_count'))
            if selected_count is not None:
                active_points.append(('acquisition/selected_count', selected_count, round_index))
            duration = _round_duration(summary)
            if duration is not None:
                active_points.append(('runtime/round_duration_sec', duration, round_index))

        active_learning_points += _write_active_learning_events(
            writer_type,
            log_dir / ('seed_%d' % seed) / 'active_learning',
            active_points,
        )

    aggregate_points = _write_active_learning_events(
        writer_type,
        log_dir / 'aggregate',
        _aggregate_points(run_dir),
    )
    summary = TensorboardExportSummary(
        run_path=str(run_dir),
        log_dir=str(log_dir),
        seed_count=len(seeds),
        round_count=round_count,
        training_points=training_points,
        active_learning_points=active_learning_points,
        aggregate_points=aggregate_points,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        **summary.to_dict(),
    }
    with (log_dir / 'tensorboard_export.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    return summary


def export_run_directory(run_dir: Path) -> TensorboardExportSummary:
    """Discover and export the run rooted exactly at ``run_dir``."""

    resolved = Path(run_dir).resolve()
    matches = [run for run in scan_runs(resolved) if Path(run.path).resolve() == resolved]
    if len(matches) != 1:
        raise ValueError('Expected one ALOD run at %s, found %d.' % (resolved, len(matches)))
    return export_run(matches[0])


def export_work_dir(work_dir: Path) -> List[TensorboardExportSummary]:
    """Export every ALOD run discovered below ``work_dir``."""

    return [export_run(run) for run in scan_runs(Path(work_dir))]
