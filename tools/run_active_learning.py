"""Windows-safe active learning runner for ALOD."""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.catalog import build_experiment_config, list_presets, resolve_experiment, resolve_method_alias

from methods.common.candidates import (
    build_candidate_artifact,
    write_candidate_artifact,
)
from methods.common.coco_pool import write_next_round_pool_split
from methods.common.io import read_json, write_json as write_common_json
from methods.common.results import acquisition_result, write_diagnostics
from methods.entropy.sampler import sample as entropy_sample
from methods.pal.acquisition import sample_pal_from_files
from methods.ppal.acquisition import run_diversity_acquisition, run_uncertainty_acquisition
from methods.random.sampler import sample as random_sample
from tools.common.paths import (
    assert_not_code_refs as tool_assert_not_code_refs,
    display_path as tool_display_path,
    is_relative_to,
    resolve_repo_path as tool_resolve_repo_path,
)
from tools.common.preparation import prepare_required_inputs


@dataclass
class CommandPlan:
    name: str
    argv: List[str]
    cwd: str
    round_index: int = 0
    log_path: Optional[str] = None
    note: str = ''

    def to_dict(self) -> Dict[str, Any]:
        data = {'name': self.name, 'argv': self.argv, 'cwd': self.cwd}
        if self.round_index:
            data['round_index'] = self.round_index
        if self.log_path:
            data['log_path'] = self.log_path
        if self.note:
            data['note'] = self.note
        return data


@dataclass
class AcquisitionPlan:
    name: str
    method: str
    round_index: int
    ppal_stage: Optional[str] = None
    note: str = ''

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'name': self.name,
            'type': 'acquisition',
            'method': self.method,
            'round_index': self.round_index,
        }
        if self.ppal_stage:
            data['ppal_stage'] = self.ppal_stage
        if self.note:
            data['note'] = self.note
        return data


PlanStep = Union[CommandPlan, AcquisitionPlan]


def _assert_not_code_refs(path: Path) -> None:
    tool_assert_not_code_refs(path, ROOT)


def _resolve_repo_path(value: str, must_be_relative: bool = True) -> Path:
    return tool_resolve_repo_path(value, ROOT, must_be_relative=must_be_relative)


def _display_path(path: Path) -> str:
    return tool_display_path(path, ROOT)


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    _assert_not_code_refs(config_path)
    raw = runpy.run_path(str(config_path))
    return {key: value for key, value in raw.items() if not key.startswith('__')}


def validate_experiment_config_paths(cfg: Dict[str, Any]) -> None:
    path_keys = (
        'oracle_path',
        'init_label_json',
        'init_unlabeled_json',
        'train_config',
        'uncertainty_infer_config',
        'image_feature_infer_config',
        'detection_feature_infer_config',
        'pal_infer_config',
        'ecpal_infer_config',
        'pal_embedding_path',
        'output_dir',
    )
    for key in path_keys:
        value = cfg.get(key)
        if value:
            _resolve_repo_path(str(value))
    if cfg.get('init_model'):
        _resolve_repo_path(str(cfg['init_model']))


def validate_initial_pool_files(cfg: Dict[str, Any]) -> None:
    missing = []
    for key in ('init_label_json', 'init_unlabeled_json'):
        value = cfg.get(key)
        if not value:
            missing.append('%s is not configured' % key)
            continue
        path = _resolve_repo_path(str(value))
        if not path.exists():
            missing.append('%s=%s' % (key, _display_path(path)))
    if missing:
        raise SystemExit(
            'Missing initial pool file(s) after automatic preparation: %s'
            % ', '.join(missing)
        )


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.python_path:
        cfg['python_path'] = args.python_path
    if args.port is not None:
        cfg['port'] = args.port
    if args.gpus is not None:
        cfg['gpus'] = args.gpus
        if 'budget' in cfg and 'budget_expand_ratio' in cfg:
            expanded = int(cfg['budget']) * int(cfg['budget_expand_ratio'])
            gpus = max(int(args.gpus), 1)
            cfg['uncertainty_pool_size'] = expanded + gpus - (expanded % gpus)
            sampler_cfg = cfg.get('uncertainty_sampler_config')
            if isinstance(sampler_cfg, dict):
                sampler_cfg['n_sample_images'] = cfg['uncertainty_pool_size']


def _cfg_options(options: Dict[str, Any]) -> List[str]:
    return ['%s=%s' % (key, value) for key, value in options.items()]


def _use_distributed(cfg: Dict[str, Any]) -> bool:
    return int(cfg.get('gpus', 1)) > 1


def _python_prefix(cfg: Dict[str, Any]) -> List[str]:
    return [str(cfg.get('python_path', 'python'))]


def _distributed_prefix(cfg: Dict[str, Any]) -> List[str]:
    python_path = str(cfg.get('python_path', 'python'))
    return [
        python_path,
        '-m',
        'torch.distributed.launch',
        '--nproc_per_node=%d' % int(cfg.get('gpus', 1)),
        '--master_port=%d' % int(cfg.get('port', 29500)),
    ]


def _command_prefix(cfg: Dict[str, Any]) -> List[str]:
    return _distributed_prefix(cfg) if _use_distributed(cfg) else _python_prefix(cfg)


def _launcher_value(cfg: Dict[str, Any]) -> str:
    return 'pytorch' if _use_distributed(cfg) else 'none'


def _round_dir(output_dir: Path, round_index: int) -> Path:
    return output_dir / ('round_%02d' % round_index)


def _round_log_path(output_dir: Path, round_index: int, filename: str) -> Path:
    return _round_dir(output_dir, round_index) / 'logs' / filename


def _round_annotations(output_dir: Path, round_index: int) -> Dict[str, Path]:
    ann_dir = _round_dir(output_dir, round_index) / 'annotations'
    if round_index == 0:
        return {
            'labeled': ann_dir / 'labeled.json',
            'unlabeled': ann_dir / 'unlabeled.json',
        }
    return {
        'labeled': ann_dir / 'new_labeled.json',
        'unlabeled': ann_dir / 'new_unlabeled.json',
        'uncertainty_pool': ann_dir / 'uncertainty_pool.json',
    }


def initialize_round_zero(cfg: Dict[str, Any], output_dir: Path) -> List[str]:
    ann_dir = _round_dir(output_dir, 0) / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    actions = []
    missing_initial_pool = False
    copies = (
        ('init_label_json', ann_dir / 'labeled.json'),
        ('init_unlabeled_json', ann_dir / 'unlabeled.json'),
    )
    for key, target in copies:
        source_value = cfg.get(key)
        if not source_value:
            actions.append('%s is not configured; skipped %s' % (key, _display_path(target)))
            continue
        source = _resolve_repo_path(str(source_value))
        if not source.exists():
            actions.append('missing %s; skipped %s' % (_display_path(source), _display_path(target)))
            missing_initial_pool = True
            continue
        if target.exists():
            actions.append('kept existing %s' % _display_path(target))
            continue
        shutil.copy2(str(source), str(target))
        actions.append('copied %s -> %s' % (_display_path(source), _display_path(target)))
    if missing_initial_pool:
        actions.append('initial pool files are missing after automatic preparation')
    return actions


def _input_pool_paths(output_dir: Path, round_index: int) -> Dict[str, Path]:
    if round_index == 1:
        return _round_annotations(output_dir, 0)
    return _round_annotations(output_dir, round_index - 1)


def _train_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    input_paths: Dict[str, Path],
    seed: int,
) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    options = {
        'data.train.ann_file': input_paths['labeled'],
    }
    options.update(cfg.get('mmdet_common_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/train.py',
            str(cfg['train_config']),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--seed',
            str(seed),
            '--deterministic',
        ]
    )
    if not _use_distributed(cfg):
        argv += ['--gpus', str(int(cfg.get('gpus', 1)))]
    argv += ['--cfg-options'] + _cfg_options(options)
    return CommandPlan(
        'train_round_%02d' % round_index,
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, 'train.log')),
    )


def _eval_plan(cfg: Dict[str, Any], output_dir: Path, round_index: int) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    latest_ckpt = round_work_dir / 'latest.pth'
    options = cfg.get('mmdet_eval_cfg_options', {})
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg['train_config']),
            str(latest_ckpt),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--eval',
            'mAP',
        ]
    )
    if options:
        argv += ['--cfg-options'] + _cfg_options(options)
    return CommandPlan(
        'eval_round_%02d' % round_index,
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, 'eval.log')),
    )


def _uncertainty_infer_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    input_paths: Dict[str, Path],
) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    prefix = round_work_dir / 'unlabeled_inference_result'
    latest_ckpt = round_work_dir / 'latest.pth'
    options = {
        'data.test.ann_file': input_paths['unlabeled'],
    }
    options.update(cfg.get('mmdet_common_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg['uncertainty_infer_config']),
            str(latest_ckpt),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--format-only',
            '--eval-options',
            'jsonfile_prefix=%s' % prefix,
            '--cfg-options',
        ]
        + _cfg_options(options)
    )
    return CommandPlan(
        'uncertainty_inference_round_%02d' % round_index,
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, 'uncertainty_inference.log')),
    )


def _feature_artifact_npz(cfg: Dict[str, Any], output_dir: Path, round_index: int, key: str, default: str) -> Path:
    round_work_dir = _round_dir(output_dir, round_index)
    path = _round_relative_file(round_work_dir, str(cfg.get(key, default)))
    text = str(path)
    if text.endswith('.npz'):
        return path
    return Path(text + '.npz')


def _ppal_candidate_features_npz(cfg: Dict[str, Any], output_dir: Path, round_index: int) -> Path:
    return _feature_artifact_npz(
        cfg,
        output_dir,
        round_index,
        'ppal_candidate_features',
        'ppal_candidate_features.npz',
    )


def _coreset_features_npz(cfg: Dict[str, Any], output_dir: Path, round_index: int, pool_name: str) -> Path:
    return _feature_artifact_npz(
        cfg,
        output_dir,
        round_index,
        'coreset_%s_features' % pool_name,
        'coreset_%s_features.npz' % pool_name,
    )


def _feature_infer_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    ann_file: Path,
    feature_npz: Path,
    name: str,
    log_name: str,
    expected_total: Optional[int] = None,
    config_key: str = 'image_feature_infer_config',
) -> CommandPlan:
    if config_key not in cfg:
        raise ValueError('%s is required for feature inference' % config_key)

    round_work_dir = _round_dir(output_dir, round_index)
    prefix = round_work_dir / ('%s_result' % name)
    latest_ckpt = round_work_dir / 'latest.pth'
    head = 'roi_head' if cfg.get('model_name') == 'fasterrcnn' else 'bbox_head'
    pool_size = _count_annotation_items(ann_file)
    if pool_size is None and expected_total is not None:
        pool_size = int(expected_total)
    if pool_size is None:
        raise ValueError('Cannot determine feature inference pool size: %s' % ann_file)
    options = {
        'data.test.ann_file': ann_file,
        'model.%s.total_images' % head: int(pool_size),
        'model.%s.output_path' % head: feature_npz,
    }
    options.update(cfg.get('mmdet_common_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg[config_key]),
            str(latest_ckpt),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--format-only',
            '--eval-options',
            'jsonfile_prefix=%s' % prefix,
            '--cfg-options',
        ]
        + _cfg_options(options)
    )
    return CommandPlan(
        '%s_round_%02d' % (name, round_index),
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, log_name)),
    )


def _round_relative_file(round_work_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Round output file must be relative: %s' % value)
    resolved = (round_work_dir / path).resolve()
    if not is_relative_to(resolved, round_work_dir.resolve()):
        raise ValueError('Round output file must stay inside the round work dir: %s' % value)
    _assert_not_code_refs(resolved)
    return resolved


def _bbox_json_prefix(path: Path) -> Path:
    text = str(path)
    if text.endswith('.bbox.json'):
        return Path(text[:-len('.bbox.json')])
    if text.endswith('.json'):
        return Path(text[:-len('.json')])
    return path


def _pal_detection_json(cfg: Dict[str, Any], output_dir: Path, round_index: int, pool_name: str) -> Path:
    round_work_dir = _round_dir(output_dir, round_index)
    key = 'pal_%s_detections' % pool_name
    default = 'pal_%s_detections.bbox.json' % pool_name
    path = _round_relative_file(round_work_dir, str(cfg.get(key, default)))
    text = str(path)
    if text.endswith('.bbox.json'):
        return path
    if text.endswith('.json'):
        return Path(text[:-len('.json')] + '.bbox.json')
    return Path(text + '.bbox.json')


def _pal_infer_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    input_paths: Dict[str, Path],
    pool_name: str,
) -> CommandPlan:
    if 'pal_infer_config' not in cfg:
        raise ValueError('pal_infer_config is required for PAL inference')

    round_work_dir = _round_dir(output_dir, round_index)
    latest_ckpt = round_work_dir / 'latest.pth'
    detection_json = _pal_detection_json(cfg, output_dir, round_index, pool_name)
    prefix = _bbox_json_prefix(detection_json)
    options = {
        'data.test.ann_file': input_paths[pool_name],
    }
    options.update(cfg.get('mmdet_common_cfg_options', {}))
    options.update(cfg.get('mmdet_pal_infer_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg['pal_infer_config']),
            str(latest_ckpt),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--format-only',
            '--eval-options',
            'jsonfile_prefix=%s' % prefix,
            '--cfg-options',
        ]
        + _cfg_options(options)
    )
    return CommandPlan(
        'pal_%s_inference_round_%02d' % (pool_name, round_index),
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, 'pal_%s_inference.log' % pool_name)),
    )


def _json_prefix(path: Path) -> Path:
    text = str(path)
    if text.endswith('.json'):
        return Path(text[:-len('.json')])
    return path


def _ecpal_feature_json(cfg: Dict[str, Any], output_dir: Path, round_index: int, pool_name: str) -> Path:
    round_work_dir = _round_dir(output_dir, round_index)
    key = 'ecpal_%s_features' % pool_name
    default = 'ecpal_%s_features.json' % pool_name
    path = _round_relative_file(round_work_dir, str(cfg.get(key, default)))
    text = str(path)
    if text.endswith('.json'):
        return path
    return Path(text + '.json')


def _ecpal_infer_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    input_paths: Dict[str, Path],
    pool_name: str,
) -> CommandPlan:
    if 'ecpal_infer_config' not in cfg:
        raise ValueError('ecpal_infer_config is required for ECPAL inference')

    round_work_dir = _round_dir(output_dir, round_index)
    latest_ckpt = round_work_dir / 'latest.pth'
    feature_json = _ecpal_feature_json(cfg, output_dir, round_index, pool_name)
    prefix = _json_prefix(feature_json)
    options = {
        'data.test.ann_file': input_paths[pool_name],
    }
    options.update(cfg.get('mmdet_common_cfg_options', {}))
    options.update(cfg.get('mmdet_ecpal_infer_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg['ecpal_infer_config']),
            str(latest_ckpt),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
            '--format-only',
            '--eval-options',
            'jsonfile_prefix=%s' % prefix,
            '--cfg-options',
        ]
        + _cfg_options(options)
    )
    return CommandPlan(
        'ecpal_%s_inference_round_%02d' % (pool_name, round_index),
        argv,
        str(ROOT),
        round_index=round_index,
        log_path=str(_round_log_path(output_dir, round_index, 'ecpal_%s_inference.log' % pool_name)),
    )


def build_round_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    method: str,
    round_index: int,
    seed: int,
) -> List[PlanStep]:
    input_paths = _input_pool_paths(output_dir, round_index)
    plan: List[PlanStep] = [
        _train_plan(cfg, output_dir, round_index, input_paths, seed),
        _eval_plan(cfg, output_dir, round_index),
    ]
    if method == 'random':
        plan.append(AcquisitionPlan('random_acquisition_round_%02d' % round_index, method, round_index))
    elif method == 'entropy':
        plan.append(_uncertainty_infer_plan(cfg, output_dir, round_index, input_paths))
        plan.append(AcquisitionPlan('entropy_acquisition_round_%02d' % round_index, method, round_index))
    elif method == 'ppal':
        plan.append(_uncertainty_infer_plan(cfg, output_dir, round_index, input_paths))
        plan.append(AcquisitionPlan(
            'ppal_uncertainty_acquisition_round_%02d' % round_index,
            method,
            round_index,
            ppal_stage='uncertainty',
        ))
        annotations = _round_annotations(output_dir, round_index)
        plan.append(_feature_infer_plan(
            cfg,
            output_dir,
            round_index,
            annotations['uncertainty_pool'],
            _ppal_candidate_features_npz(cfg, output_dir, round_index),
            'ppal_feature_inference',
            'ppal_feature_inference.log',
            expected_total=int(cfg.get('uncertainty_pool_size', cfg.get('budget', 0))),
            config_key='detection_feature_infer_config',
        ))
        plan.append(AcquisitionPlan(
            'ppal_diversity_acquisition_round_%02d' % round_index,
            method,
            round_index,
            ppal_stage='diversity',
        ))
    elif method == 'pal':
        plan.append(_pal_infer_plan(cfg, output_dir, round_index, input_paths, 'labeled'))
        plan.append(_pal_infer_plan(cfg, output_dir, round_index, input_paths, 'unlabeled'))
        plan.append(AcquisitionPlan('pal_acquisition_round_%02d' % round_index, method, round_index))
    elif method == 'ecpal':
        plan.append(_ecpal_infer_plan(cfg, output_dir, round_index, input_paths, 'labeled'))
        plan.append(_ecpal_infer_plan(cfg, output_dir, round_index, input_paths, 'unlabeled'))
        plan.append(AcquisitionPlan('ecpal_acquisition_round_%02d' % round_index, method, round_index))
    elif method == 'coreset':
        plan.append(_feature_infer_plan(
            cfg,
            output_dir,
            round_index,
            input_paths['labeled'],
            _coreset_features_npz(cfg, output_dir, round_index, 'labeled'),
            'coreset_labeled_feature_inference',
            'coreset_labeled_feature_inference.log',
            config_key='image_feature_infer_config',
        ))
        plan.append(_feature_infer_plan(
            cfg,
            output_dir,
            round_index,
            input_paths['unlabeled'],
            _coreset_features_npz(cfg, output_dir, round_index, 'unlabeled'),
            'coreset_unlabeled_feature_inference',
            'coreset_unlabeled_feature_inference.log',
            config_key='image_feature_infer_config',
        ))
        plan.append(AcquisitionPlan('coreset_acquisition_round_%02d' % round_index, method, round_index))
    else:
        raise ValueError('Unsupported method: %s' % method)
    return plan


def _write_plan_log(output_dir: Path, plan: List[PlanStep]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'active_learning_plan.json'
    _assert_not_code_refs(path)
    write_common_json(path, [step.to_dict() for step in plan], indent=2)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _assert_not_code_refs(path)
    write_common_json(path, payload, indent=2)


def _print_plan(plan: Iterable[PlanStep]) -> None:
    for step in plan:
        print(json.dumps(step.to_dict(), indent=2))


def _step_label(step: PlanStep) -> str:
    name = step.name
    labels = (
        ('ecpal_labeled_inference', 'ecpal labeled inference'),
        ('ecpal_unlabeled_inference', 'ecpal unlabeled inference'),
        ('coreset_labeled_feature_inference', 'coreset labeled feature inference'),
        ('coreset_unlabeled_feature_inference', 'coreset unlabeled feature inference'),
        ('ppal_feature_inference', 'ppal feature inference'),
        ('pal_labeled_inference', 'pal labeled inference'),
        ('pal_unlabeled_inference', 'pal unlabeled inference'),
        ('uncertainty_inference', 'uncertainty inference'),
        ('ecpal_acquisition', 'ecpal acquisition'),
        ('coreset_acquisition', 'coreset acquisition'),
        ('ppal_uncertainty_acquisition', 'ppal uncertainty acquisition'),
        ('ppal_diversity_acquisition', 'ppal diversity acquisition'),
        ('pal_acquisition', 'pal acquisition'),
        ('random_acquisition', 'random acquisition'),
        ('entropy_acquisition', 'entropy acquisition'),
        ('train', 'train'),
        ('eval', 'eval'),
    )
    for prefix, label in labels:
        if name.startswith(prefix):
            return label
    return name


class RunnerStepError(RuntimeError):
    pass


def _subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
    env.setdefault('PYTHONUNBUFFERED', '1')
    return env


_TRAIN_ITER_RE = re.compile(r'Iter \[(\d+)/(\d+)\]')
_PROGRESS_COUNT_RE = re.compile(r'(\d+)\s*/\s*(\d+)')
_LOSS_RE = re.compile(r'(?:^|,\s)loss:\s*([0-9.eE+-]+)')
_ETA_RE = re.compile(r'eta:\s*([^,]+)')


def _count_coco_images(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        data = read_json(path)
        return len(data.get('images', []))
    except (OSError, ValueError, TypeError):
        return None


def _count_nonempty_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _count_annotation_items(path: Path) -> Optional[int]:
    if path.suffix.lower() == '.json':
        return _count_coco_images(path)
    return _count_nonempty_lines(path)


def _runtime_read_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _train_max_epochs(cfg: Dict[str, Any]) -> Optional[int]:
    train_config = cfg.get('train_config')
    if not train_config:
        return None
    try:
        raw = runpy.run_path(str(_resolve_repo_path(str(train_config))))
    except Exception:
        return None
    runner = raw.get('runner')
    if isinstance(runner, dict) and runner.get('max_epochs') is not None:
        return int(runner['max_epochs'])
    return None


def _progress_kind(step: CommandPlan) -> str:
    if step.name.startswith('train'):
        return 'train'
    if (
        step.name.startswith('eval')
        or step.name.startswith('ecpal_labeled_inference')
        or step.name.startswith('ecpal_unlabeled_inference')
        or step.name.startswith('pal_labeled_inference')
        or step.name.startswith('pal_unlabeled_inference')
        or step.name.startswith('uncertainty_inference')
        or step.name.startswith('ppal_feature_inference')
        or step.name.startswith('coreset_labeled_feature_inference')
        or step.name.startswith('coreset_unlabeled_feature_inference')
    ):
        return 'test'
    return 'command'


def _progress_total_for_step(
    step: CommandPlan,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Optional[int]:
    round_index = step.round_index
    input_paths = _input_pool_paths(output_dir, round_index)
    annotations = _round_annotations(output_dir, round_index)

    if step.name.startswith('train'):
        image_count = _count_annotation_items(input_paths['labeled'])
        max_epochs = _train_max_epochs(cfg)
        if image_count is not None and max_epochs is not None:
            return image_count * max_epochs
        return image_count
    if step.name.startswith('eval'):
        eval_options = cfg.get('mmdet_eval_cfg_options', {})
        ann_file = eval_options.get('data.test.ann_file') if isinstance(eval_options, dict) else None
        if ann_file:
            return _count_annotation_items(_runtime_read_path(str(ann_file)))
        return None
    if step.name.startswith('ecpal_labeled_inference'):
        return _count_annotation_items(input_paths['labeled'])
    if step.name.startswith('ecpal_unlabeled_inference'):
        return _count_annotation_items(input_paths['unlabeled'])
    if step.name.startswith('pal_labeled_inference'):
        return _count_annotation_items(input_paths['labeled'])
    if step.name.startswith('pal_unlabeled_inference'):
        return _count_annotation_items(input_paths['unlabeled'])
    if step.name.startswith('uncertainty_inference'):
        return _count_annotation_items(input_paths['unlabeled'])
    if step.name.startswith('ppal_feature_inference'):
        total = _count_annotation_items(annotations['uncertainty_pool'])
        if total is not None:
            return total
        if cfg.get('uncertainty_pool_size') is not None:
            return int(cfg['uncertainty_pool_size'])
    if step.name.startswith('coreset_labeled_feature_inference'):
        return _count_annotation_items(input_paths['labeled'])
    if step.name.startswith('coreset_unlabeled_feature_inference'):
        return _count_annotation_items(input_paths['unlabeled'])
    return None


def _new_step_progress(
    step: CommandPlan,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    if tqdm is None:
        return None
    kind = _progress_kind(step)
    unit = 'iter' if kind == 'train' else 'img'
    bar = tqdm(
        total=_progress_total_for_step(step, cfg, output_dir),
        desc=_step_label(step),
        unit=unit,
        leave=True,
    )
    return {'bar': bar, 'kind': kind, 'last': 0}


def _set_progress_total(progress: Dict[str, Any], total: int) -> None:
    bar = progress['bar']
    if bar.total != total:
        bar.total = total
        bar.refresh()


def _advance_progress(progress: Dict[str, Any], current: int) -> None:
    bar = progress['bar']
    current = max(current, int(progress.get('last', 0)))
    if bar.total is not None:
        current = min(current, int(bar.total))
    delta = current - int(progress.get('last', 0))
    if delta > 0:
        bar.update(delta)
        progress['last'] = current


def _update_progress_from_record(progress: Optional[Dict[str, Any]], record: str) -> None:
    if progress is None:
        return
    kind = progress['kind']
    if kind == 'train':
        match = _TRAIN_ITER_RE.search(record)
    else:
        match = _PROGRESS_COUNT_RE.search(record)
    if not match:
        return
    current = int(match.group(1))
    total = int(match.group(2))
    _set_progress_total(progress, total)
    _advance_progress(progress, current)
    if kind == 'train':
        details = []
        loss = _LOSS_RE.search(record)
        eta = _ETA_RE.search(record)
        if loss:
            details.append('loss=%s' % loss.group(1))
        if eta:
            details.append('eta=%s' % eta.group(1).strip())
        if details:
            progress['bar'].set_postfix_str(' '.join(details))


def _finish_progress(progress: Optional[Dict[str, Any]]) -> None:
    if progress is None:
        return
    bar = progress['bar']
    if bar.total is not None:
        _advance_progress(progress, int(bar.total))
    bar.close()


def _run_subprocess_plan(
    step: CommandPlan,
    cfg: Dict[str, Any],
    output_dir: Path,
    verbose: bool = False,
) -> None:
    stdout_handle = None
    progress = None
    try:
        if step.log_path:
            log_path = Path(step.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = log_path.open('w', encoding='utf-8')
        if not verbose:
            progress = _new_step_progress(step, cfg, output_dir)
        process = subprocess.Popen(
            step.argv,
            cwd=step.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            env=_subprocess_env(),
        )
        assert process.stdout is not None
        record = ''
        while True:
            chunk = process.stdout.read(1)
            if not chunk:
                break
            if stdout_handle is not None:
                stdout_handle.write(chunk)
            if verbose:
                print(chunk, end='')
            if chunk in ('\r', '\n'):
                if record:
                    _update_progress_from_record(progress, record)
                    record = ''
            else:
                record += chunk
        if record:
            _update_progress_from_record(progress, record)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, step.argv)
        _finish_progress(progress)
        progress = None
    except subprocess.CalledProcessError as exc:
        log_text = _display_path(Path(step.log_path)) if step.log_path else 'terminal'
        raise RunnerStepError(
            'step failed: round=%s step=%s exit_code=%s log=%s'
            % (step.round_index or '?', _step_label(step), exc.returncode, log_text)
        ) from exc
    finally:
        if progress is not None:
            progress['bar'].close()
        if stdout_handle is not None:
            stdout_handle.close()


def _execute_lightweight_acquisition(
    cfg: Dict[str, Any],
    output_dir: Path,
    method: str,
    round_index: int,
    seed: int,
) -> Dict[str, Any]:
    input_paths = _input_pool_paths(output_dir, round_index)
    round_work_dir = _round_dir(output_dir, round_index)
    annotations = _round_annotations(output_dir, round_index)
    budget = int(cfg.get('budget', 0))
    diagnostics_path = None
    diagnostics_stage = None
    candidate_outputs: Dict[str, str] = {}

    if method == 'random':
        selected = random_sample(input_paths['unlabeled'], budget=budget, seed=seed)
    elif method == 'entropy':
        results_json = round_work_dir / 'unlabeled_inference_result.bbox.json'
        selected = entropy_sample(input_paths['unlabeled'], budget=budget, results_json=results_json, seed=seed)
    elif method == 'pal':
        labeled_dets = _pal_detection_json(cfg, output_dir, round_index, 'labeled')
        unlabeled_dets = _pal_detection_json(cfg, output_dir, round_index, 'unlabeled')
        embedding_path = None
        if cfg.get('pal_embedding_path'):
            embedding_path = _resolve_repo_path(str(cfg['pal_embedding_path']))
        pal_mode = str(cfg.get('pal_mode', 'lius'))
        pal_mode_normalized = pal_mode.lower()
        diagnostics = sample_pal_from_files(
            labeled_pool_json=input_paths['labeled'],
            unlabeled_pool_json=input_paths['unlabeled'],
            labeled_detections_json=labeled_dets,
            unlabeled_detections_json=unlabeled_dets,
            budget=budget,
            mode=pal_mode,
            iou_threshold=float(cfg.get('pal_iou_threshold', 0.5)),
            seed=seed,
            alpha=float(cfg.get('pal_alpha', 0.9)),
            beta=float(cfg.get('pal_beta', 0.04)),
            gamma=float(cfg.get('pal_gamma', 0.02)),
            embedding_source=str(cfg.get('pal_embedding_source', 'external')),
            embedding_path=embedding_path,
        )
        selected = diagnostics['selected_image_ids']
        diagnostics_file = cfg.get('pal_diagnostics_file')
        if not diagnostics_file:
            diagnostics_file = (
                'pal_diagnostics.json'
                if pal_mode_normalized in ('full', 'guide')
                else 'pal_lius_diagnostics.json'
            )
        diagnostics_path = _round_relative_file(round_work_dir, str(diagnostics_file))
        diagnostics_stage = 'guide' if pal_mode_normalized in ('full', 'guide') else 'lius'
        diagnostics_extra = dict(diagnostics)
        diagnostics_extra.pop('selected_image_ids', None)
        candidate_records = list(diagnostics_extra.pop('candidate_records', []))
        diagnostics_extra.pop('candidate_scores', None)
        candidate_file = (
            'pal_candidates.json'
            if diagnostics_stage == 'guide'
            else 'pal_lius_candidates.json'
        )
        candidate_artifact_path = _round_relative_file(round_work_dir, candidate_file)
        candidate_outputs = {
            'candidates_json': str(candidate_artifact_path),
        }
        candidate_artifact = build_candidate_artifact(
            method='pal',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            candidates=candidate_records,
            selected_image_ids=selected,
        )
        write_candidate_artifact(candidate_artifact_path, candidate_artifact)
        diagnostics_payload = acquisition_result(
            method='pal',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            selected_image_ids=selected,
            inputs={
                'labeled_pool_json': str(input_paths['labeled']),
                'unlabeled_pool_json': str(input_paths['unlabeled']),
                'labeled_detections_json': str(labeled_dets),
                'unlabeled_detections_json': str(unlabeled_dets),
                'embedding_path': str(embedding_path) if embedding_path else None,
            },
            outputs={
                'labeled_pool_json': str(annotations['labeled']),
                'unlabeled_pool_json': str(annotations['unlabeled']),
                'diagnostics_json': str(diagnostics_path),
                **candidate_outputs,
            },
            **diagnostics_extra,
        )
        write_diagnostics(diagnostics_path, diagnostics_payload)
    elif method == 'ecpal':
        from methods.ecpal.acquisition import sample_ecpal_from_files

        labeled_features = _ecpal_feature_json(cfg, output_dir, round_index, 'labeled')
        unlabeled_features = _ecpal_feature_json(cfg, output_dir, round_index, 'unlabeled')
        diagnostics = sample_ecpal_from_files(
            labeled_pool_json=input_paths['labeled'],
            unlabeled_pool_json=input_paths['unlabeled'],
            labeled_features_json=labeled_features,
            unlabeled_features_json=unlabeled_features,
            budget=budget,
            candidate_expand_ratio=int(cfg.get('ecpal_candidate_expand_ratio', 2)),
            foreground_iou_threshold=float(cfg.get('ecpal_foreground_iou_threshold', 0.5)),
            background_iou_threshold=float(cfg.get('ecpal_background_iou_threshold', 0.1)),
            eps=float(cfg.get('ecpal_eps', 1e-12)),
            weight_eps=float(cfg.get('ecpal_weight_eps', 1e-6)),
            seed=seed,
        )
        selected = diagnostics['selected_image_ids']
        diagnostics_path = _round_relative_file(
            round_work_dir,
            str(cfg.get('ecpal_diagnostics_file', 'ecpal_diagnostics.json')),
        )
        diagnostics_stage = 'ecd'
        diagnostics_extra = dict(diagnostics)
        diagnostics_extra.pop('selected_image_ids', None)
        diagnostics_extra.pop('stage', None)
        candidate_records = list(diagnostics_extra.pop('candidate_records', []))
        diagnostics_extra.pop('candidate_scores', None)
        candidate_artifact_path = _round_relative_file(
            round_work_dir,
            str(cfg.get('ecpal_candidates_file', 'ecpal_candidates.json')),
        )
        candidate_outputs = {
            'candidates_json': str(candidate_artifact_path),
        }
        candidate_artifact = build_candidate_artifact(
            method='ecpal',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            candidates=candidate_records,
            selected_image_ids=selected,
        )
        write_candidate_artifact(candidate_artifact_path, candidate_artifact)
        diagnostics_payload = acquisition_result(
            method='ecpal',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            selected_image_ids=selected,
            inputs={
                'labeled_pool_json': str(input_paths['labeled']),
                'unlabeled_pool_json': str(input_paths['unlabeled']),
                'labeled_features_json': str(labeled_features),
                'unlabeled_features_json': str(unlabeled_features),
            },
            outputs={
                'labeled_pool_json': str(annotations['labeled']),
                'unlabeled_pool_json': str(annotations['unlabeled']),
                'diagnostics_json': str(diagnostics_path),
                **candidate_outputs,
            },
            **diagnostics_extra,
        )
        write_diagnostics(diagnostics_path, diagnostics_payload)
    elif method == 'coreset':
        from methods.coreset.acquisition import sample_coreset_from_files

        labeled_features = _coreset_features_npz(cfg, output_dir, round_index, 'labeled')
        unlabeled_features = _coreset_features_npz(cfg, output_dir, round_index, 'unlabeled')
        diagnostics = sample_coreset_from_files(
            labeled_pool_json=input_paths['labeled'],
            unlabeled_pool_json=input_paths['unlabeled'],
            labeled_features_npz=labeled_features,
            unlabeled_features_npz=unlabeled_features,
            budget=budget,
            batch_size=int(cfg.get('coreset_distance_batch_size', 512)),
            center_batch_size=int(cfg.get('coreset_center_batch_size', 2048)),
        )
        selected = diagnostics['selected_image_ids']
        diagnostics_path = _round_relative_file(
            round_work_dir,
            str(cfg.get('coreset_diagnostics_file', 'coreset_diagnostics.json')),
        )
        diagnostics_stage = 'kcenter'
        diagnostics_extra = dict(diagnostics)
        diagnostics_extra.pop('selected_image_ids', None)
        diagnostics_extra.pop('stage', None)
        candidate_records = list(diagnostics_extra.pop('candidate_records', []))
        candidate_artifact_path = _round_relative_file(
            round_work_dir,
            str(cfg.get('coreset_candidates_file', 'coreset_candidates.json')),
        )
        candidate_outputs = {
            'candidates_json': str(candidate_artifact_path),
        }
        candidate_artifact = build_candidate_artifact(
            method='coreset',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            candidates=candidate_records,
            selected_image_ids=selected,
        )
        write_candidate_artifact(candidate_artifact_path, candidate_artifact)
        diagnostics_payload = acquisition_result(
            method='coreset',
            stage=diagnostics_stage,
            round_index=round_index,
            budget=budget,
            selected_image_ids=selected,
            inputs={
                'labeled_pool_json': str(input_paths['labeled']),
                'unlabeled_pool_json': str(input_paths['unlabeled']),
                'labeled_features_npz': str(labeled_features),
                'unlabeled_features_npz': str(unlabeled_features),
            },
            outputs={
                'labeled_pool_json': str(annotations['labeled']),
                'unlabeled_pool_json': str(annotations['unlabeled']),
                'diagnostics_json': str(diagnostics_path),
                **candidate_outputs,
            },
            **diagnostics_extra,
        )
        write_diagnostics(diagnostics_path, diagnostics_payload)
    else:
        raise ValueError('Unsupported lightweight acquisition method: %s' % method)

    write_next_round_pool_split(
        _resolve_repo_path(str(cfg['oracle_path'])),
        input_paths['labeled'],
        selected,
        annotations['labeled'],
        annotations['unlabeled'],
    )
    return {
        'selected_count': len(selected),
        'diagnostics_path': str(diagnostics_path) if diagnostics_path is not None else None,
        'stage': diagnostics_stage,
        'outputs': {
            'labeled_pool_json': str(annotations['labeled']),
            'unlabeled_pool_json': str(annotations['unlabeled']),
            **candidate_outputs,
        },
    }


def _load_ppal_diagnostic_stages(diagnostics_path: Path) -> List[Dict[str, Any]]:
    if not diagnostics_path.exists():
        return []
    payload = read_json(diagnostics_path)
    return list(payload.get('stages', []))


def _execute_ppal_acquisition(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    ppal_stage: str,
    seed: int,
) -> Dict[str, Any]:
    if ppal_stage not in ('uncertainty', 'diversity'):
        raise ValueError('Unsupported PPAL acquisition stage: %s' % ppal_stage)

    input_paths = _input_pool_paths(output_dir, round_index)
    round_work_dir = _round_dir(output_dir, round_index)
    annotations = _round_annotations(output_dir, round_index)
    result_json = round_work_dir / 'unlabeled_inference_result.bbox.json'

    if ppal_stage == 'uncertainty':
        output = run_uncertainty_acquisition(
            cfg=cfg,
            repo_root=ROOT,
            round_index=round_index,
            result_json=result_json,
            last_labeled_json=input_paths['labeled'],
            out_candidate_json=annotations['uncertainty_pool'],
        )
    else:
        output = run_diversity_acquisition(
            cfg=cfg,
            repo_root=ROOT,
            round_index=round_index,
            result_json=result_json,
            feature_npz=_ppal_candidate_features_npz(cfg, output_dir, round_index),
            last_labeled_json=input_paths['labeled'],
            out_labeled_json=annotations['labeled'],
            out_unlabeled_json=annotations['unlabeled'],
            seed=seed,
        )

    diagnostics_path = round_work_dir / 'ppal_diagnostics.json'
    stages = _load_ppal_diagnostic_stages(diagnostics_path)
    runner_stage = str(output.get('runner_stage', ppal_stage))
    stages = [stage for stage in stages if stage.get('runner_stage') != runner_stage]
    stages.append(output)
    stage_names = {str(stage.get('runner_stage')) for stage in stages}
    summary_stage = 'all' if {'uncertainty', 'diversity'}.issubset(stage_names) else ppal_stage
    diagnostics_payload = acquisition_result(
        method='ppal',
        stage=summary_stage,
        round_index=round_index,
        budget=int(output.get('budget', cfg.get('budget', 0))),
        selected_image_ids=output.get('selected_image_ids', []),
        inputs={
            'labeled_pool_json': str(input_paths['labeled']),
            'unlabeled_pool_json': str(input_paths['unlabeled']),
            'uncertainty_result_json': str(result_json),
            'feature_npz': str(_ppal_candidate_features_npz(cfg, output_dir, round_index)),
        },
        outputs=dict(output.get('outputs', {}), diagnostics_json=str(diagnostics_path)),
        stages=stages,
    )
    write_diagnostics(diagnostics_path, diagnostics_payload)
    output['diagnostics_path'] = str(diagnostics_path)
    return output


def _run_acquisition_plan(
    step: AcquisitionPlan,
    cfg: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    if step.method == 'ppal':
        if step.ppal_stage is None:
            raise ValueError('PPAL acquisition step requires ppal_stage')
        output = _execute_ppal_acquisition(cfg, output_dir, step.round_index, step.ppal_stage, seed)
        return {
            'selected_count': int(output.get('selected_count', 0)),
            'diagnostics_path': output.get('diagnostics_path'),
            'stage': step.ppal_stage,
            'outputs': output.get('outputs', {}),
        }

    return _execute_lightweight_acquisition(
        cfg,
        output_dir,
        step.method,
        step.round_index,
        seed=seed,
    )


def _run_plan_step(
    step: PlanStep,
    cfg: Dict[str, Any],
    output_dir: Path,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    result: Dict[str, Any] = {
        'name': step.name,
        'label': _step_label(step),
        'round_index': step.round_index,
        'status': 'running',
        'started_at': started_at,
    }
    if isinstance(step, CommandPlan):
        result['type'] = 'command'
        if step.log_path:
            result['log_path'] = str(step.log_path)
        _run_subprocess_plan(step, cfg, output_dir, verbose=verbose)
    elif isinstance(step, AcquisitionPlan):
        result['type'] = 'acquisition'
        result.update(_run_acquisition_plan(step, cfg, output_dir, seed))
    else:
        raise TypeError('Unsupported plan step: %r' % (step,))
    result['status'] = 'done'
    result['duration_sec'] = round(time.time() - started, 3)
    result['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return result


def _failed_step_result(step: PlanStep, exc: BaseException, started: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'name': step.name,
        'label': _step_label(step),
        'round_index': step.round_index,
        'status': 'failed',
        'duration_sec': round(time.time() - started, 3),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'error': str(exc),
    }
    if isinstance(step, CommandPlan) and step.log_path:
        result['type'] = 'command'
        result['log_path'] = str(step.log_path)
    elif isinstance(step, AcquisitionPlan):
        result['type'] = 'acquisition'
    return result


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return '%d:%02d:%02d' % (hours, minutes, seconds)
    return '%02d:%02d' % (minutes, seconds)


def _format_step_result(index: int, total: int, result: Dict[str, Any]) -> str:
    parts = [
        '[%d/%d]' % (index, total),
        '%-26s' % result['label'],
        result['status'],
        _format_duration(float(result.get('duration_sec', 0))),
    ]
    if result.get('selected_count') is not None:
        parts.append('selected=%s' % result['selected_count'])
    if result.get('log_path'):
        parts.append('log=%s' % _display_path(Path(str(result['log_path']))))
    return ' '.join(parts)


def _format_acquisition_result(result: Dict[str, Any]) -> str:
    parts = ['%s result:' % result['label']]
    if result.get('selected_count') is not None:
        parts.append('selected=%s' % result['selected_count'])
    if result.get('diagnostics_path'):
        parts.append('diagnostics=%s' % _display_path(Path(str(result['diagnostics_path']))))
    outputs = result.get('outputs') if isinstance(result.get('outputs'), dict) else {}
    if outputs.get('labeled_pool_json'):
        parts.append('labeled=%s' % _display_path(Path(str(outputs['labeled_pool_json']))))
    if outputs.get('unlabeled_pool_json'):
        parts.append('unlabeled=%s' % _display_path(Path(str(outputs['unlabeled_pool_json']))))
    if outputs.get('candidates_json'):
        parts.append('candidates=%s' % _display_path(Path(str(outputs['candidates_json']))))
    return ' '.join(parts)


def _round_summary_path(output_dir: Path, round_index: int) -> Path:
    return _round_dir(output_dir, round_index) / 'round_summary.json'


def _run_summary_path(output_dir: Path) -> Path:
    return output_dir / 'run_summary.json'


def _preparation_requested(cfg: Dict[str, Any]) -> bool:
    return any(key in cfg for key in ('dataset_prep', 'pretrained', 'pal_embedding_prep'))


def _print_preparation_summary(results: List[Dict[str, object]]) -> None:
    if not results:
        return
    parts = []
    for result in results:
        component = str(result.get('component', 'input'))
        action = str(result.get('action', 'ready'))
        parts.append('%s=%s' % (component, action))
    print('prepared inputs: %s' % ', '.join(parts), flush=True)


def _latest_acquisition_result(round_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    acquisitions = [
        result for result in round_results
        if result.get('type') == 'acquisition' and result.get('status') == 'done'
    ]
    return acquisitions[-1] if acquisitions else {}


def _round_summary_payload(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    status: str,
    round_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    annotations = _round_annotations(output_dir, round_index)
    acquisition = _latest_acquisition_result(round_results)
    outputs = dict(acquisition.get('outputs', {}))
    outputs.setdefault('labeled_pool_json', str(annotations['labeled']))
    outputs.setdefault('unlabeled_pool_json', str(annotations['unlabeled']))
    if acquisition.get('diagnostics_path'):
        outputs['diagnostics_json'] = acquisition.get('diagnostics_path')
    return {
        'round_index': round_index,
        'status': status,
        'budget': int(cfg.get('budget', 0)),
        'selected_count': acquisition.get('selected_count'),
        'outputs': outputs,
        'steps': round_results,
    }


def _config_path_summary(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    keys = (
        'train_config',
        'uncertainty_infer_config',
        'image_feature_infer_config',
        'detection_feature_infer_config',
        'pal_infer_config',
        'ecpal_infer_config',
        'pal_embedding_path',
    )
    return {key: str(cfg.get(key)) if cfg.get(key) else None for key in keys}


def _run_summary_base(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    output_dir: Path,
    plan_log: Path,
    total_rounds: int,
    seed: int,
) -> Dict[str, Any]:
    catalog_selection = selection.get('catalog_selection')
    detector = None
    dataset = None
    if catalog_selection is not None:
        detector = catalog_selection.preset.detector
        dataset = catalog_selection.preset.dataset
    return {
        'status': 'running',
        'method': args.method,
        'method_arg': cfg.get('_method_arg'),
        'detector': detector,
        'dataset': dataset,
        'preset': selection.get('preset_name'),
        'rounds': total_rounds,
        'start_round': args.start_round,
        'budget': int(cfg.get('budget', 0)),
        'seed': seed,
        'gpus': int(cfg.get('gpus', 1)),
        'output_dir': str(output_dir),
        'plan_path': str(plan_log),
        'config_paths': _config_path_summary(cfg),
        'round_summaries': [],
    }


def _run_timestamp() -> str:
    return datetime.now().strftime('%m-%d-%Y_%H;%M')


def _seed_output_dir(run_dir: Path, seed: int) -> Path:
    return run_dir / ('seed_%d' % seed)


def _ensure_new_timestamp_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            'Timestamp output directory already exists and is not empty: %s'
            % _display_path(run_dir)
        )


def _selected_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds is not None:
        if args.seed is not None:
            raise SystemExit('Do not combine --seed with --seeds.')
        seeds = [int(seed) for seed in args.seeds]
    else:
        seeds = [int(args.seed) if args.seed is not None else 0]
    if not seeds:
        raise SystemExit('At least one seed is required.')
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise SystemExit('Duplicate seeds are not allowed: %s' % duplicates)
    return seeds


def _print_timestamp_run_header(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    run_dir: Path,
    seeds: List[int],
    total_rounds: int,
) -> None:
    catalog_selection = selection.get('catalog_selection')
    detector = catalog_selection.preset.detector if catalog_selection is not None else 'custom'
    dataset = catalog_selection.preset.dataset if catalog_selection is not None else 'custom'
    print('ALOD run: %s / %s / %s' % (args.method, detector, dataset))
    print(
        'seeds=%s rounds=%s budget=%s gpus=%s'
        % (
            ','.join(str(seed) for seed in seeds),
            total_rounds,
            int(cfg.get('budget', 0)),
            int(cfg.get('gpus', 1)),
        )
    )
    print('output=%s' % _display_path(run_dir))


def _read_eval_json_metrics(round_dir: Path) -> Optional[Dict[str, float]]:
    paths = sorted(round_dir.glob('eval_*.json'))
    for path in reversed(paths):
        try:
            payload = read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        metric = payload.get('metric') if isinstance(payload, dict) else None
        if not isinstance(metric, dict):
            continue
        values = {}
        for key in ('mAP', 'AP50'):
            if metric.get(key) is not None:
                values[key] = float(metric[key])
        if values:
            return values
    return None


def _read_eval_log_metrics(round_summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    steps = round_summary.get('steps', [])
    if not isinstance(steps, list):
        return None
    eval_log = None
    for step in steps:
        if isinstance(step, dict) and step.get('label') == 'eval' and step.get('log_path'):
            eval_log = Path(str(step['log_path']))
            break
    if eval_log is None:
        return None
    if not eval_log.is_absolute():
        eval_log = ROOT / eval_log
    if not eval_log.exists():
        return None
    try:
        text = eval_log.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    values = {}
    for key in ('mAP', 'AP50'):
        match = re.search(r"\('%s',\s*([0-9.eE+-]+)\)" % re.escape(key), text)
        if match:
            values[key] = float(match.group(1))
    return values or None


def _round_metrics(seed_dir: Path, round_index: int, round_summary: Dict[str, Any]) -> Dict[str, float]:
    round_dir = _round_dir(seed_dir, round_index)
    metrics = _read_eval_json_metrics(round_dir)
    if metrics is None:
        metrics = _read_eval_log_metrics(round_summary)
    return metrics or {}


def _round_duration_sec(round_summary: Dict[str, Any]) -> Optional[float]:
    steps = round_summary.get('steps', [])
    if not isinstance(steps, list):
        return None
    durations = [
        float(step['duration_sec'])
        for step in steps
        if isinstance(step, dict) and step.get('duration_sec') is not None
    ]
    if not durations:
        return None
    return round(sum(durations), 3)


def _numeric_summary(values_by_seed: Dict[int, float], seeds: List[int]) -> Dict[str, Any]:
    values = [float(values_by_seed[seed]) for seed in seeds if seed in values_by_seed]
    payload: Dict[str, Any] = {
        'values': {str(seed): values_by_seed[seed] for seed in seeds if seed in values_by_seed},
        'count': len(values),
        'missing_seeds': [seed for seed in seeds if seed not in values_by_seed],
        'mean': None,
        'std': None,
    }
    if values:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        payload['mean'] = mean
        payload['std'] = variance ** 0.5
    return payload


def _read_round_summary(seed_dir: Path, round_index: int) -> Optional[Dict[str, Any]]:
    path = _round_summary_path(seed_dir, round_index)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _aggregate_summary_path(run_dir: Path) -> Path:
    return run_dir / 'aggregate_summary.json'


def _build_aggregate_summary(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    base_output_dir: Path,
    run_dir: Path,
    run_id: str,
    seeds: List[int],
    seed_run_summaries: Dict[int, Dict[str, Any]],
    total_rounds: int,
) -> Dict[str, Any]:
    catalog_selection = selection.get('catalog_selection')
    detector = catalog_selection.preset.detector if catalog_selection is not None else None
    dataset = catalog_selection.preset.dataset if catalog_selection is not None else None
    round_indexes = list(range(args.start_round, args.start_round + total_rounds))

    seed_runs = []
    round_metrics_by_seed: Dict[int, Dict[int, Dict[str, float]]] = {}
    for seed in seeds:
        seed_dir = _seed_output_dir(run_dir, seed)
        summary = seed_run_summaries.get(seed, {})
        per_round_metrics: Dict[int, Dict[str, float]] = {}
        round_summary_paths = []
        for round_index in round_indexes:
            round_summary = _read_round_summary(seed_dir, round_index)
            if round_summary is None:
                continue
            round_summary_paths.append(str(_round_summary_path(seed_dir, round_index)))
            per_round_metrics[round_index] = _round_metrics(seed_dir, round_index, round_summary)
        round_metrics_by_seed[seed] = per_round_metrics
        final_metrics = {}
        for round_index in reversed(round_indexes):
            if per_round_metrics.get(round_index):
                final_metrics = per_round_metrics[round_index]
                break
        seed_runs.append({
            'seed': seed,
            'status': summary.get('status'),
            'output_dir': str(seed_dir),
            'run_summary_json': str(_run_summary_path(seed_dir)),
            'round_summary_jsons': round_summary_paths,
            'final_metrics': final_metrics,
        })

    rounds_summary = []
    for round_index in round_indexes:
        metric_payload = {}
        for metric_name in ('mAP', 'AP50'):
            values_by_seed = {
                seed: round_metrics_by_seed.get(seed, {}).get(round_index, {}).get(metric_name)
                for seed in seeds
                if round_metrics_by_seed.get(seed, {}).get(round_index, {}).get(metric_name) is not None
            }
            metric_payload[metric_name] = _numeric_summary(values_by_seed, seeds)

        selected_by_seed: Dict[int, float] = {}
        duration_by_seed: Dict[int, float] = {}
        for seed in seeds:
            round_summary = _read_round_summary(_seed_output_dir(run_dir, seed), round_index)
            if round_summary is None:
                continue
            if round_summary.get('selected_count') is not None:
                selected_by_seed[seed] = float(round_summary['selected_count'])
            duration = _round_duration_sec(round_summary)
            if duration is not None:
                duration_by_seed[seed] = duration

        rounds_summary.append({
            'round_index': round_index,
            'metrics': metric_payload,
            'selected_count': _numeric_summary(selected_by_seed, seeds),
            'duration_sec': _numeric_summary(duration_by_seed, seeds),
        })

    return {
        'schema_version': 1,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_id': run_id,
        'base_output_dir': str(base_output_dir),
        'run_dir': str(run_dir),
        'method': args.method,
        'method_arg': cfg.get('_method_arg'),
        'detector': detector,
        'dataset': dataset,
        'preset': selection.get('preset_name'),
        'rounds': total_rounds,
        'start_round': args.start_round,
        'budget': int(cfg.get('budget', 0)),
        'gpus': int(cfg.get('gpus', 1)),
        'seeds': seeds,
        'seed_runs': seed_runs,
        'rounds_summary': rounds_summary,
    }


def _write_aggregate_summary(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    base_output_dir: Path,
    run_dir: Path,
    run_id: str,
    seeds: List[int],
    seed_run_summaries: Dict[int, Dict[str, Any]],
    total_rounds: int,
) -> Path:
    path = _aggregate_summary_path(run_dir)
    payload = _build_aggregate_summary(
        args,
        cfg,
        selection,
        base_output_dir,
        run_dir,
        run_id,
        seeds,
        seed_run_summaries,
        total_rounds,
    )
    _write_json(path, payload)
    return path


def _catalog_epilog() -> str:
    lines = [
        'Examples:',
        '  python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1',
        '  python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2',
        '  python -B tools/run_active_learning.py --preset ppal-retinanet-voc --rounds 1 --gpus 1',
        '',
        'Catalog presets:',
    ]
    for preset in list_presets():
        lines.append(
            '  {name}: method={method}, detector={detector}, dataset={dataset}'.format(
                name=preset.name,
                method=preset.method,
                detector=preset.detector,
                dataset=preset.dataset,
            )
        )
    return '\n'.join(lines)


def _print_presets() -> None:
    for preset in list_presets():
        print(json.dumps({
            'name': preset.name,
            'method': preset.method,
            'detector': preset.detector,
            'dataset': preset.dataset,
            'aliases': list(preset.aliases),
            'description': preset.description,
        }, indent=2))


def _unsupported_catalog_message() -> str:
    supported = ', '.join(preset.name for preset in list_presets())
    return (
        'Provide a positional config, --preset, or --method/--detector/--dataset '
        'matching the catalog. Supported presets: %s' % supported
    )


def resolve_runner_selection(args: argparse.Namespace) -> Dict[str, Any]:
    if args.preset and args.config:
        raise SystemExit('Do not combine a positional config with --preset.')
    if args.preset and args.method:
        raise SystemExit('Do not combine --preset with --method; the preset selects the method.')

    if args.config is None:
        requested_method = args.method or 'ppal'
        selection = resolve_experiment(
            method=args.method if args.preset else requested_method,
            detector=args.detector,
            dataset=args.dataset,
            preset=args.preset,
        )
        if selection is None:
            raise SystemExit(_unsupported_catalog_message())
        return {
            'config_path': None,
            'catalog_selection': selection,
            'method': selection.method,
            'method_arg': selection.method_alias,
            'cfg_overrides': dict(selection.cfg_overrides),
            'preset_name': selection.preset.name,
        }

    requested_method = args.method or 'ppal'
    method_selection = resolve_method_alias(requested_method)
    if method_selection is None:
        raise SystemExit('Unsupported method: %s' % requested_method)
    method, cfg_overrides = method_selection
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    return {
        'config_path': config_path.resolve(),
        'catalog_selection': None,
        'method': method,
        'method_arg': requested_method,
        'cfg_overrides': dict(cfg_overrides),
        'preset_name': None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ALOD active learning runner',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=_catalog_epilog(),
    )
    parser.add_argument(
        'config',
        nargs='?',
        type=Path,
        help='Experiment config path. Optional when catalog args resolve a preset.',
    )
    parser.add_argument(
        '--method',
        default=None,
        help='Method or alias: ppal, pal, pal:guide, pal/full, pal:lius, ecpal, coreset, random, entropy',
    )
    parser.add_argument('--detector', default=None, help='Catalog detector, e.g. retinanet')
    parser.add_argument('--dataset', default=None, help='Catalog dataset, e.g. voc')
    parser.add_argument('--preset', default=None, help='Catalog preset name, e.g. pal-retinanet-voc')
    parser.add_argument('--list-presets', action='store_true', help='List supported catalog presets and exit')
    parser.add_argument('--rounds', type=int, default=None, help='Number of AL rounds to execute')
    parser.add_argument('--start-round', type=int, default=1, help='First AL round index to execute')
    parser.add_argument('--seed', type=int, default=None, help='Single run seed. Defaults to 0 when --seeds is omitted')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Run one timestamped experiment over multiple seeds')
    parser.add_argument('--gpus', type=int, default=None, help='Override config gpus for command planning/execution')
    parser.add_argument('--port', type=int, default=None, help='Override distributed master port')
    parser.add_argument('--python-path', default=None, help='Override config python executable')
    parser.add_argument('--verbose', action='store_true', help='Print the full plan and stream subprocess output')
    return parser.parse_args()


def _run_seed(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    output_dir: Path,
    total_rounds: int,
    seed: int,
    preparation_results: List[Dict[str, object]],
) -> Dict[str, Any]:
    init_actions = initialize_round_zero(cfg, output_dir)

    plan: List[PlanStep] = []
    for round_index in range(args.start_round, args.start_round + total_rounds):
        plan.extend(build_round_plan(cfg, output_dir, args.method, round_index, seed))

    plan_log = _write_plan_log(output_dir, plan)
    run_summary = _run_summary_base(args, cfg, selection, output_dir, plan_log, total_rounds, seed)
    run_summary['preparation'] = preparation_results
    _write_json(_run_summary_path(output_dir), run_summary)
    if args.verbose:
        for action in init_actions:
            print(action)
        print('active learning plan:')
        _print_plan(plan)
    elif init_actions:
        print('initial pools: round_00 annotations ready')

    for round_offset, round_index in enumerate(range(args.start_round, args.start_round + total_rounds), start=1):
        round_plan = [step for step in plan if step.round_index == round_index]
        round_results: List[Dict[str, Any]] = []
        print('')
        print('Round %d/%d' % (round_offset, total_rounds))
        for step_index, step in enumerate(round_plan, start=1):
            label = _step_label(step)
            if args.verbose or (tqdm is None and isinstance(step, CommandPlan)):
                print('[%d/%d] %-26s running...' % (step_index, len(round_plan), label), flush=True)
            started = time.time()
            try:
                result = _run_plan_step(step, cfg, output_dir, seed=seed, verbose=args.verbose)
            except Exception as exc:
                result = _failed_step_result(step, exc, started)
                round_results.append(result)
                round_payload = _round_summary_payload(cfg, output_dir, round_index, 'failed', round_results)
                round_summary_path = _round_summary_path(output_dir, round_index)
                _write_json(round_summary_path, round_payload)
                run_summary['status'] = 'failed'
                run_summary['failed_round'] = round_index
                run_summary['failed_step'] = result
                run_summary['round_summaries'].append(str(round_summary_path))
                _write_json(_run_summary_path(output_dir), run_summary)
                if isinstance(exc, RunnerStepError):
                    raise SystemExit(str(exc))
                raise
            round_results.append(result)
            if isinstance(step, AcquisitionPlan) and not args.verbose:
                print(_format_acquisition_result(result), flush=True)
            elif args.verbose or tqdm is None:
                print(_format_step_result(step_index, len(round_plan), result), flush=True)

        round_payload = _round_summary_payload(cfg, output_dir, round_index, 'done', round_results)
        round_summary_path = _round_summary_path(output_dir, round_index)
        _write_json(round_summary_path, round_payload)
        run_summary['round_summaries'].append(str(round_summary_path))
        _write_json(_run_summary_path(output_dir), run_summary)

    run_summary['status'] = 'done'
    _write_json(_run_summary_path(output_dir), run_summary)
    return run_summary


def main() -> None:
    args = parse_args()
    if args.list_presets:
        _print_presets()
        return

    selection = resolve_runner_selection(args)
    args.method = selection['method']
    seeds = _selected_seeds(args)
    if selection['catalog_selection'] is None:
        config_path = selection['config_path']
        cfg = load_experiment_config(config_path)
    else:
        config_path = None
        cfg = build_experiment_config(selection['catalog_selection'])
    cfg.update(selection['cfg_overrides'])
    cfg['_method_arg'] = selection['method_arg']
    apply_cli_overrides(cfg, args)
    validate_experiment_config_paths(cfg)
    output_default = Path('work_dirs') / (config_path.stem if config_path is not None else selection['preset_name'])
    base_output_dir = _resolve_repo_path(str(cfg.get('output_dir', output_default)))

    preparation_results: List[Dict[str, object]] = []
    if _preparation_requested(cfg):
        print('preparing inputs...', flush=True)
        preparation_results = prepare_required_inputs(cfg, ROOT)
        _print_preparation_summary(preparation_results)

    validate_initial_pool_files(cfg)

    total_rounds = int(args.rounds if args.rounds is not None else cfg.get('round_num', 1))
    if total_rounds < 1:
        raise ValueError('round count must be positive')

    run_id = _run_timestamp()
    run_dir = base_output_dir / run_id
    _ensure_new_timestamp_run_dir(run_dir)
    _print_timestamp_run_header(args, cfg, selection, run_dir, seeds, total_rounds)

    seed_run_summaries: Dict[int, Dict[str, Any]] = {}
    for seed_offset, seed in enumerate(seeds, start=1):
        print('')
        print('Seed %d/%d: seed=%d' % (seed_offset, len(seeds), seed))
        seed_dir = _seed_output_dir(run_dir, seed)
        seed_run_summaries[seed] = _run_seed(
            args,
            cfg,
            selection,
            seed_dir,
            total_rounds,
            seed,
            preparation_results,
        )

    aggregate_path = _write_aggregate_summary(
        args,
        cfg,
        selection,
        base_output_dir,
        run_dir,
        run_id,
        seeds,
        seed_run_summaries,
        total_rounds,
    )
    print('')
    print('ALOD run complete')
    print('aggregate=%s' % _display_path(aggregate_path))


if __name__ == '__main__':
    main()
