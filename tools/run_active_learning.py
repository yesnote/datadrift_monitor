"""Windows-safe active learning runner for ALOD."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.catalog import build_experiment_config, list_presets, resolve_experiment, resolve_method_alias

VOC_PREP_HINT = (
    'prepare VOC pools with: python -B datasets/prepare_voc_active_learning.py '
    '--vocdevkit data/VOCdevkit --n-labeled 827 --n-diff 1 --seed 0'
)
PRETRAIN_PREP_HINT = (
    'prepare pretrained weights with: python -B tools/prepare_pretrain_models.py '
    '--output-dir data/pretrain_models'
)
from methods.common.coco_pool import update_labeled_unlabeled_from_oracle
from methods.common.diagnostics import print_acquisition_summary, write_diagnostics
from methods.entropy.sampler import sample as entropy_sample
from methods.pal.acquisition import sample_pal_from_files
from methods.ppal.acquisition import run_diversity_acquisition, run_uncertainty_acquisition
from methods.random.sampler import sample as random_sample


@dataclass
class CommandPlan:
    name: str
    argv: List[str]
    cwd: str
    log_path: Optional[str] = None
    note: str = ''

    def to_dict(self) -> Dict[str, Any]:
        data = {'name': self.name, 'argv': self.argv, 'cwd': self.cwd}
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_not_code_refs(path: Path) -> None:
    resolved = path.resolve()
    code_refs = (ROOT / 'code_refs').resolve()
    if _is_relative_to(resolved, code_refs):
        raise ValueError('Refusing to read/write runtime output under code_refs: %s' % path)


def _resolve_repo_path(value: str, must_be_relative: bool = True) -> Path:
    path = Path(value)
    if must_be_relative and path.is_absolute():
        raise ValueError('Config path must be relative to the repo root: %s' % value)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    root = ROOT.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError('Config path must stay inside the repo root: %s' % value)
    _assert_not_code_refs(resolved)
    return resolved


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


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
        'diversity_infer_config',
        'pal_infer_config',
        'pal_embedding_path',
        'output_dir',
    )
    for key in path_keys:
        value = cfg.get(key)
        if value:
            _resolve_repo_path(str(value))
    if cfg.get('init_model'):
        _resolve_repo_path(str(cfg['init_model']))
    for value in cfg.get('required_files', []):
        _resolve_repo_path(str(value))


def validate_required_files(cfg: Dict[str, Any]) -> None:
    missing = []
    for value in cfg.get('required_files', []):
        path = _resolve_repo_path(str(value))
        if not path.exists():
            missing.append(_display_path(path))
    if missing:
        raise SystemExit(
            'Missing required file(s): %s\n%s'
            % (', '.join(missing), PRETRAIN_PREP_HINT)
        )


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
        hint = str(cfg.get('initial_pool_prep_hint', VOC_PREP_HINT))
        raise SystemExit(
            'Missing initial pool file(s): %s\n%s'
            % (', '.join(missing), hint)
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
        'uncertainty_labeled': ann_dir / 'uncertainty_new_labeled.json',
        'uncertainty_unlabeled': ann_dir / 'uncertainty_new_unlabeled.json',
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
        actions.append(VOC_PREP_HINT)
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
) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    options = {
        'labeled_data': input_paths['labeled'],
        'unlabeled_data': input_paths['unlabeled'],
        'data.train.ann_file': input_paths['labeled'],
    }
    options.update(cfg.get('common_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/train.py',
            str(cfg['train_config']),
            '--work-dir',
            str(round_work_dir),
            '--launcher',
            _launcher_value(cfg),
        ]
    )
    if not _use_distributed(cfg):
        argv += ['--gpus', str(int(cfg.get('gpus', 1)))]
    argv += ['--cfg-options'] + _cfg_options(options)
    return CommandPlan('train_round_%02d' % round_index, argv, str(ROOT))


def _eval_plan(cfg: Dict[str, Any], output_dir: Path, round_index: int) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    latest_ckpt = round_work_dir / 'latest.pth'
    eval_log = round_work_dir / 'eval.txt'
    options = cfg.get('eval_cfg_options', {})
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
    return CommandPlan('eval_round_%02d' % round_index, argv, str(ROOT), log_path=str(eval_log))


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
        'unlabeled_data': input_paths['unlabeled'],
        'data.test.ann_file': input_paths['unlabeled'],
    }
    options.update(cfg.get('common_cfg_options', {}))
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
    return CommandPlan('uncertainty_inference_round_%02d' % round_index, argv, str(ROOT))


def _diversity_infer_plan(cfg: Dict[str, Any], output_dir: Path, round_index: int) -> CommandPlan:
    round_work_dir = _round_dir(output_dir, round_index)
    annotations = _round_annotations(output_dir, round_index)
    prefix = round_work_dir / 'diversity_inference_result'
    latest_ckpt = round_work_dir / 'latest.pth'
    image_dis = round_work_dir / 'image_dis.npy'
    head = 'roi_head' if cfg.get('model_name') == 'fasterrcnn' else 'bbox_head'
    pool_size = cfg.get('uncertainty_pool_size', cfg.get('budget'))
    options = {
        'unlabeled_data': annotations['uncertainty_labeled'],
        'data.test.ann_file': annotations['uncertainty_labeled'],
        'model.%s.total_images' % head: int(pool_size),
        'model.%s.output_path' % head: image_dis,
    }
    options.update(cfg.get('common_cfg_options', {}))
    argv = (
        _command_prefix(cfg)
        + [
            'tools/test.py',
            str(cfg['diversity_infer_config']),
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
    return CommandPlan('diversity_inference_round_%02d' % round_index, argv, str(ROOT))


def _round_relative_file(round_work_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Round output file must be relative: %s' % value)
    resolved = (round_work_dir / path).resolve()
    if not _is_relative_to(resolved, round_work_dir.resolve()):
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
        'unlabeled_data': input_paths[pool_name],
        'data.test.ann_file': input_paths[pool_name],
    }
    options.update(cfg.get('common_cfg_options', {}))
    options.update(cfg.get('pal_cfg_options', {}))
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
        str(ROOT))


def build_round_plan(
    cfg: Dict[str, Any],
    output_dir: Path,
    method: str,
    round_index: int,
) -> List[PlanStep]:
    input_paths = _input_pool_paths(output_dir, round_index)
    plan: List[PlanStep] = [
        _train_plan(cfg, output_dir, round_index, input_paths),
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
        plan.append(_diversity_infer_plan(cfg, output_dir, round_index))
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
    else:
        raise ValueError('Unsupported method: %s' % method)
    return plan


def _write_plan_log(output_dir: Path, plan: List[PlanStep]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'active_learning_plan.json'
    _assert_not_code_refs(path)
    with path.open('w', encoding='utf-8') as handle:
        json.dump([step.to_dict() for step in plan], handle, indent=2)
    return path


def _print_plan(plan: Iterable[PlanStep]) -> None:
    for step in plan:
        print(json.dumps(step.to_dict(), indent=2))


def _run_subprocess_plan(step: CommandPlan) -> None:
    stdout_handle = None
    try:
        if step.log_path:
            log_path = Path(step.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = log_path.open('w', encoding='utf-8')
        env = os.environ.copy()
        existing = env.get('PYTHONPATH')
        env['PYTHONPATH'] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
        subprocess.run(step.argv, cwd=step.cwd, check=True, stdout=stdout_handle, env=env)
    finally:
        if stdout_handle is not None:
            stdout_handle.close()


def _execute_lightweight_acquisition(
    cfg: Dict[str, Any],
    output_dir: Path,
    method: str,
    round_index: int,
    seed: int,
) -> List[Any]:
    input_paths = _input_pool_paths(output_dir, round_index)
    round_work_dir = _round_dir(output_dir, round_index)
    annotations = _round_annotations(output_dir, round_index)
    budget = int(cfg.get('budget', 0))
    diagnostics_path = None
    diagnostics_stage = None

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
        diagnostics_payload = dict(diagnostics)
        diagnostics_payload.update({
            'method': 'pal',
            'stage': diagnostics_stage,
            'round_index': round_index,
            'budget': budget,
            'selected_count': len(selected),
            'inputs': {
                'labeled_pool_json': str(input_paths['labeled']),
                'unlabeled_pool_json': str(input_paths['unlabeled']),
                'labeled_detections_json': str(labeled_dets),
                'unlabeled_detections_json': str(unlabeled_dets),
                'embedding_path': str(embedding_path) if embedding_path else None,
            },
            'outputs': {
                'labeled_pool_json': str(annotations['labeled']),
                'unlabeled_pool_json': str(annotations['unlabeled']),
                'diagnostics_json': str(diagnostics_path),
            },
        })
        write_diagnostics(diagnostics_path, diagnostics_payload)
    else:
        raise ValueError('Unsupported lightweight acquisition method: %s' % method)

    update_labeled_unlabeled_from_oracle(
        _resolve_repo_path(str(cfg['oracle_path'])),
        input_paths['labeled'],
        selected,
        annotations['labeled'],
        annotations['unlabeled'],
    )
    if diagnostics_path is not None:
        print_acquisition_summary(
            method,
            round_index,
            len(selected),
            diagnostics_path,
            stage=diagnostics_stage,
            labeled_json=annotations['labeled'],
            unlabeled_json=annotations['unlabeled'],
        )
    return selected


def _load_ppal_diagnostic_stages(diagnostics_path: Path) -> List[Dict[str, Any]]:
    if not diagnostics_path.exists():
        return []
    with diagnostics_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return list(payload.get('stages', []))


def _execute_ppal_acquisition(
    cfg: Dict[str, Any],
    output_dir: Path,
    round_index: int,
    ppal_stage: str,
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
            out_labeled_json=annotations['uncertainty_labeled'],
            out_unlabeled_json=annotations['uncertainty_unlabeled'],
        )
    else:
        output = run_diversity_acquisition(
            cfg=cfg,
            repo_root=ROOT,
            round_index=round_index,
            result_json=result_json,
            image_distance_npy=round_work_dir / 'image_dis.npy',
            last_labeled_json=input_paths['labeled'],
            out_labeled_json=annotations['labeled'],
            out_unlabeled_json=annotations['unlabeled'],
        )

    diagnostics_path = round_work_dir / 'ppal_diagnostics.json'
    stages = _load_ppal_diagnostic_stages(diagnostics_path)
    runner_stage = str(output.get('runner_stage', ppal_stage))
    stages = [stage for stage in stages if stage.get('runner_stage') != runner_stage]
    stages.append(output)
    stage_names = {str(stage.get('runner_stage')) for stage in stages}
    summary_stage = 'all' if {'uncertainty', 'diversity'}.issubset(stage_names) else ppal_stage
    diagnostics_payload = {
        'method': 'ppal',
        'stage': summary_stage,
        'round_index': round_index,
        'budget': int(output.get('budget', cfg.get('budget', 0))),
        'selected_image_ids': output.get('selected_image_ids', []),
        'selected_count': int(output.get('selected_count', 0)),
        'inputs': {
            'labeled_pool_json': str(input_paths['labeled']),
            'unlabeled_pool_json': str(input_paths['unlabeled']),
            'uncertainty_result_json': str(result_json),
            'image_distance_npy': str(round_work_dir / 'image_dis.npy'),
        },
        'outputs': {
            'labeled_pool_json': output.get('outputs', {}).get('labeled_pool_json'),
            'unlabeled_pool_json': output.get('outputs', {}).get('unlabeled_pool_json'),
            'diagnostics_json': str(diagnostics_path),
        },
        'stages': stages,
    }
    write_diagnostics(diagnostics_path, diagnostics_payload)
    print_acquisition_summary(
        'ppal',
        round_index,
        int(output.get('selected_count', 0)),
        diagnostics_path,
        stage=ppal_stage,
        labeled_json=Path(output.get('outputs', {}).get('labeled_pool_json')),
        unlabeled_json=Path(output.get('outputs', {}).get('unlabeled_pool_json')),
    )
    return output


def _run_acquisition_plan(
    step: AcquisitionPlan,
    cfg: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> None:
    if step.method == 'ppal':
        if step.ppal_stage is None:
            raise ValueError('PPAL acquisition step requires ppal_stage')
        _execute_ppal_acquisition(cfg, output_dir, step.round_index, step.ppal_stage)
        return

    selected = _execute_lightweight_acquisition(
        cfg,
        output_dir,
        step.method,
        step.round_index,
        seed=seed,
    )
    if step.method != 'pal':
        print(json.dumps({'selected_image_ids': selected}, indent=2))


def _run_plan_step(
    step: PlanStep,
    cfg: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> None:
    if isinstance(step, CommandPlan):
        _run_subprocess_plan(step)
        return
    if isinstance(step, AcquisitionPlan):
        _run_acquisition_plan(step, cfg, output_dir, seed)
        return
    raise TypeError('Unsupported plan step: %r' % (step,))


def _catalog_epilog() -> str:
    lines = [
        'Examples:',
        '  python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1',
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
        help='Method or alias: ppal, pal, pal:guide, pal/full, pal:lius, random, entropy',
    )
    parser.add_argument('--detector', default=None, help='Catalog detector, e.g. retinanet')
    parser.add_argument('--dataset', default=None, help='Catalog dataset, e.g. voc')
    parser.add_argument('--preset', default=None, help='Catalog preset name, e.g. pal-retinanet-voc')
    parser.add_argument('--list-presets', action='store_true', help='List supported catalog presets and exit')
    parser.add_argument('--rounds', type=int, default=None, help='Number of AL rounds to execute')
    parser.add_argument('--start-round', type=int, default=1, help='First AL round index to execute')
    parser.add_argument('--seed', type=int, default=0, help='Sampler seed for deterministic methods')
    parser.add_argument('--gpus', type=int, default=None, help='Override config gpus for command planning/execution')
    parser.add_argument('--port', type=int, default=None, help='Override distributed master port')
    parser.add_argument('--python-path', default=None, help='Override config python executable')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        _print_presets()
        return

    selection = resolve_runner_selection(args)
    args.method = selection['method']
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
    output_dir = _resolve_repo_path(str(cfg.get('output_dir', output_default)))
    if selection['preset_name']:
        print('resolved catalog preset: %s' % selection['preset_name'])

    validate_required_files(cfg)
    validate_initial_pool_files(cfg)

    init_actions = initialize_round_zero(cfg, output_dir)
    for action in init_actions:
        print(action)

    total_rounds = int(args.rounds if args.rounds is not None else cfg.get('round_num', 1))
    if total_rounds < 1:
        raise ValueError('round count must be positive')

    plan: List[PlanStep] = []
    for round_index in range(args.start_round, args.start_round + total_rounds):
        plan.extend(build_round_plan(cfg, output_dir, args.method, round_index))

    plan_log = _write_plan_log(output_dir, plan)
    print('wrote command plan: %s' % _display_path(plan_log))
    _print_plan(plan)

    for step in plan:
        _run_plan_step(step, cfg, output_dir, seed=args.seed)


if __name__ == '__main__':
    main()
