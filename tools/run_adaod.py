'''Run an active domain-adaptation method from its resolved configuration.'''

import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch deterministic CUDA matrix multiplication requires this to be set
# before the first CUDA context is created. Preserve either user-selected
# CuBLAS workspace mode and provide the larger documented mode by default.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from methods.common.artifacts import ArtifactStore, atomic_write_json
from methods.common.engine.context import ExecutionContext
from methods.common.engine.executor_loader import load_executor_factory
from methods.common.engine.runner import StageRunner
from methods.common.engine.state import RunStateStore
from methods.common.progress import ProgressReporter
from methods.common.registry import discover_methods, get_method
from tools.common.config import compose_config, config_fingerprint
from tools.common.paths import repository_relative_path, repository_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='ada-fnp')
    parser.add_argument('--list-methods', action='store_true')
    parser.add_argument('--budget-percent', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset')
    parser.add_argument('--detector')
    parser.add_argument('--runtime', default='default')
    parser.add_argument(
        '--run-directory',
        help='repository-relative run directory (a timestamped default is used)',
    )
    return parser


def _run_directory(config, configured_value=None) -> Path:
    if configured_value is None:
        run_timestamp = datetime.now().astimezone().strftime(
            '%m-%d-%Y_%H_%M'
        )
        configured_value = '{}/runs/{}/{}/{}/seed-{}/{}'.format(
            config['runtime']['work_root'],
            config['method'],
            config['scenario'],
            config['detector']['name'],
            config['seed'],
            run_timestamp,
        )
    relative = PurePosixPath(repository_relative_path(configured_value))
    return repository_root().joinpath(*relative.parts).resolve()


def _prepare_run(config, run_directory: Path) -> RunStateStore:
    state_store = RunStateStore(run_directory / 'state.json')
    config_path = run_directory / 'resolved_config.json'
    fingerprint = config_fingerprint(config)
    if run_directory.exists() and not run_directory.is_dir():
        raise SystemExit('run directory path exists and is not a directory')
    if run_directory.exists() and any(run_directory.iterdir()):
        raise SystemExit(
            'run directory is not empty; choose a new --run-directory or '
            'remove the previous run explicitly'
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config_path,
        {'config_fingerprint': fingerprint, 'config': config},
    )
    state_store.save(state_store.load())
    return state_store


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifests = discover_methods()
    if args.list_methods:
        for key in sorted(manifests):
            print(f'{key}: {manifests[key].description}')
        return 0
    manifest = get_method(args.method)
    overrides = {}
    if args.budget_percent is not None:
        overrides['acquisition'] = {'budget_percent': args.budget_percent}
    if args.seed is not None:
        overrides['seed'] = args.seed
    config = compose_config(
        manifest,
        dataset_key=args.dataset,
        detector_key=args.detector,
        runtime_key=args.runtime,
        overrides=overrides,
    )
    plan = manifest.plan_factory(config)
    run_directory = _run_directory(config, args.run_directory)
    state_store = _prepare_run(config, run_directory)
    progress = ProgressReporter(enabled=os.environ.get('RANK', '0') == '0')
    try:
        context = ExecutionContext(
            config=config,
            repository_root=repository_root(),
            run_directory=run_directory,
            state_store=state_store,
            artifact_store=ArtifactStore(run_directory),
            progress=progress,
        )
        registry = load_executor_factory(manifest)(context)
        StageRunner(registry, state_store, context).run(plan)
    finally:
        progress.close()
    state = state_store.load()
    summary = {
        'run_directory': (
            run_directory.relative_to(repository_root()).as_posix()
        ),
        'status': state.status,
        'completed_stages': len(state.completed_stages),
    }
    if state.completed_stages:
        final_result = state.completed_stages[-1].get('result', {})
        metrics = final_result.get('metrics', {})
        if 'AP50' in metrics:
            summary['AP50'] = metrics['AP50']
    print(json.dumps(summary, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
