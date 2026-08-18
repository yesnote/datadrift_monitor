'''Resolve a method plugin and inspect its serial stage plan.'''

import argparse
import json
from pathlib import Path, PurePosixPath
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.common.artifacts import atomic_write_json
from methods.common.engine import (
    ArtifactStore,
    ExecutionContext,
    RunStateStore,
    StageRunner,
    load_executor_factory,
)
from tools.common.config import compose_config, config_fingerprint
from tools.common.discovery import discover_method_manifests, get_method_manifest
from tools.common.paths import repository_relative_path, repository_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='ada-fnp')
    parser.add_argument('--list-methods', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--budget-percent', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset')
    parser.add_argument('--detector')
    parser.add_argument('--runtime', default='default')
    parser.add_argument(
        '--run-directory',
        help='repository-relative run directory (a deterministic default is used)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='resume an existing run after verifying its resolved configuration',
    )
    parser.add_argument(
        '--offline', action='store_true',
        help='forbid downloads and require cached external assets',
    )
    return parser


def _run_directory(config, configured_value=None) -> Path:
    if configured_value is None:
        configured_value = '{}/runs/{}/{}/{}/seed-{}'.format(
            config['runtime']['work_root'],
            config['method'],
            config['scenario'],
            config['detector']['name'],
            config['seed'],
        )
    relative = PurePosixPath(repository_relative_path(configured_value))
    return repository_root().joinpath(*relative.parts).resolve()


def _prepare_run(config, run_directory: Path, resume: bool) -> RunStateStore:
    state_store = RunStateStore(run_directory / 'state.json')
    config_path = run_directory / 'resolved_config.json'
    fingerprint = config_fingerprint(config)
    if resume:
        if not config_path.is_file():
            raise SystemExit('--resume requires an existing resolved config')
        with config_path.open('r', encoding='utf-8') as stream:
            saved = json.load(stream)
        if saved.get('config_fingerprint') != fingerprint:
            raise SystemExit('resolved config differs from the run being resumed')
        if not state_store.path.is_file():
            unexpected = [
                path for path in run_directory.iterdir()
                if path.resolve() != config_path.resolve()
            ]
            if unexpected:
                raise SystemExit(
                    'run state is missing but stage outputs exist; refusing recovery'
                )
            state_store.save(state_store.load())
        return state_store
    if state_store.path.exists() or config_path.exists():
        raise SystemExit(
            'run state already exists; use --resume or choose --run-directory'
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
    manifests = discover_method_manifests()
    if args.list_methods:
        for key in sorted(manifests):
            print(f'{key}: {manifests[key].description}')
        return 0
    manifest = get_method_manifest(args.method)
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
    if args.dry_run:
        print(json.dumps({
            'config': config,
            'config_fingerprint': config_fingerprint(config),
            'plan': plan.to_dict(),
        }, indent=2))
        return 0
    run_directory = _run_directory(config, args.run_directory)
    state_store = _prepare_run(config, run_directory, args.resume)
    context = ExecutionContext(
        config=config,
        repository_root=repository_root(),
        run_directory=run_directory,
        state_store=state_store,
        artifact_store=ArtifactStore(run_directory),
        resume=args.resume,
        offline=args.offline,
    )
    registry = load_executor_factory(manifest)(context)
    StageRunner(registry, state_store, context).run(plan)
    state = state_store.load()
    print(json.dumps({
        'run_directory': run_directory.relative_to(repository_root()).as_posix(),
        'status': state.status,
        'completed_stages': len(state.completed_stages),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
