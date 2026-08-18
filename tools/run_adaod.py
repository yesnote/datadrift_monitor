'''Resolve a method plugin and inspect its serial stage plan.'''

import argparse
import json

from tools.common.config import compose_config, config_fingerprint
from tools.common.discovery import discover_method_manifests, get_method_manifest


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
    return parser


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
    raise SystemExit(
        'execution is disabled until environment and executors pass their gates; '
        'use --dry-run'
    )


if __name__ == '__main__':
    raise SystemExit(main())
