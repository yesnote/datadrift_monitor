'''Inspect deterministic catalog composition without executing a stage.'''

import argparse
import json

from tools.common.config import compose_config, config_fingerprint
from tools.common.discovery import get_method_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--detector')
    parser.add_argument('--runtime', default='default')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = get_method_manifest(args.method)
    config = compose_config(
        manifest,
        dataset_key=args.dataset,
        detector_key=args.detector,
        runtime_key=args.runtime,
    )
    output = {
        'config': config,
        'fingerprint': config_fingerprint(config),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

