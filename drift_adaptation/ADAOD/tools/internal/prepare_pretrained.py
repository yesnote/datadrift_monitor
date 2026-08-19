"""Prepare a checksum-pinned pretrained checkpoint for an ADAOD run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.common.external_assets import (
    AssetPreparationError,
    prepare_verified_asset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify a pretrained checkpoint atomically."
    )
    parser.add_argument("--url", required=True, help="Absolute HTTPS download URL.")
    parser.add_argument(
        "--sha256", required=True, help="Expected 64-character SHA-256 digest."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only verify an existing cached file; do not access the network.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    try:
        prepared_path = prepare_verified_asset(
            output,
            url=args.url,
            expected_sha256=args.sha256,
            allow_download=not args.offline,
        )
    except (AssetPreparationError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    print(prepared_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
