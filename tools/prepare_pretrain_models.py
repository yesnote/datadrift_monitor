"""Download pretrained backbone weights used by ALOD experiments."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
RESNET50_SHA256_PREFIX = '19c8e357'


def sha256_prefix(path: Path, prefix_length: int) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:prefix_length]


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    print('Downloading %s' % url)
    print('Output: %s' % output)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(output)


def ensure_resnet50(output_dir: Path) -> Path:
    output = output_dir / 'resnet50-19c8e357.pth'
    if output.exists():
        actual = sha256_prefix(output, len(RESNET50_SHA256_PREFIX))
        if actual != RESNET50_SHA256_PREFIX:
            raise RuntimeError(
                'Existing file hash prefix mismatch: %s, expected %s for %s'
                % (actual, RESNET50_SHA256_PREFIX, output)
            )
        print('Already present: %s' % output)
        return output

    download(RESNET50_URL, output)
    actual = sha256_prefix(output, len(RESNET50_SHA256_PREFIX))
    if actual != RESNET50_SHA256_PREFIX:
        raise RuntimeError(
            'Downloaded file hash prefix mismatch: %s, expected %s'
            % (actual, RESNET50_SHA256_PREFIX)
        )
    print('Verified sha256 prefix: %s' % actual)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare pretrained model files')
    parser.add_argument('--output-dir', type=Path, default=Path('data/pretrain_models'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_resnet50(args.output_dir)


if __name__ == '__main__':
    main()
