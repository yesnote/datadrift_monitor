'''Shared atomic file writing and content hashing.'''

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=str(path.parent), prefix=f'.{path.name}.',
            suffix='.tmp', delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')
    atomic_write_bytes(path, payload)
