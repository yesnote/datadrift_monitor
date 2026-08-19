'''Atomic file operations and content-addressed run artifacts.'''

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

from methods.common.contracts import ArtifactRef


PathLike = Union[str, os.PathLike]


def sha256_bytes(payload: bytes) -> str:
    '''Return the hexadecimal SHA-256 digest of an in-memory payload.'''

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: PathLike) -> str:
    '''Return the hexadecimal SHA-256 digest of a file.'''

    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    '''Serialize a JSON-compatible value deterministically.'''

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def resolve_artifact_path(
    path: PathLike,
    run_directory: PathLike,
) -> Tuple[Path, str]:
    '''Resolve one artifact file beneath a run directory.'''

    run_root = Path(run_directory).resolve()
    candidate = Path(path)
    target = (
        candidate.resolve()
        if candidate.is_absolute()
        else (run_root / candidate).resolve()
    )
    try:
        relative_path = target.relative_to(run_root).as_posix()
    except ValueError as error:
        raise ValueError('artifact path must stay inside the run directory') from error
    if target == run_root:
        raise ValueError('artifact path must identify a file')
    return target, relative_path


def atomic_write_bytes(path: PathLike, payload: bytes) -> None:
    '''Atomically replace a file with the supplied bytes.'''

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=str(target.parent),
            prefix='.{}.'.format(target.name),
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(target))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: PathLike, value: Mapping[str, Any]) -> None:
    '''Atomically write a human-readable JSON mapping.'''

    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')
    atomic_write_bytes(path, payload)


class ArtifactStore:
    '''Create and verify artifacts contained by one experiment run.'''

    def __init__(self, run_directory: PathLike) -> None:
        self.run_directory = Path(run_directory).resolve()

    def write_json(
        self,
        relative_path: str,
        value: Mapping[str, Any],
        artifact_type: str,
        producer_stage_id: str,
    ) -> ArtifactRef:
        target, resolved_relative_path = resolve_artifact_path(
            relative_path,
            self.run_directory,
        )
        atomic_write_json(target, value)
        return self.reference_file(
            target,
            artifact_type,
            producer_stage_id,
            expected_relative_path=resolved_relative_path,
        )

    def reference_file(
        self,
        path: PathLike,
        artifact_type: str,
        producer_stage_id: str,
        *,
        expected_relative_path: Optional[str] = None,
    ) -> ArtifactRef:
        target, relative_path = resolve_artifact_path(path, self.run_directory)
        if expected_relative_path is not None and relative_path != expected_relative_path:
            raise ValueError('artifact relative path changed during creation')
        if not target.is_file():
            raise FileNotFoundError(
                'artifact file does not exist: {!s}'.format(target)
            )
        digest = sha256_file(target)
        return ArtifactRef(
            artifact_id=digest,
            artifact_type=artifact_type,
            schema_version=1,
            producer_stage_id=producer_stage_id,
            relative_path=relative_path,
            sha256=digest,
        )

    def verify(self, artifact: ArtifactRef) -> None:
        target, _ = resolve_artifact_path(
            artifact.relative_path,
            self.run_directory,
        )
        if not target.is_file() or sha256_file(target) != artifact.sha256:
            raise RuntimeError(
                'artifact verification failed: {}'.format(artifact.artifact_id)
            )
