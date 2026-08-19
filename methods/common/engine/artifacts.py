'''Content-addressed run artifacts and atomic JSON writes.'''

from pathlib import Path
from typing import Any, Mapping, Optional

from methods.common.artifacts import (
    atomic_write_json,
    resolve_artifact_path,
    sha256_file,
)
from methods.common.contracts import ArtifactRef


class ArtifactStore:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory.resolve()

    def write_json(
        self, relative_path: str, value: Mapping[str, Any],
        artifact_type: str, producer_stage_id: str,
    ) -> ArtifactRef:
        target, resolved_relative_path = resolve_artifact_path(
            relative_path, self.run_directory
        )
        atomic_write_json(target, value)
        return self.reference_file(
            target, artifact_type, producer_stage_id,
            expected_relative_path=resolved_relative_path,
        )

    def reference_file(
        self,
        path: Path,
        artifact_type: str,
        producer_stage_id: str,
        *,
        expected_relative_path: Optional[str] = None,
    ) -> ArtifactRef:
        target, relative_path = resolve_artifact_path(path, self.run_directory)
        if expected_relative_path is not None and relative_path != expected_relative_path:
            raise ValueError('artifact relative path changed during creation')
        if not target.is_file():
            raise FileNotFoundError('artifact file does not exist: {!s}'.format(target))
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
            artifact.relative_path, self.run_directory
        )
        if not target.is_file() or sha256_file(target) != artifact.sha256:
            raise RuntimeError(f'artifact verification failed: {artifact.artifact_id}')
