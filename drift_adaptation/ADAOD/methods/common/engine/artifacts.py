'''Content-addressed run artifacts and atomic JSON writes.'''

from pathlib import Path
from typing import Any, Mapping

from methods.common.artifacts import atomic_write_json, sha256_file
from methods.common.contracts import ArtifactRef


class ArtifactStore:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory.resolve()

    def write_json(
        self, relative_path: str, value: Mapping[str, Any],
        artifact_type: str, producer_stage_id: str,
    ) -> ArtifactRef:
        target = (self.run_directory / relative_path).resolve()
        if self.run_directory not in target.parents:
            raise ValueError('artifact path escapes the run directory')
        atomic_write_json(target, value)
        digest = sha256_file(target)
        return ArtifactRef(
            artifact_id=digest,
            artifact_type=artifact_type,
            schema_version=1,
            producer_stage_id=producer_stage_id,
            relative_path=target.relative_to(self.run_directory).as_posix(),
            sha256=digest,
        )

    def verify(self, artifact: ArtifactRef) -> None:
        target = self.run_directory / artifact.relative_path
        if not target.is_file() or sha256_file(target) != artifact.sha256:
            raise RuntimeError(f'artifact verification failed: {artifact.artifact_id}')
