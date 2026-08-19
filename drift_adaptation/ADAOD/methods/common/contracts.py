'''Small public contracts shared by ADAOD method plugins.'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Tuple


@dataclass(frozen=True)
class StageSpec:
    '''One serial, resumable experiment stage.'''

    stage_id: str
    executor_key: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage_id or '/' in self.stage_id or chr(92) in self.stage_id:
            raise ValueError('stage_id must be a non-empty path-safe token')
        if not self.executor_key:
            raise ValueError('executor_key must not be empty')

@dataclass(frozen=True)
class ExperimentPlan:
    '''Ordered stage plan produced by a method plugin.'''

    stages: Tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError('experiment plan contains duplicate stage IDs')


ConfigFactory = Callable[[], Mapping[str, Any]]
PlanFactory = Callable[[Mapping[str, Any]], ExperimentPlan]


@dataclass(frozen=True)
class MethodManifest:
    '''Discovery metadata for a concrete method.'''

    key: str
    api_version: int
    description: str
    config_factory: ConfigFactory
    plan_factory: PlanFactory
    custom_imports: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key or self.key == 'common':
            raise ValueError('method key must identify a concrete method')
        if self.api_version < 1:
            raise ValueError('api_version must be positive')


@dataclass(frozen=True)
class ArtifactRef:
    '''Content-addressed reference to a generated run artifact.'''

    artifact_id: str
    artifact_type: str
    schema_version: int
    producer_stage_id: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('artifact path must be relative to the run directory')
        if len(self.sha256) != 64:
            raise ValueError('artifact sha256 must contain 64 hexadecimal characters')
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError('artifact sha256 is not hexadecimal') from error
