'''Resolved resources shared by all executors in one run.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from methods.common.artifacts import ArtifactStore

from .state import RunStateStore


@dataclass(frozen=True)
class ExecutionContext:
    '''Immutable run-level inputs plus the stores used for durable state.'''

    config: Mapping[str, Any]
    repository_root: Path
    run_directory: Path
    state_store: RunStateStore
    artifact_store: ArtifactStore
    offline: bool = False

    def __post_init__(self) -> None:
        repository_root = Path(self.repository_root).resolve()
        run_directory = Path(self.run_directory).resolve()
        try:
            run_directory.relative_to(repository_root)
        except ValueError as error:
            raise ValueError('run directory must stay inside the repository') from error
        object.__setattr__(self, 'repository_root', repository_root)
        object.__setattr__(self, 'run_directory', run_directory)
        if self.artifact_store.run_directory != run_directory:
            raise ValueError('artifact store belongs to a different run directory')
        if self.state_store.path.resolve().parent != run_directory:
            raise ValueError('state store must be located directly in the run directory')
