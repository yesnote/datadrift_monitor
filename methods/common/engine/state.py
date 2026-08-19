'''Versioned mutable run state.'''

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from methods.common.artifacts import atomic_write_json


@dataclass
class RunState:
    schema_version: int = 2
    status: str = 'pending'
    active_stage_id: Optional[str] = None
    completed_stages: List[Dict[str, Any]] = field(default_factory=list)
    failed_stage_id: Optional[str] = None
    global_detector_iteration: int = 0
    active_round: int = 0
    artifact_ids: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> 'RunState':
        if value.get('schema_version') != 2:
            raise ValueError('unsupported run state schema')
        return cls(**value)


class RunStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> RunState:
        if not self.path.exists():
            return RunState()
        with self.path.open('r', encoding='utf-8') as stream:
            return RunState.from_dict(json.load(stream))

    def save(self, state: RunState) -> None:
        atomic_write_json(self.path, state.to_dict())
