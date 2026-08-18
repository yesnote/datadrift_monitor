'''Canonical JSON acquisition artifacts with content hashes.'''

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from methods.common.artifacts import atomic_write_bytes, sha256_file
from methods.common.contracts import ArtifactRef
from methods.common.data.image_identity import SampleIdentity


ARTIFACT_SCHEMA_VERSION = 1


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('{} contains a non-finite number'.format(path))
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('{} contains a non-string JSON key'.format(path))
            frozen[key] = _freeze_json(item, '{}.{}'.format(path, key))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, '{}[{}]'.format(path, index))
            for index, item in enumerate(value)
        )
    raise TypeError('{} contains a non-JSON value: {!r}'.format(path, type(value)))


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_token(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('{} must be a non-empty, trimmed string'.format(name))


@dataclass(frozen=True)
class JsonArtifactRecord:
    '''One sample-keyed record in a canonical JSON artifact.'''

    sample: SampleIdentity
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, SampleIdentity):
            raise TypeError('artifact record sample must be a SampleIdentity')
        object.__setattr__(self, 'fields', _freeze_json(self.fields, 'record.fields'))

    def to_dict(self) -> dict:
        return {'sample': self.sample.to_dict(), 'fields': _thaw_json(self.fields)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'JsonArtifactRecord':
        if set(value) != {'sample', 'fields'}:
            raise ValueError('artifact record must contain sample and fields')
        return cls(
            sample=SampleIdentity.from_dict(value['sample']),
            fields=value['fields'],
        )


@dataclass(frozen=True)
class JsonArtifact:
    '''Versioned collection of immutable sample records.'''

    artifact_type: str
    producer_stage_id: str
    records: Tuple[JsonArtifactRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_token('artifact_type', self.artifact_type)
        _require_token('producer_stage_id', self.producer_stage_id)
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError('unsupported JSON artifact schema version')
        object.__setattr__(self, 'records', tuple(self.records))
        if any(not isinstance(record, JsonArtifactRecord) for record in self.records):
            raise TypeError('artifact records must be JsonArtifactRecord instances')
        samples = [record.sample for record in self.records]
        if len(samples) != len(set(samples)):
            raise ValueError('artifact contains duplicate sample records')
        object.__setattr__(self, 'metadata', _freeze_json(self.metadata, 'metadata'))

    def to_dict(self) -> dict:
        return {
            'schema_version': self.schema_version,
            'artifact_type': self.artifact_type,
            'producer_stage_id': self.producer_stage_id,
            'metadata': _thaw_json(self.metadata),
            'records': [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'JsonArtifact':
        expected_keys = {
            'schema_version',
            'artifact_type',
            'producer_stage_id',
            'metadata',
            'records',
        }
        if set(value) != expected_keys:
            raise ValueError('JSON artifact has an invalid top-level schema')
        return cls(
            schema_version=value['schema_version'],
            artifact_type=value['artifact_type'],
            producer_stage_id=value['producer_stage_id'],
            metadata=value['metadata'],
            records=tuple(
                JsonArtifactRecord.from_dict(record) for record in value['records']
            ),
        )


def canonical_json_bytes(artifact: JsonArtifact) -> bytes:
    '''Serialize an artifact deterministically for hashing and storage.'''

    if not isinstance(artifact, JsonArtifact):
        raise TypeError('artifact must be a JsonArtifact')
    return json.dumps(
        artifact.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def _resolve_artifact_path(path: Path, run_directory: Path) -> Tuple[Path, str]:
    run_root = Path(run_directory).resolve()
    candidate = Path(path)
    target = candidate.resolve() if candidate.is_absolute() else (run_root / candidate).resolve()
    try:
        relative_path = target.relative_to(run_root).as_posix()
    except ValueError as error:
        raise ValueError('artifact path must stay inside the run directory') from error
    if target == run_root:
        raise ValueError('artifact path must identify a file')
    return target, relative_path


def write_json_artifact(
    path: Path,
    artifact: JsonArtifact,
    *,
    run_directory: Path,
) -> ArtifactRef:
    '''Atomically write an artifact and return its content-addressed reference.'''

    target, relative_path = _resolve_artifact_path(path, run_directory)
    payload = canonical_json_bytes(artifact)
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(target, payload)
    return ArtifactRef(
        artifact_id=digest,
        artifact_type=artifact.artifact_type,
        schema_version=artifact.schema_version,
        producer_stage_id=artifact.producer_stage_id,
        relative_path=relative_path,
        sha256=digest,
    )


def read_json_artifact(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> JsonArtifact:
    '''Read an artifact, optionally verifying its exact stored bytes.'''

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError('JSON artifact SHA256 mismatch')
    try:
        decoded = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('artifact is not valid UTF-8 JSON') from error
    if not isinstance(decoded, Mapping):
        raise ValueError('JSON artifact root must be an object')
    return JsonArtifact.from_dict(decoded)
