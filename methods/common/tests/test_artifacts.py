import hashlib

import pytest

from methods.common.acquisition.artifacts import (
    JsonArtifact,
    JsonArtifactRecord,
    canonical_json_bytes,
    read_json_artifact,
    sha256_file,
    write_json_artifact,
)
from methods.common.data.image_identity import SampleIdentity


def _artifact(metadata=None) -> JsonArtifact:
    sample = SampleIdentity('cityscapes', 'frame-001')
    return JsonArtifact(
        artifact_type='acquisition_scores',
        producer_stage_id='round-1-score',
        metadata={} if metadata is None else metadata,
        records=(
            JsonArtifactRecord(
                sample,
                {'final_score': 1.25, 'components': {'fn': 1.0, 'loc': 1.25}},
            ),
        ),
    )


def test_canonical_json_and_hash_ignore_mapping_insertion_order() -> None:
    first = _artifact({'z': 1, 'a': {'right': 2, 'left': 1}})
    second = _artifact({'a': {'left': 1, 'right': 2}, 'z': 1})

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert hashlib.sha256(canonical_json_bytes(first)).hexdigest() == hashlib.sha256(
        canonical_json_bytes(second)
    ).hexdigest()


def test_artifact_write_read_and_sha256_round_trip(tmp_path) -> None:
    artifact = _artifact()

    reference = write_json_artifact(
        'artifacts/scores.json', artifact, run_directory=tmp_path
    )
    stored_path = tmp_path / reference.relative_path

    assert reference.sha256 == sha256_file(stored_path)
    assert reference.artifact_id == reference.sha256
    assert read_json_artifact(
        stored_path, expected_sha256=reference.sha256
    ) == artifact


def test_artifact_hash_verification_detects_modified_bytes(tmp_path) -> None:
    artifact = _artifact()
    reference = write_json_artifact('scores.json', artifact, run_directory=tmp_path)
    stored_path = tmp_path / reference.relative_path
    stored_path.write_bytes(stored_path.read_bytes() + b' ')

    with pytest.raises(ValueError, match='SHA256'):
        read_json_artifact(stored_path, expected_sha256=reference.sha256)


def test_artifact_rejects_non_json_numbers_and_duplicate_records() -> None:
    sample = SampleIdentity('cityscapes', 'frame-001')
    with pytest.raises(ValueError, match='non-finite'):
        JsonArtifactRecord(sample, {'score': float('nan')})

    record = JsonArtifactRecord(sample, {'score': 1.0})
    with pytest.raises(ValueError, match='duplicate'):
        JsonArtifact('scores', 'stage', (record, record))


def test_artifact_cannot_escape_run_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match='inside'):
        write_json_artifact('../outside.json', _artifact(), run_directory=tmp_path)
