'''Run-local paths, active pools, and detector checkpoints.'''

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from methods.common.contracts import ArtifactRef
from methods.common.data.cityscapes.layout import TARGET_TRAIN_NAMESPACE
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine.context import ExecutionContext


def _round_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('round index must be a non-negative integer')
    return value


def _read_json_mapping(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError('{} is missing: {!s}'.format(description, path))
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError('{} root must be a JSON object'.format(description))
    return value


def dataset_cache_directory(context: ExecutionContext) -> Path:
    configured = PurePosixPath(
        context.config['runtime']['dataset_cache_root']
    )
    if configured.is_absolute() or '..' in configured.parts:
        raise ValueError('dataset cache root must be repository-relative')
    path = context.repository_root.joinpath(
        *configured.parts,
        context.config['scenario'],
    ).resolve()
    try:
        path.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError(
            'dataset cache directory escapes the repository'
        ) from error
    return path


def pool_state_path(context: ExecutionContext, round_index: int) -> Path:
    return (
        context.run_directory
        / 'artifacts'
        / 'pool'
        / 'round_{:02d}.json'.format(_round_index(round_index))
    )


def target_labeled_manifest_path(
    context: ExecutionContext,
    round_index: int,
) -> Path:
    return (
        context.run_directory
        / 'datasets'
        / 'target_train_labeled_round_{:02d}.json'.format(
            _round_index(round_index)
        )
    )


def target_unlabeled_manifest_path(
    context: ExecutionContext,
    round_index: int,
) -> Path:
    return (
        context.run_directory
        / 'datasets'
        / 'target_train_unlabeled_pool_{:02d}.json'.format(
            _round_index(round_index)
        )
    )


def find_completed_checkpoint(
    context: ExecutionContext,
    artifact_type: str,
) -> Optional[Path]:
    for completed in reversed(context.state_store.load().completed_stages):
        artifact = completed.get('result', {}).get('checkpoint_artifact')
        if artifact and artifact.get('artifact_type') == artifact_type:
            reference = ArtifactRef(**artifact)
            context.artifact_store.verify(reference)
            return context.run_directory / reference.relative_path
    return None


def find_completed_detector_checkpoint(
    context: ExecutionContext,
    iteration: int,
    executor_key: str,
) -> Optional[Path]:
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError('detector checkpoint iteration must be an integer')
    if iteration <= 0:
        raise ValueError('detector checkpoint iteration must be positive')
    expected_path = 'checkpoints/detector_{:05d}.pth'.format(iteration)
    for completed in reversed(context.state_store.load().completed_stages):
        if completed.get('executor_key') != executor_key:
            continue
        artifact = completed.get('result', {}).get('checkpoint_artifact')
        if not artifact:
            continue
        if (
            artifact.get('artifact_type') != 'detector_checkpoint'
            or artifact.get('relative_path') != expected_path
        ):
            continue
        reference = ArtifactRef(**artifact)
        if reference.producer_stage_id != completed.get('stage_id'):
            raise ValueError(
                'detector checkpoint producer does not match its stage'
            )
        if reference.artifact_id != reference.sha256:
            raise ValueError(
                'detector checkpoint artifact ID must equal its SHA256'
            )
        context.artifact_store.verify(reference)
        return context.run_directory / reference.relative_path
    return None


def read_pool_state(
    context: ExecutionContext,
    round_index: int,
) -> PoolState:
    value = _read_json_mapping(pool_state_path(context, round_index), 'pool state')
    return PoolState.from_dict(value)


def write_pool_state(
    context: ExecutionContext,
    pool: PoolState,
    round_index: int,
    producer_stage_id: str,
) -> ArtifactRef:
    if not isinstance(pool, PoolState):
        raise TypeError('pool must be a PoolState')
    output = pool_state_path(context, round_index)
    relative_path = output.relative_to(context.run_directory).as_posix()
    return context.artifact_store.write_json(
        relative_path,
        pool.to_dict(),
        'target_pool_state',
        producer_stage_id,
    )


def read_active_pool(context: ExecutionContext) -> PoolState:
    return read_pool_state(
        context,
        context.state_store.load().active_round,
    )


def _target_pool_cache(context: ExecutionContext) -> Mapping[str, Any]:
    path = dataset_cache_directory(context) / 'target_train_unlabeled.json'
    source = _read_json_mapping(path, 'target-unlabeled dataset cache')
    images = source.get('images')
    if not isinstance(images, list):
        raise ValueError('target-unlabeled dataset cache images must be a list')
    if source.get('annotations') != []:
        raise ValueError(
            'target-unlabeled dataset cache must not contain annotations'
        )
    if not isinstance(source.get('categories'), list):
        raise ValueError(
            'target-unlabeled dataset cache categories must be a list'
        )
    return source


def materialize_unlabeled_pool_manifest(
    context: ExecutionContext,
    samples: Sequence[SampleIdentity],
    producer_stage_id: str,
    *,
    pool: Optional[PoolState] = None,
) -> Path:
    if pool is None:
        pool = read_active_pool(context)
    samples = tuple(samples)
    if samples != pool.unlabeled:
        raise ValueError('requested samples do not match the active unlabeled pool')
    if not samples:
        raise RuntimeError('active DA requires a nonempty target-unlabeled pool')
    source = _target_pool_cache(context)
    images_by_sample = {}
    for image in source['images']:
        if not isinstance(image, Mapping):
            raise ValueError('target pool cache images must be JSON objects')
        sample = SampleIdentity.parse(image.get('sample_id'))
        if sample.namespace != TARGET_TRAIN_NAMESPACE:
            raise ValueError('target pool cache contains an invalid namespace')
        if sample in images_by_sample:
            raise ValueError('target pool cache contains duplicate sample IDs')
        images_by_sample[sample] = image
    if set(images_by_sample) != set(pool.universe):
        raise ValueError('target pool cache does not match the committed universe')
    output = target_unlabeled_manifest_path(
        context,
        context.state_store.load().active_round,
    )
    relative_path = output.relative_to(context.run_directory).as_posix()
    context.artifact_store.write_json(
        relative_path,
        {
            'info': dict(source.get('info', {})),
            'images': [images_by_sample[sample] for sample in samples],
            'annotations': [],
            'categories': source['categories'],
        },
        'target_unlabeled_annotations',
        producer_stage_id,
    )
    return output


def map_pool_samples_by_image_id(
    manifest_path: Path,
    expected_samples: Sequence[SampleIdentity],
) -> Mapping[int, SampleIdentity]:
    expected_samples = tuple(expected_samples)
    if len(expected_samples) != len(set(expected_samples)):
        raise ValueError('expected pool contains duplicate sample identities')
    manifest = _read_json_mapping(
        Path(manifest_path),
        'target-unlabeled pool manifest',
    )
    images = manifest.get('images')
    if not isinstance(images, list):
        raise ValueError('target pool manifest images must be a list')
    mapping = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError('target pool manifest images must be JSON objects')
        image_id = image.get('id')
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError('target pool image ID must be an integer')
        sample = SampleIdentity.parse(image.get('sample_id'))
        if sample.namespace != TARGET_TRAIN_NAMESPACE:
            raise ValueError('target pool manifest contains an invalid namespace')
        if image_id in mapping:
            raise ValueError('target pool manifest contains duplicate image IDs')
        mapping[image_id] = sample
    if (
        len(mapping) != len(expected_samples)
        or set(mapping.values()) != set(expected_samples)
    ):
        raise ValueError('target pool manifest does not cover expected samples')
    return mapping
