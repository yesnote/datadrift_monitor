'''Reusable stages for ADA-FNP-protocol active detection experiments.'''

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol

from methods.common.acquisition.budget import resolve_percentage_budget
from methods.common.acquisition.score_artifacts import (
    AcquisitionArtifact,
    AcquisitionArtifactRecord,
    read_acquisition_artifact,
    write_acquisition_artifact,
)
from methods.common.acquisition.selection import AcquisitionScore, select_top_k
from methods.common.artifacts import sha256_file
from methods.common.contracts import ArtifactRef, StageSpec
from methods.common.data.cityscapes.conversion import prepare_cityscapes_to_foggy
from methods.common.data.cityscapes.reveal import reveal_selected_annotations
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine.context import ExecutionContext
from methods.common.execution.run_files import (
    dataset_cache_directory,
    find_completed_checkpoint,
    find_completed_detector_checkpoint,
    read_active_pool,
    read_pool_state,
    target_labeled_manifest_path,
    target_unlabeled_manifest_path,
    write_pool_state,
)
from methods.common.external_assets import prepare_verified_asset
from methods.common.mmdet.models.backbones.vgg16_caffe_checkpoint import (
    CHECKPOINT_PATH as VGG16_CAFFE_PATH,
    DOWNLOAD_URL as VGG16_CAFFE_URL,
    SHA256 as VGG16_CAFFE_SHA256,
)
from methods.common.protocols.ada_fnp_detection import (
    DETECTOR_CHECKPOINT_ITERATIONS,
    resolve_detector_training_phase,
)


class ActiveDetectionBackend(Protocol):

    def train_detector(self, stage, context, phase, checkpoint_path,
                       continuation_checkpoint) -> Path:
        ...

    def evaluate(self, stage, context, checkpoint_path) -> Mapping[str, float]:
        ...


def repository_path(
    context: ExecutionContext,
    value: str,
    *,
    allow_junction_target: bool = False,
) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('execution path must be repository-relative')
    candidate = context.repository_root.joinpath(*path.parts)
    try:
        candidate.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError('execution path escapes the repository') from error
    if allow_junction_target:
        return candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError('execution path escapes the repository') from error
    return resolved


def artifact_result(artifact: ArtifactRef) -> dict:
    return asdict(artifact)


def update_state(context: ExecutionContext, **updates: Any) -> None:
    state = context.state_store.load()
    for name, value in updates.items():
        if not hasattr(state, name):
            raise AttributeError('unknown run-state field: {}'.format(name))
        setattr(state, name, value)
    context.state_store.save(state)


def record_artifact(
    context: ExecutionContext,
    key: str,
    artifact: ArtifactRef,
) -> None:
    state = context.state_store.load()
    artifact_ids = dict(state.artifact_ids)
    artifact_ids[key] = artifact.artifact_id
    state.artifact_ids = artifact_ids
    context.state_store.save(state)


def prepare_vgg16_caffe_weights(
    stage: StageSpec,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    path = prepare_verified_asset(
        repository_path(context, VGG16_CAFFE_PATH),
        url=VGG16_CAFFE_URL,
        expected_sha256=VGG16_CAFFE_SHA256,
        progress=context.progress,
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError('pretrained asset preparer returned no file')
    try:
        relative_path = path.relative_to(context.repository_root).as_posix()
    except ValueError as error:
        raise ValueError(
            'pretrained asset must stay inside the repository'
        ) from error
    digest = sha256_file(path)
    metadata = context.artifact_store.write_json(
        'artifacts/pretrained.json',
        {'path': relative_path, 'sha256': digest},
        'pretrained_checkpoint_metadata',
        stage.stage_id,
    )
    record_artifact(context, 'pretrained_checkpoint_metadata', metadata)
    return {
        'path': relative_path,
        'sha256': digest,
        'metadata_artifact': artifact_result(metadata),
    }


def prepare_cityscapes_to_foggy_stage(
    stage: StageSpec,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    dataset = context.config['dataset']
    manifest = dict(prepare_cityscapes_to_foggy(
        repository_path(
            context,
            dataset['source']['image_root'],
            allow_junction_target=True,
        ),
        repository_path(
            context,
            dataset['target']['image_root'],
            allow_junction_target=True,
        ),
        repository_path(
            context,
            dataset['source']['annotation_root'],
            allow_junction_target=True,
        ),
        dataset_cache_directory(context),
        context.repository_root,
        expected_train_images=int(dataset['target']['expected_train_images']),
        expected_val_images=int(dataset['target']['expected_eval_images']),
        progress=context.progress,
    ))
    unlabeled_path = (
        dataset_cache_directory(context) / 'target_train_unlabeled.json'
    )
    if not unlabeled_path.is_file():
        raise FileNotFoundError(
            'dataset preparer did not create {!s}'.format(unlabeled_path)
        )
    with unlabeled_path.open('r', encoding='utf-8') as stream:
        unlabeled = json.load(stream)
    samples = tuple(
        SampleIdentity.parse(image['sample_id'])
        for image in unlabeled['images']
    )
    pool = PoolState.initialize(
        samples,
        total_budget=resolve_percentage_budget(context.config),
    )
    pool_artifact = write_pool_state(context, pool, 0, stage.stage_id)
    record_artifact(context, 'target_pool_round_00', pool_artifact)
    return {
        'dataset_manifest': manifest,
        'pool_artifact': artifact_result(pool_artifact),
    }


def train_detector_stage(
    stage: StageSpec,
    context: ExecutionContext,
    backend: ActiveDetectionBackend,
) -> Mapping[str, Any]:
    pool = read_active_pool(context)
    phase = resolve_detector_training_phase(
        int(stage.payload['start_iteration']),
        int(stage.payload['end_iteration']),
        len(pool.labeled),
    )
    checkpoint_path = (
        context.run_directory
        / 'checkpoints'
        / 'detector_{:05d}.pth'.format(phase.end_iteration)
    )
    continuation_checkpoint = find_completed_checkpoint(
        context,
        'detector_checkpoint',
    )
    written = backend.train_detector(
        stage,
        context,
        phase,
        checkpoint_path,
        continuation_checkpoint,
    )
    artifact = context.artifact_store.reference_file(
        written,
        'detector_checkpoint',
        stage.stage_id,
    )
    update_state(context, global_detector_iteration=phase.end_iteration)
    record_artifact(context, 'detector_checkpoint', artifact)
    pool_manifest = target_unlabeled_manifest_path(
        context,
        context.state_store.load().active_round,
    )
    result = {
        'phase': phase.mode.value,
        'checkpoint_artifact': artifact_result(artifact),
    }
    if pool_manifest.is_file():
        pool_artifact = context.artifact_store.reference_file(
            pool_manifest,
            'target_unlabeled_annotations',
            stage.stage_id,
        )
        result['unlabeled_pool_artifact'] = artifact_result(pool_artifact)
    return result


def select_samples_stage(
    stage: StageSpec,
    context: ExecutionContext,
    validate_final_score: Callable[[AcquisitionArtifactRecord], float],
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = read_pool_state(context, round_index - 1)
    score_path = (
        context.run_directory
        / 'artifacts'
        / 'rounds'
        / '{:02d}'.format(round_index)
        / 'scores.json'
    )
    score_key = 'acquisition_scores_round_{:02d}'.format(round_index)
    expected_sha256 = context.state_store.load().artifact_ids.get(score_key)
    if expected_sha256 is None:
        raise RuntimeError('acquisition score artifact is not recorded')
    artifact = read_acquisition_artifact(
        score_path,
        expected_sha256=expected_sha256,
    )
    if artifact.artifact_type != 'acquisition_scores':
        raise ValueError('selection input is not an acquisition-score artifact')
    if artifact.metadata.get('round') != round_index:
        raise ValueError('selection input belongs to a different round')
    if {record.sample for record in artifact.records} != set(pool.unlabeled):
        raise ValueError(
            'selection scores do not cover the unlabeled pool exactly'
        )
    scores = []
    for record in artifact.records:
        final_score = float(validate_final_score(record))
        if not math.isfinite(final_score) or final_score < 0:
            raise ValueError('acquisition final score must be finite and nonnegative')
        scores.append(AcquisitionScore(
            record.sample,
            {'final': final_score},
            1,
        ))
    selected = select_top_k(tuple(scores), int(stage.payload['budget']))
    next_pool = pool.acquire(score.sample for score in selected)
    pool_artifact = write_pool_state(
        context,
        next_pool,
        round_index,
        stage.stage_id,
    )
    selection = AcquisitionArtifact(
        artifact_type='acquisition_selection',
        producer_stage_id=stage.stage_id,
        metadata={'round': round_index, 'budget': len(selected)},
        records=tuple(
            AcquisitionArtifactRecord(
                score.sample,
                {'final_score': score.final_score},
            )
            for score in selected
        ),
    )
    selection_artifact = write_acquisition_artifact(
        context.run_directory
        / 'artifacts'
        / 'rounds'
        / '{:02d}'.format(round_index)
        / 'selection.json',
        selection,
        run_directory=context.run_directory,
    )
    record_artifact(
        context,
        'target_pool_round_{:02d}'.format(round_index),
        pool_artifact,
    )
    record_artifact(
        context,
        'acquisition_selection_round_{:02d}'.format(round_index),
        selection_artifact,
    )
    return {
        'pool_artifact': artifact_result(pool_artifact),
        'selection_artifact': artifact_result(selection_artifact),
    }


def reveal_selected_annotations_stage(
    stage: StageSpec,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = read_pool_state(context, round_index)
    manifest = reveal_selected_annotations(
        dataset_cache_directory(context) / 'target_train_oracle.json',
        pool,
        target_labeled_manifest_path(context, round_index),
    )
    artifact = context.artifact_store.reference_file(
        manifest.path,
        'target_labeled_annotations',
        stage.stage_id,
    )
    update_state(context, active_round=round_index)
    record_artifact(
        context,
        'target_labeled_round_{:02d}'.format(round_index),
        artifact,
    )
    return {
        'image_count': manifest.image_count,
        'annotation_count': manifest.annotation_count,
        'annotation_artifact': artifact_result(artifact),
    }


def evaluate_detector_stage(
    stage: StageSpec,
    context: ExecutionContext,
    backend: ActiveDetectionBackend,
) -> Mapping[str, Any]:
    iteration = int(stage.payload['iteration'])
    if iteration not in DETECTOR_CHECKPOINT_ITERATIONS:
        raise ValueError(
            'evaluation iteration must be a protocol detector checkpoint'
        )
    if context.state_store.load().global_detector_iteration != iteration:
        raise RuntimeError(
            'checkpoint evaluation requires detector iteration {}'.format(
                iteration
            )
        )
    checkpoint = find_completed_detector_checkpoint(
        context,
        iteration,
        stage.payload['detector_executor_key'],
    )
    if checkpoint is None:
        raise FileNotFoundError(
            'evaluation requires completed detector checkpoint {}'.format(
                iteration
            )
        )
    metrics = dict(backend.evaluate(stage, context, checkpoint))
    metric_name = str(stage.payload['metric'])
    if metric_name not in metrics:
        raise ValueError(
            'evaluation backend did not return {}'.format(metric_name)
        )
    if not math.isfinite(float(metrics[metric_name])):
        raise ValueError('evaluation metric must be finite')
    evaluation = {'iteration': iteration, **metrics}
    artifact = context.artifact_store.write_json(
        'artifacts/evaluations/detector_{:05d}.json'.format(iteration),
        evaluation,
        'checkpoint_evaluation_metrics',
        stage.stage_id,
    )
    record_artifact(
        context,
        'evaluation_metrics_{:05d}'.format(iteration),
        artifact,
    )
    context.progress.write_message(
        'checkpoint {:05d}: {}={:.3f}'.format(
            iteration,
            metric_name,
            float(metrics[metric_name]),
        )
    )
    result = {
        'iteration': iteration,
        'metrics': metrics,
        'metrics_artifact': artifact_result(artifact),
    }
    if iteration == int(context.config['training']['max_iterations']):
        final_artifact = context.artifact_store.write_json(
            'artifacts/evaluation.json',
            metrics,
            'evaluation_metrics',
            stage.stage_id,
        )
        record_artifact(context, 'evaluation_metrics', final_artifact)
        result['final_metrics_artifact'] = artifact_result(final_artifact)
    return result
