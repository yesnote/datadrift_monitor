'''Serial ADA-FNP experiment stages and their executor registry.'''

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from methods.ada_fnp.acquisition.scoring import (
    RawAcquisitionScore,
    normalize_acquisition_scores,
)
from methods.ada_fnp.schedule import (
    ACQUISITION_MILESTONES,
    resolve_detector_training_phase,
    resolve_total_budget,
    validate_false_negative_training_round,
)
from methods.ada_fnp.training.false_negative_training import (
    build_false_negative_checkpoint_payload,
    build_false_negative_round_optimization,
    run_false_negative_training_steps,
)
from methods.common.acquisition.score_artifacts import (
    AcquisitionArtifact,
    AcquisitionArtifactRecord,
    read_acquisition_artifact,
    write_acquisition_artifact,
)
from methods.common.acquisition.selection import AcquisitionScore, select_top_k
from methods.common.artifacts import sha256_file
from methods.common.contracts import ArtifactRef, StageSpec
from methods.common.data.cityscapes.conversion import (
    prepare_cityscapes_to_foggy,
)
from methods.common.data.cityscapes.reveal import reveal_selected_annotations
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine.checkpoint import (
    save_checkpoint,
)
from methods.common.engine.context import ExecutionContext
from methods.common.engine.runner import StageExecutorRegistry
from methods.common.external_assets import prepare_verified_asset
from methods.common.mmdet.models.backbones.vgg16_caffe_checkpoint import (
    CHECKPOINT_PATH as VGG16_CAFFE_PATH,
    DOWNLOAD_URL as VGG16_CAFFE_URL,
    SHA256 as VGG16_CAFFE_SHA256,
)

from .mmdet_backend import MmdetExecutionBackend
from .run_files import (
    dataset_cache_directory,
    find_completed_checkpoint,
    read_active_pool,
    read_pool_state,
    target_labeled_manifest_path,
    target_unlabeled_manifest_path,
    write_pool_state,
)


def _repository_path(
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


def _artifact_result(artifact: ArtifactRef) -> dict:
    return asdict(artifact)


def _update_state(context: ExecutionContext, **updates: Any) -> None:
    state = context.state_store.load()
    for name, value in updates.items():
        if not hasattr(state, name):
            raise AttributeError('unknown run-state field: {}'.format(name))
        setattr(state, name, value)
    context.state_store.save(state)


def _record_artifact(
    context: ExecutionContext,
    key: str,
    artifact: ArtifactRef,
) -> None:
    state = context.state_store.load()
    artifact_ids = dict(state.artifact_ids)
    artifact_ids[key] = artifact.artifact_id
    state.artifact_ids = artifact_ids
    context.state_store.save(state)


def _prepare_vgg16_caffe_weights(
    stage: StageSpec,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    path = prepare_verified_asset(
        _repository_path(context, VGG16_CAFFE_PATH),
        url=VGG16_CAFFE_URL,
        expected_sha256=VGG16_CAFFE_SHA256,
        allow_download=not context.offline,
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
    _record_artifact(context, 'pretrained_checkpoint_metadata', metadata)
    return {
        'path': relative_path,
        'sha256': digest,
        'metadata_artifact': _artifact_result(metadata),
    }


def _prepare_cityscapes_to_foggy(
    stage: StageSpec,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    dataset = context.config['dataset']
    manifest = dict(prepare_cityscapes_to_foggy(
        _repository_path(
            context,
            dataset['source']['image_root'],
            allow_junction_target=True,
        ),
        _repository_path(
            context,
            dataset['target']['image_root'],
            allow_junction_target=True,
        ),
        _repository_path(
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
        total_budget=resolve_total_budget(context.config),
    )
    pool_artifact = write_pool_state(
        context,
        pool,
        0,
        stage.stage_id,
    )
    _record_artifact(context, 'target_pool_round_00', pool_artifact)
    return {
        'dataset_manifest': manifest,
        'pool_artifact': _artifact_result(pool_artifact),
    }


def _train_detector(
    stage: StageSpec,
    context: ExecutionContext,
    backend: MmdetExecutionBackend,
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
    _update_state(
        context,
        global_detector_iteration=phase.end_iteration,
    )
    _record_artifact(context, 'detector_checkpoint', artifact)
    pool_manifest = target_unlabeled_manifest_path(
        context,
        context.state_store.load().active_round,
    )
    result = {
        'phase': phase.mode.value,
        'checkpoint_artifact': _artifact_result(artifact),
    }
    if pool_manifest.is_file():
        pool_artifact = context.artifact_store.reference_file(
            pool_manifest,
            'target_unlabeled_annotations',
            stage.stage_id,
        )
        result['unlabeled_pool_artifact'] = _artifact_result(pool_artifact)
    return result


def _train_false_negative_predictor(
    stage: StageSpec,
    context: ExecutionContext,
    backend: MmdetExecutionBackend,
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    end_iteration = int(stage.payload['iterations'])
    validate_false_negative_training_round(round_index)
    expected_detector_iteration = ACQUISITION_MILESTONES[round_index - 1]
    if (
        context.state_store.load().global_detector_iteration
        != expected_detector_iteration
    ):
        raise RuntimeError(
            'false-negative predictor training requires detector iteration '
            '{}'.format(expected_detector_iteration)
        )
    checkpoint_path = (
        context.run_directory
        / 'checkpoints'
        / 'false_negative_predictor_round_{:02d}.pth'.format(round_index)
    )
    session = backend.create_false_negative_training_session(
        stage,
        context,
    )
    predictor_config = context.config['false_negative_predictor']
    optimizer, scheduler = build_false_negative_round_optimization(
        session.predictor,
        float(predictor_config['lr']),
    )
    context.progress.start_task(end_iteration, 'iter')
    result = run_false_negative_training_steps(
        session.predictor,
        session.teacher,
        optimizer,
        scheduler,
        session.source_batch_provider,
        session.teacher_batch_extractor,
        start_iteration=0,
        end_iteration=end_iteration,
        labeled_target_batch_provider=(
            session.labeled_target_batch_provider
        ),
        step_completed_callback=lambda completed_iteration, loss: (
            context.progress.set_completed(
                completed_iteration,
                loss=loss,
            )
        ),
    )
    save_checkpoint(
        checkpoint_path,
        {
            'false_negative_predictor': (
                build_false_negative_checkpoint_payload(
                    session.predictor,
                    round_index=round_index,
                )
            ),
        },
    )
    artifact = context.artifact_store.reference_file(
        checkpoint_path,
        'false_negative_predictor_checkpoint',
        stage.stage_id,
    )
    _record_artifact(
        context,
        'false_negative_predictor_checkpoint',
        artifact,
    )
    return {
        'iteration': result.iteration,
        'mean_loss': (
            sum(result.losses) / len(result.losses)
            if result.losses
            else None
        ),
        'checkpoint_artifact': _artifact_result(artifact),
    }


def _score_unlabeled_pool(
    stage: StageSpec,
    context: ExecutionContext,
    backend: MmdetExecutionBackend,
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = read_pool_state(context, round_index - 1)
    raw_records = tuple(backend.score_pool(stage, context, pool.unlabeled))
    if any(
        not isinstance(record, RawAcquisitionScore)
        for record in raw_records
    ):
        raise TypeError(
            'score backend must return RawAcquisitionScore records'
        )
    if {record.sample for record in raw_records} != set(pool.unlabeled):
        raise ValueError(
            'score backend must cover the current unlabeled pool exactly'
        )
    acquisition_config = context.config['acquisition']
    scores = normalize_acquisition_scores(
        raw_records,
        constant_component_value=float(
            acquisition_config['constant_score_normalized_value']
        ),
        empty_detection_score=float(
            acquisition_config['empty_detection_final_score']
        ),
    )
    raw_by_sample = {record.sample: record for record in raw_records}
    artifact = AcquisitionArtifact(
        artifact_type='acquisition_scores',
        producer_stage_id=stage.stage_id,
        metadata={'round': round_index},
        records=tuple(
            AcquisitionArtifactRecord(
                score.sample,
                {
                    'raw': {
                        'false_negative': (
                            raw_by_sample[score.sample].false_negative
                        ),
                        'localization': (
                            raw_by_sample[score.sample].localization
                        ),
                        'entropy': raw_by_sample[score.sample].entropy,
                        'diversity': raw_by_sample[score.sample].diversity,
                    },
                    'normalized': dict(score.components),
                    'source_domain_probability': (
                        raw_by_sample[
                            score.sample
                        ].source_domain_probability
                    ),
                    'detection_count': score.detection_count,
                    'final_score': score.final_score,
                },
            )
            for score in scores
        ),
    )
    reference = write_acquisition_artifact(
        context.run_directory
        / 'artifacts'
        / 'rounds'
        / '{:02d}'.format(round_index)
        / 'scores.json',
        artifact,
        run_directory=context.run_directory,
    )
    _record_artifact(
        context,
        'acquisition_scores_round_{:02d}'.format(round_index),
        reference,
    )
    pool_manifest = target_unlabeled_manifest_path(context, round_index - 1)
    result = {
        'score_artifact': _artifact_result(reference),
        'sample_count': len(scores),
    }
    if pool_manifest.is_file():
        pool_artifact = context.artifact_store.reference_file(
            pool_manifest,
            'target_unlabeled_annotations',
            stage.stage_id,
        )
        result['unlabeled_pool_artifact'] = _artifact_result(pool_artifact)
    return result


def _select_samples(
    stage: StageSpec,
    context: ExecutionContext,
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
    raw_records = []
    for record in artifact.records:
        expected_fields = {
            'raw',
            'normalized',
            'source_domain_probability',
            'detection_count',
            'final_score',
        }
        if set(record.fields) != expected_fields:
            raise ValueError(
                'acquisition score record has an invalid schema'
            )
        raw = record.fields['raw']
        if set(raw) != {
            'false_negative',
            'localization',
            'entropy',
            'diversity',
        }:
            raise ValueError(
                'raw acquisition components have an invalid schema'
            )
        normalized = record.fields['normalized']
        if not isinstance(normalized, Mapping) or set(normalized) != set(raw):
            raise ValueError(
                'normalized acquisition components have an invalid schema'
            )
        detection_count = record.fields['detection_count']
        if isinstance(detection_count, bool) or not isinstance(
            detection_count,
            int,
        ):
            raise TypeError(
                'acquisition detection_count must be an integer'
            )
        raw_records.append(RawAcquisitionScore(
            sample=record.sample,
            false_negative=float(raw['false_negative']),
            localization=float(raw['localization']),
            entropy=float(raw['entropy']),
            diversity=float(raw['diversity']),
            source_domain_probability=float(
                record.fields['source_domain_probability']
            ),
            detection_count=detection_count,
        ))
    acquisition_config = context.config['acquisition']
    normalized_by_sample = {
        score.sample: score
        for score in normalize_acquisition_scores(
            tuple(raw_records),
            constant_component_value=float(
                acquisition_config['constant_score_normalized_value']
            ),
            empty_detection_score=float(
                acquisition_config['empty_detection_final_score']
            ),
        )
    }
    scores = []
    for record in artifact.records:
        score = AcquisitionScore(
            record.sample,
            record.fields['normalized'],
            record.fields['detection_count'],
            empty_detection_score=float(
                acquisition_config['empty_detection_final_score']
            ),
        )
        expected = normalized_by_sample[record.sample]
        if dict(score.components) != dict(expected.components):
            raise ValueError(
                'stored normalized acquisition components are inconsistent'
            )
        if not math.isclose(
            score.final_score,
            float(record.fields['final_score']),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('stored acquisition product is inconsistent')
        scores.append(score)
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
    _record_artifact(
        context,
        'target_pool_round_{:02d}'.format(round_index),
        pool_artifact,
    )
    _record_artifact(
        context,
        'acquisition_selection_round_{:02d}'.format(round_index),
        selection_artifact,
    )
    return {
        'pool_artifact': _artifact_result(pool_artifact),
        'selection_artifact': _artifact_result(selection_artifact),
    }


def _reveal_selected_annotations(
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
    _update_state(context, active_round=round_index)
    _record_artifact(
        context,
        'target_labeled_round_{:02d}'.format(round_index),
        artifact,
    )
    return {
        'image_count': manifest.image_count,
        'annotation_count': manifest.annotation_count,
        'annotation_artifact': _artifact_result(artifact),
    }


def _evaluate_final_teacher(
    stage: StageSpec,
    context: ExecutionContext,
    backend: MmdetExecutionBackend,
) -> Mapping[str, Any]:
    checkpoint = find_completed_checkpoint(context, 'detector_checkpoint')
    if checkpoint is None:
        raise FileNotFoundError(
            'evaluation requires a completed detector checkpoint'
        )
    metrics = dict(backend.evaluate(stage, context, checkpoint))
    metric_name = str(stage.payload['metric'])
    if metric_name not in metrics:
        raise ValueError(
            'evaluation backend did not return {}'.format(metric_name)
        )
    if not math.isfinite(float(metrics[metric_name])):
        raise ValueError('evaluation metric must be finite')
    artifact = context.artifact_store.write_json(
        'artifacts/evaluation.json',
        metrics,
        'evaluation_metrics',
        stage.stage_id,
    )
    _record_artifact(context, 'evaluation_metrics', artifact)
    return {
        'metrics': metrics,
        'metrics_artifact': _artifact_result(artifact),
    }


def create_executor_registry(
    context: ExecutionContext,
) -> StageExecutorRegistry:
    '''Build ADA-FNP stage bindings without method-name branching.'''

    if not isinstance(context, ExecutionContext):
        raise TypeError('context must be an ExecutionContext')
    backend = MmdetExecutionBackend()
    registry = StageExecutorRegistry()
    registry.register(
        'ada_fnp.prepare_vgg16_caffe_weights',
        _prepare_vgg16_caffe_weights,
    )
    registry.register(
        'ada_fnp.prepare_cityscapes_to_foggy',
        _prepare_cityscapes_to_foggy,
    )
    registry.register(
        'ada_fnp.train_detector',
        lambda stage, ctx: _train_detector(stage, ctx, backend),
    )
    registry.register(
        'ada_fnp.train_false_negative_predictor',
        lambda stage, ctx: _train_false_negative_predictor(
            stage,
            ctx,
            backend,
        ),
    )
    registry.register(
        'ada_fnp.score_unlabeled_pool',
        lambda stage, ctx: _score_unlabeled_pool(stage, ctx, backend),
    )
    registry.register('ada_fnp.select_samples', _select_samples)
    registry.register(
        'ada_fnp.reveal_selected_annotations',
        _reveal_selected_annotations,
    )
    registry.register(
        'ada_fnp.evaluate_final_teacher',
        lambda stage, ctx: _evaluate_final_teacher(stage, ctx, backend),
    )
    return registry
