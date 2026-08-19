'''Concrete serial-stage bindings for ADA-FNP.'''

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from methods.ada_fnp.acquisition.records import RawAdaFnpScore, normalize_scores
from methods.ada_fnp.phases import (
    RevealRequest,
    execute_reveal,
    resolve_detector_phase,
    validate_fnpm_round_phase,
)
from methods.ada_fnp.plan import resolve_total_budget
from methods.ada_fnp.training.fnpm_trainer import (
    build_fnpm_resume_payload,
    build_fnpm_round_optimization,
    restore_fnpm_resume_payload,
    run_fnpm_steps,
)
from methods.common.acquisition.artifacts import (
    JsonArtifact,
    JsonArtifactRecord,
    read_json_artifact,
    write_json_artifact,
)
from methods.common.acquisition.selection import AcquisitionScore, select_top_k
from methods.common.artifacts import sha256_file
from methods.common.assets import prepare_verified_asset
from methods.common.contracts import ArtifactRef, StageSpec
from methods.common.data.cityscapes import prepare_cityscapes_to_foggy
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from methods.common.engine.context import ExecutionContext
from methods.common.engine.runner import StageExecutorRegistry
from methods.common.mmdet.models.backbones.vgg16_caffe import (
    CHECKPOINT_PATH as VGG16_CAFFE_PATH,
    DOWNLOAD_URL as VGG16_CAFFE_URL,
    SHA256 as VGG16_CAFFE_SHA256,
)

from .backend import AdaFnpExecutionBackend, MmdetExecutionBackend
from .artifacts import completed_checkpoint
from .paths import (
    dataset_cache_directory,
    pool_state_path,
    target_labeled_manifest_path,
    target_unlabeled_manifest_path,
)


def _repository_path(
    context: ExecutionContext,
    value: str,
    *,
    allow_external_target: bool = False,
) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('execution path must be repository-relative')
    candidate = context.repository_root.joinpath(*path.parts)
    try:
        candidate.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError('execution path escapes the repository') from error
    if allow_external_target:
        return candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError('execution path escapes the repository') from error
    return resolved


def _prepare_asset(context: ExecutionContext) -> Path:
    return prepare_verified_asset(
        _repository_path(context, VGG16_CAFFE_PATH),
        url=VGG16_CAFFE_URL,
        expected_sha256=VGG16_CAFFE_SHA256,
        allow_download=not context.offline,
    )


def _prepare_dataset(context: ExecutionContext) -> Mapping[str, Any]:
    dataset = context.config['dataset']
    return prepare_cityscapes_to_foggy(
        _repository_path(
            context,
            dataset['source']['image_root'],
            allow_external_target=True,
        ),
        _repository_path(
            context,
            dataset['target']['image_root'],
            allow_external_target=True,
        ),
        _repository_path(
            context,
            dataset['source']['annotation_root'],
            allow_external_target=True,
        ),
        dataset_cache_directory(context),
        context.repository_root,
        expected_train_images=int(dataset['target']['expected_train_images']),
        expected_val_images=int(dataset['target']['expected_eval_images']),
    )
def _artifact_result(artifact: ArtifactRef) -> dict:
    return asdict(artifact)


def _update_state(context: ExecutionContext, **updates: Any) -> None:
    state = context.state_store.load()
    for name, value in updates.items():
        if not hasattr(state, name):
            raise AttributeError('unknown run-state field: {}'.format(name))
        setattr(state, name, value)
    context.state_store.save(state)


def _read_pool(context: ExecutionContext, round_index: int) -> PoolState:
    path = pool_state_path(context, round_index)
    if not path.is_file():
        raise FileNotFoundError('pool state is missing: {!s}'.format(path))
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    return PoolState.from_dict(value)


def _write_pool(
    context: ExecutionContext, stage: StageSpec, pool: PoolState, round_index: int
) -> ArtifactRef:
    relative = pool_state_path(context, round_index).relative_to(
        context.run_directory
    ).as_posix()
    return context.artifact_store.write_json(
        relative, pool.to_dict(), 'target_pool_state', stage.stage_id
    )


def _prepare_pretrained(
    stage: StageSpec, context: ExecutionContext
) -> Mapping[str, Any]:
    path = _prepare_asset(context).resolve()
    if not path.is_file():
        raise FileNotFoundError('pretrained asset preparer returned no file')
    try:
        relative_path = path.relative_to(context.repository_root).as_posix()
    except ValueError as error:
        raise ValueError('pretrained asset must stay inside the repository') from error
    digest = sha256_file(path)
    metadata = context.artifact_store.write_json(
        'artifacts/pretrained.json',
        {'path': relative_path, 'sha256': digest},
        'pretrained_checkpoint_metadata',
        stage.stage_id,
    )
    return {'path': relative_path, 'sha256': digest,
            'metadata_artifact': _artifact_result(metadata)}


def _prepare_datasets(
    stage: StageSpec, context: ExecutionContext
) -> Mapping[str, Any]:
    manifest = dict(_prepare_dataset(context))
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
        SampleIdentity.parse(image['sample_id']) for image in unlabeled['images']
    )
    pool = PoolState.initialize(
        samples, total_budget=resolve_total_budget(context.config)
    )
    pool_artifact = _write_pool(context, stage, pool, 0)
    _update_state(context, pool_artifact_id=pool_artifact.artifact_id)
    return {'dataset_manifest': manifest, 'pool_artifact': _artifact_result(pool_artifact)}


def _train_detector(
    stage: StageSpec, context: ExecutionContext, backend: AdaFnpExecutionBackend
) -> Mapping[str, Any]:
    pool = _read_pool(context, context.state_store.load().active_round)
    phase = resolve_detector_phase(
        int(stage.payload['start_iteration']),
        int(stage.payload['end_iteration']),
        len(pool.labeled),
    )
    checkpoint_path = (
        context.run_directory / 'checkpoints' /
        'detector_{:05d}.pth'.format(phase.end_iteration)
    )
    resume_from = completed_checkpoint(context, 'detector_checkpoint')
    written = backend.train_detector(
        stage, context, phase, checkpoint_path, resume_from
    )
    artifact = context.artifact_store.reference_file(
        written, 'detector_checkpoint', stage.stage_id
    )
    _update_state(
        context,
        global_detector_iteration=phase.end_iteration,
        detector_checkpoint_artifact_id=artifact.artifact_id,
    )
    pool_manifest = target_unlabeled_manifest_path(
        context, context.state_store.load().active_round
    )
    result = {
        'phase': phase.mode.value,
        'checkpoint_artifact': _artifact_result(artifact),
    }
    if pool_manifest.is_file():
        result['unlabeled_pool_artifact'] = _artifact_result(
            context.artifact_store.reference_file(
                pool_manifest, 'target_unlabeled_annotations', stage.stage_id
            )
        )
    return result


def _train_fnpm(
    stage: StageSpec, context: ExecutionContext, backend: AdaFnpExecutionBackend
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    detector_iteration = int(stage.payload['detector_iteration'])
    end_iteration = int(stage.payload['iterations'])
    validate_fnpm_round_phase(round_index, detector_iteration, 0)
    checkpoint_path = (
        context.run_directory / 'checkpoints' /
        'fnpm_round_{:02d}.pth'.format(round_index)
    )
    resume_path = checkpoint_path if context.resume and checkpoint_path.is_file() else None
    session = backend.create_fnpm_session(stage, context, resume_path)
    optimizer, scheduler = build_fnpm_round_optimization(
        session.fnpm, float(context.config['fnpm']['lr'])
    )
    start_iteration = 0
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        if set(checkpoint) != {'fnpm', 'rng_state'}:
            raise ValueError('FNPM checkpoint wrapper has an invalid schema')
        start_iteration = restore_fnpm_resume_payload(
            checkpoint['fnpm'],
            session.fnpm,
            optimizer,
            scheduler,
            expected_round_index=round_index,
        )
        restore_rng_state(checkpoint['rng_state'])
    if start_iteration > end_iteration:
        raise ValueError('FNPM checkpoint is ahead of the requested stage')
    losses = []
    iteration = start_iteration
    while iteration < end_iteration:
        chunk_end = min(iteration + 100, end_iteration)
        result = run_fnpm_steps(
            session.fnpm,
            session.teacher,
            optimizer,
            scheduler,
            session.source_batch_provider,
            session.teacher_batch_extractor,
            start_iteration=iteration,
            end_iteration=chunk_end,
            labeled_target_batch_provider=session.labeled_target_batch_provider,
        )
        losses.extend(result.losses)
        iteration = result.iteration
        save_checkpoint(
            checkpoint_path,
            {
                'fnpm': build_fnpm_resume_payload(
                    session.fnpm,
                    optimizer,
                    scheduler,
                    round_index=round_index,
                    iteration=iteration,
                ),
                'rng_state': capture_rng_state(),
            },
        )
    artifact = context.artifact_store.reference_file(
        checkpoint_path, 'fnpm_checkpoint', stage.stage_id
    )
    _update_state(context, fnpm_checkpoint_artifact_id=artifact.artifact_id)
    return {
        'iteration': iteration,
        'mean_loss': sum(losses) / len(losses) if losses else None,
        'checkpoint_artifact': _artifact_result(artifact),
    }


def _score_pool(
    stage: StageSpec, context: ExecutionContext, backend: AdaFnpExecutionBackend
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = _read_pool(context, round_index - 1)
    raw_records = tuple(
        backend.score_pool(stage, context, pool.unlabeled)
    )
    if any(not isinstance(record, RawAdaFnpScore) for record in raw_records):
        raise TypeError('score backend must return RawAdaFnpScore records')
    if {record.sample for record in raw_records} != set(pool.unlabeled):
        raise ValueError('score backend must cover the current unlabeled pool exactly')
    acquisition_config = context.config['acquisition']
    scores = normalize_scores(
        raw_records,
        constant_component_value=float(
            acquisition_config['constant_score_normalized_value']
        ),
        empty_detection_score=float(
            acquisition_config['empty_detection_final_score']
        ),
    )
    raw_by_sample = {record.sample: record for record in raw_records}
    artifact = JsonArtifact(
        artifact_type='acquisition_scores',
        producer_stage_id=stage.stage_id,
        metadata={'round': round_index},
        records=tuple(
            JsonArtifactRecord(
                score.sample,
                {
                    'raw': {
                        'false_negative': raw_by_sample[score.sample].false_negative,
                        'localization': raw_by_sample[score.sample].localization,
                        'entropy': raw_by_sample[score.sample].entropy,
                        'diversity': raw_by_sample[score.sample].diversity,
                    },
                    'normalized': dict(score.components),
                    'source_domain_probability': (
                        raw_by_sample[score.sample].source_domain_probability
                    ),
                    'detection_count': score.detection_count,
                    'final_score': score.final_score,
                },
            )
            for score in scores
        ),
    )
    reference = write_json_artifact(
        'artifacts/rounds/{:02d}/scores.json'.format(round_index),
        artifact,
        run_directory=context.run_directory,
    )
    pool_manifest = target_unlabeled_manifest_path(context, round_index - 1)
    result = {
        'score_artifact': _artifact_result(reference),
        'sample_count': len(scores),
    }
    if pool_manifest.is_file():
        result['unlabeled_pool_artifact'] = _artifact_result(
            context.artifact_store.reference_file(
                pool_manifest, 'target_unlabeled_annotations', stage.stage_id
            )
        )
    return result


def _select(stage: StageSpec, context: ExecutionContext) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = _read_pool(context, round_index - 1)
    score_path = (
        context.run_directory / 'artifacts' / 'rounds' /
        '{:02d}'.format(round_index) / 'scores.json'
    )
    artifact = read_json_artifact(score_path)
    if artifact.artifact_type != 'acquisition_scores':
        raise ValueError('selection input is not an acquisition-score artifact')
    if artifact.metadata.get('round') != round_index:
        raise ValueError('selection input belongs to a different round')
    if {record.sample for record in artifact.records} != set(pool.unlabeled):
        raise ValueError('selection scores do not cover the unlabeled pool exactly')
    raw_records = []
    for record in artifact.records:
        expected_fields = {
            'raw', 'normalized', 'source_domain_probability',
            'detection_count', 'final_score',
        }
        if set(record.fields) != expected_fields:
            raise ValueError('acquisition score record has an invalid schema')
        raw = record.fields['raw']
        if set(raw) != {
            'false_negative', 'localization', 'entropy', 'diversity'
        }:
            raise ValueError('raw acquisition components have an invalid schema')
        normalized = record.fields['normalized']
        if not isinstance(normalized, Mapping) or set(normalized) != set(raw):
            raise ValueError(
                'normalized acquisition components have an invalid schema'
            )
        detection_count = record.fields['detection_count']
        if isinstance(detection_count, bool) or not isinstance(
            detection_count, int
        ):
            raise TypeError('acquisition detection_count must be an integer')
        raw_records.append(RawAdaFnpScore(
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
        score.sample: score for score in normalize_scores(
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
            raise ValueError('stored normalized acquisition components are inconsistent')
        if not math.isclose(
            score.final_score, float(record.fields['final_score']),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError('stored acquisition product is inconsistent')
        scores.append(score)
    selected = select_top_k(tuple(scores), int(stage.payload['budget']))
    next_pool = pool.acquire(score.sample for score in selected)
    pool_artifact = _write_pool(context, stage, next_pool, round_index)
    selection = JsonArtifact(
        artifact_type='acquisition_selection',
        producer_stage_id=stage.stage_id,
        metadata={'round': round_index, 'budget': len(selected)},
        records=tuple(
            JsonArtifactRecord(score.sample, {'final_score': score.final_score})
            for score in selected
        ),
    )
    selection_artifact = write_json_artifact(
        'artifacts/rounds/{:02d}/selection.json'.format(round_index),
        selection,
        run_directory=context.run_directory,
    )
    _update_state(context, pool_artifact_id=pool_artifact.artifact_id)
    return {
        'pool_artifact': _artifact_result(pool_artifact),
        'selection_artifact': _artifact_result(selection_artifact),
    }


def _reveal(stage: StageSpec, context: ExecutionContext) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = _read_pool(context, round_index)
    oracle_path = dataset_cache_directory(context) / 'target_train_oracle.json'
    output_path = target_labeled_manifest_path(context, round_index)
    manifest = execute_reveal(RevealRequest(oracle_path, output_path, pool))
    artifact = context.artifact_store.reference_file(
        manifest.path, 'target_labeled_annotations', stage.stage_id
    )
    _update_state(context, active_round=round_index)
    return {
        'image_count': manifest.image_count,
        'annotation_count': manifest.annotation_count,
        'annotation_artifact': _artifact_result(artifact),
    }


def _evaluate(
    stage: StageSpec, context: ExecutionContext, backend: AdaFnpExecutionBackend
) -> Mapping[str, Any]:
    checkpoint = completed_checkpoint(context, 'detector_checkpoint')
    if checkpoint is None:
        raise FileNotFoundError('evaluation requires a completed detector checkpoint')
    metrics = dict(backend.evaluate(stage, context, checkpoint))
    metric_name = str(stage.payload['metric'])
    if metric_name not in metrics:
        raise ValueError('evaluation backend did not return {}'.format(metric_name))
    if not math.isfinite(float(metrics[metric_name])):
        raise ValueError('evaluation metric must be finite')
    artifact = context.artifact_store.write_json(
        'artifacts/evaluation.json', metrics, 'evaluation_metrics', stage.stage_id
    )
    return {'metrics': metrics, 'metrics_artifact': _artifact_result(artifact)}


def create_executor_registry(
    context: ExecutionContext,
) -> StageExecutorRegistry:
    '''Build all common and ADA-FNP stage bindings without method-name branches.'''

    if not isinstance(context, ExecutionContext):
        raise TypeError('context must be an ExecutionContext')
    backend = MmdetExecutionBackend()
    registry = StageExecutorRegistry()
    registry.register(
        'common.prepare_pretrained',
        _prepare_pretrained,
    )
    registry.register(
        'common.prepare_datasets',
        _prepare_datasets,
    )
    registry.register('common.select', _select)
    registry.register('common.reveal_annotations', _reveal)
    registry.register(
        'common.evaluate', lambda stage, ctx: _evaluate(stage, ctx, backend)
    )
    registry.register(
        'ada_fnp.train_detector',
        lambda stage, ctx: _train_detector(stage, ctx, backend),
    )
    registry.register(
        'ada_fnp.train_fnpm',
        lambda stage, ctx: _train_fnpm(stage, ctx, backend),
    )
    registry.register(
        'ada_fnp.score_pool',
        lambda stage, ctx: _score_pool(stage, ctx, backend),
    )
    return registry
