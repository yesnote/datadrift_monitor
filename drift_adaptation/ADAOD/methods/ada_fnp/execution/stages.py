'''ADA-FNP-specific stages bound to shared active-detection execution.'''

from __future__ import annotations

import math
from typing import Any, Mapping

from methods.ada_fnp.acquisition.scoring import (
    RawAcquisitionScore,
    normalize_acquisition_scores,
)
from methods.ada_fnp.schedule import validate_false_negative_training_round
from methods.ada_fnp.training.false_negative_training import (
    build_false_negative_checkpoint_payload,
    build_false_negative_round_optimization,
    run_false_negative_training_steps,
)
from methods.common.acquisition.score_artifacts import (
    AcquisitionArtifact,
    AcquisitionArtifactRecord,
    write_acquisition_artifact,
)
from methods.common.acquisition.selection import AcquisitionScore
from methods.common.contracts import StageSpec
from methods.common.engine.checkpoint import save_checkpoint
from methods.common.engine.context import ExecutionContext
from methods.common.engine.runner import StageExecutorRegistry
from methods.common.execution.active_detection_stages import (
    artifact_result,
    evaluate_detector_stage,
    prepare_cityscapes_to_foggy_stage,
    prepare_vgg16_caffe_weights,
    record_artifact,
    reveal_selected_annotations_stage,
    select_samples_stage,
    train_detector_stage,
)
from methods.common.execution.run_files import (
    find_completed_checkpoint,
    read_pool_state,
    target_unlabeled_manifest_path,
)
from methods.common.protocols.ada_fnp_detection import ACQUISITION_MILESTONES

from .mmdet_backend import MmdetExecutionBackend


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
    session = backend.create_false_negative_training_session(stage, context)
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
            context.progress.set_completed(completed_iteration, loss=loss)
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
    record_artifact(
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
        'checkpoint_artifact': artifact_result(artifact),
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
        raise TypeError('score backend must return RawAcquisitionScore records')
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
    pseudo_label_config = context.config['pseudo_label']
    artifact = AcquisitionArtifact(
        artifact_type='acquisition_scores',
        producer_stage_id=stage.stage_id,
        metadata={
            'round': round_index,
            'method': 'ada-fnp',
            'mc_passes': int(acquisition_config['mc_passes']),
            'bbox_variance_space': 'roi_bbox_delta',
            'localization_variance_threshold': float(
                pseudo_label_config['localization_variance_threshold']
            ),
            'confidence_threshold': float(
                pseudo_label_config['confidence_threshold']
            ),
        },
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
                        raw_by_sample[score.sample].source_domain_probability
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
    record_artifact(
        context,
        'acquisition_scores_round_{:02d}'.format(round_index),
        reference,
    )
    pool_manifest = target_unlabeled_manifest_path(context, round_index - 1)
    result = {
        'score_artifact': artifact_result(reference),
        'sample_count': len(scores),
    }
    if pool_manifest.is_file():
        pool_artifact = context.artifact_store.reference_file(
            pool_manifest,
            'target_unlabeled_annotations',
            stage.stage_id,
        )
        result['unlabeled_pool_artifact'] = artifact_result(pool_artifact)
    return result


def _validate_ada_fnp_final_score(record: AcquisitionArtifactRecord) -> float:
    expected_fields = {
        'raw',
        'normalized',
        'source_domain_probability',
        'detection_count',
        'final_score',
    }
    if set(record.fields) != expected_fields:
        raise ValueError('ADA-FNP score record has an invalid schema')
    raw = record.fields['raw']
    expected_components = {
        'false_negative',
        'localization',
        'entropy',
        'diversity',
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_components:
        raise ValueError('ADA-FNP raw components have an invalid schema')
    normalized = record.fields['normalized']
    if not isinstance(normalized, Mapping) or set(normalized) != expected_components:
        raise ValueError('ADA-FNP normalized components have an invalid schema')
    detection_count = record.fields['detection_count']
    if isinstance(detection_count, bool) or not isinstance(
        detection_count,
        int,
    ):
        raise TypeError('ADA-FNP detection_count must be an integer')
    score = AcquisitionScore(
        record.sample,
        normalized,
        detection_count,
        empty_detection_score=0.0,
    )
    stored = float(record.fields['final_score'])
    if not math.isclose(
        score.final_score,
        stored,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError('stored ADA-FNP acquisition product is inconsistent')
    return stored


def create_executor_registry(
    context: ExecutionContext,
) -> StageExecutorRegistry:
    if not isinstance(context, ExecutionContext):
        raise TypeError('context must be an ExecutionContext')
    backend = MmdetExecutionBackend()
    registry = StageExecutorRegistry()
    registry.register(
        'ada_fnp.prepare_vgg16_caffe_weights',
        prepare_vgg16_caffe_weights,
    )
    registry.register(
        'ada_fnp.prepare_cityscapes_to_foggy',
        prepare_cityscapes_to_foggy_stage,
    )
    registry.register(
        'ada_fnp.train_detector',
        lambda stage, ctx: train_detector_stage(stage, ctx, backend),
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
    registry.register(
        'ada_fnp.select_samples',
        lambda stage, ctx: select_samples_stage(
            stage,
            ctx,
            _validate_ada_fnp_final_score,
        ),
    )
    registry.register(
        'ada_fnp.reveal_selected_annotations',
        reveal_selected_annotations_stage,
    )
    registry.register(
        'ada_fnp.evaluate_teacher_checkpoint',
        lambda stage, ctx: evaluate_detector_stage(stage, ctx, backend),
    )
    return registry
