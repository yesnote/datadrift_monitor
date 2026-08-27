'''AADA-specific stages bound to shared active-detection execution.'''

import math
from typing import Any, Mapping

from methods.aada.acquisition.scoring import RawAadaScore
from methods.common.acquisition.score_artifacts import (
    AcquisitionArtifact,
    AcquisitionArtifactRecord,
    write_acquisition_artifact,
)
from methods.common.contracts import StageSpec
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
    read_pool_state,
    target_unlabeled_manifest_path,
)

from .mmdet_backend import MmdetExecutionBackend


def _score_unlabeled_pool(
    stage: StageSpec,
    context: ExecutionContext,
    backend: MmdetExecutionBackend,
) -> Mapping[str, Any]:
    round_index = int(stage.payload['round'])
    pool = read_pool_state(context, round_index - 1)
    raw_records = tuple(backend.score_pool(stage, context, pool.unlabeled))
    if any(not isinstance(record, RawAadaScore) for record in raw_records):
        raise TypeError('AADA score backend returned an invalid record')
    if {record.sample for record in raw_records} != set(pool.unlabeled):
        raise ValueError('AADA scores must cover the unlabeled pool exactly')
    artifact = AcquisitionArtifact(
        artifact_type='acquisition_scores',
        producer_stage_id=stage.stage_id,
        metadata={
            'round': round_index,
            'method': 'aada',
            'formula': 'entropy*((1-source_probability)/source_probability)',
        },
        records=tuple(
            AcquisitionArtifactRecord(
                record.sample,
                {
                    'raw': {
                        'entropy': record.entropy,
                        'diversity': record.diversity,
                    },
                    'source_domain_probability': (
                        record.source_domain_probability
                    ),
                    'detection_count': record.detection_count,
                    'final_score': record.final_score,
                },
            )
            for record in raw_records
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
        'sample_count': len(raw_records),
    }
    if pool_manifest.is_file():
        pool_artifact = context.artifact_store.reference_file(
            pool_manifest,
            'target_unlabeled_annotations',
            stage.stage_id,
        )
        result['unlabeled_pool_artifact'] = artifact_result(pool_artifact)
    return result


def _validate_aada_final_score(record: AcquisitionArtifactRecord) -> float:
    expected_fields = {
        'raw',
        'source_domain_probability',
        'detection_count',
        'final_score',
    }
    if set(record.fields) != expected_fields:
        raise ValueError('AADA score record has an invalid schema')
    raw = record.fields['raw']
    if not isinstance(raw, Mapping) or set(raw) != {'entropy', 'diversity'}:
        raise ValueError('AADA raw score components have an invalid schema')
    detection_count = record.fields['detection_count']
    if isinstance(detection_count, bool) or not isinstance(
        detection_count,
        int,
    ):
        raise TypeError('AADA detection_count must be an integer')
    reconstructed = RawAadaScore(
        sample=record.sample,
        entropy=float(raw['entropy']),
        diversity=float(raw['diversity']),
        source_domain_probability=float(
            record.fields['source_domain_probability']
        ),
        detection_count=detection_count,
    )
    stored = float(record.fields['final_score'])
    if not math.isclose(
        reconstructed.final_score,
        stored,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError('stored AADA acquisition score is inconsistent')
    return stored


def create_executor_registry(
    context: ExecutionContext,
) -> StageExecutorRegistry:
    if not isinstance(context, ExecutionContext):
        raise TypeError('context must be an ExecutionContext')
    backend = MmdetExecutionBackend()
    registry = StageExecutorRegistry()
    registry.register(
        'aada.prepare_vgg16_caffe_weights',
        prepare_vgg16_caffe_weights,
    )
    registry.register(
        'aada.prepare_cityscapes_to_foggy',
        prepare_cityscapes_to_foggy_stage,
    )
    registry.register(
        'aada.train_detector',
        lambda stage, ctx: train_detector_stage(stage, ctx, backend),
    )
    registry.register(
        'aada.score_unlabeled_pool',
        lambda stage, ctx: _score_unlabeled_pool(stage, ctx, backend),
    )
    registry.register(
        'aada.select_samples',
        lambda stage, ctx: select_samples_stage(
            stage,
            ctx,
            _validate_aada_final_score,
        ),
    )
    registry.register(
        'aada.reveal_selected_annotations',
        reveal_selected_annotations_stage,
    )
    registry.register(
        'aada.evaluate_detector_checkpoint',
        lambda stage, ctx: evaluate_detector_stage(stage, ctx, backend),
    )
    return registry
