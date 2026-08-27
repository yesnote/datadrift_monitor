'''MMDetection backend for AADA training, acquisition, and evaluation.'''

from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from methods.aada.acquisition.scoring import RawAadaScore
from methods.common.acquisition.domain_uncertainty import (
    compute_domain_diversity_score,
    compute_foreground_entropy,
)
from methods.common.contracts import StageSpec
from methods.common.data.image_identity import SampleIdentity
from methods.common.engine.context import ExecutionContext
from methods.common.execution.mmdet_checkpoints import unwrap_distributed_model
from methods.common.execution.run_files import (
    find_completed_checkpoint,
    map_pool_samples_by_image_id,
    materialize_unlabeled_pool_manifest,
)
from methods.common.mmdet.configuration import (
    build_single_dataset_dataloader,
    configure_dataloader,
)
from methods.common.mmdet.runtime import (
    build_inference_model,
    evaluate_detector_checkpoint,
    require_cuda_runtime,
    train_detector_segment,
)
from methods.common.protocols.ada_fnp_detection import (
    DetectorTrainingMode,
    DetectorTrainingPhase,
)

from .mmdet_config import build_detector_stage_config, load_base_config


_SOURCE_TRAIN_LOG_KEYS = (
    'source.loss_rpn_cls',
    'source.loss_rpn_bbox',
    'source.loss_cls',
    'source.acc',
    'source.loss_bbox',
)
_INITIAL_TRAIN_LOG_KEYS = (*_SOURCE_TRAIN_LOG_KEYS, 'domain.loss_adv')
_ADAPTATION_TRAIN_LOG_KEYS = (
    *_SOURCE_TRAIN_LOG_KEYS,
    'domain.loss_adv',
    'target_labeled.loss_rpn_cls',
    'target_labeled.loss_rpn_bbox',
    'target_labeled.loss_cls',
    'target_labeled.acc',
    'target_labeled.loss_bbox',
)


class MmdetExecutionBackend:
    '''Execute AADA without teacher, FN predictor, or MC Dropout.'''

    def train_detector(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        phase: DetectorTrainingPhase,
        checkpoint_path: Path,
        continuation_checkpoint: Optional[Path],
    ) -> Path:
        required_keys = {
            DetectorTrainingMode.INITIALIZATION: _INITIAL_TRAIN_LOG_KEYS,
            DetectorTrainingMode.UNLABELED_ADAPTATION: (
                _INITIAL_TRAIN_LOG_KEYS
            ),
            DetectorTrainingMode.ADAPTATION: _ADAPTATION_TRAIN_LOG_KEYS,
        }
        return train_detector_segment(
            stage,
            context,
            phase,
            checkpoint_path,
            continuation_checkpoint,
            config_builder=build_detector_stage_config,
            required_log_keys=required_keys,
        )

    def score_pool(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        samples: Sequence[SampleIdentity],
    ) -> Sequence[RawAadaScore]:
        context.progress.start_task(len(samples), 'image')
        runtime = require_cuda_runtime()
        detector_checkpoint = find_completed_checkpoint(
            context,
            'detector_checkpoint',
        )
        if detector_checkpoint is None:
            raise FileNotFoundError(
                'AADA pool scoring requires a detector checkpoint'
            )
        model, config = build_inference_model(
            runtime,
            context,
            detector_checkpoint,
            load_base_config,
        )
        model = unwrap_distributed_model(model)
        acquisition_dataset = dict(config['target_acquisition_dataset'])
        pool_manifest = materialize_unlabeled_pool_manifest(
            context,
            samples,
            stage.stage_id,
        )
        acquisition_dataset['ann_file'] = str(pool_manifest)
        samples_by_image_id = map_pool_samples_by_image_id(
            pool_manifest,
            samples,
        )
        dataloader_config = build_single_dataset_dataloader(
            acquisition_dataset,
            int(context.config['inference']['acquisition_batch_size']),
            shuffle=False,
            drop_last=False,
        )
        configure_dataloader(dataloader_config, context)
        dataloader = runtime.build_dataloader(
            dataloader_config,
            int(context.config['seed']),
        )
        records = []
        seen_samples = set()
        epsilon = float(
            context.config['acquisition']['domain_probability_epsilon']
        )
        with torch.no_grad():
            for batch in dataloader:
                processed = model.detector.data_preprocessor(
                    batch,
                    training=False,
                )
                inputs = processed['inputs']
                data_samples = processed['data_samples']
                features = model.extract_domain_feature(inputs)
                predictions = model.predict_with_class_probabilities(
                    inputs,
                    data_samples,
                )
                source_probabilities = (
                    model.domain_discriminator.source_probability(features)
                )
                source_probabilities = source_probabilities.reshape(
                    len(data_samples),
                    -1,
                ).mean(dim=1)
                entropy_scores = torch.stack([
                    compute_foreground_entropy(
                        prediction.class_probabilities
                    )
                    for prediction in predictions
                ])
                diversity_scores = compute_domain_diversity_score(
                    source_probabilities,
                    epsilon=epsilon,
                )
                entropy_values = entropy_scores.detach().cpu().tolist()
                diversity_values = diversity_scores.detach().cpu().tolist()
                probability_values = (
                    source_probabilities.detach().cpu().tolist()
                )
                for index, (prediction, data_sample) in enumerate(
                    zip(predictions, data_samples)
                ):
                    image_id = int(data_sample.metainfo['img_id'])
                    if image_id not in samples_by_image_id:
                        raise ValueError(
                            'AADA dataloader returned an unknown image ID'
                        )
                    sample = samples_by_image_id[image_id]
                    if sample in seen_samples:
                        raise ValueError(
                            'AADA dataloader returned a sample twice'
                        )
                    records.append(RawAadaScore(
                        sample=sample,
                        entropy=float(entropy_values[index]),
                        diversity=float(diversity_values[index]),
                        source_domain_probability=float(
                            probability_values[index]
                        ),
                        detection_count=len(prediction.bboxes),
                    ))
                    seen_samples.add(sample)
                    context.progress.advance(1)
        if seen_samples != set(samples):
            raise ValueError('AADA acquisition does not cover the pool')
        return tuple(records)

    def evaluate(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        checkpoint_path: Path,
    ) -> Mapping[str, float]:
        return evaluate_detector_checkpoint(
            stage,
            context,
            checkpoint_path,
            config_loader=load_base_config,
            configure_test_dataloader=configure_dataloader,
        )
