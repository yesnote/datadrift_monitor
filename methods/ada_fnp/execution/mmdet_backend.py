'''MMEngine and MMDetection execution backend for ADA-FNP.'''

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from typing import Sequence, Tuple

import torch
from torch import nn

from methods.ada_fnp.acquisition.scoring import RawAcquisitionScore
from methods.common.acquisition.domain_uncertainty import (
    compute_domain_diversity_score,
    compute_foreground_entropy,
)
from methods.ada_fnp.models.false_negative_predictor import (
    FalseNegativePredictor,
)
from methods.common.protocols.ada_fnp_detection import (
    DetectorTrainingMode,
    DetectorTrainingPhase,
)
from methods.ada_fnp.training.false_negative_matching import (
    count_false_negatives,
)
from methods.ada_fnp.training.false_negative_training import (
    restore_false_negative_checkpoint_payload,
)
from methods.common.contracts import StageSpec
from methods.common.data.image_identity import SampleIdentity
from methods.common.engine.checkpoint import load_checkpoint
from methods.common.engine.context import ExecutionContext

from methods.common.execution.mmdet_checkpoints import (
    unwrap_distributed_model,
)
from methods.common.mmdet.runtime import (
    MmdetRuntime,
    build_inference_model,
    evaluate_detector_checkpoint,
    require_cuda_runtime,
    train_detector_segment,
)
from .mmdet_config import (
    build_detector_stage_config,
    build_single_dataset_dataloader,
    configure_dataloader,
    load_base_config,
)
from methods.common.execution.run_files import (
    find_completed_checkpoint,
    map_pool_samples_by_image_id,
    materialize_unlabeled_pool_manifest,
    target_labeled_manifest_path,
)


_DETECTOR_CHECKPOINT_TYPE = 'detector_checkpoint'
_FALSE_NEGATIVE_CHECKPOINT_TYPE = 'false_negative_predictor_checkpoint'

_SOURCE_TRAIN_LOG_KEYS = (
    'source.loss_rpn_cls',
    'source.loss_rpn_bbox',
    'source.loss_cls',
    'source.acc',
    'source.loss_bbox',
)
_INITIAL_TRAIN_LOG_KEYS = (
    *_SOURCE_TRAIN_LOG_KEYS,
    'domain.loss_adv',
)
_UNLABELED_ADAPTATION_TRAIN_LOG_KEYS = (
    *_SOURCE_TRAIN_LOG_KEYS,
    'domain.loss_adv',
    'target_unlabeled_strong.loss_rpn_cls',
    'target_unlabeled_strong.loss_cls',
)
_ADAPTATION_TRAIN_LOG_KEYS = (
    *_SOURCE_TRAIN_LOG_KEYS,
    'domain.loss_adv',
    'target_labeled.loss_rpn_cls',
    'target_labeled.loss_rpn_bbox',
    'target_labeled.loss_cls',
    'target_labeled.acc',
    'target_labeled.loss_bbox',
    'target_unlabeled_strong.loss_rpn_cls',
    'target_unlabeled_strong.loss_cls',
)


@dataclass(frozen=True)
class FalseNegativeTrainingSession:
    '''Runtime objects consumed by false-negative predictor training.'''

    predictor: nn.Module
    teacher: nn.Module
    source_batch_provider: Callable[[int], Any]
    teacher_batch_extractor: Callable[
        [nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]
    ]
    labeled_target_batch_provider: Optional[Callable[[int], Any]] = None


class _CyclingBatchProvider:
    def __init__(self, dataloader: Iterable, branch: str) -> None:
        self._dataloader = dataloader
        self._branch = branch
        self._iterator = iter(dataloader)

    def __call__(self, iteration: int):
        del iteration
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)
        return self._branch, batch


def _extract_teacher_supervision(
    teacher: nn.Module,
    value: Any,
    *,
    iou_threshold: float,
    max_detections: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    branch, batch = value
    branch_batch = {
        'inputs': batch['inputs'][branch],
        'data_samples': batch['data_samples'][branch],
    }
    processed = teacher.detector.data_preprocessor(
        branch_batch,
        training=False,
    )
    inputs = processed['inputs']
    data_samples = processed['data_samples']
    features = teacher.extract_domain_feature(inputs)
    predictions = teacher.predict(inputs, data_samples, rescale=False)
    counts = []
    for prediction, data_sample in zip(predictions, data_samples):
        instances = getattr(prediction, 'pred_instances', prediction)
        ground_truth = data_sample.gt_instances
        counts.append(
            count_false_negatives(
                instances.bboxes,
                instances.scores,
                instances.labels,
                ground_truth.bboxes,
                ground_truth.labels,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
        )
    return features, features.new_tensor(counts)


class MmdetExecutionBackend:
    '''Run detector, predictor, acquisition, and evaluation operations.'''

    def _runtime(self) -> MmdetRuntime:
        return require_cuda_runtime()

    def _inference_model(
        self,
        runtime: MmdetRuntime,
        context: ExecutionContext,
        checkpoint_path: Path,
    ) -> Tuple[nn.Module, Mapping[str, Any]]:
        return build_inference_model(
            runtime,
            context,
            checkpoint_path,
            load_base_config,
        )

    def train_detector(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        phase: DetectorTrainingPhase,
        checkpoint_path: Path,
        continuation_checkpoint: Optional[Path],
    ) -> Path:
        required_keys_by_mode = {
            DetectorTrainingMode.INITIALIZATION: _INITIAL_TRAIN_LOG_KEYS,
            DetectorTrainingMode.UNLABELED_ADAPTATION: (
                _UNLABELED_ADAPTATION_TRAIN_LOG_KEYS
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
            required_log_keys=required_keys_by_mode,
            initialize_completed_model=lambda model: (
                model.teacher.load_state_dict(
                    model.student.state_dict(),
                    strict=True,
                )
            ),
        )

    def create_false_negative_training_session(
        self,
        stage: StageSpec,
        context: ExecutionContext,
    ) -> FalseNegativeTrainingSession:
        runtime = self._runtime()
        detector_checkpoint = find_completed_checkpoint(
            context,
            _DETECTOR_CHECKPOINT_TYPE,
        )
        if detector_checkpoint is None:
            raise FileNotFoundError(
                'false-negative training requires a detector checkpoint'
            )
        model, config = self._inference_model(
            runtime,
            context,
            detector_checkpoint,
        )
        model = unwrap_distributed_model(model)

        predictor = FalseNegativePredictor(in_channels=512)
        previous_checkpoint = find_completed_checkpoint(
            context,
            _FALSE_NEGATIVE_CHECKPOINT_TYPE,
        )
        round_index = int(stage.payload['round'])
        if previous_checkpoint is not None:
            payload = load_checkpoint(previous_checkpoint)
            if set(payload) != {'false_negative_predictor'}:
                raise ValueError(
                    'false-negative predictor checkpoint wrapper is invalid'
                )
            restore_false_negative_checkpoint_payload(
                payload['false_negative_predictor'],
                predictor,
                expected_round_index=round_index - 1,
            )
        predictor = predictor.cuda()

        initial_datasets = config['stage_overrides']['initial'][
            'train_dataloader'
        ]['dataset']['datasets']
        source_loader_config = build_single_dataset_dataloader(
            initial_datasets[0],
            int(context.config['training']['source_batch_size']),
        )
        configure_dataloader(source_loader_config, context)
        source_loader = runtime.build_dataloader(
            source_loader_config,
            int(context.config['seed']),
        )

        labeled_target_provider = None
        active_round = context.state_store.load().active_round
        if active_round > 0:
            adaptation_datasets = config['stage_overrides']['adaptation'][
                'train_dataloader'
            ]['dataset']['datasets']
            labeled_loader_config = build_single_dataset_dataloader(
                adaptation_datasets[1],
                int(
                    context.config['training'][
                        'target_labeled_batch_size'
                    ]
                ),
            )
            labeled_manifest = target_labeled_manifest_path(
                context,
                active_round,
            )
            if not labeled_manifest.is_file():
                raise FileNotFoundError(
                    'selected-target manifest is missing: {!s}'.format(
                        labeled_manifest
                    )
                )
            configure_dataloader(
                labeled_loader_config,
                context,
                labeled_manifest=labeled_manifest,
            )
            labeled_target_provider = _CyclingBatchProvider(
                runtime.build_dataloader(
                    labeled_loader_config,
                    int(context.config['seed']),
                ),
                'target_labeled',
            )

        predictor_config = context.config['false_negative_predictor']
        return FalseNegativeTrainingSession(
            predictor=predictor,
            teacher=model.teacher,
            source_batch_provider=_CyclingBatchProvider(
                source_loader,
                'source',
            ),
            teacher_batch_extractor=partial(
                _extract_teacher_supervision,
                iou_threshold=float(
                    predictor_config['matcher_iou_threshold']
                ),
                max_detections=int(predictor_config['max_detections']),
            ),
            labeled_target_batch_provider=labeled_target_provider,
        )

    def score_pool(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        samples: Sequence[SampleIdentity],
    ) -> Sequence[RawAcquisitionScore]:
        context.progress.start_task(len(samples), 'image')
        runtime = self._runtime()
        detector_checkpoint = find_completed_checkpoint(
            context,
            _DETECTOR_CHECKPOINT_TYPE,
        )
        predictor_checkpoint = find_completed_checkpoint(
            context,
            _FALSE_NEGATIVE_CHECKPOINT_TYPE,
        )
        if detector_checkpoint is None or predictor_checkpoint is None:
            raise FileNotFoundError(
                'pool scoring requires detector and false-negative predictor '
                'checkpoints'
            )
        model, config = self._inference_model(
            runtime,
            context,
            detector_checkpoint,
        )
        model = unwrap_distributed_model(model)
        predictor = FalseNegativePredictor(in_channels=512)
        payload = load_checkpoint(predictor_checkpoint)
        if set(payload) != {'false_negative_predictor'}:
            raise ValueError(
                'false-negative predictor checkpoint wrapper is invalid'
            )
        restore_false_negative_checkpoint_payload(
            payload['false_negative_predictor'],
            predictor,
            expected_round_index=int(stage.payload['round']),
        )
        predictor = predictor.cuda()
        predictor.eval()

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
            context.config['inference']['acquisition_batch_size'],
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
        teacher = model.teacher
        with torch.no_grad():
            for batch in dataloader:
                processed = teacher.detector.data_preprocessor(
                    batch,
                    training=False,
                )
                inputs = processed['inputs']
                data_samples = processed['data_samples']
                features = teacher.extract_domain_feature(inputs)
                predictions = model.predict_teacher_fixed_proposals(
                    inputs,
                    data_samples,
                    passes=int(context.config['acquisition']['mc_passes']),
                )
                false_negative_scores = predictor(features)
                source_probabilities = (
                    teacher.domain_discriminator.source_probability(features)
                )
                if len(predictions) != len(data_samples):
                    raise ValueError(
                        'acquisition predictions and data samples differ'
                    )
                for index, (prediction, data_sample) in enumerate(
                    zip(predictions, data_samples)
                ):
                    image_id = int(data_sample.metainfo['img_id'])
                    if image_id not in samples_by_image_id:
                        raise ValueError(
                            'acquisition dataloader returned an unknown image ID'
                        )
                    sample = samples_by_image_id[image_id]
                    if sample in seen_samples:
                        raise ValueError(
                            'acquisition dataloader returned a sample twice'
                        )
                    detection_count = len(prediction.bboxes)
                    localization_score = (
                        prediction.box_variances.mean()
                        if detection_count
                        else features.new_zeros(())
                    )
                    entropy_score = compute_foreground_entropy(
                        prediction.class_probabilities
                    )
                    source_probability = source_probabilities[index].mean()
                    diversity_score = compute_domain_diversity_score(
                        source_probability,
                        epsilon=float(
                            context.config['acquisition'][
                                'domain_probability_epsilon'
                            ]
                        ),
                    )
                    records.append(
                        RawAcquisitionScore(
                            sample=sample,
                            false_negative=float(
                                false_negative_scores[index].detach().cpu()
                            ),
                            localization=float(
                                localization_score.detach().cpu()
                            ),
                            entropy=float(entropy_score.detach().cpu()),
                            diversity=float(diversity_score.detach().cpu()),
                            source_domain_probability=float(
                                source_probability.detach().cpu()
                            ),
                            detection_count=detection_count,
                        )
                    )
                    seen_samples.add(sample)
                    context.progress.advance(1)
        if seen_samples != set(samples):
            raise ValueError('acquisition dataloader does not cover the pool')
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
