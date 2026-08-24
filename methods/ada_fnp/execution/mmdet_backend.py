'''MMEngine and MMDetection execution backend for ADA-FNP.'''

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional
from typing import Sequence, Tuple

import torch
from torch import nn

from methods.ada_fnp.acquisition.scoring import (
    RawAcquisitionScore,
    compute_domain_diversity_score,
    compute_foreground_entropy,
)
from methods.ada_fnp.models.false_negative_predictor import (
    FalseNegativePredictor,
)
from methods.ada_fnp.schedule import (
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

from .mmdet_checkpoints import (
    bind_exact_continuation_iteration,
    save_atomic_runner_checkpoint,
    unwrap_distributed_model,
    validate_detector_continuation_checkpoint,
)
from .mmdet_config import (
    build_detector_stage_config,
    build_single_dataset_dataloader,
    configure_dataloader,
    load_base_config,
)
from .run_files import (
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


class MissingMmdetDependencyError(EnvironmentError):
    '''Raised before execution when the pinned OpenMMLab stack is unavailable.'''


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


@dataclass(frozen=True)
class _MmdetRuntime:
    '''OpenMMLab operations used by the concrete execution backend.'''

    load_config: Callable[[Path], MutableMapping[str, Any]]
    import_custom_modules: Callable[[Mapping[str, Any]], None]
    build_runner: Callable[[Mapping[str, Any]], Any]
    progress_hook: Callable[..., Any]
    build_model: Callable[[Mapping[str, Any]], nn.Module]
    build_dataloader: Callable[[Mapping[str, Any], int], Iterable]
    load_model_checkpoint: Callable[[nn.Module, Path], None]


def _load_mmdet_runtime() -> _MmdetRuntime:
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner, load_checkpoint as mm_load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from methods.common.mmdet.progress import (
            AdaodConsoleQuietRunner,
            TqdmProgressHook,
        )
    except ImportError as error:
        raise MissingMmdetDependencyError(
            'ADA-FNP execution requires the pinned MMCV/MMEngine/MMDetection '
            'environment from requirements/runtime.txt'
        ) from error

    # Runner initializes this scope, while predictor training and acquisition
    # also build components directly from the MMDetection registries.
    init_default_scope('mmdet')

    def import_custom_modules(config: Mapping[str, Any]) -> None:
        custom_imports = config.get('custom_imports')
        if custom_imports:
            import_modules_from_strings(**custom_imports)

    return _MmdetRuntime(
        load_config=lambda path: Config.fromfile(
            str(path),
            import_custom_modules=False,
        ),
        import_custom_modules=import_custom_modules,
        build_runner=AdaodConsoleQuietRunner.from_cfg,
        progress_hook=TqdmProgressHook,
        build_model=MODELS.build,
        build_dataloader=lambda config, seed: Runner.build_dataloader(
            config,
            seed=seed,
        ),
        load_model_checkpoint=lambda model, path: mm_load_checkpoint(
            model,
            str(path),
            map_location='cpu',
            strict=True,
        ),
    )


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

    def _runtime(self) -> _MmdetRuntime:
        runtime = _load_mmdet_runtime()
        if not torch.cuda.is_available():
            raise MissingMmdetDependencyError(
                'ADA-FNP model stages require a CUDA-enabled PyTorch runtime'
            )
        return runtime

    def _inference_model(
        self,
        runtime: _MmdetRuntime,
        context: ExecutionContext,
        checkpoint_path: Path,
    ) -> Tuple[nn.Module, MutableMapping[str, Any]]:
        config = load_base_config(runtime, context)
        model = runtime.build_model(config['model'])
        runtime.load_model_checkpoint(model, checkpoint_path)
        model = model.cuda()
        model.eval()
        return model, config

    def train_detector(
        self,
        stage: StageSpec,
        context: ExecutionContext,
        phase: DetectorTrainingPhase,
        checkpoint_path: Path,
        continuation_checkpoint: Optional[Path],
    ) -> Path:
        runtime = self._runtime()
        config = build_detector_stage_config(
            runtime,
            context,
            phase,
            continuation_checkpoint,
            stage.stage_id,
        )
        runner = runtime.build_runner(config)
        required_keys = (
            _INITIAL_TRAIN_LOG_KEYS
            if phase.mode is DetectorTrainingMode.INITIALIZATION
            else _ADAPTATION_TRAIN_LOG_KEYS
        )
        runner.register_hook(
            runtime.progress_hook(
                context.progress,
                task_total=phase.end_iteration - phase.start_iteration,
                task_unit='iter',
                required_keys=required_keys,
            ),
            priority='LOWEST',
        )
        if continuation_checkpoint is not None:
            validate_detector_continuation_checkpoint(
                continuation_checkpoint,
                runner.model,
                (phase.start_iteration, phase.end_iteration),
                context=context,
            )
            bind_exact_continuation_iteration(
                runner,
                continuation_checkpoint,
                phase.start_iteration,
            )
        runner.train()
        if int(runner.iter) != phase.end_iteration:
            raise RuntimeError(
                'MMEngine stopped at iteration {}, expected {}'.format(
                    runner.iter,
                    phase.end_iteration,
                )
            )
        model = unwrap_distributed_model(runner.model)
        if phase.mode is DetectorTrainingMode.INITIALIZATION:
            model.teacher.load_state_dict(
                model.student.state_dict(),
                strict=True,
            )
        return save_atomic_runner_checkpoint(
            runner,
            checkpoint_path,
            stage,
            phase.end_iteration,
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
            1,
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
        del stage
        runtime = self._runtime()
        config = load_base_config(runtime, context)
        configure_dataloader(config['test_dataloader'], context)
        config['load_from'] = str(checkpoint_path)
        config['resume'] = False
        config['work_dir'] = str(
            context.run_directory / 'mmengine/evaluation'
        )
        runner = runtime.build_runner(config)
        runner.register_hook(
            runtime.progress_hook(
                context.progress,
                task_unit='batch',
            ),
            priority='LOWEST',
        )
        metrics = dict(runner.test())
        ap50_keys = [
            key for key in metrics if key.split('/')[-1] == 'AP50'
        ]
        if len(ap50_keys) != 1:
            raise ValueError(
                'Detectron2 VOC evaluator must return exactly one AP50 metric'
            )
        ap50 = float(metrics[ap50_keys[0]])
        if not math.isfinite(ap50) or not 0.0 <= ap50 <= 100.0:
            raise ValueError(
                'Detectron2 VOC AP50 must be a finite percentage in [0, 100]'
            )
        return {'AP50': ap50}
