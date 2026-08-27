'''AADA values projected onto shared MMDetection configuration plumbing.'''

from pathlib import Path
from typing import Any, MutableMapping, Optional

from methods.common.engine.context import ExecutionContext
from methods.common.mmdet.configuration import (
    build_segment_config,
    configure_dataloader,
    load_method_config,
)
from methods.common.protocols.ada_fnp_detection import DetectorTrainingPhase


def _config_path(context: ExecutionContext) -> Path:
    return context.repository_root / 'methods/aada/configs/cityscapes_to_foggy.py'


def apply_resolved_experiment_config(
    config: MutableMapping[str, Any],
    context: ExecutionContext,
) -> None:
    domain_adaptation = context.config['domain_adaptation']
    model = config['model']
    model['grl_scale'] = float(
        domain_adaptation['gradient_reversal_scale']
    )
    model['domain_loss_weight'] = float(domain_adaptation['loss_weight'])


def load_base_config(
    runtime,
    context: ExecutionContext,
    *,
    load_pretrained_backbone: bool = False,
) -> MutableMapping[str, Any]:
    return load_method_config(
        runtime,
        context,
        _config_path(context),
        apply_resolved_experiment_config,
        load_pretrained_backbone=load_pretrained_backbone,
    )


def build_detector_stage_config(
    runtime,
    context: ExecutionContext,
    phase: DetectorTrainingPhase,
    continuation_checkpoint: Optional[Path],
    producer_stage_id: str,
) -> MutableMapping[str, Any]:
    return build_segment_config(
        runtime,
        context,
        phase,
        continuation_checkpoint,
        producer_stage_id,
        base_config_loader=load_base_config,
    )


__all__ = [
    'build_detector_stage_config',
    'configure_dataloader',
    'load_base_config',
]
