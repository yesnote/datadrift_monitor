"""Detector specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .datasets import normalize_token


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    aliases: Tuple[str, ...]
    model_name: str
    train_config: str
    uncertainty_infer_config: str
    diversity_infer_config: str
    pal_infer_config: str
    pretrained: Dict[str, object] = field(default_factory=dict)

    def to_config(self) -> Dict[str, object]:
        return {
            'train_config': self.train_config,
            'uncertainty_infer_config': self.uncertainty_infer_config,
            'diversity_infer_config': self.diversity_infer_config,
            'pal_infer_config': self.pal_infer_config,
            'model_name': self.model_name,
            'pretrained': dict(self.pretrained),
        }


RETINANET = DetectorSpec(
    name='retinanet',
    aliases=('retinanet', 'retina-net', 'retina_net'),
    model_name='retinanet',
    train_config='configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py',
    uncertainty_infer_config='configs/alod_mmdet/retinanet_voc_infer_uncertainty.py',
    diversity_infer_config='configs/alod_mmdet/retinanet_voc_infer_features.py',
    pal_infer_config='configs/alod_mmdet/retinanet_voc_infer_pal_detections.py',
    pretrained=dict(
        type='resnet50',
        output_path='data/pretrain_models/resnet50-19c8e357.pth',
    ),
)


def list_detectors() -> List[DetectorSpec]:
    return [RETINANET]


def resolve_detector(name: str) -> Optional[DetectorSpec]:
    normalized = normalize_token(name)
    for detector in list_detectors():
        aliases = {normalize_token(value) for value in detector.aliases}
        if normalized == detector.name or normalized in aliases:
            return detector
    return None
