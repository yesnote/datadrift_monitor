"""Detector specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .datasets import normalize_token


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    aliases: Tuple[str, ...]
    model_name: str
    train_config: str
    smoke_train_config: str
    uncertainty_infer_config: str
    diversity_infer_config: str
    pal_infer_config: str
    required_files: Tuple[str, ...] = ()

    def to_config(self, smoke: bool = False) -> Dict[str, object]:
        return {
            'train_config': self.smoke_train_config if smoke else self.train_config,
            'uncertainty_infer_config': self.uncertainty_infer_config,
            'diversity_infer_config': self.diversity_infer_config,
            'pal_infer_config': self.pal_infer_config,
            'model_name': self.model_name,
            'required_files': list(self.required_files),
        }


RETINANET = DetectorSpec(
    name='retinanet',
    aliases=('retinanet', 'retina-net', 'retina_net'),
    model_name='retinanet',
    train_config='configs/alod_mmdet/retinanet_voc_train_26e.py',
    smoke_train_config='configs/alod_mmdet/retinanet_voc_train_smoke_1e.py',
    uncertainty_infer_config='configs/alod_mmdet/retinanet_voc_uncertainty.py',
    diversity_infer_config='configs/alod_mmdet/retinanet_voc_diversity.py',
    pal_infer_config='configs/alod_mmdet/retinanet_voc_pal.py',
    required_files=('data/pretrain_models/resnet50-19c8e357.pth',),
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
