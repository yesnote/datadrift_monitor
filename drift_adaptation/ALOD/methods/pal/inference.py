"""PAL detection record parsing.

PAL LIUS needs detector confidence and a pre-NMS box count for each detection.
The exact detector hook is added later; this module defines the lightweight
schema consumed by the sampler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_detection_records(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [dict(record) for record in data]
    if isinstance(data, dict):
        for key in ('detections', 'annotations', 'results'):
            if isinstance(data.get(key), list):
                return [dict(record) for record in data[key]]
    raise ValueError('Unsupported PAL detection JSON schema: %s' % path)


def detection_confidence(det: Dict[str, Any]) -> float:
    return float(det.get('confidence', det.get('score', 0.0)))


def detection_pre_nms_count(det: Dict[str, Any]) -> float:
    for key in ('pre_nms_count', 'pre_nms_boxes', 'num_pre_nms', 'pre_nms_num'):
        if key in det:
            return float(det[key])
    return 0.0


def lius_feature(det: Dict[str, Any]) -> Tuple[float, float]:
    return detection_pre_nms_count(det), detection_confidence(det)


def filter_pool_detections(
    detections: Iterable[Dict[str, Any]],
    image_ids: Iterable[Any],
) -> List[Dict[str, Any]]:
    image_id_set = set(image_ids)
    return [dict(det) for det in detections if det.get('image_id') in image_id_set]
