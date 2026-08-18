'''Numeric coordinate-contract tests for PTVOCMetric.'''

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BBOX_OVERLAPS_PATH = (
    REPOSITORY_ROOT / 'mmdet' / 'evaluation' / 'functional'
    / 'bbox_overlaps.py'
)
METRIC_PATH = (
    REPOSITORY_ROOT / 'methods' / 'common' / 'mmdet' / 'metrics'
    / 'pt_voc_metric.py'
)


def _bbox_overlaps():
    namespace = {}
    exec(compile(
        BBOX_OVERLAPS_PATH.read_text(encoding='utf-8'),
        str(BBOX_OVERLAPS_PATH),
        'exec',
    ), namespace)
    return namespace['bbox_overlaps']


def test_half_open_and_legacy_iou_differ_at_ap50_boundary():
    bbox_overlaps = _bbox_overlaps()
    ground_truth = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32)
    prediction = np.array([[1.0, 0.0, 3.0, 2.0]], dtype=np.float32)

    half_open_iou = bbox_overlaps(
        prediction, ground_truth, use_legacy_coordinate=False)[0, 0]
    legacy_iou = bbox_overlaps(
        prediction, ground_truth, use_legacy_coordinate=True)[0, 0]

    assert np.isclose(half_open_iou, 1.0 / 3.0)
    assert np.isclose(legacy_iou, 0.5)
    assert half_open_iou < 0.5 <= legacy_iou


def test_pt_voc_metric_selects_half_open_ap_result(monkeypatch):
    bbox_overlaps = _bbox_overlaps()
    calls = []

    def fake_eval_map(
            preds,
            gts,
            *,
            iou_thr,
            tpfp_fn,
            use_legacy_coordinate,
            **kwargs):
        calls.append((preds, use_legacy_coordinate, tpfp_fn))
        tp, _ = tpfp_fn(
            preds[0][0],
            gts[0]['bboxes'],
            np.empty((0, 4), dtype=np.float32),
            iou_thr,
            None,
            use_legacy_coordinate,
        )
        return float(tp[0, 0]), []

    class FakeLogger:
        @staticmethod
        def get_current_instance():
            return FakeLogger()

        def info(self, message):
            return None

    class FakeVOCMetric:
        def __init__(
                self,
                iou_thrs=0.5,
                scale_ranges=None,
                metric='mAP',
                eval_mode='11points',
                **kwargs):
            self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) else iou_thrs
            self.scale_ranges = scale_ranges
            self.metric = metric
            self.eval_mode = eval_mode
            self.dataset_meta = {'classes': ('object', )}

    class FakeRegistry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    modules = {
        'mmengine': ModuleType('mmengine'),
        'mmengine.logging': ModuleType('mmengine.logging'),
        'mmdet': ModuleType('mmdet'),
        'mmdet.evaluation': ModuleType('mmdet.evaluation'),
        'mmdet.evaluation.functional': ModuleType(
            'mmdet.evaluation.functional'
        ),
        'mmdet.evaluation.metrics': ModuleType('mmdet.evaluation.metrics'),
        'mmdet.registry': ModuleType('mmdet.registry'),
    }
    modules['mmengine.logging'].MMLogger = FakeLogger
    modules['mmdet.evaluation.functional'].eval_map = fake_eval_map
    modules['mmdet.evaluation.functional'].bbox_overlaps = bbox_overlaps
    modules['mmdet.evaluation.metrics'].VOCMetric = FakeVOCMetric
    modules['mmdet.registry'].METRICS = FakeRegistry()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        'adaod_pt_voc_metric_contract_test', METRIC_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    metric = module.PTVOCMetric(iou_thrs=0.5, eval_mode='area')
    ground_truth = {
        'bboxes': np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32)
    }
    detections = [[np.array(
        [[0.04, 0.06, 2.04, 2.06, 0.1236]], dtype=np.float32
    )]]
    result = metric.compute_metrics([(ground_truth, detections[0])])

    quantized_predictions, legacy_coordinate, tpfp_fn = calls[0]
    assert legacy_coordinate is False
    assert tpfp_fn is module.tpfp_pt_voc
    assert np.allclose(
        quantized_predictions[0][0],
        [[0.0, 0.1, 2.0, 2.1, 0.124]],
    )
    assert result == {'mAP': 100.0, 'AP50': 100.0}

    exact_boundary_detection = np.array(
        [[0.0, 0.0, 1.0, 2.0, 0.9]], dtype=np.float32
    )
    tp, fp = module.tpfp_pt_voc(
        exact_boundary_detection,
        ground_truth['bboxes'],
        np.empty((0, 4), dtype=np.float32),
        iou_thr=0.5,
        use_legacy_coordinate=False,
    )
    assert tp.tolist() == [[0.0]]
    assert fp.tolist() == [[1.0]]
