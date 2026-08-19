'''Detectron2-compatible AP evaluation for zero-based, half-open boxes.'''

import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
from mmengine.logging import MMLogger

from mmdet.evaluation.functional import bbox_overlaps, eval_map
from mmdet.evaluation.metrics import VOCMetric


def compute_detectron2_voc_true_false_positives(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=None,
        iou_thr=0.5,
        area_ranges=None,
        use_legacy_coordinate=False,
        **kwargs):
    '''Classify detections with Detectron2's strict VOC IoU threshold.'''

    extra_length = 1.0 if use_legacy_coordinate else 0.0
    if gt_bboxes_ignore is None:
        gt_bboxes_ignore = np.empty((0, 4), dtype=gt_bboxes.dtype)
    gt_ignore_inds = np.concatenate((
        np.zeros(gt_bboxes.shape[0], dtype=bool),
        np.ones(gt_bboxes_ignore.shape[0], dtype=bool),
    ))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    tp = np.zeros((len(area_ranges), num_dets), dtype=np.float32)
    fp = np.zeros((len(area_ranges), num_dets), dtype=np.float32)

    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (
                det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length
            ) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length
            )
            for index, (min_area, max_area) in enumerate(area_ranges):
                inside = (det_areas >= min_area) & (det_areas < max_area)
                fp[index, inside] = 1
        return tp, fp

    ious = bbox_overlaps(
        det_bboxes,
        gt_bboxes,
        use_legacy_coordinate=use_legacy_coordinate,
    )
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_indices = np.argsort(-det_bboxes[:, -1])
    for scale_index, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (
                gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length
            ) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
            )
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)

        for det_index in sort_indices:
            # Detectron2's Pascal evaluator follows the official VOC code:
            # an overlap exactly equal to the threshold is not a match.
            if ious_max[det_index] > iou_thr:
                matched_gt = ious_argmax[det_index]
                if not (
                    gt_ignore_inds[matched_gt]
                    or gt_area_ignore[matched_gt]
                ):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[scale_index, det_index] = 1
                    else:
                        fp[scale_index, det_index] = 1
            elif min_area is None:
                fp[scale_index, det_index] = 1
            else:
                bbox = det_bboxes[det_index, :4]
                area = (
                    bbox[2] - bbox[0] + extra_length
                ) * (
                    bbox[3] - bbox[1] + extra_length
                )
                if area >= min_area and area < max_area:
                    fp[scale_index, det_index] = 1
    return tp, fp


def _quantize_detections(predictions):
    '''Round predictions exactly as Detectron2's Pascal text serialization.'''

    quantized_predictions = []
    for image_predictions in predictions:
        quantized_classes = []
        for detections in image_predictions:
            quantized = detections.copy()
            for row in quantized:
                row[:4] = [float('{:.1f}'.format(value)) for value in row[:4]]
                row[4] = float('{:.3f}'.format(row[4]))
            quantized_classes.append(quantized)
        quantized_predictions.append(quantized_classes)
    return quantized_predictions


class Detectron2PascalVocMetric(VOCMetric):
    '''Evaluate Detectron2's VOC2012 AP with half-open box arithmetic.

    The Cityscapes cache reproduces the boxes returned by Detectron2's VOC
    loader: zero-based lower bounds and half-open upper bounds. MMDetection's
    stock :class:`VOCMetric` forces legacy ``+1`` widths and heights, so this
    subclass preserves its collection behavior while reproducing Detectron2's
    prediction serialization, strict ``IoU > 0.5`` match, and non-legacy
    coordinate arithmetic. Reported AP values use the paper's percent scale.
    '''

    default_prefix: Optional[str] = 'detectron2_voc'

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault('metric', 'mAP')
        kwargs.setdefault('eval_mode', 'area')
        super().__init__(*args, **kwargs)
        if self.metric != 'mAP':
            raise ValueError(
                'Detectron2PascalVocMetric supports only mAP evaluation'
            )

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        gts, raw_preds = zip(*results)
        preds = _quantize_detections(raw_preds)
        dataset_type = self.dataset_meta.get('dataset_type')
        if dataset_type in ('VOC2007', 'VOC2012'):
            dataset_name = 'voc'
            if dataset_type == 'VOC2007' and self.eval_mode != '11points':
                warnings.warn(
                    'Pascal VOC2007 uses `11points` as its default evaluation '
                    'mode, but Detectron2PascalVocMetric is using {!r}.'.format(
                        self.eval_mode
                    )
                )
            elif dataset_type == 'VOC2012' and self.eval_mode != 'area':
                warnings.warn(
                    'Pascal VOC2012 uses `area` as its default evaluation '
                    'mode, but Detectron2PascalVocMetric is using {!r}.'.format(
                        self.eval_mode
                    )
                )
        else:
            dataset_name = self.dataset_meta['classes']

        eval_results = OrderedDict()
        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(
                '\n{}iou_thr: {}{}'.format('-' * 15, iou_thr, '-' * 15)
            )
            mean_ap, _ = eval_map(
                preds,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                dataset=dataset_name,
                logger=logger,
                eval_mode=self.eval_mode,
                tpfp_fn=(
                    compute_detectron2_voc_true_false_positives
                ),
                use_legacy_coordinate=False,
            )
            mean_aps.append(mean_ap)
            eval_results['AP{:02d}'.format(int(iou_thr * 100))] = round(
                mean_ap * 100.0, 3
            )
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps) * 100.0
        eval_results.move_to_end('mAP', last=False)
        return eval_results
