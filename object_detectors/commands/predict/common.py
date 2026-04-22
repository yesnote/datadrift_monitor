import csv
import json
import math
import queue
import shutil
import threading
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloaders.dataloader_yolo import build_dataset, create_dataloader, yolo_collate_fn
from object_detectors.commands.utils.predict_utils import (
    assign_tp_to_predictions,
    build_detector,
    enable_forced_mc_dropout_on_yolov5_head,
    collect_bbox_gradients_per_target,
    collect_bbox_layer_grads_per_target,
    collect_batch_feature_gradients_per_target,
    collect_image_features_per_layer,
    collect_batch_image_layer_grads_per_target,
    create_layer_grad_buffer,
    draw_predictions,
    get_fn_gt_indices,
    get_pre_nms_keep_indices,
    has_fn_for_image,
    load_gt_category_maps,
    map_boxes_to_letterbox,
    map_grad_tensor_to_numbers,
    parse_output_config,
    preprocess_with_letterbox,
    expand_layer_names,
    resolve_layer_parameter,
)


def _xywh_to_xyxy_tensor(xywh: torch.Tensor) -> torch.Tensor:
    out = xywh.clone()
    out[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    out[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    out[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    out[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return out


def _build_summary(total_images, fn_images, output_csv):
    return {
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv),
    }


def _as_image_list(images):
    if isinstance(images, list):
        return images
    return [images[i] for i in range(images.shape[0])]


def _prepare_infer_batch(detector, images, device, auto=False):
    image_list = _as_image_list(images)
    infer_tensors = []
    ratios = []
    pads = []
    resized_chws = []
    for img in image_list:
        infer_tensor, ratio, pad, resized_chw = preprocess_with_letterbox(
            detector, img, device, requires_grad=False, auto=auto
        )
        infer_tensors.append(infer_tensor)
        ratios.append(ratio)
        pads.append(pad)
        resized_chws.append(resized_chw)
    infer_batch = torch.cat(infer_tensors, dim=0)
    return infer_batch, ratios, pads, resized_chws


def _resolve_detector_nms_kwargs(detector):
    return {
        "conf_thres": float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
        "iou_thres": float(getattr(detector, "iou_thresh", 0.45)),
        "classes": getattr(detector, "filter_classes", None),
        "agnostic": bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
        "max_det": int(getattr(detector, "max_det", 300)),
    }


def _resolve_nms_logits(raw_prediction, raw_logits, num_classes_hint=80):
    if raw_logits is not None:
        return raw_logits
    if raw_prediction is None:
        return None
    if raw_prediction.ndim == 3 and raw_prediction.shape[2] > 5:
        return raw_prediction[:, :, 5:]
    b = int(raw_prediction.shape[0]) if raw_prediction.ndim >= 1 else 0
    n = int(raw_prediction.shape[1]) if raw_prediction.ndim >= 2 else 0
    return torch.zeros((b, n, int(num_classes_hint)), dtype=raw_prediction.dtype, device=raw_prediction.device)


def _resolve_gt_class_names(target, catid_to_name):
    gt_names = target.get("gt_class_names")
    if gt_names is not None:
        return [str(v) for v in gt_names]
    gt_labels_tensor = target["labels"]
    return [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]


def _vector_from_grad_value(grad_value):
    if isinstance(grad_value, torch.Tensor):
        if grad_value.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = grad_value.detach().float().reshape(-1).cpu().numpy()
        return np.abs(arr.astype(np.float32, copy=False))
    if isinstance(grad_value, list):
        if len(grad_value) == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = np.asarray(grad_value, dtype=np.float32).reshape(-1)
        return np.abs(arr)
    if isinstance(grad_value, (int, float)):
        return np.asarray([abs(float(grad_value))], dtype=np.float32)
    if isinstance(grad_value, dict):
        # Fallback: when vector_reduction is configured, grad value may be a stats dict.
        vals = []
        for k in sorted(grad_value.keys()):
            try:
                vals.append(float(grad_value[k]))
            except Exception:
                continue
        if not vals:
            return np.zeros((0,), dtype=np.float32)
        return np.abs(np.asarray(vals, dtype=np.float32).reshape(-1))
    return np.zeros((0,), dtype=np.float32)


def _mc_dropout_single_csv_writer(write_queue, output_csv, fieldnames):
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            batch_rows = write_queue.get()
            if batch_rows is None:
                break
            if batch_rows:
                writer.writerows(batch_rows)

__all__ = [name for name in globals().keys() if not name.startswith("__")]
