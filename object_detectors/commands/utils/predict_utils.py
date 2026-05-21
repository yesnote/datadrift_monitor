import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from dataloaders.utils.data_utils import DATASET_CLASS_NAMES
from models.faster_rcnn import FasterRCNNTorchObjectDetector
from models.yolo.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


def resolve_project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists() or path.parts[:1] == ("object_detectors",):
        return repo_candidate
    return (PROJECT_ROOT / path).resolve()


def _sync_timing_device(device=None):
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _start_timing(device=None):
    _sync_timing_device(device)
    return time.perf_counter()


def _add_elapsed_timing(timing_accumulator, stage_name, start_time, device=None):
    _sync_timing_device(device)
    if timing_accumulator is not None:
        timing_accumulator[stage_name] = float(timing_accumulator.get(stage_name, 0.0)) + (
            time.perf_counter() - start_time
        )


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_count_or_inf(value, *, key_name):
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "inf":
            return math.inf
        if raw == "":
            raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.")
        try:
            parsed = int(raw)
        except Exception as e:
            raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.") from e
    elif value is None:
        raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.")
    else:
        parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{key_name} must be >= 0 or 'inf'.")
    return parsed


def get_dataset_cfg(config):
    dataset_root_cfg = config["dataset"]
    used_raw = dataset_root_cfg["used_dataset"]
    if isinstance(used_raw, str):
        used_list = [used_raw.strip().lower()]
    elif isinstance(used_raw, (list, tuple)):
        used_list = [str(v).strip().lower() for v in used_raw if str(v).strip()]
    else:
        raise ValueError("dataset.used_dataset must be a string or list of strings.")
    if not used_list:
        raise ValueError("dataset.used_dataset is empty.")
    for used_dataset in used_list:
        if used_dataset not in dataset_root_cfg:
            raise ValueError(f"dataset.{used_dataset} is missing in config.")
    if len(used_list) == 1:
        used_dataset = used_list[0]
        return used_dataset, dataset_root_cfg[used_dataset]
    return "__multi__", {name: dataset_root_cfg[name] for name in used_list}


def get_annotation_path(config, split):
    used_dataset, dataset_cfg = get_dataset_cfg(config)
    if used_dataset != "coco":
        return None
    root = Path(dataset_cfg["root"])
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    ann_dir = dataset_cfg.get("annotation_dir")
    ann_name = dataset_cfg.get(f"{split}_annotation_file")
    if not ann_dir or not ann_name:
        return None
    return root / ann_dir / ann_name


def load_coco_category_maps(annotation_path):
    if annotation_path is None:
        return {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(c["id"]): c["name"] for c in payload.get("categories", [])}


def load_gt_category_maps(config, split):
    used_dataset, dataset_cfg = get_dataset_cfg(config)
    if used_dataset == "coco":
        annotation_path = get_annotation_path(config, split)
        return load_coco_category_maps(annotation_path)
    return {}


def _resolve_detector_class_names(config):
    model_cfg = config.get("model", {})
    configured_names = model_cfg.get("class_names")
    if configured_names:
        if isinstance(configured_names, str):
            return [name.strip() for name in configured_names.split(",") if name.strip()]
        return [str(v) for v in configured_names]

    used_dataset, _dataset_cfg = get_dataset_cfg(config)
    if used_dataset in DATASET_CLASS_NAMES:
        return list(DATASET_CLASS_NAMES[used_dataset])
    if used_dataset == "__multi__":
        dataset_root_cfg = config.get("dataset", {})
        used_raw = dataset_root_cfg.get("used_dataset", [])
        if isinstance(used_raw, str):
            names = [used_raw.strip().lower()]
        else:
            names = [str(v).strip().lower() for v in used_raw if str(v).strip()]
        if names and all(name in DATASET_CLASS_NAMES for name in names):
            first = list(DATASET_CLASS_NAMES[names[0]])
            if all(list(DATASET_CLASS_NAMES[name]) == first for name in names):
                return first
            raise ValueError("Mixed datasets with different class spaces are not supported.")
    return None


def build_detector(config, model_weight=None):
    torch.backends.cudnn.benchmark = False
    model_cfg = config["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    confidence = model_cfg.get("confidence_threshold", 0.4)
    iou_thresh = model_cfg.get("iou_threshold", 0.45)

    model_type = str(model_cfg.get("type", "yolov5")).strip().lower()
    weight_source = model_weight if model_weight is not None else model_cfg.get("weights", "")
    if isinstance(weight_source, (list, tuple)):
        if len(weight_source) == 0:
            raise ValueError("model.weights is empty.")
        weight_source = weight_source[0]
    weight_path = None
    if str(weight_source).strip():
        weight_path = Path(weight_source)
        if not weight_path.is_absolute():
            weight_path = resolve_project_path(weight_path)
    img_size = model_cfg.get("img_size", 640)
    img_size_tuple = (int(img_size), int(img_size)) if isinstance(img_size, int) else tuple(img_size)
    if model_type in {"yolov5", "yolo", "yolo_v5"}:
        detector = YOLOV5TorchObjectDetector(
            model_weight=str(weight_path),
            device=device,
            img_size=img_size_tuple,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            iou_thresh=iou_thresh,
        )
    elif model_type in {"faster_rcnn", "faster-rcnn", "frcnn"}:
        detector = FasterRCNNTorchObjectDetector(
            model_weight=str(weight_path) if weight_path is not None else None,
            device=device,
            img_size=img_size_tuple,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            iou_thresh=iou_thresh,
            max_det=int(model_cfg.get("max_det", 300)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")
    detector.eval().to(device)
    return detector, device


def resolve_module_by_name(model, layer_name):
    node = model
    for token in layer_name.split("."):
        if token.isdigit():
            idx = int(token)
            if hasattr(node, "__getitem__"):
                node = node[idx]
            elif token in node._modules:
                node = node._modules[token]
            else:
                raise ValueError(f"Cannot index '{token}' in layer path '{layer_name}'")
            continue

        if hasattr(node, token):
            node = getattr(node, token)
            continue
        if hasattr(node, "_modules") and token in node._modules:
            node = node._modules[token]
            continue
        raise ValueError(f"Layer token '{token}' not found in path '{layer_name}'")

    return node


def resolve_layer_parameter(model, layer_name):
    module = resolve_module_by_name(model, layer_name)
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        return module.weight

    for param in module.parameters():
        if isinstance(param, torch.Tensor):
            return param
    raise ValueError(f"Layer '{layer_name}' has no parameters.")


def expand_layer_names(model, layer_names):
    resolved = []
    for name in normalize_to_list(layer_names):
        token = str(name).strip()
        if not token:
            continue
        if token not in resolved:
            resolved.append(token)
    return resolved


def parse_output_config(output_cfg):
    def as_dict(v):
        return v if isinstance(v, dict) else {}

    def as_int(v, d):
        try:
            return int(v)
        except Exception:
            return d

    def as_float(v, d):
        try:
            return float(v)
        except Exception:
            return d

    uncertainty = str(output_cfg.get("uncertainty", "gt")).strip().lower()
    if not uncertainty:
        uncertainty = "gt"
    supported = {
        "deterministic",
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "null_detect",
        "layer_grad",
    }
    if uncertainty not in supported:
        raise ValueError(f"Unsupported uncertainty: {uncertainty}")

    active = as_dict(output_cfg.get(uncertainty, {}))
    save_csv_cfg = active.get("save_csv", {})
    save_image_cfg = as_dict(active.get("save_image", {}))
    save_csv = as_dict(save_csv_cfg)
    save_csv_enabled = bool(save_csv.get("enabled", bool(save_csv_cfg) if isinstance(save_csv_cfg, bool) else False))
    unit = "bbox"
    pre_nms = False
    pre_nms_ratio = 1.0

    gt_cfg = active if uncertainty == "gt" else {}
    meta_detect_cfg = active if uncertainty == "meta_detect" else {}
    null_detect_cfg = active if uncertainty == "null_detect" else {}
    mc_dropout_cfg = active if uncertainty == "mc_dropout" else {}
    layer_grad_cfg = active if uncertainty == "layer_grad" else {}

    gt_iou_match_threshold = as_float(gt_cfg.get("iou_match_threshold", 0.5), 0.5)
    meta_detect_score_threshold = as_float(meta_detect_cfg.get("score_threshold", 0.0), 0.0)
    meta_detect_iou_threshold = as_float(meta_detect_cfg.get("iou_threshold", 0.45), 0.45)
    mc_num_runs = as_int(mc_dropout_cfg.get("num_runs", 30), 30)
    mc_dropout_rate = as_float(mc_dropout_cfg.get("dropout_rate", 0.5), 0.5)

    target_values = []
    target_layers = []
    layer_target_values = []
    layer_target_layers = []
    layer_map_reduction = "none"
    layer_gradient_reduction = []
    layer_pseudo_gt = "cand"
    layer_cand_score_threshold = 0.01
    layer_bbox_loss = "ciou"
    layer_cls_loss = "bcewithlogits"
    layer_obj_loss = "bcewithlogits"
    layer_bbox_direction = "pred_to_target"
    layer_cls_direction = "pred_to_target"
    layer_obj_direction = "pred_to_target"

    loss_aliases = {
        "bce": "bcewithlogits",
        "bce_with_logits": "bcewithlogits",
        "bcewithlogits": "bcewithlogits",
        "ciou": "ciou",
        "l1": "l1",
        "l1_loss": "l1",
        "l2": "l2",
        "l2_loss": "l2",
        "mse": "l2",
        "kl": "kl",
        "kl_div": "kl",
        "kl_divergence": "kl",
        "ce": "ce",
        "cross_entropy": "ce",
        "abs": "abs_diff",
        "abs_diff": "abs_diff",
        "absolute_diff": "abs_diff",
        "signed": "signed_diff",
        "signed_diff": "signed_diff",
    }
    direction_aliases = {
        "pred_to_target": "pred_to_target",
        "prediction_to_target": "pred_to_target",
        "target": "pred_to_target",
        "target_to_pred": "target_to_pred",
        "target_to_prediction": "target_to_pred",
        "reverse": "target_to_pred",
    }

    def normalize_loss_option(raw, default, supported, key_name):
        normalized = loss_aliases.get(str(raw if raw is not None else default).strip().lower().replace("-", "_"))
        if normalized not in supported:
            raise ValueError(f"Unsupported {key_name}: {raw}. Supported values: {', '.join(sorted(supported))}.")
        return normalized

    def normalize_direction_option(raw, key_name):
        normalized = direction_aliases.get(
            str(raw if raw is not None else "pred_to_target").strip().lower().replace("-", "_")
        )
        if normalized is None:
            raise ValueError(f"Unsupported {key_name}: {raw}. Supported values: pred_to_target, target_to_pred.")
        return normalized

    def validate_loss_directions(bbox_direction, cls_loss, obj_loss, cls_direction, obj_direction):
        if bbox_direction != "pred_to_target":
            raise ValueError("bbox_direction=target_to_pred is not supported because ciou, l1, and l2 bbox losses are symmetric.")
        if cls_direction == "target_to_pred" and cls_loss in {"bcewithlogits", "ce"}:
            raise ValueError("cls_direction=target_to_pred is only supported when cls_loss=kl.")
        if obj_direction == "target_to_pred" and obj_loss != "signed_diff":
            raise ValueError("obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")

    if uncertainty == "layer_grad":
        g = as_dict(layer_grad_cfg.get("gradient", {}))
        layer_target_values = [v.lower() for v in normalize_to_list(g.get("scalar", ["loss"]))]
        if "loss" in layer_target_values:
            exp = []
            for v in layer_target_values:
                exp.extend(["obj_loss", "cls_loss", "bbox_loss"] if v == "loss" else [v])
            layer_target_values = list(dict.fromkeys(exp))
        layer_target_layers = normalize_to_list(g.get("layer", []))
        reduction_aliases = {
            "1-norm": "l1_norm",
            "1_norm": "l1_norm",
            "l1": "l1_norm",
            "l1_norm": "l1_norm",
            "2-norm": "l2_norm",
            "2_norm": "l2_norm",
            "l2": "l2_norm",
            "l2_norm": "l2_norm",
            "min": "min",
            "max": "max",
            "mean": "mean",
            "std": "std",
        }
        layer_gradient_reduction = []
        for reduction_name in normalize_to_list(g.get("reduction", [])):
            key = str(reduction_name).strip().lower().replace("-", "_")
            normalized = reduction_aliases.get(key)
            if normalized is None:
                raise ValueError(
                    "Unsupported layer_grad.gradient.reduction value "
                    f"'{reduction_name}'. Supported values: l1_norm, l2_norm, min, max, mean, std."
                )
            if normalized not in layer_gradient_reduction:
                layer_gradient_reduction.append(normalized)
        t = g.get("target", "cand_target")
        t_policy = str(t).strip().lower() if t is not None else "null_target"
        layer_pseudo_gt = "uniform" if t_policy in {"null_target", "null"} else "cand"
        layer_cand_score_threshold = as_float(g.get("cand_score_threshold", 0.01), 0.01)

        layer_bbox_loss = normalize_loss_option(g.get("bbox_loss", "ciou"), "ciou", {"ciou", "l1", "l2"}, "layer_grad.gradient.bbox_loss")
        layer_cls_loss = normalize_loss_option(
            g.get("cls_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "kl", "ce"},
            "layer_grad.gradient.cls_loss",
        )
        layer_obj_loss = normalize_loss_option(
            g.get("obj_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "abs_diff", "signed_diff"},
            "layer_grad.gradient.obj_loss",
        )
        layer_bbox_direction = normalize_direction_option(g.get("bbox_direction", "pred_to_target"), "layer_grad.gradient.bbox_direction")
        layer_cls_direction = normalize_direction_option(g.get("cls_direction", "pred_to_target"), "layer_grad.gradient.cls_direction")
        layer_obj_direction = normalize_direction_option(g.get("obj_direction", "pred_to_target"), "layer_grad.gradient.obj_direction")
        validate_loss_directions(layer_bbox_direction, layer_cls_loss, layer_obj_loss, layer_cls_direction, layer_obj_direction)

    null_bbox_loss = normalize_loss_option(null_detect_cfg.get("bbox_loss", "ciou"), "ciou", {"ciou", "l1", "l2"}, "null_detect.bbox_loss")
    null_cls_loss = normalize_loss_option(
        null_detect_cfg.get("cls_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "kl", "ce"},
        "null_detect.cls_loss",
    )
    null_obj_loss = normalize_loss_option(
        null_detect_cfg.get("obj_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "abs_diff", "signed_diff"},
        "null_detect.obj_loss",
    )
    null_bbox_direction = normalize_direction_option(null_detect_cfg.get("bbox_direction", "pred_to_target"), "null_detect.bbox_direction")
    null_cls_direction = normalize_direction_option(null_detect_cfg.get("cls_direction", "pred_to_target"), "null_detect.cls_direction")
    null_obj_direction = normalize_direction_option(null_detect_cfg.get("obj_direction", "pred_to_target"), "null_detect.obj_direction")
    validate_loss_directions(null_bbox_direction, null_cls_loss, null_obj_loss, null_cls_direction, null_obj_direction)
    null_feature_set = str(null_detect_cfg.get("feature_set", "full")).strip().lower().replace("-", "_")
    if null_feature_set not in {"full", "losses_only"}:
        raise ValueError("Unsupported null_detect.feature_set. Supported values: full, losses_only.")

    save_image_enabled = bool(save_image_cfg.get("enabled", bool(save_image_cfg)))
    gt_image_step = as_int(save_image_cfg.get("step", 1), 1)
    gt_image_max_num = as_int(save_image_cfg.get("max_num", 1), 1)

    return {
        "save_csv_enabled": save_csv_enabled,
        "uncertainty": uncertainty,
        "pre_nms": pre_nms,
        "pre_nms_ratio": float(pre_nms_ratio),
        "unit": unit,
        "gt_iou_match_threshold": gt_iou_match_threshold,
        "meta_detect_score_threshold": meta_detect_score_threshold,
        "meta_detect_iou_threshold": meta_detect_iou_threshold,
        "mc_num_runs": mc_num_runs,
        "mc_dropout_rate": mc_dropout_rate,
        "target_values": target_values,
        "target_layers": target_layers,
        "layer_target_values": layer_target_values,
        "layer_target_layers": layer_target_layers,
        "layer_map_reduction": layer_map_reduction,
        "layer_gradient_reduction": layer_gradient_reduction,
        "layer_pseudo_gt": layer_pseudo_gt,
        "layer_cand_score_threshold": float(layer_cand_score_threshold),
        "layer_bbox_loss": layer_bbox_loss,
        "layer_cls_loss": layer_cls_loss,
        "layer_obj_loss": layer_obj_loss,
        "layer_bbox_direction": layer_bbox_direction,
        "layer_cls_direction": layer_cls_direction,
        "layer_obj_direction": layer_obj_direction,
        "null_detect_bbox_loss": null_bbox_loss,
        "null_detect_cls_loss": null_cls_loss,
        "null_detect_obj_loss": null_obj_loss,
        "null_detect_bbox_direction": null_bbox_direction,
        "null_detect_cls_direction": null_cls_direction,
        "null_detect_obj_direction": null_obj_direction,
        "null_detect_feature_set": null_feature_set,
        "layer_ref_mode": "none",
        "layer_ref_type": "none",
        "layer_ref_prototype_mode": "none",
        "layer_ref_subspace_mode": "none",
        "layer_ref_subspace_centering": "centered",
        "layer_ref_subspace_rank_mode": "energy",
        "layer_ref_subspace_energy_threshold": 0.95,
        "layer_ref_subspace_k": 10,
        "layer_ref_subspace_max_samples": 1000,
        "layer_disc_separation_score": "effect_size",
        "layer_disc_topk": 3,
        "layer_disc_fn_non_fn_map_root": "",
        "layer_ref_map_root": "",
        "save_image_enabled": save_image_enabled,
        "save_image_gt_step": gt_image_step,
        "save_image_gt_max_num": gt_image_max_num,
    }


def build_target_scalar_pre_nms(target_value, raw_prediction, raw_logits):
    if target_value == "obj":
        if raw_prediction is None or raw_prediction.numel() == 0:
            return None
        # NMS 이전: 모든 후보 bbox의 objectness(sigmoid) 합
        return raw_prediction[..., 4].sum()

    if target_value == "cls":
        if raw_logits is None or raw_logits.numel() == 0:
            return None
        # NMS 이전: 모든 후보 bbox의 max(class logit) 합
        return raw_logits.max(dim=-1).values.sum()

    raise ValueError(f"Unsupported target_value: {target_value}")


def get_pre_nms_keep_indices(pred_row, logit_row=None, pre_nms_ratio=1.0):
    if pred_row is None or pred_row.numel() == 0:
        device = pred_row.device if isinstance(pred_row, torch.Tensor) else "cpu"
        return torch.zeros((0,), dtype=torch.long, device=device)
    n = int(pred_row.shape[0])
    ratio = float(pre_nms_ratio)
    if ratio <= 0.0:
        return torch.zeros((0,), dtype=torch.long, device=pred_row.device)
    if ratio >= 1.0:
        return torch.arange(n, device=pred_row.device, dtype=torch.long)

    obj = pred_row[:, 4] if pred_row.shape[1] > 4 else torch.ones((n,), device=pred_row.device)
    if logit_row is not None and isinstance(logit_row, torch.Tensor) and logit_row.numel() > 0:
        cls_prob = torch.sigmoid(logit_row)
        cls_max = cls_prob.max(dim=1).values if cls_prob.ndim == 2 and cls_prob.shape[1] > 0 else torch.ones_like(obj)
    else:
        cls_scores = pred_row[:, 5:] if pred_row.shape[1] > 5 else None
        cls_max = cls_scores.max(dim=1).values if cls_scores is not None and cls_scores.numel() > 0 else torch.ones_like(obj)
    score = obj * cls_max

    keep = int(np.ceil(n * ratio))
    keep = max(0, min(n, keep))
    if keep == 0:
        return torch.zeros((0,), dtype=torch.long, device=pred_row.device)
    if keep == n:
        return torch.arange(n, device=pred_row.device, dtype=torch.long)
    top_idx = torch.topk(score, k=keep, largest=True, sorted=False).indices
    top_idx, _ = torch.sort(top_idx)
    return top_idx


def collect_gradients_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    layer_buffer,
    pre_nms=True,
    pre_nms_ratio=1.0,
):
    grad_stats = {}
    for target_value in target_values:
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()

        grad_input = input_tensor.detach().requires_grad_(True)
        model_output = detector.model(grad_input, augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None

        target_scalar = None
        if target_value in {"obj", "cls"}:
            if pre_nms:
                pred_img = raw_prediction[0] if raw_prediction is not None and raw_prediction.ndim == 3 else None
                logit_img = raw_logits[0] if raw_logits is not None and raw_logits.ndim == 3 else None
                keep_idx = get_pre_nms_keep_indices(pred_img, logit_img, pre_nms_ratio=pre_nms_ratio)
                if int(keep_idx.shape[0]) > 0:
                    if target_value == "obj":
                        target_scalar = pred_img[keep_idx, 4].sum()
                    else:
                        logits_for_cls = logit_img if logit_img is not None else pred_img[:, 5:]
                        if logits_for_cls is not None and logits_for_cls.numel() > 0:
                            target_scalar = logits_for_cls[keep_idx].max(dim=1).values.sum()
            else:
                with torch.no_grad():
                    _selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                        raw_prediction.detach(),
                        raw_logits.detach() if raw_logits is not None else raw_logits,
                        detector.confidence,
                        detector.iou_thresh,
                        classes=None,
                        agnostic=detector.agnostic,
                        return_indices=True,
                    )
                    raw_keep_indices = (
                        selected_indices[0]
                        if selected_indices
                        else torch.zeros((0,), dtype=torch.long, device=grad_input.device)
                    )
                if int(raw_keep_indices.shape[0]) > 0:
                    pred_img = raw_prediction[0]
                    logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]
                    if target_value == "obj":
                        target_scalar = pred_img[raw_keep_indices, 4].sum()
                    else:
                        if logit_img is not None and logit_img.numel() > 0:
                            target_scalar = logit_img[raw_keep_indices].max(dim=1).values.sum()
        else:
            # Loss targets use the final NMS predictions.
            with torch.no_grad():
                _selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    raw_prediction.detach(),
                    raw_logits.detach() if raw_logits is not None else raw_logits,
                    detector.confidence,
                    detector.iou_thresh,
                    classes=None,
                    agnostic=detector.agnostic,
                    return_indices=True,
                )
                raw_keep_indices = (
                    selected_indices[0]
                    if selected_indices
                    else torch.zeros((0,), dtype=torch.long, device=grad_input.device)
                )

            pred_img = raw_prediction[0]
            iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
            loss_terms = []
            for bbox_idx in range(int(raw_keep_indices.shape[0])):
                raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                losses = build_pseudo_label_losses_for_candidates(
                    pred_img=pred_img,
                    raw_idx=raw_idx,
                    iou_threshold=iou_threshold,
                )
                if losses is not None and target_value in losses:
                    loss_terms.append(losses[target_value])
            if loss_terms:
                target_scalar = torch.stack(loss_terms).mean()

        if target_scalar is None:
            for layer_name in target_layers:
                grad_stats[f"{target_value}_{layer_name}"] = []
            if grad_input.grad is not None:
                grad_input.grad = None
            del grad_input, model_output, raw_prediction, raw_logits
            layer_buffer.clear()
            continue

        target_scalar.backward()
        layer_stats = list(layer_buffer.gradients["value"])
        layer_stats.reverse()

        for layer_idx, layer_name in enumerate(target_layers):
            key = f"{target_value}_{layer_name}"
            grad_stats[key] = layer_stats[layer_idx] if layer_idx < len(layer_stats) else []

        if grad_input.grad is not None:
            grad_input.grad = None
        del grad_input, model_output, raw_prediction, raw_logits, target_scalar, layer_stats
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()
    return grad_stats


def map_grad_tensor_to_numbers(v):
    if v is None:
        return {"l1_norm": 0.0, "l2_norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    v = v.detach().float().reshape(-1)
    if v.numel() == 0:
        return {"l1_norm": 0.0, "l2_norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "l1_norm": torch.norm(v, p=1),
        "l2_norm": torch.norm(v, p=2),
        "min": v.min(),
        "max": v.max(),
        "mean": torch.mean(v),
        "std": torch.std(v, unbiased=False),
    }


def configure_mc_dropout(model: torch.nn.Module, dropout_rate: float) -> int:
    model.eval()
    count = 0
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.p = float(dropout_rate)
            module.train()
            count += 1
    return count


def enable_forced_mc_dropout_on_yolov5_head(model: torch.nn.Module, dropout_rate: float):
    # DiL-style behavior for YOLO: apply dropout in inference path explicitly.
    detect_module = None
    for module in model.modules():
        if hasattr(module, "m") and hasattr(module, "nc") and hasattr(module, "na"):
            try:
                if isinstance(module.m, nn.ModuleList) and len(module.m) > 0 and isinstance(module.m[0], nn.Conv2d):
                    detect_module = module
            except Exception:
                continue

    if detect_module is None:
        return []

    handles = []
    p = float(dropout_rate)
    for conv in detect_module.m:
        def _pre_hook(_module, inputs, p_drop=p):
            if not inputs:
                return inputs
            x = inputs[0]
            x = F.dropout(x, p=p_drop, training=True)
            if len(inputs) == 1:
                return (x,)
            return (x,) + tuple(inputs[1:])

        handles.append(conv.register_forward_pre_hook(_pre_hook))
    return handles


def format_gradient_output(grad_tensor, vector_reduction, map_reduction="none"):
    if grad_tensor is None:
        return []
    grad_tensor = grad_tensor.detach().float()
    map_mode = str(map_reduction).strip().lower()
    if map_mode not in {"none", "energy"}:
        raise ValueError("layer_grad.map_reduction must be 'none' or 'energy'.")

    if map_mode == "energy":
        if grad_tensor.ndim == 0:
            reduced = grad_tensor.abs().reshape(1)
        elif grad_tensor.ndim == 1:
            reduced = grad_tensor.abs()
        else:
            first_dim = int(grad_tensor.shape[0])
            reduced = grad_tensor.reshape(first_dim, -1).abs().mean(dim=1)
    else:
        reduced = grad_tensor.reshape(-1)

    if not vector_reduction:
        return reduced.detach()
    stats = map_grad_tensor_to_numbers(reduced)
    return {k: stats[k] for k in vector_reduction}


def zero_grad_numbers():
    return {
        "l1_norm": 0.0,
        "l2_norm": 0.0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }


def build_pseudo_label_losses(pred_row):
    eps = 1e-6
    obj_prob = pred_row[4].clamp(eps, 1.0 - eps)
    cls_prob = pred_row[5:].clamp(eps, 1.0 - eps)
    cls_idx = int(torch.argmax(cls_prob.detach()).item())
    cls_target = torch.zeros_like(cls_prob)
    cls_target[cls_idx] = 1.0

    obj_loss = F.binary_cross_entropy(obj_prob, torch.ones_like(obj_prob))
    cls_loss = F.binary_cross_entropy(cls_prob, cls_target)
    return {
        "obj_loss": obj_loss,
        "cls_loss": cls_loss,
        "loss": obj_loss + cls_loss,
    }


def _xywh_to_xyxy_tensor(xywh: torch.Tensor) -> torch.Tensor:
    out = xywh.clone()
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def _bbox_ciou_xywh_tensor(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.ndim == 1:
        box1 = box1.view(1, 4)
    if box2.ndim == 1:
        box2 = box2.view(1, 4)
    if box2.shape[0] == 1 and box1.shape[0] > 1:
        box2 = box2.expand(box1.shape[0], -1)

    b1_x1 = box1[:, 0] - box1[:, 2] / 2.0
    b1_y1 = box1[:, 1] - box1[:, 3] / 2.0
    b1_x2 = box1[:, 0] + box1[:, 2] / 2.0
    b1_y2 = box1[:, 1] + box1[:, 3] / 2.0
    b2_x1 = box2[:, 0] - box2[:, 2] / 2.0
    b2_y1 = box2[:, 1] - box2[:, 3] / 2.0
    b2_x2 = box2[:, 0] + box2[:, 2] / 2.0
    b2_y2 = box2[:, 1] + box2[:, 3] / 2.0

    inter = (
        (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    )
    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (box1[:, 0] - box2[:, 0]).pow(2) + (box1[:, 1] - box2[:, 1]).pow(2)
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def _box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _box_iou_1vN_tensor(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=box.device)
    if box.ndim > 1:
        box = box.reshape(-1, 4)[0]
    lt = torch.max(box[:2], boxes[:, :2])
    rb = torch.min(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-12)


def _flatten_raw_prediction_layers(pred_layers):
    if not isinstance(pred_layers, list):
        return None
    flat = []
    for layer_pred in pred_layers:
        if not isinstance(layer_pred, torch.Tensor) or layer_pred.ndim != 5:
            return None
        flat.append(layer_pred.reshape(layer_pred.shape[0], -1, layer_pred.shape[-1]))
    if not flat:
        return None
    return torch.cat(flat, dim=1)


def _plain_iou_xywh_tensor(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    box1_xyxy = _xywh_to_xyxy_tensor(box1.reshape(-1, 4))
    box2_xyxy = _xywh_to_xyxy_tensor(box2.reshape(-1, 4))
    if box2_xyxy.shape[0] == 1 and box1_xyxy.shape[0] > 1:
        box2_xyxy = box2_xyxy.expand(box1_xyxy.shape[0], -1)
    return _box_iou_1vN_tensor(box2_xyxy[0], box1_xyxy)


def _apply_direction(pred_value: torch.Tensor, target_value: torch.Tensor, direction: str):
    if direction == "target_to_pred":
        return target_value, pred_value
    return pred_value, target_value


def _bbox_loss_xywh_tensor(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    mode: str = "ciou",
    reduction: str = "mean",
    direction: str = "pred_to_target",
):
    if direction != "pred_to_target":
        raise ValueError("bbox_direction=target_to_pred is not supported because ciou, l1, and l2 bbox losses are symmetric.")
    mode = str(mode).strip().lower()
    target_xywh = target_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device)
    left, right = pred_xywh, target_xywh

    if mode == "ciou":
        loss = 1.0 - _bbox_ciou_xywh_tensor(left, right)
    elif mode == "l1":
        loss = torch.abs(left - right)
    elif mode == "l2":
        loss = torch.square(left - right)
    else:
        raise ValueError("bbox_loss must be one of: ciou, l1, l2")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _objectness_loss_tensor(
    raw_obj: torch.Tensor,
    target_prob: torch.Tensor,
    mode: str = "bcewithlogits",
    direction: str = "pred_to_target",
    reduction: str = "sum",
):
    target_prob = target_prob.to(dtype=raw_obj.dtype, device=raw_obj.device)
    mode = str(mode).strip().lower()
    if mode == "bcewithlogits":
        if direction != "pred_to_target":
            raise ValueError("obj_direction=target_to_pred is not supported when obj_loss=bcewithlogits.")
        loss = F.binary_cross_entropy_with_logits(raw_obj, target_prob, reduction="none")
    elif mode in {"abs_diff", "signed_diff"}:
        if direction == "target_to_pred" and mode != "signed_diff":
            raise ValueError("obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")
        pred_prob = torch.sigmoid(raw_obj)
        left, right = _apply_direction(pred_prob, target_prob, direction)
        loss = torch.abs(left - right) if mode == "abs_diff" else (left - right)
    else:
        raise ValueError("obj_loss must be one of: bcewithlogits, abs_diff, signed_diff")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _smooth_distribution(target_prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target_prob = target_prob.clamp(min=eps)
    return target_prob / target_prob.sum(dim=-1, keepdim=True).clamp(min=eps)


def _class_loss_tensor(
    raw_cls_logits: torch.Tensor,
    target_prob: torch.Tensor,
    class_idx: int = None,
    mode: str = "bcewithlogits",
    direction: str = "pred_to_target",
    reduction: str = "sum",
):
    if raw_cls_logits.ndim == 1:
        raw_cls_logits = raw_cls_logits.view(1, -1)
    if target_prob.ndim == 1:
        target_prob = target_prob.view(1, -1)
    target_prob = target_prob.to(dtype=raw_cls_logits.dtype, device=raw_cls_logits.device)
    if target_prob.shape[0] == 1 and raw_cls_logits.shape[0] > 1:
        target_prob = target_prob.expand(raw_cls_logits.shape[0], -1)

    mode = str(mode).strip().lower()
    if mode == "bcewithlogits":
        if direction != "pred_to_target":
            raise ValueError("cls_direction=target_to_pred is not supported when cls_loss=bcewithlogits.")
        loss = F.binary_cross_entropy_with_logits(raw_cls_logits, target_prob, reduction="none").sum(dim=-1)
    elif mode == "ce":
        if direction != "pred_to_target":
            raise ValueError("cls_direction=target_to_pred is not supported when cls_loss=ce.")
        if class_idx is None:
            loss = -(target_prob * F.log_softmax(raw_cls_logits, dim=-1)).sum(dim=-1)
        else:
            class_target = torch.full(
                (raw_cls_logits.shape[0],),
                int(class_idx),
                dtype=torch.long,
                device=raw_cls_logits.device,
            )
            loss = F.cross_entropy(raw_cls_logits, class_target, reduction="none")
    elif mode == "kl":
        pred_log_prob = F.log_softmax(raw_cls_logits, dim=-1)
        if direction == "target_to_pred":
            pred_prob = pred_log_prob.exp()
            target_log_prob = _smooth_distribution(target_prob).log()
            loss = (pred_prob * (pred_log_prob - target_log_prob)).sum(dim=-1)
        else:
            safe_target = target_prob.clamp(min=1e-12)
            loss = torch.where(
                target_prob > 0,
                target_prob * (safe_target.log() - pred_log_prob),
                torch.zeros_like(target_prob),
            ).sum(dim=-1)
    else:
        raise ValueError("cls_loss must be one of: bcewithlogits, kl, ce")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def build_pseudo_label_losses_for_candidates(
    pred_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
    score_threshold: float = 0.01,
    bbox_loss: str = "ciou",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    raw_img: torch.Tensor = None,
    timing_accumulator=None,
    timing_device=None,
):
    if raw_idx >= pred_img.shape[0]:
        return None

    t_candidate = _start_timing(timing_device)
    with torch.no_grad():
        pseudo_row = pred_img[raw_idx].detach()
        pseudo_cls = int(torch.argmax(pseudo_row[5:]).item())
        pred_boxes_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4].detach())
        pseudo_box_xyxy = _xywh_to_xyxy_tensor(pseudo_row[:4].view(1, 4))
        ious = _box_iou_1vN_tensor(pseudo_box_xyxy, pred_boxes_xyxy)
        pred_cls = torch.argmax(pred_img[:, 5:].detach(), dim=1)
        obj = pred_img[:, 4].detach()
        cls_max = pred_img[:, 5:].detach().max(dim=1).values if pred_img.shape[1] > 5 else torch.ones_like(obj)
        score = obj * cls_max
        candidate_mask = (ious >= float(iou_threshold)) & (pred_cls == pseudo_cls) & (score >= float(score_threshold))
        if not bool(candidate_mask.any()):
            candidate_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
            candidate_mask[raw_idx] = True
    _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)

    t_loss = _start_timing(timing_device)
    candidate_pred = pred_img[candidate_mask]
    if raw_img is not None and raw_img.shape[0] == pred_img.shape[0]:
        raw_obj = raw_img[:, 4]
        raw_cls = raw_img[:, 5:]
    else:
        eps = 1e-6
        raw_obj = torch.logit(pred_img[:, 4].clamp(eps, 1.0 - eps))
        raw_cls = torch.logit(pred_img[:, 5:].clamp(eps, 1.0 - eps))
    candidate_raw_cls = raw_cls[candidate_mask]

    pseudo_box_target = pseudo_row[:4].view(1, 4).expand(candidate_pred.shape[0], -1)
    bbox_loss_value = _bbox_loss_xywh_tensor(
        candidate_pred[:, :4],
        pseudo_box_target,
        mode=bbox_loss,
        reduction="sum",
        direction=bbox_direction,
    )

    candidate_raw_obj = raw_obj[candidate_mask]
    obj_target = ious[candidate_mask].detach().to(dtype=candidate_raw_obj.dtype).clamp(min=0.0, max=1.0)
    obj_loss_value = _objectness_loss_tensor(
        candidate_raw_obj,
        obj_target,
        mode=obj_loss,
        direction=obj_direction,
        reduction="sum",
    )

    cls_target = torch.zeros_like(candidate_raw_cls)
    cls_target[:, pseudo_cls] = 1.0
    cls_loss_value = _class_loss_tensor(
        candidate_raw_cls,
        cls_target,
        class_idx=pseudo_cls,
        mode=cls_loss,
        direction=cls_direction,
        reduction="sum",
    )

    losses = {
        "bbox_loss": bbox_loss_value,
        "obj_loss": obj_loss_value,
        "cls_loss": cls_loss_value,
    }
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
    return losses


def build_layer_target_scalar_bbox(
    target_value,
    pred_img,
    logit_img,
    raw_img,
    raw_idx,
    iou_threshold,
    pseudo_gt="cand",
    anchor_xywh=None,
    cand_score_threshold=0.01,
    bbox_loss: str = "ciou",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    if target_value == "obj":
        if raw_idx >= pred_img.shape[0]:
            return None
        return pred_img[raw_idx, 4]
    if target_value == "cls":
        if raw_idx >= logit_img.shape[0]:
            return None
        return logit_img[raw_idx].max()

    if pseudo_gt == "uniform":
        if raw_idx >= pred_img.shape[0]:
            return None
        t_loss = _start_timing(timing_device)
        pred_row = pred_img[raw_idx]
        raw_row = raw_img[raw_idx] if raw_img is not None and raw_idx < raw_img.shape[0] else None
        if target_value == "obj_loss":
            if anchor_xywh is None:
                return None
            if raw_row is not None:
                raw_obj = raw_row[4]
            else:
                raw_obj = torch.logit(pred_row[4].clamp(1e-6, 1.0 - 1e-6))
            target_iou = _plain_iou_xywh_tensor(pred_row[:4].detach(), anchor_xywh.detach()).to(
                dtype=raw_obj.dtype,
                device=raw_obj.device,
            )
            obj_target = target_iou.reshape(()).clamp(min=0.0, max=1.0)
            loss = _objectness_loss_tensor(raw_obj, obj_target, mode=obj_loss, direction=obj_direction, reduction="sum")
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        if target_value == "cls_loss":
            if raw_row is not None:
                cls_logits = raw_row[5:]
            else:
                cls_logits = logit_img[raw_idx] if (logit_img is not None and raw_idx < logit_img.shape[0]) else pred_row[5:]
            if cls_logits.numel() == 0:
                return None
            uniform_target = torch.full_like(cls_logits, 1.0 / float(cls_logits.numel()))
            loss = _class_loss_tensor(
                cls_logits,
                uniform_target,
                class_idx=None,
                mode=cls_loss,
                direction=cls_direction,
                reduction="sum",
            )
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        if target_value == "bbox_loss":
            if anchor_xywh is None:
                return None
            pred_xywh = pred_row[:4]
            anchor_xywh = anchor_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device)
            loss = _bbox_loss_xywh_tensor(
                pred_xywh,
                anchor_xywh,
                mode=bbox_loss,
                reduction="mean",
                direction=bbox_direction,
            )
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None

    losses = build_pseudo_label_losses_for_candidates(
        pred_img=pred_img,
        raw_idx=raw_idx,
        iou_threshold=iou_threshold,
        score_threshold=cand_score_threshold,
        bbox_loss=bbox_loss,
        cls_loss=cls_loss,
        obj_loss=obj_loss,
        bbox_direction=bbox_direction,
        cls_direction=cls_direction,
        obj_direction=obj_direction,
        raw_img=raw_img,
        timing_accumulator=timing_accumulator,
        timing_device=timing_device,
    )
    if losses is not None and target_value in losses:
        return losses[target_value]
    return None


def collect_bbox_layer_grads_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    map_reduction="none",
    vector_reduction=None,
    pseudo_gt="cand",
    cand_score_threshold=0.01,
    bbox_loss: str = "ciou",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    t_detector = _start_timing(timing_device)
    model_output = detector.model(input_tensor.detach(), augment=False)
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    pred_layers = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 and isinstance(model_output[2], list) else None
    raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
    with torch.no_grad():
        nms_prediction = raw_prediction.detach().clone()
        nms_logits = raw_logits.detach().clone() if raw_logits is not None else None
        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            nms_prediction,
            nms_logits,
            detector.confidence,
            detector.iou_thresh,
            classes=None,
            agnostic=detector.agnostic,
            return_indices=True,
        )
    _add_elapsed_timing(timing_accumulator, "detector_inference_sec", t_detector, timing_device)

    det = selected_preds[0] if selected_preds else torch.zeros((0, 6), device=input_tensor.device)
    raw_keep_indices = selected_indices[0] if selected_indices else torch.zeros((0,), dtype=torch.long, device=input_tensor.device)
    pred_img = raw_prediction[0]
    logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]
    raw_flat = _flatten_raw_prediction_layers(pred_layers)
    raw_img = raw_flat[0] if raw_flat is not None and raw_flat.ndim == 3 else None
    anchor_img = raw_anchor_priors[0] if raw_anchor_priors is not None else None

    rows = []
    num_boxes = int(det.shape[0])
    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    try:
        for bbox_idx in range(num_boxes):
            raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
            grad_stats = {}
            for target_value in target_values:
                detector.zero_grad(set_to_none=True)

                anchor_row = anchor_img[raw_idx] if (anchor_img is not None and raw_idx < anchor_img.shape[0]) else None

                target_scalar = build_layer_target_scalar_bbox(
                    target_value=target_value,
                    pred_img=pred_img,
                    logit_img=logit_img,
                    raw_img=raw_img,
                    raw_idx=raw_idx,
                    iou_threshold=iou_threshold,
                    pseudo_gt=pseudo_gt,
                    anchor_xywh=anchor_row,
                    cand_score_threshold=cand_score_threshold,
                    bbox_loss=bbox_loss,
                    cls_loss=cls_loss,
                    obj_loss=obj_loss,
                    bbox_direction=bbox_direction,
                    cls_direction=cls_direction,
                    obj_direction=obj_direction,
                    timing_accumulator=timing_accumulator,
                    timing_device=timing_device,
                )

                if target_scalar is None:
                    for layer_name in target_layers:
                        grad_stats[f"{target_value}_{layer_name}"] = (
                            {metric: 0.0 for metric in vector_reduction} if vector_reduction else []
                        )
                    continue

                t_backprop = _start_timing(timing_device)
                grads = torch.autograd.grad(
                    target_scalar,
                    layer_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                _add_elapsed_timing(timing_accumulator, "backpropagation_sec", t_backprop, timing_device)
                t_feature = _start_timing(timing_device)
                for layer_idx, layer_name in enumerate(target_layers):
                    key = f"{target_value}_{layer_name}"
                    grad_tensor = grads[layer_idx]
                    grad_stats[key] = format_gradient_output(
                        grad_tensor,
                        vector_reduction=vector_reduction,
                        map_reduction=map_reduction,
                    )
                _add_elapsed_timing(timing_accumulator, "feature_compute_sec", t_feature, timing_device)

                del target_scalar, grads

            cls_idx = int(det[bbox_idx, 5].detach().cpu().item())
            rows.append(
                {
                    "pred_idx": bbox_idx,
                    "raw_pred_idx": raw_idx,
                    "xmin": float(det[bbox_idx, 0].detach().cpu().item()),
                    "ymin": float(det[bbox_idx, 1].detach().cpu().item()),
                    "xmax": float(det[bbox_idx, 2].detach().cpu().item()),
                    "ymax": float(det[bbox_idx, 3].detach().cpu().item()),
                    "score": float(det[bbox_idx, 4].detach().cpu().item()),
                    "pred_class": detector.names[cls_idx] if detector.names is not None else cls_idx,
                    "grad_stats": grad_stats,
                }
            )
    finally:
        for param, req_grad in zip(layer_params, original_requires_grad):
            param.requires_grad_(req_grad)
        detector.zero_grad(set_to_none=True)
        del model_output, raw_prediction, raw_logits, raw_anchor_priors, pred_img, logit_img, raw_flat, raw_img, anchor_img

    return rows


def preprocess_with_letterbox(detector, image_tensor, device, requires_grad=True, auto=True):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # For most modes we keep YOLO letterbox default behavior (auto=True).
    # MC-dropout batched path can set auto=False to force fixed-size tensors.
    resized, ratio, pad = detector.yolo_resize(image_np, new_shape=detector.img_size, auto=auto)
    resized = resized.transpose((2, 0, 1))
    resized = np.ascontiguousarray(resized)
    input_tensor = torch.from_numpy(resized).float().unsqueeze(0).to(device) / 255.0
    input_tensor.requires_grad_(requires_grad)
    return input_tensor, ratio, pad, resized


def map_boxes_to_letterbox(boxes_tensor, ratio, pad):
    if boxes_tensor.numel() == 0:
        return []
    ratio_w, ratio_h = ratio
    pad_w, pad_h = pad
    boxes = boxes_tensor.clone().float()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_w + pad_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_h + pad_h
    return boxes.tolist()


def normalize_class_name(name) -> str:
    return str(name).strip().lower()


def classes_match(name_a, name_b) -> bool:
    return normalize_class_name(name_a) == normalize_class_name(name_b)


def draw_predictions(image_chw, boxes, labels, scores):
    # image_chw: C,H,W uint8
    image = np.transpose(image_chw, (1, 2, 0)).copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            image,
            f"{label}:{float(score):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def has_fn_for_image(gt_boxes, gt_class_names, pred_boxes, pred_class_names, iou_match_threshold):
    matched_pred_indices = set()
    for gt_box, gt_name in zip(gt_boxes, gt_class_names):
        found_match = False
        for pred_idx, (pred_box, pred_name) in enumerate(zip(pred_boxes, pred_class_names)):
            if pred_idx in matched_pred_indices:
                continue
            if not classes_match(gt_name, pred_name):
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            return 1
    return 0


def get_fn_gt_indices(gt_boxes, gt_class_names, pred_boxes, pred_class_names, iou_match_threshold):
    matched_pred_indices = set()
    fn_gt_indices = []
    for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
        found_match = False
        for pred_idx, (pred_box, pred_name) in enumerate(zip(pred_boxes, pred_class_names)):
            if pred_idx in matched_pred_indices:
                continue
            if not classes_match(gt_name, pred_name):
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            fn_gt_indices.append(gt_idx)
    return fn_gt_indices


def assign_tp_to_predictions(
    gt_boxes,
    gt_class_names,
    pred_boxes,
    pred_class_names,
    pred_scores,
    iou_match_threshold,
):
    n_pred = len(pred_boxes)
    matched_gt_indices = set()
    tp_flags = [0] * n_pred
    best_ious = [0.0] * n_pred

    sorted_indices = list(range(n_pred))
    if pred_scores is not None:
        sorted_indices.sort(key=lambda i: float(pred_scores[i]), reverse=True)

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_name = pred_class_names[pred_idx]

        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
            if not classes_match(gt_name, pred_name):
                continue
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        best_ious[pred_idx] = float(best_iou)
        if best_gt_idx >= 0 and best_gt_idx not in matched_gt_indices and best_iou >= iou_match_threshold:
            tp_flags[pred_idx] = 1
            matched_gt_indices.add(best_gt_idx)

    return tp_flags, best_ious


def analyze_prediction_error_types(
    gt_boxes,
    gt_class_names,
    pred_boxes,
    pred_class_names,
    pred_scores,
    iou_match_threshold,
    background_iou_threshold=0.1,
):
    n_pred = len(pred_boxes)
    tp_flags = [0] * n_pred
    best_same_class_ious = [0.0] * n_pred
    best_any_class_ious = [0.0] * n_pred
    best_same_class_gt_indices = [-1] * n_pred
    best_any_class_gt_indices = [-1] * n_pred
    best_same_class_gt_classes = [""] * n_pred
    best_any_class_gt_classes = [""] * n_pred
    matched_gt_indices = [-1] * n_pred
    duplicate_flags = [0] * n_pred
    error_types = ["localization_error"] * n_pred

    sorted_indices = list(range(n_pred))
    if pred_scores is not None:
        sorted_indices.sort(key=lambda i: float(pred_scores[i]), reverse=True)

    claimed_gt_indices = set()
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_name = pred_class_names[pred_idx]

        best_same_iou = 0.0
        best_same_idx = -1
        best_any_iou = 0.0
        best_any_idx = -1
        for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_idx = gt_idx
            if classes_match(gt_name, pred_name) and iou > best_same_iou:
                best_same_iou = iou
                best_same_idx = gt_idx

        best_same_class_ious[pred_idx] = float(best_same_iou)
        best_any_class_ious[pred_idx] = float(best_any_iou)
        best_same_class_gt_indices[pred_idx] = int(best_same_idx)
        best_any_class_gt_indices[pred_idx] = int(best_any_idx)
        if best_same_idx >= 0:
            best_same_class_gt_classes[pred_idx] = str(gt_class_names[best_same_idx])
        if best_any_idx >= 0:
            best_any_class_gt_classes[pred_idx] = str(gt_class_names[best_any_idx])

        if best_same_idx >= 0 and best_same_iou >= iou_match_threshold:
            matched_gt_indices[pred_idx] = int(best_same_idx)
            if best_same_idx not in claimed_gt_indices:
                tp_flags[pred_idx] = 1
                error_types[pred_idx] = "tp"
                claimed_gt_indices.add(best_same_idx)
            else:
                duplicate_flags[pred_idx] = 1
                error_types[pred_idx] = "localization_error"
        elif best_any_idx >= 0 and best_any_iou >= iou_match_threshold:
            error_types[pred_idx] = "classification_error"
        elif best_same_idx >= 0 and best_same_iou >= background_iou_threshold:
            error_types[pred_idx] = "localization_error"
        else:
            error_types[pred_idx] = "localization_error"

    rows = []
    for pred_idx in range(n_pred):
        error_type = error_types[pred_idx]
        rows.append(
            {
                "tp": int(tp_flags[pred_idx]),
                "max_iou": float(best_same_class_ious[pred_idx]),
                "gt_iou": float(best_same_class_ious[pred_idx]),
                "error_type": error_type,
                "best_same_class_iou": float(best_same_class_ious[pred_idx]),
                "best_any_class_iou": float(best_any_class_ious[pred_idx]),
                "best_same_class_gt_idx": int(best_same_class_gt_indices[pred_idx]),
                "best_any_class_gt_idx": int(best_any_class_gt_indices[pred_idx]),
                "best_same_class_gt_class": best_same_class_gt_classes[pred_idx],
                "best_any_class_gt_class": best_any_class_gt_classes[pred_idx],
                "matched_gt_idx": int(matched_gt_indices[pred_idx]),
                "is_duplicate": int(duplicate_flags[pred_idx]),
                "is_localization_error": int(error_type == "localization_error"),
                "is_classification_error": int(error_type == "classification_error"),
            }
        )
    return rows


