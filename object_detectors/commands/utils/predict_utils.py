import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.yolo.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def build_detector(config, model_weight=None):
    torch.backends.cudnn.benchmark = False
    model_cfg = config["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    confidence = model_cfg.get("confidence_threshold", 0.4)
    iou_thresh = model_cfg.get("iou_threshold", 0.45)

    weight_source = model_weight if model_weight is not None else model_cfg["weights"]
    if isinstance(weight_source, (list, tuple)):
        raise ValueError("build_detector expects a single weight path, got a list.")
    weight_path = Path(weight_source)
    if not weight_path.is_absolute():
        weight_path = (PROJECT_ROOT / weight_path).resolve()

    detector = YOLOV5TorchObjectDetector(
        model_weight=str(weight_path),
        device=device,
        img_size=(model_cfg["img_size"], model_cfg["img_size"]),
        names=None,
        mode="eval",
        confidence=confidence,
        iou_thresh=iou_thresh,
    )
    # We only need gradients wrt feature maps, not model weights.
    detector.model.requires_grad_(False)
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


class LayerGradBuffer:
    def __init__(self, model, target_layers, map_reduction="energy", vector_reduction=None):
        self.target_layers = list(target_layers)
        self.map_reduction = str(map_reduction).lower()
        self.vector_reduction = list(vector_reduction or [])
        self.gradients = {"value": []}
        self.forward_handles = []
        self.backward_handles = []

        for layer_name in self.target_layers:
            module = resolve_module_by_name(model, layer_name)
            self.forward_handles.append(module.register_forward_hook(self._forward_hook))
            if hasattr(module, "register_full_backward_hook"):
                self.backward_handles.append(module.register_full_backward_hook(self._backward_hook))
            else:
                self.backward_handles.append(module.register_backward_hook(self._backward_hook))

    def _forward_hook(self, _module, _inputs, output):
        # Keep DiL-style forward hook wiring, but do not retain feature tensors.
        return None

    def _backward_hook(self, _module, _grad_input, grad_output):
        if grad_output is None or len(grad_output) == 0:
            return None
        grad = grad_output[0]
        if grad is not None:
            self.gradients["value"].append(
                get_feature_grad_stats(
                    grad,
                    map_reduction=self.map_reduction,
                    vector_reduction=self.vector_reduction,
                )
            )
        return None

    def clear(self):
        self.gradients["value"].clear()

    def remove(self):
        self.clear()
        for handle in self.forward_handles:
            handle.remove()
        self.forward_handles = []
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles = []


def create_layer_grad_buffer(model, target_layers, map_reduction="energy", vector_reduction=None):
    return LayerGradBuffer(
        model=model,
        target_layers=target_layers,
        map_reduction=map_reduction,
        vector_reduction=vector_reduction,
    )


def expand_layer_names(model, layer_names):
    resolved = []
    for name in normalize_to_list(layer_names):
        token = str(name).strip()
        if not token:
            continue
        if token.lower() == "all_conv":
            for mod_name, mod in model.named_modules():
                if mod_name and isinstance(mod, nn.Conv2d) and mod_name not in resolved:
                    resolved.append(mod_name)
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

    def as_count_or_inf(v, d):
        if isinstance(v, str) and v.strip().lower() == "inf":
            return math.inf
        try:
            return int(v)
        except Exception:
            return d

    uncertainty = str(output_cfg.get("uncertainty", "gt")).lower()
    active = as_dict(output_cfg.get(uncertainty, {}))
    save_csv_cfg = active.get("save_csv", {})
    save_image_cfg = as_dict(active.get("save_image", {}))
    save_csv = as_dict(save_csv_cfg)
    layer_grad_data_cfg = as_dict(save_csv.get("data", {})) if uncertainty == "layer_grad" else save_csv

    save_csv_enabled = bool(layer_grad_data_cfg.get("enabled", bool(save_csv_cfg) if isinstance(save_csv_cfg, bool) else False))
    unit = str(layer_grad_data_cfg.get("unit", "image")).lower()
    pre_nms_cfg = as_dict(layer_grad_data_cfg.get("pre_nms", {}))
    pre_nms = bool(pre_nms_cfg.get("enabled", False))
    pre_nms_ratio = as_float(pre_nms_cfg.get("pre_nms_ratio", 1.0), 1.0)

    gt_cfg = save_csv if uncertainty == "gt" else {}
    score_cfg = save_csv if uncertainty == "score" else {}
    meta_detect_cfg = save_csv if uncertainty == "meta_detect" else {}
    mc_dropout_cfg = save_csv if uncertainty == "mc_dropout" else {}
    ensemble_cfg = save_csv if uncertainty == "ensemble" else {}
    energy_cfg = save_csv if uncertainty == "energy" else {}
    entropy_cfg = save_csv if uncertainty == "entropy" else {}
    full_softmax_cfg = save_csv if uncertainty == "full_softmax" else {}
    feature_cfg = save_csv if uncertainty == "feature" else {}
    feature_grad_cfg = save_csv if uncertainty == "feature_grad" else {}
    layer_grad_cfg = layer_grad_data_cfg if uncertainty == "layer_grad" else {}

    gt_iou_match_threshold = as_float(gt_cfg.get("iou_match_threshold", 0.5), 0.5)
    meta_detect_score_threshold = as_float(meta_detect_cfg.get("score_threshold", 0.0), 0.0)
    meta_detect_iou_threshold = as_float(meta_detect_cfg.get("iou_threshold", 0.45), 0.45)
    meta_detect_vector_reduction = normalize_vector_reduction(meta_detect_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    mc_num_runs = as_int(mc_dropout_cfg.get("num_runs", 30), 30)
    mc_dropout_rate = as_float(mc_dropout_cfg.get("dropout_rate", 0.5), 0.5)
    mc_queue_maxsize = as_int(mc_dropout_cfg.get("queue_maxsize", 8), 8)
    mc_vector_reduction = normalize_vector_reduction(mc_dropout_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    ensemble_vector_reduction = normalize_vector_reduction(ensemble_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    score_vector_reduction = normalize_vector_reduction(score_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    energy_vector_reduction = normalize_vector_reduction(energy_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    entropy_vector_reduction = normalize_vector_reduction(entropy_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))
    full_softmax_vector_reduction = normalize_vector_reduction(full_softmax_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))

    feature_target_layers = normalize_to_list(feature_cfg.get("target_layer", []))
    feature_map_reduction = str(feature_cfg.get("map_reduction", "energy")).strip().lower()
    feature_vector_reduction = normalize_vector_reduction(feature_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"]))

    target_values = []
    target_layers = []
    layer_target_values = []
    layer_target_layers = []
    layer_map_reduction = "none"
    layer_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    layer_pseudo_gt = "cand"

    if uncertainty == "feature_grad":
        g = as_dict(feature_grad_cfg.get("gradient", {}))
        r = as_dict(feature_grad_cfg.get("reduction", {}))
        target_values = [v.lower() for v in normalize_to_list(g.get("scalar", ["obj"]))]
        if "loss" in target_values:
            exp = []
            for v in target_values:
                exp.extend(["obj_loss", "cls_loss", "bbox_loss"] if v == "loss" else [v])
            target_values = list(dict.fromkeys(exp))
        target_layers = normalize_to_list(g.get("layer", []))
        feature_map_reduction = str(r.get("map", "energy")).strip().lower()
        feature_vector_reduction = normalize_vector_reduction(r.get("vector", ["L1", "L2", "min", "max", "mean", "std"]))
    elif uncertainty == "layer_grad":
        g = as_dict(layer_grad_cfg.get("gradient", {}))
        r = as_dict(layer_grad_cfg.get("reduction", {}))
        layer_target_values = [v.lower() for v in normalize_to_list(g.get("scalar", ["loss"]))]
        if "loss" in layer_target_values:
            exp = []
            for v in layer_target_values:
                exp.extend(["obj_loss", "cls_loss", "bbox_loss"] if v == "loss" else [v])
            layer_target_values = list(dict.fromkeys(exp))
        layer_target_layers = normalize_to_list(g.get("layer", []))
        layer_map_reduction = str(r.get("map", "none")).strip().lower()
        layer_vector_reduction = normalize_vector_reduction(r.get("vector", ["L1", "L2", "min", "max", "mean", "std"]))
        t = g.get("target", "cand_target")
        t_policy = str(t).strip().lower() if t is not None else "null_target"
        layer_pseudo_gt = "uniform" if t_policy in {"null_target", "null"} else "cand"

    save_image_enabled = bool(save_image_cfg.get("enabled", bool(save_image_cfg)))
    gt_image_cfg = as_dict(save_image_cfg.get("gt", {}))
    gt_image_step = as_int(gt_image_cfg.get("step", 1), 1)
    gt_image_max_num = as_int(gt_image_cfg.get("max_num", 1), 1)

    layer_grad_image_per_image_enabled = False
    layer_grad_image_per_image_step = 1
    layer_grad_image_per_image_max_num = 0
    layer_grad_image_ref_enabled = False
    layer_grad_image_ref_groups = ["fn", "non_fn"]
    layer_grad_image_save_final_raw_map = True
    layer_grad_image_save_final_norm_map = True
    layer_grad_image_save_profile = True
    layer_grad_image_gt_csv = ""
    layer_grad_image_num_fn = math.inf
    layer_grad_image_num_non_fn = math.inf
    layer_img_target_values = ["obj_loss", "cls_loss", "bbox_loss"]
    layer_img_target_layers = []
    layer_grad_image_pseudo_gt = "cand"
    layer_grad_image_delta_metric = "l2"
    layer_grad_image_delta_l2_tol = 1e-4
    layer_grad_image_patience = 20
    layer_grad_image_min_samples = 200
    layer_grad_image_max_samples = 20000
    layer_grad_ref_save_running_log = True
    layer_grad_ref_save_final_raw_map_csv = True
    layer_grad_ref_save_final_norm_map_csv = True

    if uncertainty == "layer_grad":
        lg = save_image_cfg
        per_image_cfg = as_dict(lg.get("per_image", {}))
        ref_img_cfg = as_dict(lg.get("reference", {}))
        conv_cfg = ref_img_cfg
        layer_grad_image_per_image_enabled = bool(per_image_cfg.get("enabled", False))
        layer_grad_image_per_image_step = as_int(per_image_cfg.get("step", 1), 1)
        layer_grad_image_per_image_max_num = as_int(per_image_cfg.get("max_num", 0), 0)
        layer_grad_image_ref_enabled = bool(ref_img_cfg.get("enabled", False))
        layer_grad_image_ref_groups = [g.lower() for g in normalize_to_list(ref_img_cfg.get("group", ["fn", "non_fn"]))]
        layer_grad_image_save_final_raw_map = bool(ref_img_cfg.get("save_final_raw_map", True))
        layer_grad_image_save_final_norm_map = bool(ref_img_cfg.get("save_final_norm_map", True))
        layer_grad_image_save_profile = bool(ref_img_cfg.get("save_profile", True))

        gt_dir = str(lg.get("gt", "")).strip()
        if gt_dir:
            layer_grad_image_gt_csv = str((Path(gt_dir) / "fn.csv").as_posix())

        g = as_dict(lg.get("gradient", {}))
        layer_img_target_values = [v.lower() for v in normalize_to_list(g.get("scalar", ["loss"]))]
        if "loss" in layer_img_target_values:
            exp = []
            for v in layer_img_target_values:
                exp.extend(["obj_loss", "cls_loss", "bbox_loss"] if v == "loss" else [v])
            layer_img_target_values = list(dict.fromkeys(exp))
        layer_img_target_layers = normalize_to_list(g.get("layer", []))
        t = g.get("target", "cand_target")
        t_policy = str(t).strip().lower() if t is not None else "null_target"
        layer_grad_image_pseudo_gt = "uniform" if t_policy in {"null_target", "null"} else "cand"

        layer_grad_image_delta_l2_tol = as_float(conv_cfg.get("delta", 1e-4), 1e-4)
        layer_grad_image_patience = as_int(conv_cfg.get("patience", 20), 20)
        layer_grad_image_min_samples = as_int(conv_cfg.get("min_samples", 200), 200)
        layer_grad_image_max_samples = as_int(conv_cfg.get("max_samples", 20000), 20000)

        ref_csv_cfg = as_dict(save_csv.get("reference", {}))
        layer_grad_ref_save_running_log = bool(ref_csv_cfg.get("save_running_log", True))
        layer_grad_ref_save_final_raw_map_csv = bool(ref_csv_cfg.get("save_final_raw_map_csv", True))
        layer_grad_ref_save_final_norm_map_csv = bool(ref_csv_cfg.get("save_final_norm_map_csv", True))

    return {
        "save_csv_enabled": save_csv_enabled,
        "uncertainty": uncertainty,
        "pre_nms": pre_nms,
        "pre_nms_ratio": float(pre_nms_ratio),
        "unit": unit,
        "gt_iou_match_threshold": gt_iou_match_threshold,
        "meta_detect_score_threshold": meta_detect_score_threshold,
        "meta_detect_iou_threshold": meta_detect_iou_threshold,
        "meta_detect_vector_reduction": meta_detect_vector_reduction,
        "mc_num_runs": mc_num_runs,
        "mc_dropout_rate": mc_dropout_rate,
        "mc_queue_maxsize": mc_queue_maxsize,
        "mc_vector_reduction": mc_vector_reduction,
        "ensemble_vector_reduction": ensemble_vector_reduction,
        "score_vector_reduction": score_vector_reduction,
        "energy_vector_reduction": energy_vector_reduction,
        "entropy_vector_reduction": entropy_vector_reduction,
        "full_softmax_vector_reduction": full_softmax_vector_reduction,
        "feature_target_layers": feature_target_layers,
        "feature_map_reduction": feature_map_reduction,
        "feature_vector_reduction": feature_vector_reduction,
        "target_values": target_values,
        "target_layers": target_layers,
        "layer_target_values": layer_target_values,
        "layer_target_layers": layer_target_layers,
        "layer_map_reduction": layer_map_reduction,
        "layer_vector_reduction": layer_vector_reduction,
        "layer_pseudo_gt": layer_pseudo_gt,
        "save_image_enabled": save_image_enabled,
        "save_image_gt_step": gt_image_step,
        "save_image_gt_max_num": gt_image_max_num,
        "save_image_layer_grad_target_values": layer_img_target_values,
        "save_image_layer_grad_target_layers": layer_img_target_layers,
        "save_image_layer_grad_pseudo_gt": layer_grad_image_pseudo_gt,
        "save_image_layer_grad_num_fn": layer_grad_image_num_fn,
        "save_image_layer_grad_num_non_fn": layer_grad_image_num_non_fn,
        "save_image_layer_grad_gt_csv": layer_grad_image_gt_csv,
        "save_image_layer_grad_per_image_enabled": layer_grad_image_per_image_enabled,
        "save_image_layer_grad_per_image_step": layer_grad_image_per_image_step,
        "save_image_layer_grad_per_image_max_num": layer_grad_image_per_image_max_num,
        "save_image_layer_grad_reference_enabled": layer_grad_image_ref_enabled,
        "save_image_layer_grad_reference_groups": layer_grad_image_ref_groups,
        "save_image_layer_grad_convergence_delta_metric": layer_grad_image_delta_metric,
        "save_image_layer_grad_convergence_delta_l2_tol": layer_grad_image_delta_l2_tol,
        "save_image_layer_grad_convergence_patience": layer_grad_image_patience,
        "save_image_layer_grad_convergence_min_samples": layer_grad_image_min_samples,
        "save_image_layer_grad_convergence_max_samples": layer_grad_image_max_samples,
        "save_image_layer_grad_save_final_raw_map": layer_grad_image_save_final_raw_map,
        "save_image_layer_grad_save_final_norm_map": layer_grad_image_save_final_norm_map,
        "save_image_layer_grad_save_profile": layer_grad_image_save_profile,
        "save_image_layer_grad_csv_save_running_log": layer_grad_ref_save_running_log,
        "save_image_layer_grad_csv_save_final_raw_map_csv": layer_grad_ref_save_final_raw_map_csv,
        "save_image_layer_grad_csv_save_final_norm_map_csv": layer_grad_ref_save_final_norm_map_csv,
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
            # unit=image + *loss: mean over final NMS predictions
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
def collect_bbox_gradients_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    layer_buffer,
):
    # Step 1) compute final NMS boxes and their raw indices without building autograd graph.
    with torch.no_grad():
        model_output = detector.model(input_tensor.detach(), augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            raw_prediction,
            raw_logits,
            detector.confidence,
            detector.iou_thresh,
            classes=None,
            agnostic=detector.agnostic,
            return_indices=True,
        )

    det = selected_preds[0] if selected_preds else torch.zeros((0, 6), device=input_tensor.device)
    raw_keep_indices = selected_indices[0] if selected_indices else torch.zeros((0,), dtype=torch.long, device=input_tensor.device)

    rows = []
    num_boxes = int(det.shape[0])
    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    for bbox_idx in range(num_boxes):
        raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
        grad_stats = {}
        for target_value in target_values:
            detector.zero_grad(set_to_none=True)
            layer_buffer.clear()

            # Step 2) re-run forward and build scalar only from selected raw prediction index.
            grad_input = input_tensor.detach().requires_grad_(True)
            model_output = detector.model(grad_input, augment=False)
            raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
            pred_img = raw_prediction[0]
            logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]

            target_scalar = None
            if raw_idx < pred_img.shape[0]:
                if target_value == "obj":
                    target_scalar = pred_img[raw_idx, 4]
                elif target_value == "cls":
                    target_scalar = logit_img[raw_idx].max()
                else:
                    losses = build_pseudo_label_losses_for_candidates(
                        pred_img=pred_img,
                        raw_idx=raw_idx,
                        iou_threshold=iou_threshold,
                    )
                    if losses is not None and target_value in losses:
                        target_scalar = losses[target_value]

            if target_scalar is None:
                for layer_name in target_layers:
                    grad_stats[f"{target_value}_{layer_name}"] = []
                if grad_input.grad is not None:
                    grad_input.grad = None
                del grad_input, model_output, raw_prediction, raw_logits, pred_img, logit_img
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
            del grad_input, model_output, raw_prediction, raw_logits, pred_img, logit_img, target_scalar
            layer_buffer.clear()

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

    del selected_preds, selected_indices, det, raw_keep_indices
    detector.zero_grad(set_to_none=True)
    layer_buffer.clear()
    return rows


def map_grad_tensor_to_numbers(v):
    if v is None:
        return {"1-norm": 0.0, "2-norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    v = v.detach().float().reshape(-1)
    if v.numel() == 0:
        return {"1-norm": 0.0, "2-norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "1-norm": float(torch.norm(v, p=1).detach().cpu().item()),
        "2-norm": float(torch.norm(v, p=2).detach().cpu().item()),
        "min": float(v.min().detach().cpu().item()),
        "max": float(v.max().detach().cpu().item()),
        "mean": float(torch.mean(v).detach().cpu().item()),
        "std": float(torch.std(v, unbiased=False).detach().cpu().item()),
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


def normalize_vector_reduction(value):
    items = [v.strip().lower() for v in normalize_to_list(value)]
    if not items:
        return []

    alias = {
        "l1": "1-norm",
        "1": "1-norm",
        "1-norm": "1-norm",
        "l2": "2-norm",
        "2": "2-norm",
        "2-norm": "2-norm",
        "min": "min",
        "max": "max",
        "mean": "mean",
        "std": "std",
    }
    normalized = []
    for item in items:
        if item not in alias:
            continue
        key = alias[item]
        if key not in normalized:
            normalized.append(key)
    return normalized


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
        return reduced.detach().cpu().tolist()
    stats = map_grad_tensor_to_numbers(reduced)
    return {k: stats[k] for k in vector_reduction}


def zero_grad_numbers():
    return {
        "1-norm": 0.0,
        "2-norm": 0.0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }


def collect_image_features_per_layer(
    detector,
    input_tensor,
    target_layers,
    map_reduction="energy",
    vector_reduction=None,
):
    collected = {layer_name: None for layer_name in target_layers}
    handles = []
    try:
        for layer_name in target_layers:
            module = resolve_module_by_name(detector.model, layer_name)

            def _hook(_module, _inputs, output, _layer_name=layer_name):
                out = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(out, torch.Tensor):
                    collected[_layer_name] = get_feature_grad_stats(
                        out,
                        map_reduction=map_reduction,
                        vector_reduction=vector_reduction,
                    )

            handles.append(module.register_forward_hook(_hook))

        with torch.no_grad():
            detector.model(input_tensor, augment=False)
    finally:
        for h in handles:
            h.remove()

    out_dict = {}
    for layer_name in target_layers:
        value = collected.get(layer_name, None)
        if value is None:
            out_dict[layer_name] = zero_grad_numbers() if vector_reduction else []
        else:
            out_dict[layer_name] = value
    return out_dict


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


def _box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def build_pseudo_label_losses_for_candidates(
    pred_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
):
    if raw_idx >= pred_img.shape[0]:
        return None

    eps = 1e-6
    with torch.no_grad():
        pseudo_row = pred_img[raw_idx].detach()
        pseudo_cls = int(torch.argmax(pseudo_row[5:]).item())
        pred_boxes_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4].detach())
        pseudo_box_xyxy = _xywh_to_xyxy_tensor(pseudo_row[:4].view(1, 4))
        ious = _box_iou_tensor(pred_boxes_xyxy, pseudo_box_xyxy).squeeze(1)
        pred_cls = torch.argmax(pred_img[:, 5:].detach(), dim=1)
        candidate_mask = (ious >= float(iou_threshold)) & (pred_cls == pseudo_cls)
        if not bool(candidate_mask.any()):
            candidate_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
            candidate_mask[raw_idx] = True

    candidate_pred = pred_img[candidate_mask]
    pseudo_box_target = pseudo_row[:4].view(1, 4).expand(candidate_pred.shape[0], -1)
    bbox_loss = F.smooth_l1_loss(candidate_pred[:, :4], pseudo_box_target, reduction="sum")

    obj_prob = candidate_pred[:, 4].clamp(eps, 1.0 - eps)
    obj_target = torch.ones_like(obj_prob)
    obj_loss = F.binary_cross_entropy(obj_prob, obj_target, reduction="sum")

    cls_prob = candidate_pred[:, 5:].clamp(eps, 1.0 - eps)
    cls_target = torch.zeros_like(cls_prob)
    cls_target[:, pseudo_cls] = 1.0
    cls_loss = F.binary_cross_entropy(cls_prob, cls_target, reduction="sum")

    return {
        "bbox_loss": bbox_loss,
        "obj_loss": obj_loss,
        "cls_loss": cls_loss,
        "loss": bbox_loss + obj_loss + cls_loss,
    }


def build_layer_target_scalar_image(target_value, raw_prediction, raw_logits):
    if raw_prediction is None or raw_prediction.numel() == 0:
        return None
    if target_value == "obj":
        return raw_prediction[..., 4].sum()
    if target_value == "cls":
        logits = raw_logits if raw_logits is not None else raw_prediction[..., 5:]
        if logits is None or logits.numel() == 0:
            return None
        return logits.max(dim=-1).values.sum()
    return None


def build_layer_target_scalar_bbox(
    target_value,
    pred_img,
    logit_img,
    raw_idx,
    iou_threshold,
    pseudo_gt="cand",
    anchor_xywh=None,
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
        eps = 1e-6
        pred_row = pred_img[raw_idx]
        if target_value == "obj_loss":
            obj_prob = pred_row[4].clamp(eps, 1.0 - eps)
            obj_target = torch.full_like(obj_prob, 0.5)
            return F.binary_cross_entropy(obj_prob, obj_target)
        if target_value == "cls_loss":
            cls_logits = logit_img[raw_idx] if (logit_img is not None and raw_idx < logit_img.shape[0]) else pred_row[5:]
            if cls_logits.numel() == 0:
                return None
            log_probs = F.log_softmax(cls_logits, dim=-1)
            uniform_target = torch.full_like(log_probs, 1.0 / float(log_probs.numel()))
            # Soft-target cross-entropy with uniform pseudo GT.
            return -(uniform_target * log_probs).sum()
        if target_value == "bbox_loss":
            if anchor_xywh is None:
                return None
            pred_xywh = pred_row[:4]
            anchor_xywh = anchor_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device)
            return F.smooth_l1_loss(pred_xywh, anchor_xywh, reduction="mean")
        return None

    losses = build_pseudo_label_losses_for_candidates(
        pred_img=pred_img,
        raw_idx=raw_idx,
        iou_threshold=iou_threshold,
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
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    with torch.no_grad():
        model_output = detector.model(input_tensor.detach(), augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
        raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            raw_prediction,
            raw_logits,
            detector.confidence,
            detector.iou_thresh,
            classes=None,
            agnostic=detector.agnostic,
            return_indices=True,
        )

    det = selected_preds[0] if selected_preds else torch.zeros((0, 6), device=input_tensor.device)
    raw_keep_indices = selected_indices[0] if selected_indices else torch.zeros((0,), dtype=torch.long, device=input_tensor.device)

    rows = []
    num_boxes = int(det.shape[0])
    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    try:
        for bbox_idx in range(num_boxes):
            raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
            grad_stats = {}
            for target_value in target_values:
                detector.zero_grad(set_to_none=True)

                model_output = detector.model(input_tensor.detach(), augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
                pred_img = raw_prediction[0]
                logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]
                anchor_img = raw_anchor_priors[0] if raw_anchor_priors is not None else None
                anchor_row = anchor_img[raw_idx] if (anchor_img is not None and raw_idx < anchor_img.shape[0]) else None

                target_scalar = build_layer_target_scalar_bbox(
                    target_value=target_value,
                    pred_img=pred_img,
                    logit_img=logit_img,
                    raw_idx=raw_idx,
                    iou_threshold=iou_threshold,
                    pseudo_gt=pseudo_gt,
                    anchor_xywh=anchor_row,
                )

                if target_scalar is None:
                    for layer_name in target_layers:
                        grad_stats[f"{target_value}_{layer_name}"] = []
                    del model_output, raw_prediction, raw_logits, raw_anchor_priors, pred_img, logit_img, anchor_img, anchor_row
                    continue

                grads = torch.autograd.grad(
                    target_scalar,
                    layer_params,
                    retain_graph=False,
                    allow_unused=True,
                )
                for layer_idx, layer_name in enumerate(target_layers):
                    key = f"{target_value}_{layer_name}"
                    grad_tensor = grads[layer_idx]
                    grad_stats[key] = format_gradient_output(
                        grad_tensor,
                        vector_reduction=vector_reduction,
                        map_reduction=map_reduction,
                    )

                del model_output, raw_prediction, raw_logits, raw_anchor_priors, pred_img, logit_img, anchor_img, anchor_row, target_scalar, grads

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

    del selected_preds, selected_indices, det, raw_keep_indices, raw_anchor_priors
    return rows


def collect_image_layer_grads_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    map_reduction="none",
    vector_reduction=None,
    pre_nms=True,
    pre_nms_ratio=1.0,
    pseudo_gt="cand",
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    with torch.no_grad():
        model_output = detector.model(input_tensor.detach(), augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
        raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
        _selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            raw_prediction,
            raw_logits,
            detector.confidence,
            detector.iou_thresh,
            classes=None,
            agnostic=detector.agnostic,
            return_indices=True,
        )
        raw_keep_indices = selected_indices[0] if selected_indices else torch.zeros((0,), dtype=torch.long, device=input_tensor.device)

    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    grad_stats = {}
    try:
        for target_value in target_values:
            detector.zero_grad(set_to_none=True)
            model_output = detector.model(input_tensor.detach(), augment=False)
            raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
            raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
            pred_img = raw_prediction[0]
            logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]
            anchor_img = raw_anchor_priors[0] if raw_anchor_priors is not None else None

            target_scalar = None
            if target_value in {"obj", "cls"}:
                if pre_nms:
                    keep_idx = get_pre_nms_keep_indices(pred_img, logit_img, pre_nms_ratio=pre_nms_ratio)
                    if int(keep_idx.shape[0]) > 0:
                        if target_value == "obj":
                            target_scalar = pred_img[keep_idx, 4].sum()
                        else:
                            if logit_img is not None and logit_img.numel() > 0:
                                target_scalar = logit_img[keep_idx].max(dim=1).values.sum()
                else:
                    if int(raw_keep_indices.shape[0]) > 0:
                        if target_value == "obj":
                            target_scalar = pred_img[raw_keep_indices, 4].sum()
                        else:
                            if logit_img is not None and logit_img.numel() > 0:
                                target_scalar = logit_img[raw_keep_indices].max(dim=1).values.sum()
            else:
                loss_terms = []
                if pseudo_gt == "uniform":
                    if pre_nms:
                        idx_tensor = get_pre_nms_keep_indices(pred_img, logit_img, pre_nms_ratio=pre_nms_ratio)
                    else:
                        idx_tensor = raw_keep_indices
                else:
                    idx_tensor = raw_keep_indices

                for bbox_idx in range(int(idx_tensor.shape[0])):
                    raw_idx = int(idx_tensor[bbox_idx].detach().cpu().item())
                    anchor_row = anchor_img[raw_idx] if (anchor_img is not None and raw_idx < anchor_img.shape[0]) else None
                    scalar = build_layer_target_scalar_bbox(
                        target_value=target_value,
                        pred_img=pred_img,
                        logit_img=logit_img,
                        raw_idx=raw_idx,
                        iou_threshold=iou_threshold,
                        pseudo_gt=pseudo_gt,
                        anchor_xywh=anchor_row,
                    )
                    if scalar is not None:
                        loss_terms.append(scalar)
                if loss_terms:
                    target_scalar = torch.stack(loss_terms).mean()

            if target_scalar is None:
                for layer_name in target_layers:
                    grad_stats[f"{target_value}_{layer_name}"] = zero_grad_numbers() if vector_reduction else []
                del model_output, raw_prediction, raw_logits, raw_anchor_priors, pred_img, logit_img, anchor_img
                continue

            grads = torch.autograd.grad(
                target_scalar,
                layer_params,
                retain_graph=False,
                allow_unused=True,
            )
            for layer_idx, layer_name in enumerate(target_layers):
                key = f"{target_value}_{layer_name}"
                grad_tensor = grads[layer_idx]
                if grad_tensor is None:
                    grad_stats[key] = zero_grad_numbers() if vector_reduction else []
                else:
                    grad_stats[key] = format_gradient_output(
                        grad_tensor,
                        vector_reduction=vector_reduction,
                        map_reduction=map_reduction,
                    )

            del model_output, raw_prediction, raw_logits, raw_anchor_priors, pred_img, logit_img, anchor_img, target_scalar, grads
    finally:
        for param, req_grad in zip(layer_params, original_requires_grad):
            param.requires_grad_(req_grad)
        detector.zero_grad(set_to_none=True)

    return grad_stats


def get_feature_grad_stats(grad_tensor, map_reduction="energy", vector_reduction=None):
    # Expect [B, C, H, W] from conv feature maps.
    grad_tensor = grad_tensor.detach().float()
    if grad_tensor.ndim >= 1 and grad_tensor.shape[0] == 1:
        grad_tensor = grad_tensor[0]

    map_mode = str(map_reduction).lower()
    if map_mode not in {"none", "energy"}:
        raise ValueError("feature_grad.map_reduction must be 'none' or 'energy'.")

    if grad_tensor.numel() == 0:
        return zero_grad_numbers() if vector_reduction else []

    if map_mode == "energy":
        if grad_tensor.ndim == 0:
            vec = grad_tensor.abs().reshape(1)
        elif grad_tensor.ndim == 1:
            vec = grad_tensor.abs()
        else:
            c = grad_tensor.shape[0]
            flat = grad_tensor.reshape(c, -1)
            vec = flat.abs().mean(dim=1)
    else:
        vec = grad_tensor.reshape(-1)

    if not vector_reduction:
        return vec.detach().cpu().tolist()

    stats = map_grad_tensor_to_numbers(vec)
    return {k: stats[k] for k in vector_reduction}


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
            if gt_idx in matched_gt_indices:
                continue
            if not classes_match(gt_name, pred_name):
                continue
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        best_ious[pred_idx] = float(best_iou)
        if best_gt_idx >= 0 and best_iou >= iou_match_threshold:
            tp_flags[pred_idx] = 1
            matched_gt_indices.add(best_gt_idx)

    return tp_flags, best_ious


