import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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


def get_dataset_cfg(config):
    dataset_root_cfg = config["dataset"]
    used_dataset = dataset_root_cfg["used_dataset"]
    if used_dataset.lower() != "coco":
        raise ValueError("Predict mode currently supports COCO only.")
    if used_dataset not in dataset_root_cfg:
        raise ValueError(f"dataset.{used_dataset} is missing in config.")
    return dataset_root_cfg[used_dataset]


def get_annotation_path(config, split):
    dataset_cfg = get_dataset_cfg(config)
    root = Path(dataset_cfg["root"])
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    ann_dir = dataset_cfg["annotation_dir"]
    ann_name = dataset_cfg[f"{split}_annotation_file"]
    return root / ann_dir / ann_name


def load_coco_category_maps(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(c["id"]): c["name"] for c in payload.get("categories", [])}


def build_detector(config):
    torch.backends.cudnn.benchmark = False
    model_cfg = config["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    confidence = model_cfg.get("confidence_threshold", 0.4)
    iou_thresh = model_cfg.get("iou_threshold", 0.45)

    weight_path = Path(model_cfg["weights"])
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


def parse_output_config(output_cfg):
    save_csv_cfg = output_cfg.get("save_csv", {})
    if isinstance(save_csv_cfg, bool):
        save_csv_enabled = save_csv_cfg
        uncertainty = "gt"
        gt_cfg = {}
        score_cfg = {}
        energy_cfg = {}
        entropy_cfg = {}
        full_softmax_cfg = {}
        feature_grad_cfg = {}
        layer_grad_cfg = {}
        unit = "image"
    else:
        save_csv_enabled = bool(save_csv_cfg.get("enabled", True))
        uncertainty = str(save_csv_cfg.get("uncertainty", "gt")).lower()
        gt_cfg = save_csv_cfg.get("gt", {})
        score_cfg = save_csv_cfg.get("score", {})
        energy_cfg = save_csv_cfg.get("energy", {})
        entropy_cfg = save_csv_cfg.get("entropy", {})
        full_softmax_cfg = save_csv_cfg.get("full_softmax", {})
        feature_grad_cfg = save_csv_cfg.get("feature_grad", {})
        layer_grad_cfg = save_csv_cfg.get("layer_grad", {})
        unit = str(save_csv_cfg.get("unit", "image")).lower()

    if uncertainty not in {"gt", "score", "energy", "entropy", "full_softmax", "feature_grad", "layer_grad"}:
        raise ValueError(
            f"Unsupported output.save_csv.uncertainty='{uncertainty}'. "
            "Use 'gt', 'score', 'energy', 'entropy', 'full_softmax', 'feature_grad' or 'layer_grad'."
        )

    gt_iou_match_threshold = float(gt_cfg.get("iou_match_threshold", 0.5))
    score_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    entropy_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    full_softmax_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    target_values = []
    target_layers = []
    feature_map_reduction = "energy"
    feature_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    layer_target_values = []
    layer_target_layers = []
    layer_vector_reduction = ["1-norm", "2-norm", "min", "max", "mean", "std"]
    if uncertainty == "feature_grad":
        if unit not in {"image", "bbox"}:
            raise ValueError("output.save_csv.unit must be 'image' or 'bbox' when uncertainty is 'feature_grad'.")
        target_values = [v.lower() for v in normalize_to_list(feature_grad_cfg.get("target_value", ["obj"]))]
        valid_values = {"obj", "cls", "loss", "obj_loss", "cls_loss", "bbox_loss"}
        invalid_values = [v for v in target_values if v not in valid_values]
        if invalid_values:
            raise ValueError(f"Unsupported target_value(s): {invalid_values}. Use {sorted(valid_values)}")
        if "loss" in target_values:
            expanded = []
            for v in target_values:
                if v == "loss":
                    expanded.extend(["obj_loss", "cls_loss", "bbox_loss"])
                else:
                    expanded.append(v)
            target_values = list(dict.fromkeys(expanded))

        target_layers = normalize_to_list(feature_grad_cfg.get("target_layer", []))
        if not target_layers and save_csv_enabled:
            raise ValueError("output.save_csv.feature_grad.target_layer must contain at least one layer name.")
        feature_map_reduction = str(feature_grad_cfg.get("map_reduction", "energy")).strip().lower()
        if feature_map_reduction != "energy":
            raise ValueError("output.save_csv.feature_grad.map_reduction currently supports only 'energy'.")
        feature_vector_reduction = normalize_vector_reduction(
            feature_grad_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"])
        )
    elif uncertainty == "layer_grad":
        if unit not in {"image", "bbox"}:
            msg = "Invalid config: output.save_csv.uncertainty='layer_grad' requires output.save_csv.unit in {'image','bbox'}."
            warnings.warn(msg)
            raise ValueError(msg)
        layer_target_values = [v.lower() for v in normalize_to_list(layer_grad_cfg.get("target_value", ["loss"]))]
        valid_values = {"loss", "obj_loss", "cls_loss", "bbox_loss", "obj", "cls"}
        invalid_values = [v for v in layer_target_values if v not in valid_values]
        if invalid_values:
            raise ValueError(f"Unsupported layer_grad target_value(s): {invalid_values}. Use {sorted(valid_values)}")
        # 'loss' means all pseudo-label loss components.
        if "loss" in layer_target_values:
            expanded = []
            for v in layer_target_values:
                if v == "loss":
                    expanded.extend(["obj_loss", "cls_loss", "bbox_loss"])
                else:
                    expanded.append(v)
            # keep order while removing duplicates
            layer_target_values = list(dict.fromkeys(expanded))
        layer_target_layers = normalize_to_list(layer_grad_cfg.get("target_layer", []))
        if not layer_target_layers and save_csv_enabled:
            raise ValueError("output.save_csv.layer_grad.target_layer must contain at least one layer name.")
        layer_vector_reduction = normalize_vector_reduction(
            layer_grad_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"])
        )
    elif uncertainty == "gt":
        if unit not in {"image", "bbox"}:
            msg = "Invalid config: output.save_csv.uncertainty='gt' requires output.save_csv.unit in {'image','bbox'}."
            warnings.warn(msg)
            raise ValueError(msg)
    elif uncertainty == "score":
        if unit not in {"image", "bbox"}:
            msg = "Invalid config: output.save_csv.uncertainty='score' requires output.save_csv.unit in {'image','bbox'}."
            warnings.warn(msg)
            raise ValueError(msg)
        score_vector_reduction = normalize_vector_reduction(
            score_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"])
        )
    elif uncertainty == "energy":
        if unit != "bbox":
            msg = "Invalid config: output.save_csv.uncertainty='energy' requires output.save_csv.unit='bbox'."
            warnings.warn(msg)
            raise ValueError(msg)
    elif uncertainty == "entropy":
        if unit not in {"image", "bbox"}:
            msg = "Invalid config: output.save_csv.uncertainty='entropy' requires output.save_csv.unit in {'image','bbox'}."
            warnings.warn(msg)
            raise ValueError(msg)
        entropy_vector_reduction = normalize_vector_reduction(
            entropy_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"])
        )
    elif uncertainty == "full_softmax":
        if unit not in {"image", "bbox"}:
            msg = "Invalid config: output.save_csv.uncertainty='full_softmax' requires output.save_csv.unit in {'image','bbox'}."
            warnings.warn(msg)
            raise ValueError(msg)
        full_softmax_vector_reduction = normalize_vector_reduction(
            full_softmax_cfg.get("vector_reduction", ["L1", "L2", "min", "max", "mean", "std"])
        )

    save_image_cfg = output_cfg.get("save_image", {})
    if isinstance(save_image_cfg, bool):
        save_image_enabled = save_image_cfg
        image_step = 1
        image_max_num = 1
    else:
        save_image_enabled = bool(save_image_cfg.get("enabled", False))
        image_step = int(save_image_cfg.get("step", 1))
        image_max_num = int(save_image_cfg.get("max_num", 1))

    if image_step <= 0:
        raise ValueError("output.save_image.step must be >= 1.")
    if image_max_num <= 0:
        raise ValueError("output.save_image.max_num must be >= 1.")

    return {
        "save_csv_enabled": save_csv_enabled,
        "uncertainty": uncertainty,
        "unit": unit,
        "gt_iou_match_threshold": gt_iou_match_threshold,
        "score_vector_reduction": score_vector_reduction,
        "entropy_vector_reduction": entropy_vector_reduction,
        "full_softmax_vector_reduction": full_softmax_vector_reduction,
        "target_values": target_values,
        "target_layers": target_layers,
        "feature_map_reduction": feature_map_reduction,
        "feature_vector_reduction": feature_vector_reduction,
        "layer_target_values": layer_target_values,
        "layer_target_layers": layer_target_layers,
        "layer_vector_reduction": layer_vector_reduction,
        "save_image_enabled": save_image_enabled,
        "image_step": image_step,
        "image_max_num": image_max_num,
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
def collect_gradients_per_target(detector, input_tensor, target_values, target_layers, layer_buffer):
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
            target_scalar = build_target_scalar_pre_nms(target_value, raw_prediction, raw_logits)
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
    return {
        "1-norm": float(torch.norm(v, p=1).detach().cpu().item()),
        "2-norm": float(torch.norm(v, p=2).detach().cpu().item()),
        "min": float(v.min().detach().cpu().item()),
        "max": float(v.max().detach().cpu().item()),
        "mean": float(torch.mean(v).detach().cpu().item()),
        "std": float(torch.std(v).detach().cpu().item()),
    }


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
            raise ValueError(
                "Unsupported output.save_csv.feature_grad.vector_reduction value: "
                f"'{item}'. Use L1, L2, min, max, mean, std."
            )
        key = alias[item]
        if key not in normalized:
            normalized.append(key)
    return normalized


def format_gradient_output(grad_tensor, vector_reduction):
    if grad_tensor is None:
        return []
    grad_tensor = grad_tensor.detach().float()
    if not vector_reduction:
        return grad_tensor.reshape(-1).detach().cpu().tolist()
    stats = map_grad_tensor_to_numbers(grad_tensor)
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


def build_layer_target_scalar_bbox(target_value, pred_img, logit_img, raw_idx, iou_threshold):
    if target_value == "obj":
        if raw_idx >= pred_img.shape[0]:
            return None
        return pred_img[raw_idx, 4]
    if target_value == "cls":
        if raw_idx >= logit_img.shape[0]:
            return None
        return logit_img[raw_idx].max()

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
    vector_reduction=None,
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

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
    try:
        for bbox_idx in range(num_boxes):
            raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
            grad_stats = {}
            for target_value in target_values:
                detector.zero_grad(set_to_none=True)

                model_output = detector.model(input_tensor.detach(), augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                pred_img = raw_prediction[0]
                logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]

                target_scalar = build_layer_target_scalar_bbox(
                    target_value=target_value,
                    pred_img=pred_img,
                    logit_img=logit_img,
                    raw_idx=raw_idx,
                    iou_threshold=iou_threshold,
                )

                if target_scalar is None:
                    for layer_name in target_layers:
                        grad_stats[f"{target_value}_{layer_name}"] = []
                    del model_output, raw_prediction, raw_logits, pred_img, logit_img
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
                    grad_stats[key] = format_gradient_output(grad_tensor, vector_reduction=vector_reduction)

                del model_output, raw_prediction, raw_logits, pred_img, logit_img, target_scalar, grads

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

    del selected_preds, selected_indices, det, raw_keep_indices
    return rows


def collect_image_layer_grads_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    vector_reduction=None,
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    with torch.no_grad():
        model_output = detector.model(input_tensor.detach(), augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
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
            pred_img = raw_prediction[0]
            logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]

            target_scalar = None
            if target_value in {"obj", "cls"}:
                target_scalar = build_layer_target_scalar_image(target_value, raw_prediction, raw_logits)
            else:
                loss_terms = []
                for bbox_idx in range(int(raw_keep_indices.shape[0])):
                    raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                    scalar = build_layer_target_scalar_bbox(
                        target_value=target_value,
                        pred_img=pred_img,
                        logit_img=logit_img,
                        raw_idx=raw_idx,
                        iou_threshold=iou_threshold,
                    )
                    if scalar is not None:
                        loss_terms.append(scalar)
                if loss_terms:
                    target_scalar = torch.stack(loss_terms).mean()

            if target_scalar is None:
                for layer_name in target_layers:
                    grad_stats[f"{target_value}_{layer_name}"] = zero_grad_numbers() if vector_reduction else []
                del model_output, raw_prediction, raw_logits, pred_img, logit_img
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
                    grad_stats[key] = format_gradient_output(grad_tensor, vector_reduction=vector_reduction)

            del model_output, raw_prediction, raw_logits, pred_img, logit_img, target_scalar, grads
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

    if str(map_reduction).lower() != "energy":
        raise ValueError("feature_grad.map_reduction currently supports only 'energy'.")

    if grad_tensor.numel() == 0:
        return zero_grad_numbers() if vector_reduction else []

    if grad_tensor.ndim == 0:
        vec = grad_tensor.abs().reshape(1)
    elif grad_tensor.ndim == 1:
        vec = grad_tensor.abs()
    else:
        c = grad_tensor.shape[0]
        flat = grad_tensor.reshape(c, -1)
        vec = flat.abs().mean(dim=1)

    if not vector_reduction:
        return vec.detach().cpu().tolist()

    stats = map_grad_tensor_to_numbers(vec)
    return {k: stats[k] for k in vector_reduction}


def preprocess_with_letterbox(detector, image_tensor, device, requires_grad=True):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # Match DiL preprocessing: use YOLO letterbox default behavior (auto=True),
    # which keeps aspect ratio and applies only stride-aligned padding.
    resized, ratio, pad = detector.yolo_resize(image_np, new_shape=detector.img_size)
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
            if gt_name != pred_name:
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            return 1
    return 0


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
            if gt_name != pred_name:
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


