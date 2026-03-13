import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

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


class LayerGradBuffer:
    def __init__(self, model, target_layers):
        self.target_layers = list(target_layers)
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
            # Store per-channel L1 energy vector, not raw tensors.
            self.gradients["value"].append(get_channel_stats(grad))
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


def create_layer_grad_buffer(model, target_layers):
    return LayerGradBuffer(model=model, target_layers=target_layers)


def parse_output_config(output_cfg):
    save_csv_cfg = output_cfg.get("save_csv", {})
    if isinstance(save_csv_cfg, bool):
        save_csv_enabled = save_csv_cfg
        cue = "fn"
        fn_cfg = {}
        tp_cfg = {}
        feature_grad_cfg = {}
        unit = "image"
    else:
        save_csv_enabled = bool(save_csv_cfg.get("enabled", True))
        cue = str(save_csv_cfg.get("cue", "fn")).lower()
        fn_cfg = save_csv_cfg.get("fn", {})
        tp_cfg = save_csv_cfg.get("tp", {})
        feature_grad_cfg = save_csv_cfg.get("feature_grad", {})
        unit = str(save_csv_cfg.get("unit", "image")).lower()

    if cue not in {"fn", "tp", "feature_grad"}:
        raise ValueError(f"Unsupported output.save_csv.cue='{cue}'. Use 'fn', 'tp' or 'feature_grad'.")

    iou_match_threshold = float(fn_cfg.get("iou_match_threshold", 0.5))
    tp_iou_match_threshold = float(tp_cfg.get("iou_match_threshold", 0.5))
    target_values = []
    target_layers = []
    if cue == "feature_grad":
        if unit not in {"image", "bbox"}:
            raise ValueError("output.save_csv.unit must be 'image' or 'bbox' when cue is 'feature_grad'.")
        target_values = [v.lower() for v in normalize_to_list(feature_grad_cfg.get("target_value", ["obj"]))]
        valid_values = {"obj", "cls"}
        invalid_values = [v for v in target_values if v not in valid_values]
        if invalid_values:
            raise ValueError(f"Unsupported target_value(s): {invalid_values}. Use {sorted(valid_values)}")

        target_layers = normalize_to_list(feature_grad_cfg.get("target_layer", []))
        if not target_layers and save_csv_enabled:
            raise ValueError("output.save_csv.feature_grad.target_layer must contain at least one layer name.")
    elif cue == "fn":
        if unit != "image":
            msg = "Invalid config: output.save_csv.cue='fn' requires output.save_csv.unit='image'."
            warnings.warn(msg)
            raise ValueError(msg)
    elif cue == "tp":
        if unit != "bbox":
            msg = "Invalid config: output.save_csv.cue='tp' requires output.save_csv.unit='bbox'."
            warnings.warn(msg)
            raise ValueError(msg)

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
        "cue": cue,
        "unit": unit,
        "iou_match_threshold": iou_match_threshold,
        "tp_iou_match_threshold": tp_iou_match_threshold,
        "target_values": target_values,
        "target_layers": target_layers,
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


def build_target_scalar_post_nms(target_value, bbox_idx, selected_logits, selected_objectness):
    if target_value == "obj":
        if selected_objectness is None or selected_objectness.numel() == 0:
            return None
        return selected_objectness[bbox_idx]

    if target_value == "cls":
        if selected_logits is None or selected_logits.numel() == 0:
            return None
        return selected_logits[bbox_idx].max()

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
        target_scalar = build_target_scalar_pre_nms(target_value, raw_prediction, raw_logits)

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
    detector.zero_grad(set_to_none=True)
    grad_input = input_tensor.detach().requires_grad_(True)
    model_output = detector.model(grad_input, augment=False)
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None

    selected_preds, selected_logits, selected_objectness = detector.non_max_suppression(
        raw_prediction,
        raw_logits,
        detector.confidence,
        detector.iou_thresh,
        classes=None,
        agnostic=detector.agnostic,
    )

    det = selected_preds[0] if selected_preds else torch.zeros((0, 6), device=grad_input.device)
    det_logits = selected_logits[0] if selected_logits else torch.zeros((0, 0), device=grad_input.device)
    det_objectness = selected_objectness[0] if selected_objectness else torch.zeros((0,), device=grad_input.device)

    rows = []
    num_boxes = int(det.shape[0])
    for bbox_idx in range(num_boxes):
        grad_stats = {}
        for target_value in target_values:
            detector.zero_grad(set_to_none=True)
            layer_buffer.clear()

            target_scalar = build_target_scalar_post_nms(
                target_value=target_value,
                bbox_idx=bbox_idx,
                selected_logits=det_logits,
                selected_objectness=det_objectness,
            )
            if target_scalar is None:
                for layer_name in target_layers:
                    grad_stats[f"{target_value}_{layer_name}"] = []
                layer_buffer.clear()
                continue

            target_scalar.backward(retain_graph=True)
            layer_stats = list(layer_buffer.gradients["value"])
            layer_stats.reverse()

            for layer_idx, layer_name in enumerate(target_layers):
                key = f"{target_value}_{layer_name}"
                grad_stats[key] = layer_stats[layer_idx] if layer_idx < len(layer_stats) else []

            layer_buffer.clear()

        cls_idx = int(det[bbox_idx, 5].detach().cpu().item())
        rows.append(
            {
                "pred_idx": bbox_idx,
                "xmin": float(det[bbox_idx, 0].detach().cpu().item()),
                "ymin": float(det[bbox_idx, 1].detach().cpu().item()),
                "xmax": float(det[bbox_idx, 2].detach().cpu().item()),
                "ymax": float(det[bbox_idx, 3].detach().cpu().item()),
                "score": float(det[bbox_idx, 4].detach().cpu().item()),
                "pred_class": detector.names[cls_idx] if detector.names is not None else cls_idx,
                "grad_stats": grad_stats,
            }
        )

    if grad_input.grad is not None:
        grad_input.grad = None
    del grad_input, model_output, raw_prediction, raw_logits, selected_preds, selected_logits, selected_objectness
    detector.zero_grad(set_to_none=True)
    layer_buffer.clear()
    return rows


def get_channel_stats(grad_tensor):
    # Expect [B, C, H, W] from conv feature maps and return per-channel
    # L1 energy: mean(abs(grad)) over spatial dimensions.
    grad_tensor = grad_tensor.detach().float()
    if grad_tensor.ndim == 4:
        grad_tensor = grad_tensor[0]

    # For tensor-like gradients, treat dim-0 as channel axis and average |grad|
    # over the remaining dimensions.
    if grad_tensor.ndim == 0:
        return [float(grad_tensor.abs().item())]
    if grad_tensor.ndim == 1:
        return grad_tensor.abs().detach().cpu().tolist()

    c = grad_tensor.shape[0]
    flat = grad_tensor.reshape(c, -1)
    l1_energy = flat.abs().mean(dim=1)
    return l1_energy.detach().cpu().tolist()


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
