import json
from pathlib import Path

import cv2
import numpy as np
import torch

from models.yolo.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector


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

    detector = YOLOV5TorchObjectDetector(
        model_weight=model_cfg["weights"],
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
        self.activations = {"value": []}
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
        out = output[0] if isinstance(output, (tuple, list)) else output
        self.activations["value"].append(out)
        return None

    def _backward_hook(self, _module, _grad_input, grad_output):
        if grad_output is None or len(grad_output) == 0:
            return None
        grad = grad_output[0]
        if grad is not None:
            self.gradients["value"].append(grad)
        return None

    def clear(self):
        self.activations["value"] = []
        self.gradients["value"] = []

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
        cue = "grad"
        grad_cfg = {}
    else:
        save_csv_enabled = bool(save_csv_cfg.get("enabled", True))
        cue = str(save_csv_cfg.get("cue", "grad")).lower()
        grad_cfg = save_csv_cfg.get("grad", {})

    if cue != "grad":
        raise ValueError(f"Unsupported output.save_csv.cue='{cue}'. Only 'grad' is supported.")

    iou_match_threshold = float(grad_cfg.get("iou_match_threshold", 0.5))
    target_values = [v.lower() for v in normalize_to_list(grad_cfg.get("target_value", ["obj"]))]
    valid_values = {"obj", "cls"}
    invalid_values = [v for v in target_values if v not in valid_values]
    if invalid_values:
        raise ValueError(f"Unsupported target_value(s): {invalid_values}. Use {sorted(valid_values)}")

    target_layers = normalize_to_list(grad_cfg.get("target_layer", []))
    if not target_layers and save_csv_enabled:
        raise ValueError("output.save_csv.grad.target_layer must contain at least one layer name.")

    save_image_cfg = output_cfg.get("save_image", {})
    if isinstance(save_image_cfg, bool):
        save_image_enabled = save_image_cfg
        image_step = 1
        image_num = 1
    else:
        save_image_enabled = bool(save_image_cfg.get("enabled", False))
        image_step = int(save_image_cfg.get("step", 1))
        image_num = int(save_image_cfg.get("num", 1))

    if image_step <= 0:
        raise ValueError("output.save_image.step must be >= 1.")
    if image_num <= 0:
        raise ValueError("output.save_image.num must be >= 1.")

    return {
        "save_csv_enabled": save_csv_enabled,
        "cue": cue,
        "iou_match_threshold": iou_match_threshold,
        "target_values": target_values,
        "target_layers": target_layers,
        "save_image_enabled": save_image_enabled,
        "image_step": image_step,
        "image_num": image_num,
    }


def build_target_scalar(target_value, preds, logits, objectness):
    if target_value == "obj":
        if len(objectness) == 0 or objectness[0].numel() == 0:
            return None
        return objectness[0].sum()

    if target_value == "cls":
        if len(logits) == 0 or logits[0].numel() == 0:
            return None
        # Match DiL behavior: sum of max(class logit) over final detections.
        return torch.stack([torch.max(logit) for logit in logits[0]]).sum()

    raise ValueError(f"Unsupported target_value: {target_value}")


def collect_gradients_per_target(detector, input_tensor, target_values, target_layers, layer_buffer):
    grad_stats = {}
    for target_value in target_values:
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()

        grad_input = input_tensor.detach().requires_grad_(True)
        preds, logits, objectness, _features = detector(grad_input)
        target_scalar = build_target_scalar(target_value, preds, logits, objectness)

        if target_scalar is None:
            for layer_name in target_layers:
                grad_stats[f"d{target_value}_d{layer_name}"] = []
            if grad_input.grad is not None:
                grad_input.grad = None
            del grad_input, preds, logits, objectness, _features
            layer_buffer.clear()
            continue

        target_scalar.backward()
        grads = list(layer_buffer.gradients["value"])
        grads.reverse()

        for layer_idx, layer_name in enumerate(target_layers):
            key = f"d{target_value}_d{layer_name}"
            grad = grads[layer_idx] if layer_idx < len(grads) else None
            grad_stats[key] = [] if grad is None else get_channel_stats(grad.detach())

        if grad_input.grad is not None:
            grad_input.grad = None
        del grad_input, preds, logits, objectness, _features, target_scalar, grads
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()
    return grad_stats


def get_channel_stats(grad_tensor):
    # Expect [B, C, H, W] from conv feature maps; reduce over spatial dims per channel.
    grad_tensor = grad_tensor.detach().float()
    if grad_tensor.ndim == 4:
        grad_tensor = grad_tensor[0]
    if grad_tensor.ndim != 3:
        grad_tensor = grad_tensor.view(grad_tensor.shape[0], -1, 1)

    c = grad_tensor.shape[0]
    flat = grad_tensor.reshape(c, -1)

    l1 = flat.abs().sum(dim=1)
    l2 = torch.linalg.vector_norm(flat, ord=2, dim=1)
    min_v = flat.min(dim=1).values
    max_v = flat.max(dim=1).values
    mean_v = flat.mean(dim=1)
    std_v = flat.std(dim=1, unbiased=False)

    stacked = torch.stack([l1, l2, min_v, max_v, mean_v, std_v], dim=1)
    return stacked.detach().cpu().tolist()


def preprocess_with_letterbox(detector, image_tensor, device, requires_grad=True):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # Keep input shape fixed across images to reduce CUDA allocator fragmentation.
    resized, ratio, pad = detector.yolo_resize(image_np, new_shape=detector.img_size, auto=False)
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
