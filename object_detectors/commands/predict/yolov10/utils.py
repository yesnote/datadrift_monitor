from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn.functional as F


def parse_yolov10_output_config(config):
    output = config.get("output", {}) if isinstance(config, dict) and "output" in config else config
    output = output if isinstance(output, dict) else {}
    uncertainty = str(output.get("uncertainty", "")).strip().lower()
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
        "null_detect",
        "layer_grad",
    }
    if uncertainty not in supported and uncertainty != "meta_detect":
        raise ValueError(f"Unsupported YOLOv10 uncertainty: {uncertainty}")
    if uncertainty == "meta_detect":
        raise NotImplementedError("YOLOv10 does not support meta_detect because it is NMS-free.")

    def as_dict(value):
        return value if isinstance(value, dict) else {}

    def as_bool_save(value):
        if isinstance(value, bool):
            return value
        value = as_dict(value)
        return bool(value.get("enabled", False))

    def as_int(value, default):
        try:
            return int(value)
        except Exception:
            return default

    def as_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    def normalize_loss(raw, default, supported_values, key):
        value = str(raw if raw is not None else default).strip().lower().replace("-", "_")
        aliases = {
            "bce": "bcewithlogits",
            "bce_with_logits": "bcewithlogits",
            "bcewithlogits": "bcewithlogits",
            "kl": "kl",
            "kl_div": "kl",
            "kl_divergence": "kl",
            "l1": "l1",
            "l1_loss": "l1",
            "l2": "l2",
            "l2_loss": "l2",
            "mse": "l2",
        }.get(value)
        if value in {"box_l1", "box_l2", "offset_l1", "offset_l2", "obj_loss", "obj"}:
            raise ValueError(f"YOLOv10 does not support YOLOv5-style {key}: {raw}")
        if value not in supported_values and aliases not in supported_values:
            raise ValueError(f"Unsupported YOLOv10 {key}: {raw}. Supported values: {', '.join(sorted(supported_values))}.")
        return aliases or value

    def normalize_direction(raw, default="pred_to_target"):
        value = str(raw if raw is not None else default).strip().lower().replace("-", "_")
        aliases = {
            "pred_to_target": "pred_to_target",
            "prediction_to_target": "pred_to_target",
            "target": "pred_to_target",
            "target_to_pred": "target_to_pred",
            "target_to_prediction": "target_to_pred",
            "reverse": "target_to_pred",
        }
        if value not in aliases:
            raise ValueError(f"Unsupported YOLOv10 direction: {raw}")
        return aliases[value]

    active = as_dict(output.get(uncertainty, {}))
    parsed = {
        "uncertainty": uncertainty,
        "unit": "bbox",
        "save_csv_enabled": as_bool_save(active.get("save_csv", False)),
        "save_image_enabled": as_bool_save(active.get("save_image", False)),
        "gt_iou_match_threshold": 0.5,
        "mc_num_runs": 30,
        "mc_dropout_rate": 0.5,
        "null_detect_cls_loss": "kl",
        "null_detect_cls_direction": "pred_to_target",
        "null_detect_feature_set": "full",
    }
    if uncertainty == "gt":
        parsed["gt_iou_match_threshold"] = as_float(active.get("iou_match_threshold", 0.5), 0.5)
    elif uncertainty == "mc_dropout":
        parsed["mc_num_runs"] = as_int(active.get("num_runs", 30), 30)
        parsed["mc_dropout_rate"] = as_float(active.get("dropout_rate", 0.5), 0.5)
    elif uncertainty == "null_detect":
        for forbidden in ("obj_loss", "bbox_loss", "bbox_direction"):
            if forbidden in active:
                raise ValueError(f"YOLOv10 null_detect does not support {forbidden}.")
        parsed["null_detect_cls_loss"] = normalize_loss(active.get("cls_loss", "kl"), "kl", {"bcewithlogits", "kl"}, "null_detect.cls_loss")
        parsed["null_detect_cls_direction"] = normalize_direction(active.get("cls_direction", "pred_to_target"))
        feature_set = str(active.get("feature_set", "full")).strip().lower()
        if feature_set not in {"full", "losses_only"}:
            raise ValueError("YOLOv10 null_detect.feature_set must be full or losses_only.")
        parsed["null_detect_feature_set"] = feature_set
    elif uncertainty == "layer_grad":
        layer_cfg = as_dict(output.get("layer_grad", {}))
        grad = as_dict(layer_cfg.get("gradient", {}))
        target = str(grad.get("target", "null_target")).strip().lower()
        if target != "null_target":
            raise NotImplementedError("YOLOv10 layer_grad supports only target=null_target.")
        for forbidden in ("obj_loss", "obj_direction", "cand_score_threshold"):
            if forbidden in grad:
                raise ValueError(f"YOLOv10 layer_grad does not support gradient.{forbidden}.")
        normalize_loss(grad.get("bbox_loss", "l1"), "l1", {"l1", "l2"}, "layer_grad.gradient.bbox_loss")
        normalize_loss(grad.get("cls_loss", "bcewithlogits"), "bcewithlogits", {"bcewithlogits", "kl"}, "layer_grad.gradient.cls_loss")
    return parsed


def split_yolov10_raw_pred_idx(raw_pred_idx, num_classes):
    raw_pred_idx = int(raw_pred_idx)
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("YOLOv10 num_classes must be positive.")
    return raw_pred_idx // num_classes, raw_pred_idx % num_classes


@dataclass
class YoloV10ForwardResult:
    model_output: object
    raw_levels: object
    decoded_prediction: torch.Tensor
    raw_logits: torch.Tensor
    selected_preds: list
    selected_logits: list
    selected_probs: list
    selected_indices: list
    source_points: torch.Tensor
    detector_inference_sec: float


def run_yolov10_forward(detector, infer_batch=None, timing=None, grad=False, feature_cache=None):
    t_detector = timing.start() if timing is not None else None
    output = detector.forward_layer_grad(infer_batch) if grad else detector.forward_nms_free(infer_batch, feature_cache=feature_cache)
    detector_inference_sec = timing.elapsed(t_detector) if timing is not None else 0.0
    return YoloV10ForwardResult(
        model_output=output["model_output"],
        raw_levels=output["raw_levels"],
        decoded_prediction=output["decoded_prediction"],
        raw_logits=output["raw_logits"],
        selected_preds=output["selected_preds"],
        selected_logits=output["selected_logits"],
        selected_probs=output["selected_probs"],
        selected_indices=output["selected_indices"],
        source_points=output["source_points"],
        detector_inference_sec=detector_inference_sec,
    )


def yolov10_class_name(detector, cls_idx):
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def iter_yolov10_detection_rows(detector, targets, selected_preds, selected_indices, device):
    num_classes = len(detector.names) if detector.names is not None else getattr(detector, "num_classes", 80)
    for sample_idx in range(len(targets)):
        target = targets[sample_idx]
        image_id = int(target["image_id"][0].item())
        image_path = target["path"]
        det = (
            selected_preds[sample_idx]
            if selected_preds and sample_idx < len(selected_preds)
            else torch.zeros((0, 6), dtype=torch.float32, device=device)
        )
        raw_keep = (
            selected_indices[sample_idx]
            if selected_indices and sample_idx < len(selected_indices)
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        for pred_idx in range(int(det.shape[0])):
            if pred_idx >= int(raw_keep.shape[0]):
                raise RuntimeError(
                    "YOLOv10 selected_indices is shorter than selected predictions. "
                    f"sample_idx={sample_idx}, pred_idx={pred_idx}, indices={int(raw_keep.shape[0])}"
                )
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            raw_box_idx, raw_class_idx = split_yolov10_raw_pred_idx(raw_pred_idx, num_classes)
            box = det[pred_idx]
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            if raw_class_idx != cls_idx:
                raise RuntimeError(
                    "YOLOv10 raw_pred_idx class component does not match selected prediction class. "
                    f"raw_pred_idx={raw_pred_idx}, raw_class_idx={raw_class_idx}, pred_class_idx={cls_idx}"
                )
            yield {
                "sample_idx": sample_idx,
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "raw_box_idx": raw_box_idx,
                "raw_class_idx": raw_class_idx,
                "box": box,
                "base_row": {
                    "image_id": image_id,
                    "image_path": image_path,
                    "pred_idx": pred_idx,
                    "raw_pred_idx": raw_pred_idx,
                    "xmin": float(box[0].detach().cpu().item()),
                    "ymin": float(box[1].detach().cpu().item()),
                    "xmax": float(box[2].detach().cpu().item()),
                    "ymax": float(box[3].detach().cpu().item()),
                    "score": float(box[4].detach().cpu().item()),
                    "pred_class": yolov10_class_name(detector, cls_idx),
                },
            }


def selected_yolov10_sigmoid_probs(forward, sample_idx, device):
    if forward.selected_probs and sample_idx < len(forward.selected_probs):
        return forward.selected_probs[sample_idx].to(device=device, dtype=torch.float32)
    return torch.zeros((0, 0), dtype=torch.float32, device=device)


def selected_yolov10_logits(forward, sample_idx, device):
    if forward.selected_logits and sample_idx < len(forward.selected_logits):
        return forward.selected_logits[sample_idx].to(device=device, dtype=torch.float32)
    return torch.zeros((0, 0), dtype=torch.float32, device=device)


def source_point_box(source_points, raw_box_idx, device):
    point = source_points[int(raw_box_idx)].to(device=device, dtype=torch.float32)
    return torch.stack([point[0], point[1], point[0], point[1]])


def yolov10_feature_tensor(forward):
    from models.yolov10.core import xywh2xyxy

    boxes = xywh2xyxy(forward.decoded_prediction[..., :4].detach().float())
    probs = torch.sigmoid(forward.raw_logits.detach().float())
    return torch.cat([boxes, probs], dim=-1)


def enable_forced_yolov10_dropout(model, dropout_rate):
    head = model.model[-1]
    original = head.forward_feat
    p = float(dropout_rate)

    def patched(self, x, cv2, cv3):
        outputs = original(x, cv2, cv3)
        return [F.dropout(out, p=p, training=True) for out in outputs]

    head.forward_feat = MethodType(patched, head)

    def restore():
        head.forward_feat = original

    return restore
