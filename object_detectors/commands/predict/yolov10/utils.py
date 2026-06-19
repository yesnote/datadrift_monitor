from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn.functional as F


def parse_yolov10_output_config(config):
    from commands.utils.predict_utils import parse_output_config

    output = config.get("output", {}) if isinstance(config, dict) and "output" in config else config
    output = output if isinstance(output, dict) else {}
    uncertainty = str(output.get("uncertainty", "")).strip().lower()
    if uncertainty == "meta_detect":
        raise NotImplementedError("YOLOv10 does not support meta_detect because it is NMS-free.")
    layer_cfg = output.get("layer_grad", {}) if isinstance(output.get("layer_grad", {}), dict) else {}
    grad = layer_cfg.get("gradient", {}) if isinstance(layer_cfg.get("gradient", {}), dict) else {}
    if uncertainty == "layer_grad" or grad:
        target = str(grad.get("target", "null_target")).strip().lower()
        if target != "null_target":
            raise NotImplementedError("YOLOv10 layer_grad supports only target=null_target.")
        for forbidden in ("obj_loss", "obj_direction", "cand_score_threshold"):
            if forbidden in grad:
                raise ValueError(f"YOLOv10 layer_grad does not support gradient.{forbidden}.")
        bbox_loss = str(grad.get("bbox_loss", "l1")).strip().lower()
        if bbox_loss in {"box_l1", "box_l2", "offset_l1", "offset_l2"}:
            raise ValueError("YOLOv10 layer_grad bbox_loss supports l1/l2 only, not YOLOv5 box/offset losses.")
        if bbox_loss not in {"l1", "l2"}:
            raise ValueError("YOLOv10 layer_grad bbox_loss supports only l1 or l2.")
    null_cfg = output.get("null_detect", {}) if isinstance(output.get("null_detect", {}), dict) else {}
    for forbidden in ("obj_loss", "bbox_loss", "bbox_direction"):
        if forbidden in null_cfg:
            raise ValueError(f"YOLOv10 null_detect does not support {forbidden}.")
    return parse_output_config(output)


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


def run_yolov10_forward(detector, infer_batch, timing=None, grad=False):
    t_detector = timing.start() if timing is not None else None
    output = detector.forward_layer_grad(infer_batch) if grad else detector.forward_nms_free(infer_batch)
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
