from dataclasses import dataclass

import torch


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
            box = det[pred_idx]
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            yield {
                "sample_idx": sample_idx,
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
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


def source_point_box(source_points, raw_pred_idx, device):
    point = source_points[raw_pred_idx].to(device=device, dtype=torch.float32)
    return torch.stack([point[0], point[1], point[0], point[1]])
