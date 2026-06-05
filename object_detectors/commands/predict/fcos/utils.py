from dataclasses import dataclass

from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms, unpack_fcos_model_output


@dataclass
class FcosForwardNMSResult:
    infer_batch: object
    ratios: list
    pads: list
    resized_chws: object
    processed_images: list
    raw_prediction: list
    raw_logits: object
    raw_indices: object
    selected_preds: list
    selected_logits: list
    selected_indices: list
    pre_nms_prediction: object
    detector_inference_sec: float


@dataclass
class FcosDetectionRow:
    sample_idx: int
    pred_idx: int
    raw_pred_idx: int
    box: torch.Tensor
    score: float
    cls_idx: int
    base: dict


@dataclass
class FcosCandidateCache:
    prediction: torch.Tensor
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    score_threshold: float


def fcos_class_name(detector, cls_idx):
    cls_idx = int(cls_idx)
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def run_fcos_forward_nms(
    *,
    detector,
    image_list,
    device,
    timing,
    keep_pre_nms=False,
    keep_class_outputs=True,
    pre_nms_threshold=None,
    conf_thres=None,
):
    infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
    t_detector = timing.start()
    with torch.no_grad():
        processed_images = detector.preprocess_images(infer_batch)
        with detector.temporary_pre_nms_threshold(pre_nms_threshold):
            model_output = detector.forward_preprocessed(
                processed_images,
                keep_pre_nms=keep_pre_nms,
                keep_class_outputs=keep_class_outputs,
            )
        raw_prediction, raw_logits, raw_indices = unpack_fcos_model_output(model_output)
        pre_nms_prediction = None
        if keep_pre_nms:
            pre_nms_prediction, _pre_nms_logits, _pre_nms_indices = detector.get_last_pre_nms_predictions()
        selected = select_fcos_post_nms(
            detector,
            raw_prediction,
            raw_logits,
            raw_indices,
            conf_thres=conf_thres,
        )
        selected_preds = selected[0]
        selected_logits = selected[1]
        selected_indices = selected[3]
    detector_inference_sec = timing.elapsed(t_detector)
    return FcosForwardNMSResult(
        infer_batch=infer_batch,
        ratios=ratios,
        pads=pads,
        resized_chws=resized_chws,
        processed_images=processed_images,
        raw_prediction=raw_prediction,
        raw_logits=raw_logits,
        raw_indices=raw_indices,
        selected_preds=selected_preds,
        selected_logits=selected_logits,
        selected_indices=selected_indices,
        pre_nms_prediction=pre_nms_prediction,
        detector_inference_sec=detector_inference_sec,
    )


def ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx):
    det = (
        selected_preds[sample_idx]
        if selected_preds and sample_idx < len(selected_preds)
        else torch.zeros((0, 6))
    )
    raw_keep = (
        selected_indices[sample_idx]
        if selected_indices and sample_idx < len(selected_indices)
        else torch.zeros((0,), dtype=torch.long, device=det.device)
    )
    if int(raw_keep.shape[0]) < int(det.shape[0]):
        raise RuntimeError(
            f"FCOS selected_indices length mismatch: image_idx={sample_idx}, "
            f"num_detections={int(det.shape[0])}, num_indices={int(raw_keep.shape[0])}"
        )
    return raw_keep


def iter_fcos_detection_rows(detector, targets, selected_preds, selected_indices, device):
    for sample_idx in range(len(targets)):
        target = targets[sample_idx]
        image_id = int(target["image_id"][0].item())
        image_path = target["path"]
        det = (
            selected_preds[sample_idx]
            if selected_preds and sample_idx < len(selected_preds)
            else torch.zeros((0, 6), dtype=torch.float32, device=device)
        )
        raw_keep = ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx).to(device=device)
        for pred_idx in range(int(det.shape[0])):
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            if raw_pred_idx < 0:
                raise RuntimeError(
                    f"FCOS raw_pred_idx is negative: image_idx={sample_idx}, pred_idx={pred_idx}, raw_pred_idx={raw_pred_idx}"
                )
            box = det[pred_idx]
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            base = {
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "xmin": float(box[0].detach().cpu().item()),
                "ymin": float(box[1].detach().cpu().item()),
                "xmax": float(box[2].detach().cpu().item()),
                "ymax": float(box[3].detach().cpu().item()),
                "score": float(box[4].detach().cpu().item()),
                "pred_class": fcos_class_name(detector, cls_idx),
            }
            yield FcosDetectionRow(
                sample_idx=sample_idx,
                pred_idx=pred_idx,
                raw_pred_idx=raw_pred_idx,
                box=box,
                score=float(box[4].detach().cpu().item()),
                cls_idx=cls_idx,
                base=base,
            )


def selected_fcos_class_logits(result, sample_idx, num_classes, device):
    det = (
        result.selected_preds[sample_idx]
        if result.selected_preds and sample_idx < len(result.selected_preds)
        else torch.zeros((0, 6), dtype=torch.float32, device=device)
    )
    if result.selected_logits and sample_idx < len(result.selected_logits):
        logits = result.selected_logits[sample_idx]
        if logits is not None and int(logits.shape[0]) == int(det.shape[0]) and int(logits.shape[-1]) > 0:
            if int(logits.shape[-1]) == int(num_classes):
                return logits.to(device=device, dtype=torch.float32)
            if int(logits.shape[-1]) > int(num_classes):
                return logits[:, :num_classes].to(device=device, dtype=torch.float32)
            return torch.nn.functional.pad(
                logits.to(device=device, dtype=torch.float32),
                (0, int(num_classes) - int(logits.shape[-1])),
            )
    if int(det.shape[0]) == 0:
        return torch.zeros((0, int(num_classes)), dtype=torch.float32, device=device)
    raise RuntimeError(
        f"FCOS selected class logits are unavailable or misaligned: "
        f"image_idx={sample_idx}, num_detections={int(det.shape[0])}"
    )


def selected_fcos_class_probs(result, sample_idx, num_classes, device):
    return torch.sigmoid(selected_fcos_class_logits(result, sample_idx, num_classes, device))


def build_fcos_candidate_cache(candidate_prediction, score_threshold):
    if candidate_prediction is None or candidate_prediction.numel() == 0:
        device = candidate_prediction.device if isinstance(candidate_prediction, torch.Tensor) else torch.device("cpu")
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return FcosCandidateCache(
            prediction=torch.zeros((0, 6), dtype=torch.float32, device=device),
            boxes_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=empty,
            classes=torch.zeros((0,), dtype=torch.long, device=device),
            score_threshold=float(score_threshold),
        )
    pred = candidate_prediction.detach().float()
    boxes_xyxy = _xywh_to_xyxy_tensor(pred[:, :4])
    scores = pred[:, 4].detach().float()
    classes = pred[:, 5].detach().long() if pred.shape[1] > 5 else torch.zeros_like(scores, dtype=torch.long)
    return FcosCandidateCache(
        prediction=pred,
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        classes=classes,
        score_threshold=float(score_threshold),
    )


def fcos_candidate_mask_from_cache(cache, final_box_xyxy, final_cls, iou_threshold):
    final_cls = int(final_cls)
    candidate_count = int(cache.boxes_xyxy.shape[0])
    mask = torch.zeros((candidate_count,), dtype=torch.bool, device=cache.boxes_xyxy.device)
    ious = torch.zeros((candidate_count,), dtype=cache.boxes_xyxy.dtype, device=cache.boxes_xyxy.device)
    if candidate_count == 0:
        return mask, ious
    score_class_mask = (cache.scores >= float(cache.score_threshold)) & (cache.classes == final_cls)
    if bool(score_class_mask.any()):
        candidate_indices = torch.nonzero(score_class_mask, as_tuple=False).flatten()
        candidate_ious = _box_iou_1vN_tensor(final_box_xyxy.view(1, 4), cache.boxes_xyxy[candidate_indices])
        ious[candidate_indices] = candidate_ious
        mask[candidate_indices] = candidate_ious > float(iou_threshold)
    return mask, ious


__all__ = [
    "FcosForwardNMSResult",
    "FcosDetectionRow",
    "FcosCandidateCache",
    "run_fcos_forward_nms",
    "iter_fcos_detection_rows",
    "ensure_fcos_selected_indices",
    "selected_fcos_class_logits",
    "selected_fcos_class_probs",
    "build_fcos_candidate_cache",
    "fcos_candidate_mask_from_cache",
    "fcos_class_name",
]
