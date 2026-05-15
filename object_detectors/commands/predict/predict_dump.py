import torch.nn.functional as F

from commands.predict.common import *


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _bbox_ciou_xywh(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if pred_xywh.ndim == 1:
        pred_xywh = pred_xywh.view(1, 4)
    if target_xywh.ndim == 1:
        target_xywh = target_xywh.view(1, 4)
    if target_xywh.shape[0] == 1 and pred_xywh.shape[0] > 1:
        target_xywh = target_xywh.expand(pred_xywh.shape[0], -1)

    b1_x1 = pred_xywh[:, 0] - pred_xywh[:, 2] / 2.0
    b1_y1 = pred_xywh[:, 1] - pred_xywh[:, 3] / 2.0
    b1_x2 = pred_xywh[:, 0] + pred_xywh[:, 2] / 2.0
    b1_y2 = pred_xywh[:, 1] + pred_xywh[:, 3] / 2.0
    b2_x1 = target_xywh[:, 0] - target_xywh[:, 2] / 2.0
    b2_y1 = target_xywh[:, 1] - target_xywh[:, 3] / 2.0
    b2_x2 = target_xywh[:, 0] + target_xywh[:, 2] / 2.0
    b2_y2 = target_xywh[:, 1] + target_xywh[:, 3] / 2.0

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
    rho2 = (pred_xywh[:, 0] - target_xywh[:, 0]).pow(2) + (pred_xywh[:, 1] - target_xywh[:, 1]).pow(2)
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def _bbox_ciou_loss_xywh(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, reduction="sum") -> torch.Tensor:
    loss = 1.0 - _bbox_ciou_xywh(pred_xywh, target_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device))
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    return loss.sum()


def _resolve_yolo_class_confidences(pred_img: torch.Tensor) -> torch.Tensor:
    if pred_img.shape[1] <= 5:
        return torch.zeros((pred_img.shape[0], 0), dtype=torch.float32, device=pred_img.device)
    return pred_img[:, 5:].float().clamp(0.0, 1.0)


def _resolve_softmax_class_probabilities(logit_img: torch.Tensor, pred_img: torch.Tensor) -> torch.Tensor:
    if logit_img is not None and logit_img.numel() > 0:
        return torch.softmax(logit_img.float(), dim=-1)
    if pred_img.shape[1] <= 5:
        return torch.zeros((pred_img.shape[0], 0), dtype=torch.float32, device=pred_img.device)
    return torch.softmax(pred_img[:, 5:].float(), dim=-1)


def _prediction_dump_losses(
    pred_img: torch.Tensor,
    cls_conf_img: torch.Tensor,
    softmax_prob_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
    score_threshold: float,
    anchor_xywh: torch.Tensor = None,
):
    eps = 1e-6
    if raw_idx >= int(pred_img.shape[0]):
        zero = torch.zeros((), dtype=torch.float32, device=pred_img.device)
        return {
            "num_candidate_boxes": zero,
            "score_cand_diff": zero,
            "obj_cand_bce_loss": zero,
            "cls_cand_onehot_bce_loss": zero,
            "bbox_cand_ciou_loss": zero,
            "score_null_diff": zero,
            "obj_null_bce_loss": zero,
            "cls_uniform_kl": zero,
            "bbox_null_ciou_loss": zero,
        }

    pred_row = pred_img[raw_idx].float()
    pred_cls_conf_vec = cls_conf_img[raw_idx].float() if cls_conf_img.numel() > 0 else torch.zeros((0,), dtype=torch.float32, device=pred_img.device)
    pred_softmax_prob = softmax_prob_img[raw_idx].float() if softmax_prob_img.numel() > 0 else torch.zeros((0,), dtype=torch.float32, device=pred_img.device)
    pred_obj = pred_row[4].clamp(eps, 1.0 - eps)
    pred_cls = int(torch.argmax(pred_cls_conf_vec).item()) if pred_cls_conf_vec.numel() > 0 else 0
    pred_cls_conf = pred_cls_conf_vec[pred_cls].clamp(eps, 1.0 - eps) if pred_cls_conf_vec.numel() > 0 else torch.ones((), device=pred_img.device)
    pred_score = (pred_obj * pred_cls_conf).clamp(eps, 1.0 - eps)

    raw_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4].detach())
    pred_xyxy = _xywh_to_xyxy_tensor(pred_row[:4].view(1, 4)).view(4)
    ious = _box_iou_1vN_tensor(pred_xyxy, raw_xyxy)
    all_cls = torch.argmax(cls_conf_img, dim=1) if cls_conf_img.numel() > 0 else torch.zeros((pred_img.shape[0],), dtype=torch.long, device=pred_img.device)
    all_obj = pred_img[:, 4].float().clamp(eps, 1.0 - eps)
    all_cls_conf = cls_conf_img.max(dim=1).values.clamp(eps, 1.0 - eps) if cls_conf_img.numel() > 0 else torch.ones_like(all_obj)
    all_score = all_obj * all_cls_conf
    cand_mask = (ious >= float(iou_threshold)) & (all_cls == pred_cls) & (all_score >= float(score_threshold))
    if not bool(cand_mask.any()):
        cand_mask = torch.zeros_like(all_cls, dtype=torch.bool)
        cand_mask[raw_idx] = True

    cand_rows = pred_img[cand_mask].float()
    cand_cls_conf = cls_conf_img[cand_mask].float() if cls_conf_img.numel() > 0 else torch.zeros((cand_rows.shape[0], 0), dtype=torch.float32, device=pred_img.device)
    cand_cls = all_cls[cand_mask]
    cand_score = all_score[cand_mask]

    score_cand_diff = torch.abs(pred_score.expand_as(cand_score) - cand_score).sum()
    obj_cand_bce_loss = F.binary_cross_entropy(
        pred_obj.expand_as(cand_rows[:, 4]),
        cand_rows[:, 4].float().clamp(eps, 1.0 - eps),
        reduction="sum",
    )
    if pred_cls_conf_vec.numel() > 0 and cand_cls_conf.numel() > 0:
        cls_target = torch.zeros_like(cand_cls_conf)
        cls_target[torch.arange(cand_cls_conf.shape[0], device=cand_cls_conf.device), cand_cls] = 1.0
        cls_cand_onehot_bce_loss = F.binary_cross_entropy(
            pred_cls_conf_vec.clamp(eps, 1.0 - eps).view(1, -1).expand_as(cls_target),
            cls_target,
            reduction="sum",
        )
    else:
        cls_cand_onehot_bce_loss = torch.zeros((), dtype=torch.float32, device=pred_img.device)
    bbox_cand_ciou_loss = _bbox_ciou_loss_xywh(
        pred_row[:4].view(1, 4).expand(cand_rows.shape[0], -1),
        cand_rows[:, :4],
        reduction="sum",
    )

    num_classes = int(pred_cls_conf_vec.numel()) if pred_cls_conf_vec.numel() > 0 else 1
    null_score = torch.full_like(pred_score, 0.5 * (1.0 / float(num_classes)))
    score_null_diff = torch.abs(pred_score - null_score)
    obj_null_bce_loss = F.binary_cross_entropy(pred_obj, torch.full_like(pred_obj, 0.5), reduction="sum")
    if pred_softmax_prob.numel() > 0:
        p = pred_softmax_prob.clamp(eps, 1.0)
        uniform = torch.full_like(p, 1.0 / float(p.numel()))
        cls_uniform_kl = (p * (torch.log(p) - torch.log(uniform))).sum()
    else:
        cls_uniform_kl = torch.zeros((), dtype=torch.float32, device=pred_img.device)
    if anchor_xywh is None or anchor_xywh.numel() < 4:
        raise RuntimeError(
            "predict_dump requires YOLO anchor priors for bbox_null_ciou_loss, "
            "but the model output did not provide an anchor for this prediction."
        )
    target_bbox = anchor_xywh.to(dtype=pred_row.dtype, device=pred_row.device).view(1, 4)
    bbox_null_ciou_loss = _bbox_ciou_loss_xywh(pred_row[:4].view(1, 4), target_bbox, reduction="sum")

    return {
        "num_candidate_boxes": torch.tensor(float(cand_rows.shape[0]), dtype=torch.float32, device=pred_img.device),
        "score_cand_diff": score_cand_diff,
        "obj_cand_bce_loss": obj_cand_bce_loss,
        "cls_cand_onehot_bce_loss": cls_cand_onehot_bce_loss,
        "bbox_cand_ciou_loss": bbox_cand_ciou_loss,
        "score_null_diff": score_null_diff,
        "obj_null_bce_loss": obj_null_bce_loss,
        "cls_uniform_kl": cls_uniform_kl,
        "bbox_null_ciou_loss": bbox_null_ciou_loss,
    }


def run_predict_dump_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "predict_dump"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    active_cfg = config.get("output", {}).get("predict_dump", {}) or {}
    score_threshold = float(active_cfg.get("score_threshold", 0.0))
    iou_threshold = float(active_cfg.get("iou_threshold", 0.45))

    if not save_csv:
        return

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    prob_columns = [f"prob_{i}" for i in range(max(0, num_classes))]
    output_csv = run_dir / "predict_dump.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "max_iou", "gt_iou", "tp", "error_type",
        "objectness", "cls_conf", "class_probability",
        "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "bbox_area",
        "anchor_cx", "anchor_cy", "anchor_w", "anchor_h", "anchor_area",
        *prob_columns,
        "num_candidate_boxes",
        "score_cand_diff", "obj_cand_bce_loss", "cls_cand_onehot_bce_loss", "bbox_cand_ciou_loss",
        "score_null_diff", "obj_null_bce_loss", "cls_uniform_kl", "bbox_null_ciou_loss",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
                nms_logits = _resolve_nms_logits(raw_prediction, raw_logits, num_classes)
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=raw_prediction,
                    logits=nms_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_img = raw_prediction[sample_idx].float()
                logit_img = nms_logits[sample_idx].float() if nms_logits is not None else None
                cls_conf_img = _resolve_yolo_class_confidences(pred_img)
                softmax_prob_img = _resolve_softmax_class_probabilities(logit_img, pred_img)
                if raw_anchor_priors is None:
                    raise RuntimeError(
                        "predict_dump requires YOLO anchor priors, but detector.model() did not return them. "
                        "Expected model_output[3] with shape [batch, num_raw_predictions, 4]."
                    )
                anchor_img = raw_anchor_priors[sample_idx]
                if int(anchor_img.shape[0]) != int(pred_img.shape[0]):
                    raise RuntimeError(
                        "predict_dump anchor prior count does not match raw prediction count: "
                        f"{int(anchor_img.shape[0])} vs {int(pred_img.shape[0])}."
                    )

                pred_boxes = det_b[:, :4].detach().cpu().tolist()
                pred_scores = det_b[:, 4].detach().cpu().tolist() if det_b.shape[1] > 4 else []
                pred_cls_ids = det_b[:, 5].long() if det_b.shape[1] > 5 else torch.zeros((det_b.shape[0],), dtype=torch.long, device=device)
                pred_class_names = [
                    detector.names[int(c.item())] if detector.names is not None else int(c.item())
                    for c in pred_cls_ids
                ]
                gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                error_rows = analyze_prediction_error_types(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=iou_match_threshold,
                )

                for pred_idx, (box, score, pred_class) in enumerate(zip(pred_boxes, pred_scores, pred_class_names)):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    if raw_pred_idx >= int(pred_img.shape[0]):
                        continue
                    raw_row = pred_img[raw_pred_idx].float()
                    pred_cls_conf_vec = cls_conf_img[raw_pred_idx].float() if cls_conf_img.numel() > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
                    cls_conf = pred_cls_conf_vec.max() if pred_cls_conf_vec.numel() > 0 else torch.ones((), dtype=torch.float32, device=device)
                    if raw_pred_idx >= int(anchor_img.shape[0]):
                        raise RuntimeError(
                            f"raw_pred_idx {raw_pred_idx} is out of range for anchor priors "
                            f"with length {int(anchor_img.shape[0])}."
                        )
                    anchor_xywh = anchor_img[raw_pred_idx].float()
                    loss_values = _prediction_dump_losses(
                        pred_img=pred_img,
                        cls_conf_img=cls_conf_img,
                        softmax_prob_img=softmax_prob_img,
                        raw_idx=raw_pred_idx,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                        anchor_xywh=anchor_xywh,
                    )
                    prob_values = {
                        f"prob_{i}": (_to_float(pred_cls_conf_vec[i]) if i < int(pred_cls_conf_vec.shape[0]) else 0.0)
                        for i in range(max(0, num_classes))
                    }
                    error_row = error_rows[pred_idx]
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                            "score": float(score),
                            "pred_class": pred_class,
                            "max_iou": float(error_row["max_iou"]),
                            "gt_iou": float(error_row["gt_iou"]),
                            "tp": int(error_row["tp"]),
                            "error_type": error_row["error_type"],
                            "objectness": _to_float(raw_row[4]),
                            "cls_conf": _to_float(cls_conf),
                            "class_probability": _to_float(cls_conf),
                            "bbox_cx": _to_float(raw_row[0]),
                            "bbox_cy": _to_float(raw_row[1]),
                            "bbox_w": _to_float(raw_row[2]),
                            "bbox_h": _to_float(raw_row[3]),
                            "bbox_area": _to_float(raw_row[2].clamp(min=0.0) * raw_row[3].clamp(min=0.0)),
                            "anchor_cx": _to_float(anchor_xywh[0]),
                            "anchor_cy": _to_float(anchor_xywh[1]),
                            "anchor_w": _to_float(anchor_xywh[2]),
                            "anchor_h": _to_float(anchor_xywh[3]),
                            "anchor_area": _to_float(anchor_xywh[2].clamp(min=0.0) * anchor_xywh[3].clamp(min=0.0)),
                            **prob_values,
                            **{k: _to_float(v) for k, v in loss_values.items()},
                        }
                    )
            del infer_batch, model_output, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_predict_dump_csv"]
