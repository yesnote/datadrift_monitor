from commands.predict.common import *

def run_meta_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "meta_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_threshold = float(parsed["meta_detect_score_threshold"])
    iou_threshold = float(parsed["meta_detect_iou_threshold"])

    if not save_csv:
        return

    def _stats_tensor(v: torch.Tensor, device):
        if v is None or v.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return zero, zero, zero, zero
        x = v.detach().float().reshape(-1)
        return torch.min(x), torch.max(x), torch.mean(x), torch.std(x, unbiased=False)

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _iou_1vN(box: torch.Tensor, boxes: torch.Tensor):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=box.device)
        lt = torch.max(box[:2], boxes[:, :2])
        rb = torch.min(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-12)

    def _boxes_to_original_xyxy(boxes: torch.Tensor, ratio, pad, image_tensor: torch.Tensor):
        if boxes.numel() == 0:
            return boxes.clone()
        out = boxes.clone()
        ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
        pad_w, pad_h = float(pad[0]), float(pad[1])
        img_h = int(image_tensor.shape[-2])
        img_w = int(image_tensor.shape[-1])
        out[..., [0, 2]] = (out[..., [0, 2]] - pad_w) / max(ratio_w, 1e-12)
        out[..., [1, 3]] = (out[..., [1, 3]] - pad_h) / max(ratio_h, 1e-12)
        out[..., [0, 2]] = out[..., [0, 2]].clamp(0, img_w)
        out[..., [1, 3]] = out[..., [1, 3]].clamp(0, img_h)
        return out

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]

    output_csv = run_dir / "meta_detect.csv"
    meta_feature_names = [
        "num_candidate_boxes",
        "x_min", "x_max", "x_mean", "x_std",
        "y_min", "y_max", "y_mean", "y_std",
        "w_min", "w_max", "w_mean", "w_std",
        "h_min", "h_max", "h_mean", "h_std",
        "size", "size_min", "size_max", "size_mean", "size_std",
        "circum", "circum_min", "circum_max", "circum_mean", "circum_std",
        "size_circum", "size_circum_min", "size_circum_max", "size_circum_mean", "size_circum_std",
        "score_min", "score_max", "score_mean", "score_std",
        "iou_pb_min", "iou_pb_max", "iou_pb_mean", "iou_pb_std",
    ]
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        *output_feature_names,
        *meta_feature_names,
    ]

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "candidate_search_sec", "feature_compute_sec"],
        device=device,
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            t_detector = timing.start()
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
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
            detector_inference_sec = timing.elapsed(t_detector)

            candidate_search_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_boxes = det_b[:, :4].detach().cpu().tolist()
                pred_scores = det_b[:, 4].detach().cpu().tolist() if det_b.shape[1] > 4 else []
                pred_class_ids = det_b[:, 5].long().detach().cpu().tolist() if det_b.shape[1] > 5 else []
                pred_class_names = [
                    detector.names[int(cls_id)] if detector.names is not None else int(cls_id)
                    for cls_id in pred_class_ids
                ]

                t_candidate = timing.start()
                raw = raw_prediction[sample_idx].detach().float()
                if raw.numel() == 0:
                    candidate_search_sec += timing.elapsed(t_candidate)
                    continue
                raw_xyxy = _xywh_to_xyxy_tensor(raw[:, :4])
                raw_xyxy_orig = _boxes_to_original_xyxy(raw_xyxy, ratios[sample_idx], pads[sample_idx], image_list[sample_idx])
                raw_obj = raw[:, 4] if raw.shape[1] > 4 else torch.ones((raw.shape[0],), device=device)
                raw_cls = raw[:, 5:] if raw.shape[1] > 5 else torch.zeros((raw.shape[0], 0), device=device)
                if raw_cls.numel() > 0:
                    raw_cls_max, raw_cls_idx = raw_cls.max(dim=1)
                else:
                    raw_cls_max = torch.ones_like(raw_obj)
                    raw_cls_idx = torch.zeros((raw.shape[0],), dtype=torch.long, device=device)
                raw_score = raw_obj * raw_cls_max
                candidate_search_sec += timing.elapsed(t_candidate)

                for pred_idx, (box, score, pred_class_name) in enumerate(zip(pred_boxes, pred_scores, pred_class_names)):
                    t_candidate = timing.start()
                    fbox = torch.tensor(box, dtype=torch.float32, device=device)
                    fbox_orig = _boxes_to_original_xyxy(fbox.view(1, 4), ratios[sample_idx], pads[sample_idx], image_list[sample_idx]).view(4)
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    cls_idx = int(pred_class_ids[pred_idx]) if pred_idx < len(pred_class_ids) else -1
                    ious = _iou_1vN(fbox, raw_xyxy)
                    class_mask = (raw_cls_idx == cls_idx) if cls_idx >= 0 else torch.ones_like(raw_score, dtype=torch.bool)
                    cand_mask = class_mask & (raw_score > score_threshold) & (ious > iou_threshold)

                    cand_boxes = raw_xyxy_orig[cand_mask]
                    cand_scores = raw_score[cand_mask]
                    cand_ious = ious[cand_mask]
                    if cand_boxes.numel() == 0:
                        cand_boxes = fbox_orig.view(1, 4)
                        cand_scores = torch.tensor([float(score)], dtype=torch.float32, device=device)
                        cand_ious = torch.zeros((1,), dtype=torch.float32, device=device)
                    candidate_search_sec += timing.elapsed(t_candidate)

                    t_feature = timing.start()
                    if 0 <= raw_pred_idx < int(raw_cls.shape[0]) and raw_cls.numel() > 0:
                        pred_probs = raw_cls[raw_pred_idx].detach().float()
                    else:
                        pred_probs = torch.zeros((num_classes,), dtype=torch.float32, device=device)
                    prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)}
                    for prob_idx in range(max(0, num_classes)):
                        prob_values[f"prob_{prob_idx}"] = (
                            pred_probs[prob_idx]
                            if prob_idx < int(pred_probs.shape[0])
                            else torch.zeros((), dtype=torch.float32, device=device)
                        )

                    x = 0.5 * (cand_boxes[:, 0] + cand_boxes[:, 2])
                    y = 0.5 * (cand_boxes[:, 1] + cand_boxes[:, 3])
                    w = torch.abs(0.5 * (cand_boxes[:, 0] - cand_boxes[:, 2]))
                    h = torch.abs(0.5 * (cand_boxes[:, 1] - cand_boxes[:, 3]))
                    size_vals = (0.5 * (x - w)) * (0.5 * (y - h))
                    circum_vals = (cand_boxes[:, 2] - cand_boxes[:, 0]) + (cand_boxes[:, 3] - cand_boxes[:, 1])
                    size_circum_vals = (w * h) / (torch.abs(cand_boxes[:, 2] - cand_boxes[:, 0]) + torch.abs(cand_boxes[:, 3] - cand_boxes[:, 1])).clamp(min=1e-12)

                    iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
                    iou_pb_pos = iou_pb[iou_pb > 0]

                    fx1_t, fy1_t, fx2_t, fy2_t = fbox_orig.unbind()
                    fsize = (0.5 * ((0.5 * (fx1_t + fx2_t)) - torch.abs(0.5 * (fx1_t - fx2_t)))) * (
                        0.5 * ((0.5 * (fy1_t + fy2_t)) - torch.abs(0.5 * (fy1_t - fy2_t)))
                    )
                    fcircum = torch.abs(fx2_t - fx1_t) + torch.abs(fy2_t - fy1_t)
                    fsize_circum = ((0.5 * torch.abs(fx2_t - fx1_t)) * (0.5 * torch.abs(fy2_t - fy1_t))) / fcircum.clamp(min=1e-12)

                    x_min, x_max, x_mean, x_std = _stats_tensor(x, device)
                    y_min, y_max, y_mean, y_std = _stats_tensor(y, device)
                    w_min, w_max, w_mean, w_std = _stats_tensor(w, device)
                    h_min, h_max, h_mean, h_std = _stats_tensor(h, device)
                    size_min, size_max, size_mean, size_std = _stats_tensor(size_vals, device)
                    circum_min, circum_max, circum_mean, circum_std = _stats_tensor(circum_vals, device)
                    size_circum_min, size_circum_max, size_circum_mean, size_circum_std = _stats_tensor(size_circum_vals, device)
                    score_min, score_max, score_mean, score_std = _stats_tensor(cand_scores, device)
                    iou_pb_min, iou_pb_max, iou_pb_mean, iou_pb_std = _stats_tensor(iou_pb_pos, device)

                    feature_row = {
                        "num_candidate_boxes": float(cand_boxes.shape[0]),
                        "x_min": x_min, "x_max": x_max, "x_mean": x_mean, "x_std": x_std,
                        "y_min": y_min, "y_max": y_max, "y_mean": y_mean, "y_std": y_std,
                        "w_min": w_min, "w_max": w_max, "w_mean": w_mean, "w_std": w_std,
                        "h_min": h_min, "h_max": h_max, "h_mean": h_mean, "h_std": h_std,
                        "size": fsize, "size_min": size_min, "size_max": size_max, "size_mean": size_mean, "size_std": size_std,
                        "circum": fcircum, "circum_min": circum_min, "circum_max": circum_max, "circum_mean": circum_mean, "circum_std": circum_std,
                        "size_circum": fsize_circum, "size_circum_min": size_circum_min, "size_circum_max": size_circum_max,
                        "size_circum_mean": size_circum_mean, "size_circum_std": size_circum_std,
                        "score_min": score_min, "score_max": score_max, "score_mean": score_mean, "score_std": score_std,
                        "iou_pb_min": iou_pb_min, "iou_pb_max": iou_pb_max, "iou_pb_mean": iou_pb_mean, "iou_pb_std": iou_pb_std,
                    }
                    feature_compute_sec += timing.elapsed(t_feature)
                    prob_values = {key: _to_float(value) for key, value in prob_values.items()}
                    feature_row = {key: _to_float(value) for key, value in feature_row.items()}
                    fx1, fy1, fx2, fy2 = [_to_float(v) for v in fbox_orig]
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": fx1,
                            "ymin": fy1,
                            "xmax": fx2,
                            "ymax": fy2,
                            "score": float(score),
                            "pred_class": pred_class_name,
                            **prob_values,
                            **feature_row,
                        }
                    )
                batch_items += int(len(pred_boxes))
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "candidate_search_sec": candidate_search_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del infer_batch, model_output, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_meta_detect_csv"]
