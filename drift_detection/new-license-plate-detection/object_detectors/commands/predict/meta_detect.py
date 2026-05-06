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
    vector_reduction = parsed["meta_detect_vector_reduction"]

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='meta_detect' requires output.unit in {'image','bbox'}.")

    def _stats(v: torch.Tensor):
        if v is None or v.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        x = v.detach().float().reshape(-1)
        return float(torch.min(x).item()), float(torch.max(x).item()), float(torch.mean(x).item()), float(torch.std(x, unbiased=False).item())

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
    if unit == "bbox":
        fieldnames = [
            "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
            *meta_feature_names,
        ]
    else:
        fieldnames = ["image_id", "image_path"]
        for feature_name in meta_feature_names:
            for metric_name in vector_reduction:
                fieldnames.append(f"{feature_name}_{metric_name}")
        fieldnames.append("num_preds")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    raw_prof = RawComputeProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=unit)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
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

            t_raw = raw_prof.start()
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

                raw = raw_prediction[sample_idx].detach().float()
                if raw.numel() == 0:
                    continue
                raw_xyxy = _xywh_to_xyxy_tensor(raw[:, :4])
                raw_obj = raw[:, 4] if raw.shape[1] > 4 else torch.ones((raw.shape[0],), device=device)
                raw_cls = raw[:, 5:] if raw.shape[1] > 5 else torch.zeros((raw.shape[0], 0), device=device)
                if raw_cls.numel() > 0:
                    raw_cls_max, raw_cls_idx = raw_cls.max(dim=1)
                else:
                    raw_cls_max = torch.ones_like(raw_obj)
                    raw_cls_idx = torch.zeros((raw.shape[0],), dtype=torch.long, device=device)
                raw_score = raw_obj * raw_cls_max

                image_feature_rows = []
                for pred_idx, (box, score, pred_class_name) in enumerate(zip(pred_boxes, pred_scores, pred_class_names)):
                    fbox = torch.tensor(box, dtype=torch.float32, device=device)
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    cls_idx = int(pred_class_ids[pred_idx]) if pred_idx < len(pred_class_ids) else -1
                    ious = _iou_1vN(fbox, raw_xyxy)
                    class_mask = (raw_cls_idx == cls_idx) if cls_idx >= 0 else torch.ones_like(raw_score, dtype=torch.bool)
                    cand_mask = class_mask & (raw_score > score_threshold) & (ious > iou_threshold)

                    cand_boxes = raw_xyxy[cand_mask]
                    cand_scores = raw_score[cand_mask]
                    cand_ious = ious[cand_mask]
                    if cand_boxes.numel() == 0:
                        cand_boxes = fbox.view(1, 4)
                        cand_scores = torch.tensor([float(score)], dtype=torch.float32, device=device)
                        cand_ious = torch.zeros((1,), dtype=torch.float32, device=device)

                    x = 0.5 * (cand_boxes[:, 0] + cand_boxes[:, 2])
                    y = 0.5 * (cand_boxes[:, 1] + cand_boxes[:, 3])
                    w = torch.abs(0.5 * (cand_boxes[:, 0] - cand_boxes[:, 2]))
                    h = torch.abs(0.5 * (cand_boxes[:, 1] - cand_boxes[:, 3]))
                    size_vals = (0.5 * (x - w)) * (0.5 * (y - h))
                    circum_vals = (cand_boxes[:, 2] - cand_boxes[:, 0]) + (cand_boxes[:, 3] - cand_boxes[:, 1])
                    size_circum_vals = (w * h) / (torch.abs(cand_boxes[:, 2] - cand_boxes[:, 0]) + torch.abs(cand_boxes[:, 3] - cand_boxes[:, 1])).clamp(min=1e-12)

                    iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
                    iou_pb_pos = iou_pb[iou_pb > 0]

                    fx1, fy1, fx2, fy2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    fsize = float((0.5 * ((0.5 * (fx1 + fx2)) - abs(0.5 * (fx1 - fx2)))) * (0.5 * ((0.5 * (fy1 + fy2)) - abs(0.5 * (fy1 - fy2)))))
                    fcircum = float(abs(fx2 - fx1) + abs(fy2 - fy1))
                    fsize_circum = float(((0.5 * abs(fx2 - fx1)) * (0.5 * abs(fy2 - fy1))) / max(abs(fx2 - fx1) + abs(fy2 - fy1), 1e-12))

                    x_min, x_max, x_mean, x_std = _stats(x)
                    y_min, y_max, y_mean, y_std = _stats(y)
                    w_min, w_max, w_mean, w_std = _stats(w)
                    h_min, h_max, h_mean, h_std = _stats(h)
                    size_min, size_max, size_mean, size_std = _stats(size_vals)
                    circum_min, circum_max, circum_mean, circum_std = _stats(circum_vals)
                    size_circum_min, size_circum_max, size_circum_mean, size_circum_std = _stats(size_circum_vals)
                    score_min, score_max, score_mean, score_std = _stats(cand_scores)
                    iou_pb_min, iou_pb_max, iou_pb_mean, iou_pb_std = _stats(iou_pb_pos)

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
                    if unit == "bbox":
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
                                **feature_row,
                            }
                        )
                    else:
                        image_feature_rows.append(feature_row)
                if unit == "image":
                    batch_items += 1
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "num_preds": len(image_feature_rows),
                    }
                    for feature_name in meta_feature_names:
                        if len(image_feature_rows) == 0:
                            stats = {
                                "1-norm": 0.0,
                                "2-norm": 0.0,
                                "min": 0.0,
                                "max": 0.0,
                                "mean": 0.0,
                                "std": 0.0,
                            }
                        else:
                            vec = torch.tensor(
                                [float(r[feature_name]) for r in image_feature_rows],
                                dtype=torch.float32,
                                device=device,
                            )
                            stats = map_grad_tensor_to_numbers(vec)
                        for metric_name in vector_reduction:
                            row[f"{feature_name}_{metric_name}"] = float(stats[metric_name])
                    writer.writerow(row)
                else:
                    batch_items += int(len(pred_boxes))
            raw_prof.end(t_raw, batch_items)
            del infer_batch, model_output, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = raw_prof.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved raw compute timing: {timing_csv}")
    print(f"Saved raw compute timing summary: {timing_json}")

__all__ = ["run_meta_detect_csv"]
