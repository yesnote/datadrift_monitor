from commands.predict.common import *


def _class_name(names, cls_idx):
    if isinstance(names, dict):
        return names.get(cls_idx, str(cls_idx))
    if isinstance(names, list) and 0 <= cls_idx < len(names):
        return names[cls_idx]
    return str(cls_idx)


def _null_target_stats(raw_row, logit_row, cls_idx, obj, score):
    if logit_row is not None and getattr(logit_row, "numel", lambda: 0)() > 0:
        cls_probs = torch.softmax(logit_row.detach().float(), dim=-1)
    elif raw_row is not None and raw_row.shape[0] > 5:
        cls_probs = raw_row[5:].detach().float().clamp(min=0.0)
        prob_sum = cls_probs.sum()
        if float(prob_sum.detach().cpu().item()) > 1e-12:
            cls_probs = cls_probs / prob_sum
    else:
        cls_probs = torch.zeros((0,), dtype=torch.float32)

    n_cls = int(cls_probs.shape[0]) if cls_probs.numel() else 0
    null_obj = 0.5
    null_cls_conf = (1.0 / float(n_cls)) if n_cls > 0 else 0.0
    null_score = null_obj * null_cls_conf
    cls_prob = (
        float(cls_probs[cls_idx].detach().cpu().item())
        if cls_idx >= 0 and n_cls > cls_idx
        else 0.0
    )
    if n_cls > 0:
        eps = 1e-12
        probs = cls_probs.clamp(min=eps)
        entropy = float((-(probs * torch.log(probs)).sum()).detach().cpu().item())
        max_entropy = float(np.log(float(n_cls)))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        uniform_kl = max_entropy - entropy
    else:
        entropy = 0.0
        entropy_norm = 0.0
        uniform_kl = 0.0
    return {
        "null_obj": null_obj,
        "null_cls_conf": null_cls_conf,
        "null_score": null_score,
        "obj_null_abs_diff": abs(float(obj) - null_obj),
        "cls_null_abs_diff": abs(float(cls_prob) - null_cls_conf),
        "score_null_abs_diff": abs(float(score) - null_score),
        "cls_entropy": entropy,
        "cls_entropy_norm": entropy_norm,
        "cls_uniform_kl": uniform_kl,
    }


def run_prediction_dump_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "prediction_dump"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    if unit != "bbox":
        raise ValueError("output.uncertainty='prediction_dump' requires output.unit='bbox'.")
    if not save_csv:
        return

    output_csv = run_dir / "prediction_dump.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "cx",
        "cy",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "anchor_cx",
        "anchor_cy",
        "anchor_w",
        "anchor_h",
        "anchor_area",
        "anchor_aspect_ratio",
        "bbox_anchor_dx",
        "bbox_anchor_dy",
        "bbox_anchor_center_l2",
        "bbox_anchor_w_ratio",
        "bbox_anchor_h_ratio",
        "bbox_anchor_area_ratio",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_aspect_ratio_diff",
        "obj",
        "cls_conf",
        "score",
        "pred_class",
        "pred_class_id",
        "null_obj",
        "null_cls_conf",
        "null_score",
        "obj_null_abs_diff",
        "cls_null_abs_diff",
        "score_null_abs_diff",
        "cls_entropy",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "max_iou",
        "tp",
    ]

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    raw_prof = RawComputeProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=unit)
    total_images = 0
    total_predictions = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            total_images += len(image_list)
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = (
                    model_output[1]
                    if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                    else None
                )
                raw_priors = (
                    model_output[3]
                    if isinstance(model_output, (tuple, list)) and len(model_output) > 3
                    else None
                )
                nms_logits = _resolve_nms_logits(raw_prediction, raw_logits)
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

            t_raw = raw_prof.start()
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = selected_preds[sample_idx]
                raw_keep_b = selected_indices[sample_idx]
                raw_b = raw_prediction[sample_idx].detach().float()
                pred_boxes = det[:, :4].detach().cpu().tolist()
                pred_scores = det[:, 4].detach().cpu().tolist() if det.shape[1] > 4 else []
                pred_cls_ids = (
                    det[:, 5].long().detach().cpu().tolist()
                    if det.shape[1] > 5
                    else [0 for _ in range(int(det.shape[0]))]
                )
                pred_class_names = [_class_name(detector.names, int(cls_idx)) for cls_idx in pred_cls_ids]
                gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                tp_flags, best_ious = assign_tp_to_predictions(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=iou_match_threshold,
                )
                batch_items += int(det.shape[0])
                total_predictions += int(det.shape[0])

                for pred_idx, box in enumerate(det):
                    raw_pred_idx = (
                        int(raw_keep_b[pred_idx].detach().cpu().item())
                        if pred_idx < int(raw_keep_b.shape[0])
                        else pred_idx
                    )
                    cls_idx = int(pred_cls_ids[pred_idx]) if pred_idx < len(pred_cls_ids) else -1
                    xmin = float(box[0].detach().cpu().item())
                    ymin = float(box[1].detach().cpu().item())
                    xmax = float(box[2].detach().cpu().item())
                    ymax = float(box[3].detach().cpu().item())
                    w = max(0.0, xmax - xmin)
                    h = max(0.0, ymax - ymin)
                    area = w * h
                    obj = 0.0
                    cls_conf = 0.0
                    raw_row = None
                    logit_row = None
                    prior_row = None
                    if 0 <= raw_pred_idx < int(raw_b.shape[0]):
                        raw_row = raw_b[raw_pred_idx]
                        obj = float(raw_row[4].detach().cpu().item()) if raw_row.shape[0] > 4 else 1.0
                        if cls_idx >= 0 and raw_row.shape[0] > 5 + cls_idx:
                            cls_conf = float(raw_row[5 + cls_idx].detach().cpu().item())
                    if raw_logits is not None and sample_idx < int(raw_logits.shape[0]) and 0 <= raw_pred_idx < int(raw_logits.shape[1]):
                        logit_row = raw_logits[sample_idx, raw_pred_idx]
                    if raw_priors is not None and sample_idx < int(raw_priors.shape[0]) and 0 <= raw_pred_idx < int(raw_priors.shape[1]):
                        prior_row = raw_priors[sample_idx, raw_pred_idx].detach().float()
                    score = float(box[4].detach().cpu().item())
                    if cls_conf == 0.0 and obj > 1e-12:
                        cls_conf = score / obj
                    null_stats = _null_target_stats(raw_row, logit_row, cls_idx, obj, score)
                    anchor_cx = anchor_cy = anchor_w = anchor_h = 0.0
                    if prior_row is not None and prior_row.shape[0] >= 4:
                        anchor_cx = float(prior_row[0].detach().cpu().item())
                        anchor_cy = float(prior_row[1].detach().cpu().item())
                        anchor_w = float(prior_row[2].detach().cpu().item())
                        anchor_h = float(prior_row[3].detach().cpu().item())
                    anchor_area = max(0.0, anchor_w) * max(0.0, anchor_h)
                    anchor_aspect_ratio = anchor_w / max(anchor_h, 1e-12)
                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)
                    dx = cx - anchor_cx
                    dy = cy - anchor_cy
                    w_ratio = w / max(anchor_w, 1e-12)
                    h_ratio = h / max(anchor_h, 1e-12)
                    area_ratio = area / max(anchor_area, 1e-12)

                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "cx": cx,
                            "cy": cy,
                            "w": w,
                            "h": h,
                            "area": area,
                            "aspect_ratio": w / max(h, 1e-12),
                            "anchor_cx": anchor_cx,
                            "anchor_cy": anchor_cy,
                            "anchor_w": anchor_w,
                            "anchor_h": anchor_h,
                            "anchor_area": anchor_area,
                            "anchor_aspect_ratio": anchor_aspect_ratio,
                            "bbox_anchor_dx": dx,
                            "bbox_anchor_dy": dy,
                            "bbox_anchor_center_l2": float((dx * dx + dy * dy) ** 0.5),
                            "bbox_anchor_w_ratio": w_ratio,
                            "bbox_anchor_h_ratio": h_ratio,
                            "bbox_anchor_area_ratio": area_ratio,
                            "bbox_anchor_log_w_ratio": float(np.log(max(w_ratio, 1e-12))),
                            "bbox_anchor_log_h_ratio": float(np.log(max(h_ratio, 1e-12))),
                            "bbox_anchor_log_area_ratio": float(np.log(max(area_ratio, 1e-12))),
                            "bbox_anchor_aspect_ratio_diff": (w / max(h, 1e-12)) - anchor_aspect_ratio,
                            "obj": obj,
                            "cls_conf": cls_conf,
                            "score": score,
                            "pred_class": pred_class_names[pred_idx] if pred_idx < len(pred_class_names) else _class_name(detector.names, cls_idx),
                            "pred_class_id": cls_idx,
                            **null_stats,
                            "max_iou": float(best_ious[pred_idx]) if pred_idx < len(best_ious) else 0.0,
                            "tp": int(tp_flags[pred_idx]) if pred_idx < len(tp_flags) else 0,
                        }
                    )
            raw_prof.end(t_raw, batch_items)
            del infer_batch, raw_prediction, raw_logits, raw_priors, selected_preds, selected_indices

    summary = {
        "output_csv": str(output_csv),
        "total_images": int(total_images),
        "total_predictions": int(total_predictions),
        "mean_predictions_per_image": (float(total_predictions) / total_images) if total_images else 0.0,
    }
    with open(run_dir / "prediction_dump_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = raw_prof.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved prediction dump summary: {run_dir / 'prediction_dump_summary.json'}")
    print(f"Saved raw compute timing: {timing_csv}")
    print(f"Saved raw compute timing summary: {timing_json}")


__all__ = ["run_prediction_dump_csv"]
