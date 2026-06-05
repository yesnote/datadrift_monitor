from commands.predict.common import *
from commands.predict.fcos.utils import (
    build_fcos_candidate_cache,
    fcos_candidate_mask_from_cache,
    iter_fcos_detection_rows,
    run_fcos_forward_nms,
    selected_fcos_class_probs,
)


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


def _run_fcos_meta_forward(detector, image_list, device, timing, score_threshold):
    normal_threshold = float(getattr(detector, "confidence", 0.05))
    candidate_threshold = min(normal_threshold, float(score_threshold))
    if candidate_threshold < normal_threshold:
        result = run_fcos_forward_nms(
            detector=detector,
            image_list=image_list,
            device=device,
            timing=timing,
            keep_pre_nms=False,
            keep_class_outputs=True,
            pre_nms_threshold=normal_threshold,
        )
        t_detector = timing.start()
        with torch.no_grad():
            with detector.temporary_pre_nms_threshold(candidate_threshold):
                detector.forward_preprocessed(
                    result.processed_images,
                    keep_pre_nms=True,
                    keep_class_outputs=False,
                )
            pre_nms_prediction, _pre_logits, _pre_indices = detector.get_last_pre_nms_predictions()
        result.pre_nms_prediction = pre_nms_prediction
        result.detector_inference_sec += timing.elapsed(t_detector)
        return result
    return run_fcos_forward_nms(
        detector=detector,
        image_list=image_list,
        device=device,
        timing=timing,
        keep_pre_nms=True,
        keep_class_outputs=True,
        pre_nms_threshold=normal_threshold,
    )


def run_meta_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "meta_detect"

    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_threshold = float(parsed["meta_detect_score_threshold"])
    iou_threshold = float(parsed["meta_detect_iou_threshold"])

    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
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
    output_csv = run_dir / "meta_detect.csv"

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
            detector.zero_grad(set_to_none=True)
            result = _run_fcos_meta_forward(detector, image_list, device, timing, score_threshold)
            pre_nms_prediction = result.pre_nms_prediction if result.pre_nms_prediction is not None else result.raw_prediction

            candidate_search_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            candidate_caches = {}
            probs_by_sample = {
                sample_idx: selected_fcos_class_probs(result, sample_idx, num_classes, device)
                for sample_idx in range(len(image_list))
            }
            for sample_idx in range(len(image_list)):
                candidate_img = (
                    pre_nms_prediction[sample_idx].float()
                    if pre_nms_prediction is not None and sample_idx < len(pre_nms_prediction)
                    else torch.zeros((0, 6 + num_classes), dtype=torch.float32, device=device)
                )
                t_candidate = timing.start()
                candidate_caches[sample_idx] = build_fcos_candidate_cache(candidate_img, score_threshold)
                candidate_search_sec += timing.elapsed(t_candidate)

            for det_row in iter_fcos_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                t_candidate = timing.start()
                cache = candidate_caches[det_row.sample_idx]
                cand_mask, ious = fcos_candidate_mask_from_cache(
                    cache,
                    det_row.box[:4].detach().float(),
                    det_row.cls_idx,
                    iou_threshold,
                )
                candidate_search_sec += timing.elapsed(t_candidate)

                t_feature = timing.start()
                cand_boxes = cache.boxes_xyxy[cand_mask]
                cand_scores = cache.scores[cand_mask]
                cand_ious = ious[cand_mask]
                probs = probs_by_sample[det_row.sample_idx]
                pred_probs = probs[det_row.pred_idx].detach().float() if det_row.pred_idx < int(probs.shape[0]) else torch.zeros((num_classes,), dtype=torch.float32, device=device)
                prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)}
                for prob_idx in range(max(0, num_classes)):
                    prob_values[f"prob_{prob_idx}"] = pred_probs[prob_idx] if prob_idx < int(pred_probs.shape[0]) else torch.zeros((), dtype=torch.float32, device=device)

                x = 0.5 * (cand_boxes[:, 0] + cand_boxes[:, 2])
                y = 0.5 * (cand_boxes[:, 1] + cand_boxes[:, 3])
                w = torch.abs(0.5 * (cand_boxes[:, 0] - cand_boxes[:, 2]))
                h = torch.abs(0.5 * (cand_boxes[:, 1] - cand_boxes[:, 3]))
                size_vals = (0.5 * (x - w)) * (0.5 * (y - h))
                circum_vals = (cand_boxes[:, 2] - cand_boxes[:, 0]) + (cand_boxes[:, 3] - cand_boxes[:, 1])
                size_circum_vals = (w * h) / (torch.abs(cand_boxes[:, 2] - cand_boxes[:, 0]) + torch.abs(cand_boxes[:, 3] - cand_boxes[:, 1])).clamp(min=1e-12)

                iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
                iou_pb_pos = iou_pb[iou_pb > 0]

                fx1_t, fy1_t, fx2_t, fy2_t = det_row.box[:4].detach().float().unbind()
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
                writer.writerow(
                    {
                        **det_row.base,
                        **{key: _to_float(value) for key, value in prob_values.items()},
                        **{key: _to_float(value) for key, value in feature_row.items()},
                    }
                )
                batch_items += 1

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": result.detector_inference_sec,
                    "candidate_search_sec": candidate_search_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            if hasattr(detector, "_clear_last_pre_nms_predictions"):
                detector._clear_last_pre_nms_predictions()
            del result, candidate_caches, probs_by_sample

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_meta_detect_csv"]
