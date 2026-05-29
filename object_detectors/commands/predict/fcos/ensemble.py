from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms, unpack_fcos_model_output


def _match_post_nms_feature(det_box, det_cls, run_features_b, run_labels_b):
    if run_features_b is None or int(run_features_b.shape[0]) == 0:
        return None
    if run_labels_b is None or int(run_labels_b.shape[0]) != int(run_features_b.shape[0]):
        return None
    cls_mask = run_labels_b.to(det_box.device) == int(det_cls)
    if not bool(cls_mask.any()):
        return None
    candidate_indices = torch.where(cls_mask)[0]
    ious = _box_iou_1vN_tensor(det_box.view(1, 4), run_features_b[candidate_indices, :4])
    best_pos = int(torch.argmax(ious).detach().cpu().item())
    return run_features_b[candidate_indices[best_pos]]

def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]

    if not save_csv:
        return

    ensemble_cfg = config.get("output", {}).get("ensemble", {})
    weights_cfg = ensemble_cfg.get("weights", [])
    if isinstance(weights_cfg, str):
        weight_paths = [weights_cfg]
    elif isinstance(weights_cfg, (list, tuple)):
        weight_paths = [str(w) for w in weights_cfg if str(w).strip()]
    else:
        weight_paths = []
    if not weight_paths:
        raise ValueError("output.uncertainty='ensemble' requires output.ensemble.weights to be a non-empty string/list.")

    # Keep loading deterministic and stable across repeated passes.
    dataset = build_dataset(config, split=split)
    dl_cfg = config["dataloader"]
    shuffle = dl_cfg["shuffle_train"] if split == "train" else dl_cfg["shuffle_eval"]
    dataloader = DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=yolo_collate_fn,
    )
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    output_csv = run_dir / "ensemble.csv"

    n_classes_hint = None
    class_names_hint = None
    n_classes_actual = None
    device = torch.device("cpu")
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )

    detectors = []
    try:
        for model_weight in weight_paths:
            detector, device = build_detector(config, model_weight=model_weight)
            if n_classes_hint is None:
                n_classes_hint = len(detector.names) if detector.names is not None else 80
                class_names_hint = detector.names
            detectors.append(detector)
        timing.device = device

        if n_classes_hint is None:
            n_classes_hint = 80

        fieldnames = [
            "image_id", "image_path", "pred_idx", "raw_pred_idx",
            "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
            "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean",
            "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std",
        ]
        for class_idx in range(n_classes_hint):
            fieldnames.append(f"prob_{class_idx}_mean")
            fieldnames.append(f"prob_{class_idx}_std")

        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for images, targets in tqdm(
                dataloader,
                desc=f"Object Detector ({mode} - {uncertainty})",
                total=len(dataloader),
            ):
                base_detector = detectors[0]
                infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                    base_detector, images, device, auto=False
                )
                fcos_preprocessed = (
                    base_detector.preprocess_images(infer_batch)
                    if bool(getattr(base_detector, "is_fcos", False)) and hasattr(base_detector, "preprocess_images")
                    else None
                )
                batch_size = len(infer_batch) if isinstance(infer_batch, list) else int(infer_batch.shape[0])
                image_ids = [int(targets[i]["image_id"][0].item()) for i in range(batch_size)]
                image_paths = [targets[i]["path"] for i in range(batch_size)]

                feature_runs = []
                variable_candidate_runs = False
                det_boxes = None
                raw_keep_indices = None
                detector_inference_total_sec = 0.0
                prediction_matching_sec = 0.0
                feature_compute_sec = 0.0
                for det_idx, detector in enumerate(detectors):
                    t_detector = timing.start()
                    with torch.no_grad():
                        if fcos_preprocessed is not None and bool(getattr(detector, "is_fcos", False)) and hasattr(detector, "forward_preprocessed"):
                            det_output = detector.forward_preprocessed(fcos_preprocessed)
                        else:
                            det_output = detector.model(infer_batch, augment=False)
                        det_raw_pred, det_raw_logits, det_raw_indices = unpack_fcos_model_output(det_output)
                        nms_kwargs = _resolve_detector_nms_kwargs(detector)
                        selected_preds, _selected_logits, _selected_objectness, selected_indices = select_fcos_post_nms(
                            detector,
                            det_raw_pred,
                            det_raw_logits,
                            det_raw_indices,
                        )
                    detector_inference_total_sec += timing.elapsed(t_detector)

                    t_feature = timing.start()
                    if isinstance(det_raw_pred, list):
                        variable_candidate_runs = True
                        run_features = []
                        run_labels = []
                        class_count = None
                        for pred_img in det_raw_pred:
                            pred_img = pred_img.detach().float()
                            bbox_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4])
                            score_vec = pred_img[:, 4:5]
                            prob_mat = get_prediction_class_probs(detector, pred_img).detach().float()
                            run_features.append(torch.cat([bbox_xyxy, score_vec, prob_mat], dim=1).detach())
                            run_labels.append(pred_img[:, 5].detach().long() if pred_img.shape[1] > 5 else None)
                            if class_count is None:
                                class_count = int(prob_mat.shape[-1])
                        if class_count is None:
                            class_count = 0
                    else:
                        pred_batch = det_raw_pred.detach().float()
                        bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                        score_vec = pred_batch[..., 4].unsqueeze(-1)
                        prob_mat = get_prediction_class_probs(detector, pred_batch).detach().float()
                        if prob_mat.numel() == 0 and det_raw_logits is not None:
                            prob_mat = torch.sigmoid(det_raw_logits.detach().float())
                        run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2).detach()
                        class_count = int(run_features.shape[2] - 5)
                    if n_classes_actual is None:
                        n_classes_actual = class_count
                    elif n_classes_actual != class_count:
                        raise ValueError(
                            f"All ensemble weights must have the same class count: {n_classes_actual} vs {class_count}."
                        )
                    feature_runs.append({"features": run_features, "labels": run_labels} if variable_candidate_runs else run_features)
                    feature_compute_sec += timing.elapsed(t_feature)

                    if det_idx == 0:
                        t_matching = timing.start()
                        det_boxes = []
                        raw_keep_indices = []
                        for b in range(batch_size):
                            det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                            raw_keep_b = (
                                selected_indices[b]
                                if selected_indices and b < len(selected_indices)
                                else torch.zeros((0,), dtype=torch.long, device=device)
                            )
                            det_boxes.append(det_b.detach().cpu())
                            raw_keep_indices.append([int(v) for v in raw_keep_b.detach().cpu().tolist()])
                        prediction_matching_sec += timing.elapsed(t_matching)

                t_feature = timing.start()
                mean = None
                std = None
                if not variable_candidate_runs:
                    runs_tensor = torch.stack(feature_runs, dim=0)  # [M, B, N, F]
                    mean = runs_tensor.mean(dim=0)
                    std = runs_tensor.std(dim=0, unbiased=False)
                    del runs_tensor
                feature_compute_sec += timing.elapsed(t_feature)
                del infer_batch
                if fcos_preprocessed is not None:
                    del fcos_preprocessed

                batch_items = 0
                t_matching = timing.start()
                mean_cpu = mean.detach().float().cpu() if mean is not None else None
                std_cpu = std.detach().float().cpu() if std is not None else None
                for b in range(len(image_ids)):
                    image_id = int(image_ids[b])
                    image_path = str(image_paths[b])
                    if not variable_candidate_runs:
                        mean_b = mean_cpu[b]
                        std_b = std_cpu[b]
                        n_candidates = int(mean_b.shape[0])

                    det_b = det_boxes[b]
                    raw_keep_b = [int(v) for v in raw_keep_indices[b]]
                    num_final = int(det_b.shape[0])
                    for pred_idx in range(num_final):
                        if pred_idx >= len(raw_keep_b):
                            continue
                        raw_idx = int(raw_keep_b[pred_idx])
                        cls_idx = int(det_b[pred_idx, 5].item()) if det_b.shape[1] > 5 else -1
                        if variable_candidate_runs:
                            per_model_values = []
                            for run_features in feature_runs:
                                features_by_image = run_features["features"] if isinstance(run_features, dict) else run_features
                                labels_by_image = run_features.get("labels") if isinstance(run_features, dict) else None
                                if b < len(features_by_image):
                                    matched = _match_post_nms_feature(
                                        det_b[pred_idx, :4].to(features_by_image[b].device),
                                        cls_idx,
                                        features_by_image[b],
                                        labels_by_image[b] if labels_by_image is not None and b < len(labels_by_image) else None,
                                    )
                                    if matched is not None:
                                        per_model_values.append(matched)
                            if not per_model_values:
                                class_count = int(n_classes_actual) if n_classes_actual is not None else int(n_classes_hint)
                                prob_vec = torch.zeros((class_count,), dtype=det_b.dtype)
                                if 0 <= cls_idx < class_count:
                                    prob_vec[cls_idx] = 1.0
                                mean_vec = torch.cat([det_b[pred_idx, :5].detach().float(), prob_vec], dim=0)
                                std_vec = torch.zeros_like(mean_vec)
                            else:
                                values = torch.stack(per_model_values, dim=0)
                                mean_vec = values.mean(dim=0).detach().float().cpu()
                                std_vec = values.std(dim=0, unbiased=False).detach().float().cpu()
                        else:
                            if raw_idx < 0 or raw_idx >= n_candidates:
                                continue
                            mean_vec = mean_b[raw_idx]
                            std_vec = std_b[raw_idx]
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_idx,
                            "xmin": float(det_b[pred_idx, 0].item()),
                            "ymin": float(det_b[pred_idx, 1].item()),
                            "xmax": float(det_b[pred_idx, 2].item()),
                            "ymax": float(det_b[pred_idx, 3].item()),
                            "score": float(det_b[pred_idx, 4].item()) if det_b.shape[1] > 4 else 0.0,
                            "pred_class": (
                                class_names_hint[cls_idx]
                                if (class_names_hint is not None and cls_idx >= 0 and cls_idx < len(class_names_hint))
                                else int(cls_idx)
                            ),
                            "xmin_mean": float(mean_vec[0].item()),
                            "ymin_mean": float(mean_vec[1].item()),
                            "xmax_mean": float(mean_vec[2].item()),
                            "ymax_mean": float(mean_vec[3].item()),
                            "score_mean": float(mean_vec[4].item()),
                            "xmin_std": float(std_vec[0].item()),
                            "ymin_std": float(std_vec[1].item()),
                            "xmax_std": float(std_vec[2].item()),
                            "ymax_std": float(std_vec[3].item()),
                            "score_std": float(std_vec[4].item()),
                        }
                        for class_idx in range(n_classes_hint):
                            if class_idx < n_classes_actual:
                                row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                                row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
                            else:
                                row[f"prob_{class_idx}_mean"] = 0.0
                                row[f"prob_{class_idx}_std"] = 0.0
                        writer.writerow(row)
                    batch_items += int(num_final)
                prediction_matching_sec += timing.elapsed(t_matching)
                timing.record(
                    num_images=batch_size,
                    num_predictions=batch_items,
                    stage_seconds={
                        "detector_inference_sec": detector_inference_total_sec,
                        "prediction_matching_sec": prediction_matching_sec,
                        "feature_compute_sec": feature_compute_sec,
                    },
                )
                del feature_runs
                if mean is not None:
                    del mean, std, mean_cpu, std_cpu
    except Exception:
        raise
    finally:
        for detector in detectors:
            del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_ensemble_csv"]
