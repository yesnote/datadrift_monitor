from copy import deepcopy

from commands.predict.common import *


def _match_faster_rcnn_mc_feature(det_box, det_cls, run_features_b, run_labels_b):
    if run_features_b is None or int(run_features_b.shape[0]) == 0:
        return None
    if run_labels_b is not None and int(run_labels_b.shape[0]) == int(run_features_b.shape[0]):
        cls_mask = run_labels_b.to(det_box.device) == int(det_cls)
        if bool(cls_mask.any()):
            candidate_indices = torch.where(cls_mask)[0]
            candidate_boxes = run_features_b[candidate_indices, :4]
            ious = _box_iou_1vN_tensor(det_box.view(1, 4), candidate_boxes)
            best_pos = int(torch.argmax(ious).detach().cpu().item())
            return run_features_b[candidate_indices[best_pos]]
    return None


def _deterministic_mc_feature(detector, det_row, det_raw_pred_b, raw_idx, class_count):
    device = det_row.device
    dtype = det_row.dtype
    bbox_score = det_row[:5].detach().to(dtype=torch.float32)
    probs = torch.zeros((class_count,), dtype=torch.float32, device=device)
    if det_raw_pred_b is not None and 0 <= int(raw_idx) < int(det_raw_pred_b.shape[0]):
        raw_probs = get_prediction_class_probs(
            detector,
            det_raw_pred_b[int(raw_idx) : int(raw_idx) + 1],
        ).detach().float().reshape(-1)
        n = min(class_count, int(raw_probs.shape[0]))
        if n > 0:
            probs[:n] = raw_probs[:n].to(device=device)
    elif det_row.shape[0] > 5:
        cls_idx = int(det_row[5].detach().cpu().item())
        if 0 <= cls_idx < class_count:
            probs[cls_idx] = det_row[4].detach().to(dtype=torch.float32)
    return torch.cat([bbox_score.to(device=device, dtype=dtype), probs.to(device=device, dtype=dtype)], dim=0)


def _is_yolov5_detector(detector):
    return not bool(getattr(detector, "is_faster_rcnn", False)) and not bool(getattr(detector, "is_fcos", False))


def _get_yolov5_detect_module(detector):
    model = getattr(detector, "model", None)
    modules = getattr(model, "model", None)
    if modules is None or len(modules) == 0:
        raise RuntimeError("YOLOv5 MC-dropout requires detector.model.model modules.")
    detect_module = modules[-1]
    if not (hasattr(detect_module, "m") and hasattr(detect_module, "nc") and hasattr(detect_module, "na")):
        raise RuntimeError("YOLOv5 MC-dropout requires a YOLO Detect head as the final module.")
    return detect_module


def _forward_yolov5_features_to_head(model, img):
    modules = model.model
    detect_module = modules[-1]
    if not (hasattr(detect_module, "m") and hasattr(detect_module, "nc") and hasattr(detect_module, "na")):
        raise RuntimeError("YOLOv5 MC-dropout requires a YOLO Detect head as the final module.")
    y = []
    x = img
    for module in modules[:-1]:
        if module.f != -1:
            x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
        x = module(x)
        y.append(x if module.i in model.save else None)
    head_input = y[detect_module.f] if isinstance(detect_module.f, int) else [x if j == -1 else y[j] for j in detect_module.f]
    if isinstance(head_input, torch.Tensor):
        return [head_input.detach()]
    return [feature.detach() for feature in head_input]


def _forward_yolov5_head_from_cache(detector, cached_features):
    features = [feature.clone() for feature in cached_features]
    return detector.model.model[-1](features)


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    num_runs = int(parsed["mc_num_runs"])
    dropout_rate = float(parsed["mc_dropout_rate"])

    if not save_csv:
        return

    # Use the same dataset ordering and aspect-ratio grouping as gt/deterministic.
    # Keep MC-dropout single-process to avoid Windows OpenMP worker conflicts.
    dataloader_config = deepcopy(config)
    dataloader_config.setdefault("dataloader", {})
    dataloader_config["dataloader"]["num_workers"] = 0
    dataloader = create_dataloader(dataloader_config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )
    n_classes_hint = len(detector.names) if detector.names is not None else 80

    output_csv = run_dir / "mc_dropout.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean",
        "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std",
    ]
    for class_idx in range(n_classes_hint):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")

    # Probe once to notify if forced-dropout hooks are unavailable on this model.
    if hasattr(detector, "set_dropout_rate"):
        probe_handles = []
    else:
        probe_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)
        if len(probe_handles) == 0:
            print("[WARN] YOLOv5 detect head not found for forced MC-dropout hooks.")
        for h in probe_handles:
            h.remove()

    write_queue: queue.Queue = queue.Queue()
    writer_thread = threading.Thread(
        target=_mc_dropout_single_csv_writer,
        args=(write_queue, output_csv, fieldnames),
        daemon=True,
    )
    writer_thread.start()

    had_error = False
    try:
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            batch_size = len(image_list)
            image_ids = []
            image_paths = []
            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                image_ids.append(int(target["image_id"][0].item()))
                image_paths.append(target["path"])
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                detector, image_list, device, auto=False
            )
            fcos_preprocessed = (
                detector.preprocess_images(infer_batch)
                if bool(getattr(detector, "is_fcos", False)) and hasattr(detector, "preprocess_images")
                else None
            )
            yolo_cached_features = None
            use_yolo_head_cache = _is_yolov5_detector(detector)

            # 1) Deterministic forward once: get final NMS predictions and raw pre-NMS indices.
            detector_inference_sec = 0.0
            prediction_matching_sec = 0.0
            feature_compute_sec = 0.0

            # Deterministic forward once: get final NMS predictions and raw pre-NMS indices.
            # Count this as detector inference because it is a model forward plus NMS.
            t_detector = timing.start()
            with torch.no_grad():
                roi_cache = None
                if bool(getattr(detector, "is_faster_rcnn", False)) and hasattr(detector, "prepare_roi_cache"):
                    roi_cache = detector.prepare_roi_cache(infer_batch)
                    det_output = detector.forward_from_roi_cache(roi_cache)
                elif fcos_preprocessed is not None and hasattr(detector, "forward_preprocessed"):
                    det_output = detector.forward_preprocessed(fcos_preprocessed)
                elif use_yolo_head_cache:
                    _get_yolov5_detect_module(detector)
                    yolo_cached_features = _forward_yolov5_features_to_head(detector.model, infer_batch)
                    det_output = _forward_yolov5_head_from_cache(detector, yolo_cached_features)
                else:
                    det_output = detector.model(infer_batch, augment=False)
                det_raw_pred = det_output[0] if isinstance(det_output, (tuple, list)) else det_output
                det_raw_logits = det_output[1] if isinstance(det_output, (tuple, list)) and len(det_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=det_raw_pred,
                    logits=det_raw_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )
            detector_inference_sec += timing.elapsed(t_detector)

            feature_runs = []
            n_candidates = None
            n_classes = None
            variable_candidate_runs = False
            if hasattr(detector, "set_dropout_rate"):
                detector.set_dropout_rate(dropout_rate)
                mc_handles = []
            else:
                mc_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)

            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        detector.zero_grad(set_to_none=True)
                        t_detector = timing.start()
                        if roi_cache is not None:
                            model_output = detector.forward_from_roi_cache(roi_cache)
                        elif fcos_preprocessed is not None and hasattr(detector, "forward_preprocessed"):
                            model_output = detector.forward_preprocessed(fcos_preprocessed)
                        elif use_yolo_head_cache:
                            model_output = _forward_yolov5_head_from_cache(detector, yolo_cached_features)
                        else:
                            model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = (
                            model_output[1]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                            else None
                        )
                        detector_inference_sec += timing.elapsed(t_detector)

                        t_feature = timing.start()
                        if isinstance(raw_prediction, list):
                            variable_candidate_runs = True
                            needs_iou_matching = bool(getattr(detector, "is_faster_rcnn", False)) or bool(getattr(detector, "is_fcos", False))
                            run_features = []
                            run_labels = []
                            for pred_img in raw_prediction:
                                pred_img = pred_img.detach().float()
                                bbox_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4])
                                score_vec = pred_img[:, 4:5]
                                prob_mat = get_prediction_class_probs(detector, pred_img).detach().float()
                                run_features.append(torch.cat([bbox_xyxy, score_vec, prob_mat], dim=1))
                                if needs_iou_matching and pred_img.shape[1] > 5:
                                    run_labels.append(pred_img[:, 5].detach().long())
                                else:
                                    run_labels.append(None)
                                if n_classes is None:
                                    n_classes = int(prob_mat.shape[-1])
                            feature_runs.append({"features": run_features, "labels": run_labels} if needs_iou_matching else run_features)
                        else:
                            pred_batch = raw_prediction.detach().float()
                            bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                            score_vec = pred_batch[..., 4].unsqueeze(-1)
                            prob_mat = get_prediction_class_probs(detector, pred_batch).detach().float()
                            if prob_mat.numel() == 0 and raw_logits is not None:
                                prob_mat = torch.sigmoid(raw_logits.detach().float())
                            run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2)

                            if n_candidates is None:
                                n_candidates = int(run_features.shape[1])
                                n_classes = int(run_features.shape[2] - 5)

                            if int(run_features.shape[1]) != n_candidates:
                                raise ValueError("Raw candidate count changed across MC runs; expected fixed pre-NMS candidates.")

                            feature_runs.append(run_features.detach())
                        feature_compute_sec += timing.elapsed(t_feature)
            finally:
                for h in mc_handles:
                    h.remove()
                if hasattr(detector, "set_dropout_rate"):
                    detector.set_dropout_rate(0.0)

            if (not variable_candidate_runs) and n_candidates is None:
                del infer_batch
                if fcos_preprocessed is not None:
                    del fcos_preprocessed
                if yolo_cached_features is not None:
                    del yolo_cached_features
                continue

            feat_mean = None
            feat_std = None
            if not variable_candidate_runs:
                t_feature = timing.start()
                runs_tensor = torch.stack(feature_runs, dim=0)  # [R, B, N, F]
                feat_mean = runs_tensor.mean(dim=0)
                feat_std = runs_tensor.std(dim=0, unbiased=False)
                feature_compute_sec += timing.elapsed(t_feature)
                del runs_tensor
            batch_rows = []
            batch_items = 0

            t_matching = timing.start()
            for b in range(batch_size):
                image_id = image_ids[b]
                image_path = image_paths[b]
                det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = (
                    selected_indices[b]
                    if selected_indices and b < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                num_final = int(det_b.shape[0])
                det_raw_pred_b = (
                    det_raw_pred[b]
                    if isinstance(det_raw_pred, list) and b < len(det_raw_pred)
                    else None
                )

                for pred_idx in range(num_final):
                    raw_idx = (
                        int(raw_keep_b[pred_idx].detach().cpu().item())
                        if pred_idx < int(raw_keep_b.shape[0])
                        else -1
                    )
                    cls_idx = int(det_b[pred_idx, 5].detach().cpu().item()) if det_b.shape[1] > 5 else -1
                    class_count = int(n_classes) if n_classes is not None else int(n_classes_hint)
                    if variable_candidate_runs:
                        per_run_values = []
                        for run_features in feature_runs:
                            if isinstance(run_features, dict):
                                features_by_image = run_features["features"]
                                labels_by_image = run_features["labels"]
                                if b < len(features_by_image):
                                    det_box = det_b[pred_idx, :4].to(features_by_image[b].device)
                                    det_cls = int(det_b[pred_idx, 5].detach().cpu().item()) if det_b.shape[1] > 5 else -1
                                    matched = _match_faster_rcnn_mc_feature(
                                        det_box,
                                        det_cls,
                                        features_by_image[b],
                                        labels_by_image[b] if b < len(labels_by_image) else None,
                                    )
                                    if matched is not None:
                                        per_run_values.append(matched)
                            elif b < len(run_features) and 0 <= raw_idx < int(run_features[b].shape[0]):
                                per_run_values.append(run_features[b][raw_idx])
                        t_feature = timing.start()
                        if per_run_values:
                            run_values = torch.stack(per_run_values, dim=0)
                            mean_vec = run_values.mean(dim=0).detach().float().cpu()
                            std_vec = run_values.std(dim=0, unbiased=False).detach().float().cpu()
                        else:
                            det_vec = _deterministic_mc_feature(
                                detector,
                                det_b[pred_idx],
                                det_raw_pred_b,
                                raw_idx,
                                class_count,
                            )
                            mean_vec = det_vec.detach().float().cpu()
                            std_vec = torch.zeros_like(mean_vec)
                        feature_compute_sec += timing.elapsed(t_feature)
                    else:
                        if 0 <= raw_idx < n_candidates:
                            mean_vec = feat_mean[b, raw_idx].detach().float().cpu()
                            std_vec = feat_std[b, raw_idx].detach().float().cpu()
                        else:
                            det_vec = _deterministic_mc_feature(
                                detector,
                                det_b[pred_idx],
                                det_raw_pred_b,
                                raw_idx,
                                class_count,
                            )
                            mean_vec = det_vec.detach().float().cpu()
                            std_vec = torch.zeros_like(mean_vec)
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": pred_idx,
                        "raw_pred_idx": raw_idx,
                        "xmin": float(det_b[pred_idx, 0].detach().cpu().item()),
                        "ymin": float(det_b[pred_idx, 1].detach().cpu().item()),
                        "xmax": float(det_b[pred_idx, 2].detach().cpu().item()),
                        "ymax": float(det_b[pred_idx, 3].detach().cpu().item()),
                        "score": float(det_b[pred_idx, 4].detach().cpu().item()) if det_b.shape[1] > 4 else 0.0,
                        "pred_class": detector.names[cls_idx] if (detector.names is not None and cls_idx >= 0) else cls_idx,
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
                    for class_idx in range(class_count):
                        row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                        row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
                    batch_rows.append(row)
                batch_items += num_final
            prediction_matching_sec += timing.elapsed(t_matching)

            write_queue.put(batch_rows)
            timing.record(
                num_images=batch_size,
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "prediction_matching_sec": prediction_matching_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )

            del selected_preds, selected_indices
            del feature_runs
            if feat_mean is not None:
                del feat_mean, feat_std
            if "roi_cache" in locals() and roi_cache is not None:
                del roi_cache
            del infer_batch
            if fcos_preprocessed is not None:
                del fcos_preprocessed
            if yolo_cached_features is not None:
                del yolo_cached_features
    except Exception:
        had_error = True
        raise
    finally:
        if had_error:
            write_queue.put(None)
            writer_thread.join()
        else:
            write_queue.put(None)
            writer_thread.join()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_mc_dropout_csv"]
