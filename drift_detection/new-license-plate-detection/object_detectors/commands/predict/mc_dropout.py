from commands.predict.common import *

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
    queue_maxsize = int(parsed["mc_queue_maxsize"])
    vector_reduction = parsed["mc_vector_reduction"]

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='mc_dropout' requires output.unit in {'image','bbox'}.")

    # Windows OpenMP + subprocess workers can conflict in MC-dropout runs.
    # Force single-process data loading here to avoid libiomp duplicate init crashes.
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

    detector, device = build_detector(config)
    raw_prof = RawComputeProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=unit)
    n_classes_hint = len(detector.names) if detector.names is not None else 80

    output_csv = run_dir / "mc_dropout.csv"
    stat_keys = list(vector_reduction)
    stat_alias = {
        "1-norm": "l1",
        "2-norm": "l2",
        "min": "min",
        "max": "max",
        "mean": "mean",
        "std": "std",
    }

    def stats_from_tensor(vec):
        if vec is None or vec.numel() == 0:
            return {
                "1-norm": 0.0,
                "2-norm": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
            }
        v = vec.detach().float().reshape(-1)
        return {
            "1-norm": float(torch.norm(v, p=1).item()),
            "2-norm": float(torch.norm(v, p=2).item()),
            "min": float(torch.min(v).item()),
            "max": float(torch.max(v).item()),
            "mean": float(torch.mean(v).item()),
            "std": float(torch.std(v, unbiased=False).item()),
        }

    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(
            [
                "pred_idx",
                "raw_pred_idx",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "score",
                "pred_class",
                "xmin_mean",
                "ymin_mean",
                "xmax_mean",
                "ymax_mean",
                "score_mean",
                "xmin_std",
                "ymin_std",
                "xmax_std",
                "ymax_std",
                "score_std",
            ]
        )
        for class_idx in range(n_classes_hint):
            fieldnames.append(f"prob_{class_idx}_mean")
            fieldnames.append(f"prob_{class_idx}_std")
    else:
        fieldnames.append("num_preds")
        for prefix in (
            "xmin_mean",
            "ymin_mean",
            "xmax_mean",
            "ymax_mean",
            "xmin_std",
            "ymin_std",
            "xmax_std",
            "ymax_std",
            "score_mean",
            "score_std",
        ):
            for key in stat_keys:
                fieldnames.append(f"{prefix}_{stat_alias[key]}")
        for class_idx in range(n_classes_hint):
            for key in stat_keys:
                fieldnames.append(f"prob_{class_idx}_mean_{stat_alias[key]}")
            for key in stat_keys:
                fieldnames.append(f"prob_{class_idx}_std_{stat_alias[key]}")

    # Probe once to notify if forced-dropout hooks are unavailable on this model.
    probe_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)
    if len(probe_handles) == 0:
        print("[WARN] YOLOv5 detect head not found for forced MC-dropout hooks.")
    for h in probe_handles:
        h.remove()

    write_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
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
            batch_size = len(images)
            batch_tensors = []
            image_ids = []
            image_paths = []
            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                image_ids.append(int(target["image_id"][0].item()))
                image_paths.append(target["path"])
                infer_tensor, _ratio, _pad, _resized_chw = preprocess_with_letterbox(
                    detector, images[sample_idx], device, requires_grad=False, auto=False
                )
                batch_tensors.append(infer_tensor)

            infer_batch = torch.cat(batch_tensors, dim=0)
            del batch_tensors

            # 1) Deterministic forward once: get final NMS predictions and raw pre-NMS indices.
            with torch.no_grad():
                det_output = detector.model(infer_batch, augment=False)
                det_raw_pred = det_output[0] if isinstance(det_output, (tuple, list)) else det_output
                det_raw_logits = det_output[1] if isinstance(det_output, (tuple, list)) and len(det_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    det_raw_pred,
                    det_raw_logits,
                    detector.confidence,
                    detector.iou_thresh,
                    classes=None,
                    agnostic=detector.agnostic,
                    return_indices=True,
                )

            t_raw = raw_prof.start()
            feat_mean = None
            feat_m2 = None
            n_candidates = None
            n_classes = None
            run_count = 0
            mc_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)

            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        detector.zero_grad(set_to_none=True)
                        model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = (
                            model_output[1]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                            else None
                        )

                        pred_batch = raw_prediction.detach().float()
                        bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                        score_vec = pred_batch[..., 4].unsqueeze(-1)
                        prob_mat = pred_batch[..., 5:].detach().float()
                        if prob_mat.numel() == 0 and raw_logits is not None:
                            prob_mat = torch.sigmoid(raw_logits.detach().float())
                        run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2)

                        if n_candidates is None:
                            n_candidates = int(run_features.shape[1])
                            n_classes = int(run_features.shape[2] - 5)
                            feat_dim = 5 + n_classes
                            feat_mean = torch.zeros((batch_size, n_candidates, feat_dim), device=device)
                            feat_m2 = torch.zeros((batch_size, n_candidates, feat_dim), device=device)

                        if int(run_features.shape[1]) != n_candidates:
                            raise ValueError("Raw candidate count changed across MC runs; expected fixed pre-NMS candidates.")

                        run_count += 1
                        delta = run_features - feat_mean
                        feat_mean = feat_mean + delta / run_count
                        feat_m2 = feat_m2 + delta * (run_features - feat_mean)
            finally:
                for h in mc_handles:
                    h.remove()

            if n_candidates is None:
                del infer_batch
                continue

            feat_std = torch.sqrt(torch.clamp(feat_m2 / max(run_count, 1), min=0.0))
            batch_rows = []
            batch_items = 0

            for b in range(batch_size):
                image_id = image_ids[b]
                image_path = image_paths[b]
                det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = (
                    selected_indices[b]
                    if selected_indices and b < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                feat_mean_cpu = feat_mean[b].detach().float().cpu()
                feat_std_cpu = feat_std[b].detach().float().cpu()
                num_final = int(det_b.shape[0])
                valid_pairs = []
                for pred_idx in range(num_final):
                    raw_idx = int(raw_keep_b[pred_idx].detach().cpu().item())
                    if 0 <= raw_idx < n_candidates:
                        valid_pairs.append((pred_idx, raw_idx))

                if unit == "bbox":
                    for pred_idx, raw_idx in valid_pairs:
                        cls_idx = int(det_b[pred_idx, 5].detach().cpu().item()) if det_b.shape[1] > 5 else -1
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
                            "xmin_mean": float(feat_mean_cpu[raw_idx, 0].item()),
                            "ymin_mean": float(feat_mean_cpu[raw_idx, 1].item()),
                            "xmax_mean": float(feat_mean_cpu[raw_idx, 2].item()),
                            "ymax_mean": float(feat_mean_cpu[raw_idx, 3].item()),
                            "score_mean": float(feat_mean_cpu[raw_idx, 4].item()),
                            "xmin_std": float(feat_std_cpu[raw_idx, 0].item()),
                            "ymin_std": float(feat_std_cpu[raw_idx, 1].item()),
                            "xmax_std": float(feat_std_cpu[raw_idx, 2].item()),
                            "ymax_std": float(feat_std_cpu[raw_idx, 3].item()),
                            "score_std": float(feat_std_cpu[raw_idx, 4].item()),
                        }
                        class_count = int(n_classes) if n_classes is not None else 0
                        for class_idx in range(class_count):
                            row[f"prob_{class_idx}_mean"] = float(feat_mean_cpu[raw_idx, 5 + class_idx].item())
                            row[f"prob_{class_idx}_std"] = float(feat_std_cpu[raw_idx, 5 + class_idx].item())
                        batch_rows.append(row)
                    batch_items += int(len(valid_pairs))
                else:
                    raw_indices = list(range(n_candidates))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": len(raw_indices)}
                    if len(raw_indices) == 0:
                        for prefix in (
                            "xmin_mean",
                            "ymin_mean",
                            "xmax_mean",
                            "ymax_mean",
                            "xmin_std",
                            "ymin_std",
                            "xmax_std",
                            "ymax_std",
                            "score_mean",
                            "score_std",
                        ):
                            for key in stat_keys:
                                row[f"{prefix}_{stat_alias[key]}"] = 0.0
                        for class_idx in range(n_classes_hint):
                            for key in stat_keys:
                                row[f"prob_{class_idx}_mean_{stat_alias[key]}"] = 0.0
                            for key in stat_keys:
                                row[f"prob_{class_idx}_std_{stat_alias[key]}"] = 0.0
                    else:
                        raw_indices_tensor = torch.tensor(raw_indices, dtype=torch.long, device=feat_mean_cpu.device)
                        feat_mean_sel = feat_mean_cpu.index_select(0, raw_indices_tensor)
                        feat_std_sel = feat_std_cpu.index_select(0, raw_indices_tensor)
                        xmin_mean_vec = feat_mean_sel[:, 0].reshape(-1)
                        ymin_mean_vec = feat_mean_sel[:, 1].reshape(-1)
                        xmax_mean_vec = feat_mean_sel[:, 2].reshape(-1)
                        ymax_mean_vec = feat_mean_sel[:, 3].reshape(-1)
                        xmin_std_vec = feat_std_sel[:, 0].reshape(-1)
                        ymin_std_vec = feat_std_sel[:, 1].reshape(-1)
                        xmax_std_vec = feat_std_sel[:, 2].reshape(-1)
                        ymax_std_vec = feat_std_sel[:, 3].reshape(-1)
                        score_mean_vec = feat_mean_sel[:, 4].reshape(-1)
                        score_std_vec = feat_std_sel[:, 4].reshape(-1)

                        for key, val in stats_from_tensor(xmin_mean_vec).items():
                            row[f"xmin_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymin_mean_vec).items():
                            row[f"ymin_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmax_mean_vec).items():
                            row[f"xmax_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymax_mean_vec).items():
                            row[f"ymax_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmin_std_vec).items():
                            row[f"xmin_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymin_std_vec).items():
                            row[f"ymin_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmax_std_vec).items():
                            row[f"xmax_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymax_std_vec).items():
                            row[f"ymax_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(score_mean_vec).items():
                            row[f"score_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(score_std_vec).items():
                            row[f"score_std_{stat_alias[key]}"] = val

                        class_count = int(n_classes) if n_classes is not None else 0
                        for class_idx in range(n_classes_hint):
                            if class_idx < class_count:
                                prob_mean_vec = feat_mean_sel[:, 5 + class_idx].reshape(-1)
                                prob_std_vec = feat_std_sel[:, 5 + class_idx].reshape(-1)
                            else:
                                prob_mean_vec = torch.zeros((0,), dtype=torch.float32, device=feat_mean_cpu.device)
                                prob_std_vec = torch.zeros((0,), dtype=torch.float32, device=feat_mean_cpu.device)
                            for key, val in stats_from_tensor(prob_mean_vec).items():
                                row[f"prob_{class_idx}_mean_{stat_alias[key]}"] = val
                            for key, val in stats_from_tensor(prob_std_vec).items():
                                row[f"prob_{class_idx}_std_{stat_alias[key]}"] = val
                    batch_rows.append(row)
                    batch_items += 1

            write_queue.put(batch_rows)
            raw_prof.end(t_raw, batch_items)

            del selected_preds, selected_indices
            del infer_batch
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
    timing_csv, timing_json = raw_prof.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved raw compute timing: {timing_csv}")
    print(f"Saved raw compute timing summary: {timing_json}")

__all__ = ["run_mc_dropout_csv"]
