from object_detectors.commands.predict.common import *

def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    vector_reduction = parsed["ensemble_vector_reduction"]

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='ensemble' requires output.unit in {'image','bbox'}.")

    model_cfg = config.get("model", {})
    weights_cfg = model_cfg.get("weights", [])
    if isinstance(weights_cfg, str):
        weight_paths = [weights_cfg]
    elif isinstance(weights_cfg, (list, tuple)):
        weight_paths = [str(w) for w in weights_cfg if str(w).strip()]
    else:
        weight_paths = []
    if not weight_paths:
        raise ValueError("output.uncertainty='ensemble' requires model.weights to be a non-empty string/list.")

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
    temp_dir = run_dir / "_ensemble_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

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

    def load_state(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    n_classes_hint = None
    class_names_hint = None
    n_classes_actual = None
    device = torch.device("cpu")
    had_error = False
    try:
        for weight_idx, model_weight in enumerate(weight_paths):
            detector, device = build_detector(config, model_weight=model_weight)
            if n_classes_hint is None:
                n_classes_hint = len(detector.names) if detector.names is not None else 80
                class_names_hint = detector.names

            for batch_idx, (images, targets) in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Object Detector ({mode} - {uncertainty}) [{weight_idx + 1}/{len(weight_paths)}]",
                    total=len(dataloader),
                )
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

                pred_batch = det_raw_pred.detach().float()
                bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                score_vec = pred_batch[..., 4].unsqueeze(-1)
                prob_mat = pred_batch[..., 5:].detach().float()
                if prob_mat.numel() == 0 and det_raw_logits is not None:
                    prob_mat = torch.sigmoid(det_raw_logits.detach().float())
                run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2).detach().cpu()
                class_count = int(run_features.shape[2] - 5)
                if n_classes_actual is None:
                    n_classes_actual = class_count
                elif n_classes_actual != class_count:
                    raise ValueError(
                        f"All ensemble weights must have the same class count: {n_classes_actual} vs {class_count}."
                    )

                state_path = temp_dir / f"batch_{batch_idx:06d}.pt"
                if weight_idx == 0:
                    det_boxes_cpu = []
                    raw_keep_cpu = []
                    for b in range(batch_size):
                        det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                        raw_keep_b = (
                            selected_indices[b]
                            if selected_indices and b < len(selected_indices)
                            else torch.zeros((0,), dtype=torch.long, device=device)
                        )
                        det_boxes_cpu.append(det_b.detach().cpu())
                        raw_keep_cpu.append([int(v) for v in raw_keep_b.detach().cpu().tolist()])

                    state = {
                        "count": 1,
                        "mean": run_features,
                        "m2": torch.zeros_like(run_features),
                        "image_ids": image_ids,
                        "image_paths": image_paths,
                        "det_boxes": det_boxes_cpu,
                        "raw_keep_indices": raw_keep_cpu,
                    }
                else:
                    state = load_state(state_path)
                    if list(state.get("image_ids", [])) != image_ids or list(state.get("image_paths", [])) != image_paths:
                        raise ValueError(
                            "Data order mismatch across ensemble passes. Set dataloader.shuffle_eval=false for predict mode."
                        )
                    mean = state["mean"]
                    m2 = state["m2"]
                    count = int(state["count"])
                    if tuple(mean.shape) != tuple(run_features.shape):
                        raise ValueError(
                            f"Candidate tensor shape mismatch across ensemble weights: {tuple(mean.shape)} vs {tuple(run_features.shape)}."
                        )
                    count_new = count + 1
                    delta = run_features - mean
                    mean = mean + delta / count_new
                    delta2 = run_features - mean
                    m2 = m2 + delta * delta2
                    state["count"] = count_new
                    state["mean"] = mean
                    state["m2"] = m2

                torch.save(state, state_path)
                del infer_batch, det_raw_pred, det_raw_logits, selected_preds, selected_indices, run_features, state

            del detector
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if n_classes_hint is None:
            n_classes_hint = 80
        if n_classes_actual is None:
            n_classes_actual = n_classes_hint

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

        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for batch_idx in range(len(dataloader)):
                state_path = temp_dir / f"batch_{batch_idx:06d}.pt"
                state = load_state(state_path)
                count = int(state["count"])
                mean = state["mean"].detach().float()
                m2 = state["m2"].detach().float()
                std = torch.sqrt(torch.clamp(m2 / max(count, 1), min=0.0))

                image_ids = state["image_ids"]
                image_paths = state["image_paths"]
                det_boxes = state["det_boxes"]
                raw_keep_indices = state["raw_keep_indices"]
                for b in range(len(image_ids)):
                    image_id = int(image_ids[b])
                    image_path = str(image_paths[b])
                    mean_b = mean[b]
                    std_b = std[b]
                    n_candidates = int(mean_b.shape[0])

                    if unit == "bbox":
                        det_b = det_boxes[b]
                        raw_keep_b = [int(v) for v in raw_keep_indices[b]]
                        num_final = int(det_b.shape[0])
                        for pred_idx in range(num_final):
                            if pred_idx >= len(raw_keep_b):
                                continue
                            raw_idx = int(raw_keep_b[pred_idx])
                            if raw_idx < 0 or raw_idx >= n_candidates:
                                continue
                            cls_idx = int(det_b[pred_idx, 5].item()) if det_b.shape[1] > 5 else -1
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
                                "xmin_mean": float(mean_b[raw_idx, 0].item()),
                                "ymin_mean": float(mean_b[raw_idx, 1].item()),
                                "xmax_mean": float(mean_b[raw_idx, 2].item()),
                                "ymax_mean": float(mean_b[raw_idx, 3].item()),
                                "score_mean": float(mean_b[raw_idx, 4].item()),
                                "xmin_std": float(std_b[raw_idx, 0].item()),
                                "ymin_std": float(std_b[raw_idx, 1].item()),
                                "xmax_std": float(std_b[raw_idx, 2].item()),
                                "ymax_std": float(std_b[raw_idx, 3].item()),
                                "score_std": float(std_b[raw_idx, 4].item()),
                            }
                            for class_idx in range(n_classes_hint):
                                if class_idx < n_classes_actual:
                                    row[f"prob_{class_idx}_mean"] = float(mean_b[raw_idx, 5 + class_idx].item())
                                    row[f"prob_{class_idx}_std"] = float(std_b[raw_idx, 5 + class_idx].item())
                                else:
                                    row[f"prob_{class_idx}_mean"] = 0.0
                                    row[f"prob_{class_idx}_std"] = 0.0
                            writer.writerow(row)
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
                            raw_indices_tensor = torch.tensor(raw_indices, dtype=torch.long)
                            feat_mean_sel = mean_b.index_select(0, raw_indices_tensor)
                            feat_std_sel = std_b.index_select(0, raw_indices_tensor)

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

                            for class_idx in range(n_classes_hint):
                                if class_idx < n_classes_actual:
                                    prob_mean_vec = feat_mean_sel[:, 5 + class_idx].reshape(-1)
                                    prob_std_vec = feat_std_sel[:, 5 + class_idx].reshape(-1)
                                else:
                                    prob_mean_vec = torch.zeros((0,), dtype=torch.float32)
                                    prob_std_vec = torch.zeros((0,), dtype=torch.float32)
                                for key, val in stats_from_tensor(prob_mean_vec).items():
                                    row[f"prob_{class_idx}_mean_{stat_alias[key]}"] = val
                                for key, val in stats_from_tensor(prob_std_vec).items():
                                    row[f"prob_{class_idx}_std_{stat_alias[key]}"] = val
                        writer.writerow(row)
                del state, mean, m2, std
    except Exception:
        had_error = True
        raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if had_error and device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved results CSV: {output_csv}")

__all__ = ["run_ensemble_csv"]
