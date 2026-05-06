from commands.predict.common import *

def run_score_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "score"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_vector_reduction = parsed["score_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return

    output_csv = run_dir / "score.csv"
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
            ]
        )
    else:
        fieldnames.extend(score_vector_reduction)
        fieldnames.append("num_preds")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    raw_prof = RawComputeProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=unit)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            raw_prediction = None
            raw_logits = None
            selected_preds = None
            selected_indices = None
            with torch.no_grad():
                if unit == "bbox":
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                    raw_logits = (
                        model_output[1]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 1
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
                else:
                    preds, _logits, _objectness, _features = detector(infer_batch)
                    if pre_nms:
                        model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output

            t_raw = raw_prof.start()
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                if unit == "bbox":
                    det = selected_preds[sample_idx]
                    batch_items += int(det.shape[0])
                    raw_keep_b = selected_indices[sample_idx]
                    for pred_idx, box in enumerate(det):
                        raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                        cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                        if isinstance(detector.names, dict):
                            pred_class = detector.names.get(cls_idx, str(cls_idx))
                        elif isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
                            pred_class = detector.names[cls_idx]
                        else:
                            pred_class = str(cls_idx)
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
                                "score": float(box[4]),
                                "pred_class": pred_class,
                            }
                        )
                else:
                    pred_scores = preds[3][sample_idx]
                    if pre_nms and raw_prediction is not None:
                        pre = raw_prediction[sample_idx].detach().float()
                        if pre.numel() == 0:
                            score_tensor = torch.zeros((0,), dtype=torch.float32, device=device)
                        else:
                            obj = pre[:, 4]
                            cls_max = pre[:, 5:].max(dim=1).values if pre.shape[1] > 5 else torch.ones_like(obj)
                            score_tensor = obj * cls_max
                            keep_idx = get_pre_nms_keep_indices(pre, pre_nms_ratio=pre_nms_ratio)
                            if int(keep_idx.shape[0]) > 0:
                                score_tensor = score_tensor[keep_idx]
                            else:
                                score_tensor = torch.zeros((0,), dtype=torch.float32, device=device)
                        num_preds = int(score_tensor.shape[0])
                    else:
                        score_tensor = torch.as_tensor(pred_scores, dtype=torch.float32, device=device)
                        num_preds = len(pred_scores)
                    batch_items += int(num_preds)

                    if num_preds == 0:
                        stat_all = {
                            "1-norm": 0.0,
                            "2-norm": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    else:
                        stat_all = map_grad_tensor_to_numbers(score_tensor.reshape(-1))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": num_preds}
                    for metric_name in score_vector_reduction:
                        row[metric_name] = float(stat_all[metric_name])
                    writer.writerow(row)
            raw_prof.end(t_raw, batch_items)
            if unit == "bbox":
                del infer_batch, raw_prediction, raw_logits, selected_preds, selected_indices
            else:
                del infer_batch, preds, _logits, _objectness, _features, raw_prediction

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = raw_prof.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved raw compute timing: {timing_csv}")
    print(f"Saved raw compute timing summary: {timing_json}")

__all__ = ["run_score_csv"]
