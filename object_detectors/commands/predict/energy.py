from object_detectors.commands.predict.common import *

def run_energy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "energy"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    energy_vector_reduction = parsed["energy_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='energy' requires output.unit in {'image','bbox'}.")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "energy.csv"
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
                "energy",
            ]
        )
    else:
        fieldnames.extend(energy_vector_reduction)
        fieldnames.append("num_preds")

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
                    preds, logits, _objectness, _features = detector(infer_batch)
                    if pre_nms:
                        model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = (
                            model_output[1]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                            else None
                        )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                if unit == "bbox":
                    det = selected_preds[sample_idx]
                    raw_keep_b = selected_indices[sample_idx]
                    if raw_logits is not None:
                        selected_logits = (
                            raw_logits[sample_idx][raw_keep_b]
                            if int(raw_keep_b.shape[0]) > 0
                            else torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        )
                    else:
                        selected_logits = (
                            raw_prediction[sample_idx][raw_keep_b, 5:]
                            if int(raw_keep_b.shape[0]) > 0 and raw_prediction[sample_idx].shape[1] > 5
                            else torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        )
                    selected_probs = torch.softmax(selected_logits, dim=-1) if selected_logits.numel() else selected_logits
                    if selected_probs.numel():
                        probs_clipped = selected_probs.clamp(min=1e-8, max=1.0 - 1e-8)
                        pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
                        pred_energy = -100.0 * torch.log(
                            torch.clamp(
                                torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1),
                                min=1e-8,
                            )
                        )
                    else:
                        pred_energy = torch.zeros((0,), device=device)
                    for pred_idx, box in enumerate(det):
                        raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                        cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                        if isinstance(detector.names, dict):
                            pred_class = detector.names.get(cls_idx, str(cls_idx))
                        elif isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
                            pred_class = detector.names[cls_idx]
                        else:
                            pred_class = str(cls_idx)
                        energy_val = (
                            float(pred_energy[pred_idx].detach().cpu().item())
                            if pred_idx < pred_energy.shape[0]
                            else 0.0
                        )
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
                                "energy": energy_val,
                            }
                        )
                else:
                    pred_logits = logits[sample_idx] if logits else torch.zeros((0, num_classes), device=device)
                    pred_probs = torch.softmax(pred_logits, dim=-1) if pred_logits.numel() else pred_logits
                    if pred_probs.numel():
                        probs_clipped = pred_probs.clamp(min=1e-8, max=1.0 - 1e-8)
                        pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
                        pred_energy = -100.0 * torch.log(
                            torch.clamp(
                                torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1),
                                min=1e-8,
                            )
                        )
                    else:
                        pred_energy = torch.zeros((0,), device=device)
                    if pre_nms and raw_prediction is not None:
                        if raw_logits is not None:
                            pre_logits = raw_logits[sample_idx].detach().float()
                            pre_probs = torch.softmax(pre_logits, dim=-1) if pre_logits.numel() else pre_logits
                            keep_idx = get_pre_nms_keep_indices(
                                raw_prediction[sample_idx].detach().float(),
                                pre_logits,
                                pre_nms_ratio=pre_nms_ratio,
                            )
                        else:
                            pre_raw = raw_prediction[sample_idx].detach().float()
                            cls_scores = (
                                pre_raw[:, 5:]
                                if pre_raw.shape[1] > 5
                                else torch.zeros((pre_raw.shape[0], num_classes), device=device)
                            )
                            pre_probs = torch.softmax(cls_scores, dim=-1) if cls_scores.numel() else cls_scores
                            keep_idx = get_pre_nms_keep_indices(pre_raw, pre_nms_ratio=pre_nms_ratio)
                        if int(keep_idx.shape[0]) > 0:
                            pre_probs = pre_probs[keep_idx]
                        else:
                            pre_probs = torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        if pre_probs.numel():
                            probs_clipped = pre_probs.clamp(min=1e-8, max=1.0 - 1e-8)
                            pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
                            energy_tensor = -100.0 * torch.log(
                                torch.clamp(
                                    torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1),
                                    min=1e-8,
                                )
                            )
                        else:
                            energy_tensor = torch.zeros((0,), device=device)
                    else:
                        energy_tensor = pred_energy

                    num_preds = int(energy_tensor.shape[0])
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
                        stat_all = map_grad_tensor_to_numbers(energy_tensor.detach().float().reshape(-1))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": num_preds}
                    for metric_name in energy_vector_reduction:
                        row[metric_name] = float(stat_all[metric_name])
                    writer.writerow(row)
            if unit == "bbox":
                del infer_batch, raw_prediction, raw_logits, selected_preds, selected_indices
            else:
                del infer_batch, preds, logits, _objectness, _features, raw_prediction, raw_logits

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")

__all__ = ["run_energy_csv"]
