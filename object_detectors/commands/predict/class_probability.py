from commands.predict.common import *

def run_class_probability_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "class_probability"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]

    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec"],
        device=device,
    )
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "class_probability.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "score",
        "pred_class",
    ] + [f"prob_{i}" for i in range(num_classes)]

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
            t_detector = timing.start()
            with torch.no_grad():
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
            detector_inference_sec = timing.elapsed(t_detector)

            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                det = selected_preds[sample_idx]
                batch_items += int(det.shape[0])
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
                pred_probs = torch.softmax(selected_logits, dim=-1) if selected_logits.numel() else selected_logits
                for pred_idx, box in enumerate(det):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                    if isinstance(detector.names, dict):
                        pred_class = detector.names.get(cls_idx, str(cls_idx))
                    elif isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
                        pred_class = detector.names[cls_idx]
                    else:
                        pred_class = str(cls_idx)
                    row = {
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
                    probs = pred_probs[pred_idx].detach().cpu().tolist() if pred_idx < pred_probs.shape[0] else [0.0] * num_classes
                    for class_idx in range(num_classes):
                        row[f"prob_{class_idx}"] = float(probs[class_idx]) if class_idx < len(probs) else 0.0
                    writer.writerow(row)
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={"detector_inference_sec": detector_inference_sec},
            )
            del infer_batch, raw_prediction, raw_logits, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_class_probability_csv"]
