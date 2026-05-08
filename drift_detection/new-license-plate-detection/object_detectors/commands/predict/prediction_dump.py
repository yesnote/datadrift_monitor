from commands.predict.common import *


def _class_name(names, cls_idx):
    if isinstance(names, dict):
        return names.get(cls_idx, str(cls_idx))
    if isinstance(names, list) and 0 <= cls_idx < len(names):
        return names[cls_idx]
    return str(cls_idx)


def run_prediction_dump_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "prediction_dump"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
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
        "obj",
        "cls_conf",
        "score",
        "pred_class",
        "pred_class_id",
    ]

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
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
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

            t_raw = raw_prof.start()
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = selected_preds[sample_idx]
                raw_keep_b = selected_indices[sample_idx]
                raw_b = raw_prediction[sample_idx].detach().float()
                batch_items += int(det.shape[0])
                total_predictions += int(det.shape[0])

                for pred_idx, box in enumerate(det):
                    raw_pred_idx = (
                        int(raw_keep_b[pred_idx].detach().cpu().item())
                        if pred_idx < int(raw_keep_b.shape[0])
                        else pred_idx
                    )
                    cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else -1
                    xmin = float(box[0].detach().cpu().item())
                    ymin = float(box[1].detach().cpu().item())
                    xmax = float(box[2].detach().cpu().item())
                    ymax = float(box[3].detach().cpu().item())
                    w = max(0.0, xmax - xmin)
                    h = max(0.0, ymax - ymin)
                    area = w * h
                    obj = 0.0
                    cls_conf = 0.0
                    if 0 <= raw_pred_idx < int(raw_b.shape[0]):
                        raw_row = raw_b[raw_pred_idx]
                        obj = float(raw_row[4].detach().cpu().item()) if raw_row.shape[0] > 4 else 1.0
                        if cls_idx >= 0 and raw_row.shape[0] > 5 + cls_idx:
                            cls_conf = float(raw_row[5 + cls_idx].detach().cpu().item())
                    score = float(box[4].detach().cpu().item())
                    if cls_conf == 0.0 and obj > 1e-12:
                        cls_conf = score / obj

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
                            "cx": 0.5 * (xmin + xmax),
                            "cy": 0.5 * (ymin + ymax),
                            "w": w,
                            "h": h,
                            "area": area,
                            "aspect_ratio": w / max(h, 1e-12),
                            "obj": obj,
                            "cls_conf": cls_conf,
                            "score": score,
                            "pred_class": _class_name(detector.names, cls_idx),
                            "pred_class_id": cls_idx,
                        }
                    )
            raw_prof.end(t_raw, batch_items)
            del infer_batch, raw_prediction, raw_logits, selected_preds, selected_indices

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
