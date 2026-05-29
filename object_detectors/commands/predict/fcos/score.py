from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms, unpack_fcos_model_output

def run_score_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "score"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]

    if not save_csv:
        return

    output_csv = run_dir / "score.csv"
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
    ]

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
            raw_indices = None
            selected_preds = None
            selected_indices = None
            t_detector = timing.start()
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction, raw_logits, raw_indices = unpack_fcos_model_output(model_output)
                selected_preds, _selected_logits, _selected_objectness, selected_indices = select_fcos_post_nms(
                    detector, raw_prediction, raw_logits, raw_indices
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
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={"detector_inference_sec": detector_inference_sec},
            )
            del infer_batch, raw_prediction, raw_logits, raw_indices, selected_preds, selected_indices

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_score_csv"]
