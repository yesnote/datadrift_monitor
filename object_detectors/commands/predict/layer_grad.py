from commands.predict.common import *


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = [str(v) for v in parsed["layer_target_values"]]
    target_layers = parsed["layer_target_layers"]
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_pseudo_gt = parsed.get("layer_pseudo_gt", "cand")
    layer_cand_score_threshold = float(parsed.get("layer_cand_score_threshold", 0.01))

    if not save_csv:
        return

    output_csv = run_dir / "layer_grad.csv"

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "candidate_search_sec", "loss_compute_sec", "backpropagation_sec"],
        device=device,
    )
    target_layers = expand_layer_names(detector.model, target_layers)

    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
    ]
    for target_value in target_values:
        for layer_name in target_layers:
            fieldnames.append(f"{target_value}_{layer_name}")

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file_handle:
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            stage_seconds = {
                "detector_inference_sec": 0.0,
                "candidate_search_sec": 0.0,
                "loss_compute_sec": 0.0,
                "backpropagation_sec": 0.0,
            }
            batch_items = 0

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                bbox_rows = collect_bbox_layer_grads_per_target(
                    detector=detector,
                    input_tensor=infer_batch[sample_idx: sample_idx + 1],
                    target_values=target_values,
                    target_layers=target_layers,
                    map_reduction=layer_map_reduction,
                    vector_reduction=[],
                    pseudo_gt=layer_pseudo_gt,
                    cand_score_threshold=layer_cand_score_threshold,
                    bbox_loss="ciou",
                    timing_accumulator=stage_seconds,
                    timing_device=device,
                )
                for bbox_row in bbox_rows:
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": bbox_row["pred_idx"],
                        "raw_pred_idx": bbox_row["raw_pred_idx"],
                        "xmin": bbox_row["xmin"],
                        "ymin": bbox_row["ymin"],
                        "xmax": bbox_row["xmax"],
                        "ymax": bbox_row["ymax"],
                        "score": bbox_row["score"],
                        "pred_class": bbox_row["pred_class"],
                    }
                    for grad_key, grad_value in bbox_row["grad_stats"].items():
                        row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                    csv_writer.writerow(row)
                batch_items += int(len(bbox_rows))
                del bbox_rows

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds=stage_seconds,
            )
            del infer_batch

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_layer_grad_csv"]
