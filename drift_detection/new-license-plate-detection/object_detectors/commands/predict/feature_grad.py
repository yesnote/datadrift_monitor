from commands.predict.common import *

def run_feature_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "feature_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))

    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = parsed["target_values"]
    target_layers = parsed["target_layers"]
    feature_map_reduction = parsed["feature_map_reduction"]
    feature_vector_reduction = parsed["feature_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return

    output_csv = run_dir / "feature_grad.csv"
    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(["pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"])
    for target_value in target_values:
        for layer_name in target_layers:
            fieldnames.append(f"{target_value}_{layer_name}")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    layer_buffer = None
    if unit == "bbox":
        layer_buffer = create_layer_grad_buffer(
            detector.model,
            target_layers,
            map_reduction=feature_map_reduction,
            vector_reduction=feature_vector_reduction,
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for images, targets in tqdm(
                dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
            ):
                image_list = _as_image_list(images)
                infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
                batch_grad_stats = None
                if unit != "bbox":
                    batch_grad_stats = collect_batch_feature_gradients_per_target(
                        detector=detector,
                        input_tensor=infer_batch,
                        target_values=target_values,
                        target_layers=target_layers,
                        map_reduction=feature_map_reduction,
                        vector_reduction=feature_vector_reduction,
                        pre_nms=pre_nms,
                        pre_nms_ratio=pre_nms_ratio,
                    )
                for sample_idx in range(len(image_list)):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    if unit == "bbox":
                        infer_tensor = infer_batch[sample_idx: sample_idx + 1]
                        bbox_rows = collect_bbox_gradients_per_target(
                            detector=detector,
                            input_tensor=infer_tensor,
                            target_values=target_values,
                            target_layers=target_layers,
                            layer_buffer=layer_buffer,
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
                            writer.writerow(row)
                        del bbox_rows
                    else:
                        grad_stats = batch_grad_stats[sample_idx]
                        row = {"image_id": image_id, "image_path": image_path}
                        for grad_key, grad_value in grad_stats.items():
                            row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                        writer.writerow(row)
                        del grad_stats
                if batch_grad_stats is not None:
                    del batch_grad_stats
                del infer_batch
        finally:
            if layer_buffer is not None:
                layer_buffer.remove()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")

__all__ = ["run_feature_grad_csv"]
