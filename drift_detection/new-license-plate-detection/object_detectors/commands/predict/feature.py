from object_detectors.commands.predict.common import *

def run_feature_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "feature"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_layers = parsed["feature_target_layers"]
    map_reduction = parsed["feature_map_reduction"]
    vector_reduction = parsed["feature_vector_reduction"]

    if not save_csv:
        return
    if unit != "image":
        raise ValueError("output.uncertainty='feature' currently supports only output.unit='image'.")

    output_csv = run_dir / "feature.csv"
    fieldnames = ["image_id", "image_path"] + target_layers

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                infer_tensor = infer_batch[sample_idx: sample_idx + 1]
                feature_stats = collect_image_features_per_layer(
                    detector=detector,
                    input_tensor=infer_tensor,
                    target_layers=target_layers,
                    map_reduction=map_reduction,
                    vector_reduction=vector_reduction,
                )
                row = {"image_id": image_id, "image_path": image_path}
                for layer_name in target_layers:
                    row[layer_name] = json.dumps(feature_stats[layer_name], separators=(",", ":"))
                writer.writerow(row)
            del infer_batch

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")

__all__ = ["run_feature_csv"]
