import csv
import json

import cv2
import torch
from tqdm import tqdm

from dataloaders.dataloader_yolo import create_dataloader
from modes.utils.predict_utils import (
    build_detector,
    collect_gradients_per_target,
    draw_predictions,
    get_annotation_path,
    has_fn_for_image,
    load_coco_category_maps,
    map_boxes_to_letterbox,
    parse_output_config,
    preprocess_with_letterbox,
    register_layer_hooks,
)


def run_predict(config, run_dir):
    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    output_cfg = config.get("output", {})
    parsed = parse_output_config(output_cfg)

    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["iou_match_threshold"]
    target_values = parsed["target_values"]
    target_layers = parsed["target_layers"]
    save_image = parsed["save_image_enabled"]
    image_step = parsed["image_step"]
    image_num = parsed["image_num"]
    compute_grads = save_csv and bool(target_layers)

    output_csv = run_dir / "fn_results.csv"
    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError(
            "Loaded 0 images. Check dataset root/image_dir/split configuration in YAML."
        )

    detector, device = build_detector(config)
    activations, hook_handles = register_layer_hooks(detector.model, target_layers) if target_layers else ({}, [])

    fieldnames = ["image_id", "image_path", "has_fn"]
    if compute_grads:
        for target_value in target_values:
            for layer_name in target_layers:
                fieldnames.append(f"d{target_value}_d{layer_name}")

    total_images = 0
    fn_images = 0

    writer = None
    file_handle = None
    if save_csv:
        file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for step_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Predict ({split})", total=len(dataloader))
        ):
            batch_size = images.shape[0]
            should_save_step = save_image and (step_idx % image_step == 0)
            step_dir = run_dir / "images" / f"0_{step_idx}"
            if should_save_step:
                step_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx in range(batch_size):
                detector.zero_grad(set_to_none=True)
                activations.clear()
                infer_tensor, ratio, pad, resized_chw = preprocess_with_letterbox(
                    detector, images[sample_idx], device, requires_grad=False
                )
                with torch.no_grad():
                    preds, _logits, _objectness, _features = detector(infer_tensor)

                target = targets[sample_idx]
                row = {
                    "image_id": int(target["image_id"][0].item()),
                    "image_path": target["path"],
                }

                pred_boxes = preds[0][0]
                pred_class_names = preds[2][0]
                pred_scores = preds[3][0]
                gt_boxes_tensor = target["boxes"]
                gt_labels_tensor = target["labels"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratio, pad)
                gt_class_names = [
                    catid_to_name.get(int(label), "__unknown__")
                    for label in gt_labels_tensor.tolist()
                ]
                has_fn = has_fn_for_image(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    iou_match_threshold=iou_match_threshold,
                )
                row["has_fn"] = has_fn
                fn_images += int(has_fn)

                if compute_grads:
                    grad_stats = collect_gradients_per_target(
                        detector=detector,
                        input_tensor=infer_tensor,
                        target_values=target_values,
                        target_layers=target_layers,
                        activations=activations,
                    )
                    for key, stats in grad_stats.items():
                        row[key] = json.dumps(stats, separators=(",", ":"))

                if writer is not None:
                    writer.writerow(row)

                if should_save_step and sample_idx < image_num:
                    vis_image = draw_predictions(
                        resized_chw,
                        pred_boxes,
                        pred_class_names,
                        pred_scores,
                    )
                    out_path = step_dir / f"{row['image_id']}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                total_images += 1

                del infer_tensor, preds, _logits, _objectness, _features
                if compute_grads:
                    del grad_stats
                activations.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if file_handle is not None:
            file_handle.close()

    for handle in hook_handles:
        handle.remove()

    summary = {
        "mode": "predict",
        "cue": parsed["cue"],
        "split": split,
        "save_csv": save_csv,
        "save_image": save_image,
        "save_image_step": image_step,
        "save_image_num": image_num,
        "target_values": target_values,
        "target_layers": target_layers,
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv) if save_csv else "",
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
