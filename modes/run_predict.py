import csv
import json

import torch

from dataloaders.dataloader_yolo import create_dataloader
from modes.utils.predict_utils import (
    build_detector,
    collect_gradients,
    get_annotation_path,
    has_fn_for_image,
    load_coco_category_maps,
    map_boxes_to_letterbox,
    parse_grad_config,
    preprocess_with_letterbox,
    register_layer_hooks,
)


def run_predict(config, run_dir):
    predict_cfg = config.get("predict", {})
    split = predict_cfg.get("split", "val")
    save_fn = bool(predict_cfg.get("save_fn", True))
    iou_match_threshold, target_values, target_layers = parse_grad_config(predict_cfg)

    output_csv = run_dir / "fn_results.csv"
    grad_dir = run_dir / "grad_tensors"
    grad_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError(
            "Loaded 0 images. Check dataset root/image_dir/split configuration in YAML."
        )
    detector, device = build_detector(config)
    activations, hook_handles = register_layer_hooks(detector.model, target_layers)

    rows = []
    for images, targets in dataloader:
        if images.shape[0] != 1:
            raise ValueError("For predict export, dataloader batch_size must be 1.")

        detector.zero_grad(set_to_none=True)
        input_tensor, ratio, pad = preprocess_with_letterbox(detector, images[0], device)
        preds, logits, objectness, _features = detector(input_tensor)

        target = targets[0]
        row = {
            "image_id": int(target["image_id"][0].item()),
            "image_path": target["path"],
        }

        if save_fn:
            pred_boxes = preds[0][0]
            pred_class_names = preds[2][0]
            gt_boxes_tensor = target["boxes"]
            gt_labels_tensor = target["labels"]
            gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratio, pad)
            gt_class_names = [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]
            row["has_fn"] = has_fn_for_image(
                gt_boxes=gt_boxes,
                gt_class_names=gt_class_names,
                pred_boxes=pred_boxes,
                pred_class_names=pred_class_names,
                iou_match_threshold=iou_match_threshold,
            )

        grad_tensors = collect_gradients(
            target_values=target_values,
            target_layers=target_layers,
            preds=preds,
            logits=logits,
            objectness=objectness,
            activations=activations,
        )

        grad_path = grad_dir / f"{row['image_id']}.pt"
        torch.save(grad_tensors, grad_path)

        rows.append(row)

    for handle in hook_handles:
        handle.remove()

    if save_fn:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_id", "image_path", "has_fn"],
            )
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "mode": "predict",
        "cue": "grad",
        "split": split,
        "save_fn": save_fn,
        "target_values": target_values,
        "target_layers": target_layers,
        "total_images": len(rows),
        "grad_dir": str(grad_dir),
    }
    if save_fn:
        fn_images = sum(r["has_fn"] for r in rows)
        summary["fn_images"] = fn_images
        summary["fn_ratio"] = (fn_images / len(rows)) if rows else 0.0
        summary["output_csv"] = str(output_csv)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
