import csv
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from dataloaders.dataloader_yolo import create_dataloader
from commands.utils.predict_utils import (
    assign_tp_to_predictions,
    build_detector,
    collect_bbox_gradients_per_target,
    collect_bbox_layer_grads_per_target,
    collect_gradients_per_target,
    create_layer_grad_buffer,
    draw_predictions,
    get_annotation_path,
    has_fn_for_image,
    load_coco_category_maps,
    map_boxes_to_letterbox,
    parse_output_config,
    preprocess_with_letterbox,
)


def _build_summary(total_images, fn_images, output_csv):
    return {
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv),
    }


def run_fn_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    cue = "fn"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["iou_match_threshold"]
    save_image = parsed["save_image_enabled"]
    image_step = parsed["image_step"]
    image_max_num = parsed["image_max_num"]

    output_csv = run_dir / "fn.csv"
    summary_json = run_dir / "summary.json"

    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    total_images = 0
    fn_images = 0
    csv_writer = None
    csv_file_handle = None
    if save_csv:
        csv_file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=["image_id", "image_path", "fn"])
        csv_writer.writeheader()

    try:
        for step_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Object Detector ({mode} - {cue})", total=len(dataloader))
        ):
            batch_size = images.shape[0]
            should_save_step = save_image and (step_idx % image_step == 0)
            step_dir = run_dir / "images" / f"0_{step_idx}"
            if should_save_step:
                step_dir.mkdir(parents=True, exist_ok=True)

            for sample_idx in range(batch_size):
                detector.zero_grad(set_to_none=True)
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
                gt_class_names = [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]
                fn = has_fn_for_image(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    iou_match_threshold=iou_match_threshold,
                )
                row["fn"] = fn
                fn_images += int(fn)

                if csv_writer is not None:
                    csv_writer.writerow(row)

                if should_save_step and sample_idx < image_max_num:
                    vis_image = draw_predictions(resized_chw, pred_boxes, pred_class_names, pred_scores)
                    out_path = step_dir / f"{row['image_id']}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                total_images += 1
                del infer_tensor, preds, _logits, _objectness, _features
    finally:
        if csv_file_handle is not None:
            csv_file_handle.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = _build_summary(total_images, fn_images, output_csv if save_csv else "")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    print(f"Saved summary: {summary_json}")


def run_feature_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    cue = "feature_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))

    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = parsed["target_values"]
    target_layers = parsed["target_layers"]

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
    layer_buffer = create_layer_grad_buffer(detector.model, target_layers)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for images, targets in tqdm(
                dataloader, desc=f"Object Detector ({mode} - {cue})", total=len(dataloader)
            ):
                for sample_idx in range(images.shape[0]):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]

                    infer_tensor, _ratio, _pad, _resized_chw = preprocess_with_letterbox(
                        detector, images[sample_idx], device, requires_grad=False
                    )
                    if unit == "bbox":
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
                        del infer_tensor, bbox_rows
                    else:
                        grad_stats = collect_gradients_per_target(
                            detector=detector,
                            input_tensor=infer_tensor,
                            target_values=target_values,
                            target_layers=target_layers,
                            layer_buffer=layer_buffer,
                        )

                        row = {"image_id": image_id, "image_path": image_path}
                        for grad_key, grad_value in grad_stats.items():
                            row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                        writer.writerow(row)
                        del infer_tensor, grad_stats
        finally:
            layer_buffer.remove()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    cue = "tp"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["tp_iou_match_threshold"]

    if not save_csv:
        return

    output_csv = run_dir / "tp.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "score",
        "pred_class",
        "max_iou",
        "tp",
    ]

    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {cue})", total=len(dataloader)
        ):
            for sample_idx in range(images.shape[0]):
                detector.zero_grad(set_to_none=True)
                infer_tensor, ratio, pad, _resized_chw = preprocess_with_letterbox(
                    detector, images[sample_idx], device, requires_grad=False
                )
                with torch.no_grad():
                    preds, _logits, _objectness, _features = detector(infer_tensor)

                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                pred_boxes = preds[0][0]
                pred_class_names = preds[2][0]
                pred_scores = preds[3][0]
                gt_boxes_tensor = target["boxes"]
                gt_labels_tensor = target["labels"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratio, pad)
                gt_class_names = [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]

                tp_flags, best_ious = assign_tp_to_predictions(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=iou_match_threshold,
                )

                for pred_idx, (box, score, pred_class) in enumerate(
                    zip(pred_boxes, pred_scores, pred_class_names)
                ):
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                            "score": float(score),
                            "pred_class": pred_class,
                            "max_iou": float(best_ious[pred_idx]),
                            "tp": int(tp_flags[pred_idx]),
                        }
                    )

                del infer_tensor, preds, _logits, _objectness, _features

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    cue = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    target_values = parsed["layer_target_values"]
    target_layers = parsed["layer_target_layers"]

    if not save_csv:
        return

    output_csv = run_dir / "layer_grad.csv"
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"]
    for target_value in target_values:
        for layer_name in target_layers:
            fieldnames.append(f"{target_value}_{layer_name}")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    layer_buffer = create_layer_grad_buffer(detector.model, target_layers)

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for images, targets in tqdm(
                dataloader, desc=f"Object Detector ({mode} - {cue})", total=len(dataloader)
            ):
                for sample_idx in range(images.shape[0]):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]

                    infer_tensor, _ratio, _pad, _resized_chw = preprocess_with_letterbox(
                        detector, images[sample_idx], device, requires_grad=False
                    )
                    bbox_rows = collect_bbox_layer_grads_per_target(
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
                    del infer_tensor, bbox_rows
        finally:
            layer_buffer.remove()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_predict(config, run_dir):
    parsed = parse_output_config(config.get("output", {}))
    cue = parsed["cue"]

    if cue == "fn":
        run_fn_csv(config, run_dir)
        return
    if cue == "tp":
        run_tp_csv(config, run_dir)
        return
    if cue == "feature_grad":
        run_feature_grad_csv(config, run_dir)
        return
    if cue == "layer_grad":
        run_layer_grad_csv(config, run_dir)
        return
    raise ValueError(f"Unsupported cue: {cue}")
