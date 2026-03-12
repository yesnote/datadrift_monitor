import csv
import json
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from dataloaders.dataloader_yolo import create_dataloader
from modes.utils.predict_utils import (
    build_detector,
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


def build_row_key(image_id, image_path):
    return f"{image_id}|{image_path}"


def should_run_grad_pass(config):
    parsed = parse_output_config(config.get("output", {}))
    return parsed["save_csv_enabled"] and bool(parsed["target_layers"])


def _build_fieldnames(target_values, target_layers, compute_grads):
    fieldnames = ["image_id", "image_path", "has_fn"]
    if compute_grads:
        for target_value in target_values:
            for layer_name in target_layers:
                fieldnames.append(f"d{target_value}_d{layer_name}")
    return fieldnames


def _build_predict_stats(parsed, split, total_images, fn_images, output_csv):
    return {
        "mode": "predict",
        "cue": parsed["cue"],
        "split": split,
        "save_csv": parsed["save_csv_enabled"],
        "save_image": parsed["save_image_enabled"],
        "save_image_step": parsed["image_step"],
        "save_image_num": parsed["image_num"],
        "target_values": parsed["target_values"],
        "target_layers": parsed["target_layers"],
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv) if parsed["save_csv_enabled"] else "",
    }


def _mb(num_bytes):
    return num_bytes / (1024 ** 2)


def _create_memory_logger(run_dir, pass_name, enabled):
    if not enabled:
        return None, None, None, None
    memory_log_path = run_dir / f"cuda_memory_{pass_name}.csv"
    handle = open(memory_log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "image_index",
            "allocated_bytes",
            "reserved_bytes",
            "max_allocated_bytes",
            "max_reserved_bytes",
            "allocated_mb",
            "reserved_mb",
            "max_allocated_mb",
            "max_reserved_mb",
            "delta_allocated_bytes",
            "delta_reserved_bytes",
            "delta_allocated_mb",
            "delta_reserved_mb",
        ],
    )
    writer.writeheader()
    baseline = {"allocated": None, "reserved": None}
    return handle, writer, memory_log_path, baseline


def _log_cuda_memory(writer, handle, device, image_index, baseline):
    if writer is None or device.type != "cuda":
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    max_allocated = torch.cuda.max_memory_allocated(device_index)
    max_reserved = torch.cuda.max_memory_reserved(device_index)

    if baseline["allocated"] is None:
        baseline["allocated"] = allocated
    if baseline["reserved"] is None:
        baseline["reserved"] = reserved

    delta_allocated = allocated - baseline["allocated"]
    delta_reserved = reserved - baseline["reserved"]

    writer.writerow(
        {
            "image_index": image_index,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "max_allocated_bytes": max_allocated,
            "max_reserved_bytes": max_reserved,
            "allocated_mb": round(_mb(allocated), 3),
            "reserved_mb": round(_mb(reserved), 3),
            "max_allocated_mb": round(_mb(max_allocated), 3),
            "max_reserved_mb": round(_mb(max_reserved), 3),
            "delta_allocated_bytes": delta_allocated,
            "delta_reserved_bytes": delta_reserved,
            "delta_allocated_mb": round(_mb(delta_allocated), 3),
            "delta_reserved_mb": round(_mb(delta_reserved), 3),
        }
    )
    handle.flush()


def run_predict_pass(config, run_dir):
    run_dir = Path(run_dir)
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
    memory_log_interval = int(output_cfg.get("memory_log_interval", 50))

    output_csv = run_dir / "fn_results.csv"
    base_csv = run_dir / "fn_base_rows.csv"
    stats_json = run_dir / "predict_pass_stats.json"
    summary_json = run_dir / "summary.json"

    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError(
            "Loaded 0 images. Check dataset root/image_dir/split configuration in YAML."
        )

    detector, device = build_detector(config)

    total_images = 0
    fn_images = 0
    base_writer = None
    base_file_handle = None
    memory_handle = None
    memory_writer = None
    memory_log_path = None
    memory_baseline = None
    if save_csv:
        base_file_handle = open(base_csv, "w", newline="", encoding="utf-8")
        base_writer = csv.DictWriter(
            base_file_handle, fieldnames=["image_id", "image_path", "has_fn"]
        )
        base_writer.writeheader()
    if device.type == "cuda":
        memory_handle, memory_writer, memory_log_path, memory_baseline = _create_memory_logger(
            run_dir, "predict_pass", enabled=True
        )

    try:
        for step_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Predict Pass ({split})", total=len(dataloader))
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

                if base_writer is not None:
                    base_writer.writerow(row)

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
                if device.type == "cuda":
                    if memory_log_interval > 0 and (
                        total_images == 1 or total_images % memory_log_interval == 0
                    ):
                        _log_cuda_memory(
                            memory_writer,
                            memory_handle,
                            device,
                            total_images,
                            memory_baseline,
                        )
                        print(
                            f"[CUDA][Predict Pass] image={total_images} "
                            f"alloc={round(_mb(torch.cuda.memory_allocated()), 1)}MB "
                            f"reserved={round(_mb(torch.cuda.memory_reserved()), 1)}MB"
                        )
    finally:
        if base_file_handle is not None:
            base_file_handle.close()
        if memory_handle is not None:
            memory_handle.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stats = _build_predict_stats(parsed, split, total_images, fn_images, output_csv)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if save_csv and not compute_grads:
        os.replace(base_csv, output_csv)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved results CSV: {output_csv}")
        if memory_log_path is not None:
            print(f"Saved CUDA memory log: {memory_log_path}")
        print(f"Saved summary: {summary_json}")
        return

    if not save_csv:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        if memory_log_path is not None:
            print(f"Saved CUDA memory log: {memory_log_path}")
        print(f"Saved summary: {summary_json}")
        return

    if memory_log_path is not None:
        print(f"Saved CUDA memory log: {memory_log_path}")
    print(f"Saved intermediate base CSV: {base_csv}")


def run_grad_pass(config, run_dir):
    run_dir = Path(run_dir)
    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    output_cfg = config.get("output", {})
    parsed = parse_output_config(output_cfg)

    save_csv = parsed["save_csv_enabled"]
    target_values = parsed["target_values"]
    target_layers = parsed["target_layers"]
    compute_grads = save_csv and bool(target_layers)
    memory_log_interval = int(output_cfg.get("memory_log_interval", 50))
    output_csv = run_dir / "fn_results.csv"
    base_csv = run_dir / "fn_base_rows.csv"
    stats_json = run_dir / "predict_pass_stats.json"
    summary_json = run_dir / "summary.json"

    if not save_csv:
        if stats_json.exists():
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        return

    if not compute_grads:
        if base_csv.exists() and not output_csv.exists():
            os.replace(base_csv, output_csv)
        if stats_json.exists():
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        return

    if not base_csv.exists():
        raise FileNotFoundError(
            f"Base CSV not found for Grad Pass: {base_csv}. Run Predict Pass first."
        )

    stats = {}
    if stats_json.exists():
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

    fieldnames = _build_fieldnames(target_values, target_layers, compute_grads=True)

    base_rows = {}
    with open(base_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_rows[build_row_key(row["image_id"], row["image_path"])] = row

    grad_loader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    layer_buffer = create_layer_grad_buffer(detector.model, target_layers)
    grad_image_count = 0
    memory_handle = None
    memory_writer = None
    memory_log_path = None
    memory_baseline = None
    if device.type == "cuda":
        memory_handle, memory_writer, memory_log_path, memory_baseline = _create_memory_logger(
            run_dir, "grad_pass", enabled=True
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for images, targets in tqdm(
                grad_loader, desc=f"Grad Pass ({split})", total=len(grad_loader)
            ):
                for sample_idx in range(images.shape[0]):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    key = build_row_key(str(image_id), image_path)
                    base_row = base_rows.get(key)

                    infer_tensor, _ratio, _pad, _resized_chw = preprocess_with_letterbox(
                        detector, images[sample_idx], device, requires_grad=False
                    )
                    grad_stats = collect_gradients_per_target(
                        detector=detector,
                        input_tensor=infer_tensor,
                        target_values=target_values,
                        target_layers=target_layers,
                        layer_buffer=layer_buffer,
                    )

                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "has_fn": int(base_row["has_fn"]) if base_row is not None else 0,
                    }
                    for grad_key, grad_value in grad_stats.items():
                        row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                    writer.writerow(row)

                    del infer_tensor, grad_stats
                    grad_image_count += 1
                    if device.type == "cuda":
                        if memory_log_interval > 0 and (
                            grad_image_count == 1 or grad_image_count % memory_log_interval == 0
                        ):
                            _log_cuda_memory(
                                memory_writer,
                                memory_handle,
                                device,
                                grad_image_count,
                                memory_baseline,
                            )
                            print(
                                f"[CUDA][Grad Pass] image={grad_image_count} "
                                f"alloc={round(_mb(torch.cuda.memory_allocated()), 1)}MB "
                                f"reserved={round(_mb(torch.cuda.memory_reserved()), 1)}MB"
                            )
        finally:
            layer_buffer.remove()
            if memory_handle is not None:
                memory_handle.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if base_csv.exists():
        base_csv.unlink()

    if stats:
        stats["output_csv"] = str(output_csv)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved results CSV: {output_csv}")
    if memory_log_path is not None:
        print(f"Saved CUDA memory log: {memory_log_path}")
    print(f"Saved summary: {summary_json}")


def run_predict(config, run_dir):
    run_predict_pass(config, run_dir)
    if should_run_grad_pass(config):
        run_grad_pass(config, run_dir)
