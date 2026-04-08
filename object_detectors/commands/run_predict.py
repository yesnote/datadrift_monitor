import csv
import json
import queue
import threading
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloaders.dataloader_yolo import build_dataset, create_dataloader, yolo_collate_fn
from commands.utils.predict_utils import (
    assign_tp_to_predictions,
    build_detector,
    enable_forced_mc_dropout_on_yolov5_head,
    collect_bbox_gradients_per_target,
    collect_bbox_layer_grads_per_target,
    collect_image_features_per_layer,
    collect_gradients_per_target,
    collect_image_layer_grads_per_target,
    create_layer_grad_buffer,
    draw_predictions,
    get_fn_gt_indices,
    get_pre_nms_keep_indices,
    has_fn_for_image,
    load_gt_category_maps,
    map_boxes_to_letterbox,
    map_grad_tensor_to_numbers,
    parse_output_config,
    preprocess_with_letterbox,
    resolve_layer_parameter,
)


def _xywh_to_xyxy_tensor(xywh: torch.Tensor) -> torch.Tensor:
    out = xywh.clone()
    out[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    out[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    out[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    out[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return out


def _build_summary(total_images, fn_images, output_csv):
    return {
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv),
    }


def _as_image_list(images):
    if isinstance(images, list):
        return images
    return [images[i] for i in range(images.shape[0])]


def _prepare_infer_batch(detector, images, device, auto=False):
    image_list = _as_image_list(images)
    infer_tensors = []
    ratios = []
    pads = []
    resized_chws = []
    for img in image_list:
        infer_tensor, ratio, pad, resized_chw = preprocess_with_letterbox(
            detector, img, device, requires_grad=False, auto=auto
        )
        infer_tensors.append(infer_tensor)
        ratios.append(ratio)
        pads.append(pad)
        resized_chws.append(resized_chw)
    infer_batch = torch.cat(infer_tensors, dim=0)
    return infer_batch, ratios, pads, resized_chws


def _resolve_gt_class_names(target, catid_to_name):
    gt_names = target.get("gt_class_names")
    if gt_names is not None:
        return [str(v) for v in gt_names]
    gt_labels_tensor = target["labels"]
    return [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]


def _vector_from_grad_value(grad_value):
    if isinstance(grad_value, list):
        if len(grad_value) == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = np.asarray(grad_value, dtype=np.float32).reshape(-1)
        return np.abs(arr)
    if isinstance(grad_value, (int, float)):
        return np.asarray([abs(float(grad_value))], dtype=np.float32)
    if isinstance(grad_value, dict):
        # Fallback: when vector_reduction is configured, grad value may be a stats dict.
        vals = []
        for k in sorted(grad_value.keys()):
            try:
                vals.append(float(grad_value[k]))
            except Exception:
                continue
        if not vals:
            return np.zeros((0,), dtype=np.float32)
        return np.abs(np.asarray(vals, dtype=np.float32).reshape(-1))
    return np.zeros((0,), dtype=np.float32)


def _build_layer_filter_map_from_grad_stats(grad_stats, target_values, target_layers, layer_param_shapes=None):
    layer_vectors = []
    for layer_name in target_layers:
        per_target = []
        max_len = 0
        expected_shape = None
        if layer_param_shapes is not None:
            expected_shape = layer_param_shapes.get(layer_name)
        for target_value in target_values:
            key = f"{target_value}_{layer_name}"
            raw_vec = _vector_from_grad_value(grad_stats.get(key, []))
            vec = raw_vec
            if expected_shape and raw_vec.size > 0:
                numel = int(np.prod(expected_shape))
                if raw_vec.size == numel:
                    reshaped = raw_vec.reshape(expected_shape)
                    if len(expected_shape) == 1:
                        vec = np.abs(reshaped).astype(np.float32, copy=False)
                    else:
                        first_dim = int(expected_shape[0])
                        vec = np.abs(reshaped).reshape(first_dim, -1).mean(axis=1).astype(np.float32, copy=False)
            per_target.append(vec)
            if vec.shape[0] > max_len:
                max_len = vec.shape[0]
        if max_len == 0:
            layer_vectors.append(np.zeros((0,), dtype=np.float32))
            continue
        mat = np.full((len(per_target), max_len), np.nan, dtype=np.float32)
        for i, vec in enumerate(per_target):
            if vec.shape[0] > 0:
                mat[i, : vec.shape[0]] = vec
        layer_vectors.append(np.nanmean(mat, axis=0))

    f_max = max((v.shape[0] for v in layer_vectors), default=0)
    out = np.full((len(target_layers), f_max), np.nan, dtype=np.float32)
    for li, vec in enumerate(layer_vectors):
        if vec.shape[0] > 0:
            out[li, : vec.shape[0]] = vec
    return out


def _normalize_layer_map(layer_map, mode="layer_minmax"):
    if mode == "none":
        return layer_map.astype(np.float32, copy=True)
    out = layer_map.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            continue
        vals = row[finite_mask]
        if mode == "layer_trimmed_minmax":
            vmin = float(np.percentile(vals, 1.0))
            vmax = float(np.percentile(vals, 99.0))
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        if vmax > vmin:
            normed = (vals - vmin) / (vmax - vmin)
            row[finite_mask] = np.clip(normed, 0.0, 1.0)
        else:
            row[finite_mask] = 0.0
    return out


def _stack_nanmean_maps(maps):
    if not maps:
        return np.zeros((0, 0), dtype=np.float32)
    l_max = max(m.shape[0] for m in maps)
    f_max = max(m.shape[1] for m in maps)
    arr = np.full((len(maps), l_max, f_max), np.nan, dtype=np.float32)
    for i, m in enumerate(maps):
        arr[i, : m.shape[0], : m.shape[1]] = m
    valid = np.isfinite(arr)
    count = valid.sum(axis=0).astype(np.float32)
    total = np.where(valid, arr, 0.0).sum(axis=0).astype(np.float32)
    out = np.full((l_max, f_max), np.nan, dtype=np.float32)
    mask = count > 0
    out[mask] = total[mask] / count[mask]
    return out


def _profile_stats(profiles):
    if not profiles:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    l_max = max(p.shape[0] for p in profiles)
    arr = np.full((len(profiles), l_max), np.nan, dtype=np.float32)
    for i, p in enumerate(profiles):
        arr[i, : p.shape[0]] = p
    valid = np.isfinite(arr)
    count = valid.sum(axis=0)
    idx = np.where(count > 0)[0]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    mean = np.nanmean(arr[:, idx], axis=0).astype(np.float32)
    std = np.nanstd(arr[:, idx], axis=0).astype(np.float32)
    return mean, std, idx.astype(np.int64)


def _save_layer_profile_plot(fn_profiles, non_fn_profiles, out_path, log_scale=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fn_mean, fn_std, fn_idx = _profile_stats(fn_profiles)
    non_mean, non_std, non_idx = _profile_stats(non_fn_profiles)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    eps = 1e-12
    plotted = False
    if fn_idx.size > 0:
        y = np.maximum(fn_mean, eps) if log_scale else fn_mean
        lo = np.maximum(fn_mean - fn_std, eps) if log_scale else (fn_mean - fn_std)
        hi = np.maximum(fn_mean + fn_std, eps) if log_scale else (fn_mean + fn_std)
        ax.plot(fn_idx, y, color="#d62728", linewidth=2.0, label="FN mean")
        ax.fill_between(fn_idx, lo, hi, color="#d62728", alpha=0.18, linewidth=0)
        plotted = True
    if non_idx.size > 0:
        y = np.maximum(non_mean, eps) if log_scale else non_mean
        lo = np.maximum(non_mean - non_std, eps) if log_scale else (non_mean - non_std)
        hi = np.maximum(non_mean + non_std, eps) if log_scale else (non_mean + non_std)
        ax.plot(non_idx, y, color="#1f77b4", linewidth=2.0, label="non-FN mean")
        ax.fill_between(non_idx, lo, hi, color="#1f77b4", alpha=0.18, linewidth=0)
        plotted = True

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Layer Sum(|grad|)")
    ax.set_title("Layer-wise Gradient Profile (mean ± std)")
    ax.grid(True, which="both", axis="both", alpha=0.2)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No profile data", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_heatmap_png(map_2d, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#efefef")

    if map_2d.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Filter Number")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    m = map_2d.astype(np.float32, copy=True)
    finite_mask = np.isfinite(m)
    if not finite_mask.any():
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Layer Number")
        ax.set_ylabel("Filter Number")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    vals = m[finite_mask]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax > vmin:
        m[finite_mask] = (vals - vmin) / (vmax - vmin)
    else:
        m[finite_mask] = 0.0

    layer_idx, filter_idx = np.where(finite_mask)
    color_vals = m[finite_mask]
    sc = ax.scatter(
        layer_idx.astype(np.float32),
        filter_idx.astype(np.float32),
        c=color_vals,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        s=90,
        alpha=0.32,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Gradient")
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Filter Number")
    ax.set_xlim(-0.5, m.shape[0] - 0.5)
    ax.set_ylim(-0.5, m.shape[1] - 0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _mc_dropout_single_csv_writer(write_queue, output_csv, fieldnames):
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            batch_rows = write_queue.get()
            if batch_rows is None:
                break
            if batch_rows:
                writer.writerows(batch_rows)


def run_fn_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    save_image = parsed["save_image_enabled"]
    image_step = parsed["save_image_gt_step"]
    image_max_num = parsed["save_image_gt_max_num"]

    output_csv = run_dir / "fn.csv"
    summary_json = run_dir / "summary.json"

    catid_to_name = load_gt_category_maps(config, split)
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
            tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader))
        ):
            image_list = _as_image_list(images)
            batch_size = len(image_list)
            should_save_step = save_image and (step_idx % image_step == 0)
            step_dir = run_dir / "images" / f"0_{step_idx}"
            if should_save_step:
                step_dir.mkdir(parents=True, exist_ok=True)

            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)

            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                row = {
                    "image_id": int(target["image_id"][0].item()),
                    "image_path": target["path"],
                }

                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
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
                    vis_image = draw_predictions(resized_chws[sample_idx], pred_boxes, pred_class_names, pred_scores)
                    fn_gt_indices = get_fn_gt_indices(
                        gt_boxes=gt_boxes,
                        gt_class_names=gt_class_names,
                        pred_boxes=pred_boxes,
                        pred_class_names=pred_class_names,
                        iou_match_threshold=iou_match_threshold,
                    )
                    for gt_idx in fn_gt_indices:
                        gt_box = gt_boxes[gt_idx]
                        gt_name = gt_class_names[gt_idx]
                        x1, y1, x2, y2 = [int(v) for v in gt_box]
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            vis_image,
                            f"{gt_name}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    out_path = step_dir / f"{row['image_id']}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                total_images += 1
            del infer_batch, preds, _logits, _objectness, _features
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
                for sample_idx in range(len(image_list)):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]

                    infer_tensor = infer_batch[sample_idx: sample_idx + 1]
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
                        del bbox_rows
                    else:
                        grad_stats = collect_gradients_per_target(
                            detector=detector,
                            input_tensor=infer_tensor,
                            target_values=target_values,
                            target_layers=target_layers,
                            layer_buffer=layer_buffer,
                            pre_nms=pre_nms,
                            pre_nms_ratio=pre_nms_ratio,
                        )

                        row = {"image_id": image_id, "image_path": image_path}
                        for grad_key, grad_value in grad_stats.items():
                            row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                        writer.writerow(row)
                        del grad_stats
                del infer_batch
        finally:
            layer_buffer.remove()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


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


def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]

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

    catid_to_name = load_gt_category_maps(config, split)
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
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)

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
            del infer_batch, preds, _logits, _objectness, _features

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_meta_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "meta_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_threshold = float(parsed["meta_detect_score_threshold"])
    iou_threshold = float(parsed["meta_detect_iou_threshold"])
    vector_reduction = parsed["meta_detect_vector_reduction"]

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='meta_detect' requires output.unit in {'image','bbox'}.")

    def _stats(v: torch.Tensor):
        if v is None or v.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        x = v.detach().float().reshape(-1)
        return float(torch.min(x).item()), float(torch.max(x).item()), float(torch.mean(x).item()), float(torch.std(x, unbiased=False).item())

    def _iou_1vN(box: torch.Tensor, boxes: torch.Tensor):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=box.device)
        lt = torch.max(box[:2], boxes[:, :2])
        rb = torch.min(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-12)

    output_csv = run_dir / "meta_detect.csv"
    meta_feature_names = [
        "num_candidate_boxes",
        "x_min", "x_max", "x_mean", "x_std",
        "y_min", "y_max", "y_mean", "y_std",
        "w_min", "w_max", "w_mean", "w_std",
        "h_min", "h_max", "h_mean", "h_std",
        "size", "size_min", "size_max", "size_mean", "size_std",
        "circum", "circum_min", "circum_max", "circum_mean", "circum_std",
        "size_circum", "size_circum_min", "size_circum_max", "size_circum_mean", "size_circum_std",
        "score_min", "score_max", "score_mean", "score_std",
        "iou_pb_min", "iou_pb_max", "iou_pb_mean", "iou_pb_std",
    ]
    if unit == "bbox":
        fieldnames = [
            "image_id", "image_path", "pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
            *meta_feature_names,
        ]
    else:
        fieldnames = ["image_id", "image_path"]
        for feature_name in meta_feature_names:
            for metric_name in vector_reduction:
                fieldnames.append(f"{feature_name}_{metric_name}")
        fieldnames.append("num_preds")

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
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                pred_boxes = preds[0][sample_idx]
                pred_scores = preds[3][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_class_ids = preds[1][sample_idx] if len(preds) > 1 else []

                raw = raw_prediction[sample_idx].detach().float()
                if raw.numel() == 0:
                    continue
                raw_xyxy = _xywh_to_xyxy_tensor(raw[:, :4])
                raw_obj = raw[:, 4] if raw.shape[1] > 4 else torch.ones((raw.shape[0],), device=device)
                raw_cls = raw[:, 5:] if raw.shape[1] > 5 else torch.zeros((raw.shape[0], 0), device=device)
                if raw_cls.numel() > 0:
                    raw_cls_max, raw_cls_idx = raw_cls.max(dim=1)
                else:
                    raw_cls_max = torch.ones_like(raw_obj)
                    raw_cls_idx = torch.zeros((raw.shape[0],), dtype=torch.long, device=device)
                raw_score = raw_obj * raw_cls_max

                image_feature_rows = []
                for pred_idx, (box, score, pred_class_name) in enumerate(zip(pred_boxes, pred_scores, pred_class_names)):
                    fbox = torch.tensor(box, dtype=torch.float32, device=device)
                    cls_idx = int(pred_class_ids[pred_idx]) if pred_idx < len(pred_class_ids) else -1
                    ious = _iou_1vN(fbox, raw_xyxy)
                    class_mask = (raw_cls_idx == cls_idx) if cls_idx >= 0 else torch.ones_like(raw_score, dtype=torch.bool)
                    cand_mask = class_mask & (raw_score > score_threshold) & (ious > iou_threshold)

                    cand_boxes = raw_xyxy[cand_mask]
                    cand_scores = raw_score[cand_mask]
                    cand_ious = ious[cand_mask]
                    if cand_boxes.numel() == 0:
                        cand_boxes = fbox.view(1, 4)
                        cand_scores = torch.tensor([float(score)], dtype=torch.float32, device=device)
                        cand_ious = torch.zeros((1,), dtype=torch.float32, device=device)

                    x = 0.5 * (cand_boxes[:, 0] + cand_boxes[:, 2])
                    y = 0.5 * (cand_boxes[:, 1] + cand_boxes[:, 3])
                    w = torch.abs(0.5 * (cand_boxes[:, 0] - cand_boxes[:, 2]))
                    h = torch.abs(0.5 * (cand_boxes[:, 1] - cand_boxes[:, 3]))
                    size_vals = (0.5 * (x - w)) * (0.5 * (y - h))
                    circum_vals = (cand_boxes[:, 2] - cand_boxes[:, 0]) + (cand_boxes[:, 3] - cand_boxes[:, 1])
                    size_circum_vals = (w * h) / (torch.abs(cand_boxes[:, 2] - cand_boxes[:, 0]) + torch.abs(cand_boxes[:, 3] - cand_boxes[:, 1])).clamp(min=1e-12)

                    iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
                    iou_pb_pos = iou_pb[iou_pb > 0]

                    fx1, fy1, fx2, fy2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    fsize = float((0.5 * ((0.5 * (fx1 + fx2)) - abs(0.5 * (fx1 - fx2)))) * (0.5 * ((0.5 * (fy1 + fy2)) - abs(0.5 * (fy1 - fy2)))))
                    fcircum = float(abs(fx2 - fx1) + abs(fy2 - fy1))
                    fsize_circum = float(((0.5 * abs(fx2 - fx1)) * (0.5 * abs(fy2 - fy1))) / max(abs(fx2 - fx1) + abs(fy2 - fy1), 1e-12))

                    x_min, x_max, x_mean, x_std = _stats(x)
                    y_min, y_max, y_mean, y_std = _stats(y)
                    w_min, w_max, w_mean, w_std = _stats(w)
                    h_min, h_max, h_mean, h_std = _stats(h)
                    size_min, size_max, size_mean, size_std = _stats(size_vals)
                    circum_min, circum_max, circum_mean, circum_std = _stats(circum_vals)
                    size_circum_min, size_circum_max, size_circum_mean, size_circum_std = _stats(size_circum_vals)
                    score_min, score_max, score_mean, score_std = _stats(cand_scores)
                    iou_pb_min, iou_pb_max, iou_pb_mean, iou_pb_std = _stats(iou_pb_pos)

                    feature_row = {
                        "num_candidate_boxes": float(cand_boxes.shape[0]),
                        "x_min": x_min, "x_max": x_max, "x_mean": x_mean, "x_std": x_std,
                        "y_min": y_min, "y_max": y_max, "y_mean": y_mean, "y_std": y_std,
                        "w_min": w_min, "w_max": w_max, "w_mean": w_mean, "w_std": w_std,
                        "h_min": h_min, "h_max": h_max, "h_mean": h_mean, "h_std": h_std,
                        "size": fsize, "size_min": size_min, "size_max": size_max, "size_mean": size_mean, "size_std": size_std,
                        "circum": fcircum, "circum_min": circum_min, "circum_max": circum_max, "circum_mean": circum_mean, "circum_std": circum_std,
                        "size_circum": fsize_circum, "size_circum_min": size_circum_min, "size_circum_max": size_circum_max,
                        "size_circum_mean": size_circum_mean, "size_circum_std": size_circum_std,
                        "score_min": score_min, "score_max": score_max, "score_mean": score_mean, "score_std": score_std,
                        "iou_pb_min": iou_pb_min, "iou_pb_max": iou_pb_max, "iou_pb_mean": iou_pb_mean, "iou_pb_std": iou_pb_std,
                    }
                    if unit == "bbox":
                        writer.writerow(
                            {
                                "image_id": image_id,
                                "image_path": image_path,
                                "pred_idx": pred_idx,
                                "xmin": fx1,
                                "ymin": fy1,
                                "xmax": fx2,
                                "ymax": fy2,
                                "score": float(score),
                                "pred_class": pred_class_name,
                                **feature_row,
                            }
                        )
                    else:
                        image_feature_rows.append(feature_row)
                if unit == "image":
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "num_preds": len(image_feature_rows),
                    }
                    for feature_name in meta_feature_names:
                        if len(image_feature_rows) == 0:
                            stats = {
                                "1-norm": 0.0,
                                "2-norm": 0.0,
                                "min": 0.0,
                                "max": 0.0,
                                "mean": 0.0,
                                "std": 0.0,
                            }
                        else:
                            vec = torch.tensor(
                                [float(r[feature_name]) for r in image_feature_rows],
                                dtype=torch.float32,
                                device=device,
                            )
                            stats = map_grad_tensor_to_numbers(vec)
                        for metric_name in vector_reduction:
                            row[f"{feature_name}_{metric_name}"] = float(stats[metric_name])
                    writer.writerow(row)
            del infer_batch, preds, _logits, _objectness, _features, raw_prediction

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved results CSV: {output_csv}")


def run_score_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "score"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_vector_reduction = parsed["score_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return

    output_csv = run_dir / "score.csv"
    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(
            [
                "pred_idx",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "score",
                "pred_class",
            ]
        )
    else:
        fieldnames.extend(score_vector_reduction)
        fieldnames.append("num_preds")

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
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            raw_prediction = None
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)
                if unit == "image" and pre_nms:
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]

                if unit == "bbox":
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
                            }
                        )
                else:
                    if pre_nms and raw_prediction is not None:
                        pre = raw_prediction[sample_idx].detach().float()
                        if pre.numel() == 0:
                            score_tensor = torch.zeros((0,), dtype=torch.float32, device=device)
                        else:
                            obj = pre[:, 4]
                            cls_max = pre[:, 5:].max(dim=1).values if pre.shape[1] > 5 else torch.ones_like(obj)
                            score_tensor = obj * cls_max
                            keep_idx = get_pre_nms_keep_indices(pre, pre_nms_ratio=pre_nms_ratio)
                            if int(keep_idx.shape[0]) > 0:
                                score_tensor = score_tensor[keep_idx]
                            else:
                                score_tensor = torch.zeros((0,), dtype=torch.float32, device=device)
                        num_preds = int(score_tensor.shape[0])
                    else:
                        score_tensor = torch.as_tensor(pred_scores, dtype=torch.float32, device=device)
                        num_preds = len(pred_scores)

                    if num_preds == 0:
                        stat_all = {
                            "1-norm": 0.0,
                            "2-norm": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    else:
                        stat_all = map_grad_tensor_to_numbers(score_tensor.reshape(-1))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": num_preds}
                    for metric_name in score_vector_reduction:
                        row[metric_name] = float(stat_all[metric_name])
                    writer.writerow(row)
            del infer_batch, preds, _logits, _objectness, _features, raw_prediction

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_full_softmax_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "full_softmax"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    vector_reduction = parsed["full_softmax_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='full_softmax' requires output.unit in {'image','bbox'}.")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "full_softmax.csv"
    if unit == "bbox":
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
        ] + [f"prob_{i}" for i in range(num_classes)]
    else:
        fieldnames = ["image_id", "image_path"] + [f"{k}_vector" for k in vector_reduction] + ["num_preds"]

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
            with torch.no_grad():
                preds, logits, _objectness, _features = detector(infer_batch)
                if unit == "image" and pre_nms:
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                    raw_logits = (
                        model_output[1]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                        else None
                    )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                pred_logits = logits[sample_idx] if logits else torch.zeros((0, num_classes), device=device)
                pred_probs = torch.softmax(pred_logits, dim=-1) if pred_logits.numel() else pred_logits

                if unit == "bbox":
                    for pred_idx, (box, score, pred_class) in enumerate(
                        zip(pred_boxes, pred_scores, pred_class_names)
                    ):
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                            "score": float(score),
                            "pred_class": pred_class,
                        }
                        if pred_idx < pred_probs.shape[0]:
                            probs = pred_probs[pred_idx].detach().cpu().tolist()
                        else:
                            probs = [0.0] * num_classes
                        for class_idx in range(num_classes):
                            row[f"prob_{class_idx}"] = float(probs[class_idx]) if class_idx < len(probs) else 0.0
                        writer.writerow(row)
                else:
                    if pre_nms and raw_prediction is not None:
                        if raw_logits is not None:
                            pre_logits = raw_logits[sample_idx].detach().float()
                            pre_probs = torch.softmax(pre_logits, dim=-1) if pre_logits.numel() else pre_logits
                            keep_idx = get_pre_nms_keep_indices(
                                raw_prediction[sample_idx].detach().float(),
                                pre_logits,
                                pre_nms_ratio=pre_nms_ratio,
                            )
                        else:
                            pre_raw = raw_prediction[sample_idx].detach().float()
                            cls_scores = pre_raw[:, 5:] if pre_raw.shape[1] > 5 else torch.zeros((pre_raw.shape[0], num_classes), device=device)
                            pre_probs = torch.softmax(cls_scores, dim=-1) if cls_scores.numel() else cls_scores
                            keep_idx = get_pre_nms_keep_indices(pre_raw, pre_nms_ratio=pre_nms_ratio)
                        if int(keep_idx.shape[0]) > 0:
                            pre_probs = pre_probs[keep_idx]
                        else:
                            pre_probs = torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        probs_np = pre_probs.detach().cpu().numpy() if pre_probs.numel() else None
                        num_preds = int(pre_probs.shape[0])
                    else:
                        probs_np = pred_probs.detach().cpu().numpy() if pred_probs.numel() else None
                        num_preds = int(pred_probs.shape[0])
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "num_preds": num_preds,
                    }
                    for metric_name in vector_reduction:
                        if probs_np is None or num_preds == 0:
                            vec = [0.0] * num_classes
                        elif metric_name == "1-norm":
                            vec = np.sum(np.abs(probs_np), axis=0).astype(float).tolist()
                        elif metric_name == "2-norm":
                            vec = np.sqrt(np.sum(np.square(probs_np), axis=0)).astype(float).tolist()
                        elif metric_name == "min":
                            vec = np.min(probs_np, axis=0).astype(float).tolist()
                        elif metric_name == "max":
                            vec = np.max(probs_np, axis=0).astype(float).tolist()
                        elif metric_name == "mean":
                            vec = np.mean(probs_np, axis=0).astype(float).tolist()
                        elif metric_name == "std":
                            vec = np.std(probs_np, axis=0).astype(float).tolist()
                        else:
                            vec = [0.0] * num_classes
                        row[f"{metric_name}_vector"] = json.dumps(vec, separators=(",", ":"))
                    writer.writerow(row)
            del infer_batch, preds, logits, _objectness, _features, raw_prediction, raw_logits

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_energy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "energy"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    energy_vector_reduction = parsed["energy_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='energy' requires output.unit in {'image','bbox'}.")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "energy.csv"
    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(
            [
                "pred_idx",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "score",
                "pred_class",
                "energy",
            ]
        )
    else:
        fieldnames.extend(energy_vector_reduction)
        fieldnames.append("num_preds")

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
            with torch.no_grad():
                preds, logits, _objectness, _features = detector(infer_batch)
                if unit == "image" and pre_nms:
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                    raw_logits = (
                        model_output[1]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                        else None
                    )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                pred_logits = logits[sample_idx] if logits else torch.zeros((0, num_classes), device=device)
                pred_probs = torch.softmax(pred_logits, dim=-1) if pred_logits.numel() else pred_logits
                if pred_probs.numel():
                    probs_clipped = pred_probs.clamp(min=1e-8, max=1.0 - 1e-8)
                    pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
                    pred_energy = -100.0 * torch.log(
                        torch.clamp(
                            torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1),
                            min=1e-8,
                        )
                    )
                else:
                    pred_energy = torch.zeros((0,), device=device)

                if unit == "bbox":
                    for pred_idx, (box, score, pred_class) in enumerate(
                        zip(pred_boxes, pred_scores, pred_class_names)
                    ):
                        energy_val = (
                            float(pred_energy[pred_idx].detach().cpu().item())
                            if pred_idx < pred_energy.shape[0]
                            else 0.0
                        )
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
                                "energy": energy_val,
                            }
                        )
                else:
                    if pre_nms and raw_prediction is not None:
                        if raw_logits is not None:
                            pre_logits = raw_logits[sample_idx].detach().float()
                            pre_probs = torch.softmax(pre_logits, dim=-1) if pre_logits.numel() else pre_logits
                            keep_idx = get_pre_nms_keep_indices(
                                raw_prediction[sample_idx].detach().float(),
                                pre_logits,
                                pre_nms_ratio=pre_nms_ratio,
                            )
                        else:
                            pre_raw = raw_prediction[sample_idx].detach().float()
                            cls_scores = (
                                pre_raw[:, 5:]
                                if pre_raw.shape[1] > 5
                                else torch.zeros((pre_raw.shape[0], num_classes), device=device)
                            )
                            pre_probs = torch.softmax(cls_scores, dim=-1) if cls_scores.numel() else cls_scores
                            keep_idx = get_pre_nms_keep_indices(pre_raw, pre_nms_ratio=pre_nms_ratio)
                        if int(keep_idx.shape[0]) > 0:
                            pre_probs = pre_probs[keep_idx]
                        else:
                            pre_probs = torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        if pre_probs.numel():
                            probs_clipped = pre_probs.clamp(min=1e-8, max=1.0 - 1e-8)
                            pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
                            energy_tensor = -100.0 * torch.log(
                                torch.clamp(
                                    torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1),
                                    min=1e-8,
                                )
                            )
                        else:
                            energy_tensor = torch.zeros((0,), device=device)
                    else:
                        energy_tensor = pred_energy

                    num_preds = int(energy_tensor.shape[0])
                    if num_preds == 0:
                        stat_all = {
                            "1-norm": 0.0,
                            "2-norm": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    else:
                        stat_all = map_grad_tensor_to_numbers(energy_tensor.detach().float().reshape(-1))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": num_preds}
                    for metric_name in energy_vector_reduction:
                        row[metric_name] = float(stat_all[metric_name])
                    writer.writerow(row)
            del infer_batch, preds, logits, _objectness, _features, raw_prediction, raw_logits

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_entropy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "entropy"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    entropy_vector_reduction = parsed["entropy_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='entropy' requires output.unit in {'image','bbox'}.")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "entropy.csv"
    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(
            [
                "pred_idx",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "score",
                "pred_class",
                "entropy",
            ]
        )
    else:
        fieldnames.extend(entropy_vector_reduction)
        fieldnames.append("num_preds")

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
            with torch.no_grad():
                preds, logits, _objectness, _features = detector(infer_batch)
                if unit == "image" and pre_nms:
                    model_output = detector.model(infer_batch, augment=False)
                    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                    raw_logits = (
                        model_output[1]
                        if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                        else None
                    )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                pred_logits = logits[sample_idx] if logits else torch.zeros((0, num_classes), device=device)
                pred_probs = torch.softmax(pred_logits, dim=-1) if pred_logits.numel() else pred_logits
                if pred_probs.numel():
                    pred_entropy = -torch.sum(pred_probs * torch.log(pred_probs.clamp(min=1e-12)), dim=-1)
                else:
                    pred_entropy = torch.zeros((0,), device=device)

                if unit == "bbox":
                    for pred_idx, (box, score, pred_class) in enumerate(
                        zip(pred_boxes, pred_scores, pred_class_names)
                    ):
                        entropy_val = (
                            float(pred_entropy[pred_idx].detach().cpu().item())
                            if pred_idx < pred_entropy.shape[0]
                            else 0.0
                        )
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
                                "entropy": entropy_val,
                            }
                        )
                else:
                    if pre_nms and raw_prediction is not None:
                        if raw_logits is not None:
                            pre_logits = raw_logits[sample_idx].detach().float()
                            pre_probs = torch.softmax(pre_logits, dim=-1) if pre_logits.numel() else pre_logits
                            keep_idx = get_pre_nms_keep_indices(
                                raw_prediction[sample_idx].detach().float(),
                                pre_logits,
                                pre_nms_ratio=pre_nms_ratio,
                            )
                        else:
                            pre_raw = raw_prediction[sample_idx].detach().float()
                            cls_scores = pre_raw[:, 5:] if pre_raw.shape[1] > 5 else torch.zeros((pre_raw.shape[0], num_classes), device=device)
                            pre_probs = torch.softmax(cls_scores, dim=-1) if cls_scores.numel() else cls_scores
                            keep_idx = get_pre_nms_keep_indices(pre_raw, pre_nms_ratio=pre_nms_ratio)
                        if int(keep_idx.shape[0]) > 0:
                            pre_probs = pre_probs[keep_idx]
                        else:
                            pre_probs = torch.zeros((0, num_classes), dtype=torch.float32, device=device)
                        if pre_probs.numel():
                            entropy_tensor = -torch.sum(pre_probs * torch.log(pre_probs.clamp(min=1e-12)), dim=-1)
                        else:
                            entropy_tensor = torch.zeros((0,), device=device)
                    else:
                        entropy_tensor = pred_entropy

                    num_preds = int(entropy_tensor.shape[0])
                    if num_preds == 0:
                        stat_all = {
                            "1-norm": 0.0,
                            "2-norm": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    else:
                        stat_all = map_grad_tensor_to_numbers(entropy_tensor.detach().float().reshape(-1))
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": num_preds}
                    for metric_name in entropy_vector_reduction:
                        row[metric_name] = float(stat_all[metric_name])
                    writer.writerow(row)
            del infer_batch, preds, logits, _objectness, _features, raw_prediction, raw_logits

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    num_runs = int(parsed["mc_num_runs"])
    dropout_rate = float(parsed["mc_dropout_rate"])
    queue_maxsize = int(parsed["mc_queue_maxsize"])
    vector_reduction = parsed["mc_vector_reduction"]
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))

    if not save_csv:
        return
    if unit not in {"image", "bbox"}:
        raise ValueError("output.uncertainty='mc_dropout' requires output.unit in {'image','bbox'}.")

    # Windows OpenMP + subprocess workers can conflict in MC-dropout runs.
    # Force single-process data loading here to avoid libiomp duplicate init crashes.
    dataset = build_dataset(config, split=split)
    dl_cfg = config["dataloader"]
    shuffle = dl_cfg["shuffle_train"] if split == "train" else dl_cfg["shuffle_eval"]
    dataloader = DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=yolo_collate_fn,
    )
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    n_classes_hint = len(detector.names) if detector.names is not None else 80

    output_csv = run_dir / "mc_dropout.csv"
    stat_keys = list(vector_reduction)
    stat_alias = {
        "1-norm": "l1",
        "2-norm": "l2",
        "min": "min",
        "max": "max",
        "mean": "mean",
        "std": "std",
    }

    def stats_from_tensor(vec):
        if vec is None or vec.numel() == 0:
            return {
                "1-norm": 0.0,
                "2-norm": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
            }
        v = vec.detach().float().reshape(-1)
        return {
            "1-norm": float(torch.norm(v, p=1).item()),
            "2-norm": float(torch.norm(v, p=2).item()),
            "min": float(torch.min(v).item()),
            "max": float(torch.max(v).item()),
            "mean": float(torch.mean(v).item()),
            "std": float(torch.std(v, unbiased=False).item()),
        }

    fieldnames = ["image_id", "image_path"]
    if unit == "bbox":
        fieldnames.extend(
            [
                "pred_idx",
                "raw_pred_idx",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "score",
                "pred_class",
                "xmin_mean",
                "ymin_mean",
                "xmax_mean",
                "ymax_mean",
                "score_mean",
                "xmin_std",
                "ymin_std",
                "xmax_std",
                "ymax_std",
                "score_std",
            ]
        )
        for class_idx in range(n_classes_hint):
            fieldnames.append(f"prob_{class_idx}_mean")
            fieldnames.append(f"prob_{class_idx}_std")
    else:
        fieldnames.append("num_preds")
        for prefix in (
            "xmin_mean",
            "ymin_mean",
            "xmax_mean",
            "ymax_mean",
            "xmin_std",
            "ymin_std",
            "xmax_std",
            "ymax_std",
            "score_mean",
            "score_std",
        ):
            for key in stat_keys:
                fieldnames.append(f"{prefix}_{stat_alias[key]}")
        for class_idx in range(n_classes_hint):
            for key in stat_keys:
                fieldnames.append(f"prob_{class_idx}_mean_{stat_alias[key]}")
            for key in stat_keys:
                fieldnames.append(f"prob_{class_idx}_std_{stat_alias[key]}")

    # Probe once to notify if forced-dropout hooks are unavailable on this model.
    probe_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)
    if len(probe_handles) == 0:
        print("[WARN] YOLOv5 detect head not found for forced MC-dropout hooks.")
    for h in probe_handles:
        h.remove()

    write_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
    writer_thread = threading.Thread(
        target=_mc_dropout_single_csv_writer,
        args=(write_queue, output_csv, fieldnames),
        daemon=True,
    )
    writer_thread.start()

    had_error = False
    try:
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            batch_size = len(images)
            batch_tensors = []
            image_ids = []
            image_paths = []
            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                image_ids.append(int(target["image_id"][0].item()))
                image_paths.append(target["path"])
                infer_tensor, _ratio, _pad, _resized_chw = preprocess_with_letterbox(
                    detector, images[sample_idx], device, requires_grad=False, auto=False
                )
                batch_tensors.append(infer_tensor)

            infer_batch = torch.cat(batch_tensors, dim=0)
            del batch_tensors

            # 1) Deterministic forward once: get final NMS predictions and raw pre-NMS indices.
            with torch.no_grad():
                det_output = detector.model(infer_batch, augment=False)
                det_raw_pred = det_output[0] if isinstance(det_output, (tuple, list)) else det_output
                det_raw_logits = det_output[1] if isinstance(det_output, (tuple, list)) and len(det_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    det_raw_pred,
                    det_raw_logits,
                    detector.confidence,
                    detector.iou_thresh,
                    classes=None,
                    agnostic=detector.agnostic,
                    return_indices=True,
                )

            feat_mean = None
            feat_m2 = None
            n_candidates = None
            n_classes = None
            run_count = 0
            mc_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)

            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        detector.zero_grad(set_to_none=True)
                        model_output = detector.model(infer_batch, augment=False)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = (
                            model_output[1]
                            if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                            else None
                        )

                        pred_batch = raw_prediction.detach().float()
                        bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                        score_vec = pred_batch[..., 4].unsqueeze(-1)
                        prob_mat = pred_batch[..., 5:].detach().float()
                        if prob_mat.numel() == 0 and raw_logits is not None:
                            prob_mat = torch.sigmoid(raw_logits.detach().float())
                        run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2)

                        if n_candidates is None:
                            n_candidates = int(run_features.shape[1])
                            n_classes = int(run_features.shape[2] - 5)
                            feat_dim = 5 + n_classes
                            feat_mean = torch.zeros((batch_size, n_candidates, feat_dim), device=device)
                            feat_m2 = torch.zeros((batch_size, n_candidates, feat_dim), device=device)

                        if int(run_features.shape[1]) != n_candidates:
                            raise ValueError("Raw candidate count changed across MC runs; expected fixed pre-NMS candidates.")

                        run_count += 1
                        delta = run_features - feat_mean
                        feat_mean = feat_mean + delta / run_count
                        feat_m2 = feat_m2 + delta * (run_features - feat_mean)
            finally:
                for h in mc_handles:
                    h.remove()

            if n_candidates is None:
                del infer_batch
                continue

            feat_std = torch.sqrt(torch.clamp(feat_m2 / max(run_count, 1), min=0.0))
            batch_rows = []

            for b in range(batch_size):
                image_id = image_ids[b]
                image_path = image_paths[b]
                det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = (
                    selected_indices[b]
                    if selected_indices and b < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                feat_mean_cpu = feat_mean[b].detach().float().cpu()
                feat_std_cpu = feat_std[b].detach().float().cpu()
                num_final = int(det_b.shape[0])
                valid_pairs = []
                for pred_idx in range(num_final):
                    raw_idx = int(raw_keep_b[pred_idx].detach().cpu().item())
                    if 0 <= raw_idx < n_candidates:
                        valid_pairs.append((pred_idx, raw_idx))

                if unit == "bbox":
                    for pred_idx, raw_idx in valid_pairs:
                        cls_idx = int(det_b[pred_idx, 5].detach().cpu().item()) if det_b.shape[1] > 5 else -1
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_idx,
                            "xmin": float(det_b[pred_idx, 0].detach().cpu().item()),
                            "ymin": float(det_b[pred_idx, 1].detach().cpu().item()),
                            "xmax": float(det_b[pred_idx, 2].detach().cpu().item()),
                            "ymax": float(det_b[pred_idx, 3].detach().cpu().item()),
                            "score": float(det_b[pred_idx, 4].detach().cpu().item()) if det_b.shape[1] > 4 else 0.0,
                            "pred_class": detector.names[cls_idx] if (detector.names is not None and cls_idx >= 0) else cls_idx,
                            "xmin_mean": float(feat_mean_cpu[raw_idx, 0].item()),
                            "ymin_mean": float(feat_mean_cpu[raw_idx, 1].item()),
                            "xmax_mean": float(feat_mean_cpu[raw_idx, 2].item()),
                            "ymax_mean": float(feat_mean_cpu[raw_idx, 3].item()),
                            "score_mean": float(feat_mean_cpu[raw_idx, 4].item()),
                            "xmin_std": float(feat_std_cpu[raw_idx, 0].item()),
                            "ymin_std": float(feat_std_cpu[raw_idx, 1].item()),
                            "xmax_std": float(feat_std_cpu[raw_idx, 2].item()),
                            "ymax_std": float(feat_std_cpu[raw_idx, 3].item()),
                            "score_std": float(feat_std_cpu[raw_idx, 4].item()),
                        }
                        class_count = int(n_classes) if n_classes is not None else 0
                        for class_idx in range(class_count):
                            row[f"prob_{class_idx}_mean"] = float(feat_mean_cpu[raw_idx, 5 + class_idx].item())
                            row[f"prob_{class_idx}_std"] = float(feat_std_cpu[raw_idx, 5 + class_idx].item())
                        batch_rows.append(row)
                else:
                    if pre_nms:
                        raw_keep = get_pre_nms_keep_indices(
                            det_raw_pred[b].detach().float(),
                            det_raw_logits[b].detach().float() if det_raw_logits is not None else None,
                            pre_nms_ratio=pre_nms_ratio,
                        )
                        raw_indices = [int(i.item()) for i in raw_keep]
                    else:
                        raw_indices = [raw_idx for _pred_idx, raw_idx in valid_pairs]
                    row = {"image_id": image_id, "image_path": image_path, "num_preds": len(raw_indices)}
                    if len(raw_indices) == 0:
                        for prefix in (
                            "xmin_mean",
                            "ymin_mean",
                            "xmax_mean",
                            "ymax_mean",
                            "xmin_std",
                            "ymin_std",
                            "xmax_std",
                            "ymax_std",
                            "score_mean",
                            "score_std",
                        ):
                            for key in stat_keys:
                                row[f"{prefix}_{stat_alias[key]}"] = 0.0
                        for class_idx in range(n_classes_hint):
                            for key in stat_keys:
                                row[f"prob_{class_idx}_mean_{stat_alias[key]}"] = 0.0
                            for key in stat_keys:
                                row[f"prob_{class_idx}_std_{stat_alias[key]}"] = 0.0
                    else:
                        raw_indices_tensor = torch.tensor(raw_indices, dtype=torch.long, device=feat_mean_cpu.device)
                        feat_mean_sel = feat_mean_cpu.index_select(0, raw_indices_tensor)
                        feat_std_sel = feat_std_cpu.index_select(0, raw_indices_tensor)
                        xmin_mean_vec = feat_mean_sel[:, 0].reshape(-1)
                        ymin_mean_vec = feat_mean_sel[:, 1].reshape(-1)
                        xmax_mean_vec = feat_mean_sel[:, 2].reshape(-1)
                        ymax_mean_vec = feat_mean_sel[:, 3].reshape(-1)
                        xmin_std_vec = feat_std_sel[:, 0].reshape(-1)
                        ymin_std_vec = feat_std_sel[:, 1].reshape(-1)
                        xmax_std_vec = feat_std_sel[:, 2].reshape(-1)
                        ymax_std_vec = feat_std_sel[:, 3].reshape(-1)
                        score_mean_vec = feat_mean_sel[:, 4].reshape(-1)
                        score_std_vec = feat_std_sel[:, 4].reshape(-1)

                        for key, val in stats_from_tensor(xmin_mean_vec).items():
                            row[f"xmin_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymin_mean_vec).items():
                            row[f"ymin_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmax_mean_vec).items():
                            row[f"xmax_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymax_mean_vec).items():
                            row[f"ymax_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmin_std_vec).items():
                            row[f"xmin_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymin_std_vec).items():
                            row[f"ymin_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(xmax_std_vec).items():
                            row[f"xmax_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(ymax_std_vec).items():
                            row[f"ymax_std_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(score_mean_vec).items():
                            row[f"score_mean_{stat_alias[key]}"] = val
                        for key, val in stats_from_tensor(score_std_vec).items():
                            row[f"score_std_{stat_alias[key]}"] = val

                        class_count = int(n_classes) if n_classes is not None else 0
                        for class_idx in range(n_classes_hint):
                            if class_idx < class_count:
                                prob_mean_vec = feat_mean_sel[:, 5 + class_idx].reshape(-1)
                                prob_std_vec = feat_std_sel[:, 5 + class_idx].reshape(-1)
                            else:
                                prob_mean_vec = torch.zeros((0,), dtype=torch.float32, device=feat_mean_cpu.device)
                                prob_std_vec = torch.zeros((0,), dtype=torch.float32, device=feat_mean_cpu.device)
                            for key, val in stats_from_tensor(prob_mean_vec).items():
                                row[f"prob_{class_idx}_mean_{stat_alias[key]}"] = val
                            for key, val in stats_from_tensor(prob_std_vec).items():
                                row[f"prob_{class_idx}_std_{stat_alias[key]}"] = val
                    batch_rows.append(row)

            write_queue.put(batch_rows)

            del selected_preds, selected_indices
            del infer_batch
    except Exception:
        had_error = True
        raise
    finally:
        if had_error:
            write_queue.put(None)
            writer_thread.join()
        else:
            write_queue.put(None)
            writer_thread.join()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved results CSV: {output_csv}")


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = parsed["layer_target_values"]
    target_layers = parsed["layer_target_layers"]
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_vector_reduction = parsed["layer_vector_reduction"]
    layer_pseudo_gt = parsed.get("layer_pseudo_gt", "cand")
    pre_nms = bool(parsed.get("pre_nms", False))
    pre_nms_ratio = float(parsed.get("pre_nms_ratio", 1.0))
    save_image_enabled = bool(parsed.get("save_image_enabled", False))
    viz_enabled = bool(save_image_enabled and unit == "image")
    viz_normalize = str(parsed.get("save_image_layer_grad_normalize", "layer_minmax")).strip().lower()
    viz_target_layer_cfg = parsed.get("save_image_layer_grad_target_layer", "target_layer")
    viz_max_per_group = int(parsed.get("save_image_layer_grad_max_num_per_group", 200))
    viz_save_mean_maps = bool(parsed.get("save_image_layer_grad_save_mean_maps", True))
    viz_save_diff_map = bool(parsed.get("save_image_layer_grad_save_diff_map", True))
    viz_save_per_image = bool(parsed.get("save_image_layer_grad_save_per_image", False))

    if not save_csv and not viz_enabled:
        return

    output_csv = run_dir / "layer_grad.csv"
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
    viz_target_layers = list(target_layers)
    if unit == "image" and viz_enabled:
        if isinstance(viz_target_layer_cfg, str):
            viz_layer_mode = viz_target_layer_cfg.strip().lower()
            if viz_layer_mode == "all_conv":
                viz_target_layers = [
                    name
                    for name, module in detector.model.named_modules()
                    if name and isinstance(module, torch.nn.Conv2d)
                ]
            else:
                viz_target_layers = list(target_layers)
        elif isinstance(viz_target_layer_cfg, (list, tuple)):
            viz_target_layers = [str(v).strip() for v in viz_target_layer_cfg if str(v).strip()]
        if not viz_target_layers:
            viz_target_layers = list(target_layers)

    layer_param_shapes = {}
    if unit == "image" and viz_enabled:
        for layer_name in viz_target_layers:
            try:
                layer_param_shapes[layer_name] = tuple(resolve_layer_parameter(detector.model, layer_name).shape)
            except Exception:
                layer_param_shapes[layer_name] = None
    all_layers_for_grad = list(target_layers)
    if unit == "image" and viz_enabled:
        all_layers_for_grad = list(dict.fromkeys(list(target_layers) + list(viz_target_layers)))
    catid_to_name = load_gt_category_maps(config, split) if viz_enabled else {}
    iou_match_threshold = parsed["gt_iou_match_threshold"] if viz_enabled else 0.45
    viz_maps = {"fn": [], "non_fn": []}
    viz_profiles = {"fn": [], "non_fn": []}
    viz_counts = {"fn": 0, "non_fn": 0}
    viz_saved_per_image = {"fn": 0, "non_fn": 0}
    viz_maps_saved = False
    viz_saved_early = False
    viz_dir = run_dir / "images"
    if viz_enabled and viz_save_per_image:
        (viz_dir / "per_image" / "fn").mkdir(parents=True, exist_ok=True)
        (viz_dir / "per_image" / "non_fn").mkdir(parents=True, exist_ok=True)

    def _save_layer_grad_group_maps_if_needed():
        if not viz_enabled:
            return False
        if not (viz_save_mean_maps or viz_save_diff_map):
            return False
        fn_mean = _stack_nanmean_maps(viz_maps["fn"])
        non_fn_mean = _stack_nanmean_maps(viz_maps["non_fn"])
        if viz_save_mean_maps:
            _save_heatmap_png(fn_mean, viz_dir / "fn_mean_map.png")
            _save_heatmap_png(non_fn_mean, viz_dir / "non_fn_mean_map.png")
        if viz_save_diff_map:
            l_max = max(fn_mean.shape[0], non_fn_mean.shape[0]) if (fn_mean.size or non_fn_mean.size) else 0
            f_max = max(fn_mean.shape[1], non_fn_mean.shape[1]) if (fn_mean.size or non_fn_mean.size) else 0
            fn_pad = np.full((l_max, f_max), np.nan, dtype=np.float32)
            non_fn_pad = np.full((l_max, f_max), np.nan, dtype=np.float32)
            if fn_mean.size:
                fn_pad[: fn_mean.shape[0], : fn_mean.shape[1]] = fn_mean
            if non_fn_mean.size:
                non_fn_pad[: non_fn_mean.shape[0], : non_fn_mean.shape[1]] = non_fn_mean
            diff_map = fn_pad - non_fn_pad
            _save_heatmap_png(diff_map, viz_dir / "diff_map.png")
        _save_layer_profile_plot(
            fn_profiles=viz_profiles["fn"],
            non_fn_profiles=viz_profiles["non_fn"],
            out_path=viz_dir / "layer_profile_mean_std_log.png",
            log_scale=True,
        )
        return True

    csv_file_handle = None
    csv_writer = None
    if save_csv:
        csv_file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, ratios, pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            batch_preds = None
            need_viz_batch = bool(
                viz_enabled
                and (not viz_maps_saved)
                and (viz_save_mean_maps or viz_save_diff_map or viz_save_per_image)
            )
            if need_viz_batch:
                detector.zero_grad(set_to_none=True)
                with torch.no_grad():
                    batch_preds, _bz_logits, _bz_obj, _bz_feat = detector(infer_batch)

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                infer_tensor = infer_batch[sample_idx: sample_idx + 1]
                if unit == "bbox":
                    bbox_rows = collect_bbox_layer_grads_per_target(
                        detector=detector,
                        input_tensor=infer_tensor,
                        target_values=target_values,
                        target_layers=target_layers,
                        map_reduction=layer_map_reduction,
                        vector_reduction=layer_vector_reduction,
                        pseudo_gt=layer_pseudo_gt,
                    )
                    if csv_writer is not None:
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
                    del bbox_rows
                else:
                    group_key = None
                    need_viz_for_sample = False
                    if need_viz_batch and batch_preds is not None:
                        pred_boxes = batch_preds[0][sample_idx]
                        pred_class_names = batch_preds[2][sample_idx]
                        gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                        gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                        is_fn = has_fn_for_image(
                            gt_boxes=gt_boxes,
                            gt_class_names=gt_class_names,
                            pred_boxes=pred_boxes,
                            pred_class_names=pred_class_names,
                            iou_match_threshold=iou_match_threshold,
                        )
                        group_key = "fn" if is_fn else "non_fn"
                        viz_counts[group_key] += 1
                        need_viz_for_sample = (
                            len(viz_maps[group_key]) < viz_max_per_group
                            and (viz_save_mean_maps or viz_save_diff_map or viz_save_per_image)
                        )

                    required_layers = []
                    if csv_writer is not None:
                        required_layers.extend(target_layers)
                    if need_viz_for_sample:
                        required_layers.extend(viz_target_layers)
                    required_layers = list(dict.fromkeys(required_layers))

                    grad_stats_all = {}
                    if required_layers:
                        use_raw_grad = bool(need_viz_for_sample)
                        grad_stats_all = collect_image_layer_grads_per_target(
                            detector=detector,
                            input_tensor=infer_tensor,
                            target_values=target_values,
                            target_layers=required_layers,
                            map_reduction=layer_map_reduction,
                            vector_reduction=[] if use_raw_grad else layer_vector_reduction,
                            pre_nms=pre_nms,
                            pre_nms_ratio=pre_nms_ratio,
                            pseudo_gt=layer_pseudo_gt,
                        )

                    if csv_writer is not None:
                        row = {"image_id": image_id, "image_path": image_path}
                        for target_value in target_values:
                            for layer_name in target_layers:
                                grad_key = f"{target_value}_{layer_name}"
                                grad_value = grad_stats_all.get(grad_key, [])
                                if use_raw_grad and layer_vector_reduction:
                                    vec = torch.tensor(_vector_from_grad_value(grad_value), dtype=torch.float32)
                                    stats = map_grad_tensor_to_numbers(vec)
                                    row[grad_key] = json.dumps(
                                        {k: float(stats[k]) for k in layer_vector_reduction},
                                        separators=(",", ":"),
                                    )
                                else:
                                    row[grad_key] = json.dumps(grad_value, separators=(",", ":"))
                        csv_writer.writerow(row)

                    if need_viz_batch and need_viz_for_sample and group_key is not None:
                        if len(viz_maps[group_key]) < viz_max_per_group:
                            grad_map = _build_layer_filter_map_from_grad_stats(
                                grad_stats=grad_stats_all,
                                target_values=target_values,
                                target_layers=viz_target_layers,
                                layer_param_shapes=layer_param_shapes,
                            )
                            grad_map_norm = _normalize_layer_map(grad_map, mode=viz_normalize)
                            with np.errstate(invalid="ignore"):
                                layer_profile = np.nanmean(grad_map_norm, axis=1).astype(np.float32)
                            viz_profiles[group_key].append(layer_profile)
                            viz_maps[group_key].append(grad_map_norm)
                            if viz_save_per_image:
                                out_path = (
                                    viz_dir
                                    / "per_image"
                                    / group_key
                                    / f"{image_id}_{viz_saved_per_image[group_key]:05d}.png"
                                )
                                _save_heatmap_png(grad_map_norm, out_path)
                                viz_saved_per_image[group_key] += 1
                        if (
                            not viz_maps_saved
                            and len(viz_maps["fn"]) >= viz_max_per_group
                            and len(viz_maps["non_fn"]) >= viz_max_per_group
                        ):
                            viz_maps_saved = _save_layer_grad_group_maps_if_needed()
                            viz_saved_early = bool(viz_maps_saved)
                    del grad_stats_all
            del infer_batch
            if batch_preds is not None:
                del batch_preds
    finally:
        if csv_file_handle is not None:
            csv_file_handle.close()

    if viz_enabled:
        if not viz_maps_saved:
            viz_maps_saved = _save_layer_grad_group_maps_if_needed()

        viz_summary = {
            "normalize": viz_normalize,
            "max_num_per_group": int(viz_max_per_group),
            "group_total_counts": {k: int(v) for k, v in viz_counts.items()},
            "group_used_counts": {k: int(len(viz_maps[k])) for k in ("fn", "non_fn")},
            "group_used_profile_counts": {k: int(len(viz_profiles[k])) for k in ("fn", "non_fn")},
            "target_layers_for_map": viz_target_layers,
            "save_mean_maps": bool(viz_save_mean_maps),
            "save_diff_map": bool(viz_save_diff_map),
            "save_per_image": bool(viz_save_per_image),
            "maps_saved_early": bool(viz_saved_early),
        }
        with open(viz_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(viz_summary, f, ensure_ascii=False, indent=2)

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    if viz_enabled:
        print(f"Saved layer-grad maps: {viz_dir}")


def run_predict(config, run_dir):
    parsed = parse_output_config(config.get("output", {}))
    uncertainty = parsed["uncertainty"]
    unit = parsed["unit"]

    if uncertainty == "gt":
        if unit == "image":
            run_fn_csv(config, run_dir)
            return
        if unit == "bbox":
            run_tp_csv(config, run_dir)
            return
        raise ValueError("output.uncertainty='gt' requires output.unit in {'image','bbox'}.")
    if uncertainty == "score":
        run_score_csv(config, run_dir)
        return
    if uncertainty == "meta_detect":
        run_meta_detect_csv(config, run_dir)
        return
    if uncertainty == "mc_dropout":
        run_mc_dropout_csv(config, run_dir)
        return
    if uncertainty == "full_softmax":
        run_full_softmax_csv(config, run_dir)
        return
    if uncertainty == "energy":
        run_energy_csv(config, run_dir)
        return
    if uncertainty == "entropy":
        run_entropy_csv(config, run_dir)
        return
    if uncertainty == "feature":
        run_feature_csv(config, run_dir)
        return
    if uncertainty == "feature_grad":
        run_feature_grad_csv(config, run_dir)
        return
    if uncertainty == "layer_grad":
        run_layer_grad_csv(config, run_dir)
        return
    raise ValueError(f"Unsupported uncertainty: {uncertainty}")
