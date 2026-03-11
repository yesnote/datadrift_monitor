import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dataloaders.dataloader_yolo import create_dataloader, load_config
from models.yolo.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector


YOLO_DEFAULT_CONFIDENCE = 0.4
YOLO_DEFAULT_IOU_THRESH = 0.45
TARGET_VALUE_OPTIONS = {"obj", "cls"}
TARGET_LAYER_TO_INDEX = {"P3": 0, "P4": 1, "P5": 2}


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def create_run_dir():
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_predict_targets(predict_cfg):
    target_values = [v.lower() for v in normalize_to_list(predict_cfg.get("target_value", ["obj"]))]
    target_layers = [v.upper() for v in normalize_to_list(predict_cfg.get("target_layer", ["P3", "P4", "P5"]))]

    invalid_values = [v for v in target_values if v not in TARGET_VALUE_OPTIONS]
    if invalid_values:
        raise ValueError(f"Unsupported target_value(s): {invalid_values}. Use {sorted(TARGET_VALUE_OPTIONS)}")

    invalid_layers = [v for v in target_layers if v not in TARGET_LAYER_TO_INDEX]
    if invalid_layers:
        raise ValueError(f"Unsupported target_layer(s): {invalid_layers}. Use {sorted(TARGET_LAYER_TO_INDEX)}")

    return target_values, target_layers


def get_dataset_cfg(config):
    dataset_root_cfg = config["dataset"]
    used_dataset = dataset_root_cfg["used_dataset"]
    if used_dataset.lower() != "coco":
        raise ValueError("This script currently supports COCO only.")
    if used_dataset not in dataset_root_cfg:
        raise ValueError(f"dataset.{used_dataset} is missing in config.")
    return dataset_root_cfg[used_dataset]


def get_annotation_path(config, split):
    dataset_cfg = get_dataset_cfg(config)
    root = Path(dataset_cfg["root"])
    ann_dir = dataset_cfg["annotation_dir"]
    ann_name = dataset_cfg[f"{split}_annotation_file"]
    return root / ann_dir / ann_name


def load_coco_category_maps(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(c["id"]): c["name"] for c in payload.get("categories", [])}


def build_detector(config):
    model_cfg = config["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    confidence = model_cfg.get("confidence_threshold", YOLO_DEFAULT_CONFIDENCE)
    iou_thresh = model_cfg.get("iou_threshold", YOLO_DEFAULT_IOU_THRESH)

    detector = YOLOV5TorchObjectDetector(
        model_weight=model_cfg["weights"],
        device=device,
        img_size=(model_cfg["img_size"], model_cfg["img_size"]),
        names=None,
        mode="eval",
        confidence=confidence,
        iou_thresh=iou_thresh,
    )
    detector.eval().to(device)
    return detector, device


def build_target_scalar(target_value, preds, logits, objectness):
    if target_value == "obj":
        if len(objectness) == 0 or objectness[0].numel() == 0:
            return None
        return objectness[0].sum()

    if target_value == "cls":
        if len(logits) == 0 or logits[0].numel() == 0:
            return None
        pred_classes = preds[1][0]
        if len(pred_classes) == 0:
            return None
        cls_idx = torch.tensor(pred_classes, device=logits[0].device, dtype=torch.long)
        cls_idx = cls_idx.clamp_(0, logits[0].shape[1] - 1)
        det_idx = torch.arange(cls_idx.shape[0], device=logits[0].device)
        return logits[0][det_idx, cls_idx].sum()

    raise ValueError(f"Unsupported target_value: {target_value}")


def collect_gradients(
    target_values,
    target_layers,
    preds,
    logits,
    objectness,
    features,
    save_grad_tensors=False,
):
    grad_metrics = {}
    grad_tensors = {}

    for target_value in target_values:
        target_scalar = build_target_scalar(target_value, preds, logits, objectness)

        for layer_name in target_layers:
            layer_idx = TARGET_LAYER_TO_INDEX[layer_name]
            fmap = features[layer_idx]
            key = f"d{target_value}_d{layer_name}"

            if target_scalar is None:
                grad_metrics[key] = 0.0
                if save_grad_tensors:
                    grad_tensors[key] = np.zeros(tuple(fmap.shape), dtype=np.float32)
                continue

            grad = torch.autograd.grad(
                target_scalar,
                fmap,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad = torch.zeros_like(fmap)

            grad_abs = grad.detach().abs()
            grad_metrics[key] = float(grad_abs.mean().item())
            if save_grad_tensors:
                grad_tensors[key] = grad.detach().cpu().numpy().astype(np.float32)

    return grad_metrics, grad_tensors


def has_fn_for_image(gt_boxes, gt_class_names, pred_boxes, pred_class_names, iou_match_threshold):
    matched_pred_indices = set()

    for gt_box, gt_name in zip(gt_boxes, gt_class_names):
        found_match = False
        for pred_idx, (pred_box, pred_name) in enumerate(zip(pred_boxes, pred_class_names)):
            if pred_idx in matched_pred_indices:
                continue
            if gt_name != pred_name:
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            return 1
    return 0


def run_predict(config, run_dir):
    predict_cfg = config.get("predict", {})
    split = predict_cfg.get("split", "val")
    iou_match_threshold = predict_cfg.get("iou_match_threshold", 0.5)
    output_name = predict_cfg.get("output_csv_name", f"fn_gt_{split}.csv")
    output_csv = run_dir / output_name
    save_grad_tensors = bool(predict_cfg.get("save_grad_tensors", True))
    target_values, target_layers = parse_predict_targets(predict_cfg)
    grad_tensor_dir = run_dir / "grad_tensors"
    if save_grad_tensors:
        grad_tensor_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = get_annotation_path(config, split)
    catid_to_name = load_coco_category_maps(annotation_path)
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)

    rows = []
    for images, targets in dataloader:
        if images.shape[0] != 1:
            raise ValueError("For FN+gradient export, dataloader batch_size must be 1.")

        detector.zero_grad(set_to_none=True)
        images = images.to(device)
        preds, logits, objectness, features = detector(images)

        pred_boxes = preds[0][0]
        pred_class_names = preds[2][0]

        target = targets[0]
        gt_boxes_tensor = target["boxes"]
        gt_labels_tensor = target["labels"]
        gt_boxes = gt_boxes_tensor.tolist() if gt_boxes_tensor.numel() else []
        gt_class_names = [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]

        has_fn = has_fn_for_image(
            gt_boxes=gt_boxes,
            gt_class_names=gt_class_names,
            pred_boxes=pred_boxes,
            pred_class_names=pred_class_names,
            iou_match_threshold=iou_match_threshold,
        )

        grad_metrics, grad_tensors = collect_gradients(
            target_values=target_values,
            target_layers=target_layers,
            preds=preds,
            logits=logits,
            objectness=objectness,
            features=features,
            save_grad_tensors=save_grad_tensors,
        )

        image_id = int(target["image_id"][0].item())
        grad_file = ""
        if save_grad_tensors:
            grad_path = grad_tensor_dir / f"{image_id}.npz"
            np.savez_compressed(grad_path, **grad_tensors)
            grad_file = str(grad_path)

        rows.append(
            {
                "image_id": image_id,
                "image_path": target["path"],
                "has_fn": has_fn,
                "grad_file": grad_file,
                **grad_metrics,
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "image_path", "has_fn", "grad_file"]
        for target_value in target_values:
            for layer_name in target_layers:
                fieldnames.append(f"d{target_value}_d{layer_name}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    fn_images = sum(r["has_fn"] for r in rows)
    summary = {
        "mode": "predict",
        "split": split,
        "target_values": target_values,
        "target_layers": target_layers,
        "save_grad_tensors": save_grad_tensors,
        "total_images": total,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total) if total else 0.0,
        "output_csv": str(output_csv),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved FN GT CSV: {output_csv}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
    print(f"Total images: {total}")
    print(f"Images with FN (has_fn=1): {fn_images}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/predict_yolov5.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    mode = str(config.get("mode", "")).lower()

    run_dir = create_run_dir()
    shutil.copy2(config_path, run_dir / "used_config.yaml")

    if mode == "predict":
        run_predict(config, run_dir)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Only 'predict' is implemented in main.py.")


if __name__ == "__main__":
    main()
