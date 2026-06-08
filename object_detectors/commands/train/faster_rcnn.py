import json
from datetime import datetime
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from commands.train.common import (
    active_dataset_names,
    count_total_params,
    count_trainable_params,
    load_matching_state_dict,
    resolve_train_class_names,
    torch_load,
)
from dataloaders.dataloader_yolo import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _ensure_default_coco_weight(path: Path = DEFAULT_FASTER_RCNN_COCO_WEIGHT) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url, str(path), progress=True)
    return path


def _default_coco_names():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return categories[1:] if categories and categories[0] == "__background__" else categories


def _resolve_class_names(config):
    names = resolve_train_class_names(config)
    if names is not None:
        return list(names)
    active = active_dataset_names(config)
    if active == ["coco"]:
        return _default_coco_names()
    return None


def _target_to_faster_rcnn(target, device):
    boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device=device, dtype=torch.float32)
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).to(device=device, dtype=torch.int64)
    dataset_name = str(target.get("dataset_name", "")).lower()
    if dataset_name == "coco":
        labels = labels.clamp(min=1)
    else:
        labels = labels + 1
    valid = (
        (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if boxes.numel()
        else torch.zeros((0,), dtype=torch.bool, device=device)
    )
    return {"boxes": boxes[valid], "labels": labels[valid]}


def _build_model(config, device):
    model_cfg = config.get("model", {})
    pretrained = bool(model_cfg.get("pretrained", True))
    weights = None
    img_size = int(model_cfg.get("img_size", 640))
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None if weights is None else None,
        min_size=img_size,
        max_size=img_size,
        box_score_thresh=float(model_cfg.get("confidence_threshold", 0.25)),
        box_nms_thresh=float(model_cfg.get("iou_threshold", 0.45)),
        box_detections_per_img=int(model_cfg.get("max_det", 300)),
    )
    class_names = _resolve_class_names(config)
    if not class_names:
        raise ValueError("Could not resolve Faster R-CNN class names for training.")
    num_classes = len(class_names) + 1
    current_classes = int(model.roi_heads.box_predictor.cls_score.out_features)
    if current_classes != num_classes:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    weights_path = str(model_cfg.get("weights", "") or "").strip()
    use_finetune = False
    if weights_path and Path(weights_path).is_file():
        payload = torch_load(weights_path, map_location=device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None and isinstance(payload, dict):
            state_dict = payload.get("state_dict", payload)
        load_matching_state_dict(model, state_dict)
        use_finetune = True
    elif weights_path and pretrained and Path(weights_path).name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name:
        state_dict = torch_load(_ensure_default_coco_weight(Path(weights_path)), map_location=device)
        load_matching_state_dict(model, state_dict)
    elif pretrained:
        state_dict = torch_load(_ensure_default_coco_weight(), map_location=device)
        load_matching_state_dict(model, state_dict)
    model.to(device)
    model.train()
    return model, use_finetune, class_names


def _run_one_epoch(model, dataloader, optimizer, device, train_mode=True):
    model.train()
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    for images, targets in pbar:
        image_list = [img.to(device=device, dtype=torch.float32) for img in images]
        target_list = [_target_to_faster_rcnn(t, device) for t in targets]
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(image_list, target_list)
            loss = sum(v for v in loss_dict.values())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = model(image_list, target_list)
                loss = sum(v for v in loss_dict.values())
        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        total_steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        del image_list, target_list, loss_dict, loss
    return (total_loss / total_steps) if total_steps else 0.0


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss, class_names):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "class_names": list(class_names),
            "num_classes": int(len(class_names) + 1),
            "date": datetime.now().isoformat(),
        },
        path,
    )


def run_train(config, run_dir, device, epochs, lr, weight_decay):
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    training_cfg = config.get("training", {})

    train_loader = create_dataloader(config, split="train")
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None
    model, use_finetune, class_names = _build_model(config, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    print(f"[train] trainable_params={count_trainable_params(model)} total_params={count_total_params(model)}")

    best_metric = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_one_epoch(model, train_loader, optimizer, device, train_mode=True)
        val_loss = None
        if val_loader is not None:
            val_loss = _run_one_epoch(model, val_loader, optimizer, device, train_mode=False)
        metric = val_loss if val_loss is not None else train_loss
        _save_ckpt(weights_dir / "last.pt", epoch, model, optimizer, train_loss, val_loss, class_names)
        if metric <= best_metric:
            best_metric = metric
            _save_ckpt(weights_dir / "best.pt", epoch, model, optimizer, train_loss, val_loss, class_names)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "best_metric": float(best_metric),
            }
        )
        print(
            f"[train] epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={'none' if val_loss is None else f'{val_loss:.6f}'}"
        )

    summary = {
        "mode": "train",
        "model_type": "faster_rcnn",
        "epochs": int(epochs),
        "seed": None if training_cfg.get("seed") is None else int(training_cfg.get("seed")),
        "device": str(device),
        "finetune": bool(use_finetune),
        "num_classes": int(len(class_names)),
        "num_classes_with_background": int(len(class_names) + 1),
        "class_names": list(class_names),
        "history": history,
        "best_metric": float(best_metric),
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "weights": {
            "last": str((weights_dir / "last.pt").resolve()),
            "best": str((weights_dir / "best.pt").resolve()),
        },
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["run_train"]
