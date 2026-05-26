import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dataloaders.dataloader_yolo import create_dataloader
from dataloaders.utils.data_utils import DATASET_CLASS_NAMES
from losses.loss import build_loss
from models.fcos import FCOSTorchObjectDetector
from models.yolo.models.experimental import attempt_load
from models.yolo.utils.general import coco80_to_coco91_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_default_faster_rcnn_coco_weight(path: Path = DEFAULT_FASTER_RCNN_COCO_WEIGHT) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url, str(path), progress=True)
    return path


def _load_matching_state_dict(model, state_dict):
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(filtered, strict=False)


def _set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _resolve_device(device_str):
    device_str = str(device_str or "cuda").strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _prepare_batch(images, img_size, device):
    if isinstance(img_size, int):
        out_h = out_w = int(img_size)
    else:
        out_h, out_w = int(img_size[0]), int(img_size[1])

    infer_tensors = []
    ratios = []
    pads = []
    pad_value = float(114.0 / 255.0)
    for img in images:
        # img: [C,H,W] float tensor in [0,1]
        c, h, w = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        if c != 3:
            raise ValueError(f"Expected 3-channel image tensor, got shape={tuple(img.shape)}")

        scale = min(float(out_h) / float(h), float(out_w) / float(w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized = F.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h = out_h - new_h
        pad_w = out_w - new_w
        top = int(pad_h // 2)
        bottom = int(pad_h - top)
        left = int(pad_w // 2)
        right = int(pad_w - left)
        resized = F.pad(resized, (left, right, top, bottom), value=pad_value)

        infer_tensors.append(resized)
        ratios.append((scale, scale))
        pads.append((float(left), float(top)))

    infer_batch = torch.stack(infer_tensors, dim=0).to(device=device, non_blocking=True)
    return infer_batch, ratios, pads


def _to_yolo_targets(targets, ratios, pads, img_h, img_w, device):
    rows = []
    coco91_to_80 = {int(cat_id): int(i) for i, cat_id in enumerate(coco80_to_coco91_class())}
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.numel() == 0:
            continue

        ratio_w, ratio_h = ratios[batch_idx]
        pad_w, pad_h = pads[batch_idx]
        b = boxes.clone().float()
        b[:, [0, 2]] = b[:, [0, 2]] * ratio_w + pad_w
        b[:, [1, 3]] = b[:, [1, 3]] * ratio_h + pad_h

        cls = labels.clone().long()
        if str(target.get("dataset_name", "")).lower() == "coco":
            mapped = []
            keep = []
            for i, v in enumerate(cls.tolist()):
                if v in coco91_to_80:
                    mapped.append(coco91_to_80[v])
                    keep.append(i)
            if not keep:
                continue
            b = b[keep]
            cls = torch.tensor(mapped, dtype=torch.long)

        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        valid = (w > 0.0) & (h > 0.0)
        if not valid.any():
            continue
        x1, y1, w, h, cls = x1[valid], y1[valid], w[valid], h[valid], cls[valid]

        xc = (x1 + w * 0.5) / float(img_w)
        yc = (y1 + h * 0.5) / float(img_h)
        wn = w / float(img_w)
        hn = h / float(img_h)
        batch_col = torch.full((xc.shape[0],), int(batch_idx), dtype=torch.float32)

        t = torch.stack(
            [
                batch_col,
                cls.float(),
                xc.float(),
                yc.float(),
                wn.float(),
                hn.float(),
            ],
            dim=1,
        )
        rows.append(t)

    if not rows:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)


def _active_dataset_names(config):
    raw = config.get("dataset", {}).get("used_dataset", "")
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        names = []
    return names


def _resolve_train_class_names(config):
    model_cfg = config.get("model", {})
    configured_names = model_cfg.get("class_names")
    if configured_names:
        if isinstance(configured_names, str):
            return [name.strip() for name in configured_names.split(",") if name.strip()]
        return [str(v) for v in configured_names]

    names = _active_dataset_names(config)
    if names and all(name in DATASET_CLASS_NAMES for name in names):
        first = list(DATASET_CLASS_NAMES[names[0]])
        if all(list(DATASET_CLASS_NAMES[name]) == first for name in names):
            return first
        raise ValueError("Mixed datasets with different class spaces are not supported.")
    if any(name in DATASET_CLASS_NAMES for name in names):
        raise ValueError("Mixed custom datasets with non-matching class spaces are not supported.")
    return None


def _default_faster_rcnn_coco_names():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return categories[1:] if categories and categories[0] == "__background__" else categories


def _resolve_faster_rcnn_class_names(config):
    names = _resolve_train_class_names(config)
    if names is not None:
        return list(names)
    active = _active_dataset_names(config)
    if active == ["coco"]:
        return _default_faster_rcnn_coco_names()
    return None


def _rebuild_detect_head_for_class_count(model, class_names, device):
    if not class_names:
        return model

    target_nc = int(len(class_names))
    detect = model.model[-1]
    current_nc = int(getattr(detect, "nc", target_nc))

    if current_nc != target_nc:
        detect.nc = target_nc
        detect.no = target_nc + 5
        rebuilt = []
        for conv in detect.m:
            new_conv = nn.Conv2d(
                conv.in_channels,
                detect.na * detect.no,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode,
            )
            new_conv.to(device=device, dtype=conv.weight.dtype)
            rebuilt.append(new_conv)
        detect.m = nn.ModuleList(rebuilt)
        detect.grid = [torch.empty(0, device=device) for _ in range(detect.nl)]
        detect.anchor_grid = [torch.empty(0, device=device) for _ in range(detect.nl)]
        if hasattr(model, "_initialize_biases"):
            model._initialize_biases()

    model.names = list(class_names)
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["nc"] = target_nc
        model.yaml["names"] = list(class_names)
    model.nc = target_nc
    return model


def _build_model_for_train(config, device):
    model_cfg = config.get("model", {})
    weights = model_cfg.get("weights")
    if isinstance(weights, str):
        weights = weights.strip()
    use_finetune = bool(weights)

    if use_finetune:
        model = attempt_load(weights, device=device, fuse=False)
    else:
        arch_ref = str((PROJECT_ROOT / "models" / "yolo" / "weights" / "yolov5x.pt").resolve())
        model = attempt_load(arch_ref, device=device, fuse=False)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                try:
                    module.reset_parameters()
                except Exception:
                    continue

    class_names = _resolve_train_class_names(config)
    model = _rebuild_detect_head_for_class_count(model, class_names, device)

    hyp = {
        "box": float(config.get("loss", {}).get("box", 0.05)),
        "cls": float(config.get("loss", {}).get("cls", 0.5)),
        "obj": float(config.get("loss", {}).get("obj", 1.0)),
        "cls_pw": 1.0,
        "obj_pw": 1.0,
        "fl_gamma": 0.0,
        "label_smoothing": 0.0,
        "anchor_t": 4.0,
    }
    model.hyp = hyp
    model.requires_grad_(True)
    model.float().to(device)
    model.train()
    return model, use_finetune


def _run_one_epoch(model, dataloader, loss_fn, optimizer, img_size, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")

    def _normalize_preds_for_yolo_loss(model_output):
        # ComputeLoss expects a list of per-detection-layer tensors:
        # [bs, na, ny, nx, no] for each detection layer.
        if isinstance(model_output, list):
            return model_output
        if isinstance(model_output, tuple):
            # In eval mode our Detect head returns:
            # (cat_pred, cat_logits, per_layer_preds, priors)
            if len(model_output) > 2 and isinstance(model_output[2], list):
                return model_output[2]
            if len(model_output) > 0 and isinstance(model_output[0], list):
                return model_output[0]
        raise TypeError(
            f"Unsupported model output type for YOLO loss: {type(model_output)}"
        )

    for images, targets in pbar:
        infer_batch, ratios, pads = _prepare_batch(images, img_size=img_size, device=device)
        img_h, img_w = int(infer_batch.shape[2]), int(infer_batch.shape[3])
        yolo_targets = _to_yolo_targets(targets, ratios, pads, img_h=img_h, img_w=img_w, device=device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            preds = _normalize_preds_for_yolo_loss(model(infer_batch))
            loss, _loss_items = loss_fn(preds, yolo_targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = _normalize_preds_for_yolo_loss(model(infer_batch))
                loss, _loss_items = loss_fn(preds, yolo_targets)

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        total_steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")

        del infer_batch, preds, yolo_targets, loss

    mean_loss = (total_loss / total_steps) if total_steps else 0.0
    return mean_loss


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss):
    ckpt = {
        "epoch": int(epoch),
        "model": deepcopy(model).half().cpu(),
        "optimizer": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "date": datetime.now().isoformat(),
    }
    torch.save(ckpt, path)


def _target_to_faster_rcnn(target, device):
    boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device=device, dtype=torch.float32)
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).to(device=device, dtype=torch.int64)
    dataset_name = str(target.get("dataset_name", "")).lower()
    if dataset_name == "coco":
        # Torchvision COCO Faster R-CNN uses COCO category ids directly, including category id gaps.
        labels = labels.clamp(min=1)
    else:
        labels = labels + 1
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) if boxes.numel() else torch.zeros((0,), dtype=torch.bool, device=device)
    return {"boxes": boxes[valid], "labels": labels[valid]}


def _resolve_fcos_class_names(config):
    names = _resolve_train_class_names(config)
    if names is not None:
        return list(names)
    active = _active_dataset_names(config)
    if active == ["coco"]:
        return list(DATASET_CLASS_NAMES["coco"])
    return None


def _target_to_fcos(target, image, device):
    from fcos_core.structures.bounding_box import BoxList

    boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device=device, dtype=torch.float32)
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).to(device=device, dtype=torch.int64)
    dataset_name = str(target.get("dataset_name", "")).lower()
    if dataset_name == "coco":
        coco91_to_80 = {int(cat_id): int(i) for i, cat_id in enumerate(coco80_to_coco91_class())}
        mapped = []
        keep = []
        for idx, value in enumerate(labels.tolist()):
            if int(value) in coco91_to_80:
                mapped.append(coco91_to_80[int(value)] + 1)
                keep.append(idx)
        if keep:
            boxes = boxes[keep]
            labels = torch.tensor(mapped, dtype=torch.int64, device=device)
        else:
            boxes = boxes[:0]
            labels = labels[:0]
    else:
        labels = labels + 1

    if boxes.numel():
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (labels > 0)
        boxes = boxes[valid]
        labels = labels[valid]
    width = int(image.shape[-1])
    height = int(image.shape[-2])
    boxlist = BoxList(boxes, (width, height), mode="xyxy")
    boxlist.add_field("labels", labels)
    return boxlist


def _build_fcos_for_train(config, device):
    model_cfg = config.get("model", {})
    class_names = _resolve_fcos_class_names(config)
    if not class_names:
        raise ValueError("Could not resolve FCOS class names for training.")

    weights_path = str(model_cfg.get("weights", "") or "").strip()
    use_finetune = bool(weights_path)
    detector = FCOSTorchObjectDetector(
        model_weight=weights_path if weights_path else None,
        device=str(device),
        names=class_names,
        mode="train",
        confidence=float(model_cfg.get("confidence_threshold", 0.05)),
        iou_thresh=float(model_cfg.get("iou_threshold", 0.6)),
    )
    model = detector.detector_model
    model.to(device)
    model.train()
    return detector, model, use_finetune, class_names, detector.cfg


def _run_fcos_one_epoch(detector, model, dataloader, optimizer, device, train_mode=True):
    model.train()
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    for images, targets in pbar:
        image_list = detector.preprocess_images(images)
        target_list = [_target_to_fcos(t, img, device) for img, t in zip(image_list, targets)]
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


def _save_fcos_ckpt(path, epoch, model, optimizer, train_loss, val_loss, class_names, cfg):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "class_names": list(class_names),
            "num_classes": int(len(class_names) + 1),
            "fcos_cfg": cfg.dump() if hasattr(cfg, "dump") else None,
            "date": datetime.now().isoformat(),
        },
        path,
    )


def _run_fcos_train(config, run_dir, device, epochs, lr, weight_decay):
    train_loader = create_dataloader(config, split="train")
    val_loader = None
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None
    detector, model, use_finetune, class_names, cfg = _build_fcos_for_train(config, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    weights_dir = Path(run_dir) / "weights"
    best_metric = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_fcos_one_epoch(detector, model, train_loader, optimizer, device, train_mode=True)
        val_loss = None
        if val_loader is not None:
            val_loss = _run_fcos_one_epoch(detector, model, val_loader, optimizer, device, train_mode=False)
        metric = val_loss if val_loss is not None else train_loss
        _save_fcos_ckpt(weights_dir / "last.pt", epoch, model, optimizer, train_loss, val_loss, class_names, cfg)
        if metric <= best_metric:
            best_metric = metric
            _save_fcos_ckpt(weights_dir / "best.pt", epoch, model, optimizer, train_loss, val_loss, class_names, cfg)
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
    return model, use_finetune, class_names, history, best_metric


def _build_faster_rcnn_for_train(config, device):
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
    class_names = _resolve_faster_rcnn_class_names(config)
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
        payload = _torch_load(weights_path, map_location=device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None and isinstance(payload, dict):
            state_dict = payload.get("state_dict", payload)
        _load_matching_state_dict(model, state_dict)
        use_finetune = True
    elif weights_path and pretrained and Path(weights_path).name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name:
        state_dict = _torch_load(_ensure_default_faster_rcnn_coco_weight(Path(weights_path)), map_location=device)
        _load_matching_state_dict(model, state_dict)
    elif pretrained:
        state_dict = _torch_load(_ensure_default_faster_rcnn_coco_weight(), map_location=device)
        _load_matching_state_dict(model, state_dict)
    model.to(device)
    model.train()
    return model, use_finetune, class_names


def _run_faster_rcnn_one_epoch(model, dataloader, optimizer, device, train_mode=True):
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


def _save_faster_rcnn_ckpt(path, epoch, model, optimizer, train_loss, val_loss, class_names):
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


def _run_faster_rcnn_train(config, run_dir, device, epochs, lr, weight_decay):
    train_loader = create_dataloader(config, split="train")
    val_loader = None
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None
    model, use_finetune, class_names = _build_faster_rcnn_for_train(config, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    weights_dir = Path(run_dir) / "weights"
    best_metric = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_faster_rcnn_one_epoch(model, train_loader, optimizer, device, train_mode=True)
        val_loss = None
        if val_loader is not None:
            val_loss = _run_faster_rcnn_one_epoch(model, val_loader, optimizer, device, train_mode=False)
        metric = val_loss if val_loss is not None else train_loss
        _save_faster_rcnn_ckpt(weights_dir / "last.pt", epoch, model, optimizer, train_loss, val_loss, class_names)
        if metric <= best_metric:
            best_metric = metric
            _save_faster_rcnn_ckpt(weights_dir / "best.pt", epoch, model, optimizer, train_loss, val_loss, class_names)
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
    return model, use_finetune, class_names, history, best_metric


def run_train(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed")
    epochs = int(training_cfg.get("epochs", 10))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    if epochs <= 0:
        raise ValueError("training.epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("training.lr must be > 0.")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0.")

    _set_seed(seed)
    device = _resolve_device(config.get("model", {}).get("device", "cuda"))
    print(f"[train] device={device}")
    if str(device) == "cpu":
        print("[train][warn] CUDA unavailable -> training on CPU (very slow).")

    model_type = str(config.get("model", {}).get("type", "yolov5")).strip().lower()
    if model_type in {"fcos"}:
        model, use_finetune, class_names, history, best_metric = _run_fcos_train(
            config=config,
            run_dir=run_dir,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        summary = {
            "mode": "train",
            "model_type": "fcos",
            "epochs": int(epochs),
            "seed": None if seed is None else int(seed),
            "device": str(device),
            "finetune": bool(use_finetune),
            "num_classes": int(len(class_names)),
            "num_classes_with_background": int(len(class_names) + 1),
            "class_names": list(class_names),
            "history": history,
            "best_metric": float(best_metric),
            "weights": {
                "last": str((weights_dir / "last.pt").resolve()),
                "best": str((weights_dir / "best.pt").resolve()),
            },
        }
        with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return
    if model_type in {"faster_rcnn", "faster-rcnn", "frcnn"}:
        model, use_finetune, class_names, history, best_metric = _run_faster_rcnn_train(
            config=config,
            run_dir=run_dir,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        summary = {
            "mode": "train",
            "model_type": "faster_rcnn",
            "epochs": int(epochs),
            "seed": None if seed is None else int(seed),
            "device": str(device),
            "finetune": bool(use_finetune),
            "num_classes": int(len(class_names)),
            "num_classes_with_background": int(len(class_names) + 1),
            "class_names": list(class_names),
            "history": history,
            "best_metric": float(best_metric),
            "weights": {
                "last": str((weights_dir / "last.pt").resolve()),
                "best": str((weights_dir / "best.pt").resolve()),
            },
        }
        with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    train_loader = create_dataloader(config, split="train")
    val_loader = None
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None

    model, use_finetune = _build_model_for_train(config, device)
    loss_fn = build_loss(config.get("model", {}).get("type", "yolov5"), model, config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_metric = float("inf")
    history = []
    img_size = config.get("model", {}).get("img_size", 640)

    for epoch in range(1, epochs + 1):
        train_loss = _run_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            img_size=img_size,
            device=device,
            train_mode=True,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _run_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                img_size=img_size,
                device=device,
                train_mode=False,
            )

        metric = val_loss if val_loss is not None else train_loss
        _save_ckpt(weights_dir / "last.pt", epoch, model, optimizer, train_loss, val_loss)
        if metric <= best_metric:
            best_metric = metric
            _save_ckpt(weights_dir / "best.pt", epoch, model, optimizer, train_loss, val_loss)

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
        "epochs": int(epochs),
        "seed": None if seed is None else int(seed),
        "device": str(device),
        "finetune": bool(use_finetune),
        "num_classes": int(getattr(model, "nc", getattr(model.model[-1], "nc", 0))),
        "class_names": list(getattr(model, "names", [])),
        "history": history,
        "best_metric": float(best_metric),
        "weights": {
            "last": str((weights_dir / "last.pt").resolve()),
            "best": str((weights_dir / "best.pt").resolve()),
        },
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
