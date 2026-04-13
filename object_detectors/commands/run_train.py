import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.dataloader_yolo import create_dataloader
from losses.loss import build_loss
from models.yolo.models.experimental import attempt_load
from models.yolo.utils.general import coco80_to_coco91_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "history": history,
        "best_metric": float(best_metric),
        "weights": {
            "last": str((weights_dir / "last.pt").resolve()),
            "best": str((weights_dir / "best.pt").resolve()),
        },
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
