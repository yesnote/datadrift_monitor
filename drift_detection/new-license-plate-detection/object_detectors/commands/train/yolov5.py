import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from commands.train.common import (
    autocast_context,
    count_total_params,
    count_trainable_params,
    make_grad_scaler,
    map_coco91_to_80,
    merge_epoch_timing,
    resolve_train_class_names,
    trainable_parameters,
    training_options,
)
from dataloaders.dataloader_yolo import create_dataloader
from losses.loss import build_loss
from models.yolo.models.experimental import attempt_load


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_batch(images, img_size, device):
    if isinstance(img_size, int):
        out_h = out_w = int(img_size)
    else:
        out_h, out_w = int(img_size[0]), int(img_size[1])

    if images and all(
        int(img.shape[0]) == 3 and int(img.shape[1]) == out_h and int(img.shape[2]) == out_w for img in images
    ):
        infer_batch = torch.stack(images, dim=0).to(device=device, non_blocking=True)
        ratios = [(1.0, 1.0)] * len(images)
        pads = [(0.0, 0.0)] * len(images)
        return infer_batch, ratios, pads

    infer_tensors = []
    ratios = []
    pads = []
    pad_value = float(114.0 / 255.0)
    for img in images:
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
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.numel() == 0:
            continue

        ratio_w, ratio_h = ratios[batch_idx]
        pad_w, pad_h = pads[batch_idx]
        b = boxes.to(device=device, dtype=torch.float32, non_blocking=True).clone()
        b[:, [0, 2]] = b[:, [0, 2]] * ratio_w + pad_w
        b[:, [1, 3]] = b[:, [1, 3]] * ratio_h + pad_h

        cls = labels.to(device=device, dtype=torch.long, non_blocking=True).clone()
        if str(target.get("dataset_name", "")).lower() == "coco":
            mapped, keep = map_coco91_to_80(cls, offset=0)
            if not keep.any():
                continue
            b = b[keep]
            cls = mapped.to(dtype=torch.long)

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
        batch_col = torch.full((xc.shape[0],), int(batch_idx), dtype=torch.float32, device=device)
        rows.append(torch.stack([batch_col, cls.float(), xc.float(), yc.float(), wn.float(), hn.float()], dim=1))

    if not rows:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0).to(dtype=torch.float32)


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


def _build_model(config, device):
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

    class_names = resolve_train_class_names(config)
    model = _rebuild_detect_head_for_class_count(model, class_names, device)
    model.hyp = {
        "box": float(config.get("loss", {}).get("box", 0.05)),
        "cls": float(config.get("loss", {}).get("cls", 0.5)),
        "obj": float(config.get("loss", {}).get("obj", 1.0)),
        "cls_pw": 1.0,
        "obj_pw": 1.0,
        "fl_gamma": 0.0,
        "label_smoothing": 0.0,
        "anchor_t": 4.0,
    }
    model.requires_grad_(True)
    model.float().to(device)
    model.train()
    return model, use_finetune


def _freeze_feature_extractor(model):
    for module in model.model[:-1]:
        module.requires_grad_(False)
        module.eval()
    model.model[-1].requires_grad_(True)
    model.model[-1].train()


def _apply_frozen_module_modes(model, freeze_feature_extractor):
    if not freeze_feature_extractor:
        return
    for module in model.model[:-1]:
        module.eval()
    model.model[-1].train()


def _normalize_preds_for_yolo_loss(model_output):
    if isinstance(model_output, list):
        return model_output
    if isinstance(model_output, tuple):
        if len(model_output) > 2 and isinstance(model_output[2], list):
            return model_output[2]
        if len(model_output) > 0 and isinstance(model_output[0], list):
            return model_output[0]
    raise TypeError(f"Unsupported model output type for YOLO loss: {type(model_output)}")


def _run_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    img_size,
    device,
    train_mode=True,
    amp=False,
    scaler=None,
    freeze_feature_extractor=False,
    log_timing=False,
):
    if train_mode:
        model.train()
        _apply_frozen_module_modes(model, freeze_feature_extractor)
    else:
        model.eval()

    timing = {"data_sec": 0.0, "target_sec": 0.0, "forward_loss_sec": 0.0, "backward_step_sec": 0.0}
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    next_data_start = time.perf_counter()

    for images, targets in pbar:
        if log_timing:
            timing["data_sec"] += time.perf_counter() - next_data_start

        t_target = time.perf_counter()
        infer_batch, ratios, pads = _prepare_batch(images, img_size=img_size, device=device)
        img_h, img_w = int(infer_batch.shape[2]), int(infer_batch.shape[3])
        yolo_targets = _to_yolo_targets(targets, ratios, pads, img_h=img_h, img_w=img_w, device=device)
        if log_timing:
            timing["target_sec"] += time.perf_counter() - t_target

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            t_forward = time.perf_counter()
            with autocast_context(device, amp):
                preds = _normalize_preds_for_yolo_loss(model(infer_batch))
                loss, _loss_items = loss_fn(preds, yolo_targets)
            if log_timing:
                timing["forward_loss_sec"] += time.perf_counter() - t_forward

            t_backward = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if log_timing:
                timing["backward_step_sec"] += time.perf_counter() - t_backward
        else:
            with torch.no_grad():
                t_forward = time.perf_counter()
                with autocast_context(device, amp):
                    preds = _normalize_preds_for_yolo_loss(model(infer_batch))
                    loss, _loss_items = loss_fn(preds, yolo_targets)
                if log_timing:
                    timing["forward_loss_sec"] += time.perf_counter() - t_forward

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        total_steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        del infer_batch, preds, yolo_targets, loss
        next_data_start = time.perf_counter()

    mean_loss = (total_loss / total_steps) if total_steps else 0.0
    if log_timing:
        return mean_loss, {key: value / max(total_steps, 1) for key, value in timing.items()}
    return mean_loss, {}


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss, save_optimizer=True):
    torch.save(
        {
            "epoch": int(epoch),
            "model": deepcopy(model).half().cpu(),
            "optimizer": optimizer.state_dict() if save_optimizer else None,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "date": datetime.now().isoformat(),
        },
        path,
    )


def run_train(config, run_dir, device, epochs, lr, weight_decay):
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    training_cfg = config.get("training", {})
    options = training_options(config, device)

    train_loader = create_dataloader(config, split="train")
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None

    model, use_finetune = _build_model(config, device)
    if options["freeze_feature_extractor"]:
        _freeze_feature_extractor(model)
    trainable_params = trainable_parameters(model)
    if not trainable_params:
        raise ValueError("No trainable YOLOv5 parameters remain after applying freeze settings.")

    print(
        f"[train] trainable_params={count_trainable_params(model)} total_params={count_total_params(model)} "
        f"amp={options['amp']} freeze_feature_extractor={options['freeze_feature_extractor']}"
    )
    loss_fn = build_loss(config.get("model", {}).get("type", "yolov5"), model, config)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(device, options["amp"])

    best_metric = float("inf")
    history = []
    img_size = config.get("model", {}).get("img_size", 640)

    for epoch in range(1, epochs + 1):
        train_loss, train_timing = _run_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            img_size=img_size,
            device=device,
            train_mode=True,
            amp=options["amp"],
            scaler=scaler,
            freeze_feature_extractor=options["freeze_feature_extractor"],
            log_timing=options["log_timing"],
        )
        val_loss = None
        val_timing = {}
        do_val = val_loader is not None and (epoch % options["val_interval"] == 0 or epoch == epochs)
        if do_val:
            val_loss, val_timing = _run_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                img_size=img_size,
                device=device,
                train_mode=False,
                amp=options["amp"],
                scaler=None,
                freeze_feature_extractor=options["freeze_feature_extractor"],
                log_timing=options["log_timing"],
            )

        metric = val_loss if val_loss is not None else train_loss
        can_update_best = val_loss is not None or val_loader is None
        if epoch % options["save_last_interval"] == 0 or epoch == epochs:
            _save_ckpt(
                weights_dir / "last.pt",
                epoch,
                model,
                optimizer,
                train_loss,
                val_loss,
                save_optimizer=options["save_optimizer"],
            )
        is_best = can_update_best and metric <= best_metric
        if is_best:
            best_metric = metric
        if options["save_best"] and is_best:
            _save_ckpt(
                weights_dir / "best.pt",
                epoch,
                model,
                optimizer,
                train_loss,
                val_loss,
                save_optimizer=options["save_optimizer"],
            )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "best_metric": float(best_metric),
        }
        if options["log_timing"]:
            row = merge_epoch_timing(row, {f"train_{k}": v for k, v in train_timing.items()})
            row = merge_epoch_timing(row, {f"val_{k}": v for k, v in val_timing.items()})
        history.append(row)
        print(
            f"[train] epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={'none' if val_loss is None else f'{val_loss:.6f}'}"
        )

    summary = {
        "mode": "train",
        "model_type": "yolov5",
        "epochs": int(epochs),
        "seed": None if training_cfg.get("seed") is None else int(training_cfg.get("seed")),
        "device": str(device),
        "finetune": bool(use_finetune),
        "num_classes": int(getattr(model, "nc", getattr(model.model[-1], "nc", 0))),
        "class_names": list(getattr(model, "names", [])),
        "history": history,
        "best_metric": float(best_metric),
        "training_options": options,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "weights": {
            "last": str((weights_dir / "last.pt").resolve()),
            "best": str((weights_dir / "best.pt").resolve()) if options["save_best"] else None,
        },
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["run_train"]
